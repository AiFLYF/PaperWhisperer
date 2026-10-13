# PaperWhisperer Hosted tier: one image, two targets, no torch in the default.
#
#     docker build -t paperwhisperer .              # base, the default
#     docker build -t paperwhisperer-parse --target parse .
#
# The base target deliberately does **not** install the ``parse`` extra. The
# plan's risk table names docling's torch chain as the heaviest thing this
# project can install, and T0.8's whole point was to keep it out of the base
# install so a personal deployment does not pay for it. A base image that
# carried torch would make that decision unrecoverable in practice: the
# download is the expensive part, and every `docker compose up --build` would
# pay it again. The `parse` target exists so the extra is one flag away rather
# than a fork of this file.
#
# The Hosted tier's app service builds the base target. A deployment that wants
# in-process Docling sets `PAPER_IMAGE_TARGET=parse` (see docker-compose.yml) or
# points the parser at a docling-serve container, which is the shape the plan
# actually recommends for a machine that cannot hold torch.

# ---------------------------------------------------------------------------
# Builder: install into a prefix the runtime stage copies.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# The source is copied *before* the install, and that ordering is deliberate
# rather than lazy. The obvious arrangement -- copy pyproject.toml, install
# dependencies, then copy the source -- is what this file did first, and it does
# not work: `requirements.txt` is a pointer file containing `-e .`, so `pip
# install -r requirements.txt` is an *editable* install of the project, which
# needs the package directory to exist. Without the source it fails with
# "package directory 'paperwhisperer' does not exist".
#
# The usual workaround is to install the dependency set without the project,
# which with this layout means parsing pyproject.toml by hand and maintaining a
# second list of every dependency in the image. That is a second source of truth
# for the dependency set, and `make check-deps` exists precisely to stop that
# class of drift elsewhere in the repository. So the trade is made explicitly:
# a source change invalidates this layer, and the install runs again. Correct
# beats cached, and a cache that only sometimes hits is a cache nobody trusts.
COPY pyproject.toml requirements.txt README.md ./
COPY paperwhisperer ./paperwhisperer
# `env_loader` is a top-level module, not a package, and `paperwhisperer.core.
# config` imports it at module scope. Without it the image builds cleanly and
# then dies on its first import with `ModuleNotFoundError: No module named
# 'env_loader'` -- which is exactly how this was found.
#
# The install does carry it now: pyproject's `[tool.setuptools] py-modules`
# ships it in the wheel, so `pip install .` above puts a copy in site-packages.
# Verified from a wheel, in a directory that did not contain the repository:
# `import paperwhisperer.core.config` resolves `env_loader` out of the install.
#
# The COPY stays anyway, and not as a habit: it is the only one of the two
# paths that has ever been exercised *in this image*. Nothing in CI builds the
# image, so dropping it would trade a line of duplication for a failure that
# only shows up on an operator's first request to a container that reported
# itself healthy. Remove it when a Docker build runs somewhere that would go
# red -- which is the same condition under which the `psycopg` workaround above
# should go.
COPY env_loader.py ./
COPY web_app.py logo.ico ./
COPY templates ./templates

# `--prefix` rather than a venv: the runtime stage copies a directory, and a venv
# carries interpreter symlinks and its own bin/ that then shadow the base image's
# PATH. `--prefix=/install` puts the packages in exactly the layout the runtime
# stage's own Python expects, so the copy is a merge into /usr/local and nothing
# else has to be configured.
#
# Non-editable on purpose. An editable install records a path to /build, which
# does not exist in the runtime stage; the result would be an app that imports
# only because the compose file bind-mounts the source over it, and a broken
# image the moment somebody runs it without the mount.
#
# **`psycopg[binary]` is installed explicitly, and it is not in pyproject.**
# The Hosted tier's store needs a PostgreSQL driver, the project's base
# dependency list does not declare one, and without it the app starts, serves
# `/api/health`, and then raises `RuntimeError: Hosted 档位需要 PostgreSQL 驱动`
# on the first request that touches the store. That is a genuinely nasty
# failure: the container is healthy, the port answers, and the tier is
# non-functional.
#
# It is here, in the image, rather than in pyproject because the image *is* the
# Hosted tier -- the personal tier never runs this image (decision D2: zero
# Docker), so a driver in it costs the personal tier nothing. The proper fix is a
# `hosted` extra in pyproject.toml, which this task does not own; it is reported
# in docs/DEPLOYMENT.md. `psycopg` 3 is the supported generation; the backend
# also accepts psycopg2, and `load_driver` prefers psycopg 3.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --prefix=/install . "psycopg[binary]>=3.1"

# ---------------------------------------------------------------------------
# Runtime: the base image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Labels the build's provenance into the image itself, so an operator holding
# an image and no repository can still tell what it is.
LABEL org.opencontainers.image.title="PaperWhisperer" \
      org.opencontainers.image.description="Evidence-first research workspace (Hosted tier)" \
      org.opencontainers.image.source="https://github.com/paperwhisperer/paperwhisperer"

# curl is for the health check below, which is the only way `docker compose up
# --wait` and CI can tell "the app started" from "the app is a container that
# exists". Without it the check has to be a Python one-liner, and a health check
# written as an unquoted Python string in YAML is a lint problem every time
# somebody edits it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTAPI_HOST=0.0.0.0 \
    FASTAPI_PORT=8000

# Merged into the base image's own prefix, which is where its Python looks.
COPY --from=builder /install /usr/local

WORKDIR /app

# The source again, for the same reason a venv is not enough: the project is
# installed, but `web_app.py` and `templates/` are the deployment surface and
# `config` resolves them as paths relative to the package, so they have to be on
# disk for `import paperwhisperer.app` to succeed at all.
COPY pyproject.toml README.md requirements.txt ./
COPY paperwhisperer ./paperwhisperer
COPY env_loader.py web_app.py logo.ico ./
COPY templates ./templates

# Non-root. Parsing an untrusted PDF is a large attack surface and the container
# should not be root when it happens. The uid is fixed rather than dynamic so a
# bind-mounted host directory can be chowned to it once, instead of to whichever
# uid the image happened to be built with.
#
# The three runtime folders are created and owned here because the app writes
# into them on its first request, and a read-only root filesystem on a fresh
# volume would otherwise turn that into a 500 rather than into a clear error.
RUN useradd --create-home --uid 10001 paperwhisperer \
    && mkdir -p /app/uploads /app/output /app/context \
    && chown -R paperwhisperer:paperwhisperer /app
USER paperwhisperer

EXPOSE 8000

# `/api/health` rather than `/`, because the index route renders a template and a
# health check that depends on template loading fails for a reason that has
# nothing to do with whether the service is up. This one reports liveness and
# whether the runtime folders are writable, which is the pair an operator
# actually needs at boot.
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
    CMD curl -fsS "http://127.0.0.1:${FASTAPI_PORT}/api/health" || exit 1

# `exec` form, and uvicorn by module rather than the `web_app:main` console
# script: the script lives in the same install and would work, but naming the
# module keeps the process tree to a single exec with no shell between PID 1 and
# the server. `--proxy-headers` because a Hosted deployment sits behind something
# that terminates TLS, and without it every request's client address is the
# proxy's.
CMD ["uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]

# ---------------------------------------------------------------------------
# Parse target: the base image plus docling.
# ---------------------------------------------------------------------------
# A separate target rather than a Compose profile, and the distinction is
# deliberate. A Compose profile would put a *service* behind the flag, and the
# thing being gated here is an *install* -- a second image that shares nothing
# with the first except this file. A build target is the shape that matches:
#
#     docker build --target parse -t paperwhisperer-parse .
#
# and the same source, the same entrypoint, one extra dependency.
#
# This is also the image that is expensive to build, which is the whole reason it
# is not the default. Expect a multi-gigabyte image and a first run that
# downloads Docling's models. The plan's alternative -- a docling-serve
# container -- is the better answer on a machine that cannot hold either, and
# nothing here prevents it.
#
# Declared *after* the runtime stage on purpose: `docker build .` builds the
# last stage, so the heavy one must be the last stage -- otherwise a default
# `docker build .` silently pulls torch and undoes T0.8 the first time somebody
# forgets `--target`. The cost is that the base target is no longer the file's
# last word, which is why the compose file pins `target: runtime` explicitly
# rather than relying on the default.
FROM runtime AS parse

# Installed into the runtime image rather than copied from a second builder, so
# `parse` is *literally* the base image plus one dependency and cannot drift from
# it. A separate builder stage here would be a second copy of the dependency
# resolution to keep in step with this one.
RUN python -m pip install docling

# The label is the difference between somebody reading `docker images` and
# knowing why theirs is four times the size, and not.
LABEL org.opencontainers.image.description="PaperWhisperer + in-process Docling (heavy)"
