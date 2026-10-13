.PHONY: help install install-locked install-parse install-render dev lock-help lock-check test lint format check check-deps \
	check-frontend check-stores check-compose serve clean parity \
	compose-config compose-up compose-smoke compose-down compose-logs \
	backup restore verify-backup restore-drill restore-hosted \
	fault-test fault-inject fault-inject-one load-probe
	gates gates-list gates-release
	eval-bakeoff eval-bakeoff-smoke eval-bakeoff-adversarial eval-bakeoff-force-ocr \
	eval-bakeoff-parsers

PYTHON ?= python
NODE ?= node
# vm.SourceTextModule needs this flag; without it every module fails to parse.
NODE_VM := $(NODE) --experimental-vm-modules

# Dotted names for the Hosted tier's own knobs, so an operator sets them in the
# environment (or a .env) and every compose target below picks them up. Empty
# by default, because every value in docker-compose.yml already has a default
# and `docker compose up` must work on a checkout with no .env at all.
COMPOSE ?= docker compose
#: Where `make backup` writes. On the same disk as the data it is not a backup;
#: point it somewhere else, which is the one thing this variable exists for.
BACKUP_DIR ?= backups
#: The archive `make verify-backup` and `make restore` operate on. Blank means
#: "the most recent one in $(BACKUP_DIR)", resolved when the recipe runs rather
#: than when the Makefile is parsed -- a `:=` here would capture the answer
#: before the backup that recipe is about to depend on exists.
BACKUP = $(shell ls -1t $(BACKUP_DIR)/paperwhisperer-*.tar.gz 2>/dev/null | head -1)
#: The personal-tier data directory the restore targets drill. Empty means an
#: empty scratch directory, which is what a drill must use: restoring over the
#: live data directory is the one thing the restore refuses to do by design.
RESTORE_DIR ?=
#: Which tier `make backup` backs up. Personal by default, because that is the
#: default tier (D2) and a Hosted backup needs a DSN the personal one never
#: asks for. Pass TIER=hosted, and DSN=... when the connection string is not
#: already in the environment.
TIER ?= personal

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# pyproject.toml is the single authoritative dependency list. requirements.txt is
# a pointer file (`-e .`) that adds no requirements of its own, so the two cannot
# disagree -- see the header of that file for the drift it used to hide.
#
# This resolves pyproject's `>=` floors and does NOT read `uv.lock`, so a given
# run of it is not reproducible. That is fine on a personal machine and wrong
# anywhere a result has to be attributed: `docling>=2.0` spans 2.x minors that
# change DoclingDocument's output, which is what made the T0.9 bake-off report
# unreproducible. Use `make install-locked` when the version set is the point.
install:  ## Install runtime dependencies (unpinned; not reproducible)
	$(PYTHON) -m pip install -r requirements.txt

# The same project, installed at the versions uv.lock records. The lock covers
# all 170 transitive packages including the extras that are easy to forget --
# docling and its torch chain, pypdfium2, psycopg -- so this is the target to
# reach for when a measurement has to be attributable to an environment.
install-locked:  ## Install exactly what uv.lock pins (dev + hosted + render)
	uv sync --locked --extra dev --extra hosted --extra render

install-parse:  ## Install the project plus the docling parse extra (pulls torch)
	$(PYTHON) -m pip install -e ".[parse]"

# The T3.3 page rasteriser. A separate target from `install` for the same
# reason `install-parse` is: the personal tier's defining constraint is the
# smallest footprint it can run with, and rendering a PDF page for the evidence
# panel is a nicety rather than core. Everything about the product works
# without it -- citations still resolve, still name their page and their box,
# and the evidence panel falls back to the layout rebuilt from the parse's own
# coordinates. Both packages are wheels: no compiler, no system poppler, and no
# second package manager needed on a personal machine.
install-render:  ## Install the project plus the render extra (PDF page images)
	$(PYTHON) -m pip install -e ".[render]"

dev:  ## Install runtime + development dependencies
	$(PYTHON) -m pip install -e ".[dev]"

lock-help:  ## Explain the dependency-locking state and how to use the lock
	@echo "The lock is uv.lock, and it is tracked. It pins all 170 transitive"
	@echo "packages, including the extras that are easy to forget: docling and"
	@echo "its torch chain, pypdfium2 for page rendering, psycopg for Hosted."
	@echo
	@echo "    make install-locked   install exactly what the lock records"
	@echo "    make lock-check       fail if the lock has gone stale"
	@echo "    uv lock               refresh it after editing pyproject.toml"
	@echo
	@echo "CI installs with 'uv sync --locked', so an edit to pyproject.toml that"
	@echo "was not followed by 'uv lock' fails the build instead of quietly"
	@echo "installing a set nobody recorded. That is the whole point: a lock file"
	@echo "nothing enforces is a decoration."
	@echo
	@echo "'make install' stays the unpinned pip path -- it resolves the '>='"
	@echo "floors and ignores the lock. Correct for a personal machine, wrong for"
	@echo "anything you intend to reproduce later."
	@echo
	@echo "The hazard that motivated the lock is 'docling>=2.0': 2.x minors change"
	@echo "DoclingDocument output, so the T0.9 bake-off report can only be"
	@echo "reproduced from a pinned environment. Reproduce it with"
	@echo "'make install-locked', not with 'make install'."

lock-check:  ## Verify uv.lock is up to date with pyproject.toml
	@uv lock --check

# Guards the invariant that makes the dependency list maintainable: a single
# source of truth. Cheap, and it fails the moment someone helpfully adds a
# package name back to requirements.txt and silently re-forks the list.
#
# Self-contained on purpose: it imports only the stdlib, so it keeps working
# when the environment is only half installed (which is exactly when someone is
# most likely to be tempted to "just fix" requirements.txt by hand).
check-deps:  ## Verify requirements.txt defers to pyproject instead of duplicating it
	@$(PYTHON) -c 'from pathlib import Path; import sys, tomllib; \
text = Path("requirements.txt").read_text(encoding="utf-8"); \
reqs = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]; \
bad = [r for r in reqs if not r.startswith(("-e", "-r", "--editable"))]; \
bad and sys.exit("requirements.txt re-declares dependencies: %s\npyproject.toml is the only dependency list; add extras there, not package names here." % bad); \
data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8")); \
extras = data["project"].get("optional-dependencies", {}); \
print("requirements.txt: %d own requirement(s) %s" % (len(reqs), reqs)); \
print("pyproject: %d base deps, extras %s" % (len(data["project"]["dependencies"]), sorted(extras)))'

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

lint:  ## Run ruff checks
	$(PYTHON) -m ruff check .

format:  ## Auto-fix lintable issues
	$(PYTHON) -m ruff check --fix .

check-frontend:  ## Verify the front-end module split (needs no reference copy)
	$(NODE_VM) tools/check_js_syntax.mjs templates/static/js
	$(PYTHON) tools/check_frontend.py

check-stores:  ## Verify every storage backend covers its interface
	$(PYTHON) tools/check_store_contract.py

# ---------------------------------------------------------------------------
# Release gates (T4.3)
# ---------------------------------------------------------------------------
# The plan's section 7 thresholds, made machine-decidable. See
# docs/RELEASE_GATES.md. Not in `check:` on purpose: `gates` takes about a
# minute and a half when eval/corpus/ is present, which is too slow for a
# target meant to run on every save -- but it is fast enough for CI, which is
# where it actually runs (the `gates` job in .github/workflows/ci.yml).
#
# `gates` uses the BUILD rule: only a gate that failed exits non-zero. The
# corpus is gitignored, so on a clean clone three gates SKIP and two are
# BLOCKED, and the run still exits 0 while printing each of those separately.
# `gates-release` uses the RELEASE rule, where an undecided gate also blocks --
# it is expected to fail today, because two gates have no labelled set.
gates:  ## Run the release gates (fails only on a gate that actually failed)
	$(PYTHON) -m eval.run_gates --out eval/reports

gates-list:  ## Print the gate definitions and their thresholds, running nothing
	$(PYTHON) -m eval.run_gates --list

gates-release:  ## Pre-release gate: refuse to ship unless every gate passed
	$(PYTHON) -m eval.run_gates --release --out eval/reports

# ---------------------------------------------------------------------------
# Parser bake-off (T0.9)
# ---------------------------------------------------------------------------
# Re-runs the parser comparison that backs docs/PARSER_BAKE_OFF.md. Not in
# `check:` and not a release gate: a full docling pass over the 52-document
# corpus is a CPU-bound job that takes tens of minutes (see
# docs/PARSER_BAKE_OFF.md section 4 for the measured figure), which is far too
# slow for either. It is also not comparable across corpora -- `corpus_version`
# is part of every report, and `--fail-on-regression` refuses to compare
# reports that disagree on it.
#
# RUN is a directory name under eval/reports/, not a path, so the three files
# a run produces (parser_eval.{md,csv,json}) stay grouped and diffable.
RUN ?= bakeoff-local
# Kept separate so `RUN=run3-force-ocr` does not yield `run3-force-ocr-force-ocr`.
FORCE_OCR_RUN ?= $(RUN)-force-ocr

# The environment the recorded run in docs/PARSER_BAKE_OFF.md section 4 was
# measured under. These are not defaults -- they are pinned so that a re-run
# reproduces those numbers instead of merely looking like them.
#
# PARSER_TIMEOUT_SECONDS=900 is load-bearing. At the 120s default, docling
# exceeds the timeout on 5 of 52 documents, and a *timeout* gets recorded as a
# *parse failure*: the run reports failures that are not the parser's fault.
# The real worst case is 424.4s, so 900 gives headroom without being unbounded
# (config clamps at 1800).
#
# Override any of these on the command line, e.g.
#   make eval-bakeoff RUN=run2 PARSER_TIMEOUT_SECONDS=1800
PARSER_TIMEOUT_SECONDS ?= 900
PARSER_BACKEND ?= gateway
PARSER_DOCLING_MODE ?= inprocess
HF_ENDPOINT ?= https://hf-mirror.com
BAKEOFF_ENV = HF_ENDPOINT=$(HF_ENDPOINT) \
	PARSER_TIMEOUT_SECONDS=$(PARSER_TIMEOUT_SECONDS) \
	PARSER_BACKEND=$(PARSER_BACKEND) \
	PARSER_DOCLING_MODE=$(PARSER_DOCLING_MODE)

# Serial, not parallel: `ms` is a reported metric, so two runs sharing a GPU and
# a page cache would each report timings inflated by the other.
eval-bakeoff:  ## Bake off legacy vs docling over the full corpus (slow: ~33 min measured)
	$(BAKEOFF_ENV) $(PYTHON) -m eval.run_parser_eval --parsers legacy,docling \
		--out eval/reports/$(RUN)

eval-bakeoff-smoke:  ## Bake off the first 3 corpus documents only (seconds; smoke check)
	$(BAKEOFF_ENV) $(PYTHON) -m eval.run_parser_eval --parsers legacy,docling \
		--limit 3 --out eval/reports/$(RUN)-smoke

# --include-excluded *adds* the 4 adversarial/ files to the batch; it does not
# run them alone, so this re-parses all 52 regular documents too (~33 min).
eval-bakeoff-adversarial:  ## Bake off regular + adversarial/ via --include-excluded (~33 min)
	$(BAKEOFF_ENV) $(PYTHON) -m eval.run_parser_eval --parsers legacy,docling \
		--include-excluded --out eval/reports/$(RUN)-adversarial

# Scoped to the 11 scanned documents, which is where forced OCR could possibly
# matter. Measured result (docs section 4.4): a no-op for docling 2.131.0, which
# already OCRs by default -- identical character counts, document by document.
eval-bakeoff-force-ocr:  ## Forced-OCR bake off over the 11 scanned documents
	$(BAKEOFF_ENV) $(PYTHON) -m eval.run_parser_eval --parsers legacy,docling \
		--force-ocr --categories scanned_clear,scanned_lowq,hybrid_scan_text \
		--out eval/reports/$(FORCE_OCR_RUN)

eval-bakeoff-parsers:  ## Print which parsers this machine can run, parsing nothing
	$(PYTHON) -m eval.run_parser_eval --list-parsers --parsers legacy,docling

# ---------------------------------------------------------------------------
# Hosted tier (Docker). None of these are in `check:`, and that is deliberate.
# ---------------------------------------------------------------------------
# The personal tier is the default and runs with zero Docker (decision D2), so
# a target that needs a Docker daemon cannot sit in an aggregate a contributor
# runs on every change — it would make "run make check" mean "install Docker",
# which is exactly the inversion this project decided against. These targets are
# opt-in, and CI runs the one that matters (`compose-smoke`) on its own schedule.

# Resolves and validates the compose file without pulling or building anything.
# Cheap enough to be worth running on every compose edit, and it catches the
# class of mistake -- a malformed mount, a service with no image -- that is
# otherwise only found after a multi-minute build.
check-compose:  ## Validate docker-compose.yml without building or pulling
	$(COMPOSE) config --quiet && $(COMPOSE) config --profiles

# The single command T2.5's acceptance criterion names.
compose-up:  ## Bring up the Hosted tier (app + PostgreSQL/pgvector)
	$(COMPOSE) up -d --build --wait

# The CI smoke, as a make target so the steps CI runs and the steps a developer
# runs are the same steps. `--wait` blocks on both services' healthchecks, which
# is what makes this a test of the stack rather than of the container runtime:
# the app's healthcheck is /api/health and the db's asserts the `vector`
# extension is actually present.
#
# PAPER_IMAGE_TARGET=parse builds the docling image instead. It is not the
# default because that target pulls the torch chain, and a developer running
# this on a laptop should not pay a multi-gigabyte download to find out that
# compose works.
compose-smoke:  ## Build, start, health-check, and tear down the Hosted tier
	$(COMPOSE) up -d --build --wait
	@echo "--- db: the vector extension is present"
	@$(COMPOSE) exec -T db psql -U $${POSTGRES_USER:-paperwhisperer} \
		-d $${POSTGRES_DB:-paperwhisperer} \
		-tAc "SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector','pg_trgm') ORDER BY extname"
	@echo "--- app: the store reports the Hosted tier"
	@$(COMPOSE) exec -T app python -c "from paperwhisperer.infra.stores import build_stores; print(build_stores().describe())"
	@echo "--- smoke passed"
	$(COMPOSE) down

compose-down:  ## Stop the Hosted tier, keeping volumes
	$(COMPOSE) down

# `down -v` is separate and says so: it drops the database volume, which is
# every document the Hosted tier holds. Never wire it to a target a CI job runs
# as a cleanup step without thinking about it twice.
compose-logs:  ## Tail the Hosted tier's logs
	$(COMPOSE) logs -f --tail=100

# ---------------------------------------------------------------------------
# Backup and restore (T2.6). Both tiers, no Docker, no PostgreSQL for the
# personal one.
# ---------------------------------------------------------------------------

# The personal tier's backup is its data directory. BACKUP_DIR must be on a
# different disk than the data it is protecting, and this default is not: a
# backup next to the thing it backs up survives exactly the events backups are
# for. The message says so rather than leaving it as a comment nobody reads.
#
# TIER=hosted switches to `pg_dump` plus the object files. Same target, because
# "back up this deployment" should not mean a different command on a different
# tier -- an operator moving to the Hosted tier should not have to re-learn the
# command, only add the DSN.
backup:  ## Back up this deployment (TIER=personal|hosted) to $(BACKUP_DIR)
	@case "$(BACKUP_DIR)" in "$(CURDIR)"/*|"$(CURDIR)") \
		echo "WARNING: $(BACKUP_DIR) is inside the project. A backup on the same" >&2; \
		echo "         disk as the data it protects survives nothing. Use BACKUP_DIR=" >&2; \
		echo "         on a separate disk or a mounted share." >&2;; \
	esac
	$(PYTHON) tools/backup.py --tier "$(TIER)" --out "$(BACKUP_DIR)" $(if $(DSN),--dsn "$(DSN)",)

# Verify without writing. The operator holding a backup on a disk they are not
# sure about needs to know it is intact *before* deciding to use it.
verify-backup:  ## Check every digest in $(BACKUP) without restoring
	@test -n "$(BACKUP)" || { echo "No backup in $(BACKUP_DIR)/ to verify." >&2; exit 1; }
	$(PYTHON) tools/restore.py --verify "$(BACKUP)"

# Deliberately has no default target directory. The restore refuses a non-empty
# one, and a make target that defaulted to the live data directory would either
# fail every time or need --force, which is a footgun wearing a convenience's
# clothes. An empty scratch directory is the only safe default.
restore:  ## Restore $(BACKUP) into $(RESTORE_DIR) (must be empty or absent)
	@test -n "$(BACKUP)" || { echo "No backup in $(BACKUP_DIR)/ to restore." >&2; exit 1; }
	$(PYTHON) tools/restore.py --to "$(RESTORE_DIR)" "$(BACKUP)"

# T2.6's acceptance criterion as a repeatable target: back up, restore into a
# scratch directory, and compare byte for byte. `diff -r` is the comparison
# because it is the one that fails on a single differing byte, and a drill that
# compared row counts would pass on a restore that mangled everything around
# the rows. It walks the three runtime folders rather than the whole checkout,
# because those are what the backup captured -- diffing the repository root
# would "fail" on every source file as a one-sided entry and teach the reader
# to ignore the output.
#
# SQLite's -wal/-shm are excluded: they are a function of whether a connection
# happened to be open, not of the data. The restore deliberately checkpoints
# them away, and its own census connection re-creates an empty pair, so their
# presence in the restored tree is not a defect.
#
# Runs the CLI, not the test suite, so what is drilled is the path an operator
# actually types.
restore-drill:  ## Back up this checkout, restore it to a scratch dir, diff byte for byte
	@set -e; \
	scratch=$$(mktemp -d); \
	out=$$scratch/backup; \
	restored=$$scratch/restored; \
	echo "scratch: $$scratch"; \
	echo "=== 1. backup ==="; \
	$(PYTHON) tools/backup.py --out "$$out" --name drill.tar.gz; \
	echo "=== 2. verify (no writes) ==="; \
	$(PYTHON) tools/restore.py --verify "$$out/drill.tar.gz"; \
	echo "=== 3. restore into an empty directory ==="; \
	$(PYTHON) tools/restore.py --to "$$restored" "$$out/drill.tar.gz"; \
	echo "=== 4. byte-for-byte comparison ==="; \
	for folder in uploads output context; do \
		if [ -d "$$folder" ]; then \
			diff -r --exclude='*-wal' --exclude='*-shm' "$$folder" "$$restored/$$folder"; \
			echo "  $$folder: identical"; \
		fi; \
	done; \
	echo "=== drill passed: every captured file came back with the bytes it went in with ==="; \
	rm -rf "$$scratch"

# The Hosted half of the restore: `pg_restore` for the database. Separate from
# `restore` because the two halves are genuinely different operations on
# different systems, and because this one is destructive enough to deserve a
# name of its own. `--clean` is opt-in: restoring over an existing schema
# should fail loudly on the first conflicting object, and that is the
# operator's call to make, not this target's.
restore-hosted:  ## pg_restore $(BACKUP:.tar.gz=.dump) into $(DSN)
	@test -n "$(DSN)" || { echo "需要 DSN=postgresql://... （或环境变量 STORAGE_PG_DSN）" >&2; exit 2; }
	@dump=$(subst .tar.gz,.dump,$(BACKUP)); \
	test -f "$$dump" || { echo "找不到 pg_dump 归档：$$dump" >&2; exit 1; }
	$(PYTHON) tools/restore.py --tier hosted --dsn "$(DSN)" --dump "$$dump" $(if $(CLEAN),--clean,)

# Not in `check:` on purpose -- see the note above compose-config. The personal
# tier must stay runnable with `make check` and no Docker anywhere near it, which
# is why the Hosted tier's compose check is its own target.
check: lint test check-deps check-frontend check-stores  ## Lint + test + deps + front-end + store contracts (no Docker)

# ---------------------------------------------------------------------------
# Fault injection and load (T4.4). Personal tier, no Docker, no network, no
# k6/Locust -- neither is installable here and neither drives the thing that
# matters on this deployment, which is a single-writer SQLite file.
# ---------------------------------------------------------------------------

# The suite. Fast (~40s) because the faults are cheap: a real `os._exit` in a
# child process, a real HTTP server on loopback, a real `BEGIN EXCLUSIVE`, a
# handful of real OS processes. No Docker, no PostgreSQL, no network.
#
# This is the target that belongs in CI, and it is green while four known
# defects are open, because those are pinned as strict xfails: they pass
# *because* the bug is still there and start failing the moment it is fixed,
# which is the signal to delete the pin and promote this into `check:`.
fault-test:  ## Run the T4.4 fault-injection suite (real faults, no Docker)
	$(PYTHON) -m pytest tests/test_fault_injection.py -q

# The same scenarios as a human-readable report, and a non-zero exit when any
# of them is not observable. Deliberately NOT in `check:`: it fails today, on
# purpose, because four failures are currently silent (docs/FAILURE_MODES.md).
# Adding it to `check:` would mean either accepting a permanently red
# `make check` or deleting the assertion that the failures are real. When the
# last SILENT line goes away, this is the target to promote.
fault-inject:  ## Trigger every T4.4 failure for real; non-zero if any is silent
	$(PYTHON) tools/fault_inject.py --all

# One scenario, for reproducing a single failure by hand.
#   make fault-inject-one SCENARIO=mineru-failures
SCENARIO ?=
EXTRA ?=
fault-inject-one:  ## Trigger one failure scenario (SCENARIO=worker-restart)
	@test -n "$(SCENARIO)" || { echo "usage: make fault-inject-one SCENARIO=worker-restart" >&2; \
		echo "available: $$($(PYTHON) tools/fault_inject.py --list | tr '\n' ' ')" >&2; exit 2; }
	$(PYTHON) tools/fault_inject.py $(SCENARIO) $(EXTRA)

# Concurrency probe for the personal tier. Stdlib only, and it prints its own
# caveats in the output: a loopback single-process number is a contention
# measurement, not a capacity plan, and the output says so where somebody
# reading the numbers will actually see it rather than only in a docstring.
#
#   make load-probe
#   make load-probe LEVELS=1,2,4,8 PROBE_JOBS=100
LEVELS ?= 1,2,4,8,16,32
PROBE_JOBS ?= 40
load-probe:  ## Measure personal-tier contention and report p50/p95 (stdlib only)
	$(PYTHON) tools/load_probe.py --concurrency "$(LEVELS)" --jobs "$(PROBE_JOBS)" $(EXTRA)

serve:  ## Run the development server
	$(PYTHON) web_app.py

parity:  ## One-off: diff the split front-end against the pre-split sources
	@echo "This compares against .parity/ originals, which are not committed."
	$(NODE_VM) .parity/check_js_syntax.mjs templates/static/js
	$(NODE_VM) .parity/check_frontend_parity.mjs .parity/orig_app.js templates/static/js
	$(NODE_VM) .parity/check_js_graph.mjs templates/static/js
	$(PYTHON) .parity/check_css_parity.py .parity/orig_style.css templates/static/css

clean:  ## Remove caches and runtime artifacts
	rm -rf .pytest_cache .ruff_cache .benchmarks
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
