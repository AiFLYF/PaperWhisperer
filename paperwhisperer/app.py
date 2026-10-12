"""FastAPI application factory.

Composition root: mounts static assets, registers the security-header
middleware and includes every router. Import this module rather than building
the app inline so tests and the CLI share one wiring path.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware

from paperwhisperer.api import (
    routes_documents,
    routes_evidence,
    routes_health,
    routes_jobs,
    routes_papers,
    routes_qa,
    routes_v1,
)
from paperwhisperer.api.auth import assert_startup_safe, describe_auth
from paperwhisperer.core import config, observability
from paperwhisperer.infra.stores.factory import resolve_backend

logger = logging.getLogger(__name__)

config.ensure_runtime_folders()

templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))

#: The OpenAPI description, which is the only place a reader learns that ``/v1``
#: is frozen and that ``/api`` is not.
#:
#: Written here rather than left to FastAPI's default because the single most
#: useful thing a client can be told about this service is which half of it may
#: change. ``/v1`` promises a version prefix, stable field names, documented
#: error codes and a 501 for capabilities this build does not have — a definite
#: "no" rather than an empty success. ``/api`` is what the bundled browser talks
#: to and is kept working for as long as that browser ships; it is *not* covered
#: by the promise. A client that reads this and picks wrongly is a misreading the
#: document did not have to allow.
API_V1_DESCRIPTION = """
## 版本化的 `/v1` 契约

`/v1` 是**对外冻结**的契约：字段名、错误码与状态码语义在 v1 内不再变更。破坏性
修改必须走新的前缀。

- **资源化**：`/v1/documents`、`/v1/documents/{id}/pages`、`/v1/jobs/{id}` 等按资源
  寻址，而非按动作。
- **错误码**：每个失败响应的 `code` 字段是稳定值，`error` 是中文说明、只面向人。
  客户端只应依赖 `code`。
- **否定也是契约**：本版本没有 PDF 页图渲染能力，`/v1/documents/{id}/pages/{page}/image`
  返回 **501** 与 `page_render_unavailable`，而不是空白 200。
- **状态流只带状态**：`/v1/jobs/{id}/stream` 只推送状态迁移，并带 SSE `id:` 序号行；
  断线后带 `Last-Event-ID` 重连即可补齐错过的事件。

未加前缀的 `/api/*` 是既有前端使用的兼容面，保持可用但不在冻结范围内。
""".strip()


def _resolve_worker_store(application: FastAPI) -> Any:
    """The job store the routes would use, resolved the way FastAPI resolves it.

    **The override table is consulted in the order FastAPI itself would resolve
    it, and that order is the whole function.** ``routes_jobs.get_job_store`` is
    a dependency whose own parameter is ``Depends(get_store_bundle)``, so the
    default in its signature is a :class:`fastapi.params.Depends` *marker
    object*, not a bundle. Calling it with no argument therefore passes that
    marker straight through as the bundle, and the store factory — which wants a
    real :class:`StoreBundle` — fails on ``bundle.backend``.

    That is why this function passes the bundle in *explicitly*: it resolves the
    inner dependency first, the way FastAPI's own solver does, and then calls the
    outer one with a real argument. The alternative — a bare ``get_job_store()``
    — reads like the honest thing and is the single reason this service has never
    once started a worker in any real deployment.

    The outer override is still checked first, so a test that overrides
    ``get_job_store`` directly keeps getting exactly the store it installed. A
    test that overrides only ``get_store_bundle`` — the seam one level down —
    also gets what it asked for, through the second branch.
    """
    override = application.dependency_overrides.get(routes_jobs.get_job_store)
    if override is not None:
        return override()

    bundle_override = application.dependency_overrides.get(routes_evidence.get_store_bundle)
    bundle = (
        bundle_override()
        if bundle_override is not None
        else routes_evidence.get_store_bundle()
    )
    return routes_jobs.get_job_store(bundle)


def _describe_storage_backend() -> str:
    """The configured backend's name, for a failure message.

    ``or "未知"`` rather than an exception: this runs on the error path, where a
    second failure raised while describing the first would replace the real
    reason with a second one. :func:`resolve_backend` raises for a misspelled
    ``STORAGE_BACKEND``, and a name that cannot be resolved is honestly
    "unknown".
    """
    try:
        return resolve_backend(config)
    except Exception:  # noqa: BLE001 - a diagnostic must not raise
        return "未知"


def start_workers(application: FastAPI) -> list:
    """Start the job workers for this application, behind ``JOBS_ENABLED``.

    The store comes through the application's *own* dependency overrides rather
    than through a fresh ``build_job_store``, and that is not a test-only
    courtesy. ``routes_jobs.get_job_store`` is a FastAPI dependency, so a test
    overrides it to point at a temporary database; if the lifespan built its own
    store instead, the worker would drain a queue in the developer's real data
    directory while the routes read and wrote a different one — two stores, one
    job table, and a worker that never sees the jobs it was started to do. Reading
    the override table is how the worker and the routes are guaranteed to be
    talking about the same database.

    **There is no supported configuration in which this returns ``[]`` while
    ``JOBS_ENABLED`` is on.** That is worth writing down because the failure it
    used to produce was invisible and total: a bare ``get_job_store()`` call
    raised ``AttributeError`` on its ``Depends`` marker default, the ``except``
    swallowed it at ERROR, and the process came up healthy with no worker at all.
    Every test that exercised the lifespan installed an override for
    ``get_job_store``, so the one branch a real deployment takes was the one
    branch nobody ran.

    **Why this stays at ERROR rather than dropping to WARNING.** The two ways a
    process legitimately has no worker are both already below ERROR and both are
    a decision rather than a malfunction: ``JOBS_ENABLED=false`` returns early
    with a line saying the queue is off. What remains here is "the store is
    configured and cannot be opened", and the personal tier has no "no store
    configured" state at all — it always builds a SQLite bundle at the shipped
    default path — so every failure past this point is a real fault. The message
    now names the backend and says what a missing worker actually costs, because
    the old one said only "长任务将保持排队", which described the symptom in the
    passive voice and named neither the cause nor the consequence an operator has
    to act on.

    Returns the workers started, so teardown can stop exactly those even if
    something else added to the registry meanwhile.
    """
    if not config.JOBS_ENABLED:
        logger.info("JOBS_ENABLED 为关闭，长任务回到同步处理，不启动 worker。")
        return []

    try:
        store = _resolve_worker_store(application)
    except Exception as exc:  # noqa: BLE001 - a queue that cannot start must not stop the app
        logger.error(
            "任务存储不可用（存储后端 %s）：%s。未启动 worker 的进程不会消费任务队列——"
            "异步提交（POST /api/jobs/analyze、POST /api/jobs/ask）会一直停在 queued，"
            "只有同步提交会退化为在请求内执行。",
            _describe_storage_backend(),
            exc,
            exc_info=True,
        )
        return []

    started = routes_jobs.start_job_worker(store)
    logger.info("已启动 %d 个任务 worker：%s。", len(started),
                "、".join(sorted(routes_jobs.PIPELINES)))
    return started


def stop_workers(application: FastAPI, started: list) -> None:
    """Stop the workers the lifespan started, and say so.

    The registry is cleared whether or not ``started`` is non-empty, because a
    worker that outlived a failed startup would otherwise be un-reachable from
    the next lifespan on the same object — the one worker a shutdown path cannot
    find is the one that leaks.

    **What is logged depends on whether the shutdown actually worked.**
    ``stop_job_workers`` removes a worker from the registry only once its thread
    has really stopped, so anything still listed afterwards is a thread that is
    still running. Reporting "stopped N" for such a worker would lose exactly the
    information the registry was reordered to keep — it is what makes a failed
    teardown indistinguishable from a clean one — so the survivors are named, and
    the count that gets logged is the count that actually stopped.
    """
    attempted = routes_jobs.stop_job_workers()
    still_running = routes_jobs.stuck_workers()
    stopped = len(attempted) - len([w for w in still_running if w in attempted])
    if still_running:
        logger.error(
            "已停止 %d 个任务 worker，但有 %d 个仍在运行（%s）；"
            "它们的句柄仍在 worker 注册表中，可被下一次关闭再次回收。"
            "未完成的任务将由下一次启动回收。",
            stopped,
            len(still_running),
            "、".join(sorted(worker.worker_id for worker in still_running)),
        )
        return
    if attempted or started:
        logger.info("已停止 %d 个任务 worker；未完成的任务将由下一次启动回收。", stopped)


class RangeSafeGZipMiddleware(GZipMiddleware):
    """GZip, minus the one response shape compression turns into a lie.

    A partial response carries ``Content-Range: bytes 0-5000/6793`` -- a claim
    about *raw* file offsets. Compress the body and the claim survives while the
    bytes no longer match it: the client decompresses 5001 bytes of gzip stream
    and is told they are offsets 0-5000 of the original. Measured against this
    app before the guard existed, with a 5001-byte range on a 6.8 KB script:

        HTTP/1.1 206 Partial Content
        content-range: bytes 0-5000/6793
        content-encoding: gzip

    Starlette does not exclude partials -- only ``text/event-stream`` -- and the
    size rule does not save it either: a *small* range slips under
    ``minimum_size`` (a 100-byte range came back uncompressed, which is luck,
    not policy) while any range over 500 bytes is compressed. Resume-a-download
    and any media element that seeks are the consumers that would get a corrupt
    file.

    The guard is on the request rather than the response because that is where
    the answer is already known: a client that sends ``Range`` is asking for
    bytes it will place at an offset, so it gets them verbatim. A range response
    is inherently a partial transfer, so losing compression on it costs nothing
    measurable.
    """

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or "range" in Headers(scope=scope):
            await self.app(scope, receive, send)
            return
        await super().__call__(scope, receive, send)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """Bring up the Hosted tier's identity, the job queue, or refuse to start.

        Four things happen here and all are deliberate:

        * ``assert_startup_safe`` raises when ``AUTH_MODE`` names a real mode
          and no credentials are configured. **A Hosted deployment that serves
          traffic with authentication unconfigured is worse than one that does
          not start**, because the first looks healthy and the second is
          obviously broken. The personal tier's default is
          ``AUTH_MODE=disabled`` and this call is then a no-op that returns the
          disabled authenticator, so the single-user deployment is untouched.
        * the store bundle is resolved through a resolver and wrapped per
          request by the tenant scope, so a query that forgets its scope means
          "my own documents" rather than "everything".
        * a :class:`~paperwhisperer.infra.jobs.JobWorker` is started per job
          kind, which is T2.3's deliverable: until something drives
          :class:`~paperwhisperer.infra.jobs.JobWorker`, the durable job system
          is an engine with nothing driving it, and every long task happens
          inside a request. It is behind ``JOBS_ENABLED`` so a deployment can
          turn the queue off and go back to synchronous handling while
          diagnosing something.
        * and on the way out the workers are stopped and the observability
          pipeline is shut down. **A worker that is not stopped on shutdown is a
          leaked thread** — and a leaked worker holding a lease is worse than
          useless, because the next process recovers its job and both processes
          bill for the same cloud parse. The stop happens before the
          observability shutdown so that a worker's last log lines still have a
          pipeline to go through.

        The recovery pass runs *before* the workers start, not on their first
        tick. A tick happens after a poll interval, and "the queue is stuck
        until the poll interval elapses" is a worse sentence than a call that is
        plainly in the code.
        """
        authenticator = assert_startup_safe()
        application.state.authenticator = authenticator
        logger.info("鉴权层已就绪：%s", describe_auth(authenticator))

        # Telemetry, if the operator asked for it. Every counter and trace
        # event in T4.1 is written by a real call site and nothing ever turned
        # collection on, so `OBSERVABILITY_ENABLED=1` was read into config and
        # then ignored — a deployment asking for metrics exported none. The
        # gate is explicit because the default is off, and the test that
        # asserts "importing the module changes nothing" would not catch an
        # activation that happened here at startup rather than at import.
        if bool(getattr(config, "OBSERVABILITY_ENABLED", False)):
            observability.configure()
            logger.info("可观测性已启用。")

        started = start_workers(application)
        try:
            yield
        finally:
            stop_workers(application, started)
            observability.shutdown()

    application = FastAPI(lifespan=lifespan,
        title=config.APP_NAME,
        version=config.APP_VERSION,
        description=API_V1_DESCRIPTION,
        openapi_tags=[
            {
                "name": "documents",
                "description": "文档资源：上传、登记、元数据、页面、重新解析、删除。",
            },
            {"name": "jobs", "description": "长任务的提交结果与状态订阅。"},
            {"name": "search", "description": "学术检索。"},
        ],
    )

    @application.middleware("http")
    async def add_security_headers(request: Request, call_next):
        started_at = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started_at) * 1000:.2f}"
        for header, value in config.SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    # Compression, added last so it is the outermost layer and therefore sees
    # the finished response -- including the static mount, which no middleware
    # registered before it would cover. Measured on this tree: the page's own
    # assets are 407 KB uncompressed and 123 KB gzipped, a 70% saving, with
    # i18n.js alone dropping 163 KB to 50 KB.
    #
    # `text/event-stream` is excluded by Starlette itself
    # (DEFAULT_EXCLUDED_CONTENT_TYPES), which is the property this app cannot do
    # without: every streaming route here is SSE, and a buffered or re-encoded
    # stream would break the one thing those routes exist for. Verified, not
    # assumed -- the analyse stream still answers with no content-encoding,
    # `x-accel-buffering: no`, and an `event: start` frame inside the first two
    # seconds.
    application.add_middleware(RangeSafeGZipMiddleware, minimum_size=500, compresslevel=6)

    application.mount(
        "/static",
        StaticFiles(directory=str(config.STATIC_DIR)),
        name="static",
    )

    application.include_router(routes_health.router)
    application.include_router(routes_documents.router)
    application.include_router(routes_qa.router)
    application.include_router(routes_evidence.router)
    application.include_router(routes_papers.router)
    # The frozen surface. Registered last so that a `/v1/...` path the legacy
    # routers also claim (`/v1/documents/{id}/evidence/{evidence_id}`) keeps
    # resolving to the router that already owned it; the two v1 routers do not
    # overlap with each other or with the evidence one.
    application.include_router(routes_v1.router)
    application.include_router(routes_jobs.router)

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        response = templates.TemplateResponse(request, "index.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    return application


app = create_app()
