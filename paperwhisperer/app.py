"""FastAPI application factory.

Composition root: mounts static assets, registers the security-header
middleware and includes every router. Import this module rather than building
the app inline so tests and the CLI share one wiring path.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from paperwhisperer.api import (
    routes_documents,
    routes_health,
    routes_papers,
    routes_qa,
)
from paperwhisperer.core import config

logger = logging.getLogger(__name__)

config.ensure_runtime_folders()

templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


def create_app() -> FastAPI:
    application = FastAPI(title=config.APP_NAME, version=config.APP_VERSION)

    @application.middleware("http")
    async def add_security_headers(request: Request, call_next):
        started_at = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started_at) * 1000:.2f}"
        for header, value in config.SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    application.mount(
        "/static",
        StaticFiles(directory=str(config.STATIC_DIR)),
        name="static",
    )

    application.include_router(routes_health.router)
    application.include_router(routes_documents.router)
    application.include_router(routes_qa.router)
    application.include_router(routes_papers.router)

    @application.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        response = templates.TemplateResponse(request, "index.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    return application


app = create_app()
