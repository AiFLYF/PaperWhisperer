"""Health check route."""

from __future__ import annotations

import os
import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from paperwhisperer.core import config
from paperwhisperer.core.errors import now_iso

router = APIRouter()

APP_STARTED_AT = time.time()


@router.get("/api/health")
async def health_check():
    """Report liveness plus whether each runtime folder is present and writable.

    The two limits are echoed here because the front end used to carry its own
    copy of the upload ceiling. It does not any more: the browser learns the
    number from this endpoint, so an operator raising ``MAX_UPLOAD_BYTES`` does
    not leave the client quoting a stale maximum to the user.
    """
    runtime_folders = {
        "uploads": config.UPLOAD_FOLDER,
        "output": config.OUTPUT_FOLDER,
        "context": config.CONTEXT_FOLDER,
    }
    return JSONResponse(
        content={
            "status": "ok",
            "app": config.APP_NAME,
            "version": config.APP_VERSION,
            "timestamp": now_iso(),
            "uptime_seconds": max(0, round(time.time() - APP_STARTED_AT, 3)),
            "max_upload_bytes": config.MAX_UPLOAD_BYTES,
            "max_document_pages": config.MAX_DOCUMENT_PAGES,
            "folders": {
                name: {
                    "exists": os.path.isdir(path),
                    "writable": os.path.isdir(path) and os.access(path, os.W_OK),
                }
                for name, path in runtime_folders.items()
            },
        },
        headers={"Cache-Control": "no-store"},
    )
