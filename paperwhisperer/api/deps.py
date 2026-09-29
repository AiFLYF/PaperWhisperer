"""Shared helpers for API routes."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

from fastapi import Request

from paperwhisperer.core.errors import build_error_response
from paperwhisperer.core.text import remove_file_safely

logger = logging.getLogger(__name__)


async def parse_json_object_request(request: Request):
    """Parse a JSON object body, returning ``(data, error_response)``.

    Exactly one of the two is ever non-None, so a route can bail out with a
    single check.
    """
    try:
        raw_body = await request.body()
    except Exception as exc:
        logger.warning("Failed to read JSON request body: %s", exc)
        return None, build_error_response("Unable to read request body.", code="body_read_failed")

    if not raw_body or not raw_body.strip():
        return {}, None

    try:
        data = json.loads(raw_body)
    except json.JSONDecodeError:
        return None, build_error_response("Invalid JSON body.", code="invalid_json")

    if not isinstance(data, dict):
        return None, build_error_response("JSON body must be an object.", code="json_not_object")
    return data, None


#: Shared worker pool for blocking LLM generators pushed off the event loop.
_STREAM_WORKERS = ThreadPoolExecutor(max_workers=8, thread_name_prefix="sse-stream")


async def pump_sync_events(
    produce,
    on_event,
    cleanup=None,
) -> AsyncIterator[bytes]:
    """Drive a blocking generator from a worker thread into an SSE response.

    ``produce`` is a synchronous generator that yields event dicts. Iterating
    it directly inside the event loop would stall every other request for the
    duration of the LLM work, so it runs in a worker thread and hands events
    back through an asyncio queue. The consumer side stays fully async.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    sentinel = object()

    def _run() -> None:
        try:
            for event in produce():
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the consumer
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    worker = loop.run_in_executor(_STREAM_WORKERS, _run)

    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            payload = on_event(item)
            if payload:
                yield payload
    finally:
        # Surface producer errors even if the client already disconnected.
        try:
            await worker
        except Exception:
            logger.debug("Stream producer ended with an error after client disconnect")
        if cleanup:
            cleanup()


def cleanup_path(file_path, description: str) -> None:
    remove_file_safely(file_path, description)
