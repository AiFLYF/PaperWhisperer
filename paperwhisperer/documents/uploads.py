"""Upload persistence with size limits and signature verification."""

from __future__ import annotations

import logging

from paperwhisperer.core.text import remove_file_safely
from paperwhisperer.documents.validation import validate_saved_file_signature

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 64


async def save_upload_file(upload_file, destination_path, max_bytes) -> int:
    """Stream an upload to disk, then verify its magic bytes.

    Writing in chunks means an oversized upload is rejected without ever
    buffering the whole payload, and a failed validation leaves nothing behind.
    """
    total_bytes = 0
    try:
        with open(destination_path, "wb") as handle:
            while True:
                chunk = await upload_file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError(
                        f"Uploaded file is too large. Limit: {max_bytes // (1024 * 1024)} MB"
                    )
                handle.write(chunk)
        if total_bytes <= 0:
            raise ValueError("Uploaded file is empty.")
        validate_saved_file_signature(destination_path)
        return total_bytes
    except Exception:
        remove_file_safely(destination_path, "failed upload file")
        raise
