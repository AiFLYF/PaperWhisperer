"""File-type validation for uploads and remote downloads.

Extensions are attacker-controlled, so every payload is additionally checked
against its magic bytes. This is what stops an HTML landing page served as
``application/pdf`` from being parsed as a document.
"""

from __future__ import annotations

import os
import urllib.parse

from paperwhisperer.core import config

_REJECTED_CONTENT_TYPES = frozenset({
    "application/json",
    "application/xhtml+xml",
    "application/xml",
    "text/xml",
})

_GENERIC_BINARY_CONTENT_TYPES = frozenset({
    "application/octet-stream",
    "binary/octet-stream",
    "application/download",
    "application/x-download",
    "application/force-download",
})

_ALLOWED_CONTENT_TYPES_BY_EXT = {
    ".pdf": frozenset({"application/pdf", "application/x-pdf", "application/acrobat",
            "application/vnd.pdf"}),
    ".txt": frozenset({"text/plain", "text/markdown"}),
    ".docx": frozenset({
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/x-zip-compressed",
    }),
    ".pptx": frozenset({
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/zip",
        "application/x-zip-compressed",
    }),
}

_ZIP_MAGIC_PREFIXES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


def normalize_content_type(value) -> str:
    return str(value or "").split(";", 1)[0].strip().lower()


def parse_content_length(headers):
    raw_value = str(headers.get("Content-Length") or "").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    return value if value >= 0 else None


def validate_remote_content_length(headers, max_bytes: int) -> None:
    """Reject empty or oversized remote payloads before downloading them."""
    content_length = parse_content_length(headers)
    if content_length is None:
        return
    if content_length <= 0:
        raise ValueError("Downloaded file is empty.")
    if content_length > max_bytes:
        raise ValueError(f"Remote file is too large. Limit: {max_bytes // (1024 * 1024)} MB")


def validate_remote_content_type(file_name: str, content_type) -> None:
    """Reject responses whose declared type cannot be the requested document."""
    normalized = normalize_content_type(content_type)
    if not normalized:
        return

    if normalized.startswith("text/html") or normalized in _REJECTED_CONTENT_TYPES:
        raise ValueError(
            "The paper link returned a non-document response instead of a downloadable file."
        )

    ext = os.path.splitext(file_name)[1].lower()
    allowed_for_ext = _ALLOWED_CONTENT_TYPES_BY_EXT.get(ext, frozenset())
    if normalized in _GENERIC_BINARY_CONTENT_TYPES or normalized in allowed_for_ext:
        return
    if ext == ".txt" and normalized.startswith("text/"):
        return
    raise ValueError(f"Remote content type is not compatible with {ext or 'the selected'} file.")


def get_sample_head(sample) -> bytes:
    head = bytes(sample or b"")[:2048].lstrip()
    if head.startswith(b"\xef\xbb\xbf"):
        head = head[3:].lstrip()
    return head.lower()


def validate_document_file_signature(file_name: str, sample, source_name: str) -> None:
    """Verify the payload's magic bytes match the claimed extension."""
    if not sample:
        raise ValueError(f"The {source_name} is empty.")

    head = get_sample_head(sample)
    if head.startswith((b"<!doctype html", b"<html")) or b"<html" in head[:512]:
        raise ValueError(f"The {source_name} appears to be an HTML page instead of a document.")

    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf" and b"%PDF-" not in sample[:1024]:
        raise ValueError(f"The {source_name} does not look like a valid PDF.")
    if ext in {".docx", ".pptx"} and not sample.startswith(_ZIP_MAGIC_PREFIXES):
        raise ValueError(f"The {source_name} does not look like a valid {ext[1:].upper()} file.")
    if ext == ".txt" and b"\x00" in sample[:2048]:
        raise ValueError(f"The {source_name} appears to be binary.")


def validate_saved_file_signature(file_path: str) -> None:
    with open(file_path, "rb") as handle:
        sample = handle.read(4096)
    validate_document_file_signature(
        os.path.basename(file_path), sample, "uploaded file"
    )


def read_remote_file_sample(response, file_name: str) -> bytes:
    sample = response.read(4096)
    validate_document_file_signature(file_name, sample, "remote file")
    return sample


def looks_like_direct_file_url(raw_url) -> bool:
    """True when the URL path ends in a supported document extension."""
    lowered = urllib.parse.urlparse(str(raw_url or "").strip()).path.lower()
    return any(lowered.endswith(ext) for ext in config.SUPPORTED_EXTENSIONS)
