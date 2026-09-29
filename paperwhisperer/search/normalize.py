"""Paper record normalization shared by search results and the reading queue."""

from __future__ import annotations

import re

from paperwhisperer.core import config
from paperwhisperer.core.text import (
    compact_text,
    normalize_author_list,
    now_iso,
    parse_year,
)


def normalize_paper_record(source: str, record: dict) -> dict:
    """Map a provider payload onto the app's canonical paper shape."""
    if source == "Semantic Scholar":
        open_access_pdf = record.get("openAccessPdf") or {}
        return {
            "source": source,
            "paper_id": str(record.get("paperId") or "").strip(),
            "title": compact_text(record.get("title") or "", limit=300),
            "abstract": compact_text(record.get("abstract") or "", limit=2000),
            "authors": normalize_author_list(record.get("authors") or []),
            "year": parse_year(record.get("year")),
            "venue": compact_text(record.get("venue") or "Semantic Scholar", limit=120),
            "url": str(record.get("url") or "").strip(),
            "pdf_url": str(open_access_pdf.get("url") or "").strip(),
        }

    return {
        "source": source,
        "paper_id": str(record.get("paper_id") or record.get("id") or "").strip(),
        "title": compact_text(record.get("title") or "", limit=300),
        "abstract": compact_text(record.get("abstract") or record.get("summary") or "", limit=2000),
        "authors": normalize_author_list(record.get("authors") or []),
        "year": parse_year(record.get("year") or record.get("published") or ""),
        "venue": compact_text(record.get("venue") or source, limit=120),
        "url": str(record.get("url") or record.get("id") or "").strip(),
        "pdf_url": str(record.get("pdf_url") or "").strip(),
    }


def normalize_paper_collection(values, max_items: int = config.READING_QUEUE_LIMIT) -> list[dict]:
    """Clean, dedupe and cap a client-supplied list of papers."""
    if not isinstance(values, list):
        return []
    items = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        title = compact_text(value.get("title"), limit=300)
        if not title:
            continue
        url = str(value.get("url") or "").strip()[:1000]
        pdf_url = str(value.get("pdf_url") or "").strip()[:1000]
        key = re.sub(r"\s+", " ", title).strip().lower() or url or pdf_url
        if key in seen:
            continue
        seen.add(key)
        items.append({
            "source": compact_text(value.get("source"), limit=80),
            "paper_id": compact_text(value.get("paper_id"), limit=160),
            "title": title,
            "abstract": compact_text(value.get("abstract"), limit=1200),
            "authors": normalize_author_list(value.get("authors") or [], limit=8),
            "year": parse_year(value.get("year")),
            "venue": compact_text(value.get("venue"), limit=120),
            "url": url,
            "pdf_url": pdf_url,
            "saved_at": compact_text(value.get("saved_at") or now_iso(), limit=40),
        })
        if len(items) >= max_items:
            break
    return items


def deduplicate_papers(items) -> list[dict]:
    """Drop repeated titles while preserving provider ordering."""
    deduplicated = []
    seen_titles = set()
    for item in items or []:
        title_key = re.sub(r"\s+", " ", str(item.get("title") or "")).strip().lower()
        if not title_key or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        deduplicated.append(item)
    return deduplicated
