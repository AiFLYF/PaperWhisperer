"""Aggregated paper search across Semantic Scholar and arXiv.

Both sources are queried concurrently and failures are reported per source, so
one provider being rate-limited does not blank out the other's results.
"""

from __future__ import annotations

import concurrent.futures
import logging
import urllib.parse
import xml.etree.ElementTree as ET

from paperwhisperer.core import config
from paperwhisperer.core.text import compact_text
from paperwhisperer.search.http import (
    build_ssl_context,
    describe_source_error,
    http_get_json,
    http_get_text,
)
from paperwhisperer.search.normalize import deduplicate_papers, normalize_paper_record

logger = logging.getLogger(__name__)

ARXIV_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv_papers(query: str, limit: int) -> list[dict]:
    """Query the arXiv Atom API and map entries onto the canonical paper shape."""
    encoded_query = urllib.parse.quote(query)
    url = f"{config.ARXIV_API_URL}?search_query=all:{encoded_query}&start=0&max_results={limit}"
    feed_text = http_get_text(
        url,
        timeout=config.ARXIV_TIMEOUT_SECONDS,
        headers={"User-Agent": config.APP_USER_AGENT},
        retries=config.ARXIV_MAX_RETRIES,
        ssl_context=build_ssl_context(),
        retry_max_delay=config.ARXIV_RETRY_MAX_DELAY,
    )
    root = ET.fromstring(feed_text)
    items = []

    for entry in root.findall("atom:entry", ARXIV_NAMESPACE):
        title = compact_text(
            entry.findtext("atom:title", default="", namespaces=ARXIV_NAMESPACE), limit=300
        )
        summary = compact_text(
            entry.findtext("atom:summary", default="", namespaces=ARXIV_NAMESPACE), limit=2000
        )
        paper_id = (
            entry.findtext("atom:id", default="", namespaces=ARXIV_NAMESPACE) or ""
        ).strip()
        published = (
            entry.findtext("atom:published", default="", namespaces=ARXIV_NAMESPACE) or ""
        ).strip()
        authors = [
            author.findtext("atom:name", default="", namespaces=ARXIV_NAMESPACE)
            for author in entry.findall("atom:author", ARXIV_NAMESPACE)
        ]
        pdf_url = ""
        for link in entry.findall("atom:link", ARXIV_NAMESPACE):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "").strip()
                break
        items.append(
            normalize_paper_record(
                "arXiv",
                {
                    "paper_id": paper_id,
                    "title": title,
                    "abstract": summary,
                    "authors": authors,
                    "published": published,
                    "venue": "arXiv",
                    "url": paper_id,
                    "pdf_url": pdf_url,
                },
            )
        )
    return items


def search_semantic_scholar_papers(query: str, limit: int) -> list[dict]:
    """Query the Semantic Scholar graph API."""
    params = urllib.parse.urlencode(
        {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,venue,url,authors,openAccessPdf,paperId",
        }
    )
    url = f"{config.SEMANTIC_SCHOLAR_SEARCH_URL}?{params}"
    headers = {"User-Agent": config.APP_USER_AGENT}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY

    payload = http_get_json(
        url,
        timeout=config.SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
        headers=headers,
        retries=config.SEMANTIC_SCHOLAR_MAX_RETRIES,
        retry_max_delay=config.SEMANTIC_SCHOLAR_RETRY_MAX_DELAY,
    )
    return [
        normalize_paper_record("Semantic Scholar", item) for item in payload.get("data", [])
    ]


def _execute_paper_source(source_name, search_fn, query, limit):
    """Run one provider, converting any failure into a per-source error string."""
    try:
        return source_name, search_fn(query, limit), None
    except Exception as exc:
        logger.warning("%s paper search failed: %s", source_name, exc)
        return source_name, [], describe_source_error(source_name, exc)


def search_papers(query, limit=None) -> dict:
    """Search every configured source and merge the results.

    Returns ``items`` (deduplicated, length-capped) plus an ``errors`` list so
    the UI can explain a partial result.
    """
    clean_query = compact_text(query, limit=240)
    if not clean_query:
        raise ValueError("Please enter a search query.")

    resolved_limit = config.clamp_int_value(
        limit,
        config.PAPER_SEARCH_RESULT_LIMIT,
        min_value=1,
        max_value=config.PAPER_SEARCH_RESULT_LIMIT,
    )

    sources = (
        ("Semantic Scholar", search_semantic_scholar_papers),
        ("arXiv", search_arxiv_papers),
    )

    results_by_source = {name: [] for name, _ in sources}
    errors_by_source = {name: None for name, _ in sources}

    if config.PAPER_SEARCH_PARALLEL and len(sources) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as executor:
            futures = [
                executor.submit(_execute_paper_source, name, fn, clean_query, resolved_limit)
                for name, fn in sources
            ]
            for future in concurrent.futures.as_completed(futures):
                name, found_items, error = future.result()
                results_by_source[name] = found_items
                errors_by_source[name] = error
    else:
        for name, fn in sources:
            name_back, found_items, error = _execute_paper_source(
                name, fn, clean_query, resolved_limit
            )
            results_by_source[name_back] = found_items
            errors_by_source[name_back] = error

    items: list[dict] = []
    for name, _ in sources:
        items.extend(results_by_source[name])

    errors = [errors_by_source[name] for name, _ in sources if errors_by_source[name]]

    return {
        "query": clean_query,
        "items": deduplicate_papers(items)[:resolved_limit],
        "errors": errors,
    }
