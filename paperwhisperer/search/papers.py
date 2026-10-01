"""Aggregated paper search across Semantic Scholar and arXiv.

Both sources are queried concurrently and failures are reported per source, so
one provider being rate-limited does not blank out the other's results.

**The two providers are combined, not concatenated.** A preprint and its
published version come back as two rows — one from arXiv carrying an arXiv id
and no DOI, one from Semantic Scholar carrying a DOI and a ``paperId`` — and
the user must see one row. Merging them is
:mod:`paperwhisperer.search.normalize`'s job and it is the identity layer's
job underneath; this module's job is to *ask each provider for enough rows
that the merged list can still be filled*. See :data:`SOURCE_OVERFETCH`.
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
from paperwhisperer.search.normalize import (
    deduplicate_papers_report,
    normalize_paper_record,
)

logger = logging.getLogger(__name__)

ARXIV_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
#: arXiv publishes its own extension elements on the same feed. ``arxiv:doi`` is
#: the DOI of the *journal* version when the author has declared one, and
#: reading it is what lets an arXiv row and a Semantic Scholar row be proven to
#: be the same paper by a shared DOI rather than merely suspected by a shared
#: title. ``arxiv:journal_ref`` is the free-text venue string, which is more
#: informative than the literal ``"arXiv"`` for a paper that has since been
#: published.
ARXIV_EXTENSION_NAMESPACE = {"arxiv": "http://arxiv.org/schemas/atom"}

#: How many results each provider is asked for per result the user wants.
#:
#: Deduplication *removes* rows, and the old code deduplicated first and then
#: took ``[:limit]``, so any cross-source collapse silently cost the user a
#: result: ask for 8, merge two pairs away, show 6. Asking each provider for
#: twice what is needed means the merged list is normally still full, and a
#: short list now means the providers genuinely had nothing more — which is a
#: fact worth being able to show the user.
#:
#: **This doubles the request volume of a search**, which is a real cost on a
#: shared, rate-limited API and is not hidden here. It is bounded three ways:
#: by the multiplier, by :data:`SOURCE_MAX_RESULTS`, and by the fact that
#: ``_open_with_retry`` already honours ``Retry-After`` rather than hammering a
#: throttled endpoint. The plan's real answer to rate limiting is the shared
#: client's per-host limiter and response cache (section 8), which belongs in
#: :mod:`paperwhisperer.search.http` and covers every caller at once — not a
#: second rate limiter built into the search path, which would be a second
#: source of truth about how hard the providers may be hit.
SOURCE_OVERFETCH = 2

#: Ceiling on the per-source request. Semantic Scholar's search endpoint
#: rejects ``limit`` above 100, and an unbounded multiplier on a 20-result
#: configured limit would eventually ask for 40 rows to show 20 — more load on
#: a rate-limited shared API in exchange for results the user will not scroll
#: to. The cap is the point where the two constraints meet.
SOURCE_MAX_RESULTS = 40



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
                    # A journal DOI and a journal reference, when the author has
                    # declared them. Both are the author's own statement that
                    # this preprint is a published paper, which is a far
                    # stronger merge signal than the title they share, so they
                    # are carried through rather than dropped.
                    "doi": entry.findtext(
                        "arxiv:doi", default="", namespaces=ARXIV_EXTENSION_NAMESPACE
                    ),
                    "venue": entry.findtext(
                        "arxiv:journal_ref", default="", namespaces=ARXIV_EXTENSION_NAMESPACE
                    )
                    or "arXiv",
                    "url": paper_id,
                    "pdf_url": pdf_url,
                },
            )
        )
    return items


def search_semantic_scholar_papers(query: str, limit: int) -> list[dict]:
    """Query the Semantic Scholar graph API.

    ``externalIds`` is in the requested field list and is the whole point of
    this call: without it the API does not send the DOI at all, and the record
    leaves here with nothing to be identified by except its title. The cost of
    asking for it is a few hundred bytes per record; the cost of not asking is
    a network round trip per record later, against a rate-limited Crossref.
    """
    params = urllib.parse.urlencode(
        {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,venue,url,authors,openAccessPdf,paperId,externalIds",
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


def _per_source_limit(resolved_limit: int) -> int:
    """How many rows to ask each provider for, given what the user wants.

    The multiplier exists because deduplication removes rows: asking for
    exactly ``resolved_limit`` per provider and then merging means the user
    sees fewer results than they asked for precisely when the search worked
    best — arXiv and Semantic Scholar overlapping on a preprint. The cap keeps
    that from becoming an unbounded multiplication on a rate-limited API.
    """
    return min(resolved_limit * SOURCE_OVERFETCH, SOURCE_MAX_RESULTS)


def _interleave(per_source: list[list[dict]]) -> list[dict]:
    """Round-robin the providers' rows so neither source is starved.

    Concatenating would put every Semantic Scholar row ahead of every arXiv
    row, and since the result list is then capped, a search would routinely
    show the user N results from one provider and none from the other. Which
    provider answered first is not a ranking, so it must not decide what the
    user sees. Interleaving also puts a preprint next to its published version
    in the merged list, which makes the pair easy to spot as a pair.
    """
    merged: list[dict] = []
    for position in range(max((len(rows) for rows in per_source), default=0)):
        for rows in per_source:
            if position < len(rows):
                merged.append(rows[position])
    return merged


def search_papers(query, limit=None) -> dict:
    """Search every configured source and merge the results.

    Returns ``items`` (deduplicated, length-capped) plus an ``errors`` list so
    the UI can explain a partial result.

    **Truncation happens last and never silently loses a result.** The old
    order — deduplicate, then ``[:limit]`` — meant every genuine cross-source
    collapse cost the user a row. Each provider is now over-fetched
    (:data:`SOURCE_OVERFETCH`), so the cap normally cuts filler rather than
    content.

    The response also carries ``possible_duplicates``: the pairs that look like
    one paper on title and author evidence but could not be *proven* to be one
    paper, and which were therefore **kept**, both rows visible. The dedupe
    layer refuses to collapse a match it cannot confirm — a wrong merge
    destroys a paper the user can no longer find, while a duplicate costs one
    glance — and it reports the pair instead of hiding it. A client that
    renders ``possible_duplicates`` turns that into a "these may be the same"
    affordance; a client that ignores the key still shows both rows, which is
    the safe default.
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
    per_source_limit = _per_source_limit(resolved_limit)

    sources = (
        ("Semantic Scholar", search_semantic_scholar_papers),
        ("arXiv", search_arxiv_papers),
    )

    results_by_source = {name: [] for name, _ in sources}
    errors_by_source = {name: None for name, _ in sources}

    if config.PAPER_SEARCH_PARALLEL and len(sources) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as executor:
            futures = [
                executor.submit(_execute_paper_source, name, fn, clean_query, per_source_limit)
                for name, fn in sources
            ]
            for future in concurrent.futures.as_completed(futures):
                name, found_items, error = future.result()
                results_by_source[name] = found_items
                errors_by_source[name] = error
    else:
        for name, fn in sources:
            name_back, found_items, error = _execute_paper_source(
                name, fn, clean_query, per_source_limit
            )
            results_by_source[name_back] = found_items
            errors_by_source[name_back] = error

    combined = _interleave([results_by_source[name] for name, _ in sources])
    report = deduplicate_papers_report(combined)

    errors = [errors_by_source[name] for name, _ in sources if errors_by_source[name]]

    return {
        "query": clean_query,
        "items": report["items"][:resolved_limit],
        "errors": errors,
        "possible_duplicates": report["possible_duplicates"],
        "merged_duplicates": report["merged"],
    }
