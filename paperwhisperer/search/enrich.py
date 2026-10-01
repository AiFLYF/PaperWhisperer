"""Filling in the metadata a paper record is missing, chiefly a lost DOI.

"DOI 丢失可回查" — a record that arrived without a DOI must be able to get
one back. In practice that happens constantly and for boring reasons:
:func:`paperwhisperer.search.normalize.normalize_paper_record` builds a flat
record out of whatever the provider sent, and Semantic Scholar puts the DOI
in ``externalIds``, which that function drops. The result is a real paper
sitting in the user's reading queue whose identity is stuck on an arXiv id or,
worse, on its title, while the authoritative identifier was available the
whole time. Enrichment goes and gets it.

**Every network call is injected.** The functions here that talk to Crossref
or OpenAlex take a *fetcher* — a plain callable taking a :class:`LookupQuery`
and returning a raw provider payload. Tests pass a lambda; production passes
:func:`build_crossref_source` / :func:`build_openalex_source`, which are thin
wrappers over :mod:`paperwhisperer.search.http`. The risk table's
"OpenAlex/Crossref/S2 限速 → 统一 http client 限速与缓存" line is deliberately
*not* implemented here: a second rate limiter in this module would be a second
source of truth about how hard the providers are allowed to be hit, and the
plan says the shared client is where that belongs.

**A candidate has to earn its DOI.** A bibliographic query for a common title
returns several works, and Crossref ranks them by relevance, not by identity.
Writing the top hit's DOI onto a record would be a guess dressed as data, and
a wrong DOI is worse than a missing one: it makes the paper un-deduplicatable
against every future record, and it sends the user to a citation they did not
ask for. So a candidate is accepted only when its title, first author and
year all check out against the record being enriched, and when more than one
*distinct* DOI passes that bar the result is reported as ambiguous and
nothing is written. "Two plausible answers" is a state this product can show
the user; "pick the first one" is not one it can.

**Failure degrades, it does not raise.** A provider being rate-limited, down,
or simply not carrying this paper must leave the record exactly as it was.
Enrichment is an improvement, never a precondition: a user who is offline
still gets their search results.
"""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from paperwhisperer.core.text import parse_year
from paperwhisperer.search.http import describe_source_error
from paperwhisperer.search.identity import (
    MIN_FALLBACK_TITLE_CHARS,
    PaperIdentity,
    normalize_arxiv_id,
    titles_equivalent,
)

# --------------------------------------------------------------------------- #
# Configuration
#
# These four read ``os.environ`` directly rather than going through
# ``core.config``. That is a known duplication, not a preference: enrichment
# belongs to this task and the configuration system is owned elsewhere, so
# rather than edit a shared module or silently fall back to hard-coded values
# the defaults are spelled out here and flagged for consolidation into
# ``core.config`` alongside the other provider settings.
# --------------------------------------------------------------------------- #


def _enrichment_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, ignoring junk."""
    try:
        return max(1, int(os.getenv(name, "").strip() or default))
    except (TypeError, ValueError):
        return default


CROSSREF_API_URL = os.getenv("CROSSREF_API_URL", "https://api.crossref.org/works").strip()
OPENALEX_API_URL = os.getenv("OPENALEX_API_URL", "https://api.openalex.org/works").strip()
#: Crossref's "polite pool" and OpenAlex's both key on a contact address; without
#: one they serve the anonymous pool, which is both slower and more likely to be
#: rate-limited. There is no sensible default address, so this stays empty.
ENRICHMENT_CONTACT_EMAIL = os.getenv("ENRICHMENT_CONTACT_EMAIL", "").strip()
ENRICHMENT_TIMEOUT_SECONDS = _enrichment_int("ENRICHMENT_TIMEOUT_SECONDS", 15)
ENRICHMENT_MAX_RETRIES = _enrichment_int("ENRICHMENT_MAX_RETRIES", 2)
#: How many candidates a bibliographic query asks for. Five is the smallest
#: number that still lets the ambiguity check see more than one candidate —
#: with ``rows=1`` a wrong top hit would be indistinguishable from a right one.
ENRICHMENT_ROWS = _enrichment_int("ENRICHMENT_ROWS", 5)

#: How far apart a candidate's publication year and the record's may be. A
#: preprint posted in December and a journal issue dated the following June are
#: one paper; anything wider is a different work that happens to share a title.
ENRICHMENT_YEAR_TOLERANCE = 1


# --------------------------------------------------------------------------- #
# The injection contract
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LookupQuery:
    """What to ask a provider for, derived from the record being enriched.

    A DOI lookup is used when the identity already has one — there is nothing
    to recover, but other metadata (year, venue, external ids) is still worth
    completing. Otherwise a bibliographic lookup on title, year and first
    author is the attempt to *recover* the DOI.
    """

    doi: str = ""
    title: str = ""
    year: str = ""
    first_author: str = ""

    @property
    def is_doi_lookup(self) -> bool:
        return bool(self.doi)

    @classmethod
    def from_identity(cls, identity: PaperIdentity) -> LookupQuery:
        return cls(
            doi=identity.doi,
            title=identity.title,
            year=identity.normalized_year,
            first_author=identity.first_author,
        )

    def bibliographic_text(self) -> str:
        """The free-text query for a bibliographic search.

        Title first, then year, then first author: the order is relevance
        order, and the year is there to break ties between a paper and a
        later paper that reuses its title.
        """
        parts = [self.title.strip(), self.year.strip(), self.author_for_query()]
        return " ".join(part for part in parts if part).strip()

    def author_for_query(self) -> str:
        """The author's surname, which is what bibliographic indexes index on.

        Sending "Kaiming He" instead of "He" measurably degrades Crossref
        recall, because a bibliographic query is matched token-wise against an
        index where the surname is the keyed field.
        """
        author_key = self.first_author.strip()
        return author_key.split()[-1] if author_key else ""


@dataclass(frozen=True)
class MetadataCandidate:
    """One provider record, normalized to the same shape as a :class:`PaperIdentity`."""

    identity: PaperIdentity
    source: str


@dataclass(frozen=True)
class MetadataSource:
    """A provider: how to ask it and how to read the answer.

    ``fetch`` returns a raw provider payload or ``None``. It is allowed to
    raise; :func:`enrich_identity` converts the exception into a user-facing
    message so that one provider being down never fails the request.
    """

    name: str
    fetch: Callable[[LookupQuery], Mapping[str, Any] | None]
    parse: Callable[[Mapping[str, Any]], list[MetadataCandidate]]
    #: Whether this provider can answer a bare DOI lookup. A provider that
    #: cannot is simply skipped when the identity already has a DOI.
    supports_doi_lookup: bool = True


@dataclass(frozen=True)
class EnrichmentResult:
    """What enrichment did, and what the user should be told about it.

    ``errors`` and ``notes`` are kept apart because they are opposites to the
    user: a note says something became more true, an error says it did not.
    Collapsing them into one "warnings" list is how a UI ends up rendering a
    successfully recovered DOI next to an orange exclamation mark.
    """

    identity: PaperIdentity
    source: str = ""
    filled_fields: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    ambiguous: bool = False

    @property
    def ok(self) -> bool:
        """Whether anything at all was filled in."""
        return bool(self.filled_fields)

    @property
    def recovered_doi(self) -> bool:
        """Whether this call is what gave the record its DOI."""
        return "doi" in self.filled_fields

    @property
    def messages_zh(self) -> tuple[str, ...]:
        """Every user-facing string, notes first: what worked, then what did not."""
        return self.notes + self.errors


# --------------------------------------------------------------------------- #
# Payload parsing (pure functions — no I/O, fully testable)
# --------------------------------------------------------------------------- #

#: mEDRA's prefix for DOIs minted for arXiv postings. Crossref records link
#: them through ``relation.has-preprint``, which is the one place a publisher
#: states "this is the preprint of me" — worth reading for the arXiv id it
#: carries, because it is an authoritative version of the link that
#: enrichment is otherwise trying to infer.
ARXIV_DOI_PREFIX = "10.48550/arxiv."


def _first_string(value) -> str:
    """Crossref returns titles and container titles as one-element lists."""
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip() if value else ""
    return str(value or "").strip()


def _crossref_year(item: Mapping[str, Any]) -> str:
    issued = item.get("issued") or {}
    date_parts = (issued.get("date-parts") or [[]])[0]
    return parse_year(date_parts[0] if date_parts else "")


def _crossref_first_author(item: Mapping[str, Any]) -> str:
    authors = item.get("author") or []
    if not authors:
        return ""
    first = authors[0] or {}
    if isinstance(first, str):
        return first.strip()
    family = str(first.get("family") or "").strip()
    given = str(first.get("given") or "").strip()
    # "Family, Given" so that normalize_author_key's comma rule can strip it
    # back to the surname without depending on word order.
    return f"{family}, {given}" if family and given else (family or given)


def _crossref_preprint_arxiv_ids(item: Mapping[str, Any]) -> list[str]:
    relation = item.get("relation") or {}
    raw_ids: list[str] = []
    for kind in ("has-preprint", "is-preprint-of"):
        for entry in relation.get(kind) or []:
            if isinstance(entry, dict) and entry.get("id"):
                raw_ids.append(str(entry["id"]).strip())
    found: list[str] = []
    for raw in raw_ids:
        if raw.lower().startswith(ARXIV_DOI_PREFIX):
            candidate = normalize_arxiv_id(raw[len(ARXIV_DOI_PREFIX) :])
            if candidate and candidate not in found:
                found.append(candidate)
    return found


def parse_crossref_payload(payload) -> list[MetadataCandidate]:
    """Turn a Crossref ``/works`` response into candidates.

    Handles both response shapes: the search endpoint returns
    ``{"message": {"items": [...]}}`` and the single-work endpoint returns
    ``{"message": {...}}`` with no ``items``. A payload from either is
    accepted because a caller that only wants to complete a known DOI should
    not have to know which endpoint it hit.
    """
    message = (payload or {}).get("message") or {}
    items = message.get("items")
    if items is None:
        items = [message] if message.get("DOI") else []
    candidates: list[MetadataCandidate] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        doi = str(item.get("DOI") or "").strip()
        preprint_ids = _crossref_preprint_arxiv_ids(item)
        candidates.append(
            MetadataCandidate(
                identity=PaperIdentity(
                    doi=doi,
                    arxiv_id=preprint_ids[0] if preprint_ids else "",
                    title=_first_string(item.get("title")),
                    year=_crossref_year(item),
                    first_author=_crossref_first_author(item),
                ),
                source="Crossref",
            )
        )
    return [candidate for candidate in candidates if candidate.identity.doi]


def parse_openalex_payload(payload) -> list[MetadataCandidate]:
    """Turn an OpenAlex ``/works`` response into candidates.

    The DOI arrives as a full ``https://doi.org/...`` URL, which
    :func:`normalize_doi` already understands, and the arXiv id — when OpenAlex
    knows one at all — has to be dug out of the landing page, since OpenAlex's
    ``ids`` object carries no arXiv field.
    """
    results = (payload or {}).get("results")
    if results is None:
        results = [payload] if payload and payload.get("id") else []
    candidates: list[MetadataCandidate] = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        identity = PaperIdentity(
            doi=item.get("doi") or item.get("id") or "",
            openalex_id=item.get("id") or "",
            arxiv_id=_openalex_landing_arxiv_id(item),
            title=str(item.get("title") or item.get("display_name") or "").strip(),
            year=parse_year(item.get("publication_year") or ""),
            first_author=_openalex_first_author(item),
        )
        if identity.doi or identity.openalex_id:
            candidates.append(MetadataCandidate(identity=identity, source="OpenAlex"))
    return candidates


def _openalex_first_author(item: Mapping[str, Any]) -> str:
    authorships = item.get("authorships") or []
    if not authorships:
        return ""
    author = (authorships[0] or {}).get("author") or {}
    return str(author.get("display_name") or "").strip()


def _openalex_landing_arxiv_id(item: Mapping[str, Any]) -> str:
    landing = (item.get("primary_location") or {}).get("landing_page_url") or ""
    return normalize_arxiv_id(str(landing))


# --------------------------------------------------------------------------- #
# The production fetchers
# --------------------------------------------------------------------------- #


def _default_get_json(url: str, timeout: int, headers: dict, retries: int):
    """The one outbound HTTP path, imported lazily from the shared client.

    The obvious alternative is to open a connection here, and it is rejected
    because ``search/http.py`` already owns the SSL context, the User-Agent
    convention and the 429-aware retry backoff. Duplicating any of that would
    mean two places to fix when a provider changes its rate-limit behaviour.
    """
    from paperwhisperer.search.http import build_ssl_context, http_get_json

    return http_get_json(
        url,
        timeout=timeout,
        headers=headers,
        retries=retries,
        ssl_context=build_ssl_context(),
        retry_max_delay=6,
    )


def _contact_params() -> str:
    if not ENRICHMENT_CONTACT_EMAIL:
        return ""
    return f"&mailto={urllib.parse.quote(ENRICHMENT_CONTACT_EMAIL)}"


def build_crossref_source(
    fetch: Callable[..., Mapping[str, Any]] | None = None,
    *,
    url: str = CROSSREF_API_URL,
    timeout: int = ENRICHMENT_TIMEOUT_SECONDS,
    rows: int = ENRICHMENT_ROWS,
    user_agent: str = "",
) -> MetadataSource:
    """Wire Crossref onto the shared HTTP client.

    ``fetch`` defaults to :func:`_default_get_json` and is injectable purely so
    the URL construction can be asserted without a network call.
    """
    get_json = fetch or _default_get_json
    headers = {"User-Agent": user_agent} if user_agent else {}

    def _fetch(query: LookupQuery) -> Mapping[str, Any] | None:
        if query.is_doi_lookup:
            target = f"{url}/{urllib.parse.quote(query.doi, safe='/')}"
        else:
            params = urllib.parse.urlencode(
                {
                    "query.bibliographic": query.bibliographic_text(),
                    "rows": rows,
                    "select": "DOI,title,author,issued,container-title,relation",
                }
            )
            target = f"{url}?{params}{_contact_params()}"
        return get_json(target, timeout=timeout, headers=headers, retries=ENRICHMENT_MAX_RETRIES)

    return MetadataSource(name="Crossref", fetch=_fetch, parse=parse_crossref_payload)


def build_openalex_source(
    fetch: Callable[..., Mapping[str, Any]] | None = None,
    *,
    url: str = OPENALEX_API_URL,
    timeout: int = ENRICHMENT_TIMEOUT_SECONDS,
    rows: int = ENRICHMENT_ROWS,
    user_agent: str = "",
) -> MetadataSource:
    """Wire OpenAlex onto the shared HTTP client."""
    get_json = fetch or _default_get_json
    headers = {"User-Agent": user_agent} if user_agent else {}

    def _fetch(query: LookupQuery) -> Mapping[str, Any] | None:
        if query.is_doi_lookup:
            target = f"{url}/doi:{urllib.parse.quote(query.doi, safe='/')}"
        else:
            params = urllib.parse.urlencode(
                {"search": query.bibliographic_text(), "per-page": rows}
            )
            target = f"{url}?{params}{_contact_params()}"
        return get_json(target, timeout=timeout, headers=headers, retries=ENRICHMENT_MAX_RETRIES)

    return MetadataSource(name="OpenAlex", fetch=_fetch, parse=parse_openalex_payload)


def default_metadata_sources() -> list[MetadataSource]:
    """The production provider set: Crossref first, then OpenAlex.

    Crossref goes first because a publisher-registered DOI is the tier this
    module's identity key ranks highest, and because Crossref carries the
    ``has-preprint`` relation that yields the arXiv id for free. OpenAlex
    second is still a first-class source, not a consolation prize: it is the
    only one of the two that indexes preprints as works in their own right.
    """
    return [build_crossref_source(), build_openalex_source()]


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def _candidate_is_verified(query: LookupQuery, candidate: MetadataCandidate) -> bool:
    """Whether a provider record really is the record being looked up.

    Three checks, each vetoing:

    ``title``
        Same title, or the same title modulo a dropped subtitle — the same
        comparison the identity layer uses, so "the record that looked like a
        match when we merged it" and "the record that looks like a match when
        we enrich it" cannot disagree.
    ``first author``
        Surname must agree, unless one side simply has no author recorded.
        A missing author is not evidence of a mismatch, so it does not veto;
        a *different* author is.
    ``year``
        Must agree within :data:`ENRICHMENT_YEAR_TOLERANCE` years, for the
        preprint-then-publication reason given on the constant. A candidate
        outside the window is not this paper.
    """
    identity = candidate.identity
    title_key = query.title.strip()
    if len(title_key) < MIN_FALLBACK_TITLE_CHARS:
        return False
    if not titles_equivalent(title_key, identity.title):
        return False

    query_surname = query.author_for_query().casefold()
    if query_surname and identity.author_key and query_surname != identity.author_key:
        return False

    if query.year and identity.normalized_year:
        gap = abs(int(query.year) - int(identity.normalized_year))
        if gap > ENRICHMENT_YEAR_TOLERANCE:
            return False
    return True


def enrich_identity(
    identity: PaperIdentity,
    sources: Sequence[MetadataSource] = (),
) -> EnrichmentResult:
    """Complete ``identity`` from the given providers, never raising.

    With a DOI present the providers are asked about that DOI and only *empty*
    fields are filled. Without one, a bibliographic lookup tries to recover a
    DOI, and a candidate is written only if it passes
    :func:`_candidate_is_verified`.

    Multiple providers are tried in order and the first one to yield a verified
    answer wins; later providers are not asked, which keeps the request count
    down on the rate-limited shared client. A provider that fails is recorded
    and the next is tried, so a Crossref outage still lets OpenAlex answer.
    """
    if not isinstance(identity, PaperIdentity):
        raise TypeError("enrich_identity expects a PaperIdentity")

    query = LookupQuery.from_identity(identity)
    errors: list[str] = []
    notes: list[str] = []

    for source in sources or ():
        if query.is_doi_lookup and not source.supports_doi_lookup:
            continue
        try:
            payload = source.fetch(query)
        except Exception as exc:  # noqa: BLE001 - a provider must never fail the request
            # describe_source_error already names the source and maps the
            # exception to a user-facing reason, so the prefix is not repeated.
            reason = describe_source_error(source.name, exc)
            errors.append(f"元数据补全失败——{reason}，保留原记录。")
            continue
        if not payload:
            continue

        try:
            candidates = source.parse(payload)
        except Exception as exc:  # noqa: BLE001 - a shape change must not crash enrichment
            errors.append(f"{source.name} 返回的数据无法解析：{exc}，保留原记录。")
            continue

        verified = [c for c in candidates if _candidate_is_verified(query, c)]
        dois = {c.identity.doi for c in verified if c.identity.doi}
        if len(dois) > 1:
            notes.append(
                f"{source.name} 返回了多个可能的 DOI（{'、'.join(sorted(dois))}），"
                "无法确定哪一个属于该记录，已保留原记录。"
            )
            continue
        if not dois:
            continue

        return _apply_candidate(identity, verified[0], source.name, errors)

    if query.is_doi_lookup and not errors:
        errors.append("未能从元数据源补全该记录的 DOI 相关信息。")
    return EnrichmentResult(
        identity=identity,
        source="",
        filled_fields=(),
        errors=tuple(errors),
        notes=tuple(notes),
        ambiguous=bool(notes),
    )


def _apply_candidate(
    identity: PaperIdentity,
    candidate: MetadataCandidate,
    source_name: str,
    errors: list[str],
) -> EnrichmentResult:
    """Write the candidate's missing identifiers onto the identity.

    Only empty fields are filled. An existing value is never overwritten even
    when the provider disagrees — a record that already says 2015 is not made
    more true by agreeing with a provider, and a silent overwrite would destroy
    the distinction between "we looked this up" and "we already knew this".
    """
    filled: list[str] = []
    merged_values: dict[str, str] = {}
    for field in ("doi", "arxiv_id", "s2_paper_id", "openalex_id", "year", "first_author"):
        if getattr(identity, field):
            continue
        value = getattr(candidate.identity, field)
        if value:
            merged_values[field] = value
            filled.append(field)

    enriched = PaperIdentity(**{**identity.to_dict(), **merged_values})
    notes: list[str] = []
    if "doi" in filled and identity.doi == "":
        notes.append(f"已从{source_name}回查到 DOI：{enriched.doi}。")
    return EnrichmentResult(
        identity=enriched,
        source=source_name,
        filled_fields=tuple(filled),
        errors=tuple(errors),
        notes=tuple(notes),
        ambiguous=False,
    )
