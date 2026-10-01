"""Why is this paper being recommended? Five named signals, each with provenance.

A recommendation list that cannot answer "why this one?" is indistinguishable
from a random list, and the user's response to an indistinguishable list is to
stop reading it. So every row this module returns carries the signal that
actually produced it, and — the part that is easy to fake and is therefore the
part this module is built around — **a row with no identifiable signal is not
returned at all**. There is no ``reason="related"`` default, no fallback label
attached after the fact, and no ``else: reason = "主题相似"`` branch. A paper
reached by the topical search is topical; a paper reached by the citation graph
is cited-by; a paper we could not attribute to either is absent.

**The five signals, and where each one comes from.**

``cited_by`` (引用 / 参考文献)
    Works the seed paper *cites* — the papers already in its bibliography. A
    citation edge is a statement somebody made about the literature, authored
    by the seed paper's own authors, which is why it outranks anything derived
    from text similarity. The OpenAlex filter is ``cited_by:<seed>``: works
    cited *by* the given work. Direction is load-bearing, so it is written out
    in :data:`OPENALEX_CITED_BY_FILTER` rather than left to a reader who knows
    the API. S2's ``/references`` is the same relation and is available as a
    second provider.
``citing`` (被引)
    Works that cite the seed — follow-up work, which is the most useful
    recommendation direction available for a paper the user is reading. OpenAlex
    filter ``cites:<seed>``; S2's ``/citations`` is the same relation.
``related`` (相关)
    S2's recommendations endpoint, and OpenAlex's ``related_to`` filter. Weaker
    than a citation edge and honestly labelled as such: "the provider thinks
    these are related" is a real signal with a real reason, and it is not the
    same claim as "these cite each other".
``topical`` (主题相似)
    LLM query expansion followed by ordinary search. The reason names the
    *expanded* query that produced the hit, not the topic in the abstract, so a
    user can see that the list came from a query the model wrote.
``same_dataset`` (同数据集)
    Search on the dataset names the seed paper names. The hardest signal and
    the one most easily faked: topic similarity over a paper that happens to
    use ImageNet will surface plenty of papers that never touch ImageNet, and
    calling that "同数据集" is a lie with a Chinese label on it. So a hit earns
    this reason only when the dataset name it was searched for literally
    appears in the returned paper's own title or abstract — see
    :func:`dataset_name_appears`.

**Deduplication is the identity layer's job, and a paper reached by three
signals is one row with three reasons.** Every candidate is grouped through
:func:`paperwhisperer.search.identity.compare_identities`, and the collapse
threshold is imported from
:func:`paperwhisperer.search.normalize.COLLAPSE_CONFIDENCE` rather than
re-spelled, so there is one definition of "proven duplicate" in the codebase
and this module cannot drift from it. The T3.1 distinction is honoured as
written: ``exact`` collapses, ``probable`` does *not* — both rows stay visible
and the pair is reported in ``possible_duplicates`` for the user to confirm.
Accumulating reasons on top of a group is the only thing this module does that
:func:`paperwhisperer.search.normalize.deduplicate_papers` does not: that
function returns records, and a record has nowhere to put "three reasons" — but
every decision it makes is made here by calling the same two identity
functions, not by a second comparison implementation.

**Every network call is injected**, exactly as in
:mod:`paperwhisperer.search.enrich`: a source takes a ``fetch`` callable, tests
pass a lambda, production passes the builders at the bottom of this file which
wrap :mod:`paperwhisperer.search.http`. A second HTTP client is not built here.
The risk-table line "OpenAlex/Crossref/S2 限速 → 统一 http client 限速与缓存"
is a property of that shared module and is not solved by a second caller-side
throttle; what this module does do about rate limiting is bound the work per
request through :class:`CallBudget` and *say so* when the bound bites, which is
the failure a user can otherwise not see.

**A failed signal is not an empty signal.** ``status`` is four-valued —
``ok`` / ``empty`` / ``unavailable`` / ``skipped`` — because "the provider
returned nothing" and "we could not ask" are different claims about the world
and the UI must be able to tell them apart. A rate-limited OpenAlex produces
``unavailable`` plus a Chinese reason naming the provider, never an empty
``cited_by`` bucket that reads as "this paper has never been cited".
"""

from __future__ import annotations

import logging
import os
import re
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from paperwhisperer.core import config
from paperwhisperer.core.text import compact_text, parse_year
from paperwhisperer.search.http import describe_source_error
from paperwhisperer.search.identity import (
    PaperIdentity,
    compare_identities,
    identity_from_record,
    merge_identities,
)
from paperwhisperer.search.normalize import (
    COLLAPSE_CONFIDENCE,
    IDENTIFIER_FIELDS,
    normalize_paper_record,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CallBudget",
    "MAX_DATASET_QUERIES",
    "OPENALEX_CITED_BY_FILTER",
    "OPENALEX_CITES_FILTER",
    "OPENALEX_RELATED_FILTER",
    "RecommendationQuery",
    "RecommendationReason",
    "SignalResult",
    "SignalSource",
    "build_openalex_cited_by_source",
    "build_openalex_citing_source",
    "build_openalex_related_source",
    "build_s2_cited_by_source",
    "build_s2_citing_source",
    "build_s2_related_source",
    "build_same_dataset_source",
    "build_topical_source",
    "dataset_name_appears",
    "dataset_names_from_facts",
    "default_recommendation_sources",
    "recommend_papers",
]

# --------------------------------------------------------------------------- #
# The five signals
# --------------------------------------------------------------------------- #

SIGNAL_CITED_BY = "cited_by"
SIGNAL_CITING = "citing"
SIGNAL_RELATED = "related"
SIGNAL_TOPICAL = "topical"
SIGNAL_SAME_DATASET = "same_dataset"

#: The short label the UI puts on a filter chip or a badge. The
#: unambiguous full sentence lives on each :class:`RecommendationReason` as
#: ``reason_zh``, because a two-character chip next to a paper the user has
#: never heard of does not disambiguate who cited whom.
#:
#: These are the plan's two words (被引/引用) assigned by **their normal
#: meaning in Chinese academic usage**, not by the English signal id. That
#: matters, because the id and the word point in opposite directions here:
#: OpenAlex's ``cited_by`` filter returns the works the seed *cites* — the
#: seed's own bibliography, which in Chinese is 引用 / 参考文献. 被引 means
#: the opposite and is unarguable to a Chinese reader: it is the citation
#: count a paper *receives*. Labelling the bibliography 被引 puts a chip that
#: says, in effect, the opposite of the truth, on every row.
#:
#: The mapping was verified against the live API rather than assumed.
#: For a seed citing 44 works and receiving 1253:
#:   ``filter=cited_by:W`` -> OQL "works where it's cited by (W)" -> 44 rows,
#:   and those ids appear in W's own ``referenced_works``;
#:   ``filter=cites:W``     -> OQL "works where it cites (W)" -> 1253 rows.
#: So OpenAlex names both filters from the *citing* work's point of view,
#: which is exactly the thing that is easy to read backwards.
SIGNAL_LABELS_ZH: dict[str, str] = {
    # Works the seed cites — the seed's own bibliography.
    SIGNAL_CITED_BY: "引用",
    # Works that cite the seed.
    SIGNAL_CITING: "被引",
    SIGNAL_RELATED: "相关",
    SIGNAL_TOPICAL: "主题相似",
    SIGNAL_SAME_DATASET: "同数据集",
}

#: The order signals are attempted in, and the order the response lists them in.
#: Citation edges come before provider similarity, which comes before the two
#: signals that depend on something we had to generate (the expanded query, the
#: extracted dataset names) — the cheap, authored, structural signals are
#: answered before we spend the request's upstream-call budget on derived ones.
SIGNAL_ORDER: tuple[str, ...] = (
    SIGNAL_CITED_BY,
    SIGNAL_CITING,
    SIGNAL_RELATED,
    SIGNAL_TOPICAL,
    SIGNAL_SAME_DATASET,
)

# --------------------------------------------------------------------------- #
# Provider filter constants
#
# OpenAlex's filter names read from the *citing* work's point of view, which is
# the single easiest thing to get backwards in this file, and getting it
# backwards produces a confidently mislabelled reason rather than an error. So
# the direction each filter runs is spelled out on the constant.
# --------------------------------------------------------------------------- #

#: Works **cited by** the seed: the seed's own bibliography. OpenAlex names the
#: filter after the citing work, so ``cited_by:W1`` returns W1's references.
OPENALEX_CITED_BY_FILTER = "cited_by"
#: Works that **cite** the seed: follow-up work. ``cites:W1`` returns the works
#: that list W1 in their reference lists.
OPENALEX_CITES_FILTER = "cites"
#: Works the provider considers related to the seed, from OpenAlex's
#: co-occurrence-derived relatedness. Not a citation claim.
OPENALEX_RELATED_FILTER = "related_to"

#: How many dataset names one request may search. Two is enough to catch the
#: common "this paper evaluates on ImageNet and on a private corpus" shape
#: without turning one user click into a loop over a prose sentence.
MAX_DATASET_QUERIES = 2

#: Default ceiling on upstream calls for one recommendation request. Four of
#: the five signals cost at most one call, so four buys a complete answer in
#: the common case while making the worst case explicit. Overridable per
#: request via :class:`CallBudget`.
DEFAULT_MAX_UPSTREAM_CALLS = 4

# --------------------------------------------------------------------------- #
# The budget
# --------------------------------------------------------------------------- #


@dataclass
class CallBudget:
    """The ceiling on upstream calls one request may make.

    A recommendation page is one user click, and one click can otherwise reach
    four providers plus a search plus a search per dataset name. On a
    rate-limited API that is how one interaction becomes a 429 across every
    other feature on the site, which is the risk the plan's section 8 names.
    The bound is enforced here, in the engine, and it is *visible*: when it
    runs out, the signals that were dropped are reported as ``skipped`` with a
    reason, rather than silently missing.

    ``charge`` returns ``False`` instead of raising. A source that cannot
    afford its call should produce an explicit "not attempted" answer, and an
    exception is the wrong tool for a condition that is a normal outcome of a
    bounded request.
    """

    limit: int = DEFAULT_MAX_UPSTREAM_CALLS
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def charge(self, units: int = 1) -> bool:
        """Consume ``units`` calls, or refuse and leave the counter untouched."""
        if units > self.remaining:
            return False
        self.used += units
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"limit": self.limit, "used": self.used, "remaining": self.remaining}


# --------------------------------------------------------------------------- #
# The request
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecommendationQuery:
    """Everything a source is allowed to know about what to recommend for.

    ``identity`` is the seed paper — the one whose references, citations and
    datasets are being walked. ``expanded_query`` is the LLM's rewrite of the
    user's request; a source that uses it must say so in its reason, because a
    hit retrieved by a query the user never typed is the hardest kind of
    recommendation to explain and therefore the one that most needs explaining.

    ``datasets`` is a tuple of *dataset names*, not prose. A caller that has
    canonical names (a future T1.8 dataset registry) passes them and the
    ``same_dataset`` reasons are marked ``declared``. A caller that only has
    the fact layer's paraphrase passes
    :func:`dataset_names_from_facts` output and the reasons are marked
    ``derived`` — see that function for why the difference is recorded rather
    than papered over.
    """

    identity: PaperIdentity
    expanded_query: str = ""
    topics: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    dataset_names_are_declared: bool = False
    limit: int = 6

    @classmethod
    def from_record(cls, record: Mapping[str, Any] | None, **overrides: Any) -> RecommendationQuery:
        """Build a query from a canonical paper record dict."""
        base = {
            "identity": identity_from_record(dict(record or {})),
        }
        base.update(overrides)
        return cls(**base)

    @property
    def seed_title(self) -> str:
        return compact_text(self.identity.title, limit=160)

    @property
    def search_text(self) -> str:
        """What the topical source searches for.

        The expanded query when the LLM produced one, otherwise the seed's own
        title. Returning the title here rather than an empty string is
        deliberate but bounded: the source still records ``which query`` in its
        reason, so a title-only search says "按标题检索" and an LLM search says
        "按扩展检索式…检索", and the two are not presented as the same thing.
        """
        return compact_text(self.expanded_query, limit=240) or self.seed_title


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecommendationReason:
    """One signal, one provider, and the sentence the UI shows for it.

    ``detail`` is machine-readable provenance — the OpenAlex filter that ran,
    the query string, the dataset name that was searched — kept separate from
    ``reason_zh`` so a test can assert *which* mechanism fired without matching
    on prose. ``confidence`` is ``declared`` when the dataset name came from
    somewhere authoritative and ``derived`` when it was lexically scanned out of
    a paraphrase; the UI is expected to say which, because "this paper searches
    the same dataset name" and "this paper is on the same dataset" are
    different claims and only the first is currently provable.
    """

    signal: str
    source: str
    reason_zh: str
    detail: str = ""
    confidence: str = ""

    @property
    def label_zh(self) -> str:
        return SIGNAL_LABELS_ZH.get(self.signal, self.signal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "label_zh": self.label_zh,
            "source": self.source,
            "reason_zh": self.reason_zh,
            "detail": self.detail,
            "confidence": self.confidence,
        }


#: ``status`` values. Four because three of the four are needed and the fourth
#: is the one that gets confused with the others.
#:
#: ``ok``
#:     The provider was asked and returned at least one paper.
#: ``empty``
#:     The provider was asked and said it has nothing. This is a *fact about
#:     the literature* and is displayed as such.
#: ``unavailable``
#:     We could not get an answer: no identifier for that provider, a network
#:     error, a 429, or a payload we could not parse. The reason says which.
#:     Never rendered as "no such papers".
#: ``skipped``
#:     We chose not to ask, because the request's upstream-call budget was
#:     spent. Says so, so the user knows a smaller list was a budget decision.
STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_UNAVAILABLE = "unavailable"
STATUS_SKIPPED = "skipped"


@dataclass(frozen=True)
class SignalHit:
    """One candidate paper, with the reason it was fetched and nothing yet
    decided about whether it survives deduplication."""

    record: dict[str, Any]
    identity: PaperIdentity
    reason: RecommendationReason


@dataclass(frozen=True)
class SignalResult:
    """What one source did, in full: the hits, the status, and the reason.

    A source that raised still returns a :class:`SignalResult`; the engine never
    sees an exception from a source, because a provider being down must not take
    down the four signals that did answer.
    """

    signal: str
    source: str
    status: str
    hits: tuple[SignalHit, ...] = ()
    reason_zh: str = ""

    @property
    def label_zh(self) -> str:
        return SIGNAL_LABELS_ZH.get(self.signal, self.signal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "label_zh": self.label_zh,
            "source": self.source,
            "status": self.status,
            "count": len(self.hits),
            "reason_zh": self.reason_zh,
        }


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SignalSource:
    """A named way of finding papers, with its result parsing kept separate.

    ``fetch`` takes a :class:`RecommendationQuery` and returns a
    :class:`SignalResult`. It is *allowed* to raise; the engine wraps it. That
    split is what lets a source be tested by passing a lambda that returns a
    hand-written provider payload — the same injection contract
    :mod:`paperwhisperer.search.enrich` uses — while the production builders at
    the bottom of this file stay one line of wiring to the shared HTTP client.
    """

    name: str
    signal: str
    fetch: Callable[[RecommendationQuery], SignalResult]


# --------------------------------------------------------------------------- #
# Payload parsing (pure functions)
# --------------------------------------------------------------------------- #


S2_SOURCE_NAME = "Semantic Scholar"
OPENALEX_SOURCE_NAME = "OpenAlex"

#: Fields requested from S2. ``externalIds`` is the whole point: without it S2
#: sends no DOI and the hit cannot be deduplicated against anything.
_S2_FIELDS = "title,abstract,year,venue,url,authors,externalIds,paperId,openAccessPdf"

_OPENALEX_SELECT = "id,doi,display_name,publication_year,authorships,primary_location"


def _s2_record(item: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one S2 item onto the app's canonical paper shape.

    Routed through :func:`normalize_paper_record` rather than hand-built,
    because that is where the ``externalIds`` → ``doi``/``arxiv_id`` reading
    lives, and a hand-built record here would reintroduce exactly the DOI loss
    T3.1 fixed.
    """
    return normalize_paper_record(S2_SOURCE_NAME, dict(item or {}))


def _openalex_record(item: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one OpenAlex work onto the app's canonical paper shape.

    OpenAlex spells the fields differently from every other provider this
    codebase consumes (``display_name``, ``publication_year``, ``authorships``),
    so the flattening happens here and *only* here. The identifiers are already
    in shapes :func:`paperwhisperer.search.identity` understands: ``doi`` is a
    full ``https://doi.org/`` URL and ``id`` is a full
    ``https://openalex.org/W…`` URL, both of which the identity layer's
    normalizers strip back to bare values.
    """
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    authors = [
        str((authorship.get("author") or {}).get("display_name") or "").strip()
        for authorship in item.get("authorships") or []
    ]
    flat = {
        "id": item.get("id") or "",
        "doi": item.get("doi") or "",
        "title": item.get("title") or item.get("display_name") or "",
        "abstract": item.get("abstract") or "",
        "year": item.get("publication_year") or item.get("year") or "",
        "authors": [author for author in authors if author],
        "venue": source.get("display_name") or "OpenAlex",
        "url": location.get("landing_page_url") or item.get("id") or "",
        "pdf_url": location.get("pdf_url") or "",
    }
    return normalize_paper_record(OPENALEX_SOURCE_NAME, flat)


def _signal_result_from_items(
    signal: str,
    source: str,
    items: Sequence[Mapping[str, Any]],
    reason_zh: str,
    detail: str,
) -> SignalResult:
    """Wrap parsed provider rows into a :class:`SignalResult`.

    The empty case gets a *positive* reason string, not a blank one: "S2 said
    it has no follow-up work for this paper" is a statement, and it is a
    different statement from the empty ``reason_zh`` a failure would produce.
    """
    hits = [
        SignalHit(
            record=record,
            identity=identity_from_record(record),
            reason=RecommendationReason(
                signal=signal,
                source=source,
                reason_zh=reason_zh,
                detail=detail,
            ),
        )
        for record in items
        if str(record.get("title") or "").strip()
    ]
    if not hits:
        return SignalResult(
            signal=signal,
            source=source,
            status=STATUS_EMPTY,
            reason_zh=f"{source} 没有返回与该论文相关的记录。",
        )
    return SignalResult(
        signal=signal,
        source=source,
        status=STATUS_OK,
        hits=tuple(hits),
        reason_zh=reason_zh,
    )


def parse_s2_recommendations(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """S2 recommendations: ``{"recommendedPapers": [...]}``.

    ``data`` is accepted as an alias because S2 has shipped both spellings
    across endpoint versions, and a payload shape change must degrade to an
    empty signal rather than to a crash on a list index.
    """
    payload = payload or {}
    items = payload.get("recommendedPapers")
    if items is None:
        items = payload.get("data") or []
    return [_s2_record(item) for item in items if isinstance(item, Mapping)]


def parse_s2_citations(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """S2 citations: ``{"data": [{"citingPaper": {...}}]}``."""
    rows = (payload or {}).get("data") or []
    items = [row.get("citingPaper") for row in rows if isinstance(row, Mapping)]
    return [_s2_record(item) for item in items if isinstance(item, Mapping)]


def parse_s2_references(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """S2 references: ``{"data": [{"citedPaper": {...}}]}``."""
    rows = (payload or {}).get("data") or []
    items = [row.get("citedPaper") for row in rows if isinstance(row, Mapping)]
    return [_s2_record(item) for item in items if isinstance(item, Mapping)]


def parse_openalex_works(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """OpenAlex: ``{"results": [...]}``, with a bare single work tolerated."""
    payload = payload or {}
    results = payload.get("results")
    if results is None:
        results = [payload] if payload.get("id") else []
    return [_openalex_record(item) for item in results if isinstance(item, Mapping)]


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


def unavailable(signal: str, source: str, reason_zh: str) -> SignalResult:
    """A signal that could not be produced, with the reason spelled out."""
    return SignalResult(
        signal=signal, source=source, status=STATUS_UNAVAILABLE, reason_zh=reason_zh
    )


def skipped(signal: str, source: str, reason_zh: str) -> SignalResult:
    """A signal that was deliberately not attempted."""
    return SignalResult(signal=signal, source=source, status=STATUS_SKIPPED, reason_zh=reason_zh)


def failed(signal: str, source: str, exc: Exception) -> SignalResult:
    """Convert a provider exception into an explicit ``unavailable``.

    :func:`paperwhisperer.search.http.describe_source_error` already maps a 429
    to a rate-limit message and a certificate failure to a TLS message, so the
    Chinese sentence here does not have to re-derive what went wrong — it only
    has to say that this signal is missing rather than empty.
    """
    return unavailable(signal, source, f"{describe_source_error(source, exc)}，该信号本次不可用。")


# --------------------------------------------------------------------------- #
# The default transport
# --------------------------------------------------------------------------- #


def _default_get_json(url: str, timeout: int, headers: dict, retries: int):
    """The one outbound HTTP path, imported lazily from the shared client.

    Copied in structure from :func:`paperwhisperer.search.enrich._default_get_json`
    on purpose. ``search/http.py`` owns the SSL context and the 429-aware retry
    backoff; opening a connection here would mean two places to fix when a
    provider changes its rate-limit behaviour. What that shared client is still
    missing for this workload — per-host rate limiting and a response cache —
    is listed in the task report rather than worked around locally.
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


#: OpenAlex's polite-pool contact address, when one is configured.
#:
#: Read from the same ``ENRICHMENT_CONTACT_EMAIL`` as
#: :mod:`paperwhisperer.search.enrich` rather than from a new variable, so an
#: operator who configured it for metadata recovery also gets the slower,
#: larger pool here. That is the cheapest rate-limit mitigation available and
#: it costs one configuration key rather than code.
def _contact_email() -> str:
    return str(getattr(config, "ENRICHMENT_CONTACT_EMAIL", "") or "").strip()


def _provider_int(name: str, default: int) -> int:
    """Read a positive integer for a provider call, ignoring junk.

    The same deliberate duplication as
    :func:`paperwhisperer.search.enrich._enrichment_int`, for the same reason:
    ``core/config.py`` is owned elsewhere and adding keys to it is not this
    task's call to make, so the defaults are spelled out here and flagged for
    consolidation alongside the enrichment ones.
    """
    try:
        return max(1, int(os.getenv(name, "").strip() or default))
    except (TypeError, ValueError):
        return default


#: Timeout and retry ceiling for the OpenAlex graph calls. OpenAlex's own
#: guidance is far more patient than the search endpoints because these are
#: single lookups by id, not ranked searches, so one retry is enough to ride out
#: a transient blip without turning one click into four sequential waits.
RECOMMEND_OPENALEX_TIMEOUT = _provider_int("RECOMMEND_OPENALEX_TIMEOUT", 15)
RECOMMEND_OPENALEX_MAX_RETRIES = _provider_int("RECOMMEND_OPENALEX_MAX_RETRIES", 2)


def _s2_headers() -> dict[str, str]:
    headers = {"User-Agent": config.APP_USER_AGENT}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY
    return headers


# --------------------------------------------------------------------------- #
# Graph sources: S2
# --------------------------------------------------------------------------- #


def _s2_url(paper_id: str, endpoint: str, **params: Any) -> str:
    """Build one S2 graph URL: ``/graph/v1/paper/{id}/{endpoint}``.

    The id is quoted and the ``/`` is *not* part of what gets quoted — building
    the path by quoting ``"{id}/{endpoint}"`` as a single unit produces
    ``.../649def34...%2Freferences``, which S2 does not route and which is
    invisible unless a test looks at the URL. The endpoint is appended after the
    id is encoded, for exactly that reason.
    """
    query = {"fields": _S2_FIELDS, **params}
    return (
        "https://api.semanticscholar.org/graph/v1/paper/"
        f"{urllib.parse.quote(paper_id, safe='')}/{endpoint}"
        f"?{urllib.parse.urlencode(query)}"
    )


def _s2_result(
    query: RecommendationQuery,
    signal: str,
    identity_field: str,
    missing_zh: str,
    url: str,
    parse: Callable[[Mapping[str, Any]], list[dict[str, Any]]],
    reason_zh: str,
    detail: str,
    get_json: Callable[..., Mapping[str, Any]],
    budget: CallBudget,
) -> SignalResult:
    """Shared body of the three S2 graph sources.

    A source whose seed has no S2 id is ``unavailable``, not ``empty``: the
    difference between "S2 has no recommendations for this paper" and "we
    cannot address this paper to S2 at all" is the difference between a fact
    about the literature and an absence of work, and only the first one may be
    displayed as an absence of results.
    """
    paper_id = str(getattr(query.identity, identity_field, "") or "").strip()
    if not paper_id:
        return unavailable(signal, S2_SOURCE_NAME, missing_zh)
    if not budget.charge():
        return skipped(
            signal,
            S2_SOURCE_NAME,
            f"本次请求的上游调用预算已用尽（上限 {budget.limit} 次），该信号未执行。",
        )
    try:
        payload = get_json(
            url,
            timeout=config.SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
            headers=_s2_headers(),
            retries=config.SEMANTIC_SCHOLAR_MAX_RETRIES,
        )
        return _signal_result_from_items(signal, S2_SOURCE_NAME, parse(payload), reason_zh, detail)
    except Exception as exc:  # noqa: BLE001 - a provider must never fail the request
        return failed(signal, S2_SOURCE_NAME, exc)


def build_s2_cited_by_source(
    fetch: Callable[..., Mapping[str, Any]] | None = None,
    *,
    budget: CallBudget | None = None,
) -> SignalSource:
    """S2 ``/references`` — the papers the seed paper itself cites (被引)."""
    get_json = fetch or _default_get_json
    call_budget = budget or CallBudget()

    def _fetch(query: RecommendationQuery) -> SignalResult:
        return _s2_result(
            query,
            SIGNAL_CITED_BY,
            "s2_paper_id",
            "当前论文没有 Semantic Scholar 编号，无法查询它引用的论文。",
            _s2_url(query.identity.s2_paper_id, "references", limit=query.limit),
            parse_s2_references,
            f"Semantic Scholar 的参考文献关系：这篇论文被《{query.seed_title}》引用。",
            "s2:/references",
            get_json,
            call_budget,
        )

    return SignalSource(name=S2_SOURCE_NAME, signal=SIGNAL_CITED_BY, fetch=_fetch)


def build_s2_citing_source(
    fetch: Callable[..., Mapping[str, Any]] | None = None,
    *,
    budget: CallBudget | None = None,
) -> SignalSource:
    """S2 ``/citations`` — the papers that cite the seed paper (引用)."""
    get_json = fetch or _default_get_json
    call_budget = budget or CallBudget()

    def _fetch(query: RecommendationQuery) -> SignalResult:
        return _s2_result(
            query,
            SIGNAL_CITING,
            "s2_paper_id",
            "当前论文没有 Semantic Scholar 编号，无法查询引用了它的论文。",
            _s2_url(query.identity.s2_paper_id, "citations", limit=query.limit),
            parse_s2_citations,
            f"Semantic Scholar 的引用关系：这篇论文引用了《{query.seed_title}》。",
            "s2:/citations",
            get_json,
            call_budget,
        )

    return SignalSource(name=S2_SOURCE_NAME, signal=SIGNAL_CITING, fetch=_fetch)


def build_s2_related_source(
    fetch: Callable[..., Mapping[str, Any]] | None = None,
    *,
    budget: CallBudget | None = None,
) -> SignalSource:
    """S2 ``/recommendations`` — the provider's own notion of related (相关)."""
    get_json = fetch or _default_get_json
    call_budget = budget or CallBudget()

    def _fetch(query: RecommendationQuery) -> SignalResult:
        paper_id = str(query.identity.s2_paper_id or "").strip()
        if not paper_id:
            return unavailable(
                SIGNAL_RELATED,
                S2_SOURCE_NAME,
                "当前论文没有 Semantic Scholar 编号，无法获取它的推荐论文。",
            )
        if not call_budget.charge():
            return skipped(
                SIGNAL_RELATED,
                S2_SOURCE_NAME,
                f"本次请求的上游调用预算已用尽（上限 {call_budget.limit} 次），该信号未执行。",
            )
        url = (
            "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/"
            f"{urllib.parse.quote(paper_id, safe='')}"
            f"?{urllib.parse.urlencode({'fields': _S2_FIELDS, 'limit': query.limit})}"
        )
        try:
            payload = get_json(
                url,
                timeout=config.SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
                headers=_s2_headers(),
                retries=config.SEMANTIC_SCHOLAR_MAX_RETRIES,
            )
            return _signal_result_from_items(
                SIGNAL_RELATED,
                S2_SOURCE_NAME,
                parse_s2_recommendations(payload),
                "Semantic Scholar 的推荐结果：该提供方认为这篇论文与《"
                f"{query.seed_title}》相关（不是引用关系）。",
                "s2:/recommendations",
            )
        except Exception as exc:  # noqa: BLE001
            return failed(SIGNAL_RELATED, S2_SOURCE_NAME, exc)

    return SignalSource(name=S2_SOURCE_NAME, signal=SIGNAL_RELATED, fetch=_fetch)


# --------------------------------------------------------------------------- #
# Graph sources: OpenAlex
# --------------------------------------------------------------------------- #


def _openalex_url(filter_name: str, work_id: str, limit: int) -> str:
    """Build the OpenAlex works URL for one graph filter.

    ``mailto`` is always present, empty when unconfigured: OpenAlex keys its
    polite pool on the contact address and serves the anonymous pool to
    everything else, so sending the key unconditionally is what makes the pool
    decision the operator's rather than the absence of one variable's.
    """
    params = {
        "filter": f"{filter_name}:{urllib.parse.quote(work_id, safe=':/')}",
        "per-page": str(limit),
        "select": _OPENALEX_SELECT,
        "mailto": _contact_email(),
    }
    # safe=":" keeps the filter's own separator literal. Percent-encoding it is
    # equally valid HTTP and much harder to read in a log line or a test
    # assertion, and this string is where a citation direction is written down.
    return f"{config.OPENALEX_API_URL}?{urllib.parse.urlencode(params, safe=':')}"


def _openalex_result(
    query: RecommendationQuery,
    signal: str,
    filter_name: str,
    missing_zh: str,
    reason_zh: str,
    get_json: Callable[..., Mapping[str, Any]],
    budget: CallBudget,
) -> SignalResult:
    work_id = str(query.identity.openalex_id or "").strip()
    if not work_id:
        return unavailable(signal, OPENALEX_SOURCE_NAME, missing_zh)
    if not budget.charge():
        return skipped(
            signal,
            OPENALEX_SOURCE_NAME,
            f"本次请求的上游调用预算已用尽（上限 {budget.limit} 次），该信号未执行。",
        )
    try:
        payload = get_json(
            _openalex_url(filter_name, work_id, query.limit),
            timeout=RECOMMEND_OPENALEX_TIMEOUT,
            headers={"User-Agent": config.APP_USER_AGENT},
            retries=RECOMMEND_OPENALEX_MAX_RETRIES,
        )
        return _signal_result_from_items(
            signal,
            OPENALEX_SOURCE_NAME,
            parse_openalex_works(payload),
            reason_zh,
            f"openalex:filter={filter_name}",
        )
    except Exception as exc:  # noqa: BLE001
        return failed(signal, OPENALEX_SOURCE_NAME, exc)


def build_openalex_cited_by_source(
    fetch: Callable[..., Mapping[str, Any]] | None = None,
    *,
    budget: CallBudget | None = None,
) -> SignalSource:
    """OpenAlex ``cited_by:<seed>`` — the seed's own bibliography (被引)."""
    get_json = fetch or _default_get_json
    call_budget = budget or CallBudget()

    def _fetch(query: RecommendationQuery) -> SignalResult:
        return _openalex_result(
            query,
            SIGNAL_CITED_BY,
            OPENALEX_CITED_BY_FILTER,
            "当前论文没有 OpenAlex 编号，无法查询它引用的论文。",
            f"OpenAlex 的引用关系：这篇论文被《{query.seed_title}》引用。",
            get_json,
            call_budget,
        )

    return SignalSource(
        name=OPENALEX_SOURCE_NAME, signal=SIGNAL_CITED_BY, fetch=_fetch
    )


def build_openalex_citing_source(
    fetch: Callable[..., Mapping[str, Any]] | None = None,
    *,
    budget: CallBudget | None = None,
) -> SignalSource:
    """OpenAlex ``cites:<seed>`` — follow-up work that cites the seed (引用)."""
    get_json = fetch or _default_get_json
    call_budget = budget or CallBudget()

    def _fetch(query: RecommendationQuery) -> SignalResult:
        return _openalex_result(
            query,
            SIGNAL_CITING,
            OPENALEX_CITES_FILTER,
            "当前论文没有 OpenAlex 编号，无法查询引用了它的论文。",
            f"OpenAlex 的引用关系：这篇论文引用了《{query.seed_title}》。",
            get_json,
            call_budget,
        )

    return SignalSource(
        name=OPENALEX_SOURCE_NAME, signal=SIGNAL_CITING, fetch=_fetch
    )


def build_openalex_related_source(
    fetch: Callable[..., Mapping[str, Any]] | None = None,
    *,
    budget: CallBudget | None = None,
) -> SignalSource:
    """OpenAlex ``related_to:<seed>`` — provider-declared relatedness (相关)."""
    get_json = fetch or _default_get_json
    call_budget = budget or CallBudget()

    def _fetch(query: RecommendationQuery) -> SignalResult:
        return _openalex_result(
            query,
            SIGNAL_RELATED,
            OPENALEX_RELATED_FILTER,
            "当前论文没有 OpenAlex 编号，无法获取它的相关论文。",
            f"OpenAlex 的相关论文关系：该提供方认为这篇论文与《{query.seed_title}》相关"
            "（不是引用关系）。",
            get_json,
            call_budget,
        )

    return SignalSource(
        name=OPENALEX_SOURCE_NAME, signal=SIGNAL_RELATED, fetch=_fetch
    )


# --------------------------------------------------------------------------- #
# The two generated signals
# --------------------------------------------------------------------------- #


def _search_hits(
    signal: str,
    source: str,
    search_result: Mapping[str, Any],
    reason_zh: str,
    detail: str,
    confidence: str = "",
) -> SignalResult:
    """Turn an injected ``search_papers``-shaped result into a signal result.

    A search that returned an error *and* no items is ``unavailable``, not
    ``empty``. Both providers in :mod:`paperwhisperer.search.papers` being
    rate-limited and "no paper on this topic" are the same empty list, and only
    one of them is a fact about the literature.
    """
    items = [item for item in (search_result.get("items") or []) if isinstance(item, Mapping)]
    errors = [str(error) for error in (search_result.get("errors") or []) if str(error).strip()]
    if not items:
        if errors:
            return unavailable(signal, source, f"检索失败——{'；'.join(errors)}，该信号本次不可用。")
        return SignalResult(
            signal=signal, source=source, status=STATUS_EMPTY, reason_zh=f"{source} 没有命中记录。"
        )
    hits = []
    for record in items:
        normalized = dict(record)
        if not str(normalized.get("title") or "").strip():
            continue
        hits.append(
            SignalHit(
                record=normalized,
                identity=identity_from_record(normalized),
                reason=RecommendationReason(
                    signal=signal,
                    source=source,
                    reason_zh=reason_zh,
                    detail=detail,
                    confidence=confidence,
                ),
            )
        )
    if not hits:
        return SignalResult(
            signal=signal, source=source, status=STATUS_EMPTY, reason_zh=f"{source} 没有命中记录。"
        )
    return SignalResult(
        signal=signal, source=source, status=STATUS_OK, hits=tuple(hits), reason_zh=reason_zh
    )


def build_topical_source(
    search_fn: Callable[[str, int], Mapping[str, Any]],
    *,
    budget: CallBudget | None = None,
    source: str = "文献检索",
) -> SignalSource:
    """LLM query expansion followed by ordinary search (主题相似).

    ``search_fn`` is the same ``(query, limit) -> dict`` seam
    :meth:`paperwhisperer.analysis.orchestrator.DocumentAnalyzer.recommend_papers`
    already injects, so production passes
    :func:`paperwhisperer.search.papers.search_papers` and tests pass a lambda
    with a hand-written payload. The expanded query is written into
    :attr:`RecommendationReason.detail`, because a hit retrieved by a query the
    user never typed is the recommendation hardest to defend and therefore the
    one that must show its work.
    """
    call_budget = budget or CallBudget()

    def _fetch(query: RecommendationQuery) -> SignalResult:
        search_text = query.search_text
        if not search_text:
            return unavailable(
                SIGNAL_TOPICAL,
                source,
                "既没有可用的扩展检索式，也没有论文标题可用于检索，主题相似信号无法执行。",
            )
        if not call_budget.charge():
            return skipped(
                SIGNAL_TOPICAL,
                source,
                f"本次请求的上游调用预算已用尽（上限 {call_budget.limit} 次），该信号未执行。",
            )
        try:
            result = search_fn(search_text, query.limit)
        except Exception as exc:  # noqa: BLE001
            return failed(SIGNAL_TOPICAL, source, exc)
        origin = "扩展检索式" if query.expanded_query else "论文标题"
        topics = "、".join(query.topics) if query.topics else ""
        suffix = f"识别到的主题：{topics}。" if topics else ""
        return _search_hits(
            SIGNAL_TOPICAL,
            source,
            result,
            f"按{origin}「{search_text}」检索到的主题相近论文。{suffix}",
            f"query:{search_text}",
        )

    return SignalSource(name=source, signal=SIGNAL_TOPICAL, fetch=_fetch)


#: A name-shaped token: a Latin-script run that may contain ``.``/``-``/``_``/
#: ``+`` separators, because that is how dataset names are actually spelled
#: (``CIFAR-10``, ``WMT14``, ``MultiNLI``). Anchored on a letter so a bare
#: number is not a name.
_NAME_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[.\-_+][A-Za-z0-9]+)*")

#: Capitalized words that are not datasets and would otherwise pass the
#: name-shape test, because the fact layer's ``datasets`` value is prose written
#: by a model and prose is full of sentence-initial capitals. Kept short and
#: English-only on purpose: an attempt to enumerate method names here would be
#: a vocabulary project, and the honest fix is the dataset registry named in
#: :func:`dataset_names_from_facts`.
_NAME_STOPWORDS = frozenset(
    {
        "Ablation",
        "Attention",
        "Appendix",
        "Background",
        "Baseline",
        "Batch",
        "Both",
        "CNN",
        "Conclusion",
        "Convolutional",
        "Data",
        "Dataset",
        "Datasets",
        "Experiments",
        "Figure",
        "Framework",
        "Introduction",
        "Layer",
        "Learning",
        "Loss",
        "Model",
        "Models",
        "Network",
        "Results",
        "Section",
        "Table",
        "Task",
        "Transformer",
        "Training",
        "We",
    }
)


def _looks_like_dataset_name(token: str) -> bool:
    """Whether a token is name-shaped enough to be worth searching for.

    Three accepted shapes, because dataset names in the wild use all three and
    the alternatives each lose a real one: an interior capital (``ImageNet``,
    ``SQuAD``, ``MultiNLI``), an all-caps run (``COCO``, ``GLUE``, ``SST-2``),
    or a digit anywhere (``WMT14``, ``CIFAR-10``).

    The obvious looser rule — "any token that starts with a capital" — is
    rejected because the fact layer's value is a Chinese paraphrase whose
    English fragments are sentence fragments, and every one of them starts with
    a capital. That rule searches for "We" and "Both".
    """
    if len(token) < 3 or token in _NAME_STOPWORDS:
        return False
    if any(char.isdigit() for char in token):
        return True
    if token.isupper():
        return True
    return any(char.isupper() for char in token[1:])


def dataset_names_from_facts(facts: Any) -> tuple[str, ...]:
    """Best-effort dataset names out of a T1.8 :class:`PaperFacts` fact layer.

    **This is the weak link in the same-dataset signal and is why the reasons
    it produces are marked ``derived``.** ``PaperFacts.datasets`` holds model
    prose in the paper's own language — "本文在 ImageNet 与 COCO 上进行评估" —
    not a list of dataset identifiers, because the fact layer's job is to
    answer "what does this paper say about its datasets", and a list of names
    would not answer that. The names therefore have to be scanned out of the
    sentence, and the scan is lexical: it finds ``ImageNet`` and misses a
    dataset the paper names only in Chinese, and it can admit a method name
    that happens to be capitalized. Both errors are *over-generation* or
    *missed matches*, both of which are visible in the reason string, and
    neither of which is a wrong merge — a dataset-named hit still has to pass
    :func:`dataset_name_appears` to earn the reason at all.

    The fix is a dataset registry with canonical names and aliases, which is a
    T1.8 follow-up and not something to be faked here. Until it exists the
    caller is expected to pass ``dataset_names_are_declared=True`` when it has
    authoritative names, and the difference is recorded in
    :attr:`RecommendationReason.confidence` so the UI can say "同名数据集"
    instead of "同数据集" when that is all we proved.

    Accepts a :class:`~paperwhisperer.ingestion.schema.PaperFacts`, its
    ``to_dict()`` payload, a bare list of ``Fact``s, or ``None``, because the
    two call sites in this codebase (the fact layer and the persisted session)
    hold different shapes and neither of them is wrong.
    """
    raw = _dataset_fact_values(facts)
    found: list[str] = []
    for value in raw:
        for token in _NAME_TOKEN.findall(value):
            if not _looks_like_dataset_name(token):
                continue
            if token not in found:
                found.append(token)
    return tuple(found)


def _dataset_fact_values(facts: Any) -> list[str]:
    """Pull the non-empty ``datasets`` fact values out of any accepted shape."""
    if facts is None:
        return []
    entries: Sequence[Any]
    if isinstance(facts, Mapping):
        entries = facts.get("datasets") or []
    elif isinstance(facts, (list, tuple)):
        entries = facts
    else:
        entries = list(getattr(facts, "datasets", []) or [])

    values: list[str] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            value = str(entry.get("value") or "").strip()
            status = str(entry.get("extraction_status") or "not_found")
        else:
            value = str(getattr(entry, "value", "") or "").strip()
            status = str(getattr(entry, "extraction_status", "not_found"))
        # "论文未说明" and its unverified-prefixed form are answers about the
        # paper, not dataset names, and searching for either finds nothing.
        if not value or "论文未说明" in value or "未通过证据校验" in value:
            continue
        if status == "not_found":
            continue
        values.append(value)
    return values


def dataset_name_appears(dataset: str, record: Mapping[str, Any]) -> bool:
    """Whether ``dataset`` is literally named in the candidate's own metadata.

    The gate on the same-dataset signal, and the reason this signal is not just
    topical similarity wearing a nicer label. A dataset search returns papers
    *about* a dataset, which is a much larger set than the papers that *evaluate
    on* it; requiring the name to appear in the returned record's title or
    abstract is a cheap, checkable filter that removes most of the false
    positives, and it is checkable — a reviewer can open the abstract and see
    the name.

    The honest limit: an abstract that says "unlike prior work on ImageNet" is
    a mention, not a use, and this function cannot tell the difference. That is
    why the reason says the name was *检索到* (retrieved) rather than asserting
    the paper trains on it, and why ``confidence`` is ``derived`` or
    ``declared`` rather than ``verified``.
    """
    needle = str(dataset or "").strip().casefold()
    if not needle:
        return False
    haystack = " ".join(
        str(record.get(field) or "") for field in ("title", "abstract", "venue")
    ).casefold()
    return needle in haystack


def build_same_dataset_source(
    search_fn: Callable[[str, int], Mapping[str, Any]],
    *,
    budget: CallBudget | None = None,
    source: str = "数据集检索",
) -> SignalSource:
    """Search the seed paper's named datasets, and gate every hit (同数据集).

    Two differences from :func:`build_topical_source`, both load-bearing:

    1. One query per dataset name rather than one combined query, so the reason
       can name the dataset that actually found the paper.
    2. Every hit goes through :func:`dataset_name_appears`. A candidate the
       search returned but whose own metadata does not name the dataset is
       **dropped**, not shown with a weakened label. Returning it under
       ``topical`` instead would be defensible; returning it under
       ``same_dataset`` would be the lie this module exists to avoid.
    """
    call_budget = budget or CallBudget()

    def _fetch(query: RecommendationQuery) -> SignalResult:
        if not query.datasets:
            return unavailable(
                SIGNAL_SAME_DATASET,
                source,
                "当前论文没有可用的数据集名称（事实层未抽取或文档未持久化 paper_facts），"
                "同数据集信号无法执行；本模块不会用主题相似冒充同数据集。",
            )
        names = query.datasets[:MAX_DATASET_QUERIES]
        hits: list[SignalHit] = []
        errors: list[str] = []
        searched: list[str] = []
        confidence = "declared" if query.dataset_names_are_declared else "derived"
        for dataset in names:
            if not call_budget.charge():
                return skipped(
                    SIGNAL_SAME_DATASET,
                    source,
                    f"本次请求的上游调用预算已用尽（上限 {call_budget.limit} 次），"
                    "该信号未执行。",
                )
            try:
                result = search_fn(dataset, query.limit)
            except Exception as exc:  # noqa: BLE001
                errors.append(describe_source_error(source, exc))
                continue
            searched.append(dataset)
            for record in result.get("items") or []:
                if not isinstance(record, Mapping) or not str(record.get("title") or "").strip():
                    continue
                if not dataset_name_appears(dataset, record):
                    continue
                normalized = dict(record)
                hits.append(
                    SignalHit(
                        record=normalized,
                        identity=identity_from_record(normalized),
                        reason=RecommendationReason(
                            signal=SIGNAL_SAME_DATASET,
                            source=source,
                            reason_zh=(
                                f"检索数据集「{dataset}」命中，且该数据集出现在论文的"
                                "标题或摘要中。"
                            ),
                            detail=f"dataset:{dataset}",
                            confidence=confidence,
                        ),
                    )
                )
        if not hits:
            if not searched:
                return unavailable(
                    SIGNAL_SAME_DATASET,
                    source,
                    f"按数据集检索失败——{'；'.join(errors) or '未知原因'}，该信号本次不可用。",
                )
            rejected = ""
            if errors:
                rejected = f"（{'；'.join(errors)} 的检索结果未计入）"
            return SignalResult(
                signal=SIGNAL_SAME_DATASET,
                source=source,
                status=STATUS_EMPTY,
                reason_zh=(
                    f"按数据集检索「{'、'.join(searched)}」没有返回在标题或摘要中"
                    f"提到该数据集的论文{rejected}。"
                ),
            )
        return SignalResult(
            signal=SIGNAL_SAME_DATASET,
            source=source,
            status=STATUS_OK,
            hits=tuple(hits),
            reason_zh=f"按数据集检索「{'、'.join(searched)}」命中。",
        )

    return SignalSource(name=source, signal=SIGNAL_SAME_DATASET, fetch=_fetch)


# --------------------------------------------------------------------------- #
# The default source set
# --------------------------------------------------------------------------- #


def default_recommendation_sources(
    *,
    budget: CallBudget | None = None,
    search_fn: Callable[[str, int], Mapping[str, Any]] | None = None,
    fetch: Callable[..., Mapping[str, Any]] | None = None,
) -> list[SignalSource]:
    """The production signal set, sharing one budget.

    Two providers per graph signal rather than one, because a seed paper that
    has an arXiv id and a DOI but no S2 id and no OpenAlex id is the *normal*
    case for a freshly uploaded preprint, and a set that only asked S2 would
    report "被引: 不可用" for a paper that OpenAlex knows perfectly well. Each
    source checks for the identifier it needs and steps aside with a reason if
    the seed lacks it, so the graph signals cost at most two calls between them
    and usually one.

    ``search_fn`` is required for the two generated signals and defaults to
    :func:`paperwhisperer.search.papers.search_papers`; the import is lazy so
    that importing this module does not drag in the search stack.
    """
    call_budget = budget or CallBudget()
    if search_fn is None:
        from paperwhisperer.search.papers import search_papers as search_fn

    return [
        build_openalex_cited_by_source(fetch, budget=call_budget),
        build_s2_cited_by_source(fetch, budget=call_budget),
        build_openalex_citing_source(fetch, budget=call_budget),
        build_s2_citing_source(fetch, budget=call_budget),
        build_s2_related_source(fetch, budget=call_budget),
        build_openalex_related_source(fetch, budget=call_budget),
        build_topical_source(search_fn, budget=call_budget),
        build_same_dataset_source(search_fn, budget=call_budget),
    ]


# --------------------------------------------------------------------------- #
# Grouping: one paper, many reasons
# --------------------------------------------------------------------------- #


@dataclass
class _Group:
    """A candidate paper and every signal that reached it, plus the duplicates
    it might be but is not yet proven to be."""

    record: dict[str, Any]
    identity: PaperIdentity
    reasons: list[RecommendationReason]
    possible_duplicates: list[dict[str, Any]]


def _reason_key(reason: RecommendationReason) -> tuple[str, str, str]:
    """Identity of a reason for de-duplication within one group.

    The *signal* and *source* and *detail* together, not the sentence: two
    providers that fired the same filter with the same query value are two
    pieces of provenance and both are worth showing, while the same provider
    being asked twice for the same thing is one. Chinese prose is never part of
    the key, so rewording a label cannot silently change which reasons dedupe.
    """
    return (reason.signal, reason.source, reason.detail)


def _is_collapse(match: Any) -> bool:
    """Whether an identity match is strong enough to merge two candidates.

    The threshold is imported from
    :mod:`paperwhisperer.search.normalize` rather than written here, so
    "proven duplicate" means the same thing in this module as everywhere else in
    the product. A ``probable`` match — same title and author, different
    identifiers — is *not* collapsed: the pair is reported so a human can
    decide, because merging two different papers destroys one of them
    irrecoverably while showing one twice costs a glance.
    """
    return bool(match.same_paper) and match.confidence == COLLAPSE_CONFIDENCE


def _apply_identity(record: dict[str, Any], identity: PaperIdentity) -> dict[str, Any]:
    """Write a merged identity's identifiers back onto a surviving record.

    Only the identifier fields are touched. Title, year, authors and venue keep
    the surviving record's values, for the same reason they do in
    :func:`paperwhisperer.search.normalize._apply_merge`: that record is what
    the user is looking at, and they are not looking at a diff.
    """
    record.update({field: getattr(identity, field) for field in IDENTIFIER_FIELDS})
    record["identity_key"] = identity.key
    return record


def _merge_candidates(
    hits: Sequence[SignalHit], seed: PaperIdentity
) -> tuple[list[_Group], list[dict[str, Any]]]:
    """Fold every hit into one row per paper, unioning the reasons.

    Three things happen per hit, in this order:

    1. **The seed is never recommended to itself.** A citation walk of a
       well-known paper can return the paper — S2's recommendations
       occasionally does, and an OpenAlex ``related_to`` walk of a survey
       returns the survey. Showing it is not a harmless duplicate; it is the
       product telling the user to go and read the document they are already
       reading. The check is the identity layer's, so it catches the arXiv-only
       and DOI-only spellings of the seed as well.
    2. **An exact match folds in.** The two identities are merged (so the
       surviving row carries the union of both sides' identifiers, which is
       what makes a later collapse possible) and the reason is appended.
    3. **A probable match is reported, not folded.** Both rows stay, and the
       pair is named in ``possible_duplicates`` with the evidence.
    """
    groups: list[_Group] = []
    possible: list[dict[str, Any]] = []

    for hit in hits:
        if not isinstance(hit.record, Mapping):
            continue
        identity = hit.identity

        if seed.key and _is_collapse(compare_identities(identity, seed)):
            continue

        target: _Group | None = None
        for group in groups:
            match = compare_identities(identity, group.identity)
            if _is_collapse(match):
                target = group
                break

        if target is not None:
            merged = merge_identities(target.identity, identity)
            target.record = _apply_identity(dict(target.record), merged.identity)
            target.identity = merged.identity
            key = _reason_key(hit.reason)
            if all(_reason_key(existing) != key for existing in target.reasons):
                target.reasons.append(hit.reason)
            continue

        duplicate_of: dict[str, Any] | None = None
        for group in groups:
            match = compare_identities(identity, group.identity)
            if not (match.same_paper and match.requires_confirmation):
                continue
            entry = {
                "title": hit.record.get("title", ""),
                "other_title": group.record.get("title", ""),
                "basis": match.basis,
                "basis_zh": match.basis_zh,
                "note_zh": (
                    match.notes[0]
                    if match.notes
                    else "这两条结果可能来自同一篇论文的两条记录，尚未确认，已同时保留。"
                ),
                "signals": sorted({hit.reason.signal, *(r.signal for r in group.reasons)}),
            }
            if entry not in possible:
                possible.append(entry)
            # Both rows carry the flag, not just the one that happened to be
            # seen first. A badge on only one of a pair leaves the other looking
            # confirmed, and the user has no way to know which of the two is
            # the one under suspicion.
            duplicate_of = {
                "title": group.record.get("title", ""),
                "basis_zh": match.basis_zh,
                "note_zh": entry["note_zh"],
            }
            if duplicate_of not in group.possible_duplicates:
                group.possible_duplicates.append(dict(duplicate_of))
            break

        groups.append(
            _Group(
                record=dict(hit.record),
                identity=identity,
                reasons=[hit.reason],
                possible_duplicates=[dict(duplicate_of)] if duplicate_of else [],
            )
        )
    return groups, possible


def _rank(groups: Sequence[_Group]) -> list[_Group]:
    """Order rows by the strongest signal that reached them.

    Not a quality score and not a popularity sort: it is the answer to "which
    row is the most defensible", computed by looking at the best signal in
    :data:`SIGNAL_ORDER` that appears on the row. A paper reached by both the
    citation graph and a topic search outranks one reached only by the topic
    search, which is the ordering a reader would choose anyway, and it is
    reproducible because :data:`SIGNAL_ORDER` is a constant.
    """
    def best_index(group: _Group) -> int:
        signals = {reason.signal for reason in group.reasons}
        for index, signal in enumerate(SIGNAL_ORDER):
            if signal in signals:
                return index
        return len(SIGNAL_ORDER)

    return sorted(groups, key=best_index)


# --------------------------------------------------------------------------- #
# Signal status aggregation
# --------------------------------------------------------------------------- #


def _aggregate_signals(results: Sequence[SignalResult]) -> list[dict[str, Any]]:
    """One entry per signal, merging the providers that reported on it.

    The per-provider detail is kept under ``providers`` rather than collapsed
    away: two providers both failing is what tells the user to try again later,
    where one failing and one succeeding means the answer is partial rather than
    missing. The aggregated ``status`` is the most optimistic status reported,
    with the rule that ``unavailable`` and ``skipped`` only surface when no
    provider produced anything at all — otherwise a working OpenAlex would
    hide a rate-limited S2 behind a cheerful ``ok``.
    """
    order = {signal: index for index, signal in enumerate(SIGNAL_ORDER)}
    buckets: dict[str, list[SignalResult]] = {}
    for result in results:
        buckets.setdefault(result.signal, []).append(result)

    aggregated: list[dict[str, Any]] = []
    for signal, bucket in buckets.items():
        providers = [item.to_dict() for item in bucket]
        statuses = {item.status for item in bucket}
        if STATUS_OK in statuses:
            status = STATUS_OK
        elif STATUS_EMPTY in statuses:
            status = STATUS_EMPTY
        elif STATUS_UNAVAILABLE in statuses:
            status = STATUS_UNAVAILABLE
        else:
            status = STATUS_SKIPPED
        reason_zh = next(
            (item.reason_zh for item in bucket if item.status == status and item.reason_zh), ""
        )
        aggregated.append(
            {
                "signal": signal,
                "label_zh": SIGNAL_LABELS_ZH.get(signal, signal),
                "status": status,
                "count": sum(len(item.hits) for item in bucket),
                "reason_zh": reason_zh,
                "providers": providers,
            }
        )
    aggregated.sort(key=lambda entry: order.get(entry["signal"], len(order)))
    return aggregated


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #


def recommend_papers(
    query: RecommendationQuery,
    sources: Sequence[SignalSource] = (),
    *,
    budget: CallBudget | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run every signal, unify the duplicates, and report what happened.

    The response is built for the UI, not for a caller that happens to be
    Python:

    ``items``
        One row per paper, each carrying ``reasons`` — a list of
        ``{signal, label_zh, source, reason_zh, detail, confidence}``. This is
        the acceptance criterion: the reason is in the payload, so a client
        renders what was actually the signal instead of reconstructing a label
        from a list of papers it happens to have grouped itself. ``signals`` on
        each row is the same set deduplicated to bare keys, for a cheap
        ``row.signals.includes('citing')`` test.
    ``signals``
        One entry per signal in :data:`SIGNAL_ORDER` — status, count, Chinese
        reason, and a ``providers`` breakdown. ``unavailable_signals`` and
        ``attempted_signals`` are derived from it for clients that want a
        one-line banner.
    ``possible_duplicates``
        The ``probable`` pairs, both rows retained. Present because T3.1's
        distinction is only worth anything if it survives to the screen.
    ``upstream_calls``
        The budget and what it was spent on, so the bound is observable rather
        than a promise in a comment.

    **A source that raises does not fail the request.** Each is wrapped, and an
    exception becomes an ``unavailable`` signal with the provider named — so
    OpenAlex being down leaves four working signals on screen instead of a 500,
    and does not silently produce an empty "被引" bucket that reads as a fact
    about the paper.
    """
    if not isinstance(query, RecommendationQuery):
        raise TypeError("recommend_papers expects a RecommendationQuery")

    call_budget = budget or CallBudget()
    active_sources = list(sources) or default_recommendation_sources(budget=call_budget)
    resolved_limit = max(1, int(limit or query.limit or 1))

    results: list[SignalResult] = []
    for source in active_sources:
        try:
            result = source.fetch(query)
        except Exception as exc:  # noqa: BLE001 - see docstring
            logger.warning(
                "Recommendation signal %s (%s) failed: %s", source.signal, source.name, exc
            )
            result = failed(source.signal, source.name, exc)
        if not isinstance(result, SignalResult):
            # A misbehaving injected source must not be able to fabricate a
            # result. Treating a non-Result as a failure is the only safe
            # reading: there is no signal in it, so there is no reason to show.
            result = unavailable(
                source.signal,
                source.name,
                "该信号返回了无法识别的结果，已忽略。",
            )
        results.append(result)

    hits: list[SignalHit] = []
    for result in results:
        hits.extend(result.hits)

    groups, possible = _merge_candidates(hits, query.identity)
    ranked = _rank(groups)[:resolved_limit]

    items: list[dict[str, Any]] = []
    for group in ranked:
        row = dict(group.record)
        row["reasons"] = [reason.to_dict() for reason in group.reasons]
        row["signals"] = sorted(
            {reason.signal for reason in group.reasons},
            key=lambda signal: SIGNAL_ORDER.index(signal) if signal in SIGNAL_ORDER else 99,
        )
        if group.possible_duplicates:
            row["possible_duplicates"] = group.possible_duplicates
        items.append(row)

    signals = _aggregate_signals(results)
    unavailable_signals = [
        {
            "signal": entry["signal"],
            "label_zh": entry["label_zh"],
            "status": entry["status"],
            "reason_zh": entry["reason_zh"],
        }
        for entry in signals
        if entry["status"] in (STATUS_UNAVAILABLE, STATUS_SKIPPED)
    ]
    errors = [
        f"{entry['label_zh']}：{entry['reason_zh']}"
        for entry in signals
        if entry["status"] == STATUS_UNAVAILABLE and entry["reason_zh"]
    ]

    return {
        "seed": {
            "title": query.seed_title,
            "year": query.identity.normalized_year or parse_year(query.identity.year),
            "identity_key": query.identity.key,
            "doi": query.identity.doi,
            "arxiv_id": query.identity.arxiv_id,
            "s2_paper_id": query.identity.s2_paper_id,
            "openalex_id": query.identity.openalex_id,
        },
        "items": items,
        "signals": signals,
        "attempted_signals": [
            entry["signal"] for entry in signals if entry["status"] in (STATUS_OK, STATUS_EMPTY)
        ],
        "unavailable_signals": unavailable_signals,
        "possible_duplicates": possible,
        "errors": errors,
        "upstream_calls": call_budget.to_dict(),
    }
