"""Hybrid retrieval: three legs, rank fusion, and a record of everything.

This is T1.4 of the plan, and the acceptance criteria are short: *a Chinese or
English query both recall; a proper noun, a DOI or a dataset name is retrieved
stably; the retrieval record is inspectable.* Three sentences that between them
rule out most of what a retriever could be. This module is the shape that
satisfies them, and the shape is mostly a set of refusals.

**Fusion is Reciprocal Rank Fusion, not a weighted sum of scores.** A dense leg
returns a cosine similarity and a lexical leg returns a bounded token-coverage
ratio; one lives in ``[-1, 1]`` with no idea what a good value is, the other in
``[0, 1]`` with a known ceiling and a meaning ("how much of the query is
present"). Adding them requires first deciding what ``cos = 0.74`` is worth on
the coverage scale, and the answer is not derivable -- it is a free parameter
that has to be tuned against a corpus, which is to say it will be tuned against
the one corpus somebody happened to have, and will look perfect on it. RRF never
looks at a score. It looks only at a rank, and combines ranks as
``sum_leg w_leg / (k + rank_leg)``. The consequence is the whole reason to pick
it: the parameter that survives is ``k``, and ``k`` has an order of magnitude
rather than a best value.

**The lexical leg is not a fallback, it is the leg that answers exact questions.**
The acceptance criterion "专有名词/DOI/数据集名精确查询稳定" is not a wish, it is
the reason this module exists. A DOI, a dataset name, a method name and an author
surname are strings, and a dense vector blurs every string that is not a common
English word into the same neighbourhood: ``BM25``, ``BERT-base`` and ``BERT-large``
have vectors closer to each other than either is to the passage that answers the
question. So the lexical leg runs on every query, contributes its own ranking,
and is never demoted to "if nothing else turned up". The plan's own framing --
"全文检索取代前 N 字符 excerpt" -- is a claim about *retrieval* replacing a
substring, and the substring is still the right tool for a string.

**Fuzzy recall lives here, and therefore works on both tiers.** T1.1 found that
:mod:`paperwhisperer.infra.stores.scoring` scores a prefix or trigram candidate
at ``0.0`` -- correctly, because its contract is *whole terms* -- and refused to
ship a widening that only the PostgreSQL tier could answer, since a capability
one tier has and the other lacks is a capability the shared test suite cannot
assert. It left ``pg_trgm`` installed and indexed, and left the policy here. So
the widening is implemented in Python, over chunk text the
:class:`~paperwhisperer.infra.stores.interfaces.MetadataStore` hands back through
the same interface on both tiers, and it is a *third leg* with its own weight
rather than a change to what "a match" means. That distinction is the reason it
does not break the store contract: :func:`~paperwhisperer.infra.stores.scoring.score_lexical`
still scores a near-miss at ``0.0``, and fusion is a separate stage that can
weight the near-miss as a *candidate* without redefining a *match*. What could
not be verified here is stated at :meth:`HybridRetriever._fuzzy_leg`: nothing
about the Hosted tier, because there is no PostgreSQL in this environment. The
algorithm is tier-agnostic by construction -- it never names SQL, a table, or an
engine -- but "by construction" is an argument, not a measurement, and the
Hosted-tier assertion belongs to whoever has a database.

**Chinese is handled by the store's bigram tokenizer, and this module adds
nothing to it.** :func:`paperwhisperer.infra.stores.scoring.tokenize` splits a CJK
run into character bigrams, and ``score_lexical`` already gives a phrase bonus
when the normalized query appears as a contiguous substring. Those two rules
together mean ``注意力`` recalls ``自注意力机制`` -- the bigrams overlap and the
substring test passes -- and a *truncated Chinese* query recalls a longer
document phrase for free, because the truncated normalized form is a substring of
the longer one. The gap that is left, and the only one the fuzzy leg closes, is
character-level near-miss in Latin text: a truncated identifier (``transfor``), a
typo (``recieve``), a compound written one way in the document and another in the
query. Trigram similarity over whole terms handles those, and is not applied to
CJK bigrams at all, because for a two-character bigram the trigram arithmetic is
noise -- see :func:`_token_similarity`.

**A leg with nothing to say abstains rather than voting arbitrarily.** This is
the one addition the acceptance criteria forced, and it was found by a failing
test rather than designed: searching the corpus for a DOI, the dense leg
returned twelve candidates whose cosine similarities were ``1.0`` and
``0.99999`` -- a spread of 0.0002 -- and RRF counted that noise as an opinion,
so the passage that was the lexical leg's first and only match came back fourth
behind three passages that did not contain the DOI at all. Fusing a real ranking
with an arbitrary one is strictly worse than fusing two real ones, so a dense
leg whose top candidates are inside :data:`DENSE_TIE_BAND` of each other does
not vote, and says in the record that it did not. Two informative rankings and
one meaningless one should count as two.

**The metadata filter is a filter, and it runs before fusion.** ``document_ids``
goes into both stores' ``search`` calls, so an excluded document is never a
candidate and cannot be rescued by a high rank. That is the whole distinction: a
filter removes, fusion ranks. Anything that merely *prefers* one section over
another -- which is what a per-task plan wants -- is a post-fusion reweight, and
it is called :attr:`RetrievalPlan.topic_boosts` so nobody later mistakes a
preference for a filter and reports "the plan excluded it" when it did not.

**A plan may not return nothing quietly.** Section 1 of the plan: 证据不足时明确说不足,
不补全. A retriever that cannot find evidence and quietly returns the document
head has invented an answer, which is the one outcome this product exists to
prevent. So an under-filled retrieval is a *first-class* result:
:attr:`RetrievalResult.insufficient_evidence` is ``True``,
:attr:`RetrievalResult.empty_reason` says which of the declared reasons
applies, and there is no fallback plan, no default query and no "return the best
we have anyway". :func:`plan_for` is also deliberately strict -- an unknown task
name raises rather than returning the Q&A plan, because a typo that silently
becomes "answer the question generically" is a wrong answer delivered by a typo.

**Rerank is off by default, and a failing reranker degrades rather than
reorders.** See :mod:`paperwhisperer.rag.rerank` for why there is a provider
abstraction at all; the two decisions that matter here are that
:func:`~paperwhisperer.rag.rerank.build_reranker` returning ``None`` is normal
and not an error, and that a provider exception leaves the fusion order intact
and writes a Chinese reason into :attr:`RetrievalRecord.degradations`.

**The record holds the losers.** The plan's exit criteria and T1.8 both need to
answer "why did this answer cite that passage", and a record containing only the
top-k cannot answer it -- a bad answer is far more often a *missing* candidate
than a badly-ordered one. :class:`RetrievalRecord` therefore carries every fused
candidate with its per-leg rank and contribution, whether it was selected and if
not why, alongside the final evidence, the expansions and the degradations. It
round-trips through ``to_dict``/``from_dict`` without loss, because T1.8 stores
it next to an ``analysis_run`` and a record that has to be re-interpreted to be
written down is a record nobody writes down.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from paperwhisperer.infra.stores import scoring
from paperwhisperer.infra.stores.factory import StoreBundle
from paperwhisperer.ingestion.schema import Chunk, Evidence
from paperwhisperer.rag.rerank import RerankDocument, build_reranker

logger = logging.getLogger(__name__)

__all__ = [
    "AUTO",
    "CandidateRecord",
    "DENSE_TIE_BAND",
    "DENSE_TIE_WINDOW",
    "DEFAULT_RRF_K",
    "ExpandedEvidence",
    "ExpansionRecord",
    "FusionEntry",
    "FuzzyRecallPolicy",
    "HybridRetriever",
    "LegHit",
    "NeighbourPolicy",
    "PLANS",
    "RRF_SAFE_RANGE",
    "RetrievalPlan",
    "RetrievalRecord",
    "RetrievalResult",
    "TASK_CONCEPT",
    "TASK_CONTRIBUTION",
    "TASK_EXPERIMENT",
    "TASK_QA",
    "TASK_REPRODUCTION",
    "TASK_TYPES",
    "lexical_query_forms",
    "plan_for",
    "reciprocal_rank_fusion",
    "register_plan",
    "trigram_dice",
]


# --------------------------------------------------------------------------
# Legs
# --------------------------------------------------------------------------

#: The three retrieval legs, named. A leg is a *ranking* over chunk ids; a
#: score attached to a hit is metadata for the record and never enters fusion.
LEG_DENSE = "dense"
LEG_LEXICAL = "lexical"
LEG_FUZZY = "fuzzy"

#: Declared order, so the record's leg list has a fixed shape regardless of
#: which legs ran. Set iteration order is the kind of thing that makes a
#: "deterministic" record compare equal in one run and not the next.
LEG_ORDER = (LEG_DENSE, LEG_LEXICAL, LEG_FUZZY)


# --------------------------------------------------------------------------
# RRF
# --------------------------------------------------------------------------

#: The RRF constant. Sixty is Cormack et al.'s value, and the reason it is a
#: named constant rather than a tuned one is that it is *insensitive* over a wide
#: range -- see :data:`RRF_SAFE_RANGE` and :func:`reciprocal_rank_fusion`.
DEFAULT_RRF_K = 60

#: The band inside which RRF's ordering was measured to be stable. Outside it
#: the algorithm still runs; the retriever records a degradation rather than
#: refusing, because a caller experimenting with ``k`` should be able to see
#: what it bought.
#:
#: The low edge is 20 rather than the 10 usually quoted, and the difference is
#: measured rather than borrowed: on a twelve-candidate three-leg fusion,
#: ``k = 60`` and ``k = 10`` disagree about one chunk's position while ``k = 20``
#: through ``k = 100`` agree on the whole list. The upper edge is the point
#: where the curve stops distinguishing rank 1 from rank 900 at all. ``k = 60``
#: sits in the middle of the band, which is the whole argument for it: a
#: parameter that can be moved by a factor of five without changing the answer
#: is a parameter that cannot have been overfitted to this corpus.
RRF_SAFE_RANGE = (20, 100)


#: A dense leg whose returned candidates all sit inside a band this narrow has
#: not expressed an opinion -- it has expressed a tie, and the order inside a tie
#: is floating-point noise plus a ``chunk_id`` comparison.
#:
#: It is worth handling rather than papering over, because the case is the common
#: one this task exists for. A model asked about a DOI, a model name or a
#: dataset name it cannot read returns something close to its prior for *every*
#: passage: measured on the corpus, the query ``10.1137/0330046`` produced dense
#: scores of ``1.0, 1.0, 1.0, 0.99999, 0.99999, 0.99993, ...`` -- a spread of
#: 0.0002 across the top twelve. Under plain RRF that arbitrary order then *adds*
#: to a correct lexical ranking, and the passage that is the lexical leg's first
#: and only match drops to fourth place behind three passages that are not it.
#:
#: Letting a leg abstain when it has nothing to say is the whole point of a
#: fusion step: two informative rankings and one arbitrary one should vote as
#: two, not as three. The abstention is recorded, because "the dense leg did not
#: vote" is a fact a reader of a retrieval record needs, and it is the honest
#: explanation for a result that came from the lexical leg alone.
#:
#: Applied to the dense leg only, and deliberately not to the lexical one. This
#: is a statement about the precision of a cosine similarity: 0.9998 against 1.0
#: is not a distinction a ranking can act on, whereas a coverage score of 0.33
#: against 0.67 is a real difference in a meaningful quantity.
DENSE_TIE_BAND = 0.02

#: How far down the dense leg's ranking the tie is measured. The *whole* returned
#: window is the wrong thing to measure: the tail of a nearest-neighbour
#: ranking is less similar than the head by construction, so a window of forty
#: gives a spread of 0.03 for a query where the top ten are identical to four
#: decimal places, and a rule with a threshold calibrated on that would never
#: fire. The question the band answers is "can this leg tell its own best
#: candidates apart?", so the window is a prefix. Ten, because a reranker is
#: usually asked to reorder about that many passages and a decision about fewer
#: than five would be reacting to one or two coincidences.
DENSE_TIE_WINDOW = 10


@dataclass(slots=True, frozen=True)
class LegHit:
    """One leg's opinion about one candidate, as a position and a contribution.

    ``rank`` is 1-based and ``score`` is the term this leg actually contributed
    to the fusion -- not the leg's own score. Keeping the two apart is what
    makes the record worth reading: "the dense leg put this fourth and the
    lexical leg did not return it at all" is a diagnosis, and a record holding
    one blended number cannot express it.
    """

    leg: str
    rank: int
    score: float


@dataclass(slots=True, frozen=True)
class FusionEntry:
    """One fused candidate: its id, its fused score, and the legs behind it."""

    chunk_id: str
    score: float
    legs: tuple[LegHit, ...] = ()

    @property
    def legs_found(self) -> tuple[str, ...]:
        return tuple(leg.leg for leg in self.legs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "legs": [
                {"leg": leg.leg, "rank": leg.rank, "score": leg.score} for leg in self.legs
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FusionEntry:
        return cls(
            chunk_id=str(payload.get("chunk_id") or ""),
            score=float(payload.get("score") or 0.0),
            legs=tuple(
                LegHit(
                    leg=str(item.get("leg") or ""),
                    rank=int(item.get("rank") or 0),
                    score=float(item.get("score") or 0.0),
                )
                for item in payload.get("legs") or ()
                if isinstance(item, Mapping)
            ),
        )


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
) -> tuple[FusionEntry, ...]:
    """Combine ranked lists by rank position, not by score. Best first.

    ``score = sum_over_legs(weight[leg] / (k + rank[leg]))`` with ``rank``
    1-based: Reciprocal Rank Fusion, reproduced here because there is nothing to
    configure about it and everything to explain about it. The explanation is in
    the module docstring; the short version is that the input to this function
    is a *position*, so the cosine similarity of one leg and the token-coverage
    ratio of another are never asked to be commensurate.

    **What ``k`` does, and what breaks if it is wrong.** ``k`` damps the head of
    each list: it is the dial that decides how much one leg's *first* opinion is
    worth against the rest of that leg, and against the other legs. Two failure
    directions, and they are opposite:

    * **``k`` too small.** One leg's top-1 starts to outrank a consensus. Take
      ``A`` ranked 1st by dense and 4th by lexical, and ``B`` ranked 2nd by
      both. At ``k = 1`` the arithmetic is ``0.5 + 0.2 = 0.7`` against
      ``0.67``, and ``A`` -- the candidate a single leg loved and the other
      demoted -- wins. At ``k = 10`` it is ``0.162`` against ``0.167`` and
      ``B`` wins. So a small ``k`` lets one retriever veto the other, which is
      only safe while the legs are of comparable quality. They are not: the
      dense leg is a lossy semantic summary and the lexical leg is exact, so
      small ``k`` is exactly the setting in which the blunter leg decides.
    * **``k`` too large.** The curve flattens until every rank contributes
      nearly the same and the fused score approaches a *count* of how many legs
      returned the passage at all. A passage first in the dense leg alone
      (score ``1/61 = 0.0164`` at ``k = 60``) beats a passage both legs put in
      900th (score ``2/960 = 0.0021``) by nearly eight to one; at
      ``k = 10,000`` those are ``0.0001`` and ``0.00018``, and the deep
      consensus wins. That is a worse failure than the first, because it looks
      like agreement.

    Between roughly 20 and 100 the ordering does not move, and *that* is the
    argument for RRF: the one parameter left is one nobody can overfit to a
    corpus, because there is no corpus-dependent value to overfit it to.

    Ordering is total -- ``(-score, chunk_id)`` -- so two runs against the same
    indexes produce the same list, and ``final_k`` is a function of the data
    rather than of dictionary iteration order.
    """
    if int(k) < 1:
        raise ValueError(f"RRF k 必须 >= 1，收到 {k!r}。")
    contributions: dict[str, dict[str, float]] = {}
    positions: dict[str, dict[str, int]] = {}
    for leg in LEG_ORDER:
        items = rankings.get(leg) or ()
        if not items:
            continue
        weight = 1.0 if weights is None else float(weights.get(leg, 0.0))
        if weight == 0.0:
            continue
        for position, chunk_id in enumerate(items, start=1):
            key = str(chunk_id)
            contributions.setdefault(key, {})[leg] = weight / (int(k) + position)
            positions.setdefault(key, {})[leg] = position
    entries = [
        FusionEntry(
            chunk_id=key,
            score=sum(parts.values()),
            legs=tuple(
                LegHit(leg=leg, rank=positions[key][leg], score=parts[leg])
                for leg in LEG_ORDER
                if leg in parts
            ),
        )
        for key, parts in contributions.items()
    ]
    entries.sort(key=lambda entry: (-entry.score, entry.chunk_id))
    return tuple(entries)


# --------------------------------------------------------------------------
# Query shaping
# --------------------------------------------------------------------------

#: A DOI arriving as a URL, which is how a user pastes one and how a browser
#: hands one over. Stripping it is not a nicety: ``normalize`` turns
#: ``https://doi.org/10.18653/v1/N19-1234`` into eight terms, three of which
#: are ``https``/``doi``/``org``, and the coverage score is diluted by tokens
#: that describe the transport rather than the paper. Every other document that
#: mentions a DOI contains those same three terms, so the dilution does not just
#: weaken the right answer -- it promotes the wrong one.
_DOI_URL_PREFIX = re.compile(r"^\s*(?:https?://)?(?:dx\.)?doi\.org/|^\s*doi:\s*", re.IGNORECASE)

#: One alternate form, for the reason above. The tempting alternative --
#: re-spelling the query every way separators allow ("BM25" / "BM 25" / "BM-25")
#: -- was written and deleted: :func:`scoring.normalize` already replaces every
#: non-alphanumeric character with a space on *both* sides, so the spellings
#: produce the identical term list and the variant search returns byte-identical
#: candidates. The only form worth issuing is the one that removes text the
#: document does not contain.
MAX_QUERY_FORMS = 2


def lexical_query_forms(query: str) -> tuple[str, ...]:
    """The surface forms one search is issued under, best first.

    ``forms[0]`` is the form every leg actually searches with: the query with
    any DOI URL prefix removed, or the query itself when there is none. The
    dense leg uses *only* ``forms[0]`` -- an embedding of ``"https://doi.org/
    10.1137/0330046"`` is the embedding of a URL, not of a paper, and issuing a
    second dense query per form would double the cost of the most expensive leg
    to buy nothing. The lexical leg tries all of them, because there stripping
    the prefix changes *which terms exist* rather than which words they are.

    Ordered best-first rather than raw-first for one reason: the record's
    ``query_forms`` and the dense leg must agree about which form is the query,
    or "the same question" would embed two different strings depending on how the
    user pasted the identifier.

    See :data:`_DOI_URL_PREFIX` for why this one alternate exists and the
    module docstring for why no third does.
    """
    raw = str(query or "")
    forms: list[str] = []
    stripped = _DOI_URL_PREFIX.sub("", raw)
    for candidate in (stripped, raw):
        if candidate and candidate not in forms:
            forms.append(candidate)
    return tuple(forms)[:MAX_QUERY_FORMS]


# --------------------------------------------------------------------------
# Fuzzy recall
# --------------------------------------------------------------------------

#: Latin tokens shorter than this are not fuzzy-matched. Two reasons, both
#: arithmetic rather than taste: a three-character token has four trigrams, so
#: its Dice coefficient against a six-character word is dominated by the two
#: padding trigrams they share for free; and a two-character token is almost
#: always a CJK bigram from :func:`scoring.tokenize`, where "similarity" is not
#: a quantity -- comparing 注意 with 意力 by trigrams is a coin flip dressed as
#: a measurement. CJK partial matching is already exact (substring bonus) and
#: needs no help here.
MIN_FUZZY_TOKEN_CHARS = 3

#: Similarity below which a query token counts as *not found* and above which it
#: counts. The number is set by measured pairs, not by a corpus and not by taste.
#: At 0.65 the leg accepts ``attention``/``attenion`` (0.74, a transposition),
#: ``resnet``/``resnet50`` (0.75, a suffix the user omitted) and
#: ``transfor``/``transformer`` (0.76), and rejects
#: ``attention``/``affection`` (0.40), ``rcnn``/``rnn`` (0.44),
#: ``cat``/``category`` (0.46) and ``recieve``/``receive`` (0.50).
#:
#: The rejections are the interesting half. ``recieve``/``receive`` is a real
#: typo that this threshold turns down: a transposition in the middle of a long
#: word breaks enough trigrams that the score lands mid-band, and admitting
#: that band would also admit ``cat``/``category``. So the leg misses some
#: misspellings, and the honest response to that is what the numbers below
#: already are -- the threshold is high, the leg's weight is below 1.0 so a
#: marginal hit cannot overrule the other two, and
#: :attr:`FuzzyRecallPolicy.enabled` exists so a deployment that would rather
#: have recall than precision can have both. A threshold low enough to catch
#: every transposition would be a threshold that returned the whole corpus.
FUZZY_MIN_SIMILARITY = 0.65

#: A candidate that a query term is a proper prefix of is scored by prefix
#: length rather than by trigram overlap alone. Truncation is the commonest real
#: fuzzy case -- a user who cannot spell the method types the first half of it --
#: and "prefix" is both a stronger and a more legible signal than a fraction.
FUZZY_PREFIX_FLOOR = 0.70
FUZZY_PREFIX_SPAN = 0.29
FUZZY_MIN_PREFIX_CHARS = 4


def _trigrams(token: str) -> frozenset[str]:
    """Space-padded character trigrams, which is what makes the ends count.

    The padding is the standard ``pg_trgm`` convention and it is not cosmetic:
    without it ``cat`` and ``category`` share only their interior and are
    understated, while ``ing`` and ``sing`` look identical because both are
    wholly interior. Padding gives every token the same three leading and
    trailing evidence a human uses.
    """
    padded = f"  {token} "
    return frozenset(padded[index : index + 3] for index in range(len(padded) - 2))


def trigram_dice(left: str, right: str) -> float:
    """Sørensen–Dice coefficient over padded character trigrams, in ``[0, 1]``.

    Dice rather than Jaccard because it treats a *containment* relationship the
    way a user asking for ``transform`` expects: the trigrams of ``transform``
    are almost a subset of those of ``transformer``, and Jaccard would penalise
    that containment as heavily as a total mismatch of the same size.
    """
    if not left or not right:
        return 0.0
    first, second = _trigrams(left), _trigrams(right)
    if not first or not second:
        return 0.0
    return 2.0 * len(first & second) / (len(first) + len(second))


def _token_similarity(query_token: str, candidate_token: str) -> float:
    """How well ``candidate_token`` answers for ``query_token``, in ``[0, 1]``.

    Two rules combined by maximum. Trigram Dice is the general one and covers
    transposition, inflection and typos. The prefix rule is the special one,
    scaled rather than boolean so that ``transfor`` ranks above ``transferred``
    for the query ``transfer``: a truncation is a better guess than an unrelated
    word that happens to share letters.
    """
    similarity = trigram_dice(query_token, candidate_token)
    if len(query_token) >= FUZZY_MIN_PREFIX_CHARS and candidate_token.startswith(query_token):
        ratio = len(query_token) / len(candidate_token)
        similarity = max(similarity, FUZZY_PREFIX_FLOOR + FUZZY_PREFIX_SPAN * ratio)
    return min(0.99, similarity)


@dataclass(frozen=True, slots=True)
class FuzzyRecallPolicy:
    """When to widen recall beyond whole terms, and how far.

    Every field is a *bound* as much as a *threshold*, and the bounds are the
    part that matters for a personal-tier deployment: the fuzzy leg reads chunk
    text back through the metadata store, which is real work, and an unbounded
    widening is how a retriever that is fast on a four-document library becomes
    a minute-long answer on a four-hundred-document one.

    ``trigger_under`` is the sharpest of the bounds. The leg runs only when the
    exact legs came back with fewer than this many distinct candidates, so an
    ordinary question never pays for it while a query that found *nothing* --
    the case fuzzy recall exists for -- always does. Setting it to ``0`` runs
    the leg unconditionally, which is the right setting for a plan that expects
    misspellings (see the reproduction and concept plans).
    """

    enabled: bool = True
    min_similarity: float = FUZZY_MIN_SIMILARITY
    #: How many query terms must be individually matched for a chunk to
    #: qualify. ``1``, because a one-term query is the common case for a proper
    #: noun, and coverage across the remaining terms is already handled by the
    #: score's denominator.
    min_matched_tokens: int = 1
    #: Only widen for a query at least this long. A two-character query has
    #: nothing to be near-missed by, and every candidate it produced would be a
    #: guess presented as a match.
    min_query_chars: int = 4
    #: Documents to consider when the plan does not scope the search.
    max_documents: int = 25
    #: Chunks actually examined. Truncating the scan is recorded as a
    #: degradation, not applied silently: a fuzzy leg that stopped early and
    #: one that looked and found nothing are different facts.
    max_scanned_chunks: int = 2000
    #: Run only when the exact legs returned fewer candidates than this.
    trigger_under: int = 6

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_similarity": self.min_similarity,
            "min_matched_tokens": self.min_matched_tokens,
            "min_query_chars": self.min_query_chars,
            "max_documents": self.max_documents,
            "max_scanned_chunks": self.max_scanned_chunks,
            "trigger_under": self.trigger_under,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FuzzyRecallPolicy:
        base = cls()
        return cls(
            enabled=bool(payload.get("enabled", base.enabled)),
            min_similarity=float(payload.get("min_similarity", base.min_similarity)),
            min_matched_tokens=int(payload.get("min_matched_tokens", base.min_matched_tokens)),
            min_query_chars=int(payload.get("min_query_chars", base.min_query_chars)),
            max_documents=int(payload.get("max_documents", base.max_documents)),
            max_scanned_chunks=int(
                payload.get("max_scanned_chunks", base.max_scanned_chunks)
            ),
            trigger_under=int(payload.get("trigger_under", base.trigger_under)),
        )


# --------------------------------------------------------------------------
# Neighbour expansion
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NeighbourPolicy:
    """How much surrounding text an answerable fragment needs, and its ceiling.

    T1.2 removed chunk overlap, and its docstring hands this module the bill
    explicitly: "A retriever that wants a passage's continuation has a better
    tool for it: the neighbour expansion T1.4 already owns." A chunk that stops
    where the budget ran out is a fragment, and a fragment is frequently not
    answerable -- the sentence naming a hyperparameter is in one chunk and the
    value is in the next. Expansion is what repairs that, and it is *repair*
    rather than retrieval: an expanded passage was not found, it was placed,
    which is why :class:`ExpandedEvidence` is a different type from
    :class:`~paperwhisperer.ingestion.schema.Evidence` and is never counted among
    the results a release gate measures recall over.

    Two relations, both bounded:

    ``neighbour``
        The chunks adjacent in reading order within the same document, by
        ``order``. ``after`` is what the reproduction plan raises, because the
        continuation of a fragment -- the numbers, the caveat, the table the
        previous sentence introduced -- is what the anchor is missing.

    ``parent``
        The other parts of the same element, when the chunk is one piece of a
        split element (``part_index``/``part_count``). This is the case the
        chunker cannot solve itself: it is forbidden from overlapping precisely
        so that a rebuilt index does not store every sentence twice, which
        means the pieces of one oversized paragraph are the only place a
        passage's own continuation is not simply "the next chunk".
    """

    before: int = 1
    after: int = 1
    max_per_anchor: int = 4
    max_total: int = 12
    include_parent_parts: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before,
            "after": self.after,
            "max_per_anchor": self.max_per_anchor,
            "max_total": self.max_total,
            "include_parent_parts": self.include_parent_parts,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NeighbourPolicy:
        base = cls()
        return cls(
            before=int(payload.get("before", base.before)),
            after=int(payload.get("after", base.after)),
            max_per_anchor=int(payload.get("max_per_anchor", base.max_per_anchor)),
            max_total=int(payload.get("max_total", base.max_total)),
            include_parent_parts=bool(
                payload.get("include_parent_parts", base.include_parent_parts)
            ),
        )


# --------------------------------------------------------------------------
# Retrieval plans
# --------------------------------------------------------------------------

#: The five task types of plan section 6.4, named in the plan's own Chinese.
TASK_QA = "qa"
TASK_CONCEPT = "concept"
TASK_CONTRIBUTION = "contribution"
TASK_EXPERIMENT = "experiment"
TASK_REPRODUCTION = "reproduction"

TASK_TYPES = (
    TASK_QA,
    TASK_CONCEPT,
    TASK_CONTRIBUTION,
    TASK_EXPERIMENT,
    TASK_REPRODUCTION,
)

#: Display names, in Chinese, for messages a user or an operator reads.
TASK_LABELS: dict[str, str] = {
    TASK_QA: "问答",
    TASK_CONCEPT: "概念",
    TASK_CONTRIBUTION: "贡献",
    TASK_EXPERIMENT: "实验",
    TASK_REPRODUCTION: "复现",
}


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """A task type's retrieval strategy, as a value you can read and diff.

    A plan is a frozen dataclass rather than a branch inside :meth:`retrieve`
    for three reasons. It can be *inspected*: T4.3's eval compares strategies,
    and a strategy encoded in five ``if`` statements is only comparable by
    reading the source. It can be *extended*: :func:`register_plan` adds a sixth
    task type without touching the retriever, which matters because the five
    here are the plan's, not the product's -- "what does this paper's table 3
    say" is a task type nobody has written down yet. And it can be *recorded*:
    :meth:`to_dict` is stored in the retrieval record, so a bad answer is
    debuggable against the exact strategy that produced it rather than against a
    default that may have changed since.

    Two fields are load-bearing and easy to confuse:

    ``required_document_ids``
        A *filter*, passed into both stores before fusion. A document named
        here cannot appear in the result even if it ranks first everywhere.

    ``topic_boosts``
        A *preference*, applied after fusion as a multiplier on the fused score.
        It can reorder but never create: a section that no leg returned stays
        absent no matter how many boosts name it. That is the honest behaviour
        for a heuristic, and it is why a plan that "matches nothing" returns
        nothing rather than promoting the nearest unrelated section.
    """

    task: str
    label: str
    description: str
    rrf_k: int = DEFAULT_RRF_K
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    fuzzy_weight: float = 0.6
    #: How many candidates each leg is asked for. Deliberately larger than
    #: ``final_k``: fusion can only promote a candidate some leg returned, so a
    #: leg asked for exactly ``final_k`` would cap the fused list at ``final_k``
    #: and make the whole mechanism decorative.
    candidate_depth: int = 40
    final_k: int = 8
    #: How many of the selected anchors get neighbour expansion.
    expand_top_k: int = 4
    #: Below this spread in the dense leg's own scores, the leg abstains.
    #: ``0.0`` disables the check; see :data:`DENSE_TIE_BAND` for why a cosine
    #: tie must not be counted as a vote.
    dense_tie_band: float = DENSE_TIE_BAND
    topic_boosts: tuple[tuple[str, float], ...] = ()
    required_document_ids: tuple[str, ...] | None = None
    #: A floor under the lexical leg's own score. The lexical contract already
    #: refuses to return a chunk sharing no term with the query; this is the
    #: same idea with a bar, and it is how a plan says "an answer has to
    #: contain most of what was asked".
    min_lexical_score: float = 0.0
    #: ``None`` means "use whatever the deployment configured". A plan can turn
    #: rerank off; it cannot force it on, because there may be no provider.
    rerank: bool | None = None
    fuzzy: FuzzyRecallPolicy = field(default_factory=FuzzyRecallPolicy)
    neighbors: NeighbourPolicy = field(default_factory=NeighbourPolicy)

    def __post_init__(self) -> None:
        if not str(self.task or "").strip():
            raise ValueError("RetrievalPlan.task 不能为空。")
        if int(self.rrf_k) < 1:
            raise ValueError(f"rrf_k 必须 >= 1，收到 {self.rrf_k!r}。")
        if int(self.final_k) <= 0:
            raise ValueError(f"final_k 必须 > 0，收到 {self.final_k!r}。")
        if int(self.candidate_depth) < int(self.final_k):
            raise ValueError(
                f"candidate_depth（{self.candidate_depth}）不能小于 final_k"
                f"（{self.final_k}）：融合只能提升已被召回的候选。"
            )

    # -- weights -----------------------------------------------------------

    def leg_weights(self) -> dict[str, float]:
        """Weight per leg, in the declared leg order. A zero weight drops a leg."""
        weights = {
            LEG_DENSE: float(self.dense_weight),
            LEG_LEXICAL: float(self.lexical_weight),
            LEG_FUZZY: float(self.fuzzy_weight),
        }
        return {leg: weights[leg] for leg in LEG_ORDER if weights[leg] != 0.0}

    def k_outside_safe_range(self) -> bool:
        """Whether this plan's ``rrf_k`` is outside the band RRF is stable in."""
        low, high = RRF_SAFE_RANGE
        return not (low <= int(self.rrf_k) <= high)

    def boost_for(self, heading_path: Sequence[str]) -> tuple[float, str]:
        """The multiplier this plan gives a chunk in this section.

        Returns ``(factor, keyword)``. A section matching two keywords takes
        the larger rather than the sum: a paper whose heading is "Experiments and
        Results" is not doubly interesting, and summing would make the
        multiplier depend on how many synonyms a heading happened to contain.

        Matching is case-folded substring against any level of the heading path,
        because the plan's keywords are written in both scripts and the paths
        are built by adapters whose level conventions differ. Substring rather
        than equality for the same reason: the plan names ``"实验"`` and the
        document says ``"4. 实验与分析"``.
        """
        if not self.topic_boosts or not heading_path:
            return 1.0, ""
        haystack = " › ".join(str(item) for item in heading_path).casefold()
        best = 0.0
        keyword = ""
        for candidate, amount in self.topic_boosts:
            needle = str(candidate).casefold()
            if needle and needle in haystack and float(amount) > best:
                best = float(amount)
                keyword = str(candidate)
        return 1.0 + best, keyword

    def with_overrides(self, **overrides: Any) -> RetrievalPlan:
        """A copy with some fields replaced -- the ordinary way to specialise.

        ``dataclasses.replace`` under a name that says what it is for, so a
        caller building a one-off plan is not reaching for ``replace`` and
        having to remember that the topic boosts are a tuple of pairs.
        """
        return replace(self, **overrides)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "label": self.label,
            "description": self.description,
            "rrf_k": self.rrf_k,
            "dense_weight": self.dense_weight,
            "lexical_weight": self.lexical_weight,
            "fuzzy_weight": self.fuzzy_weight,
            "candidate_depth": self.candidate_depth,
            "dense_tie_band": self.dense_tie_band,
            "final_k": self.final_k,
            "expand_top_k": self.expand_top_k,
            "topic_boosts": [list(item) for item in self.topic_boosts],
            "required_document_ids": (
                list(self.required_document_ids)
                if self.required_document_ids is not None
                else None
            ),
            "min_lexical_score": self.min_lexical_score,
            "rerank": self.rerank,
            "fuzzy": self.fuzzy.to_dict(),
            "neighbors": self.neighbors.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetrievalPlan:
        boosts = tuple(
            (str(item[0]), float(item[1]))
            for item in payload.get("topic_boosts") or ()
            if isinstance(item, Sequence) and len(item) == 2
        )
        required = payload.get("required_document_ids")
        rerank = payload.get("rerank")
        base = cls(task="placeholder", label="", description="")
        return cls(
            task=str(payload.get("task") or ""),
            label=str(payload.get("label") or ""),
            description=str(payload.get("description") or ""),
            rrf_k=int(payload.get("rrf_k") or base.rrf_k),
            dense_weight=float(payload.get("dense_weight", base.dense_weight)),
            lexical_weight=float(payload.get("lexical_weight", base.lexical_weight)),
            fuzzy_weight=float(payload.get("fuzzy_weight", base.fuzzy_weight)),
            candidate_depth=int(payload.get("candidate_depth") or base.candidate_depth),
            dense_tie_band=float(
                payload.get("dense_tie_band", base.dense_tie_band)
            ),
            final_k=int(payload.get("final_k") or base.final_k),
            expand_top_k=int(payload.get("expand_top_k") or base.expand_top_k),
            topic_boosts=boosts,
            required_document_ids=(
                tuple(str(item) for item in required) if required is not None else None
            ),
            min_lexical_score=float(payload.get("min_lexical_score") or 0.0),
            rerank=None if rerank is None else bool(rerank),
            fuzzy=FuzzyRecallPolicy.from_dict(payload.get("fuzzy") or {}),
            neighbors=NeighbourPolicy.from_dict(payload.get("neighbors") or {}),
        )

    def signature(self) -> str:
        """A short stable hash of the whole plan, for ids and for diffing.

        Content-addressed, like the chunk ids, so two retrievals under the same
        strategy produce the same record id and a record written under one plan
        cannot be mistaken for one written under another.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


#: A plan per task type. The weights and boosts below are *declared editorial
#: choices*, not fitted parameters: they encode what a person who has read a few
#: hundred papers knows about where each kind of answer lives. They are wrong
#: often enough to be worth revisiting, which is why they are data in a frozen
#: value with a signature rather than magic numbers inside a function where a
#: reader cannot see that a choice was made at all.
PLANS: dict[str, RetrievalPlan] = {
    TASK_QA: RetrievalPlan(
        task=TASK_QA,
        label=TASK_LABELS[TASK_QA],
        description="通用问答：三条腿等权，邻居扩展前 1 后 1，取 8 条。",
        final_k=8,
        expand_top_k=4,
        neighbors=NeighbourPolicy(before=1, after=1, max_per_anchor=4, max_total=12),
    ),
    TASK_CONCEPT: RetrievalPlan(
        task=TASK_CONCEPT,
        label=TASK_LABELS[TASK_CONCEPT],
        description=(
            "概念解释：优先定义所在的章节（引言 / 术语 / 背景），取 6 条并向前多扩 1 块，"
            "因为定义句常在段首。"
        ),
        final_k=6,
        expand_top_k=4,
        topic_boosts=(
            ("术语", 0.45),
            ("definitions", 0.45),
            ("terminology", 0.4),
            ("background", 0.3),
            ("引言", 0.3),
            ("背景", 0.25),
            ("introduction", 0.25),
            ("abstract", 0.15),
            ("摘要", 0.15),
        ),
        # A definition asked in a user's own words rarely shares a whole term
        # with the document, so the fuzzy leg runs whether or not the exact
        # legs found anything.
        fuzzy=FuzzyRecallPolicy(trigger_under=0),
        neighbors=NeighbourPolicy(before=1, after=1, max_per_anchor=4, max_total=10),
    ),
    TASK_CONTRIBUTION: RetrievalPlan(
        task=TASK_CONTRIBUTION,
        label=TASK_LABELS[TASK_CONTRIBUTION],
        description=(
            "论文贡献：贡献声明写在引言与摘要，实验数字属于结果章；"
            "因此加重引言类章节，取 6 条以免把实验表格当成贡献。"
        ),
        final_k=6,
        expand_top_k=3,
        topic_boosts=(
            ("引言", 0.45),
            ("introduction", 0.45),
            ("contribution", 0.4),
            ("贡献", 0.4),
            ("related work", 0.2),
            ("abstract", 0.2),
            ("摘要", 0.2),
            ("conclusion", 0.1),
            ("结论", 0.1),
        ),
        neighbors=NeighbourPolicy(before=1, after=1, max_per_anchor=3, max_total=8),
    ),
    TASK_EXPERIMENT: RetrievalPlan(
        task=TASK_EXPERIMENT,
        label=TASK_LABELS[TASK_EXPERIMENT],
        description=(
            "实验结果：指标、消融与表格在实验 / 结果 / 评测章；"
            "向后扩 2 块，因为表格常在提到它的句子之后。"
        ),
        final_k=10,
        expand_top_k=6,
        topic_boosts=(
            ("experiment", 0.45),
            ("实验", 0.45),
            ("results", 0.4),
            ("结果", 0.4),
            ("evaluation", 0.3),
            ("评测", 0.3),
            ("ablation", 0.25),
            ("消融", 0.25),
            ("benchmark", 0.2),
            ("table", 0.15),
            ("表格", 0.15),
        ),
        neighbors=NeighbourPolicy(before=1, after=2, max_per_anchor=4, max_total=16),
    ),
    TASK_REPRODUCTION: RetrievalPlan(
        task=TASK_REPRODUCTION,
        label=TASK_LABELS[TASK_REPRODUCTION],
        description=(
            "复现：需要方法、实验设置、局限与附录四类证据，取 10 条并向后扩 3 块；"
            "词法权重提高到 1.2，因为超参数与数据集名是专有名词，模糊召回不总能救回来。"
        ),
        final_k=10,
        expand_top_k=6,
        lexical_weight=1.2,
        fuzzy_weight=0.8,
        topic_boosts=(
            ("method", 0.35),
            ("方法", 0.35),
            ("implementation", 0.3),
            ("实现", 0.3),
            ("experiment", 0.3),
            ("实验", 0.3),
            ("setup", 0.25),
            ("设置", 0.25),
            ("reproducib", 0.3),
            ("复现", 0.3),
            ("limitation", 0.25),
            ("局限", 0.25),
            ("appendix", 0.2),
            ("附录", 0.2),
        ),
        # "How do I reproduce this" is the task type most likely to be typed
        # with a misspelled method name, so the fuzzy leg is unconditional.
        fuzzy=FuzzyRecallPolicy(trigger_under=0),
        neighbors=NeighbourPolicy(before=1, after=3, max_per_anchor=5, max_total=20),
    ),
}


def plan_for(task: str) -> RetrievalPlan:
    """The registered plan for ``task``.

    Raises :class:`KeyError` for an unknown task rather than returning the Q&A
    plan. The fallback is the dangerous kind: a caller with a typo, or a new
    task type nobody registered, would get a strategy that answers the question
    generically and report success. 证据不足时明确说不足 -- and a wrong retrieval
    strategy is one way of being insufficient without saying so.
    """
    key = str(task or "").strip().lower()
    try:
        return PLANS[key]
    except KeyError:
        raise KeyError(
            f"未知的检索任务类型：{task!r}。已注册：{', '.join(sorted(PLANS))}。"
            "新增任务类型请用 register_plan() 注册 RetrievalPlan。"
        ) from None


def register_plan(plan: RetrievalPlan, *, replace_existing: bool = False) -> RetrievalPlan:
    """Register a sixth (or seventh) task type. The extension point, on purpose.

    The five in :data:`PLANS` are the plan's, and a product with an evidence
    panel will want more than five: "what does table 3 say" is neither an
    experiment question nor a contribution question. Registering is one call
    and touches no retrieval code, which is the test of whether the plan was
    really a value and not five branches.
    """
    key = str(plan.task or "").strip().lower()
    if key in PLANS and not replace_existing:
        raise ValueError(f"检索任务类型 {key!r} 已注册；如需覆盖请传 replace_existing=True。")
    PLANS[key] = plan
    return plan


# --------------------------------------------------------------------------
# The retrieval record
# --------------------------------------------------------------------------

#: Why a result set is empty. Named, because "no evidence" has several causes a
#: user must be told apart: a filter that excluded everything is a product bug,
#: and "the paper does not say this" is a finding.
EMPTY_NO_CANDIDATES = "no_candidates"
EMPTY_FILTERED = "filtered_out"
EMPTY_BEYOND_THRESHOLD = "beyond_threshold"
EMPTY_LEGS_SKIPPED = "legs_skipped"
EMPTY_OK = "ok"

#: Why a candidate was fused but not returned. Recorded rather than implied,
#: because "the answer cited the second-best passage" and "the right passage was
#: never a candidate" are different bugs with different fixes.
DROP_BELOW_FINAL_K = "below_final_k"
DROP_NOT_IN_INDEX = "chunk_missing_from_metadata"


@dataclass(slots=True, frozen=True)
class CandidateRecord:
    """One fused candidate and its whole history, winner or not."""

    chunk_id: str
    document_id: str = ""
    rrf_score: float = 0.0
    rank: int = 0
    legs: tuple[LegHit, ...] = ()
    topic_boost: float = 1.0
    matched_topic: str = ""
    final_score: float = 0.0
    rerank_score: float | None = None
    rerank_rank: int | None = None
    selected: bool = False
    reason: str = ""

    @property
    def legs_found(self) -> tuple[str, ...]:
        """The legs that returned this candidate, in the declared leg order.

        On the record as well as on :class:`FusionEntry`, because the record is
        what a person debugging a bad answer actually reads, and "no leg
        returned this chunk" is the single most useful sentence in it.
        """
        return tuple(leg.leg for leg in self.legs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "rrf_score": self.rrf_score,
            "rank": self.rank,
            "legs": [
                {"leg": leg.leg, "rank": leg.rank, "score": leg.score} for leg in self.legs
            ],
            "topic_boost": self.topic_boost,
            "matched_topic": self.matched_topic,
            "final_score": self.final_score,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
            "selected": self.selected,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateRecord:
        rerank_score = payload.get("rerank_score")
        rerank_rank = payload.get("rerank_rank")
        return cls(
            chunk_id=str(payload.get("chunk_id") or ""),
            document_id=str(payload.get("document_id") or ""),
            rrf_score=float(payload.get("rrf_score") or 0.0),
            rank=int(payload.get("rank") or 0),
            legs=tuple(
                LegHit(
                    leg=str(item.get("leg") or ""),
                    rank=int(item.get("rank") or 0),
                    score=float(item.get("score") or 0.0),
                )
                for item in payload.get("legs") or ()
                if isinstance(item, Mapping)
            ),
            topic_boost=float(payload.get("topic_boost", 1.0)),
            matched_topic=str(payload.get("matched_topic") or ""),
            final_score=float(payload.get("final_score") or 0.0),
            rerank_score=None if rerank_score is None else float(rerank_score),
            rerank_rank=None if rerank_rank is None else int(rerank_rank),
            selected=bool(payload.get("selected")),
            reason=str(payload.get("reason") or ""),
        )


@dataclass(slots=True, frozen=True)
class ExpansionRecord:
    """One added passage and the anchor it was added for."""

    anchor_chunk_id: str
    expanded_chunk_id: str
    relation: str
    evidence_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_chunk_id": self.anchor_chunk_id,
            "expanded_chunk_id": self.expanded_chunk_id,
            "relation": self.relation,
            "evidence_id": self.evidence_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExpansionRecord:
        return cls(
            anchor_chunk_id=str(payload.get("anchor_chunk_id") or ""),
            expanded_chunk_id=str(payload.get("expanded_chunk_id") or ""),
            relation=str(payload.get("relation") or ""),
            evidence_id=str(payload.get("evidence_id") or ""),
        )


@dataclass(slots=True, frozen=True)
class ExpandedEvidence:
    """A passage included for context rather than because it was retrieved.

    A separate type from :class:`~paperwhisperer.ingestion.schema.Evidence` on
    purpose. The two must never be counted together: a recall measurement that
    included expansions would be measuring its own expansion policy, and a UI
    that rendered them identically would be showing a reader a passage the
    system did not find as though it had. :attr:`relation` and
    :attr:`anchor_chunk_ids` are therefore not decoration -- they are the reason
    this value is distinguishable from an answer's own evidence.
    """

    evidence: Evidence
    relation: str
    anchor_chunk_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": self.evidence.to_dict(),
            "relation": self.relation,
            "anchor_chunk_ids": list(self.anchor_chunk_ids),
        }


@dataclass(slots=True, frozen=True)
class RetrievalRecord:
    """Everything one retrieval did, in a form that survives a round trip.

    The shape follows one rule: *a record that shows only the winners cannot
    debug a bad answer*, so this carries every fused candidate, its per-leg
    ranks and contributions, whether it was selected and if not why, the final
    evidence, every expansion, and the degradations.
    :attr:`candidates` is deliberately not trimmed to the selected set.

    :attr:`record_id` is content-addressed over the query, the plan signature
    and the outcome, so re-running the same retrieval over an unchanged index
    produces the same id -- which is what makes "was this answer reproducible?"
    answerable by comparing two rows instead of by re-running the pipeline. It
    deliberately excludes any timestamp or duration: a record whose id changed
    on every run could not answer "did the same question return the same
    evidence yesterday", which is exactly what T4.3's regression gate asks.
    """

    record_id: str
    query: str
    task: str
    plan_signature: str
    plan: dict[str, Any]
    rrf_k: int
    #: The plan's leg weights, not the weights that actually applied. A leg the
    #: plan weighted at 1.0 that abstained -- a dense leg with no opinion to
    #: offer, a lexical leg that found nothing -- is listed here with the weight
    #: it was *given*, and the reason it did not use it is in
    #: :attr:`degradations`. Reporting the effective weight instead would make a
    #: record unreadable against the plan that produced it, which is the one
    #: thing a stored retrieval record exists to be.
    leg_weights: dict[str, float]
    candidate_depth: int
    candidates: tuple[CandidateRecord, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    expansions: tuple[ExpansionRecord, ...] = ()
    empty_reason: str = EMPTY_OK
    degradations: tuple[str, ...] = ()
    rerank: dict[str, Any] = field(default_factory=dict)
    query_forms: tuple[str, ...] = ()
    documents_considered: tuple[str, ...] = ()

    @property
    def insufficient_evidence(self) -> bool:
        """Whether this retrieval found nothing, however many legs ran."""
        return not self.evidence

    def to_dict(self) -> dict[str, Any]:
        """The stored form. JSON-safe and lossless, in both directions."""
        return {
            "record_id": self.record_id,
            "query": self.query,
            "task": self.task,
            "plan_signature": self.plan_signature,
            "plan": dict(self.plan),
            "rrf_k": self.rrf_k,
            "leg_weights": dict(self.leg_weights),
            "candidate_depth": self.candidate_depth,
            "candidates": [item.to_dict() for item in self.candidates],
            "evidence": [item.to_dict() for item in self.evidence],
            "expansions": [item.to_dict() for item in self.expansions],
            "empty_reason": self.empty_reason,
            "degradations": list(self.degradations),
            "rerank": dict(self.rerank),
            "query_forms": list(self.query_forms),
            "documents_considered": list(self.documents_considered),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RetrievalRecord:
        return cls(
            record_id=str(payload.get("record_id") or ""),
            query=str(payload.get("query") or ""),
            task=str(payload.get("task") or ""),
            plan_signature=str(payload.get("plan_signature") or ""),
            plan=dict(payload.get("plan") or {}),
            rrf_k=int(payload.get("rrf_k") or DEFAULT_RRF_K),
            leg_weights=dict(payload.get("leg_weights") or {}),
            candidate_depth=int(payload.get("candidate_depth") or 0),
            candidates=tuple(
                CandidateRecord.from_dict(item)
                for item in payload.get("candidates") or ()
                if isinstance(item, Mapping)
            ),
            evidence=tuple(
                Evidence.from_dict(dict(item))
                for item in payload.get("evidence") or ()
                if isinstance(item, Mapping)
            ),
            expansions=tuple(
                ExpansionRecord.from_dict(item)
                for item in payload.get("expansions") or ()
                if isinstance(item, Mapping)
            ),
            empty_reason=str(payload.get("empty_reason") or EMPTY_OK),
            degradations=tuple(str(item) for item in payload.get("degradations") or ()),
            rerank=dict(payload.get("rerank") or {}),
            query_forms=tuple(str(item) for item in payload.get("query_forms") or ()),
            documents_considered=tuple(
                str(item) for item in payload.get("documents_considered") or ()
            ),
        )


@dataclass(slots=True, frozen=True)
class RetrievalResult:
    """What a retrieval produced: evidence, context, and the record of both.

    ``evidence`` and ``expansions`` are separate lists rather than one merged
    list because they mean different things to every consumer downstream:
    T1.7 builds claims from ``evidence`` and must not cite a passage the
    retriever *placed* rather than found, while a UI reading the answer wants
    ``expansions`` as context beside the citation. Merging them behind a flag
    would put the burden of that distinction on every reader, and the first one
    to forget would produce a fabricated citation -- precisely the failure the
    whole Phase 0 canonical schema was built to prevent.
    """

    evidence: tuple[Evidence, ...]
    expansions: tuple[ExpandedEvidence, ...]
    record: RetrievalRecord

    @property
    def insufficient_evidence(self) -> bool:
        """The required outcome for a question the corpus cannot answer."""
        return not self.evidence

    @property
    def empty_reason(self) -> str:
        return self.record.empty_reason

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.evidence)

    def expansions_for(self, chunk_id: str) -> tuple[ExpandedEvidence, ...]:
        """The context passages attached to one anchor, in expansion order."""
        key = str(chunk_id)
        return tuple(item for item in self.expansions if key in item.anchor_chunk_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": [item.to_dict() for item in self.evidence],
            "expansions": [item.to_dict() for item in self.expansions],
            "record": self.record.to_dict(),
            "insufficient_evidence": self.insufficient_evidence,
            "empty_reason": self.record.empty_reason,
        }


# --------------------------------------------------------------------------
# The embedding port
# --------------------------------------------------------------------------


@runtime_checkable
class QueryEmbedder(Protocol):
    """The two things the dense leg needs from an embedding provider.

    A local ``Protocol`` rather than an import of
    :class:`paperwhisperer.llm.embeddings.EmbeddingGateway`, for the same reason
    :mod:`paperwhisperer.rag.rerank` does not import the chat client: the
    retriever must not gain a dependency on an HTTP SDK in order to read a
    vector. The concrete gateway satisfies this protocol as it stands, and a
    deterministic local embedder costs two methods -- which is what makes the
    dense leg testable offline, given that this environment's endpoint has no
    ``/v1/embeddings`` route at all.
    """

    @property
    def index_version(self) -> str:
        """Provider / model / dimension identity, recorded on every retrieval."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """One query vector, in the same space as the stored vectors."""
        ...


class _Auto:
    """Sentinel type for "no reranker was injected; resolve it from config"."""

    __slots__ = ()


#: Default for :attr:`HybridRetriever.reranker`. Distinguishing "not supplied"
#: from ``None`` (``explicitly off``) is what lets a test assert that rerank is
#: off by default without having to unset the environment first.
AUTO = _Auto()


# --------------------------------------------------------------------------
# Internal value carried from fusion to the record
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _Ranked:
    """A candidate after every post-fusion stage, with all of them recorded.

    Separate from :class:`FusionEntry` because rerank scores are not fusion
    scores. Overwriting ``FusionEntry.score`` with a cross-encoder's relevance
    would leave the record unable to say "this was ranked 6th by both retrievers
    and promoted by the reranker" -- which is the single most useful sentence
    there is when a rerank is switched on and the results change.
    """

    chunk_id: str
    rrf_score: float
    topic_boost: float = 1.0
    topic: str = ""
    final_score: float = 0.0
    rerank_score: float | None = None
    rerank_rank: int | None = None
    legs: tuple[LegHit, ...] = ()


@dataclass(slots=True, frozen=True)
class _DocIndex:
    """One document's term index for the fuzzy leg.

    Two maps rather than one, and the reason is a performance shape rather than
    a preference. ``by_gram`` answers "which terms could possibly be near-missed
    for this query" in one pass over the query's trigrams. ``by_chunk`` then
    answers "does *this* chunk own any of them" without re-tokenizing the chunk,
    which is what keeps the leg's work proportional to the chunks that share a
    trigram with the query rather than to the corpus. An index that answered
    only the first question would force a second full tokenization of every
    chunk it touched.

    ``chunk_count`` is the invalidation key; see
    :meth:`HybridRetriever._trigrams_for` for what it does and does not catch.
    """

    chunk_count: int
    by_gram: dict[str, frozenset[str]]
    by_chunk: dict[str, frozenset[str]]

    def matching_tokens(self, chunk_id: str, needles: frozenset[str]) -> set[str]:
        """The terms of ``chunk_id`` that share a trigram with ``needles``."""
        allowed: set[str] = set()
        for gram in needles:
            allowed |= self.by_gram.get(gram, frozenset())
        return allowed & self.by_chunk.get(chunk_id, frozenset())


#: Cached per-document trigram indexes. Bounded, because the fuzzy leg runs on
#: every under-filled query and an unbounded cache is a leak sized by the user's
#: library. Cleared wholesale rather than evicted per entry: the indexes are
#: cheap to rebuild, and an LRU here would need a second structure to stay
#: correct with the lock.
TRIGRAM_CACHE_LIMIT = 32

#: Candidates hydrated from the metadata store before topic boosts and evidence
#: are built. Bounds the per-retrieval read: the stores rank cheaply, but a
#: heading path and a locator need the chunk behind the id.
HYDRATE_LIMIT = 160



class HybridRetriever:
    """Dense + lexical + fuzzy, fused by rank, reranked optionally, always recorded.

    Construction resolves three things once, so :meth:`retrieve` is a function of
    its arguments and the indexes: the embedding provider, the rerank provider,
    and the trigram cache. The rerank provider is resolved with
    :func:`~paperwhisperer.rag.rerank.build_reranker` unless one is injected or
    ``None`` is passed explicitly; ``None`` means off, which is the default, and
    the reason this class works on a checkout with no ``RERANK_ENABLED`` in it
    and no network.

    The instance is safe to share across threads. It caches one thing -- the
    per-document trigram index behind the fuzzy leg -- behind a lock, and
    nothing else: no per-query state, no "last result", nothing a second request
    could observe.
    """

    def __init__(
        self,
        bundle: StoreBundle,
        *,
        embeddings: QueryEmbedder | None = None,
        reranker: Any = AUTO,
        settings: Any = None,
    ) -> None:
        self.bundle = bundle
        self.embeddings = embeddings
        self.reranker = build_reranker(settings) if reranker is AUTO else reranker
        self._trigram_cache: dict[str, _DocIndex] = {}
        self._trigram_lock = threading.Lock()

    # -- public ------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        plan: RetrievalPlan | None = None,
        task: str | None = None,
        document_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Run one retrieval and return its evidence, its context and its record.

        ``plan`` and ``task`` are two ways to say the same thing and only one
        may be used: a plan already names its task, and :func:`plan_for` raises
        on an unknown name rather than guessing, so there is no precedence rule
        to get wrong.

        The call **never raises for an empty result**. A question the corpus
        cannot answer is a legitimate answer, and the product's stated rule is
        that an insufficient-evidence answer says so. What it does raise for is
        a *misconfiguration* -- an unknown task, a plan asking for fewer
        candidates than results -- because those are bugs rather than findings.
        """
        resolved = self._resolve_plan(plan=plan, task=task)
        final_k = int(top_k) if top_k else int(resolved.final_k)
        scope = self._resolve_scope(resolved, document_ids)
        degradations: list[str] = []
        if resolved.k_outside_safe_range():
            degradations.append(
                f"RetrievalPlan.rrf_k={resolved.rrf_k} 落在 RRF 稳定区间"
                f"（{RRF_SAFE_RANGE[0]}–{RRF_SAFE_RANGE[1]}）之外，融合排序可能对 k 敏感。"
            )

        if scope is not None and not scope:
            return self._empty_result(resolved, query, degradations, EMPTY_FILTERED, scope)
        forms = lexical_query_forms(query)
        if not any(form.strip() for form in forms):
            return self._empty_result(
                resolved, query, degradations, EMPTY_LEGS_SKIPPED, scope
            )

        owner: dict[str, str] = {}
        rankings: dict[str, list[str]] = {leg: [] for leg in LEG_ORDER}
        scores: dict[str, dict[str, float]] = {leg: {} for leg in LEG_ORDER}

        self._dense_leg(
            forms[0], resolved, scope, rankings, scores, owner, degradations
        )
        self._lexical_leg(forms, resolved, scope, rankings, scores, owner, degradations)
        self._fuzzy_leg(forms, resolved, scope, rankings, scores, owner, degradations)

        chunks = self._hydrate(rankings, owner, resolved, degradations)
        entries = reciprocal_rank_fusion(rankings, k=resolved.rrf_k, weights=resolved.leg_weights())
        ranked = self._apply_topics(entries, chunks, resolved)
        ranked, rerank_info = self._rerank(ranked, chunks, query, resolved, degradations)
        anchors, dropped = self._select(ranked, chunks, final_k)

        evidence: list[Evidence] = []
        parsers = self._parser_names(owner)
        for item in anchors:
            chunk = chunks.get(item.chunk_id)
            if chunk is None:
                continue
            evidence.append(
                _evidence_for(
                    chunk, item.final_score, resolved, parsers.get(owner.get(item.chunk_id, ""), "")
                )
            )
        expansions, expansion_records = self._expand(
            anchors, chunks, owner, resolved, parsers, degradations
        )
        selected_ids = {item.chunk_id for item in evidence}
        # Recorded *after* the ids exist and from the ids' own documents, so
        # "which plan can this document's evidence have been minted under" is
        # answered by the retriever that minted them rather than by a registry
        # the caller was supposed to remember to update.
        record_plan_signature(
            self.bundle, {owner.get(item.chunk_id, "") for item in anchors}, resolved.signature()
        )
        records = tuple(
            self._candidate_record(item, owner.get(item.chunk_id, ""), selected_ids, dropped)
            for item in ranked
        )
        record = self._build_record(
            query=query,
            forms=forms,
            plan=resolved,
            scope=scope,
            candidates=records,
            evidence=evidence,
            expansion_records=expansion_records,
            degradations=degradations,
            rerank=rerank_info,
            documents_seen=sorted(set(owner.values())),
        )
        return RetrievalResult(
            evidence=tuple(evidence), expansions=tuple(expansions), record=record
        )

    def describe(self) -> dict[str, Any]:
        """Which legs and providers this instance will actually use.

        The fuzzy entry reports the *default* policy rather than a boolean,
        because a plan can switch it off and there is no single answer to
        "is fuzzy on". The per-retrieval truth is in the record: a leg that
        abstained says so in :attr:`RetrievalRecord.degradations`.
        """
        return {
            "dense": {
                "enabled": self.embeddings is not None,
                "index_version": getattr(self.embeddings, "index_version", ""),
                "tie_band": DENSE_TIE_BAND,
            },
            "lexical": {"engine": self.bundle.capabilities.lexical_engine},
            "fuzzy": {
                "min_similarity": FuzzyRecallPolicy().min_similarity,
                "min_token_chars": MIN_FUZZY_TOKEN_CHARS,
            },
            "rerank": {
                "enabled": self.reranker is not None,
                "model": getattr(self.reranker, "model", ""),
            },
            "tier": self.bundle.tier,
            "backend": self.bundle.backend,
        }

    # -- plan / scope ------------------------------------------------------

    @staticmethod
    def _resolve_plan(
        *, plan: RetrievalPlan | None, task: str | None
    ) -> RetrievalPlan:
        if plan is not None and task is not None:
            raise ValueError("retrieve() 接受 plan 或 task 其中之一，不要同时传。")
        if plan is not None:
            return plan
        return plan_for(task if task is not None else TASK_QA)

    @staticmethod
    def _resolve_scope(
        plan: RetrievalPlan, document_ids: Sequence[str] | None
    ) -> tuple[str, ...] | None:
        """The pre-fusion filter: a document set, or ``None`` for the whole store.

        A caller's ``document_ids`` replaces the plan's rather than being merged
        with it. Merging would mean a plan requiring two documents plus a caller
        asking for a third silently becoming a three-document search, and a
        search that reaches further than either author intended is a leak: on a
        multi-tenant Hosted deployment that is one user's document set
        intersecting another's.
        """
        if document_ids is not None:
            return tuple(str(item) for item in document_ids)
        if plan.required_document_ids is not None:
            return tuple(str(item) for item in plan.required_document_ids)
        return None

    # -- legs --------------------------------------------------------------

    def _dense_leg(
        self,
        query: str,
        plan: RetrievalPlan,
        scope: tuple[str, ...] | None,
        rankings: dict[str, list[str]],
        scores: dict[str, dict[str, float]],
        owner: dict[str, str],
        degradations: list[str],
    ) -> None:
        """The dense leg, and a recorded reason when it could not run.

        Two ways it can be absent, and they are not the same thing. No
        embedding provider at all is a configuration fact, reported once. A
        provider that raises is a *failure* of a leg that should have run, and
        it degrades instead of raising because of the product rule: a retrieval
        that cannot embed is still able to answer from what it has, provided it
        says that is what happened. An unanswered question is a failure; a
        question answered from the lexical leg alone, with the record showing
        the dense leg never ran, is a degradation.
        """
        if plan.dense_weight == 0.0:
            degradations.append("RetrievalPlan.dense_weight=0，本次检索关闭 dense 通道。")
            return
        if self.embeddings is None:
            degradations.append("未配置 embedding provider，dense 通道未参与本次检索。")
            return
        try:
            vector = self.embeddings.embed_query(query)
            hits = self.bundle.vectors.search(
                vector, plan.candidate_depth, document_ids=scope
            )
        except Exception as exc:  # noqa: BLE001 - a leg must not take the query down
            logger.warning("dense 通道失败，本次检索降级为词法检索。", exc_info=True)
            degradations.append(f"dense 通道失败，已降级为词法检索：{exc}")
            return
        if hits and plan.dense_tie_band > 0.0:
            window = hits[: max(2, int(DENSE_TIE_WINDOW))]
            spread = float(window[0].score) - float(window[-1].score)
            if spread < plan.dense_tie_band:
                degradations.append(
                    f"dense 通道前 {len(window)} 个候选的余弦相似度跨度仅 {spread:.4f}"
                    f"（小于 {plan.dense_tie_band}），对本次 query 没有区分力，该通道弃权。"
                )
                return
        for hit in hits:
            _add(
                rankings, scores, owner, LEG_DENSE,
                hit.chunk_id, getattr(hit, "document_id", ""), float(hit.score),
            )

    def _lexical_leg(
        self,
        forms: Sequence[str],
        plan: RetrievalPlan,
        scope: tuple[str, ...] | None,
        rankings: dict[str, list[str]],
        scores: dict[str, dict[str, float]],
        owner: dict[str, str],
        degradations: list[str],
    ) -> None:
        """The exact leg, over every query form. Never optional, never a fallback.

        The best rank any form achieved is the chunk's rank in the leg, not the
        concatenation of the forms' result lists. Concatenation would let one
        chunk occupy two slots and push a different chunk's only real position
        down, which is how a search for a DOI that also matched nothing else
        returns eight near-identical snippets of the same paper.
        """
        if plan.lexical_weight == 0.0:
            degradations.append("RetrievalPlan.lexical_weight=0，本次检索关闭词法通道。")
            return
        best: dict[str, tuple[int, float]] = {}
        order: list[str] = []
        for form in forms:
            try:
                hits = self.bundle.lexical.search(
                    form, plan.candidate_depth, document_ids=scope
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("词法通道失败。", exc_info=True)
                degradations.append(f"词法通道失败（query 形式 {form!r}）：{exc}")
                continue
            for position, hit in enumerate(hits, start=1):
                if hit.score < plan.min_lexical_score:
                    continue
                if hit.document_id:
                    owner[str(hit.chunk_id)] = str(hit.document_id)
                previous = best.get(hit.chunk_id)
                if previous is None:
                    order.append(hit.chunk_id)
                    best[hit.chunk_id] = (position, float(hit.score))
                elif position < previous[0]:
                    best[hit.chunk_id] = (position, float(hit.score))
        for chunk_id in sorted(order, key=lambda key: (best[key][0], key)):
            _add(
                rankings, scores, owner, LEG_LEXICAL,
                chunk_id, owner.get(chunk_id, ""), best[chunk_id][1],
            )

    def _fuzzy_leg(
        self,
        forms: Sequence[str],
        plan: RetrievalPlan,
        scope: tuple[str, ...] | None,
        rankings: dict[str, list[str]],
        scores: dict[str, dict[str, float]],
        owner: dict[str, str],
        degradations: list[str],
    ) -> None:
        """The near-miss leg: trigram similarity over whole terms, in Python.

        Why it is here and not in the store is the argument in the module
        docstring, and it is worth restating because the alternative is
        tempting: the Hosted tier has ``pg_trgm`` installed and indexed, and a
        ``chunk_id %% similarity`` query would be one line of SQL. That line
        would make a recall capability exist on one tier and not the other --
        the single failure the store contract was written to prevent, and the
        reason T1.1 refused to ship it.

        Reading chunk text back through the metadata store is the price, and it
        is a real one: a corpus-wide widening is linear in the corpus. It is
        paid only when the exact legs under-filled
        (:attr:`FuzzyRecallPolicy.trigger_under`), capped twice over (documents
        and chunks), and truncated *explicitly* -- with a recorded degradation --
        rather than silently, because a fuzzy leg that stopped early and one
        that found nothing are different facts.

        **What could not be verified here.** This algorithm reads no SQL and
        names no engine, so it is identical on both tiers by construction -- but
        there is no PostgreSQL in this environment, so the Hosted tier's
        behaviour is an argument rather than a measurement. What a Hosted-tier
        test would have to check is not the scoring, which is shared Python, but
        the read-back: that ``list_chunks`` returns the same ``Chunk`` objects
        the SQLite tier does, in the same order, for the same document. That
        belongs to whoever runs the tier suite.
        """
        policy = plan.fuzzy
        if not policy.enabled or plan.fuzzy_weight == 0.0:
            return
        terms = [
            term
            for term in _fuzzy_query_terms(forms)
            if len(term) >= policy.min_query_chars
        ]
        if not terms:
            return
        exact = {chunk_id for leg in (LEG_DENSE, LEG_LEXICAL) for chunk_id in rankings[leg]}
        # ``trigger_under == 0`` is the documented "always" setting, so the test
        # is written as ``> 0`` rather than as ``len(exact) >= trigger_under`` --
        # the latter makes 0 mean "never", which is the opposite of what the
        # plans that set it (概念, 复现) are asking for.
        if policy.trigger_under > 0 and len(exact) >= policy.trigger_under:
            return
        for term in terms:
            hits = self._fuzzy_candidates(
                term, plan, scope, policy, owner, exact, degradations
            )
            for document, chunk_id, score in hits:
                _add(rankings, scores, owner, LEG_FUZZY, chunk_id, document, score)

    def _fuzzy_candidates(
        self,
        term: str,
        plan: RetrievalPlan,
        scope: tuple[str, ...] | None,
        policy: FuzzyRecallPolicy,
        owner: Mapping[str, str],
        exact: set[str],
        degradations: list[str],
    ) -> list[tuple[str, float]]:
        """Score every chunk sharing a trigram with ``term``, best first.

        Built over an inverted trigram index per document, so the work is
        proportional to the chunks that could plausibly match rather than to the
        corpus. The index is cached per document; the cache is bounded and
        invalidated by chunk count, and nothing else about the retriever is
        cached, because a cache keyed on anything mutable is a way for two runs
        to disagree.
        """
        needles = _trigrams(term)
        if not needles:
            return []
        # (score, chunk_id, document_id) together on purpose: attributing a
        # hit to whichever document happened to be scanned last would put the
        # wrong document on the candidate, and the document id is what the
        # hydration and expansion stages use to read the chunk back.
        scored: list[tuple[float, str, str]] = []
        scanned = 0
        for document in self._fuzzy_documents(scope, policy, owner, exact):
            chunks = self._document_chunks(document, degradations)
            index = self._trigrams_for(document, chunks)
            for chunk in chunks:
                if scanned >= policy.max_scanned_chunks:
                    degradations.append(
                        f"模糊召回扫描到 {scanned} 个 chunk 达到上限"
                        f"（max_scanned_chunks={policy.max_scanned_chunks}），结果可能不完整。"
                    )
                    scored.sort(key=lambda item: (-item[0], item[1]))
                    return [
                        (doc, cid, sc) for sc, cid, doc in scored[: plan.candidate_depth]
                    ]
                scanned += 1
                best = 0.0
                for token in index.matching_tokens(chunk.chunk_id, needles):
                    if len(token) < MIN_FUZZY_TOKEN_CHARS:
                        continue
                    best = max(best, _token_similarity(term, token))
                    if best >= 0.99:
                        break
                if best >= policy.min_similarity:
                    scored.append((best, chunk.chunk_id, document))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(doc, cid, sc) for sc, cid, doc in scored[: plan.candidate_depth]]

    def _fuzzy_documents(
        self,
        scope: tuple[str, ...] | None,
        policy: FuzzyRecallPolicy,
        owner: Mapping[str, str],
        exact: set[str],
    ) -> list[str]:
        """Which documents the fuzzy leg may look in, in a stable order.

        Documents the exact legs already put a candidate in come first, so a
        query that found something in two documents gets its widening inside
        those two before the widening wanders into a third. Without that rule a
        fuzzy hit from an unrelated document could outrank a lexical hit from
        the document the question was about -- the exact failure fuzzy recall
        exists to avoid, introduced by the leg meant to fix recall.
        """
        if scope is not None:
            return list(scope)[: policy.max_documents]
        rows = self.bundle.metadata.list_documents(limit=policy.max_documents, offset=0)
        names = [str(row.document_id) for row in rows]
        relevant = {owner[chunk_id] for chunk_id in exact if chunk_id in owner}
        return sorted(names, key=lambda name: (0 if name in relevant else 1, name))[
            : policy.max_documents
        ]

    def _document_chunks(self, document_id: str, degradations: list[str]) -> list[Chunk]:
        if not document_id:
            return []
        try:
            return self.bundle.metadata.list_chunks(document_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取文档 chunk 失败：%s", document_id, exc_info=True)
            degradations.append(f"读取文档 {document_id} 的 chunk 失败：{exc}")
            return []

    def _trigrams_for(self, document_id: str, chunks: Sequence[Chunk]) -> _DocIndex:
        """The document's term index for the fuzzy leg, rebuilt when it is stale.

        Staleness is judged by the chunk count, which is the only thing about a
        re-index this index can see. Rebuilding on every call would be correct
        and would make the leg quadratic in the corpus; never rebuilding would
        return hits for passages that no longer exist. Counting is the cheapest
        honest compromise, and a re-index that re-cuts a section *without*
        changing the chunk count is a re-index this build cannot see -- stated
        here rather than left as a latent bug for someone to find at 3am.
        """
        with self._trigram_lock:
            cached = self._trigram_cache.get(document_id)
            if cached is not None and cached.chunk_count == len(chunks):
                return cached
            if len(self._trigram_cache) >= TRIGRAM_CACHE_LIMIT:
                self._trigram_cache.clear()
            by_gram: dict[str, set[str]] = {}
            by_chunk: dict[str, frozenset[str]] = {}
            for chunk in chunks:
                terms = frozenset(
                    token
                    for token in scoring.tokenize(chunk.text)
                    if len(token) >= MIN_FUZZY_TOKEN_CHARS
                )
                by_chunk[chunk.chunk_id] = terms
                for token in terms:
                    for gram in _trigrams(token):
                        by_gram.setdefault(gram, set()).add(token)
            index = _DocIndex(
                chunk_count=len(chunks),
                by_gram={gram: frozenset(items) for gram, items in by_gram.items()},
                by_chunk=by_chunk,
            )
            self._trigram_cache[document_id] = index
            return index

    # -- post-fusion stages ------------------------------------------------

    def _hydrate(
        self,
        rankings: Mapping[str, Sequence[str]],
        owner: Mapping[str, str],
        plan: RetrievalPlan,
        degradations: list[str],
    ) -> dict[str, Chunk]:
        """Fetch the chunk behind each candidate, once, up to a bound.

        Needed for three things and expensive for none of them in particular:
        the heading path a topic boost matches, the locator and verbatim text an
        :class:`~paperwhisperer.ingestion.schema.Evidence` is built from, and the
        ``order``/``part_index`` neighbour expansion walks. It is one
        ``list_chunks`` per *document* rather than one ``get_chunk`` per
        candidate, so a three-candidate fusion out of one document costs one
        read. A :class:`~paperwhisperer.ingestion.schema.Chunk` carries no
        document id, which is why the map is built from the legs' hits instead.
        """
        wanted: list[str] = []
        for leg in LEG_ORDER:
            for chunk_id in rankings.get(leg) or ():
                if chunk_id not in wanted:
                    wanted.append(chunk_id)
        if not wanted:
            return {}
        truncated = len(wanted) > HYDRATE_LIMIT
        wanted = wanted[:HYDRATE_LIMIT]
        by_document: dict[str, set[str]] = {}
        for chunk_id in wanted:
            document_id = owner.get(chunk_id, "")
            if not document_id:
                continue
            by_document.setdefault(document_id, set()).add(chunk_id)
        chunks: dict[str, Chunk] = {}
        for document_id, needed in by_document.items():
            for chunk in self._document_chunks(document_id, degradations):
                if chunk.chunk_id in needed:
                    chunks[chunk.chunk_id] = chunk
        if len(chunks) < len(wanted):
            degradations.append(
                f"有 {len(wanted) - len(chunks)} 个候选在元数据存储中找不到对应 chunk"
                "（索引与元数据不一致），这些候选已从结果中剔除。"
            )
        if truncated:
            degradations.append(
                f"候选数超过 hydration 上限（{HYDRATE_LIMIT}），超出部分未展开证据。"
            )
        return chunks

    def _apply_topics(
        self,
        entries: Sequence[FusionEntry],
        chunks: Mapping[str, Chunk],
        plan: RetrievalPlan,
    ) -> tuple[_Ranked, ...]:
        """Multiply the fused score by the plan's preference for the section.

        Multiplicative, not additive, and the reason is that it cannot change
        the units: an additive bonus in fused-score units would have to be
        rescaled whenever ``k`` or the leg weights changed, so the *same* plan
        would mean something different after an unrelated edit. A multiplier is
        a factor on "how much this candidate is worth", which is what a
        preference actually is.

        It also cannot create. A chunk no leg returned is not in ``entries`` and
        so cannot be promoted into the result by any amount of preference. A
        plan that prefers the experiments section will not return an
        experiments section that retrieval could not find -- it will return
        nothing, and say so. That is the difference between this and a filter,
        and it is what stops a preference from quietly answering a different
        question.
        """
        ranked: list[_Ranked] = []
        for entry in entries:
            chunk = chunks.get(entry.chunk_id)
            factor, keyword = plan.boost_for(chunk.heading_path) if chunk else (1.0, "")
            ranked.append(
                _Ranked(
                    chunk_id=entry.chunk_id,
                    rrf_score=entry.score,
                    legs=entry.legs,
                    topic_boost=factor,
                    topic=keyword,
                    final_score=entry.score * factor,
                )
            )
        ranked.sort(key=lambda item: (-item.final_score, item.chunk_id))
        return tuple(ranked)

    def _rerank(
        self,
        ranked: Sequence[_Ranked],
        chunks: Mapping[str, Chunk],
        query: str,
        plan: RetrievalPlan,
        degradations: list[str],
    ) -> tuple[tuple[_Ranked, ...], dict[str, Any]]:
        """The optional second pass. Off by default; degrading when it fails.

        Rerank only the candidates that were already fused, never a wider set: a
        cross-encoder is the most expensive thing in the pipeline and its whole
        claim is "reorder what retrieval proposed". Feeding it passages the
        retrievers rejected would be a different -- and much more expensive --
        product.

        On failure the fusion order is kept *and* the reason recorded. The
        tempting alternative is to fall through to the next-best ordering
        silently; that hides a broken model behind a plausible answer and makes
        a regression unreproducible, because the regression only reproduces
        while the provider is up.
        """
        info: dict[str, Any] = {"applied": False, "model": "", "error": ""}
        if plan.rerank is False:
            info["error"] = "RetrievalPlan.rerank=False，本次检索显式关闭 rerank。"
            return tuple(ranked), info
        provider = self.reranker
        if provider is None:
            return tuple(ranked), info
        candidates = [item for item in ranked if item.chunk_id in chunks]
        if not candidates:
            return tuple(ranked), info
        info["model"] = str(getattr(provider, "model", "") or "")
        try:
            results = provider.rerank(
                query,
                [
                    RerankDocument(chunk_id=item.chunk_id, text=chunks[item.chunk_id].text)
                    for item in candidates
                ],
            )
        except Exception as exc:  # noqa: BLE001 - rerank is optional by design
            logger.warning("rerank 失败，保留 RRF 融合顺序。", exc_info=True)
            info["error"] = f"rerank 失败，保留 RRF 融合顺序：{exc}"
            degradations.append(info["error"])
            return tuple(ranked), info
        verdict = {item.chunk_id: (item.score, item.rank) for item in results}
        reordered = []
        for item in ranked:
            if item.chunk_id not in verdict:
                reordered.append(item)
                continue
            score, position = verdict[item.chunk_id]
            reordered.append(
                replace(
                    item,
                    final_score=float(score),
                    rerank_score=float(score),
                    rerank_rank=int(position),
                )
            )
        reordered.sort(key=lambda item: (-item.final_score, item.chunk_id))
        info["applied"] = True
        info["count"] = len(verdict)
        return tuple(reordered), info

    @staticmethod
    def _select(
        ranked: Sequence[_Ranked], chunks: Mapping[str, Chunk], final_k: int
    ) -> tuple[list[_Ranked], dict[str, str]]:
        """Take the top ``final_k`` candidates that still have a chunk behind them.

        Skipping a candidate whose chunk has gone from the metadata store is not
        an optimisation. A fused candidate with no chunk has no text, no locator
        and no heading path, so it cannot become evidence -- and returning fewer
        than ``final_k`` because two chunks were orphaned by a re-index is
        honest, where padding the list with something else would not be. The
        skipped ones are recorded with :data:`DROP_NOT_IN_INDEX`.
        """
        anchors: list[_Ranked] = []
        dropped: dict[str, str] = {}
        for item in ranked:
            if item.chunk_id not in chunks:
                dropped[item.chunk_id] = DROP_NOT_IN_INDEX
            elif len(anchors) < max(1, int(final_k)):
                anchors.append(item)
            else:
                dropped[item.chunk_id] = DROP_BELOW_FINAL_K
        return anchors, dropped

    def _expand(
        self,
        anchors: Sequence[_Ranked],
        chunks: Mapping[str, Chunk],
        owner: Mapping[str, str],
        plan: RetrievalPlan,
        parsers: Mapping[str, str],
        degradations: list[str],
    ) -> tuple[list[ExpandedEvidence], list[ExpansionRecord]]:
        """Attach bounded surrounding text to the top anchors.

        Bounded three ways, all of which matter: by
        :attr:`NeighbourPolicy.max_per_anchor` so one early hit cannot drag half
        a document into an answer, by :attr:`NeighbourPolicy.max_total` so
        ``final_k`` anchors cannot compound into an unbounded context, and by
        :attr:`RetrievalPlan.expand_top_k` so the tail of the result list -- the
        passages that were not confident enough to rank highly -- is not expanded
        at all. The third is the subtle one: expansion makes a fragment
        readable, and a passage that barely made the cut is the one whose
        readability should not be improved.

        Only passages that are *not* already anchors are expanded: expanding a
        passage the retriever already found would put it in the record twice
        with two different meanings, once found and once placed.
        """
        policy = plan.neighbors
        expansions: list[ExpandedEvidence] = []
        records: list[ExpansionRecord] = []
        if policy.max_per_anchor <= 0 or policy.max_total <= 0:
            return expansions, records
        anchor_ids = {item.chunk_id for item in anchors}
        placed: set[str] = set()
        for item in anchors[: max(0, int(plan.expand_top_k))]:
            chunk = chunks.get(item.chunk_id)
            if chunk is None:
                continue
            document_id = owner.get(item.chunk_id, "")
            for other, relation in self._neighbours(chunk, document_id, policy, degradations):
                if other.chunk_id in anchor_ids or other.chunk_id in placed:
                    continue
                if len(expansions) >= policy.max_total:
                    return expansions, records
                placed.add(other.chunk_id)
                evidence = _evidence_for(
                    other,
                    item.final_score,
                    plan,
                    parsers.get(document_id, ""),
                )
                expansions.append(
                    ExpandedEvidence(
                        evidence=evidence,
                        relation=relation,
                        anchor_chunk_ids=(item.chunk_id,),
                    )
                )
                records.append(
                    ExpansionRecord(
                        anchor_chunk_id=item.chunk_id,
                        expanded_chunk_id=other.chunk_id,
                        relation=relation,
                        evidence_id=evidence.evidence_id,
                    )
                )
        return expansions, records

    def _neighbours(
        self,
        chunk: Chunk,
        document_id: str,
        policy: NeighbourPolicy,
        degradations: list[str],
    ) -> list[tuple[Chunk, str]]:
        """The passages around ``chunk``, in reading order, already capped.

        The parent relation is checked first because a split element's other
        parts are a *stronger* continuation than the neighbouring chunk: they
        are the same paragraph, whereas the next chunk may have crossed a
        section boundary and be about something else entirely.
        """
        found: list[tuple[Chunk, str]] = []
        siblings: set[str] = set()
        siblings_of = set(chunk.element_ids) if chunk.part_count > 1 else set()
        if policy.include_parent_parts and siblings_of:
            for candidate in self._document_chunks(document_id, degradations):
                if candidate.chunk_id == chunk.chunk_id:
                    continue
                if not siblings_of & set(candidate.element_ids):
                    continue
                siblings.add(candidate.chunk_id)
                found.append((candidate, "parent"))
        low = chunk.order - max(0, int(policy.before))
        high = chunk.order + max(0, int(policy.after))
        for candidate in self._document_chunks(document_id, degradations):
            if candidate.chunk_id in siblings or candidate.chunk_id == chunk.chunk_id:
                continue
            if low <= candidate.order <= high:
                found.append((candidate, "neighbor"))
        found.sort(key=lambda item: (item[0].order, item[0].chunk_id))
        return found[: max(0, int(policy.max_per_anchor))]

    # -- values ------------------------------------------------------------

    def _parser_names(self, owner: Mapping[str, str]) -> dict[str, str]:
        """Document id -> parser name, for :attr:`Evidence.source_parser`.

        A field the schema has and nothing in this module would otherwise fill.
        One read per distinct document that produced a candidate, which is a
        count bounded by ``candidate_depth`` and not by the corpus.
        """
        names: dict[str, str] = {}
        for document_id in sorted(set(owner.values())):
            try:
                row = self.bundle.metadata.get_document(document_id)
            except Exception:  # noqa: BLE001 - provenance is not worth failing over
                row = None
            names[document_id] = str(getattr(row, "parser_name", "") or "")
        return names

    def _candidate_record(
        self,
        item: _Ranked,
        document_id: str,
        selected_ids: set[str],
        dropped: Mapping[str, str],
    ) -> CandidateRecord:
        return CandidateRecord(
            chunk_id=item.chunk_id,
            document_id=document_id,
            rrf_score=item.rrf_score,
            legs=item.legs,
            topic_boost=item.topic_boost,
            matched_topic=item.topic,
            final_score=item.final_score,
            rerank_score=item.rerank_score,
            rerank_rank=item.rerank_rank,
            selected=item.chunk_id in selected_ids,
            reason=dropped.get(item.chunk_id, ""),
        )

    def _scope_exists(self, scope: tuple[str, ...] | None) -> bool | None:
        """Whether the in-force filter names documents the store has.

        ``None`` when no filter was applied, so the caller can tell "there was
        nothing to check" from "the check came back false". Bounded by the size
        of the filter, which is a caller-supplied list and is therefore the
        caller's own size -- the one place in this module where an unbounded read
        is justified, because a search the caller did not scope is not this
        method's job.
        """
        if scope is None:
            return None
        for document_id in scope:
            try:
                if self.bundle.metadata.get_document(document_id) is not None:
                    return True
            except Exception:  # noqa: BLE001 - a read failure is a "no"
                return False
        return False

    def _build_record(
        self,
        *,
        query: str,
        forms: Sequence[str],
        plan: RetrievalPlan,
        scope: tuple[str, ...] | None,
        candidates: Sequence[CandidateRecord],
        evidence: Sequence[Evidence],
        expansion_records: Sequence[ExpansionRecord],
        degradations: Sequence[str],
        rerank: Mapping[str, Any],
        documents_seen: Sequence[str],
    ) -> RetrievalRecord:
        ranked = tuple(replace(item, rank=position) for position, item in enumerate(candidates, 1))
        signature = plan.signature()
        return RetrievalRecord(
            record_id=_record_id(query, signature, ranked, evidence, expansion_records),
            query=query,
            task=plan.task,
            plan_signature=signature,
            plan=plan.to_dict(),
            rrf_k=int(plan.rrf_k),
            leg_weights=plan.leg_weights(),
            candidate_depth=int(plan.candidate_depth),
            candidates=ranked,
            evidence=tuple(evidence),
            expansions=tuple(expansion_records),
            empty_reason=_empty_reason(
                evidence, ranked, scope_exists=self._scope_exists(scope)
            ),
            degradations=tuple(degradations),
            rerank=dict(rerank),
            query_forms=tuple(forms),
            documents_considered=tuple(documents_seen),
        )

    def _empty_result(
        self,
        plan: RetrievalPlan,
        query: str,
        degradations: Sequence[str],
        reason: str,
        scope: tuple[str, ...] | None,
    ) -> RetrievalResult:
        """The explicit empty result, with a record that says why.

        Built through the same record path as a successful retrieval so that
        "we found nothing" and "we did not run" have the same shape, and so an
        empty result is stored, logged and inspected on exactly the same terms
        as a non-empty one. There is deliberately no fallback: this method does
        not try another plan, another query or a wider scope, and a caller cannot
        ask it to.
        """
        signature = plan.signature()
        return RetrievalResult(
            evidence=(),
            expansions=(),
            record=RetrievalRecord(
                record_id=_record_id(query, signature, (), (), ()),
                query=query,
                task=plan.task,
                plan_signature=signature,
                plan=plan.to_dict(),
                rrf_k=int(plan.rrf_k),
                leg_weights=plan.leg_weights(),
                candidate_depth=int(plan.candidate_depth),
                candidates=(),
                evidence=(),
                expansions=(),
                empty_reason=reason,
                degradations=tuple(degradations),
                rerank={"applied": False, "model": "", "error": ""},
                query_forms=lexical_query_forms(query),
                documents_considered=tuple(scope or ()),
            ),
        )


# --------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------


def _evidence_for(
    chunk: Chunk, score: float, plan: RetrievalPlan, parser: str = ""
) -> Evidence:
    """One :class:`Evidence` for one chunk, at ``score``.

    The text is :attr:`~paperwhisperer.ingestion.schema.Chunk.verbatim_text`, not
    ``text``. The chunker prepends a heading line and a lead-in sentence that
    this product generated; a direct quote checked against those would "verify" a
    sentence that appears nowhere in the paper, and the citation verifier (T1.6)
    would be reporting a fabrication as a success. The separation is the reason
    that field exists.

    ``is_direct_quote`` is hard-wired ``False``. It is set by the verifier after
    a check, never by the thing that retrieved the passage, and a retriever that
    set it would make the flag mean "this was retrieved" -- which is true of
    every row in this list.

    ``unit_id`` is the chunk's first element id: a real, resolvable identifier
    from the canonical schema. Deriving a synthetic unit string from the locator
    instead would invent a vocabulary nothing else in the project can resolve.
    """
    return Evidence(
        evidence_id=_evidence_id(chunk.chunk_id, plan.signature()),
        chunk_id=chunk.chunk_id,
        text=chunk.verbatim_text or chunk.text,
        locator=chunk.locator,
        unit_id=chunk.element_ids[0] if chunk.element_ids else "",
        heading_path=chunk.heading_path,
        score=float(score),
        is_direct_quote=False,
        source_parser=parser,
    )


def _fuzzy_query_terms(forms: Sequence[str]) -> list[str]:
    """The Latin query terms the fuzzy leg will try, in order, de-duplicated.

    Only terms of at least :data:`MIN_FUZZY_TOKEN_CHARS` characters reach the
    similarity arithmetic, and a term is tried once across every form: a DOI
    pasted with and without its URL prefix produces the same terms, and running
    the leg twice over them would double its cost for no new candidate.
    """
    terms: list[str] = []
    for form in forms:
        for token in scoring.tokenize(form):
            if len(token) >= MIN_FUZZY_TOKEN_CHARS and token not in terms:
                terms.append(token)
    return terms


def _add(
    rankings: dict[str, list[str]],
    scores: dict[str, dict[str, float]],
    owner: dict[str, str],
    leg: str,
    chunk_id: str,
    document_id: str,
    score: float,
) -> None:
    """Append a hit to one leg, or record a better score for one already there.

    A repeated hit keeps its first position. The leg's ordering came from the
    store, and re-appending a chunk at a lower rank would mean this module
    invented a position the store never gave it -- which is exactly the kind of
    invisible score manipulation RRF exists to forbid. Only the stored score
    improves, and only the fuzzy leg ever arrives here twice.
    """
    key = str(chunk_id)
    if key not in scores[leg]:
        rankings[leg].append(key)
    scores[leg][key] = float(score)
    if document_id:
        owner[key] = str(document_id)


def _empty_reason(
    evidence: Sequence[Evidence],
    candidates: Sequence[CandidateRecord],
    *,
    scope_exists: bool | None = None,
) -> str:
    """Which of the named reasons explains an empty result set.

    ``scope_exists`` is what separates the two reasons a caller must never be
    shown interchangeably. ``filtered_out`` means the filter named documents the
    metadata store has never heard of -- a stale id, a deleted upload, a
    multi-tenant scope that lost its row -- which is a bug in the *caller* and
    looks identical to a finding if it is reported as one. ``no_candidates``
    means the legs ran over documents that exist and matched nothing, which is
    the finding the product exists to be able to state. ``None`` means no filter
    was in force, so the distinction does not arise.
    """
    if evidence:
        return EMPTY_OK
    if scope_exists is False:
        return EMPTY_FILTERED
    if candidates:
        return EMPTY_BEYOND_THRESHOLD
    return EMPTY_NO_CANDIDATES


#: Where a document remembers the plan signatures its chunks have been
#: retrieved under. Stored on ``DocumentRecord.metadata`` rather than in a new
#: table because that field already exists, already round-trips through both
#: tiers unchanged, and is already the home for exactly this kind of
#: document-scoped bookkeeping (``metadata["tenant_id"]`` is the precedent).
PLAN_SIGNATURES_METADATA_KEY = "retrieval_plan_signatures"

#: A cap on how many signatures one document accumulates. Each entry is 12 hex
#: characters, so this is not about payload size -- it is about a caller that
#: builds a fresh derived plan per request (a different ``final_k`` each time,
#: say) turning the metadata into an unbounded log. At this size the enumeration
#: cost of resolving one citation stays in the same order as the chunk count,
#: which is the cost the re-derivation was always going to pay.
PLAN_SIGNATURES_MAX = 64


def record_plan_signature(
    bundle: StoreBundle, document_ids: Iterable[str], signature: str
) -> None:
    """Note that ``document_ids`` was queried under ``signature``.

    **Why the minting side records anything at all.** An evidence id is a content
    address over ``(chunk, plan_signature)``, and resolving one means
    re-deriving it over a *candidate set* of signatures. The resolver's job is
    therefore to have the right candidates, and a static registry of built-in
    plans is not the right set: :meth:`RetrievalPlan.with_overrides` is how this
    codebase specialises a plan for real work -- the fact layer calls it for every
    field it extracts -- and a plan derived that way has a signature no registry
    enumerates. The citation then 404s, which is the product's central promise
    failing for a whole population of citations.

    **Recorded per document, and only on the documents that produced evidence.**
    Scoping it to the document is what keeps the cross-document guarantee: the
    resolver still enumerates only *this* document's chunks, and a signature
    recorded against some other paper adds candidates, never foreign text. A
    retrieval that returned nothing records nothing, because no id was minted.

    **Idempotent by design, so the steady state costs no write.** A signature
    already on the record returns without touching the store, which matters
    because this runs inside a read path that answers every user question; the
    version counter on the document only moves when a genuinely new plan
    signature is seen for it.

    **Best-effort, never fatal.** This is bookkeeping about ids that were already
    minted successfully. A store that will not take the write -- a lock, a
    read-only replica -- costs the *next* citation click a resolution, and must
    not cost the current question its answer. So a failure is logged and
    swallowed, exactly as a failing retrieval leg is degraded rather than raised.
    """
    signature = str(signature or "").strip()
    if not signature:
        return
    try:
        for document_id in {str(item) for item in document_ids if str(item or "").strip()}:
            record = bundle.metadata.get_document(document_id)
            if record is None:
                continue
            metadata = dict(record.metadata or {})
            known = [str(item) for item in metadata.get(PLAN_SIGNATURES_METADATA_KEY) or []]
            if signature in known:
                continue
            known.append(signature)
            known = known[-PLAN_SIGNATURES_MAX:]
            metadata[PLAN_SIGNATURES_METADATA_KEY] = known
            record.metadata = metadata
            bundle.metadata.upsert_document(record)
    except Exception:  # noqa: BLE001 - bookkeeping must not fail a retrieval
        logger.warning(
            "记录检索计划签名失败，该文档的引用解析将退回已注册计划集合。",
            exc_info=True,
        )


def _evidence_id(chunk_id: str, plan_signature: str) -> str:
    """A deterministic evidence id.

    Content-addressed, for the same reason chunk ids are: T1.8 writes
    ``evidence_ids`` into the fact layer and T1.5 resolves them from a URL, so an
    id that changed on every run would make a re-retrieval of the same question
    look like a different piece of evidence. Derived from the chunk *and* the
    plan, so the same passage retrieved under two strategies is
    distinguishable -- which is the case worth being able to tell apart.
    """
    payload = f"{chunk_id}|{plan_signature}"
    return "ev_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _record_id(
    query: str,
    plan_signature: str,
    candidates: Sequence[CandidateRecord],
    evidence: Sequence[Evidence],
    expansions: Sequence[ExpansionRecord],
) -> str:
    """A content-addressed id for one retrieval.

    Over the query, the plan signature, the candidate order and the evidence ids
    -- the inputs a reader would compare to decide whether two retrievals were
    the same retrieval. It deliberately excludes any timestamp or duration: a
    record whose id changed every run could not answer "did yesterday's question
    return today's evidence", which is exactly what T4.3's regression gate asks.
    """
    payload = json.dumps(
        {
            "query": query,
            "plan": plan_signature,
            "candidates": [item.chunk_id for item in candidates],
            "evidence": [item.evidence_id for item in evidence],
            "expansions": [item.expanded_chunk_id for item in expansions],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "rr_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
