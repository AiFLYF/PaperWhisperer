"""Paper record normalization shared by search results and the reading queue.

This module is the boundary between what a bibliographic provider sends and
what the rest of the product stores. Two things happen here, and both of them
used to happen *wrongly in the direction of losing information*.

**Identifiers are read, not dropped.** Semantic Scholar returns a record's DOI
and arXiv id inside an ``externalIds`` object; arXiv returns the journal DOI of
an already-published preprint in its own ``arxiv:doi`` element. Both were
discarded, which left a real paper sitting in the user's queue with nothing but
a title to identify it by. That is the "DOI 丢失可回查" case: recovery then
costs a Crossref round trip, and the round trip is the slow, rate-limited,
sometimes-wrong path, for an answer the provider had already sent. Every
identifier a provider supplies is therefore read, and it is read *through*
:func:`paperwhisperer.search.identity.identity_from_record`, which owns
recognition and normalization. Deciding whether a string is a DOI, an arXiv id
or neither is done in exactly one place in this codebase; a second set of
ad-hoc regexes here is what produced the original bug.

**Deduplication asks who the paper is, not what it is called.** The old key was
the lowercased title. That merges two different papers whenever they share a
title — a preprint and an unrelated survey with the same name, two venue
versions of one workshop paper, a retitled republication — and it fails to
merge a preprint with its published version whenever the typesetter re-worded
the subtitle, which is the case this whole layer exists for. Both are silent.
:func:`deduplicate_papers` now delegates the decision to
:func:`paperwhisperer.search.identity.compare_identities`, which weighs DOI,
arXiv, provider ids, then title/year/author, and which *refuses* to merge when
the two records carry conflicting identifiers.

**Two identity namespaces, never one.** A canonical identity key
(``doi:10.…``) and a URL are different kinds of claim about a record. Mixing
them in one comparable space was harmless in practice only because a title is
never string-equal to a URL; it was still wrong, and it is now explicit: a
record with no usable identity is recognised by its link, and a record that
*has* an identity is judged by it.
"""

from __future__ import annotations

import urllib.parse

from paperwhisperer.core import config
from paperwhisperer.core.text import (
    compact_text,
    normalize_author_list,
    now_iso,
    parse_year,
)
from paperwhisperer.search.identity import (
    IdentityMatch,
    PaperIdentity,
    compare_identities,
    identity_from_record,
    merge_identities,
)

#: The one confidence level at which a search list is allowed to collapse two
#: records into one. See :func:`deduplicate_papers` for why ``probable`` is not
#: in it — the short version is that a wrong merge is unrecoverable and a
#: duplicate is not, and this constant exists so that the rule is a single
#: greppable line rather than a comparison scattered through the loop.
COLLAPSE_CONFIDENCE = "exact"

#: Fields carried through from a provider payload into the canonical record.
#: They are emitted as a group, always present, empty string when unknown, so
#: that every consumer can read them without a ``.get(...) or {}`` dance and
#: without mistaking a missing key for "the provider said there is none".
IDENTIFIER_FIELDS: tuple[str, ...] = ("doi", "arxiv_id", "s2_paper_id", "openalex_id")

#: S2 spells its identifiers with provider-specific names inside ``externalIds``.
#: Only the two that the identity tier order knows are read; ``MAG``,
#: ``DBLP``, ``PubMed`` and ``CorpusId`` are ignored on purpose — a value with
#: no tier has nowhere to live in :class:`PaperIdentity`, and inventing a field
#: for it would put a second, unvalidated id namespace into the record.
_S2_EXTERNAL_ID_FIELDS = {"DOI": "doi", "ArXiv": "arxiv_id"}

#: Flat spellings accepted from any provider, not just Semantic Scholar, so
#: that a newly wired source does not have to be taught to this module before
#: its identifiers are kept.
_FLAT_IDENTIFIER_FIELDS = {
    "doi": "doi",
    "arxiv_id": "arxiv_id",
    "arxiv": "arxiv_id",
    "s2_paper_id": "s2_paper_id",
    "openalex_id": "openalex_id",
}


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #


def _provider_identifier_hints(source: str, record: dict) -> dict[str, str]:
    """The raw, *unvalidated* identifier strings a provider supplied.

    Only the mapping from a provider's field names to the canonical names
    happens here. Whether ``"CorpusId:2215645"`` is a DOI is not a question
    this module answers — it hands the value to
    :func:`identity_from_record` and lets the identity layer's recognizers
    reject it, which is what happens in production: S2 puts
    ``CorpusId:<n>`` in the DOI slot for records that have no real DOI, and a
    hand-rolled ``if value:`` here would have written that onto the record as
    a DOI, making the paper permanently un-deduplicatable against everything
    else.
    """
    hints: dict[str, str] = {}
    if source == "Semantic Scholar":
        external = record.get("externalIds") or {}
        if isinstance(external, dict):
            for provider_key, canonical in _S2_EXTERNAL_ID_FIELDS.items():
                hints[canonical] = str(external.get(provider_key) or "").strip()
    for provider_key, canonical in _FLAT_IDENTIFIER_FIELDS.items():
        value = str(record.get(provider_key) or "").strip()
        if value:
            hints[canonical] = value
    return hints


def _attach_identity(record: dict, hints: dict[str, str]) -> dict:
    """Read the identifiers off ``record`` and write them back, normalized.

    The identity is computed from the record *plus* the provider's raw hints,
    so the existing ``paper_id`` / ``url`` fallbacks inside
    :func:`identity_from_record` still apply for sources that name their
    identifier positionally — an arXiv ``paper_id`` of
    ``http://arxiv.org/abs/1512.03385v1`` becomes ``arxiv_id`` with no
    per-source code here at all.
    """
    identity = identity_from_record({**record, **hints})
    record.update({field: getattr(identity, field) for field in IDENTIFIER_FIELDS})
    record["identity_key"] = identity.key
    return record


# --------------------------------------------------------------------------- #
# Record normalization
# --------------------------------------------------------------------------- #


def normalize_paper_record(source: str, record: dict) -> dict:
    """Map a provider payload onto the app's canonical paper shape.

    The emitted record always carries the four identifier fields plus
    ``identity_key`` (see :data:`IDENTIFIER_FIELDS`), so a record that arrived
    from Semantic Scholar with a DOI in ``externalIds`` leaves here with that
    DOI, and one that arrived with ``CorpusId:…`` in the DOI slot leaves here
    with an empty ``doi`` and an ``s2_paper_id`` instead of a fake identifier.
    """
    if source == "Semantic Scholar":
        open_access_pdf = record.get("openAccessPdf") or {}
        normalized = {
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
    else:
        normalized = {
            "source": source,
            "paper_id": str(record.get("paper_id") or record.get("id") or "").strip(),
            "title": compact_text(record.get("title") or "", limit=300),
            "abstract": compact_text(
                record.get("abstract") or record.get("summary") or "", limit=2000
            ),
            "authors": normalize_author_list(record.get("authors") or []),
            "year": parse_year(record.get("year") or record.get("published") or ""),
            "venue": compact_text(record.get("venue") or source, limit=120),
            "url": str(record.get("url") or record.get("id") or "").strip(),
            "pdf_url": str(record.get("pdf_url") or "").strip(),
        }
    return _attach_identity(normalized, _provider_identifier_hints(source, record))


def normalize_paper_collection(values, max_items: int = config.READING_QUEUE_LIMIT) -> list[dict]:
    """Clean, dedupe and cap a client-supplied list of papers.

    A record with no title is dropped. That is pre-existing and deliberate: the
    queue is a list of things the user will read, and a titled nothing is not
    one of them.

    Deduplication runs on the same identity rules as search — see
    :func:`deduplicate_papers` — with one addition, because this input is
    *client-supplied* and therefore less trustworthy than a provider payload:
    a record that carries no usable identity is recognised by its link, and
    one that does is judged by its identity alone.

    **Two identical titles are no longer one row.** This function used to key
    on the title alone, which meant the reading queue would silently fuse two
    different papers that happened to share a name, and — because a title key
    and a URL were mixed in one namespace — would also fuse a record with
    another record's URL. Records with no author and no year, which is what a
    bare client-side save looks like, are now kept separately unless they share
    a link. That is the same asymmetry the search path applies: a duplicate the
    user can see costs them a glance, a wrong merge costs them a paper.
    """
    if not isinstance(values, list):
        return []
    candidates: list[dict] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        title = compact_text(value.get("title"), limit=300)
        if not title:
            continue
        url = str(value.get("url") or "").strip()[:1000]
        pdf_url = str(value.get("pdf_url") or "").strip()[:1000]
        record = {
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
        }
        hints = {
            canonical: compact_text(value.get(canonical), limit=200)
            for canonical in IDENTIFIER_FIELDS
        }
        candidates.append(_attach_identity(record, hints))

    kept = _run_dedupe(candidates, use_links=True)[0]
    return kept[:max_items]


# --------------------------------------------------------------------------- #
# Deduplication
# --------------------------------------------------------------------------- #


def _link_key(value) -> str:
    """A comparable form of a URL: same link, same key.

    Scheme and host are case-folded, ``www.`` and a trailing slash are
    dropped, and the fragment is discarded. The query string is *kept*,
    because some publisher landing pages identify an article by a query
    parameter and dropping it would merge distinct papers — the wrong
    direction, and the only one that is not recoverable by the user.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    parts = urllib.parse.urlsplit(text)
    if not parts.scheme and not parts.netloc:
        return text.casefold()
    netloc = parts.netloc.casefold()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path.rstrip("/")
    return urllib.parse.urlunsplit((parts.scheme.casefold(), netloc, path, parts.query, ""))


def _record_links(record: dict) -> set[str]:
    """The link namespace for a record: its landing page and its PDF."""
    return {key for key in (_link_key(record.get("url")), _link_key(record.get("pdf_url"))) if key}


def _is_collapse(match: IdentityMatch) -> bool:
    """Whether ``match`` is strong enough to remove a record from the list.

    ``exact`` means a shared identifier — a DOI, an arXiv id, or a provider id
    — said the two records are the same published object. Nothing weaker is
    allowed to remove a row: see :func:`deduplicate_papers`.
    """
    return bool(match.same_paper) and match.confidence == COLLAPSE_CONFIDENCE


def _describe_possible_duplicate(match: IdentityMatch) -> str:
    """A Chinese sentence explaining why two kept records might be one paper."""
    if match.notes:
        return match.notes[0]
    return (
        f"这条结果与列表中的另一条可能是同一篇论文（{match.basis_zh}）。"
        "两条都已保留，请人工核对后再合并。"
    )


def _run_dedupe(
    items, *, use_links: bool = False
) -> tuple[list[dict], list[dict], list[dict]]:
    """Collapse confirmed duplicates, in provider order, and report the rest.

    Returns ``(kept, merged, possible)``:

    ``kept``
        The surviving records, first occurrence wins, each carrying the union
        of the identifiers of everything collapsed into it.
    ``merged``
        One entry per collapse, for the UI to explain where a result came
        from. Without it, an arXiv row that quietly absorbed its published
        twin looks like a search that simply found one hit.
    ``possible``
        The ``probable`` pairs, at most one per surviving record — the first
        kept record it looks like a twin of. Each entry names both records
        and the evidence. These are the pairs the user is asked to confirm.

    ``use_links`` is the extra, weaker signal available to client-supplied
    input: a record with *no* usable identity whose link has already been seen
    is the same link twice. It is deliberately restricted to records that
    identify nothing, because a link that disagrees with an identity is a
    metadata problem the user should see rather than a duplicate to hide.
    """
    kept: list[dict] = []
    kept_identities: list[PaperIdentity] = []
    kept_links: list[set[str]] = []
    merged: list[dict] = []
    possible: list[dict] = []

    for item in items or []:
        if not isinstance(item, dict):
            continue
        record = dict(item)
        identity = identity_from_record(record)
        links = _record_links(record)

        # A record that identifies nothing at all cannot be weighed against
        # another such record — compare_identities would answer "insufficient
        # evidence" and so would we. The only thing left to compare is the
        # link, and only for client-supplied input, where a repeated URL is a
        # real failure mode (the same row submitted twice, or a source echoing
        # a landing page back under a new title).
        if use_links and not identity.key and any(links & seen for seen in kept_links):
            continue

        collapse_at: int | None = None
        collapse_match: IdentityMatch | None = None
        probable: tuple[int, IdentityMatch] | None = None

        for index, existing in enumerate(kept_identities):
            match = compare_identities(identity, existing)
            if _is_collapse(match):
                collapse_at = index
                collapse_match = match
                break
            if match.same_paper and match.requires_confirmation and probable is None:
                probable = (index, match)

        if collapse_at is not None and collapse_match is not None:
            result = merge_identities(kept_identities[collapse_at], identity)
            survivor = _apply_merge(kept[collapse_at], result.identity)
            kept[collapse_at] = survivor
            kept_identities[collapse_at] = result.identity
            kept_links[collapse_at] |= links
            merged.append(
                {
                    "title": survivor.get("title", ""),
                    "kept_source": survivor.get("source", ""),
                    "merged_source": record.get("source", ""),
                    "basis": collapse_match.basis,
                    "basis_zh": collapse_match.basis_zh,
                }
            )
            continue

        if probable is not None:
            other_index, probable_match = probable
            possible.append(
                {
                    "title": record.get("title", ""),
                    "other_title": kept[other_index].get("title", ""),
                    "basis": probable_match.basis,
                    "basis_zh": probable_match.basis_zh,
                    "note_zh": _describe_possible_duplicate(probable_match),
                }
            )

        record.setdefault("identity_key", identity.key)
        kept.append(record)
        kept_identities.append(identity)
        kept_links.append(links)

    return kept, merged, possible


def _apply_merge(record: dict, identity: PaperIdentity) -> dict:
    """Write a merged identity's identifiers back onto a surviving record.

    Only the identifier fields are touched. Title, year, authors and venue keep
    the surviving record's values, because that record is the one already on
    screen and the user is not looking at a diff.
    """
    record.update({field: getattr(identity, field) for field in IDENTIFIER_FIELDS})
    record["identity_key"] = identity.key
    return record


def deduplicate_papers(items) -> list[dict]:
    """Drop confirmed duplicates of one paper, preserving provider ordering.

    **A match on ``probable`` is never collapsed.** ``compare_identities``
    returns ``probable`` for merges that rest on title, year and author
    evidence — the arXiv-only record versus the DOI-only record of the same
    paper, which is exactly the pair enrichment exists to settle. Collapsing
    it would hide the reason the DOI is missing, and would do so silently, on
    the one comparison the product cannot verify. Keeping it shows the same
    paper twice, which the user can recognise and the report below explains.
    The asymmetry is deliberate and is the whole reason the collapse rule is a
    named constant: :data:`COLLAPSE_CONFIDENCE` is ``"exact"`` and nothing
    else, and no caller can pass a looser one.

    Use :func:`deduplicate_papers_report` to get the ``probable`` pairs and the
    record of what was collapsed, both of which the search route surfaces so
    that the behaviour is visible rather than merely correct.
    """
    return _run_dedupe(items)[0]


def deduplicate_papers_report(items) -> dict:
    """:func:`deduplicate_papers` plus what it decided and what it declined.

    The three lists are the honest description of the pass: ``merged`` are the
    rows that were removed as confirmed duplicates, ``possible_duplicates``
    are the pairs a human still has to look at, and ``kept`` is what the user
    sees.
    """
    kept, merged, possible = _run_dedupe(items)
    return {
        "items": kept,
        "merged": merged,
        "possible_duplicates": possible,
    }
