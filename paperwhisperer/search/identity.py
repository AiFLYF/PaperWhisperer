"""A paper's identity: which records are *the same paper*.

A preprint and its published version are one paper. Today they arrive as two
records — one from arXiv with an arXiv id and no DOI, one from Semantic
Scholar with a DOI and no arXiv id — and :func:`paperwhisperer.search.normalize
.deduplicate_papers` compares their titles, so the user sees the paper twice.
Deduplicating on nothing is no better. Both failure modes destroy trust in
everything else the product claims, which is why this module exists as a value
type rather than a helper function: the identity of a paper is something the
rest of the system can carry around, compare and store.

**The priority order and why it is that order.**
``DOI > arXiv > S2 paperId > OpenAlex > normalized(title, year, first_author)``
is not a ranking of "how authoritative the source is" in the bibliographic
sense. It ranks identifiers by *stability of assignment*, and those are not
the same thing:

``doi``
    A DOI is registered once by a publisher and identifies a specific
    published object. It is the only tier that the *publisher* asserts is
    unique, which is why it goes first: when a preprint and its journal
    version both know the DOI, that single value is the strongest possible
    statement that they are one paper, and keying on it makes the two records
    collapse even though everything else about them differs (year, venue,
    page numbers, sometimes even the title).
``arxiv``
    An arXiv identifier is issued once per submission and never reassigned
    to a different paper. It is the one identifier that reliably links a
    preprint to *itself* across time, including the common case where the
    authors update the arXiv entry to name the journal DOI. It outranks the
    graph providers because a submission is a fact about a physical act,
    whereas an S2 or OpenAlex id is an *entry in a database that maintains
    itself* — those get merged, split, deprecated and re-pointed underneath
    you. The obvious counter-argument is that a preprint is "less official"
    than a publication, so it should rank below. That argument confuses
    authority with stability: what the identity key needs is a value that
    does not move, not one that a publisher stamped.
``s2`` / ``openalex``
    Provider-assigned graph ids, ranked above the title fallback and below
    each other for the same stability argument, S2 first only because the
    plan fixes the order. They are *derived* ids, and a provider that merges
    or retires a record can change what a given id means; a conflict between
    two of them is treated as proof of difference (see below) precisely
    because we were willing to trust them this much.
``title``
    The last resort and never a proof. It is the only tier that is not unique
    by construction. It exists so that two bare records — a user-pasted
    reference and a search hit, neither carrying any identifier — can still be
    recognised as one paper. See :func:`titles_equivalent` for why this tier
    is treated as evidence rather than as a key of equal standing.

**Conflicts are reported, never resolved.** If two records carry different
DOIs, they are not the same published object, and this module refuses to
merge them and names the disagreement. Silently preferring one side would
produce a record whose key belongs to a paper the user never asked about.
:func:`merge_identities` still returns the union of the two sides' data
(because a partially-known identity is better than none) but keeps the
conflict in :class:`MergeResult` so the caller can show it. Where a value
must be picked to keep the merge commutative, the lexicographic minimum is
kept — an arbitrary but *reproducible* choice, and the conflict record, not
the retained value, is the authority.

**Ambiguity is a first-class outcome.** :func:`compare_identities` returns
``confidence="probable"`` for merges that rest on title evidence and
``"exact"`` only for those that rest on a shared identifier, and it returns
``same_paper=False`` with a reason whenever identifiers are present on both
sides but shared by neither. "I am not sure" is a state this product can
render; silently guessing is not one it can.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from paperwhisperer.core.text import parse_year

#: The identity priority order from T3.1, spelled out once so the tiers, the
#: canonical key and the conflict check can never drift from each other. The
#: test suite asserts this tuple equals the plan's ordering, which makes the
#: order a checked contract rather than a comment.
IDENTITY_TIER_ORDER: tuple[str, ...] = ("doi", "arxiv", "s2", "openalex", "title")

#: Tiers that are unique by construction. ``title`` is absent because it is
#: evidence, not an identifier.
STRONG_TIERS: tuple[str, ...] = ("doi", "arxiv", "s2", "openalex")

#: Chinese labels for user-facing explanations of which tier decided a match.
TIER_LABELS_ZH: dict[str, str] = {
    "doi": "DOI",
    "arxiv": "arXiv 编号",
    "s2": "Semantic Scholar 编号",
    "openalex": "OpenAlex 编号",
    "title": "标题、年份与第一作者",
}

# --------------------------------------------------------------------------- #
# Identifier normalization
# --------------------------------------------------------------------------- #

_DOI_URL_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "https://www.doi.org/",
    "doi.org/",
    "dx.doi.org/",
    "doi:",
)

# A DOI is required to look like ``10.<registrant>/<suffix>``. The obvious
# alternative is to accept anything with a slash in it, and it is rejected
# because it turns every URL-shaped string into a false identity. Pre-2000
# legacy DOIs without the ``10.`` prefix are consequently not supported: they
# are vanishingly rare, and the only way to recognise one is to trust a shape
# that ordinary URLs share.
_DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")

# New-style (``YYMM.NNNNN``) and old-style (``math.GT/0309136``) arXiv ids. Both
# are matched with ``fullmatch`` against the whole cleaned string, never with
# ``search``: a DOI such as ``10.1145/1234.56789`` contains a substring that
# looks exactly like a new-style arXiv id, and substring matching would hand
# a DOI an arXiv identity.
_ARXIV_NEW_STYLE = re.compile(r"\d{4}\.\d{4,5}")
#: An old-style id is ``<archive>/<YYMMNNN>`` with an optional subject class
#: attached to the archive name. The class is matched so that ``math.GT`` is
#: recognized as a valid archive spelling and can then be dropped; rejecting it
#: outright would silently lose the arXiv id on exactly the pre-2007 papers
#: that never got a DOI.
_ARXIV_ARCHIVE = re.compile(r"[a-z][a-z-]*(?:\.[a-z]{2})?", re.IGNORECASE)
_ARXIV_OLD_NUMBER = re.compile(r"\d{7}")
_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
_ARXIV_URL_PREFIX = re.compile(r"^https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/", re.IGNORECASE)

_S2_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_OPENALEX_PATTERN = re.compile(r"^[A-Za-z]\d{4,}$")
_OPENALEX_URL_PREFIX = re.compile(
    r"^https?://(?:api\.)?openalex\.org/(?:works/)?", re.IGNORECASE
)

_HTML_TAG = re.compile(r"<[^>]{1,200}>")

#: Split point between a title and its subtitle. Evaluated on the *folded*
#: title, before punctuation is stripped — a colon becomes a space during
#: punctuation stripping, so splitting afterwards would find nothing.
_SUBTITLE_SPLIT = re.compile(r"\s*[:：]\s*|\s+[-–—]\s+")

#: Titles that identify nothing. Two records both titled "Editorial" are two
#: different editorials.
GENERIC_TITLES: frozenset[str] = frozenset(
    {
        "abstract",
        "acknowledgements",
        "acknowledgments",
        "appendix",
        "author index",
        "back matter",
        "backmatter",
        "bibliography",
        "contents",
        "contributors",
        "correction",
        "corrigendum",
        "cover",
        "draft",
        "editorial",
        "editorial board",
        "erratum",
        "foreword",
        "front matter",
        "index",
        "introduction",
        "list of figures",
        "list of tables",
        "manuscript",
        "notes",
        "preface",
        "references",
        "retraction",
        "subject index",
        "title page",
        "untitled",
    }
)

#: A fallback key needs at least this many normalized characters to be worth
#: anything. Below it the title is closer to a category label than a title.
MIN_FALLBACK_TITLE_CHARS = 4

#: A short main-title stem ("BERT") is only accepted as a bridge when the short
#: title is a literal truncation of the long one. A long distinctive stem
#: ("contrastive language pretraining") is accepted on its own.
MIN_CORE_TITLE_CHARS = 4
MIN_DISTINCTIVE_CORE_TITLE_CHARS = 12

#: Name particles that are not part of a name.
_AUTHOR_NOISE = frozenset(
    {"dr", "prof", "mr", "mrs", "ms", "phd", "md", "jr", "sr", "ii", "iii", "iv"}
)

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def normalize_doi(value) -> str:
    """Return a canonical DOI, or ``""`` when the input is not one.

    Case-folded (DOIs are case-insensitive by specification and the two cases
    of the same DOI are routinely used to evade a naive dedupe), with
    ``doi:`` / ``https://doi.org/`` prefixes removed and trailing citation
    punctuation stripped. That last part is not cosmetic: every reference
    manager renders a citation as ``... doi:10.1234/abc.`` and naively
    comparing that string with a bare DOI from a search API is a real,
    reproducible source of duplicate entries.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    for prefix in _DOI_URL_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    # Strip punctuation the surrounding sentence supplied. A DOI never ends in
    # any of these characters, so removing them cannot damage a real one.
    text = text.rstrip(".,;:").strip()
    if not _DOI_PATTERN.match(text.lower()):
        return ""
    return text.lower()


def normalize_arxiv_id(value) -> str:
    """Return a canonical, version-free arXiv identifier, or ``""``.

    **Version suffixes are dropped.** ``2301.00001v1`` and
    ``2301.00001v2`` are the author's successive revisions of one submission,
    i.e. one paper, and the revised version is very often the one that gains
    the journal DOI. Keeping the version would make "the preprint I found in
    January" and "the preprint I found in April" two different papers, which
    is the same bug this module exists to fix, just one tier up the stack. The
    version is not discarded information — it is simply not *identity*; if a
    user needs to cite a specific revision, that is a link property, not a
    deduplication property.

    Accepts bare ids, ``arXiv:`` prefixes, ``arxiv.org/abs`` and
    ``arxiv.org/pdf`` URLs and old-style category ids. The old-style subject
    class is *dropped* rather than case-folded: the canonical URL for a
    1998 maths preprint has always been ``arxiv.org/abs/math/0309136`` with
    no subject class, and ``math.GT/0309136`` is the same paper with a class
    attached after the fact. Collapsing both spellings is the whole point.

    The version suffix is stripped, and the archive name lower-cased, because
    reference lists disagree on both and neither disagreement is meaningful.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    for _ in range(2):  # some citation strings carry the prefix twice
        stripped = _ARXIV_URL_PREFIX.sub("", text).strip()
        if stripped == text:
            break
        text = stripped
    text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE).strip()
    text = _ARXIV_VERSION.sub("", text).strip()
    if _ARXIV_NEW_STYLE.fullmatch(text):
        return text.lower()
    archive, separator, number = text.partition("/")
    if (
        separator
        and _ARXIV_ARCHIVE.fullmatch(archive)
        and _ARXIV_OLD_NUMBER.fullmatch(number)
    ):
        # Drop the subject class: "math.GT/0309136" and "math/0309136" are the
        # same paper, and the URL spelling is the one without the class.
        return f"{archive.split('.')[0].lower()}{separator}{number}"
    return ""


def normalize_s2_paper_id(value) -> str:
    """Return a canonical Semantic Scholar ``paperId``, or ``""``.

    S2 paper ids are 40 hex characters. Anything else — a ``CorpusId``, a
    title that a provider put in the wrong field, a URL — is rejected rather
    than hashed, because a wrong value here becomes a permanent wrong key.
    """
    text = str(value or "").strip().lower()
    if _S2_PATTERN.match(text):
        return text
    return ""


def normalize_openalex_id(value) -> str:
    """Return a canonical OpenAlex work id (``W``/``A``/… plus digits), or ``""``."""
    text = _OPENALEX_URL_PREFIX.sub("", str(value or "").strip()).strip()
    if _OPENALEX_PATTERN.match(text):
        return text.upper()
    return ""


# --------------------------------------------------------------------------- #
# Fallback key normalization
# --------------------------------------------------------------------------- #


def _fold_unicode(text) -> str:
    """NFKD-decompose, drop combining marks and case-fold, keeping punctuation.

    Diacritics are folded because the preprint and the journal copy of the
    same paper are routinely re-typeset with different accent conventions
    ("Müller" vs "Muller"), and a fallback key that cannot survive that will
    miss exactly the merges it exists to catch. This costs a little precision
    on author names, where folding is a genuine collision risk, which is why
    the author key is required to agree *as well* before anything is merged.

    Apostrophes are *removed* rather than treated as separators. Treating them
    as separators turns "Müller's Approach" into "muller s approach" and, far
    worse, "O'Neill" into a surname key of "neill" instead of "oneill".
    Removal makes "Muller's" and "Mullers" — which are the same name spelled
    two ways in two bibliographies — collide, which is the correct direction.
    """
    text = _HTML_TAG.sub(" ", str(text or ""))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    for quote in ("'", "‘", "’", "ʼ"):
        text = text.replace(quote, "")
    return text.casefold()


def normalize_title_key(title) -> str:
    """Return the normalized full title used by the fallback identity tier.

    Case-folded, diacritic-folded, punctuation flattened to spaces, whitespace
    collapsed. CJK is preserved because ``\\w`` is Unicode-aware here: dropping
    the punctuation between Han characters is a Latin-centric rule that would
    destroy Chinese titles, and the product is explicitly bilingual.

    The obvious alternative — keeping punctuation and hyphens — is rejected
    because hyphenation varies by typesetter ("self-supervised" vs "self
    supervised") and an en dash in a subtitle renders three different ways
    across arXiv, Crossref and OpenAlex. The obvious opposite — stripping all
    non-alphanumerics *including* the ones that carry meaning, or truncating
    at the first subtitle marker — is rejected for the same reason recorded on
    :func:`core_title_key`: "BERT: Pre-training" and "BERT" are the same
    paper, and truncating loses the discrimination that the subtitle
    legitimately provides.
    """
    folded = _fold_unicode(title)
    folded = re.sub(r"[^\w\s]", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def core_title_key(title) -> str:
    """Return the main-title stem, dropping any subtitle.

    This is deliberately *not* used for the canonical key — it is a comparison
    aid. Publishers add, drop and reword subtitles, so "BERT: Pre-training of
    Deep Bidirectional Transformers for Language Understanding" and "BERT"
    must still be recognisable as one paper, but a bare stem like "BERT" is
    too weak to key on: it would merge every BERT paper, every "Deep
    Learning" survey and every workshop note that opens with a colon. The
    stem is therefore usable as a bridge only under the two conditions
    described on :func:`_stem_bridge`, and a short stem whose subtitle was
    *reworded* rather than merely dropped is deliberately not bridged — that
    miss is visible and recoverable, the alternative merge is neither.
    """
    folded = _fold_unicode(title)
    parts = [part for part in _SUBTITLE_SPLIT.split(folded) if part.strip()]
    if not parts:
        return normalize_title_key(title)
    return normalize_title_key(parts[0])


def _stem_bridge(
    left_title_key: str, left_core: str, right_title_key: str, right_core: str
) -> bool:
    """Whether two titles differing only in their subtitles may be one paper.

    Two routes, because the strength of a main-title stem varies enormously:

    1. **Truncation.** The shorter full title is a whitespace-delimited prefix
       of the longer one, i.e. one side really is the other with its subtitle
       cut off. This is what turns "BERT" and "BERT: Pre-training ..." into a
       match, and it is safe at a very low length threshold because the
       evidence is structural, not lexical: nothing else in a bibliography
       shortens a title that way.
    2. **Distinctiveness.** A long, specific stem stands on its own even when
       the subtitles were reworded in both directions. A 12-character stem
       like "contrastive language pretraining" names one line of work;
       "attention" does not.

    A generic stem is rejected by both routes. No stem is used without the
    caller also checking the author, and (in the merge path) the year.
    """
    if not left_core or left_core != right_core or left_core in GENERIC_TITLES:
        return False
    shorter, longer = sorted((left_title_key, right_title_key), key=len)
    if shorter and longer.startswith(f"{shorter} "):
        return len(left_core) >= MIN_CORE_TITLE_CHARS
    return len(left_core) >= MIN_DISTINCTIVE_CORE_TITLE_CHARS



def normalize_author_key(author) -> str:
    """Return a surname-shaped key for an author name, or ``""``.

    "Kaiming He", "K. He" and "He, Kaiming" all reduce to ``he``, because the
    one formatting difference every bibliography has is whether the surname
    came first. For CJK names the whole string is kept instead: a Han name has
    no internal space to split on, so taking the "last token" would key on the
    given name.

    Known and accepted limitation: for pinyin names written family-name-first
    ("Zhang Xiaolong") the last-token rule yields ``xiaolong``, while the
    Western order yields ``zhang``. The failure mode is a *missed* merge
    rather than a wrong one, and that asymmetry is deliberate — a user who
    sees a paper twice can open both, a user whose two different papers were
    silently merged can no longer trust anything the workspace shows them.
    """
    text = str(author or "").strip()
    if not text:
        return ""
    if "," in text:
        text = text.split(",", 1)[0]
    text = re.sub(r"[^\w\s]", " ", _fold_unicode(text))
    tokens = [token for token in text.split() if token and token not in _AUTHOR_NOISE]
    if not tokens:
        return ""
    if _CJK.search("".join(tokens)):
        return " ".join(tokens)
    return tokens[-1]


def titles_equivalent(left_title, right_title) -> bool:
    """Whether two titles may be the same title, ignoring subtitles and casing.

    Full-key equality, or main-title-stem equality where the stem is
    distinctive (long enough and not in :data:`GENERIC_TITLES`). Exported for
    enrichment, which must decide whether a Crossref hit is really the paper
    being looked up before it dares write a DOI onto it.
    """
    left_key = normalize_title_key(left_title)
    right_key = normalize_title_key(right_title)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    return _stem_bridge(
        left_key, core_title_key(left_title), right_key, core_title_key(right_title)
    )


# --------------------------------------------------------------------------- #
# The value type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PaperIdentity:
    """Every identifier a paper is known by, plus the fallback key material.

    The identifier fields are normalized in ``__post_init__`` and the display
    fields (title, year, first author) are not, because they are what the user
    reads while the identifiers are only ever compared. The normalization is
    the same "small and total" kind that
    :mod:`paperwhisperer.ingestion.schema` documents: an identity built from a
    provider payload a moment ago should be correct without a re-validation
    pass, and building a wrong key silently is the one mistake this type must
    not let through.
    """

    doi: str = ""
    arxiv_id: str = ""
    s2_paper_id: str = ""
    openalex_id: str = ""
    title: str = ""
    year: str = ""
    first_author: str = ""

    def __post_init__(self) -> None:
        assign = object.__setattr__
        assign(self, "doi", normalize_doi(self.doi))
        assign(self, "arxiv_id", normalize_arxiv_id(self.arxiv_id))
        assign(self, "s2_paper_id", normalize_s2_paper_id(self.s2_paper_id))
        assign(self, "openalex_id", normalize_openalex_id(self.openalex_id))

    # -- keys ------------------------------------------------------------- #

    @property
    def title_key(self) -> str:
        """The normalized full title, regardless of whether it can key anything."""
        return normalize_title_key(self.title)

    @property
    def core_title(self) -> str:
        """The normalized main-title stem; see :func:`core_title_key`."""
        return core_title_key(self.title)

    @property
    def author_key(self) -> str:
        """The first author's surname-shaped key, or ``""``."""
        return normalize_author_key(self.first_author)

    @property
    def normalized_year(self) -> str:
        """A four-digit year, or ``""``."""
        return parse_year(self.year)

    @property
    def fallback_key(self) -> str:
        """``normalized(title)|year|first_author``, or ``""`` when unusable.

        All three components are required. A key built from a title and a year
        alone merges "Attention Is All You Need" with a survey of the same name
        two years later; a key built from a title alone merges the dozens of
        papers called "Deep Learning". Refusing to produce a key at all is
        better than producing a key that merges distinct work, because a
        missing key costs a missed merge the user can notice, while a wrong
        key costs a wrong paper they cannot.
        """
        title_key = self.title_key
        if len(title_key) < MIN_FALLBACK_TITLE_CHARS or title_key in GENERIC_TITLES:
            return ""
        year = self.normalized_year
        author = self.author_key
        if not year or not author:
            return ""
        return f"{title_key}|{year}|{author}"

    @property
    def keys(self) -> dict[str, str]:
        """Every usable tier of this identity, in :data:`IDENTITY_TIER_ORDER`.

        Insertion order is fixed by the tier order rather than by field order,
        so iteration over this mapping is deterministic — which matters once
        the first key is written into a store and read back a year later.
        """
        available: dict[str, str] = {}
        if self.doi:
            available["doi"] = self.doi
        if self.arxiv_id:
            available["arxiv"] = self.arxiv_id
        if self.s2_paper_id:
            available["s2"] = self.s2_paper_id
        if self.openalex_id:
            available["openalex"] = self.openalex_id
        fallback = self.fallback_key
        if fallback:
            available["title"] = fallback
        return available

    @property
    def key(self) -> str:
        """The single canonical identity key: ``"<tier>:<value>"``, or ``""``.

        The tier prefix is part of the key on purpose. A bare DOI and an arXiv
        id are different namespaces, and a key without it would make
        ``10.1234/abc`` and an arXiv id string that happened to look alike
        collide. The empty string means "this record identifies nothing" — a
        real state, and one a caller must handle rather than coerce into a
        made-up key.
        """
        for tier in IDENTITY_TIER_ORDER:
            value = self.keys.get(tier)
            if value:
                return f"{tier}:{value}"
        return ""

    @property
    def key_tier(self) -> str:
        """The tier that decided :attr:`key` (``""`` when there is no key)."""
        for tier in IDENTITY_TIER_ORDER:
            if self.keys.get(tier):
                return tier
        return ""

    @property
    def has_strong_identifier(self) -> bool:
        """Whether at least one unique-by-construction identifier is present."""
        return any(getattr(self, _FIELD_BY_TIER[tier]) for tier in STRONG_TIERS)

    @property
    def needs_enrichment(self) -> bool:
        """Whether a DOI is missing, i.e. whether enrichment could help."""
        return not self.doi

    def to_dict(self) -> dict[str, str]:
        """Serialize for storage. The derived key is recomputed on read, never
        stored, so a change to the tier order cannot leave stale keys behind."""
        return {
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "s2_paper_id": self.s2_paper_id,
            "openalex_id": self.openalex_id,
            "title": self.title,
            "year": self.year,
            "first_author": self.first_author,
        }

    @classmethod
    def from_dict(cls, payload) -> PaperIdentity:
        """Inverse of :meth:`to_dict`; unknown keys are ignored."""
        data = dict(payload or {})
        return cls(
            doi=data.get("doi", ""),
            arxiv_id=data.get("arxiv_id", ""),
            s2_paper_id=data.get("s2_paper_id", ""),
            openalex_id=data.get("openalex_id", ""),
            title=data.get("title", ""),
            year=data.get("year", ""),
            first_author=data.get("first_author", ""),
        )


_FIELD_BY_TIER = {"doi": "doi", "arxiv": "arxiv_id", "s2": "s2_paper_id", "openalex": "openalex_id"}


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IdentityMatch:
    """The verdict on whether two records describe one paper.

    ``confidence`` is deliberately three-valued rather than a score. A
    continuous confidence number would invite callers to threshold it, and a
    threshold invites them to pick 0.6 and merge things this module declined
    to merge. The states are "a shared identifier says yes", "the titles and
    authors say yes and nothing says no", and "something says no".
    """

    same_paper: bool
    confidence: str  # "exact" | "probable" | "none"
    basis: str
    key: str = ""
    notes: tuple[str, ...] = ()

    @property
    def requires_confirmation(self) -> bool:
        """True when the verdict rests on title evidence alone.

        The caller is expected to say so in the UI rather than presenting the
        two records as an established duplicate.
        """
        return self.same_paper and self.confidence == "probable"

    @property
    def basis_zh(self) -> str:
        """A user-facing explanation of what decided the verdict."""
        return MATCH_BASIS_ZH.get(self.basis, self.basis)


MATCH_BASIS_ZH: dict[str, str] = {
    "shared_doi": "DOI 相同",
    "shared_arxiv": "arXiv 编号相同",
    "shared_s2": "Semantic Scholar 编号相同",
    "shared_openalex": "OpenAlex 编号相同",
    "fallback_key": "标题、年份与第一作者完全一致",
    "core_title_key": "主标题一致，且年份与第一作者吻合",
    "fallback_key_bridge": "标题与作者一致，但标识符互不相同（需回查 DOI 确认）",
    "core_title_key_bridge": "主标题与作者一致，但标识符互不相同（需回查 DOI 确认）",
    "doi_conflict": "两条记录的 DOI 不同，已判定为不同论文",
    "arxiv_conflict": "两条记录的 arXiv 编号不同，已判定为不同论文",
    "s2_conflict": "两条记录的 Semantic Scholar 编号不同，已判定为不同论文",
    "openalex_conflict": "两条记录的 OpenAlex 编号不同，已判定为不同论文",
    "disjoint_identifiers": "两条记录各自带有不同标识符，无法判断是否为同一篇",
    "fallback_incomplete": "记录缺少标题、年份或作者，无法用标题比对",
    "no_shared_identifier": "标题、年份或作者不一致，且没有共同的标识符",
}


def _fallback_agreement(left: PaperIdentity, right: PaperIdentity) -> tuple[str, str]:
    """Return ``(basis, "")`` when title evidence agrees, ``("", "")`` otherwise.

    Two independent routes, tried in order of strength:

    1. The complete fallback key — identical title, year and first author.
    2. The main-title stem plus the author's surname, *without* requiring the
       years to match. A preprint posted in December and published the
       following June is one paper, and requiring the year to be equal would
       refuse exactly the merge this module exists to perform. The year is
       still checked, just loosely: a gap larger than one year means the two
       records are almost certainly about different things.
    """
    left_author = left.author_key
    right_author = right.author_key
    if not left_author or left_author != right_author:
        return "", ""

    left_fallback = left.fallback_key
    right_fallback = right.fallback_key
    if left_fallback and left_fallback == right_fallback:
        return "fallback_key", ""

    left_year = left.normalized_year
    right_year = right.normalized_year
    if not left_year or not right_year or abs(int(left_year) - int(right_year)) > 1:
        return "", ""

    left_core = left.core_title
    right_core = right.core_title
    if _stem_bridge(left.title_key, left_core, right.title_key, right_core):
        return "core_title_key", ""
    return "", ""


def compare_identities(left: PaperIdentity, right: PaperIdentity) -> IdentityMatch:
    """Decide whether two identities describe the same paper, and say why.

    The order of the checks is the argument:

    1. **A conflict on any strong identifier is a hard no.** Two different
       DOIs are two different published objects, whatever the titles say. This
       is the check that stops the classic false merge: a preprint and a
       survey that happen to share a title, or a title collision across
       venues. It is also why the identifier tiers are trusted enough to
       outrank the title at all.
    2. **A shared identifier is a yes**, at the highest tier both sides have,
       which is what makes the canonical key single-valued.
    3. **Otherwise, title evidence decides**, and only if nothing contradicts
       it. When both sides carry identifiers but share none — the arXiv-only
       versus DOI-only case that enrichment exists to solve — the verdict is
       still allowed to be positive on strong title evidence, but it is
       downgraded to ``"probable"`` and flagged for confirmation, because the
       missing DOI is precisely what would have settled it.
    """
    left_keys = left.keys
    right_keys = right.keys

    for tier in STRONG_TIERS:
        field = _FIELD_BY_TIER[tier]
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value and right_value and left_value != right_value:
            return IdentityMatch(
                same_paper=False,
                confidence="none",
                basis=f"{tier}_conflict",
                key="",
                notes=(
                    f"两条记录的{TIER_LABELS_ZH[tier]}不一致（{left_value} / {right_value}），"
                    "已按不同论文处理，请核对来源。",
                ),
            )

    for tier in STRONG_TIERS:
        value = left_keys.get(tier)
        if value and right_keys.get(tier) == value:
            return IdentityMatch(
                same_paper=True,
                confidence="exact",
                basis=f"shared_{tier}",
                key=f"{tier}:{value}",
            )

    basis, _ = _fallback_agreement(left, right)
    if not basis:
        # Neither the identifiers nor the title evidence settled it. Which of
        # the two "no" reasons is the more useful one depends on the records:
        # "disjoint identifiers" says the answer exists and enrichment must go
        # find it, "incomplete" says the record is simply thin. The disjoint
        # case is tested first because a record with no year is still more usefully
        # described by the identifiers it does carry.
        if left.has_strong_identifier and right.has_strong_identifier:
            return IdentityMatch(
                same_paper=False,
                confidence="none",
                basis="disjoint_identifiers",
                notes=(
                    "两条记录各自带有不同的标识符且互不冲突，"
                    "无法判断是否为同一篇论文，建议先回查 DOI。",
                ),
            )
        if not left.fallback_key or not right.fallback_key:
            return IdentityMatch(
                same_paper=False,
                confidence="none",
                basis="fallback_incomplete",
            )
        return IdentityMatch(
            same_paper=False,
            confidence="none",
            basis="no_shared_identifier",
        )
    both_strong = left.has_strong_identifier and right.has_strong_identifier
    matched_basis = f"{basis}_bridge" if both_strong else basis
    key = f"title:{left.fallback_key}" if basis == "fallback_key" else f"title:{left.core_title}"
    notes: tuple[str, ...] = ()
    if both_strong:
        notes = ("依据标题与作者判定为同一篇论文，但两条记录的标识符不同，建议回查 DOI 后再合并。",)
    return IdentityMatch(
        same_paper=True,
        confidence="probable",
        basis=matched_basis,
        key=key,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IdentityConflict:
    """One field where the two records disagreed, and what was kept."""

    field: str
    left: str
    right: str
    kept: str
    message: str


@dataclass(frozen=True)
class MergeResult:
    """The union of two identities, plus everything a caller must not ignore."""

    identity: PaperIdentity
    same_paper: bool
    basis: str
    confidence: str
    conflicts: tuple[IdentityConflict, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def requires_confirmation(self) -> bool:
        """Whether this merge rests on title evidence and should be confirmed."""
        return self.same_paper and self.confidence == "probable" and bool(self.notes)

    @property
    def is_trustworthy(self) -> bool:
        """False when the two records disagreed about an identifier."""
        return not self.conflicts


def merge_identities(left: PaperIdentity, right: PaperIdentity) -> MergeResult:
    """Union two identities, recording every disagreement instead of hiding it.

    **Identifiers are order-independent.** Fields are combined, and where both
    sides have *different* non-empty values the lexicographic minimum is
    retained. That choice is arbitrary, and it is documented as arbitrary
    precisely because the retained value is not a verdict —
    :attr:`MergeResult.conflicts` is. Picking by argument order would make the
    result depend on which source the caller happened to list first, and a
    dedupe result whose key changes when the query is re-run is not something
    a user can trust or a test can assert on.

    The merged *display* fields (title, year, first author) are explicitly
    **not** order-independent: they keep the left record's value when both
    sides have one, because the caller passes the record the user is already
    looking at first. What has to be stable, and is tested to be, is the
    identifier set and therefore :attr:`PaperIdentity.key`.
    """
    values: dict[str, str] = {}
    conflicts: list[IdentityConflict] = []
    for tier in STRONG_TIERS:
        field = _FIELD_BY_TIER[tier]
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value and right_value and left_value != right_value:
            kept = min(left_value, right_value)
            conflicts.append(
                IdentityConflict(
                    field=field,
                    left=left_value,
                    right=right_value,
                    kept=kept,
                    message=(
                        f"{TIER_LABELS_ZH[tier]}冲突：{left_value} 与 {right_value} 不一致，"
                        f"暂保留 {kept}，请人工核对。"
                    ),
                )
            )
            values[field] = kept
        else:
            values[field] = left_value or right_value

    first_author = left.first_author or right.first_author
    left_author = left.author_key
    right_author = right.author_key
    if left_author and right_author and left_author != right_author:
        kept = min(left_author, right_author)
        first_author = left.first_author if left_author == kept else right.first_author
        conflicts.append(
            IdentityConflict(
                field="first_author",
                left=left.first_author,
                right=right.first_author,
                kept=first_author,
                message=(
                    f"第一作者不一致（{left.first_author} / {right.first_author}），"
                    f"暂保留 {first_author}，请人工核对。"
                ),
            )
        )

    identity = PaperIdentity(
        title=left.title or right.title,
        year=left.year or right.year,
        first_author=first_author,
        **values,
    )
    match = compare_identities(left, right)
    return MergeResult(
        identity=identity,
        same_paper=match.same_paper,
        basis=match.basis,
        confidence=match.confidence,
        conflicts=tuple(conflicts),
        notes=match.notes,
    )


# --------------------------------------------------------------------------- #
# Building identities from provider records
# --------------------------------------------------------------------------- #


def identity_from_record(record) -> PaperIdentity:
    """Build a :class:`PaperIdentity` from a canonical paper record dict.

    Every identifier field is probed with every recognizer rather than routing
    on ``record["source"]``, because the recognizers validate strictly and a
    DOI is not shaped like an arXiv id is not shaped like an S2 paperId. The
    obvious alternative — switch on the source name and read the matching
    field — breaks the moment a record arrives from a source this function
    has not been taught, which is exactly the case that needs it most.

    Explicit identifier fields (``doi``, ``arxiv_id``, ``s2_paper_id``,
    ``openalex_id``) are honoured when present. ``normalize_paper_record``
    does not emit them today, which is why DOI recovery has to go to the
    network for records that already carry the answer.
    """
    data = record if isinstance(record, dict) else {}
    paper_id = str(data.get("paper_id") or "").strip()
    url = str(data.get("url") or "").strip()

    authors = data.get("authors") or []
    first_author = ""
    if isinstance(authors, (list, tuple)) and authors:
        first_author = str(authors[0] or "").strip()
    elif isinstance(authors, str):
        first_author = authors.strip()

    return PaperIdentity(
        doi=data.get("doi") or paper_id or url or "",
        arxiv_id=data.get("arxiv_id") or paper_id or url or "",
        s2_paper_id=data.get("s2_paper_id") or paper_id or "",
        openalex_id=data.get("openalex_id") or paper_id or url or "",
        title=data.get("title") or "",
        year=data.get("year") or "",
        first_author=first_author,
    )
