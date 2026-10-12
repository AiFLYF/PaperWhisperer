"""Structure-aware chunking: elements and sections in, retrieval units out.

The chunker this module replaces cut a flat string every 4000 characters with
200 characters of overlap. That approach is defensible only while the input is
a flat string, and T1.2 exists because the input stopped being one at the end
of Phase 0. Three things are now known at chunking time that were not known
when the offsets were chosen, and each of them makes a character offset the
wrong place to cut:

* **Where the element is.** A ``Table`` arrives as a two-dimensional grid with
  a caption, not as a run of text. Slicing at 4000 characters through the
  middle of a grid produces prose with pipe characters in it: the column
  headers end up on a different chunk from the row they label, so a question
  about one row of Table 3 retrieves a chunk containing half the table and no
  column names. The grid survives the whole pipeline in
  :class:`~paperwhisperer.ingestion.schema.Table` precisely so that this
  module does not have to rebuild it from characters.
* **Where the section is.** A heading element is a boundary that a reader
  would notice. Cutting through it produces a chunk that is half limitations
  and half conclusions, which is the one thing a research workspace must not
  do, because the two halves support opposite claims.
* **Where the sentence is.** Even with structure in hand, an oversized
  paragraph has to be cut somewhere, and the only acceptable somewhere is a
  sentence end.

So the order of the pipeline is: structure first, size second. Elements are
grouped into sections by their heading path, structural elements become
chunks of their own, prose is merged up to a token budget inside a section,
and only then is anything split -- at a sentence end, never inside a formula
and never inside a table.

Boundaries, in one place
------------------------
The boundary rules are stated once, here, because the acceptance criterion
for this task ("no cross-semantic break in a long paper") is only meaningful
if "semantic break" has a single definition that both the code and its tests
share:

* An element boundary, always. Chunks are assembled from whole element texts
  that are either verbatim or a verbatim slice of one element, so a chunk can
  never contain half a table or a fragment of a sentence the source does not
  contain.
* A section boundary, always. Two prose elements are merged only when their
  heading paths are equal, so a chunk never spans two heading levels. This is
  checked against the *element* heading paths rather than against a heading
  level integer, because
  :mod:`paperwhisperer.ingestion.adapters.docling_adapter` deliberately leaves
  ``Element.level`` unset -- depth inference is the adapter's job and this
  module has no better information than it does.
* A sentence boundary, when a split is required. With a target that some
  sentences exceed, the window widens to :attr:`ChunkPolicy.hard_max_tokens`
  before a mid-sentence cut is accepted, because a chunk that overruns its
  budget says so in ``token_estimate`` while a mid-sentence cut says nothing
  at all. An overrun is visible; a broken sentence is invisible. Only a single
  sentence longer than the hard limit is cut mid-sentence, and then only
  between words, where the whitespace makes the cut findable by a reader.

What the run could not do
-------------------------
Two compromises in the rules above are visible rather than silent, and both
are reported by :meth:`StructuralChunker.chunk_with_report` rather than left
for a caller to infer from a chunk list. A table larger than the ceiling is
emitted whole, because a half-grid cannot be cited or re-rendered, and it
comes out of a run flagged over-budget so the embedding provider's eventual
truncation of it is a known fact. An element with no retrievable text -- a
captionless figure, a table grid the quality gate already suspects is a
picture -- produces no chunk, and the run says which element it was and why.
Both belong in a report rather than on a chunk: "too large to embed" is a fact
about the document as a whole, and a retriever reading one chunk has nothing
to compare it against.

No overlap, and why
-------------------
The legacy chunker overlapped consecutive windows by 200 characters so that a
sentence near a cut would survive in one of the two chunks. That is a repair
for a cut placed at an arbitrary offset: overlap buys the sentence back at the
cost of storing it twice, and every duplicated sentence is a near-duplicate
vector in the index, so the retriever returns two chunks that differ by a
sentence and the caller has to deduplicate to tell which is the better one.
Once the cut is placed at a sentence end there is no sentence to rescue, and
the duplication buys nothing. A retriever that wants a passage's continuation
has a better tool for it: the neighbour expansion T1.4 already owns. So
overlap is absent here rather than merely defaulted to zero, and a caller who
wants it can supply it at the retrieval stage, where it can be weighted rather
than baked into the stored text.

Index identity
--------------
Every chunk carries an ``index_version`` that encodes the embedding provider,
model and dimension alongside a hash of the chunker's own boundary
parameters. Plan section 8 names the risk this answers: OpenAI-compatible
embedding endpoints disagree about dimension and vocabulary, and mixing their
vectors produces a store that answers queries with confident nonsense. The
obvious alternative -- keying vectors on the document alone -- is what makes
that silent, so the version is a *comparison*, not a label: a store holding
chunks whose ``index_version`` differs from the one the current configuration
computes is stale by construction, and the rebuild is a comparison rather than
a guess.

The chunker signature belongs in that string for the same reason. Changing
``max_tokens`` from 400 to 512 does not change what any chunk says, but it
changes every boundary, and an index that kept the old vectors would embed
different text than the text stored beside them. Bumping
:data:`CHUNKER_VERSION` after a change to the boundary rules does the same
thing on purpose, which is what makes T4.3's "a parser or model upgrade must
run the regression suite" checkable: the corpus is re-chunked, the ids are
diffed, and any difference is a change a human decided to make.

Determinism
-----------
Chunk ids are content-addressed, not generated: they are a hash of the index
version, the document identity, the chunk's position and the element ids it
covers. :func:`chunk_document` is therefore a pure function of
``(ParsedDocument, ChunkPolicy, provider, model, dim)``, and chunking the same
stored document twice produces the same ids in the same order, so a rebuild
can diff the old index against the new one and see exactly which passages
moved. A uuid per chunk -- the schema's own default -- would make every chunk
an orphan the moment a document is re-chunked, and a regression suite that
cannot diff cannot detect a regression.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

from paperwhisperer.core import config
from paperwhisperer.ingestion.schema import (
    SCHEMA_VERSION,
    Chunk,
    Element,
    ParsedDocument,
    SourceLocator,
)

__all__ = [
    "CHUNKER_VERSION",
    "HEADING_PREFIX",
    "ChunkingReport",
    "ChunkPolicy",
    "StructuralChunker",
    "TokenEstimator",
    "build_index_version",
    "chunk_document",
    "chunk_document_from_config",
    "element_content_text",
    "estimate_tokens",
    "index_version_parts",
    "parse_index_version",
    "split_prose_at_safe_boundaries",
]

#: Bumped when a change to this module would move a boundary in a document
#: that parsed the same way yesterday. It is folded into ``index_version``, so
#: bumping it invalidates every stored vector rather than leaving chunks that
#: were cut one way beside embeddings of text cut another way.
CHUNKER_VERSION = "1.0"

#: Prefix on the rendered heading path that opens every chunk's text. Chosen
#: to be markdown because the same string is what
#: :meth:`ParsedDocument.to_full_text` emits for a heading, so a chunk and the
#: document it came from look the same when a user compares them side by side.
HEADING_PREFIX = "## "

#: Joins the levels of a heading path, matching the separator
#: :attr:`~paperwhisperer.ingestion.schema.Evidence.citation_label` uses for
#: the same data. One separator for one concept, so "3 › Model Scaling" in a
#: chunk and in a citation label are recognisably the same path.
PATH_SEPARATOR = " › "

#: Chunk kinds that own a chunk outright. Merging these into prose is the
#: single failure this task exists to prevent: a table's cells interleaved
#: with the sentence that mentions them cannot be cited, cannot be re-rendered
#: as a grid, and cannot be answered with a number.
STRUCTURAL_KINDS = ("table", "formula", "figure", "caption")

#: Roles that open a new section.
_HEADING_ROLES = frozenset({"title", "heading"})

#: Rough characters-per-token for the Latin script, used by
#: :func:`estimate_tokens`. Four is the figure that matches the published
#: English averages of byte-pair encoders; it is an approximation, and
#: :func:`estimate_tokens` says exactly how wrong it can be.
_CHARS_PER_TOKEN = 4

#: Ideographic and kana ranges, which are roughly one token per character
#: rather than one per four. A CJK document measured at four characters per
#: token would be budgeted at a quarter of its real size, so the two scripts
#: are counted separately instead of averaged together.
_CJK_RE = re.compile("[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")

#: Every non-word, non-space character. Charged one token each, which is how
#: byte-pair encoders actually treat punctuation, and which is the only reason
#: a markdown table grid is not budgeted at a third of its real cost: eight
#: pipes in a two-column row are eight tokens, not two.
_SYMBOL_RE = re.compile(r"[^\w\s]")

#: Characters that may end a sentence. ``;`` is included because a semicolon
#: separates claims and a cut there is one a reader can see; the alternatives
#: are "only break on full stops", which ignores Chinese text entirely, and
#: "break anywhere", which is the legacy behaviour.
_TERMINATORS = frozenset(".!?。．！？；;…")

#: How far past a period the sentence-end rule looks for the next word. Forty
#: characters is far more than any gap between a full stop and the word that
#: follows it, including an indented one.
_LOOKAHEAD = 40

#: LaTeX spans, protected from every cut. Display formulas arrive wrapped in
#: ``$$…$$`` and inline ones in ``$…$``, so a cut inside a matched span would
#: emit a chunk of unbalanced braces -- text that a citation cannot verify and
#: an embedding cannot represent.
_MATH_SPAN_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.DOTALL)

#: The signature of an estimator: text in, a token count out.
TokenEstimator = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """An estimate of how many tokens ``text`` costs, without a tokenizer.

    **What it approximates.** The token count a byte-pair encoder would
    produce, to within roughly fifteen percent on English prose: Latin text is
    charged at :data:`_CHARS_PER_TOKEN` characters per token, CJK and kana at
    one token per character, and every punctuation or symbol character at one
    token. The result is the *maximum* of those three readings rather than
    their sum, because they are three estimates of the same thing and adding
    them would triple-count every letter that is also a word character.

    **What it costs.** It is wrong in both directions and knowingly so.
    Undercounting: code, URLs and dense digit strings tokenize worse than four
    characters per token. Overcounting: LaTeX and markdown, where the symbol
    rule charges a token to every brace and backslash even though an encoder
    frequently merges them. Tables land nearer the truth than a pure
    characters-per-token rule would put them, because a grid is mostly
    punctuation -- that is the case this heuristic exists for, since a table
    chunk is the one place where a low estimate translates directly into a
    request the embedding provider silently truncates.

    **Why not a real tokenizer.** This project has no tokenizer dependency and
    is not acquiring one: chunking runs over an entire corpus during indexing,
    and an exact count has to be paid once per document, on machines that are
    already the slow part of the pipeline. More importantly, an exact count is
    not what a *budget* needs. A budget exists to keep a chunk inside the
    embedding model's window, and the honest way to hold that line with an
    approximate count is headroom: the defaults leave roughly a fifth of a
    512-token window unused, which is several times the error above. The
    estimate is recorded on every chunk as ``token_estimate``, so when a real
    tokenizer is eventually available its verdict can be diffed against ours
    instead of argued about.

    :attr:`ChunkPolicy.estimator` takes this as a parameter, so a caller who
    disagrees can pass ``tiktoken`` in without this module changing.
    """
    if not text:
        return 0
    cjk = _CJK_RE.findall(text)
    remainder = _CJK_RE.sub("", text)
    latin = math.ceil(len(remainder) / _CHARS_PER_TOKEN)
    return len(cjk) + max(latin, len(remainder.split()), len(_SYMBOL_RE.findall(remainder)))


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    """Every knob that decides where a boundary falls.

    Frozen, because a policy is part of a chunk's identity: it is hashed into
    ``index_version``, and a policy mutated after the fact would describe an
    index that no longer corresponds to the vectors stored beside it. A caller
    wanting a different budget builds a different policy.

    The two size limits are not redundant. ``max_tokens`` is the target, and
    it is the only one that applies to a boundary chosen for meaning: a
    section end or a table edge is taken wherever it falls and the resulting
    chunk is whatever size that makes it. ``hard_max_tokens`` is the ceiling
    that applies only when a split is forced, and the gap between the two is
    where a sentence longer than the target is allowed to end rather than
    being cut in half. A single ceiling would have to be either too small to
    hold a long sentence or too large to keep a chunk from carrying a page of
    unrelated material; the acceptance criterion for this task is specifically
    about not breaking a sentence, so the larger number wins inside a split
    and the smaller one wins everywhere else.
    """

    #: Target size for a merged prose chunk, in estimated tokens.
    max_tokens: int = 400
    #: Ceiling for a single split piece. Exceeding it means a sentence is
    #: being cut, which is a last resort, not a preference.
    hard_max_tokens: int = 800
    #: Below this a flushed chunk is merged into the preceding prose chunk of
    #: the same section instead of standing alone.
    min_merge_tokens: int = 80
    #: Cap on the lead-in sentence attached to a table, formula or caption.
    lead_in_tokens: int = 48
    #: Whether the heading path is rendered into the chunk text as well as
    #: recorded as metadata.
    include_heading_path_in_text: bool = True
    #: The estimator the budget is measured in. Injectable so the whole module
    #: can be pointed at a real tokenizer without being rewritten.
    estimator: TokenEstimator = estimate_tokens

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.hard_max_tokens < self.max_tokens:
            raise ValueError("hard_max_tokens must be >= max_tokens")
        if self.min_merge_tokens < 0:
            raise ValueError("min_merge_tokens must be >= 0")
        if self.lead_in_tokens < 0:
            raise ValueError("lead_in_tokens must be >= 0")

    def signature_payload(self) -> dict[str, object]:
        """The parameters that move a boundary, for hashing into the version.

        Only the fields that change where a cut lands. ``estimator`` is absent
        on purpose: it is a measurement, not a boundary, and two policies that
        differ only in how they count produce identical chunks and must not
        force a rebuild between them.
        """
        return {
            "chunker": CHUNKER_VERSION,
            "max_tokens": self.max_tokens,
            "hard_max_tokens": self.hard_max_tokens,
            "min_merge_tokens": self.min_merge_tokens,
            "lead_in_tokens": self.lead_in_tokens,
            "include_heading_path_in_text": self.include_heading_path_in_text,
        }

    def size(self, text: str) -> int:
        """The estimated size of ``text`` under this policy's estimator."""
        return self.estimator(text)


@dataclass(slots=True)
class _Unit:
    """One indivisible piece of chunk text, with where it came from.

    A unit is a whole element, or a whole slice of one element produced by a
    split. Carrying the element on the unit rather than only its id is what
    lets the packer take a locator from the source element and never compute
    one: the locator is copied, never derived, so a chunk cannot end up
    pointing at a page it was assembled near rather than a page it contains.
    """

    element: Element
    text: str
    heading_path: tuple[str, ...]
    kind: str = "prose"
    #: Placed before this unit's text when it continues the previous unit of
    #: the same element. See :func:`_split_segments`.
    joiner: str = ""
    #: A verbatim excerpt of the element immediately before, given to a
    #: structural chunk so that a grid is never indexed without the sentence
    #: that introduces it.
    lead_in: str = ""
    #: True for a caption that belongs to the structural element before it.
    attaches_to_previous: bool = False

    @property
    def is_structural(self) -> bool:
        return self.kind in STRUCTURAL_KINDS


@dataclass(frozen=True, slots=True)
class ChunkingReport:
    """What one chunking run could not do, phrased for whoever has to say it.

    Two lists, both of things that did *not* happen quietly: chunks that came
    out far over budget, and elements that produced no chunk at all. The
    release gate allows neither to be invisible, and neither fits on a
    :class:`~paperwhisperer.ingestion.schema.Chunk` -- "this table is too large
    to embed whole" is a fact about the run rather than about a retrieval unit,
    and a chunk is read back one at a time by a retriever holding no other
    document to compare it with. A separate report is where a fact that only
    exists relative to the rest of the document belongs.

    ``oversize`` holds ``(order, tokens, kind)`` for chunks that came out above
    the *hard* ceiling, not above the target. The distinction is what keeps the
    report worth reading: a chunk a little over the target is the tail-merge
    rule working as designed, and a report that listed those would bury the
    one case a user has to be told about. A table is never split, so a
    full-page results grid lands here rather than being halved; the embedding
    provider will truncate it, and reporting it is what turns that truncation
    into something a caller can show a user instead of something they
    discover from a bad answer.

    ``dropped`` holds ``(element_id, reason)``. An element with no text cannot
    be retrieved, and dropping it is not a smaller index but an index quietly
    missing a passage.

    :attr:`warnings` renders both for a log line or a user-visible warning;
    :attr:`is_clean` is the one-line check a release gate wants.
    """

    oversize: tuple[tuple[int, int, str], ...] = ()
    dropped: tuple[tuple[str, str], ...] = ()

    @property
    def is_clean(self) -> bool:
        return not self.oversize and not self.dropped

    def warnings(self, ceiling: int) -> list[str]:
        """The report as sentences, for a log line or a visible warning."""
        messages = [
            f"chunk {order} is {tokens} estimated tokens of {kind} content, over the "
            f"{ceiling}-token ceiling; the embedding provider will truncate it"
            for order, tokens, kind in self.oversize
        ]
        messages.extend(
            f"element {element_id} produced no chunk: {reason}"
            for element_id, reason in self.dropped
        )
        return messages


# --------------------------------------------------------------------------
# Index version
# --------------------------------------------------------------------------

#: The first field of every index version, so a reader that meets a shape it
#: does not recognise can refuse rather than misparse the fields after it.
_INDEX_VERSION_TAG = "iv1"


def _field(value: object) -> str:
    """Render one index-version field so the separator cannot occur in it.

    Percent-encoding rather than a character filter, because a model name is
    part of the value a human reads when deciding whether to rebuild and
    ``"text-embedding-3-large"`` has to survive that read intact. A name that
    genuinely contains the separator round-trips through
    :func:`parse_index_version` instead of being silently rewritten.
    """
    return urllib.parse.quote(str(value), safe="")


def build_index_version(
    *,
    provider: str,
    model: str,
    dim: int,
    policy: ChunkPolicy | None = None,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """Compose the version string every chunk of one index carries.

    The three embedding fields are parameters rather than configuration reads
    because the index's identity is a fact about the vectors, not about this
    process: a test has no configuration, a bake-off compares two providers
    against one document, and a caller that has already resolved a provider
    should not have it read again behind its back.

    The chunker signature and the parsed-document schema version are included
    for the reason given in the module docstring. The schema version is the
    less obvious of the two and matters just as much: a locator vocabulary
    that changed shape makes every stored locator mean something adjacent to
    what it meant, and an index that kept those chunks is not merely stale, it
    is wrong.
    """
    if dim <= 0:
        raise ValueError(f"embedding dimension must be > 0, got {dim!r}")
    resolved = policy or ChunkPolicy()
    canonical = json.dumps(resolved.signature_payload(), sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    fields = {
        "chunk": signature,
        "schema": schema_version,
        "provider": provider,
        "model": model,
        "dim": dim,
    }
    rendered = "|".join(f"{key}={_field(value)}" for key, value in fields.items())
    return f"{_INDEX_VERSION_TAG}|{rendered}"


def parse_index_version(value: str) -> dict[str, str]:
    """Read a version string back into its fields.

    Returns ``{}`` for anything that is not a version this build wrote,
    including a well-formed string carrying an unknown tag. A caller asking
    "can this store be reused" has to be able to answer no, and a partially
    understood mapping is the one answer that cannot be acted on.
    """
    parts = str(value).split("|")
    if not parts or parts[0] != _INDEX_VERSION_TAG:
        return {}
    fields: dict[str, str] = {}
    for part in parts[1:]:
        key, separator, raw = part.partition("=")
        if not separator or key in fields:
            return {}
        fields[key] = urllib.parse.unquote(raw)
    return fields


def index_version_parts(value: str) -> dict[str, object]:
    """The parsed version with the dimension as an ``int``, or ``{}``.

    The typed accessor the store uses, separated from
    :func:`parse_index_version` so that the string field does not have to be
    cast at each call site, each of which would otherwise invent its own cast
    and its own idea of what to do with a value that is not a number.
    """
    fields = parse_index_version(value)
    if not fields:
        return {}
    try:
        fields["dim"] = int(fields.get("dim", ""))
    except ValueError:
        return {}
    return dict(fields)


# --------------------------------------------------------------------------
# Boundary finding
# --------------------------------------------------------------------------


def _math_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Character ranges a cut may not land inside."""
    return tuple(match.span() for match in _MATH_SPAN_RE.finditer(text))


def _is_sentence_end(text: str, position: int) -> bool:
    """Whether a cut immediately after ``text[position - 1]`` ends a sentence.

    A period only counts when whitespace follows it and the next non-space
    character is not a lowercase letter. Both halves earn their place: without
    the first, "3.14" and "et al.5" split; without the second, "see Fig. 3"
    and "e.g. the model" split mid-clause. The residual error is the
    abbreviation followed by a capital -- "Ref. Smith" still splits -- and the
    alternative, a stop-word list of abbreviations, is a hundred lines of
    guesswork with a worse failure mode, because such a list breaks the first
    time the corpus meets a paper whose field writes abbreviations this build
    has not heard of. A rare early cut is visible to a reader; a list that
    misses an abbreviation is not.
    """
    if position <= 0 or position > len(text):
        return False
    previous = text[position - 1]
    if previous in _TERMINATORS:
        return True
    if previous != ".":
        return False
    rest = text[position : position + 1 + _LOOKAHEAD]
    if len(rest) < 2 or not rest[0].isspace():
        return False
    # The lookahead is bounded because this runs at every position of every
    # window: an unbounded lstrip here would make the scan quadratic in the
    # paragraph, for a check that only ever needs the first word.
    return not rest.lstrip()[:1].islower()


def _safe_cut(text: str, high: int, protected: tuple[tuple[int, int], ...]) -> int:
    """The last cut position in ``text[:high]``, or 0 when there is none.

    Prefers a sentence end, falls back to the last whitespace so that a forced
    cut at least lands between words, and refuses any position inside a
    formula. The whitespace fallback is also what keeps a CJK paragraph from
    being cut mid-syllable: Chinese has no inter-word spaces, so for that
    script a forced cut lands on a sentence end whenever the paragraph
    contains one at all, and on nothing at all when it does not -- which is
    the honest answer for a script with no word boundaries to find.

    The whole window is scanned rather than stopping at the first candidate,
    because a sentence end further back is the better answer, and only a
    window with no sentence end in it at all should fall back to whitespace.
    """
    fallback = 0
    for position in range(min(high, len(text)), 0, -1):
        if any(start < position < end for start, end in protected):
            continue
        if _is_sentence_end(text, position):
            return position
        if not fallback and position < len(text) and text[position].isspace():
            # One past the space, not onto it: a head ending on the space and
            # a remainder that skips it is what makes the next piece start on a
            # whole word. Cutting *at* the space would leave the head holding
            # the first half of that word.
            fallback = position + 1
    return fallback


def _fit(text: str, tokens: int, estimator: TokenEstimator) -> int:
    """The length of the longest prefix of ``text`` the estimator bills at ``tokens``.

    Binary search on the prefix length rather than a characters-per-token
    multiplication, because the estimator is a parameter and a caller who
    passes a real tokenizer should get that tokenizer's answer rather than an
    approximation of it. The cost is ``log2(len)`` estimator calls per piece,
    which is nothing next to the work the same piece does downstream.
    """
    if estimator(text) <= tokens:
        return len(text)
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimator(text[:middle]) <= tokens:
            low = middle
        else:
            high = middle - 1
    return low


def _split_segments(text: str, policy: ChunkPolicy) -> list[tuple[str, str]]:
    """Split one element's text into pieces, each with its joining string.

    The joiner comes back with each piece because pieces re-join when they are
    packed: a cut that followed a period has to return with a space or the
    passage reads "results.Conclusion" and a reader sees a rendering bug. A
    cut taken at whitespace joins with nothing, because that whitespace was
    the separator and re-adding one would invent a second.

    The first piece is never given a joiner: it starts the unit, and a leading
    separator would be stripped again by every consumer downstream.
    """
    if policy.size(text) <= policy.max_tokens:
        return [(text, "")]

    protected = _math_spans(text)
    segments: list[tuple[str, str]] = []
    remaining = text
    joiner = ""
    while policy.size(remaining) > policy.max_tokens:
        # The window widens past the target looking for a sentence end before
        # it settles for a word boundary, because a chunk that overruns its
        # budget is visible in its token estimate and a broken sentence is not.
        limit = max(_fit(remaining, policy.max_tokens, policy.estimator), 1)
        hard = max(_fit(remaining, policy.hard_max_tokens, policy.estimator), 1)
        position = _safe_cut(remaining, hard, protected)
        if position <= 0:
            # Nothing safe inside even the hard window: one sentence longer
            # than the ceiling, or a paragraph with neither whitespace nor a
            # terminator. Cutting at the fitted width is the only option left,
            # and it is the one case where a word may be broken.
            position = limit
        head, remaining = remaining[:position], remaining[position:]
        segments.append((head.strip(), joiner))
        joiner = " " if head.rstrip()[-1:] in _TERMINATORS else ""
        remaining = remaining.lstrip()
    if remaining.strip():
        segments.append((remaining.strip(), joiner))
    return [segment for segment in segments if segment[0]]


def split_prose_at_safe_boundaries(
    text: str, policy: ChunkPolicy | None = None
) -> list[str]:
    """Split ``text`` into pieces that each begin at a sentence or word boundary.

    Public because the property it guarantees -- a piece never starts in the
    middle of a sentence, unless one sentence is longer than the ceiling -- is
    the acceptance criterion for T1.2, and a property that can only be checked
    through the packaging of a chunk is a property that rots quietly. The tests
    assert it directly on real paper text here instead of inferring it from
    chunk boundaries, where the heading prefix and the lead-in sit between the
    reader and the evidence.

    Pieces are stripped, so this is the text as it appears in a chunk rather
    than as it appears in the source. Use the internal split when the
    difference matters.
    """
    return [segment for segment, _ in _split_segments(text, policy or ChunkPolicy())]


def _leading_sentence(text: str, policy: ChunkPolicy) -> str:
    """The trailing run of ``text`` that fits the lead-in budget, verbatim.

    Starts at the *earliest* sentence end whose remainder still fits, so the
    result is the longest lead-in the budget allows rather than only the last
    sentence. Two sentences of lead-in beat one when the sentence naming the
    table is not the sentence immediately before it -- and it frequently is
    not, because a paragraph usually ends by interpreting the figure and the
    figure is introduced two clauses earlier.

    Verbatim because the lead-in is a copy of source text that a citation has
    to be able to point at. A summarising lead-in would put words in a chunk
    that exist nowhere in the document, and the citation verifier would fail
    every quote that touched it. The end of ``text`` rather than its start for
    the same reason: the first sentence of a paragraph introduces the
    paragraph, and the last introduces whatever comes after it, which for a
    table is the thing the reader is about to ask about.

    An empty string when not even the final sentence fits. A lead-in that
    costs more than the content it contextualises is not context, and a
    truncated one would be a third partial sentence somewhere in the index.
    """
    if policy.lead_in_tokens <= 0 or not text:
        return ""
    starts = [
        position
        for position in range(1, len(text) + 1)
        if _is_sentence_end(text, position)
    ]
    if not starts:
        return text.strip() if policy.size(text) <= policy.lead_in_tokens else ""
    for start in starts:
        candidate = text[start:].strip()
        if candidate and policy.size(candidate) <= policy.lead_in_tokens:
            return candidate
    return ""


# --------------------------------------------------------------------------
# Element -> unit
# --------------------------------------------------------------------------


def _copy_locator(locator: SourceLocator) -> SourceLocator:
    """An independent copy of a locator, rectangle included.

    A shared rectangle would be a shared mutable object between a parsed
    document that outlives the index and a chunk that a rebuild will replace.
    The document is the source of truth; nothing downstream should be able to
    move a box in it.
    """
    return replace(
        locator,
        bbox=replace(locator.bbox) if locator.bbox is not None else None,
        heading_path=tuple(locator.heading_path),
    )


def element_content_text(element: Element) -> str:
    """The text a chunk should carry for ``element``, verbatim or with its caption.

    Public because it is the rule, not an implementation detail: a citation
    verifier checking that a quote came from a chunk has to know which part of
    the chunk is source text, and re-deriving the caption-prefix rule would be
    a second copy of it that drifts.

    A table's markdown grid omits the caption, which is the one line that says
    what the columns *are*. Prefixing it restores the sentence the reader has,
    verbatim, and it matters more here than anywhere else: a grid of numbers
    with no column names shares no vocabulary with any question a reader would
    ask about it.
    """
    text = element.text.strip()
    table = element.table
    if table is not None and table.caption.strip():
        caption = table.caption.strip()
        return f"{caption}\n\n{text}" if text else caption
    return text


def _is_indexable(element: Element) -> bool:
    """Whether this element has anything a retriever could return.

    The rule is "non-empty content", with one exception taken from the
    docling adapter so that this module and the quality gate cannot disagree
    about what a table is. The gate's ``empty_table_shells`` rule warns that a
    grid of blanks is usually a picture or a chart frame; the adapter already
    drops such a grid when it carries no caption, and the chunker applies the
    same test, so a document from a weaker adapter cannot smuggle one back in
    as a chunk whose entire text is ``| | |``. A shell *with* a caption is
    kept: the caption is real, retrievable text, and dropping the element would
    delete the only evidence that the region exists at all.

    Everything else is admitted on its text alone, which is the same admission
    test the quality gate's density and coverage rules reason about.
    """
    if element.is_furniture:
        return False
    table = element.table
    if table is not None and table.is_empty_shell and not table.caption.strip():
        return False
    return bool(element_content_text(element))


def _drop_reason(element: Element) -> str:
    """Why an element produced no chunk, in words a user could be shown."""
    table = element.table
    if table is not None and table.is_empty_shell and not table.caption.strip():
        return "a table grid with no content and no caption"
    if element.role == "figure":
        return "a figure with no caption, which has no text to retrieve"
    return "no text"


def _kind_of(element: Element) -> str:
    """The chunk kind for an element, with inline formulas demoted to prose.

    An inline formula belongs inside the sentence that contains it, and the
    canonical schema already says so by distinguishing display from inline.
    Treating both as their own chunk produces a two-token chunk that says only
    ``$x$`` and, worse, cuts the sentence around it in half. A *display*
    formula keeps its own chunk: it is a block in the source, it is what the
    surrounding prose refers to as "Equation 3", and merging it into a
    paragraph would be the flattening this module exists to prevent.
    """
    if element.role == "formula":
        if element.formula is not None and not element.formula.display:
            return "prose"
        return "formula"
    if element.role in {"table", "figure", "caption"}:
        return element.role
    return "prose"


def _heading_level(element: Element) -> int:
    """The depth a heading element opens, defaulting when the adapter gave none."""
    if element.level is not None:
        return element.level
    return 1 if element.role == "title" else 2


def _pieces_for(text: str, kind: str, policy: ChunkPolicy) -> list[tuple[str, str]]:
    """The pieces one element contributes, and how each joins what follows it.

    A structural element is never split, whatever its size. A grid cut in half
    is the exact failure this module replaces, and an over-budget table is
    something a caller can be shown -- it lands in :class:`ChunkingReport` --
    where a shredded one would be indistinguishable from a correct one.
    """
    if kind == "prose":
        return _split_segments(text, policy)
    return [(text, "")]


def _iter_units(
    document: ParsedDocument,
    policy: ChunkPolicy,
    dropped: list[tuple[str, str]] | None = None,
) -> Iterator[_Unit]:
    """Walk the document once, yielding the units the packer consumes.

    One pass, in the document's own reading order, because the lead-in for a
    structural chunk is the element immediately before it; looking backwards in
    a second pass would be a second source of truth about reading order.

    ``dropped`` collects the elements that produce nothing, with the reason.
    It is optional because a caller who only wants chunks has no use for it,
    and a report nobody can switch off is a report nobody keeps accurate.
    """
    stack: list[tuple[int, str]] = []
    previous: _Unit | None = None

    def decline(element: Element, reason: str) -> None:
        if dropped is not None:
            dropped.append((element.element_id, reason))

    for element in document.iter_reading_order():
        text = element_content_text(element)
        if element.role in _HEADING_ROLES:
            if not text:
                # A heading with no text is a running page header a parser
                # mistyped as a section title. It neither opens a section nor
                # carries content, and letting it into the stack would put an
                # empty level into every heading path below it.
                decline(element, "a heading with no text")
                continue
            level = _heading_level(element)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        if not _is_indexable(element):
            decline(element, _drop_reason(element))
            continue
        # The element's own path wins when it has one: both shipped adapters
        # maintain it during their own reading-order pass, and the docling
        # adapter deliberately leaves ``level`` unset, so a path derived from
        # levels here would be strictly worse information than the adapter's.
        # The stack is the fallback for what carries no path of its own -- a
        # plain-text upload, or a document built by hand.
        path = tuple(item for item in element.heading_path if item) or tuple(
            item for _, item in stack
        )
        if (
            element.role in _HEADING_ROLES
            and policy.include_heading_path_in_text
            and path
            and path[-1] == text
        ):
            # The heading is already on every chunk of its section as the
            # rendered path, and indexing its words twice buys one extra copy
            # of the section name per chunk. The unit is still built when the
            # path does not end in this heading's own words, or when the caller
            # turned the prefix off: no configuration may drop source text
            # from the index, and a heading is source text.
            continue

        kind = _kind_of(element)
        lead_in = ""
        attaches = False
        if kind in STRUCTURAL_KINDS and previous is not None and previous.heading_path == path:
            if previous.kind == "prose":
                lead_in = _leading_sentence(previous.text, policy)
            else:
                # A caption written as its own element belongs to the figure or
                # table above it: docling folds captions into their parent, so
                # a standalone caption element means an adapter that did not,
                # and it is the same sentence either way.
                attaches = True
                lead_in = _leading_sentence(previous.lead_in, policy)

        pieces = _pieces_for(text, kind, policy)

        for segment, joiner in pieces:
            unit = _Unit(
                element=element,
                text=segment,
                heading_path=path,
                kind=kind,
                joiner=joiner,
                lead_in=lead_in,
                attaches_to_previous=attaches,
            )
            previous = unit
            yield unit


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def _draft_text(draft: list[_Unit]) -> str:
    """The body text of a draft, without the context that opens a chunk.

    Units from the same element re-join through their own joiner rather than
    through a paragraph break, because a paragraph break would be an
    invention: two pieces of one paragraph are one paragraph, and the only gap
    between them is the one the split consumed.
    """
    parts: list[str] = []
    previous: _Unit | None = None
    for unit in draft:
        if (
            previous is not None
            and unit.joiner
            and not unit.is_structural
            and unit.element.element_id == previous.element.element_id
        ):
            parts[-1] = f"{parts[-1]}{unit.joiner}{unit.text}"
        else:
            parts.append(unit.text)
        previous = unit
    return "\n\n".join(part for part in parts if part)


def _is_prose_draft(draft: list[_Unit]) -> bool:
    return bool(draft) and all(not unit.is_structural for unit in draft)


def _pack(units: list[_Unit], policy: ChunkPolicy) -> list[list[_Unit]]:
    """Group units into the chunks they will become.

    The flush conditions, in the order they are checked, are the whole
    algorithm: a structural element is a wall, a changed heading path is a
    wall, and the token budget is a soft wall. Soft because the budget is
    beaten by meaning every time -- a section that is 600 tokens long produces
    one 600-token chunk rather than a cut in half, and a table is a wall
    whatever it costs.
    """
    drafts: list[list[_Unit]] = []
    buffer: list[_Unit] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        size = policy.size(_draft_text(buffer))
        previous = drafts[-1] if drafts else None
        if (
            size < policy.min_merge_tokens
            and previous is not None
            and _is_prose_draft(previous)
            and previous[0].heading_path == buffer[0].heading_path
        ):
            # An orphan tail -- a nine-word paragraph after a full chunk -- is
            # worse than a slightly oversized one. It embeds to a vector nobody
            # queries for, and it makes the document's coverage look worse than
            # it is. Merging it into the chunk before is safe by construction,
            # because that chunk is in the same section, so this can never
            # span a heading.
            previous.extend(buffer)
        else:
            drafts.append(list(buffer))
        buffer = []

    for unit in units:
        if unit.attaches_to_previous and drafts:
            anchor = drafts[-1]
            if anchor and all(item.is_structural for item in anchor) and (
                anchor[0].heading_path == unit.heading_path
            ):
                anchor.append(unit)
                continue
        if unit.is_structural:
            flush()
            drafts.append([unit])
            continue
        if buffer and buffer[-1].heading_path != unit.heading_path:
            flush()
        if buffer and policy.size(_draft_text([*buffer, unit])) > policy.max_tokens:
            flush()
        buffer.append(unit)
    flush()
    return drafts


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def _anchor_kind(draft: list[_Unit]) -> str:
    """What a draft is made of, named for a message a user will read."""
    kinds = {unit.kind for unit in draft if unit.is_structural}
    if not kinds:
        return "prose"
    return " and ".join(sorted(kinds))


def _chunk_id(index_version: str, identity: str, order: int, element_ids: list[str]) -> str:
    """A content-addressed id for one chunk.

    Derived, not generated, and that is the whole determinism argument. The
    inputs are the index version (so a rebuild under a new embedding model
    produces a different id and cannot be mistaken for the old one), the
    document identity, the chunk's position, and the elements it covers.
    Position is included because one oversized element contributes several
    chunks sharing its element id, and those pieces must not collide.
    """
    payload = "|".join([index_version, identity, str(order), *element_ids])
    return "chunk_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class StructuralChunker:
    """Turns one :class:`ParsedDocument` into a list of :class:`Chunk`.

    A class rather than a bare function so the index version is resolved once
    for the document instead of once per chunk, which is what makes "every
    chunk in this index carries one version" a structural fact rather than a
    convention four call sites each have to remember.

    Instances hold no state between documents and cache nothing: a cache keyed
    on a mutable document is one more way for two runs to disagree, and the
    determinism this task is judged on is worth more than the microseconds it
    would save.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        dim: int,
        policy: ChunkPolicy | None = None,
        index_version: str | None = None,
    ) -> None:
        self.policy = policy or ChunkPolicy()
        self.index_version = index_version or build_index_version(
            provider=provider, model=model, dim=dim, policy=self.policy
        )

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk ``document``, deterministically and in reading order."""
        return self.chunk_with_report(document)[0]

    def chunk_with_report(
        self, document: ParsedDocument
    ) -> tuple[list[Chunk], ChunkingReport]:
        """Chunk ``document`` and say what the run could not do.

        The chunks are identical to :meth:`chunk`'s, byte for byte; the report
        is what a caller shows a user, or what a release gate reads, when it
        needs to know that a chunk came out over budget or that an element
        produced nothing. A second method rather than a parameter, so the
        common call site stays a single expression and skipping the honest
        answer to "why is this another call" -- because most callers do not
        need it -- costs nothing.
        """
        dropped: list[tuple[str, str]] = []
        drafts = _pack(list(_iter_units(document, self.policy, dropped)), self.policy)
        # The document key is the file's content hash, so re-parsing the same
        # bytes produces the same chunk ids and a rebuild can diff. The
        # fallback is for documents that carry no hash; it is stable for as
        # long as the document id is, which is exactly as long as a stored
        # canonical document lives.
        identity = document.sha256 or document.document_id
        chunks = [
            self._materialize(draft, order, identity) for order, draft in enumerate(drafts)
        ]
        _number_split_parts(chunks, drafts)
        # The ceiling rather than the target: a chunk a little over the target
        # is the tail-merge rule working, and listing those would bury the one
        # case worth telling a user about.
        oversize = tuple(
            (item.order, item.token_estimate, _anchor_kind(drafts[item.order]))
            for item in chunks
            if item.token_estimate > self.policy.hard_max_tokens
        )
        return chunks, ChunkingReport(oversize=oversize, dropped=tuple(dropped))

    def _materialize(self, draft: list[_Unit], order: int, identity: str) -> Chunk:
        """Assemble one chunk: context block, verbatim body, provenance."""
        anchor = draft[0]
        blocks: list[str] = []
        if self.policy.include_heading_path_in_text and anchor.heading_path:
            blocks.append(HEADING_PREFIX + PATH_SEPARATOR.join(anchor.heading_path))
        lead_in = next((unit.lead_in for unit in draft if unit.lead_in), "")
        if lead_in:
            blocks.append(lead_in)
        # Everything above is generated: the heading line is built from the
        # section path, and the lead-in is a sentence copied from the document
        # purely so a structural chunk (a table, a formula) is not naked. It
        # is kept as ``context_prefix`` so a consumer can tell it apart from
        # the body. A citation verifier that cannot tell them apart will
        # "find" a quote that exists only in text this product wrote and report
        # a fabricated citation as verified.
        context_prefix = "\n\n".join(block for block in blocks if block)
        body = _draft_text(draft)
        text = "\n\n".join(block for block in (*blocks, body) if block)
        element_ids: list[str] = []
        for unit in draft:
            if unit.element.element_id not in element_ids:
                element_ids.append(unit.element.element_id)
        return Chunk(
            chunk_id=_chunk_id(self.index_version, identity, order, element_ids),
            text=text,
            element_ids=tuple(element_ids),
            locator=_copy_locator(anchor.element.locator),
            heading_path=anchor.heading_path,
            order=order,
            token_estimate=self.policy.size(text),
            index_version=self.index_version,
            context_prefix=context_prefix,
        )



def _number_split_parts(chunks: list[Chunk], drafts: list[list[_Unit]]) -> None:
    """Label the pieces an oversized element was split into.

    When one element exceeds the token budget it becomes several chunks that
    all carry the same ``locator`` and the same single ``element_id``. Without
    a part number a consumer cannot address "the second of three", and worse,
    neighbour expansion's "expand to the parent" step can never fire against
    a real index: every piece looks like the whole element. (The `Chunk`
    dataclass grew ``part_index``/``part_count`` for exactly this; a field
    nothing assigns is a field nothing reads.)

    Only drafts that are a single unit of a single element are numbered. A
    merged draft is not a piece of anything -- it is the chunk -- and
    numbering it would claim a split that did not happen.

    Mutates in place: the chunks were just built here and are not yet shared
    with any cache.
    """
    groups: dict[str, list[int]] = {}
    for position, draft in enumerate(drafts):
        if len(draft) != 1:
            continue
        element_id = draft[0].element.element_id
        if not element_id:
            continue
        groups.setdefault(element_id, []).append(position)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        total = len(positions)
        for nth, position in enumerate(positions, start=1):
            # part_index is 1-based: "part 0 of 3" reads as a bug, and 0 is
            # already the value that means "not split".
            chunks[position].part_index = nth
            chunks[position].part_count = total


def chunk_document(
    document: ParsedDocument,
    *,
    provider: str,
    model: str,
    dim: int,
    policy: ChunkPolicy | None = None,
    index_version: str | None = None,
) -> list[Chunk]:
    """Chunk ``document`` for the given embedding configuration.

    ``provider``, ``model`` and ``dim`` are required keyword arguments with no
    defaults. A default would be the fastest route to two indexes built from
    different models and one store, which is the failure plan section 8 calls
    out: the vectors would be dimensionally compatible, would be served
    together, and would answer with nonsense at full confidence. A caller that
    cannot name its embedding configuration does not yet have an index.

    ``index_version`` overrides the composed string, for a caller replaying a
    previous build and needing the new chunks to carry the old version so that
    a diff shows moved passages rather than a wholesale id change.
    """
    return StructuralChunker(
        provider=provider,
        model=model,
        dim=dim,
        policy=policy,
        index_version=index_version,
    ).chunk(document)


def chunk_document_from_config(
    document: ParsedDocument, policy: ChunkPolicy | None = None
) -> list[Chunk]:
    """Chunk ``document`` using the embedding settings in ``core.config``.

    The convenience wrapper, for the ordinary path where the running
    deployment already knows its embedding configuration. It reads
    ``EMBEDDING_MODEL`` and ``EMBEDDING_DIMENSION``, which T0.8 put in config
    for every other consumer of the embedding endpoint, and asks
    :func:`paperwhisperer.core.config.embedding_provider_name` for the provider
    name.

    The provider comes from that function rather than from the
    ``EMBEDDING_PROVIDER`` constant, and the difference is the whole point: the
    constant is computed once at import, so a name read from it cannot be
    checked against the embedder that will fill the index -- and a test that
    wants to verify the rule has to be able to ask it about settings it chooses.
    The function derives the name from the same endpoint the embedder names
    itself after, so the two are equal by construction. They used to be derived
    independently from a constant, which under the shipped defaults made every
    index write fail. The model and dimension are still read with ``getattr``
    and defaults, for the reason
    :func:`paperwhisperer.ingestion.quality.thresholds_from_config` does it: the
    config block and this module are written in parallel, and a module that
    raises ``AttributeError`` because a settings name has not landed yet takes
    indexing down with it.

    A caller that knows its provider passes it to :func:`chunk_document`
    instead. Having configuration to read is not a reason to guess: the caller
    that can name its provider should.
    """
    provider = config.embedding_provider_name()
    model = str(getattr(config, "EMBEDDING_MODEL", "") or "")
    dimension = getattr(config, "EMBEDDING_DIMENSION", 0)
    try:
        dim = int(dimension)
    except (TypeError, ValueError):
        dim = 0
    if not model or dim <= 0:
        raise ValueError(
            "core.config carries no usable embedding model and dimension "
            "(EMBEDDING_MODEL / EMBEDDING_DIMENSION); pass provider, model and "
            "dim to chunk_document() instead"
        )
    return chunk_document(document, provider=provider, model=model, dim=dim, policy=policy)
