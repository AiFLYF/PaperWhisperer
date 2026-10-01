"""Server-side quote verification: the gate between a claim and quotation marks.

This module answers one question, deterministically: **does this string appear
in the passage this citation points at?** It is task T1.6 (PW-010) of the
development plan, and it exists because the product is Evidence-first: a
direct quote is a claim that the paper says exactly these words, so that claim
has to be checked by something other than the model that produced it.

The obvious implementation is to ask the model. "Here is the source, is this a
verbatim quote? Answer yes or no." That is rejected for three reasons that
have nothing to do with model quality:

* it is not reproducible — the same quote verifies on Tuesday and not on
  Wednesday, and a citation that appears and disappears is worse than one
  that was never offered;
* it cannot be audited — "the model said it was a quote" is not evidence, and
  the failure is invisible in every log and every field;
* it fails silently in the direction that matters. A model asked to confirm
  its own quote says yes, so the failure mode is not a wrong answer but no
  answer at all.

So verification here is string comparison, and nothing else. No LLM, no
network, no clock, no randomness: the same inputs produce the same verdict
every time, which is what makes the acceptance criterion ("100%
deterministic unit tests") a property of the code rather than a hope about a
test run.

Three design commitments follow from that, and they are the reason this file
is not thirty lines of ``quote in text``:

1. **The scope is an element, not a document.** A quote is verified against
   the specific :class:`~paperwhisperer.ingestion.schema.Element` the citation
   names, and the caller may widen that to a set of elements (a page, a
   chunk) but never silently to the whole document. Searching the whole
   document is the failure this module is most careful about: a model writes
   a fluent sentence that does not appear in the paragraph it cited, the
   sentence turns out to appear on page 9, and the citation now points at
   page 9 wearing the clothes of an answer about page 3. The reader follows
   the citation, lands on page 9, and concludes the paper said something in a
   context that changes its meaning. Nothing crashed; the product lied.

2. **Normalisation is a written-down list, not a fuzzy matcher.** The
   obvious move is to reach for the existing
   :func:`paperwhisperer.core.text.clean_extracted_text`, and it is the wrong
   tool for this one job. That helper is a *cleanup*: it rewrites the text it
   is given, collapsing runs of whitespace and flattening line structure,
   which is right for text about to be handed to a model and fatal here — a
   citation has to name character offsets in the text as stored, and a
   collapsed copy cannot be mapped back to them. So the comparison projection
   is built beside it, keeps an index map into the original, and never
   replaces the document's text with anything. Every
   transformation below is one a *renderer* would make invisible, and each one
   is individually tested, individually named on the result, and individually
   defensible. What is deliberately not here matters as much: no case
   folding, no punctuation stripping, no edit distance, no token overlap. A
   quote that differs from the source in a visible way is not a quote of that
   source.

3. **A failed verification cannot be rendered as a quote.** The result type
   does not return a boolean, it returns a presentation verdict:
   :class:`ClaimPresentation.DIRECT_QUOTE` or
   :class:`ClaimPresentation.PARAPHRASE`. Only a verified result can reach the
   first, the rendered text of a paraphrase has its surrounding quotation
   marks stripped before it is handed out, and
   :attr:`CitationVerification.verified_text` raises rather than returning
   the model's string when verification did not pass. A UI cannot accidentally
   wrap an unverified sentence in quotation marks, because the text it would
   wrap is not available in a form that invites wrapping. This is the release
   gate in section 7 of the plan: alignment at 100%, and anything that does
   not align degrades to a paraphrase that carries no quotation marks.

Hard cases, decided rather than assumed (each is pinned in the tests):

* **A quote spanning an element boundary** is refused, with its own outcome
  (:attr:`AlignmentOutcome.SPANS_ELEMENTS`) rather than being reported as
  "not found". The text really is in the passage, and saying so is more
  useful than "no match" — but it cannot be attributed to one element, and an
  element's locator is the smallest thing this schema can point at. A bbox
  highlight would be wrong at one end at least. Refusing is the only answer
  that cannot produce a citation pointing at the wrong region.
* **A quote repeated in the same element** is not ambiguous. The lowest
  offset wins — ``str.find`` semantics — and the result reports
  ``occurrence_count`` plus the other places it occurs, so a UI can disclose
  the repetition. Every occurrence is a real occurrence, and the one being
  cited is a real one; refusing would be refusing a true statement.
* **A quote repeated across a multi-element scope** resolves the same way, to
  the first element in reading order, with the matched element's own locator
  reported. ``ParsedDocument`` fixes reading order in its constructor, so
  "first" is a stable property of the document and not of a dict iteration.
* **A line-wrapped hyphen has a known cost, and the cost is paid
  deliberately.** "transduc-\n tion" de-hyphenates to "transduction", so a
  reader who retypes the line break as a space and writes "transduc- tion"
  does not align; and a compound split at its real hyphen, "English-\nto-
  German", loses a character the page shows. The obvious fix is to delete a
  space after a hyphen, and it is refused: a space is visible, and a rule
  that deletes visible characters to make a citation succeed is a rule that
  can point at the wrong thing ("state of the-art" folding onto
  "state-of-the-art"). Both quotes degrade to a paraphrase instead. On the
  current corpus this accounts for *every* retyped quote that fails to align,
  which the corpus test asserts rather than estimates: if a quote ever fails
  for a different reason, that assertion fails with it rather than joining a
  statistic.
* **LaTeX is compared byte-for-byte** after the general normalisations. The
  whitespace collapse applies (a formula's spacing is not a different
  formula), but nothing else does: no ``\\,``/``~``/``\\left`` folding, no
  equivalence between ``$$x^2$$`` and ``x^2``. A TeX-aware comparison is a
  large, subtle surface where two different renderings are declared
  identical, and the release gate asks for exact alignment, not for a
  semantic one.
* **A table cell** is quoted against the element's markdown grid, because
  that grid is the addressable rendering of the table, and the result names
  the cell (``R2C3``) when the quote falls inside exactly one. A quote that
  runs across cells is still a genuine substring of the table's own text, so
  it verifies and reports the cell *range* it covers rather than pretending it
  came from one cell.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from paperwhisperer.ingestion.schema import Element, Evidence, ParsedDocument, SourceLocator

#: Provenance marker for a character that no rule changed.
_IDENTITY = "identity"


class AlignmentMode(str, Enum):
    """How a verified quote was found, in decreasing order of strength.

    ``EXACT`` is a byte-for-byte substring of the element's own text.
    ``NORMALIZED`` is a match after the documented rules in
    :data:`NORMALIZATION_RULES`. Both are alignment; the difference is what
    the reader is entitled to assume about the quotation, and the enum keeps
    that difference on the value instead of in a log line.
    """

    EXACT = "exact"
    NORMALIZED = "normalized"


class ClaimPresentation(str, Enum):
    """How a claim carrying this verdict may be shown to a reader.

    This is the field the release gate is made of. ``DIRECT_QUOTE`` is the
    only value under which quotation marks are allowed, and a result can only
    carry it when alignment actually passed. ``PARAPHRASE`` is not a softer
    version of the same thing: it is the instruction that the claim must be
    rendered without quotation marks, because the string is the model's
    wording, not the paper's.
    """

    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"


class AlignmentOutcome(str, Enum):
    """Why a verification ended the way it did, success or failure.

    Every value maps to exactly one user-visible explanation, and the
    failures are not interchangeable: "your quote is not in this passage" and
    "your quote runs across two paragraphs" call for different things from the
    caller — the first is a prompt problem, the second is a chunking one.
    """

    ALIGNED = "aligned"
    NOT_FOUND = "not_found"
    SPANS_ELEMENTS = "spans_elements"
    EMPTY_QUOTE = "empty_quote"
    NO_SCOPE = "no_scope"


#: Outcomes under which the claim may never be presented as a quote. Written
#: out rather than inferred from "not aligned", so that adding an outcome
#: forces a decision about its presentation instead of inheriting a default
#: that happens to be safe today.
_DEGRADED_OUTCOMES = frozenset({
    AlignmentOutcome.NOT_FOUND,
    AlignmentOutcome.SPANS_ELEMENTS,
    AlignmentOutcome.EMPTY_QUOTE,
    AlignmentOutcome.NO_SCOPE,
})


class UnverifiedQuoteError(LookupError):
    """Raised when a caller asks for the verified text of a failed check.

    Exists because the failure it prevents is not hypothetical: a template
    that does ``{{ claim.quote }}`` is one refactor away from rendering an
    unverified string, and the string is right there on the result. Making
    the accessor raise turns "someone forgot to branch on the outcome" into a
    visible error at the first call instead of a wrong quote in production.
    """


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class NormalizationRule:
    """One documented, named, individually tested transformation.

    ``rule_id`` is what lands on a result, in a test name and in an eval
    report, so it is a wire value: renaming one invalidates a stored report.
    ``description`` is the sentence that has to survive review — the reason
    the transformation cannot change what the reader sees, which is the only
    justification for applying it to a citation at all.
    """

    rule_id: str
    description: str


#: The complete set of normalisations, in the order they are applied. The
#: order is load-bearing and is not interchangeable: de-hyphenation has to see
#: the line break before whitespace collapse removes it, and the ligature
#: expansion has to see the ligature before anything can look inside it. The
#: list is a closed world — a comparison that cannot be written down here does
#: not happen.
NORMALIZATION_RULES: tuple[NormalizationRule, ...] = (
    NormalizationRule(
        rule_id="soft_hyphen",
        description=(
            "A soft hyphen (U+00AD) is deleted. It is a discretionary hyphen a "
            "typesetter embedded and no renderer draws, so a quote that omits "
            "it is quoting the same visible characters."
        ),
    ),
    NormalizationRule(
        rule_id="line_break_hyphen",
        description=(
            "A hyphen at the end of a line, with a letter on both sides, is "
            "removed together with the break. This is one word that a "
            "line-wrap split in two; the page shows one word and the "
            "extracted text shows its two halves."
        ),
    ),
    NormalizationRule(
        rule_id="fullwidth_ascii",
        description=(
            "Full-width ASCII (U+FF01-U+FF5E) folds to its half-width "
            "counterpart. These glyphs occupy a full em and are what a CJK "
            "PDF's text layer emits for a keyboard layout the reader never "
            "sees; the characters they name are the ASCII ones."
        ),
    ),
    NormalizationRule(
        rule_id="quote_marks",
        description=(
            "Curly, guillemet and CJK corner quotation marks fold to the "
            "straight double quote, and the single forms to the apostrophe. "
            "Only the mark changes; which mark a typesetter picked is not a "
            "property of the sentence being quoted."
        ),
    ),
    NormalizationRule(
        rule_id="ligature",
        description=(
            "The ﬁ/ﬂ/ﬀ/ﬃ/ﬄ ligatures expand to their letter sequences. PDF "
            "text layers emit them and nobody types them, so a quote written "
            "on a keyboard cannot otherwise match a perfectly ordinary word."
        ),
    ),
    NormalizationRule(
        rule_id="whitespace_collapse",
        description=(
            "Every run of whitespace becomes one space, and leading and "
            "trailing whitespace is removed. Line wrapping, indentation and "
            "the several space characters a PDF text layer emits are all "
            "invisible on the page."
        ),
    ),
)

#: Curly, guillemet and CJK quotation marks, mapped to what they name.
#:
#: The single forms are kept separate from the double forms because "don't"
#: and "don’t" are the same word while 'x' and ‘x’ are not the same
#: quotation. Folding the inner marks to the double quote as well would let a
#: quote swap its nesting and still verify.
_QUOTE_MARK_FOLD = {
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "«": '"', "»": '"', "‹": "'", "›": "'",
    "「": '"', "」": '"', "『": '"', "』": '"',
    "【": '"', "】": '"', "〝": '"', "〞": '"', "〟": '"',
}

#: Ligatures expanded, and the ones deliberately left alone.
#:
#: U+FB05 and U+FB06 (the long-s ligatures) are excluded even though the other
#: five are not, because 'ſ' is a real letter change and not a typographic
#: shortcut: folding it would make "past" and "paſt" verify against each
#: other.
_LIGATURE_FOLD = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

#: Space characters that a text layer emits and ``str.isspace`` is not always
#: reached for by whatever produced the text. Both are invisible.
_EXTRA_WHITESPACE = frozenset({"　", " "})

#: Word-shaped characters either side of a de-hyphenation. A hyphen
#: between two letters is a split word; a hyphen between a bullet and a
#: number is a list item, and joining those would invent a word.
_SOFT_HYPHEN = "­"
#: Whitespace that may sit between a trailing hyphen and the letter that
#: continues its word. The run has to contain a line break, and the character
#: after it has to be a letter — an indented continuation line is ordinary in
#: a PDF text layer, and the indentation is removed by the whitespace rule
#: anyway, so refusing to look past it would fail on real documents for no
#: safety gained.
_LINE_BREAK_PADDING = " \t\r\n"


def _normalize_soft_hyphen(
    chars: list[tuple[str, int, str]],
) -> tuple[list[tuple[str, int, str]], bool]:
    """Drop U+00AD. See :data:`NORMALIZATION_RULES` for why it is invisible."""
    kept: list[tuple[str, int, str]] = []
    fired = False
    for char, index, provenance in chars:
        if char == _SOFT_HYPHEN:
            fired = True
            # The deleted character leaves no entry behind, so the rule would
            # be unreportable on a match that needed it. It is attributed to
            # the character before the hyphen, which every match spanning the
            # deletion necessarily contains: a match cannot begin at a
            # character that is not there.
            if kept:
                previous = kept[-1]
                kept[-1] = (previous[0], previous[1], "soft_hyphen")
            continue
        kept.append((char, index, provenance))
    return kept, fired


def _normalize_line_break_hyphen(
    chars: list[tuple[str, int, str]],
) -> tuple[list[tuple[str, int, str]], bool]:
    """Join a word that a line wrap split with a hyphen.

    The hyphen, the padding and the line break are all removed together and
    the first character of the second half is kept, because the span this
    produces has to be reported in the source's coordinates: the character
    that survives at the seam is the one whose offset bounds the match, so the
    deleted hyphen and break fall inside the reported span and the highlight
    covers the whole word as the page draws it.
    """
    folded: list[tuple[str, int, str]] = []
    fired = False
    position = 0
    while position < len(chars):
        char, index, provenance = chars[position]
        if (
            char == "-"
            and position > 0
            and chars[position - 1][0].isalnum()
            and position + 1 < len(chars)
        ):
            lookahead = position + 1
            saw_line_break = False
            while lookahead < len(chars) and chars[lookahead][0] in _LINE_BREAK_PADDING:
                if chars[lookahead][0] == "\n":
                    saw_line_break = True
                lookahead += 1
            if (
                saw_line_break
                and lookahead < len(chars)
                and chars[lookahead][0].isalnum()
            ):
                folded.append(
                    (chars[lookahead][0], chars[lookahead][1], "line_break_hyphen")
                )
                fired = True
                position = lookahead + 1
                continue
        folded.append((char, index, provenance))
        position += 1
    return folded, fired


def _normalize_fullwidth_ascii(
    chars: list[tuple[str, int, str]],
) -> tuple[list[tuple[str, int, str]], bool]:
    """Fold U+FF01-U+FF5E onto ASCII. The glyphs differ; the characters do not."""
    folded: list[tuple[str, int, str]] = []
    fired = False
    for char, index, provenance in chars:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            folded.append((chr(code - 0xFEE0), index, "fullwidth_ascii"))
            fired = True
        else:
            folded.append((char, index, provenance))
    return folded, fired


def _normalize_quote_marks(
    chars: list[tuple[str, int, str]],
) -> tuple[list[tuple[str, int, str]], bool]:
    """Fold quotation-mark glyphs. See ``_QUOTE_MARK_FOLD`` for the pairs."""
    folded: list[tuple[str, int, str]] = []
    fired = False
    for char, index, provenance in chars:
        replacement = _QUOTE_MARK_FOLD.get(char)
        if replacement is None:
            folded.append((char, index, provenance))
            continue
        folded.append((replacement, index, "quote_marks"))
        fired = True
    return folded, fired


def _normalize_ligature(
    chars: list[tuple[str, int, str]],
) -> tuple[list[tuple[str, int, str]], bool]:
    """Expand the five ligatures a PDF text layer emits and a keyboard does not.

    One source character becomes several normalised ones, all carrying the same
    source offset, which is what keeps the reported span correct: the span is
    derived from the outermost surviving characters, and the ligature they came
    from is inside it.
    """
    expanded: list[tuple[str, int, str]] = []
    fired = False
    for char, index, provenance in chars:
        expansion = _LIGATURE_FOLD.get(char)
        if expansion is None:
            expanded.append((char, index, provenance))
            continue
        fired = True
        expanded.extend((piece, index, "ligature") for piece in expansion)
    return expanded, fired


def _normalize_whitespace_collapse(
    chars: list[tuple[str, int, str]],
) -> tuple[list[tuple[str, int, str]], bool]:
    """Collapse whitespace runs to one space and drop both ends.

    A run that is already a single space in the middle of the string is left
    completely alone, provenance included. Reporting it as a change would make
    ``applied_rules`` say "whitespace collapse" about a match that used no
    normalisation at all, and a verdict that names rules it did not use is
    worse than one that names none.

    A synthesised space carries this rule's provenance even when a literal
    space was among the characters collapsed, because "these two sides lined
    up only because they wrapped their lines differently" is the honest
    description of why the match happened.
    """
    collapsed: list[tuple[str, int, str]] = []
    fired = False
    position = 0
    while position < len(chars):
        char, index, provenance = chars[position]
        if not (char.isspace() or char in _EXTRA_WHITESPACE):
            collapsed.append((char, index, provenance))
            position += 1
            continue
        run_end = position
        while run_end < len(chars) and (
            chars[run_end][0].isspace() or chars[run_end][0] in _EXTRA_WHITESPACE
        ):
            run_end += 1
        # Leading and trailing runs are dropped outright; neither is inside a
        # quote the reader would see, and emitting a space for one would make
        # "a" match " a".
        at_edge = not collapsed or run_end >= len(chars)
        if not at_edge and run_end - position == 1 and char == " ":
            collapsed.append(chars[position])
        elif at_edge:
            # A dropped run leaves no provenance behind, so the rule is
            # attributed to the character next to it — the first one when the
            # run led the string, the last one when it ended it. A match can
            # neither start nor end with whitespace that is not in the output.
            fired = True
            if collapsed:
                last = collapsed[-1]
                collapsed[-1] = (last[0], last[1], "whitespace_collapse")
        else:
            collapsed.append((" ", index, "whitespace_collapse"))
            fired = True
        position = run_end
    return collapsed, fired


#: The pipeline, in application order, bound to the ids they report. A rule
#: that is not in this table cannot fire, and a rule in
#: :data:`NORMALIZATION_RULES` that is not in this table has a description and
#: no implementation — which is exactly the inconsistency worth having a test
#: for rather than a comment.
_RULE_PIPELINE = (
    ("soft_hyphen", _normalize_soft_hyphen),
    ("line_break_hyphen", _normalize_line_break_hyphen),
    ("fullwidth_ascii", _normalize_fullwidth_ascii),
    ("quote_marks", _normalize_quote_marks),
    ("ligature", _normalize_ligature),
    ("whitespace_collapse", _normalize_whitespace_collapse),
)


def normalization_rule_ids() -> tuple[str, ...]:
    """Every rule that can run, in the order it is applied.

    Public because the order is a documented part of the contract — a caller
    reporting a normalized match lists these, and an eval report pins the
    corpus against them — and because the way to check that a declared rule is
    a real rule is to read this list, not the pipeline table inside it.
    """
    return tuple(rule.rule_id for rule in NORMALIZATION_RULES)


@dataclass(slots=True, frozen=True)
class MatchedSpan:
    """Half-open ``[start, end)`` character offsets into the source text.

    The source is the *element's own text*, not the normalised projection, so
    a caller can slice, highlight or log it directly. That matters for the
    normalised case: the span of a de-hyphenated match covers the hyphen and
    the line break that normalisation removed, because the reader needs the
    whole word highlighted, not the two halves the extracted text happens to
    have been cut into.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"a matched span must be non-empty and non-negative: {self}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def extract(self, source: str) -> str:
        """The matched characters of ``source``."""
        return source[self.start : self.end]

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MatchedSpan:
        return cls(start=int(payload["start"]), end=int(payload["end"]))


@dataclass(slots=True, frozen=True)
class NormalizedText:
    """Normalised text plus everything needed to talk about the original.

    Three parallel sequences over one length, because a normalised string on
    its own has lost everything a citation needs:

    * ``text`` — the characters being compared;
    * ``offsets`` — for each normalised character, the index of the *source*
      character it came from, so a match found in normalised space can be
      reported as a span the evidence viewer can highlight;
    * ``provenance`` — the rule that produced each character, so a result can
      name the rules that actually mattered for the match it found rather than
      every rule that happened to fire somewhere in a 3000-character element.
    """

    text: str
    offsets: tuple[int, ...]
    provenance: tuple[str, ...]
    #: Rules that changed this string somewhere, including by deleting a
    #: character. A deletion leaves no provenance behind — the trimmed
    #: leading space of " a " simply is not in the output — so without this
    #: the rules that matter for a *whole string* would be unreportable, and
    #: a quote aligned only because both sides were trimmed would be reported
    #: as having used no normalisation at all.
    fired_rules: tuple[str, ...] = ()

    @property
    def applied_rules(self) -> tuple[str, ...]:
        """Rules that changed something, in canonical order."""
        fired = set(self.fired_rules) | set(self.provenance)
        return tuple(
            rule.rule_id for rule in NORMALIZATION_RULES if rule.rule_id in fired
        )

    def rules_within(self, start: int, end: int) -> tuple[str, ...]:
        """Rules that produced the characters in ``[start, end)``."""
        fired = set(self.provenance[start:end]) - {_IDENTITY}
        return tuple(
            rule.rule_id for rule in NORMALIZATION_RULES if rule.rule_id in fired
        )

    def map_span(self, start: int, end: int) -> MatchedSpan:
        """Translate a span in this text into a span in the source text."""
        if start < 0 or end > len(self.offsets) or start >= end:
            raise ValueError(
                f"span [{start}, {end}) is outside a normalised text of "
                f"length {len(self.offsets)}"
            )
        return MatchedSpan(start=self.offsets[start], end=self.offsets[end - 1] + 1)


def normalize_for_alignment(text: str) -> NormalizedText:
    """Normalise ``text`` for comparison, keeping the map back to the original.

    Exposed because the eval runner and the tests need to assert on what a
    rule did, not only on whether a match eventually happened. Deterministic
    and total: the same string always produces the same ``NormalizedText``,
    and any string produces one.
    """
    staged: list[tuple[str, int, str]] = [
        (char, index, _IDENTITY) for index, char in enumerate(str(text))
    ]
    fired: set[str] = set()
    for rule_id, rule in _RULE_PIPELINE:
        staged, did_fire = rule(staged)
        if did_fire:
            fired.add(rule_id)
    return NormalizedText(
        text="".join(char for char, _, _ in staged),
        offsets=tuple(index for _, index, _ in staged),
        provenance=tuple(rule_id for _, _, rule_id in staged),
        fired_rules=tuple(
            rule.rule_id for rule in NORMALIZATION_RULES if rule.rule_id in fired
        ),
    )


# --------------------------------------------------------------------------
# Matching within one element
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _Match:
    """One alignment found in one element's text."""

    mode: AlignmentMode
    span: MatchedSpan
    occurrence_count: int
    applied_rules: tuple[str, ...]


def _count_overlapping(haystack: str, needle: str) -> int:
    """Occurrences of ``needle``, counting overlaps.

    ``str.count`` is used for nothing here. "aa" occurs four times in "aaaaa"
    by the only definition a reader would use, and ``str.count`` says two,
    because it resumes past a whole match. A citation whose count cannot be
    checked by the reader is not a count.
    """
    if not needle:
        return 0
    total = 0
    position = haystack.find(needle)
    while position != -1:
        total += 1
        position = haystack.find(needle, position + 1)
    return total


def _align_in_text(quote: str, source: str, normalized_source: NormalizedText) -> _Match | None:
    """Find ``quote`` in one element's text, exactly or after normalising.

    Exact is tried first and wins outright. A quote that appears verbatim has
    no business being reported as a normalised match: a caller that wanted the
    weaker guarantee asked for it, and a caller that did not would rather know
    the passage is byte-identical.
    """
    if not quote:
        return None

    exact_at = source.find(quote)
    if exact_at != -1:
        return _Match(
            mode=AlignmentMode.EXACT,
            span=MatchedSpan(exact_at, exact_at + len(quote)),
            occurrence_count=_count_overlapping(source, quote),
            applied_rules=(),
        )

    normalized_quote = normalize_for_alignment(quote)
    if not normalized_quote.text:
        return None
    found_at = normalized_source.text.find(normalized_quote.text)
    if found_at == -1:
        return None
    end = found_at + len(normalized_quote.text)
    fired = set(normalized_source.rules_within(found_at, end))
    fired.update(normalized_quote.applied_rules)
    return _Match(
        mode=AlignmentMode.NORMALIZED,
        span=normalized_source.map_span(found_at, end),
        occurrence_count=_count_overlapping(normalized_source.text,
                                            normalized_quote.text),
        applied_rules=tuple(
            rule.rule_id for rule in NORMALIZATION_RULES if rule.rule_id in fired
        ),
    )


def _table_cell_spans(element: Element) -> list[tuple[int, int, int, int]]:
    """Where each cell of an element's table lives in its rendered text.

    Recovered by walking the cells in the order ``Table.to_markdown`` writes
    them and locating each one in the element's own text, rather than by
    reimplementing the markdown layout. A second copy of that layout would be
    a second opinion waiting to drift: the day the renderer changes, the copy
    keeps reporting the old cell boundaries and every table citation starts
    pointing at the wrong cell with no test failing.

    Cells whose text does not appear at all — empty cells, or a cell the
    renderer rewrote — are skipped rather than guessed at.
    """
    table = element.table
    if table is None:
        return []
    spans: list[tuple[int, int, int, int]] = []
    cursor = 0
    for row_index, row in enumerate(table.rows):
        for column_index, cell in enumerate(row):
            raw = str(cell)
            if not raw.strip():
                continue
            escaped = raw.replace("|", "\\|")
            found_at = element.text.find(raw, cursor)
            found_text = raw
            if found_at == -1 and escaped != raw:
                found_at = element.text.find(escaped, cursor)
                found_text = escaped
            if found_at == -1:
                continue
            spans.append((row_index, column_index, found_at, found_at + len(found_text)))
            cursor = found_at + len(found_text)
    return spans


def _cell_of(element: Element, span: MatchedSpan) -> tuple[str, str]:
    """The single cell containing ``span``, or the range of cells it covers.

    Returns ``(cell, cell_range)`` where ``cell`` is ``"R2C3"`` when the whole
    match sits inside one cell and empty otherwise, and ``cell_range`` is
    always populated when the element has a table, in the same notation.
    """
    spans = _table_cell_spans(element)
    if not spans:
        return "", ""
    covered = [
        (row, column)
        for row, column, start, end in spans
        if start < span.end and span.start < end
    ]
    if not covered:
        return "", ""
    first, last = covered[0], covered[-1]
    cell_range = (
        f"R{first[0] + 1}C{first[1] + 1}"
        if first == last
        else f"R{first[0] + 1}C{first[1] + 1}:R{last[0] + 1}C{last[1] + 1}"
    )
    # A cell is named only when it *contains* the match. Overlapping one cell
    # and spilling into the "| " that renders between cells is a match the
    # table does not contain, and naming that cell would be a claim about a
    # value the source never wrote as one cell.
    inside = [
        (row, column, start, end)
        for row, column, start, end in spans
        if start <= span.start and span.end <= end
    ]
    cell = (
        f"R{inside[0][0] + 1}C{inside[0][1] + 1}"
        if len(inside) == 1
        else ""
    )
    return cell, cell_range


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


def _reading_order(elements: Iterable[Element]) -> list[Element]:
    """The elements sorted the way a reader meets them.

    ``element_id`` is the final tiebreaker, and it is not decoration: a scope
    of two elements that share an order and a locator — which a two-column
    layout produces routinely — has to resolve to the same winner on every run
    and on every machine. Python's sort is stable, so without the tiebreaker
    the result would depend on the order the caller's list happened to be in.
    """
    return sorted(
        elements, key=lambda element: (element.order, element.locator.sort_key, element.element_id)
    )


@dataclass(slots=True)
class _ScopeText:
    """Several elements' text joined into one searchable string.

    Built only to answer the boundary question — "is this quote in the
    passage, but not in any one element?" — and never to produce a match. A
    match that comes from here is refused, so the only thing that has to be
    right about it is the answer it gives.
    """

    text: str
    #: Which element each character came from, ``-1`` for the injected
    #: separators.
    element_index: tuple[int, ...]


def _scope_text(elements: Sequence[Element]) -> _ScopeText:
    """Join elements for the boundary check, recording which element each char came from."""
    pieces: list[str] = []
    owners: list[int] = []
    for position, element in enumerate(elements):
        normalized = normalize_for_alignment(element.text)
        pieces.append(normalized.text)
        owners.extend([position] * len(normalized.text))
        if position < len(elements) - 1:
            pieces.append(" ")
            owners.append(-1)
    return _ScopeText(text="".join(pieces), element_index=tuple(owners))


def _spanning_elements(quote: str, elements: Sequence[Element]) -> tuple[str, ...]:
    """Element ids a quote runs across, or empty when it is not in the passage.

    Reported as its own outcome rather than folded into "not found" because
    the two mean different things to whoever has to fix something: a quote
    that is nowhere in the passage is a hallucination to be regenerated, and
    one that runs across a paragraph break is a chunking boundary the model
    cannot see.
    """
    if len(elements) < 2:
        return ()
    scope = _scope_text(elements)
    normalized_quote = normalize_for_alignment(quote)
    if not normalized_quote.text:
        return ()
    found_at = scope.text.find(normalized_quote.text)
    if found_at == -1:
        return ()
    end = found_at + len(normalized_quote.text)
    touched = {
        position
        for position in scope.element_index[found_at:end]
        if position >= 0
    }
    if len(touched) < 2 and -1 not in scope.element_index[found_at:end]:
        return ()
    return tuple(elements[position].element_id for position in sorted(touched))


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

#: User-visible explanations, one per failure. Every one of them says the same
#: thing in the end — the claim is being shown as a paraphrase, without
#: quotation marks — because a reader who sees a quote-shaped sentence with no
#: quotation marks is owed an explanation, not just an absence.
_MESSAGES = {
    AlignmentOutcome.EMPTY_QUOTE: (
        "引文为空，已改为转述呈现（不带引号）"
    ),
    AlignmentOutcome.NO_SCOPE: (
        "引用的证据没有可校验的原文，已改为转述呈现（不带引号）"
    ),
    AlignmentOutcome.NOT_FOUND: (
        "引文未在所引证据的原文中逐字找到，已改为转述呈现（不带引号）"
    ),
    AlignmentOutcome.SPANS_ELEMENTS: (
        "引文跨越了多个段落，无法归属到单一来源位置，已改为转述呈现（不带引号）"
    ),
}


@dataclass(slots=True, frozen=True)
class CitationVerification:
    """The verdict on one quote, in the form a caller has to act on.

    Note what is not here: a score, and a boolean. A similarity number would
    let a caller draw a threshold and a threshold is a place where this
    project has promised not to put judgement — the same argument the quality
    gate makes about collapsing its flags, and for the same reason. What is
    here is the verdict (``outcome``), how it was reached (``mode``), where
    (``span``, ``locator``, ``element_id``), and what may be shown to a reader
    (``presentation``).

    ``presentation`` is a property derived from ``outcome``, not a field a
    caller can set, so there is no way to construct a result that calls a
    failed verification a quote. That is the release gate expressed as a
    type: the state that must not be rendered in quotation marks is a state
    that cannot be built.

    ``source_text`` is the element's own text, carried rather than looked up,
    because a verdict stored today has to still render next month against a
    document that may since have been re-parsed. It is the same argument the
    schema makes about locators being values rather than pointers.
    """

    quote: str
    outcome: AlignmentOutcome
    span: MatchedSpan | None = None
    mode: AlignmentMode | None = None
    source_text: str = ""
    element_id: str = ""
    locator: SourceLocator | None = None
    evidence_id: str = ""
    occurrence_count: int = 0
    #: Every other place the quote occurs inside the same scope. A reader can
    #: be told "this phrase also appears on p. 7", which is information a
    #: single span throws away; the cap keeps a boilerplate phrase from
    #: turning one citation into a page of identical locators.
    occurrence_locators: tuple[dict[str, Any], ...] = ()
    applied_rules: tuple[str, ...] = ()
    matched_cell: str = ""
    matched_cell_range: str = ""
    spanned_element_ids: tuple[str, ...] = ()

    @property
    def aligned(self) -> bool:
        return self.outcome is AlignmentOutcome.ALIGNED

    @property
    def is_direct_quote(self) -> bool:
        """Whether this text may be shown as a quote of the source."""
        return self.presentation is ClaimPresentation.DIRECT_QUOTE

    @property
    def quote_marks_allowed(self) -> bool:
        """The single question a template asks. Spelled out rather than
        aliased to ``is_direct_quote`` so that a caller reading the field name
        has to read the whole name."""
        return self.is_direct_quote

    @property
    def presentation(self) -> ClaimPresentation:
        """How this claim may be shown, classified rather than defaulted.

        The classification is a lookup with no fallback on purpose. A new
        outcome added to :class:`AlignmentOutcome` without being classified
        here raises instead of quietly rendering as a paraphrase — which is
        the safe direction to fail, but still a failure, because "safe by
        accident" is how an unclassified outcome becomes a quoted one after
        the next refactor.
        """
        if self.outcome is AlignmentOutcome.ALIGNED:
            return ClaimPresentation.DIRECT_QUOTE
        if self.outcome in _DEGRADED_OUTCOMES:
            return ClaimPresentation.PARAPHRASE
        raise ValueError(
            f"alignment outcome {self.outcome.value!r} has no declared "
            "presentation; add it to the degraded set or give it a rule"
        )

    @property
    def message(self) -> str:
        """The user-visible explanation. Chinese, because the UI is."""
        if self.outcome is AlignmentOutcome.ALIGNED:
            if self.mode is AlignmentMode.EXACT:
                return "已逐字回源校验，与原文完全一致"
            rules = "、".join(self.applied_rules) or "无"
            return f"归一化后与原文一致（{rules}）"
        return _MESSAGES[self.outcome]

    @property
    def verified_text(self) -> str:
        """The matched characters, taken from the source and nothing else.

        Raises when verification failed, rather than returning the quote the
        caller supplied. See :class:`UnverifiedQuoteError` for why the
        difference is worth a crash.
        """
        if not self.aligned or self.span is None:
            raise UnverifiedQuoteError(
                f"quote was not verified ({self.outcome.value}); it must be "
                "presented as a paraphrase, not rendered as a quotation"
            )
        return self.span.extract(self.source_text)

    def render(self) -> RenderedClaim:
        """The claim as a caller may show it.

        A verified claim carries the *source's* characters, not the model's,
        so what the reader sees inside the quotation marks is the paper's
        wording even if the model sent something subtly different that still
        aligned. A failed claim carries the model's text with any surrounding
        quotation marks removed, because the one thing a degraded render must
        never contain is a pair of quotation marks around unverified text.
        """
        if self.aligned and self.span is not None:
            return RenderedClaim(
                text=self.verified_text,
                presentation=ClaimPresentation.DIRECT_QUOTE,
                outcome=self.outcome,
                mode=self.mode,
                span=self.span,
                locator=self.locator,
                element_id=self.element_id,
                evidence_id=self.evidence_id,
                applied_rules=self.applied_rules,
                matched_cell=self.matched_cell,
                matched_cell_range=self.matched_cell_range,
                message=self.message,
            )
        return RenderedClaim(
            text=_strip_wrapping_quotes(self.quote),
            presentation=ClaimPresentation.PARAPHRASE,
            outcome=self.outcome,
            locator=None,
            element_id=self.element_id,
            evidence_id=self.evidence_id,
            spanned_element_ids=self.spanned_element_ids,
            message=self.message,
        )

    def mark_evidence(self, evidence: Evidence) -> Evidence:
        """The evidence with ``is_direct_quote`` set from this verdict.

        A copy, never an in-place write. ``Evidence.is_direct_quote`` is
        defined by the schema as "set only after the quote has been checked",
        and a verifier that mutates the evidence it was handed would leave
        the flag true after a later re-check came back ``NOT_FOUND``.
        """
        if not self.aligned or self.span is None:
            return replace(evidence, is_direct_quote=False)
        return replace(evidence, is_direct_quote=True, text=self.verified_text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quote": self.quote,
            "outcome": self.outcome.value,
            "presentation": self.presentation.value,
            "quote_marks_allowed": self.quote_marks_allowed,
            "mode": self.mode.value if self.mode else None,
            "span": self.span.to_dict() if self.span else None,
            "source_text": self.source_text,
            "element_id": self.element_id,
            "locator": self.locator.to_dict() if self.locator else None,
            "evidence_id": self.evidence_id,
            "occurrence_count": self.occurrence_count,
            "occurrence_locators": [dict(item) for item in self.occurrence_locators],
            "applied_rules": list(self.applied_rules),
            "matched_cell": self.matched_cell,
            "matched_cell_range": self.matched_cell_range,
            "spanned_element_ids": list(self.spanned_element_ids),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CitationVerification:
        """Rebuild a stored verdict.

        ``presentation`` is not read back: it is a function of ``outcome``, and
        accepting it here would be a way to persist a result whose outcome
        says one thing and whose presentation says another.
        """
        locator_payload = payload.get("locator")
        span_payload = payload.get("span")
        mode_value = payload.get("mode")
        return cls(
            quote=str(payload.get("quote") or ""),
            outcome=AlignmentOutcome(str(payload.get("outcome") or AlignmentOutcome.NOT_FOUND)),
            span=MatchedSpan.from_dict(span_payload) if span_payload else None,
            mode=AlignmentMode(str(mode_value)) if mode_value else None,
            source_text=str(payload.get("source_text") or ""),
            element_id=str(payload.get("element_id") or ""),
            locator=SourceLocator.from_dict(locator_payload) if locator_payload else None,
            evidence_id=str(payload.get("evidence_id") or ""),
            occurrence_count=int(payload.get("occurrence_count") or 0),
            occurrence_locators=tuple(
                dict(item) for item in payload.get("occurrence_locators") or ()
            ),
            applied_rules=tuple(str(item) for item in payload.get("applied_rules") or ()),
            matched_cell=str(payload.get("matched_cell") or ""),
            matched_cell_range=str(payload.get("matched_cell_range") or ""),
            spanned_element_ids=tuple(
                str(item) for item in payload.get("spanned_element_ids") or ()
            ),
        )


#: Pairs stripped from a degraded render. A single outer pair only: repeated
#: stripping would eat a quote that legitimately ends inside a quotation, which
#: is a smaller loss than leaving marks around an unverified sentence, but
#: both are bad and one is bounded while the other is not.
_WRAPPING_QUOTE_PAIRS = (
    ('"', '"'),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
    ("《", "》"),
    ("〝", "〞"),
    ("＂", "＂"),
)


def _strip_wrapping_quotes(text: str) -> str:
    """Remove one matched pair of surrounding quotation marks, if present.

    An inner value that is only whitespace is *not* treated as a stripped
    quote: a text of exactly ``"…"`` carries no claim, and returning an empty
    string here would turn a formatting oddity into a claim that renders as
    nothing at all.
    """
    stripped = str(text).strip()
    for opening, closing in _WRAPPING_QUOTE_PAIRS:
        if (
            len(stripped) < len(opening) + len(closing)
            or not stripped.startswith(opening)
            or not stripped.endswith(closing)
        ):
            continue
        inner = stripped[len(opening) : len(stripped) - len(closing)].strip()
        if inner:
            return inner
    return stripped


@dataclass(slots=True, frozen=True)
class RenderedClaim:
    """A claim that is ready to be displayed, carrying its own presentation.

    The guard lives in ``__post_init__``: a ``DIRECT_QUOTE`` without a span is
    refused, because that combination is the misuse this whole module exists
    to prevent and a frozen dataclass that accepted it would be a hole in the
    contract. ``uses_quotation_marks`` is a property derived from
    ``presentation``, so a caller cannot store a claim and then read back a
    flag that disagrees with it.

    ``locator`` is populated only for a verified claim. A degraded render has
    no location, and that is deliberate rather than a missing field: a
    location chip next to a paraphrase is a chip the reader will read as a
    citation, which is the impression the release gate exists to prevent.
    """

    text: str
    presentation: ClaimPresentation
    outcome: AlignmentOutcome
    message: str = ""
    mode: AlignmentMode | None = None
    span: MatchedSpan | None = None
    locator: SourceLocator | None = None
    element_id: str = ""
    evidence_id: str = ""
    applied_rules: tuple[str, ...] = ()
    matched_cell: str = ""
    matched_cell_range: str = ""
    spanned_element_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.presentation is ClaimPresentation.DIRECT_QUOTE and self.span is None:
            raise ValueError(
                "a direct-quote render must carry the span it was verified "
                "against; there is no way to present a quotation whose source "
                "is unknown"
            )
        if self.presentation is ClaimPresentation.PARAPHRASE and self.locator is not None:
            raise ValueError(
                "a paraphrase must not carry a locator: a location chip beside "
                "an unverified sentence reads as a citation"
            )

    @property
    def uses_quotation_marks(self) -> bool:
        return self.presentation is ClaimPresentation.DIRECT_QUOTE

    @property
    def quote_text(self) -> str:
        """The text that may be wrapped in quotation marks; raises otherwise."""
        if not self.uses_quotation_marks:
            raise UnverifiedQuoteError(
                f"claim was not verified ({self.outcome.value}) and must be "
                "rendered as a paraphrase without quotation marks"
            )
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "presentation": self.presentation.value,
            "uses_quotation_marks": self.uses_quotation_marks,
            "outcome": self.outcome.value,
            "message": self.message,
            "mode": self.mode.value if self.mode else None,
            "span": self.span.to_dict() if self.span else None,
            "locator": self.locator.to_dict() if self.locator else None,
            "element_id": self.element_id,
            "evidence_id": self.evidence_id,
            "applied_rules": list(self.applied_rules),
            "matched_cell": self.matched_cell,
            "matched_cell_range": self.matched_cell_range,
            "spanned_element_ids": list(self.spanned_element_ids),
        }


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def _degradation(outcome: AlignmentOutcome, quote: str, **extra: Any) -> CitationVerification:
    """Build a failed verdict, with every field that says "do not quote this"."""
    return CitationVerification(quote=quote, outcome=outcome, **extra)


def _occurrence_payload(element: Element, span: MatchedSpan) -> dict[str, Any]:
    return {
        "element_id": element.element_id,
        "locator": element.locator.to_dict(),
        "span": span.to_dict(),
    }


#: How many other places a verified quote is allowed to advertise itself.
#: Four is enough for "this phrase is boilerplate and recurs through the
#: paper", which is the only thing a reader can act on, and small enough that
#: a citation payload stays readable.
_MAX_DISCLOSED_OCCURRENCES = 4


def verify_quote(quote: str, element: Element) -> CitationVerification:
    """Verify ``quote`` against exactly one element.

    The narrowest and the most-used entry point: an element *is* the smallest
    thing this schema can point a reader at, so a quote verified against an
    element is verified against something a citation can name. Nothing
    outside this element is searched, which is why a fluent sentence that
    appears on page 9 does not rescue a quote the model attributed to page 3.
    """
    if not str(quote).strip():
        return _degradation(AlignmentOutcome.EMPTY_QUOTE, str(quote),
                            element_id=element.element_id)
    if not element.text.strip():
        return _degradation(AlignmentOutcome.NO_SCOPE, str(quote),
                            element_id=element.element_id, locator=element.locator)

    match = _align_in_text(str(quote), element.text, normalize_for_alignment(element.text))
    if match is None:
        return _degradation(AlignmentOutcome.NOT_FOUND, str(quote),
                            element_id=element.element_id, locator=element.locator)

    return _aligned_result(str(quote), element, match, [(element, match)], evidence_id="")


def verify_quote_in_scope(
    quote: str,
    elements: Sequence[Element],
    *,
    evidence_id: str = "",
) -> CitationVerification:
    """Verify ``quote`` against a set of elements — a page, a chunk, a unit.

    Widening the scope is the caller's decision and stays visible: the
    elements are sorted into reading order first, so "the first match" means
    the first one a reader would meet, and the result reports which element
    actually matched. A quote that is in the scope but in no single element
    gets its own outcome rather than being blamed on neither.
    """
    scope = _reading_order(elements)
    if not scope:
        return _degradation(AlignmentOutcome.NO_SCOPE, str(quote), evidence_id=evidence_id)
    if len(scope) == 1:
        return replace(verify_quote(quote, scope[0]), evidence_id=evidence_id)

    if not str(quote).strip():
        return _degradation(AlignmentOutcome.EMPTY_QUOTE, str(quote),
                            element_id=scope[0].element_id, evidence_id=evidence_id)

    matches: list[tuple[Element, _Match]] = []
    for element in scope:
        if not element.text.strip():
            continue
        match = _align_in_text(str(quote), element.text, normalize_for_alignment(element.text))
        if match is not None:
            matches.append((element, match))

    if not matches:
        spanned = _spanning_elements(str(quote), scope)
        outcome = (
            AlignmentOutcome.SPANS_ELEMENTS if spanned else AlignmentOutcome.NOT_FOUND
        )
        return _degradation(
            outcome,
            str(quote),
            element_id=scope[0].element_id,
            evidence_id=evidence_id,
            spanned_element_ids=spanned,
        )

    return _aligned_result(str(quote), matches[0][0], matches[0][1], matches,
                           evidence_id=evidence_id)


def _aligned_result(
    quote: str,
    element: Element,
    match: _Match,
    all_matches: Sequence[tuple[Element, _Match]],
    *,
    evidence_id: str,
) -> CitationVerification:
    """Assemble the success verdict, including where else the quote occurs."""
    cell, cell_range = _cell_of(element, match.span)
    occurrences = [
        _occurrence_payload(occ_element, occ_match.span)
        for occ_element, occ_match in all_matches
        if not (occ_element is element and occ_match.span == match.span)
    ]
    return CitationVerification(
        quote=quote,
        outcome=AlignmentOutcome.ALIGNED,
        span=match.span,
        mode=match.mode,
        source_text=element.text,
        element_id=element.element_id,
        locator=element.locator,
        evidence_id=evidence_id,
        occurrence_count=sum(occ_match.occurrence_count for _, occ_match in all_matches),
        occurrence_locators=tuple(occurrences[:_MAX_DISCLOSED_OCCURRENCES]),
        applied_rules=match.applied_rules,
        matched_cell=cell,
        matched_cell_range=cell_range,
    )


def _same_source_unit(one: SourceLocator, other: SourceLocator) -> bool:
    """Whether two locators name the same page, slide, sheet or paragraph.

    A bbox is deliberately ignored. The locator on an ``Evidence`` is the
    chunk's, and a chunk rarely carries the geometry its elements carry;
    comparing boxes would make every lookup from a chunk miss and push
    callers toward the whole-document search this module refuses to offer.
    An unresolvable locator matches nothing, because "somewhere in this
    document" is the scope this module exists to refuse.
    """
    if one.kind != other.kind:
        return False
    if one.kind == "pdf_page":
        return one.page is not None and one.page == other.page
    if one.kind == "slide":
        return one.slide is not None and one.slide == other.slide
    if one.kind == "docx_element":
        return one.paragraph is not None and one.paragraph == other.paragraph
    if one.kind == "sheet_range":
        return bool(one.sheet) and one.sheet == other.sheet
    if one.kind == "txt_offset":
        return one.offset_start is not None and one.offset_start == other.offset_start
    return False


def _scope_for_evidence(evidence: Evidence, document: ParsedDocument) -> list[Element]:
    """The elements a retrieved passage is allowed to be verified against.

    Two ways to recover them, tried in order, and never anything wider:

    1. every element sharing the evidence's page/slide/sheet/paragraph — the
       common case, since a chunk does not leave its page;
    2. every element whose text appears inside the evidence's own text — used
       when the evidence carries no usable locator, e.g. one built from a
       chunk whose elements inherited ``unknown``.

    The second is a containment test and cannot widen the scope by accident:
    an element that is not part of the cited passage does not appear inside
    the passage's text.
    """
    if evidence.locator.is_resolvable:
        same_page = [
            element
            for element in document.iter_reading_order()
            if _same_source_unit(element.locator, evidence.locator)
        ]
        if same_page:
            return same_page

    passage = normalize_for_alignment(evidence.text).text
    if not passage:
        return []
    return [
        element
        for element in document.iter_reading_order()
        if element.text.strip()
        and normalize_for_alignment(element.text).text in passage
    ]


def verify_evidence_quote(
    quote: str,
    evidence: Evidence,
    document: ParsedDocument,
) -> CitationVerification:
    """Verify ``quote`` against the passage ``evidence`` was retrieved from.

    This is the entry point the Q&A path uses, and it is deliberately the only
    one that starts from a retrieval artifact. The scope it derives is the
    evidence's own page or its own text, never the document: a citation that
    cannot say which passage it came from has nothing to verify against, and
    answering "not found" for that is the correct and useful result. The
    elements it settled on are on the result, so a caller can see how wide the
    search actually was.
    """
    scope = _scope_for_evidence(evidence, document)
    if not scope:
        return _degradation(
            AlignmentOutcome.NO_SCOPE,
            str(quote),
            evidence_id=evidence.evidence_id,
            locator=evidence.locator if evidence.locator.is_resolvable else None,
        )
    return verify_quote_in_scope(quote, scope, evidence_id=evidence.evidence_id)
