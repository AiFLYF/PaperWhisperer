"""The scoring rules every tier computes the same way.

Both tiers' searches have the same shape: the backend narrows the candidate
set, and then something in Python decides what each candidate is actually
worth. This module is that something. Putting it in one place is what makes
"the same test suite passes on both tiers" mean something beyond "both tiers
return some rows" — the expected scores in the shared suite are exact
numbers, and a backend-native ``ts_rank_cd`` or SQLite's ``bm25()`` would
make them two different expectations.

The cost of that decision is real and worth stating: a term-frequency-weighted
score computed by the server is cheaper than this one. It is given up so that
a retrieval regression reproduces on the machine you are standing on, and so
that the CJK handling below is *the* handling rather than whatever the
database's tokenizer happened to do.

Two rules are worth more than the rest:

**Chinese is handled by character bigrams, not by a word segmenter.** A
Chinese query has no spaces, so a whitespace tokenizer hands back the whole
phrase as one term, and no document that says ``自注意力机制`` will ever match
a query for ``注意力``. Bigrams of the character run are the standard fix and
are what both tiers index. A run of one character has no bigram, and is
emitted as itself rather than dropped — a one-character CJK query is rare but
real, and dropping it would answer "no results" to a question the document
does answer.

**The score is bounded, and the order is total.** Coverage is a ratio, the
phrase bonus is capped so a perfect token match cannot score above a perfect
phrase match, and every consumer sorts by ``(-score, chunk_id)``. Without the
tiebreak, two equally-good chunks swap places between runs, and ``top_k``
becomes a coin toss.
"""

from __future__ import annotations

import math
import struct
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: Characters treated as one script and indexed by character bigram. The
#: ranges are the ones that actually occur in papers this product parses; a
#: full Unicode script table would be a dependency for no visible gain.
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x2E80, 0x2EFF),  # CJK radicals supplement
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0x20000, 0x2A6DF),  # CJK extension B
)

#: How much a full-phrase match adds on top of token coverage. Half, rather
#: than "a lot", because a phrase match is a stronger signal than a bag of
#: terms but not a certainty — and the cap keeps the total inside [0, 1] so
#: the number means "how much of the query is present", which is what a
#: caller can reason about.
PHRASE_BONUS = 0.5

#: Width of the excerpt :func:`make_snippet` cuts, in characters.
SNIPPET_WIDTH = 160


def _is_cjk(char: str) -> bool:
    point = ord(char)
    return any(low <= point <= high for low, high in _CJK_RANGES)


def normalize(text: str) -> str:
    """Fold text to the form matching and scoring are defined on.

    NFKC first, then case folding, then every non-alphanumeric character
    replaced by a space, then runs of whitespace collapsed. NFKC matters for
    more than tidiness: full-width Latin (``ＦＰＳ``), compatibility ideographs
    and the various Unicode space characters all reach this project through
    pasted abstracts, and without it ``ＣＦＴ`` and ``CFT`` are different
    documents.
    """
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in folded).split())


def _script_segments(run: str) -> list[tuple[bool, str]]:
    """Split a normalized run into maximal same-script pieces.

    ``bool`` is ``True`` for a CJK run. The split exists because
    ``使用attention机制`` is one whitespace token and one search term, which
    matches nothing a user would call relevant.
    """
    segments: list[tuple[bool, str]] = []
    current: list[str] = []
    current_is_cjk: bool | None = None
    for char in run:
        is_cjk = _is_cjk(char)
        if current and current_is_cjk is not None and is_cjk != current_is_cjk:
            segments.append((current_is_cjk, "".join(current)))
            current = []
        current_is_cjk = is_cjk
        current.append(char)
    if current:
        segments.append((bool(current_is_cjk), "".join(current)))
    return segments


def tokenize(text: str) -> tuple[str, ...]:
    """Break text into the terms matching and scoring work on.

    Non-CJK runs become one term each. CJK runs become character bigrams,
    with a single-character run emitted as itself.
    """
    tokens: list[str] = []
    for run in normalize(text).split():
        for is_cjk, segment in _script_segments(run):
            if not is_cjk or len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(segment[index : index + 2] for index in range(len(segment) - 1))
    return tuple(tokens)


@dataclass(slots=True)
class LexicalQuery:
    """A query prepared once and reused across every candidate.

    Preparing is O(length of the query); scoring against N candidates
    re-tokenizing the same query N times would make a single lookup scale with
    the index, which is precisely what the index exists to prevent.
    """

    raw: str
    normalized: str
    tokens: tuple[str, ...]
    unique_tokens: frozenset[str]

    @classmethod
    def prepare(cls, query: str) -> LexicalQuery:
        tokens = tokenize(query)
        return cls(
            raw=str(query or ""),
            normalized=normalize(query),
            tokens=tokens,
            unique_tokens=frozenset(tokens),
        )

    @property
    def is_empty(self) -> bool:
        return not self.unique_tokens


def score_lexical(
    query: LexicalQuery, document_normalized: str, document_tokens: Iterable[str]
) -> float:
    """Score one candidate against a prepared query. ``0.0`` means no match.

    Two components: the share of the query's distinct terms the document
    contains, plus a bonus when the normalized query appears as a contiguous
    run. Contiguity is the difference between "these words co-occur somewhere
    in a section" and "this is the sentence you were looking for", and for a
    product whose output is a citable quote that is worth more than any
    term-frequency weighting.
    """
    if query.is_empty:
        return 0.0
    present = query.unique_tokens & frozenset(document_tokens)
    if not present:
        return 0.0
    coverage = len(present) / len(query.unique_tokens)
    phrase = bool(query.normalized) and query.normalized in document_normalized
    return min(1.0, coverage + (PHRASE_BONUS if phrase else 0.0))


def make_snippet(text: str, query: LexicalQuery, width: int = SNIPPET_WIDTH) -> str:
    """Cut a readable excerpt around the first place the query matched.

    The window is taken from the *original* text, not the normalized one.
    Normalized text would be lowercased, stripped of punctuation and cut into
    bigrams, which is the perfect index and a terrible quotation — and the
    snippet exists to be shown to a reader deciding whether to open the
    evidence panel.

    When nothing is found the excerpt is the head of the text. A caller
    displaying a result the scorer said matches, but which the match window
    cannot locate, is better served by the opening of the passage than by an
    empty string.
    """
    body = str(text or "")
    if not body.strip():
        return ""
    span = max(1, int(width))
    haystack = body.casefold()
    # Longest term first, then alphabetical, so the anchor is the most
    # informative match and the choice does not depend on set iteration.
    anchors = sorted({token for token in query.tokens if token}, key=lambda t: (-len(t), t))
    position = -1
    for anchor in anchors:
        found = haystack.find(anchor)
        if found >= 0 and (position < 0 or found < position):
            position = found
    if position < 0:
        return body[:span]
    start = max(0, position - span // 4)
    excerpt = body[start : start + span]
    return f"…{excerpt}…" if start > 0 or start + span < len(body) else excerpt


def sort_key(hit_score: float, key: str) -> tuple[float, str]:
    """The total order every search result is returned in.

    Negated score, then the key ascending. The tiebreak is what makes
    ``top_k`` a function of the data rather than of the iteration order the
    backend happened to use.
    """
    return (-float(hit_score), str(key))


# --------------------------------------------------------------------------
# Vector maths
# --------------------------------------------------------------------------


def l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """Scale a vector to unit length, refusing the zero vector.

    With unit-length inputs, cosine similarity *is* the dot product, so the
    similarity search is one matrix product and nothing else. The zero vector
    has no direction, so its similarity to anything is undefined rather than
    zero; returning ``0.0`` for it would put a number in a field that
    promises a measured similarity, and a caller filtering on ``min_score``
    could not tell the difference between "orthogonal" and "not a vector".
    """
    values = [float(item) for item in vector]
    if not values:
        raise ValueError("向量维度为 0：无法归一化。")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("零向量无法计算余弦相似度，请检查 embedding 输出。")
    return tuple(value / norm for value in values)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two equal-width vectors.

    The result is clamped to ``[-1, 1]``. Accumulated floating-point error can
    push an exactly-parallel pair to ``1.0000000000000002``, and a caller
    asserting ``score == pytest.approx(1.0)`` on a self-query would then fail
    for a reason that has nothing to do with the code under test.
    """
    if len(left) != len(right):
        raise ValueError(f"向量维度不一致：{len(left)} != {len(right)}")
    total = sum(a * b for a, b in zip(left, right, strict=True))
    return max(-1.0, min(1.0, total))


def pack_f32(values: Sequence[float]) -> bytes:
    """Pack floats as little-endian float32, the one on-disk vector format.

    Little-endian is written out explicitly rather than inherited from the
    host. The bytes are what lands in the file, and a store written on one
    machine and read on another must not depend on the second machine's byte
    order.
    """
    return struct.pack(f"<{len(values)}f", *(float(value) for value in values))


def unpack_f32(blob: bytes) -> tuple[float, ...]:
    """Inverse of :func:`pack_f32`."""
    if len(blob) % 4:
        raise ValueError(f"向量 BLOB 长度 {len(blob)} 不是 4 的倍数，数据已损坏。")
    return struct.unpack(f"<{len(blob) // 4}f", blob)
