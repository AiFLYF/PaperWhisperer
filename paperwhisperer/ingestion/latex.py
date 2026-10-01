"""What "the same formula" means, written down as a contract.

A quote gate that passes only when the model echoes LaTeX byte for byte fails
constantly, because a PDF text layer and a language model do not spell a
formula the same way — and a gate that fails constantly is a gate users learn
to ignore. So this module states, as data, exactly which differences are
treated as noise and which are not, and refuses everything it cannot decide.

Three properties matter, and each of them exists because of a failure this
project has already had.

**A verdict of ``DIFFERENT`` is a claim, so ``UNCOMPARABLE`` has to exist.**
A fraction that survived parsing as an inline image has no LaTeX at all. Its
text is a caption-sized gap. Calling that "different from ``\\frac{a}{b}``"
would report a fabricated formula for one the parser could not read — a false
accusation produced by a comparison that never happened. The honest answer to
"are these the same formula" when one side is not a formula is *I cannot tell*,
and it is a separate verdict rather than a ``False`` folded into ``True``.

**Every tolerance is a rule with a name and a reason.** The list is
:data:`LATEX_EQUIVALENCE_RULES`, and the ones in it are not guesses: each is
something a real PDF in ``eval/corpus`` demonstrably produces. Ligatures
because a Type 1 font emits U+FB01 for ``fi`` (72 times in
``formula_dense/maldacena_large_n``). Unicode maths because the same text
layer emits ``α`` and ``≤`` rather than ``\\alpha`` and ``\\le``. Font glyph
names leaking through as ``/greaterorequalslant`` because
``formula_dense/green_tao_primes`` does exactly that. A tolerance nobody can
trace to an observation is a tolerance that will eventually fold two different
formulas together.

**What is deliberately *not* folded.** Bold, blackboard-bold and calligraphic
wrappers survive, because weight is something a reader can see in the
rendered formula and folding ``\\mathbf{v}`` onto ``v`` would declare two
visually different expressions identical. Upright wrappers (``\\mathrm``,
``\\operatorname``, ``\\text``) are folded, because uprightness is invisible
in a text layer and a gate that demands it fails for no reader-visible reason.
The line is *would a reader looking at the two rendered formulas see a
difference?*, and it is the same line the quote gate draws when it refuses to
delete a visible space.

The entry points are :func:`canonical_latex` (one formula, normalised, with
the rules that fired), :func:`compare_latex` (two formulas and a verdict) and
:func:`latex_defects` (what is wrong with this string, for a renderer to say
out loud). :func:`find_inline_math` and :func:`join_latex` exist for the two
robustness cases that are not about comparison: a formula the model wrote into
its own prose, and a formula the parser split across two elements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


#: A named tolerance, with the reason it exists.
#:
#: Deliberately a data structure rather than a comment attached to a function.
#: The rules run in the order listed, and a test asserts that each one has an
#: effect, so a rule that stops mattering — because a backend changed, or a
#: later rule subsumes it — fails a test instead of quietly widening the
#: tolerance of a release gate.
@dataclass(frozen=True, slots=True)
class NormalizationRule:
    """One documented tolerance."""

    rule_id: str
    description: str


LATEX_EQUIVALENCE_RULES: tuple[NormalizationRule, ...] = (
    NormalizationRule(
        "outer_delimiters",
        "Outer $...$, $$...$$, \\(...\\) and \\[...\\] are removed. The same "
        "formula arrives delimited from a text layer, from MinerU and from a "
        "model, and the delimiters are a property of the carrier rather than "
        "of the expression.",
    ),
    NormalizationRule(
        "ligature_forms",
        "The typographic ligatures U+FB00-U+FB06 become their ASCII letter "
        "pairs. A Type 1 font maps 'fi' to a single glyph and the text layer "
        "hands back U+FB01; eval/corpus/formula_dense/maldacena_large_n.pdf "
        "contains 72 of them.",
    ),
    NormalizationRule(
        "unicode_maths",
        "Unicode maths and Greek letters become the LaTeX commands they are "
        "the glyphs of (α -> \\alpha, ≤ and ⩽ -> \\le, √ -> \\sqrt, U+2212 "
        "MINUS SIGN -> '-'). The text layer of a TeX-produced PDF emits the "
        "glyph, and every document in the formula_dense category does.",
    ),
    NormalizationRule(
        "font_glyph_names",
        "A leaked font glyph name becomes the command it stands for "
        "(/greaterorequalslant -> \\ge). Some encodings have no ToUnicode "
        "entry and the extractor emits the name itself; "
        "eval/corpus/formula_dense/green_tao_primes.pdf contains "
        "'/greaterorequalslant' in running text.",
    ),
    NormalizationRule(
        "upright_wrappers",
        "\\mathrm{X}, \\operatorname{X}, \\text{X}, \\mathit{X} and \\mbox{X} "
        "reduce to X, with the space inside a \\text group preserved. Upright "
        "versus italic is invisible in a text layer, so requiring it would "
        "reject a formula for a difference no reader can see.",
    ),
    NormalizationRule(
        "size_and_delimiters",
        "\\left, \\right, \\big and its relatives, \\displaystyle, \\textstyle "
        "and \\limits are removed. They choose a glyph size and a delimiter "
        "shape; the expression they enclose is unchanged.",
    ),
    NormalizationRule(
        "fraction_variants",
        "\\dfrac, \\tfrac and \\cfrac become \\frac. They differ only in the "
        "size the fraction is set at, and a display formula reproduced inline "
        "comes back as \\dfrac half the time.",
    ),
    NormalizationRule(
        "single_token_arguments",
        "A macro given a bare single-token argument gets braces around it, so "
        "\\frac12 and \\frac{1}{2} are the same expression. A text layer "
        "routinely loses the braces on one- and two-character arguments.",
    ),
    NormalizationRule(
        "spacing",
        "Spacing commands (\\, \\; \\: \\! \\quad \\hspace{..} ~) and all "
        "inter-token whitespace are removed. LaTeX ignores spaces in math "
        "mode, so a space is a typesetting artefact rather than a difference "
        "in what was written.",
    ),
    NormalizationRule(
        "equation_number",
        "A trailing \\tag{..}, \\label{..}, '(3.1)' or '[7]' is removed. A "
        "numbered display equation is the same expression with a number beside "
        "it, and the number is page furniture: it changes on the next page "
        "and is not part of the formula.",
    ),
)

#: Characters that mean "not a formula" for the purposes of a verdict.
_NO_FORMULA = ("", "()", "[]", "\\(", "\\)", "\\[", "\\]", "$", "$$")

#: Unicode ligatures a TeX text layer emits. Expansion happens before the
#: maths table so that 'ﬁ' becomes 'fi' rather than a maths lookup failing.
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ſt",
    "ﬆ": "st",
}

#: Unicode maths -> LaTeX. Every entry is a glyph a formula-bearing PDF
#: actually puts on the page; anything not listed is left alone, because
#: guessing at an unlisted codepoint is how two different symbols start
#: comparing equal.
_UNICODE_MATHS = {
    # Greek lower case
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ϵ": r"\epsilon", "ζ": r"\zeta", "η": r"\eta",
    "θ": r"\theta", "ϑ": r"\vartheta", "ι": r"\iota", "κ": r"\kappa",
    "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi",
    "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "ς": r"\varsigma",
    "τ": r"\tau", "υ": r"\upsilon", "ϕ": r"\phi", "φ": r"\varphi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    # Greek upper case (those distinct from Latin lookalikes)
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    # Relations and operators
    "≤": r"\le", "⩽": r"\le", "≥": r"\ge", "⩾": r"\ge",
    "≠": r"\neq", "≈": r"\approx", "≡": r"\equiv", "≃": r"\simeq",
    "≅": r"\cong", "∼": r"\sim", "≁": r"\nsim", "∝": r"\propto",
    "≪": r"\ll", "≫": r"\gg", "∈": r"\in", "∉": r"\notin",
    "⊂": r"\subset", "⊆": r"\subseteq", "⊃": r"\supset",
    "⊇": r"\supseteq", "∪": r"\cup", "∩": r"\cap",
    "∧": r"\wedge", "∨": r"\vee", "¬": r"\neg",
    "→": r"\to", "←": r"\gets", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "×": r"\times", "÷": r"\div", "⨯": r"\times",
    "±": r"\pm", "∓": r"\mp", "⋅": r"\cdot", "·": r"\cdot",
    "∘": r"\circ", "∗": r"\ast", "⊗": r"\otimes", "⊕": r"\oplus",
    # Large operators and radicals
    "∫": r"\int", "∬": r"\iint", "∮": r"\oint",
    "∑": r"\sum", "∏": r"\prod", "∐": r"\coprod",
    "⋃": r"\bigcup", "⋂": r"\bigcap",
    "√": r"\sqrt", "∛": r"\sqrt[3]", "∜": r"\sqrt[4]",
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla",
    "∀": r"\forall", "∃": r"\exists", "∄": r"\nexists",
    "ℏ": r"\hbar", "ℓ": r"\ell", "ℜ": r"\Re", "ℑ": r"\Im",
    "ℝ": r"\mathbb{R}", "ℂ": r"\mathbb{C}",
    "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}",
    # Punctuation a typesetter uses inside a formula
    "′": "'", "″": "''", "°": r"^{\circ}",
    "−": "-",  # MINUS SIGN, not a hyphen: the same stroke, a different codepoint
    "‐": "-", "‑": "-",  # HYPHEN and NON-BREAKING HYPHEN
}

#: Font glyph names an extractor emits verbatim when the font has no ToUnicode
#: entry. Observed in eval/corpus/formula_dense/green_tao_primes.pdf.
_GLYPH_NAMES = {
    "/equal": "=",
    "/greaterorequalslant": r"\ge",
    "/lessorequalslant": r"\le",
    "/notequal": r"\neq",
    "/approxequal": r"\approx",
    "/similar": r"\sim",
    "/arrowright": r"\to",
    "/arrowleft": r"\gets",
    "/arrowboth": r"\leftrightarrow",
    "/partialdiff": r"\partial",
    "/summation": r"\sum",
    "/product": r"\prod",
    "/integral": r"\int",
    "/radical": r"\sqrt",
    "/bullet": r"\cdot",
    "/multiply": r"\times",
    "/divide": r"\div",
    "/plusminus": r"\pm",
    "/element": r"\in",
    "/notelement": r"\notin",
    "/infinity": r"\infty",
}

#: Wrappers whose contents are text rather than maths, so a space inside one
#: is content. Recorded by ``rule_id`` so a report can name what was folded.
_UPRIGHT_WRAPPERS = ("text", "mathrm", "operatorname", "mathit", "mbox", "textbf")

#: Wrappers that are purely typographic. ``mathbb``/``mathbf``/``mathcal`` are
#: NOT here: they change how the symbol is drawn, and a reader sees that.
_SIZE_ONLY = (
    "left", "right", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
    "Biggl", "Biggr", "displaystyle", "textstyle", "scriptstyle",
    "scriptscriptstyle", "limits", "nolimits", "bigl.", "bigr.",
)

_FRACTION_VARIANTS = ("dfrac", "tfrac", "cfrac", "dbinom", "tbinom", "binom")

#: Macros whose single-token argument is routinely written bare. ``frac`` is
#: here as well as its variants: ``\frac12`` is the shape a text layer
#: produces when it loses the braces, and ``\dfrac12`` is what the same
#: extraction produces when the formula was set as a display equation.
_BARE_ARGUMENT_MACROS = _FRACTION_VARIANTS + (
    "frac", "sqrt", "hat", "bar", "vec", "tilde", "dot", "ddot",
    "overline", "underline", "widetilde", "widehat", "overrightarrow",
)

_SPACING_COMMAND = re.compile(
    r"\\(?:,|;|:|!|quad|qquad|thinspace|medspace|thickspace|enspace|hspace)"
    r"\s*\{[^}]*\}?|\\ |~"
)
_SIZE_COMMAND = re.compile(r"\\(" + "|".join(_SIZE_ONLY) + r")\b")
_FRACTION_COMMAND = re.compile(r"\\(" + "|".join(_FRACTION_VARIANTS) + r")(?![a-zA-Z])")
_EQUATION_TAG = re.compile(r"\\(?:tag|label)\s*\{[^}]*\}")
_TRAILING_NUMBER = re.compile(r"\s*[\(\[]\s*[A-Za-z]?\d+(?:\.\d+)*\s*[\)\]]\s*$")
#: A full stop set off from the last token by a space, with nothing after it.
#: LaTeX writes the stop tight against the expression ('\alpha'. ); a text
#: layer puts a space in front of it, and the space is a justification
#: artefact. Only the spaced form is stripped, so an expression that genuinely
#: ends in a full stop is left alone.
_TYPESETTING_STOP = re.compile(r"\s+\.\s*$")
#: A bare single-character argument to a macro that takes one: \frac12, \sqrt2.
#:
#: The alternation is ordered longest-first where it matters, because ``\frac``
#: also *ends* the longer names: without ``dfrac`` ahead of ``frac`` the
#: pattern matches the tail of ``\dfrac12`` and leaves a stray ``d`` in the
#: canonical form.
_BARE_ARGUMENT = re.compile(
    r"\\(" + "|".join(sorted(_BARE_ARGUMENT_MACROS, key=len, reverse=True))
    + r")(?=[A-Za-z0-9])([A-Za-z0-9])"
)
#: A superscript or subscript whose argument is one character inside braces.
#:
#: The counterpart to :data:`_BARE_ARGUMENT`, and it exists so that the pair is
#: *symmetric*. A model writes ``x^{2}``; a text layer gives back ``x^2``. Adding
#: braces to the bare form without removing them from the braced one would make
#: the canonical form depend on which side of the comparison came from where.
_SUPERSCRIPT_BRACES = re.compile(r"([\^_])\{([A-Za-z0-9])\}")
#: The denominator of a fraction written as a bare token: ``\frac{1}2``.
#:
#: TeX reads this as ``\frac{1}{2}`` -- a macro takes its two arguments whether
#: or not they are braced -- so a comparison that treats the braced and unbraced
#: spellings as different expressions is comparing a *typesetting* choice. It
#: exists because ``\tfrac12`` normalises to ``\frac12`` and one pass of
#: :data:`_BARE_ARGUMENT` only braces the numerator, leaving exactly the
#: ``\frac{1}2`` that no source wrote on purpose.
_FRACTION_DENOMINATOR = re.compile(
    r"\\(" + "|".join(sorted(_BARE_ARGUMENT_MACROS, key=len, reverse=True))
    + r")\{([A-Za-z0-9])\}(?=[A-Za-z0-9])([A-Za-z0-9])"
)
_WHITESPACE = re.compile(r"\s+")
_ENVIRONMENT = re.compile(r"\\begin\s*\{([^}]*)\}")
_ENVIRONMENT_END = re.compile(r"\\end\s*\{([^}]*)\}")

#: Sentinel for a protected group while the rest of the string is normalised.
#: NUL cannot appear in a formula that came out of a text layer, and if one
#: does the comparison is wrong rather than the normaliser crashing.
_SENTINEL = "\x00"

_MATH_DELIMITER = "$"


class LatexVerdict(str, Enum):
    """The answer to "are these the same formula".

    Written out rather than collapsed into a boolean because the failures are
    not interchangeable, and a caller that must decide whether to *render* a
    formula as verified needs to tell them apart. ``DIFFERENT`` licenses
    saying the model invented it. ``UNCOMPARABLE`` does not: it licenses
    saying nothing was established, which is what a formula the parser
    recovered as an image deserves.
    """

    #: The two strings were already the same.
    IDENTICAL = "identical"
    #: Different strings, equal once the named tolerances in
    #: :data:`LATEX_EQUIVALENCE_RULES` are applied.
    EQUIVALENT = "equivalent"
    #: Different expressions. This is the only verdict that may be reported as
    #: a fabricated formula.
    DIFFERENT = "different"
    #: At least one side is not a formula — an image with no LaTeX, a
    #: placeholder, an empty string. No claim either way.
    UNCOMPARABLE = "uncomparable"


#: Verdicts under which the two formulas may be presented as one.
MATCHING_VERDICTS = (LatexVerdict.IDENTICAL, LatexVerdict.EQUIVALENT)


@dataclass(frozen=True, slots=True)
class CanonicalLatex:
    """One formula, normalised, plus what it took to get there.

    ``applied_rules`` is the audit trail. A gate that widened without anyone
    deciding to widen it is the failure this project is built against, so the
    rules that fired travel with the answer and a caller can log them.
    """

    text: str
    applied_rules: tuple[str, ...] = ()
    defects: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether normalisation left nothing that could be a formula."""
        return self.text.strip() in _NO_FORMULA

    def __bool__(self) -> bool:
        return not self.is_empty


@dataclass(frozen=True, slots=True)
class LatexComparison:
    """A verdict on two formulas, with the evidence for it."""

    verdict: LatexVerdict
    left: CanonicalLatex
    right: CanonicalLatex
    reason: str

    @property
    def matches(self) -> bool:
        """Whether the two may be presented as the same formula.

        ``UNCOMPARABLE`` is ``False`` here, and that is the safe direction:
        an unestablished match may never be *rendered* as a verified one, or a
        parser that cannot read a formula would be indistinguishable from a
        model that invented it.
        """
        return self.verdict in MATCHING_VERDICTS


# --------------------------------------------------------------------------
# Defects
# --------------------------------------------------------------------------


def _unescaped_brace_balance(text: str) -> int:
    """Brace depth ignoring ``\\{`` and ``\\}``."""
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    return depth


def _open_environments(text: str) -> list[str]:
    """Environments opened and never closed, innermost first."""
    opened: list[str] = []
    for match in _ENVIRONMENT.finditer(text):
        opened.append(match.group(1).strip())
    for match in _ENVIRONMENT_END.finditer(text):
        name = match.group(1).strip()
        if name in opened:
            opened.remove(name)
    return opened


def latex_defects(latex: str) -> tuple[str, ...]:
    """What is wrong with this LaTeX, named rather than guessed at.

    A defect is a *report*, never a refusal. A formula with an unclosed
    environment still renders, and dropping it would be a corruption the
    reader cannot see: the export would simply be missing an equation the
    paper has. What changes is that the renderer says so next to it, so the
    reader knows not to trust the typesetting of that one line.

    Never raises. A caller in a render path must be able to ask "is this
    sound?" about a string that arrived from a model, and the answer to a
    malformed question is a malformed string, not an exception.
    """
    text = str(latex or "")
    found: list[str] = []
    balance = _unescaped_brace_balance(text)
    if balance > 0:
        found.append("unbalanced_braces")
    elif balance < 0:
        found.append("stray_closing_brace")
    unclosed = _open_environments(text)
    if unclosed:
        found.append("unclosed_environment")
    stripped = text.rstrip()
    if stripped.endswith("\\") or re.search(r"\\[a-zA-Z]+$", stripped):
        found.append("dangling_command")
    if canonical_latex(text).is_empty and text.strip():
        found.append("no_latex_content")
    return tuple(found)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def _strip_outer_delimiters(text: str) -> str:
    stripped = text.strip()
    for opening, closing in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if len(stripped) > len(opening) + len(closing) - 1 and (
            stripped.startswith(opening) and stripped.endswith(closing)
        ):
            return stripped[len(opening) : len(stripped) - len(closing)]
    return stripped


def _protect_upright_groups(text: str) -> tuple[str, list[str]]:
    """Pull ``\\text{...}``-style groups out before whitespace is collapsed.

    A space inside ``\\text{a b}`` is content — it is the difference between
    two words and one. Collapsing it would make a formula compare equal to one
    that says something else, which is exactly the false-equal the tolerance
    list is supposed to rule out. So the groups are lifted out, the
    placeholder they leave behind contains no whitespace, and they go back in
    at the end.
    """
    saved: list[str] = []
    for wrapper in _UPRIGHT_WRAPPERS:
        needle = f"\\{wrapper}{{"
        while True:
            start = text.find(needle)
            if start < 0:
                break
            body_start = start + len(needle)
            depth = 1
            index = body_start
            while index < len(text) and depth:
                char = text[index]
                if char == "\\" and index + 1 < len(text):
                    index += 2
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                index += 1
            body = text[body_start : index - 1] if depth == 0 else text[body_start:index]
            placeholder = f"{_SENTINEL}{len(saved)}{_SENTINEL}"
            saved.append(" ".join(body.split()))
            text = text[:start] + placeholder + text[index:]
    return text, saved


def _restore_upright_groups(text: str, saved: list[str]) -> str:
    for index, body in enumerate(saved):
        text = text.replace(f"{_SENTINEL}{index}{_SENTINEL}", body)
    return text


def canonical_latex(latex: str) -> CanonicalLatex:
    """Normalise one formula, recording which tolerances fired.

    Total by construction: every step is a substitution over a string, and a
    step that cannot apply is skipped rather than raised on. The one thing
    that can produce an empty result is a formula that was never a formula —
    an image, a placeholder, a caption — and that comes back as
    :attr:`CanonicalLatex.is_empty` so the caller can say "I cannot tell"
    instead of "different".
    """
    original = str(latex or "")
    text = original
    applied: list[str] = []

    stripped = _strip_outer_delimiters(text)
    if stripped != text:
        applied.append("outer_delimiters")
    text = stripped

    expanded = "".join(_LIGATURES.get(char, char) for char in text)
    if expanded != text:
        applied.append("ligature_forms")
    text = expanded

    translated = "".join(_UNICODE_MATHS.get(char, char) for char in text)
    if translated != text:
        applied.append("unicode_maths")
    text = translated

    def _replace_glyph_names(match: re.Match[str]) -> str:
        # ``.get`` with a fallback, not a subscript. The pattern is deliberately
        # broad -- it has to catch a name nobody has seen yet -- and a real
        # text layer leaks internal PDF names alongside the maths ones:
        # dbrane_theory_review.pdf contains "/CFT" in its running text. Raising
        # on the first unknown name would make a whole document unparseable
        # because of one stray token, and the correct response to a name with no
        # translation is to leave it alone and let it fail to typeset.
        return _GLYPH_NAMES.get(match.group(0), match.group(0))

    glyphed = re.sub(r"/[A-Za-z]{3,}", _replace_glyph_names, text)
    if glyphed != text:
        applied.append("font_glyph_names")
    text = glyphed

    text, saved = _protect_upright_groups(text)
    if saved:
        applied.append("upright_wrappers")

    sized = _SIZE_COMMAND.sub("", text)
    if sized != text:
        applied.append("size_and_delimiters")
    text = sized

    fracs = _FRACTION_COMMAND.sub(lambda match: "\\frac", text)
    if fracs != text:
        applied.append("fraction_variants")
    text = fracs

    # A single-token argument may or may not have arrived braced. Bracing it
    # is the symmetric choice: both sides get the same treatment, so it can
    # never make two genuinely different fractions compare equal.
    #
    # Two passes, not one. ``\frac12`` is *two* bare arguments, so bracing the
    # first leaves ``\frac{1}2`` -- still different from ``\frac{1}{2}``, which
    # is the exact case this rule exists to handle. And an argument that was
    # already braced must lose its braces again, or the canonical form depends
    # on which side of the comparison the bracing came from.
    for _ in range(3):
        braced = _BARE_ARGUMENT.sub(lambda match: f"\\{match.group(1)}{{{match.group(2)}}}", text)
        braced = _FRACTION_DENOMINATOR.sub(
            lambda match: f"\\{match.group(1)}{{{match.group(2)}}}{{{match.group(3)}}}", braced
        )
        braced = _SUPERSCRIPT_BRACES.sub(r"\1\2", braced)
        if braced == text:
            break
        text = braced
    if text != fracs:
        applied.append("single_token_arguments")

    untagged = _EQUATION_TAG.sub("", text)
    untagged = _TRAILING_NUMBER.sub("", untagged)
    untagged = _TYPESETTING_STOP.sub("", untagged)
    if untagged != text:
        applied.append("equation_number")
    text = untagged

    spaced = _SPACING_COMMAND.sub("", text)
    spaced = _WHITESPACE.sub("", spaced)
    if spaced != untagged:
        applied.append("spacing")
    text = spaced

    text = _restore_upright_groups(text, saved)

    return CanonicalLatex(text=text.strip(), applied_rules=tuple(applied))


def compare_latex(left: str, right: str) -> LatexComparison:
    """Decide whether two formulas are the same expression.

    The order of the checks is the design. Emptiness comes first, because a
    comparison between a formula and a non-formula has no answer and must not
    be allowed to reach the string comparison, where an empty canonical form
    on one side and a non-empty one on the other would read as
    ``DIFFERENT``. Only after both sides are known to be formulas is the
    question "same expression?" a real one, and only then can the answer
    honestly be ``DIFFERENT``.
    """
    left_canonical = canonical_latex(left)
    right_canonical = canonical_latex(right)

    if left_canonical.is_empty or right_canonical.is_empty:
        # Name the side that is not a formula. "One of these is an image" is
        # actionable; "they differ" is not, when the truth is that there is
        # nothing on one side to compare.
        empty_side = "left" if left_canonical.is_empty else "right"
        return LatexComparison(
            verdict=LatexVerdict.UNCOMPARABLE,
            left=left_canonical,
            right=right_canonical,
            reason=(
                f"{'公式左' if empty_side == 'left' else '公式右'}侧没有可比较的 LaTeX"
                "（可能是解析为图片的公式或占位符），无法判定是否同一个公式。"
            ),
        )

    if left_canonical.text == right_canonical.text:
        # IDENTICAL means the two arrived the same; EQUIVALENT means they
        # arrived different and a named tolerance closed the gap. The reader
        # is entitled to know which, because a gate that silently widens is
        # the thing this project is built against.
        rules = tuple(
            dict.fromkeys(left_canonical.applied_rules + right_canonical.applied_rules)
        )
        return LatexComparison(
            verdict=LatexVerdict.EQUIVALENT if rules else LatexVerdict.IDENTICAL,
            left=left_canonical,
            right=right_canonical,
            reason=(
                "归一化后一致（已应用：" + "、".join(rules) + "）。"
                if rules
                else "两个公式本来就完全一致。"
            ),
        )

    return LatexComparison(
        verdict=LatexVerdict.DIFFERENT,
        left=left_canonical,
        right=right_canonical,
        reason="归一化后仍是两个不同的表达式。",
    )


def formulas_match(left: str, right: str) -> bool:
    """Whether two formulas may be presented as one.

    The narrow form of :func:`compare_latex` for the caller that only needs
    the decision. ``UNCOMPARABLE`` is ``False``, because a match that was
    never established may not be rendered as one.
    """
    return compare_latex(left, right).matches


# --------------------------------------------------------------------------
# Formulas embedded in prose
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MathSpan:
    """One stretch of maths inside a piece of text.

    ``delimiters`` is what was actually there, so a renderer can reproduce it.
    ``defect`` is set when the span could not be closed, because the reader
    still has to be shown the text the model wrote — a truncated ``$`` is
    ambiguous with a literal dollar sign, and discarding the span would
    silently remove a formula from an analysis.
    """

    latex: str
    display: bool
    start: int
    end: int
    delimiters: str
    defect: str = ""


#: A ``$`` that starts a word and is followed by a digit is a currency amount.
#: "$5" is not a formula, and treating it as the opening delimiter makes every
#: span after it in the paragraph mis-delimited -- which is the whole run of
#: text after the false opener, mis-read as maths. This is the same escape
#: convention every Markdown maths extension uses, and a formula that genuinely
#: starts with a digit is always written without a space in front of it.
_CURRENCY_DOLLAR = re.compile(r"(?:\A|(?<=[\s(]))\$(?=\d)")


def _dollar_tokens(source: str) -> list[tuple[int, str, str]]:
    """Every ``$`` run in ``source``, as ``(offset, delimiter, defect)``.

    Runs rather than characters, and that is the whole trick. A scanner that
    steps one ``$`` at a time cannot tell the first half of a ``$$…$$`` block
    from an opening ``$``, so a display formula after a currency amount is
    split down the middle and both halves are rendered as prose. Runs cannot
    be split, so a ``$$`` is either a delimiter or it is not.

    A run longer than the delimiter it is read as carries a defect rather than
    being trimmed: ``$$$x$$$`` has a ``$`` in it the model put somewhere, and
    deciding silently which one to drop is how a formula loses a character.
    The defect travels with the span and the renderer says so.
    """
    tokens: list[tuple[int, str, str]] = []
    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        if char == "\\" and index + 1 < length:
            index += 2
            continue
        if char != _MATH_DELIMITER:
            index += 1
            continue
        run_end = index
        while run_end < length and source[run_end] == _MATH_DELIMITER:
            run_end += 1
        run = source[index:run_end]
        if _CURRENCY_DOLLAR.match(source, index) and run == _MATH_DELIMITER:
            index = run_end
            continue
        delimiter = "$$" if run.startswith("$$") else "$"
        defect = "ambiguous_delimiters" if len(run) > len(delimiter) else ""
        tokens.append((index, delimiter, defect))
        index = run_end
    return tokens


def find_inline_math(text: str) -> tuple[MathSpan, ...]:
    """Every ``$…$`` / ``$$…$$`` stretch in a piece of prose, in order.

    Used on model output, where a formula may or may not be present and may
    or may not be balanced. Every stretch is returned, including an unclosed
    one: the caller's job is to mark it, not to drop it. An unclosed ``$`` is
    indistinguishable from a literal dollar sign, and discarding the span
    would silently remove a formula from an analysis -- a corruption the
    reader cannot see, which is the one outcome this whole module is arranged
    to rule out.

    Three escapes are honoured, each because the alternative corrupts
    something real. ``\\$`` is a literal dollar sign. A ``$`` at the start of a
    word followed by a digit is currency, not maths -- and this is not a
    cosmetic rule for a report's cost column: reading ``$5`` as an opening
    delimiter makes every formula after it in the paragraph part of the same
    span, so a $5 figure and a display equation are handed to the comparator
    as one string and neither can match the paper. A run longer than its own
    delimiter (``$$$``) carries a defect rather than being trimmed to length,
    because choosing which of the three dollars to keep is a guess, and a
    guess that silently changes a formula is the one failure this module
    exists to prevent.
    """
    source = str(text or "")
    tokens = _dollar_tokens(source)
    spans: list[MathSpan] = []
    position = 0
    while position < len(tokens):
        start, opening, opening_defect = tokens[position]
        if position + 1 >= len(tokens):
            # An opener with nothing after it. Either the model left the
            # delimiter unclosed or it was a literal dollar. The two are
            # indistinguishable from the text alone, and picking either one
            # silently is how a formula loses its closing delimiter, so the
            # span is returned with the defect attached and the renderer
            # decides what to say about it.
            spans.append(
                MathSpan(
                    latex=source[start + len(opening) :],
                    display=opening == "$$",
                    start=start,
                    end=len(source),
                    delimiters=opening,
                    defect=opening_defect or "unbalanced_delimiters",
                )
            )
            break
        close_start, closing, closing_defect = tokens[position + 1]
        if opening != closing:
            # A ``$`` cannot close a ``$$`` and the reverse is equally true.
            # The opener was not a delimiter after all -- a model writing
            # ``$x$$y$$`` means three separate things, not one formula.
            #
            # Resynchronise on the *closer* and re-examine it as a potential
            # opener. Stepping over both would consume the closer as though
            # it had been an opener, and it would then pair with the next real
            # formula in the document, reporting a span that runs from inside
            # one formula to inside the next. That is the specific corruption
            # this walk exists to prevent.
            spans.append(
                MathSpan(
                    latex=source[start + len(opening) : close_start],
                    display=opening == "$$",
                    start=start,
                    end=close_start,
                    delimiters=opening,
                    defect=opening_defect or "mismatched_delimiters",
                )
            )
            position += 1
            continue
        spans.append(
            MathSpan(
                latex=source[start + len(opening) : close_start],
                display=opening == "$$",
                start=start,
                end=close_start + len(closing),
                delimiters=opening,
                defect=opening_defect or closing_defect,
            )
        )
        position += 2
    return tuple(spans)


# --------------------------------------------------------------------------
# A formula the parser split in two
# --------------------------------------------------------------------------


def join_latex(left: str, right: str) -> str | None:
    """Rejoin a formula the parser split across two elements, or ``None``.

    The test is *structural*, not semantic: the left half ends inside an
    unclosed environment or inside an unclosed brace group, so the right half
    is a continuation rather than a new formula. A PDF that lays out a
    display equation over two lines produces exactly this, and a formula cut
    in half renders as two LaTeX errors instead of one equation.

    ``None`` means "these are two formulas", and the caller must keep both.
    Returning ``None`` rather than joining on a hunch is what keeps a real
    pair of consecutive equations from being silently fused into one
    unreadable string.
    """
    head = str(left or "").rstrip()
    tail = str(right or "").lstrip()
    if not head or not tail:
        return None
    # Joined with a space rather than glued. TeX ignores spaces in math mode,
    # so the space costs nothing -- and it is the only thing standing between a
    # head ending in ``\\`` and a tail starting with a letter, which glued would
    # read as the single control word ``\\c`` rather than a line break. The
    # control sequence silently changing meaning is the one failure here that
    # would not show up as a visible error.
    if _unescaped_brace_balance(head) > 0 or _open_environments(head):
        return f"{head} {tail}"
    if head.endswith("\\") or re.search(r"\\[a-zA-Z]+$", head):
        return f"{head} {tail}"
    return None
