"""Front-end evidence and parse-status invariants, asserted statically.

Task T1.9 and T3.4 have no browser in this repository, and adding one is out of
scope, so nothing here renders anything. Every assertion is a structural
property of the shipped files: that the evidence panel exists and is addressed
at the real Evidence API, that a degraded parse is a *visible element* rather
than an absence, and — the part that most needs a test — that a paraphrase
cannot acquire quotation marks no matter what the stylesheet does.

That last property is why this file reads the JavaScript's structure rather than
just grepping for strings. A test asserting "the word `paraphrase` appears near
the word `direct_quote`" passes happily when both are emitted from the same
branch, which is exactly the bug the release gate exists to prevent. So the
release-gate assertions here are written against the *reachability* of the mark
constants: a quotation mark is a constant in ``config.js``, the direct-quote
renderer is the only thing that may name it, and the paraphrase renderer is
required to name neither it nor the function that builds one. A paraphrase that
renders inside quotation marks would then need a new export in ``claims.js``,
which is a diff someone reviews — rather than a one-line edit to a rule in a
stylesheet that nobody diffs for meaning.

The tests read files, so they cannot drift from what ships the way a mock
contract can. What they cannot do is prove the panel *looks* right; that remains
a manual check, and the one gap is called out at the bottom of this module.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JS_DIR = PROJECT_ROOT / "templates" / "static" / "js"
TEMPLATE = PROJECT_ROOT / "templates" / "index.html"

CLAIMS = (JS_DIR / "claims.js").read_text(encoding="utf-8")
EVIDENCE = (JS_DIR / "evidence.js").read_text(encoding="utf-8")
PARSE_STATUS = (JS_DIR / "parseStatus.js").read_text(encoding="utf-8")
CONFIG = (JS_DIR / "config.js").read_text(encoding="utf-8")
CHAT = (JS_DIR / "chat.js").read_text(encoding="utf-8")
SECTIONS = (JS_DIR / "sections.js").read_text(encoding="utf-8")
MAIN = (JS_DIR / "main.js").read_text(encoding="utf-8")
INDEX_HTML = TEMPLATE.read_text(encoding="utf-8")

#: Every module the evidence work added or touched, for the no-framework check.
NEW_MODULES = ("claims.js", "evidence.js", "parseStatus.js")

#: Quotation mark characters. The gate is about *these*; a straight apostrophe
#: is a JavaScript string delimiter and is not one of them.
QUOTE_MARKS = '“”‘’「」『』《》〝〞＂"'


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def strip_js_comments(source: str) -> str:
    """Remove ``//`` and ``/* */`` comments, leaving string literals intact.

    Needed so a comment that *discusses* quotation marks cannot satisfy or
    violate a structural assertion about them. The scanner tracks string state,
    so a ``//`` inside a string is not mistaken for a comment.
    """
    out: list[str] = []
    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        nxt = source[index + 1] if index + 1 < length else ""
        if char == "/" and nxt == "/":
            while index < length and source[index] != "\n":
                index += 1
        elif char == "/" and nxt == "*":
            index += 2
            while index < length and not (source[index] == "*" and source[index + 1 : index + 2] == "/"):
                index += 1
            index += 2
        elif char in "\"'`":
            quote = char
            out.append(char)
            index += 1
            while index < length:
                out.append(source[index])
                if source[index] == "\\":
                    index += 1
                    if index < length:
                        out.append(source[index])
                        index += 1
                    continue
                if source[index] == quote:
                    index += 1
                    break
                index += 1
        else:
            out.append(char)
            index += 1
    return "".join(out)


def strip_js_imports(source: str) -> str:
    """Remove static ``import`` statements from the top of a module.

    Needed where an assertion is about what a module's *code* references: an
    import of a constant is not a use of it, and treating one as the other
    would flag every module that legitimately imports a shared name.
    """
    return re.sub(r"^\s*import\s+[^;]*?from\s+'[^']+';", "", source, flags=re.MULTILINE)


def function_body(source: str, name: str) -> str:
    """Return the body of ``function name(...)`` by brace matching.

    Brace counting is done with string state tracked, so a ``{`` inside a
    template literal or a string does not close the function early — the
    failure mode that turns a structural assertion into a flaky one.

    The opening brace is the first one *outside* the parameter list, so a
    destructured parameter (``buildHeader(view, { badge })``) is not mistaken
    for the body. The parameter list is skipped by counting parentheses, which
    also survives a default value containing one.
    """
    match = re.search(rf"function\s+{re.escape(name)}\s*\(", source)
    assert match, f"function {name}() is missing"

    index = match.end()
    depth = 1
    while index < len(source) and depth > 0:
        char = source[index]
        if char in "\"'`":
            quote = char
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    break
                index += 1
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        index += 1
    index = source.index("{", index)
    depth = 0
    start = index
    while index < len(source):
        char = source[index]
        if char in "\"'`":
            quote = char
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    break
                index += 1
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
        index += 1
    raise AssertionError(f"unbalanced braces in {name}()")


def element_by_id(tag: str, element_id: str) -> str | None:
    """Return the source of the tag carrying ``id="element_id"``, if present."""
    for match in re.finditer(rf"<{tag}\b[^>]*>", INDEX_HTML):
        if re.search(rf'\bid="{re.escape(element_id)}"', match.group(0)):
            return match.group(0)
    return None


# --------------------------------------------------------------------------
# The panel exists, and it is wired to the real Evidence API
# --------------------------------------------------------------------------


def test_evidence_panel_is_present_in_the_template_as_a_dialog():
    panel = element_by_id("aside", "evidencePanel")
    assert panel is not None, "the evidence panel is missing from index.html"
    assert 'role="dialog"' in panel
    assert 'aria-labelledby="evidencePanelHeading"' in panel
    assert 'aria-hidden="true"' in panel, "a closed panel must start hidden"
    assert "hidden" in panel


def test_evidence_panel_is_wired_to_the_evidence_api_path():
    assert "EVIDENCE_API_PATH" in CONFIG
    path_match = re.search(r"EVIDENCE_API_PATH\s*=\s*'([^']+)'", CONFIG)
    assert path_match, "EVIDENCE_API_PATH is not a literal path template"
    path = path_match.group(1)
    assert path.startswith("/v1/documents/"), path
    assert path.endswith("/evidence/{evidenceId}"), path
    assert "{documentId}" in path

    # The module must build that path rather than hard-coding a second one.
    assert "EVIDENCE_API_PATH" in EVIDENCE, "evidence.js ignores the shared path"
    assert "/v1/documents/" not in EVIDENCE, "evidence.js duplicates the endpoint path"
    # ...and both path segments are encoded rather than interpolated raw.
    url_body = function_body(EVIDENCE, "buildEvidenceUrl")
    assert url_body.count("encodeURIComponent") >= 2


def test_every_panel_state_is_a_distinct_element_rather_than_an_absence():
    """The degraded states must be visible nodes, not blank panels.

    Each state gets its own element id so a client can tell "the request
    failed" from "this citation cannot be pointed at on the page" from "the
    citation resolved but carries no text". Sharing a node would let one
    overwriting another teach the reader the wrong thing about their own paper.
    """
    mapping = re.search(
        r"PANEL_STATE_ELEMENTS\s*=\s*\{(.*?)\}", EVIDENCE, re.DOTALL
    )
    assert mapping, "the panel has no declared state-to-element mapping"
    entries = dict(re.findall(r"(\w+)\s*:\s*'([^']+)'", mapping.group(1)))
    assert set(entries) == {
        "idle",
        "loading",
        "error",
        "no_visual_targeting",
        "empty",
    }
    assert len(set(entries.values())) == len(entries), "two states share one element"
    for name, element_id in entries.items():
        assert element_by_id("div", element_id), (
            f"state {name!r} maps to #{element_id}, which is not in the template"
        )


def test_no_visual_targeting_still_shows_the_passage():
    """A locator that cannot be drawn is not a locator that has no text.

    The passage region is therefore shown in two states. If it were owned by
    the mutually-exclusive state set, a degraded citation would hide the one
    part of the response the reader can actually use.
    """
    assert re.search(
        r"PASSAGE_VISIBLE_STATES\s*=\s*new Set\(\[[^\]]*'no_visual_targeting'", EVIDENCE
    ), "the passage is hidden when visual targeting is unavailable"
    passage_id = re.search(r"PASSAGE_ELEMENT_ID\s*=\s*'([^']+)'", EVIDENCE)
    assert passage_id
    assert element_by_id("div", passage_id.group(1)), "the passage region is not in the template"
    # And the degraded state says so in words.
    assert "still shown below" in INDEX_HTML


def test_a_failed_request_and_a_missing_target_render_different_copy():
    error_copy = INDEX_HTML[INDEX_HTML.index('id="evidenceError"') : INDEX_HTML.index('id="evidenceNoTarget"')]
    no_target_copy = INDEX_HTML[
        INDEX_HTML.index('id="evidenceNoTarget"') : INDEX_HTML.index('id="evidenceEmpty"')
    ]
    assert "could not be opened" in error_copy
    assert "No visual targeting is available" in no_target_copy
    assert "could not be opened" not in no_target_copy


# --------------------------------------------------------------------------
# Direct quote vs paraphrase — the release gate
# --------------------------------------------------------------------------


def test_direct_quote_and_paraphrase_use_different_elements():
    """The distinction is in the DOM, so it survives a stylesheet that fails."""
    quote_body = function_body(CLAIMS, "renderDirectQuoteClaim")
    paraphrase_body = function_body(CLAIMS, "renderParaphraseClaim")

    assert "buildClaimElement('blockquote'" in quote_body
    assert "buildClaimElement('p'" in paraphrase_body


def test_claim_elements_carry_the_presentation_in_data_attributes():
    builder = function_body(CLAIMS, "buildClaimElement")
    assert "data-presentation" in builder or "dataset.presentation" in builder
    assert "data-quote-marks" in builder or "dataset.quoteMarks" in builder
    # The paraphrase half must set the flag to the forbidding value, not merely
    # omit the allowed one.
    assert "'forbidden'" in builder
    assert "'allowed'" in builder


def test_quotation_marks_are_markup_not_a_css_generated_pseudo_element():
    assert "DIRECT_QUOTE_OPEN_MARK" in CONFIG
    assert "DIRECT_QUOTE_CLOSE_MARK" in CONFIG
    quote_body = function_body(CLAIMS, "renderDirectQuoteClaim")
    assert "buildMarkSpan(DIRECT_QUOTE_OPEN_MARK)" in quote_body
    assert "buildMarkSpan(DIRECT_QUOTE_CLOSE_MARK)" in quote_body

    # If the marks came from a ::before/::after rule, deleting one line of CSS
    # would silently downgrade every direct quote to bare text — and adding a
    # theme could put a mark around a paraphrase. Assert no such rule exists.
    style = INDEX_HTML[INDEX_HTML.index("<style>") : INDEX_HTML.index("</style>")]
    assert not re.search(r"\.claim-quote-mark::(before|after)", style)
    assert not re.search(r"\.claim[^\n{]*::(before|after)\s*\{[^}]*content\s*:\s*[\"']", style)


def test_the_paraphrase_path_cannot_emit_a_quotation_mark():
    """The release gate: "对不齐降级 paraphrase，不带引号".

    Structural, not textual. The mark is a constant; the paraphrase renderer is
    required to name neither it nor the helper that builds one. Anything else
    would be a new export in this module, which is a reviewed diff.
    """
    body = function_body(strip_js_comments(CLAIMS), "renderParaphraseClaim")
    assert "DIRECT_QUOTE_OPEN_MARK" not in body
    assert "DIRECT_QUOTE_CLOSE_MARK" not in body
    assert "DIRECT_QUOTE" not in body.replace("PRESENTATION_DIRECT_QUOTE", ""), (
        "the paraphrase renderer must not branch into the direct-quote path"
    )
    assert "buildMarkSpan" not in body
    assert "claim-quote-mark" not in body
    # No literal quotation mark is emitted either.
    assert not any(mark in body for mark in QUOTE_MARKS), (
        "a quotation mark character is emitted from the paraphrase path"
    )


def test_only_the_direct_quote_renderer_names_the_mark_constants():
    """One function may put a quotation mark on the page, and it is this one."""
    stripped = strip_js_comments(CLAIMS)
    for constant in ("DIRECT_QUOTE_OPEN_MARK", "DIRECT_QUOTE_CLOSE_MARK"):
        referencing = {
            name
            for name in re.findall(r"function\s+(\w+)\s*\(", stripped)
            if constant in function_body(stripped, name)
        }
        assert referencing == {"renderDirectQuoteClaim"}, (
            f"{constant} is reachable from {referencing}"
        )

    # The constants must be exported from config.js and reachable from no module
    # other than the one that owns the mark, so there is exactly one place in
    # the front end that can put a quotation mark on the page. The owner was
    # already checked function-by-function above; imports are excluded here
    # because importing a constant is not using it.
    for module_name in NEW_MODULES:
        if module_name in ("config.js", "claims.js"):
            continue
        source = strip_js_imports(strip_js_comments((JS_DIR / module_name).read_text(encoding="utf-8")))
        assert "DIRECT_QUOTE_OPEN_MARK" not in source
        assert "DIRECT_QUOTE_CLOSE_MARK" not in source

    # And no *other* module in the directory may name them either, so the gate
    # cannot be quietly reimplemented somewhere without a reviewer noticing.
    for path in sorted(JS_DIR.glob("*.js")):
        if path.name in ("config.js", "claims.js"):
            continue
        source = strip_js_imports(strip_js_comments(path.read_text(encoding="utf-8")))
        assert "DIRECT_QUOTE_OPEN_MARK" not in source, f"{path.name} can emit a quotation mark"
        assert "DIRECT_QUOTE_CLOSE_MARK" not in source, f"{path.name} can emit a quotation mark"


def test_a_paraphrase_has_its_quotes_stripped_before_rendering():
    body = strip_js_comments(CLAIMS)
    assert "function stripWrappingQuotes" in body
    assert "WRAPPING_QUOTE_PAIRS" in CONFIG
    # The renderer strips, and reports a survivor rather than shipping it.
    paraphrase = function_body(body, "renderParaphraseClaim")
    assert "stripWrappingQuotes(claimText(claim))" in paraphrase
    assert "hasWrappingQuotes" in paraphrase


def test_an_unrecognised_outcome_degrades_to_a_paraphrase():
    """Failing toward the unverifiable side, never toward the quoted side."""
    body = strip_js_comments(CLAIMS)
    resolver = function_body(body, "resolvePresentation")
    # The two quote-granting branches, and the single fall-through that is a
    # paraphrase. No other default exists.
    assert resolver.count("PRESENTATION_DIRECT_QUOTE") >= 2
    assert resolver.rstrip().endswith("}")
    returns = re.findall(r"return\s+(PRESENTATION_\w+);", resolver)
    assert returns[-1] == "PRESENTATION_PARAPHRASE", (
        "resolvePresentation must not fall through to a direct quote"
    )
    assert "PRESENTATION_PARAPHRASE" in CONFIG
    degraded = re.search(r"DEGRADED_ALIGNMENT_OUTCOMES\s*=\s*\[(.*?)\]", CONFIG, re.DOTALL)
    assert degraded and "not_found" in degraded.group(1)
    assert "spans_elements" in degraded.group(1)


# --------------------------------------------------------------------------
# Citation chips
# --------------------------------------------------------------------------


def test_citation_chips_are_keyboard_reachable_buttons():
    chips = function_body(EVIDENCE, "renderCitationChips")
    assert "createElement('button')" in chips
    assert "button.type = 'button'" in chips
    assert "aria-expanded" in chips
    assert "aria-haspopup" in chips
    assert "aria-label" in chips
    # Delegated, so chips re-rendered per answer do not leak listeners.
    assert "document.addEventListener('click', handleChipClick)" in EVIDENCE


def test_only_a_verified_claim_gets_a_citation_chip():
    """A paraphrase has no verified position, so it gets no chip.

    A chip is a promise that something can be pointed at. Beside an unverified
    sentence it reads as a citation, which is the impression the gate exists to
    prevent — the same reasoning that makes the backend's `RenderedClaim`
    refuse a locator on a paraphrase.
    """
    chips = function_body(EVIDENCE, "renderCitationChips")
    assert "claims.filter(claim => isDirectQuote(claim))" in chips
    body = strip_js_comments(CLAIMS)
    assert "function isDirectQuote" in body
    assert "PRESENTATION_DIRECT_QUOTE" in function_body(body, "isDirectQuote")


# --------------------------------------------------------------------------
# Opening evidence must not cost the reader their context
# --------------------------------------------------------------------------


def test_opening_the_panel_never_touches_the_question_input():
    """Comments are stripped: what matters is what the code *does*.

    An `evidence-open` reflow, a panel opening over the composer, and a chip
    click that clears the input are all ways to make the reader's question
    disappear at the moment they are trying to check a claim inside it.
    """
    for name in NEW_MODULES:
        source = strip_js_comments((JS_DIR / name).read_text(encoding="utf-8"))
        assert "questionInput" not in source, (
            f"{name} reaches into the question input; opening evidence must not "
            "consume or clear what the user asked"
        )


def test_the_scroll_anchor_is_captured_and_restored():
    assert "evidenceScrollAnchor" in (JS_DIR / "state.js").read_text(encoding="utf-8")
    open_body = function_body(EVIDENCE, "openEvidence")
    assert "state.evidenceScrollAnchor = window.scrollY" in open_body
    assert "restoreScrollAnchor()" in open_body
    close_body = function_body(EVIDENCE, "closeEvidence")
    assert "restoreScrollAnchor()" in close_body
    assert re.search(
        r"function restoreScrollAnchor[\s\S]*window\.scrollTo\(0, anchor\)", EVIDENCE
    )


def test_focus_returns_to_the_chip_that_opened_the_panel():
    assert "evidenceReturnFocusTo" in (JS_DIR / "state.js").read_text(encoding="utf-8")
    close_body = function_body(EVIDENCE, "closeEvidence")
    assert "state.evidenceReturnFocusTo" in close_body
    assert "focus({ preventScroll: true })" in close_body
    assert "document.contains(target)" in close_body
    # preventScroll matters for the same reason the anchor does.
    focus_body = function_body(EVIDENCE, "focusPanel")
    assert "preventScroll: true" in focus_body


def test_the_sheet_uses_no_body_scroll_lock():
    """A scroll-locked body jumps on close; an overlay does not.

    The common mobile-sheet technique pins ``body`` with ``position: fixed`` and
    restores it on close, which is a scroll jump wearing a disguise and a
    full-document reflow at the worst moment. The panel here is an overlay, so
    the page behind keeps its offset for free. The assertion is therefore about
    the *body* specifically — the panel and backdrop are meant to be fixed.
    """
    style = INDEX_HTML[INDEX_HTML.index("<style>") : INDEX_HTML.index("</style>")]

    body_rules = re.findall(r"(?:^|[},]\s*)body[^{]*\{([^}]*)\}", style)
    assert body_rules, "no body rules to check"
    for rule in body_rules:
        assert "position: fixed" not in rule, "the body is pinned; opening the panel will jump"
        assert "overflow: hidden" not in rule, "the body is scroll-locked; closing the panel will jump"
        assert "overflow: clip" not in rule

    # Nor from JavaScript, which is where such a lock is usually applied.
    for name in NEW_MODULES:
        source = strip_js_comments((JS_DIR / name).read_text(encoding="utf-8"))
        assert "document.body.style.overflow" not in source
        assert "document.body.style.position" not in source

    # The sheet contains its own scroll chain instead, so reading a long
    # passage does not scroll the conversation behind it.
    assert "overscroll-behavior: contain" in style
    assert "overflow-y: auto" in style


# --------------------------------------------------------------------------
# Desktop column and mobile bottom sheet
# --------------------------------------------------------------------------


def test_both_the_desktop_column_and_the_mobile_sheet_exist():
    style = INDEX_HTML[INDEX_HTML.index("<style>") : INDEX_HTML.index("</style>")]
    query = re.search(r"EVIDENCE_SHEET_MEDIA_QUERY\s*=\s*'([^']+)'", CONFIG)
    assert query, "no breakpoint for the sheet/column switch is declared"
    media = query.group(1)

    # The same query must appear in the stylesheet, or the JS would trap focus
    # in a column the reader can still read past (or vice versa).
    assert f"@media {media}" in style, f"{media} is not the stylesheet breakpoint"

    column = re.search(r"\.evidence-panel\s*\{([^}]*)\}", style)
    assert column, "no desktop panel rule"
    assert "right: 0" in column.group(1)
    assert "width: var(--evidence-panel-width)" in column.group(1)

    sheet = re.search(rf"@media {re.escape(media)}\s*\{{(.*?)\n        \}}\n", style, re.DOTALL)
    assert sheet, "no bottom-sheet rules"
    sheet_body = sheet.group(1)
    assert "bottom: 0" in sheet_body
    assert "width: 100%" in sheet_body
    assert "border-radius: var(--radius-lg) var(--radius-lg) 0 0" in sheet_body

    # Side by side, not overlaid: the container is padded rather than covered.
    assert re.search(r"body\.evidence-open \.container\s*\{[^}]*padding-right", style)


def test_the_sheet_is_modal_only_when_it_is_a_sheet():
    """As a column the conversation stays usable, so it must not be modal."""
    assert "function isSheetLayout" in EVIDENCE
    assert "EVIDENCE_SHEET_MEDIA_QUERY" in EVIDENCE
    open_shell = function_body(EVIDENCE, "openPanelShell")
    assert "isSheetLayout() ? 'sheet' : 'column'" in open_shell
    assert re.search(r"aria-modal.*isSheetLayout\(\)", open_shell)
    # The focus trap is scoped to the sheet for the same reason.
    trap = function_body(EVIDENCE, "trapFocus")
    assert "isSheetLayout()" in trap


def test_the_sheet_backdrop_only_dismisses_when_it_is_a_sheet():
    body = function_body(EVIDENCE, "initializeEvidencePanel")
    assert "isSheetLayout()" in body
    assert "evidenceBackdrop" in body
    assert element_by_id("div", "evidenceBackdrop")


# --------------------------------------------------------------------------
# Accessibility
# --------------------------------------------------------------------------


def test_the_panel_closes_on_escape_and_from_the_close_button():
    assert element_by_id("button", "evidenceCloseBtn")
    keydown = function_body(EVIDENCE, "handlePanelKeydown")
    assert "event.key === 'Escape'" in keydown
    assert "closeEvidence()" in keydown
    assert "document.addEventListener('keydown', handlePanelKeydown)" in EVIDENCE


def test_focus_moves_into_the_panel_when_it_opens():
    body = function_body(EVIDENCE, "focusPanel")
    assert "evidencePanelHeading" in body
    assert "focusPanel()" in function_body(EVIDENCE, "openEvidence")
    heading = element_by_id("h2", "evidencePanelHeading")
    assert heading and 'tabindex="-1"' in heading


def test_the_focus_trap_covers_the_elements_a_reader_can_reach():
    assert "FOCUSABLE_SELECTOR" in EVIDENCE
    selector = re.search(r"FOCUSABLE_SELECTOR\s*=\s*\[(.*?)\]\.join", EVIDENCE, re.DOTALL)
    assert selector
    for expected in ("button:not([disabled])", "a[href]", "[tabindex]:not([tabindex=\"-1\"])"):
        assert expected in selector.group(1)
    trap = function_body(EVIDENCE, "trapFocus")
    assert "event.key !== 'Tab'" in trap
    assert "event.shiftKey" in trap


def test_a_skip_link_reaches_the_evidence_aware_workspace():
    """Keyboard entry points must not have been broken by the new overlay."""
    assert 'href="#uploadWorkspace"' in INDEX_HTML
    assert 'href="#result"' in INDEX_HTML
    skip = INDEX_HTML[INDEX_HTML.index('<nav class="skip-links"') :]
    assert "</nav>" in skip


# --------------------------------------------------------------------------
# Parse status (T3.4)
# --------------------------------------------------------------------------


def test_the_parse_status_card_names_every_signal_t34_requires():
    assert element_by_id("section", "parseStatusCard")
    assert element_by_id("div", "parseStatus")
    status_card = re.search(
        r'<section class="result-card" id="parseStatusCard".*?</section>', INDEX_HTML, re.DOTALL
    )
    assert status_card, "the parse status card is missing"
    card = status_card.group(0)

    render = function_body(PARSE_STATUS, "renderParseStatus")
    # Which parser ran.
    assert "parsed_by" in render
    # The quality verdict, and what to say when it was not reported.
    assert "parse.quality" in render or "renderParseQuality" in render
    # The reader-facing sentences live in the catalogue, not in the module: the
    # card's strings are produced at render time and are resolved with `t()` so
    # a language switch repaints them. So these assertions read the catalogue
    # entry the module looks up, rather than a literal in the module source.
    catalogue = (JS_DIR / "i18n.js").read_text(encoding="utf-8")
    assert "'parse.quality.absent'" in catalogue
    assert "t('parse.quality.absent')" in function_body(PARSE_STATUS, "renderParseQuality")
    # A fallback that fired.
    assert "fallback_used" in render
    # Whether it left the machine — all three facts, kept apart.
    for field in ("used_cloud", "cloud_attempted", "touched_network"):
        assert field in render, f"{field} is not surfaced to the user"
    for key in ("parse.notice.used_cloud", "parse.notice.cloud_discarded"):
        assert f"t('{key}')" in render, f"{key} is not said through the catalogue"
        assert f"'{key}'" in catalogue, f"{key} has no catalogue entry"
    assert "left this machine" in catalogue
    assert "charged" in catalogue
    assert card


def test_parse_warnings_are_listed_rather_than_swallowed():
    render = function_body(PARSE_STATUS, "renderParseStatus")
    assert "parse.warnings" in render
    assert "parse-warning-list" in render
    assert "parse.error_message" in render


def test_quality_flags_stay_individually_addressable():
    """No score. A collapsed number is a threshold, and thresholds are where
    this project has promised not to put judgement."""
    quality = function_body(PARSE_STATUS, "renderParseQuality")
    assert "quality.flags" in quality
    assert "rule_id" in quality
    assert "severity" in quality
    assert "toFixed" not in quality
    assert "score" not in quality


def test_the_reparse_control_offers_both_force_ocr_and_a_parser_choice():
    assert element_by_id("button", "reparseBtn")
    assert element_by_id("fieldset", "reparseControls") is None  # a div, per the template
    assert element_by_id("input", "reparseForceOcr")
    assert element_by_id("select", "reparseParser")
    assert element_by_id("button", "reparseSubmitBtn")

    controls = INDEX_HTML[INDEX_HTML.index('id="reparseControls"') :]
    assert 'for="reparseForceOcr"' in controls
    assert "Force OCR" in controls
    assert 'for="reparseParser"' in controls
    assert "Parser" in controls

    options = re.search(r"REPARSE_PARSER_OPTIONS\s*=\s*\[(.*?)\];", CONFIG, re.DOTALL)
    assert options, "the reparse control declares no parser options"
    assert "docling" in options.group(1)
    assert "mineru_cloud" in options.group(1)
    assert "leavesMachine" in options.group(1)


def test_the_reparse_control_warns_before_the_consequence_happens():
    """A cloud parse leaves the machine, and the user has to be told first."""
    # The warning sentences are catalogue keys now, so the warning follows the
    # language switch; the strings themselves are asserted in the catalogue.
    assert "'reparse.force_ocr_warning'" in (JS_DIR / "i18n.js").read_text(encoding="utf-8")
    assert "上传到第三方服务" in CONFIG
    assert "重跑一次解析" in CONFIG

    warnings = function_body(PARSE_STATUS, "updateReparseWarnings")
    # Shown on selection, driven by the option and the checkbox.
    assert "option.leavesMachine" in warnings
    assert "forceOcr" in warnings
    assert "t('reparse.cloud_warning')" in warnings
    assert "t('reparse.force_ocr_warning')" in warnings
    assert element_by_id("div", "reparseWarning")

    # The cloud option must be flagged as leaving the machine in the data too,
    # so the warning survives a re-render.
    options = re.search(r"REPARSE_PARSER_OPTIONS\s*=\s*\[(.*?)\];", CONFIG, re.DOTALL).group(1)
    cloud = re.search(r"\{[^{}]*mineru_cloud[^{}]*\}", options)
    assert cloud and "leavesMachine: true" in cloud.group(0)


def test_a_reparse_the_server_refuses_is_reported_not_swallowed():
    """An unavailable endpoint is a visible state, not a silent no-op.

    The POST target is the T2.4 contract. This repository does not serve it
    yet, and a button that looks armed and does nothing is precisely the
    failure mode this product is built against — so the refusal is said in the
    card, and the document is reported as untouched.
    """
    path = re.search(r"REPARSE_API_PATH\s*=\s*'([^']+)'", CONFIG)
    assert path and path.group(1).endswith("/reparse")
    submit = function_body(PARSE_STATUS, "submitReparse")
    assert "buildReparseUrl(documentId)" in submit
    # The two failure sentences are catalogue entries now, so they follow the
    # language switch. What is pinned here is what each of them must state: that
    # this server offers no reparse endpoint, and that the document was left
    # untouched — in both languages, because a Chinese reader told only that a
    # reparse failed is not told their paper is intact.
    assert "'reparse.result.no_route'" in submit
    catalogue = (JS_DIR / "i18n.js").read_text(encoding="utf-8")
    assert "'reparse.result.no_route'" in catalogue
    assert "'reparse.result.failed'" in catalogue
    assert "does not offer a reparse endpoint" in catalogue
    assert "left untouched" in catalogue
    assert "文档没有被改动" in catalogue, (
        "the Chinese locale must say the document was untouched too"
    )
    # Disabled with no document id, rather than live and failing.
    warnings = function_body(PARSE_STATUS, "updateReparseWarnings")
    assert "state.currentDocumentId" in warnings
    assert "setControlDisabled(submit, !hasDocument)" in warnings


def test_the_document_id_is_not_confused_with_the_session_id():
    """The two are minted independently; conflating them produces a 404 that
    the reader cannot tell from a typo."""
    finalize = function_body(SECTIONS, "finalizeAnalysisResultState")
    assert "state.currentDocumentId" in finalize
    assert "data.document_id" in finalize
    assert "state.currentDocumentId = state.currentSessionId" not in SECTIONS


def test_a_new_analysis_closes_the_panel_and_clears_the_parse_status():
    reset = function_body(SECTIONS, "resetResultView")
    assert "closeEvidence()" in reset
    assert "resetParseStatus()" in reset
    assert "state.currentParseProvenance = null" in reset
    assert "state.currentDocumentId = ''" in reset


def test_the_panel_survives_a_failed_re_analysis_restore():
    """A new analysis that fails restores the previous workspace — evidence
    panel and parse status included, or the reader is left looking at a status
    for a document that is no longer displayed."""
    capture = function_body(SECTIONS, "captureWorkspaceState")
    assert "currentDocumentId" in capture
    assert "currentParseProvenance" in capture
    assert "parseStatus" in capture
    restore = function_body(SECTIONS, "restoreWorkspaceState")
    assert "snapshot.currentDocumentId" in restore
    assert "snapshot.currentParseProvenance" in restore


# --------------------------------------------------------------------------
# WALK-DEFECT-4: a reparse option that cannot run must say so
# --------------------------------------------------------------------------


def _parser_options() -> list[dict]:
    """`REPARSE_PARSER_OPTIONS` as data, parsed from the shipped `config.js`."""
    block = re.search(r"REPARSE_PARSER_OPTIONS\s*=\s*\[(.*?)\];", CONFIG, re.DOTALL)
    assert block, "REPARSE_PARSER_OPTIONS is missing from config.js"
    options = []
    for entry in re.finditer(r"\{([^{}]*)\}", block.group(1)):
        body = entry.group(1)
        fields = dict(re.findall(r"(\w+):\s*'([^']*)'", body))
        # Booleans are written unquoted in config.js, so they are captured
        # separately rather than being silently dropped from the parse.
        for name, value in re.findall(r"(\w+):\s*(true|false)\b", body):
            fields[name] = value
        options.append(fields)
    assert options, "no parser options were parsed out of config.js"
    return options


def _catalogue() -> dict:
    """Both locale tables of the catalogue as `{key: (zh, en)}`."""
    source = (JS_DIR / "i18n.js").read_text(encoding="utf-8")
    tables = {}
    for locale in ("ZH_CN", "EN"):
        match = re.search(rf"const {locale} = \{{(.*?)\n\}};", source, re.DOTALL)
        assert match, f"the {locale} table is missing from i18n.js"
        tables[locale] = dict(
            re.findall(r"^    '([^']+)':\s*(.*?),?$", match.group(1), re.MULTILINE)
        )
    return tables


def test_a_reparse_option_that_cannot_run_in_this_deployment_is_labelled_with_its_requirement():
    """WALK-DEFECT-4: the menu must not offer a parser it cannot use, silently.

    With the shipped default (`PARSER_BACKEND=legacy`) the gateway is rolled back
    to the legacy adapter, so every *named* parser answers `409
    parser_unavailable`. The walkthrough's finding was not the 409 — it is that
    the control offered the choice without saying what the choice requires, and
    under a different deployment the same menu offers a parser that works fine.

    So the contract asserted here is: every option is either absent from the
    menu, or carries a `requirementKey` whose sentence exists in **both** locales
    and names the requirement. Filtering by availability is deliberately *not*
    the mechanism, because the browser is not told what this deployment has
    installed (no endpoint publishes the parser catalogue), so a filter would be
    wrong in the other direction on a gateway deployment. Labelling is the one
    statement that is true in every deployment.
    """
    from paperwhisperer.core import config

    tables = _catalogue()
    options = _parser_options()

    def bilingual(key: str) -> tuple[str, str]:
        zh = tables["ZH_CN"].get(key)
        en = tables["EN"].get(key)
        assert zh is not None, f"{key} has no zh-CN catalogue entry"
        assert en is not None, f"{key} has no en catalogue entry"
        return zh, en

    legacy_default = config.PARSER_BACKEND != "gateway"

    for option in options:
        # Every option is named in the reader's language.
        bilingual(option["labelKey"])
        # And every option states what it requires, in both languages.
        requirement_key = option.get("requirementKey")
        assert requirement_key, (
            f"the {option.get('value')!r} option carries no requirementKey, so a "
            "reader cannot tell whether it can run in this deployment"
        )
        zh, en = bilingual(requirement_key)
        assert zh.strip(), f"the requirement for {option.get('value')!r} is blank in zh-CN"
        assert en.strip(), f"the requirement for {option.get('value')!r} is blank in en"
        if legacy_default and option.get("value", "") in {"docling", "docling_ocr"}:
            # Under the shipped default these cannot run; the requirement must
            # say exactly that, in words, in the reader's language.
            for text in (zh, en):
                assert "PARSER_BACKEND" in text, (
                    "the docling options' requirement does not name the setting "
                    f"that decides whether they can run: {text!r}"
                )
                assert "gateway" in text, (
                    f"the requirement does not say what has to be set: {text!r}"
                )

    # The cloud option keeps its egress warning (a separate, stronger rule).
    cloud = next(option for option in options if option["value"] == "mineru_cloud")
    assert cloud.get("leavesMachine") == "true"
    warnings = function_body(PARSE_STATUS, "updateReparseWarnings")
    assert "option.requirementKey" in warnings, (
        "the selected option's own requirement is not shown before the reparse"
    )
    assert "t('reparse.cloud_warning')" in warnings

    # The default entry must not imply a single best parser: the routing decision
    # is per media type (DOCX is pinned to legacy by design), so the default
    # entry says so.
    default_option = next(option for option in options if option["value"] == "")
    zh, en = bilingual(default_option["requirementKey"])
    for text in (zh, en):
        assert "file type" in text or "文件类型" in text, (
            f"the default entry does not say the route is chosen per file type: {text!r}"
        )


def test_no_reparse_option_is_offered_without_saying_it_needs_something():
    """Every option a reader can pick names its requirement — no exceptions.

    Stated separately from the test above because it is the invariant that keeps
    a *newly added* option honest: a fifth parser added to `config.js` without a
    `requirementKey` fails here, at review, rather than shipping as a control
    that offers a choice whose consequence is undisclosed.
    """
    tables = _catalogue()
    for option in _parser_options():
        assert option.get("requirementKey"), (
            f"{option.get('value')!r} has no requirementKey"
        )
        assert option["requirementKey"] in tables["ZH_CN"]
        assert option["requirementKey"] in tables["EN"]
        assert option["labelKey"] in tables["ZH_CN"]
        assert option["labelKey"] in tables["EN"]


# --------------------------------------------------------------------------
# Wiring, and the no-build constraint
# --------------------------------------------------------------------------


def test_the_new_modules_are_wired_into_the_entry_point_and_the_result():
    assert "initializeEvidencePanel" in MAIN
    assert "initializeParseStatus" in MAIN
    assert "initializeEvidencePanel()" in MAIN
    assert "initializeParseStatus()" in MAIN
    assert "renderParseStatus" in SECTIONS
    assert "collectCitations" in CHAT
    assert "renderInlineClaims" in CHAT


def test_the_citation_reading_is_defensive_about_a_moving_contract():
    """The Q&A response shape is being changed right now.

    `collectCitations` probes every list the answer might carry its citations
    under and returns an empty list when none is present, so an answer with no
    citations renders exactly as it did before this work existed rather than
    throwing inside a stream handler.
    """
    collect = function_body(EVIDENCE, "collectCitations")
    for key in ("citations", "evidence", "claims"):
        assert key in collect
    assert "Array.isArray" in collect
    assert "continue" in collect
    # The shell carries its own container, so streaming deltas cannot wipe the
    # claims once they are rendered.
    shell = function_body(CHAT, "appendStreamingAnswerShell")
    assert "answer-claims" in shell
    done = function_body(CHAT, "askQuestion")
    assert "collectCitations(payload)" in done
    assert "renderInlineClaims(streamingShell.claims, citations)" in done


def test_no_framework_and_no_bundler_was_introduced():
    """Vanilla ES modules, loaded directly, with no npm dependency."""
    forbidden = ("react", "vue", "angular", "svelte", "jquery", "preact", "solid-js")
    for name in NEW_MODULES:
        source = (JS_DIR / name).read_text(encoding="utf-8")
        for framework in forbidden:
            assert framework not in source.lower(), f"{name} references {framework}"
        # Every import is a relative path to another module in this directory.
        for specifier in re.findall(r"from\s+'([^']+)'", source):
            assert specifier.startswith("./"), f"{name} imports {specifier}"
            assert (JS_DIR / specifier[2:]).is_file(), f"{name} imports missing {specifier}"

    for name in NEW_MODULES:
        assert "export " in (JS_DIR / name).read_text(encoding="utf-8")

    # Still exactly one entry point, and the new modules are reachable from it.
    assert INDEX_HTML.count('<script type="module"') == 1
    assert "from './evidence.js'" in MAIN
    assert "from './parseStatus.js'" in MAIN


def test_the_new_template_markup_is_reachable_and_nested_correctly():
    """A panel that never made it into the served template would pass every
    other test here, so the ids are checked against the real file."""
    for element_id in (
        "evidencePanel",
        "evidencePanelHeading",
        "evidenceCloseBtn",
        "evidenceBackdrop",
        "evidenceIdle",
        "evidenceLoading",
        "evidenceError",
        "evidenceNoTarget",
        "evidenceEmpty",
        "evidenceContent",
        "evidenceTarget",
        "evidenceSnippet",
        "evidenceTable",
        "evidenceWarnings",
        "parseStatus",
        "parseStatusCard",
        "reparseBtn",
        "reparseControls",
        "reparseParser",
        "reparseForceOcr",
        "reparseWarning",
        "reparseResult",
        "reparseSubmitBtn",
    ):
        assert f'id="{element_id}"' in INDEX_HTML, f"#{element_id} is not in index.html"

    assert INDEX_HTML.count("<html") == 1
    assert INDEX_HTML.rstrip().endswith("</html>")
    # Every id the JS reads through getElementById in the new modules exists.
    for name in NEW_MODULES:
        source = (JS_DIR / name).read_text(encoding="utf-8")
        for element_id in re.findall(r"getElementById\('([^']+)'\)", source):
            assert f'id="{element_id}"' in INDEX_HTML, (
                f"{name} reads #{element_id}, which is not in the template"
            )


# ==========================================================================
#
# The viewer's language
# ==========================================================================


VIEWER = (JS_DIR / "evidenceViewer.js").read_text(encoding="utf-8")
VIEWER_BODY = strip_js_comments(VIEWER)

#: Every sentence the panel invents for itself, and the render function that
#: puts it on screen. Each is asserted to go through the catalogue, because a
#: partially translated panel is worse than an untranslated one: the reader has
#: no way to know which half of the evidence they can trust.
VIEWER_SURFACES = {
    "buildHeader": ("evidence.view_kind", "evidence.not_a_raster"),
    "buildHighlightStatus": ("evidence.highlight_drawn", "evidence.highlight_absent"),
    "buildTruncationNotes": ("evidence.view.blocks_truncated", "evidence.view.omitted_relative"),
    "renderPageMap": ("evidence.view.preview_truncated",),
    "renderSlideView": ("evidence.view.no_text",),
    "renderElementView": ("evidence.paragraph_label", "evidence.view.no_text"),
    "renderEvidenceView": ("evidence.unsupported_view",),
}


def test_the_viewer_routes_every_interface_string_through_the_catalogue():
    """One reader-facing sentence per key, and the key is looked up.

    Not a grep for Chinese: the point is that the *lookup* exists, so the string
    follows the reader's switch. A literal in any of these functions is a
    sentence that will render in Chinese under an English interface, and the
    panel is the last surface a reader looks at before deciding to trust a
    citation.
    """
    assert "from './i18n.js'" in VIEWER
    catalogue = (JS_DIR / "i18n.js").read_text(encoding="utf-8")
    for function_name, keys in VIEWER_SURFACES.items():
        body = function_body(VIEWER_BODY, function_name)
        for key in keys:
            assert key in body, f"{function_name}() does not use '{key}'"
            assert key in catalogue, f"{key} is not shipped in the catalogue"
        # The lookup, not the constant: a table read straight into `textContent`
        # is the failure this file's whole no-framework section is about.
        assert re.search(r"\bt\(", body) or "localised(" in body, (
            f"{function_name}() builds user-visible text with no catalogue lookup"
        )


def test_the_viewer_does_not_route_the_documents_own_words_through_the_catalogue():
    """The other half of the boundary, and the one that is easy to over-apply.

    Routing the *cited passage* through the catalogue would translate a
    reader's paper because they switched their interface. So the passage, the
    block previews and the heading path are passed through untouched. Asserted
    per call site rather than by grep, because the shapes differ — a block's
    preview is the third argument of `element()`, the passage is the second of
    `element()` too, and a reader who "helpfully" wrapped one of them in `t()`
    would still leave the other three alone.

    The view's `note` used to be in this list and is not any more: it travels
    inside the payload, which is what made it look like the document's words,
    and it is not. It is a statement about the drawing, so it is translated —
    see {@link test_the_views_disclaimer_is_looked_up_by_code} for the other
    side of that move.
    """
    cases = {
        "renderPageMap": "block.text_preview",
        "renderSlideView": "block.text",
        "renderElementView": "block.text",
    }
    for function_name, expression in cases.items():
        body = function_body(VIEWER_BODY, function_name)
        assert expression in body, f"{function_name}() no longer reads {expression}"
        # Every catalogue call in the function, with its argument list, so the
        # document expression can be caught *inside* one rather than merely
        # somewhere on the same line as one.
        arguments = re.findall(r"\b(?:t|localised)\([^()]*", body)
        for call in arguments:
            assert expression not in call, (
                f"{function_name}() routes {expression} through the catalogue: {call!r}"
            )
    # The heading path is the document's own outline, joined as sent.
    element_view = function_body(VIEWER_BODY, "renderElementView")
    assert "path.join(' › ')" in element_view
    assert "view.label" in function_body(VIEWER_BODY, "buildHeader")


def test_the_views_disclaimer_is_looked_up_by_code_and_only_speaks_if_cannot():
    """The note is read through a key, with the server's own sentence as its floor.

    This is the structural half of the fix for the one Chinese sentence the
    language switch could not reach. The payload carries `note_code`; the
    viewer turns that code into `evidence.view_note.<code>` and looks it up, so
    the sentence follows the reader's switch. The server's `note` is not
    discarded — it is what renders when there is no code, when the code is one
    this build has never heard of, or when the catalogue entry has been lost —
    because a disclaimer that degrades to nothing tells a reader the layout is
    the page.

    `describeViewNote()` is asserted separately from the render site because
    the two failures are different: a viewer that forgets to *call* it renders
    the server's Chinese sentence, and one that calls it without a floor renders
    a dotted key.
    """
    describe = function_body(VIEWER_BODY, "describeViewNote")
    assert "view?.note" in describe or "view.note" in describe, (
        "describeViewNote() no longer reads the server's sentence"
    )
    assert "view?.note_code" in describe or "view.note_code" in describe, (
        "describeViewNote() does not read the code the server sent beside the note"
    )
    # The lookup, and the server's sentence as its second argument. `localised()`
    # returning the key itself is exactly the case the floor exists for.
    calls = re.findall(r"\blocalised\([^;]*", describe)
    assert calls, "describeViewNote() never calls localised()"
    assert any("serverText" in call for call in calls), (
        "describeViewNote() looks the note up without keeping the server's "
        f"sentence as a floor: {calls!r}"
    )
    assert re.search(r"const NOTE_CODE_KEY_PREFIX = 'evidence\.view_note\.';", VIEWER), (
        "the key prefix the disclaimer is assembled from has moved or changed"
    )

    render = function_body(VIEWER_BODY, "renderEvidenceView")
    assert "describeViewNote(view)" in render, (
        "renderEvidenceView() no longer puts the disclaimer through "
        "describeViewNote()"
    )
    # The note is only added when there is a sentence to show: a view with no
    # note and no code must not grow an empty element, which reads as a layout
    # with nothing to say about itself.
    assert re.search(r"if \(noteText\)", render), (
        "the disclaimer is appended without checking it has text"
    )


def test_the_viewer_repaints_the_open_panel_on_a_locale_change():
    """Structural half of the repaint; `tests/test_i18n.py` runs it.

    Read here so that unsubscribing is caught by this file's structural pass
    rather than by a behavioural test in another one, and asserted here because
    this is the file that owns the viewer.
    """
    assert "onLocaleChange" in VIEWER
    match = re.search(r"onLocaleChange\((.*?)\);", VIEWER, re.DOTALL)
    assert match, "evidenceViewer.js does not register a locale listener"
    assert "repaintLocalizedView" in match.group(1)
    assert "repaintLocalizedView" in function_body(VIEWER_BODY, "renderEvidenceView") or (
        "lastRenderedView" in VIEWER_BODY
    ), "the repaint has nothing to repaint without the remembered view"
