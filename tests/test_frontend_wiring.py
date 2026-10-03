"""Front-end wiring for T3.2 / T3.9: reasons, four-state status, the 404 path.

Three back-end capabilities landed — multi-signal recommendations
(``POST /api/paper-recommendations``), a reparse control pointed at
``/v1/documents/{id}/reparse``, and citation deep links — and this file holds
the front end to them. There is no browser here and adding one is out of scope,
so nothing renders anything; the assertions are structural properties of the
shipped files, the same shape ``tests/test_frontend_evidence.py`` established.

The load-bearing assertions, and why each is structural rather than textual:

* **The recommendation UI reads the per-signal ``reasons``.** T3.2's
  acceptance criterion is "UI 有'为什么推荐'". An API can return a reason
  forever and the criterion stays unmet until a client draws it, so the test
  requires the reasons to be read off each row and labelled from the payload —
  not reconstructed by the client from the papers it happens to group.

* **``unavailable`` renders differently from ``empty``.** These are the pair
  that gets conflated, and conflating them is a falsehood about the user's own
  paper: "被引：没有" is a claim about the literature, "被引：未能获取" is a
  claim about us (rate limited, no identifier, network down). The four states
  are asserted to be four distinct values with four distinct sentences, and the
  ``unavailable`` copy is required to *deny* the "none" reading explicitly.

* **``possible_duplicates`` has a visible home.** T3.1 established that a
  possible match is shown as a possibility and never silently merged. That
  decision only pays off if the user can see the pair, the basis, and the fact
  that both rows were kept — all three are required here.

* **A reparse 404 is handled honestly.** The route is a T2.4 contract that may
  not exist yet on the server. The button must say the server does not offer
  it and must leave the document untouched — a control that looks armed and
  does nothing is the failure mode this product is built against.

* **The deep link works from a cold load.** ``#evidence=…&document=…`` is read
  on init and on ``hashchange``; both are asserted to be wired, and the
  degrade-to-a-message path is asserted to exist so a link into a document the
  browser has not loaded says so rather than opening an empty panel.

Comments are stripped before the structural assertions wherever a comment
merely *discusses* a guarantee, so prose cannot satisfy an assertion about
behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JS_DIR = PROJECT_ROOT / "templates" / "static" / "js"
TEMPLATE = PROJECT_ROOT / "templates" / "index.html"

PAPERS = (JS_DIR / "papers.js").read_text(encoding="utf-8")
PAPER_LIST = (JS_DIR / "paperList.js").read_text(encoding="utf-8")
SIGNALS = (JS_DIR / "recommendationSignals.js").read_text(encoding="utf-8")
PARSE_STATUS = (JS_DIR / "parseStatus.js").read_text(encoding="utf-8")
EVIDENCE = (JS_DIR / "evidence.js").read_text(encoding="utf-8")
EXPORT = (JS_DIR / "export.js").read_text(encoding="utf-8")
CONFIG = (JS_DIR / "config.js").read_text(encoding="utf-8")
SECTIONS = (JS_DIR / "sections.js").read_text(encoding="utf-8")
MAIN = (JS_DIR / "main.js").read_text(encoding="utf-8")
STATE = (JS_DIR / "state.js").read_text(encoding="utf-8")
INDEX_HTML = TEMPLATE.read_text(encoding="utf-8")
CATALOGUE = (JS_DIR / "i18n.js").read_text(encoding="utf-8")


def _catalogue_source() -> str:
    """The translation catalogue, as source.

    Several assertions below check that a module names a message by key rather
    than carrying the sentence inline. Checking the *name* is only half of it:
    a key that is referenced but never defined renders as a dotted string onto
    the screen, which is a worse failure than prose frozen into one language.
    So the reference and the definition are asserted together, here.
    """
    return CATALOGUE

#: Modules this task added or touched, for the no-framework / relative-import
#: checks.
NEW_MODULES = ("recommendationSignals.js",)


# --------------------------------------------------------------------------
# Helpers (same scanners the sibling suites established)
# --------------------------------------------------------------------------


def strip_js_comments(source: str) -> str:
    """Remove ``//`` and ``/* */`` comments, leaving string literals intact."""
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
            while index < length and not (
                source[index] == "*" and source[index + 1 : index + 2] == "/"
            ):
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
    """Remove static ``import`` statements, so importing a constant is not a use."""
    return re.sub(r"^\s*import\s+[^;]*?from\s+'[^']+';", "", source, flags=re.MULTILINE)


def function_body(source: str, name: str) -> str:
    """Return the body of ``function name(...)`` by string-aware brace matching."""
    match = re.search(rf"function\s+{re.escape(name)}\s*\(", source)
    assert match, f"function {name}() is missing"

    index = source.index("{", match.end())
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
# The recommendation client talks to the multi-signal route, and sends a seed
# --------------------------------------------------------------------------


def test_the_recommendation_client_calls_the_multi_signal_route():
    """T3.2 is ``/api/paper-recommendations``; the legacy route has no reasons."""
    path = re.search(r"RECOMMENDATIONS_API_PATH\s*=\s*'([^']+)'", CONFIG)
    assert path, "RECOMMENDATIONS_API_PATH is not a literal path"
    assert path.group(1) == "/api/paper-recommendations"

    body = function_body(PAPERS, "recommendPapers")
    assert "RECOMMENDATIONS_API_PATH" in body, "the client must call the documented path"
    # The legacy route returns a single top-level `reason` and no per-row
    # provenance, so pointing at it would silently drop the whole feature.
    assert "/api/recommend-papers" not in PAPERS, "the client still calls the legacy route"
    # Issued through the transport choke point rather than `fetch` directly, so
    # this call inherits the timeout, the size cap and the error taxonomy.
    assert "apiRequest(RECOMMENDATIONS_API_PATH" in body


def test_a_recommendation_carries_the_seed_paper_the_route_requires():
    """The route 400s ``missing_seed_paper`` without a ``paper``, so the client
    must always send one it can actually build."""
    assert "function buildRecommendationSeed" in PAPERS
    assert "function setSeedPaper" in PAPERS
    assert "function recommendationSeedTitle" in PAPERS
    seed = function_body(PAPERS, "buildRecommendationSeed")
    assert "state.currentSeedPaper" in seed
    assert "recommendationSeedTitle()" in seed, "the title fallback is not wired"

    body = function_body(PAPERS, "recommendPapers")
    assert "buildRecommendationSeed()" in body
    assert "paper: seed" in body, "the seed is built but not sent"

    # The four identifier fields T3.1 added must be copied onto the seed — they
    # are what the citation-graph signals walk, and omitting them is what makes
    # those four signals report unavailable for every document.
    setter = function_body(PAPERS, "setSeedPaper")
    for field in ("doi", "arxiv_id", "s2_paper_id", "openalex_id"):
        assert field in setter, f"the seed does not carry {field}"

    # And a seed with neither a title nor an identifier is refused outright,
    # because the route would answer 400 `missing_seed_paper` and the failure
    # would read as a server fault rather than as "there is nothing to
    # recommend from".
    #
    # The sentence itself is a catalogue key now: the test pins the *guard*, not
    # the wording, and pins that the key resolves in both languages — a literal
    # assertion here would have failed the moment the copy was localised, which
    # is the opposite of what this test is for.
    assert "papers.recommend.no_paper_identity" in body
    assert _catalogue_source().count("'papers.recommend.no_paper_identity'") == 2, (
        "the refusal is shown from the catalogue, so the key must be defined in "
        "both locales — once in each table, not twice in one"
    )
    assert "seed.title" in body and "seed.doi" in body, (
        "the no-seed guard must check every identifier the route recognises"
    )


# --------------------------------------------------------------------------
# The four-state signal status
# --------------------------------------------------------------------------


def test_the_four_signal_states_are_declared_and_all_distinct():
    """``ok`` / ``empty`` / ``unavailable`` / ``skipped`` — four, not three.

    The gap that matters is between ``empty`` and ``unavailable``, but the four
    are asserted together because a two-state vocabulary cannot express
    "we chose not to ask", and that absence is what makes a budget-limited run
    read as a thin result rather than as a budget decision.
    """
    block = re.search(
        r"export const RECOMMENDATION_SIGNAL_STATES = \{(.*?)\n\};", CONFIG, re.DOTALL
    )
    assert block, "the four-state signal vocabulary is missing from config.js"
    keys = set(re.findall(r"^\s{4}(ok|empty|unavailable|skipped):", block.group(1), re.MULTILINE))
    assert keys == {"ok", "empty", "unavailable", "skipped"}

    # Each state carries a tone and a label, and they are not all the same
    # string — a status badge that reads "已获取" for an unavailable signal is
    # the same defect as no badge at all.
    tones = re.findall(r"tone:\s*'([^']+)'", block.group(1))
    labels = re.findall(r"label:\s*'([^']+)'", block.group(1))
    assert len(tones) == 4 and len(set(tones)) == 4, "the four states must have four tones"
    assert len(labels) == 4 and len(set(labels)) == 4, "the four states must have four labels"


def test_unavailable_is_worded_as_could_not_check_and_never_as_none():
    """The single most important sentence in the feature.

    A rate-limited provider rendered as an empty list tells the user a falsehood
    about their own paper. So the ``unavailable`` summary must explicitly deny
    the "there are none" reading, and must be worded differently from ``empty``,
    which *is* allowed to say "nothing matched".
    """
    states = dict(
        re.findall(
            r"^\s{4}(ok|empty|unavailable|skipped):\s*\{(.*?)\n    \}",
            re.search(
                r"export const RECOMMENDATION_SIGNAL_STATES = \{(.*?)\n\};",
                CONFIG,
                re.DOTALL,
            ).group(1),
            re.DOTALL | re.MULTILINE,
        )
    )
    assert set(states) == {"ok", "empty", "unavailable", "skipped"}

    def summary(name: str) -> str:
        match = re.search(r"summary:\s*'([^']*)'", states[name], re.DOTALL)
        return match.group(1) if match else ""

    unavailable, empty = summary("unavailable"), summary("empty")

    assert unavailable, "the unavailable state has no sentence"
    assert empty, "the empty state has no sentence"
    # The explicit denial of the "none" reading. The word 没有 appears here
    # precisely because it is being negated — the sentence must read "this does
    # NOT mean there are none" — so the check is for the denial form
    # 不代表没有, not for the absence of 没有.
    assert "不代表没有" in unavailable, (
        "the unavailable sentence must explicitly deny that there are no papers"
    )
    assert "未能" in unavailable, "the unavailable sentence must name the failure"
    # ...and the two must not be the same sentence.
    assert unavailable != empty, "unavailable and empty render the same copy"


def test_a_signal_row_distinguishes_the_four_states_in_the_dom():
    """``data-status`` carries the state, never a count.

    The whole visual distinction hangs off this attribute, so a row that wrote
    ``data-status="empty"`` for an unavailable provider — or keyed the badge off
    ``count === 0`` — would put the false "none" reading back on screen through
    the back door, no matter what the copy says.
    """
    row = function_body(SIGNALS, "buildSignalRow")
    assert "dataset.status" in row
    assert "stateForStatus(entry?.status)" in row

    resolver = function_body(SIGNALS, "stateForStatus")
    # An unrecognised status degrades to `unavailable`, not to `empty`: failing
    # toward the "we could not check" side is the safe direction, because unknown
    # is not evidence that there is nothing.
    assert "RECOMMENDATION_UNKNOWN_SIGNAL_STATE" in resolver
    unknown = re.search(
        r"export const RECOMMENDATION_UNKNOWN_SIGNAL_STATE = '([^']+)'", CONFIG
    )
    assert unknown and unknown.group(1) == "unavailable"

    # A count is not printed for the two states that have no meaningful count,
    # or "0 篇" would sit next to an unavailable badge and reassert the reading
    # the sentence just denied.
    assert "state.tone !== 'unavailable'" in row
    assert "state.tone !== 'skipped'" in row


def test_the_signal_region_reads_the_per_signal_status_array():
    """The client renders the ``signals`` array it is sent, not a guess."""
    render = function_body(SIGNALS, "renderRecommendationSignals")
    assert "recommendationSignals" in render
    assert "recommendationDuplicates" in render
    assert "Array.isArray(signals)" in render

    body = function_body(PAPERS, "recommendPapers")
    assert "data.signals" in body, "the per-signal status is never read"
    assert "renderRecommendationSignals(" in body


def test_the_signal_region_is_a_visible_container_not_an_absence():
    """Both regions are declared in the template with the ids the module reads.

    A module that mounts into a container the template never declares returns
    early and shows nothing, and "nothing shown" is indistinguishable from
    "every signal was fine" — the exact false reading this UI exists to avoid.
    """
    for element_id in ("recommendationSignals", "recommendationDuplicates"):
        assert f'id="{element_id}"' in INDEX_HTML, f"#{element_id} is not in index.html"
        assert element_id in SIGNALS, f"no module reads #{element_id}"

    for element_id in ("recommendationSignals", "recommendationDuplicates"):
        source = strip_js_imports(strip_js_comments(SIGNALS))
        assert f"getElementById('{element_id}')" in source

    # And the check tool requires both containers, so a rename that orphans one
    # fails CI rather than shipping a silently-missing region.
    tool = (PROJECT_ROOT / "tools" / "check_frontend.py").read_text(encoding="utf-8")
    assert '"recommendationSignals.js"' in tool, (
        "recommendationSignals.js is not a required-container module"
    )


# --------------------------------------------------------------------------
# The per-row reasons — "why is this paper here"
# --------------------------------------------------------------------------


def test_each_recommendation_row_renders_its_per_signal_reasons():
    """The reasons are read off the row and labelled from the payload.

    The client must not reconstruct a label from the set of papers it grouped —
    that reconstruction is how a "被引" chip ends up on a paper that was
    actually reached by a topic search. So ``label_zh`` / ``reason_zh`` /
    ``signal`` are read straight through.
    """
    item = function_body(PAPER_LIST, "buildReasonItem")
    for field in ("reason?.signal", "reason?.label_zh", "reason?.reason_zh", "reason?.source"):
        assert field in item, f"the reason item does not read {field}"

    reasons = function_body(PAPER_LIST, "buildRecommendationReasons")
    assert "reasons" in reasons
    assert "RECOMMENDATION_REASONS_HEADING" in reasons
    heading = re.search(
        r"export const RECOMMENDATION_REASONS_HEADING = '([^']+)'", CONFIG
    )
    assert heading and "为什么推荐" in heading.group(1)

    # ...and the block is actually mounted on the card.
    card = function_body(PAPER_LIST, "renderPaperList")
    assert "buildRecommendationReasons(item.reasons)" in card
    assert "buildPossibleDuplicateNote(item.possible_duplicates)" in card


def test_a_row_with_no_reasons_gets_no_reason_heading():
    """Search rows carry no ``reasons``; a heading over nothing is worse than
    no heading, because it promises an explanation the row does not have."""
    reasons = function_body(PAPER_LIST, "buildRecommendationReasons")
    assert "return null" in reasons, "a row with no reasons must render nothing"
    assert reasons.index("return null") < reasons.index("RECOMMENDATION_REASONS_HEADING"), (
        "the no-reasons guard must run before the heading is emitted"
    )


def test_reason_confidence_is_kept_attached_to_the_reason():
    """``declared`` and ``derived`` prove different things and only the first
    is authoritative, so the qualifier travels with the reason it qualifies."""
    item = function_body(PAPER_LIST, "buildReasonItem")
    assert "reason?.confidence" in item
    assert "RECOMMENDATION_CONFIDENCE_LABELS" in item
    labels = re.search(
        r"export const RECOMMENDATION_CONFIDENCE_LABELS = \{(.*?)\n\};", CONFIG, re.DOTALL
    )
    assert labels and "declared" in labels.group(1) and "derived" in labels.group(1)


# --------------------------------------------------------------------------
# possible_duplicates — shown, never silently merged
# --------------------------------------------------------------------------


def test_possible_duplicates_have_a_visible_run_level_home():
    """The pair is named in its own region, with both titles and the basis.

    T3.1's decision — a probable match is a possibility, never merged — only
    pays off if the user can see the pair, what was matched on, and that both
    rows were kept.
    """
    panel = function_body(SIGNALS, "buildDuplicatePanel")
    assert "pair?.title" in panel and "pair?.other_title" in panel, (
        "both titles of a suspect pair must be shown"
    )
    assert "pair?.basis_zh" in panel, "the basis the identity matched on must be shown"
    assert "POSSIBLE_DUPLICATE_POLICY" in panel, (
        "the reason the pair was NOT merged must be stated in the user's words"
    )

    policy = re.search(r"export const POSSIBLE_DUPLICATE_POLICY\s*=\s*(.+?);", CONFIG, re.DOTALL)
    assert policy, "the non-merge policy is not written down"
    assert "没有合并" in policy.group(1), "the policy must say both rows were kept"


def test_a_possible_duplicate_is_badged_on_the_card_too():
    """Both rows of a suspect pair carry the marker, so neither looks confirmed.

    The backend deliberately flags both rows. A badge on only one would leave
    the other looking like an ordinary result, and a reader who scrolls only the
    cards would never learn the pair is in question.
    """
    note = function_body(PAPER_LIST, "buildPossibleDuplicateNote")
    assert "POSSIBLE_DUPLICATE_BADGE" in note
    assert "pairs.filter(Boolean)" in note
    assert "return null" in note

    badge = re.search(r"export const POSSIBLE_DUPLICATE_BADGE = '([^']+)'", CONFIG)
    assert badge and badge.group(1), "the per-card duplicate badge has no label"


def test_the_client_reads_possible_duplicates_off_the_payload():
    body = function_body(PAPERS, "recommendPapers")
    assert "data.possible_duplicates" in body, "the run-level duplicate pairs are never read"
    assert "state.currentRecommendationDuplicates" in body


# --------------------------------------------------------------------------
# The reparse control when the route is absent
# --------------------------------------------------------------------------


def test_a_reparse_404_says_the_server_does_not_offer_it_and_leaves_the_document_alone():
    """A 404 is a named, visible outcome — not a silent no-op.

    The reparse route is a T2.4 contract that may not exist yet on the server
    this build runs against. The button must name that absence and must leave
    the document untouched, because a control that looks armed and does nothing
    is the failure mode this product is built against.
    """
    submit = function_body(PARSE_STATUS, "submitReparse")
    # The status is now read off the normalised `ApiError` rather than off a raw
    # `Response`, because the request goes through the transport choke point
    # (`apiRequest`) that turns any failure into that shape. The property being
    # asserted is unchanged: a 404 is still called out separately.
    assert "error.status === 404" in submit, "a 404 is not called out separately"

    # The 404 branch names its message by KEY, not by carrying the sentence
    # inline. These two assertions used to read the English prose out of the
    # module body, which made the test a photograph of one locale's wording:
    # moving the sentence into the catalogue — the migration that let a reader
    # switch to 中文 — turned the pin red without changing the behaviour it
    # claimed to protect.
    #
    # What is actually worth pinning is stronger than the sentence was: the
    # failure message is *localisable*, and the key it names resolves in every
    # shipped locale. A key that stopped resolving would render as a dotted
    # string onto the card, which is worse than prose in one language, so the
    # second assertion is the load-bearing one.
    assert "reparse.result.no_route" in submit, (
        "the missing-route branch must name a catalogue key, not inline prose"
    )
    assert "reparse.result.no_route" in _catalogue_source(), (
        "the key the 404 branch names does not exist in the catalogue"
    )

    # A non-404 failure gets its own branch, so it is never misreported as a
    # missing route.
    assert submit.count("setReparseResult(") >= 2, (
        "a non-404 failure must not reuse the missing-route message"
    )
    assert "reparse.result.failed" in submit, (
        "other failures need their own catalogue key, not the missing-route one"
    )

    # Nothing on the failure path writes to the parse provenance or re-renders
    # the status card, which is what "left untouched" means operationally.
    stripped = strip_js_comments(submit)
    failure_branch = stripped[stripped.index("const reason =") : stripped.index("finally")]
    assert "renderParseStatus" not in failure_branch, (
        "a failed reparse must not overwrite the parse status"
    )
    assert "state.currentParseProvenance" not in failure_branch


def test_the_reparse_route_the_client_calls_is_the_documented_one():
    path = re.search(r"REPARSE_API_PATH\s*=\s*'([^']+)'", CONFIG)
    assert path, "REPARSE_API_PATH is not a literal path"
    assert path.group(1) == "/v1/documents/{documentId}/reparse"
    submit = function_body(PARSE_STATUS, "submitReparse")
    assert "buildReparseUrl(documentId)" in submit


# --------------------------------------------------------------------------
# Citation deep links from a cold load
# --------------------------------------------------------------------------


def test_the_deep_link_is_read_on_load_and_on_hashchange():
    """A link in an exported file has to resolve, so the fragment is read on
    init (cold load) and on ``hashchange`` (pasted into an open tab)."""
    init = function_body(EVIDENCE, "initializeEvidencePanel")
    # On load: the cold-load path, called from initialise without a click.
    assert "openDeepLinkedEvidence()" in init
    assert "hashchange" in init

    body = strip_js_comments(EVIDENCE)
    assert "addEventListener('hashchange'" in body

    read = function_body(body, "readDeepLinkTarget")
    assert "EVIDENCE_DEEP_LINK_PARAMS" in read
    assert "URLSearchParams" in read


def test_a_deep_link_adopts_the_document_the_link_names():
    """A link into a *different* document is still a valid link.

    The reader followed it deliberately, so the document id from the fragment is
    adopted rather than answered with the current session's document id, which
    would open the wrong paper.
    """
    open_link = function_body(EVIDENCE, "openDeepLinkedEvidence")
    assert "state.currentDocumentId = target.documentId" in open_link
    assert "openEvidence(state.currentDocumentId, target.evidenceId, null)" in open_link


def test_a_deep_link_to_an_unloaded_document_degrades_to_a_message():
    """The panel must say so rather than opening empty.

    A fresh tab has no analysed document in state. The deep link supplies the
    document id and the evidence fetch runs; if the server does not know that
    document the panel's error state is what the reader sees — a named message,
    not a blank panel.
    """
    open_evidence = function_body(EVIDENCE, "openEvidence")
    assert "setPanelState('error')" in open_evidence

    # Same correction as the reparse test above: the panel's failure title is
    # named by key so it can be translated, and the assertion is that the key
    # resolves rather than that one language's sentence is present in the file.
    assert "evidence.error_title" in open_evidence
    assert "evidence.error_title" in _catalogue_source(), (
        "the key the evidence error state names does not exist in the catalogue"
    )

    # The no-document-id branch exists too, so a link with no document and no
    # loaded session says so instead of issuing a request to a malformed path.
    assert "evidence.no_document_id" in open_evidence


def test_the_exported_deep_link_and_the_client_agree_on_the_parameters():
    """Both ends of the contract, so a link in an exported file is real.

    ``export.js`` builds ``#evidence=<id>&document=<id>`` from
    ``EVIDENCE_DEEP_LINK_PARAMS``; ``evidence.js`` reads the same two names. The
    names are the shared contract and both sides read from the one constant.
    """
    params = re.search(
        r"export const EVIDENCE_DEEP_LINK_PARAMS = \{(.*?)\};", CONFIG, re.DOTALL
    )
    assert params
    names = dict(re.findall(r"(\w+):\s*'([^']+)'", params.group(1)))
    assert names == {"evidence": "evidence", "document": "document"}

    builder = function_body(EXPORT, "buildCitationDeepLink")
    assert "EVIDENCE_DEEP_LINK_PARAMS.evidence" in builder
    assert "EVIDENCE_DEEP_LINK_PARAMS.document" in builder


# --------------------------------------------------------------------------
# State, reset, and the no-build constraint
# --------------------------------------------------------------------------


def test_a_new_analysis_clears_the_signal_status_and_the_seed():
    """A previous run signal status must not survive into a new document, or it
    attributes one paper coverage to another."""
    reset = function_body(PAPER_LIST, "resetPaperPanels")
    assert "state.currentRecommendationSignals = []" in reset
    assert "state.currentRecommendationDuplicates = []" in reset
    assert "resetRecommendationSignals()" in reset

    result_reset = function_body(SECTIONS, "resetResultView")
    assert "state.currentSeedPaper = null" in result_reset


def test_the_signal_status_survives_a_failed_re_analysis_restore():
    """A restore that brought back the rows but not their signal status would
    leave a list of papers claiming nothing about coverage — the clean-looking
    false reading the four-state rendering exists to prevent."""
    capture = function_body(SECTIONS, "captureWorkspaceState")
    assert "currentRecommendationSignals" in capture
    assert "currentRecommendationDuplicates" in capture
    restore = function_body(SECTIONS, "restoreWorkspaceState")
    assert "renderRecommendationSignals(" in restore
    assert "snapshot.currentRecommendationSignals" in restore


def test_the_new_module_adds_no_framework_and_no_dependency():
    """Vanilla ES modules, loaded directly, with no npm dependency."""
    for name in NEW_MODULES:
        source = (JS_DIR / name).read_text(encoding="utf-8")
        for framework in ("react", "vue", "angular", "svelte", "jquery", "preact", "solid-js"):
            assert framework not in source.lower(), f"{name} references {framework}"
        # Every import is a relative path to another module in this directory.
        for specifier in re.findall(r"from\s+'([^']+)'", source):
            assert specifier.startswith("./"), f"{name} imports {specifier}"
            assert (JS_DIR / specifier[2:]).is_file(), f"{name} imports missing {specifier}"
        assert "export " in source

    # The signal renderer builds text with textContent, never innerHTML — the
    # reason strings are server-authored and must not become markup.
    body = strip_js_comments(SIGNALS)
    assert "innerHTML" not in body, (
        "the signal renderer must build nodes, not interpolate server strings as HTML"
    )
    assert "textContent" in body

    # Still exactly one entry point, and the new module is reachable from the
    # graph the entry point pulls in.
    assert INDEX_HTML.count("<script type=\"module\"") == 1
    assert "recommendationSignals.js" in PAPER_LIST
    assert "recommendationSignals.js" in PAPERS
    assert "recommendationSignals.js" in SECTIONS


def test_the_recommendation_state_fields_exist():
    for field in (
        "currentRecommendationSignals",
        "currentRecommendationDuplicates",
        "currentSeedPaper",
    ):
        assert field in STATE, f"state.js does not declare {field}"
