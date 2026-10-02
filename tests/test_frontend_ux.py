"""Front-end UX invariants, asserted statically and by execution.

This project has no browser harness and adding one is out of scope, so nothing
here renders anything. What it does do is split along two lines.

**Structural assertions** read the shipped files and check properties that a
stylesheet or a stylesheet failure cannot fake: that a long analysis has a
visible stage, that an answer carries the evidence verdict the server computed,
that an irreversible control names its consequence, that a degraded state is not
dressed as a success.

**Executed assertions** extract the pure functions out of the modules and run
them under ``node`` against real payloads. That is the part worth having. The
single most important defect this file guards — an answer that puts a whole
retrieved passage inside real quotation marks — is a *data* defect, and a grep
for the word ``verified_text`` would pass just as happily on a function that
reads it and throws the value away. So the claim/citation join is not grepped,
it is run, against a payload carrying a 3000-character passage and a
twenty-character verified span, and the assertion is that the twenty-character
one is what reaches the renderer.

Every payload here is the real wire shape, taken from ``AnswerResult.to_dict()``
in ``paperwhisperer/analysis/orchestrator.py`` rather than invented: ``claims``
carry ``evidence_ids`` (plural) and ``verified_text``; ``citations`` carry
``evidence_id`` (singular) and a ``text`` that is the whole retrieved passage.
Getting that distinction wrong is the bug under test, so a test that guessed the
shape would be testing its own guess.

The gap this cannot close is stated at the bottom of the module.
"""

from __future__ import annotations

import json
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JS_DIR = PROJECT_ROOT / "templates" / "static" / "js"

CHAT = (JS_DIR / "chat.js").read_text(encoding="utf-8")
ANALYZE = (JS_DIR / "analyze.js").read_text(encoding="utf-8")
SECTIONS = (JS_DIR / "sections.js").read_text(encoding="utf-8")
PAPERS = (JS_DIR / "papers.js").read_text(encoding="utf-8")
QUEUE = (JS_DIR / "queue.js").read_text(encoding="utf-8")
UPLOAD = (JS_DIR / "upload.js").read_text(encoding="utf-8")

#: Every module, for the invariants that are about a *pair* of files — a writer
#: and the container it writes into — rather than about one function in one.
ALL_JS = {
    path.name: path.read_text(encoding="utf-8")
    for path in sorted(JS_DIR.glob("*.js"))
}
SUGGESTIONS = (JS_DIR / "suggestions.js").read_text(encoding="utf-8")
MAIN = (JS_DIR / "main.js").read_text(encoding="utf-8")
STATE = (JS_DIR / "state.js").read_text(encoding="utf-8")

NODE = "node"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def strip_js_comments(source: str) -> str:
    """Remove ``//`` and ``/* */`` comments, leaving string literals intact.

    Needed wherever an assertion is about what a module *does*: a comment
    discussing quotation marks, or a confidence level, cannot satisfy a
    structural claim about either. String state is tracked so a ``//`` inside a
    string is not mistaken for a comment.
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
            while (
                index < length
                and not (source[index] == "*" and source[index + 1 : index + 2] == "/")
            ):
                index += 1
            index += 2
        elif char in "\"'`":
            quote = char
            out.append(char)
            index += 1
            while index < length:
                if source[index] == "\\":
                    out.append(source[index : index + 2])
                    index += 2
                    continue
                out.append(source[index])
                if source[index] == quote:
                    index += 1
                    break
                index += 1
        else:
            out.append(char)
            index += 1
    return "".join(out)


def function_body(source: str, name: str) -> str:
    """Return the body of ``function name(...)`` by brace matching.

    Brace counting tracks string-literal state, so a ``{`` inside a string or a
    template literal does not close the function early. Comments are expected to
    have been stripped by the caller: this scanner has no notion of them, which
    is exactly why ``strip_js_comments`` exists.
    """
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


def run_node(script: str, tmp_path: Path) -> dict:
    """Execute a JavaScript snippet under node and return its JSON output.

    The only way to assert on this front end's *logic* rather than on the text
    of its source. Fails the test rather than skipping: there is no environment
    in which skipping is honest here, because node is a build-time dependency of
    ``tools/check_js_syntax.mjs`` and is already required to pass.
    """
    script_path = tmp_path / "harness.mjs"
    script_path.write_text(script, encoding="utf-8")
    completed = subprocess.run(
        [NODE, str(script_path)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=PROJECT_ROOT,
    )
    assert completed.returncode == 0, (
        f"node harness failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def function_span(source: str, name: str) -> tuple[int, int]:
    """Return the ``(start, end)`` offsets of one function's full declaration.

    The same brace match as :func:`function_body`, but reporting where the
    declaration ends so the whole ``function name(...) {...}`` can be lifted out
    for execution. Emitting only the braces would make every ``return`` inside
    it a syntax error at module top level.
    """
    match = re.search(rf"function\s+{re.escape(name)}\s*\(", source)
    assert match, f"function {name}() is missing"

    index = source.index("{", match.end())
    depth = 0
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
                return match.start(), index + 1
        index += 1
    raise AssertionError(f"unbalanced braces in {name}()")


def extract_function(source: str, name: str) -> str:
    """Return one function's full declaration, ready to paste into a harness."""
    stripped = strip_js_comments(source)
    start, end = function_span(stripped, name)
    return stripped[start:end]


def extract_pure_function(source: str, name: str, preamble: str = "") -> str:
    """Return one self-contained function, runnable under node.

    Only valid for functions with no DOM or module dependencies. The three used
    here (``buildAnswerClaims``, ``describeAnalysisOutcome``,
    ``expectedSectionCount``) are deliberately pure: the first takes the already
    collected citations as an argument precisely so it can be executed here
    rather than only read.
    """
    return preamble + "\n" + extract_function(source, name) + "\n"


#: The real ``AnswerResult.to_dict()`` shape, with the fields this front end
#: actually reads. The passage is deliberately far longer than the verified
#: span, because the entire defect is that the former is displayed as though it
#: were the latter.
LONG_PASSAGE = "PARAGRAPH " + ("retrieved prose " * 200)
VERIFIED_SPAN = "one sentence the verifier actually aligned"

CITATION_PAYLOAD = {
    "answer": "prose",
    "claims": [
        {
            "text": "The first assertion, in the answer's own words.",
            "quote": "one sentence the verifier actually aligned",
            "evidence_ids": ["ev_1"],
            "presentation": "direct_quote",
            "uses_quotation_marks": True,
            "verified_text": VERIFIED_SPAN,
            "outcome": "aligned",
            "message": "",
            "citation_label": "p3",
            "locator": {"kind": "page", "page": 3},
        },
        {
            "text": "The second assertion, which the verifier could not align.",
            "quote": "a span the model appears to have invented",
            "evidence_ids": ["ev_2"],
            "presentation": "paraphrase",
            "uses_quotation_marks": False,
            "verified_text": "",
            "outcome": "misaligned",
            "message": "引文与原文不一致",
            "citation_label": "p4",
            "locator": {"kind": "page", "page": 4},
        },
    ],
    "citations": [
        {
            "evidence_id": "ev_1",
            "chunk_id": "c1",
            "citation_label": "p3",
            "text": LONG_PASSAGE,
            "presentation": "direct_quote",
            "is_direct_quote": True,
            "heading_path": ["Results"],
            "score": 0.91,
        },
        {
            "evidence_id": "ev_2",
            "chunk_id": "c2",
            "citation_label": "p4",
            "text": "the whole second retrieved passage",
            "presentation": "paraphrase",
            "is_direct_quote": False,
            "heading_path": [],
            "score": 0.62,
        },
    ],
    "confidence": "partially_supported",
    "insufficient_evidence": False,
    "retrieval": {
        "record_id": "rec_1",
        "evidence_count": 2,
        "degradations": ["dense leg abstained"],
    },
    "notes": ["问答使用的文档结构由会话保存的纯文本重建。"],
    "answer_mode": "evidence",
}


# ==========================================================================
#
# The release gate, executed
#
# ==========================================================================


def test_a_direct_quote_renders_the_verified_span_not_the_whole_passage(tmp_path):
    """A quotation mark may only ever wrap text the source was asked to confirm.

    ``citations[].text`` is the whole retrieved passage, up to 3000 characters
    of it. ``claims[].verified_text`` is the single span the verifier aligned
    against the document. The old client handed the renderer the citation list,
    so a passage the model had quoted one sentence of was rendered inside real
    quotation marks under a "verbatim quote" heading — the reader is told the
    paper said all of it, word for word, and the quotation marks are precisely
    the signal they have been trained to read as "checked against the source".

    This runs the join rather than grepping for ``verified_text``, because a
    function can read the field and discard it.
    """
    payload = dict(CITATION_PAYLOAD)
    script = extract_pure_function(
        CHAT, "buildAnswerClaims", "const PRESENTATION_DIRECT_QUOTE = 'direct_quote';"
    )
    script += textwrap.dedent(
        """
        const payload = __PAYLOAD__;
        const out = buildAnswerClaims(payload, payload.citations);
        console.log(JSON.stringify(out));
        """
        ).replace("__PAYLOAD__", json.dumps(payload))

    result = run_node(script, tmp_path)
    direct = [claim for claim in result if claim.get("presentation") == "direct_quote"]

    assert direct, "no claim was rendered as a direct quote"
    assert len(direct) == 1
    assert direct[0]["text"] == VERIFIED_SPAN, (
        "the direct quote must carry the span the verifier aligned, not the "
        f"retrieved passage: got {direct[0]['text'][:80]!r}"
    )
    assert LONG_PASSAGE not in json.dumps(result), (
        "the retrieved passage leaked into the rendered claim; a passage the "
        "model quoted one sentence of must never appear as a verbatim quote"
    )


def test_a_paraphrase_shows_the_assertion_not_the_source_passage(tmp_path):
    """A paraphrase must not be rendered as though the paper said it.

    The inverse error, and the one the previous behaviour produced: a
    paraphrase claim's citation carries the paper's *genuine* retrieved text.
    Rendering that citation under a "did not pass verification" heading labels
    real source text as unverified — which trains the reader to distrust the
    quotes the product got right, and hides the model output that actually
    failed. What failed is the model's quote, not the passage.
    """
    payload = dict(CITATION_PAYLOAD)
    script = extract_pure_function(
        CHAT, "buildAnswerClaims", "const PRESENTATION_DIRECT_QUOTE = 'direct_quote';"
    )
    script += textwrap.dedent(
        """
        const payload = __PAYLOAD__;
        const out = buildAnswerClaims(payload, payload.citations);
        console.log(JSON.stringify(out));
        """
        ).replace("__PAYLOAD__", json.dumps(payload))

    result = run_node(script, tmp_path)
    paraphrases = [claim for claim in result if claim.get("presentation") == "paraphrase"]

    assert len(paraphrases) == 1
    assert paraphrases[0]["text"] == "The second assertion, which the verifier could not align.", (
        "a paraphrase must display the assertion that failed verification, not "
        "the retrieved passage that was fine"
    )
    assert paraphrases[0]["uses_quotation_marks"] is False


def test_every_rendered_claim_keeps_a_path_to_its_evidence(tmp_path):
    """An answer is never rendered without a way to reach the passage.

    A chip is a promise that something can be opened. A claim whose evidence id
    resolves to nothing would render as a sentence attributed to the paper with
    no way to check it — the one outcome a citation must never produce. So a
    claim with no resolvable id is dropped rather than rendered unaddressable.
    """
    payload = json.loads(json.dumps(CITATION_PAYLOAD))
    # A claim citing an id the retriever never returned, alongside a real one.
    payload["claims"].append(
        {
            "text": "An assertion citing a passage that was never retrieved.",
            "quote": "invented",
            "evidence_ids": ["ev_never_returned"],
            "presentation": "direct_quote",
            "verified_text": "text the verifier never saw",
            "outcome": "aligned",
            "citation_label": "p9",
        }
    )
    script = extract_pure_function(
        CHAT, "buildAnswerClaims", "const PRESENTATION_DIRECT_QUOTE = 'direct_quote';"
    )
    script += textwrap.dedent(
        """
        const payload = __PAYLOAD__;
        const out = buildAnswerClaims(payload, payload.citations);
        console.log(JSON.stringify(out));
        """
        ).replace("__PAYLOAD__", json.dumps(payload))

    result = run_node(script, tmp_path)

    assert all(claim.get("evidence_id") for claim in result), (
        "a claim was rendered with no evidence id; it would be a sentence "
        "attributed to the paper that the reader cannot open"
    )
    assert all(
        claim["evidence_id"] != "ev_never_returned" for claim in result
    ), "a claim citing an id retrieval never returned survived the join"
    # The two addressable claims still render. Dropping the bad one must not
    # take the rest of the answer with it.
    assert len(result) == 2


def test_a_direct_quote_with_nothing_verified_degrades_to_a_paraphrase(tmp_path):
    """No verified text means no quotation marks, even if the server said quote.

    Fails toward the unverifiable side, which is the direction
    ``templates/static/js/claims.js`` already commits to. A payload that says
    ``direct_quote`` while carrying no ``verified_text`` is one this client
    cannot source, and the only safe rendering of it is without marks.
    """
    payload = json.loads(json.dumps(CITATION_PAYLOAD))
    payload["claims"] = [payload["claims"][0]]
    payload["claims"][0]["verified_text"] = ""
    script = extract_pure_function(
        CHAT, "buildAnswerClaims", "const PRESENTATION_DIRECT_QUOTE = 'direct_quote';"
    )
    script += textwrap.dedent(
        """
        const payload = __PAYLOAD__;
        const out = buildAnswerClaims(payload, payload.citations);
        console.log(JSON.stringify(out));
        """
        ).replace("__PAYLOAD__", json.dumps(payload))

    result = run_node(script, tmp_path)

    assert result, "the claim disappeared instead of degrading"
    assert result[0]["presentation"] == "paraphrase"
    assert result[0]["uses_quotation_marks"] is False
    assert LONG_PASSAGE not in json.dumps(result)


def test_a_payload_without_claims_still_renders_its_citations(tmp_path):
    """The pre-T1.7 shape must keep rendering rather than going blank.

    A contract move should degrade to the old rendering, not to an empty
    answer. This is the path a client that has never heard of ``claims`` would
    take, and it is also what an ``insufficient_evidence`` response with a
    non-empty citation list looks like.
    """
    payload = {"citations": CITATION_PAYLOAD["citations"]}
    script = extract_pure_function(
        CHAT, "buildAnswerClaims", "const PRESENTATION_DIRECT_QUOTE = 'direct_quote';"
    )
    script += textwrap.dedent(
        """
        const payload = __PAYLOAD__;
        const out = buildAnswerClaims(payload, payload.citations);
        console.log(JSON.stringify(out));
        """
        ).replace("__PAYLOAD__", json.dumps(payload))

    result = run_node(script, tmp_path)

    assert len(result) == 2
    assert all(claim.get("evidence_id") for claim in result)


# ==========================================================================
#
# The answer's support verdict
#
# ==========================================================================


def test_every_answer_states_how_well_it_is_supported():
    """A refusal must not arrive in the same chrome as a verified answer.

    The backend derives ``confidence`` from the claims and explicitly never asks
    the model for it. Rendering the answer without it means an
    ``insufficient_evidence`` response — which carries no claims at all, so the
    claims container is hidden and the verdict would be the *only* signal — looks
    exactly like a supported one: same header, same Copy button, same status
    tone. The one fact a researcher most needs is the one the interface omitted.
    """
    body = function_body(strip_js_comments(CHAT), "renderAnswerVerdict")

    assert "confidence" in body, "the verdict never reads the server's confidence"
    # Rendered from the payload on every settled answer, not only when there are
    # claims to render.
    assert "renderAnswerVerdict(streamingShell.verdict, payload)" in strip_js_comments(CHAT)
    # Machine-readable, so it is assertable without a browser and so the design
    # agent has a hook that does not depend on a colour.
    assert "dataset.confidence" in body
    # An unrecognised value is reported as unrecognised. Defaulting it to
    # success is how a contract change silently upgrades every answer.
    assert "'unknown'" in body
    assert "CONFIDENCE_COPY" in body


def test_all_three_confidence_values_have_distinct_wording():
    """The three states must be separable in words, not only in tone.

    A signal that exists only as a colour or an attribute is erased by a
    stylesheet that fails to load, a high-contrast mode, or a screen reader —
    which is the same argument ``claims.js`` makes for carrying the
    quote/paraphrase distinction four ways over.
    """
    for value in ("supported", "partially_supported", "insufficient_evidence"):
        assert f"{value}:" in CHAT, f"no wording is defined for confidence {value}"

    match = re.search(r"CONFIDENCE_COPY\s*=\s*\{(.*?)\n\}", CHAT, re.DOTALL)
    assert match, "the confidence wording table is missing"
    values = re.findall(r":\s*'([^']*)'", match.group(1))
    assert len(values) == 3
    assert len(set(values)) == 3, "two confidence states share the same wording"


def test_the_answer_records_its_retrieval_and_its_notes():
    """The reader can see why an answer was thin, not only that it was.

    ``retrieval.degradations`` and ``notes`` are the two facts the response
    carries precisely so a reader can tell "the paper does not say this" from
    "this deployment could not search properly" — the retriever's own
    ``empty_reason`` table is written per-cause for the same reason. A user who
    cannot see why an answer was weak cannot know what to ask next.
    """
    verdict = function_body(strip_js_comments(CHAT), "renderAnswerVerdict")
    describe = function_body(strip_js_comments(CHAT), "describeRetrieval")
    # The rendering is split across two functions; both halves matter.
    assert "retrieval" in verdict and "degradations" in describe
    assert "evidence_count" in describe
    assert "notes" in verdict, (
        "notes is where the server says a citation locator came from re-derived "
        "plain text rather than a real parse; dropping it is what makes a "
        "character offset and a page number look like the same kind of thing"
    )


def test_a_refusal_does_not_take_the_success_colour():
    """The status chip must not say success for an unsupported answer.

    ``updateStatus`` takes a tone, and the chip has distinct styling for
    ``success`` and ``error``. Putting success on a refusal is the same defect
    as the missing verdict, reached by the shortest route: the one glance a
    reader takes while scrolling.
    """
    done = re.search(
        r"done:\s*payload\s*=>\s*\{(.*?)\n\s{12}\},", strip_js_comments(CHAT), re.DOTALL
    )
    assert done, "the done handler could not be located"
    body = done.group(1)

    assert "insufficient_evidence" in body
    assert "'error'" in body, "a refusal must not be reported with a success tone"
    # The success tone may only be the fall-through, after both degraded states
    # have been handled.
    assert body.index("insufficient_evidence") < body.rindex("'success'")
    assert body.index("partially_supported") < body.rindex("'success'")


# ==========================================================================
#
# A long-running analysis has a visible, honest stage
#
# ==========================================================================


def test_the_analysis_stage_counts_sections_instead_of_advancing_on_the_first():
    """A 2-minute run must not claim to be rendering after 1 of 6 sections.

    The progress steps are Upload → Analyze → Render → Ready. The old code moved
    to ``render`` on the *first* section event and stayed there, so a run with
    six sections spent almost all of its time labelled "Rendering" while five
    were still being generated. A stage indicator that names a stage the work
    has not reached is a spinner with extra steps.
    """
    body = function_body(strip_js_comments(ANALYZE), "expectedSectionCount")
    # summary, quotes and mindmap are always planned; the rest are optional.
    assert "3 +" in body.replace("(", " (")
    handler = re.search(
        r"section:\s*payload\s*=>\s*\{(.*?)\n\s{12}\},", strip_js_comments(ANALYZE), re.DOTALL
    )
    assert handler, "the section handler could not be located"
    section_body = handler.group(1)
    assert "plannedSections" in section_body and "seenSections" in section_body
    # Render only once every planned section is in; otherwise stay on analyze.
    assert "complete ? 'render' : 'analyze'" in section_body


def test_the_analysis_does_not_claim_to_be_analysing_before_it_is():
    """The ``start`` event precedes the parse, so it cannot mean "analysing".

    ``routes_documents.py`` yields ``start`` and only then iterates
    ``analyze_stream``, whose first action is ``_load_document`` — the parse. For
    a scanned PDF that parse is the long part, and it is the step that may re-run
    OCR. Telling the reader "AI is analyzing structure, citations and research
    signals" at that moment is a promise the server has not made.
    """
    start = re.search(
        r"start:\s*payload\s*=>\s*\{(.*?)\n\s{12}\},", strip_js_comments(ANALYZE), re.DOTALL
    )
    assert start, "the start handler could not be located"
    body = start.group(1)
    assert "'analyze'" not in body, (
        "the start handler advances to the analyze step before the document has "
        "been parsed"
    )
    assert "OCR" in body, (
        "the read step does not say that a scan may re-run OCR, which is the "
        "one thing a two-minute pause is usually about"
    )


def test_a_stream_that_never_settles_is_reported_rather_than_silently_dropped():
    """A stream that ends without ``done`` resolves normally at the transport.

    Without a settled flag the reader gets the spinner taken down, an empty
    workspace and a status chip still claiming sections are streaming — a
    failure that presents as a success. This is the same watchdog the
    robustness layer added for the transport, applied to the missing event.
    """
    stripped = strip_js_comments(ANALYZE)
    assert "let settled = false" in stripped
    # The flag is set by the terminal handler, not by the transport resolving:
    # a resolved stream is exactly the case that must not read as success.
    assert re.search(
        r"done:\s*async payload => \{\s*if \(requestId !== state\.analyzeRequestId\) return;\s*settled = true;",
        stripped,
    ), "the done handler does not mark the run settled"
    assert "if (!settled && requestId === state.analyzeRequestId)" in stripped
    assert "reportInterruptedAnalysis(previousWorkspace, seenSections, plannedSections)" in stripped
    report = function_body(stripped, "reportInterruptedAnalysis")
    assert "notifyFailure" in report, "an interrupted run must be reported, not swallowed"
    assert "restoreWorkspaceState" in report, (
        "an interrupted run is a failed run and must restore the previous "
        "workspace the way a thrown failure does"
    )


def test_a_cancelled_run_is_never_reported_as_a_retryable_failure():
    """Retry offered on a cancel re-runs work the reader deliberately stopped.

    An abort can surface as a clean end-of-stream rather than a thrown
    AbortError, so the interrupted-run report checks the signal directly. This
    is the same invariant ``test_frontend_robustness.py`` asserts structurally
    for the modules it owns; it is repeated here because the path is new.
    """
    report = function_body(strip_js_comments(ANALYZE), "reportInterruptedAnalysis")
    assert "aborted" in report
    assert report.index("aborted") < report.index("notifyFailure")

    chat = strip_js_comments(CHAT)
    guard = re.search(r"if \(!settled[^)]*\) \{", chat)
    assert guard, "the answer stream has no interrupted-run guard"
    condition = guard.group(0)
    assert "aborted" in condition, (
        "an abort can end the answer stream cleanly rather than throwing, so the "
        "guard must test the signal and not rely on the transport"
    )


# ==========================================================================
#
# Work is not lost
#
# ==========================================================================


def test_a_failed_import_restores_the_workspace_the_upload_path_restores():
    """The upload card promises failed attempts keep the last workspace.

    ``templates/index.html`` says a failed analysis leaves the previous
    completed workspace available, and ``analyze()`` honours it with a snapshot.
    The import path did not: it tore the workspace down on the way in and, on
    failure, left the reader with no analysis, no transcript and no reading
    queue, with no way back. The promise was simply false for one of the two
    ways to start an analysis.
    """
    body = function_body(strip_js_comments(PAPERS), "addPaperToAnalysis")
    assert "captureWorkspaceState()" in body, "the import path takes no snapshot"
    assert "restoreWorkspaceState(previousWorkspace)" in body
    # The snapshot is taken before the workspace is torn down, or it captures
    # nothing.
    assert body.index("captureWorkspaceState()") < body.index("resetResultView()")


def test_a_restored_transcript_keeps_working_controls():
    """Restoring the transcript as innerHTML must not restore dead buttons.

    ``restoreWorkspaceState`` writes the transcript back with ``innerHTML``,
    which produces fresh nodes carrying no listeners. The Copy button and the
    follow-up chips used to be bound per node, so after a failed re-analysis
    every restored control was inert while looking completely normal — a
    workspace that appears intact and silently does nothing. Delegation is the
    fix: a data attribute survives serialisation, a listener does not.
    """
    finalize = function_body(strip_js_comments(CHAT), "finalizeStreamingAnswer")
    assert "addEventListener" not in finalize, (
        "a per-node listener cannot survive an innerHTML restore"
    )
    shell = function_body(strip_js_comments(CHAT), "appendStreamingAnswerShell")
    assert "dataset.copyAnswerId" in shell, (
        "the Copy control must be described by a data attribute so a restored "
        "transcript is still functional"
    )
    delegate = function_body(strip_js_comments(CHAT), "handleChatHistoryClick")
    assert "data-copy-answer-id" in delegate
    assert "copyText" in delegate
    assert "initializeChatDelegates" in MAIN, "the delegate is never bound"


def test_a_cancelled_question_comes_back_to_the_composer():
    """Cancelling deletes the half-streamed answer and empties the input.

    The question stays in the transcript, so without this the reader is left
    looking at their own question followed by nothing, which reads as a bug
    rather than as their decision — and the most expensive field in this
    interface to retype is gone as a result of a button they pressed
    deliberately.
    """
    assert "pendingQuestion" in STATE, "state has no record of the in-flight question"
    cancel = function_body(strip_js_comments(CHAT), "cancelAskRequest")
    assert "state.pendingQuestion" in cancel
    assert "questionInput" in cancel, "the question is not returned to the composer"
    # And it is only put back when the box is empty, so it cannot overwrite a
    # question typed since.
    assert "!questionInput.value.trim()" in cancel


def test_a_typed_question_is_never_destroyed_by_a_suggestion_chip():
    """Every chip routes through one helper that used to assign unconditionally.

    A reader who had half typed a question and clicked a starter question to
    compare the two lost their draft, with no warning and no undo. The draft
    wins, and the status line says why the chip did not apply rather than
    reporting a load that did not happen.
    """
    fill = function_body(strip_js_comments(SUGGESTIONS), "fillQuestionInput")
    assert "draft" in fill, "the helper cannot see what is already typed"
    assert re.search(r"if\s*\(\s*draft\s*&&", fill), "the draft is overwritten unconditionally"
    assert "return" in fill, "the early return that protects the draft is missing"
    assert "updateStatus" in fill, "declining to fill is not acknowledged"

    # The starter-question path must not then report a load that was declined.
    starter = function_body(strip_js_comments(SUGGESTIONS), "useStarterQuestion")
    assert "updateStatus" not in starter, (
        "useStarterQuestion overwrites the explanation with a statement that is "
        "false when the draft was protected"
    )


# ==========================================================================
#
# Irreversible actions name their consequence
#
# ==========================================================================


def test_clearing_the_reading_queue_states_what_it_will_destroy():
    """One stray click currently destroys a dozen deliberate ones.

    "Clear" sits beside the queue heading, carries no count, says nothing about
    what is lost, and writes straight through to the server — so one accidental
    press destroys a reading trail that took a dozen "Save" presses to build,
    with no undo. A ``window.confirm`` would be worse: unstyleable, announced
    inconsistently, and modal for a decision the label could have made obvious.
    So the first press arms and names, and the window closes on its own.
    """
    body = function_body(strip_js_comments(QUEUE), "clearReadingQueue")
    assert "clearArmed" in body, "the first press does not arm the control"
    # The armed state must return before anything is destroyed.
    first_press = body.index("if (!clearArmed)")
    destruction = body.index("state.currentReadingQueue = []")
    assert first_press < destruction
    # The consequence is named in words, with the count, on the control itself.
    assert "press again to confirm" in body
    assert "removePaperFromQueue" not in body.split("state.currentReadingQueue = []")[0]

    render = function_body(strip_js_comments(QUEUE), "renderReadingQueue")
    assert "aria-label" in render, (
        "the consequence must be in the accessible name; a hover-only title is "
        "not a name a screen reader can rely on"
    )
    assert "cannot be undone" in render
    # A re-render disarms, or an armed button outlives the count it named.
    assert "resetClearArm()" in render


def test_reparsing_names_what_it_costs_before_it_happens():
    """The one irreversible action that already warns must keep warning.

    A reparse discards the current parse and re-runs OCR, which can be slow and
    can introduce recognition errors. ``parseStatus.js`` is not this module's,
    so this asserts only that the affordance is reachable and still carries its
    warning — a regression guard, not a re-implementation.

    The warning *sentences* are catalogue keys rather than constants imported
    out of ``config.js``, because ``config.js`` holds only the Chinese half of
    each pair; a warning read from the constant is Chinese in both locales,
    which is the mixed-language defect WALK-DEFECT-3 is about. So this asserts
    that the module looks the copy up through ``t()`` and that the catalogue
    actually carries it, rather than asserting on a string literal.
    """
    parse_status = (JS_DIR / "parseStatus.js").read_text(encoding="utf-8")
    i18n = (JS_DIR / "i18n.js").read_text(encoding="utf-8")
    assert "t('reparse.force_ocr_warning')" in parse_status
    assert "t('reparse.cloud_warning')" in parse_status
    assert "'reparse.force_ocr_warning'" in i18n
    assert "'reparse.cloud_warning'" in i18n
    assert 'id="reparseSubmitBtn"' in (PROJECT_ROOT / "templates" / "index.html").read_text(
        encoding="utf-8"
    )


# ==========================================================================
#
# `cloud_outcome`: the backend's verdict, which the card used to discard
#
# ==========================================================================
#
# The backend promotes a machine-readable cloud outcome to the client and the
# client never read it: `ParseResult.cloud_outcome` →
# `ParseOutcome.cloud_outcome` → `build_parse_provenance`
# (`analysis/orchestrator.py`). Rendering `status`, `used_cloud`,
# `cloud_attempted` and `error_message` without it made four different cloud
# failures one sentence, so a reader whose cloud parse ran out of quota was told
# exactly what a reader whose network failed was told, and neither could act.
#
# These are executed rather than grepped, and deliberately so. The property that
# matters is a *data* property — that six values produce six different sentences,
# and that the retry policy attached to each one is the policy its code already
# carried. A grep for the word "quota" would pass just as happily on a map that
# sent every outcome to the same entry.


#: The six values `ingestion/adapters/mineru_cloud.py` records. Taken from the
#: adapter rather than invented here, because the whole point is that the client
#: and the server agree on the vocabulary.
CLOUD_OUTCOMES = ["success", "partial", "quota_exhausted", "timed_out", "blocked", "failed"]

#: The outcomes that mean "this document has no cloud text behind it". The other
#: two produced something and are not failures.
CLOUD_FAILURE_OUTCOMES = ["quota_exhausted", "timed_out", "blocked", "failed"]


def run_cloud_harness(body: str, tmp_path: Path) -> dict:
    """Run JS against the real ``parseStatus``/``config``/``i18n`` modules.

    Unlike :func:`extract_pure_function`, this imports the shipped files rather
    than lifting one function out of them. That matters here because the decision
    under test is a *join* across three modules — the outcome maps to a code in
    ``config.js``, the code carries a kind, the kind carries the retry policy,
    and the catalogue carries the words. Extracting ``describeCloudOutcome``
    alone would test a function with no config table and no catalogue, which is
    the version of this test that passes trivially.
    """
    script = (
        "import { describeCloudOutcome } from %r;\n"
        "import { setLocale, t, errorPresentation } from %r;\n"
        "import { ERROR_PRESENTATIONS, FAILURE_KINDS, NOTICE_POLICY } from %r;\n"
        "const result = {};\n%s\n"
        "console.log(JSON.stringify(result));\n"
        % (
            str(JS_DIR / "parseStatus.js"),
            str(JS_DIR / "i18n.js"),
            str(JS_DIR / "config.js"),
            body,
        )
    )
    return run_node(script, tmp_path)


def cloud_outcome_record(locale: str, payload: dict, tmp_path: Path) -> dict:
    """Everything the interface derives from one ``parse`` payload, in one locale."""
    body = textwrap.dedent(
        """
        setLocale(%r);
        const d = describeCloudOutcome(%s);
        const kind = d.code ? (ERROR_PRESENTATIONS[d.code] || {}).kind : null;
        const policy = kind ? NOTICE_POLICY[kind] : null;
        result.record = {
            raw: d.raw,
            known: d.known,
            code: d.code,
            reports: d.reports,
            hasCloudLeg: d.hasCloudLeg,
            tone: d.tone,
            sentence: d.hasCloudLeg ? t(d.sentenceKey) : '',
            kind: kind,
            actionable: policy ? policy.actionable : null,
            message: d.reports ? errorPresentation(d.code).message : '',
            hint: d.reports ? errorPresentation(d.code).hint : '',
            title: d.reports ? errorPresentation(d.code).title : ''
        };
        """
        % (locale, json.dumps(payload))
    )
    return run_cloud_harness(body, tmp_path)["record"]


@pytest.mark.parametrize("outcome", CLOUD_OUTCOMES)
@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_every_cloud_outcome_says_something_of_its_own(tmp_path, outcome, locale):
    """Six outcomes, six different sentences — in both languages.

    The reader-facing sentences are what makes this worth asserting rather than
    asserting six map entries. Two outcomes sharing a sentence is the original
    defect in a smaller size, and it would pass a test that only counted the
    keys.
    """
    record = cloud_outcome_record(
        locale, {"status": "success", "cloud_attempted": True, "cloud_outcome": outcome}, tmp_path
    )
    assert record["raw"] == outcome
    assert record["known"] is True, f"{outcome} is one of the adapter's own values and must be recognised"
    assert record["sentence"], f"{outcome} renders no sentence in {locale}"

    others = [
        cloud_outcome_record(
            other,
            {"status": "success", "cloud_attempted": True, "cloud_outcome": other},
            tmp_path,
        )["sentence"]
        for other in CLOUD_OUTCOMES
        if other != outcome
    ]
    assert record["sentence"] not in others, (
        f"{locale}: cloud outcome {outcome!r} shares its sentence with another outcome, so "
        f"the reader cannot tell those two cloud failures apart"
    )


@pytest.mark.parametrize("outcome", CLOUD_FAILURE_OUTCOMES)
def test_a_cloud_failure_reports_the_reason_it_has_a_code_for(tmp_path, outcome):
    """The four "no cloud text" outcomes each resolve to a code the runtime knows.

    A code is the only handle the failure surface branches on, so an outcome
    that reached the card without one would have nothing to render — the exact
    "a title and no sentence" hole the taxonomy exists to prevent.
    """
    record = cloud_outcome_record(
        "en", {"status": "success", "cloud_attempted": True, "cloud_outcome": outcome}, tmp_path
    )
    assert record["reports"] is True, f"{outcome} means no cloud text and must be reported"
    assert record["code"], f"{outcome} is reported but has no code, so nothing can present it"
    assert record["kind"] in {"retryable", "terminal"}, f"{outcome} has kind {record['kind']!r}"
    assert record["message"], f"{outcome} has a code but no reader-facing sentence"
    assert record["title"] != record["message"]


@pytest.mark.parametrize("outcome", ["success", "partial"])
def test_a_cloud_run_that_produced_text_is_not_a_failure(tmp_path, outcome):
    """``success`` and ``partial`` are facts, not failures.

    Both mean the cloud leg *returned something*. Raising a notice for either
    would train the reader to dismiss the notice surface, and would put a
    failure above a document that has its text. They are still shown, because a
    fact the reader cannot see is the same silence as a failure they cannot
    distinguish — the difference is that they raise no notice.
    """
    record = cloud_outcome_record(
        "en", {"status": "success", "cloud_attempted": True, "cloud_outcome": outcome}, tmp_path
    )
    assert record["reports"] is False, f"{outcome} produced text and must not be reported as a failure"
    assert record["code"] == "", f"{outcome} must not mint a failure code"
    assert record["kind"] is None
    assert record["sentence"], f"{outcome} is still a fact the reader has to be shown"


def test_a_spent_quota_is_terminal_and_a_stalled_cloud_is_retryable(tmp_path):
    """The retry policy, decided by the code each outcome maps to.

    This is the assertion that matters most in this section, and it is a join,
    not a pair of constants: each outcome's ``code`` is looked up in
    ``ERROR_PRESENTATIONS`` and the resulting kind is looked up in
    ``NOTICE_POLICY``, exactly as ``runtime.notifyFailure`` does. So it fails if
    a code is renamed, if a kind is flipped, or if the actionable flag moves —
    not only if someone edits a literal.
    """
    records = {
        outcome: cloud_outcome_record(
            "en",
            {"status": "success", "cloud_attempted": True, "cloud_outcome": outcome},
            tmp_path,
        )
        for outcome in CLOUD_FAILURE_OUTCOMES
    }

    # A spent quota is the canonical trap: a retry is free to offer and
    # guaranteed to be refused, so the button is worse than nothing. It must be
    # terminal — and therefore offered no button at all, not a disabled one.
    quota = records["quota_exhausted"]
    assert quota["kind"] == "terminal", "a spent quota is unchanged by asking again"
    assert quota["actionable"] is False, "a terminal failure must not be offered a retry"

    # A blocked document is a refusal by policy, so the identical property.
    blocked = records["blocked"]
    assert blocked["kind"] == "terminal", "policy will refuse the identical request again"
    assert blocked["actionable"] is False

    # A timeout is the opposite: the work was not lost, it was merely not
    # finished in time, so asking again is a real chance at a real result.
    timed_out = records["timed_out"]
    assert timed_out["kind"] == "retryable", (
        "timed_out means the work may still finish, so it must be retryable"
    )
    assert timed_out["actionable"] is True, "a retryable failure must be offered a retry"

    # And the generic one follows the backend's own treatment of an
    # unclassified cloud failure: retryable, like the four `mineru_*`
    # transport/task failures already in the table.
    assert records["failed"]["kind"] == "retryable"


def test_the_four_cloud_failures_do_not_overlap_in_what_they_tell_the_reader(tmp_path):
    """No two of them say the same thing, in either language.

    A reader told to top up a quota when they were actually blocked will not
    trust the next message either, so overlap between two failure sentences is a
    trust defect and not a style preference. This asserts the *sentences* are
    pairwise distinct, not merely that the keys are — which is the same reason
    the sentence test above exists.
    """
    for locale in ("zh-CN", "en"):
        titles = {}
        messages = {}
        for outcome in CLOUD_FAILURE_OUTCOMES:
            record = cloud_outcome_record(
                locale,
                {"status": "success", "cloud_attempted": True, "cloud_outcome": outcome},
                tmp_path,
            )
            titles[outcome] = record["title"]
            messages[outcome] = record["message"]
        for outcome, title in titles.items():
            for other, other_title in titles.items():
                if other == outcome:
                    continue
                assert title != other_title, (
                    f"{locale}: {outcome} and {other} share a failure title, so the reader "
                    f"is given one instruction for two different conditions"
                )
                assert title not in messages[other], (
                    f"{locale}: {outcome}'s title is the body of {other}'s message — the two "
                    f"conditions have been given the same words"
                )


def test_a_payload_with_no_cloud_outcome_says_something_rather_than_nothing(tmp_path):
    """The degradation paths, and each one is a different sentence.

    Three absences, three correct answers, and getting any of them wrong is the
    defect this test exists for:

    * **a local parse that never left the machine** — no row, because there is
      no cloud leg to describe. Announcing one would put "the server reported an
      outcome this version does not know" above every healthy local document.
    * **a cloud leg on the record whose document produced nothing** — the
      generic message, not silence. This is the billed-but-unrecorded case and
      the reason an empty outcome degrades to `failed` rather than to blank.
    * **a value this build cannot name** — says it is unrecognised, because that
      is a different fact from "the server sent nothing" and it needs a different
      response from whoever operates this.
    """
    local_only = cloud_outcome_record(
        "en", {"status": "success", "touched_network": False}, tmp_path
    )
    assert local_only["hasCloudLeg"] is False
    assert local_only["sentence"] == "", (
        "a parse that never left the machine has no cloud outcome to report, and "
        "inventing a row for it would be a false statement about the document"
    )
    assert local_only["reports"] is False

    unrecorded = cloud_outcome_record(
        "en", {"status": "failed", "cloud_attempted": True}, tmp_path
    )
    assert unrecorded["hasCloudLeg"] is True, "a cloud leg on the record must produce a row"
    assert unrecorded["reports"] is True
    assert unrecorded["sentence"], "a charged cloud call with no outcome must not degrade to a blank"
    assert unrecorded["message"], "and it must carry a sentence, not just a title"

    unrecognised = cloud_outcome_record(
        "en",
        {"status": "success", "cloud_attempted": True, "cloud_outcome": "quantum_flux"},
        tmp_path,
    )
    assert unrecognised["known"] is False
    assert unrecognised["reports"] is True, "an outcome we cannot name is a failure, not a success"
    assert unrecognised["sentence"] != unrecorded["sentence"], (
        "'the server sent nothing' and 'the server sent something we cannot read' are "
        "different facts and must not be given the same sentence"
    )

    # And a local parse that *failed* with no cloud attempt is still not a cloud
    # problem. Announcing a cloud failure over a corrupt PDF is the loudest
    # possible way to be wrong about the reader's own document.
    local_failure = cloud_outcome_record(
        "en", {"status": "failed", "touched_network": False}, tmp_path
    )
    assert local_failure["hasCloudLeg"] is False
    assert local_failure["reports"] is False


def test_a_successful_cloud_parse_does_not_announce_a_broken_one(tmp_path):
    """Every *successful* cloud parse arrives with an empty ``cloud_outcome``.

    This is the trap that makes the rule above necessary rather than cautious:
    ``_cloud_outcome_of`` in ``ingestion/orchestrator.py`` reads the outcome off a
    *failed* attempt, so the wire value for a healthy cloud run is `""`. A client
    that treated an empty outcome as a failure would put a cloud-failure notice
    above every successful cloud parse — and a notice that appears on success is
    a notice the reader learns to ignore.
    """
    for outcome in ("success", "partial"):
        record = cloud_outcome_record(
            "en",
            {"status": "success", "used_cloud": True, "cloud_outcome": outcome},
            tmp_path,
        )
        assert record["reports"] is False
    # And the shape a *healthy* successful cloud run actually has: used_cloud,
    # no outcome value at all.
    healthy = cloud_outcome_record(
        "en", {"status": "success", "used_cloud": True, "cloud_attempted": True}, tmp_path
    )
    assert healthy["hasCloudLeg"] is True
    assert healthy["reports"] is False, (
        "a successful cloud parse with an empty outcome must not be reported as a failure"
    )


def test_the_cloud_outcome_goes_through_the_shared_failure_surface():
    """The card must not grow its own second failure path.

    ``runtime.js`` owns failures — four kinds and ``notifyFailure``. If
    ``parseStatus.js`` rendered the outcome's sentence and hint itself, the retry
    button would have to be re-implemented beside it, and the two would drift:
    this is the mistake that let the outcome be discarded in the first place.
    So the card names the outcome as a *fact* and the explanation goes through
    the one function that decides what a failure looks like.
    """
    source = strip_js_comments((JS_DIR / "parseStatus.js").read_text(encoding="utf-8"))

    reporter = function_body(source, "reportCloudOutcome")
    assert "notifyFailure" in reporter, (
        "the cloud outcome's explanation must go through runtime.notifyFailure"
    )
    # It supplies the code and lets the taxonomy decide; it does not decide.
    assert "outcome.code" in reporter, "notifyFailure must be handed the resolved code"
    assert not re.search(r"kind:\s*['\"]", reporter), (
        "parseStatus.js must not name a failure kind; ERROR_PRESENTATIONS owns that"
    )
    # The retry re-issues the same request, and is offered for every reported
    # outcome — the policy, not this branch, decides which ones actually show it.
    assert "onRetry" in reporter
    assert "submitReparse" in reporter

    # Re-rendering must replace the notice, not stack another copy of it.
    assert "dismissCloudOutcomeNotice()" in reporter
    reset = function_body(source, "resetParseStatus")
    assert "dismissCloudOutcomeNotice()" in reset, (
        "resetting the card must retire its notice, or a quota failure outlives "
        "the document that caused it"
    )

    # And the card's own row is a fact row, not a second notice.
    row = function_body(source, "appendCloudOutcomeRow")
    assert "appendFact(" in row, "the outcome belongs in the card's facts"
    assert "notifyFailure" not in row, "the card must not raise its own failure"
    assert "t(" in row, "the card's sentence must come from the catalogue"


def test_the_card_resets_the_outcome_attributes_it_publishes():
    """The card publishes its outcome on ``data-*``; those attributes are state.

    A card re-rendered for a new document that did not clear them would leave the
    *previous* document's quota verdict on screen as a dataset attribute — and
    the first reader of that attribute is an operator deciding whether to top up
    a quota that was never actually spent.
    """
    source = strip_js_comments((JS_DIR / "parseStatus.js").read_text(encoding="utf-8"))
    body = function_body(source, "renderParseStatus")
    for attribute in ("cloudOutcome", "cloudOutcomeCode"):
        assert f"container.dataset.{attribute} =" in body, (
            f"renderParseStatus must set data-{attribute} on every render, or a stale "
            f"value from the previous document survives"
        )


def test_the_card_never_says_the_same_thing_twice():
    """One render, one copy of each sentence.

    A duplicated block here is not a cosmetic regression: the card is the
    provenance record, and a fallback notice that appears twice reads as "two
    things went wrong" where one did. It is also invisible to a test that only
    checks that a branch *exists* — which is how a copy-paste during the
    ``cloud_outcome`` wiring survived long enough to be noticed by hand.
    """
    source = strip_js_comments((JS_DIR / "parseStatus.js").read_text(encoding="utf-8"))
    body = function_body(source, "renderParseStatus")
    for call in re.findall(r"appendNotice\((.*?)\);", body, re.S):
        literal = re.findall(r"'([^']{12,})'", call)
        if not literal:
            continue
        assert body.count(call.strip()) == 1, (
            f"the card renders the same notice twice in one pass: {literal[0]!r}"
        )
    # And each notice is reached by exactly one condition.
    conditions = re.findall(r"if \((parse\.[a-z_]+)[^)]*\) \{\s*appendNotice", body)
    assert len(conditions) == len(set(conditions)), (
        f"two different branches render the same notice: {conditions}"
    )
    assert body.count("reportCloudOutcome(cloudOutcome)") == 1, (
        "the cloud outcome is reported once per render; a second call would raise a "
        "second notice for the same document"
    )


# ==========================================================================
#
# A degraded run is not a successful one
#
# ==========================================================================


def test_a_partial_analysis_is_not_reported_as_a_success(tmp_path):
    """``analysis_status`` is computed by the server and was read by nobody.

    The workspace already renders a "Section needs review" card per failed
    section, so the detail was not lost — the headline was. The status chip said
    "Analysis ready for follow-up questions" in a success tone on runs where a
    third of the analysis had failed, and a degraded state wearing the success
    colour is worse than no signal because it removes the reader's reason to
    look at the sections.
    """
    script = extract_pure_function(SECTIONS, "describeAnalysisOutcome")

    cases = {
        "complete": ("complete", [], "success"),
        "partial": ("partial", [], "error"),
        "failed": (
            "partial",
            ["概览", "引用片段"],
            "error",
        ),
        "absent": ("", [], "idle"),
    }
    for name, (quality, failed, expected_tone) in cases.items():
        script += textwrap.dedent(
            f"""
            console.log(JSON.stringify(describeAnalysisOutcome({{
                analysis_status: {{ quality: "{quality}", failed_sections: {json.dumps(failed)} }}
            }})));
            """
        )
        outcome = run_node(script, tmp_path)
        assert outcome["tone"] == expected_tone, (
            f"{name} analysis reported tone {outcome['tone']!r}, expected {expected_tone!r}"
        )
        if failed:
            for label in failed:
                assert label in outcome["text"], (
                    "a partial run must name the parts that did not generate, so "
                    "the reader can find the cards for them"
                )

    # An absent analysis_status is not a success either: the server sends it on
    # every result, so its absence means the contract moved.
    script += textwrap.dedent(
        """
        console.log(JSON.stringify(describeAnalysisOutcome({})));
        """
    )
    assert run_node(script, tmp_path)["tone"] == "idle"


def test_both_paths_into_an_analysis_judge_the_run_the_same_way():
    """An uploaded paper and an imported one are the same analysis.

    Two copies of "did this finish" would be two chances for one of them to keep
    saying success on a partial run, which is exactly the defect above.
    """
    assert "describeAnalysisOutcome" in ANALYZE
    assert "describeAnalysisOutcome" in PAPERS
    for module in (ANALYZE, PAPERS):
        import_line = [line for line in module.splitlines() if "describeAnalysisOutcome" in line and "import" in line]
        assert import_line, "the shared verdict is reimplemented locally instead of imported"
    # One definition only.
    assert len(re.findall(r"function describeAnalysisOutcome", SECTIONS)) == 1


def test_the_import_path_reports_a_stage_and_a_clock():
    """An import is a download, a parse and six sections. It showed a spinner.

    The upload path calls ``beginProgressClock`` and names a step; the import
    path did neither, so two runs of the same analysis reported themselves
    completely differently and the longer one was the silent one.
    """
    body = function_body(strip_js_comments(PAPERS), "addPaperToAnalysis")
    assert "beginProgressClock()" in body
    assert "endProgressClock(" in body
    assert "updateAnalysisProgress(" in body


def test_the_import_index_resolves_against_the_list_that_was_clicked():
    """The handler read a fixed list while the button declared its own.

    ``data-paper-list`` is set on every paper control, and the add branch
    ignored it, indexing the search results regardless. Today the control is
    only rendered on the search list, so the bug is latent — but the day it is
    rendered on the recommendation list, clicking "Add" there imports the search
    result at the same index. A user who clicks "analyse this recommended paper"
    and gets a different paper analysed has been told something false about
    their own click.
    """
    handler = function_body(strip_js_comments(PAPERS), "handlePaperResultClick")
    assert "addPaperToAnalysisByIndex(elementId, index)" in handler
    resolve = function_body(strip_js_comments(PAPERS), "addPaperToAnalysisByIndex")
    assert "getPaperListByElementId(elementId)" in resolve
    assert "state.currentPaperSearchResults[index]" not in resolve, (
        "the import path still resolves against a hard-coded list"
    )


# ==========================================================================
#
# The question history
#
# ==========================================================================


def test_the_server_is_asked_for_history_rather_than_the_client_supplying_it():
    """The transcript is the record of the session, and the server keeps it.

    ``routes_qa.py`` reads ``qa_history`` from the session file and
    ``record_qa_turn`` appends to it, so follow-up questions are answered with
    the earlier turns in context without the client re-sending anything. The
    invariant worth protecting is that this stays true: a client that started
    posting its own history would be a second, divergent transcript.
    """
    routes_qa = (PROJECT_ROOT / "paperwhisperer" / "api" / "routes_qa.py").read_text(
        encoding="utf-8"
    )
    assert "session_payload.get(\"qa_history\"" in routes_qa
    assert "record_qa_turn" in routes_qa

    body = function_body(strip_js_comments(CHAT), "askQuestion")
    assert "history" not in body, (
        "the client is posting its own history; the session is the transcript "
        "of record and two copies would diverge"
    )
    # What the server needs to continue the conversation is sent, though.
    assert "session_id" in body and "session_token" in body


def test_a_settled_turn_is_recorded_with_what_the_server_vouched_for():
    """The transcript entry carries the verdict, not only the prose.

    An answer stored as a bare string is an answer whose support cannot be
    reconstructed later — by the export, by a later turn's history, or by a
    reader six months from now. The fields the response already carries are kept
    on the turn so nothing has to be guessed back into existence.
    """
    done = re.search(
        r"done:\s*payload\s*=>\s*\{(.*?)\n\s{12}\},", strip_js_comments(CHAT), re.DOTALL
    )
    assert done
    body = done.group(1)
    turn = re.search(r"state\.currentChatTurns\.push\(\{(.*?)\}\);", body, re.DOTALL)
    assert turn, "the settled turn is not recorded"
    fields = turn.group(1)
    for field in ("question", "answer", "answer_mode", "confidence", "citations", "retrieval", "notes"):
        assert field in fields, f"the transcript entry does not carry {field}"


def test_a_turn_is_filed_under_the_mode_it_was_asked_in():
    """The mode was read again when the stream settled, not when it was sent.

    A reader who switches mode while an answer is streaming got a turn filed
    under a mode they did not ask in — in the transcript, and in the exported
    report, which is the record of what was asked.
    """
    body = function_body(strip_js_comments(CHAT), "askQuestion")
    assert "const answerMode = state.currentAnswerMode" in body
    done = re.search(
        r"done:\s*payload\s*=>\s*\{(.*?)\n\s{12}\},", strip_js_comments(CHAT), re.DOTALL
    )
    assert done and "answer_mode: answerMode" in done.group(1)
    assert "answer_mode: state.currentAnswerMode" not in strip_js_comments(CHAT)


# ==========================================================================
#
# No new machinery
#
# ==========================================================================


def test_the_ux_changes_stayed_within_the_existing_stack():
    """Vanilla ES modules, no bundler, no framework, no npm."""
    for name in ("chat.js", "analyze.js", "sections.js", "queue.js", "papers.js", "suggestions.js"):
        source = (JS_DIR / name).read_text(encoding="utf-8")
        for framework in ("react", "vue", "angular", "svelte", "jquery", "preact", "solid-js"):
            assert framework not in source.lower(), f"{name} references {framework}"
        for specifier in re.findall(r"from\s+'([^']+)'", source):
            assert specifier.startswith("./"), f"{name} imports {specifier}"
            assert (JS_DIR / specifier[2:]).is_file(), f"{name} imports missing {specifier}"

    # Still exactly one entry point.
    assert (PROJECT_ROOT / "templates" / "index.html").read_text(
        encoding="utf-8"
    ).count('<script type="module"') == 1


def test_every_ux_verdict_element_is_hidden_until_it_has_something_to_say():
    """An always-present empty container is an absence, not a signal.

    The panel's own rule — a degraded state must be a visible element rather
    than a blank — cuts both ways. A container that is present and empty on every
    answer trains the reader to stop looking at it, which is the same failure as
    never having built it.
    """
    shell = function_body(strip_js_comments(CHAT), "appendStreamingAnswerShell")
    assert "answer-verdict-host" in shell
    assert re.search(r"verdict\.hidden = true", shell), (
        "the verdict container starts visible on an answer that has no verdict yet"
    )
    render = function_body(strip_js_comments(CHAT), "renderAnswerVerdict")
    assert "container.hidden = false" in render


# --------------------------------------------------------------------------
# What this cannot verify
# --------------------------------------------------------------------------
#
# There is no browser in this repository, so none of the following is checked
# here and all of it is a manual pass:
#
# * that the verdict, the stage counter and the armed Clear button *look*
#   right — the classes used here (``.answer-verdict``, ``.answer-verdict-label``,
#   ``.answer-verdict-detail``, ``.answer-verdict-host``) have no stylesheet
#   rules yet, because the CSS belongs to another agent. Until one exists they
#   render as unstyled text, which is legible but plain.
# * that the live colour distinction of the status chip survives a theme switch.
# * that the two-press Clear is reachable and discoverable by keyboard, and that
#   the re-arm window feels right in use.
# * that a real ``AnswerResult`` from a real parse produces the same join
#   observed here: the payloads are the true wire shape, but they were built by
#   hand rather than captured from a running server.
# * that the delegated Copy button still behaves under a rapid double click.
#
# What *is* checked here is the logic: the claim/citation join, the confidence
# wording, the analysis outcome verdict and the section count are all executed
# under node against real payload shapes. The rest is structural, and structural
# assertions on a file whose whole purpose is to survive a missing stylesheet
# are the right kind of assertion for this front end.


# ==========================================================================
#
# A dropped analysis re-attaches, and never charges the reader twice
#
# ==========================================================================
#
# The release gate for this section is one decision, and it is executed rather
# than grepped: given what the server says happened to the job, is a retry
# offered? A "重试" on a job that *completed* is a second full parse and a
# second bill for a document that has already been analysed, and the interface
# is the only place that can prevent it.


def _run_pure(module_source, name, expression, tmp_path, preamble=""):
    """Run one pure function from a module and return the JSON it produces."""
    script = extract_pure_function(module_source, name, preamble)
    script += textwrap.dedent(
        """
        console.log(JSON.stringify(%s));
        """
    ) % expression
    return run_node(script, tmp_path)


def test_a_completed_job_is_never_offered_a_retry(tmp_path):
    """The double charge, closed by one boolean and asserted on the real states.

    ``POST /api/analyze/stream`` submits the parse and the model calls to
    ``infra/jobs`` and then *watches* the row, so a browser that gives up has not
    thrown the work away — the document is very likely still being analysed on
    the server. The reader's only previous way forward was "重试", which
    re-uploads the same file and analyses it again.

    The feed is consulted first, and a job that finished is reported as finished
    with *no* retry attached. Retrying there is not recovering the run, it is
    paying for it twice.
    """
    cases = {
        "completed": ("completed", False),
        "partial": ("partial", False),
        "failed": ("failed", False),
        "cancelled": ("failed", False),
        # A non-terminal state is the one case where a retry is honest: the job
        # is alive, we could not learn its fate, and re-running is a choice the
        # reader is entitled to make with the button in front of them.
        "analyzing": ("unknown", True),
        "parsing": ("unknown", True),
        "": ("unknown", True),
    }
    for job_state, (expected_outcome, expected_retryable) in cases.items():
        outcome = _run_pure(
            ANALYZE, "describeJobOutcome",
            f"describeJobOutcome('{job_state}', 'job_1')", tmp_path,
        )
        assert outcome["outcome"] == expected_outcome, (
            f"a job that ended as {job_state!r} was reported as "
            f"{outcome['outcome']!r}, expected {expected_outcome!r}"
        )
        assert outcome["retryable"] is expected_retryable, (
            f"a job that ended as {job_state!r} "
            + (
                "was offered a retry, which re-analyses an already-analysed "
                "document"
                if not expected_retryable
                else "was not offered a retry, but the outcome was unknown so a "
                "rerun is the reader's call"
            )
        )
        if expected_outcome == "unknown":
            # Deliberately empty. "Unknown" means the job could not be reached,
            # and the caller falls back to the ordinary interrupted-run report
            # rather than showing this — so a sentence here would be a claim
            # about the run that nothing supports.
            assert not outcome["text"], (
                "the unknown outcome carries a message, which would be shown in "
                "place of the real report on a run whose fate is unknown"
            )
        else:
            assert outcome["text"], (
                "a decided outcome must say something; an empty string is not "
                "an explanation"
            )


def test_a_recovered_job_names_its_own_id(tmp_path):
    """A report the reader can act on has to say which run it is about.

    The job is the only durable handle on work that is still going somewhere the
    reader cannot see, so an outcome that does not name it leaves them unable to
    ask what became of their document.
    """
    outcome = _run_pure(ANALYZE, "describeJobOutcome", "describeJobOutcome('completed', 'job_42')", tmp_path)
    assert "job_42" in outcome["text"], (
        "a completed job's report does not name the job, so the reader cannot "
        "look up what happened to it"
    )


def test_the_job_state_machine_drives_the_visible_stage(tmp_path):
    """The stage track must show the stage the server is actually in.

    The server's spine is finer than the interface's: ``queued``, ``parsing``,
    ``quality_check``, ``fallback``, ``normalizing`` and ``indexing`` are all
    still "getting the document into the product", and only ``analyzing`` is the
    model work. Mapping them is what makes the re-attach useful rather than
    decorative — it is the only path on which the server's own ``message`` (the
    one that says a scan had to be re-OCRed) reaches the reader.
    """
    expected = {
        "queued": "upload",
        "parsing": "upload",
        "quality_check": "upload",
        "fallback": "upload",
        "normalizing": "upload",
        "indexing": "upload",
        "analyzing": "analyze",
        "completed": "ready",
        "partial": "ready",
        "failed": "ready",
        "cancelled": "ready",
    }
    for job_state, step in expected.items():
        actual = _run_pure(
            ANALYZE, "jobStageStep", f"jobStageStep('{job_state}')", tmp_path
        )
        assert actual == step, (
            f"a job in state {job_state!r} was shown as stage {actual!r}, "
            f"expected {step!r}"
        )


def test_the_analysis_stream_keeps_the_server_job_id():
    """Re-attach needs an id, and the id arrives on the ``start`` event.

    Asserted on the handler body rather than anywhere in the file, because the
    whole value of the re-attach rests on this one assignment and an id
    captured too late — after the sections begin — is an id the reader can no
    longer use.
    """
    start = re.search(
        r"start:\s*payload\s*=>\s*\{(.*?)\n\s{12}\},", strip_js_comments(ANALYZE), re.DOTALL
    )
    assert start, "the start handler could not be located"
    body = start.group(1)
    assert "payload.job_id" in body, (
        "the job id on the start event is not read, so a dropped connection has "
        "nothing to re-attach to and the reader's only option is a re-run"
    )
    assert "currentAnalysisJobId" in body, (
        "the job id is read but not kept, so it cannot be re-attached to"
    )


def test_a_dropped_stream_asks_the_job_before_reporting_a_failure():
    """A dropped connection is not a failed analysis, so it must not be reported
    as one until the server has been asked.

    Both shapes of the failure reach the catch: a thrown
    ``client_stream_interrupted``/``client_timeout`` from the transport's stall
    watchdog, and a stream that resolves cleanly with no terminal ``done``. The
    first is the common one — a dead network on a two-minute parse — and it is
    the one that used to produce a "重试" button for a document the server was
    still working on.
    """
    body = function_body(strip_js_comments(ANALYZE), "reportInterruptedAnalysis")
    assert "reattachAnalysisJob" in body, (
        "an interrupted stream is reported as a failure without asking the job "
        "whether it survived"
    )
    # The re-attach has to come first: anything that reports, restores or
    # raises a notice before it is unreachable.
    assert body.index("reattachAnalysisJob") < body.index("notifyFailure"), (
        "the run is reported as failed before the job is consulted"
    )
    assert "describeJobOutcome" in body, (
        "the job's state is fetched but not turned into an outcome for the reader"
    )

    # And the thrown path gets the same treatment, since that is the one the
    # stall watchdog actually produces.
    analyze_body = function_body(strip_js_comments(ANALYZE), "analyze")
    assert "reattachAnalysisJob" in analyze_body, (
        "a thrown stream interruption is not re-attached to the surviving job"
    )


def test_a_recovered_job_does_not_restore_a_workspace_it_never_received():
    """Nothing may be rendered as though a finished job delivered an analysis.

    The job's result payload is parked process-locally and taken once
    (``routes_jobs.take_job_result``), so a job that completed while this client
    was disconnected generally cannot be handed back. The honest pair of
    statements — the analysis completed, and the workspace could not be
    rebuilt — is the only one available, and a blank workspace presented as a
    finished run would be a claim about the reader's paper that nothing
    supports.
    """
    body = function_body(strip_js_comments(ANALYZE), "restoreAfterRecoveredJob")
    assert "resetResultView" in body, (
        "a recovered job leaves whatever was on screen rather than clearing it"
    )
    # The retry is attached from the outcome's own flag, not unconditionally.
    assert "recovered.retryable" in body, (
        "the retry affordance is not decided by whether the job succeeded"
    )
    guard = body.index("if (recovered.retryable)")
    assert guard < body.index("notifyFailure"), (
        "the failure notice is raised before the retry decision is consulted"
    )


def test_cancelling_does_not_claim_the_server_work_stopped():
    """The cancel button stops watching, and the copy has to say so.

    ``cancelAnalyzeRequest`` aborts this page's connection and nothing else. The
    route's own disconnect path returns without cancelling the job, so the
    worker keeps parsing and keeps calling the model. "Analysis canceled" was a
    claim the product could not support: the reader believes the work stopped and
    it has not.
    """
    body = function_body(strip_js_comments(ANALYZE), "cancelAnalyzeRequest")
    assert "Analysis canceled" not in body, (
        "the cancel path still claims the analysis was cancelled, but the "
        "server-side job was not cancelled"
    )
    assert "currentAnalysisJobId" in body, (
        "the cancel path discards the job id, so the reader is not told which "
        "run may still be going"
    )
    # The id must be read before it is cleared, or the message cannot name it.
    assert body.index("currentAnalysisJobId") < body.rindex("currentAnalysisJobId = ''"), (
        "the job id is cleared before the message that has to name it is written"
    )


# ==========================================================================
#
# A finished run does not leave a status line behind
#
# ==========================================================================


def test_a_completed_run_retires_its_own_pending_notice():
    """The "in progress" line must not outlive the work it describes.

    ``notifyPending`` is answered by ``endFlight({ succeeded: true })``, which
    retires notices *by operation*. An unscoped notice carries an operation of
    ``''``, which matches nothing — so "分析进行中" and "回答生成中" stayed on
    screen after the run finished, next to a completed workspace, with nothing
    wrong to look at and nothing to dismiss. A reader arriving ten seconds
    later is told work is in progress that finished ten seconds ago.

    This is asserted in both directions: the scope has to be passed *and* the
    success has to close the flight, because either one alone leaves the notice
    up.
    """
    for module, operation, pending_call in (
        ("analyze.js", "analyze", "分析进行中"),
        ("chat.js", "ask", "回答生成中"),
    ):
        source = ALL_JS[module]
        stripped = strip_js_comments(source)
        assert pending_call in stripped, f"{module} no longer raises its pending notice"
        pending = re.search(r"notifyPending\((.*?)\);", stripped, re.DOTALL)
        assert pending, f"{module} does not call notifyPending"
        scope = f"operation: {operation.upper()}_OPERATION" if module == "analyze.js" else "operation: ASK_OPERATION"
        assert scope in pending.group(1), (
            f"{module} raises its pending notice without the operation its "
            "flight closes under, so the success cannot retire it"
        )
        # And the closing endFlight is a success, scoped to the same operation.
        end = re.search(r"endFlight\(flight, \{(.*?)\}\);", stripped, re.DOTALL)
        assert end, f"{module} never closes the flight"
        assert "succeeded:" in end.group(1), (
            f"{module} closes the flight without saying whether the run "
            "succeeded, so its pending notice is never retired"
        )


def test_a_cancelled_run_retires_its_pending_notice_by_hand():
    """A cancellation reaches none of the success paths, so it must close its own.

    Cancelling bumps the request token, so the run's own ``finally`` sees a
    stale token and skips its settled branch — and a cancel is not a success, so
    the by-operation retirement does not apply either. Nothing else reaches the
    notice, and it stays above a transcript the reader has just emptied.
    """
    assert "pendingNoticeIds" in STATE, (
        "state has no record of the pending notice each long operation raised"
    )
    for module, key, operation in (
        ("analyze.js", "analyze", "ANALYZE_OPERATION"),
        ("chat.js", "ask", "ASK_OPERATION"),
    ):
        cancel_name = "cancelAnalyzeRequest" if module == "analyze.js" else "cancelAskRequest"
        cancel = function_body(strip_js_comments(ALL_JS[module]), cancel_name)
        assert f"pendingNoticeIds.{key}" in cancel, (
            f"{module}'s cancel path does not retire the pending notice, so the "
            "in-progress line outlives the run the reader stopped"
        )
        assert "dismissNotice" in cancel, (
            f"{module}'s cancel path records the notice but never dismisses it"
        )


def test_a_success_does_not_delete_another_operations_notices():
    """One operation finishing must not silence another operation's problem.

    ``clearNotices()`` was the only thing retiring the pending notice on the
    answer path, and it retires *every* notice: asking one question silently
    deleted a paper-search failure, a reparse failure or a queue-save failure
    that had nothing to do with it. A real problem vanishing because the reader
    did something else is a worse defect than the notice it replaced.
    """
    chat = strip_js_comments(CHAT)
    assert "clearNotices" not in chat, (
        "the answer path clears every notice, so a successful question removes "
        "failures belonging to other operations"
    )
    assert re.search(r"endFlight\(flight, \{[^}]*succeeded:", chat, re.DOTALL), (
        "the answer path does not retire its own notices on success either"
    )


def test_every_spinner_clearing_path_also_closes_the_run_indicator():
    """The run indicator has to be hidden wherever the spinner is taken down.

    Once the indicator is hidden until a run starts, the three places that take
    the spinner down have to put it back, or a *second* analysis starts with no
    indicator at all — the mirror image of the bug this fixes, and harder to
    notice, because the reader has already seen the run start once.
    """
    for module, function_name in (
        ("analyze.js", "analyze"),
        ("analyze.js", "cancelAnalyzeRequest"),
        ("papers.js", "addPaperToAnalysis"),
    ):
        body = function_body(strip_js_comments(ALL_JS[module]), function_name)
        assert "hideRunIndicator" in body, (
            f"{module}.{function_name} takes the spinner down but leaves the run "
            "indicator visible, so it describes a run that is not happening"
        )
    for module, function_name in (
        ("analyze.js", "analyze"),
        ("papers.js", "addPaperToAnalysis"),
    ):
        body = function_body(strip_js_comments(ALL_JS[module]), function_name)
        assert "showRunIndicator" in body, (
            f"{module}.{function_name} starts a run without showing the run "
            "indicator, so the reader loses the stage and the elapsed clock"
        )
    main = strip_js_comments(MAIN)
    assert re.search(r"function initialize\(\)[\s\S]*?hideRunIndicator\(\)", main), (
        "the run indicator is not hidden on load, so the interface opens with a "
        "spinner and an active Upload stage describing work nobody started"
    )


# ==========================================================================
#
# A degraded state stays distinguishable from a success
#
# ==========================================================================


def test_an_interrupted_answer_is_marked_in_the_place_the_reader_can_see():
    """The "this is not a verified answer" line must be visible.

    It was appended into ``.answer-verdict-host``, which
    ``appendStreamingAnswerShell`` creates *hidden* because a settled answer
    fills it in ``renderAnswerVerdict`` — and the interrupted path never reaches
    that. The single sentence warning the reader that this text is unverified was
    therefore rendered inside a container the stylesheet sets to
    ``display: none``: a partial answer, with its Copy button enabled, looking
    exactly like a finished one.

    The existing test suite caught the *copy* of this case in spirit — an answer
    must never be presented as more verified than it is — and this is the same
    gate applied to the case where the answer never settled at all.
    """
    body = function_body(strip_js_comments(CHAT), "askQuestion")
    guard = re.search(
        r"if \(!settled[^{]*?\{([\s\S]*?)\n        \}", body
    )
    assert guard, "the interrupted-answer path could not be located"
    incomplete = guard.group(1)
    assert "verdictHost.hidden = false" in incomplete, (
        "the incomplete marker is written into a container that stays hidden, so "
        "the reader is never told the answer is unverified"
    )
    assert "querySelector('.answer-verdict-host') ||" not in incomplete, (
        "the marker is still appended into whatever host it finds, which is the "
        "hidden one on this path"
    )
    # And the escape hatch is withdrawn: copying is how text leaves the
    # interface, and a partial answer pasted out is indistinguishable from a
    # verified one.
    assert "setControlDisabled(streamingShell.button, true)" in incomplete, (
        "the Copy control is left enabled on an answer that never settled, so "
        "unverified text can leave the interface as though it were verified"
    )
    assert "streamingShell.answerDiv.dataset.incomplete = 'true'" in incomplete, (
        "the incomplete answer is not marked as such on the element itself"
    )


# ==========================================================================
#
# A cancellation is the reader's decision, not a failure
#
# ==========================================================================


@pytest.mark.parametrize(
    "function_name",
    ["searchPapers", "recommendPapers"],
)
def test_a_cancelled_search_is_not_reported_as_a_failure(function_name):
    """Both spellings of a cancellation, because the transport rewrites it.

    ``apiRequest`` turns an abort into an ``ApiError`` whose ``name`` is
    ``'ApiError'`` and whose ``code`` is ``'client_cancelled'``. Testing only
    ``name`` meant a cancelled search ran the failure path: it cleared the
    results the cancel handler had just re-rendered, replaced them with "Search
    results will appear here", and raised a failure notice with a retry button —
    for work the reader deliberately stopped, and at the cost of the previous
    results.
    """
    body = function_body(strip_js_comments(PAPERS), function_name)
    catch = re.search(r"\} catch \(error\) \{(.*)", body, re.DOTALL)
    assert catch, f"{function_name} has no catch block"
    guard = re.search(r"if \(([^\n]*)\)\s*\{\s*return;\s*\}", catch.group(1))
    assert guard, f"{function_name} has no cancellation guard in its catch"
    condition = guard.group(1)
    assert "AbortError" in condition, f"{function_name} does not recognise a raw abort"
    assert "client_cancelled" in condition, (
        f"{function_name} does not recognise the ApiError the transport actually "
        "throws for a cancellation, so cancelling is reported as a failure"
    )
    # And the guard has to come before anything destructive.
    assert catch.group(1).index(condition) < catch.group(1).index("notifyFailure"), (
        f"{function_name} raises a failure notice before the cancellation guard"
    )


def test_the_drop_zone_does_not_open_two_file_dialogs_on_one_keypress():
    """The file input is a child of the drop zone, so its keys bubble.

    ``event.target !== fileInput`` guarded only the click handler. A keypress
    while the input itself had focus therefore bubbled to the zone handler,
    which called ``fileInput.click()`` a second time on top of the input's own
    default — two file dialogs, the second stacked over the first, from one
    keypress.
    """
    body = function_body(strip_js_comments(UPLOAD), "initializeUploadDropZone")
    keydown = re.search(r"addEventListener\('keydown', event => \{(.*?)\n    \}\);", body, re.DOTALL)
    assert keydown, "the drop zone's keydown handler could not be located"
    assert "event.target === fileInput" in keydown.group(1), (
        "a keypress on the file input still reaches the drop zone, which opens "
        "a second file dialog on top of the input's own"
    )
    assert keydown.group(1).index("event.target === fileInput") < keydown.group(1).index("fileInput.click()"), (
        "the guard is after the click, so it prevents nothing"
    )


# ==========================================================================
#
# The default language is Chinese, and the modules must not undo that
#
# ==========================================================================


def test_no_module_overwrites_a_translated_container_with_an_english_literal():
    """The i18n layer translates the static markup; these writers undid it.

    ``initializeI18n`` runs first, so ``data-i18n`` containers hold Chinese on
    first paint. Each of the functions below then wrote an English literal
    straight over the top — from the config tables, which hold the *English*
    locale's values, or from a hard-coded sentence. The result was a
    Chinese-default interface whose own description of itself, its reading
    queue, its answer-mode hint and all five section placeholders were in the
    wrong language from the moment it loaded.

    Each of these keys already existed in the catalogue; the tables in
    ``config.js`` are what i18n.js reads them *from*, so reading the table
    directly and reading the catalogue are the same sentence in two languages,
    and only one of them follows the reader's switch.
    """
    cases = [
        ("upload.js", "updateFileMeta", "upload.file_meta_empty"),
        ("queue.js", "renderReadingQueue", "queue.meta"),
        ("suggestions.js", "setAnswerMode", "mode.detail."),
        ("sections.js", "setSectionContent", "section.empty."),
    ]
    for module, function_name, key in cases:
        body = function_body(strip_js_comments(ALL_JS[module]), function_name)
        assert "t(" in body, (
            f"{module}.{function_name} writes into a translated container with no "
            f"catalogue lookup, so '{key}' is replaced by English on first paint"
        )
        assert key in body, (
            f"{module}.{function_name} does not use the existing catalogue key "
            f"'{key}'"
        )
    # The keys must actually be shipped, or this is asserting a translation
    # that does not exist. Three are declared on the markup the reader first
    # sees; `section.empty.*` is catalogue-only, because those containers are
    # filled by JS and carry no static text.
    template = (PROJECT_ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    catalogue = (JS_DIR / "i18n.js").read_text(encoding="utf-8")
    for key in ("upload.file_meta_empty", "queue.meta", "mode.detail.evidence"):
        assert key in template, f"{key} is not declared in index.html"
    for key in ("upload.file_meta_empty", "queue.meta", "mode.detail.evidence", "section.empty.summary"):
        assert f"'{key}'" in catalogue, f"{key} is not in the catalogue"
