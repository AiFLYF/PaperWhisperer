"""T3.5: an exported file must say what produced it, and keep the gate intact.

The acceptance criterion is "导出物可追溯生成条件" — someone opening an exported
file six months later can tell exactly what produced it and from which version
of which document. That is a property of *files*, not of functions, so most of
what follows builds a real result, calls a real renderer, and reads the bytes
back: the Markdown as text, the HTML as a string that must not contain a single
unescaped tag, and the DOCX as a ZIP that is unzipped and read as XML.

Two things are being protected, and the second is the more fragile.

**Traceability.** Every field T1.8's
:meth:`~paperwhisperer.analysis.paper_facts.AnalysisRun.as_analysis_run_columns`
names appears in the export, and every field the run did *not* record appears
as ``（未记录）`` rather than being absent. A reader who cannot tell "this run
did not record a cost" from "this build does not write costs down" cannot use
the table to decide anything, and an absent field reads as the second.

**The quotation-mark gate at the point where it is weakest.** In the
application, ``claims.js`` structurally cannot put quotation marks around a
paraphrase. An exported file has no such guard and nobody will ever re-audit
it, so the gate has to be re-established inside the exporter and tested there:
for every renderer, a claim the verifier did not align must come out with no
marks, and the *only* way to reach a mark must be the branch a verified quote
takes. That is asserted structurally — the mark is emitted on one branch of one
function, and no code path reaches it from a paraphrase — because a test
asserting "the output contains no quote marks" would pass for a paraphrase that
happens to have none in its text while a sibling entry in the same file was
wrongly marked.
"""

from __future__ import annotations

import ast
import io
import re
import zipfile

import pytest

from paperwhisperer.analysis import service
from paperwhisperer.analysis.service import (
    EVIDENCE_API_PATH_TEMPLATE,
    EXPORT_UNKNOWN_VALUE,
    build_citation_index,
    build_export_provenance,
    evidence_api_url,
    evidence_deep_link,
    render_docx_report,
    render_html_report,
    render_markdown_report,
    render_provenance_markdown,
)

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG_JS = (PROJECT_ROOT / "templates" / "static" / "js" / "config.js").read_text(
    encoding="utf-8"
)
EXPORT_JS = (PROJECT_ROOT / "templates" / "static" / "js" / "export.js").read_text(
    encoding="utf-8"
)

#: The quotation-mark characters. A straight apostrophe is a Python/JS string
#: delimiter and is not one of these; a straight double quote *is*, because in a
#: Markdown file a bare ``"`` around a sentence reads as a quotation to whoever
#: opens it.
QUOTE_MARKS = "“”‘’「」『』《》〝〞＂\""

#: The subset that are marks when scanning *source* rather than output. The
#: straight double quote is excluded because in Python and JavaScript it is a
#: delimiter: counting it as a mark would make every function that indexes a
#: dictionary look like one that emits quotation marks.
SOURCE_MARKS = "“”‘’「」『』《》〝〞＂"


# --------------------------------------------------------------------------
# A result that actually has both presentations in it
# --------------------------------------------------------------------------


@pytest.fixture
def result() -> dict:
    """A fact-layer result: one field verified, one field not, one quote.

    The asymmetry is the point. A single-paradrase fixture would pass a
    "paraphrases are never marked" assertion vacuously, and a single-quote
    fixture would prove nothing about the paraphrase path. Both have to be in
    the same file for the gate to mean anything.
    """
    return {
        "document_id": "doc_1",
        "session_id": "sess_1",
        "source_filename": "attention-is-all-you-need.pdf",
        "summary": "A summary.",
        "quotes": "## 引用片段\n",
        "mindmap": "## Text Structure\n",
        "elapsed_seconds": 12,
        "char_count": 12345,
        "paper_facts": {
            "research_question": {
                "field_name": "research_question",
                "value": "本文研究如何只用注意力机制做序列到序列的转换。",
                "evidence_ids": ["ev_q", "ev_p"],
                "extraction_status": "extracted",
            },
            "method": {
                "field_name": "method",
                "value": "作者用多头自注意力替换了循环与卷积。",
                "evidence_ids": ["ev_p"],
                "extraction_status": "extracted",
            },
            "limitations": {
                "field_name": "limitations",
                "value": "论文未说明训练时的并行度选择。",
                "evidence_ids": [],
                "extraction_status": "not_found",
            },
            "__not_stated_fields__": ["reproducibility"],
        },
        "fact_extraction": {
            "verified_quotes": [
                {
                    "field_name": "research_question",
                    "value": "本文研究如何只用注意力机制做序列到序列的转换。",
                    "evidence_id": "ev_q",
                    "quote": "The dominant sequence models rely on recurrent or convolutional networks.",
                }
            ],
            "degradations": ["抽取引用了 1 个不存在于本次证据中的 evidence_id，已全部丢弃：ev_zz。"],
        },
        "analysis_run": {
            "run_id": "run_2026_09_30",
            "document_id": "doc_1",
            "created_at": "2026-09-30T10:11:12Z",
            "model": "test-model-1",
            "prompt_version": "facts-v3",
            "evidence_version": "idx-2026-09-28",
            "llm_calls": 4,
            "llm_input_tokens": 8123,
            "llm_output_tokens": 1456,
            "llm_cost": 0.004213,
            "fact_count": 3,
            "grounded_fact_count": 2,
            "not_stated_fields": ["reproducibility"],
            "degradations": ["抽取引用了 1 个不存在于本次证据中的 evidence_id，已全部丢弃：ev_zz。"],
            "retrieval_record_ids": ["rec_1", "rec_2"],
            "legacy_llm_calls": 9,
            "legacy_input_tokens": 40211,
        },
        "parse": {
            "parsed_by": "docling",
            "status": "success",
            "used_cloud": False,
            "cloud_attempted": True,
            "cache_hit": False,
            "fallback_used": False,
            "media_type": "application/pdf",
            "warnings": ["第 12 页没有文本层"],
        },
    }


# --------------------------------------------------------------------------
# The citation index
# --------------------------------------------------------------------------


def test_a_verified_quote_and_a_degraded_claim_are_both_present_and_distinguished(result):
    """Two entries for one field, because the field cites two passages.

    One passage aligned, one did not. The aligned one is quotable; the other
    is our own sentence about the same field and may not be marked. Collapsing
    them into "the field is cited" would be exactly the flattening the gate
    names, so they are separate rows with separate presentations.
    """
    entries = build_citation_index(result)
    by_id = {entry["evidence_id"]: entry for entry in entries}

    assert by_id["ev_q"]["presentation"] == "direct_quote"
    assert by_id["ev_q"]["text"] == (
        "The dominant sequence models rely on recurrent or convolutional networks."
    )
    assert by_id["ev_p"] is not None
    assert by_id["ev_p"]["presentation"] in {"direct_quote", "paraphrase"}
    # The method field cites ev_p, which never aligned, so it is a paraphrase.
    paraphrases = [entry for entry in entries if entry["presentation"] == "paraphrase"]
    assert any("多头自注意力" in entry["text"] for entry in paraphrases)


def test_a_fact_with_no_evidence_is_not_exported_as_a_citation(result):
    """"The paper does not say" is a finding, not a citation.

    An entry with no evidence id would produce a link to nothing, and a link to
    nothing in a traceability table is a link a reader follows and finds a 404
    — which teaches them the table is decorative.
    """
    entries = build_citation_index(result)

    assert all(entry["evidence_id"] for entry in entries)
    assert not any("训练时的并行度" in entry["text"] for entry in entries)


def test_every_citation_carries_a_followable_link(result):
    """Two links per citation: one a reader can click, one a machine can check."""
    for entry in build_citation_index(result):
        assert entry["deep_link"].startswith("/#evidence=")
        assert entry["evidence_id"] in entry["deep_link"]
        assert "document=doc_1" in entry["deep_link"]
        assert entry["api_url"] == f"/v1/documents/doc_1/evidence/{entry['evidence_id']}"


def test_the_exported_path_matches_the_one_the_server_serves():
    """The Python template and the JavaScript constant are one contract.

    There is no shared module across the language boundary, so this comparison
    is the only thing that notices when an export starts advertising a path the
    API does not serve — a link in every exported file, dead, invisible to
    everyone who is not reading this test.
    """
    match = re.search(r"EVIDENCE_API_PATH\s*=\s*'([^']+)'", CONFIG_JS)
    assert match, "config.js no longer declares EVIDENCE_API_PATH"
    js_path = match.group(1)
    py_path = (
        EVIDENCE_API_PATH_TEMPLATE.replace("{document_id}", "{documentId}")
        .replace("{evidence_id}", "{evidenceId}")
    )
    assert py_path == js_path, f"the export writes {py_path}, the client calls {js_path}"


def test_the_deep_link_fragment_matches_the_parameter_names_the_client_reads():
    params = re.search(
        r"export const EVIDENCE_DEEP_LINK_PARAMS = \{(.*?)\};", CONFIG_JS, re.DOTALL
    )
    assert params
    names = dict(re.findall(r"(\w+):\s*'([^']+)'", params.group(1)))
    link = evidence_deep_link("ev_q", "doc_1")
    for key in names.values():
        assert f"{key}=" in link, f"the export writes no {key}= for the client to read"


def test_an_evidence_id_with_punctuation_is_encoded_not_concatenated():
    """An id is used as a path segment, so it is percent-encoded.

    ``urlencode`` in the JavaScript twin and :func:`urllib.parse.quote` here are
    not the same function; a slash in an id would otherwise become a path
    separator and address a *different* evidence, which is the one place an
    unencoded id is a wrong answer rather than a broken one.
    """
    url = evidence_api_url("ev/../other", "doc_1")
    assert "/" not in url.split("/evidence/")[1]
    assert "..%2F" in url


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def test_provenance_carries_every_run_record_column_the_plan_names(result):
    """Model, prompt version, evidence version, cost — and their neighbours.

    The four the requirement names explicitly, plus the rest of
    ``as_analysis_run_columns``, because a half-copied row is a table whose
    missing half is indistinguishable from a value that was never recorded.
    """
    provenance = build_export_provenance(result, "paper.pdf", "0.9.0")

    assert provenance["model"] == "test-model-1"
    assert provenance["prompt_version"] == "facts-v3"
    assert provenance["evidence_version"] == "idx-2026-09-28"
    assert provenance["llm_cost"] == pytest.approx(0.004213)
    assert provenance["llm_calls"] == 4
    assert provenance["llm_input_tokens"] == 8123
    assert provenance["llm_output_tokens"] == 1456
    assert provenance["fact_coverage"] == "2/3"
    assert provenance["run_id"] == "run_2026_09_30"
    assert provenance["legacy_llm_calls"] == 9
    assert provenance["not_stated_fields"] == ["reproducibility"]
    assert provenance["degradations"], "a degraded run must say so in the export"


def test_an_unrecorded_field_says_so_rather_than_disappearing(result):
    """The distinction that makes the table usable at all.

    A field that silently vanishes reads as "not important". A field that reads
    ``（未记录）`` reads as "this run did not write it", which is a statement
    about the run that a reader can act on. Both look identical in JSON; only
    one survives being rendered.
    """
    provenance = build_export_provenance(result, "paper.pdf", "0.9.0")
    table = render_provenance_markdown(provenance)

    # document_version is genuinely unavailable to this module and says so.
    assert EXPORT_UNKNOWN_VALUE in table
    assert "文档版本" in table
    # ...and a field that *is* recorded is not blanked out by that.
    assert "test-model-1" in table
    assert "0.004213" in table
    # A recorded zero and a recorded False are answers, not gaps.
    assert "| 是否经过云端解析 | 否 |" in table


def test_the_document_version_is_reported_as_missing_unless_it_is_supplied(result):
    """The known gap is stated, and a caller can close it.

    The analysis result carries no ``document_version``: it lives on the
    ``documents`` row and this module has no store. Writing a plausible-looking
    number would be a lie in a table whose whole job is to be believed, so the
    default is ``（未记录）`` and the renderers accept the real one.
    """
    bare = build_export_provenance(result, "paper.pdf", "0.9.0")
    assert bare["document_version"] is None

    supplied = build_export_provenance(
        result,
        "paper.pdf",
        "0.9.0",
        document_version={"version": 4, "index_version": "idx-2026-09-28"},
    )
    assert supplied["document_version"]["version"] == 4
    assert "version=4" in render_provenance_markdown(supplied)


def test_a_legacy_run_with_no_fact_layer_still_reports_what_it_can(result):
    """The old six-section path has no run record; the table must not lie."""
    legacy = {key: value for key, value in result.items() if key not in {"paper_facts", "analysis_run", "fact_extraction"}}
    provenance = build_export_provenance(legacy, "paper.pdf", "0.9.0")

    assert provenance["has_fact_layer"] is False
    assert provenance["run_id"] is None
    table = render_provenance_markdown(provenance)
    assert EXPORT_UNKNOWN_VALUE in table
    # The parse provenance is still real and is still reported.
    assert "docling" in table


# --------------------------------------------------------------------------
# The release gate, in every exported format
# --------------------------------------------------------------------------


def test_the_markdown_export_never_quotes_an_unverified_claim(result):
    """Structural, not textual.

    The sentence-level assertion would pass vacuously — a Chinese paraphrase
    may contain no quotation marks anyway. What has to be guaranteed is that
    there is *no code path* from a paraphrase entry to a mark, so the assertion
    is that the mark is emitted on exactly one branch of exactly one function
    and that the branch tests the presentation.
    """
    source = (
        (PROJECT_ROOT / "paperwhisperer" / "analysis" / "service.py")
        .read_text(encoding="utf-8")
    )
    body = _function_body(source, "render_citation_index_markdown")
    marks = [mark for mark in SOURCE_MARKS if mark in body]
    assert marks == ["“", "”"], f"unexpected mark characters in the exporter: {marks}"

    # The mark is emitted by one conditional expression, and the condition that
    # selects it is the presentation. Splitting that expression on its own
    # `` if `` / `` else `` is what makes this a *reachability* claim: the
    # else arm is the paraphrase's text with nothing added to it.
    line = next(row for row in body.splitlines() if "“" in row)
    quote_arm, _, arms = line.partition(" if ")
    condition, _, paraphrase_arm = arms.partition(" else ")
    assert quote_arm.strip().startswith("body =")
    assert condition.strip() == 'entry["presentation"] == "direct_quote"', condition
    assert "“" in quote_arm and "”" in quote_arm
    assert paraphrase_arm.strip() == 'entry["text"]', paraphrase_arm
    # And nowhere else in the function is a mark emitted.
    assert body.count("“") == 1 and body.count("”") == 1


def test_the_html_export_uses_a_blockquote_only_for_a_verified_quote(result):
    """In HTML the element type is the mark.

    A stylesheet can be deleted; a ``<blockquote>`` around a paraphrase cannot
    be produced by accident, because the branch that emits it tests the
    presentation. ``data-presentation`` is on the wrapper as well, so the
    distinction survives a reader that strips all CSS.
    """
    html = render_html_report(result, "paper.pdf", "0.9.0")

    assert '<blockquote class="pw-citation">' in html
    for line in html.splitlines():
        if "<blockquote" in line and "本文研究如何只用注意力" in line:
            pytest.fail("a Chinese paraphrase was rendered inside a blockquote")
    # The paraphrase's own text appears as a paragraph, and the wrapper says so.
    assert 'data-presentation="paraphrase"' in html
    assert "<p>作者用多头自注意力替换了循环与卷积。</p>" in html


def test_the_docx_export_quotes_only_what_the_verifier_aligned(result):
    """Word has no blockquote element, so the label and the style carry it.

    Both are asserted, and the label is asserted on the *paraphrase* too — a
    reader who ignores the indentation still sees 转述 next to the sentence,
    and a reader who greps the text sees it in the file.
    """
    payload = render_docx_report(result, "paper.pdf", "0.9.0")

    assert payload[:2] == b"PK", "a .docx is a ZIP container"
    archive = zipfile.ZipFile(io.BytesIO(payload))
    assert "word/document.xml" in archive.namelist()
    xml = archive.read("word/document.xml").decode("utf-8")

    assert "The dominant sequence models rely on recurrent or convolutional networks." in xml
    assert "作者用多头自注意力替换了循环与卷积。" in xml
    # The label is present for both, and the mark only inside the quote's run.
    assert xml.count("直接引语") >= 1
    assert xml.count("转述（未逐字校验，不加引号）") >= 1
    quote_run = xml[xml.index("The dominant sequence models") - 400 : xml.index("The dominant sequence models") + 200]
    assert "“" in quote_run
    paraphrase_at = xml.index("作者用多头自注意力")
    paraphrase_run = xml[paraphrase_at - 400 : paraphrase_at + 200]
    assert "“" not in paraphrase_run, "a paraphrase acquired a quotation mark in the DOCX"


def test_the_markdown_report_keeps_the_quotation_marks_off_the_paraphrase(result):
    """The end-to-end version: the file a user downloads.

    This is the assertion a reader of the release gate would actually write,
    and it is checkable because the paraphrases in the fixture are Chinese
    sentences with no marks of their own — so any mark on them came from the
    exporter.
    """
    report = render_markdown_report(result, "paper.pdf", "0.9.0")

    for sentence in ("本文研究如何只用注意力机制做序列到序列的转换。", "作者用多头自注意力替换了循环与卷积。"):
        for mark in QUOTE_MARKS:
            assert f"{mark}{sentence}" not in report
            assert f"{sentence}{mark}" not in report
    # The verified quote is marked, because it is the paper's own words.
    assert "“The dominant sequence models rely on recurrent or convolutional networks.”" in report


# --------------------------------------------------------------------------
# The files themselves
# --------------------------------------------------------------------------


def test_the_markdown_report_carries_both_traceability_sections(result):
    report = render_markdown_report(result, "paper.pdf", "0.9.0")

    assert "## 引用与证据链接" in report
    assert "## 生成条件与溯源" in report
    # Every citation is reachable from the report as a link.
    for entry in build_citation_index(result):
        assert entry["api_url"] in report
        assert entry["deep_link"] in report
    # ...and the run record is in it.
    for token in ("test-model-1", "facts-v3", "idx-2026-09-28", "0.004213", "run_2026_09_30"):
        assert token in report, token


def test_a_legacy_report_gains_no_heading_it_cannot_fill(result):
    """No fact layer, no run record: no traceability headings at all.

    A heading over an empty table says "we checked and there was nothing",
    which is a different and stronger claim than "this build did not record
    that". The absence is documented at the bottom of the report instead.
    """
    legacy = {key: value for key, value in result.items() if key not in {"paper_facts", "analysis_run", "fact_extraction"}}
    report = render_markdown_report(legacy, "paper.pdf", "0.9.0")

    assert "## 引用与证据链接" not in report
    assert "## 生成条件与溯源" not in report
    # The report is still a report.
    assert "# PaperWhisperer 分析报告" in report
    assert "## AI 摘要" in report


def test_the_html_export_is_a_standalone_document_with_nothing_unescaped(result):
    """A file that leaves the application, with model text in it.

    Every value is escaped, and the one place markup is emitted is the document
    scaffolding this function writes. An export that could emit a tag from model
    text would be a file that executes in a reader's browser, and the cost of
    not escaping here is a security bug rather than a formatting one.
    """
    hostile = dict(result)
    hostile["summary"] = "A summary with <script>alert(1)</script> and \"quotes\" & <b>bold</b>."
    hostile["paper_facts"] = dict(result["paper_facts"])
    hostile["paper_facts"]["method"] = {
        "field_name": "method",
        "value": "用 <img src=x onerror=alert(1)> 描述方法",
        "evidence_ids": ["ev_p"],
        "extraction_status": "extracted",
    }
    html = render_html_report(hostile, "paper.pdf", "0.9.0")

    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    # The attribute text survives as *text*; what must not survive is a live
    # tag, so the assertion is about the angle bracket, not the attribute name.
    assert "<img" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    # The document is self-contained: no external stylesheet or script.
    assert "<link" not in html
    assert html.count("<style>") == 1


def test_the_docx_export_is_readable_and_carries_the_provenance(result):
    """Unzip it and read the text, rather than trusting that bytes appeared."""
    payload = render_docx_report(result, "paper.pdf", "0.9.0")
    archive = zipfile.ZipFile(io.BytesIO(payload))
    xml = archive.read("word/document.xml").decode("utf-8")

    assert "test-model-1" in xml
    assert "facts-v3" in xml
    assert "idx-2026-09-28" in xml
    assert "0.004213" in xml
    assert "doc_1" in xml
    assert "/v1/documents/doc_1/evidence/ev_q" in xml


def test_the_provenance_rows_are_the_same_in_every_format(result):
    """One table, three renderers.

    The row list is module-level, so all three files are guaranteed to answer
    the same questions in the same order. The test that matters is the one
    below it, which checks the *content*; this checks that there is only one
    list to get out of step.
    """
    keys = [key for _, key in service._EXPORT_PROVENANCE_ROWS]

    for key in ("model", "prompt_version", "evidence_version", "llm_cost", "document_version"):
        assert key in keys
    assert len(keys) == len(set(keys)), "a provenance row is listed twice"

    markdown = render_markdown_report(result, "paper.pdf", "0.9.0")
    html = render_html_report(result, "paper.pdf", "0.9.0")
    for label, _ in service._EXPORT_PROVENANCE_ROWS:
        assert f"| {label} |" in markdown, label
        assert f">{label}</th>" in html, label


def test_the_client_export_keeps_the_same_gate_the_server_does(result):
    """The browser exporter is the other half of T3.5, and it is gated too.

    A user who exports from the page and a user who exports from the server get
    two files for the same analysis, and a paraphrase that acquires quotation
    marks in one of them is still a paraphrase wearing quotation marks in
    somebody's literature review. The mark constants are the ones in
    ``config.js``; the paraphrase branch must name neither of them.
    """
    body = _js_function_body(EXPORT_JS, "formatCitationMarkdown")
    marks = [mark for mark in SOURCE_MARKS if mark in body]
    assert marks == ["“", "”"], f"unexpected mark characters in the client exporter: {marks}"

    # One ternary, and the mark is in the arm the condition selects. Splitting
    # on the ternary's own ``?`` and ``:`` is what makes this a reachability
    # claim rather than a coincidence: the other arm is the entry's own text.
    line = next(row for row in body.splitlines() if "“" in row)
    guard, _, arms = line.partition("?")
    quote_arm, _, paraphrase_arm = arms.partition(":")
    assert "entry.presentation === PRESENTATION_DIRECT_QUOTE" in guard
    assert quote_arm.strip() == "`“${entry.text}”`", quote_arm
    assert paraphrase_arm.strip() == "entry.text;", paraphrase_arm
    assert body.count("“") == 1 and body.count("”") == 1

    html_body = _js_function_body(EXPORT_JS, "buildSessionHtml")
    assert "<blockquote>" in html_body
    assert 'data-presentation="${escapeHtml(entry.presentation)}"' in html_body


def test_the_client_export_writes_the_same_provenance_fields_as_the_server():
    """Two provenance tables in one product is two things to keep in step."""
    body = _js_function_body(EXPORT_JS, "buildProvenanceRows")

    for key in (
        "model",
        "prompt_version",
        "evidence_version",
        "llm_cost",
        "used_cloud",
        "parsed_by",
    ):
        assert key in body, key
    # And it does not invent a document version either.
    assert "document_id" in body
    # The "not recorded" sentinel is written rather than a row omitted. It is
    # reached through `unknownValue()`, which resolves the catalogue entry with
    # `EXPORT_UNKNOWN_VALUE` kept as the floor — so the table still writes the
    # exact value it always has, in the reader's language. The token itself moved
    # out of this function body when the labels were routed through the
    # catalogue; the behaviour did not.
    assert "unknownValue()" in body or "EXPORT_UNKNOWN_VALUE" in body


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------


def _function_body(source: str, name: str) -> str:
    """Return the source of ``def name(...)`` — its signature through its end.

    Extracted with :mod:`ast` rather than by counting braces, which is the
    obvious way and the wrong one. An f-string delimiter is a quote to a
    brace-matcher but a token to a parser, and a scanner that mis-pairs the two
    desynchronises its depth and returns a slice of the middle of a
    *different* function. A structural assertion over such a slice is not
    flaky; it is simply an assertion about code that is not there.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return "".join(source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"function {name}() is missing")


def _js_function_body(source: str, name: str) -> str:
    """The JavaScript twin of :func:`_function_body`, for the client assertions.

    Brace matching is correct here: JavaScript has no f-string whose delimiter
    is a quote of the other kind, and template literals are delimited by a
    backtick the scanner tracks explicitly.
    """
    match = re.search(rf"function\s+{re.escape(name)}\s*\(", source)
    assert match, f"function {name}() is missing"
    return _scan_braces(source, source.index("{", match.end()))


def _scan_braces(source: str, index: int) -> str:
    """The body of a brace-delimited block, tracking string and template state."""
    depth = 0
    start = index
    length = len(source)
    while index < length:
        if source.startswith('"""', index) or source.startswith("'''", index):
            quote = source[index : index + 3]
            index += 3
            while index < length and not source.startswith(quote, index):
                index += 1
            index += 3
            continue
        char = source[index]
        if char in "\"'`":
            quote = char
            index += 1
            while index < length:
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
    raise AssertionError("unbalanced braces")


class TestExportRouteReachable:
    """The renderers existed with no route to them.

    `render_html_report` and `render_docx_report` were complete and tested but
    unreachable: the browser composes its own Markdown, so DOCX had no way out
    of the process. These drive `GET /api/export/{session_id}` through the real
    app, because a renderer with no route is a renderer nobody has run.
    """

    @pytest.fixture
    def stored_session(self, runtime_dirs):
        from paperwhisperer.sessions import store

        token = store.generate_session_token()
        payload = store.build_session_payload(
            session_id="exports",
            source_filename="paper.pdf",
            document_content="body",
            analysis={
                "summary": "概要",
                "quotes": "“可验证的原句”",
                "mindmap": "脑图",
                "mermaid": "graph TD\nA-->B",
            },
            session_token=token,
        )
        store.write_session_payload("exports", payload)
        return token

    def test_markdown_export_is_served_with_the_right_type(self, stored_session):
        from fastapi.testclient import TestClient

        from paperwhisperer.app import app

        response = TestClient(app).get(
            "/api/export/exports", params={"session_token": stored_session}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert "paper" in response.headers["content-disposition"]
        assert response.headers["content-disposition"].endswith('.md"') or ".md" in (
            response.headers["content-disposition"]
        )

    def test_docx_export_returns_a_real_docx(self, stored_session):
        from fastapi.testclient import TestClient

        from paperwhisperer.app import app

        response = TestClient(app).get(
            "/api/export/exports",
            params={"session_token": stored_session, "format": "docx"},
        )
        assert response.status_code == 200
        assert response.content[:2] == b"PK", "a DOCX is a zip; anything else is wrong"
        assert "wordprocessingml" in response.headers["content-type"]

    def test_an_unknown_format_is_refused_with_the_usable_ones_named(self, stored_session):
        from fastapi.testclient import TestClient

        from paperwhisperer.app import app

        response = TestClient(app).get(
            "/api/export/exports",
            params={"session_token": stored_session, "format": "pdf"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "unsupported_export_format"
        for name in ("markdown", "html", "docx"):
            assert name in response.json()["error"], (
                f"the error does not name {name}, so the caller is not told how "
                "to fix it"
            )

    def test_a_wrong_token_is_refused(self, stored_session):
        from fastapi.testclient import TestClient

        from paperwhisperer.app import app

        response = TestClient(app).get(
            "/api/export/exports", params={"session_token": "not-the-token"}
        )
        assert response.status_code in (400, 403)
        assert "概要" not in response.text, (
            "a refused export still leaked the stored analysis"
        )
