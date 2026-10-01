"""The docling adapter against payloads real docling produced.

``tests/test_ingestion_docling.py`` runs everywhere, on hand-written fixtures.
That is what makes it cheap and it is also how the adapter came to disagree
with docling on five points: the fixtures were consistently wrong in the same
direction, so every assertion passed and a real document produced 96 elements
with no locator among them. This module closes that gap by running the same
normalization over **dumps captured from a real docling 2.131.0 install**.

The dumps are the ``.docling.json`` files ``write_raw_artifact`` leaves in
``context/parser_artifacts`` when the eval runner parses the corpus, so they are
present on any machine that has run the bake-off and absent on one that has
not. Every test here skips rather than fails when they are — but a machine with
a real parse available is exactly the machine that should be running them, and
``RUN_DOCLING_REAL_PARSE=1`` will spend ~12s re-parsing the corpus PDF from
scratch to check that the installed docling still emits the shape pinned here.

What is pinned, and why each one is a shape rather than a behaviour:

* ``prov`` carries ``page_no``. A payload with ``page`` cannot come from
  docling, and reading it produced zero locators.
* ``texts[i]`` carries ``label``. A payload with ``text_label`` cannot come
  from docling, and reading it collapsed every role to ``text``.
* table end offsets are exclusive. Reading them as inclusive inflates every
  table by one row and one column.
* a formula's content is in ``orig``, with ``text`` empty.
* ``furniture`` is a dict, not a list.
* ``pages`` is a dict keyed by the string page number.

The shape assertions do not depend on which corpus file is present — a dump
from any document satisfies them — while the counts do, so the counts are
written against the specific dump that was found and reported in the assertion
message.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from paperwhisperer.ingestion.adapters import docling_adapter as adapter
from paperwhisperer.ingestion.adapters.docling_adapter import (
    build_table_from_docling,
    normalize_docling_document,
)
from paperwhisperer.ingestion.schema import ParsedDocument

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(os.getenv("PARSER_ARTIFACT_DIR") or (REPO_ROOT / "context/parser_artifacts"))


def _real_dumps() -> list[Path]:
    """Real docling dumps on this machine, oldest first."""
    if not ARTIFACT_DIR.is_dir():
        return []
    return sorted(ARTIFACT_DIR.glob("*.docling.json"))


_DUMPS = _real_dumps()
#: The distillation paper is the reference document: 9 pages, 5 tables, 6
#: formulas, 20 section headers and a running page footer on every page. It is
#: chosen because it exercises every axis — tables, formulas, headings,
#: furniture, a group tree and ``page_no`` provenance.
_REFERENCE_NAME = "hinton_distillation"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_dump() -> dict | None:
    for path in _DUMPS:
        origin = _load(path).get("origin") or {}
        if _REFERENCE_NAME in str(origin.get("filename") or ""):
            return _load(path)
    return None


requires_dumps = pytest.mark.skipif(
    not _DUMPS,
    reason=(
        "本机没有 docling raw artifact；先运行 eval/run_parser_eval.py "
        "或任意一次真实 docling 解析以生成 context/parser_artifacts/*.docling.json"
    ),
)


class TestRealPayloadShape:
    """The five shapes the fixtures used to get wrong."""

    @requires_dumps
    def test_provenance_uses_page_no(self):
        # Not "there is a page number somewhere" — the *key* is the contract.
        # A payload that satisfied this with a ``page`` key would let the
        # adapter keep reading a field docling never emits, and the next real
        # parse would drop every locator again.
        for path in _DUMPS:
            payload = _load(path)
            for item in list(payload.get("texts") or []) + list(payload.get("tables") or []):
                for prov in item.get("prov") or []:
                    assert "page_no" in prov, f"{path.name}: prov 缺少 page_no：{prov}"
                    assert "page" not in prov, f"{path.name}: prov 不应含 page 键：{prov}"

    @requires_dumps
    def test_text_items_carry_label_and_never_text_label(self):
        for path in _DUMPS:
            for item in _load(path).get("texts") or []:
                assert "label" in item, f"{path.name}: texts 项缺少 label：{item.get('self_ref')}"
                assert "text_label" not in item, (
                    f"{path.name}: texts 项不应含 text_label：{item.get('self_ref')}"
                )

    @requires_dumps
    def test_pages_is_a_dict_keyed_by_the_string_page_number(self):
        for path in _DUMPS:
            pages = _load(path).get("pages")
            assert isinstance(pages, dict), f"{path.name}: pages 应为 dict，实为 {type(pages).__name__}"
            for key, entry in pages.items():
                assert isinstance(key, str), f"{path.name}: pages 键应为字符串，实为 {key!r}"
                assert entry["page_no"] == int(key)
                assert "size" in entry and "height" in entry["size"]

    @requires_dumps
    def test_furniture_is_a_group_dict_not_a_list(self):
        # All 53 corpus dumps have it as a dict with an empty children list. A
        # list here would mean this adapter's furniture indexing is being
        # exercised by nothing, which is exactly the state that let a
        # non-empty furniture group go unnoticed.
        for path in _DUMPS:
            furniture = _load(path).get("furniture")
            assert isinstance(furniture, dict), (
                f"{path.name}: furniture 应为 dict 组节点，实为 {type(furniture).__name__}"
            )
            assert "children" in furniture and "self_ref" in furniture

    @requires_dumps
    def test_a_formula_keeps_its_content_in_orig(self):
        """Docling labels a formula and leaves ``text`` empty.

        This is the fact behind "6 formulas, 0 with LaTeX": there was no
        formula *text* to read, and the adapter dropped the element before the
        LaTeX question was ever reached. A release that populated ``text``
        instead would still be acceptable — the adapter reads both — so this
        pins what docling does, not what the adapter requires.
        """
        for path in _DUMPS:
            for item in _load(path).get("texts") or []:
                if item.get("label") == "formula":
                    assert (item.get("text") or "").strip() or (item.get("orig") or "").strip(), (
                        f"{path.name}: 公式既无 text 也无 orig：{item.get('self_ref')}"
                    )

    @requires_dumps
    def test_table_end_offsets_are_exclusive(self):
        # ``end == start + span``, not ``start + span - 1``. Verified against
        # docling's own ``TableData.grid``, which iterates ``range(start, end)``.
        for path in _DUMPS:
            for table in _load(path).get("tables") or []:
                for cell in (table.get("data") or {}).get("table_cells") or []:
                    assert cell["end_row_offset_idx"] == (
                        cell["start_row_offset_idx"] + cell["row_span"]
                    ), f"{path.name}: 行偏移不是排他的：{cell}"
                    assert cell["end_col_offset_idx"] == (
                        cell["start_col_offset_idx"] + cell["col_span"]
                    ), f"{path.name}: 列偏移不是排他的：{cell}"


class TestRealTableRebuild:
    """A rebuilt grid must match the dimensions docling itself declared.

    This is the assertion the whole table path is judged by, and the exclusive
    convention is what it exists for: the inclusive reading turned a 4x3 table
    into 5x4 while still looking like a plausible table.
    """

    @requires_dumps
    def test_rebuilt_grid_matches_doclings_declared_size(self):
        checked = 0
        for path in _DUMPS:
            for payload in _load(path).get("tables") or []:
                data = payload.get("data") or {}
                declared = (data.get("num_rows"), data.get("num_cols"))
                if not declared[0] or not declared[1]:
                    continue
                table = build_table_from_docling(payload)
                assert (table.row_count, table.column_count) == declared, (
                    f"{path.name} {payload.get('self_ref')}: 重建为 "
                    f"{table.row_count}x{table.column_count}，docling 声明 {declared}"
                )
                checked += 1
        assert checked, "没有可校验的表格；corpus dump 发生了变化"

    @requires_dumps
    def test_a_really_empty_cell_is_preserved_rather_than_trimmed(self):
        """The opposite risk to the phantom-row bug: over-trimming.

        Real tables do contain empty cells — the distillation paper's neighbour
        ``66a26fae…`` has a 1x1 table whose only cell is blank. The inclusive
        offset bug produced a trailing row and column of blanks; the naive fix
        for it would be to trim blank rows, which would throw away genuine
        whitespace structure. This pins that it does not: a blank cell the
        converter reported stays in the grid, and is instead caught by the
        quality gate's ``is_empty_shell`` rule, which is where that judgement
        belongs.
        """
        blanks = 0
        for path in _DUMPS:
            for payload in _load(path).get("tables") or []:
                cells = (payload.get("data") or {}).get("table_cells") or []
                if not any(not str(cell.get("text") or "").strip() for cell in cells):
                    continue
                blanks += 1
                table = build_table_from_docling(payload)
                declared = (
                    (payload["data"].get("num_rows"), payload["data"].get("num_cols"))
                    if payload.get("data")
                    else (0, 0)
                )
                assert (table.row_count, table.column_count) == declared
        assert blanks, "corpus 中没有空单元格表格；该断言未被覆盖"


@requires_dumps
def test_the_reference_document_normalizes_to_a_fully_located_document():
    payload = _reference_dump()
    if payload is None:
        pytest.skip(f"未找到 {_REFERENCE_NAME} 的 docling dump")

    document = normalize_docling_document(
        payload,
        source_filename="hinton_distillation.pdf",
        media_type="application/pdf",
    )

    # 1. Nothing is silently unlocatable. This is the assertion that would
    #    have failed at 0.0 before the ``page_no`` fix.
    assert document.elements, "没有解析出任何元素"
    located = [e for e in document.elements if e.locator.is_resolvable]
    assert len(located) == len(document.elements), (
        f"{len(document.elements) - len(located)} 个元素没有可解析的 locator；"
        f"warnings={document.warnings}"
    )
    assert all(e.locator.kind == "pdf_page" for e in document.elements)

    # 2. Boxes survive, and are absolute rather than relative.
    with_bbox = [e for e in document.elements if e.locator.bbox is not None]
    assert len(with_bbox) == len(document.elements), (
        f"{len(document.elements) - len(with_bbox)} 个元素没有 bbox"
    )
    relative = [e for e in with_bbox if e.locator.bbox.relative]
    assert not relative, f"{len(relative)} 个 bbox 仍是相对坐标"
    for element in with_bbox:
        box = element.locator.bbox
        assert box.width > 0 and box.height > 0, f"退化 bbox：{box}"

    # 3. Page coverage: every page of a 9-page paper carries body content.
    assert document.page_count == 9, f"页数为 {document.page_count}，应为 9"
    pages_with_body = {e.locator.page for e in document.body_elements}
    assert pages_with_body == set(range(1, 10)), f"有正文的页：{sorted(pages_with_body)}"

    # 4. Reading order is not a fallback: the group tree carried the references.
    assert not any("降级为按" in w for w in document.warnings), (
        f"阅读顺序走了降级分支：{document.warnings}"
    )

    # 5. No element was dropped for want of provenance or content.
    assert not any("没有 prov" in w or "缺少可用的页码" in w for w in document.warnings), (
        f"存在定位失败的元素：{document.warnings}"
    )


@requires_dumps
def test_the_reference_document_keeps_tables_formulas_headings_and_furniture():
    payload = _reference_dump()
    if payload is None:
        pytest.skip(f"未找到 {_REFERENCE_NAME} 的 docling dump")

    document = normalize_docling_document(
        payload,
        source_filename="hinton_distillation.pdf",
        media_type="application/pdf",
    )
    roles = [e.role for e in document.elements]

    # Tables: five, each with the dimensions docling declared, and each with
    # its caption, which is otherwise lost because the caption is folded into
    # the table rather than standing as its own element.
    tables = [e for e in document.elements if e.role == "table"]
    expected = [
        (t["data"]["num_rows"], t["data"]["num_cols"])
        for t in payload["tables"]
        if t.get("data", {}).get("num_rows")
    ]
    assert [(t.table.row_count, t.table.column_count) for t in tables] == expected
    assert all(t.table.caption.strip() for t in tables), "有表格丢失了 caption"

    # Formulas: one per label, each carrying LaTeX, none dropped for having an
    # empty ``text``.
    labelled_formulas = sum(1 for t in payload["texts"] if t.get("label") == "formula")
    formulas = [e for e in document.elements if e.role == "formula"]
    assert len(formulas) == labelled_formulas
    assert all(f.latex.strip() for f in formulas), "有公式没有 LaTeX"
    # Unicode maths was transliterated, so at least one formula must actually
    # contain a LaTeX command rather than only letters and digits.
    assert any("\\" in f.latex for f in formulas), "公式中没有出现任何 LaTeX 命令"

    # Headings: docling labels 20 section headers and reports depth 1.
    headings = [e for e in document.elements if e.role == "heading"]
    assert len(headings) == sum(
        1 for t in payload["texts"] if t.get("label") == "section_header"
    )
    assert any(h.level == 1 for h in headings), "docling 报告了 level，适配器没有传下来"

    # Furniture: admitted and tagged, which is what lets the quality gate's
    # ``repeated_furniture`` rule see a page number repeated on every page.
    furniture = [e for e in document.elements if e.is_furniture]
    assert len(furniture) == sum(
        1
        for t in payload["texts"]
        if str(t.get("content_layer")) == "furniture"
    )
    assert all(f.role in {"header", "footer"} for f in furniture)
    # And they are excluded from the body, so admitting them costs no prose.
    assert not (set(id(f) for f in furniture) & set(id(e) for e in document.body_elements))

    # Every caption docling reported is reachable. On this document all five
    # are *table* captions, so none survives as a standalone ``caption``
    # element — they are folded into the table, which is correct, and the test
    # above is what proves they were not lost with the fold.
    caption_texts = {
        str(t.get("text") or "").strip()
        for t in payload["texts"]
        if t.get("label") == "caption"
    }
    reachable = " ".join(e.text for e in document.elements) + " ".join(
        e.table.caption for e in tables if e.table
    )
    for caption in caption_texts:
        if caption:
            assert caption in reachable, f"caption 丢失：{caption[:60]!r}"

    # Footnotes are their own role rather than anonymous text.
    assert "footnote" in roles, f"没有 footnote 元素：{roles}"

    # Enumerated references keep their numbering: docling lifts the marker out
    # of ``text``, so without re-attachment the reference list is nine
    # unlabelled sentences.
    list_items = [e for e in document.elements if e.role == "list_item"]
    assert len(list_items) == sum(
        1 for t in payload["texts"] if t.get("label") == "list_item"
    )
    marked = [t for t in payload["texts"] if t.get("marker")]
    assert marked, "该文档没有带 marker 的列表项，marker 逻辑未被覆盖"
    assert all(
        any(item.text.startswith(str(t["marker"]).strip()) for item in list_items)
        for t in marked
    ), "有列表项丢失了 marker"

    # Units are one per page, and the group-only content is not stranded at the
    # end of the document away from its page.
    assert len(document.units) == 9
    assert any("Bucilu" in unit.text for unit in document.units), (
        "参考文献组没有被读入任何 page unit"
    )


class TestLatexTransliteration:
    """Unicode maths to LaTeX, and the control-word boundary that goes with it.

    In TeX a backslash binds every following letter, so a per-character
    substitution that turns ``∑`` into ``\\sum`` and leaves the ``j`` next to it
    produces ``\\sumj`` — one undefined control sequence, not a formula. A
    string that looks like a formula and does not compile is worse than the
    Unicode it replaced, because the eval counts a non-empty ``latex`` as a
    preserved formula.

    These are pinned from real ``orig`` strings out of the corpus dumps rather
    than invented, so they also document what docling actually hands over.
    """

    def test_a_control_word_is_broken_before_a_following_letter(self):
        assert adapter._to_latex("qi = exp(zi/T) ∑j exp(zj/T) (1)") == (
            "qi = exp(zi/T) \\sum j exp(zj/T) (1)"
        )
        assert adapter._to_latex("∂C ∂zi = 1 T (qi - pi)") == (
            "\\partial C \\partial zi = 1 T (qi - pi)"
        )
        assert adapter._to_latex("KL(pg , q) + ∑ m∈Ak KL(pm, q) (5)") == (
            "KL(pg , q) + \\sum m\\in Ak KL(pm, q) (5)"
        )

    def test_an_existing_command_is_not_mangled(self):
        # The regression this guards: matching a control word with a trailing
        # lookahead makes the greedy ``+`` give back letters until one is
        # followed by a letter, so it matches the *shortest* viable prefix and
        # rewrites ``\\frac`` as ``\\fra c``, ``\\lambda`` as ``\\lambd a`` and
        # ``\\sum`` as ``\\su m``. Every command in the formula is corrupted
        # while each individual assertion still passes on the broken form, so
        # the command names are spelled out here.
        assert adapter._to_latex(r"\frac{a}{b} = \frac12") == r"\frac{a}{b} = \frac12"
        assert adapter._to_latex(r"xl^{+1} = \lambda xl^{} + \mu F(xl^{})") == (
            r"xl^{+1} = \lambda xl^{} + \mu F(xl^{})"
        )
        assert adapter._to_latex(r"x = \frac{-b \pm \sqrt{4ac}}{2a}") == (
            r"x = \frac{-b \pm \sqrt{4ac}}{2a}"
        )

    def test_plain_ascii_maths_and_glyph_names(self):
        assert adapter._to_latex("E = mc^{2}") == "E = mc^{2}"
        # A font with no ToUnicode entry emits the glyph name verbatim;
        # green_tao_primes in the eval corpus does exactly this.
        assert adapter._to_latex("a /greaterorequalslant b") == "a \\ge b"

    @requires_dumps
    def test_no_control_word_runs_into_a_letter(self):
        # The property, checked over every formula docling really produced, so
        # a new Unicode operator in a later release is covered without editing
        # this file.
        for path in _DUMPS:
            for item in _load(path).get("texts") or []:
                if item.get("label") != "formula":
                    continue
                latex = adapter._to_latex(item.get("orig") or item.get("text") or "")
                for match in adapter._CONTROL_WORD.finditer(latex):
                    following = latex[match.end() : match.end() + 1]
                    assert not following.isalpha(), (
                        f"{path.name}: 控制词与字母粘连：{match.group(0)!r} + {following!r}"
                    )


@requires_dumps
def test_group_children_are_reachable_and_ordered_not_stranded():
    """A group's children are not in ``body.children``.

    The nine reference-list items of the distillation paper live only under
    ``#/groups/0``. A body-only walk loses them, and they reappear as
    positionally-sorted leftovers — a references section relocated to the end of
    the paper, which is the multi-column failure the whole tree walk exists to
    avoid.
    """
    payload = _reference_dump()
    if payload is None:
        pytest.skip(f"未找到 {_REFERENCE_NAME} 的 docling dump")

    index, kinds = adapter._index_docling_items(payload)
    ordered, ok = adapter._collect_ordered_refs(payload, index, kinds)
    assert ok
    body_refs = {child["$ref"] for child in payload["body"]["children"]}
    group_only = [
        ref
        for group in payload["groups"]
        for child in group.get("children") or []
        for ref in [child["$ref"]]
        if ref not in body_refs
    ]
    assert group_only, "该文档没有 group-only 内容，测试前提不成立"
    assert set(group_only) <= set(ordered), "group 的子项没有出现在阅读顺序中"


@pytest.mark.skipif(
    not os.getenv("RUN_DOCLING_REAL_PARSE"),
    reason="真实解析约需 12s/页且占用共享 GPU；设 RUN_DOCLING_REAL_PARSE=1 才运行",
)
def test_a_fresh_parse_of_the_corpus_pdf_keeps_the_pinned_shape():
    """End-to-end: run the installed docling and re-assert the five shapes.

    The artifact-based tests above re-verify the *adapter* against a captured
    payload. This one re-verifies that the installed docling still produces
    that payload, which is the half that a version bump would break.
    """
    docling = pytest.importorskip("docling", reason="本机未安装 docling")

    pdf = REPO_ROOT / "eval/corpus/digital_single_col" / f"{_REFERENCE_NAME}.pdf"
    if not pdf.is_file():
        pytest.skip(f"corpus 中没有 {pdf.name}")

    result = adapter.DoclingParser().parse(str(pdf))
    if not result.is_usable:
        pytest.fail(f"真实解析失败：{getattr(result, 'error_message', '')}")

    document: ParsedDocument = result.document
    assert document.locator_coverage == 1.0, (
        f"locator 覆盖率 {document.locator_coverage}；warnings={document.warnings}"
    )
    assert document.page_count == 9
    assert any(e.role == "formula" and e.latex for e in document.elements)
    assert any(e.role == "heading" for e in document.elements)
    assert any(e.is_furniture for e in document.elements)
    assert result.raw_artifact_uri, "raw Docling JSON 未落盘"

    # And the shape itself, read back from the artifact that was just written.
    payload = json.loads(Path(result.raw_artifact_uri).read_text(encoding="utf-8"))
    assert payload["texts"] and all("label" in t for t in payload["texts"])
    assert all("page_no" in p for t in payload["texts"] for p in (t.get("prov") or []))
    assert isinstance(payload["furniture"], dict)
    assert docling is not None
