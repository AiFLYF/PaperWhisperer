"""Chunking: where a boundary may fall, and what a chunk must be able to prove.

The acceptance criteria for T1.2 are three sentences -- a long paper is not cut
across a semantic boundary, tables are not mixed into prose, an index can be
rebuilt idempotently -- and each is asserted here as written rather than as a
proxy for it. That is why the corpus tests are guarded individually: a skipped
acceptance criterion is still a visible one, while an assertion that quietly
tests something else is not a criterion at all.

Two things are checked over and over because they are the ones that fail
silently. A chunk that carries a locator nobody can resolve still *looks* like
evidence in a UI, and a chunker that emits a different id per run makes every
rebuild an unreviewable full replacement. Both are tested by reconstruction
rather than by inspection: the locator is compared against the element the
chunk claims to come from, and the ids are compared across a round trip
through the document's own serialization.

The corpus-driven tests build their :class:`ParsedDocument` with a small
stand-in parser over ``pypdf``. It is crude, and that is stated where it is
written: what is under test is what the chunker does with a real paper's real
sentences and a real table's real cells, not whether a text layer can be
turned into a layout tree. ``docling`` is not installed in this repository and
this test must not require it.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import pytest

from paperwhisperer.core import config
from paperwhisperer.ingestion.contracts import ChunkContract
from paperwhisperer.ingestion.schema import (
    BoundingBox,
    Element,
    Figure,
    Formula,
    ParsedDocument,
    SourceLocator,
    Table,
)
from paperwhisperer.rag.chunking import (
    ChunkingReport,
    ChunkPolicy,
    StructuralChunker,
    build_index_version,
    chunk_document,
    chunk_document_from_config,
    element_content_text,
    estimate_tokens,
    index_version_parts,
    parse_index_version,
    split_prose_at_safe_boundaries,
)

PROVIDER = "openai_compatible"
MODEL = "text-embedding-3-small"
DIM = 1536


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def page(number: int, with_box: bool = True) -> SourceLocator:
    box = BoundingBox(10.0, 10.0, 200.0, 40.0) if with_box else None
    return SourceLocator(kind="pdf_page", page=number, bbox=box)


def element(role: str, text: str, number: int, order: int, **kwargs) -> Element:
    return Element(
        role=role,
        text=text,
        locator=kwargs.pop("locator", None) or page(number),
        order=order,
        **kwargs,
    )


def prose(count: int, marker: str = "alpha") -> str:
    """``count`` sentences, long enough that their count dominates their size."""
    return " ".join(
        f"Sentence {marker} {index} states that the model improves on prior work."
        for index in range(count)
    )


def document(*elements: Element, **kwargs) -> ParsedDocument:
    kwargs.setdefault("document_id", "doc_fixture")
    kwargs.setdefault("sha256", "sha_fixture")
    return ParsedDocument(elements=list(elements), **kwargs)


def chunks_of(
    doc: ParsedDocument,
    policy: ChunkPolicy | None = None,
    *,
    provider: str = PROVIDER,
    model: str = MODEL,
    dim: int = DIM,
):
    return chunk_document(doc, provider=provider, model=model, dim=dim, policy=policy)


def grid(rows: list[list[str]], caption: str = "", header_rows: int = 1) -> Table:
    return Table(rows=rows, header_rows=header_rows, caption=caption)


#: Characters that end a sentence, as the chunker's boundary rule sees them.
_SENTENCE_END = ".!?;…。！？；"


def starts_at_boundary(text: str, offset: int) -> bool:
    """Whether a piece beginning at ``offset`` begins at a sentence end.

    Whitespace between the terminator and the piece is not a break of its own:
    the split consumes it, so the character immediately before a piece is
    usually a space. What matters is the last non-space character, and it has
    to be one that ends a sentence.
    """
    if offset <= 0:
        return True
    index = offset - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    return text[index] in _SENTENCE_END


# --------------------------------------------------------------------------
# The token estimator
# --------------------------------------------------------------------------


class TestTokenEstimator:
    """What the budget is measured in, and the honest width of its error."""

    def test_nothing_costs_nothing(self):
        assert estimate_tokens("") == 0

    def test_non_empty_text_always_costs_something(self):
        assert estimate_tokens("a") >= 1

    def test_english_prose_is_about_four_characters_per_token(self):
        sentence = "We fine-tune the model on a held-out split of the benchmark data."

        # The published byte-pair average for English is close to four
        # characters per token. A twenty percent band is what "approximate"
        # means here: the estimate is a budget, not a count.
        assert 0.8 * len(sentence) / 4 <= estimate_tokens(sentence) <= 1.2 * len(sentence) / 4

    def test_chinese_is_charged_per_character_not_per_four(self):
        chinese = "该模型在基准数据集上取得了显著的效果提升。" * 10
        latin = "abcdefghij" * 10

        # A Latin-only average would budget a CJK document at a quarter of its
        # real size, which is the failure this separate count exists to stop.
        assert estimate_tokens(chinese) >= len(chinese) * 0.9
        assert estimate_tokens(chinese) > estimate_tokens(latin)

    def test_a_table_grid_is_not_budgeted_as_if_it_were_prose(self):
        grid_text = "\n".join("| model | top-1 | params |" for _ in range(10))
        prose_text = "model top-1 params " * 10

        # Eight pipes in a row are eight tokens to an encoder. Undercounting
        # here is the one case that reaches the embedding provider as a
        # silently truncated request.
        assert estimate_tokens(grid_text) > estimate_tokens(prose_text)

    def test_the_estimate_grows_with_the_text(self):
        short = estimate_tokens(prose(2))
        long = estimate_tokens(prose(20))

        assert 0 < short < long

    def test_the_estimate_is_deterministic(self):
        assert estimate_tokens(prose(5)) == estimate_tokens(prose(5))

    def test_a_different_estimator_replaces_the_count(self):
        # The policy takes the estimator as a parameter precisely so a caller
        # with tiktoken can use it without this module being rewritten.
        counting_words = ChunkPolicy(estimator=lambda text: len(text.split()))

        assert counting_words.size("one two three") == 3
        assert counting_words.size("one two three") != estimate_tokens("one two three")


# --------------------------------------------------------------------------
# Safe boundaries
# --------------------------------------------------------------------------


class TestSafeBoundaries:
    """The sentence-level half of the acceptance criterion."""

    def offsets_of(self, pieces: list[str], text: str) -> list[int]:
        offsets, cursor = [], 0
        for piece in pieces:
            found = text.find(piece, cursor)
            assert found >= 0, f"piece is not a verbatim slice of its source: {piece[:40]!r}"
            offsets.append(found)
            cursor = found + len(piece)
        return offsets

    def test_a_fitting_paragraph_is_left_alone(self):
        text = prose(3)

        assert split_prose_at_safe_boundaries(text) == [text]

    def test_an_oversized_paragraph_splits_at_sentence_ends(self):
        text = prose(40)
        policy = ChunkPolicy(max_tokens=100, hard_max_tokens=200)
        pieces = split_prose_at_safe_boundaries(text, policy)

        assert len(pieces) > 1
        for offset in self.offsets_of(pieces, text):
            assert starts_at_boundary(text, offset)

    def test_the_pieces_cover_the_whole_paragraph(self):
        text = prose(40)
        policy = ChunkPolicy(max_tokens=90, hard_max_tokens=200)
        pieces = split_prose_at_safe_boundaries(text, policy)
        offsets = self.offsets_of(pieces, text)

        # Nothing may be dropped on the way to an index. A truncation that is
        # not visible anywhere is the failure the release gate names.
        assert offsets[-1] + len(pieces[-1]) == len(text)

    def test_a_sentence_longer_than_the_ceiling_is_cut_between_words(self):
        giant = " ".join(["word"] * 400)
        policy = ChunkPolicy(max_tokens=50, hard_max_tokens=60)
        pieces = split_prose_at_safe_boundaries(giant, policy)

        assert len(pieces) > 1
        for offset in self.offsets_of(pieces, giant)[1:]:
            # The only unavoidable mid-sentence case, and it still has to land
            # between words: a piece starts a word, never the tail of one.
            assert giant[offset - 1] == " "

    def test_a_display_formula_is_never_cut_in_half(self):
        formula = "We minimise the objective $$\\frac{a}{b} + c$$ throughout the run. "
        text = formula * 8
        policy = ChunkPolicy(max_tokens=40, hard_max_tokens=60)
        pieces = split_prose_at_safe_boundaries(text, policy)

        assert len(pieces) > 1
        for piece in pieces:
            assert piece.count("$$") % 2 == 0

    def test_an_inline_formula_is_never_cut_in_half(self):
        text = "The gain scales as $a_{i} + b_{i}$ for every model in the suite. " * 8
        policy = ChunkPolicy(max_tokens=40, hard_max_tokens=60)

        for piece in split_prose_at_safe_boundaries(text, policy):
            assert piece.count("$") % 2 == 0

    def test_a_decimal_is_not_a_sentence_end(self):
        text = "The gain was 3.14159 points and the loss 0.50000 in every run. " * 6
        policy = ChunkPolicy(max_tokens=40, hard_max_tokens=80)
        pieces = split_prose_at_safe_boundaries(text, policy)

        assert len(pieces) > 1
        for offset in self.offsets_of(pieces, text):
            assert starts_at_boundary(text, offset)

    def test_no_piece_is_empty(self):
        text = prose(30)
        policy = ChunkPolicy(max_tokens=60, hard_max_tokens=120)

        assert all(piece.strip() for piece in split_prose_at_safe_boundaries(text, policy))


# --------------------------------------------------------------------------
# Tables, formulas and captions keep their own chunk
# --------------------------------------------------------------------------


class TestStructuralIsolation:
    """Tables, formulas and figure captions are not flattened into prose."""

    def sample(self) -> ParsedDocument:
        table = element(
            "table",
            "",
            4,
            3,
            table=grid(
                [["model", "top-1"], ["B0", "77.1"], ["B7", "84.3"]],
                caption="Table 3: ImageNet accuracy for the scaled family.",
            ),
        )
        return document(
            element("heading", "3.2 Model Scaling", 3, 0, level=2),
            element(
                "paragraph",
                "We sweep each coefficient independently. Table 3 reports the "
                "resulting accuracy for every budget.",
                3,
                1,
            ),
            table,
            element(
                "paragraph",
                "Larger models dominate the smaller ones on every dataset we tried.",
                5,
                4,
            ),
        )

    def test_a_table_becomes_a_chunk_of_its_own(self):
        chunks = chunks_of(self.sample())

        assert any(chunk.locator.page == 4 for chunk in chunks)
        assert any("| B7 | 84.3 |" in chunk.text for chunk in chunks)

    def test_no_prose_chunk_contains_a_cell_of_the_table(self):
        chunks = chunks_of(self.sample())
        holder = next(item for item in chunks if "| B7 | 84.3 |" in item.text)

        for other in chunks:
            if other.chunk_id == holder.chunk_id:
                continue
            assert "84.3" not in other.text
            assert "| model | top-1 |" not in other.text

    def test_the_table_chunk_names_the_table_element_alone(self):
        doc = self.sample()
        holder = next(item for item in chunks_of(doc) if "| B7 | 84.3 |" in item.text)

        assert holder.element_ids == (doc.elements[2].element_id,)

    def test_the_table_chunk_carries_its_section_and_its_lead_in(self):
        # A grid of numbers with no column names and no claim attached shares
        # no vocabulary with any question a reader would ask about it.
        table_chunk = next(
            chunk for chunk in chunks_of(self.sample()) if "| B7 | 84.3 |" in chunk.text
        )

        assert table_chunk.heading_path == ("3.2 Model Scaling",)
        assert table_chunk.text.startswith("## 3.2 Model Scaling")
        assert "Table 3 reports the resulting accuracy" in table_chunk.text
        assert "ImageNet accuracy" in table_chunk.text

    def test_the_lead_in_is_verbatim_source_text(self):
        # A paraphrased lead-in would put words in the index that exist
        # nowhere in the document, and every quote touching it would fail
        # verification.
        source = self.sample().elements[1].text
        table_chunk = next(
            chunk for chunk in chunks_of(self.sample()) if "| B7 | 84.3 |" in chunk.text
        )

        assert any(line in source for line in table_chunk.text.split("\n\n"))

    def test_a_display_formula_keeps_its_latex_and_its_own_chunk(self):
        formula = element("formula", "", 2, 1, formula=Formula(latex="L = \\sum_i w_i"))
        chunks = chunks_of(
            document(
                element("heading", "2 Method", 1, 0, level=1),
                element("paragraph", "The objective is defined below.", 1, 1),
                formula,
            )
        )
        holder = next(chunk for chunk in chunks if "$$" in chunk.text)

        assert holder.element_ids == (formula.element_id,)
        assert "\\sum_i w_i" in holder.text

    def test_an_inline_formula_stays_inside_its_sentence(self):
        # The schema already distinguishes display from inline, and an inline
        # formula pulled out on its own would cut the sentence around it.
        inline = element(
            "formula", "  $x_i$  ", 1, 1, formula=Formula(latex="x_i", display=False)
        )
        chunks = chunks_of(
            document(
                element("heading", "2 Method", 1, 0, level=1),
                element("paragraph", "We normalise every input by", 1, 1),
                inline,
                element("paragraph", "before the encoder runs.", 1, 2),
            )
        )

        assert all("$$" not in chunk.text for chunk in chunks)
        assert all("$x_i$" in chunk.text for chunk in chunks if "x_i" in chunk.text)

    def test_a_caption_joins_the_figure_it_describes(self):
        chunks = chunks_of(
            document(
                element("heading", "4 Results", 4, 0, level=1),
                element(
                    "figure",
                    "Figure 2: accuracy against parameter count.",
                    9,
                    1,
                    figure=Figure(caption="Figure 2: accuracy against parameter count."),
                ),
                element("caption", "Accuracy rises with size until it flattens.", 9, 2),
                element("paragraph", "The curve is monotonic.", 10, 3),
            )
        )
        figure_chunk = next(chunk for chunk in chunks if "Figure 2" in chunk.text)

        assert "Accuracy rises with size" in figure_chunk.text

    def test_an_empty_table_shell_is_not_indexed(self):
        # The quality gate flags a grid of blanks as a probable picture; the
        # docling adapter drops it. The chunker applies the same admission
        # test, so a weaker adapter cannot smuggle one back in as a chunk
        # whose entire text is pipes.
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            element("paragraph", prose(4), 1, 1),
            element("table", "", 1, 2, table=Table(rows=[["", ""], ["", ""]])),
        )

        assert all("|" not in chunk.text for chunk in chunks_of(doc))

    def test_a_shell_with_a_caption_is_kept(self):
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            element(
                "table",
                "",
                2,
                1,
                table=Table(rows=[["", ""]], caption="Figure 4: the ablation grid."),
            ),
        )

        assert any("ablation grid" in chunk.text for chunk in chunks_of(doc))

    def test_an_oversized_table_is_not_split(self):
        rows = [[f"model-{row}", f"{row}.{row}", f"{row * 7}"] for row in range(400)]
        table = element("table", "", 6, 1, table=grid(rows, caption="Table 9: the full sweep."))
        chunks = chunks_of(
            document(
                element("heading", "5 Ablation", 6, 0, level=1),
                table,
                element("paragraph", prose(3), 7, 2),
            ),
            policy=ChunkPolicy(max_tokens=100, hard_max_tokens=200),
        )
        holder = next(chunk for chunk in chunks if "model-399" in chunk.text)

        # Over budget and recorded as such, but whole. A shredded grid is the
        # one failure with no trace anywhere in the output.
        assert holder.token_estimate > 200
        assert "model-0" in holder.text and "model-399" in holder.text
        assert all("model-200" not in other.text for other in chunks if other is not holder)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


class TestSectionBoundaries:
    """A chunk never spans two heading levels."""

    def two_sections(self) -> ParsedDocument:
        return document(
            element("heading", "5 Limitations", 3, 0, level=1),
            element("paragraph", prose(6, "limits"), 3, 1),
            element("paragraph", prose(6, "limits"), 4, 2),
            element("heading", "6 Conclusion", 5, 3, level=1),
            element("paragraph", prose(6, "conclusion"), 5, 4),
        )

    def test_no_chunk_spans_two_sections(self):
        chunks = chunks_of(self.two_sections())
        by_path = {chunk.heading_path for chunk in chunks}

        assert by_path == {("5 Limitations",), ("6 Conclusion",)}

    def test_a_section_starts_a_new_chunk(self):
        chunks = chunks_of(self.two_sections())

        assert chunks[0].text.startswith("## 5 Limitations")
        assert "Conclusion" not in chunks[0].text
        assert "Limits" not in chunks[-1].text.split("\n\n")[-1]

    def test_every_chunk_of_a_section_carries_its_heading_path(self):
        for chunk in chunks_of(self.two_sections()):
            assert chunk.heading_path
            assert chunk.text.splitlines()[0] == "## " + " › ".join(chunk.heading_path)

    def test_the_heading_path_is_metadata_a_retriever_can_filter_on(self):
        chunks = chunks_of(self.two_sections())
        limitations = [chunk for chunk in chunks if chunk.heading_path == ("5 Limitations",)]

        assert limitations
        assert all("5 Limitations" in chunk.text for chunk in limitations)

    def test_nested_headings_build_a_path_rather_than_a_flat_one(self):
        chunks = chunks_of(
            document(
                element("heading", "2 Approach", 1, 0, level=1),
                element("heading", "2.1 Dataset", 1, 1, level=2),
                element("paragraph", prose(4, "dataset"), 1, 2),
                element("heading", "2.2 Training", 2, 3, level=2),
                element("paragraph", prose(4, "training"), 2, 4),
            )
        )
        paths = {chunk.heading_path for chunk in chunks}

        assert ("2 Approach", "2.1 Dataset") in paths
        assert ("2 Approach", "2.2 Training") in paths

    def test_a_heading_is_not_indexed_twice(self):
        # The section name opens every chunk as a path. Emitting the heading
        # element's own words as well would pay for a second copy of the same
        # string in every chunk of the section.
        chunks = [
            item
            for item in chunks_of(self.two_sections())
            if item.heading_path == ("5 Limitations",)
        ]

        assert chunks
        assert all(item.text.count("5 Limitations") == 1 for item in chunks)

    def test_turning_the_prefix_off_does_not_drop_the_heading(self):
        # The heading's words are the only place the section name exists when
        # no path is rendered, so no configuration may lose them.
        chunks = chunks_of(self.two_sections(), policy=ChunkPolicy(include_heading_path_in_text=False))

        assert any("5 Limitations" in chunk.text for chunk in chunks)
        assert any("6 Conclusion" in chunk.text for chunk in chunks)

    def test_an_adapter_supplied_path_wins_over_a_reconstructed_one(self):
        # Docling leaves ``level`` unset, so a path rebuilt from levels would
        # be worse information than the adapter's own.
        chunks = chunks_of(
            document(
                element(
                    "paragraph",
                    prose(4, "body"),
                    1,
                    0,
                    heading_path=("Introduction", "Background"),
                ),
            )
        )

        assert chunks[0].heading_path == ("Introduction", "Background")


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


class TestLocatorProvenance:
    """Every chunk points at somewhere, and the somewhere came from the source."""

    def rich(self) -> ParsedDocument:
        return document(
            element("heading", "1 Intro", 1, 0, level=1),
            element("paragraph", prose(5, "intro"), 1, 1),
            element("paragraph", prose(5, "later"), 7, 2),
        )

    def test_a_chunk_locator_is_the_anchor_element_locator(self):
        doc = self.rich()
        by_id = {item.element_id: item for item in doc.elements}

        for chunk in chunks_of(doc):
            assert chunk.locator == by_id[chunk.element_ids[0]].locator

    def test_a_chunk_owns_its_locator_rather_than_sharing_it(self):
        doc = self.rich()
        by_id = {item.element_id: item for item in doc.elements}

        for chunk in chunks_of(doc):
            anchor = by_id[chunk.element_ids[0]]
            assert chunk.locator is not anchor.locator
            assert chunk.locator.bbox is not anchor.locator.bbox

    def test_a_chunk_cites_the_page_its_passage_opens_on(self):
        doc = self.rich()
        chunks = chunks_of(doc)
        merged = next(
            item for item in chunks if item.locator.page == 1 and len(item.element_ids) > 1
        )

        # A page break is not a semantic boundary, so a merged chunk runs on to
        # the next page and cites the one it starts on. The elements it names are
        # what a consumer needs to reach the rest of the passage.
        assert merged.element_ids == (doc.elements[1].element_id, doc.elements[2].element_id)

    def test_running_headers_and_footers_are_excluded(self):
        chunks = chunks_of(
            document(
                element("header", "Proceedings of the 2019 Conference", 1, 0),
                element("heading", "1 Intro", 1, 1, level=1),
                element("paragraph", prose(4), 1, 2),
                element("footer", "12", 1, 3),
            )
        )

        assert all("Proceedings" not in item.text for item in chunks)
        assert all("12" not in item.text.splitlines()[0] for item in chunks)

    def test_every_chunk_names_at_least_one_element(self):
        # ChunkContract refuses an empty element_ids, and rightly: it is a
        # passage that cannot be traced back to the document it quotes.
        for chunk in chunks_of(self.rich()):
            assert chunk.element_ids
            assert all(item.strip() for item in chunk.element_ids)

    def test_a_chunk_survives_the_persistence_contract(self):
        # The index reads chunks back days later. If this module emits a chunk
        # the store refuses, the failure surfaces at retrieval instead of here.
        for chunk in chunks_of(self.rich()):
            assert ChunkContract.from_dataclass(chunk).to_dataclass() == chunk


# --------------------------------------------------------------------------
# Token-aware merge and split
# --------------------------------------------------------------------------


class TestMergeAndSplit:
    """Size decides when to cut; structure decides where."""

    def test_small_neighbours_merge_up_to_the_budget(self):
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            *[element("paragraph", prose(2, f"part{index}"), 1, index + 1) for index in range(6)],
        )
        chunks = chunks_of(doc)

        assert len(chunks) < 6
        assert any(len(item.element_ids) == 6 for item in chunks)

    def test_merging_never_crosses_a_heading(self):
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            *[element("paragraph", prose(2, f"a{index}"), 1, index + 1) for index in range(4)],
            element("heading", "2 Method", 2, 5, level=1),
            *[element("paragraph", prose(2, f"b{index}"), 2, index + 6) for index in range(4)],
        )
        policy = ChunkPolicy(max_tokens=4000, hard_max_tokens=8000)
        chunks = chunks_of(doc, policy=policy)

        assert len(chunks) == 2
        assert chunks[0].heading_path == ("1 Intro",)
        assert chunks[1].heading_path == ("2 Method",)

    def test_an_orphan_tail_joins_the_chunk_before_it(self):
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            element("paragraph", prose(30, "long"), 1, 1),
            element("paragraph", "It follows.", 2, 2),
        )
        chunks = chunks_of(doc, policy=ChunkPolicy(max_tokens=120, hard_max_tokens=400,
                                                min_merge_tokens=80))

        # A nine-word paragraph after a full one embeds to a vector nobody
        # queries for. It is still inside the same section, so merging it back
        # is safe -- and "safe" has to mean it landed, not merely that nothing
        # was lost.
        assert len(chunks) == 2
        assert chunks[-1].text.endswith("It follows.")
        assert "It follows." not in chunks[0].text

    def test_one_oversized_element_becomes_several_chunks_naming_it(self):
        big = element("paragraph", prose(60, "big"), 3, 0)
        chunks = chunks_of(
            document(big), policy=ChunkPolicy(max_tokens=100, hard_max_tokens=200)
        )

        assert len(chunks) > 1
        assert all(item.element_ids == (big.element_id,) for item in chunks)

    def test_a_prose_chunk_respects_the_budget(self):
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            *[element("paragraph", prose(3, f"p{index}"), 1, index + 1) for index in range(40)],
        )
        policy = ChunkPolicy(max_tokens=200, min_merge_tokens=0)
        chunks = chunks_of(doc, policy=policy)

        # Nothing in this document forces a boundary for meaning -- no section
        # is long enough on its own, no element is a table -- so every chunk
        # should come out at or under the target, and the slack allowed is the
        # estimator's own error rather than a designed overrun.
        for item in chunks:
            assert item.token_estimate <= policy.max_tokens + 40, item.text[:80]

    def test_a_table_does_not_count_against_the_prose_budget(self):
        # A table is a wall either way, so the prose on each side of it fills
        # the budget independently. Had the table counted against that budget,
        # the prose after it would come out short.
        table = element(
            "table", "", 2, 5, table=grid([["a", "b"], ["1", "2"]], caption="Table 1: small.")
        )
        chunks = chunks_of(
            document(
                element("heading", "1 Intro", 1, 0, level=1),
                *[
                    element("paragraph", prose(3, f"before{index}"), 1, index + 1)
                    for index in range(4)
                ],
                table,
                *[
                    element("paragraph", prose(3, f"after{index}"), 2, index + 6)
                    for index in range(4)
                ],
            )
        )
        prose_chunks = [item for item in chunks if len(item.element_ids) > 1]

        assert len(prose_chunks) == 2


# --------------------------------------------------------------------------
# Index version
# --------------------------------------------------------------------------


class TestIndexVersion:
    """Changing the embedding model has to force a rebuild, not blend vectors."""

    def version(self, **kwargs) -> str:
        kwargs.setdefault("provider", PROVIDER)
        kwargs.setdefault("model", MODEL)
        kwargs.setdefault("dim", DIM)
        return build_index_version(**kwargs)

    def test_the_version_names_the_provider_model_and_dimension(self):
        parts = index_version_parts(self.version())

        assert parts["provider"] == PROVIDER
        assert parts["model"] == MODEL
        assert parts["dim"] == DIM

    def test_changing_the_model_changes_the_version(self):
        assert self.version() != self.version(model="text-embedding-3-large")

    def test_changing_the_dimension_changes_the_version(self):
        assert self.version() != self.version(dim=3072)

    def test_changing_the_provider_changes_the_version(self):
        assert self.version() != self.version(provider="siliconflow")

    def test_changing_the_boundary_policy_changes_the_version(self):
        # max_tokens moves every boundary in the document, so vectors stored
        # beside the old chunks embed different text than the text itself.
        assert self.version() != self.version(policy=ChunkPolicy(max_tokens=512))
        assert self.version() != self.version(
            policy=ChunkPolicy(include_heading_path_in_text=False)
        )

    def test_changing_the_estimator_does_not_change_the_version(self):
        # A measurement is not a boundary. Two policies that count differently
        # and cut identically produce identical chunks and must share an index.
        assert self.version() == self.version(
            policy=ChunkPolicy(estimator=lambda text: len(text))
        )

    def test_the_chunker_signature_is_in_the_version(self):
        assert index_version_parts(self.version())["chunk"]

    def test_the_parsed_document_schema_version_is_in_the_version(self):
        assert index_version_parts(self.version())["schema"]

    def test_a_dimension_that_is_not_a_dimension_is_refused(self):
        with pytest.raises(ValueError, match="dimension"):
            self.version(dim=0)

    def test_an_unusual_model_name_round_trips(self):
        version = build_index_version(
            provider="oai|proxy", model="bge-m3:free?x=1", dim=1024
        )

        assert index_version_parts(version)["model"] == "bge-m3:free?x=1"
        assert index_version_parts(version)["provider"] == "oai|proxy"

    def test_a_version_this_build_did_not_write_reads_as_nothing(self):
        assert parse_index_version("") == {}
        assert parse_index_version("some-other-format|model=x") == {}
        assert parse_index_version("iv1|model=x|model=y") == {}

    def test_every_chunk_of_one_index_carries_the_same_version(self):
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            element("paragraph", prose(30), 1, 1),
        )
        chunks = chunks_of(doc)

        assert {item.index_version for item in chunks} == {chunks[0].index_version}

    def test_two_documents_under_one_configuration_share_a_version(self):
        # Otherwise "is this store current" is a question per document, and a
        # store holding several papers cannot answer it at all.
        first = chunks_of(document(element("paragraph", prose(3, "a"), 1, 0)))
        second = chunks_of(document(element("paragraph", prose(3, "b"), 1, 0)))

        assert first[0].index_version == second[0].index_version


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


class TestDeterminism:
    """Same document, same answer -- or a rebuild is unreviewable."""

    def sample(self) -> ParsedDocument:
        return document(
            element("heading", "2 Method", 1, 0, level=1),
            element("paragraph", prose(30, "one"), 1, 1),
            element("table", "", 2, 2,
                    table=grid([["a", "b"], ["1", "2"]], caption="Table 1: values.")),
            element("paragraph", prose(30, "two"), 2, 3),
        )

    def test_chunking_the_same_document_twice_is_byte_identical(self):
        doc = self.sample()

        first, second = chunks_of(doc), chunks_of(doc)

        assert [item.to_dict() for item in first] == [item.to_dict() for item in second]

    def test_a_rebuild_from_the_stored_document_produces_the_same_ids(self):
        # This is what a real rebuild is: the canonical document is read back
        # out of storage and re-chunked. If the ids moved, every vector in the
        # index is orphaned and the diff is unreadable.
        doc = self.sample()
        stored = ParsedDocument.from_dict(doc.to_dict())
        original = [item.chunk_id for item in chunks_of(doc)]
        rebuilt = [item.chunk_id for item in chunks_of(stored)]

        assert original == rebuilt
        assert len(set(original)) == len(original)

    def test_ids_are_unique_within_a_document(self):
        ids = [item.chunk_id for item in chunks_of(self.sample())]

        assert len(set(ids)) == len(ids)

    def test_a_changed_index_version_changes_every_id(self):
        # A rebuild under a new embedding model must not look like the old
        # index with different vectors; the ids have to say the world moved.
        doc = self.sample()
        policy = ChunkPolicy(max_tokens=400)
        before = [item.chunk_id for item in chunks_of(doc, policy=policy)]
        after = [item.chunk_id for item in chunks_of(doc, policy=policy, dim=3072)]

        assert set(before).isdisjoint(after)

    def test_a_new_chunker_instance_matches_an_old_one(self):
        doc = self.sample()
        first = StructuralChunker(provider=PROVIDER, model=MODEL, dim=DIM).chunk(doc)
        second = StructuralChunker(provider=PROVIDER, model=MODEL, dim=DIM).chunk(doc)

        assert [item.to_dict() for item in first] == [item.to_dict() for item in second]


# --------------------------------------------------------------------------
# What the run could not do
# --------------------------------------------------------------------------


class TestChunkingReport:
    """A truncated request and a dropped passage must both be visible.

    Both are invisible in the chunk list itself -- an over-budget table is
    still one well-formed chunk, and a dropped element is simply absent -- and
    the release gate allows neither to stay that way. This is also the honest
    home for the one compromise this design makes: a full-page results grid is
    never split, so on a real corpus some tables cannot be embedded whole and
    the only question left is who finds out.
    """

    def chunker(self, policy: ChunkPolicy | None = None) -> StructuralChunker:
        return StructuralChunker(provider=PROVIDER, model=MODEL, dim=DIM, policy=policy)

    def test_the_report_never_changes_the_chunks(self):
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            element("paragraph", prose(40), 1, 1),
            element("table", "", 2, 2, table=grid([["a", "b"], ["1", "2"]])),
        )
        plain, (reported, _) = self.chunker().chunk(doc), self.chunker().chunk_with_report(doc)

        assert [item.to_dict() for item in plain] == [item.to_dict() for item in reported]

    def test_an_oversized_table_is_reported_rather_than_silently_emitted(self):
        rows = [[f"m{index}", f"{index}.0", f"{index * 3}"] for index in range(300)]
        doc = document(
            element("heading", "5 Ablation", 6, 0, level=1),
            element("table", "", 6, 1, table=grid(rows, caption="Table 9: the full sweep.")),
        )
        chunks, report = self.chunker(ChunkPolicy(max_tokens=200, hard_max_tokens=400)
                                      ).chunk_with_report(doc)

        assert report.is_clean is False
        assert report.oversize
        order, tokens, kind = report.oversize[0]
        assert tokens > 400
        assert kind == "table"
        assert chunks[order].token_estimate == tokens

    def test_a_chunk_only_a_little_over_the_target_is_not_reported(self):
        # The tail-merge rule overshoots the target on purpose; listing every
        # such chunk would bury the table that genuinely cannot be embedded.
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            element("paragraph", prose(30, "long"), 1, 1),
            element("paragraph", "It follows.", 2, 2),
        )
        chunks, report = self.chunker(
            ChunkPolicy(max_tokens=120, hard_max_tokens=400, min_merge_tokens=80)
        ).chunk_with_report(doc)

        assert max(item.token_estimate for item in chunks) > 120
        assert report.is_clean is True

    def test_an_element_with_no_retrievable_text_is_reported(self):
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            element("paragraph", prose(3), 1, 1),
            element("figure", "", 2, 2, figure=Figure()),
            element("table", "", 2, 3, table=Table(rows=[["", ""], ["", ""]])),
        )
        _, report = self.chunker().chunk_with_report(doc)
        reasons = dict(report.dropped)

        assert len(reasons) == 2
        assert "caption" in next(r for r in reasons.values() if "figure" in r)
        assert "grid" in next(r for r in reasons.values() if "table" in r)

    def test_an_element_with_no_text_does_not_lose_its_neighbour(self):
        doc = document(
            element("heading", "1 Intro", 1, 0, level=1),
            element("paragraph", "   ", 1, 1),
            element("paragraph", prose(3), 1, 2),
        )
        chunks, _ = self.chunker().chunk_with_report(doc)

        assert "Sentence alpha 0" in chunks[0].text

    def test_the_warnings_are_written_for_a_user(self):
        report = ChunkingReport(
            oversize=((4, 3862, "table"),), dropped=(("el_1", "no text"),)
        )

        rendered = report.warnings(800)

        assert len(rendered) == 2
        assert "3862" in rendered[0] and "table" in rendered[0]
        assert "el_1" in rendered[1]

    def test_a_clean_document_says_so(self):
        doc = document(element("heading", "1 Intro", 1, 0, level=1),
                       element("paragraph", prose(3), 1, 1))
        _, report = self.chunker().chunk_with_report(doc)

        assert report.is_clean is True
        assert report.warnings(400) == []


# --------------------------------------------------------------------------
# Configuration wrapper
# --------------------------------------------------------------------------


class TestConfigWrapper:
    def test_the_wrapper_reads_the_running_deployment_settings(self, monkeypatch):
        monkeypatch.setattr(config, "EMBEDDING_MODEL", "bge-m3", raising=False)
        monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 1024, raising=False)
        monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "siliconflow", raising=False)
        doc = document(element("paragraph", prose(2), 1, 0))

        chunks = chunk_document_from_config(doc)
        parts = index_version_parts(chunks[0].index_version)

        assert parts["model"] == "bge-m3"
        assert parts["dim"] == 1024
        assert parts["provider"] == "siliconflow"

    def test_the_wrapper_refuses_to_guess(self, monkeypatch):
        # A missing configuration is a reason to raise, not to index the corpus
        # under a default model and discover the mismatch when retrieval
        # returns nonsense.
        monkeypatch.setattr(config, "EMBEDDING_MODEL", "", raising=False)
        monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 0, raising=False)
        doc = document(element("paragraph", prose(2), 1, 0))

        with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
            chunk_document_from_config(doc)


# --------------------------------------------------------------------------
# The real corpus
# --------------------------------------------------------------------------

_CORPUS = Path(__file__).resolve().parents[1] / "eval" / "corpus"

#: The corpus is gitignored, so these acceptance checks only mean something
#: where it has been materialised. Skipping keeps the criterion visible in a CI
#: report instead of quietly unverified.
requires_corpus = pytest.mark.skipif(
    not _CORPUS.is_dir(),
    reason=(
        f"eval corpus is not present at {_CORPUS}; it is gitignored, so this "
        "acceptance check only runs where the corpus has been materialised "
        "(manifest.yaml is tracked, the PDFs are not)"
    ),
)

#: The long-paper category, whose stated focus in the plan's corpus standard
#: is reproducing the excerpt defect: a paper whose interesting content is
#: past the first N characters.
LONG_PAPERS = (
    "long_paper/gpt3_language_models.pdf",
    "long_paper/planck2018_cosmo_params.pdf",
    "long_paper/ligo_tests_of_gr.pdf",
)

#: The table-dense category. The GLUE paper is used because its main results
#: table is real text in the text layer -- a table drawn as vector graphics is
#: invisible to a text reader, and asserting that an invisible table is
#: isolated would be asserting nothing.
TABLE_PAPER = "table_dense/glue_benchmark.pdf"

_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")
_NUMERIC = re.compile(r"\d+(?:\.\d+)?")


def _paragraph_blocks(lines: list[str]) -> list[list[str]]:
    """Split justified PDF lines back into paragraphs.

    In a two-column journal layout every line of a paragraph is justified to
    the column edge except the last, so a line materially shorter than the
    page's median line is a paragraph's final line. This is the standard
    reconstruction and it is deliberately a *stand-in parser*: it knows
    nothing about figures or column order, and it exists only to hand the
    chunker real sentences, real section titles and real paragraph lengths to
    be judged on. ``docling`` is the real parser and is not installed here, and
    a test that needed it would skip the acceptance criterion on every machine
    this repository actually runs on.
    """
    if not lines:
        return []
    width = statistics.median(len(line) for line in lines) or 1.0
    blocks: list[list[str]] = [[lines[0]]]
    for line in lines[1:]:
        if len(blocks[-1][-1]) < 0.9 * width:
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return blocks


def _is_heading(text: str) -> bool:
    match = _NUMBERED_HEADING.match(text)
    if match and len(text) < 90 and not text.endswith("."):
        return True
    return text.isupper() and 3 < len(text) < 70


def _gridish(line: str) -> bool:
    """Whether one line looks like a row of a results table."""
    return len(_NUMERIC.findall(line)) >= 5 and len(line.split()) >= 4


def _table_from_lines(lines: list[str]) -> Table:
    """A grid out of consecutive table-shaped lines, with its real header.

    Detection only. A real grid arrives from a parser with its cells and its
    caption; what is under test here is what the chunker does once one exists,
    and a table invented by the test would test the invention.
    """
    width = max(len(line.split()) for line in lines)
    rows = [line.split() + [""] * (width - len(line.split())) for line in lines]
    header_rows = 0
    for index, row in enumerate(rows):
        if not any(_NUMERIC.fullmatch(cell) for cell in row):
            rows.insert(index + 1, [])  # the real column-header line
            rows.pop()
            header_rows = 1
            break
    return Table(rows=rows, header_rows=header_rows)


def _append(elements: list[Element], role: str, text: str, page: int, level: int | None) -> None:
    elements.append(
        Element(
            role=role,
            text=text,
            level=level,
            locator=SourceLocator(kind="pdf_page", page=page),
            order=len(elements),
        )
    )


def _document_from_pdf(path: Path) -> ParsedDocument:
    """The parsed document a layout-blind reader would hand the chunker.

    Reading order is the page's own line order, which is *wrong* for a
    two-column paper -- and that is fine, and worth being explicit about: the
    chunker's contract is that it never breaks a semantic boundary, and a
    document whose reading order is already wrong cannot demonstrate that a
    correct reading order is preserved. Everything asserted below holds for any
    element sequence, which is the property under test.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    elements: list[Element] = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = pdf_page.extract_text() or ""
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        prose: list[str] = []
        index = 0
        while index < len(lines):
            if not _gridish(lines[index]):
                prose.append(lines[index])
                index += 1
                continue
            run = []
            while index < len(lines) and _gridish(lines[index]):
                run.append(lines[index])
                index += 1
            if len(run) < 4:
                prose.extend(run)
                continue
            for block in _paragraph_blocks(prose):
                body = " ".join(part.strip() for part in block).strip()
                if not body:
                    continue
                match = _NUMBERED_HEADING.match(body)
                _append(
                    elements,
                    "heading" if _is_heading(body) else "paragraph",
                    body,
                    number,
                    match.group(1).count(".") + 1 if match else None,
                )
            prose = []
            table = _table_from_lines(run)
            elements.append(
                Element(
                    role="table",
                    text=table.to_markdown(),
                    table=table,
                    locator=SourceLocator(kind="pdf_page", page=number),
                    order=len(elements),
                )
            )
        for block in _paragraph_blocks(prose):
            body = " ".join(part.strip() for part in block).strip()
            if not body:
                continue
            match = _NUMBERED_HEADING.match(body)
            _append(
                elements,
                "heading" if _is_heading(body) else "paragraph",
                body,
                number,
                match.group(1).count(".") + 1 if match else None,
            )
    return ParsedDocument(
        source_filename=path.name,
        media_type="application/pdf",
        page_count=len(reader.pages),
        elements=elements,
        parser_name="pypdf-stand-in",
    )


def corpus_path(relative: str) -> Path:
    path = _CORPUS / relative
    if not path.is_file():
        pytest.skip(f"corpus file is missing: {path}")
    return path


def element_index(doc: ParsedDocument) -> dict[str, int]:
    return {item.element_id: position for position, item in enumerate(doc.elements)}


# --------------------------------------------------------------------------
# The real corpus
# --------------------------------------------------------------------------

_CORPUS = Path(__file__).resolve().parents[1] / "eval" / "corpus"

#: The corpus is gitignored, so these acceptance checks only mean something
#: where it has been materialised. Skipping keeps the criterion visible in a CI
#: report rather than quietly unverified.
requires_corpus = pytest.mark.skipif(
    not _CORPUS.is_dir(),
    reason=(
        f"eval corpus is not present at {_CORPUS}; it is gitignored, so this "
        "acceptance check only runs where the corpus has been materialised "
        "(manifest.yaml is tracked, the PDFs are not)"
    ),
)

#: The long-paper category, whose stated focus in the plan's corpus standard is
#: reproducing the excerpt defect: a paper whose interesting content lies past
#: the first N characters.
LONG_PAPERS = (
    "long_paper/gpt3_language_models.pdf",
    "long_paper/planck2018_cosmo_params.pdf",
    "long_paper/ligo_tests_of_gr.pdf",
)

#: The table-dense category. The GLUE paper is used because its main results
#: table is real text in the text layer; a table drawn as vector graphics is
#: invisible to a text reader, and asserting that an invisible table is
#: isolated would be asserting nothing.
TABLE_PAPER = "table_dense/glue_benchmark.pdf"

_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\s+(\S.*)$")
_NUMERIC = re.compile(r"\d+(?:\.\d+)?")


def _paragraph_blocks(lines: list[str]) -> list[list[str]]:
    """Split justified PDF lines back into paragraphs.

    In a two-column journal layout every line of a paragraph is justified to
    the column edge except the last, so a line materially shorter than the
    page's median line is a paragraph's final line. This is the standard
    reconstruction and it is deliberately a *stand-in parser*: it knows nothing
    about figures or column order, and it exists only to hand the chunker real
    sentences, real section titles and real paragraph lengths to be judged on.
    ``docling`` is the real parser, it is not installed here, and a test that
    needed it would skip the acceptance criterion on every machine this
    repository actually runs on.
    """
    if not lines:
        return []
    width = statistics.median(len(line) for line in lines) or 1.0
    blocks: list[list[str]] = [[lines[0]]]
    for line in lines[1:]:
        if len(blocks[-1][-1]) < 0.9 * width:
            blocks.append([line])
        else:
            blocks[-1].append(line)
    return blocks


def _is_heading(text: str) -> bool:
    match = _NUMBERED_HEADING.match(text)
    if match and len(text) < 90 and not text.endswith("."):
        return True
    return text.isupper() and 3 < len(text) < 70


def _gridish(line: str) -> bool:
    """Whether one line looks like a row of a results table."""
    return len(_NUMERIC.findall(line)) >= 5 and len(line.split()) >= 4


def _table_from_lines(lines: list[str]) -> Table:
    """A grid out of consecutive table-shaped lines.

    Detection only, and the point of it is to hand the chunker a table that
    came out of a real PDF rather than one this test invented. What is under
    test is what the chunker does once a grid exists.
    """
    width = max(len(line.split()) for line in lines)
    return Table(
        rows=[line.split() + [""] * (width - len(line.split())) for line in lines]
    )


def _document_from_pdf(path: Path) -> ParsedDocument:
    """The parsed document a layout-blind reader would hand the chunker.

    Reading order is the page's own line order, which is *wrong* for a
    two-column paper, and that is worth being explicit about: the property
    under test is that a semantic boundary is never crossed, and that holds for
    any element sequence, so a reading order this stand-in gets wrong cannot
    hide a chunker that gets one wrong.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    elements: list[Element] = []

    def emit(role: str, text: str, page: int, level: int | None) -> None:
        elements.append(
            Element(
                role=role,
                text=text,
                level=level,
                locator=SourceLocator(kind="pdf_page", page=page),
                order=len(elements),
            )
        )

    def emit_prose(lines: list[str], page: int) -> None:
        for block in _paragraph_blocks(lines):
            body = " ".join(part.strip() for part in block).strip()
            if not body:
                continue
            match = _NUMBERED_HEADING.match(body)
            if _is_heading(body):
                # A depth for every heading, so nothing downstream has to
                # guess one: a parser that emits a heading without a level
                # makes its section tree ambiguous, and the chunker can only be
                # as sure of it as the fixture is.
                depth = match.group(1).count(".") + 1 if match else 1
                emit("heading", body, page, depth)
            else:
                emit("paragraph", body, page, None)

    for number, pdf_page in enumerate(reader.pages, start=1):
        text = pdf_page.extract_text() or ""
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        prose: list[str] = []
        index = 0
        while index < len(lines):
            if not _gridish(lines[index]):
                prose.append(lines[index])
                index += 1
                continue
            run: list[str] = []
            while index < len(lines) and _gridish(lines[index]):
                run.append(lines[index])
                index += 1
            if len(run) < 4:
                prose.extend(run)
                continue
            emit_prose(prose, number)
            prose = []
            table = _table_from_lines(run)
            emit("table", table.to_markdown(), number, None)
            elements[-1].table = table
        emit_prose(prose, number)

    return ParsedDocument(
        source_filename=path.name,
        media_type="application/pdf",
        page_count=len(reader.pages),
        elements=elements,
        parser_name="pypdf-stand-in",
    )



def expected_paths(doc: ParsedDocument) -> list[tuple[str, ...]]:
    """The section each element belongs to, computed independently of the chunker.

    Written out here rather than read back off a chunk so that a chunker which
    tracked the wrong section fails instead of agreeing with itself. Headings
    carry an explicit depth because the fixture gives them one, so the only
    thing being modelled is "a heading at depth d replaces the last d or
    shallower entries on the stack".
    """
    paths: list[tuple[str, ...]] = []
    stack: list[tuple[int, str]] = []
    for item in doc.elements:
        if item.role in {"heading", "title"}:
            assert item.level, f"fixture heading without a level: {item.text[:40]!r}"
            while stack and stack[-1][0] >= item.level:
                stack.pop()
            stack.append((item.level, item.text))
        paths.append(tuple(text for _, text in stack))
    return paths


@pytest.fixture(scope="module")
def long_paper() -> ParsedDocument:
    """The first long paper in the corpus, as a stand-in parse."""
    return _document_from_pdf(corpus_path(LONG_PAPERS[0]))


@pytest.fixture(scope="module")
def table_paper() -> ParsedDocument:
    """The table-dense paper, as a stand-in parse."""
    return _document_from_pdf(corpus_path(TABLE_PAPER))


# --------------------------------------------------------------------------
# Acceptance: a long paper is not cut across a semantic boundary
# --------------------------------------------------------------------------


@requires_corpus
class TestLongPaperBoundaries:
    """T1.2's first acceptance criterion, on the corpus category named for it."""

    def test_the_fixture_is_actually_a_long_paper(self, long_paper):
        # Without this, every assertion below could pass over a document too
        # small to have contained a split or a merge, and a green suite would
        # be saying nothing.
        assert long_paper.page_count > 20
        assert len(long_paper.elements) > 300

    def test_no_split_piece_starts_inside_a_sentence(self, long_paper):
        policy = ChunkPolicy()
        offenders: list[str] = []
        for item in long_paper.elements:
            if item.role in {"table", "formula", "figure", "caption"}:
                # Never split, by construction: there is no row or equation
                # boundary to split *at*. The unit tests hold that rule; here
                # the subject is prose.
                continue
            content = element_content_text(item)
            for piece in split_prose_at_safe_boundaries(content, policy):
                if not starts_at_boundary(content, content.find(piece)):
                    offenders.append(piece[:60])
        assert not offenders, f"{len(offenders)} piece(s) start mid-sentence: {offenders[:3]}"

    def test_every_chunk_ends_where_its_last_element_is_cut(self, long_paper):
        # The chunk body is assembled from whole elements and from safe slices
        # of them, so its tail is always one of those. This is the chunk-level
        # form of the same property, checked through the packaging.
        policy = ChunkPolicy()
        by_id = {item.element_id: item for item in long_paper.elements}
        for item in chunks_of(long_paper, policy=policy):
            content = element_content_text(by_id[item.element_ids[-1]])
            allowed = {content} | set(split_prose_at_safe_boundaries(content, policy))
            assert any(item.text.endswith(candidate) for candidate in allowed), (
                f"chunk {item.order} does not end on a boundary of its last element"
            )

    def test_no_chunk_spans_two_sections(self, long_paper):
        # The criterion, asserted as stated: every element in a chunk belongs
        # to the one section that chunk is filed under.
        expected = expected_paths(long_paper)
        order = element_index(long_paper)
        for item in chunks_of(long_paper):
            paths = {expected[order[element_id]] for element_id in item.element_ids}
            assert paths == {item.heading_path}, (
                f"chunk {item.order} names elements from {paths} under "
                f"{item.heading_path}"
            )

    def test_a_chunk_drops_nothing_between_the_elements_it_names(self, long_paper):
        order = element_index(long_paper)
        for item in chunks_of(long_paper):
            positions = sorted(order[element_id] for element_id in item.element_ids)
            named = set(positions)
            skipped = [
                position
                for position in range(positions[0] + 1, positions[-1])
                if position not in named
                and long_paper.elements[position].role not in {"heading", "title"}
            ]
            # A heading between two of a chunk's elements is not a skipped
            # passage: its words are the section the chunk is filed under. A
            # paragraph between them would be text the index lost.
            assert not skipped, f"chunk {item.order} skips elements at {skipped}"

    def test_heading_metadata_matches_the_document(self, long_paper):
        expected = expected_paths(long_paper)
        by_id = element_index(long_paper)

        for item in chunks_of(long_paper):
            wanted = expected[by_id[item.element_ids[0]]]
            assert item.heading_path == wanted, (
                f"chunk {item.order} is filed under {item.heading_path} and opens on "
                f"an element belonging to {wanted}"
            )

    def test_merging_actually_happens(self, long_paper):
        assert any(len(item.element_ids) > 1 for item in chunks_of(long_paper))

    def test_splitting_actually_happens(self, long_paper):
        # A paper's own paragraphs fit a 400-token budget, so the default
        # policy never has to split one. A tighter budget is the only way to
        # exercise the split path over real sentences, and the split path is
        # where the acceptance criterion lives.
        chunks = chunks_of(
            long_paper, policy=ChunkPolicy(max_tokens=120, hard_max_tokens=240)
        )

        assert any(
            sum(1 for other in chunks if other.element_ids == item.element_ids) > 1
            for item in chunks
        ), "no split occurred"

    def test_a_long_paper_is_chunked_the_same_way_twice(self, long_paper):
        first = [item.to_dict() for item in chunks_of(long_paper)]
        second = [item.to_dict() for item in chunks_of(long_paper)]
        stored = ParsedDocument.from_dict(long_paper.to_dict())
        rebuilt = [item.chunk_id for item in chunks_of(stored)]

        assert first == second
        assert [entry["chunk_id"] for entry in first] == rebuilt

    def test_the_report_names_what_cannot_be_embedded_whole(self, long_paper):
        # A full-page results grid is never split, so on a real paper some
        # tables are too large for the provider. That is a compromise worth
        # knowing about rather than discovering from a truncated embedding.
        _, report = StructuralChunker(
            provider=PROVIDER, model=MODEL, dim=DIM
        ).chunk_with_report(long_paper)

        assert report.oversize
        assert all(kind == "table" for _, _, kind in report.oversize)
        assert any(tokens > 1000 for _, tokens, _ in report.oversize)
        assert report.dropped == ()

    @pytest.mark.parametrize("relative", LONG_PAPERS)
    def test_every_long_paper_in_the_corpus_survives_chunking(self, relative):
        chunks = chunks_of(_document_from_pdf(corpus_path(relative)))

        assert len(chunks) > 50
        assert all(item.element_ids for item in chunks)
        assert all(item.text.strip() for item in chunks)
        assert all(item.token_estimate > 0 for item in chunks)


# --------------------------------------------------------------------------
# Acceptance: a table is not mixed into body prose
# --------------------------------------------------------------------------


@requires_corpus
class TestTablesAreNotProse:
    """T1.2's second acceptance criterion, on a real results table."""

    def tables(self, doc: ParsedDocument) -> list[Element]:
        found = [item for item in doc.elements if item.role == "table" and item.table]
        assert found, "no table-shaped region found in the fixture"
        return found

    def marker_row(self, doc: ParsedDocument, table: Table) -> str:
        """One rendered row that occurs exactly once in the whole document.

        Uniqueness is what makes a containment assertion mean anything. A
        single cell does not qualify: the numbers in a results table are drawn
        from a small range, so "65.1" appears in the second table on the same
        page, and an assertion built on it would pass whether or not the chunker
        did anything. A whole rendered row is a fingerprint.
        """
        whole = "\n".join(element_content_text(item) for item in doc.elements)
        for row in table.rows:
            rendered = " | ".join(row)
            if whole.count(rendered) == 1:
                return rendered
        raise AssertionError("no uniquely identifying row in the fixture table")

    def unique_cells(self, doc: ParsedDocument, table: Table) -> list[str]:
        """The table's cells that appear in no prose element of the document.

        Uniqueness is against the prose only, and the other tables are excluded
        from the comparison for the same reason: a row label such as "+ELMo"
        belongs to three tables on that page, and treating it as evidence that
        one of them leaked into body text would fail on a document that is
        entirely correct. What the assertion below is really about is a cell
        that exists *only* inside a grid ending up in a run of sentences.
        """
        others = [
            element_content_text(item) for item in doc.elements if item.role != "table"
        ]
        return sorted(
            {
                cell
                for row in table.rows
                for cell in row
                if len(cell) > 3 and all(cell not in other for other in others)
            }
        )

    def test_a_real_table_owns_a_chunk_of_its_own(self, table_paper):
        for table in self.tables(table_paper):
            marker = self.marker_row(table_paper, table.table)
            holders = [item for item in chunks_of(table_paper) if marker in item.text]

            assert len(holders) == 1, f"{marker!r} appears in {len(holders)} chunks"
            assert holders[0].element_ids == (table.element_id,)

    def test_no_prose_chunk_contains_a_cell_of_the_table(self, table_paper):
        chunks = chunks_of(table_paper)
        table_ids = {item.element_id for item in self.tables(table_paper)}
        seen = 0
        for table in self.tables(table_paper):
            cells = self.unique_cells(table_paper, table.table)
            if not cells:
                # Every one of its cells also occurs somewhere in the paper's
                # prose -- a number like 65.1 is quoted in the results
                # discussion as well as in the grid -- so containment in a body
                # chunk would say nothing about this table.
                continue
            seen += 1
            for other in chunks:
                if other.element_ids[0] in table_ids:
                    # Another table's chunk, which is where a shared row label
                    # belongs.
                    continue
                assert not any(cell in other.text for cell in cells), (
                    f"a body chunk carries cells of the table on page {table.locator.page}"
                )
        assert seen

    def test_the_table_chunk_keeps_the_grid_intact(self, table_paper):
        chunks = chunks_of(table_paper)
        for table in self.tables(table_paper):
            marker = self.marker_row(table_paper, table.table)
            holder = next(item for item in chunks if marker in item.text)
            for row in table.table.rows:
                assert " | ".join(row) in holder.text

    def test_the_table_chunk_cites_the_table_and_carries_its_section(self, table_paper):
        chunks = chunks_of(table_paper)
        for table in self.tables(table_paper):
            marker = self.marker_row(table_paper, table.table)
            holder = next(item for item in chunks if marker in item.text)

            assert holder.locator == table.locator
            assert holder.heading_path
            assert holder.text.startswith("## ")

    def test_the_table_is_given_context_it_did_not_have(self, table_paper):
        chunks = chunks_of(table_paper)
        table = self.tables(table_paper)[0]
        marker = self.marker_row(table_paper, table.table)
        holder = next(item for item in chunks if marker in item.text)
        preceding = table_paper.elements[element_index(table_paper)[table.element_id] - 1]

        assert preceding.role not in {"table", "figure"}
        assert holder.heading_path
        assert any(
            line and line in preceding.text for line in holder.text.split("\n\n")
        ), "the table chunk carries no lead-in from the sentence before it"
