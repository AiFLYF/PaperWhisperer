"""What every tier's lexical store must do, asserted once.

The score a caller sees is computed in Python by
:mod:`~paperwhisperer.infra.stores.scoring` on every tier, so the expected
numbers below are real numbers rather than "something was returned". That is
the whole reason the backends are allowed to differ: a tier with an index
narrows candidates faster, and both tiers then agree, exactly, about what
each candidate is worth.

The tests query with whole words. That is the guaranteed contract — a query
term that is a token of the document is found on both tiers — and it is what
makes the shared suite meaningful. A term *inside* a word is a capability
some tiers have and some do not, and it is tested in the tier-named section
at the bottom rather than here, because asserting it in the shared suite would
be asserting that the two tiers have the same extras.
"""

from __future__ import annotations

import threading

import pytest

from .test_stores_tiers import (
    TIER_HOSTED,
    TIER_PERSONAL,
    open_harness,
    requires_postgres,
)

#: Three passages chosen so that every ranking assertion below has exactly
#: one defensible answer: one contains the whole query as a phrase, one
#: contains the same words scattered, one is unrelated.
DOC_FULL = (
    "We propose a new simple network architecture, the Transformer, based solely "
    "on attention mechanisms, dispensing with recurrence and convolutions entirely."
)
DOC_SCATTERED = (
    "Attention is used in the encoder. The decoder is autoregressive, so recurrence "
    "was retained there in early work, and convolution remained useful for local "
    "patterns."
)
DOC_UNRELATED = "The cat sat on the mat. Household furniture is unrelated to parsing."


#: Chunk id prefix per seeded document, so a test that seeds two documents
#: gets two sets of rows rather than one set overwritten twice.
SEED_IDS = ("full", "scattered", "unrelated")
SEED_TEXTS = (DOC_FULL, DOC_SCATTERED, DOC_UNRELATED)


def seed(store, *, document_id="doc-lex", prefix="c-") -> None:
    """Index the three fixture passages under one document.

    ``prefix`` exists because chunk ids are the primary key: seeding a second
    document with the same ids would overwrite the first document's rows
    rather than adding to them, and a test about deleting one document would
    delete zero.
    """
    for suffix, text in zip(SEED_IDS, SEED_TEXTS, strict=True):
        store.index_chunk(f"{prefix}{suffix}", text, document_id=document_id)


@pytest.fixture
def harness(request, tmp_path):
    """The store bundle for whichever tier the test class names.

    The class attribute is the whole mechanism: ``tier = "sqlite"`` on one
    subclass and ``tier = "hosted"`` on the other, so the two differ by one
    line and cannot drift in their assertions.
    """
    yield from open_harness(request.node.cls.tier, tmp_path)


class StoreContract:
    """Behaviour that must be identical on every tier."""

    def test_an_empty_index_answers_nothing(self, harness):
        assert harness.lexical.count_chunks() == 0
        assert harness.lexical.search("transformer") == []

    def test_a_single_term_finds_its_document(self, harness):
        seed(harness.lexical)
        hits = harness.lexical.search("transformer", top_k=5)
        assert [hit.chunk_id for hit in hits] == ["c-full"]
        assert hits[0].score == pytest.approx(1.0)
        assert hits[0].document_id == "doc-lex"

    def test_a_phrase_outranks_a_scattered_word_list(self, harness):
        """The distinction that makes the score worth anything.

        Coverage alone would rate "attention ... convolution" the same as
        "based solely on attention mechanisms, dispensing with recurrence and
        convolutions entirely". For a product whose output is a citable
        quote, "these words co-occur somewhere in a section" and "this is the
        sentence you were looking for" are different answers, and only the
        second one is worth putting in front of a reader.
        """
        seed(harness.lexical)
        hits = harness.lexical.search("attention mechanisms", top_k=5)
        assert [hit.chunk_id for hit in hits] == ["c-full", "c-scattered"]
        assert hits[0].score > hits[1].score
        # Both contain the term, so neither is dropped: the ordering carries
        # the signal, and padding the list with unrelated rows would not.
        assert all(hit.score > 0.0 for hit in hits)

    def test_a_document_sharing_no_term_is_never_returned(self, harness):
        seed(harness.lexical)
        hits = harness.lexical.search("transformer", top_k=10)
        assert "c-unrelated" not in {hit.chunk_id for hit in hits}

    def test_an_unrelated_query_returns_nothing_rather_than_filler(self, harness):
        # Padding a result list to reach top_k would let an unanswered
        # question look answered, which is the one outcome this product
        # exists to prevent.
        seed(harness.lexical)
        assert harness.lexical.search("photosynthesis chlorophyll") == []

    def test_a_query_with_no_indexable_characters_returns_nothing(self, harness):
        seed(harness.lexical)
        assert harness.lexical.search("") == []
        assert harness.lexical.search("   ...  ") == []
        assert harness.lexical.search("!!! ???") == []

    def test_matching_ignores_case_and_punctuation(self, harness):
        seed(harness.lexical)
        assert [hit.chunk_id for hit in harness.lexical.search("TRANSFORMER!")] == ["c-full"]
        assert [hit.chunk_id for hit in harness.lexical.search("  transformer  ")] == ["c-full"]

    def test_full_width_characters_fold_onto_ascii(self, harness):
        """NFKC is doing real work, not decoration.

        Full-width Latin reaches this project through pasted abstracts, and
        ``ＴＲＡＮＳＦＯＲＭＥＲ`` is the same word as ``transformer``. A
        store that normalized with case folding alone would answer "no
        results" to a query a user could see on screen.
        """
        seed(harness.lexical)
        assert [hit.chunk_id for hit in harness.lexical.search("ＴＲＡＮＳＦＯＲＭＥＲ")] == [
            "c-full"
        ]

    def test_chinese_is_matched_by_character_bigrams(self, harness):
        """No whitespace tokenizer can do this, and this product is bilingual.

        A Chinese phrase is one whitespace-delimited token, so an index built
        on raw text would need the document to contain the query's *exact*
        character run. Querying for 注意力 must find 自注意力机制, which is the
        question the plan's mixed-language regression requirement is about.
        """
        harness.lexical.index_chunk("c-zh", "本文提出自注意力机制，替代循环与卷积。")
        hits = harness.lexical.search("注意力", top_k=5)
        assert [hit.chunk_id for hit in hits] == ["c-zh"]
        assert hits[0].score > 0.0
        assert "自注意力机制" in hits[0].snippet

    def test_a_mixed_language_query_finds_both_sides(self, harness):
        harness.lexical.index_chunk("c-zh", "本文提出自注意力机制，替代循环与卷积。")
        harness.lexical.index_chunk("c-en", DOC_FULL)
        hits = harness.lexical.search("attention 注意力", top_k=5)
        assert {hit.chunk_id for hit in hits} == {"c-zh", "c-en"}

    def test_the_snippet_quotes_the_original_text(self, harness):
        seed(harness.lexical)
        snippet = harness.lexical.search("convolution", top_k=1)[0].snippet
        # Lowercased, punctuation-stripped and bigrammed text is a perfect
        # index and a terrible quotation. The snippet exists to be shown to a
        # reader, so it has to be a window of the sentence they would find on
        # the page — verbatim, not normalized.
        assert "convolution" in snippet.lower()
        assert snippet.lstrip("…").rstrip("…") in DOC_SCATTERED

    def test_the_snippet_is_cut_around_the_match(self, harness):
        long_body = ("filler sentence. " * 200) + "The needle is here. " + ("more filler. " * 200)
        harness.lexical.index_chunk("c-long", long_body)
        snippet = harness.lexical.search("needle", top_k=1)[0].snippet
        assert "needle" in snippet
        assert len(snippet) < len(long_body)

    def test_the_snippet_of_a_head_match_has_no_leading_ellipsis(self, harness):
        seed(harness.lexical)
        snippet = harness.lexical.search("propose", top_k=1)[0].snippet
        assert snippet.startswith("We propose")

    def test_top_k_truncates_and_zero_returns_nothing(self, harness):
        seed(harness.lexical)
        assert len(harness.lexical.search("attention", top_k=1)) == 1
        assert harness.lexical.search("attention", top_k=0) == []

    def test_equal_scores_break_ties_on_the_chunk_id(self, harness):
        for chunk_id in ("chunk-c", "chunk-a", "chunk-b"):
            harness.lexical.index_chunk(chunk_id, "attention mechanism")
        hits = harness.lexical.search("attention mechanism", top_k=3)
        assert [hit.chunk_id for hit in hits] == ["chunk-a", "chunk-b", "chunk-c"]

    def test_a_document_filter_narrows_the_result(self, harness):
        harness.lexical.index_chunk("c-a", "attention mechanism", document_id="doc-a")
        harness.lexical.index_chunk("c-b", "attention mechanism", document_id="doc-b")
        assert len(harness.lexical.search("attention", top_k=10)) == 2
        assert [
            hit.chunk_id for hit in harness.lexical.search("attention", document_ids=["doc-b"])
        ] == ["c-b"]
        assert harness.lexical.search("attention", document_ids=[]) == []

    # -- maintenance -------------------------------------------------------

    def test_re_indexing_replaces_the_text(self, harness):
        harness.lexical.index_chunk("c1", "attention mechanism")
        assert [hit.chunk_id for hit in harness.lexical.search("attention")] == ["c1"]

        harness.lexical.index_chunk("c1", "nothing relevant here")
        assert harness.lexical.search("attention") == []
        assert [hit.chunk_id for hit in harness.lexical.search("relevant")] == ["c1"]
        assert harness.lexical.count_chunks() == 1

    def test_a_stale_replacement_does_not_leak_into_results(self, harness):
        """The failure a stale index entry causes, stated as a test.

        A row left behind in the accelerator after its text was replaced
        would still be a candidate, and would then be scored against text the
        user never sees. This asserts the *old* text no longer matches, which
        is the only observable difference.
        """
        harness.lexical.index_chunk("c1", "attention mechanism")
        harness.lexical.index_chunk("c1", "convolution recurrence")
        assert harness.lexical.search("attention") == []
        assert harness.lexical.search("mechanism") == []

    def test_deleting_reports_whether_it_was_there(self, harness):
        seed(harness.lexical)
        assert harness.lexical.delete_chunk("c-full") is True
        assert harness.lexical.delete_chunk("c-full") is False
        assert harness.lexical.count_chunks() == 2
        assert harness.lexical.search("transformer") == []

    def test_deleting_a_document_removes_only_its_chunks(self, harness):
        seed(harness.lexical, document_id="doc-a", prefix="a-")
        seed(harness.lexical, document_id="doc-b", prefix="b-")
        assert harness.lexical.delete_document("doc-a") == 3
        assert harness.lexical.count_chunks() == 3
        assert {hit.document_id for hit in harness.lexical.search("attention")} == {"doc-b"}

    def test_an_empty_chunk_id_is_refused(self, harness):
        with pytest.raises(ValueError, match="chunk_id"):
            harness.lexical.index_chunk("  ", "text")

    def test_indexing_empty_text_is_allowed_and_matches_nothing(self, harness):
        harness.lexical.index_chunk("c-empty", "")
        assert harness.lexical.count_chunks() == 1
        assert harness.lexical.search("anything") == []

    def test_initialize_is_idempotent(self, harness):
        seed(harness.lexical)
        harness.lexical.initialize()
        harness.lexical.initialize()
        assert harness.lexical.count_chunks() == 3
        assert len(harness.lexical.search("attention", top_k=5)) == 2

    # -- concurrency -------------------------------------------------------

    def test_concurrent_writes_all_land(self, harness):
        workers = 8
        per_worker = 6
        errors: list[BaseException] = []
        barrier = threading.Barrier(workers)

        def _write(worker: int) -> None:
            try:
                barrier.wait(timeout=10)
                for index in range(per_worker):
                    harness.lexical.index_chunk(
                        f"w{worker}-{index}",
                        f"passage {worker} {index} about attention mechanism number {index}",
                        document_id=f"doc-{worker}",
                    )
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(worker,)) for worker in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert errors == []
        assert harness.lexical.count_chunks() == workers * per_worker
        # Every worker wrote the same terms, so all of them must be
        # retrievable. A row that was written but not indexed would be
        # invisible to every assertion above.
        assert len(harness.lexical.search("attention", top_k=100)) == workers * per_worker


class TestLexicalSqlite(StoreContract):
    tier = TIER_PERSONAL


@requires_postgres
class TestLexicalPostgres(StoreContract):
    tier = TIER_HOSTED


class TestLexicalPersonalTierOnly:
    """The two search strategies, and proving they agree where they must.

    Each of these is about the personal tier, so none of it is a contract.
    The one that matters most is
    :meth:`test_the_accelerator_and_the_scan_path_rank_identically` — that is
    the claim the shared suite rests on.
    """

    def test_the_accelerator_and_the_scan_path_rank_identically(self, tmp_path):
        """Same corpus, same engine question, same answers.

        The probe picks between a full-text index and a linear scan at
        construction time, which means the answer to "what does a query
        return" depends on the build of SQLite the user happens to have. If
        the two paths disagreed, retrieval quality would differ between two
        installations of the *same* tier — a bug report nobody could
        reproduce and no test could catch, because only one path would ever
        be exercised on any given machine.
        """
        from paperwhisperer.infra.stores import build_stores

        from .test_stores_tiers import _Settings

        path = str(tmp_path / "strategies.db")
        corpus = {
            "c-full": DOC_FULL,
            "c-scattered": DOC_SCATTERED,
            "c-unrelated": DOC_UNRELATED,
            "c-zh": "本文提出自注意力机制，替代循环与卷积。",
            "c-latin": "Eine neue Attention-Architektur ohne Rekursion oder Faltung.",
        }
        queries = [
            "transformer",
            "attention mechanisms",
            "convolution recurrence",
            "注意力",
            "attention 注意力",
            "architektur",
            "photosynthesis",
        ]

        def _run(mode: str) -> dict[str, list]:
            bundle = build_stores(_Settings(STORAGE_SQLITE_PATH=path, STORAGE_LEXICAL_FTS5=mode))
            try:
                for chunk_id, text in corpus.items():
                    bundle.lexical.index_chunk(chunk_id, text, document_id="doc-strategy")
                return {
                    query: [
                        (hit.chunk_id, round(hit.score, 9), hit.snippet)
                        for hit in bundle.lexical.search(query, top_k=10)
                    ]
                    for query in queries
                }
            finally:
                bundle.close()

        accelerated = _run("on")
        scanned = _run("off")
        assert accelerated == scanned

    def test_the_accelerator_is_actually_used_when_available(self, tmp_path):
        from paperwhisperer.infra.stores import build_stores

        from .test_stores_tiers import _Settings

        bundle = build_stores(
            _Settings(STORAGE_SQLITE_PATH=str(tmp_path / "acc.db"), STORAGE_LEXICAL_FTS5="on")
        )
        try:
            assert bundle.capabilities.lexical_engine == "fts"
            assert bundle.capabilities.scan_only is False
            assert bundle.capabilities.degradations == ()
        finally:
            bundle.close()

    def test_a_term_inside_a_word_is_not_matched_by_either_strategy(self, tmp_path):
        """The limit of the contract, stated so it is a decision and not a bug.

        ``"raph"`` is inside ``"graph"`` and neither strategy reports it. An
        earlier version of the scan path let such rows through the narrowing
        step, and they all came back at 0.0 and were dropped by the scorer —
        so the filter cost a full scan and changed no result. Writing the
        limit down here means the next person to notice it finds a test
        explaining it, rather than a comment.
        """
        from paperwhisperer.infra.stores import build_stores

        from .test_stores_tiers import _Settings

        for mode in ("on", "off"):
            bundle = build_stores(
                _Settings(
                    STORAGE_SQLITE_PATH=str(tmp_path / f"inner-{mode}.db"),
                    STORAGE_LEXICAL_FTS5=mode,
                )
            )
            try:
                bundle.lexical.index_chunk("c1", "graph neural networks")
                assert bundle.lexical.search("raph") == []
                # The whole token is found by both.
                assert [hit.chunk_id for hit in bundle.lexical.search("graph")] == ["c1"]
            finally:
                bundle.close()

    def test_an_index_written_while_disabled_is_still_there_when_it_is_turned_back_on(
        self, tmp_path
    ):
        """The stale-accelerator bug, which is invisible until it is not.

        A database written by a run with the accelerator off still gets its
        index maintained when the table exists, so turning it back on finds
        everything. Without that, the first run after the switch would return
        nothing at all for documents that are in the table — and it would
        return nothing *correctly*, silently, with no error anywhere.
        """
        from paperwhisperer.infra.stores import build_stores

        from .test_stores_tiers import _Settings

        path = str(tmp_path / "stale.db")

        warm = build_stores(_Settings(STORAGE_SQLITE_PATH=path, STORAGE_LEXICAL_FTS5="on"))
        warm.lexical.index_chunk("c1", "the transformer architecture")
        warm.close()

        disabled = build_stores(
            _Settings(STORAGE_SQLITE_PATH=path, STORAGE_LEXICAL_FTS5="off")
        )
        assert disabled.capabilities.scan_only is True
        disabled.lexical.index_chunk("c2", "a second passage about attention")
        disabled.close()

        reenabled = build_stores(
            _Settings(STORAGE_SQLITE_PATH=path, STORAGE_LEXICAL_FTS5="on")
        )
        try:
            assert reenabled.capabilities.scan_only is False
            assert [hit.chunk_id for hit in reenabled.lexical.search("transformer")] == ["c1"]
            assert [hit.chunk_id for hit in reenabled.lexical.search("attention")] == ["c2"]
        finally:
            reenabled.close()

    def test_deleting_while_disabled_also_clears_the_index(self, tmp_path):
        """The observable half of a delete, asserted while the index is off.

        Today this could not fail: the accelerated path joins the index to the
        chunk table, so a stale index row for a deleted chunk is invisible
        whether or not it was cleared. It is written anyway because the join
        is a property of the current query rather than a guarantee, and a
        delete that stops clearing one of two places is the kind of half-measure
        that becomes a bug the moment a query stops needing the join.
        """
        from paperwhisperer.infra.stores import build_stores

        from .test_stores_tiers import _Settings

        path = str(tmp_path / "stale-delete.db")
        warm = build_stores(_Settings(STORAGE_SQLITE_PATH=path, STORAGE_LEXICAL_FTS5="on"))
        warm.lexical.index_chunk("c1", "the transformer architecture")
        warm.close()

        disabled = build_stores(
            _Settings(STORAGE_SQLITE_PATH=path, STORAGE_LEXICAL_FTS5="off")
        )
        assert disabled.lexical.delete_chunk("c1") is True
        disabled.close()

        reenabled = build_stores(
            _Settings(STORAGE_SQLITE_PATH=path, STORAGE_LEXICAL_FTS5="on")
        )
        try:
            assert reenabled.lexical.search("transformer") == []
        finally:
            reenabled.close()

    def test_a_bare_prefix_is_not_recalled_on_either_strategy(self, tmp_path):
        """The other half of the exact-term contract.

        The narrowed rows come back and the shared scorer scores them at
        zero, because the document's token is ``transformer`` and the query's
        is ``transfor``. Recorded as a test so the limit is discoverable: the
        next person to widen recall finds this and the module docstring's
        argument for putting it in T1.4 instead of here.
        """
        from paperwhisperer.infra.stores import build_stores

        from .test_stores_tiers import _Settings

        for mode in ("on", "off"):
            bundle = build_stores(
                _Settings(
                    STORAGE_SQLITE_PATH=str(tmp_path / f"prefix-{mode}.db"),
                    STORAGE_LEXICAL_FTS5=mode,
                )
            )
            try:
                bundle.lexical.index_chunk("c1", "the transformer architecture")
                assert bundle.lexical.search("transfor") == []
                assert [hit.chunk_id for hit in bundle.lexical.search("transformer")] == ["c1"]
            finally:
                bundle.close()
