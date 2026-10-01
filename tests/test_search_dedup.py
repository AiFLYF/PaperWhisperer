"""The live search path: identifiers that survive, and duplicates that collapse.

Every other test in this file is downstream of the claim these tests make: a
preprint and its published version are one row, and two different papers are
two rows. Both are silent failures — one shows a paper twice, the other hides
a paper the user searched for — so both are asserted directly rather than
through whatever the list happens to look like.

**The asymmetry the whole module is built on.** Merging two different papers
destroys a paper the user can no longer find; showing one paper twice costs
them a glance. So the search list collapses only what a shared *identifier*
proves, and reports everything it merely suspects. A ``probable`` match from
:func:`paperwhisperer.search.identity.compare_identities` is kept as two
visible rows plus an entry in ``possible_duplicates``; there is no flag, no
threshold and no argument that makes a caller collapse one, because the
collapse rule is a single named constant.

No network is touched. Provider rows are built from realistic S2 and arXiv
payloads and the search functions are replaced on the module, exactly as
``tests/test_paper_identity.py`` replaces the fetchers.
"""

from __future__ import annotations

import pytest

from paperwhisperer.search import papers
from paperwhisperer.search.identity import compare_identities, identity_from_record
from paperwhisperer.search.normalize import (
    COLLAPSE_CONFIDENCE,
    deduplicate_papers,
    deduplicate_papers_report,
    normalize_paper_collection,
    normalize_paper_record,
)

# --------------------------------------------------------------------------- #
# Provider payloads
#
# One real paper, in the three shapes it actually arrives in. He et al. posted
# arXiv:1512.03385 in December 2015; it appeared at CVPR 2016 as
# 10.1109/CVPR.2016.90. The year changes, the venue changes, the author list
# is reordered, and the title is re-typeset with a trailing period on the
# publisher's copy. It is one paper.
# --------------------------------------------------------------------------- #

RESIDUAL_TITLE = "Deep Residual Learning for Image Recognition"
RESIDUAL_AUTHOR = "Kaiming He"
PREPRINT_ARXIV = "1512.03385"
PUBLISHED_DOI = "10.1109/CVPR.2016.90"


def s2_payload(
    paper_id: str = "649def34f8be52c8b66281af98ae884c09aef38b",
    *,
    title: str = RESIDUAL_TITLE,
    year: int = 2015,
    doi: str = PUBLISHED_DOI,
    arxiv_id: str = PREPRINT_ARXIV,
    authors: list[str] | None = None,
) -> dict:
    """A Semantic Scholar ``/paper/search`` row, including ``externalIds``.

    S2 nests its identifiers rather than returning them as fields, which is the
    whole reason a DOI could arrive with a record and be discarded on the way
    in. ``CorpusId`` is included because S2 really does send it, and because
    a handler that mistook it for a DOI would write a permanent wrong key.
    """
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": "  We present a   residual learning framework. ",
        "year": year,
        "venue": "2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
        "url": f"https://www.semanticscholar.org/paper/{paper_id[:8]}",
        "authors": [{"name": name} for name in (authors or [RESIDUAL_AUTHOR, "Xiangyu Zhang"])],
        "openAccessPdf": {"url": f"https://arxiv.org/pdf/{PREPRINT_ARXIV}.pdf"},
        "externalIds": {
            "CorpusId": 137498608,
            "DBLP": "conf/cvpr/He16",
            "DOI": doi,
            "ArXiv": arxiv_id,
        },
    }


def arxiv_payload(
    *,
    arxiv_id: str = PREPRINT_ARXIV,
    title: str = RESIDUAL_TITLE,
    published: str = "2015-12-11T18:41:45Z",
    doi: str = "",
) -> dict:
    """An arXiv record as :func:`search_arxiv_papers` hands it to the mapper."""
    return {
        "paper_id": f"http://arxiv.org/abs/{arxiv_id}v1",
        "title": title,
        "abstract": "We present a residual learning framework.",
        "authors": [RESIDUAL_AUTHOR, "Xiangyu Zhang"],
        "published": published,
        "doi": doi,
        "url": f"http://arxiv.org/abs/{arxiv_id}v1",
        "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}v1",
    }


def residual_pair() -> list[dict]:
    """The published version and the preprint of one paper, as two rows."""
    return [
        normalize_paper_record("Semantic Scholar", s2_payload()),
        normalize_paper_record("arXiv", arxiv_payload()),
    ]


# --------------------------------------------------------------------------- #
# The merge
# --------------------------------------------------------------------------- #


class TestPreprintAndPublishedVersionAreOneRow:
    """预印本/正式版去重正确."""

    def test_the_pair_collapses_to_a_single_result(self):
        assert len(deduplicate_papers(residual_pair())) == 1

    def test_the_collapse_happens_on_a_shared_identifier(self):
        report = deduplicate_papers_report(residual_pair())

        assert len(report["merged"]) == 1
        # Either tier is a proof; asserting on the *weaker* one that must also
        # work is the point, so both are checked by the two tests below.
        assert report["merged"][0]["basis"] in {"shared_doi", "shared_arxiv"}
        assert report["merged"][0]["basis_zh"]

    def test_the_survivor_keeps_the_identifiers_of_both_sides(self):
        kept = deduplicate_papers(residual_pair())[0]

        # The DOI came from S2's externalIds and the arXiv id from the
        # preprint row. Merging must not throw either away, because the whole
        # point of merging is that the record can now be found again.
        assert kept["doi"] == PUBLISHED_DOI.lower()
        assert kept["arxiv_id"] == PREPRINT_ARXIV
        assert kept["identity_key"] == f"doi:{PUBLISHED_DOI.lower()}"

    def test_it_collapses_even_when_the_titles_were_re_typeset(self):
        published = normalize_paper_record(
            "Semantic Scholar",
            s2_payload(title=f"{RESIDUAL_TITLE}."),
        )
        preprint = normalize_paper_record(
            "arXiv", arxiv_payload(title=f"{RESIDUAL_TITLE} (extended version)")
        )

        assert len(deduplicate_papers([published, preprint])) == 1

    def test_a_preprint_and_its_published_version_merge_in_either_order(self):
        pair = residual_pair()

        assert len(deduplicate_papers(pair)) == len(deduplicate_papers(list(reversed(pair))))

    def test_an_arxiv_version_suffix_does_not_split_the_pair(self):
        # v1 and v2 are one submission. A user who searched in January and
        # again in April must not see the paper twice.
        first = normalize_paper_record("arXiv", arxiv_payload(arxiv_id="2301.00001"))
        revised = normalize_paper_record("arXiv", arxiv_payload(arxiv_id="2301.00001"))
        revised["paper_id"] = "http://arxiv.org/abs/2301.00001v3"
        revised["url"] = "http://arxiv.org/abs/2301.00001v3"

        assert len(deduplicate_papers([first, revised])) == 1

    def test_the_two_sides_are_genuinely_the_same_paper_by_the_identity_rules(self):
        # Asserted through the identity layer too, so a failure here names the
        # actual cause rather than "the list had two rows".
        published, preprint = (identity_from_record(row) for row in residual_pair())
        match = compare_identities(published, preprint)

        assert match.same_paper is True
        assert match.confidence == "exact"


# --------------------------------------------------------------------------- #
# The other failure direction
# --------------------------------------------------------------------------- #


class TestDifferentPapersThatShareATitleAreNotMerged:
    """The false merge is worse than the false split, and it is tested first."""

    def test_two_different_papers_named_alike_stay_separate(self):
        first = normalize_paper_record(
            "Semantic Scholar",
            s2_payload("a" * 40, title="Attention Is All You Need", year=2017, doi="10.5555/3295222"),
        )
        second = normalize_paper_record(
            "arXiv",
            arxiv_payload(arxiv_id="1706.03762", title="Attention Is All You Need"),
        )
        second["doi"] = "10.5555/reprint.3295222"

        assert len(deduplicate_papers([first, second])) == 2

    def test_a_preprint_and_a_same_named_survey_do_not_merge(self):
        # The canonical false merge: a survey reusing a famous paper's title,
        # with a different author and a year far enough away.
        famous = normalize_paper_record(
            "Semantic Scholar",
            s2_payload("a" * 40, title="Deep Residual Learning for Image Recognition", doi="10.5555/resnet"),
        )
        survey = normalize_paper_record(
            "Semantic Scholar",
            s2_payload(
                "b" * 40,
                title="Deep Residual Learning for Image Recognition",
                year=2023,
                doi="10.5555/survey",
                authors=["Somebody Else"],
            ),
        )

        assert len(deduplicate_papers([famous, survey])) == 2

    def test_conflicting_dois_are_a_hard_no_even_with_an_identical_title(self):
        left = normalize_paper_record(
            "Semantic Scholar",
            s2_payload("a" * 40, title="Shared Title Here", doi="10.5555/left"),
        )
        right = normalize_paper_record(
            "Semantic Scholar",
            s2_payload("b" * 40, title="Shared Title Here", doi="10.5555/right"),
        )

        kept = deduplicate_papers([left, right])

        assert len(kept) == 2
        assert {item["doi"] for item in kept} == {"10.5555/left", "10.5555/right"}

    def test_two_generic_titled_records_are_never_merged(self):
        # "Editorial" identifies nothing. Merging on it would fuse every
        # journal's editorial into one row.
        first = normalize_paper_record(
            "Semantic Scholar",
            s2_payload("a" * 40, title="Editorial", year=2020, doi="10.5555/e1"),
        )
        second = normalize_paper_record(
            "Semantic Scholar",
            s2_payload("b" * 40, title="Editorial", year=2021, doi="10.5555/e2"),
        )

        assert len(deduplicate_papers([first, second])) == 2


# --------------------------------------------------------------------------- #
# The ambiguous case, resolved the documented way
# --------------------------------------------------------------------------- #


class TestAnUnprovableMatchIsKeptAndReported:
    """``probable`` is a state this product renders, not one it guesses at.

    The arXiv row and the publisher row of one paper, when the arXiv record
    carries no journal DOI and the publisher record carries no arXiv id. The
    title and the author say one paper; nothing *proves* it. The list keeps
    both rows and names the pair, because the alternative — silently fusing
    them — is the failure the user cannot detect and cannot undo.
    """

    def _unprovable_pair(self) -> list[dict]:
        preprint = normalize_paper_record(
            "arXiv", arxiv_payload(published="2015-12-11T18:41:45Z", doi="")
        )
        published = normalize_paper_record(
            "Semantic Scholar",
            s2_payload("f" * 40, year=2016, doi=PUBLISHED_DOI, arxiv_id=""),
        )
        return [preprint, published]

    def test_both_records_are_kept(self):
        assert len(deduplicate_papers(self._unprovable_pair())) == 2

    def test_the_pair_is_reported_with_the_evidence_and_an_explanation(self):
        possible = deduplicate_papers_report(self._unprovable_pair())["possible_duplicates"]

        assert len(possible) == 1
        entry = possible[0]
        assert entry["title"] == RESIDUAL_TITLE
        assert entry["other_title"] == RESIDUAL_TITLE
        # The basis names *why* it is suspected, so the UI can show the reason
        # rather than an unexplained badge.
        assert entry["basis"]
        assert entry["basis_zh"]
        assert entry["note_zh"]

    def test_the_verdict_is_really_only_probable(self):
        preprint, published = (identity_from_record(row) for row in self._unprovable_pair())
        match = compare_identities(preprint, published)

        assert match.same_paper is True
        assert match.confidence == "probable"
        assert match.requires_confirmation is True

    def test_no_argument_can_make_a_probable_match_collapse(self):
        # The false-merge direction must not be reachable by accident, so the
        # whole API is asserted: there is no threshold, no flag, no second
        # function that merges on weaker evidence.
        pair = self._unprovable_pair()

        assert len(deduplicate_papers(pair)) == 2
        assert len(deduplicate_papers_report(pair)["items"]) == 2
        assert COLLAPSE_CONFIDENCE == "exact"
        with pytest.raises(TypeError):
            deduplicate_papers(pair, confidence="probable")

    def test_records_that_are_not_even_probably_the_same_are_not_reported(self):
        possible = deduplicate_papers_report(
            [
                normalize_paper_record("arXiv", arxiv_payload(arxiv_id="2301.00001")),
                normalize_paper_record(
                    "arXiv", arxiv_payload(arxiv_id="2302.00002", title="Something Else Entirely")
                ),
            ]
        )["possible_duplicates"]

        assert possible == []


# --------------------------------------------------------------------------- #
# Identifiers surviving normalization
# --------------------------------------------------------------------------- #


class TestIdentifiersSurviveNormalization:
    """DOI 丢失可回查 — the identifier must not be lost on the way in.

    A record that arrives *with* a DOI and leaves *without* one turns a local
    lookup into a network round trip, against a rate-limited Crossref, to
    recover a value the provider already sent. These are the assertions that
    make that round trip unnecessary.
    """

    def test_an_s2_record_keeps_the_doi_from_external_ids(self):
        record = normalize_paper_record("Semantic Scholar", s2_payload())

        assert record["doi"] == PUBLISHED_DOI.lower()

    def test_an_s2_record_keeps_the_arxiv_id_from_external_ids(self):
        record = normalize_paper_record("Semantic Scholar", s2_payload())

        assert record["arxiv_id"] == PREPRINT_ARXIV

    def test_an_s2_record_keeps_its_own_paper_id(self):
        record = normalize_paper_record("Semantic Scholar", s2_payload())

        assert record["s2_paper_id"] == "649def34f8be52c8b66281af98ae884c09aef38b"
        assert record["identity_key"].startswith("doi:")

    def test_the_identifier_fields_are_always_present(self):
        # Present-but-empty rather than absent: a consumer must not have to
        # distinguish "the provider said there is none" from "we never looked".
        record = normalize_paper_record("arXiv", arxiv_payload())

        for field in ("doi", "arxiv_id", "s2_paper_id", "openalex_id"):
            assert field in record
        assert record["doi"] == ""
        assert record["arxiv_id"] == PREPRINT_ARXIV

    def test_a_corpus_id_in_the_doi_slot_is_not_taken_for_a_doi(self):
        # S2 really does put ``CorpusId:2215645`` in the DOI field for records
        # with no real DOI. A hand-rolled "if the value is non-empty, it is a
        # DOI" would make that paper permanently un-deduplicatable.
        payload = s2_payload("b" * 40, doi="CorpusId:2215645", arxiv_id="")
        payload["title"] = "A Paper With No Real Identifier At All"

        record = normalize_paper_record("Semantic Scholar", payload)

        assert record["doi"] == ""
        # The paperId is still usable, so the record is not left unidentified.
        assert record["s2_paper_id"] == "b" * 40

    def test_an_arxiv_url_in_paper_id_becomes_an_arxiv_id(self):
        record = normalize_paper_record(
            "arXiv", arxiv_payload(arxiv_id="2301.00001")
        )

        assert record["arxiv_id"] == "2301.00001"
        assert record["identity_key"] == "arxiv:2301.00001"

    def test_a_declared_journal_doi_lets_the_two_sides_match_exactly(self):
        # The strongest end of T3.1: with the arXiv record carrying the
        # journal DOI it declares, the two rows agree on an identifier rather
        # than merely on a title, and the merge is proven rather than probable.
        preprint = normalize_paper_record(
            "arXiv", arxiv_payload(published="2015-12-11T18:41:45Z", doi=PUBLISHED_DOI)
        )
        published = normalize_paper_record("Semantic Scholar", s2_payload())

        assert len(deduplicate_papers([preprint, published])) == 1
        assert deduplicate_papers_report([preprint, published])["merged"][0]["basis"] == "shared_doi"

    def test_the_identifier_survives_a_round_trip_through_the_reading_queue(self):
        # The queue is where a record stops being temporary. A DOI dropped on
        # save is a DOI the user will have to look up again by hand.
        saved = normalize_paper_collection(
            [normalize_paper_record("Semantic Scholar", s2_payload())]
        )

        assert saved[0]["doi"] == PUBLISHED_DOI.lower()
        assert saved[0]["arxiv_id"] == PREPRINT_ARXIV
        assert saved[0]["identity_key"] == f"doi:{PUBLISHED_DOI.lower()}"


# --------------------------------------------------------------------------- #
# The reading queue
# --------------------------------------------------------------------------- #


class TestReadingQueueNormalization:
    def test_the_preprint_pair_collapses_in_the_queue_too(self):
        kept = normalize_paper_collection(residual_pair())

        assert len(kept) == 1
        assert kept[0]["doi"] == PUBLISHED_DOI.lower()
        assert kept[0]["arxiv_id"] == PREPRINT_ARXIV

    def test_different_papers_sharing_a_title_stay_separate_in_the_queue(self):
        kept = normalize_paper_collection(
            [
                normalize_paper_record("Semantic Scholar", s2_payload("a" * 40, doi="10.5555/one")),
                normalize_paper_record("Semantic Scholar", s2_payload("b" * 40, doi="10.5555/two")),
            ]
        )

        assert len(kept) == 2

    def test_an_identical_link_does_not_produce_two_rows(self):
        # A record with no identifier and no author can only be recognised by
        # its link. That is a *weaker* signal and is used only where identity
        # is silent — see the note on ``_run_dedupe``.
        row = {
            "source": "Custom",
            "title": "A Paper With Barely Any Metadata",
            "url": "https://example.org/paper/1",
        }
        twin = {**row, "url": "https://www.example.org/paper/1/"}

        assert len(normalize_paper_collection([row, twin])) == 1

    def test_a_link_never_overrides_a_conflicting_identity(self):
        # Two different DOIs that happen to share a landing page is a metadata
        # problem the user should see, not a duplicate to hide.
        first = {
            "source": "Custom",
            "title": "First Paper",
            "url": "https://example.org/paper/shared",
            "doi": "10.5555/one",
        }
        second = {**first, "doi": "10.5555/two"}

        assert len(normalize_paper_collection([first, second])) == 2

    def test_titleless_rows_are_dropped(self):
        assert normalize_paper_collection([{"source": "X", "url": "https://example.org/1"}]) == []

    def test_junk_input_is_ignored_rather_than_raised_on(self):
        assert normalize_paper_collection(None) == []
        assert normalize_paper_collection("not a list") == []
        assert normalize_paper_collection([None, 3, "text"]) == []

    def test_the_queue_length_is_capped(self):
        rows = [
            normalize_paper_record(
                "Semantic Scholar", s2_payload(f"{i:040d}", doi=f"10.5555/paper{i}")
            )
            for i in range(10)
        ]

        assert len(normalize_paper_collection(rows, max_items=4)) == 4


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #


class TestTruncationReturnsWhatWasAskedFor:
    """The old order — deduplicate, then ``[:limit]`` — quietly lost results.

    Ask for 8 with two providers that overlap on 4 papers, collapse the 4
    pairs, and the user was shown 4. The fix over-fetches per provider so the
    merged list can still be filled; a short list now means the providers had
    nothing more, which is a fact worth being able to show.
    """

    @staticmethod
    def _overlapping_sources(paired: bool, per_source: int = 8):
        """S2 rows and arXiv rows, some of which are one paper in two listings.

        Every row gets its *own* arXiv id. That is not decoration: an arXiv id
        is a hard identifier, so a fixture that reused one across supposedly
        different papers would be asserting on a conflict rather than on a
        merge, and the rows would correctly refuse to collapse.
        """

        def s2_rows(_query, limit):
            return [
                normalize_paper_record(
                    "Semantic Scholar",
                    s2_payload(
                        f"{1000 + i:040d}",
                        title=f"Distinct Synthetic Paper Number {i} on Graph Structured Topics",
                        doi=f"10.1234/s2.{i}",
                        arxiv_id=f"2601.0{i:04d}",
                    ),
                )
                for i in range(min(limit, per_source))
            ]

        def arxiv_rows(_query, limit):
            rows = []
            for i in range(min(limit, per_source)):
                # Every third row is the published twin of the S2 row with the
                # same index, and carries its DOI, so the pair is provable.
                is_twin = paired and i % 3 == 0
                # A twin keeps the S2 row's arXiv id — that is what makes it
                # the same submission. A non-twin must be a *different* one:
                # reusing an arXiv id here would make these rows duplicates
                # again, and the test named "duplicate free" would not be.
                rows.append(
                    normalize_paper_record(
                        "arXiv",
                        arxiv_payload(
                            arxiv_id=f"2601.0{i:04d}" if is_twin else f"2509.1{i:04d}",
                            title=(
                                f"Distinct Synthetic Paper Number {i} on Graph Structured Topics"
                                if is_twin
                                else f"An Arxiv Only Preprint Number {i} about Something Else"
                            ),
                            doi=f"10.1234/s2.{i}" if is_twin else "",
                        ),
                    )
                )
            return rows

        return s2_rows, arxiv_rows

    def _install(self, monkeypatch, paired: bool, per_source: int):
        """Point the search route at the fixture providers."""
        s2_rows, arxiv_rows = self._overlapping_sources(paired=paired, per_source=per_source)
        monkeypatch.setattr(papers, "search_semantic_scholar_papers", s2_rows)
        monkeypatch.setattr(papers, "search_arxiv_papers", arxiv_rows)

    def test_a_fully_overlapping_search_still_returns_the_requested_count(self, monkeypatch):
        self._install(monkeypatch, paired=True, per_source=8)

        result = papers.search_papers("graph structured topics", limit=8)

        assert len(result["items"]) == 8
        # Something really was collapsed, so the count above is not vacuous.
        assert result["merged_duplicates"]

    def test_each_provider_is_asked_for_more_than_the_user_wants(self, monkeypatch):
        requested: list[int] = []

        def record_request(_query, limit):
            requested.append(limit)
            return []

        monkeypatch.setattr(papers, "search_semantic_scholar_papers", record_request)
        monkeypatch.setattr(papers, "search_arxiv_papers", record_request)

        papers.search_papers("anything", limit=8)

        assert requested == [16, 16]

    def test_the_per_source_request_is_capped(self, monkeypatch):
        requested: list[int] = []

        def record_request(_query, limit):
            requested.append(limit)
            return []

        monkeypatch.setattr(papers, "search_semantic_scholar_papers", record_request)
        monkeypatch.setattr(papers, "search_arxiv_papers", record_request)

        papers.search_papers("anything", limit=20)

        # Semantic Scholar's search endpoint rejects a limit above 100, and a
        # large multiplier on a large configured limit is load on a shared
        # rate-limited API in exchange for rows the user will not scroll to.
        assert all(limit <= papers.SOURCE_MAX_RESULTS for limit in requested)

    def test_a_duplicate_free_search_still_returns_the_requested_count(self, monkeypatch):
        self._install(monkeypatch, paired=False, per_source=12)

        result = papers.search_papers("graph structured topics", limit=8)

        assert len(result["items"]) == 8
        assert result["merged_duplicates"] == []

    def test_the_cap_is_still_enforced(self, monkeypatch):
        self._install(monkeypatch, paired=False, per_source=30)

        result = papers.search_papers("graph structured topics", limit=5)

        assert len(result["items"]) == 5

    def test_both_providers_reach_the_user(self, monkeypatch):
        # A concatenated list capped at N is routinely N results from one
        # provider and none from the other, purely because that provider was
        # asked first. Which one answered first is not a ranking.
        self._install(monkeypatch, paired=False, per_source=8)

        result = papers.search_papers("graph structured topics", limit=8)

        assert {item["source"] for item in result["items"]} == {"Semantic Scholar", "arXiv"}

    def test_a_shorter_list_means_the_providers_ran_out(self, monkeypatch):
        monkeypatch.setattr(papers, "search_semantic_scholar_papers", lambda q, limit: [])
        monkeypatch.setattr(papers, "search_arxiv_papers", lambda q, limit: [])

        assert papers.search_papers("nothing matches this", limit=8)["items"] == []


# --------------------------------------------------------------------------- #
# Degenerate input
# --------------------------------------------------------------------------- #


class TestDegenerateInputDoesNotCrashDeduplication:
    def test_a_source_that_returns_nothing_is_not_an_error(self, monkeypatch):
        monkeypatch.setattr(papers, "search_semantic_scholar_papers", lambda q, limit: [
            normalize_paper_record("Semantic Scholar", s2_payload())
        ])
        monkeypatch.setattr(papers, "search_arxiv_papers", lambda q, limit: [])

        result = papers.search_papers("residual learning", limit=8)

        assert len(result["items"]) == 1
        assert result["errors"] == []

    def test_both_sources_returning_nothing(self, monkeypatch):
        monkeypatch.setattr(papers, "search_semantic_scholar_papers", lambda q, limit: [])
        monkeypatch.setattr(papers, "search_arxiv_papers", lambda q, limit: [])

        result = papers.search_papers("nothing at all", limit=8)

        assert result["items"] == []
        assert result["errors"] == []
        assert result["possible_duplicates"] == []

    def test_one_failing_source_is_reported_without_losing_the_other(self, monkeypatch):
        def boom(_query, _limit):
            raise TimeoutError("read timed out")

        monkeypatch.setattr(papers, "search_semantic_scholar_papers", boom)
        monkeypatch.setattr(
            papers,
            "search_arxiv_papers",
            lambda q, limit: [
                normalize_paper_record("arXiv", arxiv_payload(arxiv_id="2301.00001"))
            ],
        )

        result = papers.search_papers("residual learning", limit=8)

        assert len(result["items"]) == 1
        assert len(result["errors"]) == 1
        assert "timed out" in result["errors"][0]

    def test_deduplication_of_nothing(self):
        assert deduplicate_papers([]) == []
        assert deduplicate_papers(None) == []

    def test_deduplication_of_junk(self):
        # The queue path is fed by a browser, so non-dicts arrive.
        assert deduplicate_papers([None, 3, "text"]) == []

    def test_a_row_with_no_title_at_all_does_not_break_the_pass(self):
        junk = {"source": "Custom", "title": "", "doi": "10.5555/untitled"}
        good = normalize_paper_record("arXiv", arxiv_payload(arxiv_id="2301.00001"))

        assert len(deduplicate_papers([good, junk])) == 2

    def test_a_record_with_no_usable_identity_is_still_listed(self):
        # Nothing to compare it on, so nothing to merge it with. Dropping it
        # would lose a result the provider actually returned.
        thin = {"source": "Custom", "title": "A Paper With No Identifiers At All"}

        kept = deduplicate_papers([thin])

        assert len(kept) == 1
        assert kept[0]["title"] == thin["title"]
        # The key is present and empty rather than absent: "this record
        # identifies nothing" is a real state, and one a caller can read.
        assert kept[0]["identity_key"] == ""

    def test_the_search_route_keeps_every_response_key(self, monkeypatch):
        monkeypatch.setattr(papers, "search_semantic_scholar_papers", lambda q, limit: [])
        monkeypatch.setattr(papers, "search_arxiv_papers", lambda q, limit: [])

        result = papers.search_papers("anything", limit=4)

        assert {"query", "items", "errors"} <= set(result)
        # A client that does not know about the new keys still works; a client
        # that does has somewhere to render the ambiguity.
        assert isinstance(result["possible_duplicates"], list)
        assert isinstance(result["merged_duplicates"], list)
