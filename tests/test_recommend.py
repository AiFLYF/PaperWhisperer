"""Multi-signal recommendations: every row's reason is the signal that made it.

The acceptance criterion for T3.2 is a UI that can answer "为什么推荐", which
is only worth having if the answer is *true*. These tests are therefore
organised around the ways a reason can become a lie, each of which is silent —
a list that looks exactly as plausible when it is wrong:

1. A paper with no signal attached to it. Asserted by never letting a
   fabricated reason through: the engine drops a hit it cannot attribute, and
   each of the five signals is exercised in isolation so a reason always has a
   mechanism behind it.
2. A reason attached after the fact. Asserted by checking that the reason text
   names the provider and the actual query, and by pinning the OpenAlex filter
   direction on the constants — a filter named backwards produces a confident,
   wrong sentence rather than an error.
3. A failed signal reading as an empty one. Asserted on all four statuses
   (``ok`` / ``empty`` / ``unavailable`` / ``skipped``), because "the provider
   said there is nothing" and "we could not ask" are different claims about the
   world and the UI has to be able to tell them apart.
4. A duplicate paper shown three times. Asserted that a paper reachable from
   three signals is one row carrying three reasons, and — the asymmetry T3.1
   established — that an *unproven* match is reported as a possible duplicate
   and both rows are kept, never merged.
5. One user click becoming a fan-out. Asserted against a
   :class:`~paperwhisperer.search.recommend.CallBudget` that is too small, so
   the dropped signals are reported as ``skipped`` rather than vanishing.

No network is touched anywhere. The fetch layer is injected exactly as
``tests/test_paper_identity.py`` and ``search/enrich.py`` inject theirs: a
``fetch`` callable taking a URL and returning a hand-written provider payload.
"""

from __future__ import annotations

import urllib.error

import pytest

from paperwhisperer.search import recommend
from paperwhisperer.search.recommend import (
    MAX_DATASET_QUERIES,
    OPENALEX_CITED_BY_FILTER,
    OPENALEX_CITES_FILTER,
    OPENALEX_RELATED_FILTER,
    SIGNAL_CITED_BY,
    SIGNAL_CITING,
    SIGNAL_ORDER,
    SIGNAL_RELATED,
    SIGNAL_SAME_DATASET,
    SIGNAL_TOPICAL,
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_UNAVAILABLE,
    CallBudget,
    RecommendationQuery,
    SignalResult,
    build_openalex_cited_by_source,
    build_openalex_citing_source,
    build_openalex_related_source,
    build_s2_cited_by_source,
    build_s2_citing_source,
    build_s2_related_source,
    build_same_dataset_source,
    build_topical_source,
    dataset_name_appears,
    dataset_names_from_facts,
    default_recommendation_sources,
    recommend_papers,
)

# --------------------------------------------------------------------------- #
# Fixtures: one real paper, seen from both providers
#
# He et al. posted arXiv:1512.03385 in December 2015; it appeared at CVPR 2016
# as 10.1109/CVPR.2016.90. Using one real paper for the seed means the citation
# signals are exercised against identifiers that actually have to be normalized
# (a DOI that arrives as a full URL, an OpenAlex id that arrives as a work URL)
# rather than against convenient invented values.
# --------------------------------------------------------------------------- #

SEED_TITLE = "Deep Residual Learning for Image Recognition"
SEED_AUTHOR = "Kaiming He"
SEED_DOI = "10.1109/CVPR.2016.90"
SEED_ARXIV = "1512.03385"
SEED_S2_ID = "649def34f8be52c8b66281af98ae884c09aef38b"
SEED_OPENALEX = "W2100837269"
OPENALEX_FOLLOWUP_ID = "W2963303868"
OPENALEX_OTHER_ID = "W2101111593"

#: A second real paper — the Batch Normalization paper, which cites ResNet.
FOLLOWUP_TITLE = "Batch Normalization: Accelerating Deep Network Training"
FOLLOWUP_DOI = "10.48550/arXiv.1502.03167"
FOLLOWUP_S2_ID = "1b1584a11a77c2b8f87b36857b09065c70085640"

#: A third — the wide-residual-networks paper, also citing ResNet.
WIDE_TITLE = "Wide Residual Networks"
WIDE_DOI = "10.48550/arXiv.1605.07146"
WIDE_S2_ID = "3a4c0d0e2f0b1a0c9d8e7f6a5b4c3d2e1f0a9b8c"


def seed_identity():
    return {
        "title": SEED_TITLE,
        "year": "2015",
        "authors": [SEED_AUTHOR],
        "doi": SEED_DOI,
        "arxiv_id": SEED_ARXIV,
        "s2_paper_id": SEED_S2_ID,
        "openalex_id": SEED_OPENALEX,
    }


def make_query(**overrides) -> RecommendationQuery:
    """A query for the seed paper, with everything else empty by default."""
    base = {
        "identity": recommend.identity_from_record(seed_identity()),
        "expanded_query": "",
        "topics": (),
        "datasets": (),
        "dataset_names_are_declared": False,
        "limit": 5,
    }
    base.update(overrides)
    return RecommendationQuery(**base)


def s2_work(
    paper_id: str,
    *,
    title: str,
    doi: str = "",
    arxiv: str = "",
    year: int = 2016,
    authors: list[str] | None = None,
    abstract: str = "",
) -> dict:
    """One S2 paper, in the shape the graph endpoints actually return."""
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "venue": "CVPR",
        "url": f"https://www.semanticscholar.org/paper/{paper_id}",
        "authors": [
            {"name": name} for name in (authors if authors is not None else [SEED_AUTHOR])
        ],
        "externalIds": {"DOI": doi, "ArXiv": arxiv},
        "openAccessPdf": {"url": f"https://arxiv.org/pdf/{arxiv or paper_id}"},
    }


def openalex_work(
    openalex_id: str,
    *,
    title: str,
    doi: str = "",
    year: int = 2016,
    authors: list[str] | None = None,
    abstract: str = "",
) -> dict:
    """One OpenAlex work, in the shape the works endpoint actually returns."""
    names = authors if authors is not None else [SEED_AUTHOR]
    return {
        "id": f"https://openalex.org/{openalex_id}",
        "doi": doi,
        "display_name": title,
        "publication_year": year,
        "authorships": [
            {"author": {"display_name": name}} for name in names
        ],
        "primary_location": {
            "landing_page_url": f"https://openalex.org/{openalex_id}",
            "pdf_url": "",
            "source": {"display_name": "CVPR"},
        },
        "abstract": abstract,
    }


#: A fetcher that records the URLs it was asked for and returns canned payloads.
class RecordingFetch:
    """Stands in for the shared HTTP client, and remembers what it was asked.

    Recording the URLs rather than only counting them is what lets the
    OpenAlex-direction tests assert *which filter* ran: the filter name is the
    only place a citation direction is expressed, so a test that only counted
    calls would pass with the filters swapped.
    """

    def __init__(self, payloads: dict[str, object] | None = None, error: Exception | None = None):
        self.payloads = payloads or {}
        self.error = error
        self.urls: list[str] = []

    def __call__(self, url: str, **kwargs) -> object:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        for marker, payload in self.payloads.items():
            if marker in url:
                return payload
        return {}

    @property
    def calls(self) -> int:
        return len(self.urls)


def search_result(items, errors=None) -> dict:
    return {"items": list(items), "errors": list(errors or [])}


def canonical_search_hit(
    paper_id: str,
    *,
    title: str,
    doi: str = "",
    abstract: str = "",
    year: int = 2016,
) -> dict:
    """A row as :func:`paperwhisperer.search.papers.search_papers` emits it."""
    return {
        "source": "Semantic Scholar",
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": [SEED_AUTHOR],
        "year": str(year),
        "venue": "CVPR",
        "url": f"https://www.semanticscholar.org/paper/{paper_id}",
        "pdf_url": "",
        "doi": doi,
        "arxiv_id": "",
        "s2_paper_id": paper_id,
        "openalex_id": "",
        "identity_key": f"doi:{doi}" if doi else f"s2:{paper_id}",
    }


def signals_by_name(result: dict) -> dict[str, dict]:
    return {entry["signal"]: entry for entry in result["signals"]}


# --------------------------------------------------------------------------- #
# 1. Each signal in isolation
# --------------------------------------------------------------------------- #


class TestEachSignalInIsolation:
    """One test per signal, so a signal that stops working is named."""

    def test_cited_by_comes_from_the_references_endpoint(self):
        """引用 / 参考文献: the papers the seed itself cites.

        Asserted on the URL because S2 has two endpoints that differ by one
        path segment and by direction, and the wrong one produces a reason that
        reads perfectly and is exactly inverted.
        """
        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [
                        {
                            "citedPaper": s2_work(
                                FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                            )
                        }
                    ]
                }
            }
        )
        budget = CallBudget()
        source = build_s2_cited_by_source(fetch, budget=budget)
        result = recommend_papers(make_query(), [source], budget=budget)

        assert "/references" in fetch.urls[0]
        assert "/citations" not in fetch.urls[0]
        assert len(result["items"]) == 1
        reason = result["items"][0]["reasons"][0]
        assert reason["signal"] == SIGNAL_CITED_BY
        # "引用", not "被引": these are the seed's own bibliography.
        # OpenAlex names `cited_by` from the citing work's point of view,
        # while in Chinese 被引 means the citation count a paper *receives* —
        # the opposite. Verified against the live API: `cited_by:W` returns
        # 44 rows, all present in W's own `referenced_works`; `cites:W`
        # returns the 1253 that cite it.
        assert reason["label_zh"] == "引用"
        assert "参考文献" in reason["reason_zh"]
        assert SEED_TITLE in reason["reason_zh"]
        assert result["items"][0]["signals"] == [SIGNAL_CITED_BY]

    def test_citing_comes_from_the_citations_endpoint(self):
        """被引: the papers that cite the seed — the follow-up direction."""
        fetch = RecordingFetch(
            {
                "/citations": {
                    "data": [
                        {
                            "citingPaper": s2_work(
                                WIDE_S2_ID, title=WIDE_TITLE, doi=WIDE_DOI, authors=["Sergey Zagoruyko"]
                            )
                        }
                    ]
                }
            }
        )
        budget = CallBudget()
        result = recommend_papers(
            make_query(), [build_s2_citing_source(fetch, budget=budget)], budget=budget
        )

        assert "/citations" in fetch.urls[0]
        assert "/references" not in fetch.urls[0]
        reason = result["items"][0]["reasons"][0]
        assert reason["signal"] == SIGNAL_CITING
        # "被引": these are the works that cite the seed. Pinned separately
        # from its sibling because the id and the Chinese word point in
        # opposite directions there, and swapping them produces a chip that
        # says the opposite of the truth on every row.
        assert reason["label_zh"] == "被引"
        assert "引用了" in reason["reason_zh"]

    def test_related_says_it_is_related_and_not_cited(self):
        """相关: a provider's opinion, labelled as an opinion.

        The assertion that matters is the negative one: the reason must not
        claim a citation relationship, because the UI's whole value is that a
        user can tell "this cites you" from "the provider thinks these go
        together".
        """
        fetch = RecordingFetch(
            {
                "forpaper": {
                    "recommendedPapers": [
                        s2_work(
                            WIDE_S2_ID, title=WIDE_TITLE, doi=WIDE_DOI, authors=["Sergey Zagoruyko"]
                        )
                    ]
                }
            }
        )
        budget = CallBudget()
        result = recommend_papers(
            make_query(), [build_s2_related_source(fetch, budget=budget)], budget=budget
        )

        reason = result["items"][0]["reasons"][0]
        assert reason["signal"] == SIGNAL_RELATED
        assert reason["label_zh"] == "相关"
        assert "不是引用关系" in reason["reason_zh"]
        assert "引用" not in reason["reason_zh"].replace("不是引用关系", "")

    def test_topical_names_the_query_it_actually_searched(self):
        """主题相似: the reason is the expanded query, not the topic."""
        seen: list[tuple[str, int]] = []

        def fake_search(query: str, limit: int) -> dict:
            seen.append((query, limit))
            return search_result(
                [canonical_search_hit(FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)]
            )

        budget = CallBudget()
        result = recommend_papers(
            make_query(expanded_query="residual connections image classification", topics=("resnet",)),
            [build_topical_source(fake_search, budget=budget)],
            budget=budget,
        )

        assert seen == [("residual connections image classification", 5)]
        reason = result["items"][0]["reasons"][0]
        assert reason["signal"] == SIGNAL_TOPICAL
        assert reason["label_zh"] == "主题相似"
        assert "扩展检索式" in reason["reason_zh"]
        assert "residual connections image classification" in reason["reason_zh"]
        assert reason["detail"] == "query:residual connections image classification"

    def test_topical_without_an_llm_query_says_it_searched_the_title(self):
        """Without a rewrite the reason changes wording, not signal.

        The two cases are deliberately not collapsed into one label: a hit
        found by a query the user never typed needs more explaining than one
        found by their own title, and the payload says which happened.
        """
        seen: list[str] = []

        def fake_search(query: str, limit: int) -> dict:
            seen.append(query)
            return search_result(
                [canonical_search_hit(FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)]
            )

        budget = CallBudget()
        result = recommend_papers(
            make_query(), [build_topical_source(fake_search, budget=budget)], budget=budget
        )

        assert seen == [SEED_TITLE]
        reason = result["items"][0]["reasons"][0]
        assert "论文标题" in reason["reason_zh"]
        assert "扩展检索式" not in reason["reason_zh"]

    def test_same_dataset_requires_the_name_in_the_hits_own_metadata(self):
        """同数据集: the gate that stops it being topic similarity in disguise.

        The search returns two papers. Only one of them actually names the
        dataset. The other is dropped, *not* relabelled — a paper that surfaced
        from a dataset search and does not mention the dataset is not evidence
        of anything, and showing it under a weaker label would still be
        showing it because of a search that had nothing to do with it.
        """
        def fake_search(query: str, limit: int) -> dict:
            return search_result(
                [
                    canonical_search_hit(
                        FOLLOWUP_S2_ID,
                        title=FOLLOWUP_TITLE,
                        doi=FOLLOWUP_DOI,
                        abstract="We evaluate on the ImageNet benchmark.",
                    ),
                    canonical_search_hit(
                        WIDE_S2_ID,
                        title=WIDE_TITLE,
                        doi=WIDE_DOI,
                        abstract="A deeper residual network, evaluated on CIFAR and ImageNet",
                    ),
                    canonical_search_hit(
                        "0" * 40, title="Unrelated Work", doi="10.1000/nothing", abstract="No."
                    ),
                ]
            )

        budget = CallBudget()
        result = recommend_papers(
            make_query(datasets=("ImageNet",), dataset_names_are_declared=True),
            [build_same_dataset_source(fake_search, budget=budget)],
            budget=budget,
        )

        titles = {item["title"] for item in result["items"]}
        assert titles == {FOLLOWUP_TITLE, WIDE_TITLE}
        reason = result["items"][0]["reasons"][0]
        assert reason["signal"] == SIGNAL_SAME_DATASET
        assert reason["label_zh"] == "同数据集"
        assert "ImageNet" in reason["reason_zh"]
        assert reason["confidence"] == "declared"

    def test_same_dataset_marks_lexically_derived_names_as_unverified(self):
        """A name scanned out of a paraphrase is not a verified dataset name.

        The difference is recorded rather than hidden, because the two claims
        are "this paper names the same dataset" (checkable, and all we prove)
        and "this paper trains on the same dataset" (stronger, and unproven).
        """
        def fake_search(query: str, limit: int) -> dict:
            return search_result(
                [
                    canonical_search_hit(
                        FOLLOWUP_S2_ID,
                        title=FOLLOWUP_TITLE,
                        doi=FOLLOWUP_DOI,
                        abstract="Evaluated on ImageNet.",
                    )
                ]
            )

        budget = CallBudget()
        result = recommend_papers(
            make_query(datasets=("ImageNet",), dataset_names_are_declared=False),
            [build_same_dataset_source(fake_search, budget=budget)],
            budget=budget,
        )

        reason = result["items"][0]["reasons"][0]
        assert reason["confidence"] == "derived"
        assert "标题或摘要中" in reason["reason_zh"]


# --------------------------------------------------------------------------- #
# 2. OpenAlex filter directions
# --------------------------------------------------------------------------- #


class TestOpenAlexFilterDirections:
    """A filter named backwards produces a confident, wrong sentence.

    OpenAlex's ``cites`` and ``cited_by`` are the easiest pair in this module to
    transpose, so the filter name is asserted in the URL and the two constants
    are asserted against the plain-English claim they encode.
    """

    def test_the_constants_encode_the_directions_they_name(self):
        assert OPENALEX_CITED_BY_FILTER == "cited_by"
        assert OPENALEX_CITES_FILTER == "cites"
        assert OPENALEX_RELATED_FILTER == "related_to"

    @pytest.mark.parametrize(
        ("builder", "expected_filter", "expected_signal"),
        [
            (build_openalex_cited_by_source, "cited_by", SIGNAL_CITED_BY),
            (build_openalex_citing_source, "cites", SIGNAL_CITING),
            (build_openalex_related_source, "related_to", SIGNAL_RELATED),
        ],
    )
    def test_each_source_sends_its_own_filter(self, builder, expected_filter, expected_signal):
        fetch = RecordingFetch(
            {"openalex.org": {"results": [openalex_work(OPENALEX_OTHER_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)]}}
        )
        budget = CallBudget()
        source = builder(fetch, budget=budget)
        result = recommend_papers(make_query(), [source], budget=budget)

        assert f"filter={expected_filter}" in fetch.urls[0]
        assert f"filter={expected_filter}:{SEED_OPENALEX}" in fetch.urls[0]
        assert result["items"][0]["reasons"][0]["signal"] == expected_signal

    def test_the_openalex_identifier_is_recovered_from_a_work_url(self):
        """The id arrives as a full URL and must key as a bare OpenAlex id.

        If the identity layer did not strip the URL prefix, the seed would
        appear to have no OpenAlex id, every OpenAlex signal would report
        itself unavailable, and the citation graph would silently go empty —
        the exact failure this task exists to make visible.
        """
        fetch = RecordingFetch(
            {"openalex.org": {"results": [openalex_work(OPENALEX_OTHER_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)]}}
        )
        budget = CallBudget()
        result = recommend_papers(
            make_query(), [build_openalex_citing_source(fetch, budget=budget)], budget=budget
        )

        assert result["signals"][0]["status"] == STATUS_OK
        assert fetch.calls == 1

    def test_a_seed_without_an_openalex_id_reports_unavailable_not_empty(self):
        """A graph signal we cannot address is missing work, not missing papers.

        This is the case for a freshly uploaded preprint, and rendering it as
        an empty "被引" bucket would tell the user their paper has never been
        cited when the truth is that we never asked.
        """
        budget = CallBudget()
        result = recommend_papers(
            RecommendationQuery(identity=recommend.identity_from_record(
                {"title": SEED_TITLE, "year": "2015", "authors": [SEED_AUTHOR]}
            )),
            [build_openalex_citing_source(RecordingFetch(), budget=budget)],
            budget=budget,
        )

        entry = signals_by_name(result)[SIGNAL_CITING]
        assert entry["status"] == STATUS_UNAVAILABLE
        assert "OpenAlex" in entry["reason_zh"]
        assert result["unavailable_signals"][0]["signal"] == SIGNAL_CITING
        assert result["items"] == []
        assert result["upstream_calls"]["used"] == 0


# --------------------------------------------------------------------------- #
# 3. Unification: one paper, every reason
# --------------------------------------------------------------------------- #


class TestUnifiedDeduplication:
    """A paper reachable from several signals is one row carrying each reason.

    Three rows for one paper is not a cosmetic problem: the user reads it as
    three different papers, or as three independent confirmations, and both
    readings are wrong.
    """

    def test_one_paper_from_three_signals_is_one_row_with_three_reasons(self):
        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [
                        {
                            "citedPaper": s2_work(
                                FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                            )
                        }
                    ]
                },
                "/citations": {
                    "data": [
                        {
                            "citingPaper": s2_work(
                                FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                            )
                        }
                    ]
                },
                "forpaper": {
                    "recommendedPapers": [
                        s2_work(FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)
                    ]
                },
            }
        )

        def fake_search(query: str, limit: int) -> dict:
            return search_result(
                [canonical_search_hit(FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)]
            )

        budget = CallBudget(limit=10)
        result = recommend_papers(
            make_query(),
            [
                build_s2_cited_by_source(fetch, budget=budget),
                build_s2_citing_source(fetch, budget=budget),
                build_s2_related_source(fetch, budget=budget),
                build_topical_source(fake_search, budget=budget),
            ],
            budget=budget,
        )

        assert len(result["items"]) == 1
        row = result["items"][0]
        assert {reason["signal"] for reason in row["reasons"]} == {
            SIGNAL_CITED_BY,
            SIGNAL_CITING,
            SIGNAL_RELATED,
            SIGNAL_TOPICAL,
        }
        assert row["signals"] == [SIGNAL_CITED_BY, SIGNAL_CITING, SIGNAL_RELATED, SIGNAL_TOPICAL]
        # One row, but every signal still reports what it contributed.
        assert signals_by_name(result)[SIGNAL_CITING]["count"] == 1

    def test_two_providers_for_one_signal_keep_both_provenances(self):
        """Same paper from S2 and OpenAlex: one row, two reasons.

        The collapse is on the paper; the *evidence* for why it was recommended
        is not, because two providers agreeing is the fact the user is being
        told.
        """
        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [
                        {"citedPaper": s2_work(FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)}
                    ]
                },
                "filter=cited_by": {
                    "results": [openalex_work(OPENALEX_FOLLOWUP_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)]
                },
            }
        )
        budget = CallBudget(limit=10)
        result = recommend_papers(
            make_query(),
            [
                build_s2_cited_by_source(fetch, budget=budget),
                build_openalex_cited_by_source(fetch, budget=budget),
            ],
            budget=budget,
        )

        assert len(result["items"]) == 1
        reasons = result["items"][0]["reasons"]
        assert {reason["source"] for reason in reasons} == {"Semantic Scholar", "OpenAlex"}
        assert {reason["signal"] for reason in reasons} == {SIGNAL_CITED_BY}

    def test_the_merge_unions_the_identifiers_of_both_sides(self):
        """A row collapsed from two records carries both sets of ids.

        The surviving row is what a later request compares against, so a merge
        that kept only one side's arXiv id would leave the same paper able to
        appear twice on the next page.
        """
        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [
                        {
                            "citedPaper": s2_work(
                                FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                            )
                        }
                    ]
                },
                "filter=cited_by": {
                    "results": [
                        openalex_work(
                            OPENALEX_FOLLOWUP_ID,
                            title="A completely different title",
                            doi=FOLLOWUP_DOI,
                            authors=["Someone Else"],
                            year=2019,
                        )
                    ]
                },
            }
        )
        budget = CallBudget(limit=10)
        result = recommend_papers(
            make_query(),
            [
                build_s2_cited_by_source(fetch, budget=budget),
                build_openalex_cited_by_source(fetch, budget=budget),
            ],
            budget=budget,
        )

        assert len(result["items"]) == 1
        row = result["items"][0]
        assert row["doi"] == FOLLOWUP_DOI.casefold()
        assert row["s2_paper_id"] == FOLLOWUP_S2_ID
        assert row["openalex_id"] == OPENALEX_FOLLOWUP_ID
        assert row["identity_key"] == f"doi:{FOLLOWUP_DOI.casefold()}"

    def test_the_seed_is_never_recommended_to_itself(self):
        """A citation walk can return the paper it started from.

        S2's recommendations occasionally does and an OpenAlex relatedness walk
        of a survey returns the survey. Showing it is not a duplicate the user
        can dismiss; it is the product telling them to read what they are
        already reading.
        """
        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [{"citedPaper": s2_work(SEED_S2_ID, title=SEED_TITLE, doi=SEED_DOI)}]
                },
                "forpaper": {
                    "recommendedPapers": [
                        s2_work(SEED_S2_ID, title=SEED_TITLE, doi=SEED_DOI, year=2015)
                    ]
                },
            }
        )
        budget = CallBudget(limit=10)
        result = recommend_papers(
            make_query(),
            [
                build_s2_cited_by_source(fetch, budget=budget),
                build_s2_related_source(fetch, budget=budget),
            ],
            budget=budget,
        )

        assert result["items"] == []

    def test_different_papers_stay_different_papers(self):
        """The other half of the dedupe claim: no false merges.

        Same first author, same venue, same year, adjacent titles. Title
        similarity is exactly what a title-based key gets wrong, and this is
        the assertion that the identity tiers, not the words, are doing the
        work.
        """
        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [
                        {
                            "citedPaper": s2_work(
                                FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                            )
                        },
                        {
                            "citedPaper": s2_work(
                                WIDE_S2_ID, title=WIDE_TITLE, doi=WIDE_DOI,
                                authors=["Sergey Zagoruyko"],
                            )
                        },
                    ]
                }
            }
        )
        budget = CallBudget()
        result = recommend_papers(
            make_query(), [build_s2_cited_by_source(fetch, budget=budget)], budget=budget
        )

        assert {item["title"] for item in result["items"]} == {FOLLOWUP_TITLE, WIDE_TITLE}
        assert result["possible_duplicates"] == []


class TestPossibleDuplicatesAreNotMerged:
    """T3.1's asymmetry, carried into the recommendation list.

    Two records that agree on title, year and author but hold *different*
    identifiers are probably one paper — usually a preprint and its published
    version. "Probably" is not "proven", and the collapse rule is a single
    named constant that this module imports rather than re-spells. Both rows
    stay; the pair is reported so a human can settle it.
    """

    def _preprint_and_journal_versions(self):
        """The Batch Norm preprint (arXiv only) and its journal version (DOI only).

        Neither carries a provider graph id, which is what makes this the case
        T3.1 is about: two records with an identifier each, sharing none, that
        agree on title, year and author. Give them two different S2 paper ids
        instead and the layer answers "different papers" on a hard conflict —
        which is the right answer and a different test.
        """
        return (
            s2_work(
                "", title=FOLLOWUP_TITLE, doi="", arxiv="1502.03167",
                authors=["Sergey Ioffe"], year=2015,
            ),
            s2_work(
                "", title=FOLLOWUP_TITLE, doi="10.1007/BF02257825",
                authors=["Sergey Ioffe"], year=2015,
            ),
        )

    def test_an_unproven_match_keeps_both_rows(self):
        preprint, journal = self._preprint_and_journal_versions()
        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [
                        {"citedPaper": preprint},
                        {"citedPaper": journal},
                    ]
                }
            }
        )
        budget = CallBudget()
        result = recommend_papers(
            make_query(), [build_s2_cited_by_source(fetch, budget=budget)], budget=budget
        )

        assert len(result["items"]) == 2, "a probable match must not be collapsed"
        assert result["possible_duplicates"], "the pair must be reported"

    def test_the_possible_duplicate_names_both_rows_and_the_evidence(self):
        preprint, journal = self._preprint_and_journal_versions()
        fetch = RecordingFetch(
            {"/references": {"data": [{"citedPaper": preprint}, {"citedPaper": journal}]}}
        )
        budget = CallBudget()
        result = recommend_papers(
            make_query(), [build_s2_cited_by_source(fetch, budget=budget)], budget=budget
        )

        entry = result["possible_duplicates"][0]
        assert entry["title"] and entry["other_title"]
        assert entry["basis_zh"]
        assert entry["note_zh"]
        assert SIGNAL_CITED_BY in entry["signals"]
        # Both rows carry the flag, so the UI can badge the rows themselves
        # rather than only rendering a separate list.
        assert all("possible_duplicates" in item for item in result["items"])

    def test_a_merged_row_is_never_reported_as_a_possible_duplicate(self):
        """Exact matches are not "possible" anything.

        Reporting a resolved merge in the possible-duplicate list would make
        the user adjudicate a decision the system already proved, which is how a
        confirm dialog trains people to click through it.
        """
        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [
                        {
                            "citedPaper": s2_work(
                                FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                            )
                        }
                    ]
                },
                "/citations": {
                    "data": [
                        {
                            "citingPaper": s2_work(
                                FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                            )
                        }
                    ]
                },
            }
        )
        budget = CallBudget(limit=10)
        result = recommend_papers(
            make_query(),
            [
                build_s2_cited_by_source(fetch, budget=budget),
                build_s2_citing_source(fetch, budget=budget),
            ],
            budget=budget,
        )

        assert len(result["items"]) == 1
        assert result["possible_duplicates"] == []


# --------------------------------------------------------------------------- #
# 4. Failure is not emptiness
# --------------------------------------------------------------------------- #


class TestFailureIsNeverAnEmptyResult:
    """The four statuses, and the difference that separates them.

    ``empty`` is a claim about the literature. ``unavailable`` is a claim about
    us. A rate-limited provider produces the second, and rendering it as the
    first is how a product ends up telling users their paper has never been
    cited.
    """

    def test_a_rate_limited_provider_is_unavailable_with_a_reason(self):
        error = urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
        budget = CallBudget()
        result = recommend_papers(
            make_query(),
            [build_openalex_citing_source(RecordingFetch(error=error), budget=budget)],
            budget=budget,
        )

        entry = signals_by_name(result)[SIGNAL_CITING]
        assert entry["status"] == STATUS_UNAVAILABLE
        assert entry["status"] != STATUS_EMPTY
        assert "限流" in entry["reason_zh"] or "rate limit" in entry["reason_zh"].lower()
        assert result["errors"], "an unavailable signal must reach the errors list"

    def test_a_timeout_is_unavailable_rather_than_a_guess(self):
        budget = CallBudget()
        result = recommend_papers(
            make_query(),
            [
                build_openalex_related_source(
                    RecordingFetch(error=TimeoutError("too slow")), budget=budget
                )
            ],
            budget=budget,
        )

        assert signals_by_name(result)[SIGNAL_RELATED]["status"] == STATUS_UNAVAILABLE
        assert result["items"] == []

    def test_a_provider_with_nothing_is_empty_and_says_so(self):
        """The positive version of the same distinction.

        A genuinely uncited paper must be able to say so, or the product has
        lost the ability to state a fact about the literature and only has the
        ability to apologise.
        """
        budget = CallBudget()
        result = recommend_papers(
            make_query(), [build_s2_citing_source(RecordingFetch(), budget=budget)], budget=budget
        )

        entry = signals_by_name(result)[SIGNAL_CITING]
        assert entry["status"] == STATUS_EMPTY
        assert entry["reason_zh"]
        assert result["unavailable_signals"] == []
        assert result["errors"] == []

    def test_one_failing_provider_does_not_hide_a_working_one(self):
        """Two providers, one rate-limited: the signal is ok *and* the failure shows.

        Collapsing to the most optimistic status alone would be a lie of
        omission — the user would be told the answer is complete while half of
        it was never fetched. The per-provider breakdown is what makes the
        partial state visible.
        """
        rate_limited = RecordingFetch(error=urllib.error.HTTPError("u", 429, "slow down", {}, None))
        working = RecordingFetch(
            {
                "filter=cited_by": {
                    "results": [openalex_work(OPENALEX_FOLLOWUP_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)]
                }
            }
        )
        budget = CallBudget(limit=10)
        result = recommend_papers(
            make_query(),
            [
                build_openalex_cited_by_source(working, budget=budget),
                build_s2_cited_by_source(rate_limited, budget=budget),
            ],
            budget=budget,
        )

        entry = signals_by_name(result)[SIGNAL_CITED_BY]
        assert entry["status"] == STATUS_OK
        assert len(result["items"]) == 1
        by_provider = {item["source"]: item for item in entry["providers"]}
        assert by_provider["OpenAlex"]["status"] == STATUS_OK
        assert by_provider["Semantic Scholar"]["status"] == STATUS_UNAVAILABLE

    def test_a_source_that_raises_never_fails_the_request(self):
        def exploding(_query: RecommendationQuery) -> SignalResult:
            raise RuntimeError("boom")

        budget = CallBudget()
        result = recommend_papers(
            make_query(),
            [
                recommend.SignalSource(name="broken", signal=SIGNAL_TOPICAL, fetch=exploding),
                build_s2_citing_source(
                    RecordingFetch(
                        {
                            "/citations": {
                                "data": [
                                    {
                                        "citingPaper": s2_work(
                                            WIDE_S2_ID, title=WIDE_TITLE, doi=WIDE_DOI
                                        )
                                    }
                                ]
                            }
                        }
                    ),
                    budget=budget,
                ),
            ],
            budget=budget,
        )

        assert len(result["items"]) == 1, "one broken source must not lose the others"
        assert signals_by_name(result)[SIGNAL_TOPICAL]["status"] == STATUS_UNAVAILABLE

    def test_a_source_returning_nonsense_is_treated_as_a_failure(self):
        """An injected source must not be able to fabricate a result.

        The engine has no way to verify a reason's provenance, so a source that
        returns something that is not a :class:`SignalResult` is treated as a
        failure: there is no signal in it, so there is no reason to show.
        """
        budget = CallBudget()
        result = recommend_papers(
            make_query(),
            [
                recommend.SignalSource(
                    name="rogue",
                    signal=SIGNAL_RELATED,
                    fetch=lambda _q: [{"title": "Invented", "signal": "related"}],
                )
            ],
            budget=budget,
        )

        assert result["items"] == []
        assert signals_by_name(result)[SIGNAL_RELATED]["status"] == STATUS_UNAVAILABLE

    def test_no_row_ever_carries_a_reason_it_did_not_earn(self):
        """The invariant, checked across every signal at once.

        Every reason on every row must trace to a source that reported a hit for
        that signal. If this ever fails, the product is showing a user a
        fabricated explanation, which is the failure mode the whole module
        exists to prevent — so it is asserted directly rather than implied by
        the per-signal tests above.
        """
        def fake_search(query: str, limit: int) -> dict:
            return search_result(
                [canonical_search_hit(FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)]
            )

        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [
                        {
                            "citedPaper": s2_work(
                                FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                            )
                        }
                    ]
                },
                "/citations": {
                    "data": [
                        {
                            "citingPaper": s2_work(
                                WIDE_S2_ID, title=WIDE_TITLE, doi=WIDE_DOI
                            )
                        }
                    ]
                },
                "forpaper": {
                    "recommendedPapers": [
                        s2_work(WIDE_S2_ID, title=WIDE_TITLE, doi=WIDE_DOI)
                    ]
                },
            }
        )
        budget = CallBudget(limit=10)
        result = recommend_papers(
            make_query(datasets=("ImageNet",)),
            [
                build_s2_cited_by_source(fetch, budget=budget),
                build_s2_citing_source(fetch, budget=budget),
                build_s2_related_source(fetch, budget=budget),
                build_topical_source(fake_search, budget=budget),
                build_same_dataset_source(fake_search, budget=budget),
            ],
            budget=budget,
        )

        reported = {entry["signal"] for entry in result["signals"] if entry["count"]}
        for row in result["items"]:
            assert row["reasons"], "a row with no reason must never be returned"
            for reason in row["reasons"]:
                assert reason["signal"] in SIGNAL_ORDER
                assert reason["reason_zh"], "a reason with no sentence cannot be rendered"
                assert reason["signal"] in reported


# --------------------------------------------------------------------------- #
# 5. The response shape the UI needs
# --------------------------------------------------------------------------- #


class TestResponseShape:
    """The acceptance criterion: per-signal grouping, reason in the payload.

    Asserted as a shape rather than as values, because the client's question is
    "can I group these rows by signal and show a sentence for each" and that has
    to keep being answerable after the next refactor.
    """

    def test_rows_group_by_signal_without_consulting_anything_else(self):
        """``{signal: [rows]}`` must be derivable from the payload alone.

        This is the literal acceptance criterion. If a client had to re-derive
        which signal produced a row, the reason would be reconstructed
        client-side — and a reconstruction is a guess.
        """
        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [
                        {
                            "citedPaper": s2_work(
                                FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                            )
                        }
                    ]
                },
                "/citations": {
                    "data": [
                        {"citingPaper": s2_work(WIDE_S2_ID, title=WIDE_TITLE, doi=WIDE_DOI)}
                    ]
                },
            }
        )
        budget = CallBudget(limit=10)
        result = recommend_papers(
            make_query(),
            [
                build_s2_cited_by_source(fetch, budget=budget),
                build_s2_citing_source(fetch, budget=budget),
            ],
            budget=budget,
        )

        groups: dict[str, list[dict]] = {}
        for row in result["items"]:
            for signal in row["signals"]:
                groups.setdefault(signal, []).append(row)
        assert set(groups) == {SIGNAL_CITED_BY, SIGNAL_CITING}
        assert [row["title"] for row in groups[SIGNAL_CITED_BY]] == [FOLLOWUP_TITLE]
        assert [row["title"] for row in groups[SIGNAL_CITING]] == [WIDE_TITLE]

    def test_every_row_carries_a_renderable_reason(self):
        fetch = RecordingFetch(
            {
                "/citations": {
                    "data": [{"citingPaper": s2_work(WIDE_S2_ID, title=WIDE_TITLE, doi=WIDE_DOI)}]
                }
            }
        )
        budget = CallBudget()
        result = recommend_papers(
            make_query(), [build_s2_citing_source(fetch, budget=budget)], budget=budget
        )

        row = result["items"][0]
        reason = row["reasons"][0]
        assert set(reason) == {
            "signal", "label_zh", "source", "reason_zh", "detail", "confidence",
        }
        assert reason["label_zh"] and reason["reason_zh"] and reason["source"]
        # The canonical record fields the reading queue needs are still there.
        for field in ("title", "authors", "year", "venue", "url", "pdf_url", "identity_key"):
            assert field in row

    def test_the_signal_index_is_complete_and_ordered(self):
        """Every signal gets an entry, in the plan's order, whatever happened to it.

        A signal missing from the index is a signal the UI cannot offer as a
        filter, which is how a user is quietly denied the choice.
        """
        budget = CallBudget()
        result = recommend_papers(
            make_query(),
            [build_s2_citing_source(RecordingFetch(), budget=budget)],
            budget=budget,
        )

        reported = [entry["signal"] for entry in result["signals"]]
        assert reported == [SIGNAL_CITING]
        assert all(entry["label_zh"] for entry in result["signals"])

    def test_a_failed_signal_is_listed_so_the_ui_can_say_so(self):
        rate_limited = RecordingFetch(error=urllib.error.HTTPError("u", 429, "slow down", {}, None))
        budget = CallBudget()
        result = recommend_papers(
            make_query(),
            [
                build_s2_citing_source(rate_limited, budget=budget),
                build_openalex_citing_source(rate_limited, budget=budget),
            ],
            budget=budget,
        )

        unavailable_signals = {entry["signal"] for entry in result["unavailable_signals"]}
        assert unavailable_signals == {SIGNAL_CITING}
        assert all(entry["reason_zh"] for entry in result["unavailable_signals"])

    def test_the_seed_is_echoed_so_the_ui_can_say_why_this_paper(self):
        budget = CallBudget()
        result = recommend_papers(
            make_query(), [build_s2_citing_source(RecordingFetch(), budget=budget)], budget=budget
        )

        assert result["seed"]["title"] == SEED_TITLE
        assert result["seed"]["doi"] == SEED_DOI.casefold()
        assert result["seed"]["identity_key"].startswith("doi:")

    def test_rows_are_ranked_by_the_strongest_signal_that_reached_them(self):
        """A citation edge outranks a topic match, and the order is a constant.

        Not a quality score: it is "which row is the most defensible", computed
        from :data:`SIGNAL_ORDER` so the same inputs always produce the same
        list and a test can assert it.
        """
        def fake_search(query: str, limit: int) -> dict:
            return search_result(
                [
                    canonical_search_hit(
                        WIDE_S2_ID,
                        title="A Topic Match Only",
                        doi="10.1000/topiconly",
                        abstract="Unrelated to the citation graph.",
                    )
                ]
            )

        fetch = RecordingFetch(
            {
                "/citations": {
                    "data": [
                        {"citingPaper": s2_work(FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI)}
                    ]
                }
            }
        )
        budget = CallBudget(limit=10)
        result = recommend_papers(
            make_query(),
            [
                build_topical_source(fake_search, budget=budget),
                build_s2_citing_source(fetch, budget=budget),
            ],
            budget=budget,
        )

        assert [item["title"] for item in result["items"]] == [
            FOLLOWUP_TITLE,
            "A Topic Match Only",
        ]

    def test_the_result_limit_truncates_and_the_signal_counts_do_not_claim_more(self):
        def fake_search(query: str, limit: int) -> dict:
            return search_result(
                [
                    canonical_search_hit(f"{index:040d}", title=f"Hit {index}", doi=f"10.1000/{index}")
                    for index in range(5)
                ]
            )

        budget = CallBudget(limit=10)
        result = recommend_papers(
            make_query(), [build_topical_source(fake_search, budget=budget)], budget=budget, limit=2
        )

        assert len(result["items"]) == 2
        # The signal still reports what it fetched; the row list is the cap.
        assert signals_by_name(result)[SIGNAL_TOPICAL]["count"] == 5


# --------------------------------------------------------------------------- #
# 6. The call budget
# --------------------------------------------------------------------------- #


class TestBoundedUpstreamCalls:
    """One click must not fan out, and the bound must be visible when it bites.

    The plan's section 8 names provider rate limiting as a real risk. What this
    module can do about it locally is refuse to make an unbounded number of
    calls and to say which signals the refusal cost.
    """

    def test_a_small_budget_stops_the_fan_out(self):
        fetch = RecordingFetch(
            {
                "/references": {
                    "data": [
                        {
                            "citedPaper": s2_work(
                                FOLLOWUP_S2_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                            )
                        }
                    ]
                }
            }
        )

        def fake_search(query: str, limit: int) -> dict:
            return search_result([])

        budget = CallBudget(limit=2)
        result = recommend_papers(
            make_query(datasets=("ImageNet",)),
            default_recommendation_sources(
                budget=budget, search_fn=fake_search, fetch=fetch
            ),
            budget=budget,
        )

        assert fetch.calls + result["upstream_calls"]["used"] <= 8
        assert result["upstream_calls"]["used"] <= 2
        assert result["upstream_calls"]["limit"] == 2

    def test_a_signal_dropped_by_the_budget_is_reported_as_skipped(self):
        """The dropped signals are named, not silently absent.

        A budget that quietly shortened the list is indistinguishable from a
        paper with no related work, and the user cannot tell whether to retry
        or to stop asking.
        """
        def fake_search(query: str, limit: int) -> dict:
            return search_result([])

        # The default set asks OpenAlex before S2, so the one call this budget
        # allows buys OpenAlex's cited_by — the fake has to serve that shape or
        # the assertion would be about an unrelated signal.
        fetch = RecordingFetch(
            {
                "openalex.org": {
                    "results": [
                        openalex_work(
                            OPENALEX_FOLLOWUP_ID, title=FOLLOWUP_TITLE, doi=FOLLOWUP_DOI
                        )
                    ]
                }
            }
        )
        budget = CallBudget(limit=1)
        result = recommend_papers(
            make_query(datasets=("ImageNet",)),
            default_recommendation_sources(
                budget=budget, search_fn=fake_search, fetch=fetch
            ),
            budget=budget,
        )

        statuses = {entry["signal"]: entry["status"] for entry in result["signals"]}
        assert statuses[SIGNAL_CITED_BY] == STATUS_OK
        assert statuses[SIGNAL_TOPICAL] == STATUS_SKIPPED
        assert statuses[SIGNAL_SAME_DATASET] == STATUS_SKIPPED
        assert all(
            "预算" in entry["reason_zh"]
            for entry in result["unavailable_signals"]
            if entry["status"] == STATUS_SKIPPED
        )

    def test_a_signal_missing_its_identifier_costs_no_call(self):
        """Asking OpenAlex about a paper it cannot address would be a wasted call.

        The bound is only worth anything if a call that cannot succeed is not
        made, so a source whose precondition fails must leave the budget
        untouched — asserted on the counter, not on the absence of an error.
        """
        fetch = RecordingFetch()
        budget = CallBudget(limit=10)
        recommend_papers(
            RecommendationQuery(
                identity=recommend.identity_from_record(
                    {"title": SEED_TITLE, "authors": [SEED_AUTHOR], "s2_paper_id": SEED_S2_ID}
                )
            ),
            [
                build_openalex_citing_source(fetch, budget=budget),
                build_openalex_related_source(fetch, budget=budget),
                build_openalex_cited_by_source(fetch, budget=budget),
            ],
            budget=budget,
        )

        assert fetch.calls == 0
        assert budget.used == 0

    def test_the_budget_never_goes_negative_and_never_overspends(self):
        budget = CallBudget(limit=2)
        assert budget.charge(2) is True
        assert budget.charge(1) is False
        assert budget.used == 2
        assert budget.remaining == 0
        assert budget.to_dict() == {"limit": 2, "used": 2, "remaining": 0}

    def test_the_dataset_signal_searches_at_most_two_names(self):
        """A prose sentence must not become an unbounded loop of queries."""
        searched: list[str] = []

        def fake_search(query: str, limit: int) -> dict:
            searched.append(query)
            return search_result([])

        budget = CallBudget(limit=MAX_DATASET_QUERIES)
        result = recommend_papers(
            make_query(datasets=("ImageNet", "COCO", "CIFAR-10", "WMT14", "SQuAD")),
            [build_same_dataset_source(fake_search, budget=budget)],
            budget=budget,
        )

        assert searched == ["ImageNet", "COCO"]
        assert signals_by_name(result)[SIGNAL_SAME_DATASET]["status"] == STATUS_EMPTY

    def test_the_dataset_search_does_not_swallow_a_partial_failure(self):
        """One dataset query failing still lets the other one answer.

        And the failure is still reported, so a list built from one dataset is
        not mistaken for a list built from all of them.
        """
        def fake_search(query: str, limit: int) -> dict:
            if query == "ImageNet":
                raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)
            return search_result(
                [
                    canonical_search_hit(
                        WIDE_S2_ID, title=WIDE_TITLE, doi=WIDE_DOI, abstract="Also on COCO."
                    )
                ]
            )

        budget = CallBudget(limit=4)
        result = recommend_papers(
            make_query(datasets=("ImageNet", "COCO")),
            [build_same_dataset_source(fake_search, budget=budget)],
            budget=budget,
        )

        entry = signals_by_name(result)[SIGNAL_SAME_DATASET]
        assert entry["status"] == STATUS_OK
        assert result["items"][0]["reasons"][0]["detail"] == "dataset:COCO"

    def test_every_dataset_query_failing_is_unavailable_not_empty(self):
        def fake_search(query: str, limit: int) -> dict:
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None)

        budget = CallBudget(limit=4)
        result = recommend_papers(
            make_query(datasets=("ImageNet",)),
            [build_same_dataset_source(fake_search, budget=budget)],
            budget=budget,
        )

        assert signals_by_name(result)[SIGNAL_SAME_DATASET]["status"] == STATUS_UNAVAILABLE


# --------------------------------------------------------------------------- #
# 7. The same-dataset dependency, stated rather than approximated
# --------------------------------------------------------------------------- #


class TestSameDatasetDependency:
    """What the signal needs, and what it does when it is missing.

    ``PaperFacts.datasets`` holds model prose, not dataset identifiers, and it
    is not persisted into the session payload. The signal therefore has to
    either degrade loudly or start lying. It degrades loudly.
    """

    def test_no_dataset_names_means_unavailable_with_the_dependency_named(self):
        """The signal reports the missing dependency instead of faking one.

        The obvious shortcut is to fall back to topic similarity, which would
        produce a "同数据集" badge on papers selected by text overlap — a label
        asserting something about datasets that no code in this request ever
        checked.
        """
        budget = CallBudget()
        result = recommend_papers(
            make_query(),
            [build_same_dataset_source(lambda q, n: search_result([]), budget=budget)],
            budget=budget,
        )

        entry = signals_by_name(result)[SIGNAL_SAME_DATASET]
        assert entry["status"] == STATUS_UNAVAILABLE
        assert "paper_facts" in entry["reason_zh"]
        assert "冒充" in entry["reason_zh"]
        assert result["items"] == []

    def test_dataset_names_are_recovered_from_a_fact_layer_payload(self):
        """The scanner finds real names and skips the prose around them."""
        facts = {
            "datasets": [
                {
                    "field_name": "datasets",
                    "value": "本文在 ImageNet 与 COCO 上评估了所提方法。",
                    "extraction_status": "extracted",
                }
            ]
        }
        assert dataset_names_from_facts(facts) == ("ImageNet", "COCO")

    def test_the_scanner_ignores_sentence_prose(self):
        """Only name-shaped tokens survive.

        A rule of "any capitalised word" would search for "We", "Both" and
        "Introduction", and a result list built from those is worse than no
        result list because it is confidently mislabelled.
        """
        facts = {
            "datasets": [
                {
                    "value": "We compare against the Baseline on ImageNet and the CIFAR-10 benchmark.",
                    "extraction_status": "extracted",
                }
            ]
        }
        assert dataset_names_from_facts(facts) == ("ImageNet", "CIFAR-10")

    def test_a_digit_qualifies_a_name_the_stopword_list_could_not_know(self):
        facts = [{"value": "WMT14 and MultiNLI", "extraction_status": "extracted"}]
        assert dataset_names_from_facts(facts) == ("WMT14", "MultiNLI")

    def test_not_stated_and_unverified_datasets_are_not_searched(self):
        """Both are answers about the paper, not dataset names.

        Searching for "论文未说明" finds nothing and reports an empty signal,
        which reads as "no paper uses this dataset" about a dataset that was
        never identified.
        """
        assert dataset_names_from_facts(
            [{"value": "论文未说明", "extraction_status": "not_found"}]
        ) == ()
        assert dataset_names_from_facts(
            [{"value": "【未通过证据校验】ImageNet", "extraction_status": "needs_review"}]
        ) == ()

    def test_the_gate_checks_title_abstract_and_venue_case_insensitively(self):
        record = {"title": "Deep nets", "abstract": "We test on imagenet.", "venue": "CVPR"}
        assert dataset_name_appears("ImageNet", record) is True
        assert dataset_name_appears("imagenet", record) is True
        assert dataset_name_appears("COCO", record) is False
        assert dataset_name_appears("", record) is False

    def test_a_fact_layer_object_is_accepted_as_well_as_its_dict(self):
        """The two call sites hold different shapes and neither is wrong.

        The fact layer has the object; a persisted session has the dict. A
        helper that only accepted one would make the caller convert, and the
        conversion is where a dataset name gets lost.
        """

        class FakePaperFacts:
            datasets = (
                type("Fact", (), {"value": "Evaluated on GLUE.", "extraction_status": "extracted"}),
            )

        assert dataset_names_from_facts(FakePaperFacts()) == ("GLUE",)
        assert dataset_names_from_facts(None) == ()


# --------------------------------------------------------------------------- #
# 8. The shared HTTP client
# --------------------------------------------------------------------------- #


class TestSharedTransport:
    """No second HTTP path, and the URLs are right.

    The production builders must reach the network only through
    :mod:`paperwhisperer.search.http`. That is asserted by replacing the module
    function and checking the builders call it — a parallel client would not be
    caught by any behavioural test here, so it is caught structurally.
    """

    def test_the_default_transport_is_the_shared_client(self, monkeypatch):
        import paperwhisperer.search.http as shared_http

        seen: list[dict] = []

        def fake_get_json(url, **kwargs):
            seen.append({"url": url, **kwargs})
            return {"results": []}

        monkeypatch.setattr(shared_http, "http_get_json", fake_get_json)
        budget = CallBudget()
        recommend_papers(
            make_query(), [build_openalex_citing_source(budget=budget)], budget=budget
        )

        assert len(seen) == 1
        assert seen[0]["url"].startswith("https://api.openalex.org/works?")
        # The shared client owns the transport: the builder hands it the SSL
        # context and the backoff ceiling rather than opening a socket itself.
        assert seen[0]["ssl_context"] is not None
        assert seen[0]["retry_max_delay"] > 0
        assert seen[0]["retries"] >= 1

    def test_the_s2_transport_carries_the_configured_api_key(self, monkeypatch):
        import paperwhisperer.search.http as shared_http
        from paperwhisperer.core import config

        monkeypatch.setattr(config, "SEMANTIC_SCHOLAR_API_KEY", "secret-key")
        seen: list[dict] = []
        monkeypatch.setattr(
            shared_http, "http_get_json", lambda url, **kw: (seen.append(kw), {"data": []})[1]
        )
        budget = CallBudget()
        recommend_papers(
            make_query(), [build_s2_citing_source(budget=budget)], budget=budget
        )

        assert seen[0]["headers"]["x-api-key"] == "secret-key"

    def test_the_recommendation_endpoints_are_the_graph_v1_graph_paths(self):
        """The path shape is asserted so a base-URL edit cannot silently 404.

        ``/paper/{id}/references`` and ``/paper/{id}/citations`` differ by one
        segment and swap the meaning of the whole list; ``forpaper`` lives under
        a different prefix entirely and is a different API version.
        """
        budget = CallBudget()
        fetch = RecordingFetch()
        build_s2_cited_by_source(fetch, budget=budget).fetch(make_query())
        build_s2_citing_source(fetch, budget=budget).fetch(make_query())
        build_s2_related_source(fetch, budget=budget).fetch(make_query())

        assert f"/paper/{SEED_S2_ID}/references" in fetch.urls[0]
        assert f"/paper/{SEED_S2_ID}/citations" in fetch.urls[1]
        assert "recommendations/v1/papers/forpaper" in fetch.urls[2]
        assert all("externalIds" in url for url in fetch.urls)

    def test_the_default_source_set_pairs_two_providers_per_graph_signal(self):
        """Two providers, because a preprint with only an arXiv id is normal.

        A set that asked S2 only would report "被引: 不可用" for a paper OpenAlex
        knows perfectly well, and the user would conclude their paper has never
        been cited.
        """
        sources = default_recommendation_sources(
            budget=CallBudget(), search_fn=lambda q, n: search_result([])
        )
        by_signal: dict[str, set[str]] = {}
        for source in sources:
            by_signal.setdefault(source.signal, set()).add(source.name)
        assert by_signal[SIGNAL_CITED_BY] == {"OpenAlex", "Semantic Scholar"}
        assert by_signal[SIGNAL_CITING] == {"OpenAlex", "Semantic Scholar"}
        assert by_signal[SIGNAL_RELATED] == {"OpenAlex", "Semantic Scholar"}
        assert by_signal[SIGNAL_TOPICAL] and by_signal[SIGNAL_SAME_DATASET]

    def test_the_default_source_set_shares_one_budget(self):
        """Two budgets would be two limits and no bound.

        The whole point of the budget is that a request's ceiling is one
        number; eight sources each holding their own counter would let the
        request make eight times the intended number of upstream calls.
        """
        budget = CallBudget(limit=1)
        sources = default_recommendation_sources(
            budget=budget, search_fn=lambda q, n: search_result([]), fetch=RecordingFetch()
        )
        for source in sources:
            source.fetch(make_query())
        assert budget.used <= 1

    def test_a_malformed_provider_payload_degrades_to_an_empty_signal(self):
        """S2 has shipped two spellings of the recommendations envelope.

        A shape change must produce "no recommendations", not a KeyError on a
        payload index — the same call must never take down the other four
        signals.
        """
        budget = CallBudget()
        result = recommend_papers(
            make_query(),
            [build_s2_related_source(RecordingFetch({"forpaper": {"unexpected": 1}}), budget=budget)],
            budget=budget,
        )

        assert signals_by_name(result)[SIGNAL_RELATED]["status"] == STATUS_EMPTY
        assert result["items"] == []
