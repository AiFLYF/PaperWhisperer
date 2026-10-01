"""Paper identity: priority tiers, the preprint/published pair, and enrichment.

The tests here are organised around the two claims the module makes that a
user would notice being wrong: a preprint and its published version are one
paper, and two different papers that merely look alike are two papers. Both
are silent failures — one shows a paper twice, the other hides a paper the
user searched for — so they are asserted directly rather than through
whatever the dedupe list happens to look like.

Enrichment tests inject a fetcher. The production fetchers in
:mod:`paperwhisperer.search.enrich` are exercised only for URL construction,
with the transport itself replaced, so this file never opens a socket.
"""

from __future__ import annotations

import urllib.error

import pytest

from paperwhisperer.search.enrich import (
    EnrichmentResult,
    LookupQuery,
    MetadataSource,
    build_crossref_source,
    build_openalex_source,
    enrich_identity,
    parse_crossref_payload,
    parse_openalex_payload,
)
from paperwhisperer.search.identity import (
    GENERIC_TITLES,
    IDENTITY_TIER_ORDER,
    PaperIdentity,
    compare_identities,
    core_title_key,
    identity_from_record,
    merge_identities,
    normalize_arxiv_id,
    normalize_author_key,
    normalize_doi,
    normalize_openalex_id,
    normalize_s2_paper_id,
    normalize_title_key,
    titles_equivalent,
)

# --------------------------------------------------------------------------- #
# Fixtures: a real preprint/published pair, used throughout.
# He et al. posted arXiv:1512.03385 in December 2015 and it appeared at CVPR
# 2016 as 10.1109/CVPR.2016.90. The year changes, the venue changes, the title
# is re-typeset, the author list is reordered — and it is one paper.
# --------------------------------------------------------------------------- #

PREPRINT_TITLE = "Deep Residual Learning for Image Recognition"
PREPRINT_YEAR = "2015"
PREPRINT_AUTHOR = "Kaiming He"
PREPRINT_ARXIV = "1512.03385"
PUBLISHED_DOI = "10.1109/CVPR.2016.90"
PUBLISHED_YEAR = "2016"


def preprint_identity(**overrides) -> PaperIdentity:
    """The arXiv-side record: an arXiv id and no DOI, as arXiv returns it."""
    values = {
        "arxiv_id": PREPRINT_ARXIV,
        "title": PREPRINT_TITLE,
        "year": PREPRINT_YEAR,
        "first_author": PREPRINT_AUTHOR,
    }
    values.update(overrides)
    return PaperIdentity(**values)


def published_identity(**overrides) -> PaperIdentity:
    """The publisher-side record: a DOI and no arXiv id, as Crossref returns it."""
    values = {
        "doi": PUBLISHED_DOI,
        "title": PREPRINT_TITLE,
        "year": PUBLISHED_YEAR,
        "first_author": "K. He",
    }
    values.update(overrides)
    return PaperIdentity(**values)


def crossref_item(doi=PUBLISHED_DOI, year=2016, family="He", given="Kaiming", title=PREPRINT_TITLE):
    """One item in the shape Crossref's ``/works`` endpoint actually returns."""
    return {
        "DOI": doi,
        "title": [title],
        "author": [{"family": family, "given": given}, {"family": "Zhang", "given": "Xiangyu"}],
        "issued": {"date-parts": [[year]]},
    }


def crossref_payload(*items) -> dict:
    return {"message": {"items": list(items)}}


def fake_source(name: str, payload, parser=parse_crossref_payload) -> MetadataSource:
    """A provider whose network call is a lambda and whose parser is the real one."""
    return MetadataSource(name=name, fetch=lambda query: payload, parse=parser)


def raising_source(name: str, exc: Exception) -> MetadataSource:
    def _fetch(query):
        raise exc

    return MetadataSource(name=name, fetch=_fetch, parse=parse_crossref_payload)


# --------------------------------------------------------------------------- #


class TestIdentityPriority:
    """The documented tier order, one test per tier."""

    def test_tier_order_matches_the_plan_verbatim(self):
        # Written as a literal, not derived from the module, so that editing
        # IDENTITY_TIER_ORDER to suit the code breaks the test rather than
        # quietly redefining what the plan asked for.
        assert IDENTITY_TIER_ORDER == ("doi", "arxiv", "s2", "openalex", "title")

    def test_doi_outranks_every_other_identifier(self):
        identity = PaperIdentity(
            doi=PUBLISHED_DOI,
            arxiv_id=PREPRINT_ARXIV,
            s2_paper_id="a" * 40,
            openalex_id="https://openalex.org/W2963402937",
            title=PREPRINT_TITLE,
            year=PUBLISHED_YEAR,
            first_author=PREPRINT_AUTHOR,
        )

        assert identity.key == f"doi:{PUBLISHED_DOI.lower()}"
        assert identity.key_tier == "doi"

    def test_arxiv_is_used_when_there_is_no_doi(self):
        identity = PaperIdentity(
            arxiv_id=PREPRINT_ARXIV,
            s2_paper_id="a" * 40,
            openalex_id="W2963402937",
            title=PREPRINT_TITLE,
            year=PREPRINT_YEAR,
            first_author=PREPRINT_AUTHOR,
        )

        assert identity.key_tier == "arxiv"
        assert identity.key == f"arxiv:{PREPRINT_ARXIV}"

    def test_s2_is_used_when_there_is_no_doi_or_arxiv(self):
        identity = PaperIdentity(
            s2_paper_id="a" * 40,
            openalex_id="W2963402937",
            title=PREPRINT_TITLE,
            year=PREPRINT_YEAR,
            first_author=PREPRINT_AUTHOR,
        )

        assert identity.key_tier == "s2"

    def test_openalex_outranks_the_title_fallback(self):
        identity = PaperIdentity(
            openalex_id="W2963402937",
            title=PREPRINT_TITLE,
            year=PREPRINT_YEAR,
            first_author=PREPRINT_AUTHOR,
        )

        assert identity.key_tier == "openalex"
        assert identity.key == "openalex:W2963402937"

    def test_the_title_fallback_is_only_reached_as_a_last_resort(self):
        identity = PaperIdentity(
            title=PREPRINT_TITLE,
            year=PREPRINT_YEAR,
            first_author=PREPRINT_AUTHOR,
        )

        assert identity.key_tier == "title"
        assert identity.key.startswith("title:")

    def test_a_record_that_identifies_nothing_has_an_empty_key(self):
        assert PaperIdentity().key == ""
        assert PaperIdentity(title=PREPRINT_TITLE).key == ""

    def test_needs_enrichment_only_when_the_doi_is_missing(self):
        assert preprint_identity().needs_enrichment is True
        assert published_identity().needs_enrichment is False


class TestIdentifierNormalization:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("10.1109/CVPR.2016.90", "10.1109/cvpr.2016.90"),
            ("DOI:10.1109/CVPR.2016.90", "10.1109/cvpr.2016.90"),
            ("doi: 10.1109/CVPR.2016.90", "10.1109/cvpr.2016.90"),
            ("https://doi.org/10.1109/CVPR.2016.90", "10.1109/cvpr.2016.90"),
            ("http://dx.doi.org/10.1109/CVPR.2016.90", "10.1109/cvpr.2016.90"),
            # The case a reference manager produces. The trailing full stop
            # belongs to the sentence, not the DOI.
            ("10.1109/CVPR.2016.90.", "10.1109/cvpr.2016.90"),
            ("10.1109/CVPR.2016.90;", "10.1109/cvpr.2016.90"),
            ("  10.1109/CVPR.2016.90  ", "10.1109/cvpr.2016.90"),
        ],
    )
    def test_doi_forms_collapse_to_one_value(self, raw, expected):
        assert normalize_doi(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["", None, "not a doi", "https://example.com/paper", "11.1234/abc", "10.1234"],
    )
    def test_non_dois_are_refused_rather_than_guessed(self, raw):
        assert normalize_doi(raw) == ""

    def test_a_doi_is_never_mistaken_for_an_arxiv_id(self):
        # "1234.56789" inside this DOI is shaped exactly like a new-style
        # arXiv id; substring matching would hand it an arXiv identity.
        assert normalize_arxiv_id("10.1145/1234.56789") == ""
        assert normalize_arxiv_id("10.1109/cvpr.2016.90") == ""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("2301.00001", "2301.00001"),
            ("2301.00001v1", "2301.00001"),
            ("2301.00001v2", "2301.00001"),
            ("2301.00001V12", "2301.00001"),
            ("arXiv:2301.00001v3", "2301.00001"),
            ("arXiv: 2301.00001", "2301.00001"),
            ("http://arxiv.org/abs/2301.00001v4", "2301.00001"),
            ("https://arxiv.org/abs/2301.00001", "2301.00001"),
            ("http://arxiv.org/pdf/2301.00001v1.pdf", "2301.00001"),
            ("math/0309136", "math/0309136"),
            ("math.GT/0309136v2", "math/0309136"),
            ("cs/0501001", "cs/0501001"),
        ],
    )
    def test_arxiv_forms_collapse_and_lose_the_version(self, raw, expected):
        assert normalize_arxiv_id(raw) == expected

    @pytest.mark.parametrize("raw", ["", None, "2301.000", "not-an-id", "2301.00001xyz"])
    def test_non_arxiv_values_are_refused(self, raw):
        assert normalize_arxiv_id(raw) == ""

    def test_a_revision_is_the_same_paper_as_its_earlier_version(self):
        v1 = PaperIdentity(arxiv_id="2301.00001v1", title="A Paper", year="2023", first_author="A Person")
        v3 = PaperIdentity(arxiv_id="2301.00001v3", title="A Paper", year="2023", first_author="A Person")

        assert v1.key == v3.key
        assert compare_identities(v1, v3).same_paper is True

    def test_s2_ids_must_look_like_s2_ids(self):
        assert normalize_s2_paper_id("A" * 40) == "a" * 40
        assert normalize_s2_paper_id("CorpusId:12345") == ""
        assert normalize_s2_paper_id("Attention Is All You Need") == ""

    def test_openalex_ids_accept_urls_and_normalize_case(self):
        assert normalize_openalex_id("https://openalex.org/W2963402937") == "W2963402937"
        assert normalize_openalex_id("w2963402937") == "W2963402937"
        assert normalize_openalex_id("10.1109/cvpr.2016.90") == ""


class TestFallbackKeyNormalization:
    def test_case_punctuation_and_whitespace_all_collapse(self):
        assert normalize_title_key("  Attention Is   ALL you  need! ") == "attention is all you need"

    def test_diacritics_fold_because_publishers_retype_them(self):
        assert normalize_title_key("Müller's Approach") == normalize_title_key("Muller's Approach")

    def test_an_apostrophe_is_removed_rather_than_split(self):
        # "Muller's" and "Mullers" are the same surname spelled two ways.
        assert normalize_author_key("Seamus O'Neill") == "oneill"
        assert normalize_author_key("Müller") == normalize_author_key("Muller")

    def test_hyphenation_variants_collapse(self):
        assert normalize_title_key("Self-supervised Learning") == normalize_title_key("Self supervised Learning")

    def test_cjk_titles_survive_normalization(self):
        key = normalize_title_key("基于深度学习的图像识别：一种方法")

        assert "图像识别" in key
        assert "：" not in key

    def test_html_markup_is_stripped_before_comparison(self):
        assert normalize_title_key("<i>Attention</i> Is All You Need") == "attention is all you need"

    def test_a_subtitle_is_split_off_for_the_stem(self):
        assert core_title_key("BERT: Pre-training of Deep Bidirectional Transformers") == "bert"
        assert core_title_key("Attention Is All You Need") == "attention is all you need"

    def test_a_hyphenated_word_is_not_mistaken_for_a_subtitle(self):
        assert core_title_key("Long Short-Term Memory") == "long short term memory"

    @pytest.mark.parametrize(
        ("left", "right"),
        [("Kaiming He", "He, Kaiming"), ("Kaiming He", "K. He"), ("  Kaiming  He ", "kaiming he")],
    )
    def test_author_name_orderings_reduce_to_the_surname(self, left, right):
        assert normalize_author_key(left) == normalize_author_key(right) == "he"

    def test_honorifics_and_suffixes_are_not_part_of_the_name(self):
        assert normalize_author_key("Kaiming He Jr.") == "he"
        assert normalize_author_key("Dr. Kaiming He") == "he"

    def test_a_cjk_name_is_kept_whole(self):
        assert normalize_author_key("张小龙") == "张小龙"

    def test_the_fallback_key_requires_all_three_components(self):
        complete = PaperIdentity(title=PREPRINT_TITLE, year="2015", first_author="He")
        no_author = PaperIdentity(title=PREPRINT_TITLE, year="2015")
        no_year = PaperIdentity(title=PREPRINT_TITLE, first_author="He")

        expected = f"{normalize_title_key(PREPRINT_TITLE)}|2015|he"
        assert complete.fallback_key == expected
        assert no_author.fallback_key == ""
        assert no_year.fallback_key == ""

    def test_a_generic_title_produces_no_key_at_all(self):
        identity = PaperIdentity(title="Editorial", year="2016", first_author="Some Editor")

        assert identity.fallback_key == ""
        assert identity.key == ""
        assert "editorial" in GENERIC_TITLES

    @pytest.mark.parametrize("title", ["AI", "x"])
    def test_a_title_too_short_to_mean_anything_is_refused(self, title):
        identity = PaperIdentity(title=title, year="2016", first_author="He")

        assert identity.fallback_key == ""


class TestTitleEquivalence:
    def test_a_dropped_subtitle_still_counts_as_the_same_title(self):
        # The publisher titled it "BERT"; the preprint kept the full title.
        assert titles_equivalent("BERT: Pre-training of Deep Bidirectional Transformers", "BERT") is True

    def test_a_reworded_subtitle_on_a_short_stem_does_not(self):
        # Deliberately a miss: "BERT" alone cannot distinguish the paper from an
        # erratum, a reproduction or a review of it.
        assert (
            titles_equivalent(
                "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "BERT: Pre-training of Bidirectional Transformers",
            )
            is False
        )

    def test_a_long_distinctive_stem_survives_a_reworded_subtitle(self):
        assert (
            titles_equivalent(
                "Contrastive Pretraining of Language Models: A Framework",
                "Contrastive Pretraining of Language Models: In Practice",
            )
            is True
        )

    def test_unrelated_titles_never_match(self):
        assert titles_equivalent("Attention Is All You Need", "Deep Residual Learning") is False

    def test_an_empty_title_matches_nothing(self):
        assert titles_equivalent("", PREPRINT_TITLE) is False


class TestPreprintPublishedPair:
    """The acceptance criterion: 预印本/正式版去重正确."""

    def test_the_pair_does_not_share_a_key_before_enrichment(self):
        # This is the problem being solved, stated as a fact: two records, two
        # different canonical keys, one paper.
        assert preprint_identity().key == f"arxiv:{PREPRINT_ARXIV}"
        assert published_identity().key == f"doi:{PUBLISHED_DOI.lower()}"
        assert preprint_identity().key != published_identity().key

    def test_title_and_author_evidence_already_pairs_them(self):
        match = compare_identities(preprint_identity(), published_identity())

        assert match.same_paper is True
        # "probable", not "exact": the year changed and no identifier is shared,
        # so this rests on the title and the author alone.
        assert match.confidence == "probable"
        assert match.requires_confirmation is True

    def test_the_verdict_asks_for_doi_recovery_rather_than_claiming_certainty(self):
        match = compare_identities(preprint_identity(), published_identity())

        assert "回查" in "".join(match.notes)

    def test_recovering_the_doi_makes_the_pair_exact(self):
        recovered = preprint_identity(doi=PUBLISHED_DOI)
        match = compare_identities(recovered, published_identity())

        assert match.same_paper is True
        assert match.confidence == "exact"
        assert match.basis == "shared_doi"
        assert match.requires_confirmation is False

    def test_merging_unifies_on_the_doi_while_keeping_the_arxiv_id(self):
        result = merge_identities(preprint_identity(doi=PUBLISHED_DOI), published_identity())

        assert result.identity.doi == PUBLISHED_DOI.lower()
        assert result.identity.arxiv_id == PREPRINT_ARXIV
        # DOI outranks arXiv, so the merged record keys on the DOI.
        assert result.identity.key_tier == "doi"
        assert result.conflicts == ()

    def test_a_shared_arxiv_id_unifies_a_preprint_against_its_own_journal_doi(self):
        # The other direction: the published record remembers the arXiv id.
        both = published_identity(arxiv_id="arXiv:1512.03385v4")
        match = compare_identities(preprint_identity(), both)

        assert match.same_paper is True
        assert match.confidence == "exact"
        assert match.basis == "shared_arxiv"

    def test_a_year_gap_wider_than_one_year_is_not_the_same_paper(self):
        match = compare_identities(preprint_identity(), published_identity(year="2019"))

        assert match.same_paper is False


class TestDistinctPapersMustNotMerge:
    """A title-only collision across genuinely different papers."""

    def test_the_same_title_by_different_authors_does_not_merge(self):
        # "Deep Learning" is a real title collision: the 2016 Handbook of
        # NIPS chapter by Bengio/Goodfellow/Courville and the 2012 NIPS tutorial
        # of the same name by Krizhevsky/Sutskever/Hinton are different works.
        handbook = PaperIdentity(title="Deep Learning", year="2016", first_author="Yoshua Bengio")
        tutorial = PaperIdentity(title="Deep Learning", year="2012", first_author="Alex Krizhevsky")

        match = compare_identities(handbook, tutorial)

        assert match.same_paper is False
        assert handbook.fallback_key != tutorial.fallback_key

    def test_the_same_title_and_author_in_different_years_does_not_merge(self):
        first = PaperIdentity(title="Deep Learning", year="2016", first_author="Yoshua Bengio")
        second = PaperIdentity(title="Deep Learning", year="2021", first_author="Yoshua Bengio")

        assert compare_identities(first, second).same_paper is False

    def test_the_same_title_with_conflicting_dois_is_a_hard_no(self):
        # Both records have a DOI and they disagree: two different published
        # objects, whatever the titles say.
        left = published_identity()
        right = published_identity(doi="10.5555/other.paper")

        match = compare_identities(left, right)

        assert match.same_paper is False
        assert match.basis == "doi_conflict"
        assert match.key == ""

    def test_a_conflict_is_explained_to_the_user_with_both_values(self):
        left = published_identity()
        right = published_identity(doi="10.5555/other.paper")

        explanation = "".join(compare_identities(left, right).notes)

        assert "10.1109/cvpr.2016.90" in explanation
        assert "10.5555/other.paper" in explanation

    @pytest.mark.parametrize("tier", ["arxiv", "s2", "openalex"])
    def test_a_conflict_on_any_strong_tier_is_a_hard_no(self, tier):
        field = {"arxiv": "arxiv_id", "s2": "s2_paper_id", "openalex": "openalex_id"}[tier]
        left = PaperIdentity(
            **{field: "2301.00001" if tier == "arxiv" else ("a" * 40 if tier == "s2" else "W1111111")},
            title=PREPRINT_TITLE,
            year="2015",
            first_author="Kaiming He",
        )
        right = PaperIdentity(
            **{field: "2401.01234" if tier == "arxiv" else ("b" * 40 if tier == "s2" else "W2222222")},
            title=PREPRINT_TITLE,
            year="2015",
            first_author="Kaiming He",
        )

        assert compare_identities(left, right).same_paper is False

    def test_two_generic_titles_do_not_merge_even_with_the_same_author_and_year(self):
        left = PaperIdentity(title="Editorial", year="2016", first_author="Jane Editor")
        right = PaperIdentity(title="Editorial", year="2016", first_author="Jane Editor")

        assert left.key == ""
        assert compare_identities(left, right).same_paper is False

    def test_records_with_unrelated_identifiers_are_reported_as_undecided(self):
        s2_only = PaperIdentity(
            s2_paper_id="a" * 40, title=PREPRINT_TITLE, year="2015", first_author="Kaiming He"
        )
        openalex_only = PaperIdentity(
            openalex_id="W2963402937", title="Something Else Entirely", year="2015",
            first_author="Kaiming He",
        )

        match = compare_identities(s2_only, openalex_only)

        assert match.same_paper is False
        assert match.basis == "disjoint_identifiers"

    def test_a_record_missing_its_year_is_reported_as_incomplete(self):
        no_year = PaperIdentity(title=PREPRINT_TITLE, first_author="Kaiming He")
        other = PaperIdentity(title="A Different Paper Entirely", year="2015", first_author="Kaiming He")

        assert compare_identities(no_year, other).basis == "fallback_incomplete"


class TestMerging:
    def test_identifiers_from_both_sides_are_unioned(self):
        left = preprint_identity(s2_paper_id="a" * 40)
        right = published_identity(openalex_id="W2963402937")

        merged = merge_identities(left, right).identity

        assert merged.arxiv_id == PREPRINT_ARXIV
        assert merged.s2_paper_id == "a" * 40
        assert merged.doi == PUBLISHED_DOI.lower()
        assert merged.openalex_id == "W2963402937"

    def test_a_conflict_is_recorded_rather_than_silently_resolved(self):
        result = merge_identities(published_identity(), published_identity(doi="10.5555/other.paper"))

        assert result.same_paper is False
        assert result.is_trustworthy is False
        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert conflict.field == "doi"
        assert {conflict.left, conflict.right} == {"10.1109/cvpr.2016.90", "10.5555/other.paper"}
        assert "不一致" in conflict.message

    def test_a_conflict_does_not_silently_drop_an_identifier(self):
        result = merge_identities(published_identity(), published_identity(doi="10.5555/other.paper"))

        assert result.identity.doi in {"10.1109/cvpr.2016.90", "10.5555/other.paper"}
        assert result.identity.key_tier == "doi"

    def test_the_merged_key_does_not_depend_on_argument_order(self):
        left = preprint_identity(doi=PUBLISHED_DOI)
        right = published_identity()

        assert merge_identities(left, right).identity.key == merge_identities(right, left).identity.key

    def test_a_differing_first_author_is_surfaced(self):
        result = merge_identities(
            PaperIdentity(title=PREPRINT_TITLE, year="2015", first_author="Kaiming He"),
            PaperIdentity(title=PREPRINT_TITLE, year="2015", first_author="Someone Else"),
        )

        assert [conflict.field for conflict in result.conflicts] == ["first_author"]

    def test_serialization_round_trips(self):
        identity = preprint_identity(doi=PUBLISHED_DOI, s2_paper_id="a" * 40)

        assert PaperIdentity.from_dict(identity.to_dict()) == identity

    def test_from_dict_ignores_keys_it_does_not_know(self):
        assert PaperIdentity.from_dict({"title": PREPRINT_TITLE, "nonsense": 1}).title == PREPRINT_TITLE


class TestIdentityFromRecord:
    def test_an_arxiv_record_yields_an_arxiv_identity(self):
        # normalize_paper_record puts the full arXiv URL in paper_id.
        record = {
            "source": "arXiv",
            "paper_id": "http://arxiv.org/abs/1512.03385v1",
            "title": PREPRINT_TITLE,
            "year": "2015",
            "authors": ["Kaiming He", "Xiangyu Zhang"],
        }

        identity = identity_from_record(record)

        assert identity.arxiv_id == PREPRINT_ARXIV
        assert identity.doi == ""
        assert identity.first_author == "Kaiming He"
        assert identity.key == f"arxiv:{PREPRINT_ARXIV}"

    def test_a_semantic_scholar_record_yields_an_s2_identity(self):
        record = {
            "source": "Semantic Scholar",
            "paper_id": "b" * 40,
            "title": PREPRINT_TITLE,
            "year": "2015",
            "authors": ["Kaiming He"],
        }

        assert identity_from_record(record).s2_paper_id == "b" * 40

    def test_an_explicit_doi_field_is_honoured(self):
        # normalize_paper_record does not emit this today; S2 puts it in
        # externalIds, which that function drops.
        record = {"source": "Semantic Scholar", "paper_id": "b" * 40, "doi": PUBLISHED_DOI}

        assert identity_from_record(record).doi == PUBLISHED_DOI.lower()

    def test_a_record_with_nothing_identifiable_has_an_empty_key(self):
        assert identity_from_record({"title": PREPRINT_TITLE}).key == ""


class TestEnrichmentRecoversAMissingDoi:
    """DOI 丢失可回查."""

    def test_crossref_supplies_the_doi_the_arxiv_record_lacked(self):
        source = fake_source(
            "Crossref", crossref_payload(crossref_item())
        )
        record = preprint_identity()

        result = enrich_identity(record, [source])

        assert result.identity.doi == PUBLISHED_DOI.lower()
        assert result.recovered_doi is True
        assert result.ok is True
        # DOI now outranks the arXiv id it was recovered for.
        assert result.identity.key_tier == "doi"

    def test_the_record_is_untouched_except_for_what_was_missing(self):
        source = fake_source("Crossref", crossref_payload(crossref_item()))
        record = preprint_identity()

        result = enrich_identity(record, [source])

        assert result.identity.arxiv_id == PREPRINT_ARXIV
        assert result.identity.title == PREPRINT_TITLE
        assert result.filled_fields == ("doi",)

    def test_the_recovery_is_reported_in_chinese(self):
        result = enrich_identity(preprint_identity(), [fake_source("Crossref", crossref_payload(crossref_item()))])

        assert any(PUBLISHED_DOI.lower() in message for message in result.messages_zh)

    def test_openalex_can_supply_the_doi_too(self):
        payload = {
            "results": [
                {
                    "id": "https://openalex.org/W2963402937",
                    "doi": f"https://doi.org/{PUBLISHED_DOI}",
                    "title": PREPRINT_TITLE,
                    "publication_year": 2016,
                    "authorships": [{"author": {"display_name": "Kaiming He"}}],
                }
            ]
        }
        source = fake_source("OpenAlex", payload, parser=parse_openalex_payload)

        result = enrich_identity(preprint_identity(), [source])

        assert result.identity.doi == PUBLISHED_DOI.lower()
        assert result.identity.openalex_id == "W2963402937"

    def test_crossref_also_yields_the_arxiv_id_of_the_preprint(self):
        item = crossref_item()
        item["relation"] = {"has-preprint": [{"id": f"10.48550/arXiv.{PREPRINT_ARXIV}"}]}
        result = enrich_identity(
            PaperIdentity(title=PREPRINT_TITLE, year="2015", first_author="Kaiming He"),
            [fake_source("Crossref", crossref_payload(item))],
        )

        assert result.identity.arxiv_id == PREPRINT_ARXIV

    def test_a_recovered_doi_makes_the_preprint_pair_merge_exactly(self):
        # End to end: recover the DOI, and the arXiv record and the CVPR record
        # stop merely "probably" being one paper.
        source = fake_source("Crossref", crossref_payload(crossref_item()))
        enriched = enrich_identity(preprint_identity(), [source]).identity

        match = compare_identities(enriched, published_identity())

        assert match.confidence == "exact"
        assert match.same_paper is True

    def test_a_record_that_already_has_a_doi_asks_about_that_doi(self):
        seen = []

        def _fetch(query):
            seen.append(query)
            return crossref_payload(crossref_item())

        enrich_identity(published_identity(), [MetadataSource("Crossref", _fetch, parse_crossref_payload)])

        assert seen[0].is_doi_lookup is True
        assert seen[0].doi == PUBLISHED_DOI.lower()


class TestEnrichmentRefusesGuesses:
    """A wrong DOI is worse than a missing one, so candidates must earn it."""

    def test_a_candidate_with_a_different_author_is_rejected(self):
        item = crossref_item(family="Somebody", given="Else")
        result = enrich_identity(preprint_identity(), [fake_source("Crossref", crossref_payload(item))])

        assert result.identity == preprint_identity()
        assert result.filled_fields == ()

    def test_a_candidate_from_too_far_away_in_time_is_rejected(self):
        item = crossref_item(year=1999)
        result = enrich_identity(preprint_identity(), [fake_source("Crossref", crossref_payload(item))])

        assert result.identity.doi == ""

    def test_a_candidate_with_an_unrelated_title_is_rejected(self):
        item = crossref_item(title="A Completely Different Paper About Something Else")
        result = enrich_identity(preprint_identity(), [fake_source("Crossref", crossref_payload(item))])

        assert result.identity.doi == ""

    def test_the_relevant_hit_wins_over_a_more_relevant_but_wrong_one(self):
        payload = crossref_payload(
            crossref_item(doi="10.5555/wrong.title", title="Deep Residual Networks Revisited"),
            crossref_item(doi=PUBLISHED_DOI),
        )
        result = enrich_identity(preprint_identity(), [fake_source("Crossref", payload)])

        assert result.identity.doi == PUBLISHED_DOI.lower()

    def test_two_different_verified_dois_are_reported_as_ambiguous_not_guessed(self):
        payload = crossref_payload(crossref_item(doi=PUBLISHED_DOI), crossref_item(doi="10.5555/also.plausible"))
        result = enrich_identity(preprint_identity(), [fake_source("Crossref", payload)])

        assert result.ambiguous is True
        assert result.identity == preprint_identity()
        assert result.identity.doi == ""
        assert "无法确定" in "".join(result.notes)

    def test_two_hits_agreeing_on_one_doi_are_not_ambiguous(self):
        payload = crossref_payload(crossref_item(doi=PUBLISHED_DOI), crossref_item(doi=PUBLISHED_DOI.upper()))
        result = enrich_identity(preprint_identity(), [fake_source("Crossref", payload)])

        assert result.ambiguous is False
        assert result.identity.doi == PUBLISHED_DOI.lower()

    def test_candidates_without_a_doi_are_dropped_entirely(self):
        item = crossref_item()
        del item["DOI"]
        result = enrich_identity(preprint_identity(), [fake_source("Crossref", crossref_payload(item))])

        assert result.identity == preprint_identity()


class TestEnrichmentDegrades:
    """A provider outage must not become a user-visible failure."""

    def test_a_network_failure_degrades_to_no_doi_instead_of_raising(self):
        source = raising_source("Crossref", urllib.error.URLError("connection refused"))

        result = enrich_identity(preprint_identity(), [source])

        assert isinstance(result, EnrichmentResult)
        assert result.identity == preprint_identity()
        assert result.identity.doi == ""
        assert result.errors

    def test_a_rate_limit_is_reported_without_breaking_anything(self):
        source = raising_source("Crossref", urllib.error.HTTPError("u", 429, "Too Many", {}, None))

        result = enrich_identity(preprint_identity(), [source])

        assert result.identity == preprint_identity()
        assert "rate limit" in "".join(result.errors)

    def test_a_timeout_is_reported(self):
        result = enrich_identity(preprint_identity(), [raising_source("Crossref", TimeoutError("timed out"))])

        assert "timed out" in "".join(result.errors)

    def test_an_empty_response_leaves_the_record_alone(self):
        result = enrich_identity(preprint_identity(), [fake_source("Crossref", None)])

        assert result.identity == preprint_identity()
        assert result.filled_fields == ()

    def test_a_malformed_payload_leaves_the_record_alone(self):
        def _parse(payload):
            raise ValueError("unexpected shape")

        source = MetadataSource("Crossref", lambda query: {"message": {}}, _parse)
        result = enrich_identity(preprint_identity(), [source])

        assert result.identity == preprint_identity()
        assert "无法解析" in "".join(result.errors)

    def test_a_failing_provider_does_not_stop_the_next_one(self):
        failing = raising_source("Crossref", TimeoutError("timed out"))
        working = fake_source("OpenAlex", {"results": [
            {
                "id": "https://openalex.org/W1",
                "doi": f"https://doi.org/{PUBLISHED_DOI}",
                "title": PREPRINT_TITLE,
                "publication_year": 2016,
                "authorships": [{"author": {"display_name": "Kaiming He"}}],
            }
        ]}, parser=parse_openalex_payload)

        result = enrich_identity(preprint_identity(), [failing, working])

        assert result.identity.doi == PUBLISHED_DOI.lower()
        assert result.errors  # the Crossref failure is still reported

    def test_enriching_nothing_is_a_no_op(self):
        calls = []

        def _fetch(query):
            calls.append(query)
            return crossref_payload(crossref_item())

        result = enrich_identity(
            published_identity(), [MetadataSource("Crossref", _fetch, parse_crossref_payload)]
        )

        assert calls  # the DOI lookup still happens
        assert result.filled_fields == ()
        assert result.identity == published_identity()

    def test_a_non_identity_argument_is_rejected_loudly(self):
        with pytest.raises(TypeError):
            enrich_identity({"title": PREPRINT_TITLE}, [])


class TestProviderPayloadParsing:
    def test_a_single_work_crossref_response_is_accepted_too(self):
        # /works/{doi} returns message without an items list.
        result = enrich_identity(
            published_identity(year=""),
            [fake_source("Crossref", {"message": crossref_item()})],
        )

        assert result.identity.year == "2016"

    def test_a_single_work_openalex_response_is_accepted_too(self):
        result = enrich_identity(
            published_identity(year=""),
            [
                fake_source(
                    "OpenAlex",
                    {
                        "id": "https://openalex.org/W1",
                        "doi": f"https://doi.org/{PUBLISHED_DOI}",
                        "title": PREPRINT_TITLE,
                        "publication_year": 2016,
                    },
                    parser=parse_openalex_payload,
                )
            ],
        )

        assert result.identity.year == "2016"

    def test_parsing_is_a_pure_function_of_the_payload(self):
        payload = crossref_payload(crossref_item())

        assert parse_crossref_payload(payload) == parse_crossref_payload(payload)

    def test_a_non_dict_payload_yields_no_candidates(self):
        assert parse_crossref_payload(None) == []
        assert parse_openalex_payload({"nonsense": 1}) == []


class TestProviderUrlsWithoutNetwork:
    """The production fetchers are exercised for URL construction only."""

    def test_a_bibliographic_lookup_asks_for_several_candidates(self):
        seen = []
        source = build_crossref_source(fetch=lambda url, **kwargs: seen.append(url))
        source.fetch(LookupQuery(title=PREPRINT_TITLE, year="2015", first_author=PREPRINT_AUTHOR))

        url = seen[0]
        assert url.startswith("https://api.crossref.org/works?")
        assert "query.bibliographic=Deep+Residual+Learning+for+Image+Recognition+2015+He" in url
        # rows=1 would make a wrong top hit indistinguishable from a right one.
        assert "rows=5" in url

    def test_a_doi_lookup_addresses_the_single_work_endpoint(self):
        seen = []
        source = build_crossref_source(fetch=lambda url, **kwargs: seen.append(url))
        source.fetch(LookupQuery(doi="10.1109/cvpr.2016.90"))

        assert seen[0] == "https://api.crossref.org/works/10.1109/cvpr.2016.90"

    def test_openalex_addresses_its_doi_endpoint(self):
        seen = []
        source = build_openalex_source(fetch=lambda url, **kwargs: seen.append(url))
        source.fetch(LookupQuery(doi="10.1109/cvpr.2016.90"))

        assert seen[0] == "https://api.openalex.org/works/doi:10.1109/cvpr.2016.90"

    def test_the_contact_address_is_appended_when_configured(self, monkeypatch):
        monkeypatch.setattr("paperwhisperer.search.enrich.ENRICHMENT_CONTACT_EMAIL", "dev@example.com")
        seen = []
        source = build_crossref_source(fetch=lambda url, **kwargs: seen.append(url))
        source.fetch(LookupQuery(title=PREPRINT_TITLE))

        assert seen[0].endswith("&mailto=dev%40example.com")

    def test_the_surname_is_what_goes_into_a_bibliographic_query(self):
        query = LookupQuery(title=PREPRINT_TITLE, year="2015", first_author=PREPRINT_AUTHOR)

        assert query.bibliographic_text() == "Deep Residual Learning for Image Recognition 2015 He"


class TestDeterminism:
    def test_the_same_record_always_yields_the_same_key(self):
        keys = {preprint_identity().key for _ in range(200)}

        assert keys == {f"arxiv:{PREPRINT_ARXIV}"}

    def test_the_key_does_not_depend_on_which_fields_were_supplied_first(self):
        one = PaperIdentity(doi=PUBLISHED_DOI, arxiv_id=PREPRINT_ARXIV, title=PREPRINT_TITLE)
        other = PaperIdentity(title=PREPRINT_TITLE, arxiv_id=PREPRINT_ARXIV, doi=PUBLISHED_DOI)

        assert one.key == other.key == f"doi:{PUBLISHED_DOI.lower()}"

    def test_iterating_the_tier_mapping_follows_the_priority_order(self):
        identity = PaperIdentity(
            openalex_id="W2963402937",
            s2_paper_id="a" * 40,
            arxiv_id=PREPRINT_ARXIV,
            doi=PUBLISHED_DOI,
            title=PREPRINT_TITLE,
            year="2015",
            first_author=PREPRINT_AUTHOR,
        )

        assert list(identity.keys) == ["doi", "arxiv", "s2", "openalex", "title"]

    def test_identities_are_hashable_and_can_deduplicate(self):
        identities = {preprint_identity(), preprint_identity(), published_identity()}

        assert len(identities) == 2

    def test_enrichment_is_deterministic(self):
        source = fake_source("Crossref", crossref_payload(crossref_item()))

        outcomes = {enrich_identity(preprint_identity(), [source]).identity for _ in range(50)}

        assert len(outcomes) == 1

    def test_repeated_comparisons_give_the_same_verdict(self):
        verdicts = {
            (compare_identities(preprint_identity(), published_identity()).basis,) for _ in range(50)
        }

        assert verdicts == {("core_title_key_bridge",)}
