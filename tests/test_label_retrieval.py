"""Tests for tools/label_retrieval.py.

The properties worth pinning, in order of how much damage their absence would do:

1. **The loader accepts the export unchanged.** A label format that does not
   load is a label format that does not exist, so one test runs the *real*
   ``eval.gates.load_labels`` over a generated export. That test fails the
   moment someone adds a field the loader rejects.
2. **An unconfirmed decision cannot reach the gate's input file.** Not as a
   convention -- as the export's actual behaviour. ``assistant_draft`` rows go
   to the pending file unless a human name says otherwise.
3. **Revising is append-only.** The log keeps every version, the head is the
   latest, and re-examining without changing anything is recorded rather than
   discarded.
4. **The audit tail exists.** Without windows the ranking did not promote, the
   recall number would be a property of the candidate list rather than of the
   retriever.

The rest is arithmetic and validation, tested where it is easy to get wrong
and hard to notice: window splitting (CJK), selection parsing, and kappa's
degenerate cases.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import label_retrieval as L  # noqa: E402

DOC = "digital_two_col/example.pdf"


# --------------------------------------------------------------------------
# Fixtures: a worklist that needs no corpus and no PDF
# --------------------------------------------------------------------------


def _windows() -> list[L.Window]:
    rows = []
    for element_index in range(2):
        element_id = f"el_{element_index:04d}"
        for window_index in range(4):
            rows.append(L.Window(
                window_id=f"{element_id}#w{window_index}",
                element_id=element_id,
                element_order=element_index,
                locator={"kind": "pdf_page", "page": element_index + 1},
                window_index=window_index,
                text=(
                    f"Sentence {window_index} about batch normalisation and gamma."
                    if element_index == 0
                    else f"Sentence {window_index} about the runner-up topic entirely."
                ),
            ))
    return rows


def _worklist() -> L.Worklist:
    worklist = L.Worklist(corpus_version="sha256:test", built_at="2026-10-02T00:00:00+00:00")
    worklist.documents[DOC] = {
        "sha256": "abc",
        "parser": "legacy",
        "elements": 2,
        "windows": [window.to_dict() for window in _windows()],
    }
    for query_id in ("q001", "q002"):
        worklist.queries.append({
            "id": query_id,
            "document": DOC,
            "query": "What does batch normalisation do to the activations?",
            "language": "en",
            "kind": "prose",
            "query_source": "assistant_draft",
            "note": "",
        })
        worklist.candidates[query_id] = L.rank_candidates(
            worklist.query(query_id)["query"], query_id, _windows(), top_k=3, audit=1
        )
    return worklist


@pytest.fixture
def worklist(tmp_path: Path) -> L.Worklist:
    item = _worklist()
    # Stashed on the instance (never serialised) so the tests can pass a real
    # path to the CLI rather than reaching for a global.
    item.path = item.save(tmp_path / "worklist.json")
    return item


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(L, "ANNOTATIONS_DIR", tmp_path / "annotations")
    monkeypatch.setattr(L, "EXPORT_PATH", tmp_path / "retrieval_labels.yaml")
    monkeypatch.setattr(L, "PENDING_PATH", tmp_path / "pending_labels.yaml")
    monkeypatch.setattr(L, "AGREEMENT_PATH", tmp_path / "agreement.json")


def _args(**kwargs: object) -> SimpleNamespace:
    base = {"worklist": None, "stream": io.StringIO("")}
    base.update(kwargs)
    return SimpleNamespace(**base)


def _record(annotator: str, query: L.Worklist, query_id: str, window_ids: list[str],
            *, source: str = L.SOURCE_HUMAN, seconds: float | None = None,
            language: str = "en", kind: str = "prose") -> dict:
    return L.make_relevance_record(
        annotator=annotator,
        query=query.query(query_id),
        window_ids=window_ids,
        element_ids=L._element_ids_for(query, str(query.query(query_id)["document"]), window_ids),
        language=language,
        kind=kind,
        source=source,
        seconds=seconds,
    )


# --------------------------------------------------------------------------
# The loader accepts the export, unchanged
# --------------------------------------------------------------------------


def test_export_loads_with_the_real_gate_loader(tmp_path: Path, worklist: L.Worklist) -> None:
    """The committed schema must be the one eval/gates.py reads, not a cousin of it."""
    from eval.gates import load_labels

    candidates = worklist.candidates_for("q001")
    chosen = [str(candidates[0]["window_id"]), str(candidates[1]["window_id"])]
    L.append_decision("human:ann", _record("human:ann", worklist, "q001", chosen))
    L.append_decision("human:ann", _record("human:ann", worklist, "q002", [chosen[0]]))
    L.append_decision(
        "human:ann",
        L.make_claim_record(
            annotator="human:ann",
            claim={"id": "q001-c1", "query_id": "q001", "element_id": "el_0000",
                   "quote": "a quote"},
            supported=True,
            source=L.SOURCE_HUMAN,
        ),
        "claims",
    )

    out = tmp_path / "retrieval_labels.yaml"
    assert L.main(["export", "--worklist", str(worklist.path),
                   "--annotator", "human:ann", "--out", str(out),
                   "--pending-out", str(tmp_path / "pending.yaml")]) == 0

    labels = load_labels(out)
    assert [item.query_id for item in labels.queries] == ["q001", "q002"]
    assert labels.queries[0].relevant_element_ids  # never empty: load_labels rejects empty
    assert labels.claims[0].supported is True
    assert labels.schema == L.LABELS_SCHEMA


def test_export_keeps_window_ids_so_element_collapse_is_visible(tmp_path: Path,
                                                                worklist: L.Worklist) -> None:
    two_windows_one_element = ["el_0000#w0", "el_0000#w2"]
    L.append_decision("human:ann", _record("human:ann", worklist, "q001", two_windows_one_element))
    payload, _pending, _warnings = L.collect_export(worklist, ["human:ann"])
    row = payload["queries"][0]
    assert row["relevant_element_ids"] == ["el_0000"]
    assert row["relevant_window_ids"] == two_windows_one_element


# --------------------------------------------------------------------------
# Provenance: the line between proposed and confirmed
# --------------------------------------------------------------------------


def test_assistant_labels_do_not_reach_the_gate_file_without_an_explicit_flag(
    worklist: L.Worklist,
) -> None:
    candidates = worklist.candidates_for("q001")
    chosen = [str(candidates[0]["window_id"])]
    L.append_decision(
        "agent:draft",
        _record("agent:draft", worklist, "q001", chosen, source=L.SOURCE_ASSISTANT),
    )
    payload, pending, _warnings = L.collect_export(worklist, ["agent:draft"])
    assert payload["queries"] == []
    assert pending["queries"][0]["human_confirmed"] is False
    assert pending["queries"][0]["provenance"] == "assistant_draft"
    assert payload["human_confirmed"] is True  # nothing unconfirmed got in


def test_accepting_assistant_labels_stamps_every_row(tmp_path: Path, worklist: L.Worklist) -> None:
    """The override exists, and it makes the file say what it is in three places."""
    out = tmp_path / "labels.yaml"
    candidates = worklist.candidates_for("q001")
    for annotator in ("human:ann", "agent:draft"):
        L.append_decision(annotator, _record(
            annotator, worklist, "q001", [str(candidates[0]["window_id"])],
            source=L.SOURCE_HUMAN if annotator.startswith("human") else L.SOURCE_ASSISTANT,
        ))
    assert L.main(["export", "--worklist", str(worklist.path),
                   "--annotator", "human:ann", "--annotator", "agent:draft",
                   "--accept-assistant-labels", "--out", str(out),
                   "--pending-out", str(tmp_path / "pending.yaml")]) == 0
    text = out.read_text(encoding="utf-8")
    assert "NOT GATE-GRADE" in text
    payload = yaml.safe_load(text)
    assert payload["human_confirmed"] is False
    # One row per id even though two annotators labelled it: consolidated.
    row = payload["queries"][0]
    assert set(row["annotated_by"].split(",")) == {"human:ann", "agent:draft"}
    assert row["annotated_by_kind"] == "mixed"  # not purely human
    assert row["human_confirmed"] is False


def test_two_annotators_on_one_query_collapse_to_one_row(worklist: L.Worklist) -> None:
    """The scorer keys recall by query_id, so a duplicated id would double-count."""
    # Deliberately different elements: the union must be visible in the row.
    L.append_decision("human:a", _record("human:a", worklist, "q001", ["el_0000#w0"]))
    L.append_decision("human:b", _record("human:b", worklist, "q001",
                                         ["el_0000#w0", "el_0001#w0"]))
    payload, _pending, _warnings = L.collect_export(worklist, ["human:a", "human:b"])
    assert [row["id"] for row in payload["queries"]] == ["q001"]
    row = payload["queries"][0]
    assert row["n_annotators"] == 2
    assert row["annotators_agreed"] is False
    assert row["relevant_element_ids"] == ["el_0000", "el_0001"]  # union, stable order
    assert set(row["annotator_element_sets"]) == {"human:a", "human:b"}


def test_consolidated_claim_is_conservative_when_votes_split(worklist: L.Worklist) -> None:
    L.append_decision("human:a", _record("human:a", worklist, "q001", ["el_0000#w0"]))
    for annotator, supported in (("human:a", True), ("human:b", False)):
        L.append_decision(annotator, L.make_claim_record(
            annotator=annotator,
            claim={"id": "q001-c1", "query_id": "q001", "element_id": "el_0000"},
            supported=supported,
            source=L.SOURCE_HUMAN,
        ), "claims")
    payload, _pending, _warnings = L.collect_export(worklist, ["human:a", "human:b"])
    row = payload["claims"][0]
    assert row["supported"] is False  # split vote -> conservative reading
    assert row["annotators_agreed"] is False
    assert row["annotator_votes"] == {"human:a": True, "human:b": False}


def test_pressing_enter_is_a_confirmation_but_is_counted_separately(
    worklist: L.Worklist,
) -> None:
    """A pin on a judgement call, not an accident.

    Entering a set by hand and pressing enter on the ranking are both a person's
    act. The first is a decision, the second is an acceptance, and they carry
    different evidence about how much the label is worth -- so they are counted
    apart, and only the first is called ``human``.
    """
    assert L.SOURCE_ACCEPTED in L.HUMAN_SOURCES
    accepted = _record("human:ann", worklist, "q001", ["el_0000#w0"],
                       source=L.SOURCE_ACCEPTED)
    assert accepted["human_confirmed"] is True
    assert accepted["source"] == L.SOURCE_ACCEPTED


# --------------------------------------------------------------------------
# Append-only log, resumability, cheap revision
# --------------------------------------------------------------------------


def test_revision_appends_and_supersedes(worklist: L.Worklist) -> None:
    first = _record("human:ann", worklist, "q001", ["el_0000#w0"])
    L.append_decision("human:ann", first)
    changed = _record("human:ann", worklist, "q001", ["el_0000#w2", "el_0001#w0"])
    L.append_decision("human:ann", changed)

    head = L.latest_decisions("human:ann")
    assert head["q001"]["window_ids"] == ["el_0000#w2", "el_0001#w0"]
    assert changed["action"] == "revise"
    assert changed["revision"] == 1
    assert len(L.read_decisions("human:ann")) == 2  # history survives


def test_reaffirmation_is_recorded_not_discarded(worklist: L.Worklist) -> None:
    L.append_decision("human:ann", _record("human:ann", worklist, "q001", ["el_0000#w0"]))
    again = _record("human:ann", worklist, "q001", ["el_0000#w0"])
    L.append_decision("human:ann", again)
    assert again["action"] == "reaffirm"
    status = L.annotator_status(worklist, "human:ann")
    assert status["reaffirmations"] == 1
    assert status["revisions"] == 0


def test_a_corrupt_log_line_does_not_lock_the_session(tmp_path: Path) -> None:
    path = L._log_path("human:ann")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"subject_id": "q001"}\n{"half-writ\n', encoding="utf-8")
    rows = L.read_decisions("human:ann")
    assert [row["subject_id"] for row in rows] == ["q001"]


def test_interactive_loop_records_element_ids_and_resumes(worklist: L.Worklist) -> None:
    stream = io.StringIO("1,3\nnone\n")  # q001: pick two, q002: none of these
    args = _args(annotator="human:ann", answers=None, review=False, limit=0, agent=False,
                 stream=stream, worklist=str(worklist.path))
    assert L.cmd_label(args) == 0
    head = L.latest_decisions("human:ann")
    assert head["q001"]["window_ids"] == [
        worklist.candidates_for("q001")[0]["window_id"],
        worklist.candidates_for("q001")[2]["window_id"],
    ]
    assert head["q001"]["element_ids"] == ["el_0000"]  # both windows, one element
    assert head["q001"]["source"] == L.SOURCE_HUMAN
    assert head["q002"]["window_ids"] == []
    assert head["q002"]["source"] == L.SOURCE_HUMAN  # "none" is a decision, not a skip

    # Resumable: a second pass has nothing pending, and review mode shows both.
    assert L.cmd_label(_args(annotator="human:ann", answers=None, review=False, limit=0,
                             agent=False, stream=io.StringIO(""),
                             worklist=str(worklist.path))) == 0
    assert L.cmd_label(_args(annotator="human:ann", answers=None, review=True, limit=1,
                             agent=False, stream=io.StringIO("2\n"),
                             worklist=str(worklist.path))) == 0
    assert L.latest_decisions("human:ann")["q001"]["action"] == "revise"


def test_skip_is_distinguishable_from_none(worklist: L.Worklist) -> None:
    stream = io.StringIO("x\n")
    L.cmd_label(_args(annotator="human:ann", answers=None, review=False, limit=0, agent=False,
                      stream=stream, worklist=str(worklist.path)))
    head = L.latest_decisions("human:ann")
    assert head["q001"]["action"] == "skip"
    assert head["q001"]["human_confirmed"] is False
    status = L.annotator_status(worklist, "human:ann")
    assert status["queries_skipped"] == 1
    assert status["queries_remaining"] == 1


# --------------------------------------------------------------------------
# Answer-sheet path (bulk labelling, and the assistant draft a person edits)
# --------------------------------------------------------------------------


def test_answer_sheet_rejects_a_window_that_is_not_in_the_document(tmp_path: Path,
                                                                   worklist: L.Worklist,
                                                                   capsys) -> None:
    sheet = tmp_path / "sheet.yaml"
    sheet.write_text(yaml.safe_dump({
        "source": "human",
        "decisions": [{"query_id": "q001", "window_ids": ["el_0000#w9"]}],
    }, allow_unicode=True), encoding="utf-8")
    args = _args(annotator="human:ann", answers=str(sheet), review=False, limit=0, agent=False,
                 stream=io.StringIO(""), worklist=str(worklist.path))
    with pytest.raises(SystemExit):
        L.cmd_label(args)
    assert not L.read_decisions("human:ann")


def test_answer_sheet_distinguishes_empty_from_none(tmp_path: Path,
                                                    worklist: L.Worklist) -> None:
    sheet = tmp_path / "sheet.yaml"
    sheet.write_text(yaml.safe_dump({
        "source": "human",
        "decisions": [{"query_id": "q001", "window_ids": []}],
    }, allow_unicode=True), encoding="utf-8")
    args = _args(annotator="human:ann", answers=str(sheet), review=False, limit=0, agent=False,
                 stream=io.StringIO(""), worklist=str(worklist.path))
    with pytest.raises(SystemExit):
        L.cmd_label(args)


def test_answer_sheet_flags_a_selection_the_ranker_never_offered(worklist: L.Worklist,
                                                                 tmp_path: Path) -> None:
    sheet = tmp_path / "sheet.yaml"
    sheet.write_text(yaml.safe_dump({
        "source": "human",
        "decisions": [{"query_id": "q001", "window_ids": ["el_0001#w3"], "seconds": 42}],
    }, allow_unicode=True), encoding="utf-8")
    L.cmd_label(_args(annotator="human:ann", answers=str(sheet), review=False, limit=0,
                      agent=False, stream=io.StringIO(""), worklist=str(worklist.path)))
    record = L.latest_decisions("human:ann")["q001"]
    assert record["outside_candidates"] == ["el_0001#w3"]
    payload, _pending, _warnings = L.collect_export(worklist, ["human:ann"])
    assert payload["queries"][0]["selected_outside_ranked_candidates"] == ["el_0001#w3"]
    assert payload["queries"][0]["revisions"] == 0


# --------------------------------------------------------------------------
# The claims axis
# --------------------------------------------------------------------------


def test_claim_candidates_come_from_the_labelled_windows(worklist: L.Worklist) -> None:
    candidates = worklist.candidates_for("q001")
    chosen = [str(candidates[0]["window_id"])]
    L.append_decision("human:ann", _record("human:ann", worklist, "q001", chosen))
    claims = L.build_claim_candidates(worklist, L.latest_decisions("human:ann"))
    assert [claim["id"] for claim in claims] == ["q001-c1"]
    assert claims[0]["window_id"] == chosen[0]
    assert claims[0]["quote"]


def test_claim_sheet_refuses_to_guess_supported(worklist: L.Worklist, tmp_path: Path) -> None:
    candidates = worklist.candidates_for("q001")
    L.append_decision("human:ann", _record("human:ann", worklist, "q001",
                                            [str(candidates[0]["window_id"])]))
    claims = L.build_claim_candidates(worklist, L.latest_decisions("human:ann"))
    sheet = tmp_path / "claims.yaml"
    sheet.write_text(yaml.safe_dump({
        "source": "human",
        "decisions": [{"claim_id": claims[0]["id"], "supported": None}],
    }, allow_unicode=True), encoding="utf-8")
    args = _args(annotator="human:ann", answers=str(sheet), per_query=2, candidates_out=None,
                 agent=False, limit=0, full=False, stream=io.StringIO(""),
                 worklist=str(worklist.path))
    with pytest.raises(SystemExit):
        L.cmd_claims(args)


# --------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------


def test_kappa_on_a_hand_computed_matrix() -> None:
    # 4 both-relevant, 2 only-left, 1 only-right, 9 both-irrelevant: 16 items.
    # p_o = 13/16 = 0.8125; p_e = (6/16)(5/16) + (10/16)(11/16) = 0.546875;
    # kappa = (0.8125 - 0.546875) / (1 - 0.546875).
    pairs = [(True, True)] * 4 + [(True, False)] * 2 + [(False, True)] + \
        [(False, False)] * 9
    result = L.cohens_kappa(pairs)
    assert result["matrix"]["both_relevant"] == 4
    assert result["observed"] == pytest.approx(13 / 16)
    assert result["expected"] == pytest.approx(0.546875)
    assert result["kappa"] == pytest.approx((0.8125 - 0.546875) / (1 - 0.546875))


def test_kappa_undefined_when_both_only_marked_one_category() -> None:
    result = L.cohens_kappa([(True, True)] * 5)
    assert result["kappa"] is None
    assert "无定义" in result["note"]


def test_agreement_reports_set_disagreements(worklist: L.Worklist) -> None:
    L.append_decision("human:a", _record("human:a", worklist, "q001", ["el_0000#w0"]))
    L.append_decision("human:b", _record("human:b", worklist, "q001", ["el_0000#w0",
                                                                    "el_0001#w0"]))
    L.append_decision("human:a", _record("human:a", worklist, "q002", ["el_0000#w1"]))
    L.append_decision("human:b", _record("human:b", worklist, "q002", ["el_0000#w1"]))
    report = L.compare_annotators(worklist, "human:a", "human:b")
    assert report["queries_overlap"] == 2
    assert report["exact_set_agreement"] == pytest.approx(0.5)
    assert report["mean_window_jaccard"] == pytest.approx((0.5 + 1.0) / 2)
    assert [row["query_id"] for row in report["disagreements"]] == ["q001"]
    assert report["window_level"]["matrix"]["only_right"] == 1
    assert report["window_level"]["matrix"]["only_left"] == 0
    assert report["window_level"]["matrix"]["both_relevant"] == 2


def test_agreement_skips_queries_only_one_side_decided(worklist: L.Worklist) -> None:
    L.append_decision("human:a", _record("human:a", worklist, "q001", ["el_0000#w0"]))
    report = L.compare_annotators(worklist, "human:a", "human:b")
    assert report["queries_overlap"] == 0
    assert report["window_level"]["kappa"] is None


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def test_split_windows_joins_english_with_spaces_and_chinese_without() -> None:
    english = L.split_windows("One. Two. Three.", target=100)
    assert english == ["One. Two. Three."]
    chinese = L.split_windows("第一句。第二句。第三句。", target=100)
    assert chinese == ["第一句。第二句。第三句。"]
    assert " " not in chinese[0]


def test_split_windows_packs_to_target_and_keeps_every_sentence() -> None:
    windows = L.split_windows("One. Two. Three. Four.", target=14, overlap=4)
    assert windows[0] == "One. Two. Three."  # "Four." would overflow the target
    assert "Four." in windows[-1]


def test_split_windows_overlaps_so_a_straddling_answer_is_still_present() -> None:
    text = " ".join(f"Sentence number {index}." for index in range(40))
    windows = L.split_windows(text, target=200, overlap=100)
    assert len(windows) > 2
    assert windows[0][-50:] in windows[1] or windows[1].startswith(windows[0][-20:])


def test_split_windows_handles_empty_text() -> None:
    assert L.split_windows("   ") == []
    assert L.split_windows("") == []


# --------------------------------------------------------------------------
# The audit tail and the selection parser
# --------------------------------------------------------------------------


def test_audit_tail_shows_windows_the_ranker_did_not_promote() -> None:
    candidates = L.rank_candidates("gamma and beta", "q001", _windows(), top_k=2, audit=2)
    assert len(candidates) == 4
    assert [row["origin"] for row in candidates] == ["ranked", "ranked", "audit_tail", "audit_tail"]
    assert candidates[0]["rank"] == 1


def test_audit_tail_is_deterministic(worklist: L.Worklist) -> None:
    first = L.rank_candidates("query text", "q001", _windows(), top_k=3, audit=2)
    second = L.rank_candidates("query text", "q001", _windows(), top_k=3, audit=2)
    assert [row["window_id"] for row in first] == [row["window_id"] for row in second]


def test_selection_parser() -> None:
    assert L._parse_selection("1,3-5", 6) == [1, 3, 4, 5]
    assert L._parse_selection("2", 3) == [2]
    assert L._parse_selection("1,1", 3) == [1]
    assert L._parse_selection("9", 3) is None      # out of range
    assert L._parse_selection("what?", 3) is None
    assert L._parse_selection("", 3) is None       # Enter is handled separately


def test_status_counts_leftovers(worklist: L.Worklist) -> None:
    L.append_decision("human:ann", _record("human:ann", worklist, "q001", ["el_0000#w0"]))
    status = L.annotator_status(worklist, "human:ann")
    assert status["queries_total"] == 2
    assert status["queries_decided"] == 1
    assert status["queries_remaining"] == 1
    assert status["human_confirmed_rows"] == 1


def test_committed_export_still_loads() -> None:
    """The deliverable on disk is held to the same loader as the generated one."""
    from eval.gates import load_labels

    # The real repo path, not the monkeypatched one: the point is the committed file.
    committed = L.REPO_ROOT / "eval" / "labels" / "retrieval_labels.yaml"
    if not committed.is_file():
        pytest.skip("eval/labels/retrieval_labels.yaml 尚未导出")
    labels = load_labels(committed)
    assert labels.schema.startswith("paperwhisperer-eval-labels/")
    assert labels.queries
    assert all(item.language in L.LANGUAGES for item in labels.queries)
    assert all(item.kind in L.KINDS for item in labels.queries)
    assert all(item.relevant_element_ids for item in labels.queries)
    ids = [item.query_id for item in labels.queries]
    assert len(ids) == len(set(ids)), "同一条 query 不能出现两行（Recall 分母会被算两遍）"


def test_worklist_round_trips(tmp_path: Path) -> None:
    original = _worklist()
    path = original.save(tmp_path / "worklist.json")
    restored = L.Worklist.load(path)
    assert restored.to_dict() == original.to_dict()
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == L.WORKLIST_SCHEMA
