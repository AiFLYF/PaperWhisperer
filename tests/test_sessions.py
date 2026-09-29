"""Session persistence, token ownership and load-time normalization.

Session files are written by this app but can be hand-edited, truncated by a
crash, or left behind by an expired run, so ``load_session_payload`` is
responsible for both repairing the shape and deleting what cannot be repaired.
These tests cover that contract plus the bearer-token check.
"""

from __future__ import annotations

import json
import time

import pytest

from paperwhisperer.core import config
from paperwhisperer.core.text import build_session_expiry, parse_iso_datetime
from paperwhisperer.sessions import store


def write_raw(session_id, payload, folder):
    path = folder / f"{session_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


class TestTokenHandling:
    def test_hash_is_stable_and_not_the_raw_token(self):
        first = store.hash_session_token("secret-token")
        assert first == store.hash_session_token("secret-token")
        assert first != "secret-token"
        assert len(first) == 64

    def test_empty_token_still_hashes(self):
        assert store.hash_session_token(None) == store.hash_session_token("")

    def test_generated_tokens_are_unique_and_url_safe(self):
        tokens = {store.generate_session_token() for _ in range(50)}
        assert len(tokens) == 50
        assert all(" " not in token for token in tokens)

    def test_validate_accepts_the_matching_token(self):
        payload = {"session_auth": {"token_hash": store.hash_session_token("right")}}
        assert store.validate_session_token(payload, "right") is True

    @pytest.mark.parametrize("provided", ["wrong", "", None])
    def test_validate_rejects_anything_else(self, provided):
        payload = {"session_auth": {"token_hash": store.hash_session_token("right")}}
        assert store.validate_session_token(payload, provided) is False

    def test_validate_rejects_when_no_hash_is_stored(self):
        assert store.validate_session_token({"session_auth": {}}, "anything") is False
        assert store.validate_session_token({}, "anything") is False


class TestBuildSessionPayload:
    def test_never_stores_the_raw_token(self):
        payload = store.build_session_payload(
            session_id="s1",
            source_filename="paper.pdf",
            document_content="body",
            analysis={"summary": "sum"},
            session_token="raw-token",
        )
        assert payload["session_auth"] == {"token_hash": store.hash_session_token("raw-token")}
        assert "raw-token" not in json.dumps(payload)

    def test_omits_the_full_document_unless_configured(self, monkeypatch):
        monkeypatch.setattr(config, "SESSION_PERSIST_FULL_DOCUMENT", False)
        payload = store.build_session_payload("s1", "p.pdf", "the body", {}, "t")
        assert payload["document_content"] == ""
        assert payload["document_excerpt"] == "the body"

    def test_stores_the_full_document_when_configured(self, monkeypatch):
        monkeypatch.setattr(config, "SESSION_PERSIST_FULL_DOCUMENT", True)
        payload = store.build_session_payload("s1", "p.pdf", "the body", {}, "t")
        assert payload["document_content"] == "the body"

    def test_fills_every_analysis_key_even_when_absent(self):
        payload = store.build_session_payload("s1", "p.pdf", "body", {}, "t")
        analysis = payload["analysis"]
        for key in (
            "summary",
            "quotes",
            "mindmap",
            "mermaid",
            "evaluation",
            "research_brief",
            "sections",
            "char_count",
            "elapsed_seconds",
            "output_file",
            "suggested_questions",
            "next_actions",
            "analysis_status",
        ):
            assert key in analysis

    def test_starts_with_an_empty_reading_queue(self):
        payload = store.build_session_payload("s1", "p.pdf", "body", {}, "t")
        assert payload["paper_search"]["reading_queue"] == []


class TestGetSessionDocumentContent:
    def test_prefers_the_full_document(self):
        payload = {"document_content": "full", "document_excerpt": "excerpt"}
        assert store.get_session_document_content(payload) == "full"

    def test_falls_back_to_the_excerpt(self):
        assert store.get_session_document_content({"document_excerpt": "excerpt"}) == "excerpt"

    def test_returns_empty_when_neither_exists(self):
        assert store.get_session_document_content({}) == ""


class TestWriteSessionPayload:
    def test_stamps_updated_at_and_extends_expiry(self, runtime_dirs):
        store.write_session_payload("s1", {"custom": 1})

        payload = store.load_session_payload("s1")
        assert payload["custom"] == 1
        assert payload["updated_at"]
        assert parse_iso_datetime(payload["expires_at"]) > parse_iso_datetime("2000-01-01T00:00:00")

    def test_overwrites_an_existing_file(self, runtime_dirs):
        store.write_session_payload("s1", {"value": "first"})
        store.write_session_payload("s1", {"value": "second"})
        assert store.load_session_payload("s1")["value"] == "second"

    def test_session_id_cannot_escape_the_context_folder(self, runtime_dirs, monkeypatch):
        monkeypatch.setattr(store.config, "CONTEXT_FOLDER", str(runtime_dirs["context"]))
        path = store.get_session_file_path("../evil")
        assert str(runtime_dirs["context"]) in path


class TestLoadSessionPayloadRejectsUnusableFiles:
    def test_returns_none_when_the_file_is_missing(self, runtime_dirs):
        assert store.load_session_payload("never-written") is None

    def test_deletes_and_logs_unparseable_json(self, runtime_dirs, caplog):
        path = runtime_dirs["context"] / "broken.json"
        path.write_text("{not valid json", encoding="utf-8")

        with caplog.at_level("WARNING"):
            assert store.load_session_payload("broken") is None

        assert not path.exists()
        assert "Removed corrupt session file" in caplog.text

    def test_deletes_and_logs_a_non_object_payload(self, runtime_dirs, caplog):
        path = runtime_dirs["context"] / "array.json"
        path.write_text("[]", encoding="utf-8")

        with caplog.at_level("WARNING"):
            assert store.load_session_payload("array") is None

        assert not path.exists()
        assert "Removed non-object session file" in caplog.text

    def test_keeps_the_file_when_it_cannot_be_read(self, runtime_dirs, monkeypatch, caplog):
        path = runtime_dirs["context"] / "locked.json"
        path.write_text("{}", encoding="utf-8")

        def boom(_handle):
            raise OSError("permission denied")

        monkeypatch.setattr(json, "load", boom)

        with caplog.at_level("WARNING"):
            assert store.load_session_payload("locked") is None

        assert path.exists()
        assert "Unable to read session file" in caplog.text

    def test_deletes_and_logs_an_expired_session(self, runtime_dirs, caplog):
        path = runtime_dirs["context"] / "old.json"
        path.write_text(
            json.dumps({"expires_at": "2000-01-01T00:00:00", "document_content": "old"}),
            encoding="utf-8",
        )

        with caplog.at_level("INFO"):
            assert store.load_session_payload("old") is None

        assert not path.exists()
        assert "Removed expired session file" in caplog.text

    def test_replaces_a_malformed_expiry_instead_of_discarding(self, runtime_dirs):
        write_raw("weird", {"expires_at": "not-a-date", "document_content": "body"}, runtime_dirs["context"])

        payload = store.load_session_payload("weird")

        assert payload is not None
        assert parse_iso_datetime(payload["expires_at"]) is not None

    def test_a_session_about_to_expire_is_still_valid(self, runtime_dirs):
        soon = time.time() + 5
        from datetime import datetime

        expiry = datetime.fromtimestamp(soon).isoformat(timespec="seconds")
        write_raw("fresh", {"expires_at": expiry}, runtime_dirs["context"])
        assert store.load_session_payload("fresh") is not None


class TestLoadSessionPayloadNormalizesShape:
    def test_repairs_wrong_container_types(self, runtime_dirs):
        write_raw(
            "mixed",
            {
                "expires_at": build_session_expiry(),
                "qa_history": "not a list",
                "analysis": "not a dict",
                "paper_search": "not a dict",
                "session_auth": "not a dict",
            },
            runtime_dirs["context"],
        )

        payload = store.load_session_payload("mixed")

        assert payload["qa_history"] == []
        assert isinstance(payload["analysis"], dict)
        # A non-dict paper_search is replaced by a fully-populated empty one so
        # callers never have to guard their reads.
        assert payload["paper_search"] == {
            "last_query": "",
            "last_results": [],
            "last_recommendation": {},
            "reading_queue": [],
        }
        assert payload["session_auth"] == {"token_hash": ""}

    def test_backfills_the_document_excerpt(self, runtime_dirs):
        body = "x" * (config.DOCUMENT_EXCERPT_LIMIT + 500)
        write_raw(
            "excerpt",
            {"expires_at": build_session_expiry(), "document_content": body},
            runtime_dirs["context"],
        )
        payload = store.load_session_payload("excerpt")
        assert len(payload["document_excerpt"]) == config.DOCUMENT_EXCERPT_LIMIT
        assert payload["document_excerpt"] == body[: config.DOCUMENT_EXCERPT_LIMIT]

    def test_a_short_document_is_stored_verbatim_as_its_excerpt(self, runtime_dirs):
        write_raw(
            "short",
            {"expires_at": build_session_expiry(), "document_content": "x" * 500},
            runtime_dirs["context"],
        )
        assert store.load_session_payload("short")["document_excerpt"] == "x" * 500

    def test_prefers_an_explicit_excerpt(self, runtime_dirs):
        write_raw(
            "explicit",
            {
                "expires_at": build_session_expiry(),
                "document_content": "full",
                "document_excerpt": "cached",
            },
            runtime_dirs["context"],
        )
        assert store.load_session_payload("explicit")["document_excerpt"] == "cached"

    def test_backfills_identity_and_timestamp_fields(self, runtime_dirs):
        write_raw("bare", {"expires_at": build_session_expiry()}, runtime_dirs["context"])
        payload = store.load_session_payload("bare")
        assert payload["session_id"] == "bare"
        assert payload["source_filename"] == ""
        for key in ("generated_at", "created_at", "updated_at", "expires_at"):
            assert payload[key]

    def test_truncates_an_over_long_last_query(self, runtime_dirs):
        write_raw(
            "longquery",
            {
                "expires_at": build_session_expiry(),
                "paper_search": {"last_query": "  " + "query " * 80},
            },
            runtime_dirs["context"],
        )
        query = store.load_session_payload("longquery")["paper_search"]["last_query"]
        assert len(query) <= 243
        assert query.endswith("...")

    def test_normalizes_the_search_result_collection(self, runtime_dirs):
        write_raw(
            "results",
            {
                "expires_at": build_session_expiry(),
                "paper_search": {
                    "last_results": [
                        {
                            "title": "Valid Result",
                            "abstract": "A" * 1400,
                            "authors": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
                        },
                        {"title": "Valid Result", "url": "https://example.com/dup"},
                        {"abstract": "missing title"},
                        "not a dict",
                    ]
                },
            },
            runtime_dirs["context"],
        )

        results = store.load_session_payload("results")["paper_search"]["last_results"]

        assert len(results) == 1
        assert results[0]["title"] == "Valid Result"
        assert len(results[0]["abstract"]) == 1203
        assert results[0]["abstract"].endswith("...")
        assert len(results[0]["authors"]) == 8

    def test_normalizes_the_recommendation_block(self, runtime_dirs):
        write_raw(
            "rec",
            {
                "expires_at": build_session_expiry(),
                "paper_search": {
                    "last_recommendation": {
                        "original_query": "  " + "original " * 80,
                        "query": "  " + "recommended " * 80,
                        "reason": "  " + "reason " * 120,
                        "rewrite_model": "  " + "model " * 40,
                        "generated_at": "  " + "date " * 20,
                        "items": [
                            {"title": "Recommended", "year": "2024", "venue": "Venue"},
                            {"title": "Recommended", "year": "2023"},
                            {"url": "https://example.com/no-title"},
                        ],
                        "topics": ["topic", "topic", "  ", 42],
                        "errors": ["error", "error", ""],
                    }
                },
            },
            runtime_dirs["context"],
        )

        recommendation = store.load_session_payload("rec")["paper_search"]["last_recommendation"]

        assert len(recommendation["original_query"]) <= 243
        assert len(recommendation["query"]) <= 243
        assert len(recommendation["reason"]) <= 503
        assert len(recommendation["rewrite_model"]) <= 123
        assert len(recommendation["generated_at"]) <= 43
        assert len(recommendation["items"]) == 1
        assert recommendation["items"][0]["title"] == "Recommended"
        assert recommendation["topics"] == ["topic", "42"]
        assert recommendation["errors"] == ["error"]

    def test_rejects_a_non_dict_recommendation(self, runtime_dirs):
        write_raw(
            "badrec",
            {"expires_at": build_session_expiry(), "paper_search": {"last_recommendation": []}},
            runtime_dirs["context"],
        )
        assert store.load_session_payload("badrec")["paper_search"]["last_recommendation"] == {}

    def test_normalizes_the_reading_queue(self, runtime_dirs):
        write_raw(
            "queue",
            {
                "expires_at": build_session_expiry(),
                "paper_search": {
                    "reading_queue": [
                        {
                            "title": "  Paper One  ",
                            "authors": [{"name": "Alice"}],
                            "year": "2024",
                            "url": "https://example.com/1",
                        },
                        {"title": "Paper One", "authors": ["Duplicate"]},
                        {"title": ""},
                    ]
                },
            },
            runtime_dirs["context"],
        )

        queue = store.load_session_payload("queue")["paper_search"]["reading_queue"]

        assert len(queue) == 1
        entry = queue[0]
        assert entry["title"] == "Paper One"
        assert entry["authors"] == ["Alice"]
        assert entry["year"] == "2024"
        assert entry["url"] == "https://example.com/1"
        assert entry["saved_at"]
        assert set(entry) == {
            "source",
            "paper_id",
            "title",
            "abstract",
            "authors",
            "year",
            "venue",
            "url",
            "pdf_url",
            "saved_at",
        }

    def test_caps_the_reading_queue_length(self, runtime_dirs, monkeypatch):
        monkeypatch.setattr(config, "READING_QUEUE_LIMIT", 3)
        write_raw(
            "bigqueue",
            {
                "expires_at": build_session_expiry(),
                "paper_search": {
                    "reading_queue": [{"title": f"Paper {i}"} for i in range(10)]
                },
            },
            runtime_dirs["context"],
        )
        assert len(store.load_session_payload("bigqueue")["paper_search"]["reading_queue"]) == 3

    def test_normalizes_the_analysis_metadata(self, runtime_dirs):
        write_raw(
            "meta",
            {
                "expires_at": build_session_expiry(),
                "analysis": {
                    "sections": {},
                    "suggested_questions": ["  问题一？  ", "问题一？", ""],
                    "next_actions": [
                        {"label": "  行动  ", "prompt": "  请解释方法。  "},
                        {"label": "重复", "prompt": "请解释方法。"},
                        {"label": "无效"},
                    ],
                    "analysis_status": "invalid",
                },
            },
            runtime_dirs["context"],
        )

        analysis = store.load_session_payload("meta")["analysis"]

        assert analysis["suggested_questions"] == ["问题一？"]
        assert analysis["next_actions"] == [{"label": "行动", "prompt": "请解释方法。"}]
        assert analysis["analysis_status"] == {}
        assert analysis["sections"] == {}

    def test_repairs_a_non_dict_sections_map(self, runtime_dirs):
        write_raw(
            "secs",
            {"expires_at": build_session_expiry(), "analysis": {"sections": "invalid"}},
            runtime_dirs["context"],
        )
        assert store.load_session_payload("secs")["analysis"]["sections"] == {}


class TestLoadValidatedSession:
    def test_returns_the_sanitized_id_and_payload(self, runtime_dirs, session_factory):
        session_id, token = session_factory(document_content="body")

        resolved_id, payload = store.load_validated_session(session_id, token)

        assert resolved_id == session_id
        assert payload["document_content"] == "body"

    def test_raises_value_error_for_a_missing_session(self, runtime_dirs):
        with pytest.raises(ValueError, match="session_id is required"):
            store.load_validated_session("", "token")

        with pytest.raises(ValueError, match="Session expired or context not found"):
            store.load_validated_session("missing", "token")

    def test_raises_permission_error_for_a_wrong_token(self, runtime_dirs, session_factory):
        session_id, _token = session_factory()

        with pytest.raises(PermissionError, match="Invalid or missing session token"):
            store.load_validated_session(session_id, "wrong-token")

    def test_token_check_can_be_skipped(self, runtime_dirs, session_factory):
        session_id, _token = session_factory()
        resolved_id, _payload = store.load_validated_session(session_id, "anything", require_token=False)
        assert resolved_id == session_id

    def test_a_hostile_session_id_cannot_traverse_out_of_the_context_folder(self, runtime_dirs):
        with pytest.raises(ValueError, match="Session expired or context not found"):
            store.load_validated_session("../../etc", "token", require_token=False)

        # Nothing was created outside the context folder.
        assert not (runtime_dirs["context"].parent.parent / "etc.json").exists()

    def test_a_hostile_session_id_is_reduced_to_a_safe_basename(self):
        assert store.build_session_id("../../etc") == "etc"
        assert "/" not in store.build_session_id("a/b/c")
        assert "\\" not in store.build_session_id("a\\b\\c")


class TestCleanupExpiredSessions:
    def test_removes_an_expired_file(self, runtime_dirs, caplog):
        path = runtime_dirs["context"] / "old.json"
        path.write_text(
            json.dumps({"expires_at": "2000-01-01T00:00:00"}), encoding="utf-8"
        )

        with caplog.at_level("INFO"):
            store.cleanup_expired_sessions(force=True)

        assert not path.exists()

    def test_removes_unparseable_json(self, runtime_dirs, caplog):
        path = runtime_dirs["context"] / "broken.json"
        path.write_text("{not valid json", encoding="utf-8")

        with caplog.at_level("WARNING"):
            store.cleanup_expired_sessions(force=True)

        assert not path.exists()
        assert "Removed unreadable session file" in caplog.text

    def test_removes_a_non_object_payload(self, runtime_dirs, caplog):
        path = runtime_dirs["context"] / "array.json"
        path.write_text("[]", encoding="utf-8")

        with caplog.at_level("WARNING"):
            store.cleanup_expired_sessions(force=True)

        assert not path.exists()
        assert "session payload must be an object" in caplog.text

    def test_keeps_a_live_session(self, runtime_dirs, session_factory):
        session_id, _token = session_factory()
        store.cleanup_expired_sessions(force=True)
        assert (runtime_dirs["context"] / f"{session_id}.json").exists()

    def test_ignores_non_json_files(self, runtime_dirs):
        other = runtime_dirs["context"] / "notes.txt"
        other.write_text("keep me", encoding="utf-8")
        store.cleanup_expired_sessions(force=True)
        assert other.exists()

    def test_logs_a_missing_context_folder(self, runtime_dirs, monkeypatch, caplog):
        missing = runtime_dirs["context"] / "does-not-exist"
        monkeypatch.setattr(config, "CONTEXT_FOLDER", str(missing))

        with caplog.at_level("WARNING"):
            store.cleanup_expired_sessions(force=True)

        assert "Unable to list session folder" in caplog.text

    def test_rate_limits_without_force(self, runtime_dirs, monkeypatch):
        path = runtime_dirs["context"] / "old.json"
        path.write_text(json.dumps({"expires_at": "2000-01-01T00:00:00"}), encoding="utf-8")

        monkeypatch.setattr(store, "_last_cleanup_at", time.time())
        store.cleanup_expired_sessions()

        assert path.exists()
