"""Unit tests for the pure helpers in ``core.text`` and ``core.config``.

Nothing here touches the filesystem beyond a tmp dir, and nothing touches the
network — these are the functions everything else is built on, so they are the
right place to pin down exact behaviour.
"""

from __future__ import annotations

import json
import os

import pytest

from paperwhisperer.core import config
from paperwhisperer.core.text import (
    atomic_write_json,
    build_document_excerpt,
    build_safe_upload_filename,
    build_session_expiry,
    build_session_id,
    build_unique_storage_path,
    clean_extracted_text,
    close_response_safely,
    close_upload_file_safely,
    compact_text,
    compute_retry_after_delay,
    extract_json_object,
    extract_message_text,
    get_retry_delay_seconds,
    is_allowed_file,
    normalize_author_list,
    normalize_next_actions,
    normalize_text_items,
    now_iso,
    parse_iso_datetime,
    parse_retry_after_seconds,
    parse_year,
    remove_file_safely,
    secure_filename,
    trim_text_for_log,
)


class TestCompactText:
    def test_collapses_whitespace_runs(self):
        assert compact_text("  a \n\t b  ") == "a b"

    def test_returns_empty_for_none(self):
        assert compact_text(None) == ""

    def test_truncates_with_ellipsis_beyond_limit(self):
        result = compact_text("x" * 50, limit=10)
        assert result == "x" * 10 + "..."
        assert len(result) == 13

    def test_keeps_text_at_exactly_the_limit(self):
        assert compact_text("x" * 10, limit=10) == "x" * 10


class TestTrimTextForLog:
    def test_marks_truncation_with_newline(self):
        result = trim_text_for_log("y" * 50, limit=10)
        assert result == "y" * 10 + "\n...[truncated]"

    def test_short_text_untouched(self):
        assert trim_text_for_log("  hello  ") == "hello"


class TestParseYear:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Published in 2019 by ACM", "2019"),
            ("2020", "2020"),
            ("1998-12-31", "1998"),
            ("no year here", ""),
            (None, ""),
            ("1200", ""),
        ],
    )
    def test_extracts_four_digit_year(self, value, expected):
        assert parse_year(value) == expected


class TestNormalizeAuthorList:
    def test_accepts_strings_dicts_and_objects(self):
        class Obj:
            name = "Grace"

        assert normalize_author_list(["Ada", {"name": "Alan"}, Obj()]) == [
            "Ada",
            "Alan",
            "Grace",
        ]

    def test_caps_the_list(self):
        assert len(normalize_author_list([f"A{i}" for i in range(20)])) == 8

    def test_drops_empty_entries(self):
        assert normalize_author_list(["", "  ", "Ada"]) == ["Ada"]


class TestNormalizeTextItems:
    def test_dedupes_and_strips(self):
        assert normalize_text_items([" a ", "a", "b", ""]) == ["a", "b"]

    def test_rejects_non_list_input(self):
        assert normalize_text_items("not a list") == []

    def test_respects_max_items(self):
        assert normalize_text_items([f"i{i}" for i in range(20)], max_items=3) == [
            "i0",
            "i1",
            "i2",
        ]


class TestNormalizeNextActions:
    def test_keeps_label_prompt_pairs(self):
        assert normalize_next_actions(
            [{"label": " Go ", "prompt": " Do it "}]
        ) == [{"label": "Go", "prompt": "Do it"}]

    def test_drops_incomplete_and_duplicate_entries(self):
        assert normalize_next_actions(
            [
                {"label": "A", "prompt": "same"},
                {"label": "A", "prompt": "same"},
                {"label": "", "prompt": "orphan"},
                {"label": "no prompt"},
                "not a dict",
            ]
        ) == [{"label": "A", "prompt": "same"}]

    def test_rejects_non_list_input(self):
        assert normalize_next_actions({"label": "x"}) == []


class TestExtractMessageText:
    def test_string_content(self):
        assert extract_message_text("  hi  ") == "hi"

    def test_list_of_dicts_keeps_only_text_parts(self):
        assert extract_message_text(
            [{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]
        ) == "a\nb"

    def test_none_and_other_types(self):
        assert extract_message_text(None) == ""
        assert extract_message_text(42) == "42"


class TestExtractJsonObject:
    def test_strips_code_fences(self):
        assert extract_json_object('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_finds_object_embedded_in_prose(self):
        assert extract_json_object('Here you go: {"a": 1} enjoy') == '{"a": 1}'

    def test_returns_input_when_no_object_present(self):
        assert extract_json_object("plain text") == "plain text"


class TestCleanExtractedText:
    def test_collapses_blank_lines_and_strips_each_line(self):
        assert clean_extracted_text("a  \n\n\n\n  b   c \n") == "a\n\nb c"

    def test_empty_input(self):
        assert clean_extracted_text("") == ""


class TestSecureFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../etc/passwd", "etc_passwd"),
            ("my report.pdf", "my_report.pdf"),
            ("a/b\\c.txt", "a_b_c.txt"),
            ("...hidden...", "hidden"),
        ],
    )
    def test_strips_paths_and_unsafe_characters(self, raw, expected):
        assert secure_filename(raw) == expected

    def test_non_ascii_is_transliterated_away(self):
        # The CJK characters vanish, leaving only the extension, which is then
        # stripped as a leading dot — the caller's fallback handles the rest.
        assert secure_filename("论文.pdf") == "pdf"
        assert secure_filename("报告 final.pdf") == "final.pdf"

    @pytest.mark.parametrize("reserved", ["CON", "con.txt", "LPT1"])
    def test_windows_reserved_names_are_prefixed(self, reserved):
        assert secure_filename(reserved).startswith("_")


class TestIsAllowedFile:
    @pytest.mark.parametrize("name", ["a.txt", "a.PDF", "a.docx", "a.pptx"])
    def test_accepts_supported_extensions(self, name):
        assert is_allowed_file(name) is True

    @pytest.mark.parametrize("name", ["a.exe", "a.html", "a", "a.pdf.exe"])
    def test_rejects_everything_else(self, name):
        assert is_allowed_file(name) is False


class TestBuildSafeUploadFilename:
    def test_preserves_a_clean_name(self):
        assert build_safe_upload_filename("paper.pdf") == "paper.pdf"

    def test_replaces_unsupported_extension(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            build_safe_upload_filename("payload.exe")

    def test_leading_dots_are_not_mistaken_for_an_extension(self):
        # os.path.splitext("....pdf") yields ("....pdf", ""), so the extension
        # check must reject this rather than trust the trailing text.
        with pytest.raises(ValueError, match="Unsupported file type"):
            build_safe_upload_filename("....pdf")

    def test_falls_back_to_a_generated_name_when_nothing_survives(self):
        result = build_safe_upload_filename("论文.pdf")
        assert result.endswith(".pdf")
        assert result.startswith("file_")

    def test_normalizes_spaces_but_keeps_the_extension(self):
        assert build_safe_upload_filename("a b.PDF") == "a_b.PDF"


class TestBuildSessionId:
    def test_sanitizes_user_supplied_id(self):
        assert build_session_id("../evil id") == "evil_id"

    def test_generates_a_uuid_backed_id_when_empty(self):
        assert build_session_id("").startswith("session_")


class TestBuildUniqueStoragePath:
    def test_two_calls_do_not_collide(self):
        first = build_unique_storage_path("uploads", "paper.pdf")
        second = build_unique_storage_path("uploads", "paper.pdf")
        assert first != second
        assert first.startswith(os.path.join("uploads", "paper_"))
        assert first.endswith(".pdf")


class TestBuildDocumentExcerpt:
    def test_truncates_to_limit(self):
        assert build_document_excerpt("abcdef", limit=3) == "abc"

    def test_default_limit_comes_from_config(self):
        assert len(build_document_excerpt("x" * 50000)) == config.DOCUMENT_EXCERPT_LIMIT


class TestRetryBackoff:
    @pytest.mark.parametrize(
        ("attempt", "max_delay", "expected"),
        [
            (0, 6, 2),
            (2, 6, 6),
            (10, 6, 6),
            (2, 8, 6),
            (10, 8, 8),
            (-1, 6, 2),
            ("bad", 6, 2),
            (2, 0, 1),
            (2, "bad", 6),
        ],
    )
    def test_delay_is_capped(self, attempt, max_delay, expected):
        assert get_retry_delay_seconds(attempt, max_delay=max_delay) == expected

    def test_backoff_grows_linearly_then_saturates(self):
        assert [get_retry_delay_seconds(i) for i in range(5)] == [2, 4, 6, 6, 6]


class TestParseRetryAfterSeconds:
    def test_parses_numeric_seconds(self):
        assert parse_retry_after_seconds("30") == 30

    def test_caps_at_fallback(self):
        assert parse_retry_after_seconds("9999", fallback_cap=60) == 60

    def test_rejects_garbage(self):
        assert parse_retry_after_seconds("soon") is None
        assert parse_retry_after_seconds("") is None

    def test_parses_http_date_in_the_past_as_zero(self):
        # A stale Retry-After must not produce a negative sleep.
        assert parse_retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 0


class TestComputeRetryAfterDelay:
    def test_prefers_the_retry_after_header(self):
        class WithHeaders(Exception):
            headers = {"Retry-After": "9"}

        assert compute_retry_after_delay(WithHeaders(), 0, 20) == 9

    def test_falls_back_to_backoff_without_headers(self):
        assert compute_retry_after_delay(ValueError("x"), 0, 20) == 2


class TestSessionExpiry:
    def test_round_trips_through_the_parser(self):
        expiry = build_session_expiry()
        assert parse_iso_datetime(expiry) is not None

    def test_parser_rejects_garbage(self):
        assert parse_iso_datetime("not-a-date") is None
        assert parse_iso_datetime("") is None
        assert parse_iso_datetime(None) is None

    def test_expiry_is_in_the_future(self):
        from datetime import datetime

        assert parse_iso_datetime(build_session_expiry()) > datetime.now()


class TestNowIso:
    def test_is_second_precision(self):
        value = now_iso()
        assert len(value) == 19
        assert value[10] == "T"


class TestAtomicWriteJson:
    def test_writes_utf8_without_escapes(self, tmp_path):
        target = tmp_path / "nested.json"
        atomic_write_json(str(target), {"text": "中文"})
        assert json.loads(target.read_text(encoding="utf-8"))["text"] == "中文"

    def test_leaves_no_temp_file_behind(self, tmp_path):
        target = tmp_path / "out.json"
        atomic_write_json(str(target), {"a": 1})
        assert [p.name for p in tmp_path.iterdir()] == ["out.json"]

    def test_overwrites_existing_content(self, tmp_path):
        target = tmp_path / "out.json"
        atomic_write_json(str(target), {"a": 1})
        atomic_write_json(str(target), {"a": 2})
        assert json.loads(target.read_text(encoding="utf-8"))["a"] == 2


class TestCleanupHelpers:
    def test_remove_file_safely_deletes(self, tmp_path):
        target = tmp_path / "gone.txt"
        target.write_text("x", encoding="utf-8")
        remove_file_safely(str(target), "test file")
        assert not target.exists()

    def test_remove_file_safely_ignores_missing_paths(self, tmp_path):
        remove_file_safely(str(tmp_path / "nope.txt"), "test file")
        remove_file_safely(None, "test file")

    def test_remove_file_safely_logs_and_swallows_errors(self, tmp_path, monkeypatch, caplog):
        target = tmp_path / "stale.tmp"
        target.write_text("x", encoding="utf-8")

        def boom(_path):
            raise OSError("permission denied")

        monkeypatch.setattr(os, "remove", boom)

        with caplog.at_level("ERROR"):
            remove_file_safely(str(target), "test temp file")

        assert "Failed to remove test temp file" in caplog.text
        assert str(target) in caplog.text

    def test_remove_file_safely_swallows_system_exit(self, tmp_path, monkeypatch, caplog):
        """Regression: a delete guard that signals refusal via ``SystemExit``.

        These helpers run inside ``finally`` blocks on the streaming routes, so
        anything escaping them replaces the response and tears the server down.
        A sandboxed/guarded delete refusing with ``SystemExit`` was observed to
        kill a live request, which is what this pins shut.
        """
        target = tmp_path / "guarded.tmp"
        target.write_text("x", encoding="utf-8")

        def refuse(_path):
            raise SystemExit(1)

        monkeypatch.setattr(os, "remove", refuse)

        with caplog.at_level("ERROR"):
            remove_file_safely(str(target), "guarded temp file")  # must not raise

        assert "Failed to remove guarded temp file" in caplog.text

    def test_remove_file_safely_still_allows_keyboard_interrupt(self, tmp_path, monkeypatch):
        """Ctrl+C must keep working, so ``KeyboardInterrupt`` is not swallowed."""
        target = tmp_path / "interrupted.tmp"
        target.write_text("x", encoding="utf-8")

        def interrupt(_path):
            raise KeyboardInterrupt

        monkeypatch.setattr(os, "remove", interrupt)

        with pytest.raises(KeyboardInterrupt):
            remove_file_safely(str(target), "interrupted temp file")

    def test_close_response_safely_swallows_system_exit(self, caplog):
        class Refusing:
            def close(self):
                raise SystemExit(1)

        with caplog.at_level("ERROR"):
            close_response_safely(Refusing(), "refusing response")  # must not raise

        assert "Failed to close refusing response" in caplog.text

    def test_close_upload_file_safely_swallows_system_exit(self, caplog):
        import asyncio

        class Refusing:
            async def close(self):
                raise SystemExit(1)

        with caplog.at_level("ERROR"):
            asyncio.run(close_upload_file_safely(Refusing(), "refusing upload"))  # must not raise

        assert "Failed to close refusing upload" in caplog.text

    def test_a_failing_cleanup_cannot_break_a_finally_block(self, tmp_path, monkeypatch):
        """The real shape of the bug: cleanup runs while a result is propagating.

        If the helper raises, the ``finally`` it sits in replaces the original
        outcome. This mirrors the streaming route, where the response had
        already been produced when cleanup ran.
        """
        target = tmp_path / "fin.tmp"
        target.write_text("x", encoding="utf-8")

        def refuse(_path):
            raise SystemExit(1)

        monkeypatch.setattr(os, "remove", refuse)

        def route_body():
            try:
                raise RuntimeError("original failure")
            finally:
                remove_file_safely(str(target), "fin temp file")

        with pytest.raises(RuntimeError, match="original failure"):
            route_body()

    def test_close_response_safely_closes(self):
        class Response:
            closed = False

            def close(self):
                self.closed = True

        response = Response()
        close_response_safely(response, "test response")
        assert response.closed is True

    def test_close_response_safely_ignores_none(self):
        close_response_safely(None, "test response")

    def test_close_response_safely_logs_close_failure(self, caplog):
        class BrokenResponse:
            def close(self):
                raise OSError("close failed")

        with caplog.at_level("ERROR"):
            close_response_safely(BrokenResponse(), "test response")

        assert "Failed to close test response" in caplog.text

    def test_close_upload_file_safely_awaits_close(self):
        import asyncio

        class Upload:
            closed = False

            async def close(self):
                self.closed = True

        upload = Upload()
        asyncio.run(close_upload_file_safely(upload, "test upload"))
        assert upload.closed is True

    def test_close_upload_file_safely_logs_failure(self, caplog):
        import asyncio

        class BrokenUpload:
            async def close(self):
                raise OSError("close failed")

        with caplog.at_level("ERROR"):
            asyncio.run(close_upload_file_safely(BrokenUpload(), "test upload"))

        assert "Failed to close test upload" in caplog.text


class TestConfigParsers:
    def test_parse_int_env_uses_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("PW_TEST_INT", raising=False)
        assert config.parse_int_env("PW_TEST_INT", default=7) == 7

    def test_parse_int_env_clamps_to_range(self, monkeypatch):
        monkeypatch.setenv("PW_TEST_INT", "9999")
        assert config.parse_int_env("PW_TEST_INT", default=5, max_value=10) == 10

    def test_parse_int_env_falls_back_on_garbage(self, monkeypatch):
        monkeypatch.setenv("PW_TEST_INT", "abc")
        assert config.parse_int_env("PW_TEST_INT", default=3) == 3

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
    def test_parse_bool_env_truthy_values(self, monkeypatch, raw):
        monkeypatch.setenv("PW_TEST_BOOL", raw)
        assert config.parse_bool_env("PW_TEST_BOOL") is True

    @pytest.mark.parametrize("raw", ["0", "false", "no", "off", "maybe"])
    def test_parse_bool_env_falsy_values(self, monkeypatch, raw):
        monkeypatch.setenv("PW_TEST_BOOL", raw)
        assert config.parse_bool_env("PW_TEST_BOOL", default=True) is False

    def test_parse_bool_value_handles_none(self):
        assert config.parse_bool_value(None, default=True) is True

    def test_clamp_int_value_falls_back_on_garbage(self):
        assert config.clamp_int_value("x", default=4, min_value=1, max_value=6) == 4

    def test_clamp_int_value_clamps(self):
        assert config.clamp_int_value(99, default=4, min_value=1, max_value=6) == 6

    def test_resolve_api_key_prefers_explicit_value(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "server-key")
        assert config.resolve_api_key("request-key") == "request-key"

    def test_resolve_api_key_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "server-key")
        assert config.resolve_api_key("") == "server-key"

    def test_resolve_api_key_empty_when_nothing_configured(self):
        assert config.resolve_api_key("") == ""


class TestConfigInvariants:
    def test_runtime_folders_are_relative_names(self):
        assert config.RUNTIME_FOLDERS == ("uploads", "output", "context")

    def test_max_upload_is_sixteen_megabytes(self):
        assert config.MAX_CONTENT_LENGTH == 16 * 1024 * 1024

    def test_every_supported_extension_is_allowed(self):
        assert set(config.SUPPORTED_EXTENSIONS) <= config.ALLOWED_EXTENSIONS

    def test_security_headers_cover_the_expected_surface(self):
        assert set(config.SECURITY_HEADERS) == {
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Referrer-Policy",
            "Permissions-Policy",
        }

    def test_static_paths_point_inside_the_project(self):
        assert config.TEMPLATES_DIR.is_dir()
        assert config.STATIC_DIR.is_dir()
        assert config.LOGO_PATH.is_file()
