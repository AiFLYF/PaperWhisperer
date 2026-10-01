"""The configured ingestion limits are the limits that actually run.

``MAX_UPLOAD_BYTES`` only means something if the request path reads it, so
every test here drives a real request with the setting monkeypatched and
asserts on the response. Asserting the constant against itself would pass
just as happily against the bug these tests exist to catch: an ingress path
that reads the legacy ``MAX_CONTENT_LENGTH`` alias, or a frozen import-time
copy, keeps enforcing 16 MB no matter what the operator configured.

Each test moves the ceiling *both* ways around one fixed payload. Shrinking
it and watching the same bytes be refused is what proves the setting is
consulted; raising it and watching the same bytes be accepted is what proves
the path did not simply start refusing everything, and that the ceiling can
be lifted at all.
"""

from __future__ import annotations

import io

import pytest

from paperwhisperer.core import config

#: A cap of exactly 1 MB, so the rejection message renders as a real number
#: ("Limit: 1 MB") rather than the 0 MB that any sub-megabyte cap would show.
SMALL_LIMIT = 1024 * 1024
#: A cap comfortably above ``OVER_LIMIT_BODY``'s size.
LARGE_LIMIT = 8 * 1024 * 1024
#: Two megabytes of plain text: over ``SMALL_LIMIT``, under ``LARGE_LIMIT``,
#: and a legal ``.txt`` payload (no NUL bytes) so the only thing that can
#: turn this request down is the byte ceiling.
OVER_LIMIT_BODY = b"a" * (2 * 1024 * 1024)

#: A caller-supplied API key that a real browser would send. Since T2.3 the
#: analysis runs inside a job, and a caller-supplied key only reaches that job's
#: stage when it is at least eight characters
#: (:func:`paperwhisperer.api.routes_jobs.register_ephemeral_secret` — a shorter
#: "key" is a browser default, not a credential). A one-character placeholder is
#: therefore not a key the stage accepts, so the placeholder is realistic here;
#: every assertion is unchanged.
BROWSER_KEY = "sk-test-browser-key"


class RecordingAnalyzer:
    """Stand-in for ``DocumentAnalyzer`` that records that it was reached.

    The oversized cases must never construct one: proving that is what shows
    the ceiling is enforced *before* the document is handed to the pipeline
    rather than being discovered half-way through an analysis.
    """

    instances: list[RecordingAnalyzer] = []

    def __init__(self, api_key):
        self.api_key = api_key
        self.document_content = "document text"
        self.version = "test-version"
        RecordingAnalyzer.instances.append(self)

    @classmethod
    def reset(cls):
        cls.instances = []

    def analyze(self, file_path, mermaid=True, evaluation=True, research_brief=True):
        return {
            "summary": "summary 内容",
            "quotes": "quotes 内容",
            "mindmap": "mindmap 内容",
            "char_count": 10,
        }

    def analyze_stream(self, file_path, mermaid=True, evaluation=True, research_brief=True):
        yield {"type": "section", "name": "summary", "section": {"content": "summary 内容"}}
        yield {
            "type": "done",
            "result": {
                "summary": "summary 内容",
                "quotes": "quotes 内容",
                "mindmap": "mindmap 内容",
                "char_count": 10,
            },
        }


@pytest.fixture
def analyzer(monkeypatch, isolated_job_stores):
    """Replace ``DocumentAnalyzer`` in every module that binds the name.

    Since T2.3 ``/api/analyze`` does not build the analyzer in the route — it
    submits a job, and the job's analysing stage
    (:func:`paperwhisperer.api.routes_jobs._analyse_stage`) imports
    ``DocumentAnalyzer`` **at call time** from
    :mod:`paperwhisperer.analysis.orchestrator`. That is the seam that is
    actually executed, so it is the one that must be patched; patching only the
    route / service modules would let the real analyzer — and a real HTTP call to
    the model endpoint — through. ``/api/import-paper`` still calls
    ``analysis.service.analyze_saved_file`` (which binds the name itself), so
    that module is patched too.

    ``tests/conftest.py::no_unstubbed_job_stage_model`` is the suite-wide
    backstop for the same seam; this per-test stub is applied after it and so
    wins for the duration of the test.
    """
    RecordingAnalyzer.reset()
    for module in (
        "paperwhisperer.analysis.orchestrator",
        "paperwhisperer.analysis.service",
        "paperwhisperer.api.routes_documents",
    ):
        monkeypatch.setattr(f"{module}.DocumentAnalyzer", RecordingAnalyzer)
    yield RecordingAnalyzer
    RecordingAnalyzer.reset()


@pytest.fixture
def upload_client(client, analyzer, runtime_dirs):
    """A ``TestClient`` with a stubbed analyzer and isolated storage."""
    return client


def upload(body: bytes, name: str = "paper.txt"):
    """Build the ``files=`` mapping for a multipart upload."""
    return {"file": (name, io.BytesIO(body), "text/plain")}


class TestUploadRouteHonoursMaxUploadBytes:
    def test_a_payload_over_the_configured_ceiling_is_refused(
        self, upload_client, monkeypatch
    ):
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", SMALL_LIMIT)

        response = upload_client.post(
            "/api/analyze", files=upload(OVER_LIMIT_BODY), data={"api_key": BROWSER_KEY}
        )

        assert response.status_code == 400
        assert "too large" in response.json()["error"]
        # The message quotes the ceiling that is in force, not the default:
        # if this read the shipped 16 MB the body would have been accepted, so
        # the substring is really a second, independent check that the
        # configured value reached the error text.
        assert "1 MB" in response.json()["error"]
        assert RecordingAnalyzer.instances == []

    def test_the_legacy_alias_alone_no_longer_moves_the_ceiling(
        self, upload_client, monkeypatch
    ):
        # ``MAX_CONTENT_LENGTH`` is kept only so old importers keep working. If
        # a route read it, shrinking *this* would refuse the body below.
        monkeypatch.setattr(config, "MAX_CONTENT_LENGTH", SMALL_LIMIT)

        response = upload_client.post(
            "/api/analyze", files=upload(OVER_LIMIT_BODY), data={"api_key": BROWSER_KEY}
        )

        assert response.status_code == 200

    def test_the_same_payload_is_accepted_once_the_ceiling_is_raised(
        self, upload_client, monkeypatch
    ):
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", LARGE_LIMIT)

        response = upload_client.post(
            "/api/analyze", files=upload(OVER_LIMIT_BODY), data={"api_key": BROWSER_KEY}
        )

        assert response.status_code == 200
        assert response.json()["summary"] == "summary 内容"
        assert len(RecordingAnalyzer.instances) == 1

    def test_nothing_is_left_on_disk_after_a_refusal(self, upload_client, runtime_dirs,
                                                     monkeypatch):
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", SMALL_LIMIT)

        upload_client.post(
            "/api/analyze", files=upload(OVER_LIMIT_BODY), data={"api_key": BROWSER_KEY}
        )

        assert list(runtime_dirs["uploads"].iterdir()) == []


class TestStreamingUploadRouteHonoursMaxUploadBytes:
    def test_the_stream_route_refuses_the_same_payload(
        self, upload_client, runtime_dirs, monkeypatch
    ):
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", SMALL_LIMIT)

        response = upload_client.post(
            "/api/analyze/stream", files=upload(OVER_LIMIT_BODY), data={"api_key": BROWSER_KEY}
        )

        assert response.status_code == 400
        assert "too large" in response.json()["error"]
        assert RecordingAnalyzer.instances == []
        assert list(runtime_dirs["uploads"].iterdir()) == []

    def test_the_stream_route_accepts_it_once_the_ceiling_is_raised(
        self, upload_client, monkeypatch
    ):
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", LARGE_LIMIT)

        response = upload_client.post(
            "/api/analyze/stream", files=upload(OVER_LIMIT_BODY), data={"api_key": BROWSER_KEY}
        )

        assert response.status_code == 200
        assert '"summary": "summary 内容"' in response.text


class TestRemoteDownloadHonoursMaxUploadBytes:
    def test_a_declared_length_over_the_ceiling_is_refused(
        self, client, runtime_dirs, public_example_urls, patch_remote_response, fake_response,
        monkeypatch
    ):
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", SMALL_LIMIT)
        patch_remote_response(
            fake_response(
                b"%PDF-1.7\n" + b"x" * 2048,
                content_length=2 * 1024 * 1024,
            )
        )

        response = client.get(
            "/api/download-paper", params={"pdf_url": "https://example.com/paper.pdf"}
        )

        assert response.status_code == 400
        assert "too large" in response.json()["error"]
        assert "1 MB" in response.json()["error"]

    def test_import_refuses_a_body_that_only_the_chunk_loop_catches(
        self, client, runtime_dirs, public_example_urls, patch_remote_response, fake_response,
        monkeypatch
    ):
        # No Content-Length header, so the declared-length check is blind and
        # the streamed ceiling is the only thing that can catch this body.
        monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", SMALL_LIMIT)
        patch_remote_response(
            fake_response(b"%PDF-1.7\n" + b"x" * (2 * 1024 * 1024), content_length=None)
        )

        response = client.post(
            "/api/import-paper",
            json={"title": "Big paper", "pdf_url": "https://example.com/paper.pdf"},
        )

        assert response.status_code == 400
        assert "too large" in response.json()["error"]
        assert RecordingAnalyzer.instances == []
        assert list(runtime_dirs["uploads"].iterdir()) == []


class TestShippedDefault:
    def test_the_default_ceiling_is_unchanged(self):
        # 16 MB is the behaviour every existing test and deployment relies on.
        # ``MAX_CONTENT_LENGTH`` is an alias of the same value, not a second
        # literal; if that ever stops being true, the alias is a second policy
        # again and this assertion is the thing that notices.
        assert config.MAX_UPLOAD_BYTES == 16 * 1024 * 1024
        assert config.MAX_CONTENT_LENGTH == config.MAX_UPLOAD_BYTES
