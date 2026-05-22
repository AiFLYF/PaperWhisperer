import asyncio
import json
from email.message import Message

import pytest
from fastapi.testclient import TestClient

import web_app


class FakeResponse:
    def __init__(self, body, content_type="application/pdf", content_length=None, url="https://example.com/paper.pdf"):
        self.body = body
        self.offset = 0
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.url = url
        self.closed = False

    def geturl(self):
        return self.url

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.body) - self.offset
        start = self.offset
        end = min(len(self.body), start + size)
        self.offset = end
        return self.body[start:end]

    def close(self):
        self.closed = True


@pytest.fixture
def public_example_urls(monkeypatch):
    monkeypatch.setattr(web_app, "is_public_http_url", lambda raw_url: str(raw_url).startswith("https://example.com"))


def patch_remote_response(monkeypatch, response):
    monkeypatch.setattr(web_app.urllib.request, "urlopen", lambda *args, **kwargs: response)


class FakeUploadFile:
    def __init__(self, body):
        self.body = body
        self.offset = 0

    async def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.body) - self.offset
        start = self.offset
        end = min(len(self.body), start + size)
        self.offset = end
        return self.body[start:end]


def test_save_upload_file_accepts_valid_pdf(tmp_path):
    destination = tmp_path / "paper.pdf"
    body = b"%PDF-1.7\nlocal upload"

    total_bytes = asyncio.run(web_app.save_upload_file(FakeUploadFile(body), destination, web_app.MAX_CONTENT_LENGTH))

    assert total_bytes == len(body)
    assert destination.read_bytes() == body


def test_save_upload_file_rejects_disguised_html_and_cleans_up(tmp_path):
    destination = tmp_path / "paper.pdf"

    with pytest.raises(ValueError):
        asyncio.run(web_app.save_upload_file(FakeUploadFile(b"<html></html>"), destination, web_app.MAX_CONTENT_LENGTH))

    assert not destination.exists()


def test_remote_pdf_validation_preserves_initial_chunk(monkeypatch, public_example_urls):
    body = b"%PDF-1.7\n" + (b"x" * 5000)
    patch_remote_response(monkeypatch, FakeResponse(body))

    response, file_name, content_type, initial_chunk = web_app.stream_remote_paper(
        title="Valid PDF",
        pdf_url="https://example.com/paper.pdf",
        url="",
    )

    assert file_name.endswith(".pdf")
    assert content_type == "application/pdf"
    assert initial_chunk == body[:4096]
    assert b"".join(web_app.iter_remote_file_chunks(response, web_app.MAX_CONTENT_LENGTH, initial_chunk)) == body


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"<!doctype html><html></html>", "text/html"),
        (b"<html></html>", "application/octet-stream"),
        (b"not a pdf", "application/pdf"),
        (b"%PDF-1.7\n", "application/json"),
    ],
)
def test_remote_file_validation_rejects_non_documents(monkeypatch, public_example_urls, body, content_type):
    patch_remote_response(monkeypatch, FakeResponse(body, content_type=content_type))

    with pytest.raises(ValueError):
        web_app.stream_remote_paper(title="Bad PDF", pdf_url="https://example.com/paper.pdf", url="")


def test_remote_file_validation_rejects_oversized_content_length(monkeypatch, public_example_urls):
    patch_remote_response(
        monkeypatch,
        FakeResponse(b"%PDF-1.7\n", content_length=web_app.MAX_CONTENT_LENGTH + 1),
    )

    with pytest.raises(ValueError):
        web_app.stream_remote_paper(title="Large PDF", pdf_url="https://example.com/paper.pdf", url="")


def test_download_proxy_serves_validated_pdf(monkeypatch, public_example_urls):
    body = b"%PDF-1.7\nproxied body"
    patch_remote_response(monkeypatch, FakeResponse(body))

    response = TestClient(web_app.app).get(
        "/api/download-paper",
        params={"pdf_url": "https://example.com/paper.pdf", "title": "Proxy PDF"},
    )

    assert response.status_code == 200
    assert response.content == body
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_download_proxy_rejects_html_response(monkeypatch, public_example_urls):
    patch_remote_response(monkeypatch, FakeResponse(b"<html></html>", content_type="text/html"))

    response = TestClient(web_app.app).get(
        "/api/download-paper",
        params={"pdf_url": "https://example.com/paper.pdf", "title": "HTML"},
    )

    assert response.status_code == 400
    assert "error" in response.json()


def test_load_session_payload_removes_corrupted_json(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    session_file = tmp_path / "broken.json"
    session_file.write_text("{not valid json", encoding="utf-8")

    assert web_app.load_session_payload("broken") is None
    assert not session_file.exists()


def test_load_session_payload_normalizes_nested_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    session_file = tmp_path / "session.json"
    session_file.write_text(
        json.dumps(
            {
                "expires_at": web_app.build_session_expiry(),
                "document_content": "full text",
                "qa_history": "invalid",
                "analysis": {"sections": "invalid"},
                "paper_search": {"last_results": "invalid", "last_recommendation": "invalid"},
                "session_auth": "invalid",
            }
        ),
        encoding="utf-8",
    )

    payload = web_app.load_session_payload("session")

    assert payload["qa_history"] == []
    assert payload["analysis"]["sections"] == {}
    assert payload["paper_search"]["last_results"] == []
    assert payload["paper_search"]["last_recommendation"] == {}
    assert payload["session_auth"] == {"token_hash": ""}
    assert payload["document_excerpt"]
