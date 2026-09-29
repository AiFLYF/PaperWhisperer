"""Shared pytest fixtures.

Two rules shape everything here:

1. **No test may touch the real runtime folders.** ``config`` resolves
   ``UPLOAD_FOLDER`` / ``OUTPUT_FOLDER`` / ``CONTEXT_FOLDER`` as plain module
   attributes, so ``monkeypatch.setattr(config, ...)`` redirects every reader
   at once — including the ones living deep inside ``sessions.store``.
2. **No test may read the developer's real credentials.** ``.env`` is loaded at
   import time and this repository ships one, so the API-key variables are
   stripped for the whole session unless a test asks for them back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from paperwhisperer.app import app  # noqa: E402
from paperwhisperer.core import config  # noqa: E402
from paperwhisperer.remote import ssrf  # noqa: E402

from .helpers import FakeResponse, FakeUploadFile  # noqa: E402

#: Environment variables that would otherwise leak the developer's own
#: credentials into a test run and turn an offline test into a live API call.
CREDENTIAL_ENV_VARS = (
    "OPENAI_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Run every test as if no credentials were configured."""
    for name in CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def clean_hostname_cache():
    """The SSRF verdict cache is process-global; never let it bleed across tests."""
    ssrf.clear_hostname_cache()
    yield
    ssrf.clear_hostname_cache()


@pytest.fixture
def client():
    """A ``TestClient`` bound to the real application wiring."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def runtime_dirs(tmp_path, monkeypatch):
    """Point all three runtime folders at ``tmp_path`` and create them."""
    # Maps the folder name used everywhere else onto its config attribute —
    # "uploads" is the odd one out (UPLOAD_FOLDER, not UPLOADS_FOLDER).
    config_attr_by_folder = {
        "uploads": "UPLOAD_FOLDER",
        "output": "OUTPUT_FOLDER",
        "context": "CONTEXT_FOLDER",
    }
    folders = {}
    for folder_name, config_attr in config_attr_by_folder.items():
        path = tmp_path / folder_name
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, config_attr, str(path))
        folders[folder_name] = path
    return folders


@pytest.fixture
def public_example_urls(monkeypatch):
    """Allow only ``https://example.com`` through the SSRF gate."""
    from paperwhisperer.remote import download

    monkeypatch.setattr(
        download,
        "is_public_http_url",
        lambda raw_url: str(raw_url).startswith("https://example.com"),
    )


@pytest.fixture
def patch_remote_response(monkeypatch):
    """Return a callable that swaps ``urllib.request.urlopen`` for a fake."""

    def _patch(response):
        monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response)
        return response

    return _patch


@pytest.fixture
def fake_response():
    """Factory for :class:`helpers.FakeResponse`."""
    return FakeResponse


@pytest.fixture
def fake_upload_file():
    """Factory for :class:`helpers.FakeUploadFile`."""
    return FakeUploadFile


@pytest.fixture
def session_factory(runtime_dirs):
    """Persist a session and hand back ``(session_id, token)``."""
    from paperwhisperer.core.text import build_session_expiry
    from paperwhisperer.sessions.store import hash_session_token, write_session_payload

    counter = {"value": 0}

    def _create(document_content="document text", token=None, **extra):
        counter["value"] += 1
        session_id = f"session{counter['value']}"
        resolved_token = token if token is not None else f"token{counter['value']}"
        payload = {
            "expires_at": build_session_expiry(),
            "document_content": document_content,
            "qa_history": [],
            "paper_search": {},
            "session_auth": {"token_hash": hash_session_token(resolved_token)},
        }
        payload.update(extra)
        write_session_payload(session_id, payload)
        return session_id, resolved_token

    return _create


def read_sse_events(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into ``[(event_name, payload), ...]``.

    Streaming responses are the API's main output format, so tests assert on
    decoded events instead of substrings — that keeps them honest when field
    names change and strict when behaviour does not.
    """
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = None
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if name is None:
            continue
        raw = "\n".join(data_lines)
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"_raw": raw}
        events.append((name, payload))
    return events


@pytest.fixture
def sse_events():
    return read_sse_events


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def no_sleep(monkeypatch):
    """Make retry backoff instant so retry tests stay fast."""
    import time as time_module

    monkeypatch.setattr(time_module, "sleep", lambda _seconds: None)


@pytest.fixture
def api_key_env(monkeypatch):
    """Opt a test back into having a server-level API key.

    Requesting the fixture is enough — it sets the variable on setup. The
    previous version returned a setter *function*, which meant every test that
    merely listed it as a dependency silently ran with no key at all and got
    ``missing_api_key`` instead of the behaviour it was asserting.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return "test-key"
