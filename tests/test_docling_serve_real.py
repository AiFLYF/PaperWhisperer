"""The docling-serve path and the reparse route, driven against things that are real.

Two features in this repository were built against **mocks and never against a
live service**:

* ``PARSER_DOCLING_MODE=serve`` posts a document to a docling-serve instance.
  ``tests/test_ingestion_docling.py`` exercises the transport with a fake
  ``urlopen``, which proves the multipart body is well-formed and nothing more.
  No docling-serve has ever answered one of these requests.
* ``POST /v1/documents/{id}/reparse`` is reachable and returns plausible
  answers, but nothing has ever driven it end to end on a real PDF through the
  real router, so the parser-name wiring inside it has never been checked
  against the registry the gateway actually builds.

Both gaps closed the same way, and both closed badly. ``tests/test_docling_real.py``
documents the precedent: hand-written fixtures were "consistently wrong in the
same direction, so every assertion passed and a real document produced 96
elements with no locator among them." A mock cannot disagree with the thing it
stands in for unless it was generated from it.

So everything here is opt-in behind ``RUN_DOCLING_SERVE=1`` and drives real
components — a real HTTP server, a real docling, a real router. A clean clone
skips with the reason printed, which is the honest outcome when there is no
docling-serve to talk to.

    RUN_DOCLING_SERVE=1 pytest tests/test_docling_serve_real.py -v

Start a server first::

    docling-serve run --host 127.0.0.1 --port 5001

What these tests are *for* is not "serve mode works" — that a mock already
answers. It is whether serve produces **the same document** as in-process, since
serve mode is offered precisely to deployments that will not install torch, and
a silently poorer document would be invisible to exactly those operators.

The one test that needs no server (:func:`test_the_reparse_table_names_a_parser_the_registry_actually_registers`)
is ungated: it is a pure contract check between two literals in this
repository, it costs nothing, and it was failing when this file was written.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperwhisperer.api import routes_evidence, routes_jobs, routes_v1
from paperwhisperer.app import app
from paperwhisperer.core import config
from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.object_store import FilesystemObjectStore
from paperwhisperer.infra.stores import build_sqlite_bundle
from paperwhisperer.ingestion.adapters import docling_adapter as adapter
from paperwhisperer.ingestion.adapters.docling_adapter import DoclingParser

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The distillation paper: 9 pages, 5 tables, 6 formulas, headings and a running
#: footer on every page, so it exercises every axis the comparison checks.
TEXT_PDF = REPO_ROOT / "eval/corpus/digital_single_col/hinton_distillation.pdf"
#: A synthetic 300 DPI scan with no text layer, so OCR actually runs.
SCANNED_PDF = REPO_ROOT / "eval/corpus/scanned_clear/paper_en_basic.pdf"

#: What the reparse tests upload. A few lines of text, not the 9-page paper,
#: because the legacy adapter parses it in milliseconds and the reparse route
#: refuses the cloud selection *before any parse runs*. A real 2-minute docling
#: parse here buys nothing, and the long parse is what makes the job store's
#: background worker overlap its own drain and trip SQLite's transaction guard —
#: a real concurrency bug in ``infra/jobs.py`` that these tests would then be
#: reporting instead of the refusal they exist to check.
REPARSE_UPLOAD_NAME = "paper.txt"
REPARSE_UPLOAD_BODY = b"the quick brown fox jumps over the lazy dog. " * 40

SERVE_URL = os.getenv("PARSER_DOCLING_SERVE_URL") or "http://127.0.0.1:5001"


def _server_is_up(url: str = SERVE_URL) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=5) as response:
            response.read()
        return True
    except Exception:  # noqa: BLE001 - a probe must not raise
        return False


requires_serve = pytest.mark.skipif(
    os.getenv("RUN_DOCLING_SERVE") != "1",
    reason=(
        "真实 docling-serve 对照默认不运行：需要安装 docling-serve、拉起服务并对同一份文档"
        "跑两遍完整解析（每份约 20s–5min，占用 GPU）。设 RUN_DOCLING_SERVE=1 才运行。"
    ),
)

requires_server = pytest.mark.skipif(
    not _server_is_up(),
    reason=f"本机 {SERVE_URL} 没有可用的 docling-serve；先执行 docling-serve run --port 5001",
)


# --------------------------------------------------------------------------
# The endpoint and the response envelope — what the live server actually says
# --------------------------------------------------------------------------


@requires_serve
@requires_server
def test_the_endpoint_we_post_to_exists_on_a_real_server():
    """The adapter posts to a path; the server is the only authority on it.

    This is the test that would have caught the defect this file was written
    for. The adapter posted to ``/v1/parse/file``, which returns 404 on every
    docling-serve that has ever existed — the route is ``/v1/convert/file``,
    named for what the service does. Because a 404 is a non-retryable 4xx the
    failure arrived as ``docling 解析失败：HTTP Error 404``, a parse error
    blamed on the document, and a mock could never show it because the mock
    answers whatever path it is given.
    """
    with urllib.request.urlopen(f"{SERVE_URL.rstrip('/')}/openapi.json", timeout=20) as response:
        spec = json.loads(response.read().decode("utf-8"))

    posted_path = urllib.parse.urlsplit(adapter._serve_endpoint(SERVE_URL)).path
    assert posted_path in spec["paths"], (
        f"适配器向 {posted_path} 发请求，但真实 docling-serve 没有这条路由。"
        f"该服务的转换端点是 /v1/convert/file。"
    )
    assert "post" in spec["paths"][posted_path]


@requires_serve
@requires_server
def test_a_real_response_yields_a_real_document_and_not_the_envelope():
    """The parse must carry elements, not a dict of rendered strings.

    docling-serve wraps the document: ``document`` holds ``md_content``,
    ``html_content`` and ``json_content`` side by side, and only
    ``json_content`` is the DoclingDocument. Returning ``payload["document"]``
    — which *is* a dict — hands normalization an object with no ``texts`` and
    no ``pages``, and that does not raise. It produces a document with zero
    elements and total page coverage of nothing: a parse that reports success
    on an empty page.
    """
    with urllib.request.urlopen(f"{SERVE_URL.rstrip('/')}/openapi.json", timeout=20) as response:
        spec = json.loads(response.read().decode("utf-8"))
    assert "/v1/convert/file" in spec["paths"], (
        "本机 docling-serve 连转换端点都没有，可能是版本不兼容；serve 对照无法进行。"
    )

    payload = adapter.post_to_serve(SERVE_URL, TEXT_PDF, media_type="application/pdf")
    assert any(key in payload for key in ("texts", "tables", "pages", "body")), (
        "提取到的不是 DoclingDocument——多半是拿到了 json/md/html 信封。"
    )
    assert payload.get("texts") or payload.get("tables"), "DoclingDocument 里没有任何文本或表格。"


def test_the_envelope_is_unwrapped_however_it_arrives():
    """Both encodings of the same response must give the same document.

    ``json_content`` is a dict on some paths and a JSON *string* on others
    (the async route, and older builds), and a response that is the bare
    document with no envelope has to keep working. These three shapes are all
    one server's output expressed three ways; treating any of them as "not a
    document" is a parse failure with no diagnostic.
    """
    document = {"texts": [{"text": "hi"}], "tables": [], "pages": {"1": {}}, "body": {}}

    nested_dict = {"status": "success", "document": {"filename": "p.pdf", "json_content": document}}
    nested_str = {"status": "success", "document": {"filename": "p.pdf", "json_content": json.dumps(document)}}
    bare = document

    for payload in (nested_dict, nested_str, bare):
        assert adapter._extract_serve_document(payload) == document


def test_a_response_with_no_document_in_it_is_an_error_not_an_empty_parse():
    """An unrecognised envelope must raise.

    Returning an empty dict here would be the worst outcome available: the
    caller cannot tell "the server sent something I do not understand" from
    "this document has no content", and both become a successful empty parse.
    """
    with pytest.raises(ValueError):
        adapter._extract_serve_document({"status": "success", "document": {"filename": "p.pdf"}})


# --------------------------------------------------------------------------
# Does serve produce the same document?
# --------------------------------------------------------------------------


@requires_serve
@requires_server
def test_serve_produces_the_same_document_as_inprocess(tmp_path):
    """The comparison the whole feature rests on.

    Field-by-field on matched elements: page number, bbox coordinates, table
    row/column shape, formula LaTeX, heading level. Plus the document-level
    counts, because a parse that lost a page would otherwise average out.
    """
    local = DoclingParser(mode="inprocess").parse(str(TEXT_PDF))
    served = DoclingParser(mode="serve", serve_url=SERVE_URL).parse(str(TEXT_PDF))
    assert local.status == "success" and served.status == "success", (
        f"inprocess={local.status}/{local.error_message} "
        f"serve={served.status}/{served.error_message}"
    )

    local_doc, served_doc = local.document, served.document
    assert local_doc.page_count == served_doc.page_count
    assert local_doc.char_count == served_doc.char_count
    assert local_doc.locator_coverage == served_doc.locator_coverage
    assert len(local_doc.elements) == len(served_doc.elements)

    local_by_key = {
        (e.order, e.role, e.text[:80]): e for e in local_doc.elements
    }
    assert len(local_by_key) == len(served_doc.elements), "元素键不唯一，无法逐元素比对。"

    mismatches = []
    for element in served_doc.elements:
        peer = local_by_key.get((element.order, element.role, element.text[:80]))
        if peer is None:
            mismatches.append(f"serve 侧多出元素 order={element.order} role={element.role}")
            continue
        if peer.locator.page != element.locator.page:
            mismatches.append(f"[{element.order}] 页码 {peer.locator.page} != {element.locator.page}")
        peer_box = peer.locator.bbox
        box = element.locator.bbox
        if (peer_box is None) != (box is None) or (
            peer_box and box and (peer_box.x0, peer_box.y0, peer_box.x1, peer_box.y1)
            != (box.x0, box.y0, box.x1, box.y1)
        ):
            mismatches.append(f"[{element.order}] bbox 不同")
        peer_table = getattr(peer, "table", None)
        table = getattr(element, "table", None)
        if (peer_table is None) != (table is None):
            mismatches.append(f"[{element.order}] 一侧有表格一侧没有")
        elif table is not None and len(peer_table.rows) != len(table.rows):
            mismatches.append(
                f"[{element.order}] 表格行数 "
                f"{len(peer_table.rows)} != {len(table.rows)}"
            )
        if (peer.latex or "") != (element.latex or ""):
            mismatches.append(f"[{element.order}] 公式 LaTeX 不同")

    assert not mismatches, "serve 与 inprocess 产出的文档不一致：\n" + "\n".join(mismatches[:20])


@requires_serve
@requires_server
def test_a_misconfigured_serve_url_is_reported_as_a_configuration_fault():
    """A 404 must not be reported as a bad document.

    ``resolve_serve_url`` refuses a non-loopback host, which is right. But an
    operator can still point the setting at a path that does not exist, and
    that used to surface as ``HTTP Error 404`` inside a parse failure — the
    message blames the file the user uploaded. The service answered, so the
    only thing that can be wrong is the address.
    """
    parser = DoclingParser(mode="serve", serve_url=f"{SERVE_URL}/v1/convert")
    result = parser.parse(str(TEXT_PDF))
    assert result.status == "failed"
    assert result.error_message, "失败却没有给出任何原因。"
    assert "404" in result.error_message
    # Not "the document could not be parsed" — that is the wrong owner.
    assert "解析失败" not in result.error_message or "地址" in result.error_message


# --------------------------------------------------------------------------
# force_ocr — does re-OCR re-OCR?
# --------------------------------------------------------------------------


class _ForceOcr:
    """The only thing the parser reads off an options object."""

    def __init__(self, value: bool) -> None:
        self.force_ocr = value


@requires_serve
def test_force_ocr_changes_nothing_because_docling_already_ocrs():
    """Measure the control rather than trust the label on it.

    The reparse UI offers "re-read the whole document through OCR", which is a
    promise that something happens. On docling 2.131.0 it does not: the
    pipeline defaults to ``do_ocr=True``, so ``force_ocr`` restates the
    default. Both the option and the resulting text are checked here, because
    a no-op option object and a no-op parse are separate claims.
    """
    docling = pytest.importorskip("docling")
    plain, plain_warnings = adapter._build_converter(docling, force_ocr=False)
    forced, forced_warnings = adapter._build_converter(docling, force_ocr=True)

    assert plain_warnings == [] and forced_warnings == []
    assert adapter._ocr_is_active(plain) is True, (
        "force_ocr=False 时 OCR 已经开着——force_ocr 在这个版本上无法产生任何效果。"
    )
    assert adapter._ocr_is_active(forced) is True


@requires_serve
def test_a_scanned_document_parses_identically_with_and_without_force_ocr():
    """The user-visible version of the same finding, on a document with no text layer."""
    parser = DoclingParser(mode="inprocess")
    without = parser.parse(str(SCANNED_PDF), _ForceOcr(False))
    with_ocr = parser.parse(str(SCANNED_PDF), _ForceOcr(True))

    assert without.status == "success" and with_ocr.status == "success"
    assert without.document.to_full_text() == with_ocr.document.to_full_text(), (
        "force_ocr 改变了文本——请更新 docs/DOCLING_SERVE.md 与 PARSER_BAKE_OFF.md 的结论。"
    )
    assert without.document.char_count == with_ocr.document.char_count


# --------------------------------------------------------------------------
# The reparse route, end to end
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_document(tmp_path_factory):
    """The real application, with one real document uploaded and parsed once.

    Module-scoped, and it uploads a single document, for two reasons that only
    appear once the parse is real rather than faked:

    * **A real 9-page parse takes ~2 minutes.** Three tests each uploading their
      own copy is six minutes of GPU for one refusal being checked.
    * **Two overlapping ``TestClient`` lifespans deadlock the job store.** The
      ``app`` object is a module-level singleton; a second ``TestClient`` on it
      starts a second job worker against the *same* SQLite file, and the two
      workers' transactions collide — ``cannot start a transaction within a
      transaction`` — which surfaces as a 500 ``document_upload_failed`` on a
      request that has nothing wrong with it. One lifespan, one worker, one
      document.

    The app object is the real one so router registration is under test too; the
    three storage seams are swapped so nothing reaches a real data directory.
    """
    tmp_path = tmp_path_factory.mktemp("reparse")
    bundle = build_sqlite_bundle(
        SimpleNamespace(
            STORAGE_SQLITE_PATH=str(tmp_path / "v1.sqlite3"),
            STORAGE_SQLITE_TIMEOUT_SECONDS=30,
            STORAGE_LEXICAL_FTS5="auto",
        )
    )
    objects = FilesystemObjectStore(tmp_path / "objects", fsync=False)
    objects.initialize()
    # Built once, not per dependency resolution. A lambda that calls
    # ``build_job_store`` returns a *new* store every time FastAPI resolves the
    # dependency, and the job worker drains it from a different thread — two
    # connections writing the same file, which answers "database is locked".
    job_store = J.build_job_store(bundle=bundle)
    overrides = {
        routes_evidence.get_store_bundle: lambda: bundle,
        routes_v1.get_object_store: lambda: objects,
        routes_jobs.get_job_store: lambda: job_store,
    }
    app.dependency_overrides.update(overrides)
    backend, cache = config.PARSER_BACKEND, config.PARSER_CACHE_ENABLED
    config.PARSER_BACKEND = "gateway"
    config.PARSER_CACHE_ENABLED = False
    try:
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            upload = client.post(
                "/v1/documents",
                files={
                    "file": (
                        REPARSE_UPLOAD_NAME,
                        io.BytesIO(REPARSE_UPLOAD_BODY),
                        "text/plain",
                    )
                },
            )
            assert upload.status_code == 201, (
                f"前置上传失败：{upload.status_code} {upload.text[:300]}"
            )
            document_id = upload.json()["document"]["document_id"]
            yield client, document_id
    finally:
        config.PARSER_BACKEND, config.PARSER_CACHE_ENABLED = backend, cache
        for dependency in overrides:
            app.dependency_overrides.pop(dependency, None)
        objects.close()
        bundle.close()


def test_the_reparse_table_names_a_parser_the_registry_actually_registers():
    """The menu and the registry must agree on a parser's name.

    ``REPARSE_PARSERS`` is a hand-written table and the registry is built at
    runtime from whatever adapters are installed. Nothing checked that the two
    used the same spelling, and they did not: the table asked the catalog for
    ``"mineru"`` while the adapter registers ``"mineru_cloud"``. The lookup
    therefore missed, and the cloud reparse was refused with 409
    ``parser_unavailable`` — in *every* configuration, including one with the
    privacy mode opened, consent granted and a valid token. A cloud reparse
    could not run at all; it could only be refused.

    This test needs no server and no parse. It is the cheapest possible guard
    against the same drift, and it is ungated because it is free.

    An entry whose adapter is not installed on this machine is exempt, and the
    distinction is the one ``build_parser_catalog``'s own docstring draws: a
    parser this deployment cannot run is *absent* from the catalog rather than
    present and failing, and a menu entry for one is answered by the route with
    409 ``parser_unavailable``. That is product behaviour, not drift — so
    comparing it against the catalog would fail on every checkout that has not
    installed docling's extra (torch chain, hundreds of MB), which is most of
    them, and the guard would be read as a broken build rather than as the
    exemption it is. Everything else still has to be registered, which is what
    caught the cloud reparse whose name nothing produced.
    """
    monkey = dict(config.__dict__)
    monkey["DOCUMENT_PRIVACY_MODE"] = "allow_cloud"
    monkey["DOCUMENT_CLOUD_CONSENT_GRANTED"] = True
    monkey["MINERU_API_TOKEN"] = "not-a-real-token"
    monkey["PARSER_BACKEND"] = "gateway"
    original = {key: getattr(config, key) for key in monkey if hasattr(config, key)}
    try:
        for key, value in monkey.items():
            setattr(config, key, value)
        catalog = routes_v1.build_parser_catalog()
    finally:
        for key, value in original.items():
            setattr(config, key, value)

    docling_installed = importlib.util.find_spec("docling") is not None
    for selection, entry in routes_v1.REPARSE_PARSERS.items():
        if not entry.parser_name:
            continue
        if not docling_installed and entry.parser_name.startswith("docling"):
            continue
        assert entry.parser_name in catalog, (
            f"reparse 菜单里的 {selection!r} 要求解析器 {entry.parser_name!r}，"
            f"但注册表里只有 {sorted(catalog)}。该选项永远无法运行。"
        )


@requires_serve
def test_a_cloud_reparse_under_local_only_is_refused_before_anything_is_read(
    real_document, monkeypatch
):
    """``local_only`` must answer with the privacy code, not an install hint.

    The route resolves the parser against the catalog before consulting the
    privacy gate. Under ``local_only`` the cloud adapter is never registered —
    that is the whole point of not carrying a parser that can only fail — so
    the resolution missed and the user was told to "install the optional
    dependency or configure its credentials". That advice is wrong: an
    administrator forbade egress, and installing the dependency would not make
    the request legal. The gate is now consulted first, so the answer is the
    403 that names the setting to change.
    """
    real_client, document_id = real_document
    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "local_only")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)

    response = real_client.post(
        f"/v1/documents/{document_id}/reparse", json={"parser": "mineru_cloud"}
    )
    assert response.status_code == 403
    assert response.json()["code"] == "privacy_mode_local_only"
    assert "DOCUMENT_PRIVACY_MODE" in response.json()["error"], (
        "403 必须指出要改哪个设置，否则用户只能去装依赖。"
    )


@requires_serve
def test_the_two_cloud_refusals_stay_distinguishable(real_document, monkeypatch):
    """An administrator's refusal and a user's refusal are different problems.

    They send the user to different places — one to the operator, one to a
    settings screen — so they must not collapse into one message. This is only
    checkable with the real adapter importable, because ``local_only`` keeps it
    out of the registry entirely.
    """
    real_client, document_id = real_document
    monkeypatch.setattr(config, "MINERU_API_TOKEN", "not-a-real-token")

    monkeypatch.setattr(config, "DOCUMENT_PRIVACY_MODE", "allow_cloud")
    monkeypatch.setattr(config, "DOCUMENT_CLOUD_CONSENT_GRANTED", False)
    response = real_client.post(
        f"/v1/documents/{document_id}/reparse", json={"parser": "mineru_cloud"}
    )
    assert response.status_code == 403
    assert response.json()["code"] == "consent_not_granted"


@requires_serve
def test_an_unlisted_parser_is_a_client_bug_not_a_policy_refusal(real_document):
    """The three refusals must not blur together.

    An unlisted selection is a 400 about spelling; a cloud selection refused by
    policy is a 403 about configuration. Answering the first with the second
    sends a developer to change an administrator's privacy setting.

    Gated even though the server is not involved, because getting to a document
    id means uploading a real PDF and parsing it for real (~2min). The refusal
    under test is cheap; the setup is not, and a default suite should not pay
    for it.
    """
    real_client, document_id = real_document
    response = real_client.post(
        f"/v1/documents/{document_id}/reparse", json={"parser": "not_a_parser"}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "unknown_parser"