#!/usr/bin/env python3
"""Parse one document twice — in-process and via docling-serve — and diff the two.

``docling-serve`` mode exists for deployments that will not install the torch
chain. That is a real reason to want it, and it makes the question this script
answers the interesting one: **not** "does serve work", which a mock answers,
but **"does serve produce the same document"**. An operator who switches to
serve to save disk is trading away nothing they can see, and if the two modes
disagree then that trade is invisible — a slightly poorer parse that still
reports success, full page coverage and a locator on every element.

So the diff below is per element, not per document: page coverage, bbox
coordinates, table row/column structure, formula LaTeX and element role are
compared field by field, and a disagreement prints as a ``DIFF`` line naming
what differed. Equal summary numbers are reported as equal too, because
"identical" is the finding and it should be legible without reading a table of
deltas.

Runs only when a real docling-serve is reachable:

    RUN_DOCLING_SERVE=1 python3 tools/docling_serve_smoke.py
    RUN_DOCLING_SERVE=1 python3 tools/docling_serve_smoke.py \
        --file eval/corpus/table_dense/glue_benchmark.pdf

Both modes execute in this process (the in-process side here, and the serve side
over HTTP), so the comparison is between two runs of the *same* docling build
where the versions match — the point being that with identical docling, any
difference is the transport, not the parser. When the server's docling differs,
the version is printed and the caveat belongs with the result.

Exit codes: ``0`` identical, ``1`` the two modes disagree, ``2`` SKIP (no
``RUN_DOCLING_SERVE``, or the configured server is not reachable).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FILE = PROJECT_ROOT / "eval/corpus/digital_single_col/hinton_distillation.pdf"
DEFAULT_URL = "http://127.0.0.1:5001"


def _header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


class Comparison:
    """Accumulates disagreements between the two parses."""

    def __init__(self) -> None:
        self.defects: list[str] = []
        self.same = 0

    def check(self, field: str, local: Any, served: Any) -> None:
        """Record one field comparison, counting equality as well as difference."""
        if local == served:
            self.same += 1
            return
        self.defects.append(f"{field}: inprocess={local!r} serve={served!r}")

    def verdict(self) -> int:
        if not self.defects:
            print(f"\nIDENTICAL — {self.same} compared fields agreed.")
            return 0
        print(f"\nDIFFERENT — {len(self.defects)} of {self.same + len(self.defects)} "
              f"compared fields disagreed:")
        for line in self.defects[:40]:
            print(f"  DIFF {line}")
        if len(self.defects) > 40:
            print(f"  ... and {len(self.defects) - 40} more")
        return 1


def _bbox_tuple(bbox: Any) -> tuple | None:
    if bbox is None:
        return None
    return (bbox.x0, bbox.y0, bbox.x1, bbox.y1)


def _table_shape(element: Any) -> tuple | None:
    """The 2D structure of a table, as (rows, cols, header rows, first-row cells)."""
    table = getattr(element, "table", None)
    if table is None:
        return None
    rows = getattr(table, "rows", None)
    grid = getattr(table, "grid", None)
    shape = (len(rows) if rows else 0, len(grid) if grid else 0)
    header = getattr(table, "header", None)
    return (shape[0], shape[1], len(header) if header else 0)


def compare(local_doc: Any, served_doc: Any) -> Comparison:
    """Field-by-field diff of two ``ParsedDocument`` objects."""
    result = Comparison()

    for field in ("page_count", "char_count", "media_type"):
        result.check(f"document.{field}", getattr(local_doc, field), getattr(served_doc, field))

    result.check(
        "document.locator_coverage",
        round(local_doc.locator_coverage, 6),
        round(served_doc.locator_coverage, 6),
    )
    result.check(
        "document.element_count", len(local_doc.elements), len(served_doc.elements)
    )

    local_by_text = {}
    for element in local_doc.elements:
        key = (element.order, element.role, element.text[:80])
        local_by_text.setdefault(key, element)

    served_keys = {
        (element.order, element.role, element.text[:80]) for element in served_doc.elements
    }
    missing = sorted(set(local_by_text) - served_keys)[:10]
    extra = sorted(served_keys - set(local_by_text))[:10]
    for key in missing:
        result.defects.append(f"element absent from serve parse: order={key[0]} role={key[1]}")
    for key in extra:
        result.defects.append(f"element absent from inprocess parse: order={key[0]} role={key[1]}")

    # Compare matched elements on the fields that carry provenance and structure.
    for element in served_doc.elements:
        key = (element.order, element.role, element.text[:80])
        peer = local_by_text.get(key)
        if peer is None:
            continue
        result.check(f"[{key[0]}] role", peer.role, element.role)
        result.check(f"[{key[0]}] page", peer.locator.page, element.locator.page)
        result.check(
            f"[{key[0]}] bbox", _bbox_tuple(peer.locator.bbox),
            _bbox_tuple(element.locator.bbox),
        )
        result.check(f"[{key[0]}] table_shape", _table_shape(peer), _table_shape(element))
        result.check(f"[{key[0]}] latex", (peer.latex or "")[:200], (element.latex or "")[:200])
        result.check(f"[{key[0]}] level", peer.level, element.level)
    return result


def _server_versions(base_url: str, timeout: float) -> tuple[str, str]:
    """``(docling-serve version, docling version)`` from the live server.

    Both matter. The docling-serve version says how much of the transport to
    trust; the **docling** version says whether this comparison means anything
    at all, because if the two sides differ then any difference found is the
    parser's, not the transport's. The keys are hyphenated on the wire
    (``docling-serve``), which is worth knowing the first time ``/version``
    answers ``?`` for a perfectly healthy server.
    """
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/version", timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        reported = json.loads(raw)
        return (
            str(reported.get("docling-serve") or "?"),
            str(reported.get("docling") or "?"),
        )
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not raise
        return (f"unavailable ({exc})", "?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=str(DEFAULT_FILE), help="PDF to parse both ways")
    parser.add_argument("--url", default=os.getenv("PARSER_DOCLING_SERVE_URL") or DEFAULT_URL)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--force-ocr", action="store_true", help="parse both ways with force_ocr on"
    )
    args = parser.parse_args()

    if os.getenv("RUN_DOCLING_SERVE") != "1":
        print("SKIP: set RUN_DOCLING_SERVE=1 to run the real serve-vs-inprocess comparison.")
        return 2

    document = Path(args.file)
    if not document.is_file():
        print(f"SKIP: no document at {document}")
        return 2

    _header(f"docling-serve smoke — {document.name}")
    print(f"document : {document}  ({document.stat().st_size} bytes)")
    print(f"serve url: {args.url}")

    # Refuse before spending two parses on a server that is not there.
    try:
        with urllib.request.urlopen(f"{args.url.rstrip('/')}/health", timeout=15) as resp:
            resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"\nSKIP: no docling-serve at {args.url} — {exc}")
        print("Start one with: docling-serve run --host 127.0.0.1 --port 5001")
        return 2

    from paperwhisperer.ingestion.adapters.docling_adapter import DoclingParser

    serve_version, serve_docling = _server_versions(args.url, 15.0)
    local_docling = DoclingParser(mode="inprocess").version
    print(f"docling-serve version        : {serve_version}  (server-side)")
    print(f"docling version              : {local_docling} (local) / {serve_docling} (server)")
    if serve_docling != local_docling:
        print(
            f"\nWARNING: 两侧 docling 版本不同（{local_docling} vs {serve_docling}）。"
            "下面的差异属于解析器版本差异，不是传输方式的差异——本对照只在版本相同时成立。"
        )
    print(f"force_ocr                    : {args.force_ocr}")

    _header("in-process parse")
    t0 = time.time()
    local = DoclingParser(mode="inprocess").parse(str(document))
    print(f"status={local.status} ms={local.duration_ms} wall={round(time.time() - t0, 1)}s")
    if local.error_message:
        print(f"error={local.error_message}")

    _header("docling-serve parse")
    t0 = time.time()
    served = DoclingParser(mode="serve", serve_url=args.url).parse(
        str(document), _Options(args.force_ocr)
    )
    print(f"status={served.status} ms={served.duration_ms} wall={round(time.time() - t0, 1)}s")
    if served.error_message:
        print(f"error={served.error_message}")

    if local.status != "success" or served.status != "success":
        print("\nINCONCLUSIVE: at least one parse did not succeed; nothing to diff.")
        return 2

    _header("comparison")
    return compare(local.document, served.document).verdict()


class _Options:
    """Minimal stand-in for a parse options object (only ``force_ocr`` is read)."""

    def __init__(self, force_ocr: bool) -> None:
        self.force_ocr = force_ocr


if __name__ == "__main__":
    sys.exit(main())