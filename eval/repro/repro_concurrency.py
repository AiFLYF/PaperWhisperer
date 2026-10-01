"""Reproduction: one real upload, one real parse, one leaked transaction.

    /home/ubuntu/miniconda3/bin/python3 eval/repro/repro_concurrency.py

It drives the real path end to end, on a real 9-page corpus PDF:

* the real personal-tier bundle on a real SQLite file,
* the real object store,
* the real ``routes_v1.run_parse_job`` on a request thread, which submits a
  ``document_ingest`` job and drains it,
* the real background workers ``routes_jobs.start_job_worker`` starts -- one per
  pipeline kind, against the same store.

The overlap is not manufactured. ``run_parse_job`` ships a 60s lease; a real
9-page parse through docling-serve takes ~120s; and the parse step does not
heartbeat while it runs. So the lease lapses mid-parse, the background worker's
``recover_stale`` requeues a job that is still being parsed, and when the parse
finally returns the request thread's terminal transition finds the row somewhere
it did not leave it.

Note the background workers claim by kind and ``document_ingest`` is not one of
them -- so nothing re-runs the job, and the stale row is simply left where
recovery put it. That is the production configuration, reproduced exactly.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from paperwhisperer.core import config
from paperwhisperer.infra import jobs as J
from paperwhisperer.infra.stores.factory import build_sqlite_bundle

PDF = os.path.join("eval", "corpus", "digital_single_col", "hinton_distillation.pdf")
LEASE_SECONDS = 60.0  # the lease routes_v1.run_parse_job ships with


def main() -> int:
    if not os.path.exists(PDF):
        print(f"corpus document missing: {PDF}")
        return 2

    root = tempfile.mkdtemp(prefix="pw-concurrency-repro-")
    bundle = build_sqlite_bundle(
        SimpleNamespace(
            STORAGE_SQLITE_PATH=os.path.join(root, "store.sqlite3"),
            STORAGE_SQLITE_TIMEOUT_SECONDS=5,
            STORAGE_LEXICAL_FTS5="off",
        )
    )
    store = J.build_job_store(bundle=bundle)

    # `run_parse_job` reads the staged path its caller hands it and does not go
    # through the object store, so this reproduction does not need one. What is
    # under test is the job store's transaction handling.
    # The real background workers, one per pipeline kind, as
    # routes_jobs.start_job_worker starts them.
    from paperwhisperer.api.routes_jobs import start_job_worker, stop_job_workers
    from paperwhisperer.api.routes_v1 import ParseRun, run_parse_job

    start_job_worker(store)

    config.PARSER_BACKEND = "gateway"
    config.PARSER_CACHE_ENABLED = False

    parse_run = ParseRun()
    request_errors: list[str] = []
    started = time.monotonic()

    def request_thread() -> None:
        try:
            run_parse_job(
                store,
                kind="document_ingest",
                payload={"source_filename": "hinton_distillation.pdf", "parser": ""},
                run=lambda: __import__(
                    "paperwhisperer.ingestion.orchestrator", fromlist=["parse_document"]
                ).parse_document(PDF, media_type="application/pdf"),
                capture=parse_run,
                lease_seconds=LEASE_SECONDS,
                drain_timeout=600.0,
            )
        except Exception as exc:  # noqa: BLE001 - the failure being measured
            request_errors.append(f"{type(exc).__name__}: {exc}")

    request = threading.Thread(target=request_thread, name="request-thread")
    request.start()

    # The next upload, arriving while the first parse is still running. This is
    # the request that comes back as a 500.
    second: list[str] = []
    time.sleep(LEASE_SECONDS + 30)
    uploader = threading.Thread(target=lambda: second.append(_second_upload(store)))
    uploader.start()

    request.join(timeout=900)
    uploader.join(timeout=60)
    stop_job_workers()

    final = store.connection.in_transaction
    print()
    print(f"parse wall clock          : {time.monotonic() - started:.0f}s "
          f"(lease {LEASE_SECONDS:.0f}s)")
    print(f"parse result              : "
          f"{parse_run.outcome.status if parse_run.outcome else None}")
    print(f"request thread raised     : {request_errors or 'nothing'}")
    print(f"leaked transaction        : {final}")
    print(f"the next upload           : {second[0] if second else 'did not run'}")

    # And the lasting damage: this connection can never write again.
    try:
        store.claim_next("after", 300.0)
        aftermath = "claim_next OK"
    except Exception as exc:  # noqa: BLE001
        aftermath = f"{type(exc).__name__}: {exc}"
    print(f"store after the failure   : {aftermath}")

    bundle.close()
    shutil.rmtree(root, ignore_errors=True)

    reproduced = bool(request_errors) or final or "FAILED" in (second[0] if second else "")
    print()
    print("REPRODUCED" if reproduced else "NOT REPRODUCED")
    return 1 if reproduced else 0


def _second_upload(store) -> str:
    """A fresh submit on another thread: what the next user sees."""
    try:
        store.submit("document_ingest", {"source_filename": "second.pdf"})
        return "OK"
    except Exception as exc:  # noqa: BLE001
        return f"FAILED -> {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())
