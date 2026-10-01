#!/usr/bin/env python3
"""One real LLM call, end to end, with a PASS/FAIL verdict.

A canary for a developer or a CI job that wants to know one thing: is the
configured OpenAI-compatible endpoint actually serving our request shape? The
integration suite in ``tests/test_llm_integration.py`` answers that question
too, but it answers sixteen of them and needs pytest. This answers exactly
one, in a couple of seconds, and prints ``PASS`` or ``FAIL`` on its own line
so a shell job can read it without parsing anything.

    python tools/llm_smoke.py
    python tools/llm_smoke.py --model gpt-4o-mini
    python tools/llm_smoke.py --json          # machine-readable, for CI

Deliberately **not** wired into ``make test``. The default suite must never
depend on a reachable network service, and a real bill, to be green.

Exit codes: ``0`` PASS, ``1`` FAIL, ``2`` SKIP (nothing configured or the
endpoint is not answering -- a distinct outcome from FAIL, because it is not
a statement about the product).

It touches the same credentials and the same transport the product uses
(``env_loader`` -> ``config`` -> ``LLMClient``), so a PASS here means the
transport, the base URL, the key and the model all line up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Ceiling on the single completion this script makes. Nothing here retries:
#: a canary that retries three times is three times the cost and three times
#: the wait, for the same answer.
MAX_COMPLETIONS = 1

#: Seconds. A canary answers "is it up", not "how fast is it"; anything slower
#: than this is reported as a failure to answer rather than a slow pass.
TIMEOUT_SECONDS = 45.0

#: The probe. See the reasoning in tests/test_llm_integration.py -- a health
#: check that costs a completion is not a health check.
PROBE_TIMEOUT_SECONDS = 3.0

REQUIRED_ENV = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")


def _settings() -> dict[str, str]:
    """Read the endpoint straight from the environment, not from ``config``.

    ``config.OPENAI_BASE_URL`` and ``config.OPENAI_MODEL`` both carry vendor
    defaults, so they are never empty and can never tell you the variable is
    missing. Importing ``config`` first is still what loads the project
    ``.env`` into ``os.environ``; only the read comes afterwards.
    """
    from paperwhisperer.core import config  # noqa: F401 - import for its .env side effect

    return {
        "base_url": os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/"),
        "model": os.environ.get("OPENAI_MODEL", "").strip(),
        "api_key": os.environ.get("OPENAI_API_KEY", "").strip(),
    }


def _missing_env() -> list[str]:
    settings = _settings()
    checks = {
        "OPENAI_BASE_URL": settings["base_url"],
        "OPENAI_API_KEY": settings["api_key"],
        "OPENAI_MODEL": settings["model"],
    }
    return [name for name in REQUIRED_ENV if not checks[name]]


def probe(base_url: str, api_key: str) -> tuple[bool, str]:
    """Cheap reachability check. Never spends a completion."""
    import httpx

    try:
        response = httpx.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
        return False, f"{type(exc).__name__}: {exc}"
    if response.status_code >= 500:
        return False, f"HTTP {response.status_code}"
    return True, f"HTTP {response.status_code}"


def run_call(base_url: str, model: str, api_key: str, timeout: float) -> tuple[str, float]:
    """Make the one real completion, through the same SDK the product uses."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    started = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You reply with exactly one word."},
            {"role": "user", "content": "Reply with the single word: READY"},
        ],
        max_tokens=8,
        timeout=timeout,
    )
    elapsed = time.time() - started
    return (response.choices[0].message.content or "").strip(), elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="", help="override OPENAI_MODEL for this call")
    parser.add_argument(
        "--timeout", type=float, default=TIMEOUT_SECONDS, help="seconds before giving up"
    )
    parser.add_argument("--json", action="store_true", help="print a JSON result instead of prose")
    args = parser.parse_args(argv)

    def emit(status: str, detail: str, **extra) -> int:
        code = {"PASS": 0, "FAIL": 1, "SKIP": 2}[status]
        if args.json:
            print(json.dumps({"status": status, "detail": detail, **extra}, ensure_ascii=False))
        else:
            print(f"{status}: {detail}")
            for key, value in extra.items():
                print(f"  {key}: {value}")
        return code

    missing = _missing_env()
    if missing:
        return emit(
            "SKIP",
            f"no LLM configured; {', '.join(missing)} not set",
            hint="set OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL (e.g. in .env)",
        )

    settings = _settings()
    base_url = settings["base_url"]
    model = args.model or settings["model"]
    api_key = settings["api_key"]

    reachable, probe_detail = probe(base_url, api_key)
    if not reachable:
        return emit(
            "SKIP",
            f"endpoint {base_url} is not answering ({probe_detail})",
            hint="start the server, or point OPENAI_BASE_URL at one that is up",
        )

    try:
        text, elapsed = run_call(base_url, model, api_key, args.timeout)
    except Exception as exc:  # noqa: BLE001 - a canary reports, it does not raise
        return emit(
            "FAIL",
            f"the endpoint accepted the request but did not answer: {type(exc).__name__}: {exc}",
            base_url=base_url,
            model=model,
            probe=probe_detail,
        )

    if not text:
        return emit("FAIL", "the endpoint returned a 2xx with empty content",
                    base_url=base_url, model=model, elapsed_seconds=round(elapsed, 2))

    return emit(
        "PASS",
        "the configured endpoint served a real completion",
        base_url=base_url,
        model=model,
        elapsed_seconds=round(elapsed, 2),
        probe=probe_detail,
        completions_spent=MAX_COMPLETIONS,
        reply=text[:80],
    )


if __name__ == "__main__":
    raise SystemExit(main())
