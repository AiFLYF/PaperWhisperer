"""Front-end robustness invariants, asserted statically.

This backend can take minutes, stream, partially fail, be mid-reparse, have no
document, have a document whose page images are unavailable, and be temporarily
unreachable. Each of those is a *different* thing a reader has to be told about,
and the failure this file exists to prevent is the interface absorbing that
variety silently — most seriously by letting a dead request look exactly like a
working one, which for a product whose whole claim is "you can trust this" is a
contradiction rather than a cosmetic bug.

There is no browser harness in this repository and adding one is out of scope, so
nothing here renders anything. Every assertion is a structural property of the
shipped files, following the pattern ``tests/test_frontend_evidence.py``
established.

**The assertions are written against code, not against prose.** A comment that
*discusses* a guarantee cannot satisfy an assertion about behaviour, so
:func:`strip_js_comments` runs before the structural checks. And a test asserting
"the word retry appears" is worthless — the whole point of the taxonomy is that
retry appears for *some* codes and not others, so the tests that matter here are
the ones that pin down **which**:

* every network call site goes through the transport choke point, so a new call
  site inherits the timeout, the size cap and the error taxonomy by importing a
  function rather than by remembering four rules;
* every error code the backend can emit has a defined presentation, checked by
  extracting the codes from the Python source — so a new server-side code that
  the client cannot explain fails *this* test rather than surfacing as a raw
  string to a reader;
* retry is offered for retryable conditions and *only* for those;
* no call site can leave a spinner up on failure;
* user input survives a failure;
* timeouts and size caps exist and are finite.

Each of those is falsifiable by an ordinary mistake — deleting a ``finally``,
adding a direct ``fetch``, adding a new server error code — which is what makes
them worth having. A test that would pass against a broken implementation is
worse than no test, so each one below names the mistake it catches.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JS_DIR = PROJECT_ROOT / "templates" / "static" / "js"
TEMPLATE = PROJECT_ROOT / "templates" / "index.html"

RUNTIME = (JS_DIR / "runtime.js").read_text(encoding="utf-8")
API = (JS_DIR / "api.js").read_text(encoding="utf-8")
CONFIG = (JS_DIR / "config.js").read_text(encoding="utf-8")
DOM = (JS_DIR / "dom.js").read_text(encoding="utf-8")
MAIN = (JS_DIR / "main.js").read_text(encoding="utf-8")
INDEX_HTML = TEMPLATE.read_text(encoding="utf-8")

#: Every module that talks to the network. Checked as a *closed set* so that a
#: new call site cannot appear by simply adding a new file — the choke-point test
#: walks the whole directory rather than a list, which is the only way it can
#: fail on the mistake it exists to catch.
ALL_MODULES = {path.name: path.read_text(encoding="utf-8") for path in sorted(JS_DIR.glob("*.js"))}

#: The one module permitted to call ``fetch``.
CHOKE_POINT = "api.js"


# --------------------------------------------------------------------------
# Helpers — the same primitives test_frontend_evidence.py established
# --------------------------------------------------------------------------


def strip_js_comments(source: str) -> str:
    """Remove ``//`` and ``/* */`` comments, leaving string literals intact.

    Needed so a comment that *discusses* a guarantee cannot satisfy or violate a
    structural assertion about it. String state is tracked, so a ``//`` inside a
    string is not mistaken for a comment.
    """
    out: list[str] = []
    index = 0
    length = len(source)
    while index < length:
        char = source[index]
        nxt = source[index + 1] if index + 1 < length else ""
        if char == "/" and nxt == "/":
            while index < length and source[index] != "\n":
                index += 1
        elif char == "/" and nxt == "*":
            index += 2
            while index < length and not (
                source[index] == "*" and source[index + 1 : index + 2] == "/"
            ):
                index += 1
            index += 2
        elif char in "\"'`":
            quote = char
            out.append(char)
            index += 1
            while index < length:
                out.append(source[index])
                if source[index] == "\\":
                    index += 1
                    if index < length:
                        out.append(source[index])
                        index += 1
                    continue
                if source[index] == quote:
                    index += 1
                    break
                index += 1
        else:
            out.append(char)
            index += 1
    return "".join(out)


def function_body(source: str, name: str) -> str:
    """Return the body of ``function name(...)`` by brace matching.

    Brace counting tracks string state so a ``{`` inside a template literal does
    not close the function early — the failure mode that turns a structural
    assertion into a flaky one.

    The parameter list is skipped before the body is located, because
    ``apiRequest(path, options = {})`` contains a brace that belongs to neither.
    Finding it first is how a helper silently starts returning ``{}`` — and a
    test asserting against ``{}`` fails in a way that points at the code rather
    than at the helper.
    """
    match = re.search(rf"function\s+{re.escape(name)}\s*\(", source)
    assert match, f"function {name}() is missing"

    # Walk past the parameter list, tracking nesting and strings.
    # `parens` starts at 1 because `match.end()` is *after* the opening paren.
    index = match.end()
    parens = 1
    while index < len(source):
        char = source[index]
        if char in "\"'`":
            quote = char
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    break
                index += 1
        elif char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
            if parens == 0:
                break
        index += 1

    index = source.index("{", index)
    depth = 0
    start = index
    while index < len(source):
        char = source[index]
        if char in "\"'`":
            quote = char
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    break
                index += 1
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
        index += 1
    raise AssertionError(f"unbalanced braces in {name}()")


def config_table(name: str) -> str:
    """Return the source of an exported object literal in ``config.js``.

    Comments are stripped first: the tables are documented at length, and a
    brace inside a doc comment is not a value.
    """
    source = strip_js_comments(CONFIG)
    match = re.search(rf"export const {re.escape(name)} = \{{", source)
    assert match, f"config.js has no exported {name}"
    index = source.index("{", match.end() - 1)
    depth = 0
    start = index
    while index < len(source):
        char = source[index]
        if char in "\"'`":
            quote = char
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                    continue
                if source[index] == quote:
                    break
                index += 1
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
        index += 1
    raise AssertionError(f"unbalanced braces in config {name}")


def config_array(name: str) -> str:
    """Return the source of an exported array literal in ``config.js``."""
    source = strip_js_comments(CONFIG)
    match = re.search(rf"export const {re.escape(name)} = \[", source)
    assert match, f"config.js has no exported {name}"
    return source[match.end() - 1 : source.index("]", match.end()) + 1]


def failure_kind_names() -> set[str]:
    """The four kind values, read from the taxonomy rather than restated."""
    return set(re.findall(r":\s*'([a-z]+)'", config_table("FAILURE_KINDS")))



def backend_error_codes() -> set[str]:
    """Every ``code`` the backend can put in an error payload, read from the source.

    Extracted from the Python rather than hard-coded here on purpose. A list
    written out in this file would drift from the server and go stale silently;
    extracting it means that *adding a new server error code* fails this test
    suite, which is the only moment the drift is worth catching.
    """
    codes: set[str] = set()
    for path in sorted((PROJECT_ROOT / "paperwhisperer").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        # `code="..."` — the argument of build_error_response / build_error_payload.
        codes |= set(re.findall(r'code="([a-z_0-9]+)"', source))
        # The privacy gate's reason codes become the response code directly.
        codes |= set(re.findall(r'reason_code="([a-z_0-9]+)"', source))
        # Error tables: "key": (status, "sentence", "code").
        for match in re.finditer(
            r'"([a-z_0-9]+)":\s*\(\s*\d{3},.*?"([a-z_0-9]+)"\s*\)', source, re.S
        ):
            codes.add(match.group(2))
        # Named constants such as PAGE_RENDER_CODE.
        codes |= set(re.findall(r'^[A-Z_]*CODE\s*=\s*"([a-z_0-9]+)"', source, re.M))
        # `_Refusal("key")` raises are answered with the key as the code.
        codes |= set(re.findall(r'raise _Refusal\(\s*"([a-z_0-9]+)"', source))

    codes.add("bad_request")  # build_error_payload's own default
    # `allowed` is a *success* reason_code from the privacy gate, not an error.
    return codes - {"allowed"}


def i18n_error_copy_codes() -> set[str]:
    """The codes the i18n catalogue has reader-facing copy for.

    The copy is a *translation* concern and lives in ``i18n.js``; the policy
    ("may retrying help?") lives in ``config.js``. Reading both here is what
    lets the tests below assert the two halves cover the same codes — a code
    with a kind and no copy renders an empty notice, which looks exactly like a
    load failure.
    """
    source = strip_js_comments((JS_DIR / "i18n.js").read_text(encoding="utf-8"))
    start = source.index("const ERROR_COPY")
    index = source.index("{", start)
    depth = 0
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                break
        index += 1
    block = source[start : index + 1]
    zh = block[block.index("'zh-CN'") + len("'zh-CN'") :]
    zh = zh[: zh.index("\n    }")]
    return set(re.findall(r"^\s+([a-z_0-9]+):\s*\[", zh, re.M))


def presentation_entries() -> dict[str, str]:
    """The ``code: { kind }`` entries of ``ERROR_PRESENTATIONS``."""
    table = config_table("ERROR_PRESENTATIONS")
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"^    ([a-z_0-9]+):\s*(\{.*\}),?$", table, re.M)
    }


def config_source_error_copy(source: str) -> dict[str, list[str]]:
    """``{code: [title, message, hint]}`` for the zh-CN half of ``ERROR_COPY``.

    Split out from :func:`i18n_error_copy_codes` because the code set and the
    copy behind it are two different questions. "Does a code have *some* copy?"
    and "what does that copy *say*?" need the same parse and only one of them
    needs the set.

    The one-line shape is load-bearing, not incidental. Every entry is a
    single-line array because ``test_i18n.py`` matches each one with a regex
    anchored to its line, and because JavaScript does not concatenate adjacent
    string literals across a line break — a wrapped entry is a syntax error at
    runtime, not a formatting preference. So the parse refuses a line that does
    not close its own bracket: a silently-missed entry here would turn a
    wrapping mistake into a green test rather than a red one.

    Commas inside a sentence are the other trap. These entries are prose, and
    splitting on ``", "`` breaks every one of them that has a clause in it, so
    the strings are read as quoted literals instead of by delimiter.
    """
    start = source.index("const ERROR_COPY")
    index = source.index("{", start)
    depth = 0
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                break
        index += 1
    block = source[start : index + 1]
    zh = block[block.index("'zh-CN'") + len("'zh-CN'") :]
    zh = zh[: zh.index("\n    }")]

    entries: dict[str, list[str]] = {}
    for code, body in re.findall(r"^\s+([a-z_0-9]+):\s*(\[.*\])\s*,?$", zh, re.M):
        assert "\n" not in body, (
            f"the ERROR_COPY entry for {code} is not on one line; JavaScript will not "
            f"join adjacent string literals across a line break"
        )
        entries[code] = re.findall(r"'((?:[^'\\]|\\.)*)'", body)
    return entries


# --------------------------------------------------------------------------
# 1. One choke point
# --------------------------------------------------------------------------


def test_api_js_is_the_only_module_that_calls_fetch():
    """Every network call must inherit the bounds, not re-derive them.

    A direct ``fetch`` in any other module is a request with no timeout, no size
    cap and no error taxonomy — precisely the three properties this work exists
    to guarantee. The check walks the whole directory rather than a list of
    known call sites, so adding a new module that calls ``fetch`` fails here.
    """
    offenders = {
        name: source.count("fetch(")
        for name, source in ALL_MODULES.items()
        if name != CHOKE_POINT and re.search(r"(?<![.\w])fetch\(", strip_js_comments(source))
    }
    assert offenders == {}, (
        f"these modules call fetch directly and bypass the transport layer: {offenders}"
    )


def test_the_choke_point_actually_bounds_the_request():
    """The single ``fetch`` must be inside a function that sets a timeout.

    Asserting "no other module calls fetch" is only half the property; without
    this, deleting the timeout from ``apiRequest`` would leave every call site
    still routed correctly and still unbounded.
    """
    assert len(re.findall(r"(?<![.\w])fetch\(", strip_js_comments(API))) == 1, (
        "api.js should contain exactly one fetch — the one every request goes through"
    )
    body = function_body(strip_js_comments(API), "apiRequest")
    assert "AbortController" in body
    assert "setTimeout" in body
    assert "readBodyCapped" in body


def test_a_new_call_site_cannot_avoid_the_choke_point_by_importing_parse_json_safely():
    """The compatibility adapter must not be a way back in.

    ``parseJsonSafely`` and ``buildApiError`` were the previous public surface
    and are kept so an unmigrated module still works. They stay *safe* — capped,
    non-throwing, code-carrying — but a new call site should use the transport
    function. Asserting the adapters still exist keeps them from being deleted,
    and asserting they delegate keeps them from becoming a second transport.
    """
    assert "readBodyCapped" in function_body(strip_js_comments(API), "parseJsonSafely")
    assert "ApiError" in function_body(strip_js_comments(API), "buildApiError")


# --------------------------------------------------------------------------
# 2. The four states, and the codes behind them
# --------------------------------------------------------------------------


def test_the_taxonomy_names_exactly_four_states():
    """Pending is a state, and that is the load-bearing one.

    A request still in flight is not a failure. If it has no name of its own it
    gets rendered as either "working" (a spinner) or "failed" (an error), and
    the first is indistinguishable from a dead request — the defect this whole
    work exists to remove.
    """
    kinds = config_table("FAILURE_KINDS")
    values = dict(re.findall(r"(\w+):\s*'([a-z]+)'", kinds))
    assert set(values) == {"PENDING", "RETRYABLE", "TERMINAL", "UNREACHABLE"}
    assert len(set(values.values())) == 4, "two states share one value"
    # The names must be the four the requirement distinguishes, not a subset.
    assert values["PENDING"] == "pending"
    assert values["UNREACHABLE"] == "unreachable"


def test_pending_is_never_presented_as_a_failure():
    """Structurally: the pending notice is built from the pending kind.

    A test that merely checked the word "pending" existed would pass against an
    implementation that never used it, so this asserts the construction instead.
    """
    body = function_body(strip_js_comments(RUNTIME), "notifyPending")
    assert "FAILURE_KINDS.PENDING" in body
    # And it carries no retry: there is nothing to act on yet.
    assert "onRetry" not in body


def test_every_backend_error_code_has_a_defined_presentation():
    """No code the server can send may reach a reader unexplained.

    This is the assertion that keeps the two halves of the contract in step.
    A new ``build_error_response(..., code="x")`` on the server produces a
    failure this client has never been told about, and the reader sees a raw
    string. Extracting the codes from the Python means the moment the server
    grows one, this test fails.
    """
    codes = backend_error_codes()
    assert len(codes) > 80, f"the extraction found only {len(codes)} codes; it has probably broken"
    missing = sorted(codes - set(presentation_entries()))
    assert not missing, f"these server error codes have no presentation: {missing}"


def test_every_presentation_declares_exactly_one_valid_kind():
    """The policy half: one kind per code, and it must be one of the four.

    Two kinds on one code would mean the table could answer the retry question
    two ways, and the notice would pick one by accident rather than by rule.
    """
    valid_kinds = {name for name in re.findall(r"(\w+):\s*'[a-z]+'", config_table("FAILURE_KINDS"))}
    valid_kinds = {name.split(":")[0] for name in valid_kinds}
    for code, body in presentation_entries().items():
        kinds = re.findall(r"kind:\s*FAILURE_KINDS\.(\w+)", body)
        assert len(kinds) == 1, f"{code} declares {len(kinds)} kinds; it must declare exactly one"
        assert kinds[0] in valid_kinds, f"{code} has an unknown kind {kinds[0]}"


def test_every_code_with_a_policy_also_has_something_to_read():
    """The copy half, and the join between the two tables.

    `config.js` decides what may be done about a code and `i18n.js` says what
    happened; `runtime.normalizeFailure` reads both. A code present in one and
    absent from the other renders a notice with a title and no sentence — which
    looks to a reader like a load failure, and is the exact silent degradation
    this work exists to prevent. So the two tables are asserted to cover the
    same codes.
    """
    policy = set(presentation_entries())
    copy = i18n_error_copy_codes()
    assert policy, "no policy entries found; this test has stopped working"
    missing_copy = sorted(policy - copy)
    assert not missing_copy, f"these codes have a retry policy but no reader-facing copy: {missing_copy}"
    missing_policy = sorted(copy - policy)
    assert not missing_policy, (
        f"these codes have copy but no retry policy, so no module can decide whether to offer one: {missing_policy}"
    )


def test_the_policy_and_the_copy_are_joined_by_code_and_never_by_prose():
    """The two tables meet at exactly one place, and it is the code.

    `runtime.normalizeFailure` asks the catalogue for wording and this file for
    policy, both keyed by the same `code`, and carries the server's own sentence
    alongside as `detail` without using it as the headline. Asserting the join
    exists and is keyed on `code` is what keeps a second, prose-matching lookup
    from being added later.
    """
    assert "errorPresentation" in RUNTIME
    body = function_body(strip_js_comments(RUNTIME), "normalizeFailure")
    assert "errorPresentation(code" in body, "the copy must be looked up by code"
    assert "ERROR_PRESENTATIONS[code]" in body, "the policy must be looked up by code"
    # The server's sentence is kept, and explicitly not promoted.
    assert "serverDetail" in body or "detail" in body


def test_an_unknown_code_degrades_to_a_named_state_and_shows_itself():
    """An unrecognised code must never be swallowed or silently retried.

    Two halves. It defaults to terminal, because an unrecognised code is more
    likely to be a new refusal than a transient blip, and offering a retry on a
    condition that cannot change is worse than offering none. And the fallback
    text says the code is unknown and shows it, so an operator has something to
    act on rather than a blank notice.
    """
    assert re.search(
        r"export const UNKNOWN_FAILURE_KIND = FAILURE_KINDS\.TERMINAL;", CONFIG
    ), "an unknown code must not default to retryable"
    fallback = config_table("UNKNOWN_ERROR_PRESENTATION")
    assert re.search(r"title:\s*'[^']+'", fallback)
    assert re.search(r"message:\s*'[^']+'", fallback)
    # The raw code is kept on the failure so it can be shown.
    body = function_body(strip_js_comments(RUNTIME), "normalizeFailure")
    assert "known" in body, "normalizeFailure must record whether the code was recognised"
    assert "known: presentation ? true : copy.known" in body, (
        "an unrecognised code must be marked unknown rather than inheriting a presentation"
    )
    # And it still carries a code, so an operator has something to report.
    assert "client_unknown_error" in (JS_DIR / "i18n.js").read_text(encoding="utf-8"), (
        "the fallback path must still produce a named, reportable state"
    )


def test_no_module_decides_its_own_failure_kind():
    """The kind is resolved once, in one place, and never re-derived.

    A call site that computed a kind from the message or the status would be
    able to disagree with the notice — offering a retry for a terminal failure,
    or hiding one for a retryable — and the disagreement would be invisible.
    """
    for name, source in ALL_MODULES.items():
        stripped = strip_js_comments(source)
        # Reading the taxonomy is fine; writing a kind field is not.
        assert not re.search(r"kind:\s*['\"](?:pending|retryable|terminal|unreachable)['\"]", stripped), (
            f"{name} hardcodes a failure kind instead of deriving it from the code"
        )


def test_the_prose_is_never_pattern_matched_for_a_presentation():
    """The original defect, asserted against.

    ``dom.js`` used to read the server's sentence and offer an API-key tip
    because the sentence contained "401". That is a decision made from prose:
    reword the message and the advice changes; the same words in an unrelated
    message produce a wrong tip. The code is the API.
    """
    stripped = strip_js_comments(DOM)
    assert "notifyFailure" in stripped, "dom.js must delegate failure presentation to the runtime"
    for pattern in ("/401", "/429", "OPENAI_BASE_URL", "认证失败"):
        assert pattern not in DOM, f"dom.js still pattern-matches the prose for {pattern!r}"


# --------------------------------------------------------------------------
# 3. Retry policy
# --------------------------------------------------------------------------


def test_retry_is_offered_only_for_retryable_conditions():
    """The single most important property in this file.

    A "重试" button whose underlying condition is unchanged is worse than no
    button: it teaches the reader that the buttons here are decorative. The
    button is therefore built inside a branch guarded by the *policy* for the
    resolved kind, and no presentation that is not retryable may even name the
    retry label.
    """
    body = function_body(strip_js_comments(RUNTIME), "notifyFailure")
    assert "policy.actionable" in body
    assert "buildRetryControl" in body
    # The only place the label is used.
    assert len(re.findall(r"RETRY_LABEL", strip_js_comments(RUNTIME))) == 2, (
        "RETRY_LABEL must appear in its definition and in exactly one render site"
    )
    retry_body = function_body(strip_js_comments(RUNTIME), "buildRetryControl")
    assert "RETRY_LABEL" in retry_body

    # And the policy that decides: actionable is true for retryable and
    # unreachable only.
    policy = config_table("NOTICE_POLICY")
    entries = dict(re.findall(r"\[FAILURE_KINDS\.(\w+)\]:\s*\{([^}]*)\}", policy))
    assert set(entries) == {"PENDING", "RETRYABLE", "TERMINAL", "UNREACHABLE"}
    for kind, entry in entries.items():
        is_actionable = "actionable: true" in entry
        should_be = kind in {"RETRYABLE", "UNREACHABLE"}
        assert is_actionable == should_be, (
            f"{kind} has actionable={is_actionable}; only RETRYABLE and UNREACHABLE may offer a retry"
        )


def test_a_retryable_code_and_a_terminal_code_are_actually_distinguished():
    """The policy has to be applied somewhere, not merely declared.

    A table that marked everything terminal would satisfy the test above. So
    this asserts the split exists and is non-trivial in both directions: a
    genuinely transient code and a genuinely final one must land on opposite
    sides of it.
    """
    entries = presentation_entries()
    transient = ["storage_unavailable", "job_store_unavailable", "search_failed", "mineru_network_error"]
    final = [
        "missing_api_key",
        "malformed_id",
        "document_not_found",
        "mineru_quota_exhausted",
        "too_many_pages",
    ]
    for code in transient:
        assert "RETRYABLE" in entries[code], f"{code} is transient and must be retryable"
    for code in final:
        assert "TERMINAL" in entries[code], f"{code} is final and must not be retryable"
    # A quota that is spent is the canonical trap: retrying is free to offer
    # and guaranteed to fail, so it must be terminal.
    assert "RETRYABLE" not in entries["mineru_quota_exhausted"]


def test_automatic_retry_is_confined_to_safe_methods_and_finite_attempts():
    """Two ways a retry policy becomes a hang.

    Retrying a ``POST`` that starts work is not safe — the second copy may
    become a second parse. And retrying forever is indistinguishable from a hung
    request, which defeats the timeouts. Both are bounded in ``config.js`` and
    both are enforced in ``withRetry``.
    """
    methods = config_array("AUTO_RETRY_METHODS")
    assert "'GET'" in methods and "'POST'" not in methods, (
        "a POST that starts work must not be retried silently"
    )
    body = function_body(strip_js_comments(API), "withRetry")
    assert "AUTO_RETRY_METHODS" in body
    assert "MAX_AUTOMATIC_RETRIES" in body
    assert re.search(r"export const MAX_AUTOMATIC_RETRIES = \d+;", CONFIG)
    assert re.search(r"export const RETRY_BACKOFF_MS = \[", CONFIG)


def test_a_dropped_long_request_re_attaches_rather_than_restarting():
    """The job/subscription model exists for exactly this case.

    A two-minute parse that loses its connection must not charge the reader for
    the document twice. So the stream interruption is *retryable* and the
    transport offers a re-attach through the resumable job feed, carrying the
    resume cursor the server reads back as ``Last-Event-ID``.
    """
    entries = presentation_entries()
    assert "RETRYABLE" in entries["client_stream_interrupted"], (
        "a stream that dies mid-flight must be retryable, because the work may have completed"
    )
    api = strip_js_comments(API)
    assert "/v1/jobs/{jobId}/stream" in api
    assert "Last-Event-ID" in api
    attach = function_body(strip_js_comments(API), "attachToJobStream")
    assert "sinceSequence" in attach, "a re-attach must resume from the last event actually seen"
    assert "readSseStream" in attach


def test_the_sse_resume_cursor_is_actually_captured():
    """A re-attach is only resumable if the client recorded where it stopped.

    The event `id:` line is the cursor. If the parser discarded it, the
    re-attach would silently restart from zero and replay every transition,
    which is a worse failure than not re-attaching at all.
    """
    body = function_body(strip_js_comments(API), "readSseStream")
    assert "id:" in body
    assert "lastEventId" in body


# --------------------------------------------------------------------------
# 4. Nothing is swallowed, and nothing is left spinning
# --------------------------------------------------------------------------


def test_the_guarded_request_opens_and_closes_its_flight_in_one_call():
    """The spinner-clearing guarantee has to be structural.

    A spinner that outlives a dead request is *the* defect: the reader is told
    the work is continuing when it is not. Relying on each call site to remember
    a ``finally`` is relying on nine separate chances to forget. So the opening
    and the closing are both inside one function, and the closing is in a
    ``finally`` — which is the construct that runs on every path.
    """
    body = function_body(strip_js_comments(API), "withGuardedRequest")
    assert "beginFlight" in body
    assert "endFlight" in body
    assert "finally" in body, "the flight must be closed in a finally, not on one branch"
    # The failure branch must not be able to skip it.
    assert body.index("try") < body.index("catch") < body.index("finally")


def test_every_call_site_that_turns_a_spinner_on_turns_it_off_in_a_finally():
    """Per call site, not just in the abstraction.

    The guarded wrapper only helps if the call sites actually use it, and the
    belt-and-braces version of that is: every function that shows a busy state
    has a ``finally`` that clears it. This is the assertion that would catch a
    new feature added with a spinner and no cleanup.
    """
    spinner_call_sites = []
    for name, source in ALL_MODULES.items():
        stripped = strip_js_comments(source)
        for function_name in re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", stripped):
            try:
                body = function_body(stripped, function_name)
            except AssertionError:
                continue
            # Two real busy signals: a button put into its loading state, and the
            # loading block itself. `#result` gaining `.active` is *not* one —
            # that is the workspace panel appearing, not a spinner — and matching
            # on it would make this test pass for the wrong reason.
            shows_busy = "setButtonLoading(" in body and ", true)" in body
            # The loading block being turned *on*, not merely touched — a function
            # that only clears it is not a spinner it can strand.
            shows_busy = shows_busy or (
                "getElementById('loading')" in body
                and "classList.add('active')" in body
            )
            if not shows_busy:
                continue
            tail = body.split("finally")[-1]
            clears = "finally" in body and (
                "setButtonLoading(" in tail
                or ("getElementById('loading')" in tail and "classList.remove('active')" in tail)
            )
            spinner_call_sites.append((name, function_name, clears))

    assert spinner_call_sites, "no spinner call sites found; this test has stopped working"
    unprotected = [f"{n}.{f}" for n, f, cleared in spinner_call_sites if not cleared]
    assert not unprotected, (
        f"these functions show a busy state without clearing it in a finally: {unprotected}"
    )


def test_a_cancellation_is_not_reported_as_a_failure():
    """The reader cancelled it; they know. A notice would be noise.

    It also matters for the taxonomy: a cancellation is the one outcome that is
    neither a failure nor a success, and folding it into either teaches the
    reader that a control they pressed did something they did not ask for.
    """
    body = function_body(strip_js_comments(API), "withGuardedRequest")
    assert "client_cancelled" in body
    assert re.search(r"if \(failure && failure\.code !== 'client_cancelled'\)", body), (
        "a cancellation must be recognised and excluded from the failure path"
    )


def test_a_cancellation_is_never_offered_a_retry():
    """Retrying a cancellation re-runs work the reader deliberately stopped."""
    entries = presentation_entries()
    assert "RETRYABLE" not in entries["client_cancelled"]
    for name, source in ALL_MODULES.items():
        stripped = strip_js_comments(source)
        for function_name in re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", stripped):
            try:
                body = function_body(stripped, function_name)
            except AssertionError:
                continue
            if "notifyFailure" not in body:
                continue
            # A cancellation check must come before the notice, never after.
            if "client_cancelled" in body:
                assert body.index("client_cancelled") < body.index("notifyFailure"), (
                    f"{name}.{function_name} reports a cancellation as a failure"
                )


def test_a_malformed_payload_is_a_reported_degradation_not_an_exception():
    """Requirement 7, asserted against the actual construction.

    A `TypeError` thrown from inside a `catch` that is itself reporting a
    failure is how one bad response becomes a white screen. So the parse helper
    returns a value the caller can render, and the cap breach on an *error*
    response is also returned rather than thrown.
    """
    body = function_body(strip_js_comments(API), "parseJsonBody")
    assert "try" in body and "JSON.parse" in body
    assert "client_malformed_response" in body
    # And nothing is re-thrown: every path returns.
    assert "throw" not in body

    safe = function_body(strip_js_comments(API), "parseJsonSafely")
    assert "instanceof ApiError" in safe, (
        "a cap breach while reading an error response must be returned, not thrown"
    )


# --------------------------------------------------------------------------
# 5. The reader's work is not lost
# --------------------------------------------------------------------------


def test_every_field_the_reader_types_is_snapshotted_and_restorable():
    """The "more humane" half, and the half that is usually missed.

    Each field in ``PRESERVED_FIELDS`` needs a capture branch and a restore
    branch. A field that is snapshotted but never restored is dead code, and one
    that is restored but never captured cannot exist — so the test checks both
    halves of the round trip rather than the table alone.
    """
    capture = function_body(strip_js_comments(RUNTIME), "capturePreservedFields")
    restore = function_body(strip_js_comments(RUNTIME), "restorePreservedFields")
    for field in ("questionInput", "paperSearchInput", "exportFormat", "reparseParser", "reparseForceOcr"):
        assert field in CONFIG, f"{field} is not in the preserved-field table"
    # The table drives the capture; the restore walks whatever it was handed.
    # Asserting the table against the restore would be asserting the wrong half.
    assert "PRESERVED_FIELDS" in capture, "capture must iterate the declared table"
    assert "Object.entries(snapshot)" in restore, "restore must walk the captured snapshot"
    # The three kinds of field each need their own branch, because a checkbox is
    # not a text input and a file input cannot be assigned.
    for kind in ("checked", "file", "value"):
        assert f"spec.kind === '{kind}'" in capture, (
            f"capture has no branch for {kind}; a kind without one would be "
            "captured as the wrong shape and restored wrongly"
        )
    assert "entry.kind === 'checked'" in restore
    assert "entry.kind === 'file'" in restore


def test_the_preserved_table_covers_the_four_fields_the_requirement_names():
    """A Q&A question, an upload selection, an export format, a reparse setting.

    Named individually so that dropping one is a named failure. The upload
    selection is the hard one: a file input's `files` list is read-only, so the
    `File` object has to be held live and re-attached through a `DataTransfer`.
    """
    table = config_table("PRESERVED_FIELDS")
    for field in ("questionInput", "file", "exportFormat", "reparseParser", "reparseForceOcr"):
        assert re.search(rf"^\s+{field}:\s*\{{", table, re.M), f"{field} is not preserved"
    # And the file entry says so, because it is the one that cannot be stored
    # as a string and would otherwise be silently dropped.
    assert "kind: 'file'" in table
    assert function_body(strip_js_comments(RUNTIME), "restoreFileSelection").count("DataTransfer") >= 1


def test_a_restore_does_not_clobber_something_the_reader_changed_since():
    """Restoring blindly is its own kind of data loss.

    Between the failure and the restore the reader may have started typing a new
    question, and a restore that overwrites it destroys the newer text in favour
    of the older one. So a field is only written when it actually differs, and
    the ids written are returned so a caller that cleared a field on purpose can
    see that it was not restored.
    """
    body = function_body(strip_js_comments(RUNTIME), "restorePreservedFields")
    assert "element.value === entry.value" in body, (
        "a field whose value already matches must not be rewritten"
    )
    assert "restored.push" in body
    assert "return restored" in body


def test_the_failure_paths_actually_restore():
    """The mechanism existing is not the same as it being used.

    The three call sites where losing work would actually hurt — a two-minute
    analysis, a typed question, a failed search — must each capture before the
    request and restore in the catch. A test that only checked the helpers would
    pass against a front end that never called them.
    """
    for module, function_name, trigger in (
        ("analyze.js", "analyze", "apiRequest"),
        ("chat.js", "askQuestion", "apiRequest"),
        ("papers.js", "searchPapers", "apiRequest"),
    ):
        body = function_body(strip_js_comments(ALL_MODULES[module]), function_name)
        assert "capturePreservedFields()" in body, f"{module}.{function_name} captures nothing"
        assert "restorePreservedFields(" in body, f"{module}.{function_name} restores nothing"
        # Capture must come *before* the request: by the time the catch runs, an
        # optimistic clear has already happened and there is nothing to restore.
        assert body.index("capturePreservedFields()") < body.index(trigger), (
            f"{module}.{function_name} captures after the request, which is too late"
        )


def test_a_failed_analysis_does_not_leave_an_empty_screen_where_a_workspace_was():
    """A failure that blanks a populated workspace must say what happened to it.

    Either the previous workspace is restored, or there was none — but a silent
    empty result reads as "the analysis found nothing", which is a claim about
    the reader's paper that nothing supports.
    """
    body = function_body(strip_js_comments(ALL_MODULES["analyze.js"]), "analyze")
    assert "previousWorkspace" in body
    assert "restoreWorkspaceState" in body
    assert "restored" in body, "the restored-workspace case must be stated in words"


# --------------------------------------------------------------------------
# 6. Time and motion honesty
# --------------------------------------------------------------------------


def test_progress_is_honest_about_what_it_does_not_know():
    """A progress bar that guesses is worse than one that admits ignorance.

    The server streams *which step* it is on and never how far through that step
    it is, so any percentage would be invented. The step indicator is driven by
    the server; the elapsed clock is the honest progress measure; and the note
    saying "no percentage is shown" is visible rather than implied.
    """
    story = config_table("PROGRESS_STORY")
    assert "determinate: false" in story
    assert re.search(r"unknownNote:\s*'[^']+'", story)
    assert "百分比" in story, "the note must say in words that no percentage is shown"
    assert re.search(r"elapsedLabel:\s*'[^']+'", story)
    # No module computes a percentage from a step index.
    for name, source in ALL_MODULES.items():
        stripped = strip_js_comments(source)
        assert not re.search(r"progress\w*\s*=\s*\d+\s*/", stripped), (
            f"{name} computes a progress fraction, which the server never reports"
        )


def test_the_elapsed_clock_runs_and_is_stopped_with_the_run():
    """Motion has to be bounded by the same ``finally`` as the spinner.

    A clock that keeps counting after the work ended is the same class of lie as
    a spinner that stays up — it just looks less alarming, which is worse,
    because it is the version a reader stops noticing.
    """
    for name in ("beginProgressClock", "endProgressClock", "updateProgressElapsed"):
        assert f"function {name}(" in DOM, f"dom.js has no {name}()"
    begin = function_body(strip_js_comments(ALL_MODULES["analyze.js"]), "analyze")
    assert "beginProgressClock()" in begin
    # Both the begin and the end live in `analyze`, and the end is in the
    # finally that also takes the spinner down.
    finally_block = begin.split("finally")[-1]
    assert "endProgressClock(" in finally_block, "the clock must stop in the finally"
    assert "classList.remove('active')" in finally_block


def test_a_stalled_stream_is_told_from_a_working_one():
    """The distinction the product's claim depends on, made mechanical.

    Past the stall budget with no byte and no event, the request is declared
    stalled and reported as retryable. Without the watchdog, a dead connection
    and a two-minute parse are the same picture, and the reader cannot tell them
    apart — which means the interface cannot either.
    """
    body = function_body(strip_js_comments(API), "readSseStream")
    assert "STREAM_STALL_TIMEOUT_MS" in body
    assert "armStall" in body
    assert "client_stream_interrupted" in body
    assert "Promise.race" in body, (
        "the watchdog must be able to end a blocked read, not merely be checked between reads"
    )
    # Reset by bytes, so keep-alive comments are not mistaken for a stall.
    assert body.index("armStall()") < body.index("reader.read()")
    # And the reader is released on every path, or the connection is held open.
    assert "reader.releaseLock()" in body
    assert "reader.cancel()" in body


# --------------------------------------------------------------------------
# 7. Bounds
# --------------------------------------------------------------------------


def test_every_timeout_is_a_finite_positive_number():
    """``Infinity`` and ``0`` both read as "unbounded" and are not.

    A timeout table is only a bound if every value in it is a real deadline. A
    single ``0`` — an easy thing to write meaning "use the default" — silently
    removes the bound for whichever operation used it.
    """
    for table_name in ("REQUEST_TIMEOUTS_MS", "RESPONSE_SIZE_CAPS_BYTES"):
        table = config_table(table_name)
        numbers = re.findall(r":\s*([0-9][0-9_*\s]*[0-9]?)\s*[,}]", table)
        assert numbers, f"{table_name} declares no numeric bounds"
        for value in numbers:
            assert value not in {"0"}, f"{table_name} contains a 0, which is not a bound"
            assert "Infinity" not in table, f"{table_name} contains Infinity"
    for array_name in ("RESYNC_BACKOFF_MS", "RETRY_BACKOFF_MS"):
        values = config_array(array_name)
        numbers = re.findall(r"\d+", values)
        assert numbers, f"{array_name} declares no delays"
        assert all(int(value) > 0 for value in numbers), f"{array_name} contains a zero delay"
    assert re.search(r"export const STREAM_STALL_TIMEOUT_MS = \d+;", CONFIG)
    assert re.search(r"export const RESYNC_TIMEOUT_MS = \d+;", CONFIG)


def test_the_operations_that_can_be_slow_have_a_generous_bound():
    """The bounds answer a question about the slowest legitimate request.

    A health probe is allowed ten seconds because the alternative to an answer
    is not "wait longer"; a document analysis is allowed minutes because the
    server legitimately takes minutes. What must not happen is a parse bounded
    by the same deadline as a liveness probe, which would report a working
    server as a dead one.
    """
    timeouts = config_table("REQUEST_TIMEOUTS_MS")
    values = dict(
        (name, int(re.sub(r"[^0-9]", "", value)))
        for name, value in re.findall(r"(\w+):\s*([0-9_*\s]+[0-9]?)\s*,", timeouts)
    )
    assert values["health"] < values["default"] < values["analysis"]
    assert values["analysis"] >= 600_000, "a multi-minute parse needs a multi-minute bound"
    assert values["health"] <= 15_000


def test_a_response_is_capped_and_the_cap_is_enforced_mid_read():
    """A cap that is only checked afterwards is not a cap.

    The declared `Content-Length` is honoured before reading a byte, and the
    running total is checked as the body streams — otherwise a response that
    lies about its size, or omits the header, is read into memory regardless.
    """
    body = function_body(strip_js_comments(API), "readBodyCapped")
    assert "content-length" in body.lower()
    assert "byteLength" in body
    assert "releaseLock" in body, "a cap breach must still release the connection"
    caps = config_table("RESPONSE_SIZE_CAPS_BYTES")
    for cap in ("json", "streamEvent", "streamTotal"):
        assert re.search(rf"{cap}:\s*[0-9]", caps), f"there is no cap for {cap}"
    # The stream total is enforced in the read loop, not only per event.
    stream = function_body(strip_js_comments(API), "readSseStream")
    assert "streamTotal" in stream


def test_the_text_fallback_cap_measures_bytes_rather_than_characters():
    """Where streaming bodies are unavailable, the cap would otherwise vanish.

    `text.length` counts UTF-16 code units and under-reports non-ASCII by up to
    a factor of three, so a CJK body would pass a check that is meant to stop it.
    """
    body = function_body(strip_js_comments(API), "readBodyCapped")
    assert "TextEncoder" in body, (
        "the non-streaming fallback must measure encoded bytes, not string length"
    )


# --------------------------------------------------------------------------
# 8. Connectivity, and what happens to an action taken offline
# --------------------------------------------------------------------------


def test_offline_is_its_own_state_and_not_a_failure():
    """The browser knowing there is no network is a fact about the machine.

    It has its own remedy — wait for connectivity — and collapsing it into
    "the server returned an error" would send the reader looking at a backend
    that is perfectly healthy.
    """
    entries = presentation_entries()
    assert "UNREACHABLE" in entries["client_offline"]
    assert "UNREACHABLE" in entries["client_unreachable"]
    body = function_body(strip_js_comments(API), "apiRequest")
    assert "isOnline()" in body, "an offline request must fail fast rather than wait out a timeout"
    assert "client_offline" in body


def test_an_action_taken_offline_is_queued_visibly_and_flushed_on_reconnect():
    """Silently dropped is the failure this queue exists to prevent.

    Three parts, all load-bearing: the action is *stored* rather than discarded,
    the queue says how many items are waiting, and reconnecting actually drains
    it. A queue that stored but never flushed would be a slower way to lose the
    work.
    """
    assert "enqueueWhenOffline" in RUNTIME
    enqueue = function_body(strip_js_comments(RUNTIME), "enqueueWhenOffline")
    assert "pendingActions.push" in enqueue, "a queued action must be stored, not dropped"
    assert "renderOfflineNotice" in enqueue, "a queued action must be visible"
    flush = function_body(strip_js_comments(RUNTIME), "flushOfflineQueue")
    assert "splice" in flush
    assert "await action()" in flush
    copy = config_table("OFFLINE_QUEUE_COPY")
    assert re.search(r"title:\s*'[^']+'", copy)
    assert re.search(r"message:\s*'[^']+'", copy)
    assert re.search(r"restored:\s*'[^']+'", copy)


def test_a_queued_action_that_fails_on_flush_is_reported():
    """Flushing is not success. An action that fails there has still failed.

    The tempting shortcut is to `catch` and continue in the loop, which is
    exactly the silent drop the queue was built to avoid: the reader asked for
    something, the request went out, it failed, and the interface said nothing.
    """
    body = function_body(strip_js_comments(RUNTIME), "flushOfflineQueue")
    catch_block = body.split("catch")[1] if "catch" in body else ""
    assert "notifyFailure" in catch_block, (
        "a queued action that fails on flush must be reported like any other failure"
    )


def test_connectivity_listeners_are_bound_exactly_once():
    """A module wired on import *and* called explicitly would double-fire.

    Every `online` event would then flush the queue twice, and the second flush
    would race the first over a queue that has already been spliced empty.
    """
    body = function_body(strip_js_comments(RUNTIME), "initializeConnectivity")
    assert "addEventListener('online'" in body
    assert "addEventListener('offline'" in body
    assert "ConnectivityBound" in body, "initialisation must be idempotent"


def test_the_runtime_is_started_before_anything_can_fail():
    """A failure with no surface is a failure with no report.

    The entry point initialises the runtime first, so a request issued during
    the rest of startup has somewhere to be reported and a listener already
    bound to `online`.
    """
    body = function_body(strip_js_comments(MAIN), "initialize")
    assert "initializeRuntime()" in body
    assert body.index("initializeRuntime()") < body.index("initializeTheme()")


# --------------------------------------------------------------------------
# 9. The notice surface
# --------------------------------------------------------------------------


def test_a_notice_is_assembled_in_one_place_and_nothing_else_builds_one():
    """One surface, so the shape of a failure cannot vary with its file.

    A notice is a title, a message, an optional hint, and conditionally a retry
    and a dismiss. If another module assembled its own, the retry button's
    enablement rule would be duplicated and the two could disagree.
    """
    body = function_body(strip_js_comments(RUNTIME), "notifyFailure")
    for element in ("runtime-notice-title", "runtime-notice-message", "runtime-notice-hint"):
        assert element in body, f"the notice is missing its {element}"
    # No other module may hand-build one.
    for name, source in ALL_MODULES.items():
        if name == "runtime.js":
            continue
        assert "runtime-notice" not in source, f"{name} builds its own notice element"


def test_a_notice_is_a_live_region_and_the_loading_state_is_not_assertive():
    """Two live regions with different urgencies, kept apart.

    A notice is a status update about work the reader started and is watching;
    announcing it assertively would interrupt a screen reader mid-sentence for
    something they already know is happening. The error *banner* is the
    assertive one, and it is a different element.
    """
    body = function_body(strip_js_comments(RUNTIME), "ensureContainer")
    assert "aria-live" in body
    assert "'polite'" in body
    assert 'id="error"' in INDEX_HTML
    assert 'aria-live="assertive"' in INDEX_HTML


def test_the_elements_the_runtime_looks_up_actually_exist_in_the_template():
    """The failure surface must not mount into a container the template lacks.

    If the notice container could not be created, `notifyFailure` would still
    record the failure and the reader would see nothing — a failure that is
    reported into a void. So every id the runtime looks up is checked against
    the real HTML.
    """
    declared = set(re.findall(r'\bid="([^"]+)"', INDEX_HTML))
    # The two the runtime *creates* rather than expects; everything else must
    # already be in the template.
    created = {"runtimeNotices", "connectivityChip", "progressElapsed"}
    for element_id in sorted(set(re.findall(r"getElementById\('([^']+)'\)", RUNTIME))):
        if element_id in created:
            continue
        assert element_id in declared, (
            f"runtime.js looks up #{element_id}, which index.html does not declare"
        )


def test_the_runtime_takes_no_dependency_the_project_does_not_have():
    """No framework, no bundler, no polyfill.

    `AbortSignal.any` and `structuredClone` are the two things this work could
    plausibly have reached for and did not; the first is composed by hand in
    `api.js` precisely because this project takes no polyfill, and the second is
    not used at all.
    """
    for name, source in ALL_MODULES.items():
        lowered = source.lower()
        for framework in ("react", "vue.", "angular", "svelte", "jquery", "preact"):
            assert framework not in lowered, f"{name} references {framework}"
        for specifier in re.findall(r"from\s+'([^']+)'", source):
            assert specifier.startswith("./"), f"{name} imports the bare specifier {specifier}"
            assert (JS_DIR / specifier[2:]).is_file(), f"{name} imports missing {specifier}"
    api_stripped = strip_js_comments(API)
    assert "AbortSignal.any" not in api_stripped, (
        "AbortSignal.any is not universally available and this project takes no polyfill"
    )
    # The hand-rolled composition has to exist for that to be a real choice.
    assert "function AbortAny" in API


# --------------------------------------------------------------------------
# 10. The properties that are easiest to break by accident
# --------------------------------------------------------------------------


def test_the_request_error_shape_survives_every_path():
    """One error type, carrying the code, so a `catch` can always act.

    A `catch` that only reads `.message` still gets a correct message; one that
    wants to offer a retry reads `.kind`. If any path produced a bare `Error`,
    the taxonomy would silently not apply to it.
    """
    assert "class ApiError extends Error" in API
    body = function_body(strip_js_comments(API), "withGuardedRequest")
    assert "transportFailure" in body, "a non-ApiError must be normalised before it is reported"
    # The normalisation is the only way an arbitrary throw becomes reportable.
    normalize = function_body(strip_js_comments(API), "transportFailure")
    assert "ApiError" in normalize


def test_a_timeout_is_not_reported_as_a_cancellation():
    """They look identical at the call site and mean opposite things.

    Both abort an `AbortController`, so a single `catch` sees the same error
    name for "the reader pressed cancel" and "we gave up waiting". Reporting the
    second as the first tells the reader their request was stopped when it was
    abandoned, and hides the fact that the server may still be working on it.
    """
    body = function_body(strip_js_comments(API), "apiRequest")
    assert "timeoutController.signal.aborted" in body, (
        "the timeout cause must be checked before falling through to the generic handler"
    )
    assert "client_timeout" in body
    # The *other* abort cause is a cancellation, and it is classified in exactly
    # one place so the two can never be swapped.
    assert "transportFailure" in body
    classify = function_body(strip_js_comments(API), "transportFailure")
    assert "isAbortError" in classify
    assert "client_cancelled" in classify
    # Two controllers exist precisely so the causes can be told apart: one for
    # the deadline and one for the caller's own cancellation. Counted over the
    # module because the second lives in the hand-rolled combiner, not in this
    # function — asserting only this body would pass with a single controller
    # and a caller whose cancel button silently did nothing.
    assert strip_js_comments(API).count("new AbortController") >= 2


def test_a_successful_request_retires_its_own_failures():
    """A stale failure next to a fresh success is a contradiction.

    After a retry works, the notice that said it would fail is no longer true,
    and leaving it up teaches the reader that the interface does not know what
    happened.
    """
    assert "dismissNoticesFor" in RUNTIME
    body = function_body(strip_js_comments(RUNTIME), "endFlight")
    assert "dismissNoticesFor" in body
    guarded = function_body(strip_js_comments(API), "withGuardedRequest")
    assert "succeeded: true" in guarded, "a success must retire the notices it raised"


# --------------------------------------------------------------------------
# 12. `cloud_outcome` — the field the backend sends and the client dropped
# --------------------------------------------------------------------------
#
# The backend has always promoted a machine-readable cloud verdict to the client
# (`ParseResult.cloud_outcome` → `ParseOutcome.cloud_outcome` →
# `build_parse_provenance` in `analysis/orchestrator.py`, alongside
# `error_message` "so a branch has something to show") and `parseStatus.js`
# rendered every other field on that payload while ignoring this one. The
# consequence was not cosmetic: `error_message` is prose, so four different
# conditions — a spent quota, a stall, a policy refusal and a plain failure —
# arrived as one sentence, and a reader told to top up a quota that was in fact
# blocked stops trusting the next message too.
#
# These assertions are about the *taxonomy* rather than about the rendering: that
# every value the adapter records reaches a declared code with a declared kind,
# and that the code it reaches is one the failure surface already knows.


def cloud_outcome_table() -> dict[str, str]:
    """The ``outcome: { code: '...' }`` entries of ``CLOUD_OUTCOME_PRESENTATIONS``.

    The codes only — ``kind`` is not read from this table and must not be: the
    whole point of routing through ``ERROR_PRESENTATIONS`` is that the retry
    policy is declared in exactly one place. A second ``kind`` here would be two
    answers to "may retrying help?" for one condition.
    """
    table = config_table("CLOUD_OUTCOME_PRESENTATIONS")
    return {
        match.group(1): match.group(2)
        for match in re.finditer(r"^    ([a-z_0-9]+):\s*\{[^}]*code:\s*'([a-z_0-9]*)'", table, re.M)
    }


def adapter_cloud_outcomes() -> set[str]:
    """The outcomes ``ingestion/adapters/mineru_cloud.py`` actually records.

    Extracted from the Python rather than restated, for the reason
    :func:`backend_error_codes` gives: a list written out here would drift from
    the adapter silently, and the day the adapter grew an outcome this client
    would degrade it to the generic message without a test noticing.
    """
    source = (PROJECT_ROOT / "paperwhisperer" / "ingestion" / "adapters" / "mineru_cloud.py").read_text(
        encoding="utf-8"
    )
    # Three ways the adapter names one: the `finish(document, outcome, ...)`
    # calls that give up, the assignment that picks the two shapes of success,
    # and that ternary's `else` arm. The last two are matched narrowly — a bare
    # `else "..."` would sweep up every default string in the module, and a
    # test whose extraction is too loose reports nonsense as missing copy.
    return (
        set(re.findall(r'finish\(\s*[^,]+,\s*"([a-z_]+)"', source))
        | set(re.findall(r'outcome\s*=\s*"[a-z_]+"\s+if\s+\w+\s+else\s+"([a-z_]+)"', source))
        | set(re.findall(r'outcome\s*=\s*"([a-z_]+)"', source))
    )


def test_every_outcome_the_cloud_adapter_records_is_presented():
    """The client must name every verdict the server can send it.

    This is the same contract as ``test_every_backend_error_code_has_a_defined_presentation``,
    one level along: there it is codes, here it is the *values of a field*. An
    outcome this build does not recognise degrades to the generic
    ``cloud_parse_failed``, which is a safe direction but a strictly worse one —
    so the adapter's real vocabulary is asserted against the table rather than
    left to the fallback.
    """
    outcomes = adapter_cloud_outcomes()
    assert outcomes, "the extraction found no outcomes; it has probably broken"
    assert {"success", "partial", "quota_exhausted", "timed_out", "blocked", "failed"} <= outcomes, (
        f"the adapter's recorded outcomes changed; it now records {sorted(outcomes)}"
    )
    missing = sorted(outcomes - set(cloud_outcome_table()))
    assert not missing, f"the cloud adapter records outcomes with no presentation: {missing}"


def test_every_presented_outcome_names_a_code_the_runtime_knows():
    """The join: outcome → code → kind → policy, with nothing invented along it.

    Two failure modes, and this catches both. A code that is not in
    ``ERROR_PRESENTATIONS`` would fall through to ``UNKNOWN_FAILURE_KIND``,
    turning a genuinely retryable stall into a terminal refusal. A code that
    *is* there but has no reader-facing copy would render a title with an empty
    body — the silent degradation this whole file exists to prevent.
    """
    table = cloud_outcome_table()
    assert table, "CLOUD_OUTCOME_PRESENTATIONS is empty; this test has stopped working"
    policy = presentation_entries()
    copy = i18n_error_copy_codes()
    for outcome, code in table.items():
        if not code:
            # Only the two outcomes that mean the cloud leg succeeded may mint no
            # code, and they must be marked as not reporting.
            entry = re.search(
                rf"^    {re.escape(outcome)}:\s*(\{{[^}}]*\}})", config_table("CLOUD_OUTCOME_PRESENTATIONS"), re.M
            )
            assert entry, f"{outcome} has no entry"
            assert "reports: false" in entry.group(1), (
                f"{outcome} mints no code, so it must be marked reports: false — a "
                f"failure with no code reaches the reader as nothing"
            )
            continue
        assert code in policy, f"cloud outcome {outcome} maps to {code}, which has no retry policy"
        assert code in copy, f"cloud outcome {outcome} maps to {code}, which has no reader-facing copy"


def test_the_spent_quota_outcome_reaches_the_quota_code_that_already_existed():
    """The dead-code question, answered as an assertion rather than a note.

    ``mineru_quota_exhausted`` was registered in ``ERROR_PRESENTATIONS`` and given
    copy in both languages while no route could emit it — a prepared-but-unreachable
    code path, which is a race waiting for the day the backend starts using it. The
    resolution is that the ``quota_exhausted`` outcome *reaches that same code*:
    one condition, one code, one policy, one set of words, whether it arrives as
    an HTTP error code or as a field on a payload that otherwise succeeded. Had
    the outcome minted a second, parallel code instead, the two mechanisms would
    be able to disagree about the same condition.
    """
    table = cloud_outcome_table()
    assert table["quota_exhausted"] == "mineru_quota_exhausted", (
        "the spent-quota outcome must reach the code that already says so, not a "
        "second code for the same condition"
    )
    # And no other outcome may reach it, or "the quota is spent" would become the
    # generic answer to several different failures.
    borrowed = [outcome for outcome, code in table.items() if code == "mineru_quota_exhausted"]
    assert borrowed == ["quota_exhausted"], (
        f"more than one outcome resolves to the quota code: {borrowed}"
    )


def test_the_cloud_outcomes_that_ask_for_something_get_different_something():
    """Each actionable outcome's *hint* names the action it is asking for.

    The message bodies are checked for distinctness by an executed test in
    ``test_frontend_ux.py``; what is asserted here is the property a static check
    can actually make, and which is easy to get wrong by accident: the three
    outcomes that tell the reader to do something — spend quota, switch parser,
    ask an administrator — must point at three different things, in both
    languages. Two hints that both say "try again" is the same overlap defect
    wearing different words.
    """
    source = strip_js_comments((JS_DIR / "i18n.js").read_text(encoding="utf-8"))
    zh = config_source_error_copy(source)
    quota, blocked, timed_out = (
        zh["mineru_quota_exhausted"],
        zh["cloud_parse_blocked"],
        zh["cloud_parse_timed_out"],
    )

    for copy in (quota, blocked, timed_out):
        assert len(copy) == 3, f"an ERROR_COPY entry did not parse as [title, message, hint]: {copy}"
    assert quota[2], "a spent quota has to say what to do about it"
    assert blocked[2], "a policy refusal has to say who can change it"
    assert timed_out[2], "a stall has to say that retrying is the move"
    assert len({quota[2], blocked[2], timed_out[2]}) == 3, (
        "two of the three actionable cloud outcomes give the same instruction"
    )
    # Three different remedies for three different owners of the problem: the
    # quota is the reader's (switch parser), the refusal is an administrator's,
    # and the stall is nobody's — waiting is what fixes it.
    assert "Docling" in quota[2], (
        "a spent quota's remedy is to stop using the cloud parser, and the hint must say so"
    )
    assert "管理员" in blocked[2], "a policy refusal has to name the administrator"
    assert "重试" in timed_out[2], "a stall's remedy is to ask again, and the hint must say so"

    # And the same in the other locale. This is the check that would catch an
    # English string being written from the Chinese one: the three English hints
    # are different words for the same three remedies.
    en = {
        code: re.findall(r"'((?:[^'\\]|\\.)*)'", body)
        for code, body in re.findall(
            r"^\s+(cloud_parse_[a-z_]+|mineru_quota_exhausted):\s*(\[.*\])\s*,?$",
            source[source.index("    en: {") :],
            re.M,
        )
    }
    assert len(en) == 4, f"the English cloud copy could not be read: {sorted(en)}"
    assert len({en["mineru_quota_exhausted"][2], en["cloud_parse_blocked"][2], en["cloud_parse_timed_out"][2]}) == 3, (
        "the English hints overlap even where the Chinese ones do not"
    )


def test_every_error_copy_entry_stays_on_one_line():
    """The single-line rule, asserted from the other side of the parser.

    ``tests/test_i18n.py`` matches each ``ERROR_COPY`` entry with a regex
    anchored to one line, so a wrapped entry becomes invisible to it — the
    parity checks pass and the notice renders an empty body. It is also a
    runtime break, because JavaScript does not join adjacent string literals
    across a line break, so ``['a',\n'b']`` is two array elements rather than a
    concatenation.
    """
    source = strip_js_comments((JS_DIR / "i18n.js").read_text(encoding="utf-8"))
    start = source.index("const ERROR_COPY")
    entries = re.findall(r"^\s+[a-z_0-9]+:\s*\[", source[start:], re.M)
    # Both locales, so the count is twice the zh-CN code set. Tying it to that
    # rather than a magic number means the assertion tracks the table instead of
    # silently passing or failing when the table legitimately changes size.
    assert len(entries) == 2 * len(i18n_error_copy_codes()), (
        f"{len(entries)} ERROR_COPY entries were found, which is not the two locales "
        f"× {len(i18n_error_copy_codes())} codes; the parse has probably broken"
    )
    for match in re.finditer(r"^\s+([a-z_0-9]+):\s*\[([^\]]*)\]", source[start:], re.M):
        code, body = match.group(1), match.group(2)
        assert "\n" not in body, f"the ERROR_COPY entry for {code} wraps across lines"
        assert len(re.findall(r"'((?:[^'\\]|\\.)*)'", f"[{body}]")) == 3, (
            f"the ERROR_COPY entry for {code} is not a [title, message, hint] triple"
        )


def test_a_cloud_outcome_is_never_rendered_without_the_failure_surface():
    """The card may not answer a failure on its own.

    ``runtime.js`` owns failures: four kinds, one ``notifyFailure``, one place a
    retry is decided. A card that rendered an outcome's sentence and hint itself
    would have to re-implement the retry button beside it, and the two would
    drift — which is how the outcome came to be discarded in the first place.
    """
    parse_status = strip_js_comments((JS_DIR / "parseStatus.js").read_text(encoding="utf-8"))
    reporter = function_body(parse_status, "reportCloudOutcome")
    assert "notifyFailure" in reporter, "the outcome's explanation must go to the failure surface"
    assert not re.search(r"kind:\s*['\"](?:pending|retryable|terminal|unreachable)['\"]", reporter), (
        "parseStatus.js must not name a failure kind; that is the taxonomy's job"
    )
    # The card's row is a *fact* row: no notice, no button, no copy of its own.
    row = function_body(parse_status, "appendCloudOutcomeRow")
    assert "notifyFailure" not in row
    assert "appendNotice" not in row, (
        "the outcome belongs in the card's facts; a second notice surface would be "
        "the parallel path this module was built to avoid"
    )
    # The wording still comes from the catalogue, in both languages.
    assert "t(" in row, "the card's sentence must be a catalogue lookup, not a literal"
