"""Front-end structural checks that must hold on every commit.

    python tools/check_frontend.py

These are the invariants that keep the split front-end honest over time. They
need no reference copy of the pre-split sources, so unlike ``.parity/`` they can
run in CI:

1. every ``<link>`` / ``<script src>`` in ``index.html`` resolves to a real file,
   and no stylesheet is listed twice;
2. ``tokens.css`` is the first stylesheet, because the cascade is layered and
   the later files read the custom properties it defines;
3. the module graph under ``static/js`` is acyclic;
4. the declared entry module actually exists and nothing else claims to be one.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "templates" / "index.html"
JS_DIR = PROJECT_ROOT / "templates" / "static" / "js"

#: The cascade is layered, so token definitions have to be applied first.
FIRST_STYLESHEET = "tokens.css"

#: ``<link rel="stylesheet" href="...">`` and ``<script ... src="...">``.
STYLESHEET_RE = re.compile(r'<link[^>]+rel="stylesheet"[^>]+href="([^"]+)"')
SCRIPT_SRC_RE = re.compile(r'<script[^>]+src="([^"]+)"')
#: Static imports only. Dynamic ``import()`` and re-exports are not used here.
STATIC_IMPORT_RE = re.compile(
    r"^\s*import\s+(?:[\w*{},\s]+?\s+from\s+)?['\"](\.[^'\"]+)['\"]", re.MULTILINE
)

#: Modules whose ``getElementById`` calls name containers that must exist in the
#: template. A module qualifies when every id it looks up is a container it
#: owns and renders into, so a missing container is a wiring failure rather than
#: a legitimate optional-element probe. Modules outside this set are not scanned,
#: because most of them use ``getElementById`` to ask whether something is
#: present and are required to tolerate its absence.
#:
#: ``evidenceViewer.js`` is here for the same reason ``evidence.js`` is: the
#: whole T3.3 viewer renders into ``#evidenceViewer``, and a module whose
#: container is missing from the template returns ``null`` and shows nothing —
#: a citation that opens an empty panel, which is indistinguishable from a
#: citation that could not be located.
#:
#: ``recommendationSignals.js`` mounts both of the T3.2 run-level regions and is
#: the only place the four-state signal status is drawn. If its containers were
#: missing it would return early and render nothing, and "nothing rendered"
#: looks exactly like "every signal was fine" — the one false reading this UI is
#: built to avoid. A provider being rate limited would then produce a clean list
#: of papers with no statement that four of the five signals went unchecked.
REQUIRED_CONTAINER_MODULES = (
    "evidence.js",
    "parseStatus.js",
    "evidenceViewer.js",
    "recommendationSignals.js",
)

ELEMENT_ID_RE = re.compile(r"getElementById\('([^']+)'\)")
HTML_ID_RE = re.compile(r'\bid="([^"]+)"')


def local_references(html: str) -> tuple[list[str], list[str]]:
    """Return ``(stylesheets, scripts)`` referenced by the template."""
    stylesheets = [href for href in STYLESHEET_RE.findall(html) if href.startswith("/")]
    scripts = [src for src in SCRIPT_SRC_RE.findall(html) if src.startswith("/")]
    return stylesheets, scripts


def resolve_static(reference: str) -> Path | None:
    """Map a ``/static/...`` URL onto a file, or ``None`` if it escapes the root."""
    if not reference.startswith("/static/"):
        return None
    candidate = (PROJECT_ROOT / "templates" / reference.lstrip("/")).resolve()
    try:
        candidate.relative_to((PROJECT_ROOT / "templates").resolve())
    except ValueError:
        return None
    return candidate


def check_references(html: str) -> list[str]:
    """Every locally referenced asset must exist and no stylesheet may repeat."""
    errors: list[str] = []
    stylesheets, scripts = local_references(html)

    for kind, references in (("stylesheet", stylesheets), ("script", scripts)):
        for reference in references:
            target = resolve_static(reference)
            if target is None:
                errors.append(f"{kind} {reference!r} points outside templates/")
            elif not target.is_file():
                errors.append(f"{kind} {reference!r} has no file at {target}")

    duplicates = {ref for ref in stylesheets if stylesheets.count(ref) > 1}
    for reference in sorted(duplicates):
        errors.append(f"stylesheet {reference!r} is listed more than once")

    if stylesheets:
        first = stylesheets[0].rsplit("/", 1)[-1]
        if first != FIRST_STYLESHEET:
            errors.append(
                f"first stylesheet is {first!r}, expected {FIRST_STYLESHEET!r} "
                "(the cascade depends on it being applied first)"
            )
    return errors


def check_entry_point(html: str) -> list[str]:
    """There must be exactly one module entry point, and it must exist."""
    errors: list[str] = []
    module_scripts = [
        src
        for src in SCRIPT_SRC_RE.findall(html)
        if re.search(r'type="module"', html[max(0, html.find(src) - 200) : html.find(src)])
    ]
    if len(module_scripts) != 1:
        errors.append(
            f"expected exactly one <script type=\"module\"> entry point, "
            f"found {len(module_scripts)}: {module_scripts}"
        )
    return errors


def build_import_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for path in sorted(JS_DIR.glob("*.js")):
        source = path.read_text(encoding="utf-8")
        for specifier in STATIC_IMPORT_RE.findall(source):
            target = (path.parent / specifier).resolve()
            graph[path.name].add(target.name)
    return graph


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Return every import cycle, each reported once from its lowest member."""
    cycles: list[list[str]] = []
    state: dict[str, int] = {}  # 0 = visiting, 1 = done
    stack: list[str] = []

    def walk(node: str) -> None:
        state[node] = 0
        stack.append(node)
        for neighbour in sorted(graph.get(node, ())):
            if state.get(neighbour, 1) == 0:
                start = stack.index(neighbour)
                cycle = stack[start:] + [neighbour]
                # Normalise so the same cycle is not reported from every member.
                pivot = cycle.index(min(cycle))
                rotated = tuple(cycle[pivot:-1] + cycle[:pivot] + [cycle[pivot]])
                if rotated not in [tuple(c) for c in cycles]:
                    cycles.append(list(rotated))
            elif neighbour not in state:
                walk(neighbour)
        stack.pop()
        state[node] = 1

    for node in sorted(graph):
        if node not in state:
            walk(node)
    return cycles


def check_container_ids(html: str) -> list[str]:
    """Every container a required module mounts into must exist in the template.

    The reference and cycle checks below all work inside one file at a time, so
    they cannot catch a JavaScript module that renders into a container the
    template never declares — the panel opens, finds nothing, and the degraded
    state it was supposed to show never appears. That is exactly the silent
    degradation this front end is built to avoid, so it gets a cross-file check
    here rather than only in the static test suite.
    """
    errors: list[str] = []
    declared = set(HTML_ID_RE.findall(html))

    for module_name in REQUIRED_CONTAINER_MODULES:
        module = JS_DIR / module_name
        if not module.is_file():
            errors.append(f"required module {module_name} is missing")
            continue
        source = module.read_text(encoding="utf-8")
        for element_id in sorted(set(ELEMENT_ID_RE.findall(source))):
            if element_id not in declared:
                errors.append(
                    f"{module_name} looks up #{element_id}, which index.html "
                    "does not declare"
                )
    return errors


#: The transport layer's module name, and the invariant it anchors.
#:
#: Every network request must be issued through this one module, so that a new
#: call site inherits the timeout, the response size cap and the error taxonomy
#: by importing a function rather than by remembering three rules. A bare
#: ``fetch`` anywhere else is a request with none of them.
#:
#: This is here rather than only in the test suite for the same reason the
#: container check is: it is a property of the *whole directory*, so it can fail
#: on the mistake it exists to catch. ``tests/test_frontend_robustness.py``
#: asserts the same property with more detail; this is the cheap version that
#: runs in the front-end check on every commit.
TRANSPORT_MODULE = "api.js"

#: ``fetch(`` at a call site, not ``window.fetch`` (a lookup on the global) and
#: not a member access such as ``cache.fetch``. The lookbehind is what keeps
#: ``client.fetch(`` from being counted.
FETCH_CALL_RE = re.compile(r"(?<![.\w])fetch\s*\(")


def check_transport_choke_point() -> list[str]:
    """No module but ``api.js`` may call ``fetch``.

    A direct call site is a request with no timeout, no size cap and no error
    taxonomy — and, more seriously, one whose failure is reported by whatever
    that module happens to do with it. That variety is exactly what the
    transport layer exists to absorb, so it is checked mechanically rather than
    by review.

    ``api.js`` must contain *exactly one* ``fetch``. One per call site would put
    the bounds back where they started; more than one total means something is
    reaching around the transport it just built.
    """
    errors: list[str] = []
    for path in sorted(JS_DIR.glob("*.js")):
        source = path.read_text(encoding="utf-8")
        calls = len(FETCH_CALL_RE.findall(source))
        if path.name == TRANSPORT_MODULE:
            if calls != 1:
                errors.append(
                    f"{path.name} should contain exactly one fetch — the one every "
                    f"request goes through — but contains {calls}"
                )
            continue
        if calls:
            errors.append(
                f"{path.name} calls fetch {calls} time(s) directly, bypassing "
                f"{TRANSPORT_MODULE}; import apiRequest or readSseStream instead"
            )
    return errors


def main() -> int:
    html = TEMPLATE.read_text(encoding="utf-8")

    errors = (
        check_references(html)
        + check_entry_point(html)
        + check_container_ids(html)
        + check_transport_choke_point()
    )
    for cycle in find_cycles(build_import_graph()):
        errors.append("import cycle: " + " -> ".join(cycle))

    module_count = len(list(JS_DIR.glob("*.js")))
    print(f"checked {module_count} JS modules")

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        print(f"\n{len(errors)} problem(s)")
        return 1

    print("OK   every referenced asset exists")
    print("OK   cascade order preserved")
    print("OK   every required container is declared")
    print("OK   no import cycles")
    print(f"OK   every network call goes through {TRANSPORT_MODULE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
