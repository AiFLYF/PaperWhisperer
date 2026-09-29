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


def main() -> int:
    html = TEMPLATE.read_text(encoding="utf-8")

    errors = check_references(html) + check_entry_point(html)
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
    print("OK   no import cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
