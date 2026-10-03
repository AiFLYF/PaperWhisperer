"""Localisation invariants, asserted against the files that ship.

There is no browser in this repository and adding one is out of scope, so the
assertions here are of two kinds and it is worth saying which is which.

**Executed** — the detection, fallback and storage rules are *run*, not
grepped. ``i18n.js`` is a dependency-free ES module that guards every DOM and
``localStorage`` access, so it loads under bare Node. ``tests`` copy it beside
``config.js`` as ``.mjs`` and drive it through a generated harness. That is the
only way to assert honestly that an unsupported browser language falls back to
Chinese: a regex over the source would pass just as happily against a
``DEFAULT_LOCALE`` of ``'en'``, which is the exact bug this file exists to
catch.

**Structural** — the catalogue's shape, the switcher's wiring in the template
and the absence of bare literals in the migrated modules are properties of the
source text. They are read, because that is all the source can honestly support
without a DOM.

What this file cannot do is prove the interface *looks* right in either
language, that a screen reader announces the switcher correctly, or that the
arrow-key roving focus lands where it should. Those remain manual checks; they
are listed at the bottom of :mod:`docs.I18N` rather than being asserted here,
because a test that cannot fail on the defect it names is worse than no test.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JS_DIR = PROJECT_ROOT / "templates" / "static" / "js"
TEMPLATE = PROJECT_ROOT / "templates" / "index.html"

I18N = (JS_DIR / "i18n.js").read_text(encoding="utf-8")
INDEX_HTML = TEMPLATE.read_text(encoding="utf-8")

#: The modules this task migrated in full. A bare user-visible literal in any of
#: them is a miss, so the "no literals outside the catalogue" test is scoped to
#: exactly these and does not pretend to cover modules still awaiting migration.
#:
#: `claims.js`, `evidence.js` and `parseStatus.js` joined the list when the
#: evidence and claim surfaces were routed through the catalogue (WALK-DEFECT-3).
#: They were the last three renderers whose sentences were produced at render
#: time and hardcoded, and they are the surfaces where a mixed-language page is
#: worst, because they sit immediately beside a citation the reader is about to
#: trust.
#:
#: `evidenceViewer.js` joined when the *inside* of the evidence panel was
#: migrated, which closed the gap the previous change left: a panel that
#: re-rendered its headline in English while its badges, notes and paragraph
#: numbers stayed Chinese is the partially translated interface, and a reader
#: has no way to tell which half of the panel to trust. Every remaining module
#: is still unmigrated and is deliberately not listed.
#:
#: `analyze.js`, `papers.js`, `runtime.js` and `paperList.js` joined when the
#: four modules that carry a run's status, the failure surface and the paper
#: cards were migrated. They are the ones that made the mixed-language defect
#: unavoidable rather than merely likely: `analyze.js` opened a dialog that was
#: Chinese about the server's error and hardcoded English about the workspace
#: coming back, and `runtime.js` put a hardcoded '关闭' on the one control the
#: reader presses to make a failure go away.
MIGRATED_MODULES = (
    "i18n.js",
    "a11y.js",
    "theme.js",
    "mermaid.js",
    "format.js",
    "dom.js",
    "claims.js",
    "evidence.js",
    "parseStatus.js",
    "evidenceViewer.js",
    "runtime.js",
    "analyze.js",
    "papers.js",
    "paperList.js",
)

#: The catalogue keys `evidenceViewer.js` builds at run time, and therefore the
#: ones no single `t('...')` call in the source can be found by grepping.
#:
#: A key assembled from a wire value is the one kind of catalogue reference that
#: escapes the "no bare literal" and "actually calls `t()`" assertions above:
#: there is nothing to grep, so a key that stopped existing would only surface
#: as a dotted string rendered onto a citation panel.
#:
#: It is asserted against the shipped catalogue by
#: {@link test_every_key_the_viewer_builds_at_run_time_exists_in_both_languages}.
VIEWER_DYNAMIC_KEYS = (
    "evidence.view_kind.page_map",
    "evidence.view_kind.slide_view",
    "evidence.view_kind.element_view",
    "evidence.view_kind.fallback",
    "evidence.no_target.unknown_locator",
    "evidence.no_target.relative_bbox",
    "evidence.no_target.missing_bbox",
    "evidence.no_target.missing_page",
    "evidence.no_target.malformed_bbox",
    "evidence.no_target.incomplete_locator",
    "evidence.view_target_missing",
    "evidence.view_note.page_map",
    "evidence.view_note.slide",
    "evidence.view_note.element",
    "evidence.view_note.page_raster",
    "evidence.view_note.page_render_unavailable",
)


# --------------------------------------------------------------------------
# Running the real module
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def i18n_module(tmp_path_factory):
    """`i18n.js` loadable as an ES module under bare Node.

    Copied to ``.mjs`` with its one import rewritten, because the project has
    no ``package.json`` and Node would otherwise read ``.js`` as CommonJS. The
    copy is of the shipped source, so what runs is what ships.
    """
    workdir = tmp_path_factory.mktemp("i18n")
    shutil.copy(JS_DIR / "config.js", workdir / "config.mjs")
    source = (JS_DIR / "i18n.js").read_text(encoding="utf-8").replace(
        "from './config.js'", "from './config.mjs'"
    )
    (workdir / "i18n.mjs").write_text(source, encoding="utf-8")
    return workdir


def run_i18n(workdir: Path, body: str) -> dict:
    """Execute `body` against the real `i18n.js` and return its JSON result.

    `body` receives the module namespace as `m` and must leave a JSON-
    serialisable value in `result`. Failures surface as a non-zero exit with the
    error on stderr, so a harness that throws fails the test rather than
    returning `None` and being asserted on as if it were a result.
    """
    harness = workdir / "harness.mjs"
    harness.write_text(
        "import * as m from './i18n.mjs';\n"
        # Node ships a getter-only `navigator` global, so the browser-detection
        # tests redefine it rather than assigning to it. `defineProperty` is
        # what a browser harness has to do for the same reason.
        "globalThis.setNavigator = (v) => { Object.defineProperty(globalThis, 'navigator', "
        "{ value: v, configurable: true, writable: true }); };\n"
        "globalThis.setStorage = (v) => { Object.defineProperty(globalThis, 'localStorage', "
        "{ value: v, configurable: true, writable: true }); };\n"
        "let result = null;\n"
        "try {\n"
        f"{body}\n"
        "} catch (error) {\n"
        "    result = { __threw: String(error && error.message) };\n"
        "}\n"
        "process.stdout.write(JSON.stringify(result === undefined ? null : result));\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, timeout=60, cwd=workdir
    )
    assert completed.returncode == 0, f"harness failed: {completed.stderr}"
    return json.loads(completed.stdout)


# --------------------------------------------------------------------------
# The default
# --------------------------------------------------------------------------


def test_chinese_is_the_default_locale(i18n_module):
    """The last-resort fallback is `zh-CN`, not English.

    A fresh install with no stored preference and a browser reporting something
    the catalogue does not carry must start in Chinese. This is asserted
    against the *executed* constant, not its source text, because a source
    assertion would also pass against a default that the detection chain then
    never actually returns.
    """
    assert run_i18n(i18n_module, "result = m.DEFAULT_LOCALE;") == "zh-CN"
    assert "export const DEFAULT_LOCALE = 'zh-CN'" in I18N


def test_the_default_is_listed_first_because_it_is_not_a_preference(i18n_module):
    """`SUPPORTED_LOCALES[0]` is `DEFAULT_LOCALE`.

    Nothing iterates the list to pick a language today, so this is not load
    bearing yet. It is asserted so that adding a language to the front of the
    array — the easy way to "make Japanese the default" — fails here rather than
    being discovered by a reader in a different locale.
    """
    result = run_i18n(
        i18n_module, "result = { supported: m.SUPPORTED_LOCALES, first: m.SUPPORTED_LOCALES[0] };"
    )
    assert result["first"] == result["supported"][0]
    assert "zh-CN" in result["supported"]
    assert "en" in result["supported"], "English must stay a first-class supported mode"


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag",
    ["zh", "zh-CN", "zh-cn", "zh_CN", "zh-Hans", "zh-Hans-CN", "zh-SG", "ZH"],
)
def test_chinese_tags_resolve_to_simplified_chinese(i18n_module, tag):
    """Every Chinese tag lands on `zh-CN`, including the traditional ones.

    The product ships simplified Chinese only, so `zh-TW` / `zh-Hant` have
    nothing else to resolve to. Resolving them to the locale we do have is a
    deliberate decision, and the alternative — returning `null` and landing on
    whatever the default happens to be — would make the reader's own stated
    preference a no-op.
    """
    assert run_i18n(i18n_module, f"result = m.normalizeLocale({tag!r});") == "zh-CN"


@pytest.mark.parametrize("tag", ["zh-TW", "zh-HK", "zh-MO", "zh-Hant", "zh-Hant-HK", "zh-Hant-TW"])
def test_traditional_chinese_is_handled_deliberately_not_by_accident(i18n_module, tag):
    """A traditional-script preference resolves, and resolves to simplified.

    Asserted separately from the generic Chinese cases because this is the case
    the specification calls out as needing to be handled on purpose. The
    assertion is that it produces `zh-CN` and *not* `null`: an unresolvable tag
    would silently fall through to detection and could land the reader in a
    language they did not ask for.
    """
    resolved = run_i18n(i18n_module, f"result = m.normalizeLocale({tag!r});")
    assert resolved == "zh-CN", (
        "a traditional-script tag must resolve to the one Chinese this build "
        "ships, not fall through to the default"
    )


@pytest.mark.parametrize("tag", ["en", "en-US", "en-GB", "en-AU", "en_GB", "EN"])
def test_english_tags_resolve_to_english(i18n_module, tag):
    assert run_i18n(i18n_module, f"result = m.normalizeLocale({tag!r});") == "en"


@pytest.mark.parametrize("tag", ["fr", "ja", "de", "ko", "ru", "pt-BR", "yue"])
def test_an_unsupported_tag_resolves_to_nothing_rather_than_guessing(i18n_module, tag):
    """`null`, not a guess.

    Detection must be able to *say* "I do not know this language", so that
    `resolveInitialLocale` can apply the default deliberately. A function that
    returned some language for every input would make the unsupported-language
    branch of the fallback chain unreachable, and the default would then only
    ever apply to a missing preference — which is the bug the default exists to
    cover.
    """
    assert run_i18n(i18n_module, f"result = m.normalizeLocale({tag!r});") is None


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_missing_or_empty_preference_resolves_to_nothing(i18n_module, value):
    assert run_i18n(i18n_module, f"result = m.normalizeLocale({json.dumps(value)});") is None


@pytest.mark.parametrize("value", [123, True, [], {}, 0])
def test_a_non_string_preference_resolves_to_nothing_and_does_not_throw(i18n_module, value):
    """A malformed stored value is discarded, not obeyed and not thrown.

    "A preference stored wrongly is worse than no preference" is a
    correctness requirement, not a nicety: `localStorage` holds whatever a
    previous build, a hand edit or a half-written write left there.

    A single-element array is deliberately *not* in this list. `String(['en'])`
    is `'en'` in JavaScript, so such a value would resolve — but it cannot
    reach the function in practice, because `getItem` only ever yields a
    string or `null`. Asserting on JavaScript's own array coercion would test
    the language, not this module.
    """
    assert run_i18n(i18n_module, f"result = m.normalizeLocale({json.dumps(value)});") is None


def test_detection_returns_null_when_the_browser_says_nothing(i18n_module):
    """No `navigator` at all yields `null`.

    Node ships a `navigator` global that reports `en-US`, so the "there is no
    navigator" branch has to be provoked explicitly rather than relied on. This
    is the condition a sandboxed frame or an embedded webview produces, and it
    is the branch that makes the default reachable.
    """
    body = "setNavigator(undefined); result = m.detectLocale();"
    assert run_i18n(i18n_module, body) is None


def test_detection_returns_null_when_the_browser_reports_only_unsupported_languages(
    i18n_module,
):
    """A browser that reports nothing this build carries falls through to the default.

    This is the composed assertion: `detectLocale()` returns `null` for a
    French-only browser, and `resolveInitialLocale()` therefore returns the
    Chinese default. Asserting only the first half would pass against a build
    whose fallback chain had been wired to `en` instead.
    """
    body = (
        "setNavigator({ languages: ['fr-FR', 'fr'], language: 'fr-FR' });\n"
        "result = { detected: m.detectLocale(), resolved: m.resolveInitialLocale() };"
    )
    assert run_i18n(i18n_module, body) == {"detected": None, "resolved": "zh-CN"}


def test_an_english_browser_starts_in_english(i18n_module):
    """Detection still works: `en` moves the *starting* language, once.

    Asserted together with the fact that a stored choice is consulted first, so
    the detection rule cannot be read as overriding an explicit choice.
    """
    body = (
        "setNavigator({ languages: ['en-GB', 'en'], language: 'en-GB' });\n"
        "result = { detected: m.detectLocale(), resolved: m.resolveInitialLocale() };"
    )
    assert run_i18n(i18n_module, body) == {"detected": "en", "resolved": "en"}


def test_a_chinese_browser_is_never_moved_to_english(i18n_module):
    """The specific failure the specification forbids.

    A browser reporting Chinese must land on Chinese. If detection ever grew a
    rule that preferred English, this is the assertion that stops it.
    """
    body = (
        "setNavigator({ languages: ['zh-Hant-TW', 'en-US'], language: 'zh-Hant-TW' });\n"
        "result = m.resolveInitialLocale();"
    )
    assert run_i18n(i18n_module, body) == "zh-CN"


def test_a_stored_choice_beats_the_browser_and_survives_a_reload(i18n_module):
    """The explicit choice is consulted before detection.

    Set `en` while the browser asks for Chinese, then resolve again as a fresh
    load would. If storage were consulted after detection, the second call would
    return `zh-CN` and the reader's choice would silently revert on every reload
    — which is the failure this ordering prevents.
    """
    body = """
setStorage({
    _d: { 'paperwhisperer-locale': 'en' },
    getItem(k) { return k in this._d ? this._d[k] : null; },
    setItem(k, v) { this._d[k] = String(v); },
    removeItem(k) { delete this._d[k]; }
});
setNavigator({ languages: ['zh-CN', 'zh'], language: 'zh-CN' });
const first = m.resolveInitialLocale();
m.setLocale('en');
const afterChoice = m.resolveInitialLocale();
result = { first, afterChoice };
"""
    assert run_i18n(i18n_module, body) == {"first": "en", "afterChoice": "en"}


def test_a_wrongly_stored_preference_is_discarded_rather_than_obeyed(i18n_module):
    """A stored value this build does not ship must not select a language.

    Storing `'klingon'` or a stale `'zh-TW'`-era tag must leave the reader on the
    default, not on a locale with no catalogue and no strings.
    """
    body = """
setStorage({
    _d: { 'paperwhisperer-locale': 'klingon' },
    getItem(k) { return k in this._d ? this._d[k] : null; },
    setItem(k, v) { this._d[k] = String(v); },
    removeItem(k) { delete this._d[k]; }
});
setNavigator({ languages: ['en-US'], language: 'en-US' });
result = m.resolveInitialLocale();
"""
    assert run_i18n(i18n_module, body) == "en", (
        "a stored value naming no shipped locale must be ignored; the browser "
        "preference may then be used, but never the bad value itself"
    )


# --------------------------------------------------------------------------
# Private mode / disabled storage
# --------------------------------------------------------------------------


def test_a_storage_read_that_throws_degrades_instead_of_taking_the_page_down(i18n_module):
    """A `localStorage` that throws on every access must not throw out of the module.

    This is the real private-browsing and disabled-storage condition: in both,
    touching `localStorage` throws rather than returning `null`. The test
    installs exactly that — an object whose `getItem` *and* `setItem` throw — and
    asserts that the whole resolution chain still returns a usable locale. A bare
    `localStorage.getItem` would throw out of `resolveInitialLocale` and take the
    entry module's initialisation with it.
    """
    body = """
setStorage({
    getItem() { throw new DOMException('SecurityError'); },
    setItem() { throw new DOMException('QuotaExceededError'); },
    removeItem() { throw new DOMException('SecurityError'); }
});
setNavigator({ languages: ['en-US'], language: 'en-US' });
let threw = null;
let resolved = null;
let switchOk = null;
try {
    resolved = m.resolveInitialLocale();
    switchOk = m.setLocale('en');
} catch (error) {
    threw = String(error && error.message);
}
result = { threw, resolved, switchOk };
"""
    assert run_i18n(i18n_module, body) == {
        "threw": None,
        "resolved": "en",
        "switchOk": True,
    }


def test_storage_that_throws_still_leaves_the_default_reachable(i18n_module):
    """With storage broken *and* no usable `navigator`, the default still comes back.

    The compound failure is the one that matters: a hardened browser with no
    persistent storage and no `navigator` must still get a language.
    """
    body = """
setStorage({
    getItem() { throw new Error('nope'); },
    setItem() { throw new Error('nope'); },
    removeItem() { throw new Error('nope'); }
});
setNavigator(undefined);
let threw = null;
let resolved = null;
try { resolved = m.resolveInitialLocale(); } catch (e) { threw = String(e); }
result = { threw, resolved };
"""
    assert run_i18n(i18n_module, body) == {"threw": None, "resolved": "zh-CN"}


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------


def catalogue_table(name: str) -> dict[str, str]:
    """Extract one locale table from the source as `{key: value}`.

    A source read rather than an executed import, because this test is about
    *parity*: it must compare the two tables as data even if one of them could
    not be evaluated, which is exactly the failure it is looking for.
    """
    match = re.search(rf"const {name} = \{{(.*?)\n\}};", I18N, re.DOTALL)
    assert match, f"the {name} table is missing from i18n.js"
    return dict(re.findall(r"^    '([^']+)': (.*?),?$", match.group(1), re.MULTILINE))


def test_every_catalogue_key_exists_in_both_languages():
    """Key parity between `zh-CN` and `en`.

    A key present in one locale and missing from the other is a half-finished
    translation: in the default locale it renders, and in the other it silently
    falls back, so the interface looks complete in whichever language the
    reviewer happened to be using.
    """
    zh = catalogue_table("ZH_CN")
    en = catalogue_table("EN")
    assert zh, "the zh-CN table is empty"
    assert en, "the en table is empty"
    assert not set(en) - set(zh), f"missing from zh-CN: {sorted(set(en) - set(zh))}"
    assert not set(zh) - set(en), f"missing from en: {sorted(set(zh) - set(en))}"


def test_no_catalogue_key_is_defined_twice():
    """A duplicated key is silently the last one.

    `zh-CN` and `en` are written as object literals, so a repeated key compiles,
    parses and passes every other test in this file while one of the two
    translations is unreachable. Only a source-level duplicate check sees it.
    """
    for name in ("ZH_CN", "EN"):
        match = re.search(rf"const {name} = \{{(.*?)\n\}};", I18N, re.DOTALL)
        assert match
        keys = re.findall(r"^    '([^']+)':", match.group(1), re.MULTILINE)
        duplicates = {key for key in keys if keys.count(key) > 1}
        assert not duplicates, f"{name} defines {sorted(duplicates)} more than once"


def test_no_catalogue_value_is_empty():
    """An empty string is a translation that was started and not finished.

    It renders as nothing, which is worse than an untranslated key: a key
    renders as its own dotted name, which is visible and greppable.
    """
    for name in ("ZH_CN", "EN"):
        for key, value in catalogue_table(name).items():
            assert value not in ("''", '""'), f"{name}: {key} is empty"


def test_every_html_key_resolves_in_both_languages(i18n_module):
    """No `data-i18n` in the template points at a key that does not exist.

    A dangling attribute is invisible in review and produces a visible dotted
    key in the running interface, which is the failure mode the `t()` fallback
    is designed to make *legible* — but legible is not the same as correct.
    """
    used = set(re.findall(r'data-i18n="([^"]+)"', INDEX_HTML))
    for attribute_value in re.findall(r'data-i18n-attr="([^"]+)"', INDEX_HTML):
        for pair in attribute_value.split(";"):
            parts = pair.split(":")
            if len(parts) == 2 and parts[1].strip():
                used.add(parts[1].strip())
    assert used, "the template declares no data-i18n attributes at all"

    zh = set(catalogue_table("ZH_CN"))
    en = set(catalogue_table("EN"))
    assert not used - zh, f"keys used in index.html but absent from zh-CN: {sorted(used - zh)}"
    assert not used - en, f"keys used in index.html but absent from en: {sorted(used - en)}"


def test_lookup_falls_back_to_the_key_so_a_miss_is_visible(i18n_module):
    """An unknown key returns itself, not an empty string.

    Asserted because it is the property that makes a catalogue miss findable in
    review rather than indistinguishable from an intentionally blank control.
    """
    body = "m.setLocale('en'); result = { missing: m.t('no.such.key'), present: m.t('status.waiting') };"
    result = run_i18n(i18n_module, body)
    assert result["missing"] == "no.such.key"
    assert result["present"] == "Waiting for document"


def test_the_same_key_differs_between_the_two_languages(i18n_module):
    """Spot-check that switching actually changes the rendered text.

    Without this, a catalogue in which every value were the same string in both
    locales would satisfy parity, non-emptiness and resolution — and be wrong.
    """
    body = """
const read = () => { m.setLocale('zh-CN'); const zh = m.t('status.waiting');
                     m.setLocale('en'); const en = m.t('status.waiting');
                     return { zh, en }; };
result = read();
"""
    result = run_i18n(i18n_module, body)
    assert result["zh"] == "等待文档"
    assert result["en"] == "Waiting for document"
    assert result["zh"] != result["en"]


def test_the_chinese_default_renders_chinese_before_any_choice_is_made(i18n_module):
    """The composed default path, end to end.

    No storage, no `navigator` — the fresh-install condition — and the lookup
    that the first paint depends on must already be Chinese. This is the single
    assertion that would catch an inverted default, which is why it is stated
    against the executed module rather than against the constant.
    """
    body = """
setNavigator(undefined);
setStorage(undefined);
m.setLocale(m.resolveInitialLocale());
result = { waiting: m.t('status.waiting'), analyze: m.t('upload.analyze') };
"""
    result = run_i18n(i18n_module, body)
    assert result["waiting"] == "等待文档"
    assert result["analyze"] == "分析文档"


# --------------------------------------------------------------------------
# Errors: driven by code, never by the server's prose
# --------------------------------------------------------------------------


def error_copy_table(locale: str) -> dict[str, str]:
    """`{code: raw tuple literal}` for one locale of the error copy table."""
    body = re.search(r"const ERROR_COPY = \{(.*?)\n\};", I18N, re.DOTALL)
    assert body, "the error copy table is missing from i18n.js"
    # Split into the `'zh-CN': { ... },` and `en: { ... }` halves. The two are
    # not symmetrically quoted in the source, so locate each by its own opening
    # marker and take everything up to the next locale key.
    open_match = re.search(
        rf"\n    (?:'{re.escape(locale)}'|{re.escape(locale)}): \{{(.*?)(?=\n    (?:'zh-CN'|en): \{{|\Z)",
        body.group(1),
        re.DOTALL,
    )
    assert open_match, f"no {locale} section in the error copy table"
    return dict(
        re.findall(r"^        ([a-z_]+): (\[.*?\]),?$", open_match.group(1), re.MULTILINE)
    )


def test_every_error_code_has_copy_in_both_languages():
    """Error copy parity.

    A code with Chinese copy and no English copy is the mixed-language defect
    one level down from the interface: the notice is a success in the default
    locale and an English sentence with a Chinese gap in the other.
    """
    zh = error_copy_table("zh-CN")
    en = error_copy_table("en")
    assert zh, "the zh-CN error copy table is empty"
    assert en, "the en error copy table is empty"
    assert set(zh) == set(en), (
        f"error codes missing from en: {sorted(set(zh) - set(en))}; "
        f"missing from zh-CN: {sorted(set(en) - set(zh))}"
    )


def test_every_code_the_runtime_knows_has_localised_copy():
    """`ERROR_PRESENTATIONS` (the policy table) and the copy table agree on codes.

    `config.js` decides *whether retrying helps*; `i18n.js` decides *what is
    said*. A code present in one and not the other is either a failure with no
    sentence or a sentence attached to a failure whose retry policy is unknown.
    The first is a hole; the second is worse, because it implies the runtime
    knows more about the failure than it does.
    """
    config = (JS_DIR / "config.js").read_text(encoding="utf-8")
    table = re.search(r"export const ERROR_PRESENTATIONS = \{(.*?)\n\};", config, re.DOTALL)
    assert table, "ERROR_PRESENTATIONS is missing from config.js"
    policy_codes = set(re.findall(r"^    ([a-z_]+):", table.group(1), re.MULTILINE))
    copy_codes = set(error_copy_table("en"))

    assert policy_codes, "no error codes were parsed out of the policy table"
    assert not policy_codes - copy_codes, (
        "codes the runtime will present with no localised sentence: "
        f"{sorted(policy_codes - copy_codes)}"
    )


def test_error_presentation_is_keyed_on_code_not_on_the_servers_prose(i18n_module):
    """English copy under an English interface, despite Chinese server prose.

    The defect this exists to prevent: an English interface that surfaces the
    backend's Chinese `error` sentence. The assertion is that the localised
    English sentence is returned as the *message*, and that the server's own
    prose is confined to a separate `serverDetail` field the caller may show
    beside it or drop.
    """
    body = """
m.setLocale('en');
const p = m.errorPresentation('storage_unavailable', '服务端的数据存储现在无法访问。');
result = { title: p.title, message: p.message, serverDetail: p.serverDetail, known: p.known };
"""
    result = run_i18n(i18n_module, body)
    assert result["message"] == (
        "The server’s data storage cannot be reached. This request was not "
        "processed; retrying later is enough."
    )
    assert "重试" not in result["message"]
    assert "storage" in result["message"].lower()
    assert result["serverDetail"] == "服务端的数据存储现在无法访问。"
    assert result["known"] is True


def test_the_same_code_reads_in_chinese_for_a_chinese_reader(i18n_module):
    """The same code, the default locale, gives the Chinese sentence.

    Pairing this with the English assertion is what makes the result a
    *localised* error rather than an error with an English alternate attached.
    """
    body = """
m.setLocale('zh-CN');
const p = m.errorPresentation('storage_unavailable', 'anything at all');
result = { title: p.title, message: p.message };
"""
    result = run_i18n(i18n_module, body)
    assert result["message"] == (
        "服务端的数据存储现在无法访问。这次请求没有被处理，稍后重试即可。"
    )


def test_an_unrecognised_code_says_so_rather_than_guessing_a_sentence(i18n_module):
    """An unknown code yields the "unknown code" copy, not the server's prose.

    A finding to surface, not to paper over: inventing a sentence for a code
    this build has never seen would be a guess about what went wrong. Saying
    "this version does not know this code" is true, is actionable, and is in
    the reader's language.
    """
    body = """
m.setLocale('en');
const p = m.errorPresentation('some_code_from_a_newer_server', '某个新错误。');
result = { known: p.known, title: p.title, message: p.message, serverDetail: p.serverDetail };
"""
    result = run_i18n(i18n_module, body)
    assert result["known"] is False
    assert "does not know" in result["title"]
    assert "某个新错误" in result["serverDetail"]
    assert "某个新错误" not in result["message"]


def test_an_empty_code_does_not_crash_the_presentation(i18n_module):
    """A failure with no `code` at all still produces a usable object."""
    body = "m.setLocale('en'); const p = m.errorPresentation('', 'boom'); result = { code: p.code, known: p.known };"
    result = run_i18n(i18n_module, body)
    assert result["code"] == "client_unknown_error"
    assert result["known"] is False


# --------------------------------------------------------------------------
# The interface language is not the document's language
# --------------------------------------------------------------------------


def test_the_interface_locale_does_not_decide_the_document_language(i18n_module):
    """A Chinese paper stays Chinese-labelled when the interface is English.

    The interface language and the document's language are separate axes. This
    is the assertion that they are, in both directions and with no coupling: a
    reader who switches to English is still reading a Chinese paper, and
    labelling that paper's content English would make a screen reader pronounce
    it with the wrong phonetics.
    """
    body = """
m.setLocale('en');
const chinesePaper = '本文提出了一种新的检索增强生成方法，并在大规模语料上验证了它的有效性。';
result = {
    ui: m.getLocale(),
    doc: m.documentLocale(chinesePaper)
};
"""
    result = run_i18n(i18n_module, body)
    assert result["ui"] == "en"
    assert result["doc"] == "zh-CN"


def test_an_english_paper_stays_english_under_a_chinese_interface(i18n_module):
    """The mirror image: switching the UI to Chinese must not relabel English prose."""
    body = """
m.setLocale('zh-CN');
result = m.documentLocale('This paper proposes a new retrieval-augmented generation method.');
"""
    assert run_i18n(i18n_module, body) == "en"


def test_a_mixed_paper_is_decided_by_its_own_text(i18n_module):
    """A mixed document is classified from the content, at a ratio threshold.

    A handful of ideographs in an English abstract — a cited paper title, an
    author name in Chinese — must not flip a document's language. Asserted
    because it is the failure mode a naive "contains CJK" check produces, and
    the corpus is explicitly Chinese-and-English papers.
    """
    body = """
result = {
    mostlyEnglish: m.documentLocale('We propose RAG, a 检索增强生成 approach evaluated on three benchmarks.'),
    mostlyChinese: m.documentLocale('本文提出了一种检索增强生成方法，并在三个基准数据集上进行了充分评估与验证。'),
    empty: m.documentLocale('   ')
};
"""
    result = run_i18n(i18n_module, body)
    assert result["mostlyEnglish"] == "en"
    assert result["mostlyChinese"] == "zh-CN"
    assert result["empty"] is None


# --------------------------------------------------------------------------
# The walkthrough's scenario, executed
# --------------------------------------------------------------------------
#
# WALK-DEFECT-3, as a test rather than as a description. The walkthrough found
# that switching language repainted the parse-status card but not the answer
# claims or the evidence panel: an English reader saw Chinese claim prefixes and
# Chinese "no visual targeting" reasons beside an English interface.
#
# There is still no browser, so the DOM is a stub — but a *behavioural* one this
# time, not a source read: the real `claims.js`, `evidence.js` and `i18n.js`
# modules are imported and run, a claim and an evidence panel are rendered in
# Chinese, the locale is switched, and the surfaces are re-rendered. Every
# string the reader can see is then read back out of the stub and compared with
# the catalogue. This is what makes it a regression test rather than a grep: a
# future change that puts a literal back into `evidence.js` fails here, and a
# change that makes a surface stop repainting fails here too.

#: The minimal element the modules under test actually use. Not a DOM
#: implementation — a bag of the accessors these three modules touch.
DOM_SHIM = """
class El {
    constructor(tag) {
        this.tagName = String(tag).toUpperCase();
        this.children = [];
        this.attributes = {};
        this.dataset = {};
        // `evidenceViewer.js` positions its rectangles by writing `style.left`
        // and friends, so the shim has to carry an object to write them on.
        this.style = {};
        this.hidden = false;
        this._text = null;
        this.isConnected = true;
        this.title = '';
    }
    get className() { return this.attributes.class || ''; }
    set className(v) { this.attributes.class = String(v); }
    setAttribute(k, v) {
        this.attributes[k] = String(v);
        if (k.startsWith('data-')) {
            this.dataset[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = String(v);
        }
    }
    getAttribute(k) { return k in this.attributes ? this.attributes[k] : null; }
    removeAttribute(k) { delete this.attributes[k]; }
    appendChild(child) { this.children.push(child); return child; }
    append(...nodes) { nodes.forEach(node => this.appendChild(node)); }
    replaceChildren(...nodes) { this.children = []; nodes.forEach(node => this.appendChild(node)); }
    set textContent(value) { this.children = []; this._text = String(value); }
    get textContent() {
        // The shim keeps the assigned text *and* later-appended children, which
        // is what a real node does: setting `textContent` clears the children,
        // and appending after that adds to it. Summing them is what lets
        // `buildHighlightStatus` — which writes its sentence and then appends
        // the reason code — be read back as one string.
        const own = this._text === null ? '' : this._text;
        return own + this.children.map(child => child.textContent).join('');
    }
    addEventListener() {}
    focus() {}
    closest() { return null; }
    querySelectorAll() { return []; }
    contains(node) { return node === this || this.children.indexOf(node) >= 0; }
}
const byId = new Map();
[
    'evidencePanel','evidenceIdle','evidenceLoading','evidenceError','evidenceNoTarget',
    'evidenceEmpty','evidenceContent','evidenceWarnings','evidenceTarget','evidenceViewer',
    'evidenceSnippetMeta','evidenceSnippet','evidenceTable','evidenceCloseBtn',
    'evidenceBackdrop','evidencePanelHeading','claimsHost','chipsHost'
].forEach(id => byId.set(id, new El('div')));
globalThis.document = {
    documentElement: new El('html'),
    body: new El('body'),
    createElement: tag => new El(tag),
    createTextNode: text => ({ textContent: String(text), children: [] }),
    createDocumentFragment: () => new El('#fragment'),
    getElementById: id => byId.get(id) || null,
    querySelectorAll: () => [],
    querySelector: () => null,
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => true,
    contains: () => true,
    activeElement: null
};
globalThis.HTMLElement = El;
globalThis.window = {
    matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
    scrollY: 0, scrollTo: () => {},
    addEventListener: () => {}, removeEventListener: () => {},
    getComputedStyle: () => ({ getPropertyValue: () => '' }),
    location: { hash: '', search: '' },
    setTimeout, clearTimeout
};
globalThis.getComputedStyle = globalThis.window.getComputedStyle;
globalThis.__el = (tag) => new El(tag);
// Every element carrying a class, read back out by that class. `evidenceViewer.js`
// says which sentence belongs to which surface by which element it puts it in,
// so the surface has to be selectable the same way a reader could see it.
globalThis.__byClass = (node, className) => {
    const out = [];
    const walk = current => {
        const own = String((current.attributes || {}).class || '').split(/\\s+/);
        if (own.indexOf(className) >= 0) out.push(current);
        (current.children || []).forEach(walk);
    };
    walk(node);
    return out.map(current => current.textContent);
};
// `title` is text a reader is shown on hover, and the truncated-preview marker
// lives in nothing else.
globalThis.__titles = (node, className) => {
    const out = [];
    const walk = current => {
        const own = String((current.attributes || {}).class || '').split(/\\s+/);
        if (own.indexOf(className) >= 0) out.push(String(current.title || ''));
        (current.children || []).forEach(walk);
    };
    walk(node);
    return out;
};
globalThis.__texts = (node) => {
    const out = [];
    const walk = current => {
        if (current.textContent && current.children.length === 0 && current.textContent.trim()) {
            out.push(current.textContent);
        }
        (current.children || []).forEach(walk);
    };
    walk(node);
    return out;
};
// Accessible names, which are text a screen reader announces and which the
// visible-text walk above cannot see. A verdict that lives only in an
// aria-label is still text the reader is shown.
globalThis.__ariaLabels = (node) => {
    const out = [];
    const walk = current => {
        const label = (current.attributes || {})['aria-label'];
        if (label) out.push(label);
        (current.children || []).forEach(walk);
    };
    walk(node);
    return out;
};
"""


def run_walkthrough_harness(workdir: Path, body: str) -> dict:
    """Execute `body` against the real claim/evidence/i18n modules under a DOM stub.

    The modules are imported from the shipped files by absolute path rather than
    lifted out with a regex, because the defect under test is a *join*: the
    catalogue has to be consulted by the render path, and the render path has to
    be reached again on a locale change. Extracting one function would test a
    version of the code that does not ship.
    """
    (workdir / "dom_shim.mjs").write_text(DOM_SHIM, encoding="utf-8")
    harness = workdir / "walkthrough_harness.mjs"
    js_dir = JS_DIR
    harness.write_text(
        "import './dom_shim.mjs';\n"
        "const ev = await import(%r);\n"
        "const claims = await import(%r);\n"
        "const i18n = await import(%r);\n"
        "const state = (await import(%r)).state;\n"
        "let result = null;\n"
        "try {\n"
        f"{body}\n"
        "} catch (error) {\n"
        "    result = { __threw: String(error && error.message) };\n"
        "}\n"
        "process.stdout.write(JSON.stringify(result === undefined ? null : result));\n"
        % (
            str(js_dir / "evidence.js"),
            str(js_dir / "claims.js"),
            str(js_dir / "i18n.js"),
            str(js_dir / "state.js"),
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, timeout=60, cwd=workdir
    )
    assert completed.returncode == 0, f"harness failed: {completed.stderr}"
    return json.loads(completed.stdout)


def test_the_walkthrough_scenario_switching_language_repaints_claims_and_evidence(
    i18n_module,
):
    """WALK-DEFECT-3 as a test.

    The walkthrough's smallest reproduction, executed: render a claim and an
    evidence panel in Chinese, switch to English, let the modules re-render, and
    assert that *every* string the reader can see is now the English one. Before
    the fix this fails — the claim prefix, the evidence page label, the snippet
    meta and the capability line all stayed Chinese while the interface around
    them switched — and it fails again if any of those strings is moved back out
    of the catalogue, or if a surface stops re-rendering on the locale event.
    """
    body = textwrap.dedent(
        """
        i18n.setLocale('zh-CN');
        const doc = globalThis.document;

        // A verified direct quote and a degraded paraphrase, and an evidence
        // payload that exercises the page label, the capability line and the
        // snippet meta. All four literals named in the walkthrough are here.
        const verified = {
            text: 'Transformer 在本数据集上提升了 4.2 个百分点。',
            quote: 'Transformer improves by 4.2 points',
            evidence_ids: ['ev_1'],
            presentation: 'direct_quote',
            mode: 'normalized',
            message: '',
            citation_label: 'p3',
            locator: { kind: 'page', page: 3 }
        };
        const paraphrase = {
            text: '该方法可能适用于更大的模型。',
            quote: 'the method may scale to larger models',
            evidence_ids: ['ev_2'],
            presentation: 'paraphrase',
            message: '与原文不一致',
            citation_label: 'p4'
        };
        const payload = {
            citation_label: 'p3',
            snippet: {
                text: 'Transformer improves by 4.2 points on this dataset.',
                source: 'chunk_verbatim_text',
                truncated: true
            },
            visual_targeting: {
                available: false,
                reason_code: 'relative_bbox',
                target: { kind: 'pdf_page', page_label: '3' },
                capability: 'pdf_page_highlight'
            }
        };

        function renderEverything() {
            ev.renderInlineClaims(doc.getElementById('claimsHost'), [verified, paraphrase]);
            ev.renderCitationChips(doc.getElementById('chipsHost'), [verified, paraphrase]);
            state.evidencePanelOpen = true;
            state.currentEvidence = payload;
            ev.renderEvidencePayload(payload);
        }

        // The direct-quote and paraphrase renderers each build [prefix, body,
        // ...], so children[0] is the translated prefix and children[1] is the
        // claim's own words — which are the document's and must NOT translate.
        const claimsHost = doc.getElementById('claimsHost');
        const readClaims = () => claimsHost.children.map(element => ({
            prefix: element.children[0].textContent,
            body: element.children[1].textContent
        }));

        renderEverything();
        const zhBefore = {
            claims: readClaims(),
            target: doc.getElementById('evidenceTarget').textContent,
            meta: doc.getElementById('evidenceSnippetMeta').textContent,
            chips: __texts(doc.getElementById('chipsHost')),
            chipNames: __ariaLabels(doc.getElementById('chipsHost'))
        };

        // The reader flips the switch. `setLocale` is what the switcher calls;
        // it persists, then notifies every subscriber.
        i18n.setLocale('en');
        // And the module repaints, exactly as its onLocaleChange subscription
        // does. Called explicitly rather than relying on the notification,
        // because the stub has no event-target wiring; the subscription itself
        // is asserted separately below.
        ev.repaintLocalizedSurfaces();
        renderEverything(); // idempotent: a second repaint must not double anything

        const zhAfter = {
            claims: readClaims(),
            target: doc.getElementById('evidenceTarget').textContent,
            meta: doc.getElementById('evidenceSnippetMeta').textContent,
            chips: __texts(doc.getElementById('chipsHost')),
            chipNames: __ariaLabels(doc.getElementById('chipsHost'))
        };

        result = { zhBefore, zhAfter };
        """
    )
    result = run_walkthrough_harness(i18n_module, body)
    assert "__threw" not in result, result

    # The claim *prefix* is the interface's sentence and must have switched; the
    # claim *body* is the document's words and must not have.
    before_prefixes = [entry["prefix"] for entry in result["zhBefore"]["claims"]]
    after_prefixes = [entry["prefix"] for entry in result["zhAfter"]["claims"]]
    before_bodies = [entry["body"] for entry in result["zhBefore"]["claims"]]
    after_bodies = [entry["body"] for entry in result["zhAfter"]["claims"]]

    # Fixture sanity, asserted rather than assumed: the run really did start in
    # Chinese, so "it is now English" is a statement about the switch and not
    # about a payload that happened to be English to begin with.
    assert all(re.search(r"[一-鿿]", prefix) for prefix in before_prefixes), (
        f"the fixture should have rendered Chinese prefixes first, got {before_prefixes}"
    )
    assert all(not re.search(r"[一-鿿]", prefix) for prefix in after_prefixes), (
        f"a claim prefix is still Chinese after a switch to English: {after_prefixes}"
    )
    assert after_prefixes == [
        "Aligned verbatim after normalisation:",
        "Paraphrase (not verified verbatim against the source):",
    ], f"the claim prefixes did not become the English ones: {after_prefixes}"
    # The claim text itself — the document's words — did not move.
    assert after_bodies == before_bodies

    for key in ("target", "meta"):
        assert not re.search(r"[一-鿿]", result["zhAfter"][key]), (
            f"the {key} surface still contains Chinese after a switch to English: "
            f"{result['zhAfter'][key]!r}"
        )
    assert result["zhAfter"]["target"] != result["zhBefore"]["target"], (
        "the evidence target summary did not change with the locale at all"
    )
    assert "Page 3" in result["zhAfter"]["target"]
    assert "Can highlight on the page" in result["zhAfter"]["target"]
    assert "Citation p3" in result["zhAfter"]["meta"]
    assert "From the verified source text" in result["zhAfter"]["meta"]
    assert "truncated" in result["zhAfter"]["meta"]
    # The chip's own label and accessible name follow too. The accessible name is
    # where the verdict lives, so it is checked as string text, not as a
    # colour or a class.
    assert all(not re.search(r"[一-鿿]", chip) for chip in result["zhAfter"]["chips"]), (
        f"a citation chip is still Chinese: {result['zhAfter']['chips']}"
    )
    assert any("Source-verified direct quote" in name for name in result["zhAfter"]["chipNames"]), (
        f"the chip's accessible name lost the verdict: {result['zhAfter']['chipNames']}"
    )
    assert all(
        not re.search(r"[一-鿿]", name) for name in result["zhAfter"]["chipNames"]
    ), f"a chip's accessible name is still Chinese: {result['zhAfter']['chipNames']}"


def test_switching_language_does_not_relabel_a_chinese_papers_headings(i18n_module):
    """The document language is a separate axis and stays where it is.

    The interface language says nothing about the document's, and the fix for
    WALK-DEFECT-3 must not overreach into relabelling someone's paper. So the
    claim's own text and the evidence passage — which are the *document's* words
    — are asserted to be byte-identical before and after the switch, while the
    surrounding interface changed language.
    """
    body = textwrap.dedent(
        """
        i18n.setLocale('zh-CN');
        const doc = globalThis.document;
        const claim = {
            text: '本文提出了一种新的检索增强生成方法，并在大规模语料上验证了它的有效性。',
            quote: '我们提出了一种新的检索增强生成方法',
            presentation: 'direct_quote',
            mode: 'verbatim',
            citation_label: 'p1'
        };
        const payload = {
            citation_label: 'p1',
            snippet: { text: '第 3 节  实验结果', source: 'chunk_verbatim_text', truncated: false },
            visual_targeting: {
                available: true,
                target: { kind: 'pdf_page', page_label: '7' },
                capability: 'pdf_page_highlight'
            }
        };
        ev.renderInlineClaims(doc.getElementById('claimsHost'), [claim]);
        state.evidencePanelOpen = true;
        state.currentEvidence = payload;
        ev.renderEvidencePayload(payload);
        // The direct-quote renderer builds [prefix, body, ...], so the body's
        // text is the claim's own words without the translated prefix.
        const claimWords = () => doc.getElementById('claimsHost').children[0].children[1].textContent;
        const before = {
            claimText: claimWords(),
            passage: doc.getElementById('evidenceSnippet').textContent,
            label: doc.getElementById('evidenceTarget').textContent
        };

        i18n.setLocale('en');
        ev.repaintLocalizedSurfaces();

        result = {
            beforeClaim: before.claimText,
            afterClaim: claimWords(),
            beforePassage: before.passage,
            afterPassage: doc.getElementById('evidenceSnippet').textContent,
            beforeLabel: before.label,
            afterLabel: doc.getElementById('evidenceTarget').textContent,
            documentLocale: i18n.documentLocale(before.claimText)
        };
        """
    )
    result = run_walkthrough_harness(i18n_module, body)
    assert result["afterClaim"] == result["beforeClaim"], (
        "the claim's own text was rewritten by the language switch — the document "
        "language and the interface language are separate axes"
    )
    assert re.search(r"[一-鿿]", result["afterClaim"]), "the fixture text should be Chinese"
    assert result["afterPassage"] == result["beforePassage"], (
        "the evidence passage was translated; it is the document's text, not the "
        "interface's"
    )
    assert result["documentLocale"] == "zh-CN", (
        "the Chinese paper must still be detected as Chinese under an English "
        "interface"
    )
    # And the interface around it really did change, so the assertion above is
    # not passing because nothing happened.
    assert result["beforeLabel"] != result["afterLabel"]
    assert "Page 7" in result["afterLabel"]


def test_the_evidence_and_claim_surfaces_re_render_on_a_locale_change():
    """The panels must repaint on `onLocaleChange`, not just render once.

    Reading the source for `onLocaleChange` alone would pass for a module that
    subscribes and then forgets to re-render. So this asserts the subscription
    exists *and* that calling the exported repaint actually rewrites what is on
    screen — the pairing that makes the switch work.
    """
    evidence = (JS_DIR / "evidence.js").read_text(encoding="utf-8")
    parse_status = (JS_DIR / "parseStatus.js").read_text(encoding="utf-8")
    for name, source in (("evidence.js", evidence), ("parseStatus.js", parse_status)):
        assert "onLocaleChange" in source, f"{name} does not subscribe to locale changes"

    # The evidence module's subscription must call the repaint, and the repaint
    # must be exported so the harness can drive it directly.
    assert "repaintLocalizedSurfaces" in evidence
    match = re.search(r"onLocaleChange\((.*?)\);", evidence, re.DOTALL)
    assert match, "evidence.js does not register a locale listener"
    assert "repaintLocalizedSurfaces" in match.group(1)

    # And the parse card's own subscription must repaint the card, the parser
    # menu and the last reparse result, not just the card.
    match = re.search(r"onLocaleChange\(\(\) => \{(.*?)\n\s*\}\);", parse_status, re.DOTALL)
    assert match, "parseStatus.js does not register a locale listener"
    for call in (
        "renderParserOptions",
        "updateReparseWarnings",
        "renderParseStatus",
        "lastReparseResult",
    ):
        assert call in match.group(1), (
            f"the parse-status locale listener does not re-run {call}, so the "
            "reparse control would keep the previous language"
        )


def run_viewer_harness(workdir: Path, body: str) -> dict:
    """Execute `body` against the real `evidenceViewer.js` and `i18n.js`.

    The viewer is imported by absolute path from the shipped file, and it is
    imported *first* so its own `onLocaleChange` subscription is registered
    before anything else runs — the ordering the browser has, and the ordering
    that decides which repaint wins.
    """
    (workdir / "dom_shim.mjs").write_text(DOM_SHIM, encoding="utf-8")
    harness = workdir / "viewer_harness.mjs"
    harness.write_text(
        "import './dom_shim.mjs';\n"
        "const viewer = await import(%r);\n"
        "const i18n = await import(%r);\n"
        "let result = null;\n"
        "try {\n"
        f"{body}\n"
        "} catch (error) {\n"
        "    result = { __threw: String(error && error.message) };\n"
        "}\n"
        "process.stdout.write(JSON.stringify(result === undefined ? null : result));\n"
        % (str(JS_DIR / "evidenceViewer.js"), str(JS_DIR / "i18n.js")),
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, timeout=60, cwd=workdir
    )
    assert completed.returncode == 0, f"harness failed: {completed.stderr}"
    return json.loads(completed.stdout)


#: A viewer payload per render branch, each carrying the interface strings this
#: migration was about and the document strings it must leave alone.
#:
#: The block and passage text is deliberately Chinese and deliberately
#: meaningless: a fixture in English could not distinguish "the document's words
#: survived the switch" from "the document's words happen to read the same in
#: both languages".
VIEWER_VIEWS = """
const PASSAGE = '本文提出了一种新的检索增强生成方法，并在大规模语料上验证了它的有效性。';
// Each view carries a `note_code` beside its `note`, exactly as the server
// ships them. The note text here is deliberately *not* the sentence the
// catalogue holds for the code, so a viewer that kept rendering `view.note`
// verbatim would be visible in every assertion below that reads `notes`.
const pageMap = {
    kind: 'page_map',
    label: 'p3',
    note: '这一页没有原始渲染图，下面是按解析坐标重建的布局。',
    note_code: 'page_map',
    is_raster: false,
    coordinate_space: 'pdf_user_space_points',
    frame: { x: 0, y: 0, width: 600, height: 800 },
    blocks: [
        { element_id: 'e1', role: 'text', x: 40, y: 700, width: 500, height: 24, text_preview: PASSAGE, preview_truncated: true },
        { element_id: 'e2', role: 'text', x: 40, y: 660, width: 200, height: 18, text_preview: '相关工作' }
    ],
    target: null,
    target_rejected: { reason_code: 'relative_bbox', message: '相对坐标，无法定位' },
    blocks_total: 1204,
    blocks_shown: 2,
    blocks_truncated: true,
    omitted: { relative_bbox: 3, unpositioned: 1 }
};
const slideView = {
    kind: 'slide_view',
    label: 'slide 4',
    note: '这张幻灯片没有像素坐标，下面列出全部文本元素。',
    note_code: 'slide',
    target: { text: PASSAGE },
    blocks: [
        { element_id: 's1', role: 'text', text: PASSAGE, is_target: true },
        { element_id: 's2', role: 'text', text: '' }
    ]
};
const elementView = {
    kind: 'element_view',
    label: '第 3 节',
    note: 'DOCX 没有页码坐标，下面按段落序号给出上下文。',
    note_code: 'element',
    heading_path: ['第三章 方法', '3.2 检索增强'],
    target: { text: PASSAGE },
    context: [
        { paragraph: 41, role: 'text', text: PASSAGE, is_target: true },
        { paragraph: 42, role: 'text', text: '' }
    ]
};
// Declared in the order the harness renders them, which is not alphabetical:
// the page map is last, because it is the view left on screen when the switch
// happens, and it carries every string this migration was about — the badge,
// the truncation notes, the absent-highlight sentence and the disclaimer —
// except the paragraph numbers.
const views = { slide_view: slideView, element_view: elementView, page_map: pageMap };
"""


def test_the_inside_of_the_evidence_panel_re_renders_in_the_readers_language(i18n_module):
    """The gap this migration closed: an English panel that was still Chinese.

    `claims.js`, `evidence.js` and `parseStatus.js` were routed through the
    catalogue by the previous change; `evidenceViewer.js` was left, and left
    was worse than untranslated — the panel's headline became English while the
    badge, the highlight notes, the truncation notes, the paragraph numbers and
    the "no text" placeholders stayed Chinese, inside the one surface a reader
    is looking at when deciding whether to trust a citation.

    So this renders all three view branches in Chinese, flips the switch, and
    asserts that every interface string on screen is the English one.
    """
    body = textwrap.dedent(
        """
        i18n.setLocale('zh-CN');
        """
        + VIEWER_VIEWS
        + textwrap.dedent(
            """
            const container = globalThis.document.getElementById('evidenceViewer');

            // Read the panel back out by class, because which sentence belongs
            // to which surface is the thing under test.
            const read = () => ({
                kinds: __byClass(container, 'evidence-view-kind'),
                badges: __byClass(container, 'evidence-view-badge'),
                labels: __byClass(container, 'evidence-view-label'),
                statuses: __byClass(container, 'evidence-view-status'),
                truncations: __byClass(container, 'evidence-view-truncation'),
                numbers: __byClass(container, 'evidence-paragraph-number'),
                slideText: __byClass(container, 'evidence-slide-text'),
                paragraphText: __byClass(container, 'evidence-paragraph-text'),
                pageBlocks: __byClass(container, 'evidence-page-block'),
                targets: __byClass(container, 'evidence-page-target'),
                notes: __byClass(container, 'evidence-view-note'),
                titles: __titles(container, 'evidence-page-block')
            });

            function renderAll() {
                const out = {};
                Object.entries(views).forEach(([kind, view]) => {
                    viewer.renderEvidenceView(view);
                    out[kind] = read();
                });
                return out;
            }

            const zh = renderAll();

            // The reader flips the switch. Nothing below calls the repaint
            // explicitly yet: the panel left on screen is the page map, and if
            // the viewer's own subscription is missing it is still Chinese here.
            i18n.setLocale('en');
            const afterEvent = read();

            // And the repaint is idempotent, exactly as `evidence.js`'s
            // subscription calling through `renderEvidencePayload` is.
            viewer.repaintLocalizedView();
            const afterRepaint = read();

            const en = renderAll();

            result = { zh, afterEvent, afterRepaint, en };
            """
        )
    )
    result = run_viewer_harness(i18n_module, body)
    assert "__threw" not in result, result

    # Fixture sanity: the panel really did start in Chinese, so "it is now
    # English" is a statement about the switch and not about a payload that
    # happened to be English to begin with.
    assert result["zh"]["page_map"]["kinds"] == ["页内布局（按解析坐标重建）"]
    assert result["zh"]["page_map"]["badges"] == ["非原始渲染图"]
    assert result["zh"]["element_view"]["numbers"] == ["第 41 段", "第 42 段"]
    assert result["zh"]["slide_view"]["slideText"][1] == "（无文本）"
    assert result["zh"]["page_map"]["titles"][0] == "（预览已截断）"
    # Only the first block's preview is clipped; the other carries no title at
    # all, which is the same in both languages.
    assert result["zh"]["page_map"]["titles"][1:] == [""]

    # The subscription is the mechanism, so the panel left on screen by the
    # event alone has to be English before any explicit repaint is called.
    assert result["afterEvent"]["kinds"] == ["Page layout (rebuilt from parse coordinates)"]
    assert result["afterEvent"]["badges"] == ["Not a rendered image"]

    # Idempotence: a second repaint must not change anything, so a panel that
    # repaints twice for one switch still reads once.
    assert result["afterRepaint"] == result["afterEvent"], (
        "repainting the open panel twice produced different text"
    )

    en = result["en"]
    # Every interface string on every branch, in English.
    assert en["page_map"]["kinds"] == ["Page layout (rebuilt from parse coordinates)"]
    assert en["slide_view"]["kinds"] == ["Slide content"]
    assert en["element_view"]["kinds"] == ["Paragraph location"]
    assert en["page_map"]["badges"] == ["Not a rendered image"]
    assert en["page_map"]["titles"] == ["(preview truncated)", ""]
    assert en["element_view"]["numbers"] == ["Paragraph 41", "Paragraph 42"]
    assert en["slide_view"]["slideText"][1] == "(no text)"
    assert en["element_view"]["paragraphText"][1] == "(no text)"
    assert "relative bounding box" in en["page_map"]["statuses"][0]
    assert "No highlight box drawn" in en["page_map"]["statuses"][0]
    assert en["slide_view"]["statuses"] == [
        "The highlight box comes from this evidence’s own bounding box (absolute "
        "coordinates), using the same PDF user-space coordinates as the elements "
        "inside the page."
    ]
    assert "1,204" in en["page_map"]["truncations"][0], (
        "the block count was not localised: " + repr(en["page_map"]["truncations"])
    )
    assert "3 elements carry a relative bounding box" in en["page_map"]["truncations"][1]
    assert "1 elements have no coordinates" in en["page_map"]["truncations"][2]

    # The disclaimer. It ships inside the payload, so it was the one sentence in
    # the panel the language switch could not reach; it is now read through
    # `evidence.view_note.<note_code>` and comes back in the reader's language.
    # The fixture's own `note` is Chinese and is *not* the catalogue's English,
    # so these assertions fail for a viewer that still renders `view.note`.
    for kind in ("page_map", "slide_view", "element_view"):
        assert en[kind]["notes"], f"{kind}: the disclaimer is missing"
        assert not re.search(r"[一-鿿]", en[kind]["notes"][0]), (
            f"{kind}: the disclaimer is still the server's Chinese sentence under "
            f"an English interface: {en[kind]['notes'][0]!r}"
        )
        assert re.search(r"[一-鿿]", result["zh"][kind]["notes"][0]), (
            f"{kind}: the disclaimer should read Chinese in the default locale"
        )
    assert "not a rendering of the original PDF" in en["page_map"]["notes"][0]
    assert "不是原始 PDF 的渲染图" in result["zh"]["page_map"]["notes"][0], (
        "the Chinese disclaimer should be the catalogue's own sentence for the "
        "code, which is the one the server ships"
    )
    assert en["page_map"]["notes"][0] != result["zh"]["page_map"]["notes"][0]

    # Not one interface string anywhere is still Chinese. The disclaimer is in
    # this list now: it is a statement about the drawing, not about the paper.
    for kind, surface in en.items():
        for name, values in surface.items():
            if name in ("pageBlocks", "targets", "paragraphText", "slideText", "labels"):
                continue  # the document's own words
            for value in values:
                assert not re.search(r"[一-鿿]", value), (
                    f"{kind}.{name} is still Chinese under an English interface: {value!r}"
                )


def test_the_viewer_leaves_the_documents_own_words_where_they_are(i18n_module):
    """The document-language boundary, on the surface it is easiest to break.

    The same axis `test_switching_language_does_not_relabel_a_chinese_papers_headings`
    pins for the claim and the passage, applied to the panel's interior: the
    paragraph furniture around a citation is the interface's, and the paragraph
    itself is the paper's. Relabelling a reader's paper because they switched
    their interface is a worse failure than an untranslated button.

    The view's disclaimer is the other half of that boundary and is asserted
    here for the same reason: it travels inside the payload, which is exactly
    what made it look like the document's words, and it is not. It describes the
    drawing, so it must move with the switch — otherwise it is the Chinese
    sentence stranded beside an English panel.
    """
    body = textwrap.dedent(
        """
        i18n.setLocale('zh-CN');
        """
        + VIEWER_VIEWS
        + textwrap.dedent(
            """
            const container = globalThis.document.getElementById('evidenceViewer');
            const read = () => ({
                pageBlocks: __byClass(container, 'evidence-page-block'),
                targets: __byClass(container, 'evidence-page-target'),
                paragraphText: __byClass(container, 'evidence-paragraph-text'),
                slideText: __byClass(container, 'evidence-slide-text'),
                headings: __byClass(container, 'evidence-heading-path'),
                labels: __byClass(container, 'evidence-view-label'),
                notes: __byClass(container, 'evidence-view-note'),
                numbers: __byClass(container, 'evidence-paragraph-number')
            });
            const zh = {};
            Object.entries(views).forEach(([kind, view]) => {
                viewer.renderEvidenceView(view);
                zh[kind] = read();
            });
            i18n.setLocale('en');
            viewer.repaintLocalizedView();
            const en = {};
            Object.entries(views).forEach(([kind, view]) => {
                viewer.renderEvidenceView(view);
                en[kind] = read();
            });
            result = { zh, en };
            """
        )
    )
    result = run_viewer_harness(i18n_module, body)
    assert "__threw" not in result, result

    document_surfaces = (
        "pageBlocks",
        "targets",
        "paragraphText",
        "slideText",
        "headings",
        "labels",
    )
    #: Where each branch keeps the document's own words. One passage string is
    #: used across all three fixtures, so one check covers them.
    passage_by_kind = {
        "page_map": "pageBlocks",
        "slide_view": "slideText",
        "element_view": "paragraphText",
    }
    for kind in ("page_map", "slide_view", "element_view"):
        for name in document_surfaces:
            expected = list(result["zh"][kind][name])
            if name in ("paragraphText", "slideText"):
                # Index 0 is the document's own words and must not move; the
                # rest are the "no text" placeholder, which is the interface's
                # and is asserted to have switched by the test above.
                expected = expected[:1]
            assert result["en"][kind][name][: len(expected)] == expected, (
                f"{kind}.{name} was rewritten by the language switch: "
                f"{expected!r} became {result['en'][kind][name]!r}"
            )
        # Fixture sanity: the paper is Chinese, so a translation would be
        # visible as the text losing its ideographs.
        assert re.search(r"[一-鿿]", result["en"][kind][passage_by_kind[kind]][0]), (
            f"{kind}: the cited passage should still be Chinese"
        )
    # And the furniture around it really did move, or the assertions above are
    # passing because the switch did nothing.
    assert result["en"]["element_view"]["numbers"] == ["Paragraph 41", "Paragraph 42"]
    assert result["zh"]["element_view"]["numbers"] == ["第 41 段", "第 42 段"]
    # The disclaimer is furniture, not paper: it describes the drawing, so it
    # moves with the switch on every branch. Listed separately rather than in
    # `document_surfaces` because this test is the one place that says which
    # side of that line it is on — and it used to be on the wrong side, which
    # is what stranded a Chinese sentence in an English panel.
    for kind in ("page_map", "slide_view", "element_view"):
        assert result["en"][kind]["notes"] != result["zh"][kind]["notes"], (
            f"{kind}: the disclaimer did not follow the language switch"
        )


def test_a_viewer_string_with_no_catalogue_key_falls_back_to_a_sentence(i18n_module):
    """A missing key must not become a dotted key or an empty element.

    `t()` returns the key itself for an entry the catalogue does not carry,
    which is right while reviewing the source and wrong on the page: a badge
    reading `evidence.not_a_raster` reads as a broken control, and a note
    reading nothing reads as a statement there is nothing to say. So the
    sentence the module was migrated *from* is kept as a floor.
    """
    body = textwrap.dedent(
        """
        i18n.setLocale('en');
        """
        + VIEWER_VIEWS
        + textwrap.dedent(
            """
            const container = globalThis.document.getElementById('evidenceViewer');
            viewer.renderEvidenceView(views.page_map);
            const before = __byClass(container, 'evidence-view-badge');

            // Remove the entry from both locales, which is what a build that
            // lost the key actually looks like: `t()` has nothing left and
            // answers with the key.
            delete i18n.CATALOGUE['en']['evidence.not_a_raster'];
            delete i18n.CATALOGUE['zh-CN']['evidence.not_a_raster'];
            const viaKey = i18n.t('evidence.not_a_raster');

            viewer.renderEvidenceView(views.page_map);
            const after = __byClass(container, 'evidence-view-badge');

            // The same question asked with no code at all, and with a code this
            // build has never heard of: both must still say something.
            const noCode = viewer.describeMissingHighlight({});
            const unknownCode = viewer.describeMissingHighlight({
                target_rejected: { reason_code: 'a_code_from_a_future_build' }
            });
            const knownCode = viewer.describeMissingHighlight({
                target_rejected: { reason_code: 'relative_bbox' }
            });

            // The same question asked of the view's disclaimer: a note whose
            // code is missing, and one whose code is present but whose
            // catalogue entry has been lost.
            const serverNote = '服务端随负载送来的这一句说明。';
            const noteNoCode = viewer.describeViewNote({ note: serverNote });
            const noteUnknownCode = viewer.describeViewNote({
                note: serverNote, note_code: 'a_code_from_a_future_build'
            });
            delete i18n.CATALOGUE['en']['evidence.view_note.page_map'];
            delete i18n.CATALOGUE['zh-CN']['evidence.view_note.page_map'];
            const noteLostKey = viewer.describeViewNote(views.page_map);

            result = { before, viaKey, after, noCode, unknownCode, knownCode,
                      noteNoCode, noteUnknownCode, noteLostKey };
            """
        )
    )
    result = run_viewer_harness(i18n_module, body)
    assert "__threw" not in result, result

    assert result["before"] == ["Not a rendered image"]
    assert result["viaKey"] == "evidence.not_a_raster", (
        "the premise of this test is that `t()` answers with the key itself"
    )
    assert result["after"] == ["非原始渲染图"], (
        "a missing catalogue key must degrade to the sentence, not to the key "
        f"and not to nothing: {result['after']!r}"
    )
    assert result["after"] != result["viaKey"]
    for name in ("noCode", "unknownCode", "knownCode"):
        value = result[name]
        assert value.strip(), f"{name} rendered as an empty string"
        assert not value.startswith("evidence."), (
            f"{name} fell through to a catalogue key: {value!r}"
        )
    assert result["knownCode"] == (
        "This evidence carries a relative bounding box. Mapped onto the page it "
        "would highlight the wrong region, so no highlight box is offered."
    ), (
        "a code the client knows must be described in the reader's language "
        f"from the catalogue, not from the server's prose: {result['knownCode']!r}"
    )
    # The disclaimer's floors. A note with no code, and a note with a code this
    # build has never heard of, both fall back to the server's own sentence —
    # visible prose, never a dotted key. And a *known* code whose catalogue
    # entry has been lost degrades to that same floor rather than to the key,
    # which is the only reason the `localised()` helper exists.
    assert result["noteNoCode"] == "服务端随负载送来的这一句说明。"
    assert result["noteUnknownCode"] == "服务端随负载送来的这一句说明。"
    assert result["noteLostKey"] == "这一页没有原始渲染图，下面是按解析坐标重建的布局。", (
        "a note code the catalogue lost must degrade to the server's sentence, "
        f"not to a dotted key and not to nothing: {result['noteLostKey']!r}"
    )
    for name in ("noteNoCode", "noteUnknownCode", "noteLostKey"):
        assert not result[name].startswith("evidence.view_note"), (
            f"{name} fell through to a catalogue key: {result[name]!r}"
        )


def test_every_key_the_viewer_builds_at_run_time_exists_in_both_languages():
    """The allowlist is wired into a real assertion, in both directions.

    A key assembled at run time is invisible to the "no bare literal" and
    "actually calls `t()`" passes, so the list of them has to be asserted
    against the shipped catalogue by hand — and a hand-written list that nothing
    reads is worse than no list, because it looks like coverage.

    Both directions are asserted, because the two failures are different. A key
    on the allowlist that is not in the catalogue is a rendered dotted string;
    a key the module builds that is *not* on the allowlist is a catalogue entry
    nothing checks, which is how the list silently stops describing the module.
    The set is therefore compared against the module's own vocabulary, which is
    read out of the shipped source rather than restated here.
    """
    viewer = (JS_DIR / "evidenceViewer.js").read_text(encoding="utf-8")
    config = (JS_DIR / "config.js").read_text(encoding="utf-8")

    # `VIEW_NOTE_CODES`, as data.
    codes = re.findall(
        r"const VIEW_NOTE_CODES = new Set\(\[(.*?)\]\);", viewer, re.DOTALL
    )
    assert len(codes) == 1, "VIEW_NOTE_CODES is not declared once in evidenceViewer.js"
    note_codes = re.findall(r"'([^']+)'", codes[0])
    assert note_codes, "VIEW_NOTE_CODES is empty, so no note can ever be translated"

    # The prefix they are rendered through, read from the same source.
    prefix = re.findall(r"const NOTE_CODE_KEY_PREFIX = '([^']+)';", viewer)
    assert len(prefix) == 1, "NOTE_CODE_KEY_PREFIX is not declared in evidenceViewer.js"

    # The two vocabularies that live in config.js, for the same reason.
    def _keys_of(name: str) -> list[str]:
        block = re.search(rf"export const {name} = \{{(.*?)\n\}};", config, re.DOTALL)
        assert block, f"{name} is missing from config.js"
        return re.findall(r"^    (\w+):", block.group(1), re.MULTILINE)

    expected = {
        *(f"{prefix[0]}{code}" for code in note_codes),
        *(f"evidence.view_kind.{kind}" for kind in _keys_of("VIEW_KIND_LABELS")),
        "evidence.view_kind.fallback",
        *(f"evidence.no_target.{code}" for code in _keys_of("NO_TARGET_MESSAGES")),
        "evidence.view_target_missing",
    }
    assert set(VIEWER_DYNAMIC_KEYS) == expected, (
        "the dynamically-built key allowlist no longer describes the module; "
        f"only on the allowlist: {sorted(set(VIEWER_DYNAMIC_KEYS) - expected)}; "
        f"only in the module: {sorted(expected - set(VIEWER_DYNAMIC_KEYS))}"
    )

    zh = catalogue_table("ZH_CN")
    en = catalogue_table("EN")
    for key in VIEWER_DYNAMIC_KEYS:
        assert key in zh, f"{key} is assembled at run time but absent from zh-CN"
        assert key in en, f"{key} is assembled at run time but absent from en"
        assert zh[key] != en[key], f"{key} reads identically in both languages"


def test_every_note_code_the_server_can_send_is_one_this_build_can_translate():
    """The two vocabularies on either side of the wire are the same set.

    The server sends `view.note_code`; the client recognises a code before it
    builds a key from it, and a code it does not recognise degrades to the
    server's Chinese sentence. That degradation is exactly the defect this
    change fixed, so a code the server can send and the client cannot describe
    is the same untranslated note, one release later.

    Read out of the shipped sources rather than restated, so adding a note to
    the server without translating it fails here instead of in a browser.
    """
    from paperwhisperer.rag import evidence as evidence_module

    server_codes = {
        getattr(evidence_module, name)
        for name in dir(evidence_module)
        if name.startswith("VIEW_NOTE_")
    }
    assert server_codes, "no note codes are declared on the server"

    viewer = (JS_DIR / "evidenceViewer.js").read_text(encoding="utf-8")
    block = re.search(r"const VIEW_NOTE_CODES = new Set\(\[(.*?)\]\);", viewer, re.DOTALL)
    assert block, "evidenceViewer.js no longer declares the note codes it can render"
    client_codes = set(re.findall(r"'([^']+)'", block.group(1)))

    assert client_codes == server_codes, (
        "the note codes the client recognises and the ones the server sends are "
        f"not the same set: server-only {sorted(server_codes - client_codes)}, "
        f"client-only {sorted(client_codes - server_codes)}"
    )
    # Sanity on the set itself: a vocabulary that collapsed to one member would
    # satisfy the equality above while describing almost nothing.
    assert len(client_codes) >= 3, (
        f"only {len(client_codes)} note codes are recognised: "
        f"{sorted(client_codes)}"
    )


def test_the_viewer_subscribes_to_locale_changes_and_repainits_the_open_panel():
    """The subscription, asserted on the source; the effect, asserted above.

    Splitting it this way is deliberate: reading `onLocaleChange` out of the
    file catches a module that unsubscribed, and running the module catches one
    that subscribes and forgets to re-render. Neither alone is the guarantee.
    """
    viewer = (JS_DIR / "evidenceViewer.js").read_text(encoding="utf-8")
    assert "from './i18n.js'" in viewer, "the viewer does not import the catalogue"
    match = re.search(r"onLocaleChange\((.*?)\);", viewer, re.DOTALL)
    assert match, "evidenceViewer.js does not register a locale listener"
    assert "repaintLocalizedView" in match.group(1)
    assert "repaintLocalizedView" in viewer, (
        "the repaint must be exported so a test can drive it, as evidence.js "
        "exports repaintLocalizedSurfaces"
    )


# --------------------------------------------------------------------------
# The switcher in the template
# --------------------------------------------------------------------------


def test_the_switcher_exists_and_is_a_radiogroup():
    """A radiogroup, in the header, with one option per supported locale.

    Not a toggle: with two languages a toggle hides which one is active while
    costing two tab stops to undo. `role="radiogroup"` with `role="radio"`
    children is also what makes a screen reader announce "2 of 2" rather than
    reading two unrelated buttons.
    """
    match = re.search(r'<div[^>]*id="languageSwitcher".*?</div>', INDEX_HTML, re.DOTALL)
    assert match, "the language switcher is not in the template"
    markup = match.group(0)
    assert 'role="radiogroup"' in markup
    assert 'id="languageSwitcher"' in markup
    assert markup.count('role="radio"') == 2, "there must be one radio per shipped locale"
    assert 'data-locale-option="zh-CN"' in markup
    assert 'data-locale-option="en"' in markup


def test_the_switcher_announces_its_state_without_relying_on_colour():
    """`aria-checked` on every option, and a non-colour state carrier.

    `aria-checked` is what a screen reader announces. `data-selected` is the
    hook a stylesheet can hang a border or a weight on, so the visible state is
    not carried by background colour alone.
    """
    match = re.search(r'<div[^>]*id="languageSwitcher".*?</div>', INDEX_HTML, re.DOTALL)
    markup = match.group(0)
    assert markup.count("aria-checked=") == 2
    assert markup.count("data-selected=") == 2
    assert 'aria-checked="true"' in markup
    assert 'aria-checked="false"' in markup


def test_the_switcher_group_is_labelled_through_the_catalogue():
    """The group's own label is localised, and the key exists.

    A group with no name is announced as just "group", which is the one part of
    the switcher a screen-reader user cannot find on the page.
    """
    match = re.search(r'<div[^>]*id="languageSwitcher"[^>]*>', INDEX_HTML)
    assert match, "the language switcher is not in the template"
    markup = match.group(0)
    assert 'data-i18n-attr="aria-label:switcher.group_label"' in markup
    assert "switcher.group_label" in catalogue_table("ZH_CN")
    assert "switcher.group_label" in catalogue_table("EN")


def test_the_switcher_is_initialised_from_main_before_anything_renders():
    """Wired, and wired first.

    `initializeI18n()` is called at the top of `main.js`'s `initialize()` so the
    first paint is already in the reader's language. Asserted as an ordering
    property — the call must precede `initializeTheme()`, `updateStatus()` and
    every feature initialiser — because a switcher that initialises last shows
    the whole page in the default language first and then swaps it.
    """
    main = (JS_DIR / "main.js").read_text(encoding="utf-8")
    assert "from './i18n.js'" in main
    match = re.search(r"function initialize\(\) \{(.*?)\n\}", main, re.DOTALL)
    assert match, "initialize() is missing from main.js"
    body = match.group(1)
    i18n_at = body.find("initializeI18n()")
    assert i18n_at >= 0, "initializeI18n() is never called"
    for later in ("initializeTheme()", "updateFileMeta()", "resetResultView()"):
        assert i18n_at < body.find(later), (
            f"initializeI18n() must run before {later}()"
        )


def test_the_html_lang_attribute_starts_as_the_default_locale():
    """`<html lang>` is `zh-CN` in the served markup.

    The attribute is what a screen reader, a browser's own translation prompt
    and `:lang()` rules read before any script runs. With Chinese as the
    default it must be `zh-CN` statically, or the first paint is mislabelled
    even though `initializeI18n()` corrects it a moment later.
    """
    assert '<html lang="zh-CN">' in INDEX_HTML
    assert INDEX_HTML.count("<html") == 1


def test_still_exactly_one_module_entry_point():
    """`i18n.js` is reached through `main.js`, not by a second script tag.

    The template must keep exactly one `<script type="module">`; a second one
    would break both `tools/check_frontend.py` and the existing wiring test, and
    a duplicate entry point is a load-order hazard besides.
    """
    assert INDEX_HTML.count('<script type="module"') == 1


# --------------------------------------------------------------------------
# The DOM mechanism, against a minimal stub
# --------------------------------------------------------------------------


def test_applying_a_locale_rewrites_the_static_markup_and_the_html_lang(i18n_module):
    """`applyLocale` is what actually changes the page, so it is exercised.

    There is no browser here, so the DOM is a stub: an element with the two
    accessors `translateStaticMarkup` uses (`getAttribute`, `setAttribute`) and
    a `querySelectorAll` that returns the elements carrying `data-i18n`. That is
    a narrow stub, and it is honest about what it proves — that the lookup runs,
    the text is replaced, and the attribute keys are written — but it cannot
    prove anything about layout, focus or paint. The real rendering is a manual
    check, listed in docs/I18N.md.

    Without this, a build in which `applyLocale` was never called, or was called
    with the wrong root, would still pass every catalogue test above: those all
    read the data, none of them drive the DOM.
    """
    body = """
const made = [];
function el(attributes = {}, text = '') {
    return {
        attrs: { ...attributes },
        textContent: text,
        getAttribute(name) { return name in this.attrs ? this.attrs[name] : null; },
        setAttribute(name, value) { this.attrs[name] = value; },
        removeAttribute(name) { delete this.attrs[name]; }
    };
}
const heading = el({ 'data-i18n': 'result.overview' }, 'Overview');
const status = el({ 'data-i18n': 'status.waiting' }, 'Waiting for document');
const input = el({ 'data-i18n-attr': 'placeholder:search.placeholder' }, '');
const nodes = [heading, status, input];
globalThis.document = {
    documentElement: el({ lang: 'zh-CN' }),
    getElementById: () => null,
    dispatchEvent: () => true,
    querySelectorAll: (selector) => (selector === '[data-i18n]' ? [heading, status] : [input])
};
m.setLocale('en');
m.translateStaticMarkup();
const en = {
    lang: globalThis.document.documentElement.attrs.lang,
    ui: globalThis.document.documentElement.attrs['data-ui-lang'],
    heading: heading.textContent,
    status: status.textContent,
    placeholder: input.attrs.placeholder
};
m.setLocale('zh-CN');
m.translateStaticMarkup();
result = { en, zhHeading: heading.textContent, zhStatus: status.textContent };
"""
    result = run_i18n(i18n_module, body)
    assert result["en"]["lang"] == "en"
    assert result["en"]["ui"] == "en", "the data-ui-lang hook must be set for :lang()-free CSS"
    assert result["en"]["heading"] == "Overview"
    assert result["en"]["status"] == "Waiting for document"
    assert result["en"]["placeholder"] == "Search by topic, task, method, or dataset..."
    # The same elements, re-translated, now carry the Chinese values.
    assert result["zhHeading"] == "概览"
    assert result["zhStatus"] == "等待文档"


# --------------------------------------------------------------------------
# No literals outside the catalogue, in the modules migrated here
# --------------------------------------------------------------------------


def strip_js_comments(source: str) -> str:
    """Remove comments, leaving string literals intact.

    Needed so a comment discussing a phrase cannot satisfy or violate the
    structural assertion below. String state is tracked, so a `//` inside a
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
            while index < length and not (source[index] == "*" and source[index + 1 : index + 2] == "/"):
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


#: Expressions that build a string but do not put it in front of a reader.
NON_USER_VISIBLE_PREFIXES = (
    "console.", "import ", "export ", "from '", "from \"",
    "//", "/*", "*", "@", "http", "#", ".", "$",
)

#: Sinks where a bare literal would be user-visible. Keyed by the regex that
#: finds the literal; group 1 is the literal itself.
USER_VISIBLE_SINKS = [
    re.compile(r"\.textContent\s*=\s*'([^']{3,})'"),
    re.compile(r"\.textContent\s*=\s*\"([^\"]{3,})\"\s*[;)]"),
    re.compile(r"setAttribute\(\s*'aria-label'\s*,\s*'([^']{3,})'"),
    re.compile(r"updateStatus\(\s*'([^']{3,})'"),
    re.compile(r"placeholder\s*=\s*'([^']{3,})'"),
]


def test_no_bare_user_visible_literal_remains_in_the_migrated_modules():
    """Every user-visible string in a migrated module goes through `t()`.

    Scoped to {@link MIGRATED_MODULES} on purpose. The brief for this task
    produced a ready-to-apply migration for the remaining modules rather than
    editing them mid-flight, and a test scoped wider would fail for work that has
    not landed yet — which trains the suite to be ignored. Widen this list as
    each module is migrated; the assertion gets stronger every time.

    The check is for a *bare* literal in a user-visible sink. A call through
    `t()` is a call expression, not a string literal, so it does not match.
    """
    offenders: list[str] = []
    for name in MIGRATED_MODULES:
        source = strip_js_comments((JS_DIR / name).read_text(encoding="utf-8"))
        for pattern in USER_VISIBLE_SINKS:
            for match in pattern.finditer(source):
                literal = match.group(1)
                if literal.startswith(NON_USER_VISIBLE_PREFIXES):
                    continue
                offenders.append(f"{name}: {literal!r}")
    assert not offenders, (
        "user-visible literals outside the catalogue in migrated modules: "
        + "; ".join(offenders)
    )


def test_the_migrated_modules_actually_use_the_catalogue():
    """Each migrated module imports and calls `t()`.

    Guards against the previous test passing for the wrong reason — a module
    whose literals happened to be unrecognised by the sink patterns would
    satisfy "no bare literals" while being entirely unmigrated.
    """
    for name in MIGRATED_MODULES:
        if name == "i18n.js":
            continue
        source = (JS_DIR / name).read_text(encoding="utf-8")
        assert "from './i18n.js'" in source, f"{name} does not import the catalogue"
        assert re.search(r"\bt\(", source), f"{name} imports the catalogue but never calls t()"


# --------------------------------------------------------------------------
# The four modules migrated this round, and the dialog they broke
# --------------------------------------------------------------------------


#: The modules migrated together, and the reason they were migrated together.
#:
#: They are one group because they are the modules a reader is looking at when
#: something has gone wrong: the run that failed, the failure notice beside it,
#: the workspace that came back, and the paper list that may have sent the run.
#: A hardcoded string in any one of them is a sentence in the wrong language
#: *next to* a correct one, which is the mixed-language case, and the workspace
#: dialog was the worst of it — Chinese about the server's error, then a
#: hardcoded English sentence about the work that came back.
MIGRATED_TOGETHER = ("analyze.js", "papers.js", "runtime.js", "paperList.js")


def test_the_hardcoded_half_of_the_workspace_dialog_is_gone():
    """The specific sentence the mixed-language dialog was made of.

    Before the migration, `analyze.js` (twice) and `papers.js` (once) built that
    dialog by concatenating a server error onto a hardcoded English
    "Your previous successful workspace has been restored." The default locale
    is Chinese, so a reader in the default language was shown one Chinese
    sentence and one English one in the same alert, and a reader who switched to
    English was shown the English half twice and the Chinese half once.

    A structural grep is the honest pin here. Reconstructing the dialog in a
    harness would test the harness: it would have to reimplement the same
    concatenation to pass before the fix and to fail after it, so it can only
    detect the defect it was written from. What is asserted is that the literal
    is gone *everywhere* — not merely from the three call sites, because a
    fourth one that stringifies it back together is exactly the regression this
    is here to catch.
    """
    offenders = [
        path.name
        for path in sorted(JS_DIR.glob("*.js"))
        if "Your previous successful workspace has been restored." in strip_js_comments(
            path.read_text(encoding="utf-8")
        )
        and path.name != "i18n.js"
    ]
    assert not offenders, (
        "the hardcoded half of the workspace-restore dialog is back in: "
        f"{offenders}. The whole dialog is one catalogue key now."
    )

    # And the whole dialog is that one key — three call sites, all resolved.
    key = "analysis.workspace_restored"
    zh = catalogue_table("ZH_CN")
    en = catalogue_table("EN")
    assert key in zh and key in en, (
        "the workspace-restore dialog must exist in both languages; the whole "
        "sentence is the interface's, and a half-translated alert is worse "
        "than either language"
    )
    assert "{message}" in zh[key] and "{message}" in en[key], (
        "the dialog must take the server's own message as a placeholder, so "
        "the client passes the server's words through instead of rewriting them"
    )
    # Four sites in total: the interrupted-stream report and the recovered-job
    # report in `analyze.js`, the failed-run catch in `analyze.js`, and the
    # failed import in `papers.js`. Two of the three live in one file, which is
    # why this is a count rather than a presence check — a presence check
    # cannot tell "all four are keyed" from "one was rebuilt by hand".
    for name, expected in (("analyze.js", 3), ("papers.js", 1)):
        source = strip_js_comments((JS_DIR / name).read_text(encoding="utf-8"))
        found = source.count(f"t('{key}'")
        assert found == expected, (
            f"{name} resolves the workspace-restore dialog {found} times, "
            f"expected {expected} — a call site lost its key, or a new one "
            "built the dialog some other way"
        )


#: The server's own sentence, in the server's own language. The client is
#: required to carry it through untouched, so the fixture is Chinese and the
#: default locale is Chinese — which is precisely the configuration in which a
#: client that *did* translate would look correct.
SERVER_MESSAGE = "服务端在分析这份文档时出错。"


def test_the_workspace_dialog_is_one_sentence_in_the_readers_language(i18n_module):
    """Run, not grepped: the composed dialog reads entirely in one language.

    The grep above proves the literal is gone; this proves the *result* is
    whole. A key that resolves to a Chinese sentence while the reader has
    chosen English satisfies every structural assertion in this file and is
    still the exact defect this migration was about, so the two assertions are
    paired: the string is absent from the modules, and what comes out is
    monolingual.

    The fixture is the server's Chinese sentence in both passes. That is the
    case a client-side rewrite cannot fake: under the default locale an English
    frame around it would be the visible defect, and under `en` a Chinese frame
    around it is the same defect.
    """
    body = (
        "const sample = %r;\n"
        "const read = (locale) => {\n"
        "    m.setLocale(locale);\n"
        "    return m.t('analysis.workspace_restored', { message: sample });\n"
        "};\n"
        "result = { zh: read('zh-CN'), en: read('en') };\n"
    ) % SERVER_MESSAGE
    result = run_i18n(i18n_module, body)
    assert "__threw" not in result, result
    assert SERVER_MESSAGE in result["zh"] and SERVER_MESSAGE in result["en"], (
        "the server's own message must reach the reader unchanged in both "
        "languages — the client passes it through, it does not rewrite it"
    )
    # The frame the client *owns* is the only thing under test here, so the
    # server's own Chinese is stripped before asking whether any Chinese
    # survived: it is expected to, and it is not the defect.
    zh_frame = result["zh"].replace(SERVER_MESSAGE, "")
    en_frame = result["en"].replace(SERVER_MESSAGE, "")
    assert not re.search(r"[一-鿿]", en_frame), (
        "the English dialog still contains a Chinese sentence from the frame "
        f"around the server message: {result['en']!r}"
    )
    assert re.search(r"[一-鿿]", zh_frame), (
        f"the default locale's frame is not Chinese at all: {result['zh']!r}"
    )
    assert result["zh"] != result["en"]
    # The part that used to be hardcoded English is present in both — that it
    # moved is the point; that it disappeared would be a regression.
    assert "已经恢复" in zh_frame
    assert "has been restored" in en_frame


@pytest.mark.parametrize("name", MIGRATED_TOGETHER)
def test_no_hardcoded_chinese_survives_in_the_migrated_modules(name):
    """The whole module, not the sinks the earlier test happens to recognise.

    `test_no_bare_user_visible_literal_remains_in_the_migrated_modules` matches
    a fixed list of assignment shapes. This one reads for the thing itself: a
    CJK ideograph in a string literal is a Chinese sentence the language switch
    cannot reach, whether it is handed to `textContent`, concatenated into a
    meta line, or passed as a fall-through default.

    Comments are stripped first, so a comment *discussing* a Chinese string
    cannot fail this; and identifiers and non-code text are outside the string
    literals that remain.
    """
    source = strip_js_comments((JS_DIR / name).read_text(encoding="utf-8"))
    offenders = re.findall(r"'[^'\n]*[一-鿿][^'\n]*'", source)
    # One exception, and it is not an exception of convenience: it is a scope
    # identifier. `flushOfflineQueue` raises a failure tagged
    # `operation: '离线队列'`, and `runtime.js` never renders an operation —
    # `dismissNoticesFor` only compares it. Giving it a catalogue key would make
    # the string a success retires its notice by depend on the interface
    # language, so a notice raised before a switch would stop matching the
    # success that follows it. The assertion below pins that it stays a literal.
    allowed = {"'离线队列'"}
    unexpected = [value for value in offenders if value not in allowed]
    assert not unexpected, (
        f"{name} still carries hardcoded Chinese outside the catalogue: "
        f"{unexpected}"
    )


def test_the_offline_queue_scope_is_not_reader_facing():
    """The one Chinese literal that stays in `runtime.js`, pinned deliberately.

    It is here so the exception above cannot quietly widen: if this string ever
    does reach the screen, or if it is replaced by a key, the offline queue's
    failures would stop being retired by the next success of that operation.
    """
    runtime = strip_js_comments((JS_DIR / "runtime.js").read_text(encoding="utf-8"))
    assert "notifyFailure({ code: 'client_unknown_error', operation: '离线队列' })" in runtime, (
        "the offline queue's own failures no longer carry their operation scope"
    )
    assert "runtime.offline_queue" not in runtime, (
        "the offline queue's operation scope must not become a catalogue key: "
        "a success retires notices by comparing this string, and a switch in "
        "the interface language must not change which notices a success clears"
    )
    # ...and nothing renders it: `normalizeFailure` keeps `operation` on the
    # failure for `dismissNoticesFor` to compare, and no sink writes it out.
    assert re.search(r"textContent\s*=\s*[\w.]*operation", runtime) is None, (
        "an operation scope has reached a text node — it is an identifier, not copy"
    )
    assert re.search(r"title\.textContent\s*=\s*normalized\.title", runtime), (
        "the notice title must still come from the code's resolved copy, not "
        "from the scope identifier"
    )
    assert "title: copy.title" in runtime, (
        "normalizeFailure must still take the notice title from the catalogue's "
        "code-driven copy rather than from anything the caller supplied"
    )


@pytest.mark.parametrize("name", MIGRATED_TOGETHER)
def test_every_key_these_modules_look_up_resolves_in_both_languages(name):
    """A referenced key that is not defined renders as a dotted string.

    `tests/test_i18n.py` owns parity and non-emptiness across the catalogue as a
    whole; this asks the narrower question — does anything in *these four
    modules* name a key that does not exist. A module that reached for
    `analysis.job.complete` instead of `analysis.job.completed` would satisfy
    every table-wide invariant in this file and put a dotted key on the status
    chip beside a restored workspace.

    Only literal `t('…')` / `localised('…')` arguments are readable this way.
    The keys assembled at run time are covered by
    {@link test_the_run_time_built_keys_resolve_in_both_languages}.
    """
    source = strip_js_comments((JS_DIR / name).read_text(encoding="utf-8"))
    keys = set(re.findall(r"\bt\('([^']+)'", source)) | set(
        re.findall(r"\blocalised\('([^']+)'", source)
    )
    assert keys, f"{name} names no catalogue key at all — did the migration land?"
    zh = catalogue_table("ZH_CN")
    en = catalogue_table("EN")
    assert not keys - set(zh), f"{name} looks up keys absent from zh-CN: {sorted(keys - set(zh))}"
    assert not keys - set(en), f"{name} looks up keys absent from en: {sorted(keys - set(en))}"


#: The keys these four modules assemble from a wire value, and therefore the
#: ones no single `t('...')` in the source can be found by grepping.
#:
#: There are two families and both are driven by a vocabulary the *server*
#: supplies. `paperList.js` and `papers.js` name a signal by its code, so a new
#: upstream signal has to have a label in this catalogue before an English reader
#: can see it — and a code with no label degrades to the server's Chinese
#: `label_zh`, which is the defect. The confidence qualifier is the same shape.
RUN_TIME_BUILT_KEYS = (
    "recommend.signal.cited_by",
    "recommend.signal.citing",
    "recommend.signal.related",
    "recommend.signal.topical",
    "recommend.signal.same_dataset",
    "recommend.signal.fallback",
    "recommend.confidence.declared",
    "recommend.confidence.derived",
)


def test_the_run_time_built_keys_resolve_in_both_languages():
    """The allowlist is wired into a real assertion, in both directions.

    Same argument as the viewer's: a hand-written list that nothing reads is
    worse than no list, because it looks like coverage. So the list is compared
    against the vocabularies the modules actually build from — read out of
    `config.js`, which is where the server-facing vocabulary lives — and then
    every key on it is checked against the shipped catalogue in both languages.
    """
    config = (JS_DIR / "config.js").read_text(encoding="utf-8")

    def _keys_of(name: str) -> list[str]:
        block = re.search(rf"export const {name} = \{{(.*?)\n\}};", config, re.DOTALL)
        assert block, f"{name} is missing from config.js"
        return re.findall(r"^    (\w+):", block.group(1), re.MULTILINE)

    signals = _keys_of("RECOMMENDATION_SIGNAL_LABELS")
    confidences = _keys_of("RECOMMENDATION_CONFIDENCE_LABELS")
    assert signals and confidences, "the vocabularies the keys are built from are empty"
    expected = {
        *(f"recommend.signal.{signal}" for signal in signals),
        "recommend.signal.fallback",
        *(f"recommend.confidence.{confidence}" for confidence in confidences),
    }
    assert set(RUN_TIME_BUILT_KEYS) == expected, (
        "the run-time-built key allowlist no longer describes the modules; "
        f"only on the allowlist: {sorted(set(RUN_TIME_BUILT_KEYS) - expected)}; "
        f"only built by the modules: {sorted(expected - set(RUN_TIME_BUILT_KEYS))}"
    )

    zh = catalogue_table("ZH_CN")
    en = catalogue_table("EN")
    for key in RUN_TIME_BUILT_KEYS:
        assert key in zh, f"{key} is assembled at run time but absent from zh-CN"
        assert key in en, f"{key} is assembled at run time but absent from en"
        assert zh[key] != en[key], f"{key} reads identically in both languages"

    # The prefix is read out of the modules, so renaming it in one place cannot
    # leave the other building keys this build has never heard of.
    paper_list = (JS_DIR / "paperList.js").read_text(encoding="utf-8")
    assert re.search(
        r"const SIGNAL_LABEL_KEY_PREFIX = 'recommend\.signal\.';", paper_list
    ), "paperList.js no longer declares the prefix it builds signal labels through"


def test_the_migrated_modules_leave_the_servers_own_words_alone():
    """What must *not* be keyed, pinned so the migration cannot overreach.

    Localising the interface is not localising the payload. `paperList.js` shows
    `label_zh` and `reason_zh`, and `papers.js` shows the rewrite fields, and
    those are the words that arrived with the paper — a Chinese paper stays
    Chinese under an English interface, which is the same boundary
    `documentLocale()` exists to draw. A migration that reached into them would
    produce a card that silently misreports what the server sent, which is worse
    than one that shows an untranslated line and says so.
    """
    paper_list = strip_js_comments((JS_DIR / "paperList.js").read_text(encoding="utf-8"))
    for field in ("reason?.label_zh", "reason?.reason_zh"):
        assert field in paper_list, (
            f"{field} is no longer read: the card would be reconstructing a "
            "label the client was given rather than showing it"
        )
    papers = strip_js_comments((JS_DIR / "papers.js").read_text(encoding="utf-8"))
    for field in (
        "data.original_query",
        "data.rewritten_query",
        "data.reason",
        "data.rewrite_model",
        "entry?.label_zh",
    ):
        assert field in papers, (
            f"{field} is no longer read off the payload: the meta line is "
            "supposed to report what the server said, not paraphrase it"
        )


def test_the_empty_state_decision_cannot_depend_on_the_copy():
    """The loading / no-match branch reads a key, not a sentence.

    `getPaperStateDetails` used to decide which empty state to draw by sniffing
    English prose — `/searching|generating/i`, `includes('No matching')`. That
    is the one place in this migration where translating a string would have
    silently changed what the code *does*: with Chinese copy, the sniff fails,
    and a search that is still running is drawn as a result that found nothing.
    So the branch is asserted to be key-based, and the module is asserted to
    carry no prose-sniffing left.
    """
    paper_list = strip_js_comments((JS_DIR / "paperList.js").read_text(encoding="utf-8"))
    match = re.search(r"function getPaperStateDetails\(.*?\n\}", paper_list, re.DOTALL)
    assert match, "getPaperStateDetails() is missing from paperList.js"
    body = match.group(0)
    assert not re.search(r"/searching|generating/i", body), (
        "the loading branch is decided by matching English prose, so a Chinese "
        "reader would be shown an empty result while the search is still running"
    )
    assert "text.includes(" not in body and "title.includes(" not in body, (
        "the no-match branch is decided by matching a resolved sentence rather "
        "than the key that names it"
    )
    assert "emptyKey === NO_MATCH_KEY" in body, (
        "the no-match branch must compare keys — a sentence comparison would "
        "make the empty state depend on the interface language"
    )
    # The two loading keys and the no-match key are named at module level, next
    # to the decision that consumes them — so a key that was inlined into the
    # branch, or dropped from the list, fails here rather than silently making
    # one of the three empty states unreachable.
    module = (JS_DIR / "paperList.js").read_text(encoding="utf-8")
    loading = re.search(r"const LOADING_STATE_KEYS = \[(.*?)\];", module, re.DOTALL)
    assert loading, "paperList.js no longer declares the keys its loading state branches on"
    assert loading.group(1).count("'") == 4, (
        f"expected the two loading keys in LOADING_STATE_KEYS, got {loading.group(1)!r}"
    )
    for key in (
        "papers.state.search.loading.title",
        "papers.state.recommend.loading.title",
    ):
        assert key in loading.group(1), (
            f"{key} is not one of the keys the loading decision branches on, so "
            "a running search would be drawn as an empty result"
        )
    assert "const NO_MATCH_KEY = 'papers.state.search.none.title';" in module, (
        "the no-match key must be the one the no-match branch compares against"
    )
    # `papers.state.none.title` is the *default* rather than a branch: it is what an
    # empty key resolves to, and the two cancel paths pass `''` on purpose. So
    # it is reached by the absence of a key, and is pinned as such.
    assert "t(emptyKey || 'papers.state.none.title')" in body, (
        "the fallback empty state must still be applied when no key is given — "
        "the cancel paths pass an empty key on purpose"
    )
    branched = {key.strip().strip("'") for key in loading.group(1).split(",") if key.strip()}
    branched.add("papers.state.search.none.title")
    # Every empty state a caller names must be one the branch recognises.
    # Anything else silently renders as the ordinary "nothing here" state, which
    # is how a running search would come to look like a search that found
    # nothing — and nothing else in this file can see it, because both keys are
    # in the catalogue and both resolve.
    #
    # The keys the branch does not claim are the ordinary empty result by
    # construction: that is what the function returns when no branch matches the
    # key (each panel reaches its own recommendation/search empty copy through
    # the elementId branch). They are named so they cannot be mistaken for a
    # branch that has gone missing.
    ordinary = {
        "papers.state.search.idle.title",
        "papers.state.recommend.idle.title",
        "papers.state.search.prompt.title",
        "papers.state.recommend.none.title",
    }
    asked = set()
    for caller in ("papers.js", "paperList.js"):
        source = strip_js_comments((JS_DIR / caller).read_text(encoding="utf-8"))
        for call in re.findall(r"renderPaperList\((.*?)\);", source, re.DOTALL):
            # The third argument is the empty state. Split on commas that are not
            # inside a string, so a key cannot be cut in half.
            parts, current, quote = [], "", None
            for char in call:
                if quote:
                    current += char
                    if char == quote:
                        quote = None
                    continue
                if char in "'\"":
                    quote = char
                    current += char
                    continue
                if char == ",":
                    parts.append(current.strip())
                    current = ""
                    continue
                current += char
            parts.append(current.strip())
            assert len(parts) >= 3, f"renderPaperList({call!r}) has no empty-state argument"
            key = parts[2].strip()
            if key.startswith("'") and key.endswith("'"):
                asked.add(key.strip("'"))
    assert asked, "no caller names any empty state by key"
    recognised = branched | ordinary
    assert not asked - recognised, (
        "a caller asks for empty states no branch of getPaperStateDetails "
        f"recognises: {sorted(asked - recognised)} — they would all render as "
        "the ordinary empty result"
    )
    # And the reverse: every branch is actually reachable, so a state cannot be
    # unreachable while its copy sits in the catalogue looking covered.
    unused = sorted(branched - asked)
    assert not unused, (
        "getPaperStateDetails branches on empty states no caller ever asks "
        f"for: {unused} — a dead branch is a state that cannot be drawn"
    )


# --------------------------------------------------------------------------
# The composition the runtime must adopt
# --------------------------------------------------------------------------


def test_the_runtime_derives_its_error_copy_from_the_catalogue():
    """`runtime.js` asks `i18n.js` what to say, and keeps `config.js` for policy.

    The split is the whole design: `config.js` answers "is retrying worthwhile?"
    and `i18n.js` answers "what do I say?". If `normalizeFailure` went back to
    reading a sentence out of `config.js`, the default install would show Chinese
    errors under any interface, and an English reader who switched would still
    get them.
    """
    runtime = (JS_DIR / "runtime.js").read_text(encoding="utf-8")
    assert "from './i18n.js'" in runtime
    match = re.search(r"function normalizeFailure\(.*?\n\}", runtime, re.DOTALL)
    assert match, "normalizeFailure() is missing from runtime.js"
    body = match.group(0)
    assert "errorPresentation(" in body, (
        "normalizeFailure() must take its reader-facing copy from errorPresentation()"
    )
    assert "ERROR_PRESENTATIONS" in body, (
        "normalizeFailure() must still read ERROR_PRESENTATIONS for the retry policy"
    )
    # The server's prose may be carried, but never as the headline.
    assert "title: copy.title" in body
    assert "message: copy.message" in body
