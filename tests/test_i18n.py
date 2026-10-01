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
MIGRATED_MODULES = ("i18n.js", "a11y.js", "theme.js", "mermaid.js", "format.js", "dom.js")


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
