import {
    POSSIBLE_DUPLICATE_BASIS_FALLBACK,
    POSSIBLE_DUPLICATE_HEADING,
    POSSIBLE_DUPLICATE_POLICY,
    RECOMMENDATION_SIGNAL_LABELS,
    RECOMMENDATION_SIGNAL_STATES,
    RECOMMENDATION_UNKNOWN_SIGNAL_STATE
} from './config.js';
import { t } from './i18n.js';

/**
 * Resolve a catalogue key, falling back to the default-locale floor when the key
 * is missing. A key that stopped resolving must degrade to a sentence rather than
 * render as a dotted string on screen.
 */
function localised(key, floor) {
    const value = t(key);
    return value === key ? floor : value;
}

const SIGNAL_LABEL_KEY_PREFIX = 'recommend.signal.';

/**
 * The glyph drawn between the two titles of a probable-duplicate pair.
 *
 * A symbol, not copy: `⟷` reads the same in every language the catalogue
 * supports, so it is deliberately a named constant rather than a `t()` key —
 * routing it through the catalogue would invite a translator to "translate" a
 * mark that has no language. It is `aria-hidden` for the same reason it is not
 * keyed: it carries no meaning a screen reader should announce on its own.
 */
const DUPLICATE_SEPARATOR = ' ⟷ ';

/**
 * The two pieces of a recommendation run that describe the *run* rather than any
 * one paper: the per-signal status, and the pairs that might be duplicates.
 *
 * Both exist because of the same asymmetry, which is the module's whole reason
 * to be separate from the paper list. The rows on screen answer "which papers";
 * these answer "how much of that can you trust, and what did you fail to check?"
 * A list of five papers with four working signals and one rate-limited provider
 * looks identical to a list of five papers where the rate-limited provider
 * genuinely had nothing — and the second reading is a claim about somebody's
 * paper that we did not earn. So the status is drawn as its own region, one row
 * per signal, and `unavailable` and `empty` are different elements carrying
 * different sentences (see `RECOMMENDATION_SIGNAL_STATES` in `config.js` for
 * why the `unavailable` copy explicitly denies the "none" reading).
 *
 * **The same rule governs the duplicates.** T3.1 established that a `probable`
 * identity match is a *possibility* and is never silently merged, on the
 * grounds that merging two different papers destroys one irrecoverably while
 * showing one twice costs a glance. That decision is worth nothing unless the
 * user can see it, so the pairs are drawn with both titles, the basis the
 * identity layer matched on, and the reason the merge was *not* performed.
 *
 * This module is pure rendering: no network, no state mutation beyond the
 * reset helper. The network call lives in `papers.js`, and the per-card reasons
 * live in `paperList.js` where the rest of the card chrome is. Everything is
 * built with `createElement` + `textContent`, never `innerHTML`, because
 * `reason_zh` and `note_zh` are server-authored strings that must not become
 * markup.
 */

/** The state a signal falls back to when the API sends one this build has never heard of. */
function stateForStatus(status) {
    return RECOMMENDATION_SIGNAL_STATES[String(status || '')] || RECOMMENDATION_SIGNAL_STATES[RECOMMENDATION_UNKNOWN_SIGNAL_STATE];
}

function labelForSignal(entry) {
    const raw = String(entry?.signal || '');
    // The catalogue is consulted *before* the server's `label_zh`, exactly as in
    // `paperList.js`: that field is Chinese by construction, so preferring it
    // would pin an English reader to Chinese for as long as the server kept
    // sending it. `label_zh` stays as the degradation for a signal this build
    // cannot name — the role it already plays for the reason code.
    const named = raw ? localised(`${SIGNAL_LABEL_KEY_PREFIX}${raw}`, '') : '';
    return String(named || entry?.label_zh || RECOMMENDATION_SIGNAL_LABELS[raw] || raw || t('recommend.signal.fallback'));
}

/**
 * One signal's row: its name, its four-state badge, how many papers it
 * produced, and why.
 *
 * `data-status` is the machine-readable half and it carries the whole point of
 * the module — the tone that separates "there is genuinely nothing" from "we
 * could not check" is attached to the *state*, not to a count, so a `count: 0`
 * on a `unavailable` signal can never be styled like a `count: 0` on an
 * `empty` one.
 */
function buildSignalRow(entry) {
    const state = stateForStatus(entry?.status);

    const row = document.createElement('li');
    row.className = 'signal-row';
    row.dataset.signal = String(entry?.signal || '');
    row.dataset.status = String(entry?.status || RECOMMENDATION_UNKNOWN_SIGNAL_STATE);
    row.dataset.tone = state.tone;

    const head = document.createElement('div');
    head.className = 'signal-row-head';

    const label = document.createElement('span');
    label.className = 'signal-row-label';
    label.textContent = labelForSignal(entry);
    head.appendChild(label);

    const badge = document.createElement('span');
    badge.className = 'signal-row-status';
    badge.dataset.tone = state.tone;
    badge.textContent = state.label;
    head.appendChild(badge);

    const count = document.createElement('span');
    count.className = 'signal-row-count';
    // A count is only meaningful when the provider actually answered. Printing
    // "0 篇" next to an `unavailable` badge would put the "none" reading back
    // into the row through the back door, so the count is omitted for it.
    if (state.tone !== 'unavailable' && state.tone !== 'skipped') {
        count.textContent = `${Number(entry?.count || 0)} 篇`;
    }
    head.appendChild(count);

    const summary = document.createElement('p');
    summary.className = 'signal-row-summary';
    summary.textContent = state.summary;

    row.append(head, summary);

    // The API reason (which provider failed, or which query ran) is the
    // detail under our framing, and is what makes the state actionable.
    const reason = String(entry?.reason_zh || '').trim();
    if (reason) {
        const detail = document.createElement('p');
        detail.className = 'signal-row-reason';
        detail.dataset.role = 'api-reason';
        detail.textContent = reason;
        row.appendChild(detail);
    }
    return row;
}

function buildSignalList(signals) {
    const list = document.createElement('ul');
    list.className = 'signal-list';
    const entries = Array.isArray(signals) ? signals : [];
    entries.forEach(entry => list.appendChild(buildSignalRow(entry)));
    list.dataset.count = String(entries.length);
    return list;
}

/**
 * The probable-duplicate pairs, as a distinct visible region.
 *
 * Both titles are shown, because the whole point is that the user is being
 * asked to judge whether two *named* papers are one. The basis the identity
 * layer matched on travels with it for the same reason: "same title and first
 * author" is a weaker claim than "same DOI", and a reader who cannot see which
 * one fired cannot tell how much to trust the suspicion.
 */
function buildDuplicatePanel(duplicates) {
    const pairs = Array.isArray(duplicates) ? duplicates.filter(Boolean) : [];
    if (!pairs.length) {
        return null;
    }

    const panel = document.createElement('div');
    panel.className = 'duplicate-panel';
    panel.dataset.state = 'possible';
    panel.dataset.count = String(pairs.length);

    const heading = document.createElement('p');
    heading.className = 'duplicate-heading';
    heading.textContent = POSSIBLE_DUPLICATE_HEADING;
    panel.appendChild(heading);

    const policy = document.createElement('p');
    policy.className = 'duplicate-policy';
    policy.textContent = POSSIBLE_DUPLICATE_POLICY;
    panel.appendChild(policy);

    const list = document.createElement('ul');
    list.className = 'duplicate-list';
    pairs.forEach(pair => {
        const item = document.createElement('li');
        item.className = 'duplicate-pair';

        const titles = document.createElement('p');
        titles.className = 'duplicate-titles';
        const first = document.createElement('span');
        first.className = 'duplicate-title';
        first.textContent = String(pair?.title || t('recommend.duplicate.untitled'));
        const separator = document.createElement('span');
        separator.className = 'duplicate-separator';
        separator.setAttribute('aria-hidden', 'true');
        separator.textContent = DUPLICATE_SEPARATOR;
        const second = document.createElement('span');
        second.className = 'duplicate-title';
        second.textContent = String(pair?.other_title || t('recommend.duplicate.untitled'));
        titles.append(first, separator, second);
        item.appendChild(titles);

        const basis = document.createElement('p');
        basis.className = 'duplicate-basis';
        basis.dataset.basis = String(pair?.basis || '');
        basis.textContent = String(pair?.basis_zh || POSSIBLE_DUPLICATE_BASIS_FALLBACK);
        item.appendChild(basis);

        const note = String(pair?.note_zh || '').trim();
        if (note) {
            const detail = document.createElement('p');
            detail.className = 'duplicate-note';
            detail.textContent = note;
            item.appendChild(detail);
        }
        list.appendChild(item);
    });
    panel.appendChild(list);
    return panel;
}

/**
 * Render both run-level regions for a recommendation response.
 *
 * Safe to call with either argument empty, and with neither — it then hides the
 * containers and touches nothing, so a search-only render (which has no signals
 * and no pairs) does not leave a stale "we could not check three signals"
 * banner from a previous recommendation on screen. The single-element ids are
 * literal so `tools/check_frontend.py` can require them in `index.html`: a
 * module that mounts into a container the template never declares returns
 * `null` and shows nothing, which is indistinguishable from "nothing to
 * report" — the exact failure this module exists to prevent.
 */
function renderRecommendationSignals(signals, duplicates) {
    const signalContainer = document.getElementById('recommendationSignals');
    const duplicateContainer = document.getElementById('recommendationDuplicates');
    if (!signalContainer || !duplicateContainer) {
        return 0;
    }

    const entries = Array.isArray(signals) ? signals.filter(Boolean) : [];
    const pairs = Array.isArray(duplicates) ? duplicates.filter(Boolean) : [];

    if (!entries.length) {
        signalContainer.replaceChildren();
        signalContainer.hidden = true;
        signalContainer.dataset.count = '0';
    } else {
        const heading = document.createElement('p');
        heading.className = 'signal-heading';
        heading.textContent = t('recommend.signals_heading');
        signalContainer.replaceChildren(heading, buildSignalList(entries));
        signalContainer.hidden = false;
        signalContainer.dataset.count = String(entries.length);
    }

    const duplicatePanel = buildDuplicatePanel(pairs);
    if (!duplicatePanel) {
        duplicateContainer.replaceChildren();
        duplicateContainer.hidden = true;
        duplicateContainer.dataset.count = '0';
    } else {
        duplicateContainer.replaceChildren(duplicatePanel);
        duplicateContainer.hidden = false;
        duplicateContainer.dataset.count = String(pairs.length);
    }

    return entries.length + pairs.length;
}

/**
 * Clear both regions.
 *
 * Called when a new analysis starts and when the paper panels reset, so a
 * signal status describing a *previous* paper never sits above a new one.
 */
function resetRecommendationSignals() {
    return renderRecommendationSignals([], []);
}

export {
    buildDuplicatePanel,
    buildSignalRow,
    renderRecommendationSignals,
    resetRecommendationSignals,
    stateForStatus
};
