import { state } from './state.js';
import { sanitizeUrl } from './sanitize.js';
import { setControlDisabled } from './dom.js';
import {
    POSSIBLE_DUPLICATE_BADGE,
    RECOMMENDATION_CONFIDENCE_LABELS,
    RECOMMENDATION_NO_REASONS,
    RECOMMENDATION_REASONS_HEADING,
    RECOMMENDATION_UNKNOWN_CONFIDENCE
} from './config.js';
import { resetRecommendationSignals } from './recommendationSignals.js';
import { t } from './i18n.js';

/**
 * Pure rendering for search results, recommendations and their shared card
 * chrome. No network calls live here, which keeps it importable from the
 * reading-queue and analysis layers without creating an import cycle.
 *
 * The per-card provenance — one `reasons` list per recommendation, each naming
 * the signal that reached the paper — is rendered here rather than in the
 * recommendation controller, because it is part of the card's chrome and has to
 * be drawn the same way whether the row came from a search or a
 * recommendation. `search_papers` rows carry none, so a search card simply
 * does not get the block: `item.reasons` is absent, not empty-but-present, and
 * the heading is withheld rather than shown over an empty list.
 *
 * **On language.** Every sentence this module writes is the interface's, so
 * every one is a catalogue lookup resolved at render time rather than a
 * Chinese-only constant read out of `config.js`. Two things are deliberately
 * not translated and are marked where they appear: the server's own `label_zh`
 * and `reason_zh`, which are the payload's words, and the empty-state
 * *decisions*, which compare resolved sentences rather than English prose.
 */

/**
 * A catalogue value when this build has one, and the `config.js` constant when
 * it does not.
 *
 * The same floor `evidenceViewer.js` uses, for the same reason: these labels
 * were Chinese-only constants before the catalogue carried them, so a key that
 * disappeared should degrade to the sentence the card was written with rather
 * than to a dotted key sitting beside a citation.
 */
function localised(key, floor) {
    const value = t(key);
    return value === key ? floor : value;
}

const SIGNAL_LABEL_KEY_PREFIX = 'recommend.signal.';

function labelForSignal(signal, labelZh) {
    const raw = String(signal || '');
    // The catalogue is consulted *before* `label_zh`, and not after it. The
    // server's field is Chinese by construction, so a signal this build can
    // name would stay Chinese on an English page for as long as the server's
    // copy was preferred; `label_zh` is what an unrecognised signal degrades
    // to, exactly as the reason code does in the evidence panel.
    const named = raw ? localised(`${SIGNAL_LABEL_KEY_PREFIX}${raw}`, '') : '';
    return String(named || labelZh || raw || t('recommend.signal.fallback'));
}

/**
 * One signal's reason: its Chinese label, the sentence the backend wrote for
 * it, which provider produced it, and how far the evidence goes.
 *
 * `reason_zh` is the sentence that was actually generated for *this* hit, not a
 * label reconstructed by the client from the set of papers it happens to have
 * grouped — that reconstruction is how a "被引" chip ends up on a paper that
 * was actually reached by a topic search. So the client's only job here is to
 * show what it was given, and to keep the confidence qualifier attached when
 * one is present: `declared` and `derived` prove different things and only the
 * first is currently authoritative.
 */
function buildReasonItem(reason) {
    const item = document.createElement('li');
    item.className = 'paper-reason-item';
    item.dataset.signal = String(reason?.signal || '');
    item.dataset.source = String(reason?.source || '');

    const label = document.createElement('span');
    label.className = 'paper-reason-label';
    label.textContent = labelForSignal(reason?.signal, reason?.label_zh);
    item.appendChild(label);

    const text = document.createElement('span');
    text.className = 'paper-reason-text';
    text.textContent = String(reason?.reason_zh || '');
    item.appendChild(text);

    const meta = document.createElement('span');
    meta.className = 'paper-reason-meta';
    const bits = [];
    if (reason?.source) {
        bits.push(String(reason.source));
    }
    const confidence = String(reason?.confidence || '');
    if (confidence) {
        item.dataset.confidence = confidence;
        // The qualifier that decides how far the evidence goes is interface
        // copy, so it is looked up by its code rather than read out of the
        // Chinese-only table in `config.js`.
        bits.push(localised(
            `recommend.confidence.${confidence}`,
            RECOMMENDATION_CONFIDENCE_LABELS[confidence] || RECOMMENDATION_UNKNOWN_CONFIDENCE
        ));
    }
    meta.textContent = bits.join(' · ');
    if (bits.length) {
        item.appendChild(meta);
    }
    return item;
}

/**
 * The "why is this here" block on a recommendation card.
 *
 * Returns `null` when the row carries no reasons, so the caller can skip the
 * heading as well. A heading reading "为什么推荐" above nothing is worse than
 * no heading: it promises an explanation the row does not have.
 */
function buildRecommendationReasons(reasons) {
    const entries = Array.isArray(reasons) ? reasons.filter(Boolean) : [];
    if (!entries.length) {
        return null;
    }

    const block = document.createElement('div');
    block.className = 'paper-reasons';
    block.dataset.count = String(entries.length);

    const heading = document.createElement('p');
    heading.className = 'paper-reasons-heading';
    heading.textContent = localised('recommend.reasons_heading', RECOMMENDATION_REASONS_HEADING);
    block.appendChild(heading);

    const list = document.createElement('ul');
    list.className = 'paper-reason-list';
    entries.forEach(reason => list.appendChild(buildReasonItem(reason)));
    block.appendChild(list);
    return block;
}

/**
 * The per-card marker for a row that is *possibly* a duplicate of another.
 *
 * Both rows of a suspect pair carry this (the backend does that deliberately,
 * so neither looks confirmed), and the run-level panel in
 * `recommendationSignals.js` names both. The badge here is what stops a reader
 * who scrolls only the cards from seeing one row of a suspected pair as an
 * ordinary result.
 */
function buildPossibleDuplicateNote(pairs) {
    const entries = Array.isArray(pairs) ? pairs.filter(Boolean) : [];
    if (!entries.length) {
        return null;
    }

    const note = document.createElement('div');
    note.className = 'paper-duplicate-note';
    note.dataset.state = 'possible';
    note.dataset.count = String(entries.length);

    const badge = document.createElement('span');
    badge.className = 'paper-duplicate-badge';
    badge.textContent = localised('recommend.duplicate.badge', POSSIBLE_DUPLICATE_BADGE);
    note.appendChild(badge);

    const detail = document.createElement('span');
    detail.className = 'paper-duplicate-text';
    // Names the other title, so the badge on one card points at the row it is
    // being confused with rather than raising a question the reader cannot
    // resolve from where they are looking. The titles are the payload's words
    // and are interpolated, never translated; only the sentence around them is
    // the interface's.
    const otherTitles = entries
        .map(pair => String(pair?.title || '').trim())
        .filter(Boolean);
    detail.textContent = otherTitles.length
        ? t('papers.duplicate.named', { titles: otherTitles.join(t('papers.duplicate.join')) })
        : t('papers.duplicate.unnamed');
    note.appendChild(detail);
    return note;
}

function buildPaperActionKey(item) {
    return String(item.paper_id || item.pdf_url || item.url || item.title || '').trim();
}

/**
 * The keys whose resolved sentence means "work is happening", and the key whose
 * sentence means "the search ran and matched nothing".
 *
 * Named rather than matched, because the reason they are here at all is that
 * the *decision* cannot depend on the language: the old code sniffed the
 * English prose (`/searching|generating/i`, `includes('No matching')`) to
 * decide which empty state to draw. Once these are catalogue lookups the
 * sentence a Chinese reader sees would fail that sniff and a running search
 * would be drawn as an empty result — the same false reading the states exist
 * to prevent, arrived at by translating the copy. Branching on the key keeps
 * exactly the behaviour the prose gave, in both languages.
 */
const LOADING_STATE_KEYS = ['papers.state.search.loading.title', 'papers.state.recommend.loading.title'];
const NO_MATCH_KEY = 'papers.state.search.none.title';

function getPaperStateDetails(elementId, emptyKey) {
    const title = t(emptyKey || 'papers.state.none.title');
    const loading = LOADING_STATE_KEYS.includes(emptyKey);
    if (loading) {
        return {
            tone: 'loading',
            title,
            body: elementId === 'paperRecommendations'
                ? t('papers.state.recommend.loading.body')
                : t('papers.state.search.loading.body')
        };
    }
    if (elementId === 'paperRecommendations') {
        return {
            tone: 'empty',
            title,
            body: t('papers.state.recommend.empty.body')
        };
    }
    return {
        tone: 'empty',
        title,
        body: emptyKey === NO_MATCH_KEY
            ? t('papers.state.search.narrow.body')
            : t('papers.state.search.empty.body')
    };
}

/**
 * Draw one panel: either its empty state, or a card per row.
 *
 * `emptyKey` is a catalogue key rather than a sentence, because
 * {@link getPaperStateDetails} decides which empty state to draw from it — see
 * that function for why the decision cannot read prose. `meta` carries the line
 * above the panel, whose own text the callers compose (and, for a search, from
 * the server's rewrite fields, which stay the payload's words).
 */
function renderPaperList(elementId, items, emptyKey, meta) {
    const container = document.getElementById(elementId);
    if (!container) return;
    const normalizedItems = Array.isArray(items) ? items : [];
    const allowImport = elementId === 'paperSearchResults';
    if (!normalizedItems.length) {
        const state = getPaperStateDetails(elementId, emptyKey);
        const stateElement = document.createElement('div');
        stateElement.className = `paper-state ${state.tone === 'loading' ? 'loading-state' : ''}`.trim();
        const icon = document.createElement('span');
        icon.className = 'paper-state-icon';
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = state.tone === 'loading' ? '…' : '⌕';
        const body = document.createElement('div');
        const title = document.createElement('p');
        title.className = 'paper-state-title';
        title.textContent = state.title;
        const description = document.createElement('p');
        description.className = 'paper-state-body';
        description.textContent = state.body;
        body.append(title, description);
        stateElement.append(icon, body);
        container.replaceChildren(stateElement);
    } else {
        const cards = normalizedItems.map((item, index) => {
            const article = document.createElement('article');
            article.className = 'paper-card';

            const titleWrap = document.createElement('div');
            titleWrap.className = 'paper-card-title';
            const titleLink = document.createElement('a');
            titleLink.href = sanitizeUrl(item.url || item.pdf_url || '#');
            titleLink.target = '_blank';
            titleLink.rel = 'noopener noreferrer';
            titleLink.referrerPolicy = 'strict-origin-when-cross-origin';
            titleLink.textContent = item.title || t('papers.card.untitled');
            titleWrap.appendChild(titleLink);

            const tags = document.createElement('div');
            tags.className = 'paper-card-tags';
            [item.source, item.year, item.venue].filter(Boolean).forEach(tag => {
                const tagElement = document.createElement('span');
                tagElement.className = 'paper-tag';
                tagElement.textContent = String(tag);
                tags.appendChild(tagElement);
            });

            const authors = document.createElement('div');
            authors.className = 'paper-authors';
            authors.textContent = Array.isArray(item.authors) && item.authors.length
                ? item.authors.join(', ')
                : t('papers.card.unknown_authors');

            const abstract = document.createElement('div');
            abstract.className = 'paper-abstract';
            // `abstract` is the paper own text and is shown verbatim; only the
            // placeholder for a paper that has none is the interface.
            abstract.textContent = item.abstract || t('papers.card.no_abstract');

            article.append(titleWrap, tags, authors, abstract);

            // Per-signal provenance. Present on recommendation rows (T3.2) and
            // absent on search rows; `buildRecommendationReasons` returns null
            // for the latter so the heading is not drawn over nothing.
            const reasonsBlock = buildRecommendationReasons(item.reasons);
            if (reasonsBlock) {
                article.appendChild(reasonsBlock);
            }
            // ...and the legacy single-sentence `reason`, for a search row that
            // carries one. Recommendation rows have both, so both are shown:
            // the sentence is the summary, the list is the evidence for it.
            if (item.reason) {
                const reason = document.createElement('div');
                reason.className = 'paper-reason';
                const label = document.createElement('strong');
                label.textContent = t('papers.card.why');
                reason.append(label, ` ${item.reason}`);
                article.appendChild(reason);
            }
            const duplicateNote = buildPossibleDuplicateNote(item.possible_duplicates);
            if (duplicateNote) {
                article.appendChild(duplicateNote);
            }

            const links = document.createElement('div');
            links.className = 'paper-links';
            // Keyed rather than spelled: the same four labels appear on every
            // card, and `PDF` is the one that is not a word in either language —
            // it stays a catalogue value so the Chinese copy can say 文件.
            [
                ['papers.card.open', item.url, 'link'],
                ['papers.card.pdf', item.pdf_url, 'link'],
                ['papers.card.download', item.pdf_url || item.url, 'download'],
                ['papers.card.save', true, 'save']
            ].forEach(([labelKey, value, action]) => {
                if (!value) return;
                const label = t(labelKey);
                if (action === 'link') {
                    const link = document.createElement('a');
                    link.className = 'paper-link';
                    link.href = sanitizeUrl(value);
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    link.referrerPolicy = 'strict-origin-when-cross-origin';
                    link.textContent = label;
                    links.appendChild(link);
                    return;
                }
                const button = document.createElement('button');
                button.className = 'paper-link';
                button.type = 'button';
                button.dataset.paperAction = action;
                button.dataset.paperList = elementId;
                button.dataset.paperIndex = String(index);
                button.textContent = label;
                links.appendChild(button);
            });

            if (allowImport) {
                const actionKey = buildPaperActionKey(item) || String(index);
                const addButtonDisabled = state.currentImportPaperKey === actionKey;
                const addButton = document.createElement('button');
                addButton.className = 'paper-link';
                addButton.type = 'button';
                addButton.dataset.paperAction = 'add';
                addButton.dataset.paperList = elementId;
                addButton.dataset.paperIndex = String(index);
                addButton.setAttribute('aria-disabled', String(addButtonDisabled));
                addButton.disabled = addButtonDisabled;
                addButton.textContent = addButtonDisabled ? t('papers.card.adding') : t('papers.card.add');
                links.appendChild(addButton);
            }

            article.appendChild(links);
            return article;
        });
        container.replaceChildren(...cards);
    }

    if (meta) {
        const resolvedText = meta.text || '';
        if (meta.elementId === 'paperSearchMeta') {
            state.currentPaperSearchMetaText = resolvedText;
        }
        if (meta.elementId === 'paperRecommendationMeta') {
            state.currentPaperRecommendationMetaText = resolvedText;
        }
        const metaElement = document.getElementById(meta.elementId);
        if (metaElement) {
            metaElement.textContent = resolvedText;
        }
    }
}

function resetPaperPanels() {
    state.currentPaperSearchResults = [];
    state.currentPaperRecommendations = [];
    // The same two sentences the static markup resolves to, reached by the same
    // keys — they were duplicated here as literals, so a language switch
    // repainted the static line and left the reset one behind.
    state.currentPaperSearchMetaText = t('search.meta');
    state.currentPaperRecommendationMetaText = t('result.recommend_meta');
    // A new analysis invalidates the previous run signal status: those
    // `unavailable` rows and duplicate pairs described a *different* paper, and
    // leaving them above a fresh result would attribute one paper coverage to
    // another. Cleared in the same breath as the rows themselves.
    state.currentRecommendationSignals = [];
    state.currentRecommendationDuplicates = [];
    resetRecommendationSignals();
    renderPaperList('paperSearchResults', [], 'papers.state.search.idle.title', { elementId: 'paperSearchMeta', text: state.currentPaperSearchMetaText });
    renderPaperList('paperRecommendations', [], 'papers.state.recommend.idle.title', { elementId: 'paperRecommendationMeta', text: state.currentPaperRecommendationMetaText });
    setRecommendEnabled(Boolean(state.currentAnalysisResult && state.currentSessionId));
}

function setRecommendEnabled(enabled) {
    setControlDisabled(document.getElementById('recommendBtn'), !enabled);
}

function getPaperListByElementId(elementId) {
    if (elementId === 'paperSearchResults') return state.currentPaperSearchResults;
    if (elementId === 'paperRecommendations') return state.currentPaperRecommendations;
    return [];
}


export { buildPaperActionKey };
export { buildPossibleDuplicateNote };
export { buildRecommendationReasons };
export { getPaperListByElementId };
export { getPaperStateDetails };
export { renderPaperList };
export { resetPaperPanels };
export { setRecommendEnabled };
