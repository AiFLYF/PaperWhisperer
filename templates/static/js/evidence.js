import {
    EVIDENCE_API_PATH,
    EVIDENCE_DEEP_LINK_PARAMS,
    EVIDENCE_SHEET_MEDIA_QUERY,
    NO_TARGET_MESSAGES,
    PRESENTATION_DIRECT_QUOTE,
    TARGET_CAPABILITY_LABELS
} from './config.js';
import { state } from './state.js';
import { apiRequest, buildApiError } from './api.js';
import { notifyFailure } from './runtime.js';
import { sanitizeTableHtml } from './sanitize.js';
import { scrollElementIntoView, updateStatus } from './dom.js';
import { onLocaleChange, t } from './i18n.js';
import { describeClaim, isDirectQuote, renderClaim } from './claims.js';
import { renderEvidenceView } from './evidenceViewer.js';

/**
 * The evidence panel: citation chips, the passage behind a claim, and every
 * state in which that passage cannot be shown.
 *
 * The organising rule is T1.9's hard requirement — "打开不离开问题上下文" —
 * and it decides nearly every design choice below. Opening evidence is an
 * *overlay*, never a navigation and never a re-render of the page:
 *
 * * The question in the input is never touched. No module here reads or writes
 *   the composer, and the conversation DOM is not re-rendered to show a
 *   citation — the chip's own DOM subtree is untouched too, so the answer
 *   behind it is still exactly where it was.
 * * The window scroll offset is captured on open and restored on close if
 *   anything moved it (`state.evidenceScrollAnchor`). This matters because the
 *   desktop layout does reflow: the panel is genuinely side by side, so the
 *   container is widened and padded rather than covered, and a reflow can
 *   change the document height under the reader. Restoring the anchor is what
 *   makes that reflow cost the reader nothing.
 * * The body is never scroll-locked. The usual mobile-sheet technique pins the
 *   body with `position: fixed; top: -scrollY` and restores it on close, which
 *   is a scroll jump wearing a disguise — a reflow of the whole document at the
 *   moment it closes. The sheet here is a fixed overlay with its own scroll
 *   region (`overscroll-behavior: contain`, so scrolling inside it does not
 *   chain to the page), and the page behind simply keeps its offset.
 * * Focus is returned to the chip that opened the panel. The chip is still in
 *   the document, so a keyboard user lands back on the control they pressed
 *   rather than at the top of the page.
 *
 * The second rule is that a degraded state is a *distinct visible element*, not
 * an absence. The API has two very different ways of saying "there is no
 * passage here", and a client that collapses them teaches the reader the wrong
 * thing:
 *
 *   * **the request failed** — no claim about the paper is possible at all;
 *   * **the locator resolved to nothing** — the passage is real and is shown,
 *     but it cannot be pointed at on the page, and the reason is named.
 *
 * These are `#evidenceError` and `#evidenceNoTarget`, they never share an
 * element, and `#evidenceNoTarget` deliberately still renders the snippet. A
 * third case, "loaded and genuinely nothing to show", is `#evidenceEmpty` —
 * distinct again, because conflating it with either of the others is how a
 * broken citation ends up described to the user as a limitation of the paper.
 *
 * ## Following the language switch
 *
 * Every string this module writes is produced at render time, so the interface
 * can only follow a language switch if it is *re-rendered* after one. That is
 * what {@link repaintLocalizedSurfaces} is for, and it is the reason this module
 * remembers what it last rendered into which host: an evidence panel that
 * rendered once and is never re-rendered is a panel showing the previous
 * language beside an interface in the new one, which is worse than no
 * translation at all because the reader has no way to know which parts of the
 * page they can trust.
 *
 * **The passage and the claim text are not translated, and must not be.** They
 * are the *document's* words, not the interface's. A Chinese paper stays
 * Chinese under an English interface — see `documentLocale` in `i18n.js` — so
 * everything below routes through `t()` and nothing that carries document text
 * does.
 */

/**
 * The panel's mutually-exclusive states, and the element each one owns.
 *
 * Five entries, five distinct elements, and the distinctness is the point: two
 * states sharing one node would let whichever rendered last overwrite the
 * other, and the reader would be told the wrong thing about their own paper.
 * The passage region (`#evidenceContent`) is deliberately *not* in this set,
 * because it is wanted in two of these states — see `PASSAGE_VISIBLE_STATES`.
 */
const PANEL_STATE_ELEMENTS = {
    idle: 'evidenceIdle',
    loading: 'evidenceLoading',
    error: 'evidenceError',
    no_visual_targeting: 'evidenceNoTarget',
    empty: 'evidenceEmpty'
};

/** The states in which the resolved passage is still worth showing. */
const PASSAGE_VISIBLE_STATES = new Set(['loaded', 'no_visual_targeting']);

const PASSAGE_ELEMENT_ID = 'evidenceContent';

const FOCUSABLE_SELECTOR = [
    'button:not([disabled])',
    'a[href]',
    '[tabindex]:not([tabindex="-1"])'
].join(', ');

/** Whether the panel is currently a mobile bottom sheet rather than a column. */
function isSheetLayout() {
    return window.matchMedia?.(EVIDENCE_SHEET_MEDIA_QUERY).matches === true;
}

function buildEvidenceUrl(documentId, evidenceId) {
    return EVIDENCE_API_PATH
        .replace('{documentId}', encodeURIComponent(String(documentId || '')))
        .replace('{evidenceId}', encodeURIComponent(String(evidenceId || '')));
}

/**
 * Show exactly one of the mutually-exclusive states, and hide the rest.
 *
 * The passage is handled separately from the set, because "no highlight" and
 * "no passage" are not the same statement and must not occupy the same element.
 */
function setPanelState(stateName) {
    const panel = document.getElementById('evidencePanel');
    if (panel) {
        panel.dataset.state = stateName;
    }

    Object.entries(PANEL_STATE_ELEMENTS).forEach(([name, elementId]) => {
        const element = document.getElementById(elementId);
        if (!element) return;
        element.hidden = name !== stateName;
    });

    const passage = document.getElementById(PASSAGE_ELEMENT_ID);
    if (passage) {
        passage.hidden = !PASSAGE_VISIBLE_STATES.has(stateName);
    }
}

/**
 * Show the warnings the server attached to this evidence.
 *
 * A version mismatch is the one that matters: the passage is real and its
 * coordinates may be stale, and hiding the evidence entirely (an API that 409s
 * on a stale index) would tell the reader less than qualifying it does.
 */
function setPanelWarning(text) {
    const warnings = document.getElementById('evidenceWarnings');
    if (!warnings) return;
    const message = String(text || '').trim();
    if (!message) {
        warnings.replaceChildren();
        warnings.hidden = true;
        warnings.dataset.state = 'none';
        return;
    }
    const paragraph = document.createElement('p');
    paragraph.className = 'evidence-warning-text';
    paragraph.textContent = message;
    warnings.replaceChildren(paragraph);
    warnings.hidden = false;
    warnings.dataset.state = 'warn';
}

/**
 * Put the whole `visual_targeting` verdict on screen, including the reasons.
 *
 * Keyed on `reason_code` and never on the prose, for the same reason the rest
 * of the front end branches on `code`: the vocabulary is the contract, and a
 * sentence can be reworded without breaking a client that reads it.
 */
function renderTargetSummary(visualTargeting) {
    const target = document.getElementById('evidenceTarget');
    if (!target) return;
    target.replaceChildren();

    const kind = String(visualTargeting?.target?.kind || '');
    const label = String(visualTargeting?.target?.page_label || visualTargeting?.target?.label || '');
    if (kind || label) {
        const heading = document.createElement('p');
        heading.className = 'evidence-target-label';
        heading.textContent = kind === 'pdf_page'
            ? t('evidence.page_label', { label })
            : (label || kind);
        target.appendChild(heading);
    }

    const capability = String(visualTargeting?.capability || '');
    if (capability && TARGET_CAPABILITY_LABELS[capability]) {
        const row = document.createElement('p');
        row.className = 'evidence-target-row';
        // The vocabulary in `config.js` decides *whether* this capability is one
        // this build renders; the catalogue decides what it is called. Reading
        // the table directly would say the same sentence in one language in
        // both, which is exactly the mixed-language defect this replaces.
        row.textContent = t(`evidence.capability.${capability}`);
        target.appendChild(row);
    }

    const reasonCode = String(visualTargeting?.reason_code || '');
    if (reasonCode) {
        const note = document.createElement('p');
        note.className = 'evidence-note';
        // A code this build knows is looked up in the catalogue and the
        // server's own sentence is *not* preferred over it: the server writes
        // its reason prose in Chinese, and an English interface showing a
        // Chinese sentence beside an English heading is the defect being fixed.
        // A code this build has never heard of still gets shown — the server's
        // sentence when it sent one, the raw code otherwise — because "this
        // locator could not be pointed at" is the useful part and the specific
        // reason is a refinement.
        note.textContent = NO_TARGET_MESSAGES[reasonCode]
            ? t(`evidence.no_target.${reasonCode}`)
            : (visualTargeting?.reason || reasonCode);
        target.appendChild(note);
    }
}

/** Draw the passage, and the region around it, from the resolved payload. */
function renderEvidencePayload(payload) {
    renderTargetSummary(payload.visual_targeting);
    renderEvidenceView(payload.view);

    const meta = document.getElementById('evidenceSnippetMeta');
    if (meta) {
        const snippet = payload.snippet || {};
        const truncated = snippet.truncated === true;
        meta.textContent = [
            payload.citation_label ? t('evidence.snippet.citation', { label: payload.citation_label }) : '',
            snippet.source === 'chunk_verbatim_text' ? t('evidence.snippet.verbatim') : '',
            truncated ? t('evidence.snippet.truncated') : ''
        ].filter(Boolean).join(' · ');
        meta.hidden = !meta.textContent;
    }

    const snippet = document.getElementById('evidenceSnippet');
    if (snippet) {
        // `textContent`, never `innerHTML`: this is server-prose plus model
        // text, and the panel is the one place a reader is convinced the tool
        // is trustworthy, so it cannot be the place an injected tag lands.
        snippet.textContent = String(payload.snippet?.text || '');
    }

    const table = document.getElementById('evidenceTable');
    if (table) {
        const html = String(payload.table?.html || '');
        table.innerHTML = html ? sanitizeTableHtml(html) : '';
        table.hidden = !html;
    }

    // The two states that are *not* a failure: the locator resolved but cannot
    // be pointed at, and the evidence carries no text at all. Both render, and
    // both are named, so neither can be mistaken for a load failure.
    const targeting = payload.visual_targeting || {};
    if (targeting.available === false) {
        setPanelState('no_visual_targeting');
    } else if (!String(payload.snippet?.text || '').trim()) {
        setPanelState('empty');
    } else {
        setPanelState('loaded');
    }

    const warnings = Array.isArray(payload.warnings) ? payload.warnings.filter(Boolean) : [];
    setPanelWarning(warnings.join(' '));
}

/**
 * Every chip the panel can open, rendered as real buttons.
 *
 * A chip is a promise that something can be pointed at, so only a claim that
 * passed verification gets one. A paraphrase has no verified position, and a
 * chip beside an unverified sentence reads as a citation — which is precisely
 * the impression the release gate exists to prevent.
 *
 * Delegated rather than per-element, because the chips are re-rendered for every
 * answer and a listener per chip would leak with the old DOM.
 */
function renderCitationChips(host, claims) {
    if (!host) return [];
    const verified = claims.filter(claim => isDirectQuote(claim));
    host.replaceChildren();
    rememberRender(chipHosts, host, claims);

    if (!verified.length) {
        host.hidden = true;
        return [];
    }

    const list = document.createElement('div');
    list.className = 'citation-chips';
    list.setAttribute('role', 'group');
    list.setAttribute('aria-label', t('claim.chips_group_label'));

    verified.forEach(claim => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'citation-chip';
        button.dataset.evidenceId = String(claim?.evidence_id || claim?.id || '');
        button.dataset.expanded = 'false';
        button.setAttribute('aria-expanded', 'false');
        button.setAttribute('aria-haspopup', 'dialog');
        // The accessible name carries the verdict as well as the label, so a
        // screen-reader user is told this is a verified quote rather than
        // hearing only the number.
        button.setAttribute('aria-label', describeClaim(claim));
        button.textContent = claim.citation_label || claim.label || t('claim.chip_generic_label');
        list.appendChild(button);
    });

    host.appendChild(list);
    host.hidden = false;
    return verified;
}

/**
 * Mark the chip whose evidence is on screen, so a keyboard user moving back
 * through them can tell which is open.
 */
function syncChipExpansion() {
    const openId = String(state.currentEvidence?.evidence_id || '');
    document.querySelectorAll('.citation-chip').forEach(button => {
        const isOpen = Boolean(openId) && button.dataset.evidenceId === openId;
        button.dataset.expanded = String(isOpen);
        button.setAttribute('aria-expanded', String(isOpen));
    });
}

/** One delegated listener for every chip, however many times they re-render. */
function handleChipClick(event) {
    const button = event.target.closest('.citation-chip');
    if (!button) return;
    openEvidence(state.currentDocumentId, button.dataset.evidenceId, button);
}

function restoreScrollAnchor() {
    const anchor = Number(state.evidenceScrollAnchor);
    if (Number.isFinite(anchor)) {
        window.scrollTo(0, anchor);
    }
}

function focusPanel() {
    // The heading, not the panel: it is the element a screen reader reads to
    // announce what just opened, and `tabindex="-1"` is what makes it
    // programmatically focusable without adding it to the tab order.
    const heading = document.getElementById('evidencePanelHeading');
    if (heading) {
        heading.focus({ preventScroll: true });
        return;
    }
    document.getElementById('evidencePanel')?.focus({ preventScroll: true });
}

/** Confine Tab to the sheet, where it is modal; nothing to confine on desktop. */
function trapFocus(event) {
    if (!isSheetLayout()) return;
    if (event.key !== 'Tab') return;

    const panel = document.getElementById('evidencePanel');
    if (!panel) return;
    const focusable = Array.from(panel.querySelectorAll(FOCUSABLE_SELECTOR));
    if (!focusable.length) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
    }
}

function handlePanelKeydown(event) {
    if (event.key === 'Escape') {
        closeEvidence();
        return;
    }
    trapFocus(event);
}

function openPanelShell() {
    const panel = document.getElementById('evidencePanel');
    if (!panel) {
        return;
    }
    state.evidencePanelOpen = true;
    panel.hidden = false;
    panel.dataset.layout = isSheetLayout() ? 'sheet' : 'column';
    // Modal only as a bottom sheet. As a side-by-side column the conversation
    // behind it stays readable and usable, and declaring it modal would tell a
    // screen reader to hide content the reader is entitled to keep.
    panel.setAttribute('aria-modal', String(isSheetLayout()));
    panel.setAttribute('aria-hidden', 'false');
    document.body.classList.add('evidence-open');
    const backdrop = document.getElementById('evidenceBackdrop');
    if (backdrop) {
        // Only a sheet covers the page, so only a sheet needs something to
        // dismiss it by. A backdrop over a column would dim the very
        // conversation the panel is designed to leave usable.
        backdrop.hidden = isSheetLayout() ? false : true;
    }
    syncChipExpansion();
}

function closeEvidence() {
    const panel = document.getElementById('evidencePanel');
    if (panel) {
        panel.hidden = true;
        panel.dataset.layout = 'closed';
        panel.setAttribute('aria-hidden', 'true');
        panel.setAttribute('aria-modal', 'false');
    }
    const backdrop = document.getElementById('evidenceBackdrop');
    if (backdrop) backdrop.hidden = true;

    document.body.classList.remove('evidence-open');
    state.evidencePanelOpen = false;
    state.currentEvidence = null;
    setPanelState('idle');
    syncChipExpansion();
    restoreScrollAnchor();

    // Back to the control that opened it. `document.contains` rather than a
    // bare call, because a re-render of the answer can have removed the chip
    // between open and close, and focusing a detached node silently moves focus
    // to the body — which reads to the reader as the panel having eaten the
    // page.
    const target = state.evidenceReturnFocusTo;
    if (target && document.contains(target)) {
        target.focus({ preventScroll: true });
    }
    state.evidenceReturnFocusTo = null;
}

/**
 * Open the evidence panel on one citation.
 *
 * The failure path here is the one the taxonomy in `config.js` governs, and it
 * is written so that a request which is *in flight* is visibly distinct from one
 * that has failed: `loading` is set before the request and replaced by `error`
 * only in the `catch`. Collapsing the two — leaving the loading state up
 * forever, or clearing it without saying what happened — is what makes a dead
 * request look like a working one.
 */
async function openEvidence(documentId, evidenceId, trigger) {
    const panel = document.getElementById('evidencePanel');
    if (!panel || !evidenceId) {
        return;
    }

    state.evidenceScrollAnchor = window.scrollY;
    state.evidenceReturnFocusTo = trigger instanceof HTMLElement ? trigger : null;
    openPanelShell();
    setPanelState('loading');
    setPanelWarning('');
    updateStatus(t('evidence.opening'), 'idle');

    // No document id means no evidence URL could be built. The panel says so
    // rather than issuing a request to a path with an empty segment, which the
    // API would reject as a malformed id — an error about a request this client
    // should never have made.
    if (!documentId) {
        state.currentEvidence = null;
        const box = document.getElementById('evidenceError');
        if (box) {
            box.replaceChildren();
            const title = document.createElement('p');
            title.className = 'evidence-state-title';
            title.textContent = t('evidence.no_document_id');
            const detail = document.createElement('p');
            detail.className = 'evidence-state-detail';
            detail.textContent = t('evidence.id_not_published');
            box.append(title, detail);
        }
        setPanelState('error');
        updateStatus(t('evidence.unavailable'), 'error');
        focusPanel();
        restoreScrollAnchor();
        return;
    }

    try {
        const payload = await apiRequest(buildEvidenceUrl(documentId, evidenceId), {
            method: 'GET',
            cache: 'no-store',
            operation: 'evidence'
        });
        // A 200 that still carries an error envelope is a failure the status
        // code alone would have let through. Checking it here rather than
        // trusting `response.ok` is the same reason the runtime branches on
        // `code` and not on prose: the envelope is the contract, and a success
        // status is only half of it.
        if (payload?.error || payload?.detail) {
            throw buildApiError(payload, t('evidence.failed'));
        }
        state.currentEvidence = payload;
        renderEvidencePayload(payload);
        // Mark the chip that is now driving the panel, so a screen-reader user
        // moving back through the chips can tell which one is open.
        syncChipExpansion();
        updateStatus(t('evidence.opened'), 'success');
    } catch (error) {
        state.currentEvidence = null;
        // The code is carried through to the DOM as `data-code`, so the state element of the
        // panel and the taxonomy in the runtime agree about *which*
        // failure this is — `evidence_not_in_document` and `page_out_of_range`
        // read differently, and a reader who cannot tell them apart is being
        // told less than they need.
        const failure = {
            message: error?.message || t('evidence.failed'),
            code: error?.code || '',
            status: error?.status || ''
        };
        const box = document.getElementById('evidenceError');
        if (box) {
            box.replaceChildren();
            const title = document.createElement('p');
            title.className = 'evidence-state-title';
            title.textContent = t('evidence.error_title');
            const detail = document.createElement('p');
            detail.className = 'evidence-state-detail';
            detail.dataset.code = String(failure.code || '');
            detail.textContent = `${failure.message}${failure.status ? ` (${failure.status})` : ''}`;
            box.append(title, detail);
        }
        setPanelWarning('');
        setPanelState('error');
        updateStatus(t('evidence.unavailable'), 'error');
        // Routed through the runtime so a *retryable* failure (the server was
        // briefly down) offers a 重试 and a terminal one (this evidence id does
        // not exist) names the cause instead. The error element of the panel and
        // the notice below are two renderings of the same classification, and
        // they cannot disagree because both read the same `code`.
        notifyFailure(error, { onRetry: () => openEvidence(documentId, evidenceId, trigger) });
    } finally {
        focusPanel();
        restoreScrollAnchor();
    }
}

// --------------------------------------------------------------------------
// Deep links: a citation in an exported file, opened cold
// --------------------------------------------------------------------------

/**
 * Read `#evidence=…&document=…` off the fragment.
 *
 * Both halves are required. A fragment naming an evidence id with no document
 * would send a request the server rejects as a malformed id, which is an error
 * about a link this client could have checked before making — and the reader
 * would be told their citation was broken rather than told the link was.
 */
function readDeepLinkTarget() {
    const hash = String(window.location?.hash || '').replace(/^#/, '');
    if (!hash) return null;
    const params = new URLSearchParams(hash);
    const evidenceId = String(params.get(EVIDENCE_DEEP_LINK_PARAMS.evidence) || '').trim();
    const documentId = String(params.get(EVIDENCE_DEEP_LINK_PARAMS.document) || '').trim();
    if (!evidenceId || !documentId) return null;
    return { evidenceId, documentId };
}

/**
 * Honour a deep link, adopting the document it names.
 *
 * The document id from the fragment wins over the session's, because the
 * reader followed the link deliberately: answering with the current session's
 * document would open a citation from a *different* paper and look correct.
 */
function openDeepLinkedEvidence() {
    const target = readDeepLinkTarget();
    if (!target) return false;
    state.currentDocumentId = target.documentId;
    openEvidence(state.currentDocumentId, target.evidenceId, null);
    return true;
}

// --------------------------------------------------------------------------
// Claims inside an answer
// --------------------------------------------------------------------------

/**
 * Pull the claims out of an answer payload, whatever key they arrive under.
 *
 * The Q&A response shape is being changed, so this probes every list the answer
 * might carry them under and returns an empty list when none is present. An
 * answer with no claims then renders exactly as it did before this work
 * existed, rather than throwing inside a stream handler.
 */
function collectCitations(payload) {
    const candidates = ['citations', 'evidence', 'claims'];
    for (const key of candidates) {
        const value = payload?.[key];
        if (!Array.isArray(value)) {
            continue;
        }
        return value.filter(item => item && typeof item === 'object');
    }
    return [];
}

/**
 * Render a settled answer's claims into their own container.
 *
 * Kept out of the answer's markdown container on purpose: the answer is
 * re-rendered from data on every streamed delta, so splicing verified quote
 * elements into it would put generated HTML in charge of where a quotation mark
 * goes, and would lose them on the next delta.
 */
function renderInlineClaims(host, claims) {
    if (!host) return 0;
    host.replaceChildren();
    const list = Array.isArray(claims) ? claims : [];
    host.dataset.count = String(list.length);
    // Remembered before the early return: a host that renders nothing is still
    // a host the reader is looking at, and a later switch must not leave the
    // previous render standing in the other language.
    rememberRender(claimHosts, host, list);
    if (!list.length) {
        host.hidden = true;
        return 0;
    }
    list.forEach(claim => host.appendChild(renderClaim(claim)));
    host.hidden = false;
    return list.length;
}

/**
 * Hosts this module last rendered claims or chips into, so a language switch
 * can repaint them. One entry per host, the latest render winning, because a
 * streamed answer re-renders the same host on every delta and only the last one
 * is on screen.
 */
const claimHosts = [];
const chipHosts = [];

/** Whether a remembered host is still in the document to be repainted. */
function isAttached(host) {
    if (!host) return false;
    if (typeof host.isConnected === 'boolean') return host.isConnected;
    if (typeof document.contains === 'function') return document.contains(host);
    return true;
}

function rememberRender(registry, host, claims) {
    const index = registry.findIndex(entry => entry.host === host);
    const entry = { host, claims };
    if (index >= 0) {
        registry[index] = entry;
    } else {
        registry.push(entry);
    }
}

/**
 * Re-render every surface this module painted, in the language now active.
 *
 * Subscribed to the locale event from `initializeEvidencePanel`, behind a flag,
 * because that function is callable more than once and a second subscription
 * would double every render.
 *
 * The order matters: chips first, then the panel, so the chip the panel marks
 * as open is the one the reader can actually see marked. Only the claims and the
 * panel's own sentences are repainted — the passage, the claim text and the
 * table are the document's words and are re-written from the same payload
 * unchanged.
 */
function repaintLocalizedSurfaces() {
    claimHosts
        .filter(entry => isAttached(entry.host))
        .forEach(entry => renderInlineClaims(entry.host, entry.claims));
    chipHosts
        .filter(entry => isAttached(entry.host))
        .forEach(entry => renderCitationChips(entry.host, entry.claims));
    if (state.evidencePanelOpen && state.currentEvidence) {
        renderEvidencePayload(state.currentEvidence);
        syncChipExpansion();
    }
}

/** Whether the locale subscription below has already been made. */
let localeBound = false;

function initializeEvidencePanel() {
    const panel = document.getElementById('evidencePanel');
    if (!panel) return;

    document.getElementById('evidenceCloseBtn')?.addEventListener('click', closeEvidence);
    // The backdrop dismisses only when it is a sheet. As a column it is hidden
    // entirely, so the listener is harmless — but a backdrop that dimmed a
    // still-readable conversation would be the opposite of the requirement.
    document.getElementById('evidenceBackdrop')?.addEventListener('click', () => {
        if (isSheetLayout()) closeEvidence();
    });
    document.addEventListener('keydown', handlePanelKeydown);
    document.addEventListener('click', handleChipClick);
    window.addEventListener('hashchange', openDeepLinkedEvidence);

    setPanelState('idle');
    if (!localeBound) {
        localeBound = true;
        onLocaleChange(repaintLocalizedSurfaces);
    }
    // A deep link pasted into a fresh tab has to resolve, so this runs on load
    // and not only on a click.
    openDeepLinkedEvidence();
    return true;
}

export { buildEvidenceUrl };
export { collectCitations };
export { closeEvidence };
export { initializeEvidencePanel };
export { isSheetLayout };
export { openDeepLinkedEvidence };
export { openEvidence };
export { readDeepLinkTarget };
export { renderCitationChips };
export { renderEvidencePayload };
export { renderInlineClaims };
export { repaintLocalizedSurfaces };
export { setPanelState };
export { PANEL_STATE_ELEMENTS };
