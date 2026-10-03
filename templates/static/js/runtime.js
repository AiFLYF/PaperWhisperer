/**
 * The failure surface: one place a failure is described, one place it is
 * shown, and one place a retry is decided.
 *
 * This module exists because the alternative was already the problem. Before
 * it, each calling module did whatever it liked with a failure: some called
 * `showError` with the server's prose, some cleared a panel, some printed a
 * status chip, and one path — a stream that died while the page still believed
 * it was working — printed nothing at all. The variety of backend conditions
 * (minutes-long, streaming, partially failed, mid-reparse, no document, no page
 * images, unreachable) was being absorbed by whatever the calling module
 * happened to do, so the *experience* of a failure varied with the file it came
 * from. For a product whose claim is that you can trust this, a UI that cannot
 * tell "working" from "died" is the contradiction, not a cosmetic problem.
 *
 * ## The shape of a failure
 *
 * Four states, and the distinction between the first and the rest is the whole
 * point. A request still in flight is not a failure and is never presented as
 * one. A request that failed and might succeed again is not the same as one
 * that failed and would fail identically, and a server that cannot be reached
 * is neither. Four different messages, because the reader's next move is
 * different in each case.
 *
 * ## What is not swallowed
 *
 * Three things, by name, because "never swallow an error" is not actionable
 * without saying which errors:
 *
 * 1. Anything the caller can retry. It gets a notice and, when the kind says a
 *    retry can help, a button that re-issues *the same request* — never a
 *    generic reload.
 * 2. Anything that destroyed work the user can see. A failed analysis that
 *    blanked a populated workspace is reported as a failure with the previous
 *    workspace named as restored, not as a success with an empty screen.
 * 3. Anything with no known cause. Rendered with the raw code and an explicit
 *    statement that repeating the request will not change it, rather than
 *    dropped. A silent unknown is the failure mode this whole module is built
 *    against.
 *
 * ## Input preservation
 *
 * `capturePreservedFields` / `restorePreservedFields` snapshot the composer's
 * text and the upload selection around a request. The rule: capture before the
 * request, restore only on failure, and never clear a field the reader did not
 * type. Clearing a composer optimistically and then reporting a failure leaves
 * the user retyping what they already wrote, which is the opposite of humane.
 *
 * No framework, no build step, no dependency: `fetch`, `AbortController`,
 * `TextDecoder` and the DOM are all the browser has always had.
 */

import {
    FAILURE_KINDS,
    NOTICE_POLICY,
    // Imported for its *tones* only. Its sentences moved into the catalogue
    // (`runtime.connectivity.*`, `runtime.offline.*`) because a connectivity
    // chip is the first thing on screen after a disconnect, and it was one of
    // the two strings that stayed Chinese under an English interface. The tone
    // is policy — which colours mean what — and stays where the stylesheet's
    // vocabulary is.
    OFFLINE_STATUS_COPY,
    PRESERVED_FIELDS,
    PROGRESS_STEPS,
    RETRY_LABEL,
    UNKNOWN_FAILURE_KIND,
    ERROR_PRESENTATIONS,
    NOTICE_CONTAINER_ID,
    CONNECTIVITY_CHIP_ID
} from './config.js';
import { errorPresentation, t } from './i18n.js';

// --------------------------------------------------------------------------
// The failure object
// --------------------------------------------------------------------------

/**
 * The one shape a failure takes, whatever caught it.
 *
 * `code` is the backend's machine-readable code and is what every decision is
 * made from. `message` is for the reader and is never inspected. Keeping them
 * as separate fields is what makes it possible to change one without breaking
 * the other: rewording the prose cannot accidentally make a terminal error look
 * retryable, because nothing reads the prose.
 *
 * `kind` is resolved once, here, and never re-derived downstream. A notice that
 * decided its own kind from the message would be exactly the pattern-matching
 * this design exists to remove.
 *
 * @param {object} failure
 * @param {string} [failure.code]      the backend's `code`, or a `client_*` one
 * @param {number} [failure.status]    the HTTP status, when there was a response
 * @param {string} [failure.message]   the server's Chinese prose, verbatim
 * @param {string} [failure.operation] what the reader was doing, for the title
 * @returns {object} a normalised failure
 */
function normalizeFailure(failure = {}) {
    const code = String(failure.code || '').trim();
    const presentation = ERROR_PRESENTATIONS[code];
    const kind = presentation ? presentation.kind : UNKNOWN_FAILURE_KIND;
    // The reader-facing copy comes from the i18n catalogue, keyed by `code` and
    // authored per language. `ERROR_PRESENTATIONS` above is policy only — it
    // answers "may retrying help?" and never "what do I say?" — so this build's
    // Chinese sentences are the *zh-CN* values, and the English ones live
    // beside them in the catalogue rather than here. The server's own Chinese
    // prose arrives as `detail` and is never used as the headline, because
    // surfacing it under an English interface is the mixed-language defect this
    // split exists to prevent.
    const copy = errorPresentation(code, failure.message);

    return {
        code: copy.code,
        known: presentation ? true : copy.known,
        status: Number.isFinite(failure.status) ? failure.status : null,
        kind,
        title: copy.title,
        message: copy.message,
        hint: copy.hint || '',
        // The server's own sentence, kept but never parsed and never used as
        // the primary message. Its language is whatever the server writes,
        // which is not assumed to be the reader's.
        detail: copy.serverDetail,
        operation: String(failure.operation || '').trim(),
        cause: failure.cause || null
    };
}

// --------------------------------------------------------------------------
// The notification surface
// --------------------------------------------------------------------------

const notices = new Map();
let noticeSequence = 0;
let container = null;

/** Where notices mount. Created on first use so nothing is required of the template. */
function ensureContainer() {
    if (container && container.isConnected) return container;
    const host = document.getElementById('error');
    if (!host || !host.parentNode) return null;

    container = document.createElement('div');
    container.id = NOTICE_CONTAINER_ID;
    container.className = 'runtime-notices';
    // Polite rather than assertive: a notice is a status update about work the
    // reader initiated and is watching. Assertive would interrupt whatever a
    // screen reader is currently saying, for something the reader already knows
    // is happening.
    container.setAttribute('role', 'status');
    container.setAttribute('aria-live', 'polite');
    host.parentNode.insertBefore(container, host);
    return container;
}

/** The chip that reports connectivity. Reuses the existing status chip's shape. */
function ensureConnectivityChip() {
    const host = document.getElementById('statusChip');
    if (!host || !host.parentNode) return null;
    const existing = document.getElementById(CONNECTIVITY_CHIP_ID);
    if (existing) return existing;

    const chip = document.createElement('div');
    chip.id = CONNECTIVITY_CHIP_ID;
    chip.className = 'status-chip';
    chip.dataset.tone = 'idle';
    chip.setAttribute('role', 'status');
    chip.setAttribute('aria-live', 'polite');
    chip.textContent = t('runtime.connectivity.online');
    host.parentNode.insertBefore(chip, host);
    return chip;
}

/**
 * Show a failure.
 *
 * The one function in the front end that decides what a failure looks like. It
 * takes either a normalised failure or the raw `{ code, status, message }` an
 * API error carries, and it is the only caller of the retry button's enablement
 * rule.
 *
 * @returns {string|null} the notice id, so the caller can dismiss it on success
 */
function notifyFailure(failure, options = {}) {
    const normalized = failure.kind ? failure : normalizeFailure(failure);
    const policy = NOTICE_POLICY[normalized.kind] || NOTICE_POLICY[FAILURE_KINDS.TERMINAL];
    const host = ensureContainer();
    const noticeId = `notice-${++noticeSequence}`;
    const element = document.createElement('div');
    element.className = `runtime-notice runtime-notice-${normalized.kind}`;
    element.dataset.kind = normalized.kind;
    element.dataset.code = normalized.code;
    element.dataset.noticeId = noticeId;

    const title = document.createElement('p');
    title.className = 'runtime-notice-title';
    title.textContent = normalized.title;
    element.appendChild(title);

    const body = document.createElement('p');
    body.className = 'runtime-notice-message';
    body.textContent = normalized.message;
    element.appendChild(body);

    if (normalized.hint) {
        const hint = document.createElement('p');
        hint.className = 'runtime-notice-hint';
        hint.textContent = normalized.hint;
        element.appendChild(hint);
    }

    if (policy.actionable && options.onRetry) {
        element.appendChild(buildRetryControl(noticeId, options.onRetry));
    }

    if (policy.dismissible) {
        element.appendChild(buildDismissControl(noticeId));
    }

    if (host) {
        host.appendChild(element);
    }
    notices.set(noticeId, { element, failure: normalized, onRetry: options.onRetry || null });

    // Only a pending or unreachable state is announced in the status chip: both
    // are things the reader can do something about *right now*, and both are
    // silent otherwise. A terminal failure is fully described in its notice.
    if (normalized.kind === FAILURE_KINDS.UNREACHABLE) {
        setStatusText(t('runtime.connectivity.offline'), 'error');
    }
    return noticeId;
}

/**
 * A retry control, or the reason there isn't one.
 *
 * A `RETRYABLE` failure with a callback gets a real button. Anything else gets
 * nothing rather than a disabled one: a greyed-out "重试" invites the reader to
 * wonder whether the interface is broken, which is a worse message than the
 * absence.
 */
function buildRetryControl(noticeId, onRetry) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'action-btn runtime-notice-retry';
    button.dataset.noticeId = noticeId;
    button.textContent = RETRY_LABEL;
    button.addEventListener('click', () => {
        dismissNotice(noticeId);
        onRetry();
    });
    return button;
}

function buildDismissControl(noticeId) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'action-btn runtime-notice-dismiss';
    button.dataset.noticeId = noticeId;
    // Both the visible label and its accessible name are the interface's, so
    // both are looked up: a hardcoded '关闭' beside a fully English notice is
    // the mixed-language dialog, and here it would be one untranslated word in
    // the one control the reader uses to make the failure go away.
    button.setAttribute('aria-label', t('runtime.dismiss_label'));
    button.textContent = t('runtime.dismiss');
    button.addEventListener('click', () => dismissNotice(noticeId));
    return button;
}

/**
 * Say that something is still happening.
 *
 * The counterpart to `notifyFailure`, and the reason the two are separate. A
 * long run needs to be visible as a *state* rather than as the absence of an
 * error: "分析进行中，已用时 40 秒" and a blank screen are both things a reader
 * has to interpret, and only one of them is true.
 *
 * A pending notice is `role="status"` and polite, and it is dismissible without
 * consequence — there is nothing to act on yet, so offering a button would be
 * an invitation to press something that has no effect.
 */
function notifyPending(title, message, options = {}) {
    const normalized = {
        code: options.code || 'client_pending',
        status: null,
        kind: FAILURE_KINDS.PENDING,
        known: true,
        // A pending notice raised without a title still says *something*, in
        // the reader's language. The old literal was '进行中', which left the
        // only line on screen Chinese under an English interface.
        title: String(title || t('runtime.pending_title')),
        message: String(message || ''),
        hint: '',
        detail: '',
        operation: String(options.operation || '')
    };
    return notifyFailure(normalized, {});
}

function dismissNotice(noticeId) {
    const notice = notices.get(noticeId);
    if (!notice) return false;
    notice.element.remove();
    notices.delete(noticeId);
    return true;
}

/** Remove every notice. Called when a request of the same kind succeeds. */
function clearNotices() {
    notices.forEach(notice => notice.element.remove());
    notices.clear();
}

function setStatusText(text, tone) {
    const chip = document.getElementById('statusChip');
    if (!chip) return;
    chip.textContent = text;
    chip.dataset.tone = tone;
}

// --------------------------------------------------------------------------
// In-flight state
// --------------------------------------------------------------------------

const inFlight = new Map();
let flightSequence = 0;

/**
 * Record that a request is running.
 *
 * The reason this exists is that "in flight" must be a *state* rather than the
 * absence of a failure. A UI that only ever writes on failure cannot distinguish
 * a slow request from a dead one, and the reader is the only one who can see
 * the difference. Making it explicit lets `describeFlight` say "still working"
 * — and lets a stalled stream be noticed, which is the only way a long request
 * is ever going to admit it died.
 *
 * @returns {string} a token to pass to `endFlight`
 */
function beginFlight(operation, options = {}) {
    const token = `flight-${++flightSequence}`;
    const startedAt = Date.now();
    const entry = {
        token,
        operation,
        startedAt,
        // A "总进度未知" marker rather than a percentage. See
        // `PROGRESS_STORY.determinate`: the server reports which step it is on
        // and never how far through it, so any number here would be invented.
        progressKnown: options.progressKnown === true,
        label: options.label || '',
        onStall: options.onStall || null,
        stallTimer: null,
        settled: false
    };

    if (entry.onStall) {
        const delay = Number(options.stallAfterMs) > 0 ? Number(options.stallAfterMs) : 0;
        if (delay > 0) {
            entry.stallTimer = setTimeout(() => {
                if (!entry.settled) entry.onStall(entry);
            }, delay);
        }
    }

    inFlight.set(token, entry);
    return token;
}

/**
 * Record that a request finished, one way or another.
 *
 * Every `beginFlight` has exactly one `endFlight`, in a `finally`. That is not
 * a style preference: a spinner left up after a failure is the specific
 * defect this module was written to remove, and the only reliable way to
 * prevent it is to make "clearing it" part of the same call that "setting it"
 * is.
 */
function endFlight(token, outcome = {}) {
    const entry = inFlight.get(token);
    if (!entry) return false;
    if (entry.stallTimer) clearTimeout(entry.stallTimer);
    entry.settled = true;
    inFlight.delete(token);

    if (outcome.failed) {
        notifyFailure(outcome.failure || { code: 'client_unknown_error' }, outcome);
    } else if (outcome.succeeded) {
        dismissNoticesFor(outcome.scope || operationScope(outcome));
    }
    return true;
}

/** Notices raised for one operation, so a success can retire its own failures. */
function operationScope(outcome) {
    return String(outcome.operation || '');
}

function dismissNoticesFor(scope) {
    if (!scope) return;
    Array.from(notices.entries()).forEach(([noticeId, notice]) => {
        if (notice.failure.operation === scope) dismissNotice(noticeId);
    });
}

/** What is running right now, for the progress story and for tests. */
function activeFlights() {
    return Array.from(inFlight.values()).map(entry => ({
        token: entry.token,
        operation: entry.operation,
        startedAt: entry.startedAt,
        progressKnown: entry.progressKnown,
        label: entry.label
    }));
}

function hasActiveFlight() {
    return inFlight.size > 0;
}

// --------------------------------------------------------------------------
// Progress, told honestly
// --------------------------------------------------------------------------

/**
 * Which step a long operation is on, and how long it has been there.
 *
 * The elapsed time is the part that is always knowable and therefore always
 * shown. The step is what the server reported. The percentage is absent, and
 * its absence is stated — see `PROGRESS_STORY.unknownNote` for why a bar that
 * guessed would be worse than one that admits ignorance.
 */
function describeFlight(token) {
    const entry = inFlight.get(token);
    if (!entry) return null;
    return {
        operation: entry.operation,
        step: entry.label,
        determinate: entry.progressKnown,
        elapsedMs: Date.now() - entry.startedAt,
        steps: PROGRESS_STEPS
    };
}

function formatElapsed(elapsedMs) {
    const totalSeconds = Math.max(0, Math.floor(Number(elapsedMs) / 1000));
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    // Two keys rather than one with a conditional: "1 min 5 s" and "40 s" are
    // different shapes in every language, and a single template would have to
    // carry the branch inside the catalogue. Numbers stay bare here — the
    // catalogue owns the words and the unit, this owns the arithmetic.
    return t(minutes > 0 ? 'runtime.elapsed.minutes' : 'runtime.elapsed.seconds', { minutes, seconds });
}

// --------------------------------------------------------------------------
// Input preservation
// --------------------------------------------------------------------------

/**
 * Snapshot the composer's contents and the upload selection.
 *
 * Called *before* a request, not after it fails: by the time a failure has been
 * reported the optimistic clear has already happened, and there is nothing left
 * to restore. The file is held as a live object rather than a name for the
 * reason given on `PRESERVED_FIELDS.file`.
 */
function capturePreservedFields() {
    const snapshot = {};
    Object.entries(PRESERVED_FIELDS).forEach(([elementId, spec]) => {
        const element = document.getElementById(elementId);
        if (!element) return;
        if (spec.kind === 'checked') {
            snapshot[elementId] = { kind: spec.kind, value: element.checked === true };
            return;
        }
        if (spec.kind === 'file') {
            snapshot[elementId] = { kind: spec.kind, file: element.files?.[0] || null };
            return;
        }
        // Spelled out rather than left as the fall-through, so that adding a
        // fourth kind cannot silently land in the text branch and capture a
        // checkbox as its string value.
        if (spec.kind === 'value') {
            snapshot[elementId] = { kind: spec.kind, value: element.value };
            return;
        }
        snapshot[elementId] = { kind: spec.kind, value: '' };
    });
    return snapshot;
}

/**
 * Put a captured snapshot back.
 *
 * Only fields the reader could have typed are restored, and only the ones
 * whose value actually differs — so a successful request that legitimately
 * cleared a composer (the question is now in the transcript) is not undone.
 * That is why the return value is the set of ids actually written: a caller
 * that cleared a composer on purpose gets `false` and leaves it alone.
 *
 * @returns {string[]} the ids whose value was restored
 */
function restorePreservedFields(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') return [];
    const restored = [];

    Object.entries(snapshot).forEach(([elementId, entry]) => {
        const element = document.getElementById(elementId);
        if (!element || !entry) return;

        if (entry.kind === 'checked') {
            if (element.checked !== entry.value) {
                element.checked = Boolean(entry.value);
                restored.push(elementId);
            }
            return;
        }
        if (entry.kind === 'file') {
            if (entry.file && restoreFileSelection(element, entry.file)) {
                restored.push(elementId);
            }
            return;
        }
        if (element.value === entry.value) return;
        element.value = entry.value;
        restored.push(elementId);
    });

    return restored;
}

/**
 * Re-attach a `File` to a file input.
 *
 * A file input's `files` list is read-only, so the only way back in is a
 * `DataTransfer` built from the object we kept. Where that is not supported the
 * selection is genuinely gone — a browser restriction — and returning `false`
 * tells the caller to say so rather than showing an empty picker as though the
 * user had cleared it.
 */
function restoreFileSelection(input, file) {
    try {
        const transfer = new DataTransfer();
        transfer.items.add(file);
        input.files = transfer.files;
        return input.files?.[0] === file;
    } catch {
        return false;
    }
}

/**
 * Which preserved fields could not come back, so the failure can say so.
 *
 * A reader who is told "the file is still selected" and finds an empty picker
 * has been told something false by a recovery mechanism, which is worse than
 * having lost the file silently. So the check is explicit.
 */
function unrestorableFields(snapshot) {
    const restored = new Set(restorePreservedFields(snapshot));
    return Object.keys(snapshot || {}).filter(elementId => !restored.has(elementId));
}

// --------------------------------------------------------------------------
// Connectivity
// --------------------------------------------------------------------------

const pendingActions = [];
let connectivityKnown = typeof navigator === 'undefined' ? true : navigator.onLine !== false;

/** What the interface believes about the network right now. */
function isOnline() {
    return connectivityKnown;
}

function queuedActionCount() {
    return pendingActions.length;
}

/**
 * Hold an action until the network returns.
 *
 * A queued action is *visible*, which is the requirement: an action held back
 * by a dead network has to say that it is held back, or the interface has thrown
 * the user's work away without telling them. It is stored rather than dropped,
 * so reconnecting completes it.
 */
function enqueueWhenOffline(action) {
    if (typeof action !== 'function') return false;
    pendingActions.push(action);
    renderOfflineNotice();
    return true;
}

async function flushOfflineQueue() {
    const queued = pendingActions.splice(0, pendingActions.length);
    if (!queued.length) {
        setConnectivityNotice(t('runtime.offline.reconnected_short'), false);
        return 0;
    }
    setConnectivityNotice(t('runtime.offline.queue_flushed'), false);
    let sent = 0;
    for (const action of queued) {
        try {
            await action();
            sent += 1;
        } catch {
            // A queued action that fails on flush is reported like any other
            // failure rather than vanishing: the reader asked for it, and
            // swallowing it here would be the silent drop this queue exists to
            // prevent.
            // A scope identifier, not copy. It is never rendered — `failure.operation` is
            // only ever compared against, by `dismissNoticesFor` — so it is left
            // as a literal on purpose: giving it a catalogue key would make the
            // string a success retires its notice by depend on the reader's
            // interface language, and a notice raised before a switch would stop
            // matching the success that follows it.
            notifyFailure({ code: 'client_unknown_error', operation: '离线队列' });
        }
    }
    renderOfflineNotice();
    return sent;
}

function setConnectivity(online) {
    connectivityKnown = Boolean(online);
    const chip = ensureConnectivityChip();
    if (chip) {
        chip.textContent = online ? t('runtime.connectivity.online') : t('runtime.connectivity.offline');
        chip.dataset.tone = online ? OFFLINE_STATUS_COPY.onlineTone : 'error';
    }
    if (online) {
        flushOfflineQueue();
    } else {
        renderOfflineNotice();
    }
    return connectivityKnown;
}

function renderOfflineNotice() {
    if (!pendingActions.length) {
        setConnectivityNotice('', false);
        return;
    }
    // One key rather than two joined here: this notice is a single run of prose in
    // each language, and Chinese sets its clauses side by side where English
    // needs a full stop between them. A join in the module cannot know which.
    setConnectivityNotice(t('runtime.offline.queued', { count: pendingActions.length }), true);
}

function setConnectivityNotice(text, alarming) {
    const chip = ensureConnectivityChip();
    if (!chip) return;
    // An empty notice removes the chip's text rather than leaving a stale
    // message on screen; a chip that says "offline" after the connection is back
    // is a small lie that costs the reader a reload.
    chip.textContent = text || t('runtime.connectivity.online');
    chip.dataset.tone = text ? (alarming ? 'error' : 'idle') : OFFLINE_STATUS_COPY.onlineTone;
}

/**
 * Start listening for connectivity changes.
 *
 * Idempotent, because a module that wires listeners on import and is also
 * called explicitly would otherwise double-fire every event.
 */
function initializeConnectivity() {
    if (typeof window === 'undefined') return false;
    if (window.__paperwhispererConnectivityBound) return false;
    window.__paperwhispererConnectivityBound = true;

    window.addEventListener('online', () => setConnectivity(true));
    window.addEventListener('offline', () => setConnectivity(false));
    setConnectivity(navigator.onLine !== false);
    return true;
}

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------

/**
 * Bring the failure surface up. Safe to call more than once.
 */
function initializeRuntime() {
    initializeConnectivity();
    ensureConnectivityChip();
    return true;
}


export {
    ERROR_PRESENTATIONS,
    activeFlights,
    beginFlight,
    clearNotices,
    describeFlight,
    dismissNotice,
    endFlight,
    enqueueWhenOffline,
    flushOfflineQueue,
    formatElapsed,
    hasActiveFlight,
    initializeConnectivity,
    initializeRuntime,
    isOnline,
    normalizeFailure,
    notifyFailure,
    notifyPending,
    queuedActionCount,
    restorePreservedFields,
    unrestorableFields,
    capturePreservedFields
};
