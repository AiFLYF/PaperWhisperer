import { state } from './state.js';
import { sanitizeUrl } from './sanitize.js';
import { apiRequest } from './api.js';
import { notifyFailure } from './runtime.js';
import { setControlDisabled, showError, updateStatus } from './dom.js';
import { getPaperListByElementId } from './paperList.js';
import { renderExportPreview } from './export.js';
import { t } from './i18n.js';

/** The session reading queue, persisted through the session API. */

function getReadingQueueKey(value) {
    if (!value || typeof value !== 'object') return '';
    return String(value.title || value.paper_id || value.url || value.pdf_url || '').trim().toLowerCase().replace(/\s+/g, ' ');
}

function normalizeReadingQueue(values, maxItems = 30) {
    if (!Array.isArray(values)) return [];
    const seen = new Set();
    const items = [];
    values.forEach(value => {
        if (!value || typeof value !== 'object') return;
        const title = String(value.title || '').trim();
        const key = getReadingQueueKey(value);
        if (!title || !key || seen.has(key)) return;
        seen.add(key);
        items.push({
            source: String(value.source || '').trim(),
            paper_id: String(value.paper_id || '').trim(),
            title,
            abstract: String(value.abstract || '').trim(),
            authors: Array.isArray(value.authors) ? value.authors.map(author => String(author || '').trim()).filter(Boolean).slice(0, 8) : [],
            year: String(value.year || '').trim(),
            venue: String(value.venue || '').trim(),
            url: String(value.url || '').trim(),
            pdf_url: String(value.pdf_url || '').trim(),
            saved_at: value.saved_at || new Date().toISOString()
        });
    });
    return items.slice(0, maxItems);
}

function hasReadingQueueItem(item) {
    const key = getReadingQueueKey(item);
    return Boolean(key) && state.currentReadingQueue.some(savedItem => getReadingQueueKey(savedItem) === key);
}

function renderReadingQueue(message = '') {
    const container = document.getElementById('readingQueue');
    const meta = document.getElementById('readingQueueMeta');
    const clearButton = document.getElementById('clearReadingQueueBtn');
    if (!container) return;
    // Any re-render disarms the two-press Clear. Saving or removing an item
    // changes what the armed label was claiming it would clear, so an armed
    // button that outlives the count it named would be confirming a different
    // action than the one the reader agreed to.
    resetClearArm();
    state.currentReadingQueue = normalizeReadingQueue(state.currentReadingQueue);
    setControlDisabled(clearButton, !state.currentReadingQueue.length);
    // The accessible name carries the consequence, because the visible label is
    // design-owned and a hover-only `title` is not a name a screen reader can
    // rely on. The count and the permanence are both stated.
    if (clearButton) {
        const count = state.currentReadingQueue.length;
        clearButton.setAttribute(
            'aria-label',
            count
                ? `Clear the reading queue: removes all ${count} saved paper${count === 1 ? '' : 's'} from this session. This cannot be undone.`
                : 'Clear the reading queue (empty)'
        );
    }
    if (meta) {
        // Through the catalogue rather than the bare literal this used to carry.
        // `initialize()` calls `renderReadingQueue()` at startup, and the English
        // sentence was written straight over the element's translated
        // `data-i18n="queue.meta"` text on first paint. The counted form below
        // still has no catalogue key and is reported as needing one; it is kept
        // as English rather than shown as a raw key, because a visible
        // `queue.meta.n` reads as a defect to the reader and a missing
        // translation to nobody.
        const emptyKey = 'queue.meta';
        meta.textContent = message || (state.currentReadingQueue.length
            ? `${state.currentReadingQueue.length} saved paper(s) in this session reading queue.`
            : (t(emptyKey) === emptyKey
                ? 'Save papers from search or recommendations into this session reading queue.'
                : t(emptyKey)));
    }
    if (!state.currentReadingQueue.length) {
        const empty = document.createElement('p');
        empty.className = 'empty-state';
        empty.textContent = 'No saved papers yet. Save strong search or recommendation results to build an export-ready reading trail.';
        container.replaceChildren(empty);
        return;
    }

    const cards = state.currentReadingQueue.map((item, index) => {
        const article = document.createElement('article');
        article.className = 'queue-item';

        const rank = document.createElement('div');
        rank.className = 'queue-rank';
        rank.setAttribute('aria-label', `Reading queue item ${index + 1}`);
        rank.textContent = String(index + 1);

        const body = document.createElement('div');
        body.className = 'queue-body';

        const title = document.createElement('div');
        title.className = 'queue-title';
        const titleLink = document.createElement('a');
        titleLink.href = sanitizeUrl(item.url || item.pdf_url || '#');
        titleLink.target = '_blank';
        titleLink.rel = 'noopener noreferrer';
        titleLink.referrerPolicy = 'strict-origin-when-cross-origin';
        titleLink.textContent = item.title;
        title.appendChild(titleLink);

        const details = document.createElement('div');
        details.className = 'queue-details';
        const metadata = document.createElement('span');
        metadata.textContent = [item.source, item.year, item.venue].filter(Boolean).join(' · ') || 'Metadata pending';
        const authors = document.createElement('span');
        authors.textContent = item.authors.length ? item.authors.join(', ') : 'Unknown authors';
        details.append(metadata, authors);

        const actions = document.createElement('div');
        actions.className = 'queue-actions';
        [
            ['Open', item.url],
            ['PDF', item.pdf_url]
        ].forEach(([label, url]) => {
            if (!url) return;
            const link = document.createElement('a');
            link.className = 'paper-link';
            link.href = sanitizeUrl(url);
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.referrerPolicy = 'strict-origin-when-cross-origin';
            link.textContent = label;
            actions.appendChild(link);
        });
        const removeButton = document.createElement('button');
        removeButton.className = 'paper-link';
        removeButton.type = 'button';
        removeButton.dataset.queueRemoveIndex = String(index);
        removeButton.textContent = 'Remove';
        actions.appendChild(removeButton);

        body.append(title, details, actions);
        article.append(rank, body);
        return article;
    });
    container.replaceChildren(...cards);
}

async function saveReadingQueue({ silent = true, message = '' } = {}) {
    renderReadingQueue(message);
    renderExportPreview();
    if (!state.currentSessionId || !state.currentSessionToken) return;
    try {
        const data = await apiRequest('/api/reading-queue', {
            method: 'POST',
            body: {
                session_id: state.currentSessionId,
                session_token: state.currentSessionToken,
                items: state.currentReadingQueue
            },
            operation: 'readingQueue'
        });
        state.currentReadingQueue = normalizeReadingQueue(data.items || state.currentReadingQueue);
        renderReadingQueue(message);
    } catch (error) {
        if (error.name === 'AbortError' || error.code === 'client_cancelled') return;
        // The distinction that matters here is between a queue that failed to
        // save and a queue that is gone. It is still in `state` and still on
        // screen either way, so the notice says *that* rather than reporting a
        // bare failure the reader would read as "my list was deleted".
        if (!silent) {
            notifyFailure(error, { onRetry: () => saveReadingQueue({ silent: false, message }) });
        }
    }
}

function addPaperToQueue(item) {
    if (hasReadingQueueItem(item)) {
        renderReadingQueue('This paper is already in your reading queue.');
        updateStatus('Paper already saved', 'idle');
        return;
    }
    state.currentReadingQueue = normalizeReadingQueue([...state.currentReadingQueue, item]);
    const message = `Saved “${item.title || 'paper'}” to the reading queue.`;
    renderReadingQueue(message);
    updateStatus('Paper saved to queue', 'success');
    saveReadingQueue({ silent: false, message });
}

function addPaperToQueueByIndex(elementId, index) {
    const items = getPaperListByElementId(elementId);
    const item = items[index];
    if (!item) {
        showError('Paper result not found. Please search again.');
        return;
    }
    addPaperToQueue(item);
}

function handleReadingQueueClick(event) {
    const removeButton = event.target.closest('[data-queue-remove-index]');
    if (!removeButton) return;
    const index = Number(removeButton.dataset.queueRemoveIndex);
    if (!Number.isInteger(index)) return;
    removePaperFromQueue(index);
}

function removePaperFromQueue(index) {
    const removed = state.currentReadingQueue[index];
    state.currentReadingQueue.splice(index, 1);
    state.currentReadingQueue = normalizeReadingQueue(state.currentReadingQueue);
    saveReadingQueue({ silent: false, message: removed ? `Removed “${removed.title}” from the reading queue.` : 'Removed paper from the reading queue.' });
}

/**
 * Whether the Clear control is armed, and the button to restore when it is not.
 *
 * Module-local rather than on `state`: arming is a property of one control on
 * this card, and putting it in the shared store would let a workspace snapshot
 * capture it and hand a half-armed destructive button to the next render.
 */
let clearArmed = false;
let clearDisarmTimer = null;
let clearButtonDefaultLabel = '';

const CLEAR_ARM_WINDOW_MS = 6000;

function resetClearArm() {
    clearArmed = false;
    if (clearDisarmTimer) {
        clearTimeout(clearDisarmTimer);
        clearDisarmTimer = null;
    }
    const button = document.getElementById('clearReadingQueueBtn');
    if (button) {
        if (clearButtonDefaultLabel) {
            button.textContent = clearButtonDefaultLabel;
        }
        button.removeAttribute('aria-describedby');
        button.dataset.armed = 'false';
    }
}

/**
 * Empty the reading queue, in two deliberate presses.
 *
 * This is the only irreversible control in the interface that the reader
 * triggers by accident rather than on purpose: "Clear" sits next to the queue
 * heading, it carries no count, it says nothing about what is lost, and the
 * queue is written straight through to the server — so one stray click destroys
 * a reading trail that took a dozen deliberate "Save" presses to build, with no
 * undo and no recovery path.
 *
 * A `window.confirm` would solve it and is worse: it is unstyleable, it is
 * announced inconsistently, and it interrupts with a modal for a decision the
 * label could have made obvious. So the first press arms the button and says
 * what the second one will do, naming the count, and the window closes on its
 * own. The consequence is stated before it happens, and the first press — the
 * accidental one — costs nothing.
 */
function clearReadingQueue() {
    const count = state.currentReadingQueue.length;
    if (!count) return;

    if (!clearArmed) {
        const button = document.getElementById('clearReadingQueueBtn');
        clearArmed = true;
        if (button) {
            if (!clearButtonDefaultLabel) {
                clearButtonDefaultLabel = button.textContent || 'Clear';
            }
            button.textContent = `Clear ${count} saved paper${count === 1 ? '' : 's'} — press again to confirm`;
            button.dataset.armed = 'true';
        }
        updateStatus(`Clearing the reading queue removes all ${count} saved paper${count === 1 ? '' : 's'} from this session. Press Clear again to confirm.`, 'idle');
        clearDisarmTimer = setTimeout(resetClearArm, CLEAR_ARM_WINDOW_MS);
        return;
    }

    resetClearArm();
    state.currentReadingQueue = [];
    saveReadingQueue({ silent: false, message: `Reading queue cleared. ${count} saved paper${count === 1 ? '' : 's'} removed from this session.` });
}


export { addPaperToQueueByIndex };
export { clearReadingQueue };
export { getReadingQueueKey };
export { handleReadingQueueClick };
export { normalizeReadingQueue };
export { renderReadingQueue };
export { saveReadingQueue };
