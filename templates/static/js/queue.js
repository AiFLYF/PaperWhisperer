import { state } from './state.js';
import { sanitizeUrl } from './sanitize.js';
import { buildApiError, parseJsonSafely } from './api.js';
import { setControlDisabled, showError, updateStatus } from './dom.js';
import { getPaperListByElementId } from './paperList.js';
import { renderExportPreview } from './export.js';

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
    state.currentReadingQueue = normalizeReadingQueue(state.currentReadingQueue);
    setControlDisabled(clearButton, !state.currentReadingQueue.length);
    if (meta) {
        meta.textContent = message || (state.currentReadingQueue.length
            ? `${state.currentReadingQueue.length} saved paper(s) in this session reading queue.`
            : 'Save papers from search or recommendations into this session reading queue.');
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
        const response = await fetch('/api/reading-queue', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: state.currentSessionId,
                session_token: state.currentSessionToken,
                items: state.currentReadingQueue
            })
        });
        const data = await parseJsonSafely(response);
        if (!response.ok) throw buildApiError(data, 'Reading queue save failed.');
        state.currentReadingQueue = normalizeReadingQueue(data.items || state.currentReadingQueue);
        renderReadingQueue(message);
    } catch (error) {
        if (!silent) showError(error.message || 'Reading queue save failed.');
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

function clearReadingQueue() {
    if (!state.currentReadingQueue.length) return;
    state.currentReadingQueue = [];
    saveReadingQueue({ silent: false, message: 'Reading queue cleared.' });
}


export { addPaperToQueueByIndex };
export { clearReadingQueue };
export { getReadingQueueKey };
export { handleReadingQueueClick };
export { normalizeReadingQueue };
export { renderReadingQueue };
export { saveReadingQueue };
