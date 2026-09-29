import { state } from './state.js';
import { sanitizeUrl } from './sanitize.js';
import { setControlDisabled } from './dom.js';

/**
 * Pure rendering for search results, recommendations and their shared card
 * chrome. No network calls live here, which keeps it importable from the
 * reading-queue and analysis layers without creating an import cycle.
 */

function buildPaperActionKey(item) {
    return String(item.paper_id || item.pdf_url || item.url || item.title || '').trim();
}

function getPaperStateDetails(elementId, emptyText) {
    const text = emptyText || 'No papers found.';
    const loading = /searching|generating/i.test(text);
    if (loading) {
        return {
            tone: 'loading',
            title: text,
            body: elementId === 'paperRecommendations'
                ? 'Building topics from the current paper and checking external sources.'
                : 'Checking Semantic Scholar and arXiv for relevant papers.'
        };
    }
    if (elementId === 'paperRecommendations') {
        return {
            tone: 'empty',
            title: text,
            body: 'Try analyzing a richer paper, then run recommendations again, or search manually by method, task, or dataset.'
        };
    }
    return {
        tone: 'empty',
        title: text,
        body: text.includes('No matching')
            ? 'Try broader keywords, include a dataset or method name, or search by the problem statement instead.'
            : 'Enter a topic, method, task, or dataset above to start building your reading trail.'
    };
}

function renderPaperList(elementId, items, emptyText, meta) {
    const container = document.getElementById(elementId);
    if (!container) return;
    const normalizedItems = Array.isArray(items) ? items : [];
    const allowImport = elementId === 'paperSearchResults';
    if (!normalizedItems.length) {
        const state = getPaperStateDetails(elementId, emptyText);
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
            titleLink.textContent = item.title || 'Untitled paper';
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
            authors.textContent = Array.isArray(item.authors) && item.authors.length ? item.authors.join(', ') : 'Unknown authors';

            const abstract = document.createElement('div');
            abstract.className = 'paper-abstract';
            abstract.textContent = item.abstract || 'No abstract available.';

            article.append(titleWrap, tags, authors, abstract);
            if (item.reason) {
                const reason = document.createElement('div');
                reason.className = 'paper-reason';
                const label = document.createElement('strong');
                label.textContent = 'Why:';
                reason.append(label, ` ${item.reason}`);
                article.appendChild(reason);
            }

            const links = document.createElement('div');
            links.className = 'paper-links';
            [
                ['Open', item.url, 'link'],
                ['PDF', item.pdf_url, 'link'],
                ['Download', item.pdf_url || item.url, 'download'],
                ['Save', true, 'save']
            ].forEach(([label, value, action]) => {
                if (!value) return;
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
                addButton.textContent = addButtonDisabled ? 'Adding...' : 'Add';
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
    state.currentPaperSearchMetaText = 'Search across Semantic Scholar and arXiv with a single query.';
    state.currentPaperRecommendationMetaText = 'Analyze a paper first, then generate follow-up reading suggestions from the current session.';
    renderPaperList('paperSearchResults', [], 'Search results will appear here.', { elementId: 'paperSearchMeta', text: state.currentPaperSearchMetaText });
    renderPaperList('paperRecommendations', [], 'Recommendations will appear here after analysis.', { elementId: 'paperRecommendationMeta', text: state.currentPaperRecommendationMetaText });
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
export { getPaperListByElementId };
export { getPaperStateDetails };
export { renderPaperList };
export { resetPaperPanels };
export { setRecommendEnabled };
