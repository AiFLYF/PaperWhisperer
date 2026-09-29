import { state } from './state.js';
import { buildApiError, parseJsonSafely } from './api.js';
import { hideError, setButtonLoading, setCancelVisible, showError, updateStatus } from './dom.js';
import { buildPaperActionKey, getPaperListByElementId, renderPaperList, setRecommendEnabled } from './paperList.js';
import { addPaperToQueueByIndex } from './queue.js';
import { applyAnalysisResult, resetResultView } from './sections.js';
import { renderExportPreview } from './export.js';

/** Paper search, recommendations and importing a remote paper for analysis. */

function useExampleQuery(query) {
    const input = document.getElementById('paperSearchInput');
    if (!input || !query) return;
    input.value = query;
    input.focus({ preventScroll: true });
    updateStatus('Example query loaded', 'idle');
    searchPapers();
}

async function searchPapers() {
    const queryInput = document.getElementById('paperSearchInput');
    const searchBtn = document.getElementById('paperSearchBtn');
    const apiKey = document.getElementById('apiKey').value.trim();
    const query = queryInput.value.trim();

    if (!query) {
        queryInput.focus();
        updateStatus('Search query needed', 'idle');
        renderPaperList('paperSearchResults', [], 'Enter a topic, method, task, or dataset to search papers.', {
            elementId: 'paperSearchMeta',
            text: 'Search needs a topic, method, task, dataset, or problem statement.'
        });
        return;
    }

    if (state.paperSearchController) {
        state.paperSearchController.abort();
    }
    state.paperSearchController = new AbortController();
    state.paperSearchRequestId += 1;
    const requestId = state.paperSearchRequestId;

    setButtonLoading(searchBtn, 'Searching...', 'Search Papers', true);
    setCancelVisible('cancelPaperSearchBtn', true);
    hideError();
    renderPaperList('paperSearchResults', [], 'Searching papers...', { elementId: 'paperSearchMeta', text: `Searching for: ${query}` });

    try {
        const response = await fetch('/api/search-papers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                api_key: apiKey || undefined,
                session_id: state.currentSessionToken ? state.currentSessionId : undefined,
                session_token: state.currentSessionToken || undefined
            }),
            signal: state.paperSearchController.signal
        });
        const data = await parseJsonSafely(response);
        if (requestId !== state.paperSearchRequestId) {
            return;
        }
        if (!response.ok) {
            throw buildApiError(data, 'Paper search failed.');
        }

        state.currentPaperSearchResults = Array.isArray(data.items) ? data.items : [];
        const rewriteBits = [];
        if (data.original_query) rewriteBits.push(`Original: ${data.original_query}`);
        if (data.rewritten_query) rewriteBits.push(`Rewritten: ${data.rewritten_query}`);
        if (Array.isArray(data.topics) && data.topics.length) rewriteBits.push(`Topics: ${data.topics.join(', ')}`);
        if (data.reason) rewriteBits.push(`Why: ${data.reason}`);
        if (data.rewrite_model) rewriteBits.push(`Model: ${data.rewrite_model}`);
        if (data.errors && data.errors.length) rewriteBits.push(`Partial results: ${data.errors.join(' | ')}`);
        rewriteBits.push(`${state.currentPaperSearchResults.length} paper(s)`);
        state.currentPaperSearchMetaText = rewriteBits.join(' · ');
        renderPaperList('paperSearchResults', state.currentPaperSearchResults, 'No matching papers found.', {
            elementId: 'paperSearchMeta',
            text: state.currentPaperSearchMetaText
        });
        renderExportPreview();
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        state.currentPaperSearchResults = [];
        state.currentPaperSearchMetaText = 'Search across Semantic Scholar and arXiv with a single query.';
        renderPaperList('paperSearchResults', [], 'Search results will appear here.', { elementId: 'paperSearchMeta', text: state.currentPaperSearchMetaText });
        showError(error.message || 'Paper search failed.');
    } finally {
        if (requestId === state.paperSearchRequestId) {
            setButtonLoading(searchBtn, 'Searching...', 'Search Papers', false);
            setCancelVisible('cancelPaperSearchBtn', false);
            state.paperSearchController = null;
        }
    }
}

async function recommendPapers() {
    if (!state.currentSessionId) {
        showError('Requires document analysis first.');
        return;
    }

    const apiKey = document.getElementById('apiKey').value.trim();
    const recommendBtn = document.getElementById('recommendBtn');
    if (state.recommendController) {
        state.recommendController.abort();
    }
    state.recommendController = new AbortController();
    state.recommendRequestId += 1;
    const requestId = state.recommendRequestId;

    setButtonLoading(recommendBtn, 'Recommending...', 'Recommend from current paper', true);
    setCancelVisible('cancelRecommendBtn', true);
    hideError();
    renderPaperList('paperRecommendations', [], 'Generating recommendations...', { elementId: 'paperRecommendationMeta', text: 'Generating search topics from the current paper...' });

    try {
        const response = await fetch('/api/recommend-papers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: state.currentSessionId, session_token: state.currentSessionToken, api_key: apiKey }),
            signal: state.recommendController.signal
        });
        const data = await parseJsonSafely(response);
        if (requestId !== state.recommendRequestId) {
            return;
        }
        if (!response.ok) {
            throw buildApiError(data, 'Paper recommendation failed.');
        }

        state.currentPaperRecommendations = Array.isArray(data.items)
            ? data.items.map(item => ({ ...item, reason: data.reason || '' }))
            : [];
        const recommendationBits = [];
        if (data.original_query) recommendationBits.push(`Original: ${data.original_query}`);
        if (data.query) recommendationBits.push(`Rewritten: ${data.query}`);
        if (Array.isArray(data.topics) && data.topics.length) recommendationBits.push(`Topics: ${data.topics.join(', ')}`);
        if (data.reason) recommendationBits.push(`Why: ${data.reason}`);
        if (data.rewrite_model) recommendationBits.push(`Model: ${data.rewrite_model}`);
        if (data.errors && data.errors.length) recommendationBits.push(`Partial results: ${data.errors.join(' | ')}`);
        state.currentPaperRecommendationMetaText = recommendationBits.join(' · ');
        renderPaperList('paperRecommendations', state.currentPaperRecommendations, 'No recommendations found.', {
            elementId: 'paperRecommendationMeta',
            text: state.currentPaperRecommendationMetaText
        });
        renderExportPreview();
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        state.currentPaperRecommendations = [];
        state.currentPaperRecommendationMetaText = 'Analyze a paper first, then generate follow-up reading suggestions from the current session.';
        renderPaperList('paperRecommendations', [], 'Recommendations will appear here after analysis.', { elementId: 'paperRecommendationMeta', text: state.currentPaperRecommendationMetaText });
        showError(error.message || 'Paper recommendation failed.');
    } finally {
        if (requestId === state.recommendRequestId) {
            setButtonLoading(recommendBtn, 'Recommending...', 'Recommend from current paper', false);
            setRecommendEnabled(Boolean(state.currentSessionId));
            setCancelVisible('cancelRecommendBtn', false);
            state.recommendController = null;
        }
    }
}

function cancelPaperSearchRequest() {
    if (!state.paperSearchController) return;
    state.paperSearchController.abort();
    state.paperSearchRequestId += 1;
    setButtonLoading(document.getElementById('paperSearchBtn'), 'Searching...', 'Search Papers', false);
    setCancelVisible('cancelPaperSearchBtn', false);
    state.paperSearchController = null;
    renderPaperList('paperSearchResults', state.currentPaperSearchResults, state.currentPaperSearchResults.length ? '' : 'Search results will appear here.', {
        elementId: 'paperSearchMeta',
        text: state.currentPaperSearchMetaText
    });
    updateStatus('Paper search canceled', 'idle');
}

function cancelRecommendRequest() {
    if (!state.recommendController) return;
    state.recommendController.abort();
    state.recommendRequestId += 1;
    setButtonLoading(document.getElementById('recommendBtn'), 'Recommending...', 'Recommend from current paper', false);
    setRecommendEnabled(Boolean(state.currentSessionId));
    setCancelVisible('cancelRecommendBtn', false);
    state.recommendController = null;
    renderPaperList('paperRecommendations', state.currentPaperRecommendations, state.currentPaperRecommendations.length ? '' : 'Recommendations will appear here after analysis.', {
        elementId: 'paperRecommendationMeta',
        text: state.currentPaperRecommendationMetaText
    });
    updateStatus('Recommendations canceled', 'idle');
}

function handlePaperSearchKeyPress(event) {
    if (event.key === 'Enter' && !document.getElementById('paperSearchBtn').disabled) {
        event.preventDefault();
        searchPapers();
    }
}

function downloadPaper(item) {
    if (!item || (!item.pdf_url && !item.url)) {
        showError('No downloadable paper file is available for this result.');
        return;
    }
    const params = new URLSearchParams();
    if (item.url) params.set('url', item.url);
    if (item.pdf_url) params.set('pdf_url', item.pdf_url);
    if (item.title) params.set('title', item.title);
    const targetUrl = `/api/download-paper?${params.toString()}`;
    window.open(targetUrl, '_blank', 'noopener,noreferrer');
}

function downloadPaperByIndex(elementId, index) {
    const items = getPaperListByElementId(elementId);
    const item = items[index];
    if (!item) {
        showError('Paper result not found. Please search again.');
        return;
    }
    downloadPaper(item);
}

async function addPaperToAnalysis(item) {
    const apiKey = document.getElementById('apiKey').value.trim();
    const generateMermaid = document.getElementById('generateMermaid').checked;
    const generateEvaluation = document.getElementById('generateEvaluation').checked;
    const generateResearchBrief = document.getElementById('generateResearchBrief').checked;
    const paperKey = buildPaperActionKey(item);

    if (state.importPaperController) {
        state.importPaperController.abort();
    }
    state.importPaperController = new AbortController();
    state.currentImportPaperKey = paperKey;
    hideError();
    resetResultView();
    state.currentSessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    document.getElementById('loading').classList.add('active');
    updateStatus('Importing paper from search results...', 'idle');
    renderPaperList('paperSearchResults', state.currentPaperSearchResults, 'Search results will appear here.', {
        elementId: 'paperSearchMeta',
        text: 'Importing selected paper into PaperWhisperer...'
    });

    try {
        const response = await fetch('/api/import-paper', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title: item.title || '',
                url: item.url || '',
                pdf_url: item.pdf_url || '',
                session_id: state.currentSessionId,
                session_token: state.currentSessionToken,
                api_key: apiKey,
                generate_mermaid: generateMermaid,
                generate_evaluation: generateEvaluation,
                generate_research_brief: generateResearchBrief
            }),
            signal: state.importPaperController.signal
        });
        const data = await parseJsonSafely(response);
        if (!response.ok) {
            throw buildApiError(data, 'Paper import failed.');
        }

        state.currentPaperSearchMetaText = 'Imported selected paper into PaperWhisperer.';
        await applyAnalysisResult(data, data.source_filename || item.title || 'Imported paper', generateEvaluation, generateMermaid, generateResearchBrief);
        updateStatus('Imported paper ready for follow-up questions', 'success');
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        state.currentSessionId = '';
        resetResultView();
        showError(error.message || 'Paper import failed.');
        updateStatus('Paper import failed', 'error');
    } finally {
        state.currentImportPaperKey = '';
        document.getElementById('loading').classList.remove('active');
        renderPaperList('paperSearchResults', state.currentPaperSearchResults, 'Search results will appear here.', {
            elementId: 'paperSearchMeta',
            text: state.currentPaperSearchMetaText
        });
        state.importPaperController = null;
    }
}

function addPaperToAnalysisByIndex(index) {
    const item = state.currentPaperSearchResults[index];
    if (!item) {
        showError('Paper result not found. Please search again.');
        return;
    }
    addPaperToAnalysis(item);
}

function handlePaperResultClick(event) {
    const actionButton = event.target.closest('[data-paper-action]');
    if (!actionButton) return;
    const elementId = actionButton.dataset.paperList || '';
    const index = Number(actionButton.dataset.paperIndex);
    if (!elementId || !Number.isInteger(index)) return;
    if (actionButton.dataset.paperAction === 'download') {
        downloadPaperByIndex(elementId, index);
    } else if (actionButton.dataset.paperAction === 'add') {
        addPaperToAnalysisByIndex(index);
    } else if (actionButton.dataset.paperAction === 'save') {
        addPaperToQueueByIndex(elementId, index);
    }
}


export { addPaperToAnalysisByIndex };
export { cancelPaperSearchRequest };
export { cancelRecommendRequest };
export { downloadPaperByIndex };
export { handlePaperResultClick };
export { handlePaperSearchKeyPress };
export { recommendPapers };
export { searchPapers };
export { useExampleQuery };
