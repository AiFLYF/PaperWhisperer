import { state } from './state.js';
import { apiRequest } from './api.js';
import { capturePreservedFields, notifyFailure, restorePreservedFields } from './runtime.js';
import { hideError, setButtonLoading, setCancelVisible, showError, updateStatus, beginProgressClock, endProgressClock, updateAnalysisProgress, focusAnalysisWorkspace } from './dom.js';
import { buildPaperActionKey, getPaperListByElementId, renderPaperList, setRecommendEnabled } from './paperList.js';
import { renderRecommendationSignals, resetRecommendationSignals } from './recommendationSignals.js';
import { addPaperToQueueByIndex } from './queue.js';
import { applyAnalysisResult, captureWorkspaceState, describeAnalysisOutcome, hideRunIndicator, resetResultView, restoreWorkspaceState, showRunIndicator } from './sections.js';
import { renderExportPreview } from './export.js';
import { RECOMMENDATIONS_API_PATH } from './config.js';

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
    // Captured before the request, restored on failure. See `analyze.js` for
    // why the capture cannot live in the `catch`.
    const preserved = capturePreservedFields();
    renderPaperList('paperSearchResults', [], 'Searching papers...', { elementId: 'paperSearchMeta', text: `Searching for: ${query}` });

    try {
        const data = await apiRequest('/api/search-papers', {
            method: 'POST',
            body: {
                query,
                api_key: apiKey || undefined,
                session_id: state.currentSessionToken ? state.currentSessionId : undefined,
                session_token: state.currentSessionToken || undefined
            },
            operation: 'paperSearch',
            signal: state.paperSearchController.signal
        });
        if (requestId !== state.paperSearchRequestId) {
            return;
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
        // Both spellings, because the transport rewrites a cancellation into an
        // ApiError whose name is not AbortError. Testing only the name meant a
        // cancelled search ran the failure path: it wiped the results the
        // cancel handler had just put back, replaced them with the empty-state
        // message, and raised a failure notice with a retry button, for work the
        // reader deliberately stopped.
        if (error.name === 'AbortError' || error.code === 'client_cancelled') {
            return;
        }
        state.currentPaperSearchResults = [];
        state.currentPaperSearchMetaText = 'Search across Semantic Scholar and arXiv with a single query.';
        renderPaperList('paperSearchResults', [], 'Search results will appear here.', { elementId: 'paperSearchMeta', text: state.currentPaperSearchMetaText });
        // The query stays in the input: it was captured before the request and
        // is restored here, so a failed search can be retried by pressing
        // 重试 rather than by retyping a query that may be a full sentence.
        restorePreservedFields(preserved);
        notifyFailure(error, { onRetry: () => searchPapers() });
    } finally {
        if (requestId === state.paperSearchRequestId) {
            setButtonLoading(searchBtn, 'Searching...', 'Search Papers', false);
            setCancelVisible('cancelPaperSearchBtn', false);
            state.paperSearchController = null;
        }
    }
}

/**
 * The title to search the seed by, when the client has no bibliographic record.
 *
 * A document uploaded from disk carries no DOI, arXiv id, S2 id or OpenAlex id
 * — only a filename — and the analysed result echoes none of them. So the
 * client’s honest seed for such a document is its title, which is enough for
 * the topical signal to run and not enough for the four citation-graph signals.
 * Those then report themselves `unavailable` with a reason, and the four-state
 * status is what makes that readable as "we could not check" rather than "there
 * are none". The alternative — refusing to recommend, or inventing an
 * identifier — is worse than both.
 */
function recommendationSeedTitle() {
    const result = state.currentAnalysisResult;
    const raw = String(result?.title || result?.source_filename || state.currentSourceFileName || '').trim();
    return raw.replace(/\.[^.]+$/, '').trim();
}

/**
 * Capture the bibliographic record for the paper now under analysis.
 *
 * Called when a search result is imported, because that is the only moment the
 * front end holds a record that carries the identifier set T3.1 added — a
 * parsed document never does. Everything the route recognises is copied
 * verbatim; nothing is derived, because a derived identifier here would be
 * guessed provenance for somebody’s paper.
 */
function setSeedPaper(record) {
    const data = record && typeof record === 'object' ? record : {};
    const seed = {
        title: String(data.title || '').trim(),
        year: String(data.year || '').trim(),
        authors: Array.isArray(data.authors)
            ? data.authors.map(author => String(author || '').trim()).filter(Boolean)
            : [],
        source: String(data.source || '').trim(),
        url: String(data.url || data.pdf_url || '').trim(),
        paper_id: String(data.paper_id || '').trim(),
        doi: String(data.doi || '').trim(),
        arxiv_id: String(data.arxiv_id || '').trim(),
        s2_paper_id: String(data.s2_paper_id || '').trim(),
        openalex_id: String(data.openalex_id || '').trim()
    };
    state.currentSeedPaper = seed.title ? seed : null;
    return state.currentSeedPaper;
}

/**
 * The `paper` seed sent to the recommendation route.
 *
 * A copy, so a later render cannot mutate what was sent, and never a reference
 * to the live record on state.
 */
function buildRecommendationSeed() {
    const seed = state.currentSeedPaper;
    if (seed && typeof seed === 'object') {
        return { ...seed };
    }
    const title = recommendationSeedTitle();
    return title ? { title } : {};
}

/**
 * The one-line summary of a recommendation run.
 *
 * The signal breakdown is *not* folded into this string. The counts of what
 * worked belong in the per-signal region, where each one carries its own state
 * and its own reason; a summary that says "4/5 signals" cannot say which four,
 * and a run where the missing one was rate limited would then read as a thinner
 * result rather than a gap. What belongs here is the parts that describe the
 * request rather than the answer: what was searched, why, and what could not be
 * checked at all.
 */
function buildRecommendationMetaText(data, itemCount) {
    const bits = [];
    if (data?.seed?.title) {
        bits.push(`Seed: ${data.seed.title}`);
    } else if (data?.original_query) {
        bits.push(`Original: ${data.original_query}`);
    }
    if (data?.query) {
        bits.push(`Rewritten: ${data.query}`);
    }
    if (Array.isArray(data?.topics) && data.topics.length) {
        bits.push(`Topics: ${data.topics.join(', ')}`);
    }
    if (data?.reason) {
        bits.push(`Why: ${data.reason}`);
    }
    if (data?.rewrite_model) {
        bits.push(`Model: ${data.rewrite_model}`);
    }
    const unavailable = Array.isArray(data?.unavailable_signals) ? data.unavailable_signals : [];
    if (unavailable.length) {
        // Named, not counted. "3 signals unavailable" is a number the reader
        // cannot act on; "被引、相关：未能获取" tells them which answers are
        // missing and therefore which conclusions are not yet supported.
        const named = unavailable
            .map(entry => `${entry?.label_zh || entry?.signal || '信号'}（未能获取）`)
            .join('、');
        bits.push(`未能获取的信号: ${named}`);
    }
    if (Array.isArray(data?.errors) && data.errors.length) {
        bits.push(`Partial results: ${data.errors.join(' | ')}`);
    }
    bits.push(`${itemCount} paper(s)`);
    return bits.join(' · ');
}

async function recommendPapers() {
    if (!state.currentSessionId) {
        showError('Requires document analysis first.');
        return;
    }

    const apiKey = document.getElementById('apiKey').value.trim();
    const recommendBtn = document.getElementById('recommendBtn');

    // Built before the request, and the one case that is refused outright: a
    // seed with neither an identifier nor a title. The route answers 400
    // `missing_seed_paper` for it, so sending it would be a request this client
    // knows cannot succeed — and the failure would read as a server fault rather
    // than as "there is nothing to recommend from".
    //
    // Checked *before* the controller is created, not after. Creating it first
    // and returning here leaves `state.recommendController` pointing at a live
    // controller for a request that was never sent: the `finally` that would
    // clear it is not reached, so the module believes a run is in flight
    // indefinitely, and the next cancel is wired to abort nothing.
    const seed = buildRecommendationSeed();
    if (!seed.title && !seed.doi && !seed.arxiv_id && !seed.s2_paper_id && !seed.openalex_id) {
        showError('No paper identity is available for this document, so there is nothing to recommend from. Search for or import a paper first.');
        updateStatus('Recommendation needs a paper identity', 'error');
        return;
    }

    if (state.recommendController) {
        state.recommendController.abort();
    }
    state.recommendController = new AbortController();
    state.recommendRequestId += 1;
    const requestId = state.recommendRequestId;

    setButtonLoading(recommendBtn, 'Recommending...', 'Recommend from current paper', true);
    setCancelVisible('cancelRecommendBtn', true);
    hideError();
    // The previous run signal status describes papers that are about to be
    // replaced by "generating…", so it is cleared here rather than left to sit
    // above a list it no longer describes until the new run lands.
    state.currentRecommendationSignals = [];
    state.currentRecommendationDuplicates = [];
    resetRecommendationSignals();
    renderPaperList('paperRecommendations', [], 'Generating recommendations...', { elementId: 'paperRecommendationMeta', text: 'Generating search topics from the current paper...' });

    try {
        const data = await apiRequest(RECOMMENDATIONS_API_PATH, {
            method: 'POST',
            body: {
                session_id: state.currentSessionId,
                session_token: state.currentSessionToken,
                api_key: apiKey,
                paper: seed
            },
            operation: 'recommend',
            signal: state.recommendController.signal
        });
        if (requestId !== state.recommendRequestId) {
            return;
        }

        // T3.2. `reasons` is per row and `signals` is per run; both are read
        // from the payload rather than reconstructed, and a row without a
        // `reasons` list still renders (as a plain result) rather than being
        // dropped for lacking provenance it may legitimately not have.
        state.currentPaperRecommendations = Array.isArray(data.items)
            ? data.items.map(item => ({ ...item, reason: item.reason || data.reason || '' }))
            : [];

        // The four-state signal status and the probable-duplicate pairs, kept
        // whole. Rendered from the same values that are stored, so a re-render
        // or a workspace restore cannot show a status the current rows disagree
        // with.
        state.currentRecommendationSignals = Array.isArray(data.signals) ? data.signals : [];
        state.currentRecommendationDuplicates = Array.isArray(data.possible_duplicates) ? data.possible_duplicates : [];
        renderRecommendationSignals(state.currentRecommendationSignals, state.currentRecommendationDuplicates);

        state.currentPaperRecommendationMetaText = buildRecommendationMetaText(data, state.currentPaperRecommendations.length);
        renderPaperList('paperRecommendations', state.currentPaperRecommendations, 'No recommendations found.', {
            elementId: 'paperRecommendationMeta',
            text: state.currentPaperRecommendationMetaText
        });
        renderExportPreview();
    } catch (error) {
        // Both spellings, for the same reason as in searchPapers: the transport
        // rewrites a cancellation into an ApiError whose name is not AbortError.
        // Testing only the name therefore reported a deliberate stop as a
        // failure, which reset the per-signal status and cleared the rows on
        // screen -- destroying the previous run of recommendations, the ones the
        // cancel handler had just put back.
        if (error.name === 'AbortError' || error.code === 'client_cancelled') {
            return;
        }
        state.currentPaperRecommendations = [];
        // A failed run says nothing about any signal, so the previous run
        // status is cleared rather than left describing papers that are no
        // longer on screen. "we could not check" would be a half-truth about a
        // request that never completed.
        resetRecommendationSignals();
        state.currentRecommendationSignals = [];
        state.currentRecommendationDuplicates = [];
        state.currentPaperRecommendationMetaText = 'Analyze a paper first, then generate follow-up reading suggestions from the current session.';
        renderPaperList('paperRecommendations', [], 'Recommendations will appear here after analysis.', { elementId: 'paperRecommendationMeta', text: state.currentPaperRecommendationMetaText });
        notifyFailure(error, { onRetry: () => recommendPapers() });
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
    // The cancelled run contributed nothing, so whatever rows remain on screen
    // are the previous run and must keep the previous run signal status —
    // re-rendered from the same stored values, so the two cannot disagree.
    renderRecommendationSignals(state.currentRecommendationSignals, state.currentRecommendationDuplicates);
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

    // An import is a full analysis run, and it starts by tearing the current
    // workspace down — exactly as `analyze()` does. That run captured a snapshot
    // first and restores it when the request fails; this one did not, so a
    // failed import left the reader with no analysis, no transcript and no
    // reading queue, with no way back. The upload card promises "failed attempts
    // keep the last completed workspace available", and the promise was simply
    // false for this path. The snapshot is what makes it true.
    const previousWorkspace = state.currentAnalysisResult ? captureWorkspaceState() : null;

    if (state.importPaperController) {
        state.importPaperController.abort();
    }
    state.importPaperController = new AbortController();
    state.currentImportPaperKey = paperKey;
    hideError();
    // Captured before `resetResultView()` and before the request, because both
    // the reset and an optimistic clear can consume what the failure handler
    // would want to put back.
    const preserved = capturePreservedFields();
    resetResultView();
    state.currentSessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    showRunIndicator();
    document.getElementById('loading').classList.add('active');
    // The import fetches the paper from a third-party URL, parses it and then
    // runs the same six sections as an upload. That is a two-minute operation
    // shown as a bare spinner with no stage and no clock, because this path
    // never called `beginProgressClock` and never named a step. The upload path
    // does all three, and there is no reason the two runs of the same analysis
    // should report themselves differently.
    beginProgressClock();
    updateAnalysisProgress('upload', 'Downloading the paper from the source, then reading its text...');
    updateStatus('Importing paper from search results...', 'idle');
    renderPaperList('paperSearchResults', state.currentPaperSearchResults, 'Search results will appear here.', {
        elementId: 'paperSearchMeta',
        text: 'Importing selected paper into PaperWhisperer...'
    });

    try {
        const data = await apiRequest('/api/import-paper', {
            method: 'POST',
            body: {
                title: item.title || '',
                url: item.url || '',
                pdf_url: item.pdf_url || '',
                session_id: state.currentSessionId,
                session_token: state.currentSessionToken,
                api_key: apiKey,
                generate_mermaid: generateMermaid,
                generate_evaluation: generateEvaluation,
                generate_research_brief: generateResearchBrief
            },
            // The server fetches the paper from a third-party URL before
            // analysing it, so this is bounded by the import budget rather than
            // the default: a slow origin should not read as a hung request, and
            // a hung request must not read as a slow origin.
            importing: true,
            operation: 'importPaper',
            signal: state.importPaperController.signal
        });

        state.currentPaperSearchMetaText = 'Imported selected paper into PaperWhisperer.';
        await applyAnalysisResult(data, data.source_filename || item.title || 'Imported paper', generateEvaluation, generateMermaid, generateResearchBrief);
        // The imported search result is the one record the front end ever holds
        // that carries the identifier set the citation-graph signals need, so it
        // becomes the recommendation seed. Set after `applyAnalysisResult`,
        // because the import path resets the workspace on its way in and that
        // reset clears the seed.
        setSeedPaper(item);
        // Read from the result rather than asserted: an imported paper goes
        // through the same six sections, so it can be partial in exactly the way
        // an uploaded one can, and "ready" is only true when none of them failed.
        const outcome = describeAnalysisOutcome(data);
        updateAnalysisProgress('ready', outcome.tone === 'success'
            ? 'Workspace ready. You can ask questions or export the session.'
            : `Workspace ready, but the analysis is incomplete. ${outcome.text}`);
        endProgressClock(outcome.tone === 'success' ? '分析完成。' : '分析完成，但有部分未生成。');
        updateStatus(outcome.tone === 'success' ? 'Imported paper ready for follow-up questions' : outcome.text, outcome.tone);
    } catch (error) {
        if (error.name === 'AbortError' || error.code === 'client_cancelled') {
            return;
        }
        // The reader’s previous work goes back, exactly as it does for a failed
        // upload. Before this, an import failure was the one way to lose a whole
        // workspace with no undo.
        if (previousWorkspace) {
            restoreWorkspaceState(previousWorkspace);
            showError(`${error.message || '导入失败。'}\n\nYour previous successful workspace has been restored.`);
            focusAnalysisWorkspace({ scroll: false });
        } else {
            state.currentSessionId = '';
            resetResultView();
        }
        // The three analysis option checkboxes are restored, so a retry of a
        // failed import is the same import rather than a differently-configured
        // one that fails for a second, unrelated reason.
        restorePreservedFields(preserved);
        notifyFailure(error, { onRetry: () => addPaperToAnalysis(item) });
        updateStatus('Paper import failed', 'error');
    } finally {
        state.currentImportPaperKey = '';
        endProgressClock('这次导入没有完成。');
        document.getElementById('loading').classList.remove('active');
        hideRunIndicator();
        renderPaperList('paperSearchResults', state.currentPaperSearchResults, 'Search results will appear here.', {
            elementId: 'paperSearchMeta',
            text: state.currentPaperSearchMetaText
        });
        state.importPaperController = null;
    }
}

function addPaperToAnalysisByIndex(elementId, index) {
    // Resolved against the list the click actually came from. Reading a fixed
    // list instead means the button says which list it belongs to and the
    // handler ignores it — so the day an "Add" control is rendered on the
    // recommendation list, clicking it would import the search result at the
    // same index. A user who clicks "analyse this recommended paper" and gets a
    // different paper analysed has been told something false about their own
    // click, and the workspace is replaced with it.
    const items = getPaperListByElementId(elementId);
    const item = items[index];
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
        addPaperToAnalysisByIndex(elementId, index);
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
