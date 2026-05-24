let currentSessionId = '';
let currentSessionToken = '';
let currentSections = {};
let panZoomInstance = null;
let mermaidInstance = null;
let mermaidReadyPromise = null;
let currentMermaidSource = '';
let analyzeController = null;
let askController = null;
let paperSearchController = null;
let recommendController = null;
let importPaperController = null;
let analyzeRequestId = 0;
let askRequestId = 0;
let paperSearchRequestId = 0;
let recommendRequestId = 0;
let currentImportPaperKey = '';
let currentAnalysisResult = null;
let currentSourceFileName = '';
let currentElapsedSeconds = null;
let currentOutputFile = '';
let currentChatTurns = [];
let currentPaperSearchResults = [];
let currentPaperRecommendations = [];
let currentSuggestedQuestions = [];
let currentNextActions = [];
let currentReadingQueue = [];
let currentAnswerMode = 'evidence';
let pendingAnalysisSnapshot = null;
let activeStreamingAnswerShell = null;
let currentPaperSearchMetaText = 'Search across Semantic Scholar and arXiv with a single query.';
let currentPaperRecommendationMetaText = 'Analyze a paper first, then generate follow-up reading suggestions from the current session.';

const THEME_STORAGE_KEY = 'paperwhisperer-theme';
const SUPPORTED_UPLOAD_EXTENSIONS = ['.txt', '.pdf', '.docx', '.pptx'];
const MAX_UPLOAD_BYTES = 16 * 1024 * 1024;
const SECTION_EMPTY_TEXT = {
    summary: 'Summary will appear here after analysis.',
    quotes: 'Key citations will appear here after analysis.',
    mindmap: 'Text structure will appear here after analysis.',
    evaluation: 'Critical evaluation will appear here when enabled.',
    research_brief: 'Deep research brief will appear here when enabled.'
};

const ANSWER_MODE_LABELS = {
    evidence: 'Evidence',
    explain: 'Explain',
    critique: 'Critique',
    reproduce: 'Reproduce'
};

const ANSWER_MODE_DETAILS = {
    evidence: 'Evidence mode answers first, then cites document support and uncertainty.',
    explain: 'Explain mode teaches concepts, methods, and formulas step by step.',
    critique: 'Critique mode reviews strengths, assumptions, limitations, and threats to validity.',
    reproduce: 'Reproduce mode turns the paper into steps, variables, dependencies, and risks.'
};

const sunIcon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>`;
const moonIcon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;

function loadMermaidRenderer() {
    if (mermaidInstance) return Promise.resolve(mermaidInstance);
    if (!mermaidReadyPromise) {
        mermaidReadyPromise = import('https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.esm.min.mjs')
            .then(module => {
                mermaidInstance = module.default;
                mermaidInstance.initialize({
                    startOnLoad: false,
                    theme: getCurrentTheme() === 'dark' ? 'dark' : 'base',
                    securityLevel: 'loose'
                });
                return mermaidInstance;
            })
            .catch(error => {
                console.error('Mermaid load failed:', error);
                mermaidReadyPromise = null;
                return null;
            });
    }
    return mermaidReadyPromise;
}

if (window.marked) {
    marked.setOptions({ breaks: true, gfm: true, headerIds: false, mangle: false });
}

document.addEventListener('DOMContentLoaded', () => {
    initializeTheme();
    updateFileMeta();
    resetResultView();
    renderReadingQueue();
    setAnswerMode(currentAnswerMode);
    updateStatus('Waiting for document', 'idle');

    bindClick('themeBtn', toggleTheme);
    bindClick('analyzeBtn', analyze);
    bindClick('paperSearchBtn', searchPapers);
    bindClick('clearReadingQueueBtn', clearReadingQueue);
    bindClick('exportBtn', exportSessionReport);
    bindClick('recommendBtn', recommendPapers);
    bindClick('askBtn', askQuestion);
    bindClick('aiToggleBtn', toggleAIList);
    bindClick('zoomInBtn', zoomIn);
    bindClick('zoomOutBtn', zoomOut);
    bindClick('zoomResetBtn', zoomReset);
    bindClick('downloadMermaidBtn', downloadMermaidSVG);
    bindClick('backToTopBtn', scrollToTop);
    bindClick('cancelAnalyzeBtn', cancelAnalyzeRequest);
    bindClick('cancelPaperSearchBtn', cancelPaperSearchRequest);
    bindClick('cancelRecommendBtn', cancelRecommendRequest);
    bindClick('cancelAskBtn', cancelAskRequest);

    initializeBackToTop();
    initializeUploadDropZone();
    document.getElementById('file')?.addEventListener('change', handleFileSelection);
    document.getElementById('paperSearchInput')?.addEventListener('keydown', handlePaperSearchKeyPress);
    document.getElementById('questionInput')?.addEventListener('keydown', handleKeyPress);
    document.getElementById('paperSearchResults')?.addEventListener('click', handlePaperResultClick);
    document.getElementById('paperRecommendations')?.addEventListener('click', handlePaperResultClick);
    document.getElementById('readingQueue')?.addEventListener('click', handleReadingQueueClick);
    document.addEventListener('keydown', handleWorkspaceShortcut);
    document.querySelectorAll('.mode-chip').forEach(button => {
        button.addEventListener('click', () => setAnswerMode(button.dataset.mode));
    });
    document.querySelectorAll('[data-example-query]').forEach(button => {
        button.addEventListener('click', () => useExampleQuery(button.dataset.exampleQuery));
    });
    document.querySelectorAll('[data-starter-question]').forEach(button => {
        button.addEventListener('click', () => useStarterQuestion(button.dataset.starterQuestion));
    });
    document.querySelectorAll('[data-copy-target]').forEach(button => {
        button.addEventListener('click', () => copyText(button.dataset.copyTarget, button));
    });
});

function bindClick(id, handler) {
    const element = document.getElementById(id);
    if (element) element.addEventListener('click', handler);
}

function setCancelVisible(id, isVisible) {
    const button = document.getElementById(id);
    if (!button) return;
    button.hidden = !isVisible;
    button.disabled = !isVisible;
}

function cancelAnalyzeRequest() {
    if (!analyzeController) return;
    analyzeController.abort();
    analyzeRequestId += 1;
    document.getElementById('loading').classList.remove('active');
    setButtonLoading(document.getElementById('analyzeBtn'), 'Analyzing...', 'Analyze Document', false);
    setCancelVisible('cancelAnalyzeBtn', false);
    analyzeController = null;
    if (pendingAnalysisSnapshot) {
        restoreWorkspaceState(pendingAnalysisSnapshot);
        pendingAnalysisSnapshot = null;
        focusAnalysisWorkspace({ scroll: false });
    } else {
        currentSessionId = '';
        resetResultView();
    }
    updateStatus('Analysis canceled', 'idle');
}

function cancelPaperSearchRequest() {
    if (!paperSearchController) return;
    paperSearchController.abort();
    paperSearchRequestId += 1;
    setButtonLoading(document.getElementById('paperSearchBtn'), 'Searching...', 'Search Papers', false);
    setCancelVisible('cancelPaperSearchBtn', false);
    paperSearchController = null;
    renderPaperList('paperSearchResults', currentPaperSearchResults, currentPaperSearchResults.length ? '' : 'Search results will appear here.', {
        elementId: 'paperSearchMeta',
        text: currentPaperSearchMetaText
    });
    updateStatus('Paper search canceled', 'idle');
}

function cancelRecommendRequest() {
    if (!recommendController) return;
    recommendController.abort();
    recommendRequestId += 1;
    setButtonLoading(document.getElementById('recommendBtn'), 'Recommending...', 'Recommend from current paper', false);
    setRecommendEnabled(Boolean(currentSessionId));
    setCancelVisible('cancelRecommendBtn', false);
    recommendController = null;
    renderPaperList('paperRecommendations', currentPaperRecommendations, currentPaperRecommendations.length ? '' : 'Recommendations will appear here after analysis.', {
        elementId: 'paperRecommendationMeta',
        text: currentPaperRecommendationMetaText
    });
    updateStatus('Recommendations canceled', 'idle');
}

function cancelAskRequest() {
    if (!askController) return;
    askController.abort();
    askRequestId += 1;
    setButtonLoading(document.getElementById('askBtn'), 'Processing...', 'Send', false);
    setCancelVisible('cancelAskBtn', false);
    askController = null;
    if (activeStreamingAnswerShell?.answerDiv?.parentNode) {
        activeStreamingAnswerShell.answerDiv.parentNode.removeChild(activeStreamingAnswerShell.answerDiv);
    }
    activeStreamingAnswerShell = null;
    updateStatus('Question canceled', 'idle');
}

function focusAnalysisWorkspace({ scroll = true } = {}) {
    const result = document.getElementById('result');
    if (!result || !result.classList.contains('active')) return;
    if (scroll) result.scrollIntoView({ behavior: 'smooth', block: 'start' });
    result.focus({ preventScroll: true });
}

function focusWorkspaceTarget(targetId, focusSelector) {
    const target = document.getElementById(targetId);
    if (!target) return;
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    const focusTarget = focusSelector ? target.querySelector(focusSelector) : target;
    if (focusTarget) focusTarget.focus({ preventScroll: true });
}

function handleWorkspaceShortcut(event) {
    if (!event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const key = event.key.toLowerCase();
    if (!['u', 's', 'q'].includes(key)) return;
    event.preventDefault();
    if (key === 'u') {
        focusWorkspaceTarget('uploadWorkspace', '#dropZone');
        updateStatus('Upload shortcut focused document source', 'idle');
    } else if (key === 's') {
        focusWorkspaceTarget('paperSearchTitle', null);
        document.getElementById('paperSearchInput')?.focus({ preventScroll: true });
        updateStatus('Search shortcut focused paper search', 'idle');
    } else {
        focusWorkspaceTarget('askQuestionsTitle', null);
        document.getElementById('questionInput')?.focus({ preventScroll: true });
        updateStatus('Ask shortcut focused question input', 'idle');
    }
}

function updateBackToTopVisibility() {
    const button = document.getElementById('backToTopBtn');
    if (!button) return;
    button.classList.toggle('show', window.scrollY > 640);
}

function initializeBackToTop() {
    updateBackToTopVisibility();
    window.addEventListener('scroll', updateBackToTopVisibility, { passive: true });
}

function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

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

    if (paperSearchController) {
        paperSearchController.abort();
    }
    paperSearchController = new AbortController();
    paperSearchRequestId += 1;
    const requestId = paperSearchRequestId;

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
                session_id: currentSessionToken ? currentSessionId : undefined,
                session_token: currentSessionToken || undefined
            }),
            signal: paperSearchController.signal
        });
        const data = await parseJsonSafely(response);
        if (requestId !== paperSearchRequestId) {
            return;
        }
        if (!response.ok) {
            throw new Error(data.error || 'Paper search failed.');
        }

        currentPaperSearchResults = Array.isArray(data.items) ? data.items : [];
        const rewriteBits = [];
        if (data.original_query) rewriteBits.push(`Original: ${data.original_query}`);
        if (data.rewritten_query) rewriteBits.push(`Rewritten: ${data.rewritten_query}`);
        if (Array.isArray(data.topics) && data.topics.length) rewriteBits.push(`Topics: ${data.topics.join(', ')}`);
        if (data.reason) rewriteBits.push(`Why: ${data.reason}`);
        if (data.rewrite_model) rewriteBits.push(`Model: ${data.rewrite_model}`);
        if (data.errors && data.errors.length) rewriteBits.push(`Partial results: ${data.errors.join(' | ')}`);
        rewriteBits.push(`${currentPaperSearchResults.length} paper(s)`);
        currentPaperSearchMetaText = rewriteBits.join(' · ');
        renderPaperList('paperSearchResults', currentPaperSearchResults, 'No matching papers found.', {
            elementId: 'paperSearchMeta',
            text: currentPaperSearchMetaText
        });
        renderExportPreview();
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        currentPaperSearchResults = [];
        currentPaperSearchMetaText = 'Search across Semantic Scholar and arXiv with a single query.';
        renderPaperList('paperSearchResults', [], 'Search results will appear here.', { elementId: 'paperSearchMeta', text: currentPaperSearchMetaText });
        showError(error.message || 'Paper search failed.');
    } finally {
        if (requestId === paperSearchRequestId) {
            setButtonLoading(searchBtn, 'Searching...', 'Search Papers', false);
            setCancelVisible('cancelPaperSearchBtn', false);
            paperSearchController = null;
        }
    }
}

async function recommendPapers() {
    if (!currentSessionId) {
        showError('Requires document analysis first.');
        return;
    }

    const apiKey = document.getElementById('apiKey').value.trim();
    const recommendBtn = document.getElementById('recommendBtn');
    if (recommendController) {
        recommendController.abort();
    }
    recommendController = new AbortController();
    recommendRequestId += 1;
    const requestId = recommendRequestId;

    setButtonLoading(recommendBtn, 'Recommending...', 'Recommend from current paper', true);
    setCancelVisible('cancelRecommendBtn', true);
    hideError();
    renderPaperList('paperRecommendations', [], 'Generating recommendations...', { elementId: 'paperRecommendationMeta', text: 'Generating search topics from the current paper...' });

    try {
        const response = await fetch('/api/recommend-papers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: currentSessionId, session_token: currentSessionToken, api_key: apiKey }),
            signal: recommendController.signal
        });
        const data = await parseJsonSafely(response);
        if (requestId !== recommendRequestId) {
            return;
        }
        if (!response.ok) {
            throw new Error(data.error || 'Paper recommendation failed.');
        }

        currentPaperRecommendations = Array.isArray(data.items)
            ? data.items.map(item => ({ ...item, reason: data.reason || '' }))
            : [];
        const recommendationBits = [];
        if (data.original_query) recommendationBits.push(`Original: ${data.original_query}`);
        if (data.query) recommendationBits.push(`Rewritten: ${data.query}`);
        if (Array.isArray(data.topics) && data.topics.length) recommendationBits.push(`Topics: ${data.topics.join(', ')}`);
        if (data.reason) recommendationBits.push(`Why: ${data.reason}`);
        if (data.rewrite_model) recommendationBits.push(`Model: ${data.rewrite_model}`);
        if (data.errors && data.errors.length) recommendationBits.push(`Partial results: ${data.errors.join(' | ')}`);
        currentPaperRecommendationMetaText = recommendationBits.join(' · ');
        renderPaperList('paperRecommendations', currentPaperRecommendations, 'No recommendations found.', {
            elementId: 'paperRecommendationMeta',
            text: currentPaperRecommendationMetaText
        });
        renderExportPreview();
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        currentPaperRecommendations = [];
        currentPaperRecommendationMetaText = 'Analyze a paper first, then generate follow-up reading suggestions from the current session.';
        renderPaperList('paperRecommendations', [], 'Recommendations will appear here after analysis.', { elementId: 'paperRecommendationMeta', text: currentPaperRecommendationMetaText });
        showError(error.message || 'Paper recommendation failed.');
    } finally {
        if (requestId === recommendRequestId) {
            setButtonLoading(recommendBtn, 'Recommending...', 'Recommend from current paper', false);
            setRecommendEnabled(Boolean(currentSessionId));
            setCancelVisible('cancelRecommendBtn', false);
            recommendController = null;
        }
    }
}

function getCurrentTheme() {
    return document.body.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

function initializeTheme() {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(savedTheme || (prefersDark ? 'dark' : 'light'));
}

function applyTheme(theme) {
    const body = document.body;
    const btn = document.getElementById('themeBtn');
    if (theme === 'dark') {
        body.setAttribute('data-theme', 'dark');
        btn.innerHTML = sunIcon;
    } else {
        body.removeAttribute('data-theme');
        btn.innerHTML = moonIcon;
    }
}

function toggleTheme() {
    const nextTheme = getCurrentTheme() === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
    localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    if (currentMermaidSource) {
        renderMermaidDiagram(currentMermaidSource);
    }
}

function updateStatus(text, tone = 'idle') {
    const chip = document.getElementById('statusChip');
    if (!chip) return;
    chip.textContent = text;
    chip.dataset.tone = tone;
}

function updateAnalysisProgress(step, text) {
    const loadingText = document.getElementById('loadingText');
    if (loadingText && text) loadingText.textContent = text;

    const order = ['upload', 'analyze', 'render', 'ready'];
    const activeIndex = Math.max(order.indexOf(step), 0);
    document.querySelectorAll('.progress-step').forEach(element => {
        const index = order.indexOf(element.dataset.step);
        element.classList.toggle('done', index >= 0 && index < activeIndex);
        element.classList.toggle('active', index === activeIndex);
    });
}

function toggleAIList() {
    const list = document.getElementById('aiList');
    const button = document.getElementById('aiToggleBtn');
    if (!list || !button) return;
    const shouldShow = !list.classList.contains('show');
    list.classList.toggle('show', shouldShow);
    button.setAttribute('aria-expanded', String(shouldShow));
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function sanitizeUrl(rawUrl) {
    if (!rawUrl) return '#';
    try {
        const parsed = new URL(rawUrl, window.location.origin);
        if (['http:', 'https:', 'mailto:'].includes(parsed.protocol)) {
            return parsed.href;
        }
    } catch (error) {
        console.warn('Unsafe url ignored:', rawUrl, error);
    }
    return '#';
}

function sanitizeGeneratedHtml(html) {
    const template = document.createElement('template');
    template.innerHTML = html;
    const allowedTags = new Set(['A', 'B', 'BLOCKQUOTE', 'BR', 'CODE', 'DEL', 'DIV', 'EM', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'HR', 'I', 'IMG', 'LI', 'OL', 'P', 'PRE', 'S', 'SPAN', 'STRONG', 'TABLE', 'TBODY', 'TD', 'TH', 'THEAD', 'TR', 'UL']);
    const allowedAttributes = new Set(['alt', 'class', 'colspan', 'href', 'loading', 'rel', 'rowspan', 'src', 'target', 'title']);

    template.content.querySelectorAll('*').forEach(node => {
        if (!allowedTags.has(node.tagName)) {
            node.replaceWith(document.createTextNode(node.textContent || ''));
            return;
        }

        Array.from(node.attributes).forEach(attribute => {
            const attributeName = attribute.name.toLowerCase();
            if (!allowedAttributes.has(attributeName) || attributeName.startsWith('on') || attributeName === 'style') {
                node.removeAttribute(attribute.name);
            }
        });
    });

    template.content.querySelectorAll('a').forEach(anchor => {
        anchor.href = sanitizeUrl(anchor.getAttribute('href'));
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
    });

    template.content.querySelectorAll('img').forEach(image => {
        const safeSrc = sanitizeUrl(image.getAttribute('src'));
        if (safeSrc === '#') {
            image.remove();
            return;
        }
        image.src = safeSrc;
        image.loading = 'lazy';
    });

    const container = document.createElement('div');
    container.appendChild(template.content.cloneNode(true));
    return container.innerHTML;
}

function transformPromptTags(content) {
    return String(content).replace(/<(role|context|task|constraints|output_format|input|self_check)>\s*([\s\S]*?)\s*<\/\1>/gi, (_, tag, body) => {
        const safeTag = escapeHtml(tag.replace(/_/g, ' '));
        const safeBody = escapeHtml(body.trim());
        return `\n\n<div class="prompt-block"><div class="prompt-block-title">${safeTag}</div><div class="prompt-block-body">${safeBody}</div></div>\n\n`;
    });
}

function renderMath(element) {
    if (!window.renderMathInElement) return;
    try {
        renderMathInElement(element, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '\\[', right: '\\]', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false }
            ],
            throwOnError: false,
            output: 'html'
        });
    } catch (error) {
        console.warn('Math render failed:', error);
    }
}

function formatContent(content, fallbackText) {
    if (!content) {
        return `<p class="empty-state">${escapeHtml(fallbackText || 'No content available.')}</p>`;
    }

    let processedContent = String(content).replace(/\r\n/g, '\n');
    processedContent = transformPromptTags(processedContent);
    processedContent = processedContent.replace(/\\\[([\s\S]*?)\\\]/g, (_, expr) => `$$${expr}$$`);
    processedContent = processedContent.replace(/\\\(([\s\S]*?)\\\)/g, (_, expr) => `$${expr}$`);

    const mathTokens = {};
    let counter = 0;

    processedContent = processedContent.replace(/\$\$([\s\S]*?)\$\$/g, match => {
        const token = `@@MATHBLOCK${counter}@@`;
        mathTokens[token] = match;
        counter += 1;
        return `\n\n${token}\n\n`;
    });

    processedContent = processedContent.replace(/\$((?!\s)[^$]+?(?!\s))\$/g, match => {
        const token = `@@MATHINLINE${counter}@@`;
        mathTokens[token] = match;
        counter += 1;
        return token;
    });

    let htmlContent = window.marked ? marked.parse(processedContent) : `<p>${escapeHtml(processedContent).replace(/\n{2,}/g, '</p><p>').replace(/\n/g, '<br>')}</p>`;
    Object.entries(mathTokens).forEach(([token, mathStr]) => {
        const blockPattern = new RegExp(`<p>${token}</p>`, 'g');
        if (blockPattern.test(htmlContent)) {
            htmlContent = htmlContent.replace(blockPattern, mathStr);
        } else {
            htmlContent = htmlContent.replace(new RegExp(token, 'g'), mathStr);
        }
    });

    return sanitizeGeneratedHtml(htmlContent);
}

function renderSectionStateCard(title, detail, tone = 'empty') {
    return `
        <div class="section-state section-state-${tone}">
            <span class="section-state-icon" aria-hidden="true">${tone === 'error' ? '!' : 'i'}</span>
            <div>
                <p class="section-state-title">${escapeHtml(title)}</p>
                <p class="section-state-detail">${escapeHtml(detail)}</p>
            </div>
        </div>
    `;
}

function setSectionContent(id, value, errorMessage = '') {
    const element = document.getElementById(id);
    if (!element) return;
    element.dataset.rawContent = value || '';
    if (errorMessage) {
        element.innerHTML = renderSectionStateCard('Section needs review', errorMessage, 'error');
        return;
    }
    if (!value) {
        element.innerHTML = renderSectionStateCard('Waiting for content', SECTION_EMPTY_TEXT[id] || 'No content available.');
        return;
    }
    element.innerHTML = formatContent(value, SECTION_EMPTY_TEXT[id]);
    renderMath(element);
}

function getSectionPayload(data, sectionName) {
    if (data && data.sections && data.sections[sectionName]) {
        return data.sections[sectionName];
    }
    return {
        status: 'success',
        content: data ? (data[sectionName] || '') : '',
        error: '',
        retryable: false
    };
}

function clearChatHistory() {
    document.getElementById('chatHistory').innerHTML = '';
}

function setExportEnabled(enabled) {
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.disabled = !enabled;
    }
}

function resetExportState() {
    currentAnalysisResult = null;
    currentSourceFileName = '';
    currentElapsedSeconds = null;
    currentOutputFile = '';
    currentChatTurns = [];
    currentSections = {};
    currentSessionToken = '';
    resetSmartSuggestions();
    renderExportPreview();
    setExportEnabled(false);
}

function setRecommendEnabled(enabled) {
    const recommendBtn = document.getElementById('recommendBtn');
    if (recommendBtn) {
        recommendBtn.disabled = !enabled;
    }
}

function normalizeSmartTextItems(values, maxItems = 6) {
    if (!Array.isArray(values)) return [];
    const seen = new Set();
    const items = [];
    values.forEach(value => {
        const text = String(value || '').trim();
        if (!text || seen.has(text)) return;
        items.push(text);
        seen.add(text);
    });
    return items.slice(0, maxItems);
}

function normalizeSmartActions(values, maxItems = 5) {
    if (!Array.isArray(values)) return [];
    const seen = new Set();
    const actions = [];
    values.forEach(value => {
        if (!value || typeof value !== 'object') return;
        const label = String(value.label || '').trim();
        const prompt = String(value.prompt || '').trim();
        if (!label || !prompt || seen.has(prompt)) return;
        actions.push({ label, prompt });
        seen.add(prompt);
    });
    return actions.slice(0, maxItems);
}

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
    return Boolean(key) && currentReadingQueue.some(savedItem => getReadingQueueKey(savedItem) === key);
}

function setAnswerMode(mode) {
    currentAnswerMode = ANSWER_MODE_LABELS[mode] ? mode : 'evidence';
    document.querySelectorAll('.mode-chip').forEach(button => {
        const selected = button.dataset.mode === currentAnswerMode;
        button.classList.toggle('active', selected);
        button.setAttribute('aria-pressed', selected ? 'true' : 'false');
    });
    const hint = document.getElementById('answerModeHint');
    if (hint) hint.textContent = ANSWER_MODE_DETAILS[currentAnswerMode];
}

function fillQuestionInput(prompt) {
    const questionInput = document.getElementById('questionInput');
    if (!questionInput) return;
    questionInput.value = prompt;
    questionInput.focus();
}

function useStarterQuestion(question) {
    if (!question) return;
    fillQuestionInput(question);
    updateStatus('Starter question loaded', 'idle');
}

function renderSmartPrompts(questions = currentSuggestedQuestions) {
    const container = document.getElementById('smartPrompts');
    if (!container) return;
    currentSuggestedQuestions = normalizeSmartTextItems(questions);
    container.innerHTML = '';
    container.style.display = currentSuggestedQuestions.length ? 'flex' : 'none';
    currentSuggestedQuestions.forEach(question => {
        const button = document.createElement('button');
        button.className = 'prompt-chip';
        button.type = 'button';
        button.textContent = question;
        button.addEventListener('click', () => fillQuestionInput(question));
        container.appendChild(button);
    });
}

function renderNextActions(actions = currentNextActions) {
    const container = document.getElementById('nextActions');
    if (!container) return;
    currentNextActions = normalizeSmartActions(actions);
    container.innerHTML = '';
    container.style.display = currentNextActions.length ? 'flex' : 'none';
    currentNextActions.forEach(action => {
        const button = document.createElement('button');
        button.className = 'next-action-btn';
        button.type = 'button';
        button.textContent = action.label;
        button.title = action.prompt;
        button.addEventListener('click', () => fillQuestionInput(action.prompt));
        container.appendChild(button);
    });
}

function resetSmartSuggestions() {
    currentSuggestedQuestions = [];
    currentNextActions = [];
    renderSmartPrompts([]);
    renderNextActions([]);
}

function renderReadingQueue(message = '') {
    const container = document.getElementById('readingQueue');
    const meta = document.getElementById('readingQueueMeta');
    const clearButton = document.getElementById('clearReadingQueueBtn');
    if (!container) return;
    currentReadingQueue = normalizeReadingQueue(currentReadingQueue);
    if (clearButton) clearButton.disabled = !currentReadingQueue.length;
    if (meta) {
        meta.textContent = message || (currentReadingQueue.length
            ? `${currentReadingQueue.length} saved paper(s) in this session reading queue.`
            : 'Save papers from search or recommendations into this session reading queue.');
    }
    if (!currentReadingQueue.length) {
        container.innerHTML = '<p class="empty-state">No saved papers yet. Save strong search or recommendation results to build an export-ready reading trail.</p>';
        return;
    }
    container.innerHTML = currentReadingQueue.map((item, index) => {
        const bits = [item.source, item.year, item.venue].filter(Boolean).map(value => escapeHtml(value)).join(' · ');
        const authors = item.authors.length ? escapeHtml(item.authors.join(', ')) : 'Unknown authors';
        const titleUrl = sanitizeUrl(item.url || item.pdf_url || '#');
        const openLink = item.url ? `<a class="paper-link" href="${sanitizeUrl(item.url)}" target="_blank" rel="noopener noreferrer">Open</a>` : '';
        const pdfLink = item.pdf_url ? `<a class="paper-link" href="${sanitizeUrl(item.pdf_url)}" target="_blank" rel="noopener noreferrer">PDF</a>` : '';
        return `
            <article class="queue-item">
                <div class="queue-rank" aria-label="Reading queue item ${index + 1}">${index + 1}</div>
                <div class="queue-body">
                    <div class="queue-title"><a href="${titleUrl}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.title)}</a></div>
                    <div class="queue-details">
                        ${bits ? `<span>${bits}</span>` : '<span>Metadata pending</span>'}
                        <span>${authors}</span>
                    </div>
                    <div class="queue-actions">
                        ${openLink}
                        ${pdfLink}
                        <button class="paper-link" type="button" data-queue-remove-index="${index}">Remove</button>
                    </div>
                </div>
            </article>
        `;
    }).join('');
}

async function saveReadingQueue({ silent = true, message = '' } = {}) {
    renderReadingQueue(message);
    renderExportPreview();
    if (!currentSessionId || !currentSessionToken) return;
    try {
        const response = await fetch('/api/reading-queue', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentSessionId,
                session_token: currentSessionToken,
                items: currentReadingQueue
            })
        });
        const data = await parseJsonSafely(response);
        if (!response.ok) throw new Error(data.error || 'Reading queue save failed.');
        currentReadingQueue = normalizeReadingQueue(data.items || currentReadingQueue);
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
    currentReadingQueue = normalizeReadingQueue([...currentReadingQueue, item]);
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

function handleReadingQueueClick(event) {
    const removeButton = event.target.closest('[data-queue-remove-index]');
    if (!removeButton) return;
    const index = Number(removeButton.dataset.queueRemoveIndex);
    if (!Number.isInteger(index)) return;
    removePaperFromQueue(index);
}

function removePaperFromQueue(index) {
    const removed = currentReadingQueue[index];
    currentReadingQueue.splice(index, 1);
    currentReadingQueue = normalizeReadingQueue(currentReadingQueue);
    saveReadingQueue({ silent: false, message: removed ? `Removed “${removed.title}” from the reading queue.` : 'Removed paper from the reading queue.' });
}

function clearReadingQueue() {
    if (!currentReadingQueue.length) return;
    currentReadingQueue = [];
    saveReadingQueue({ silent: false, message: 'Reading queue cleared.' });
}

function applyAnalysisSection(sectionName, sectionPayload, generateEvaluation, generateMermaid, generateResearchBrief = true) {
    document.getElementById('result').classList.add('active');
    const section = sectionPayload || { status: 'empty', content: '', error: '', retryable: false };
    const errorMessage = section.status === 'failed' ? (section.error || `${sectionName} generation failed.`) : '';

    if (sectionName === 'evaluation') {
        document.getElementById('evaluationCard').style.display = generateEvaluation ? 'block' : 'none';
    }
    if (sectionName === 'research_brief') {
        document.getElementById('researchBriefCard').style.display = generateResearchBrief ? 'block' : 'none';
    }

    if (sectionName === 'mermaid') {
        if (generateMermaid && section.content) {
            requestAnimationFrame(() => {
                renderMermaidDiagram(section.content);
            });
        } else if (!generateMermaid || !section.content) {
            resetMermaidCard();
        }
        return;
    }

    setSectionContent(sectionName, section.content, errorMessage);
}

function finalizeAnalysisResultState(data, fileName) {
    currentAnalysisResult = data;
    currentSections = data.sections || {};
    currentSessionId = data.session_id || currentSessionId;
    currentSessionToken = data.session_token || currentSessionToken;
    currentSourceFileName = fileName;
    currentElapsedSeconds = data.elapsed_seconds ?? null;
    currentOutputFile = data.output_file || '';
    currentChatTurns = [];
    currentSuggestedQuestions = normalizeSmartTextItems(data.suggested_questions || []);
    currentNextActions = normalizeSmartActions(data.next_actions || []);
    renderSmartPrompts(currentSuggestedQuestions);
    renderNextActions(currentNextActions);
    renderExportPreview();
    resetPaperPanels();
    saveReadingQueue();
    setExportEnabled(true);
    document.getElementById('askBtn').disabled = false;
}

function applyAnalysisResult(data, fileName, generateEvaluation, generateMermaid, generateResearchBrief = true) {
    document.getElementById('result').classList.add('active');
    setFileInfo(fileName, data.char_count);

    const summarySection = getSectionPayload(data, 'summary');
    const quotesSection = getSectionPayload(data, 'quotes');
    const mindmapSection = getSectionPayload(data, 'mindmap');
    const evaluationSection = getSectionPayload(data, 'evaluation');
    const researchBriefSection = getSectionPayload(data, 'research_brief');
    const mermaidSection = getSectionPayload(data, 'mermaid');

    applyAnalysisSection('summary', summarySection, generateEvaluation, generateMermaid, generateResearchBrief);
    applyAnalysisSection('quotes', quotesSection, generateEvaluation, generateMermaid, generateResearchBrief);
    applyAnalysisSection('mindmap', mindmapSection, generateEvaluation, generateMermaid, generateResearchBrief);
    applyAnalysisSection('evaluation', evaluationSection, generateEvaluation, generateMermaid, generateResearchBrief);
    applyAnalysisSection('research_brief', researchBriefSection, generateEvaluation, generateMermaid, generateResearchBrief);
    applyAnalysisSection('mermaid', mermaidSection, generateEvaluation, generateMermaid, generateResearchBrief);

    finalizeAnalysisResultState(data, fileName);
    return Promise.resolve();
}

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
        container.innerHTML = `
            <div class="paper-state ${state.tone === 'loading' ? 'loading-state' : ''}">
                <span class="paper-state-icon" aria-hidden="true">${state.tone === 'loading' ? '…' : '⌕'}</span>
                <div>
                    <p class="paper-state-title">${escapeHtml(state.title)}</p>
                    <p class="paper-state-body">${escapeHtml(state.body)}</p>
                </div>
            </div>
        `;
    } else {
        container.innerHTML = normalizedItems.map((item, index) => {
            const title = escapeHtml(item.title || 'Untitled paper');
            const titleUrl = sanitizeUrl(item.url || item.pdf_url || '#');
            const authors = Array.isArray(item.authors) && item.authors.length ? escapeHtml(item.authors.join(', ')) : 'Unknown authors';
            const abstractText = escapeHtml(item.abstract || 'No abstract available.');
            const reasonText = item.reason ? `<div class="paper-reason"><strong>Why:</strong> ${escapeHtml(item.reason)}</div>` : '';
            const tags = [item.source, item.year, item.venue].filter(Boolean).map(tag => `<span class="paper-tag">${escapeHtml(String(tag))}</span>`).join('');
            const actionKey = buildPaperActionKey(item) || String(index);
            const downloadButton = (item.pdf_url || item.url)
                ? `<button class="paper-link" type="button" data-paper-action="download" data-paper-list="${escapeHtml(elementId)}" data-paper-index="${index}">Download</button>`
                : '';
            const addButton = allowImport
                ? `<button class="paper-link" type="button" data-paper-action="add" data-paper-list="${escapeHtml(elementId)}" data-paper-index="${index}" ${currentImportPaperKey === actionKey ? 'disabled' : ''}>${currentImportPaperKey === actionKey ? 'Adding...' : 'Add'}</button>`
                : '';
            const saveButton = `<button class="paper-link" type="button" data-paper-action="save" data-paper-list="${escapeHtml(elementId)}" data-paper-index="${index}">Save</button>`;
            const links = [
                item.url ? `<a class="paper-link" href="${sanitizeUrl(item.url)}" target="_blank" rel="noopener noreferrer">Open</a>` : '',
                item.pdf_url ? `<a class="paper-link" href="${sanitizeUrl(item.pdf_url)}" target="_blank" rel="noopener noreferrer">PDF</a>` : '',
                downloadButton,
                saveButton,
                addButton
            ].join('');
            return `
                <article class="paper-card">
                    <div class="paper-card-title"><a href="${titleUrl}" target="_blank" rel="noopener noreferrer">${title}</a></div>
                    <div class="paper-card-tags">${tags}</div>
                    <div class="paper-authors">${authors}</div>
                    <div class="paper-abstract">${abstractText}</div>
                    ${reasonText}
                    <div class="paper-links">${links}</div>
                </article>
            `;
        }).join('');
    }

    if (meta) {
        const resolvedText = meta.text || '';
        if (meta.elementId === 'paperSearchMeta') {
            currentPaperSearchMetaText = resolvedText;
        }
        if (meta.elementId === 'paperRecommendationMeta') {
            currentPaperRecommendationMetaText = resolvedText;
        }
        const metaElement = document.getElementById(meta.elementId);
        if (metaElement) {
            metaElement.textContent = resolvedText;
        }
    }
}

function resetPaperPanels() {
    currentPaperSearchResults = [];
    currentPaperRecommendations = [];
    currentPaperSearchMetaText = 'Search across Semantic Scholar and arXiv with a single query.';
    currentPaperRecommendationMetaText = 'Analyze a paper first, then generate follow-up reading suggestions from the current session.';
    renderPaperList('paperSearchResults', [], 'Search results will appear here.', { elementId: 'paperSearchMeta', text: currentPaperSearchMetaText });
    renderPaperList('paperRecommendations', [], 'Recommendations will appear here after analysis.', { elementId: 'paperRecommendationMeta', text: currentPaperRecommendationMetaText });
    setRecommendEnabled(Boolean(currentAnalysisResult && currentSessionId));
}

function resetPanZoom() {
    if (panZoomInstance) {
        panZoomInstance.destroy();
        panZoomInstance = null;
    }
}

function resetMermaidCard() {
    resetPanZoom();
    currentMermaidSource = '';
    document.getElementById('mermaidCard').style.display = 'none';
    document.getElementById('mermaidChart').innerHTML = '';
}

function captureWorkspaceState() {
    return {
        currentSessionId,
        currentSessionToken,
        currentSections: { ...currentSections },
        currentMermaidSource,
        currentAnalysisResult,
        currentSourceFileName,
        currentElapsedSeconds,
        currentOutputFile,
        currentChatTurns: [...currentChatTurns],
        currentPaperSearchResults: [...currentPaperSearchResults],
        currentPaperRecommendations: [...currentPaperRecommendations],
        currentSuggestedQuestions: [...currentSuggestedQuestions],
        currentNextActions: currentNextActions.map(action => ({ ...action })),
        currentReadingQueue: currentReadingQueue.map(item => ({ ...item, authors: [...(item.authors || [])] })),
        currentAnswerMode,
        currentPaperSearchMetaText,
        currentPaperRecommendationMetaText,
        html: {
            summary: document.getElementById('summary').innerHTML,
            quotes: document.getElementById('quotes').innerHTML,
            mindmap: document.getElementById('mindmap').innerHTML,
            evaluation: document.getElementById('evaluation').innerHTML,
            researchBrief: document.getElementById('research_brief').innerHTML,
            exportPreview: document.getElementById('exportPreview').innerHTML,
            fileInfo: document.getElementById('fileInfo').innerHTML,
            chatHistory: document.getElementById('chatHistory').innerHTML,
            paperSearchResults: document.getElementById('paperSearchResults').innerHTML,
            paperRecommendations: document.getElementById('paperRecommendations').innerHTML,
            mermaidChart: document.getElementById('mermaidChart').innerHTML
        },
        ui: {
            resultActive: document.getElementById('result').classList.contains('active'),
            evaluationDisplay: document.getElementById('evaluationCard').style.display,
            mermaidDisplay: document.getElementById('mermaidCard').style.display,
            askDisabled: document.getElementById('askBtn').disabled,
            exportDisabled: document.getElementById('exportBtn').disabled,
            recommendDisabled: document.getElementById('recommendBtn').disabled
        }
    };
}

function restoreWorkspaceState(snapshot) {
    if (!snapshot) return;
    currentSessionId = snapshot.currentSessionId;
    currentSessionToken = snapshot.currentSessionToken;
    currentSections = { ...snapshot.currentSections };
    currentMermaidSource = snapshot.currentMermaidSource;
    currentAnalysisResult = snapshot.currentAnalysisResult;
    currentSourceFileName = snapshot.currentSourceFileName;
    currentElapsedSeconds = snapshot.currentElapsedSeconds;
    currentOutputFile = snapshot.currentOutputFile;
    currentChatTurns = [...snapshot.currentChatTurns];
    currentPaperSearchResults = [...snapshot.currentPaperSearchResults];
    currentPaperRecommendations = [...snapshot.currentPaperRecommendations];
    currentSuggestedQuestions = normalizeSmartTextItems(snapshot.currentSuggestedQuestions || []);
    currentNextActions = normalizeSmartActions(snapshot.currentNextActions || []);
    currentReadingQueue = normalizeReadingQueue(snapshot.currentReadingQueue || []);
    setAnswerMode(snapshot.currentAnswerMode || 'evidence');
    currentPaperSearchMetaText = snapshot.currentPaperSearchMetaText;
    currentPaperRecommendationMetaText = snapshot.currentPaperRecommendationMetaText;

    document.getElementById('summary').innerHTML = snapshot.html.summary;
    document.getElementById('quotes').innerHTML = snapshot.html.quotes;
    document.getElementById('mindmap').innerHTML = snapshot.html.mindmap;
    document.getElementById('evaluation').innerHTML = snapshot.html.evaluation;
    document.getElementById('research_brief').innerHTML = snapshot.html.researchBrief || '';
    document.getElementById('exportPreview').innerHTML = snapshot.html.exportPreview;
    document.getElementById('fileInfo').innerHTML = snapshot.html.fileInfo;
    document.getElementById('chatHistory').innerHTML = snapshot.html.chatHistory;
    document.getElementById('paperSearchResults').innerHTML = snapshot.html.paperSearchResults;
    document.getElementById('paperRecommendations').innerHTML = snapshot.html.paperRecommendations;
    document.getElementById('paperSearchMeta').textContent = currentPaperSearchMetaText;
    document.getElementById('paperRecommendationMeta').textContent = currentPaperRecommendationMetaText;
    renderSmartPrompts(currentSuggestedQuestions);
    renderNextActions(currentNextActions);
    renderReadingQueue();
    document.getElementById('mermaidChart').innerHTML = snapshot.html.mermaidChart;

    document.getElementById('result').classList.toggle('active', snapshot.ui.resultActive);
    document.getElementById('evaluationCard').style.display = snapshot.ui.evaluationDisplay;
    document.getElementById('mermaidCard').style.display = snapshot.ui.mermaidDisplay;
    document.getElementById('askBtn').disabled = snapshot.ui.askDisabled;
    document.getElementById('exportBtn').disabled = snapshot.ui.exportDisabled;
    document.getElementById('recommendBtn').disabled = snapshot.ui.recommendDisabled;
    resetPanZoom();
    if (currentMermaidSource) {
        requestAnimationFrame(() => renderMermaidDiagram(currentMermaidSource));
    }
}

function resetResultView() {
    ['summary', 'quotes', 'mindmap', 'evaluation', 'research_brief'].forEach(id => setSectionContent(id, ''));
    document.getElementById('fileInfo').innerHTML = '';
    document.getElementById('result').classList.remove('active');
    document.getElementById('askBtn').disabled = true;
    document.getElementById('evaluationCard').style.display = '';
    document.getElementById('researchBriefCard').style.display = '';
    clearChatHistory();
    resetMermaidCard();
    resetExportState();
    resetPaperPanels();
}

function initializeUploadDropZone() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('file');
    if (!dropZone || !fileInput) return;

    dropZone.addEventListener('click', event => {
        if (event.target !== fileInput) fileInput.click();
    });
    dropZone.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            fileInput.click();
        }
    });
    ['dragenter', 'dragover'].forEach(type => {
        dropZone.addEventListener(type, event => {
            event.preventDefault();
            dropZone.classList.add('drag-over');
        });
    });
    ['dragleave', 'drop'].forEach(type => {
        dropZone.addEventListener(type, event => {
            event.preventDefault();
            dropZone.classList.remove('drag-over');
        });
    });
    dropZone.addEventListener('drop', event => {
        const file = event.dataTransfer?.files?.[0];
        if (!file) return;
        const transfer = new DataTransfer();
        transfer.items.add(file);
        fileInput.files = transfer.files;
        handleFileSelection();
    });
}

function getUploadValidationError(file) {
    if (!file) return 'Please select a document first.';
    const lowerName = file.name.toLowerCase();
    const isSupported = SUPPORTED_UPLOAD_EXTENSIONS.some(extension => lowerName.endsWith(extension));
    if (!isSupported) {
        return `Unsupported file type. Please upload ${SUPPORTED_UPLOAD_EXTENSIONS.join(', ')}.`;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
        return `File is too large (${formatFileSize(file.size)}). Maximum supported size is ${formatFileSize(MAX_UPLOAD_BYTES)}.`;
    }
    return '';
}

function handleFileSelection() {
    const fileInput = document.getElementById('file');
    const file = fileInput?.files?.[0];
    const error = getUploadValidationError(file);
    updateFileMeta(error);
    if (error && file) showError(error);
    if (!error) hideError();
}

function formatFileSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let index = 0;
    while (size >= 1024 && index < units.length - 1) {
        size /= 1024;
        index += 1;
    }
    return `${size.toFixed(size >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
}

function updateFileMeta(validationError = '') {
    const fileInput = document.getElementById('file');
    const file = fileInput?.files?.[0];
    const fileMeta = document.getElementById('fileMeta');
    const dropZone = document.getElementById('dropZone');
    if (!fileMeta) return;
    if (!file) {
        fileMeta.textContent = 'No file selected. Recommended: clean PDF, TXT, DOCX, or PPTX for better structure extraction.';
        dropZone?.classList.remove('has-file', 'has-error');
        return;
    }

    dropZone?.classList.toggle('has-file', !validationError);
    dropZone?.classList.toggle('has-error', Boolean(validationError));
    const status = validationError ? 'Cannot upload' : 'Selected';
    fileMeta.textContent = `${status}: ${file.name} · ${formatFileSize(file.size)}${validationError ? ` · ${validationError}` : ''}`;
}

function setFileInfo(fileName, charCount) {
    const fileInfo = document.getElementById('fileInfo');
    fileInfo.innerHTML = '';

    const left = document.createElement('span');
    left.textContent = 'Document: ';
    const strong = document.createElement('strong');
    strong.textContent = fileName;
    left.appendChild(strong);

    const actions = document.createElement('div');
    actions.className = 'file-info-actions';

    const stats = document.createElement('span');
    stats.textContent = `Tokens/Chars: ${charCount || 'N/A'}`;

    actions.appendChild(stats);
    fileInfo.appendChild(left);
    fileInfo.appendChild(actions);
}

function showError(message) {
    const errorEl = document.getElementById('error');
    const text = String(message || 'Request failed.').trim();
    const tips = [];

    if (/API Key|认证失败|401/i.test(text)) {
        tips.push('Tip: check whether the API key is missing, invalid, or expired.');
    }
    if (/OPENAI_BASE_URL|网页内容|HTML 页面|网页地址/i.test(text)) {
        tips.push('Tip: make sure OPENAI_BASE_URL points to the API endpoint, not a web page.');
    }
    if (/429|限流/i.test(text)) {
        tips.push('Tip: slow down requests or reduce concurrency and try again later.');
    }

    errorEl.textContent = tips.length ? `${text}\n\n${tips.join('\n')}` : text;
    errorEl.style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function hideError() {
    const errorEl = document.getElementById('error');
    errorEl.textContent = '';
    errorEl.style.display = 'none';
}

async function parseJsonSafely(response) {
    const contentType = response.headers.get('content-type') || '';
    const statusText = `${response.status} ${response.statusText || ''}`.trim();
    if (contentType.includes('application/json')) {
        try {
            const data = await response.json();
            return data && typeof data === 'object' ? data : { error: `Unexpected JSON response${statusText ? ` (${statusText})` : ''}.` };
        } catch (error) {
            console.warn('Response json parse failed:', error);
            return { error: `Response returned malformed JSON${statusText ? ` (${statusText})` : ''}.` };
        }
    }

    const text = (await response.text()).trim();
    if (!text) {
        return { error: `Request failed${statusText ? ` (${statusText})` : ''}.` };
    }
    if (/^(<!doctype html|<html|<body)/i.test(text)) {
        return { error: 'Server returned an HTML page instead of an API response. Check the service endpoint or server logs.' };
    }
    return { error: text.length > 500 ? `${text.slice(0, 500)}...` : text };
}

async function readSseStream(response, handlers = {}) {
    if (!response.ok) {
        const data = await parseJsonSafely(response);
        throw new Error(data.error || 'Connection failed.');
    }

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('text/event-stream')) {
        const data = await parseJsonSafely(response);
        throw new Error(data.error || 'Streaming response was not returned.');
    }

    const reader = response.body && response.body.getReader ? response.body.getReader() : null;
    if (!reader) {
        throw new Error('Browser does not support streaming responses.');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    const dispatchBlock = block => {
        const lines = String(block || '').split('\n');
        let eventName = 'message';
        const dataLines = [];
        lines.forEach(line => {
            if (!line || line.startsWith(':')) return;
            if (line.startsWith('event:')) {
                eventName = line.slice(6).trim() || 'message';
            } else if (line.startsWith('data:')) {
                dataLines.push(line.slice(5).trim());
            }
        });
        if (!dataLines.length) return;
        let payload = {};
        const rawData = dataLines.join('\n');
        try {
            payload = JSON.parse(rawData);
        } catch (error) {
            payload = { raw: rawData };
        }
        const handler = handlers[eventName] || handlers.message;
        if (handler) {
            handler(payload);
        }
    };

    while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

        const normalized = buffer.replace(/\r\n/g, '\n');
        const blocks = normalized.split('\n\n');
        buffer = blocks.pop() || '';
        blocks.forEach(dispatchBlock);

        if (done) {
            if (buffer.trim()) {
                dispatchBlock(buffer);
            }
            break;
        }
    }
}

function setButtonLoading(button, loadingText, defaultText, isLoading) {
    if (!button) return;
    button.disabled = isLoading;
    button.textContent = isLoading ? loadingText : defaultText;
    button.classList.toggle('is-busy', isLoading);
    if (isLoading) {
        button.setAttribute('aria-busy', 'true');
    } else {
        button.removeAttribute('aria-busy');
    }
}

function normalizeMermaidSource(source) {
    return String(source || '').replace(/```mermaid\n?/gi, '').replace(/```\n?/g, '').trim();
}

async function renderMermaidDiagram(source) {
    const mermaidCard = document.getElementById('mermaidCard');
    const mermaidDiv = document.getElementById('mermaidChart');
    const cleanSource = normalizeMermaidSource(source);

    if (!cleanSource) {
        resetMermaidCard();
        return;
    }

    const instance = await loadMermaidRenderer();
    mermaidCard.style.display = 'block';
    mermaidDiv.innerHTML = '';
    currentMermaidSource = cleanSource;

    if (!instance) {
        mermaidDiv.innerHTML = '<p class="empty-state" style="padding:20px;">Mermaid failed to load.</p>';
        return;
    }

    try {
        instance.initialize({ startOnLoad: false, theme: getCurrentTheme() === 'dark' ? 'dark' : 'base', securityLevel: 'loose' });
        const id = `mermaid-${Date.now()}`;
        const { svg } = await instance.render(id, cleanSource);
        mermaidDiv.innerHTML = svg;

        const svgElement = mermaidDiv.querySelector('svg');
        if (!svgElement) return;
        svgElement.style.maxWidth = 'none';
        svgElement.style.width = '100%';
        svgElement.style.height = '100%';

        resetPanZoom();
        if (!window.svgPanZoom) return;
        panZoomInstance = svgPanZoom(svgElement, {
            zoomEnabled: true,
            controlIconsEnabled: false,
            fit: true,
            center: true,
            minZoom: 0.5,
            maxZoom: 15
        });
    } catch (error) {
        console.error('Mermaid render failed:', error);
        mermaidDiv.innerHTML = `<p style="color:#d9480f; font-size:14px; padding:20px;">Structure too complex to render. Raw data fallback:</p><pre style="text-align:left; margin:20px; white-space:pre-wrap;">${escapeHtml(cleanSource)}</pre>`;
    }
}

async function analyze() {
    const apiKey = document.getElementById('apiKey').value.trim();
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    const generateMermaid = document.getElementById('generateMermaid').checked;
    const generateEvaluation = document.getElementById('generateEvaluation').checked;
    const generateResearchBrief = document.getElementById('generateResearchBrief').checked;
    const analyzeBtn = document.getElementById('analyzeBtn');
    const askBtn = document.getElementById('askBtn');

    const uploadError = getUploadValidationError(file);
    if (uploadError) {
        updateFileMeta(file ? uploadError : '');
        showError(uploadError);
        updateStatus('Upload needs attention', 'error');
        return;
    }

    const previousWorkspace = currentAnalysisResult ? captureWorkspaceState() : null;

    if (analyzeController) {
        analyzeController.abort();
    }
    analyzeController = new AbortController();
    analyzeRequestId += 1;
    const requestId = analyzeRequestId;

    resetResultView();
    currentSessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    pendingAnalysisSnapshot = previousWorkspace;
    setButtonLoading(analyzeBtn, 'Analyzing...', 'Analyze Document', true);
    setCancelVisible('cancelAnalyzeBtn', true);
    document.getElementById('loading').classList.add('active');
    updateAnalysisProgress('upload', 'Uploading and preparing your document...');
    hideError();
    askBtn.disabled = true;
    updateStatus('Analyzing document...', 'idle');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', currentSessionId);
    formData.append('generate_mermaid', String(generateMermaid));
    formData.append('generate_evaluation', String(generateEvaluation));
    formData.append('generate_research_brief', String(generateResearchBrief));
    if (apiKey) formData.append('api_key', apiKey);

    try {
        const response = await fetch('/api/analyze/stream', {
            method: 'POST',
            body: formData,
            signal: analyzeController.signal
        });

        await readSseStream(response, {
            start: payload => {
                currentSessionId = payload.session_id || currentSessionId;
                document.getElementById('result').classList.add('active');
                setFileInfo(file.name, '...');
                updateAnalysisProgress('analyze', 'AI is analyzing structure, citations, and research signals...');
                updateStatus('Analysis started. Streaming sections...', 'idle');
            },
            section: payload => {
                if (requestId !== analyzeRequestId) return;
                const sectionName = payload.name;
                const section = payload.section || {};
                currentSections = { ...currentSections, [sectionName]: section };
                applyAnalysisSection(sectionName, section, generateEvaluation, generateMermaid, generateResearchBrief);
                updateAnalysisProgress('render', `Rendering ${sectionName.replace('_', ' ')} results...`);
                updateStatus(`Streaming ${sectionName}...`, 'idle');
            },
            done: async payload => {
                if (requestId !== analyzeRequestId) return;
                await applyAnalysisResult(payload, file.name, generateEvaluation, generateMermaid, generateResearchBrief);
                updateAnalysisProgress('ready', 'Workspace ready. You can ask questions or export the session.');
                updateStatus('Analysis ready for follow-up questions', 'success');
                focusAnalysisWorkspace();
            },
            error: payload => {
                throw new Error(payload.error || 'Analysis failed.');
            }
        });
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        if (previousWorkspace) {
            restoreWorkspaceState(previousWorkspace);
            showError(`${error.message || 'Analysis failed.'}\n\nYour previous successful workspace has been restored.`);
            focusAnalysisWorkspace({ scroll: false });
        } else {
            currentSessionId = '';
            resetResultView();
            showError(error.message || 'Analysis failed.');
        }
        updateStatus('Analysis failed', 'error');
    } finally {
        if (requestId === analyzeRequestId) {
            document.getElementById('loading').classList.remove('active');
            setButtonLoading(analyzeBtn, 'Analyzing...', 'Analyze Document', false);
            setCancelVisible('cancelAnalyzeBtn', false);
            analyzeController = null;
            pendingAnalysisSnapshot = null;
        }
    }
}

function appendChatText(role, message, className) {
    const chatHistory = document.getElementById('chatHistory');
    const item = document.createElement('div');
    item.className = `chat-item ${className}`.trim();

    const header = document.createElement('div');
    header.className = 'chat-header';
    header.textContent = role;

    const body = document.createElement('div');
    body.className = className.includes('chat-answer') ? 'chat-text content' : 'chat-text';
    if (className.includes('chat-answer')) {
        body.dataset.rawContent = message || '';
        body.innerHTML = formatContent(message, 'No content available.');
        renderMath(body);
    } else {
        body.textContent = message;
    }

    item.appendChild(header);
    item.appendChild(body);
    chatHistory.appendChild(item);
    return item;
}

function appendLoadingAnswer() {
    const chatHistory = document.getElementById('chatHistory');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chat-item chat-answer';
    loadingDiv.innerHTML = '<div class="chat-header">Assistant</div><div class="spinner" style="width:16px;height:16px;border-width:2px;margin:8px 0 0 0;"></div>';
    chatHistory.appendChild(loadingDiv);
    return loadingDiv;
}

function appendStreamingAnswerShell() {
    const chatHistory = document.getElementById('chatHistory');
    const answerId = `answer-${Date.now()}`;
    const answerDiv = document.createElement('div');
    answerDiv.className = 'chat-item chat-answer';

    const header = document.createElement('div');
    header.className = 'chat-header';

    const title = document.createElement('span');
    title.textContent = 'Assistant';

    const button = document.createElement('button');
    button.className = 'action-btn';
    button.type = 'button';
    button.textContent = 'Copy';
    button.disabled = true;

    const content = document.createElement('div');
    content.id = answerId;
    content.className = 'answer-content content';
    content.dataset.rawContent = '';
    content.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px;margin:8px 0 0 0;"></div>';

    header.appendChild(title);
    header.appendChild(button);
    answerDiv.appendChild(header);
    answerDiv.appendChild(content);
    chatHistory.appendChild(answerDiv);
    return { answerDiv, content, button, answerId };
}

function updateStreamingAnswer(shell, answer) {
    if (!shell || !shell.content) return;
    shell.content.dataset.rawContent = answer || '';
    shell.content.innerHTML = formatContent(answer, 'No answer available.');
    renderMath(shell.content);
}

function finalizeStreamingAnswer(shell, answer) {
    if (!shell || !shell.button || !shell.content) return;
    updateStreamingAnswer(shell, answer);
    shell.button.disabled = false;
    if (!shell.button.dataset.bound) {
        shell.button.addEventListener('click', () => copyText(shell.answerId, shell.button));
        shell.button.dataset.bound = 'true';
    }
    if (!shell.actions) {
        const actions = document.createElement('div');
        actions.className = 'answer-actions';
        currentNextActions.slice(0, 3).forEach(action => {
            const button = document.createElement('button');
            button.className = 'mini-action-btn';
            button.type = 'button';
            button.textContent = action.label;
            button.title = action.prompt;
            button.addEventListener('click', () => fillQuestionInput(action.prompt));
            actions.appendChild(button);
        });
        if (actions.children.length) {
            shell.answerDiv.appendChild(actions);
            shell.actions = actions;
        }
    }
}

function appendAnswer(answer) {
    const shell = appendStreamingAnswerShell();
    finalizeStreamingAnswer(shell, answer);
}

async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();
    const apiKey = document.getElementById('apiKey').value.trim();
    const askBtn = document.getElementById('askBtn');

    if (!question) {
        questionInput.focus();
        updateStatus('Question needed', 'idle');
        return;
    }
    if (!currentSessionId) {
        questionInput.focus();
        updateStatus('Analyze a document before asking', 'idle');
        showError('Requires document analysis first.');
        return;
    }

    if (askController) {
        askController.abort();
    }
    askController = new AbortController();
    askRequestId += 1;
    const requestId = askRequestId;

    hideError();
    appendChatText(`User · ${ANSWER_MODE_LABELS[currentAnswerMode]}`, question, 'chat-question');
    questionInput.value = '';
    setButtonLoading(askBtn, 'Processing...', 'Send', true);
    setCancelVisible('cancelAskBtn', true);
    updateStatus('Answering based on current document...', 'idle');
    const streamingShell = appendStreamingAnswerShell();
    activeStreamingAnswerShell = streamingShell;
    let accumulatedAnswer = '';

    try {
        const response = await fetch('/api/ask/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, session_id: currentSessionId, session_token: currentSessionToken, api_key: apiKey, answer_mode: currentAnswerMode }),
            signal: askController.signal
        });

        await readSseStream(response, {
            start: () => {
                updateStatus('Streaming answer...', 'idle');
            },
            delta: payload => {
                if (requestId !== askRequestId) return;
                accumulatedAnswer += payload.text || '';
                updateStreamingAnswer(streamingShell, accumulatedAnswer);
            },
            done: payload => {
                if (requestId !== askRequestId) return;
                accumulatedAnswer = payload.answer || accumulatedAnswer;
                finalizeStreamingAnswer(streamingShell, accumulatedAnswer);
                currentChatTurns.push({
                    question,
                    answer: accumulatedAnswer,
                    answer_mode: currentAnswerMode,
                    timestamp: new Date().toISOString()
                });
                renderExportPreview();
                activeStreamingAnswerShell = null;
                updateStatus('Answer ready', 'success');
            },
            error: payload => {
                throw new Error(payload.error || 'Question request failed.');
            }
        });
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        if (streamingShell && streamingShell.answerDiv && streamingShell.answerDiv.parentNode) {
            streamingShell.answerDiv.parentNode.removeChild(streamingShell.answerDiv);
        }
        appendChatText('System Error', error.message || 'Question request failed.', 'chat-answer chat-error');
        updateStatus('Question request failed', 'error');
    } finally {
        if (requestId === askRequestId) {
            setButtonLoading(askBtn, 'Processing...', 'Send', false);
            setCancelVisible('cancelAskBtn', false);
            askController = null;
            activeStreamingAnswerShell = null;
            questionInput.focus();
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !document.getElementById('askBtn').disabled) {
        event.preventDefault();
        askQuestion();
    }
}

function handlePaperSearchKeyPress(event) {
    if (event.key === 'Enter' && !document.getElementById('paperSearchBtn').disabled) {
        event.preventDefault();
        searchPapers();
    }
}

function getPaperListByElementId(elementId) {
    if (elementId === 'paperSearchResults') return currentPaperSearchResults;
    if (elementId === 'paperRecommendations') return currentPaperRecommendations;
    return [];
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

    if (importPaperController) {
        importPaperController.abort();
    }
    importPaperController = new AbortController();
    currentImportPaperKey = paperKey;
    hideError();
    resetResultView();
    currentSessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    document.getElementById('loading').classList.add('active');
    updateStatus('Importing paper from search results...', 'idle');
    renderPaperList('paperSearchResults', currentPaperSearchResults, 'Search results will appear here.', {
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
                session_id: currentSessionId,
                session_token: currentSessionToken,
                api_key: apiKey,
                generate_mermaid: generateMermaid,
                generate_evaluation: generateEvaluation,
                generate_research_brief: generateResearchBrief
            }),
            signal: importPaperController.signal
        });
        const data = await parseJsonSafely(response);
        if (!response.ok) {
            throw new Error(data.error || 'Paper import failed.');
        }

        currentPaperSearchMetaText = 'Imported selected paper into PaperWhisperer.';
        await applyAnalysisResult(data, data.source_filename || item.title || 'Imported paper', generateEvaluation, generateMermaid, generateResearchBrief);
        updateStatus('Imported paper ready for follow-up questions', 'success');
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        currentSessionId = '';
        resetResultView();
        showError(error.message || 'Paper import failed.');
        updateStatus('Paper import failed', 'error');
    } finally {
        currentImportPaperKey = '';
        document.getElementById('loading').classList.remove('active');
        renderPaperList('paperSearchResults', currentPaperSearchResults, 'Search results will appear here.', {
            elementId: 'paperSearchMeta',
            text: currentPaperSearchMetaText
        });
        importPaperController = null;
    }
}

function addPaperToAnalysisByIndex(index) {
    const item = currentPaperSearchResults[index];
    if (!item) {
        showError('Paper result not found. Please search again.');
        return;
    }
    addPaperToAnalysis(item);
}

async function copyText(elementId, btnElement) {
    const content = document.getElementById(elementId);
    if (!content || !btnElement) return;

    const rawText = (content.dataset && typeof content.dataset.rawContent === 'string')
        ? content.dataset.rawContent
        : '';
    const text = (rawText || content.innerText || content.textContent || '').trim();
    const originalText = btnElement.innerText;
    if (!text) {
        btnElement.innerText = 'No content';
        updateStatus('Nothing to copy yet', 'idle');
        setTimeout(() => { btnElement.innerText = originalText; }, 1600);
        return;
    }

    btnElement.disabled = true;
    btnElement.setAttribute('aria-busy', 'true');
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.setAttribute('readonly', '');
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
        }
        btnElement.innerText = 'Copied';
        btnElement.classList.add('action-success');
        updateStatus('Content copied', 'success');
    } catch (error) {
        console.warn('Copy failed:', error);
        btnElement.innerText = 'Copy failed';
        updateStatus('Copy failed', 'error');
    }
    setTimeout(() => {
        btnElement.innerText = originalText;
        btnElement.disabled = false;
        btnElement.removeAttribute('aria-busy');
        btnElement.classList.remove('action-success');
    }, 1600);
}

function zoomIn() { if (panZoomInstance) panZoomInstance.zoomIn(); }
function zoomOut() { if (panZoomInstance) panZoomInstance.zoomOut(); }
function zoomReset() { if (panZoomInstance) { panZoomInstance.resetZoom(); panZoomInstance.center(); } }

function getCurrentSvgSource() {
    const svgElement = document.querySelector('#mermaidChart svg');
    if (!svgElement) return '';
    const serializer = new XMLSerializer();
    let source = serializer.serializeToString(svgElement);
    if (!source.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    if (!source.match(/^<svg[^>]+"http\:\/\/www\.w3\.org\/1999\/xlink"/)) source = source.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
    return '<?xml version="1.0" standalone="no"?>\r\n' + source;
}

function triggerTextDownload(fileName, content, mimeType = 'text/plain;charset=utf-8') {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function sanitizeFileStem(name) {
    return String(name || 'paperwhisperer_session')
        .replace(/\.[^.]+$/, '')
        .replace(/[^\w\u4e00-\u9fa5-]+/g, '_')
        .replace(/^_+|_+$/g, '') || 'paperwhisperer_session';
}

function formatExportTimestamp(value) {
    const date = value ? new Date(value) : new Date();
    if (Number.isNaN(date.getTime())) {
        return new Date().toLocaleString();
    }
    return date.toLocaleString();
}

function formatMarkdownList(items) {
    const normalizedItems = normalizeSmartTextItems(items, 12);
    return normalizedItems.length ? normalizedItems.map(item => `- ${item}`).join('\n') : '- _None._';
}

function formatMarkdownActions(actions) {
    const normalizedActions = normalizeSmartActions(actions, 8);
    return normalizedActions.length
        ? normalizedActions.map(action => `- **${action.label}**: ${action.prompt}`).join('\n')
        : '- _None._';
}

function formatWorkspaceGuidanceMarkdown(items) {
    return items.map(item => `- **${item.label}**: ${item.value} — ${item.detail}`).join('\n');
}

function formatSectionStatusTable(status) {
    const completed = Array.isArray(status.completed_sections) ? status.completed_sections.join(', ') : 'N/A';
    const failed = Array.isArray(status.failed_sections) && status.failed_sections.length ? status.failed_sections.join(', ') : 'None';
    const disabled = Array.isArray(status.disabled_sections) && status.disabled_sections.length ? status.disabled_sections.join(', ') : 'None';
    return [
        '| Status | Sections |',
        '| --- | --- |',
        `| Completed | ${completed || 'None'} |`,
        `| Failed | ${failed} |`,
        `| Disabled | ${disabled} |`,
        `| Quality | ${status.quality || 'N/A'} |`
    ].join('\n');
}

function formatPaperTrace(items) {
    const normalizedItems = Array.isArray(items) ? items.slice(0, 8) : [];
    if (!normalizedItems.length) return '- _None._';
    return normalizedItems.map((item, index) => {
        const bits = [item.source, item.year, item.venue].filter(Boolean).join(' · ');
        const suffix = bits ? ` — ${bits}` : '';
        return `${index + 1}. ${item.title || 'Untitled paper'}${suffix}`;
    }).join('\n');
}

function getWorkspaceGuidanceItems(status, completed) {
    const failed = Array.isArray(status.failed_sections) ? status.failed_sections.length : 0;
    const disabled = Array.isArray(status.disabled_sections) ? status.disabled_sections.length : 0;
    const paperLeadCount = currentPaperSearchResults.length + currentPaperRecommendations.length;
    const nextAction = currentNextActions[0];
    return [
        {
            label: 'Workspace state',
            value: failed ? `${completed} ready · ${failed} need review` : `${completed} sections ready`,
            detail: disabled ? `${disabled} optional section(s) disabled for this run.` : 'Core analysis cards are ready for review and export.'
        },
        {
            label: 'Best next step',
            value: nextAction?.label || 'Ask a grounded follow-up',
            detail: nextAction?.prompt || 'Use Evidence mode to verify claims before switching to critique or reproduce mode.'
        },
        {
            label: 'Research trail',
            value: currentReadingQueue.length ? `${currentReadingQueue.length} saved paper(s)` : `${paperLeadCount} paper lead(s)`,
            detail: currentReadingQueue.length ? 'Saved papers are included in the session export.' : 'Search or recommend papers, then save the strongest leads to the queue.'
        }
    ];
}

function renderExportPreview() {
    const preview = document.getElementById('exportPreview');
    if (!preview) return;
    if (!currentAnalysisResult) {
        preview.innerHTML = '<p class="empty-state">Export the current analysis, Q&A history, research trace, and Mermaid assets after a successful run.</p>';
        return;
    }

    const status = currentAnalysisResult.analysis_status || {};
    const completed = Array.isArray(status.completed_sections) ? status.completed_sections.length : Object.keys(currentSections || {}).length;
    const failed = Array.isArray(status.failed_sections) ? status.failed_sections.length : 0;
    const paperLeadCount = currentPaperSearchResults.length + currentPaperRecommendations.length;
    const exportPreviewItems = [
        { label: 'Analysis', value: `${completed} ready`, detail: failed ? `${failed} section(s) need review` : 'Core sections are ready' },
        { label: 'Q&A', value: `${currentChatTurns.length} turns`, detail: currentChatTurns.length ? 'Local conversation will be included' : 'Ask follow-ups to enrich the report' },
        { label: 'Papers', value: `${paperLeadCount} leads`, detail: currentReadingQueue.length ? `${currentReadingQueue.length} saved to queue` : 'Save strong leads before export' },
        { label: 'Visuals', value: currentMermaidSource ? 'Map ready' : 'No map', detail: currentMermaidSource ? 'Mermaid SVG can be exported' : 'Enable structure diagram for visuals' }
    ];
    const guidanceItems = getWorkspaceGuidanceItems(status, completed);
    preview.innerHTML = `
        <div class="export-preview-grid" aria-label="Export contents summary">
            ${exportPreviewItems.map(item => `
                <div class="export-preview-card">
                    <span>${escapeHtml(item.label)}</span>
                    <strong>${escapeHtml(item.value)}</strong>
                    <small>${escapeHtml(item.detail)}</small>
                </div>
            `).join('')}
        </div>
        <div class="workspace-guidance" aria-label="Workspace guidance">
            ${guidanceItems.map(item => `
                <div class="workspace-guidance-item">
                    <span>${escapeHtml(item.label)}</span>
                    <strong>${escapeHtml(item.value)}</strong>
                    <small>${escapeHtml(item.detail)}</small>
                </div>
            `).join('')}
        </div>
        <p class="export-preview-note">Includes analysis, research brief, reading queue, suggested follow-ups, research trace, and local Q&A history.</p>
    `;
}

function buildSessionMarkdown() {
    if (!currentAnalysisResult) return '';

    const reportTime = new Date();
    const svgSource = getCurrentSvgSource();
    const fileStem = sanitizeFileStem(currentSourceFileName || currentAnalysisResult.session_id || 'paperwhisperer_session');
    const svgFileName = `${fileStem}_visual_map.svg`;
    const status = currentAnalysisResult.analysis_status || {};
    const completed = Array.isArray(status.completed_sections) ? status.completed_sections.length : Object.keys(currentSections || {}).length;
    const guidanceItems = getWorkspaceGuidanceItems(status, completed);
    const lines = [
        '# PaperWhisperer Session Report',
        '',
        '> Rich export of the current analysis session, including follow-up Q&A and visual assets.',
        '> Generated by [PaperWhisperer](https://github.com/AiFLYF/PaperWhisperer).',
        '',
        '---',
        '',
        '## Session Overview',
        '',
        '| Item | Value |',
        '| --- | --- |',
        `| Source file | ${currentSourceFileName || 'N/A'} |`,
        `| Session ID | ${currentAnalysisResult.session_id || currentSessionId || 'N/A'} |`,
        `| Generated at | ${formatExportTimestamp(reportTime.toISOString())} |`,
        `| Analysis duration | ${currentElapsedSeconds ?? 'N/A'} s |`,
        `| Character count | ${currentAnalysisResult.char_count ?? 'N/A'} |`,
        `| Q&A turns | ${currentChatTurns.length} |`,
        `| Saved papers | ${currentReadingQueue.length} |`,
        '',
        '### Workspace Guidance',
        '',
        formatWorkspaceGuidanceMarkdown(guidanceItems),
        '',
        '### Section Status',
        '',
        formatSectionStatusTable(status),
        '',
        '---',
        '',
        '## Overview',
        '',
        currentAnalysisResult.summary || '_No summary generated._',
        '',
        '---',
        '',
        '## Key Citations',
        '',
        currentAnalysisResult.quotes || '_No citations generated._',
        '',
        '---',
        '',
        '## Text Structure',
        '',
        currentAnalysisResult.mindmap || '_No text structure generated._'
    ];

    if (currentAnalysisResult.evaluation) {
        lines.push('', '---', '', '## Evaluation', '', currentAnalysisResult.evaluation);
    }

    if (currentAnalysisResult.research_brief) {
        lines.push('', '---', '', '## Deep Research Brief', '', currentAnalysisResult.research_brief);
    }

    if (currentSuggestedQuestions.length || currentNextActions.length) {
        lines.push(
            '',
            '---',
            '',
            '## Suggested Follow-ups',
            '',
            '### Questions',
            '',
            formatMarkdownList(currentSuggestedQuestions),
            '',
            '### Next Actions',
            '',
            formatMarkdownActions(currentNextActions)
        );
    }

    if (currentPaperSearchResults.length || currentPaperRecommendations.length || currentReadingQueue.length) {
        lines.push(
            '',
            '---',
            '',
            '## Research Trace',
            '',
            '### Reading Queue',
            '',
            formatPaperTrace(currentReadingQueue),
            '',
            '### Paper Search Results',
            '',
            formatPaperTrace(currentPaperSearchResults),
            '',
            '### Auto Recommendations',
            '',
            formatPaperTrace(currentPaperRecommendations)
        );
    }

    if (currentMermaidSource) {
        lines.push(
            '',
            '---',
            '',
            '## Mermaid Source',
            '',
            '```mermaid',
            currentMermaidSource,
            '```'
        );

        if (svgSource) {
            lines.push(
                '',
                '### Visual Map SVG',
                '',
                `The rendered SVG is exported as a companion file: \`${svgFileName}\`.`
            );
        }
    }

    if (currentChatTurns.length) {
        lines.push('', '---', '', '## Ask Questions', '');
        currentChatTurns.forEach((turn, index) => {
            const modeLabel = ANSWER_MODE_LABELS[turn.answer_mode] || ANSWER_MODE_LABELS.evidence;
            lines.push(
                `### Q${index + 1}`,
                '',
                `Mode: ${modeLabel}`,
                '',
                turn.question || '_No question text._',
                '',
                `### A${index + 1}`,
                '',
                turn.answer || '_No answer text._',
                ''
            );
        });
    }

    lines.push('', '---', '', '## Export Metadata', '', `- App: [PaperWhisperer](https://github.com/AiFLYF/PaperWhisperer)`, `- Session export time: ${formatExportTimestamp(reportTime.toISOString())}`);
    return lines.join('\n');
}

function exportSessionReport() {
    const exportBtn = document.getElementById('exportBtn');
    if (!currentAnalysisResult) {
        updateStatus('Analyze a document before exporting', 'idle');
        return;
    }

    const fileStem = sanitizeFileStem(currentSourceFileName || currentAnalysisResult.session_id || 'paperwhisperer_session');
    const markdown = buildSessionMarkdown();
    if (!markdown) {
        updateStatus('Export content is not ready', 'error');
        return;
    }

    triggerTextDownload(`${fileStem}_session_report.md`, markdown, 'text/markdown;charset=utf-8');

    const svgSource = getCurrentSvgSource();
    if (svgSource) {
        triggerTextDownload(`${fileStem}_visual_map.svg`, svgSource, 'image/svg+xml;charset=utf-8');
    }

    if (exportBtn) {
        const originalText = exportBtn.innerText;
        exportBtn.innerText = svgSource ? 'Exported Report + SVG' : 'Exported Report';
        exportBtn.classList.add('action-success');
        setTimeout(() => {
            exportBtn.innerText = originalText;
            exportBtn.classList.remove('action-success');
        }, 1800);
    }
    updateStatus(svgSource ? 'Session report and SVG exported' : 'Session report exported', 'success');
}

function downloadMermaidSVG() {
    const svgSource = getCurrentSvgSource();
    if (!svgSource) return;
    triggerTextDownload(`paper_map_${Date.now()}.svg`, svgSource, 'image/svg+xml;charset=utf-8');
}
