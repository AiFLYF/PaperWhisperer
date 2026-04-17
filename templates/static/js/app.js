let currentSessionId = '';
let currentSessionToken = '';
let currentSections = {};
let panZoomInstance = null;
let mermaidInstance = null;
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
let currentPaperSearchMetaText = 'Search across Semantic Scholar and arXiv with a single query.';
let currentPaperRecommendationMetaText = 'Analyze a paper first, then generate follow-up reading suggestions from the current session.';

const THEME_STORAGE_KEY = 'paperwhisperer-theme';
const SECTION_EMPTY_TEXT = {
    summary: 'Summary will appear here after analysis.',
    quotes: 'Key citations will appear here after analysis.',
    mindmap: 'Text structure will appear here after analysis.',
    evaluation: 'Critical evaluation will appear here when enabled.'
};

const sunIcon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>`;
const moonIcon = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;

const mermaidReady = import('https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.esm.min.mjs')
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
        return null;
    });

marked.setOptions({ breaks: true, gfm: true, headerIds: false, mangle: false });

document.addEventListener('DOMContentLoaded', () => {
    initializeTheme();
    updateFileMeta();
    resetResultView();
    updateStatus('Waiting for document', 'idle');
    document.getElementById('file').addEventListener('change', updateFileMeta);
});

async function searchPapers() {
    const queryInput = document.getElementById('paperSearchInput');
    const searchBtn = document.getElementById('paperSearchBtn');
    const query = queryInput.value.trim();

    if (!query) {
        showError('Please enter a paper search query.');
        return;
    }

    if (paperSearchController) {
        paperSearchController.abort();
    }
    paperSearchController = new AbortController();
    paperSearchRequestId += 1;
    const requestId = paperSearchRequestId;

    setButtonLoading(searchBtn, 'Searching...', 'Search Papers', true);
    hideError();
    renderPaperList('paperSearchResults', [], 'Searching papers...', { elementId: 'paperSearchMeta', text: `Searching for: ${query}` });

    try {
        const response = await fetch('/api/search-papers', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                session_id: currentSessionId || undefined,
                session_token: currentSessionId ? currentSessionToken : undefined
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

    recommendBtn.disabled = true;
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
            setRecommendEnabled(Boolean(currentSessionId));
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
    const container = document.createElement('div');
    container.innerHTML = html;

    container.querySelectorAll('*').forEach(node => {
        Array.from(node.attributes).forEach(attribute => {
            if (/^on/i.test(attribute.name)) {
                node.removeAttribute(attribute.name);
            }
        });
    });

    container.querySelectorAll('a').forEach(anchor => {
        anchor.href = sanitizeUrl(anchor.getAttribute('href'));
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
    });

    container.querySelectorAll('img').forEach(image => {
        image.src = sanitizeUrl(image.getAttribute('src'));
        image.loading = 'lazy';
    });

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

    let htmlContent = marked.parse(processedContent);
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

function setSectionContent(id, value, errorMessage = '') {
    const element = document.getElementById(id);
    if (!element) return;
    element.dataset.rawContent = value || '';
    if (errorMessage) {
        element.innerHTML = `<p class="empty-state">${escapeHtml(errorMessage)}</p>`;
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
    setExportEnabled(false);
}

function setRecommendEnabled(enabled) {
    const recommendBtn = document.getElementById('recommendBtn');
    if (recommendBtn) {
        recommendBtn.disabled = !enabled;
    }
}

function applyAnalysisSection(sectionName, sectionPayload, generateEvaluation, generateMermaid) {
    document.getElementById('result').classList.add('active');
    const section = sectionPayload || { status: 'empty', content: '', error: '', retryable: false };
    const errorMessage = section.status === 'failed' ? (section.error || `${sectionName} generation failed.`) : '';

    if (sectionName === 'evaluation') {
        document.getElementById('evaluationCard').style.display = generateEvaluation ? 'block' : 'none';
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
    resetPaperPanels();
    setExportEnabled(true);
    document.getElementById('askBtn').disabled = false;
}

function applyAnalysisResult(data, fileName, generateEvaluation, generateMermaid) {
    document.getElementById('result').classList.add('active');
    setFileInfo(fileName, data.char_count);

    const summarySection = getSectionPayload(data, 'summary');
    const quotesSection = getSectionPayload(data, 'quotes');
    const mindmapSection = getSectionPayload(data, 'mindmap');
    const evaluationSection = getSectionPayload(data, 'evaluation');
    const mermaidSection = getSectionPayload(data, 'mermaid');

    applyAnalysisSection('summary', summarySection, generateEvaluation, generateMermaid);
    applyAnalysisSection('quotes', quotesSection, generateEvaluation, generateMermaid);
    applyAnalysisSection('mindmap', mindmapSection, generateEvaluation, generateMermaid);
    applyAnalysisSection('evaluation', evaluationSection, generateEvaluation, generateMermaid);
    applyAnalysisSection('mermaid', mermaidSection, generateEvaluation, generateMermaid);

    finalizeAnalysisResultState(data, fileName);
    return Promise.resolve();
}

function buildPaperActionKey(item) {
    return String(item.paper_id || item.pdf_url || item.url || item.title || '').trim();
}

function renderPaperList(elementId, items, emptyText, meta) {
    const container = document.getElementById(elementId);
    if (!container) return;
    const normalizedItems = Array.isArray(items) ? items : [];
    const allowImport = elementId === 'paperSearchResults';
    if (!normalizedItems.length) {
        container.innerHTML = `<p class="empty-state">${escapeHtml(emptyText || 'No papers found.')}</p>`;
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
                ? `<button class="paper-link" type="button" onclick="downloadPaperByIndex('${escapeHtml(elementId)}', ${index})">Download</button>`
                : '';
            const addButton = allowImport
                ? `<button class="paper-link" type="button" onclick="addPaperToAnalysisByIndex(${index})" ${currentImportPaperKey === actionKey ? 'disabled' : ''}>${currentImportPaperKey === actionKey ? 'Adding...' : 'Add'}</button>`
                : '';
            const links = [
                item.url ? `<a class="paper-link" href="${sanitizeUrl(item.url)}" target="_blank" rel="noopener noreferrer">Open</a>` : '',
                item.pdf_url ? `<a class="paper-link" href="${sanitizeUrl(item.pdf_url)}" target="_blank" rel="noopener noreferrer">PDF</a>` : '',
                downloadButton,
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

function resetResultView() {
    ['summary', 'quotes', 'mindmap', 'evaluation'].forEach(id => setSectionContent(id, ''));
    document.getElementById('fileInfo').innerHTML = '';
    document.getElementById('result').classList.remove('active');
    document.getElementById('askBtn').disabled = true;
    document.getElementById('evaluationCard').style.display = '';
    clearChatHistory();
    resetMermaidCard();
    resetExportState();
    resetPaperPanels();
}

function updateFileMeta() {
    const file = document.getElementById('file').files[0];
    const fileMeta = document.getElementById('fileMeta');
    if (!fileMeta) return;
    if (!file) {
        fileMeta.textContent = 'No file selected. Recommended: clean PDF, TXT, DOCX, or PPTX for better structure extraction.';
        return;
    }

    const units = ['B', 'KB', 'MB', 'GB'];
    let size = file.size;
    let index = 0;
    while (size >= 1024 && index < units.length - 1) {
        size /= 1024;
        index += 1;
    }
    const formattedSize = `${size.toFixed(size >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
    fileMeta.textContent = `Selected: ${file.name} · ${formattedSize}`;
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
    if (contentType.includes('application/json')) {
        try {
            return await response.json();
        } catch (error) {
            console.warn('Response json parse failed:', error);
            return {};
        }
    }
    const text = await response.text();
    return { error: text };
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
    button.disabled = isLoading;
    button.innerText = isLoading ? loadingText : defaultText;
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

    const instance = await mermaidReady;
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
    const analyzeBtn = document.getElementById('analyzeBtn');
    const askBtn = document.getElementById('askBtn');

    if (!file) {
        showError('Please select a document first.');
        return;
    }

    if (analyzeController) {
        analyzeController.abort();
    }
    analyzeController = new AbortController();
    analyzeRequestId += 1;
    const requestId = analyzeRequestId;

    resetResultView();
    currentSessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    setButtonLoading(analyzeBtn, 'Analyzing...', 'Analyze Document', true);
    document.getElementById('loading').classList.add('active');
    hideError();
    askBtn.disabled = true;
    updateStatus('Analyzing document...', 'idle');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', currentSessionId);
    formData.append('generate_mermaid', String(generateMermaid));
    formData.append('generate_evaluation', String(generateEvaluation));
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
                updateStatus('Analysis started. Streaming sections...', 'idle');
            },
            section: payload => {
                if (requestId !== analyzeRequestId) return;
                const sectionName = payload.name;
                const section = payload.section || {};
                currentSections = { ...currentSections, [sectionName]: section };
                applyAnalysisSection(sectionName, section, generateEvaluation, generateMermaid);
                updateStatus(`Streaming ${sectionName}...`, 'idle');
            },
            done: async payload => {
                if (requestId !== analyzeRequestId) return;
                await applyAnalysisResult(payload, file.name, generateEvaluation, generateMermaid);
                updateStatus('Analysis ready for follow-up questions', 'success');
            },
            error: payload => {
                throw new Error(payload.error || 'Analysis failed.');
            }
        });
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        currentSessionId = '';
        resetResultView();
        showError(error.message || 'Analysis failed.');
        updateStatus('Analysis failed', 'error');
    } finally {
        if (requestId === analyzeRequestId) {
            document.getElementById('loading').classList.remove('active');
            setButtonLoading(analyzeBtn, 'Analyzing...', 'Analyze Document', false);
            analyzeController = null;
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

    if (!question) return;
    if (!currentSessionId) {
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
    appendChatText('User', question, 'chat-question');
    questionInput.value = '';
    setButtonLoading(askBtn, 'Processing...', 'Send', true);
    updateStatus('Answering based on current document...', 'idle');
    const streamingShell = appendStreamingAnswerShell();
    let accumulatedAnswer = '';

    try {
        const response = await fetch('/api/ask/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, session_id: currentSessionId, session_token: currentSessionToken, api_key: apiKey }),
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
                    timestamp: new Date().toISOString()
                });
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
            askController = null;
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
                generate_evaluation: generateEvaluation
            }),
            signal: importPaperController.signal
        });
        const data = await parseJsonSafely(response);
        if (!response.ok) {
            throw new Error(data.error || 'Paper import failed.');
        }

        currentPaperSearchMetaText = 'Imported selected paper into PaperWhisperer.';
        await applyAnalysisResult(data, data.source_filename || item.title || 'Imported paper', generateEvaluation, generateMermaid);
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
    if (!content) return;

    const rawText = (content.dataset && typeof content.dataset.rawContent === 'string')
        ? content.dataset.rawContent
        : '';
    const text = (rawText || content.innerText || content.textContent || '').trim();
    if (!text) return;

    const originalText = btnElement.innerText;
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
    } catch (error) {
        console.warn('Copy failed:', error);
        btnElement.innerText = 'Copy failed';
    }
    setTimeout(() => { btnElement.innerText = originalText; }, 1600);
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

function buildSessionMarkdown() {
    if (!currentAnalysisResult) return '';

    const reportTime = new Date();
    const svgSource = getCurrentSvgSource();
    const fileStem = sanitizeFileStem(currentSourceFileName || currentAnalysisResult.session_id || 'paperwhisperer_session');
    const svgFileName = `${fileStem}_visual_map.svg`;
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
        `| Backend archive | ${currentOutputFile || 'N/A'} |`,
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
            lines.push(
                `### Q${index + 1}`,
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
    if (!currentAnalysisResult) return;

    const fileStem = sanitizeFileStem(currentSourceFileName || currentAnalysisResult.session_id || 'paperwhisperer_session');
    const markdown = buildSessionMarkdown();
    if (!markdown) return;

    triggerTextDownload(`${fileStem}_session_report.md`, markdown, 'text/markdown;charset=utf-8');

    const svgSource = getCurrentSvgSource();
    if (svgSource) {
        triggerTextDownload(`${fileStem}_visual_map.svg`, svgSource, 'image/svg+xml;charset=utf-8');
    }
}

function downloadMermaidSVG() {
    const svgSource = getCurrentSvgSource();
    if (!svgSource) return;
    triggerTextDownload(`paper_map_${Date.now()}.svg`, svgSource, 'image/svg+xml;charset=utf-8');
}
