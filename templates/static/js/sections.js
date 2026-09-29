import { SECTION_EMPTY_TEXT } from './config.js';
import { state } from './state.js';
import { replaceWithFormattedContent } from './format.js';
import { setControlDisabled, setOptionalCardVisible } from './dom.js';
import { normalizeSmartActions, normalizeSmartTextItems, renderNextActions, renderSmartPrompts, resetSmartSuggestions, setAnswerMode } from './suggestions.js';
import { renderMermaidDiagram, resetMermaidCard, resetPanZoom, setMermaidCardVisible } from './mermaid.js';
import { renderExportPreview, resetExportState, setExportEnabled } from './export.js';
import { normalizeReadingQueue, renderReadingQueue, saveReadingQueue } from './queue.js';
import { resetPaperPanels, setRecommendEnabled } from './paperList.js';
import { setFileInfo } from './upload.js';
import { clearChatHistory } from './chat.js';

/**
 * Result workspace: per-section rendering, applying an analysis payload, and
 * the snapshot/restore pair that protects a previous run when a new analysis
 * fails.
 */

function createSectionStateCard(title, detail, tone = 'empty') {
    const card = document.createElement('div');
    card.className = `section-state section-state-${tone}`;

    const icon = document.createElement('span');
    icon.className = 'section-state-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = tone === 'error' ? '!' : 'i';

    const body = document.createElement('div');
    const titleElement = document.createElement('p');
    titleElement.className = 'section-state-title';
    titleElement.textContent = title;

    const detailElement = document.createElement('p');
    detailElement.className = 'section-state-detail';
    detailElement.textContent = detail;

    body.append(titleElement, detailElement);
    card.append(icon, body);
    return card;
}

function setSectionContent(id, value, errorMessage = '') {
    const element = document.getElementById(id);
    if (!element) return;
    element.dataset.rawContent = value || '';
    if (errorMessage) {
        element.replaceChildren(createSectionStateCard('Section needs review', errorMessage, 'error'));
        return;
    }
    if (!value) {
        element.replaceChildren(createSectionStateCard('Waiting for content', SECTION_EMPTY_TEXT[id] || 'No content available.'));
        return;
    }
    replaceWithFormattedContent(element, value, SECTION_EMPTY_TEXT[id]);
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

function applyAnalysisSection(sectionName, sectionPayload, generateEvaluation, generateMermaid, generateResearchBrief = true) {
    document.getElementById('result').classList.add('active');
    const section = sectionPayload || { status: 'empty', content: '', error: '', retryable: false };
    const errorMessage = section.status === 'failed' ? (section.error || `${sectionName} generation failed.`) : '';

    if (sectionName === 'evaluation') {
        setOptionalCardVisible('evaluationCard', generateEvaluation);
    }
    if (sectionName === 'research_brief') {
        setOptionalCardVisible('researchBriefCard', generateResearchBrief);
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
    state.currentAnalysisResult = data;
    state.currentSections = data.sections || {};
    state.currentSessionId = data.session_id || state.currentSessionId;
    state.currentSessionToken = data.session_token || state.currentSessionToken;
    state.currentSourceFileName = fileName;
    state.currentElapsedSeconds = data.elapsed_seconds ?? null;
    state.currentOutputFile = data.output_file || '';
    state.currentChatTurns = [];
    state.currentSuggestedQuestions = normalizeSmartTextItems(data.suggested_questions || []);
    state.currentNextActions = normalizeSmartActions(data.next_actions || []);
    renderSmartPrompts(state.currentSuggestedQuestions);
    renderNextActions(state.currentNextActions);
    renderExportPreview();
    resetPaperPanels();
    saveReadingQueue();
    setExportEnabled(true);
    setControlDisabled(document.getElementById('askBtn'), false);
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

function captureWorkspaceState() {
    return {
        currentSessionId: state.currentSessionId,
        currentSessionToken: state.currentSessionToken,
        currentSections: { ...state.currentSections },
        currentMermaidSource: state.currentMermaidSource,
        currentAnalysisResult: state.currentAnalysisResult,
        currentSourceFileName: state.currentSourceFileName,
        currentElapsedSeconds: state.currentElapsedSeconds,
        currentOutputFile: state.currentOutputFile,
        currentChatTurns: [...state.currentChatTurns],
        currentPaperSearchResults: [...state.currentPaperSearchResults],
        currentPaperRecommendations: [...state.currentPaperRecommendations],
        currentSuggestedQuestions: [...state.currentSuggestedQuestions],
        currentNextActions: state.currentNextActions.map(action => ({ ...action })),
        currentReadingQueue: state.currentReadingQueue.map(item => ({ ...item, authors: [...(item.authors || [])] })),
        currentAnswerMode: state.currentAnswerMode,
        currentPaperSearchMetaText: state.currentPaperSearchMetaText,
        currentPaperRecommendationMetaText: state.currentPaperRecommendationMetaText,
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
            evaluationHidden: document.getElementById('evaluationCard').classList.contains('is-hidden'),
            researchBriefHidden: document.getElementById('researchBriefCard').classList.contains('is-hidden'),
            mermaidHidden: document.getElementById('mermaidCard').classList.contains('is-hidden'),
            askDisabled: document.getElementById('askBtn').disabled,
            exportDisabled: document.getElementById('exportBtn').disabled,
            recommendDisabled: document.getElementById('recommendBtn').disabled
        }
    };
}

function restoreWorkspaceState(snapshot) {
    if (!snapshot) return;
    state.currentSessionId = snapshot.currentSessionId;
    state.currentSessionToken = snapshot.currentSessionToken;
    state.currentSections = { ...snapshot.currentSections };
    state.currentMermaidSource = snapshot.currentMermaidSource;
    state.currentAnalysisResult = snapshot.currentAnalysisResult;
    state.currentSourceFileName = snapshot.currentSourceFileName;
    state.currentElapsedSeconds = snapshot.currentElapsedSeconds;
    state.currentOutputFile = snapshot.currentOutputFile;
    state.currentChatTurns = [...snapshot.currentChatTurns];
    state.currentPaperSearchResults = [...snapshot.currentPaperSearchResults];
    state.currentPaperRecommendations = [...snapshot.currentPaperRecommendations];
    state.currentSuggestedQuestions = normalizeSmartTextItems(snapshot.currentSuggestedQuestions || []);
    state.currentNextActions = normalizeSmartActions(snapshot.currentNextActions || []);
    state.currentReadingQueue = normalizeReadingQueue(snapshot.currentReadingQueue || []);
    setAnswerMode(snapshot.currentAnswerMode || 'evidence');
    state.currentPaperSearchMetaText = snapshot.currentPaperSearchMetaText;
    state.currentPaperRecommendationMetaText = snapshot.currentPaperRecommendationMetaText;

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
    document.getElementById('paperSearchMeta').textContent = state.currentPaperSearchMetaText;
    document.getElementById('paperRecommendationMeta').textContent = state.currentPaperRecommendationMetaText;
    renderSmartPrompts(state.currentSuggestedQuestions);
    renderNextActions(state.currentNextActions);
    renderReadingQueue();
    document.getElementById('mermaidChart').innerHTML = snapshot.html.mermaidChart;

    document.getElementById('result').classList.toggle('active', snapshot.ui.resultActive);
    setOptionalCardVisible('evaluationCard', !snapshot.ui.evaluationHidden);
    setOptionalCardVisible('researchBriefCard', !snapshot.ui.researchBriefHidden);
    setMermaidCardVisible(!snapshot.ui.mermaidHidden);
    setControlDisabled(document.getElementById('askBtn'), snapshot.ui.askDisabled);
    setControlDisabled(document.getElementById('exportBtn'), snapshot.ui.exportDisabled);
    setControlDisabled(document.getElementById('recommendBtn'), snapshot.ui.recommendDisabled);
    resetPanZoom();
    if (state.currentMermaidSource) {
        requestAnimationFrame(() => renderMermaidDiagram(state.currentMermaidSource));
    }
}

function resetResultView() {
    ['summary', 'quotes', 'mindmap', 'evaluation', 'research_brief'].forEach(id => setSectionContent(id, ''));
    document.getElementById('fileInfo').replaceChildren();
    document.getElementById('result').classList.remove('active');
    setControlDisabled(document.getElementById('askBtn'), true);
    setOptionalCardVisible('evaluationCard', true);
    setOptionalCardVisible('researchBriefCard', true);
    clearChatHistory();
    resetMermaidCard();
    resetExportState();
    resetPaperPanels();
}


export { applyAnalysisResult };
export { applyAnalysisSection };
export { captureWorkspaceState };
export { getSectionPayload };
export { resetResultView };
export { restoreWorkspaceState };
