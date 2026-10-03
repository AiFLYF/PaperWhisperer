import { SECTION_EMPTY_TEXT } from './config.js';
import { state } from './state.js';
import { t } from './i18n.js';
import { replaceWithFormattedContent } from './format.js';
import { setControlDisabled, setOptionalCardVisible } from './dom.js';
import { normalizeSmartActions, normalizeSmartTextItems, renderNextActions, renderSmartPrompts, resetSmartSuggestions, setAnswerMode } from './suggestions.js';
import { renderMermaidDiagram, resetMermaidCard, resetPanZoom, setMermaidCardVisible } from './mermaid.js';
import { renderExportPreview, resetExportState, setExportEnabled } from './export.js';
import { normalizeReadingQueue, renderReadingQueue, saveReadingQueue } from './queue.js';
import { resetPaperPanels, setRecommendEnabled } from './paperList.js';
import { setFileInfo } from './upload.js';
import { clearChatHistory } from './chat.js';
import { closeEvidence } from './evidence.js';
import { renderParseStatus, resetParseStatus } from './parseStatus.js';
import { renderRecommendationSignals } from './recommendationSignals.js';

/**
 * Result workspace: per-section rendering, applying an analysis payload, and
 * the snapshot/restore pair that protects a previous run when a new analysis
 * fails.
 */

/**
 * Show or hide the run indicator, using the design system's own `is-hidden`.
 *
 * The block carries a spinner, a "preparing analysis" line and the four-stage
 * track, and `states.css` only ever styles it in the `.active` state — there is
 * no base rule to fall back on. So between page load and the reader's first
 * analysis, the interface showed a spinning ring, "正在准备文档分析…" and an
 * active 上传 stage: a run indicator describing a run nobody has asked for. That
 * is the same defect as a spinner that outlives its work, arriving at the other
 * end of the run, and it is worse here because every reader passes through it.
 *
 * `is-hidden` rather than a new class because it is how this stylesheet already
 * takes a block out of flow, and being `!important` it wins against
 * `.loading.active` — which is what makes starting a run a single toggle rather
 * than a pair of class writes that have to be kept in step at four call sites.
 *
 * `resetResultView` is *not* used for this: it tears the workspace down, and
 * hiding the indicator is the only thing wanted here.
 */
function hideRunIndicator() {
    document.getElementById('loading')?.classList.add('is-hidden');
}

function showRunIndicator() {
    document.getElementById('loading')?.classList.remove('is-hidden');
}

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
    // Through the catalogue rather than `SECTION_EMPTY_TEXT`, which is the
    // *English* locale's values. Reading the config table directly wrote English
    // into a Chinese-default interface, and did so on every page load:
    // `resetResultView()` calls this for all five sections during
    // `initialize()`. `section.empty.<id>` is the same sentence in the reader's
    // language.
    const emptyText = t(`section.empty.${id}`) || SECTION_EMPTY_TEXT[id];
    if (!value) {
        element.replaceChildren(createSectionStateCard('Waiting for content', emptyText));
        return;
    }
    replaceWithFormattedContent(element, value, emptyText);
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

    // The document id addresses the Evidence API, and it is not the session id:
    // the two are minted independently and conflating them would produce an
    // evidence URL that 404s on a 404 the reader cannot tell from a typo.
    // Absent a document id, the panel says so rather than guessing — see the
    // disabled submit button in `updateReparseWarnings`.
    state.currentDocumentId = String(data.document_id || data.parse?.document_id || '');

    // T3.4: which parser read this file, and did it leave the machine. Rendered
    // from the result payload, which carries it as `parse`.
    state.currentParseProvenance = data.parse || null;
    // The registration verdict rides beside the provenance rather than inside
    // it: it is not a property of the parse (the parse succeeded either way),
    // it is the answer to whether the citations on this run will resolve. Both are
    // rendered together because they are one card.
    state.currentDocumentRegistration = data.document_registration || null;
    renderParseStatus(state.currentParseProvenance, state.currentDocumentRegistration);

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

/**
 * Whether an analysis run finished whole, and if not, which parts did not.
 *
 * `analysis_status` is computed by the server from which sections actually
 * succeeded. The workspace already renders a "Section needs review" card for
 * each failed section, so the detail is not lost — the headline was. The status
 * chip used to say "Analysis ready for follow-up questions" in a success tone
 * unconditionally, on runs where a third of the analysis had failed, and a
 * degraded state wearing the success colour is worse than no signal at all
 * because it removes the reader's reason to look at the sections.
 *
 * Only a run that completed every planned section reports success. An absent
 * `analysis_status` is not treated as success either: the server sends it on
 * every result, so its absence means the contract moved, and the recoverable
 * direction is to under-claim rather than to congratulate the reader on an
 * analysis whose completeness this client cannot see.
 *
 * `failed_sections` carries the server's own labels ("概览", "引用片段"), so the
 * named parts are the ones the reader can actually see on screen.
 */
function describeAnalysisOutcome(payload) {
    const status = payload?.analysis_status || {};
    const failed = Array.isArray(status.failed_sections) ? status.failed_sections.filter(Boolean) : [];
    const quality = String(status.quality || '');

    if (failed.length) {
        return {
            text: `Analysis finished, but ${failed.length} part${failed.length === 1 ? '' : 's'} could not be generated: ${failed.join(', ')}.`,
            tone: 'error'
        };
    }
    if (quality === 'partial') {
        return {
            text: 'Analysis finished with some parts incomplete. Check each section before relying on it.',
            tone: 'error'
        };
    }
    if (!quality) {
        return {
            text: 'Analysis finished. This build was not told which parts completed, so check each section before relying on it.',
            tone: 'idle'
        };
    }
    return { text: 'Analysis ready for follow-up questions', tone: 'success' };
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
        currentRecommendationSignals: state.currentRecommendationSignals.map(entry => ({ ...entry })),
        currentRecommendationDuplicates: state.currentRecommendationDuplicates.map(entry => ({ ...entry })),
        currentSeedPaper: state.currentSeedPaper ? { ...state.currentSeedPaper } : null,
        currentDocumentId: state.currentDocumentId,
        currentParseProvenance: state.currentParseProvenance,
        currentDocumentRegistration: state.currentDocumentRegistration,
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
            mermaidChart: document.getElementById('mermaidChart').innerHTML,
            parseStatus: document.getElementById('parseStatus').innerHTML
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
    // Re-rendered rather than restored as raw HTML, because the signal status is
    // derived from the stored per-signal values. Restoring markup would let a
    // snapshot and a re-render disagree — the rows would come back from the
    // snapshot while the status came back from a different source, which is the
    // one thing the four-state rendering exists to prevent.
    state.currentRecommendationSignals = Array.isArray(snapshot.currentRecommendationSignals)
        ? snapshot.currentRecommendationSignals.map(entry => ({ ...entry }))
        : [];
    state.currentRecommendationDuplicates = Array.isArray(snapshot.currentRecommendationDuplicates)
        ? snapshot.currentRecommendationDuplicates.map(entry => ({ ...entry }))
        : [];
    state.currentSeedPaper = snapshot.currentSeedPaper ? { ...snapshot.currentSeedPaper } : null;
    renderRecommendationSignals(state.currentRecommendationSignals, state.currentRecommendationDuplicates);
    state.currentDocumentId = snapshot.currentDocumentId || '';
    state.currentParseProvenance = snapshot.currentParseProvenance || null;
    // Restored next to the provenance and *not* re-read out of the markup the
    // snapshot saved: the copy is resolved from the catalogue at render time, so
    // restoring rendered HTML would bring back the sentence in the language that
    // was active when the snapshot was taken — and the server warning string,
    // untranslated, would come back with it.
    state.currentDocumentRegistration = snapshot.currentDocumentRegistration || null;

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
    document.getElementById('parseStatus').innerHTML = snapshot.html.parseStatus || '';
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
    // The evidence panel is an overlay over a workspace that is being torn
    // down; leaving it open would leave it pointing at evidence from a
    // document that is no longer on screen.
    closeEvidence();
    resetParseStatus();
    state.currentDocumentId = '';
    state.currentParseProvenance = null;
    state.currentDocumentRegistration = null;
    // The seed is the bibliographic record of the paper that was *imported*.
    // A document uploaded from disk has no identifier set, and a new analysis
    // replaces the workspace, so carrying the previous import DOI forward
    // would seed the next recommendation run from a different paper than the
    // one on screen.
    state.currentSeedPaper = null;
}


export { applyAnalysisResult };
export { applyAnalysisSection };
export { captureWorkspaceState };
export { describeAnalysisOutcome };
export { getSectionPayload };
export { hideRunIndicator };
export { resetResultView };
export { restoreWorkspaceState };
export { showRunIndicator };
