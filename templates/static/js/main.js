/**
 * Entry point. Wires the DOM to the feature modules and performs the initial
 * render. Everything here is event binding — behaviour lives in the modules
 * this file imports.
 */
import { state } from './state.js';
import { bindClick, copyText, toggleAIList, updateStatus } from './dom.js';
import { initializeTheme, toggleTheme } from './theme.js';
import { handleFileSelection, initializeUploadDropZone, updateFileMeta } from './upload.js';
import { resetResultView } from './sections.js';
import { clearReadingQueue, handleReadingQueueClick, renderReadingQueue } from './queue.js';
import { handleAnswerModeKeydown, setAnswerMode, useStarterQuestion } from './suggestions.js';
import { handleWorkspaceShortcut, initializeBackToTop, scrollToTop } from './a11y.js';
import { downloadMermaidSVG, exportSessionReport } from './export.js';
import { zoomIn, zoomOut, zoomReset } from './mermaid.js';
import {
    cancelPaperSearchRequest,
    cancelRecommendRequest,
    handlePaperResultClick,
    handlePaperSearchKeyPress,
    recommendPapers,
    searchPapers,
    useExampleQuery
} from './papers.js';
import { askQuestion, cancelAskRequest, handleKeyPress } from './chat.js';
import { analyze, cancelAnalyzeRequest } from './analyze.js';

function bindDelegatedClick(elementId, handler) {
    document.getElementById(elementId)?.addEventListener('click', handler);
}

function bindDataAttribute(attribute, handlerFactory) {
    document.querySelectorAll(`[${attribute}]`).forEach(button => {
        button.addEventListener('click', handlerFactory(button, button.dataset));
    });
}

function initialize() {
    initializeTheme();
    updateFileMeta();
    resetResultView();
    renderReadingQueue();
    setAnswerMode(state.currentAnswerMode);
    updateStatus('Waiting for document', 'idle');

    const buttons = [
        ['themeBtn', toggleTheme],
        ['analyzeBtn', analyze],
        ['paperSearchBtn', searchPapers],
        ['clearReadingQueueBtn', clearReadingQueue],
        ['exportBtn', exportSessionReport],
        ['recommendBtn', recommendPapers],
        ['askBtn', askQuestion],
        ['aiToggleBtn', toggleAIList],
        ['zoomInBtn', zoomIn],
        ['zoomOutBtn', zoomOut],
        ['zoomResetBtn', zoomReset],
        ['downloadMermaidBtn', downloadMermaidSVG],
        ['backToTopBtn', scrollToTop],
        ['cancelAnalyzeBtn', cancelAnalyzeRequest],
        ['cancelPaperSearchBtn', cancelPaperSearchRequest],
        ['cancelRecommendBtn', cancelRecommendRequest],
        ['cancelAskBtn', cancelAskRequest]
    ];
    buttons.forEach(([id, handler]) => bindClick(id, handler));

    initializeBackToTop();
    initializeUploadDropZone();

    document.getElementById('file')?.addEventListener('change', handleFileSelection);
    document.getElementById('paperSearchInput')?.addEventListener('keydown', handlePaperSearchKeyPress);
    document.getElementById('questionInput')?.addEventListener('keydown', handleKeyPress);
    bindDelegatedClick('paperSearchResults', handlePaperResultClick);
    bindDelegatedClick('paperRecommendations', handlePaperResultClick);
    bindDelegatedClick('readingQueue', handleReadingQueueClick);
    document.addEventListener('keydown', handleWorkspaceShortcut);

    document.querySelectorAll('.mode-chip').forEach(button => {
        button.addEventListener('click', () => setAnswerMode(button.dataset.mode));
        button.addEventListener('keydown', handleAnswerModeKeydown);
    });
    bindDataAttribute('data-example-query', (button, data) => () => useExampleQuery(data.exampleQuery));
    bindDataAttribute('data-starter-question', (button, data) => () => useStarterQuestion(data.starterQuestion));
    bindDataAttribute('data-copy-target', (button, data) => () => copyText(data.copyTarget, button));
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
} else {
    initialize();
}
