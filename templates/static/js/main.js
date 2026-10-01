/**
 * Entry point. Wires the DOM to the feature modules and performs the initial
 * render. Everything here is event binding — behaviour lives in the modules
 * this file imports.
 */
import { state } from './state.js';
import { bindClick, copyText, toggleAIList, updateStatus } from './dom.js';
import { setMaxUploadBytes } from './config.js';
import { apiRequest } from './api.js';
import { initializeI18n, t } from './i18n.js';
import { initializeRuntime } from './runtime.js';
import { initializeTheme, toggleTheme } from './theme.js';
import { handleFileSelection, initializeUploadDropZone, updateFileMeta } from './upload.js';
import { hideRunIndicator, resetResultView } from './sections.js';
import { clearReadingQueue, handleReadingQueueClick, renderReadingQueue } from './queue.js';
import { handleAnswerModeKeydown, setAnswerMode, useStarterQuestion } from './suggestions.js';
import { handleWorkspaceShortcut, initializeBackToTop, scrollToTop } from './a11y.js';
import { downloadMermaidSVG, exportSessionReport, initializeExportFormats } from './export.js';
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
import { askQuestion, cancelAskRequest, handleKeyPress, initializeChatDelegates } from './chat.js';
import { analyze, cancelAnalyzeRequest } from './analyze.js';
import { initializeEvidencePanel } from './evidence.js';
import { initializeParseStatus } from './parseStatus.js';

function bindDelegatedClick(elementId, handler) {
    document.getElementById(elementId)?.addEventListener('click', handler);
}

/**
 * Adopt the server's upload ceiling so the pre-flight check quotes the number
 * the server will actually enforce.
 *
 * Deliberately not awaited: the file picker is usable before this resolves, and
 * a client that has to finish a health call before accepting a file is worse
 * than one that briefly quotes a fallback. setMaxUploadBytes rejects a bad
 * value, so a health endpoint that is down or malformed leaves the fallback in
 * place rather than switching the check off.
 */
async function syncUploadLimitFromServer() {
    try {
        const payload = await apiRequest('/api/health', {
            method: 'GET',
            cache: 'no-store',
            health: true,
            operation: 'health'
        });
        setMaxUploadBytes(payload?.max_upload_bytes);
    } catch {
        // Offline, aborted, non-JSON, or a body over the size cap. Every one of
        // those is a reason the fallback constant stands, and none of them is a
        // reason to interrupt the reader: this is a background probe for a
        // number, not the operation they asked for. `setMaxUploadBytes` also
        // rejects a bad value, so a malformed one cannot switch the check off.
    }
}

function bindDataAttribute(attribute, handlerFactory) {
    document.querySelectorAll(`[${attribute}]`).forEach(button => {
        button.addEventListener('click', handlerFactory(button, button.dataset));
    });
}

function initialize() {
    // First, before the theme, the status chip or any feature module: the
    // starting language. `initializeI18n` resolves the stored choice, then the
    // browser's, then the default, sets `<html lang>`, and translates the
    // static markup. Running it here means the first paint is already in the
    // reader's language instead of showing the default and swapping, and every
    // module that renders after this point reads `t()` in the right locale.
    initializeI18n();
    // Before anything else: the failure surface, the connectivity chip and the
    // offline queue. A request that fails during the rest of this function has
    // to have somewhere to be reported, and a request that fails later has to
    // have a listener already bound to the `online` event.
    initializeRuntime();
    initializeTheme();
    updateFileMeta();
    resetResultView();
    renderReadingQueue();
    setAnswerMode(state.currentAnswerMode);
    updateStatus(t('status.waiting'), 'idle');
    hideRunIndicator();

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
    initializeEvidencePanel();
    initializeParseStatus();
    // Before any turn can render one. The transcript's controls are delegated
    // rather than bound per answer, so a transcript restored from `innerHTML`
    // after a failed re-analysis is still functional — a listener bound per node
    // would be silently stripped by that restore.
    initializeChatDelegates();
    initializeExportFormats();
    syncUploadLimitFromServer();

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
