/**
 * Mutable application state.
 *
 * Every module shares this single store instead of relying on globals that
 * are declared in one file and read from another. Keeping it in one place
 * makes the data flow obvious and keeps the ES module graph acyclic.
 *
 * Fields are intentionally named after the historical globals so the intent
 * of each value stays searchable.
 */
export const state = {
    // --- session identity -------------------------------------------------
    currentSessionId: '',
    currentSessionToken: '',

    // --- analysis output --------------------------------------------------
    currentSections: {},
    currentAnalysisResult: null,
    currentSourceFileName: '',
    currentElapsedSeconds: null,
    currentOutputFile: '',
    currentMermaidSource: '',

    // --- renderers / third-party handles ----------------------------------
    panZoomInstance: null,
    mermaidInstance: null,
    mermaidReadyPromise: null,

    // --- in-flight requests ----------------------------------------------
    analyzeController: null,
    askController: null,
    paperSearchController: null,
    recommendController: null,
    importPaperController: null,

    // --- monotonic request tokens used to drop stale responses -----------
    analyzeRequestId: 0,
    askRequestId: 0,
    paperSearchRequestId: 0,
    recommendRequestId: 0,
    currentImportPaperKey: '',

    // --- conversation -----------------------------------------------------
    currentChatTurns: [],
    currentAnswerMode: 'evidence',
    activeStreamingAnswerShell: null,

    // --- research trail ---------------------------------------------------
    currentPaperSearchResults: [],
    currentPaperRecommendations: [],
    currentPaperSearchMetaText: 'Search across Semantic Scholar and arXiv with a single query.',
    currentPaperRecommendationMetaText: 'Analyze a paper first, then generate follow-up reading suggestions from the current session.',
    currentReadingQueue: [],
    currentSuggestedQuestions: [],
    currentNextActions: [],

    // --- transient UI bookkeeping ----------------------------------------
    pendingAnalysisSnapshot: null,
    backToTopFramePending: false,
    buttonFeedbackTimers: new WeakMap()
};
