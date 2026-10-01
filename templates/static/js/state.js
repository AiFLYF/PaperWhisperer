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
    // The question of the turn currently in flight, cleared when the turn
    // settles. A cancel removes the half-streamed answer, so without this the
    // reader is left with their question in the transcript and nothing under
    // it, and with an empty composer — the most expensive field in this
    // interface to retype, gone because of a button they pressed deliberately.
    pendingQuestion: '',

    // The id of the "in progress" notice raised by each long-running operation,
    // keyed by the operation name.
    //
    // Needed because the two ways an operation ends do not reach the same code.
    // A *success* retires its pending notice by operation, inside `endFlight`.
    // A *cancellation* does not: the reader pressed Cancel, which bumps the
    // request token so the request's own `finally` skips its settled branch —
    // and a cancel is not a success, so nothing retires the notice. The result
    // was a status line reading "回答生成中" above a transcript the reader had
    // just emptied, with nothing to dismiss it. The cancel handler reads the id
    // from here to close that.
    pendingNoticeIds: { analyze: '', ask: '' },

    // The server's job id for the analysis currently running, taken from the
    // `start` event of `/api/analyze/stream`.
    //
    // This is what makes a dropped connection resumable rather than a restart.
    // The route submits the parse and the model calls to
    // `paperwhisperer.infra.jobs` and then *watches* the row, so the work is
    // durable and still running after the browser gives up — a reader who
    // presses "重试" on a dropped stream pays for the same document twice.
    // `/v1/jobs/{id}/stream` reports how far the surviving job got, and takes
    // `Last-Event-ID`, so it can be re-entered where the connection left off.
    //
    // Cleared when the run settles, and on cancel, so a later run cannot be
    // told about this one.
    currentAnalysisJobId: '',

    // --- research trail ---------------------------------------------------
    currentPaperSearchResults: [],
    currentPaperRecommendations: [],
    currentPaperSearchMetaText: 'Search across Semantic Scholar and arXiv with a single query.',
    currentPaperRecommendationMetaText: 'Analyze a paper first, then generate follow-up reading suggestions from the current session.',
    currentReadingQueue: [],
    currentSuggestedQuestions: [],
    currentNextActions: [],

    // --- multi-signal recommendations (T3.2) -----------------------------
    // The per-signal four-state status and the probable-duplicate pairs, kept
    // apart from `currentPaperRecommendations` because they describe the *run*
    // rather than any one row. A rate-limited provider is a fact about this
    // request; it must survive a re-render and a workspace restore, or the
    // honest "we could not check this" quietly becomes an empty list.
    currentRecommendationSignals: [],
    currentRecommendationDuplicates: [],
    // The bibliographic record for the paper recommendations are seeded from,
    // captured when a paper is imported for analysis. A document uploaded from
    // disk has no DOI/arXiv/S2/OpenAlex id, only a title, so the citation-graph
    // signals legitimately report themselves unavailable for it — this field
    // is what lets a paper that *does* have identifiers get the full answer.
    currentSeedPaper: null,

    // --- transient UI bookkeeping ----------------------------------------
    pendingAnalysisSnapshot: null,
    backToTopFramePending: false,
    buttonFeedbackTimers: new WeakMap(),

    // --- parse provenance (T3.4) ----------------------------------------
    // Kept whole rather than flattened into booleans: `used_cloud`,
    // `cloud_attempted` and `touched_network` are three different facts, and
    // collapsing them to one `cloud: true` is how a deployment ends up billing
    // for a result it threw away without saying so.
    currentParseProvenance: null,
    currentDocumentId: '',
    reparseController: null,

    // --- evidence panel (T1.9) -------------------------------------------
    // `returnFocusTo` is what makes closing the panel non-destructive: the
    // panel is an overlay, so the chip the user pressed is still in the
    // document, and focus goes back to it rather than to the top of the page.
    currentEvidence: null,
    evidencePanelOpen: false,
    evidenceReturnFocusTo: null,
    // The window scroll offset captured when the panel opened. Restored on
    // close if anything moved it, so reading a citation cannot cost the user
    // their place in the conversation.
    evidenceScrollAnchor: 0
};
