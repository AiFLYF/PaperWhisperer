import { state } from './state.js';
import { apiRequest, attachToJobStream, buildApiError, readSseStream } from './api.js';
import { beginFlight, capturePreservedFields, dismissNotice, endFlight, notifyFailure, notifyPending, restorePreservedFields } from './runtime.js';
import { beginProgressClock, endProgressClock, focusAnalysisWorkspace, hideError, setButtonLoading, setCancelVisible, setControlDisabled, showError, updateAnalysisProgress, updateStatus } from './dom.js';
import { getUploadValidationError, setFileInfo, updateFileMeta } from './upload.js';
import { applyAnalysisResult, applyAnalysisSection, captureWorkspaceState, describeAnalysisOutcome, hideRunIndicator, resetResultView, restoreWorkspaceState, showRunIndicator } from './sections.js';

/**
 * Report a run whose stream ended without ever settling.
 *
 * Its own function rather than an inline block, for one reason that is also the
 * reason it is worth extracting: `analyze` already has a cancellation guard
 * below this point, and a cancellation must never be reported as a failure or
 * offered a retry that re-runs work the reader deliberately stopped. Keeping
 * this report out of the function body means the guard in the catch remains the
 * first thing that function does with a cancellation.
 *
 * An abort can also surface as a clean end-of-stream rather than a thrown
 * AbortError, so the aborted signal is checked directly rather than relying on
 * the transport to have thrown.
 *
 * Carries `client_stream_interrupted` rather than a bare message, so the
 * runtime classifies it as retryable and offers a retry. Without a code it
 * arrives as `client_unknown_error`, which is terminal: the reader is told an
 * unclassified failure happened and is given no way to act on it, when the one
 * thing to do is press the button that starts the analysis again.
 *
 * **Asynchronous because of the re-attach.** A dropped connection is not a
 * failed analysis: the route submitted the work to a durable job row before it
 * started streaming, so the parse and the model calls are very likely still
 * running. Declaring the run over from here would report a completed analysis as
 * a failure and offer "重试", which is a second parse and a second bill for the
 * same document. So before concluding anything this asks the server what
 * actually happened to the job, and only reports a failure when the job agrees
 * that it failed — or when the job cannot be reached at all, which is the one
 * case where not knowing is all there is.
 */
async function reportInterruptedAnalysis(previousWorkspace, seenSections, plannedSections) {
    if (state.analyzeController?.signal.aborted) {
        return false;
    }

    const jobId = state.currentAnalysisJobId;
    const jobState = await reattachAnalysisJob(jobId, {
        signal: state.analyzeController?.signal
    });
    if (jobState) {
        const recovered = describeJobOutcome(jobState.state, jobId);
        if (recovered.outcome !== 'unknown') {
            restoreAfterRecoveredJob(previousWorkspace, recovered, jobId);
            return true;
        }
    }

    const partial = seenSections.size;
    const error = buildApiError(
        {
            code: 'client_stream_interrupted',
            error: `The analysis stream ended before it completed. ${partial
                ? `${partial} of ${plannedSections} sections had arrived; they are not a usable analysis.`
                : 'No section was received.'}${jobId ? ` The job ${jobId} could not be reached from here either.` : ''}`
        },
        'Analysis stream interrupted.'
    );
    if (previousWorkspace) {
        restoreWorkspaceState(previousWorkspace);
        showError(`${error.message}\n\nYour previous successful workspace has been restored.`);
        focusAnalysisWorkspace({ scroll: false });
    } else {
        state.currentSessionId = '';
        resetResultView();
        showError(error.message);
    }
    state.pendingAnalysisSnapshot = null;
    notifyFailure(error, { operation: ANALYZE_OPERATION, onRetry: () => analyze() });
    updateStatus('Analysis interrupted before it completed', 'error');
    return true;
}

/**
 * Put the page back after a re-attach found out what the job actually did.
 *
 * The previous workspace comes back on the same terms as any other failed run,
 * because from this page's point of view nothing was delivered — which is the
 * point. A job that completed did not put an analysis on this screen, and
 * rendering an empty workspace as though the run had produced nothing would
 * claim something about the reader's paper that nothing supports.
 *
 * The retry is attached only when the job did *not* succeed. That is the whole
 * reason the job was asked: a successful job means the document has already
 * been parsed and analysed once, and a "重试" here is a second full run of it.
 */
function restoreAfterRecoveredJob(previousWorkspace, recovered, jobId) {
    if (previousWorkspace) {
        restoreWorkspaceState(previousWorkspace);
        focusAnalysisWorkspace({ scroll: false });
    } else {
        state.currentSessionId = '';
        resetResultView();
    }
    state.pendingAnalysisSnapshot = null;
    const tone = recovered.outcome === 'completed' ? 'idle' : 'error';
    showError(`${recovered.text}${previousWorkspace
        ? '\n\nYour previous successful workspace has been restored.'
        : ''}`);
    updateStatus(recovered.text, tone);
    if (recovered.retryable) {
        notifyFailure(
            buildApiError({ code: 'client_stream_interrupted', error: recovered.text }, 'Analysis interrupted.'),
            { operation: ANALYZE_OPERATION, onRetry: () => analyze() }
        );
    }
    return jobId;
}

/**
 * Which of the four visible stages a job state belongs to.
 *
 * The server's state machine is finer than the interface's: `queued`,
 * `parsing`, `quality_check`, `fallback`, `normalizing` and `indexing` are all
 * still "getting the document into the product", and only `analyzing` is the
 * model work the `分析` stage names. Mapping rather than inventing a fifth
 * stage means the track cannot show a stage the server does not have, and the
 * server's own `message` — which is where it says things like "re-ran OCR
 * because the scan was unreadable" — is what the reader actually sees.
 */
function jobStageStep(jobState) {
    const value = String(jobState || '');
    if (['completed', 'partial', 'failed', 'cancelled'].includes(value)) return 'ready';
    if (value === 'analyzing') return 'analyze';
    return 'upload';
}

/**
 * What a re-attached job's final state means for this page.
 *
 * Pure, and pure for the reason that matters: the decision of whether to offer
 * a retry is the difference between "start the analysis again" and "pay for
 * this document twice", and it must not be re-derived differently by the
 * interrupted-report path and the timeout path.
 *
 * `retryable` is false for a job that *succeeded*. The work is done; the
 * connection ate the payload, not the parse. Offering "重试" there invites a
 * second full analysis of a document that has already been analysed, which is
 * the double charge the re-attach exists to prevent — and a reader who is told
 * only "it failed" would reasonably press it.
 *
 * Pure for the second reason too: the result payload is parked process-locally
 * and taken once (`routes_jobs.take_job_result`), so a job that finished while
 * this client was disconnected cannot in general be handed back to it. Saying
 * the analysis completed *and* that the workspace could not be rebuilt is both
 * true and the only honest pair of statements available.
 */
function describeJobOutcome(jobState, jobId) {
    const state_ = String(jobState || '');
    if (state_ === 'completed') {
        return {
            outcome: 'completed',
            retryable: false,
            text: `分析已在服务端完成（任务 ${jobId || '未知'}），但结果没有送达这个页面。服务端只保留任务的进度，不保留分析正文，所以工作台无法重建。`
        };
    }
    if (state_ === 'partial') {
        // Kept apart from `completed` on purpose. The job finished, but it
        // finished with visible warnings — sections that did not generate — and
        // telling the reader it "completed" would report a partial analysis as
        // a whole one, which is the same over-claim the status chip already had
        // a fix for on the synchronous path.
        return {
            outcome: 'partial',
            retryable: false,
            text: `分析任务 ${jobId || '未知'} 在服务端结束，但只有部分章节生成成功。结果没有送达这个页面，工作台无法重建。`
        };
    }
    if (state_ === 'failed' || state_ === 'cancelled') {
        return {
            outcome: 'failed',
            retryable: false,
            text: `分析任务 ${jobId || '未知'} 在服务端结束为 ${state_}，不是这个页面断开了连接。`
        };
    }
    return {
        outcome: 'unknown',
        retryable: true,
        text: ''
    };
}

/**
 * Follow the surviving job after this connection died.
 *
 * The whole point of the id on the `start` event: the parse and the model calls
 * are on a worker thread against a durable row, so a browser that gives up has
 * not thrown the work away. `/v1/jobs/{id}/stream` carries the job's real
 * states and reads `Last-Event-ID`, so this re-enters where the connection left
 * off rather than replaying the run from the start.
 *
 * Returns the last state seen, or `null` when the feed itself could not be
 * resumed. `null` is deliberately not an outcome: it means "could not find
 * out", and the caller has to keep saying so rather than guessing either way.
 */
async function reattachAnalysisJob(jobId, options = {}) {
    if (!jobId) return null;
    let latest = null;
    const record = (payload) => {
        if (!payload || typeof payload !== 'object') return;
        latest = payload;
        const message = String(payload.message || '').trim();
        updateAnalysisProgress(jobStageStep(payload.state), message);
        if (message) {
            updateStatus(message, 'idle');
        }
    };
    try {
        await attachToJobStream(jobId, {
            snapshot: record,
            state: record
        }, {
            operation: ANALYZE_OPERATION,
            signal: options.signal,
            sinceSequence: options.sinceSequence
        });
    } catch {
        return null;
    }
    return latest;
}

/**
 * Document analysis: owns the SSE request, the request token that discards
 * stale events, and the cancel path that restores the previous workspace.
 */

/**
 * The operation name every notice and flight for this run carries.
 *
 * A module constant rather than a bare `'analyze'` at four call sites, because
 * the pending notice and the flight have to agree on it: the flight's success
 * retires notices *by operation*, so a typo in one of them leaves
 * "analysis in progress" on screen after a run that has finished, with no error
 * and nothing to dismiss it.
 */
const ANALYZE_OPERATION = 'analyze';

/**
 * How many sections this run will produce.
 *
 * Derived from the same three checkboxes the request is built from, so the
 * count cannot disagree with the request. It exists because "Rendering summary
 * results…" frozen on screen for ninety seconds while five more sections are
 * still being generated is the same lie as a spinner: it names a stage the work
 * has not reached. With a count, the reader can tell a run that is 1 of 6 from
 * one that is finishing, and a run that stalls at 2 of 6 is visibly stalled.
 */
function expectedSectionCount(generateMermaid, generateEvaluation, generateResearchBrief) {
    // summary, quotes and mindmap are always planned; the rest are optional.
    return 3 + (generateMermaid ? 1 : 0) + (generateEvaluation ? 1 : 0) + (generateResearchBrief ? 1 : 0);
}

/**
 * Whether the run finished whole, and if not, which parts did not.
 *
 * Defined in `sections.js`, which owns the analysis payload and which both the
 * upload and the import path already import, so the two runs of the same
 * analysis are judged by the same code rather than by two copies of it.
 */

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

    const previousWorkspace = state.currentAnalysisResult ? captureWorkspaceState() : null;

    // Captured *before* the request, not after it fails. The question is that a
    // failure handler cannot restore what an earlier optimistic clear already
    // destroyed — by the time we would want it back it is gone. A file input’s
    // `files` list is read-only, so the File object is what has to be held; see
    // `PRESERVED_FIELDS.file`.
    const preserved = capturePreservedFields();

    if (state.analyzeController) {
        state.analyzeController.abort();
    }
    state.analyzeController = new AbortController();
    state.analyzeRequestId += 1;
    const requestId = state.analyzeRequestId;

    resetResultView();
    state.currentSessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    state.pendingAnalysisSnapshot = previousWorkspace;
    setButtonLoading(analyzeBtn, 'Analyzing...', 'Analyze Document', true);
    setCancelVisible('cancelAnalyzeBtn', true);
    showRunIndicator();
    document.getElementById('loading').classList.add('active');
    beginProgressClock();
    updateAnalysisProgress('upload', 'Uploading your document and reading its text. A scanned PDF may need OCR, which takes noticeably longer.');
    hideError();
    setControlDisabled(askBtn, true);
    updateStatus('Reading the document...', 'idle');
    // Scoped to `analyze` so that the flight's own success can retire this
    // notice in `endFlight` below. Without the scope the pending notice is
    // matched by operation only, and an unlabelled one is never dismissed by a
    // success — so "analysis in progress" outlives the analysis it describes,
    // and the reader is left reading a status line for a run that finished.
    //
    // The id is kept so a cancellation can retire exactly this notice, without
    // the by-operation sweep that a success uses: that sweep is right for a
    // completed run and wrong for a cancelled one, where a genuine failure from
    // an earlier attempt should stay on screen.
    const pendingNoticeId = notifyPending(
        '分析进行中',
        '文档正在解析。这可能需要一两分钟——这里显示的是服务器报告的阶段和真实用时，不显示百分比，因为服务端没有报告这一步的完成度。',
        { operation: ANALYZE_OPERATION }
    );
    // Published so `cancelAnalyzeRequest` can retire it. A cancel bumps the
    // request token, so this run's `finally` skips its settled branch, and a
    // cancel is not a success — so the by-operation retirement in `endFlight`
    // never runs for it.
    state.pendingNoticeIds.analyze = pendingNoticeId;
    // Opened here rather than inside the request, so the run is registered as
    // in-flight for the whole of the upload-and-wait, not only for the moment
    // the socket is open.
    const flight = beginFlight(ANALYZE_OPERATION, { progressKnown: false, label: 'analysis' });

    const plannedSections = expectedSectionCount(generateMermaid, generateEvaluation, generateResearchBrief);
    const seenSections = new Set();
    // Whether the server ever sent the terminal `done` event. A stream that
    // closes without it resolves normally at the transport layer, so without
    // this the reader gets a spinner that stops, an empty workspace, and a
    // status chip still claiming sections are streaming.
    let settled = false;
    // Whether the reader cancelled. Distinct from `!settled`, which also covers
    // a run that died on its own, because the two must leave the screen saying
    // different things.
    let cancelled = false;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', state.currentSessionId);
    formData.append('generate_mermaid', String(generateMermaid));
    formData.append('generate_evaluation', String(generateEvaluation));
    formData.append('generate_research_brief', String(generateResearchBrief));
    if (apiKey) formData.append('api_key', apiKey);

    try {
        // The stall watchdog lives inside `readSseStream`, not here. That is the
        // point of routing the stream through the transport layer: a parse that
        // goes quiet is detected and reported as `client_stream_interrupted` —
        // a retryable failure with a "重试" affordance — instead of leaving a
        // spinner on screen forever, which is indistinguishable from work that
        // is still running.
        const response = await apiRequest('/api/analyze/stream', {
            method: 'POST',
            body: formData,
            stream: true,
            raw: true,
            operation: 'analyze',
            signal: state.analyzeController.signal
        });

        await readSseStream(response, {
            start: payload => {
                state.currentSessionId = payload.session_id || state.currentSessionId;
                // The one field that makes this run recoverable. The route has
                // already submitted the work to `infra/jobs` and is only
                // watching the row, so if this connection dies the job does not
                // — and the id is the way back to it. Without it the only
                // recovery is to submit the document again, which is a second
                // parse and a second bill for the same file.
                state.currentAnalysisJobId = String(payload.job_id || '');
                document.getElementById('result').classList.add('active');
                setFileInfo(file.name, '...');
                // Deliberately still on the *read* step. The `start` event is
                // emitted before the document is parsed, so claiming analysis
                // has begun here is a promise the server has not made yet — and
                // for a scanned PDF the parse is the long part. It stays on this
                // step until the first section actually arrives, which is the
                // first moment the server has told us any analysis happened.
                updateAnalysisProgress('upload', 'Reading the document text. This step can take a while on a scan, and may re-run OCR if the text is unreadable.');
                updateStatus('Reading the document...', 'idle');
            },
            section: payload => {
                if (requestId !== state.analyzeRequestId) return;
                const sectionName = payload.name;
                const section = payload.section || {};
                state.currentSections = { ...state.currentSections, [sectionName]: section };
                applyAnalysisSection(sectionName, section, generateEvaluation, generateMermaid, generateResearchBrief);
                seenSections.add(sectionName);
                const arrived = seenSections.size;
                // The step stays on "analyze" until every planned section has
                // arrived. Moving it to "render" on the first one is what made a
                // two-minute run look finished for a minute and a half of it.
                const complete = arrived >= plannedSections;
                updateAnalysisProgress(
                    complete ? 'render' : 'analyze',
                    complete
                        ? 'All sections received. Assembling the workspace...'
                        : `Analysing — section ${arrived} of ${plannedSections} received (${sectionName.replace('_', ' ')}).`
                );
                updateStatus(`Analysing: ${arrived} of ${plannedSections} sections received.`, 'idle');
            },
            done: async payload => {
                if (requestId !== state.analyzeRequestId) return;
                settled = true;
                await applyAnalysisResult(payload, file.name, generateEvaluation, generateMermaid, generateResearchBrief);
                const outcome = describeAnalysisOutcome(payload);
                updateAnalysisProgress('ready', outcome.tone === 'success'
                    ? 'Workspace ready. You can ask questions or export the session.'
                    : `Workspace ready, but the analysis is incomplete. ${outcome.text}`);
                endProgressClock(outcome.tone === 'success' ? '分析完成。' : '分析完成，但有部分未生成。');
                updateStatus(outcome.text, outcome.tone);
                focusAnalysisWorkspace();
            },
            error: payload => {
                // An `error` event carries the same `{ error, code }` envelope as
                // an HTTP failure, so it is raised rather than thrown as a bare
                // `Error`. That is what lets the notice below name the cause and
                // decide whether a retry is worth offering, instead of printing
                // the server’s prose into a banner.
                throw buildApiError(payload, 'Analysis failed.');
            }
        }, { operation: 'analyze', signal: state.analyzeController.signal });

        // The stream ended without the terminal `done` event. The transport
        // resolved normally, so without this the run looks like it simply
        // finished: the spinner comes down, the workspace stays empty, and the
        // status chip still claims sections are streaming. Whatever did arrive
        // is named and the previous workspace is put back, because this is a
        // failed run and `analyze()` already has a path for that.
        if (!settled && requestId === state.analyzeRequestId) {
            await reportInterruptedAnalysis(previousWorkspace, seenSections, plannedSections);
        }
    } catch (error) {
        if (error.name === 'AbortError' || error.code === 'client_cancelled') {
            // Not a failure, and not a success either: the reader stopped it.
            // Flagged rather than inferred below, because the stream can also
            // end cleanly on an abort, in which case the transport never throws
            // and this branch is not reached at all.
            cancelled = true;
            // The reader stopped it, so the "analysis in progress" line is now
            // false. Raised unconditionally: a cancellation that surfaces as a
            // clean end-of-stream rather than a thrown AbortError never reaches
            // this branch, and the notice is left to `cancelAnalyzeRequest`.
            if (pendingNoticeId) {
                dismissNotice(pendingNoticeId);
            }
            return;
        }

        // The reader’s work goes back first. A failure notice that arrives while
        // their question is still in the box is a failure they have to fix
        // twice; losing the selection means the retry cannot even be attempted.
        restorePreservedFields(preserved);

        // A dropped or timed-out connection is not a failed analysis, and this
        // is the same re-attach the interrupted-stream path uses. The transport
        // already classes both as `client_stream_interrupted` and
        // `client_timeout`, and the i18n copy for each says the server may
        // already have finished — so the copy has been promising a reconnect
        // that nothing performed. It performs it now, and asks the job what
        // became of the run before anything is reported as failed.
        if (state.currentAnalysisJobId) {
            const jobState = await reattachAnalysisJob(state.currentAnalysisJobId, {
                signal: state.analyzeController?.signal
            });
            if (jobState) {
                const recovered = describeJobOutcome(jobState.state, state.currentAnalysisJobId);
                if (recovered.outcome !== 'unknown') {
                    restoreAfterRecoveredJob(previousWorkspace, recovered, state.currentAnalysisJobId);
                    return;
                }
            }
        }

        if (previousWorkspace) {
            restoreWorkspaceState(previousWorkspace);
            showError(`${error.message || '分析失败。'}\n\nYour previous successful workspace has been restored.`);
            focusAnalysisWorkspace({ scroll: false });
        } else {
            state.currentSessionId = '';
            resetResultView();
            showError(error.message || '分析失败。');
        }

        // Routed through the runtime so the kind — retryable or terminal —
        // decides whether a "重试" button appears. The retry re-issues *this*
        // request, with the file still attached, not a generic reload.
        notifyFailure(error, { operation: ANALYZE_OPERATION, onRetry: () => analyze() });
        updateStatus('Analysis failed', 'error');
    } finally {
        // Every exit closes the flight, including a cancellation and a failure:
        // the run is over on all three, and a flight left open is a permanent
        // "in flight" entry in the runtime's ledger, which is what makes a
        // stall reportable at all. Closing it here rather than on the success
        // branch is the whole reason the closing cannot be forgotten.
        //
        // Success retires this run's own notices — the "analysis in progress"
        // one above among them — so a completed run does not leave a status
        // line describing work that has finished. A failure keeps its notice,
        // and a cancellation retires the pending one without raising anything:
        // the reader stopped it deliberately and already knows.
        endFlight(flight, {
            succeeded: settled,
            operation: ANALYZE_OPERATION
        });
        if (requestId === state.analyzeRequestId) {
            // Only on a run that did not settle. The `done` handler already
            // stopped the clock with the outcome it computed, and stopping it a
            // second time re-reads a `startedAt` that the first call deleted —
            // so the elapsed figure falls back to now, and a two-minute analysis
            // reports "0 seconds" on the one screen that says how long it took.
            // The only honest measure of a parse is the real one.
            if (!settled) {
                // Also chosen by what happened rather than asserted: a cancelled
                // run is the reader's own decision, not a run that failed, and
                // only the genuine non-success gets "没有完成".
                endProgressClock(cancelled ? '这次分析已被取消。' : '这次分析没有完成。');
            }
            document.getElementById('loading').classList.remove('active');
            hideRunIndicator();
            setButtonLoading(analyzeBtn, 'Analyzing...', 'Analyze Document', false);
            setCancelVisible('cancelAnalyzeBtn', false);
            state.analyzeController = null;
            state.pendingAnalysisSnapshot = null;
            // The id is spent. Left in place it would let a later failure
            // re-attach to *this* run's job and report its outcome as though it
            // were the new one's, which is the same class of confusion the
            // request token exists to prevent.
            state.currentAnalysisJobId = '';
        }
    }
}

function cancelAnalyzeRequest() {
    if (!state.analyzeController) return;
    state.analyzeController.abort();
    state.analyzeRequestId += 1;
    // Retired here because nothing else can. The request's own `finally` sees a
    // stale request token and skips its settled branch, and a cancel is not a
    // success, so the by-operation retirement in `endFlight` does not apply.
    // Without this, "分析进行中" stays above a workspace the reader has just
    // been told they cancelled.
    if (state.pendingNoticeIds.analyze) {
        dismissNotice(state.pendingNoticeIds.analyze);
        state.pendingNoticeIds.analyze = '';
    }
    document.getElementById('loading').classList.remove('active');
    hideRunIndicator();
    setButtonLoading(document.getElementById('analyzeBtn'), 'Analyzing...', 'Analyze Document', false);
    setCancelVisible('cancelAnalyzeBtn', false);
    state.analyzeController = null;
    // The id is captured *before* the message is written, because the status
    // line has to be true about what a cancel actually did.
    //
    // Cancelling here aborts this page's connection and nothing else. The route
    // has already submitted the work to a job row, and its own disconnect path
    // (`request.is_disconnected()`) simply returns — the worker thread keeps
    // parsing and keeps calling the model. So "Analysis canceled" was a claim
    // the product could not support: the reader stops paying attention, and the
    // document keeps being analysed at their expense. The id is given so the run
    // is findable, and the wording says what actually happened.
    const abandonedJobId = state.currentAnalysisJobId;
    const stillRunning = abandonedJobId
        ? ` The server-side task ${abandonedJobId} was not cancelled and may still be running.`
        : '';
    if (state.pendingAnalysisSnapshot) {
        restoreWorkspaceState(state.pendingAnalysisSnapshot);
        state.pendingAnalysisSnapshot = null;
        focusAnalysisWorkspace({ scroll: false });
        // Re-derived from the restored result rather than asserted. The previous
        // workspace may have been a partial run, and a blanket "Analysis
        // canceled" would leave a status chip that says nothing about what is
        // actually on screen.
        const outcome = describeAnalysisOutcome(state.currentAnalysisResult);
        updateStatus(`Stopped watching.${stillRunning} ${outcome.text}`, outcome.tone === 'success' ? 'idle' : outcome.tone);
    } else {
        state.currentSessionId = '';
        resetResultView();
        updateStatus(`Stopped watching.${stillRunning}`, 'idle');
    }
    state.currentAnalysisJobId = '';
}


export { analyze };
export { cancelAnalyzeRequest };
