import { ANSWER_MODE_LABELS, PRESENTATION_DIRECT_QUOTE } from './config.js';
import { state } from './state.js';
import { apiRequest, buildApiError, readSseStream } from './api.js';
import { beginFlight, capturePreservedFields, dismissNotice, endFlight, notifyFailure, notifyPending, restorePreservedFields } from './runtime.js';
import { replaceWithFormattedContent } from './format.js';
import { copyText, hideError, scrollWindowTo, setButtonLoading, setCancelVisible, setControlDisabled, showError, updateStatus } from './dom.js';
import { fillQuestionInput } from './suggestions.js';
import { renderExportPreview } from './export.js';
import { collectCitations, renderInlineClaims } from './evidence.js';

/** The Q&A panel: streaming answers, cancellation and transcript storage. */

/**
 * The operation name every notice and flight for a turn carries.
 *
 * A module constant because the pending notice and the flight have to agree on
 * it: a success retires notices *by operation*, so a typo in one of them leaves
 * "回答生成中" on screen after the answer has arrived, with nothing to dismiss
 * it and nothing wrong to look at.
 */
const ASK_OPERATION = 'ask';

function clearChatHistory() {
    document.getElementById('chatHistory').replaceChildren();
}

/**
 * The claims an answer asserts, paired with the passages they resolve to.
 *
 * `collectCitations` returns whichever list the payload happens to carry, and
 * the two lists are not interchangeable. A *citation* is a retrieved passage
 * and its `text` is that whole passage — up to 3000 characters of it. A *claim*
 * is one assertion the answer makes, and its `quote` is the single sentence the
 * model believed it was quoting. `renderInlineClaims` renders whatever it is
 * given as a claim, so handing it a citation renders the entire retrieved
 * passage inside real quotation marks under a "逐字引用" heading — asserting
 * that the paper said all of it verbatim, when verification only ever aligned
 * the one sentence in `verified_text`.
 *
 * That is the release gate failing in the direction it exists to prevent, and
 * it fails silently: the reader sees quotation marks, which is exactly the
 * signal they are told means "checked against the source".
 *
 * So the two are joined here rather than in the renderer. The merge is
 * deliberately one-directional:
 *
 * * a **claim** is the unit of display. Its `text` is the assertion, and the
 *   quotation it shows is `verified_text` — the characters the paper actually
 *   wrote, taken from the source rather than from the model’s reply. A claim
 *   with no `verified_text` is therefore rendered as a paraphrase: there is
 *   nothing verified to put in quotation marks, and `claims.js` fails toward
 *   the unverifiable side already, so we do not contradict it.
 * * a **citation** contributes only the locator and the label, looked up by the
 *   id the claim names. A claim naming an id retrieval never returned is
 *   dropped, exactly as an unaddressable chip would be — a chip that opens
 *   nothing is a promise the panel cannot keep.
 * * a payload carrying no `claims` at all falls back to the collected
 *   citations, unchanged. That is the pre-T1.7 shape and the degraded path, and
 *   it must keep rendering rather than going blank when a contract moves.
 */
function buildAnswerClaims(payload, collected) {
    const fallback = Array.isArray(collected) ? collected : [];
    const rawClaims = Array.isArray(payload?.claims) ? payload.claims : [];
    if (!rawClaims.length) {
        return fallback;
    }

    const citationsById = new Map();
    fallback.forEach(entry => {
        const id = String(entry?.evidence_id || entry?.evidenceId || entry?.chunk_id || '');
        if (id) {
            citationsById.set(id, entry);
        }
    });

    const seen = new Set();
    const claims = [];
    rawClaims.forEach(raw => {
        if (!raw || typeof raw !== 'object') return;
        const ids = Array.isArray(raw.evidence_ids) ? raw.evidence_ids : [];
        // First named id that actually resolves. An unresolvable one is not
        // carried forward into the citation list by the backend either, so
        // preferring a resolvable sibling loses nothing.
        const resolved = ids.map(String).find(id => citationsById.has(id)) || '';
        if (!resolved) return;

        const key = `${resolved}:${String(raw.text || '')}`;
        if (seen.has(key)) return;
        seen.add(key);

        const citation = citationsById.get(resolved) || {};
        const isDirect = String(raw.presentation || '') === PRESENTATION_DIRECT_QUOTE;
        const verified = String(raw.verified_text || '').trim();

        claims.push({
            text: isDirect ? verified : String(raw.text || ''),
            evidence_id: resolved,
            chunk_id: citation.chunk_id || '',
            citation_label: raw.citation_label || citation.citation_label || '',
            locator: raw.locator ?? citation.locator ?? null,
            heading_path: citation.heading_path || [],
            // A direct quote with nothing verified behind it is a paraphrase.
            // The alternative is quotation marks around text this client never
            // received from the source, which is the one thing `claims.js`
            // exists to make impossible.
            presentation: isDirect && verified ? PRESENTATION_DIRECT_QUOTE : 'paraphrase',
            uses_quotation_marks: Boolean(isDirect && verified),
            outcome: raw.outcome || citation.outcome || '',
            message: raw.message || citation.message || '',
            mode: raw.mode || '',
            score: citation.score
        });
    });

    return claims.length ? claims : fallback;
}

/**
 * What the server will and will not vouch for, in words.
 *
 * The backend computes `confidence` from the claims and explicitly never asks
 * the model for it. Rendering an answer without it means a refusal
 * (`evidence_count: 0`, no claims) arrives in the same visual chrome as a fully
 * supported one — same header, same Copy button, same follow-up chips — and the
 * only signal is a sentence in the body the reader has to notice and
 * understand. `notes` and `retrieval.degradations` are the two other facts the
 * response deliberately carries for exactly this reason and that this client was
 * dropping.
 */
const CONFIDENCE_COPY = {
    supported: 'Every claim in this answer is carried by a passage from this document, and any quotation was checked against it.',
    partially_supported: 'Some claims in this answer are not carried by a passage from this document. The unsupported ones are marked below.',
    insufficient_evidence: 'This answer is not backed by evidence from this document, so nothing in it is presented as the paper’s own claim.'
};

function describeRetrieval(retrieval) {
    const count = Number(retrieval?.evidence_count);
    const parts = [];
    if (Number.isFinite(count)) {
        parts.push(`Retrieved ${count} passage${count === 1 ? '' : 's'} from this document for this question.`);
    }
    const degradations = Array.isArray(retrieval?.degradations) ? retrieval.degradations.filter(Boolean) : [];
    if (degradations.length) {
        parts.push(`Retrieval was degraded: ${degradations.join('; ')}.`);
    }
    return parts.join(' ');
}

/**
 * The answer’s provenance, rendered under the answer it qualifies.
 *
 * `data-confidence` carries the machine-readable value so this stays assertable
 * without a browser and so the design agent has a hook; the sentences carry the
 * meaning in words, because a signal that exists only as a colour or an
 * attribute is erased by a stylesheet that fails to load, a high-contrast mode,
 * or a screen reader.
 */
function renderAnswerVerdict(container, payload) {
    if (!container) return;

    const confidence = String(payload?.confidence || '');
    const known = Object.prototype.hasOwnProperty.call(CONFIDENCE_COPY, confidence);
    // An unrecognised confidence is reported as unrecognised, never as success.
    // The backend only emits three values, so a fourth means the contract moved
    // and the recoverable direction is to under-claim.
    const verdict = document.createElement('div');
    verdict.className = 'answer-verdict';
    verdict.dataset.confidence = known ? confidence : 'unknown';
    verdict.setAttribute('role', 'status');
    verdict.setAttribute('aria-live', 'polite');

    const summary = document.createElement('p');
    summary.className = 'answer-verdict-label';
    summary.textContent = known
        ? CONFIDENCE_COPY[confidence]
        : 'This answer did not report how well it is supported, so treat it as unverified.';
    verdict.appendChild(summary);

    const retrievalText = describeRetrieval(payload?.retrieval);
    if (retrievalText) {
        const retrievalLine = document.createElement('p');
        retrievalLine.className = 'answer-verdict-detail';
        retrievalLine.textContent = retrievalText;
        verdict.appendChild(retrievalLine);
    }

    // `notes` is where the server states that a citation’s locator came from
    // re-derived plain text rather than a real parse, so a character offset and
    // a page number are not rendered as though they were the same kind of thing.
    const notes = Array.isArray(payload?.notes) ? payload.notes.filter(Boolean) : [];
    if (notes.length) {
        const notesLine = document.createElement('p');
        notesLine.className = 'answer-verdict-detail';
        notesLine.textContent = notes.join(' ');
        verdict.appendChild(notesLine);
    }

    container.replaceChildren(verdict);
    container.hidden = false;
    return verdict;
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
        replaceWithFormattedContent(body, message, 'No content available.');
    } else {
        body.textContent = message;
    }

    item.appendChild(header);
    item.appendChild(body);
    chatHistory.appendChild(item);
    return item;
}

function createInlineSpinner() {
    const spinner = document.createElement('div');
    spinner.className = 'spinner inline-spinner';
    return spinner;
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
    // Delegated rather than bound per button. `restoreWorkspaceState` puts the
    // transcript back as `innerHTML`, which produces fresh nodes carrying no
    // listeners — a listener bound here would be silently gone after a failed
    // re-analysis restored the previous workspace, leaving every restored Copy
    // button inert while looking completely normal. The data attribute survives
    // serialisation; a listener does not.
    button.dataset.copyAnswerId = answerId;
    setControlDisabled(button, true);

    const content = document.createElement('div');
    content.id = answerId;
    content.className = 'answer-content content';
    content.dataset.rawContent = '';
    content.appendChild(createInlineSpinner());

    // Claims and citation chips live in their own container beside the answer
    // rather than inside `content`. The answer is markdown rendered by
    // `format.js` and re-rendered on every streamed delta; splicing verified
    // quote elements into it would put generated HTML in charge of where a
    // quotation mark goes, and would lose them on the next delta. Claims are
    // rendered from data into real DOM nodes, once, when the answer settles.
    const claims = document.createElement('div');
    claims.className = 'answer-claims';
    claims.dataset.count = '0';
    claims.hidden = true;

    // What the server will and will not vouch for, and what it retrieved to
    // decide. A separate container from the claims because it qualifies the
    // answer as a whole, where a claim qualifies one sentence: an answer whose
    // every claim failed verification has no claims to render at all, and must
    // still say so.
    const verdict = document.createElement('div');
    verdict.className = 'answer-verdict-host';
    verdict.hidden = true;

    header.appendChild(title);
    header.appendChild(button);
    answerDiv.appendChild(header);
    answerDiv.appendChild(content);
    answerDiv.appendChild(verdict);
    answerDiv.appendChild(claims);
    chatHistory.appendChild(answerDiv);
    return { answerDiv, content, button, answerId, claims, verdict };
}

function updateStreamingAnswer(shell, answer) {
    if (!shell || !shell.content) return;
    shell.content.dataset.rawContent = answer || '';
    replaceWithFormattedContent(shell.content, answer, 'No answer available.');
}

function finalizeStreamingAnswer(shell, answer) {
    if (!shell || !shell.button || !shell.content) return;
    updateStreamingAnswer(shell, answer);
    setControlDisabled(shell.button, false);
    // The Copy listener is delegated from `initializeChatDelegates`, so there is
    // nothing to bind here and nothing that an `innerHTML` restore can strip.
    //
    // The follow-up chips that used to be appended under each answer are
    // deliberately not re-added. `state.currentNextActions` is the *analysis*
    // payload’s static list — identical for turn 1 and turn 7 — and it is
    // already rendered above the transcript by `renderNextActions`. Repeating
    // it beneath every answer implied those prompts were derived from that
    // answer, which they are not. One home for the suggestions, where
    // "what next" belongs.
}

/**
 * One delegated click handler for the whole transcript.
 *
 * Every control inside an answer is described by a data attribute rather than a
 * closure, so a transcript restored from `innerHTML` is fully functional. The
 * alternative — re-binding after each restore — is the version where a restored
 * workspace looks perfect and does nothing when pressed.
 */
function handleChatHistoryClick(event) {
    const copyButton = event.target.closest?.('[data-copy-answer-id]');
    if (copyButton) {
        copyButton.dataset.bound = 'true';
        copyText(copyButton.dataset.copyAnswerId, copyButton);
        return;
    }
    const actionButton = event.target.closest?.('[data-ask-prompt]');
    if (actionButton) {
        fillQuestionInput(actionButton.dataset.askPrompt);
    }
}

function initializeChatDelegates() {
    document.getElementById('chatHistory')
        ?.addEventListener('click', handleChatHistoryClick);
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
    if (!state.currentSessionId) {
        questionInput.focus();
        updateStatus('Analyze a document before asking', 'idle');
        showError('Requires document analysis first.');
        return;
    }

    if (state.askController) {
        state.askController.abort();
    }
    state.askController = new AbortController();
    state.askRequestId += 1;
    const requestId = state.askRequestId;

    // The mode is captured once, here, and used for the transcript entry as
    // well as the request body. Reading state.currentAnswerMode again when the
    // stream settles means a reader who switches mode while the answer is
    // streaming gets a turn filed under a mode they did not ask in — in the
    // transcript and in the exported report, which is the record of what was
    // asked.
    const answerMode = state.currentAnswerMode;

    // Recorded so a cancel can put the question back. Set here rather than at
    // the top of the turn, because this is the first point at which the question
    // is known to be committed to a request.
    state.pendingQuestion = question;

    // Captured before the composer is cleared below, because the `catch` that
    // would want it back runs after. A question is the most expensive thing in
    // this interface to retype, so it is the one field that is never allowed to
    // disappear into a failed request.
    const preserved = capturePreservedFields();

    hideError();
    appendChatText(`User · ${ANSWER_MODE_LABELS[answerMode]}`, question, 'chat-question');
    questionInput.value = '';
    setButtonLoading(askBtn, 'Processing...', 'Send', true);
    setCancelVisible('cancelAskBtn', true);
    updateStatus('Answering based on current document...', 'idle');
    // Scoped to `ask` so the flight's own success can retire it in `endFlight`.
    // An unscoped pending notice is matched by operation only, and an operation
    // of `''` matches nothing — so "回答生成中" outlives the answer it
    // describes, leaving the reader with a status line claiming work in
    // progress that finished a second ago.
    const pendingNoticeId = notifyPending('回答生成中', '正在基于当前文档生成回答。', { operation: ASK_OPERATION });
    // Published so `cancelAskRequest` can retire it. A cancel bumps the request
    // token, so this turn's `finally` skips its settled branch, and a cancel is
    // not a success — so the by-operation retirement in `endFlight` never runs
    // for it, and the notice would outlive the answer it describes.
    state.pendingNoticeIds.ask = pendingNoticeId;
    // Opened before the request so the turn is registered as in-flight for the
    // whole of the wait, not only while the socket is open.
    const flight = beginFlight(ASK_OPERATION, { progressKnown: false, label: 'answer' });
    const streamingShell = appendStreamingAnswerShell();
    state.activeStreamingAnswerShell = streamingShell;
    let accumulatedAnswer = '';
    // Whether the server ever sent the terminal done event. A stream that
    // closes without it is not a success and is not obviously a failure: the
    // transport resolves normally, so without this the reader is left staring
    // at a spinner that will never resolve, next to a question they cannot
    // tell was answered. Same shape as the analysis stream, same fix.
    let settled = false;
    // Whether the reader cancelled, which is neither a settled answer nor a
    // failed one: the text on screen and the question in the transcript are
    // both the reader's own doing, and the status line must say so.
    let cancelled = false;

    try {
        const response = await apiRequest('/api/ask/stream', {
            method: 'POST',
            body: {
                question,
                session_id: state.currentSessionId,
                session_token: state.currentSessionToken,
                api_key: apiKey,
                answer_mode: answerMode
            },
            // A model call with no streaming body is allowed to be slow, but not
            // unbounded: past this the request is abandoned and reported, and the
            // reader is told the server may still be working on it.
            generation: true,
            raw: true,
            operation: 'ask',
            signal: state.askController.signal
        });

        await readSseStream(response, {
            start: () => {
                updateStatus('Streaming answer...', 'idle');
            },
            delta: payload => {
                if (requestId !== state.askRequestId) return;
                accumulatedAnswer += payload.text || '';
                updateStreamingAnswer(streamingShell, accumulatedAnswer);
            },
            done: payload => {
                if (requestId !== state.askRequestId) return;
                settled = true;
                accumulatedAnswer = payload.answer || accumulatedAnswer;
                finalizeStreamingAnswer(streamingShell, accumulatedAnswer);
                // The claims, joined to the passages they resolve to. See
                // buildAnswerClaims: handing the renderer the citation list
                // instead would put the whole retrieved passage in quotation
                // marks, which is the release-gate failure this product exists
                // to prevent. Falls back to the collected citations when the
                // payload carries no claims, so a contract change degrades to
                // the old rendering rather than to a blank.
                const citations = buildAnswerClaims(payload, collectCitations(payload));
                renderInlineClaims(streamingShell.claims, citations);
                // Always rendered, including for a refusal that has no claims to
                // show: insufficient_evidence is the single most important
                // fact on this screen and it is the one the reader is most
                // likely not to notice buried in the prose.
                renderAnswerVerdict(streamingShell.verdict, payload);
                const confidence = String(payload?.confidence || '');
                state.currentChatTurns.push({
                    question,
                    answer: accumulatedAnswer,
                    answer_mode: answerMode,
                    confidence,
                    citations,
                    retrieval: payload?.retrieval || null,
                    notes: Array.isArray(payload?.notes) ? payload.notes : [],
                    timestamp: new Date().toISOString()
                });
                renderExportPreview();
                state.activeStreamingAnswerShell = null;
                // The pending "回答生成中" notice is retired in the `finally`
                // below, by the flight that owns it, rather than here — and
                // scoped to this operation rather than by `clearNotices()`.
                // That call was the only thing retiring the pending notice, and
                // it retired *every* notice: asking one question silently
                // deleted a paper-search failure or a reparse failure belonging
                // to a different operation, so a real problem vanished because
                // the reader did something else. A success may retire its own
                // failures; it may not delete other operations'.
                //
                // The status chip repeats the verdict in the app’s existing
                // visual language, so the signal is available even where the
                // transcript has been scrolled away. A refusal is not a success
                // and must not wear its colour.
                if (confidence === 'insufficient_evidence') {
                    updateStatus('Answer ready — not backed by this document', 'error');
                } else if (confidence === 'partially_supported') {
                    updateStatus('Answer ready — partly unverified', 'idle');
                } else {
                    updateStatus('Answer ready', 'success');
                }
            },
            error: payload => {
                // Carries the `{ error, code }` envelope, so the failure below
                // can tell "the model refused" from "the stream broke" and offer
                // a retry for only one of them.
                throw buildApiError(payload, 'Question request failed.');
            }
        }, { operation: 'ask', signal: state.askController.signal });

        // The transport resolved but the answer never settled. Report it as
        // what it is — an interrupted stream — rather than leaving the spinner
        // up, and never as an answer: the text on screen is whatever streamed
        // before the cut and has not been through the final verification pass,
        // so it is labelled as incomplete instead of being presented as the
        // paper’s answer.
        if (!settled && requestId === state.askRequestId && !state.askController?.signal.aborted) {
            const partial = accumulatedAnswer.trim();
            if (partial) {
                finalizeStreamingAnswer(streamingShell, partial);
            }
            streamingShell.answerDiv.dataset.incomplete = 'true';

            // The verdict host, not the answer body — and it has to be unhidden.
            // `appendStreamingAnswerShell` creates it hidden because a settled
            // answer fills it in `renderAnswerVerdict`, and this path never
            // reaches that. Appending into it while it stays hidden put the one
            // sentence telling the reader that this text is *not* a verified
            // answer inside a container the stylesheet renders as
            // `display: none`: the partial answer looked exactly like a complete
            // one, with its Copy button enabled and nothing marking it. A
            // degraded state that looks normal is the specific failure this
            // product exists to avoid.
            const verdictHost = streamingShell.verdict;
            verdictHost.hidden = false;
            verdictHost.dataset.state = 'incomplete';
            const incomplete = document.createElement('p');
            incomplete.className = 'answer-verdict-detail';
            incomplete.textContent = partial
                ? 'The connection ended before this answer finished. What is shown above is the partial text, not a verified answer: it has not been checked against the document, and no citation for it was verified.'
                : 'The connection ended before this answer began. Nothing was returned.';
            verdictHost.replaceChildren(incomplete);

            // The Copy control is taken away rather than left enabled. Copying
            // is how text leaves this interface — into notes, into a draft, into
            // a reply — and a partial answer pasted out of here is
            // indistinguishable from a verified one once it has left. The
            // reader who wants the partial text can still select it.
            setControlDisabled(streamingShell.button, true);
            streamingShell.button.title = 'This answer did not finish, so it cannot be copied as a verified answer.';

            appendChatText('System Error', 'The answer stream ended before it completed. Ask again to retry.', 'chat-answer chat-error');
            state.activeStreamingAnswerShell = null;
            state.pendingQuestion = '';
            updateStatus('Answer interrupted before it completed', 'error');
        }
    } catch (error) {
        if (error.name === 'AbortError' || error.code === 'client_cancelled') {
            // The reader stopped it, so "回答生成中" is now false. Retired
            // unconditionally: a cancellation that surfaces as a clean
            // end-of-stream never throws and never reaches here, so relying on
            // this branch alone would leave the notice up for that case.
            cancelled = true;
            if (pendingNoticeId) {
                dismissNotice(pendingNoticeId);
            }
            return;
        }

        // The question goes back in the box before anything else is reported.
        // It is also already in the transcript above, so restoring it is not
        // duplication — it is the difference between "press 重试" and "retype
        // this". Any field the user has since edited is left alone, which is
        // what `restorePreservedFields` checks before it writes.
        restorePreservedFields(preserved);

        if (streamingShell && streamingShell.answerDiv && streamingShell.answerDiv.parentNode) {
            streamingShell.answerDiv.parentNode.removeChild(streamingShell.answerDiv);
        }
        appendChatText('System Error', error.message || 'Question request failed.', 'chat-answer chat-error');
        notifyFailure(error, { operation: ASK_OPERATION, onRetry: () => askQuestion() });
        updateStatus('Question request failed', 'error');
    } finally {
        // Every exit closes the flight — success, failure and cancellation. A
        // flight left open is a permanent "in flight" row in the runtime's
        // ledger, and a ledger that cannot be emptied is a ledger nothing can
        // be concluded from. Closing it here rather than on the success branch
        // is what makes forgetting it impossible.
        //
        // A settled turn's success retires this operation's notices, so
        // "回答生成中" does not outlive the answer. A failure keeps its notice —
        // the reader still needs to know — and a cancellation has already
        // retired its own, since the reader stopped it and the notice was never
        // true afterwards.
        endFlight(flight, {
            succeeded: settled,
            operation: ASK_OPERATION
        });
        if (requestId === state.askRequestId) {
            // Cleared once the turn settled, so a settled question cannot
            // reappear in the composer on some later cancel. The failure path
            // does not clear it: restorePreservedFields puts the question back
            // in the box there, and the retry re-submits it. The cancel path
            // reads it before clearing it.
            if (settled) {
                state.pendingQuestion = '';
            }
            setButtonLoading(askBtn, 'Processing...', 'Send', false);
            setCancelVisible('cancelAskBtn', false);
            state.askController = null;
            state.activeStreamingAnswerShell = null;
            questionInput.focus();
            scrollWindowTo({ top: document.body.scrollHeight });
        }
    }
}

function cancelAskRequest() {
    if (!state.askController) return;
    state.askController.abort();
    state.askRequestId += 1;
    // Retired here because nothing else can: the turn's own `finally` sees a
    // stale request token and skips its settled branch, and a cancel is not a
    // success, so the by-operation retirement in `endFlight` does not apply.
    // Without this, "回答生成中" stays on screen above a transcript the reader
    // has just emptied, saying work is in progress that they stopped.
    if (state.pendingNoticeIds.ask) {
        dismissNotice(state.pendingNoticeIds.ask);
        state.pendingNoticeIds.ask = '';
    }
    setButtonLoading(document.getElementById('askBtn'), 'Processing...', 'Send', false);
    setCancelVisible('cancelAskBtn', false);
    state.askController = null;
    if (state.activeStreamingAnswerShell?.answerDiv?.parentNode) {
        state.activeStreamingAnswerShell.answerDiv.parentNode.removeChild(state.activeStreamingAnswerShell.answerDiv);
    }
    state.activeStreamingAnswerShell = null;
    // The question itself stays in the transcript, so without this the reader
    // is left looking at their own question followed by nothing — which reads
    // as a bug rather than as their decision. It is named, and the question
    // goes back in the box so asking again costs one press rather than a
    // retype. state.pendingQuestion is the record of what was in flight, which is
    // more reliable than reading it back out of the transcript’s markup.
    appendChatText('System', 'Canceled before the answer completed. Ask again to retry.', 'chat-answer chat-error');
    const questionInput = document.getElementById('questionInput');
    if (questionInput && !questionInput.value.trim() && state.pendingQuestion) {
        questionInput.value = state.pendingQuestion;
    }
    state.pendingQuestion = '';
    updateStatus('Question canceled', 'idle');
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !document.getElementById('askBtn').disabled) {
        event.preventDefault();
        askQuestion();
    }
}


export { askQuestion };
export { cancelAskRequest };
export { clearChatHistory };
export { handleKeyPress };
export { initializeChatDelegates };
export { buildAnswerClaims };
