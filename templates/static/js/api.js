/**
 * The transport layer, and the only place in the front end that calls `fetch`.
 *
 * Everything above this file deals in *outcomes*; everything below it deals in
 * bytes. Keeping the boundary here is what makes requirement 1 real rather than
 * aspirational: a new call site does not choose a timeout, a size cap, an error
 * shape, or whether a failure is retryable. It cannot forget to, because there
 * is nothing to forget — those decisions are made once, below, and a call site
 * that wanted different ones would have to change this file, which is a diff
 * someone reviews.
 *
 * ## What this file guarantees
 *
 * 1. **Every request is bounded in time and in size.** A hung request is
 *    aborted and reported as a timeout; a runaway body is stopped mid-read and
 *    reported as an oversized response. Without both, one bad response can
 *    white-screen the page, and a request that never settles holds a spinner up
 *    for ever — which is the same defect as "working looks like died", reached
 *    by a different route.
 *
 * 2. **Every failure arrives as an `ApiError` with a machine-readable `code`.**
 *    The code decides the kind, the kind decides whether a retry is offered,
 *    and the prose is only ever displayed. Nothing above this line inspects
 *    prose, so a server-side rewording cannot turn a terminal failure into
 *    something that looks retryable.
 *
 * 3. **Nothing here decides what a failure looks like on screen.** That is
 *    `runtime.js`'s job, and the separation is why this file can be tested for
 *    transport properties and the runtime for presentation ones.
 *
 * ## Why the old helpers still exist
 *
 * `buildApiError`, `parseJsonSafely` and `readSseStream` were the previous
 * public surface and several modules import them. They are kept as thin
 * adapters over the new functions rather than deleted, so a module that has not
 * been migrated yet still gets the size cap, the timeout classification and
 * the code — a partial migration inherits the behaviour instead of either
 * working or being broken.
 */

import {
    AUTO_RETRY_METHODS,
    ERROR_PRESENTATIONS,
    FAILURE_KINDS,
    MAX_AUTOMATIC_RETRIES,
    REQUEST_TIMEOUTS_MS,
    RESYNC_BACKOFF_MS,
    RESYNC_TIMEOUT_MS,
    RESPONSE_SIZE_CAPS_BYTES,
    RETRY_BACKOFF_MS,
    STREAM_STALL_TIMEOUT_MS,
    UNKNOWN_FAILURE_KIND
} from './config.js';
import { beginFlight, endFlight, isOnline } from './runtime.js';

// --------------------------------------------------------------------------
// The error type
// --------------------------------------------------------------------------

/**
 * One error type for every way a request can fail.
 *
 * Carrying the `code` and the derived `kind` on the error itself — rather than
 * leaving the presentation to whichever `catch` block runs first — is what makes
 * the four-state taxonomy survive contact with a dozen call sites. A catch
 * block that only does `error.message` still gets a correct message; a catch
 * block that wants a retry can ask `error.kind` instead of guessing.
 */
class ApiError extends Error {
    constructor(message, details = {}) {
        super(message);
        this.name = 'ApiError';
        this.code = String(details.code || 'client_unknown_error');
        this.status = Number.isFinite(details.status) ? details.status : null;
        this.kind = details.kind || UNKNOWN_FAILURE_KIND;
        this.presentation = ERROR_PRESENTATIONS[this.code] || null;
        this.operation = String(details.operation || '');
        this.cause = details.cause || null;
        // `retryable` is derived, never declared. A caller that wants to know
        // whether to offer a retry reads the same field the notice's button
        // enablement reads, so the two cannot disagree.
        this.retryable = this.kind === FAILURE_KINDS.RETRYABLE;
    }
}

function isAbortError(error) {
    return error?.name === 'AbortError';
}

/**
 * A failure the server never described, or a network that never answered.
 *
 * Split because they are different sentences: a timeout says "we gave up
 * waiting", an unreachable says "there was nobody there to wait for", and the
 * second is a fact about the machine rather than about the request.
 */
function transportFailure(error, operation) {
    if (error instanceof ApiError) return error;
    if (isAbortError(error)) {
        return new ApiError('请求已取消。', { code: 'client_cancelled', operation });
    }
    return new ApiError(error?.message || '网络请求失败。', {
        code: 'client_unreachable',
        operation,
        cause: error
    });
}

// --------------------------------------------------------------------------
// Bounded reading
// --------------------------------------------------------------------------

/**
 * Read a body, refusing to exceed `maxBytes`.
 *
 * A cap, not a truncation: a truncated body is a parse error dressed up as
 * data, and the reader would see a document that ends mid-sentence and believe
 * it whole. Stopping at the cap and *saying so* is the only honest outcome, and
 * `client_response_too_large` is a terminal code because no retry changes the
 * size of the response.
 *
 * Falls back to `response.text()` where streaming bodies are unavailable, and
 * checks the encoded length there too — otherwise a browser without
 * `ReadableStream` would have no cap at all, and the cap would be a property of
 * the browser rather than of this code.
 */
async function readBodyCapped(response, maxBytes) {
    const limit = Number(maxBytes) > 0 ? Number(maxBytes) : RESPONSE_SIZE_CAPS_BYTES.json;
    const declared = Number(response.headers?.get?.('content-length'));

    // A declared length over the cap is refused before a single byte is read.
    // This is the cheap case, and it is the common one: a server that knows how
    // big its own response is tells us, and honouring that is free.
    if (Number.isFinite(declared) && declared > limit) {
        throw new ApiError('响应超出大小上限。', { code: 'client_response_too_large' });
    }

    const body = response.body;
    if (!body || !body.getReader) {
        const text = await response.text();
        // `TextEncoder` rather than `text.length`, which counts UTF-16 code
        // units and under-reports non-ASCII by a factor of up to three — so
        // the string-length check would let a CJK body through at three times
        // the cap.
        const bytes = new TextEncoder().encode(text).length;
        if (bytes > limit) {
            throw new ApiError('响应超出大小上限。', { code: 'client_response_too_large' });
        }
        return text;
    }

    const reader = body.getReader();
    const chunks = [];
    let total = 0;
    try {
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            total += value?.byteLength || 0;
            if (total > limit) {
                throw new ApiError('响应超出大小上限。', { code: 'client_response_too_large' });
            }
            chunks.push(value);
        }
    } finally {
        // Released even on the cap path, or the connection is held open for a
        // body nobody will finish reading.
        reader.releaseLock();
    }

    const merged = new Uint8Array(total);
    let offset = 0;
    chunks.forEach(chunk => {
        merged.set(chunk, offset);
        offset += chunk.byteLength;
    });
    return new TextDecoder().decode(merged);
}

/**
 * Parse a body as JSON, or explain why it could not be.
 *
 * A malformed payload is a *reported degradation*, never an exception: the
 * alternative is a `TypeError` thrown from inside a `catch` that is itself
 * reporting a failure, which is how a bad response becomes a white screen. So
 * every parse failure returns a value the caller can render.
 */
function parseJsonBody(text, statusText = '') {
    try {
        const data = JSON.parse(text);
        if (!data || typeof data !== 'object') {
            return { error: '响应不是 JSON 对象。', code: 'client_malformed_response' };
        }
        return data;
    } catch {
        // An HTML body where JSON was expected has a well-known cause — a
        // `base_url` pointing at a web page, or a proxy answering instead of
        // the app — and naming it saves the reader from guessing. It is a
        // *diagnosis*, not a branch on the text: the code is what callers act on.
        const looksLikeHtml = /^\s*(<!doctype html|<html|<head|<body)/i.test(text);
        return {
            error: looksLikeHtml
                ? '服务端返回了一个网页，而不是接口的响应。请检查服务地址是否指向了网页。'
                : '响应不是合法 JSON。',
            code: 'client_malformed_response',
            status_hint: statusText
        };
    }
}

// --------------------------------------------------------------------------
// The choke point
// --------------------------------------------------------------------------

/**
 * The one `fetch` in the front end.
 *
 * Everything bound lives here, in the order it has to be evaluated:
 *
 * 1. **Offline, up front.** A request issued while the browser says there is no
 *    network fails fast with `unreachable` instead of waiting out a timeout
 *    that will certainly expire. The queued alternative is
 *    `runtime.enqueueWhenOffline`, and the choice between them belongs to the
 *    call site — this function only refuses, it does not silently queue.
 * 2. **A timeout, always.** Composed with the caller's signal rather than
 *    replacing it, so a cancel button still cancels while a timeout is armed.
 * 3. **A size cap, on the body.**
 * 4. **A code, from the payload or the transport.**
 *
 * @param {string} path              the request path, always same-origin
 * @param {object} [options]
 * @param {string} [options.method]
 * @param {any}    [options.body]
 * @param {object} [options.headers]
 * @param {AbortSignal} [options.signal]
 * @param {number} [options.timeoutMs]  defaults by operation class
 * @param {number} [options.maxBytes]
 * @param {string} [options.operation]   what the reader was doing
 * @param {string} [options.fallback]    message for a code-less failure
 * @param {boolean}[options.raw]         return the `Response` untouched
 */
async function apiRequest(path, options = {}) {
    const operation = options.operation || path;
    const method = String(options.method || 'GET').toUpperCase();

    if (!isOnline()) {
        throw new ApiError('当前处于离线状态。', { code: 'client_offline', operation });
    }

    const timeoutMs = Number(options.timeoutMs) > 0
        ? Number(options.timeoutMs)
        : defaultTimeoutFor(method, options);
    const maxBytes = Number(options.maxBytes) > 0
        ? Number(options.maxBytes)
        : RESPONSE_SIZE_CAPS_BYTES.json;

    // A caller-supplied signal and a timeout are two independent reasons to
    // stop, so they get two controllers joined by one: replacing the caller's
    // signal with our own would make the cancel button inert, and a single
    // controller for both would make "the request ended" ambiguous.
    const timeoutController = new AbortController();
    const timer = setTimeout(() => timeoutController.abort(), timeoutMs);
    const signals = [timeoutController.signal];
    if (options.signal) signals.push(options.signal);
    const signal = signals.length === 1 ? signals[0] : AbortAny(signals);

    let response;
    try {
        response = await fetch(path, {
            method,
            headers: buildHeaders(options),
            body: buildBody(options),
            signal,
            cache: options.cache || 'default'
        });
    } catch (error) {
        // Distinguishing the two abort causes is the whole point of the split
        // above: a timeout must not be reported as a user cancellation, or the
        // interface will claim a request was stopped when it was abandoned.
        if (timeoutController.signal.aborted && !isAbortError(options.signal?.reason)) {
            throw new ApiError('在等待响应的时间上限内没有收到答复。', {
                code: 'client_timeout',
                operation,
                cause: error
            });
        }
        throw transportFailure(error, operation);
    } finally {
        clearTimeout(timer);
    }

    if (options.raw) return response;

    const text = await readBodyCapped(response, maxBytes);
    const data = parseJsonBody(text, `${response.status} ${response.statusText || ''}`.trim());

    if (!response.ok) {
        throw new ApiError(data.error || options.fallback || '请求失败。', {
            code: data.code,
            status: response.status,
            operation,
            cause: data
        });
    }
    return data;
}

/**
 * The timeout class for a request, by what it is doing rather than by which
 * module asked. An analysis gets minutes because the server legitimately takes
 * minutes; a health probe gets ten seconds because the alternative to an answer
 * is not "wait longer".
 */
function defaultTimeoutFor(method, options) {
    if (options.stream) return REQUEST_TIMEOUTS_MS.analysis;
    if (options.generation) return REQUEST_TIMEOUTS_MS.generation;
    if (options.health) return REQUEST_TIMEOUTS_MS.health;
    if (options.importing) return REQUEST_TIMEOUTS_MS.import;
    if (method === 'GET' || method === 'HEAD') return REQUEST_TIMEOUTS_MS.default;
    return REQUEST_TIMEOUTS_MS.default;
}

function buildHeaders(options) {
    const headers = { Accept: 'application/json', ...(options.headers || {}) };
    // FormData must not get a Content-Type: the browser has to set it so the
    // multipart boundary matches the body it actually sends. Setting it by hand
    // is the classic way to produce a 400 that names the client as at fault.
    if (options.body !== undefined && !(options.body instanceof FormData) && !headers['Content-Type']) {
        headers['Content-Type'] = 'application/json';
    }
    return headers;
}

function buildBody(options) {
    if (options.body === undefined) return undefined;
    if (options.body instanceof FormData) return options.body;
    if (typeof options.body === 'string') return options.body;
    return JSON.stringify(options.body);
}

/**
 * A signal that aborts when any of its inputs does.
 *
 * `AbortSignal.any` exists in newer browsers only, and this project takes no
 * polyfill, so the combination is done by hand. Written as a returned object
 * with an `aborted` getter rather than a real `AbortSignal` because `fetch`
 * only reads `aborted` and adds a listener to it.
 */
function AbortAny(signals) {
    const controller = new AbortController();
    const abort = source => () => controller.abort(source?.reason);
    signals.forEach(signal => {
        if (signal.aborted) {
            controller.abort(signal.reason);
            return;
        }
        signal.addEventListener('abort', abort(signal), { once: true });
    });
    return controller.signal;
}

// --------------------------------------------------------------------------
// Retrying
// --------------------------------------------------------------------------

/**
 * Retry a request that failed for a reason retrying can address.
 *
 * Three rules, and the third is the one that matters:
 *
 * 1. Only a `RETRYABLE` kind is retried. A terminal failure is re-attempted
 *    zero times, because the second attempt of an identical request is
 *    guaranteed to be an identical refusal.
 * 2. Only safe methods are retried *silently*. A `POST` that starts work is not
 *    re-sent on the client's initiative; it waits for the reader to press
 *    "重试", which is a decision rather than an assumption.
 * 3. The ladder is finite. A request that has failed four times is not going to
 *    succeed on the fifth, and a client that retries forever is
 *    indistinguishable from one that has hung — which defeats the timeout
 *    bounds above.
 */
async function withRetry(path, options = {}, attempt = 0) {
    try {
        return await apiRequest(path, options);
    } catch (error) {
        const normalized = error instanceof ApiError ? error : transportFailure(error, options.operation);
        const method = String(options.method || 'GET').toUpperCase();
        const canRetrySilently = AUTO_RETRY_METHODS.includes(method);
        if (!normalized.retryable || !canRetrySilently || attempt >= MAX_AUTOMATIC_RETRIES) {
            throw normalized;
        }
        await delay(RETRY_BACKOFF_MS[Math.min(attempt, RETRY_BACKOFF_MS.length - 1)]);
        return withRetry(path, options, attempt + 1);
    }
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// --------------------------------------------------------------------------
// Streaming, with a stall watchdog
// --------------------------------------------------------------------------

/**
 * Read an SSE response, and know whether it died or is merely quiet.
 *
 * This is the function that answers the question the product's whole claim rests
 * on. A stream that produces nothing for `STREAM_STALL_TIMEOUT_MS` is declared
 * stalled — a `RETRYABLE` failure with a re-attach offered — rather than left
 * running for ever. The alternative is a parse that hangs at "Analyze" with no
 * explanation, which the reader cannot distinguish from a working parse, and
 * which is therefore indistinguishable from a tool that has lost track of its
 * own work.
 *
 * The watchdog is reset by *bytes*, not by parsed events, so a stream that
 * sends keep-alive comments is correctly not treated as stalled while one that
 * has genuinely gone quiet is.
 *
 * @param {Response} response
 * @param {object} handlers  event name (or `message`) to callback
 * @param {object} [options]
 * @param {string} [options.operation]
 * @param {AbortSignal} [options.signal]
 * @param {number} [options.stallTimeoutMs]
 * @param {(event: object) => void} [options.onEvent]  called for every event,
 *        used to track the SSE `id:` for a resumable re-attach
 */
async function readSseStream(response, handlers = {}, options = {}) {
    const operation = options.operation || 'stream';
    if (!response.ok) {
        const text = await readBodyCapped(response, RESPONSE_SIZE_CAPS_BYTES.json);
        const data = parseJsonBody(text, `${response.status} ${response.statusText || ''}`.trim());
        throw new ApiError(data.error || '连接失败。', {
            code: data.code,
            status: response.status,
            operation,
            cause: data
        });
    }

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('text/event-stream')) {
        const text = await readBodyCapped(response, RESPONSE_SIZE_CAPS_BYTES.json);
        const data = parseJsonBody(text, `${response.status} ${response.statusText || ''}`.trim());
        throw new ApiError('服务端没有返回流式响应。', {
            code: 'client_malformed_response',
            status: response.status,
            operation,
            cause: data
        });
    }

    const reader = response.body?.getReader?.();
    if (!reader) {
        throw new ApiError('这个浏览器不支持流式响应。', {
            code: 'client_malformed_response',
            operation
        });
    }

    const stallTimeoutMs = Number(options.stallTimeoutMs) > 0
        ? Number(options.stallTimeoutMs)
        : STREAM_STALL_TIMEOUT_MS;
    const decoder = new TextDecoder();
    let buffer = '';
    let totalBytes = 0;
    let lastEventId = '';

    // The watchdog: a deadline that is pushed forward by every byte and that
    // throws rather than merely resolving, so a stall unwinds the same `catch`
    // every other failure unwinds into and the caller's `finally` still runs.
    // A resolved-promise version would need its own error path, and two error
    // paths is how a spinner gets left up.
    let stallReject = null;
    let stallTimer = null;
    const armStall = () => {
        if (stallTimer) clearTimeout(stallTimer);
        if (!stallReject) return;
        stallTimer = setTimeout(() => {
            if (stallReject) {
                const reject = stallReject;
                stallReject = null;
                reject(new ApiError('数据流长时间没有任何进展。', {
                    code: 'client_stream_interrupted',
                    operation
                }));
            }
        }, stallTimeoutMs);
    };
    const stallPromise = new Promise((_resolve, reject) => {
        stallReject = reject;
    });
    armStall();

    const onAbort = () => {
        if (stallReject) {
            const reject = stallReject;
            stallReject = null;
            reject(new ApiError('请求已取消。', { code: 'client_cancelled', operation }));
        }
    };
    if (options.signal) {
        if (options.signal.aborted) onAbort();
        else options.signal.addEventListener('abort', onAbort, { once: true });
    }

    const dispatch = block => {
        const lines = String(block || '').split('\n');
        let eventName = 'message';
        let eventId = '';
        const dataLines = [];
        lines.forEach(line => {
            if (!line || line.startsWith(':')) return;
            if (line.startsWith('event:')) {
                eventName = line.slice(6).trim() || 'message';
            } else if (line.startsWith('id:')) {
                // Tracked because it is the resume cursor: a stream that numbers
                // its events can be re-attached where it stopped, and one that
                // does not cannot. Recording it is what makes that decision
                // possible without re-reading the transport.
                eventId = line.slice(3).trim();
            } else if (line.startsWith('data:')) {
                dataLines.push(line.slice(5).trim());
            }
        });
        if (eventId) lastEventId = eventId;
        if (!dataLines.length) return;

        let payload;
        const raw = dataLines.join('\n');
        try {
            payload = JSON.parse(raw);
        } catch {
            // A single unparseable event is a degradation, not a crash: the rest
            // of the stream is still worth reading, and discarding the whole
            // response over one bad frame would be a worse outcome than
            // rendering the frames that did parse.
            payload = { raw, code: 'client_malformed_response' };
        }
        if (!payload || typeof payload !== 'object') payload = { raw, code: 'client_malformed_response' };

        if (typeof options.onEvent === 'function') options.onEvent({ name: eventName, payload, id: eventId });
        const handler = handlers[eventName] || handlers.message;
        if (handler) handler(payload);
    };

    const readLoop = (async () => {
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            armStall();
            totalBytes += value?.byteLength || 0;
            if (totalBytes > RESPONSE_SIZE_CAPS_BYTES.streamTotal) {
                throw new ApiError('流式响应超出大小上限。', {
                    code: 'client_response_too_large',
                    operation
                });
            }

            buffer += decoder.decode(value || new Uint8Array(), { stream: true });
            const normalized = buffer.replace(/\r\n/g, '\n');
            const blocks = normalized.split('\n\n');
            buffer = blocks.pop() || '';
            blocks.forEach(dispatch);
        }
        if (buffer.trim()) dispatch(buffer);
    })();

    try {
        // `Promise.race`, not `await`, so the watchdog can end the read while it
        // is still blocked: the reader is abandoned via the `finally` below and
        // the caller's `catch` runs with a classified error.
        await Promise.race([readLoop, stallPromise]);
    } catch (error) {
        if (isAbortError(error)) {
            throw new ApiError('请求已取消。', { code: 'client_cancelled', operation, cause: error });
        }
        if (error instanceof ApiError) throw error;
        // A stream that dies mid-body is `client_stream_interrupted` and is
        // retryable, not terminal: the work may well have completed on the
        // server. The re-attach below is what turns that into a resumed
        // operation rather than a restart.
        throw new ApiError('数据流在传输途中断开。', {
            code: 'client_stream_interrupted',
            operation,
            cause: error
        });
    } finally {
        if (stallTimer) clearTimeout(stallTimer);
        stallReject = null;
        options.signal?.removeEventListener?.('abort', onAbort);
        try {
            reader.cancel();
        } catch {
            // Cancelling an already-errored reader throws; the lock release
            // below is what matters and must not be skipped for it.
        }
        reader.releaseLock();
    }

    return { lastEventId };
}

// --------------------------------------------------------------------------
// Re-attaching to work that is already running
// --------------------------------------------------------------------------

const JOB_STREAM_PATH = '/v1/jobs/{jobId}/stream';
const JOB_PATH = '/v1/jobs/{jobId}';

/**
 * Follow a job's state feed, and resume it rather than restart it.
 *
 * The backend has a job and subscription model for exactly this case
 * (`/v1/jobs/{id}/stream`, resumable via `Last-Event-ID`), and using it is the
 * difference between "your analysis died, start over" and "your analysis is
 * still running — here is where it got to". A two-minute parse that loses its
 * connection is the most expensive thing this interface can get wrong, and
 * restarting it charges the reader twice for the same document.
 *
 * `sinceSequence` is sent as `Last-Event-ID`, which is the header the server
 * reads first, and is also tracked locally so a caller that reconnects
 * mid-stream can resume from the last event it actually saw rather than from
 * zero.
 *
 * @param {string} jobId
 * @param {object} handlers
 * @param {object} [options]
 * @param {number} [options.sinceSequence]
 * @param {number} [options.resumeAttempts]
 * @param {AbortSignal} [options.signal]
 * @param {(state: object) => void} [options.onSnapshot]
 * @param {(error: ApiError) => boolean} [options.onStalled]  return true to stop
 */
async function attachToJobStream(jobId, handlers = {}, options = {}) {
    const operation = options.operation || 'job';
    if (!jobId) {
        throw new ApiError('没有可以重新连接的任务 id。', {
            code: 'client_unknown_error',
            operation
        });
    }

    const maxAttempts = Number(options.resumeAttempts) > 0
        ? Number(options.resumeAttempts)
        : RESYNC_BACKOFF_MS.length;
    let since = Number(options.sinceSequence) || 0;

    for (let attempt = 0; attempt <= maxAttempts; attempt += 1) {
        try {
            const response = await apiRequest(buildJobStreamPath(jobId), {
                method: 'GET',
                stream: true,
                timeoutMs: RESYNC_TIMEOUT_MS,
                operation,
                raw: true,
                headers: since > 0 ? { 'Last-Event-ID': String(since) } : {},
                signal: options.signal
            });
            const result = await readSseStream(response, {
                snapshot: payload => {
                    since = Number(payload?.sequence) || since;
                    if (typeof options.onSnapshot === 'function') options.onSnapshot(payload);
                },
                ...handlers
            }, {
                operation,
                signal: options.signal,
                onEvent: event => {
                    const sequence = Number(event?.id) || Number(event?.payload?.sequence) || 0;
                    if (sequence > since) since = sequence;
                }
            });
            if (result?.lastEventId) since = Number(result.lastEventId) || since;
            return { resumed: attempt > 0, sequence: since };
        } catch (error) {
            const normalized = error instanceof ApiError ? error : transportFailure(error, operation);
            const canResume = normalized.code === 'client_stream_interrupted'
                || normalized.code === 'client_timeout';
            const readerStopped = typeof options.onStalled === 'function' && options.onStalled(normalized);
            if (!canResume || readerStopped || attempt >= maxAttempts) throw normalized;
            await delay(RESYNC_BACKOFF_MS[Math.min(attempt, RESYNC_BACKOFF_MS.length - 1)]);
        }
    }

    throw new ApiError('无法重新连接到正在运行的任务。', {
        code: 'client_stream_interrupted',
        operation
    });
}

function buildJobStreamPath(jobId) {
    return JOB_STREAM_PATH.replace('{jobId}', encodeURIComponent(String(jobId)));
}

/** One job's current state, and its result summary once it is done. */
async function fetchJobState(jobId, options = {}) {
    const path = JOB_PATH.replace('{jobId}', encodeURIComponent(String(jobId)));
    return apiRequest(path, { ...options, operation: options.operation || 'job' });
}

// --------------------------------------------------------------------------
// A guarded action: in flight state that always settles
// --------------------------------------------------------------------------

/**
 * Run a request with its in-flight state opened and — guaranteed — closed.
 *
 * Every spinner in this interface is opened and closed here rather than in the
 * calling module, because the specific defect this was written to remove is a
 * spinner that outlives a dead request, and the only reliable way to prevent it
 * is for the opening and the closing to be the same call. A caller that forgot
 * the `finally` would be a bug no amount of discipline at the call site could
 * have caught.
 *
 * @param {object} options
 * @param {string} options.operation
 * @param {() => Promise<any>} options.run
 * @param {(error: ApiError) => void} [options.onFailure]  render the failure
 * @param {boolean} [options.showPendingNotice]
 */
async function withGuardedRequest(options = {}) {
    const flight = beginFlight(options.operation, {
        progressKnown: false,
        label: options.label || '',
        onStall: options.onStall || null,
        stallAfterMs: options.stallAfterMs || 0
    });
    let failure = null;
    try {
        return await options.run();
    } catch (error) {
        failure = error instanceof ApiError ? error : transportFailure(error, options.operation);
        throw failure;
    } finally {
        // The `finally`, not the branches. A spinner that outlives a dead request
        // is the specific defect this whole layer exists to remove, and the only
        // reliable way to prevent it is for "close the flight" to be on the same
        // path as "open it" — which `finally` is, for success, failure and an
        // exception thrown from inside the failure handler alike.
        if (failure && failure.code !== 'client_cancelled') {
            // A cancellation is the reader's own doing and needs no notice: they
            // already know, because they did it. Reporting it would be noise on
            // top of an action they just took.
            endFlight(flight, { failed: true, failure, operation: options.operation });
            if (typeof options.onFailure === 'function') options.onFailure(failure);
        } else {
            // A success also retires the notices this operation previously raised:
            // a stale "this will fail" beside a fresh success is a contradiction,
            // and it teaches the reader the interface does not know what happened.
            endFlight(flight, { succeeded: true, operation: options.operation });
        }
    }
}

// --------------------------------------------------------------------------
// Compatibility adapters
// --------------------------------------------------------------------------

/**
 * The previous error shape, as an `ApiError`.
 *
 * Kept because several modules construct their errors with it, and because it
 * is the one place that turns a raw payload into something with a `kind`. A
 * module that has not been migrated to `apiRequest` still gets the taxonomy.
 */
function buildApiError(data, fallbackMessage) {
    const payload = data || {};
    return new ApiError(payload.error || fallbackMessage || '请求失败。', {
        code: payload.code,
        status: payload.status,
        cause: payload
    });
}

/**
 * Tolerant JSON parsing, now size-capped and never throwing.
 *
 * Same contract as before — returns an object either way — so an unmigrated
 * call site keeps working, and it now additionally cannot be made to allocate
 * without limit by a response that lies about its size.
 */
async function parseJsonSafely(response) {
    try {
        const text = await readBodyCapped(response, RESPONSE_SIZE_CAPS_BYTES.json);
        if (!text.trim()) return { error: '服务端返回了空响应。', code: 'client_malformed_response' };
        return parseJsonBody(text, `${response.status} ${response.statusText || ''}`.trim());
    } catch (error) {
        // A cap breach while reading an *error* response must not turn into a
        // thrown error: the caller is already in a failure path and needs a
        // payload to report, not another exception.
        if (error instanceof ApiError) {
            return { error: error.message, code: error.code };
        }
        return { error: '响应无法解析。', code: 'client_malformed_response' };
    }
}


export {
    ApiError,
    attachToJobStream,
    apiRequest,
    buildApiError,
    fetchJobState,
    isAbortError,
    parseJsonSafely,
    readSseStream,
    transportFailure,
    withGuardedRequest,
    withRetry
};
