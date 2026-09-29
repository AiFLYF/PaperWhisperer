/**
 * HTTP plumbing: tolerant JSON parsing, structured errors and a
 * dependency-free SSE reader built on `fetch` streaming.
 */

function buildApiError(data, fallbackMessage) {
    const error = new Error(data.error || fallbackMessage);
    if (data.code) error.code = data.code;
    if (data.timestamp) error.timestamp = data.timestamp;
    return error;
}

async function parseJsonSafely(response) {
    const contentType = response.headers.get('content-type') || '';
    const statusText = `${response.status} ${response.statusText || ''}`.trim();
    if (contentType.includes('application/json')) {
        try {
            const data = await response.json();
            return data && typeof data === 'object' ? data : { error: `Unexpected JSON response${statusText ? ` (${statusText})` : ''}.` };
        } catch (error) {
            console.warn('Response json parse failed:', error);
            return { error: `Response returned malformed JSON${statusText ? ` (${statusText})` : ''}.` };
        }
    }

    const text = (await response.text()).trim();
    if (!text) {
        return { error: `Request failed${statusText ? ` (${statusText})` : ''}.` };
    }
    if (/^(<!doctype html|<html|<body)/i.test(text)) {
        return { error: 'Server returned an HTML page instead of an API response. Check the service endpoint or server logs.' };
    }
    return { error: text.length > 500 ? `${text.slice(0, 500)}...` : text };
}

async function readSseStream(response, handlers = {}) {
    if (!response.ok) {
        const data = await parseJsonSafely(response);
        throw buildApiError(data, 'Connection failed.');
    }

    const contentType = response.headers.get('content-type') || '';
    if (!contentType.includes('text/event-stream')) {
        const data = await parseJsonSafely(response);
        throw buildApiError(data, 'Streaming response was not returned.');
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

    try {
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
    } finally {
        reader.releaseLock();
    }
}


export { buildApiError };
export { parseJsonSafely };
export { readSseStream };
