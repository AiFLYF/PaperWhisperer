import { ANSWER_MODE_LABELS } from './config.js';
import { state } from './state.js';
import { buildApiError, readSseStream } from './api.js';
import { replaceWithFormattedContent } from './format.js';
import { copyText, hideError, scrollWindowTo, setButtonLoading, setCancelVisible, setControlDisabled, showError, updateStatus } from './dom.js';
import { fillQuestionInput } from './suggestions.js';
import { renderExportPreview } from './export.js';

/** The Q&A panel: streaming answers, cancellation and transcript storage. */

function clearChatHistory() {
    document.getElementById('chatHistory').replaceChildren();
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
    setControlDisabled(button, true);

    const content = document.createElement('div');
    content.id = answerId;
    content.className = 'answer-content content';
    content.dataset.rawContent = '';
    content.appendChild(createInlineSpinner());

    header.appendChild(title);
    header.appendChild(button);
    answerDiv.appendChild(header);
    answerDiv.appendChild(content);
    chatHistory.appendChild(answerDiv);
    return { answerDiv, content, button, answerId };
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
    if (!shell.button.dataset.bound) {
        shell.button.addEventListener('click', () => copyText(shell.answerId, shell.button));
        shell.button.dataset.bound = 'true';
    }
    if (!shell.actions) {
        const actions = document.createElement('div');
        actions.className = 'answer-actions';
        state.currentNextActions.slice(0, 3).forEach(action => {
            const button = document.createElement('button');
            button.className = 'mini-action-btn';
            button.type = 'button';
            button.textContent = action.label;
            button.title = action.prompt;
            button.addEventListener('click', () => fillQuestionInput(action.prompt));
            actions.appendChild(button);
        });
        if (actions.children.length) {
            shell.answerDiv.appendChild(actions);
            shell.actions = actions;
        }
    }
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

    hideError();
    appendChatText(`User · ${ANSWER_MODE_LABELS[state.currentAnswerMode]}`, question, 'chat-question');
    questionInput.value = '';
    setButtonLoading(askBtn, 'Processing...', 'Send', true);
    setCancelVisible('cancelAskBtn', true);
    updateStatus('Answering based on current document...', 'idle');
    const streamingShell = appendStreamingAnswerShell();
    state.activeStreamingAnswerShell = streamingShell;
    let accumulatedAnswer = '';

    try {
        const response = await fetch('/api/ask/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, session_id: state.currentSessionId, session_token: state.currentSessionToken, api_key: apiKey, answer_mode: state.currentAnswerMode }),
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
                accumulatedAnswer = payload.answer || accumulatedAnswer;
                finalizeStreamingAnswer(streamingShell, accumulatedAnswer);
                state.currentChatTurns.push({
                    question,
                    answer: accumulatedAnswer,
                    answer_mode: state.currentAnswerMode,
                    timestamp: new Date().toISOString()
                });
                renderExportPreview();
                state.activeStreamingAnswerShell = null;
                updateStatus('Answer ready', 'success');
            },
            error: payload => {
                throw new Error(payload.error || 'Question request failed.');
            }
        });
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        if (streamingShell && streamingShell.answerDiv && streamingShell.answerDiv.parentNode) {
            streamingShell.answerDiv.parentNode.removeChild(streamingShell.answerDiv);
        }
        appendChatText('System Error', error.message || 'Question request failed.', 'chat-answer chat-error');
        updateStatus('Question request failed', 'error');
    } finally {
        if (requestId === state.askRequestId) {
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
    setButtonLoading(document.getElementById('askBtn'), 'Processing...', 'Send', false);
    setCancelVisible('cancelAskBtn', false);
    state.askController = null;
    if (state.activeStreamingAnswerShell?.answerDiv?.parentNode) {
        state.activeStreamingAnswerShell.answerDiv.parentNode.removeChild(state.activeStreamingAnswerShell.answerDiv);
    }
    state.activeStreamingAnswerShell = null;
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
