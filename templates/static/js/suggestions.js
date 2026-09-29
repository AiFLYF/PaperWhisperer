import { ANSWER_MODE_DETAILS, ANSWER_MODE_LABELS } from './config.js';
import { state } from './state.js';
import { updateStatus } from './dom.js';

/**
 * Follow-up affordances: answer-mode selection, starter questions and the
 * suggested-question / next-action chips.
 */

function normalizeSmartTextItems(values, maxItems = 6) {
    if (!Array.isArray(values)) return [];
    const seen = new Set();
    const items = [];
    values.forEach(value => {
        const text = String(value || '').trim();
        if (!text || seen.has(text)) return;
        items.push(text);
        seen.add(text);
    });
    return items.slice(0, maxItems);
}

function normalizeSmartActions(values, maxItems = 5) {
    if (!Array.isArray(values)) return [];
    const seen = new Set();
    const actions = [];
    values.forEach(value => {
        if (!value || typeof value !== 'object') return;
        const label = String(value.label || '').trim();
        const prompt = String(value.prompt || '').trim();
        if (!label || !prompt || seen.has(prompt)) return;
        actions.push({ label, prompt });
        seen.add(prompt);
    });
    return actions.slice(0, maxItems);
}

function setAnswerMode(mode) {
    state.currentAnswerMode = ANSWER_MODE_LABELS[mode] ? mode : 'evidence';
    document.querySelectorAll('.mode-chip').forEach(button => {
        const selected = button.dataset.mode === state.currentAnswerMode;
        button.classList.toggle('active', selected);
        button.setAttribute('aria-checked', selected ? 'true' : 'false');
        button.setAttribute('aria-pressed', selected ? 'true' : 'false');
        button.tabIndex = selected ? 0 : -1;
    });
    const hint = document.getElementById('answerModeHint');
    if (hint) hint.textContent = ANSWER_MODE_DETAILS[state.currentAnswerMode];
}

function handleAnswerModeKeydown(event) {
    const navigationKeys = ['ArrowLeft', 'ArrowUp', 'ArrowRight', 'ArrowDown', 'Home', 'End'];
    if (!navigationKeys.includes(event.key)) return;

    const chips = Array.from(document.querySelectorAll('.mode-chip'));
    if (!chips.length) return;
    event.preventDefault();

    const currentIndex = Math.max(0, chips.indexOf(event.currentTarget));
    let nextIndex = currentIndex;
    if (event.key === 'Home') {
        nextIndex = 0;
    } else if (event.key === 'End') {
        nextIndex = chips.length - 1;
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
        nextIndex = (currentIndex - 1 + chips.length) % chips.length;
    } else {
        nextIndex = (currentIndex + 1) % chips.length;
    }

    const nextChip = chips[nextIndex];
    setAnswerMode(nextChip.dataset.mode);
    nextChip.focus({ preventScroll: true });
}

function fillQuestionInput(prompt) {
    const questionInput = document.getElementById('questionInput');
    if (!questionInput) return;
    questionInput.value = prompt;
    questionInput.focus();
}

function useStarterQuestion(question) {
    if (!question) return;
    fillQuestionInput(question);
    updateStatus('Starter question loaded', 'idle');
}

function renderSmartPrompts(questions = state.currentSuggestedQuestions) {
    const container = document.getElementById('smartPrompts');
    if (!container) return;
    state.currentSuggestedQuestions = normalizeSmartTextItems(questions);
    container.replaceChildren();
    container.classList.toggle('is-hidden', !state.currentSuggestedQuestions.length);
    container.setAttribute('aria-hidden', String(!state.currentSuggestedQuestions.length));
    state.currentSuggestedQuestions.forEach(question => {
        const button = document.createElement('button');
        button.className = 'prompt-chip';
        button.type = 'button';
        button.textContent = question;
        button.addEventListener('click', () => fillQuestionInput(question));
        container.appendChild(button);
    });
}

function renderNextActions(actions = state.currentNextActions) {
    const container = document.getElementById('nextActions');
    if (!container) return;
    state.currentNextActions = normalizeSmartActions(actions);
    container.replaceChildren();
    container.classList.toggle('is-hidden', !state.currentNextActions.length);
    container.setAttribute('aria-hidden', String(!state.currentNextActions.length));
    state.currentNextActions.forEach(action => {
        const button = document.createElement('button');
        button.className = 'next-action-btn';
        button.type = 'button';
        button.textContent = action.label;
        button.title = action.prompt;
        button.addEventListener('click', () => fillQuestionInput(action.prompt));
        container.appendChild(button);
    });
}

function resetSmartSuggestions() {
    state.currentSuggestedQuestions = [];
    state.currentNextActions = [];
    renderSmartPrompts([]);
    renderNextActions([]);
}


export { fillQuestionInput };
export { handleAnswerModeKeydown };
export { normalizeSmartActions };
export { normalizeSmartTextItems };
export { renderNextActions };
export { renderSmartPrompts };
export { resetSmartSuggestions };
export { setAnswerMode };
export { useStarterQuestion };
