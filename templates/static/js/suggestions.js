import { ANSWER_MODE_DETAILS, ANSWER_MODE_LABELS } from './config.js';
import { state } from './state.js';
import { updateStatus } from './dom.js';
import { t } from './i18n.js';

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
    // Through the catalogue rather than `ANSWER_MODE_DETAILS`, which holds the
    // *English* locale's sentences. Writing it here overwrote the translated
    // `data-i18n="mode.detail.evidence"` text on the very first call —
    // `initialize()` calls this at startup — so a Chinese-default interface
    // showed one English line here from the moment it loaded, above a page that
    // was otherwise Chinese.
    if (hint) {
        const detail = t(`mode.detail.${state.currentAnswerMode}`);
        hint.textContent = detail === `mode.detail.${state.currentAnswerMode}`
            ? ANSWER_MODE_DETAILS[state.currentAnswerMode]
            : detail;
    }
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

/**
 * Put a prompt in the composer without destroying what is already there.
 *
 * Every chip on this page — starter questions, the server's suggested
 * questions, the next-action buttons — routes through here. It used to assign
 * unconditionally, so a reader who had typed half a question and then clicked a
 * chip to *compare* the two lost their draft with no warning and no undo. A
 * question is the most expensive field in this interface to retype, and losing
 * one to a click that was not asking to replace it is a bug, not a shortcut.
 *
 * So the draft wins. The chip's prompt is not shown instead, because there is
 * nowhere to show it without another surface and silently doing nothing is
 * worse than the status line naming what happened: the reader is told the chip
 * did not apply and why, and their own text is still in front of them.
 */
function fillQuestionInput(prompt) {
    const questionInput = document.getElementById('questionInput');
    if (!questionInput || !prompt) return;
    const draft = questionInput.value.trim();
    if (draft && draft !== prompt) {
        updateStatus('Your question was left as you typed it — clear it to use this suggestion', 'idle');
        questionInput.focus();
        return;
    }
    questionInput.value = prompt;
    questionInput.focus();
    updateStatus('Question loaded into the box — edit it or send it as is', 'idle');
}

function useStarterQuestion(question) {
    if (!question) return;
    // No status update here: `fillQuestionInput` already acknowledged the load,
    // or explained why it declined to. Reporting "Starter question loaded" on
    // top of that would overwrite the explanation with a statement that is
    // false — the starter question is precisely what did *not* get loaded.
    fillQuestionInput(question);
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
        // The prompt — the question this button will actually ask — was
        // reachable only through `title`, which is a hover affordance. A
        // keyboard or screen-reader user gets a name that is just the label
        // ("Review the methodology") and no way to find out what is about to be
        // asked before they press it, which is the whole content of the control.
        // This is the same argument `queue.js` makes for putting the
        // consequence in an `aria-label` rather than a `title`.
        button.setAttribute('aria-label', `${action.label} — asks: ${action.prompt}`);
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
