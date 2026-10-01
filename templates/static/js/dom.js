import { state } from './state.js';
import { PROGRESS_STORY } from './config.js';
import { formatElapsed, notifyFailure } from './runtime.js';
import { t } from './i18n.js';

/**
 * Low-level DOM helpers: control state, scrolling, focus management, status
 * reporting, error banners and clipboard feedback.
 */

function bindClick(id, handler) {
    const element = document.getElementById(id);
    if (element) element.addEventListener('click', handler);
}

function setControlDisabled(element, disabled) {
    if (!element) return;
    element.disabled = disabled;
    element.setAttribute('aria-disabled', String(disabled));
}

function setCancelVisible(id, isVisible) {
    const button = document.getElementById(id);
    if (!button) return;
    button.hidden = !isVisible;
    setControlDisabled(button, !isVisible);
}

function scheduleButtonFeedbackReset(button, callback, delay = 1600) {
    if (!button) return;
    const existingTimer = state.buttonFeedbackTimers.get(button);
    if (existingTimer) {
        clearTimeout(existingTimer);
    }
    const timer = setTimeout(() => {
        state.buttonFeedbackTimers.delete(button);
        callback();
    }, delay);
    state.buttonFeedbackTimers.set(button, timer);
}

function getMotionSafeScrollBehavior() {
    return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
}

function scrollElementIntoView(element, options = {}) {
    if (!element) return;
    element.scrollIntoView({ behavior: getMotionSafeScrollBehavior(), block: 'start', ...options });
}

function scrollWindowTo(options) {
    window.scrollTo({ behavior: getMotionSafeScrollBehavior(), ...options });
}

function focusAnalysisWorkspace({ scroll = true } = {}) {
    const result = document.getElementById('result');
    if (!result || !result.classList.contains('active')) return;
    if (scroll) scrollElementIntoView(result);
    result.focus({ preventScroll: true });
}

function focusWorkspaceTarget(targetId, focusSelector) {
    const target = document.getElementById(targetId);
    if (!target) return;
    scrollElementIntoView(target);
    const focusTarget = focusSelector ? target.querySelector(focusSelector) : target;
    if (focusTarget) focusTarget.focus({ preventScroll: true });
}

function updateStatus(text, tone = 'idle') {
    const chip = document.getElementById('statusChip');
    if (!chip) return;
    chip.textContent = text;
    chip.dataset.tone = tone;
}

/**
 * Move the progress steps to `step`, and say how long this has taken.
 *
 * The step indicator is driven by what the server reported, so it cannot
 * disagree with the server. What it deliberately does *not* do is show a
 * percentage: the analysis stream reports which step it is on and never how far
 * through that step it is, so any fraction here would be invented. The elapsed
 * time is the honest progress measure, and `PROGRESS_STORY.unknownNote` says
 * out loud that no percentage is being shown and why.
 *
 * A bar that guesses is worse than one that admits ignorance: a reader who
 * watches a fabricated 60% sit still for two minutes stops trusting every
 * number this interface shows, including the ones that are real.
 */
function updateAnalysisProgress(step, text) {
    const loadingText = document.getElementById('loadingText');
    if (loadingText && text) {
        loadingText.textContent = text;
    }

    const order = ['upload', 'analyze', 'render', 'ready'];
    const activeIndex = Math.max(order.indexOf(step), 0);
    document.querySelectorAll('.progress-step').forEach(element => {
        const index = order.indexOf(element.dataset.step);
        const isActive = index === activeIndex;
        element.classList.toggle('done', index >= 0 && index < activeIndex);
        element.classList.toggle('active', isActive);
        if (isActive) {
            element.setAttribute('aria-current', 'step');
        } else {
            element.removeAttribute('aria-current');
        }
    });

    updateProgressElapsed();
}

/**
 * The one knowable measure of a long run: how long it has been going.
 *
 * Written into its own element rather than appended to the step text, so a
 * module that rewrites the step description does not erase the clock.
 */
function updateProgressElapsed() {
    const holder = document.getElementById('loading');
    if (!holder) return;
    let clock = document.getElementById('progressElapsed');
    if (!clock) {
        clock = document.createElement('div');
        clock.id = 'progressElapsed';
        clock.className = 'progress-elapsed';
        clock.setAttribute('role', 'status');
        clock.setAttribute('aria-live', 'off');
        holder.appendChild(clock);
    }
    const startedAt = Number(holder.dataset.startedAt);
    const base = Number.isFinite(startedAt) ? startedAt : Date.now();
    holder.dataset.startedAt = String(base);
    clock.textContent = `${PROGRESS_STORY.elapsedLabel}：${formatElapsed(Date.now() - base)}。${PROGRESS_STORY.unknownNote}`;
}

/** Start the elapsed clock for a run that is beginning. */
function beginProgressClock() {
    const holder = document.getElementById('loading');
    if (holder) holder.dataset.startedAt = String(Date.now());
    updateProgressElapsed();
    return holder?.dataset.startedAt || null;
}

/**
 * Stop the clock, and say plainly that no percentage was ever shown.
 *
 * Called from the same `finally` that takes the spinner down, so the elapsed
 * time is still on screen at the moment the run ends rather than vanishing with
 * the spinner.
 */
function endProgressClock(note) {
    const holder = document.getElementById('loading');
    if (!holder) return;
    const startedAt = Number(holder.dataset.startedAt);
    const clock = document.getElementById('progressElapsed');
    if (clock) {
        const base = Number.isFinite(startedAt) ? startedAt : Date.now();
        clock.textContent = `${PROGRESS_STORY.elapsedLabel}：${formatElapsed(Date.now() - base)}。${note || ''}`;
    }
    delete holder.dataset.startedAt;
}

function toggleAIList() {
    const list = document.getElementById('aiList');
    const button = document.getElementById('aiToggleBtn');
    if (!list || !button) return;
    const shouldShow = list.hidden;
    list.hidden = !shouldShow;
    list.classList.toggle('show', shouldShow);
    button.setAttribute('aria-expanded', String(shouldShow));
    button.setAttribute('aria-label', t(shouldShow ? 'footer.hide_ai' : 'footer.show_ai'));
}

function showError(message) {
    const errorEl = document.getElementById('error');
    if (!errorEl) return;

    // `message` may be a string (every existing call site) or an `ApiError`.
    // When it is a failure, the *code* decides the whole treatment: the tip, the
    // dismissal and the retry button all come from `runtime.js`, which is the
    // one place a failure is described. What is left here is the legacy path for
    // a bare string, which is a local validation message the client already
    // knows the cause of.
    const text = typeof message === 'string'
        ? message.trim()
        : String(message?.message || t('error.request_failed')).trim();

    errorEl.textContent = text;
    errorEl.classList.remove('is-hidden');
    errorEl.focus({ preventScroll: true });
    scrollWindowTo({ top: 0 });

    if (typeof message !== 'string' && message?.code) {
        notifyFailure(message, { onRetry: message.onRetry });
    }
}

function hideError() {
    const errorEl = document.getElementById('error');
    errorEl.textContent = '';
    errorEl.classList.add('is-hidden');
}

function setButtonLoading(button, loadingText, defaultText, isLoading) {
    if (!button) return;
    setControlDisabled(button, isLoading);
    button.textContent = isLoading ? loadingText : defaultText;
    button.classList.toggle('is-busy', isLoading);
    if (isLoading) {
        button.setAttribute('aria-busy', 'true');
    } else {
        button.removeAttribute('aria-busy');
    }
}

async function copyText(elementId, btnElement) {
    const content = document.getElementById(elementId);
    if (!content || !btnElement) return;

    const rawText = (content.dataset && typeof content.dataset.rawContent === 'string')
        ? content.dataset.rawContent
        : '';
    const text = (rawText || content.textContent || '').trim();
    const originalText = btnElement.textContent;
    if (!text) {
        btnElement.textContent = t('copy.no_content');
        updateStatus(t('copy.nothing_yet'), 'idle');
        scheduleButtonFeedbackReset(btnElement, () => { btnElement.textContent = originalText; });
        return;
    }

    setControlDisabled(btnElement, true);
    btnElement.setAttribute('aria-busy', 'true');
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
        } else {
            const textarea = document.createElement('textarea');
            try {
                textarea.value = text;
                textarea.setAttribute('readonly', '');
                textarea.className = 'clipboard-fallback-field';
                document.body.appendChild(textarea);
                textarea.focus();
                textarea.select();
                document.execCommand('copy');
            } finally {
                textarea.remove();
            }
        }
        btnElement.textContent = t('copy.done');
        btnElement.classList.add('action-success');
        updateStatus(t('copy.done_status'), 'success');
    } catch (error) {
        console.warn('Copy failed:', error);
        btnElement.textContent = t('copy.failed');
        updateStatus(t('copy.failed_status'), 'error');
    }
    scheduleButtonFeedbackReset(btnElement, () => {
        btnElement.textContent = originalText;
        setControlDisabled(btnElement, false);
        btnElement.removeAttribute('aria-busy');
        btnElement.classList.remove('action-success');
    });
}

function setOptionalCardVisible(cardId, visible) {
    const card = document.getElementById(cardId);
    if (!card) return;
    card.classList.toggle('is-hidden', !visible);
    card.setAttribute('aria-hidden', String(!visible));
}


export { bindClick };
export { beginProgressClock };
export { copyText };
export { endProgressClock };
export { focusAnalysisWorkspace };
export { focusWorkspaceTarget };
export { hideError };
export { scheduleButtonFeedbackReset };
export { scrollElementIntoView };
export { scrollWindowTo };
export { setButtonLoading };
export { setCancelVisible };
export { setControlDisabled };
export { setOptionalCardVisible };
export { showError };
export { toggleAIList };
export { updateAnalysisProgress };
export { updateProgressElapsed };
export { updateStatus };
