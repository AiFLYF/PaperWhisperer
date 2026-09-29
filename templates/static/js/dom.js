import { state } from './state.js';

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

function updateAnalysisProgress(step, text) {
    const loadingText = document.getElementById('loadingText');
    if (loadingText && text) loadingText.textContent = text;

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
}

function toggleAIList() {
    const list = document.getElementById('aiList');
    const button = document.getElementById('aiToggleBtn');
    if (!list || !button) return;
    const shouldShow = list.hidden;
    list.hidden = !shouldShow;
    list.classList.toggle('show', shouldShow);
    button.setAttribute('aria-expanded', String(shouldShow));
    button.setAttribute('aria-label', shouldShow ? 'Hide AI collaborators' : 'Show AI collaborators');
}

function showError(message) {
    const errorEl = document.getElementById('error');
    const text = String(message || 'Request failed.').trim();
    const tips = [];

    if (/API Key|认证失败|401/i.test(text)) {
        tips.push('Tip: check whether the API key is missing, invalid, or expired.');
    }
    if (/OPENAI_BASE_URL|网页内容|HTML 页面|网页地址/i.test(text)) {
        tips.push('Tip: make sure OPENAI_BASE_URL points to the API endpoint, not a web page.');
    }
    if (/429|限流/i.test(text)) {
        tips.push('Tip: slow down requests or reduce concurrency and try again later.');
    }

    errorEl.textContent = tips.length ? `${text}\n\n${tips.join('\n')}` : text;
    errorEl.classList.remove('is-hidden');
    errorEl.focus({ preventScroll: true });
    scrollWindowTo({ top: 0 });
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
        btnElement.textContent = 'No content';
        updateStatus('Nothing to copy yet', 'idle');
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
        btnElement.textContent = 'Copied';
        btnElement.classList.add('action-success');
        updateStatus('Content copied', 'success');
    } catch (error) {
        console.warn('Copy failed:', error);
        btnElement.textContent = 'Copy failed';
        updateStatus('Copy failed', 'error');
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
export { copyText };
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
export { updateStatus };
