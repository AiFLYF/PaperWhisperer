import { state } from './state.js';
import { focusWorkspaceTarget, scrollWindowTo, updateStatus } from './dom.js';

/**
 * Accessibility affordances: the back-to-top control and the Alt+U / Alt+S /
 * Alt+Q workspace shortcuts.
 */

function handleWorkspaceShortcut(event) {
    if (!event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    const key = event.key.toLowerCase();
    if (!['u', 's', 'q'].includes(key)) return;
    event.preventDefault();
    if (key === 'u') {
        focusWorkspaceTarget('uploadWorkspace', '#dropZone');
        updateStatus('Upload shortcut focused document source', 'idle');
    } else if (key === 's') {
        focusWorkspaceTarget('paperSearchTitle', null);
        document.getElementById('paperSearchInput')?.focus({ preventScroll: true });
        updateStatus('Search shortcut focused paper search', 'idle');
    } else {
        focusWorkspaceTarget('askQuestionsTitle', null);
        document.getElementById('questionInput')?.focus({ preventScroll: true });
        updateStatus('Ask shortcut focused question input', 'idle');
    }
}

function updateBackToTopVisibility() {
    const button = document.getElementById('backToTopBtn');
    if (!button) return;
    const isVisible = window.scrollY > 640;
    button.classList.toggle('show', isVisible);
    button.setAttribute('aria-hidden', String(!isVisible));
    button.tabIndex = isVisible ? 0 : -1;
}

function scheduleBackToTopVisibilityUpdate() {
    if (state.backToTopFramePending) return;
    state.backToTopFramePending = true;
    requestAnimationFrame(() => {
        state.backToTopFramePending = false;
        updateBackToTopVisibility();
    });
}

function initializeBackToTop() {
    updateBackToTopVisibility();
    window.addEventListener('scroll', scheduleBackToTopVisibilityUpdate, { passive: true });
}

function scrollToTop() {
    scrollWindowTo({ top: 0 });
}


export { handleWorkspaceShortcut };
export { initializeBackToTop };
export { scrollToTop };
