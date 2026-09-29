import { state } from './state.js';
import { getCurrentTheme } from './themeState.js';
import { sanitizeRenderedSvg } from './sanitize.js';
import { setOptionalCardVisible } from './dom.js';

const MERMAID_MODULE_URL = 'https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.esm.min.mjs';

/**
 * Mermaid loading, rendering and pan/zoom. Rendering runs with
 * `securityLevel: 'strict'` and the produced SVG is sanitized again before it
 * reaches the DOM, so a hostile diagram cannot inject script or handlers.
 */

const MERMAID_OPTIONS = {
    startOnLoad: false,
    // Diagram source is model output: 'strict' keeps Mermaid from emitting raw
    // HTML labels and from wiring up click handlers inside the SVG.
    securityLevel: 'strict',
    fontFamily: 'inherit'
};

function loadMermaidRenderer() {
    if (state.mermaidInstance) return Promise.resolve(state.mermaidInstance);
    if (!state.mermaidReadyPromise) {
        state.mermaidReadyPromise = import(MERMAID_MODULE_URL)
            .then(module => {
                state.mermaidInstance = module.default;
                state.mermaidInstance.initialize({
                    ...MERMAID_OPTIONS,
                    theme: getCurrentTheme() === 'dark' ? 'dark' : 'base'
                });
                return state.mermaidInstance;
            })
            .catch(error => {
                console.error('Mermaid load failed:', error);
                state.mermaidReadyPromise = null;
                return null;
            });
    }
    return state.mermaidReadyPromise;
}

function normalizeMermaidSource(source) {
    return String(source || '').replace(/```mermaid\n?/gi, '').replace(/```\n?/g, '').trim();
}

async function renderMermaidDiagram(source) {
    const mermaidDiv = document.getElementById('mermaidChart');
    const cleanSource = normalizeMermaidSource(source);

    if (!cleanSource) {
        resetMermaidCard();
        return;
    }

    const instance = await loadMermaidRenderer();
    setMermaidCardVisible(true);
    mermaidDiv.replaceChildren();
    state.currentMermaidSource = cleanSource;

    if (!instance) {
        const fallback = document.createElement('p');
        fallback.className = 'empty-state mermaid-fallback-message';
        fallback.textContent = 'Mermaid failed to load.';
        mermaidDiv.appendChild(fallback);
        return;
    }

    try {
        instance.initialize({
            ...MERMAID_OPTIONS,
            theme: getCurrentTheme() === 'dark' ? 'dark' : 'base'
        });
        const id = `mermaid-${Date.now()}`;
        const { svg } = await instance.render(id, cleanSource);
        // Second line of defence: even with 'strict' the SVG is re-checked
        // against an allowlist before it is inserted.
        mermaidDiv.innerHTML = sanitizeRenderedSvg(svg);

        const svgElement = mermaidDiv.querySelector('svg');
        if (!svgElement) return;
        svgElement.classList.add('mermaid-svg-fit');

        resetPanZoom();
        if (!window.svgPanZoom) return;
        state.panZoomInstance = svgPanZoom(svgElement, {
            zoomEnabled: true,
            controlIconsEnabled: false,
            fit: true,
            center: true,
            minZoom: 0.5,
            maxZoom: 15
        });
    } catch (error) {
        console.error('Mermaid render failed:', error);
        const message = document.createElement('p');
        message.className = 'mermaid-error-message';
        message.textContent = 'Structure too complex to render. Raw data fallback:';
        const fallback = document.createElement('pre');
        fallback.className = 'mermaid-raw-fallback';
        fallback.textContent = cleanSource;
        mermaidDiv.replaceChildren(message, fallback);
    }
}

function resetPanZoom() {
    if (state.panZoomInstance) {
        state.panZoomInstance.destroy();
        state.panZoomInstance = null;
    }
}

function setMermaidCardVisible(visible) {
    setOptionalCardVisible('mermaidCard', visible);
}

function resetMermaidCard() {
    resetPanZoom();
    state.currentMermaidSource = '';
    setMermaidCardVisible(false);
    document.getElementById('mermaidChart').replaceChildren();
}

function zoomIn() { if (state.panZoomInstance) state.panZoomInstance.zoomIn(); }

function zoomOut() { if (state.panZoomInstance) state.panZoomInstance.zoomOut(); }

function zoomReset() { if (state.panZoomInstance) { state.panZoomInstance.resetZoom(); state.panZoomInstance.center(); } }

function getCurrentSvgSource() {
    const svgElement = document.querySelector('#mermaidChart svg');
    if (!svgElement) return '';
    const serializer = new XMLSerializer();
    let source = serializer.serializeToString(svgElement);
    if (!source.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) source = source.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
    if (!source.match(/^<svg[^>]+"http\:\/\/www\.w3\.org\/1999\/xlink"/)) source = source.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
    return '<?xml version="1.0" standalone="no"?>\r\n' + source;
}


export { getCurrentSvgSource };
export { normalizeMermaidSource };
export { renderMermaidDiagram };
export { resetMermaidCard };
export { resetPanZoom };
export { setMermaidCardVisible };
export { zoomIn };
export { zoomOut };
export { zoomReset };
