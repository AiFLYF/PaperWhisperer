import { SVG_NS, THEME_ICONS } from './config.js';
import { THEME_STORAGE_KEY, getCurrentTheme, setCurrentTheme } from './themeState.js';
import { state } from './state.js';
import { renderMermaidDiagram } from './mermaid.js';

/** Theme persistence and the light/dark toggle. */

function readLocalStorageValue(key) {
    try {
        return localStorage.getItem(key);
    } catch (error) {
        console.warn('Local storage read failed:', error);
        return '';
    }
}

function writeLocalStorageValue(key, value) {
    try {
        localStorage.setItem(key, value);
    } catch (error) {
        console.warn('Local storage write failed:', error);
    }
}

function initializeTheme() {
    const savedTheme = readLocalStorageValue(THEME_STORAGE_KEY);
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(savedTheme || (prefersDark ? 'dark' : 'light'));
}

function createThemeIcon(name) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('width', '16');
    svg.setAttribute('height', '16');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    (THEME_ICONS[name] || THEME_ICONS.moon).forEach(([tagName, attributes]) => {
        const element = document.createElementNS(SVG_NS, tagName);
        Object.entries(attributes).forEach(([attribute, value]) => element.setAttribute(attribute, value));
        svg.appendChild(element);
    });
    return svg;
}

function applyTheme(theme) {
    const body = document.body;
    const btn = document.getElementById('themeBtn');
    const resolved = setCurrentTheme(theme);
    const isDark = resolved === 'dark';
    if (isDark) {
        body.setAttribute('data-theme', 'dark');
    } else {
        body.removeAttribute('data-theme');
    }
    if (!btn) return;
    btn.replaceChildren(createThemeIcon(isDark ? 'sun' : 'moon'));
    btn.setAttribute('aria-pressed', String(isDark));
    btn.setAttribute('aria-label', isDark ? 'Switch to light theme' : 'Switch to dark theme');
}

function toggleTheme() {
    const nextTheme = getCurrentTheme() === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
    writeLocalStorageValue(THEME_STORAGE_KEY, nextTheme);
    if (state.currentMermaidSource) {
        renderMermaidDiagram(state.currentMermaidSource);
    }
}


export { applyTheme };
export { initializeTheme };
export { toggleTheme };
