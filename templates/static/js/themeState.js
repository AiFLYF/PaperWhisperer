/**
 * The active theme is shared state rather than a DOM read, so the theme
 * controller and the Mermaid renderer can both consult it without importing
 * each other.
 */
const THEME_STORAGE_KEY = 'paperwhisperer-theme';

let currentTheme = 'light';

export function getCurrentTheme() {
    return currentTheme;
}

export function setCurrentTheme(theme) {
    currentTheme = theme === 'dark' ? 'dark' : 'light';
    return currentTheme;
}

export { THEME_STORAGE_KEY };
