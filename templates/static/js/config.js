/** Shared, immutable configuration for the front end. */

export const THEME_STORAGE_KEY = 'paperwhisperer-theme';

export const SUPPORTED_UPLOAD_EXTENSIONS = ['.txt', '.pdf', '.docx', '.pptx'];

export const MAX_UPLOAD_BYTES = 16 * 1024 * 1024;

export const SECTION_EMPTY_TEXT = {
    summary: 'Summary will appear here after analysis.',
    quotes: 'Key citations will appear here after analysis.',
    mindmap: 'Text structure will appear here after analysis.',
    evaluation: 'Critical evaluation will appear here when enabled.',
    research_brief: 'Deep research brief will appear here when enabled.'
};

export const ANSWER_MODE_LABELS = {
    evidence: 'Evidence',
    explain: 'Explain',
    critique: 'Critique',
    reproduce: 'Reproduce'
};

export const ANSWER_MODE_DETAILS = {
    evidence: 'Evidence mode answers first, then cites document support and uncertainty.',
    explain: 'Explain mode teaches concepts, methods, and formulas step by step.',
    critique: 'Critique mode reviews strengths, assumptions, limitations, and threats to validity.',
    reproduce: 'Reproduce mode turns the paper into steps, variables, dependencies, and risks.'
};

export const SVG_NS = 'http://www.w3.org/2000/svg';

export const THEME_ICONS = {
    sun: [
        ['circle', { cx: '12', cy: '12', r: '5' }],
        ['path', { d: 'M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42' }]
    ],
    moon: [
        ['path', { d: 'M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z' }]
    ]
};
