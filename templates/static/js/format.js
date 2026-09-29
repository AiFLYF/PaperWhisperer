import { escapeHtml, sanitizeGeneratedHtml } from './sanitize.js';

if (window.marked) {
    marked.setOptions({ breaks: true, gfm: true, headerIds: false, mangle: false });
}

/**
 * Markdown, prompt-tag and math rendering. `formatContent` returns sanitized
 * HTML, so callers may insert the result directly.
 */

function transformPromptTags(content) {
    return String(content).replace(/<(role|context|task|constraints|output_format|input|self_check)>\s*([\s\S]*?)\s*<\/\1>/gi, (_, tag, body) => {
        const safeTag = escapeHtml(tag.replace(/_/g, ' '));
        const safeBody = escapeHtml(body.trim());
        return `\n\n<div class="prompt-block"><div class="prompt-block-title">${safeTag}</div><div class="prompt-block-body">${safeBody}</div></div>\n\n`;
    });
}

function renderMath(element) {
    if (!window.renderMathInElement) return;
    try {
        renderMathInElement(element, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '\\[', right: '\\]', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false }
            ],
            throwOnError: false,
            output: 'html'
        });
    } catch (error) {
        console.warn('Math render failed:', error);
    }
}

function formatContent(content, fallbackText) {
    if (!content) {
        return `<p class="empty-state">${escapeHtml(fallbackText || 'No content available.')}</p>`;
    }

    let processedContent = String(content).replace(/\r\n/g, '\n');
    processedContent = transformPromptTags(processedContent);
    processedContent = processedContent.replace(/\\\[([\s\S]*?)\\\]/g, (_, expr) => `$$${expr}$$`);
    processedContent = processedContent.replace(/\\\(([\s\S]*?)\\\)/g, (_, expr) => `$${expr}$`);

    const mathTokens = {};
    let counter = 0;

    processedContent = processedContent.replace(/\$\$([\s\S]*?)\$\$/g, match => {
        const token = `@@MATHBLOCK${counter}@@`;
        mathTokens[token] = match;
        counter += 1;
        return `\n\n${token}\n\n`;
    });

    processedContent = processedContent.replace(/\$((?!\s)[^$]+?(?!\s))\$/g, match => {
        const token = `@@MATHINLINE${counter}@@`;
        mathTokens[token] = match;
        counter += 1;
        return token;
    });

    let htmlContent = window.marked ? marked.parse(processedContent) : `<p>${escapeHtml(processedContent).replace(/\n{2,}/g, '</p><p>').replace(/\n/g, '<br>')}</p>`;
    Object.entries(mathTokens).forEach(([token, mathStr]) => {
        const blockPattern = new RegExp(`<p>${token}</p>`, 'g');
        if (blockPattern.test(htmlContent)) {
            htmlContent = htmlContent.replace(blockPattern, mathStr);
        } else {
            htmlContent = htmlContent.replace(new RegExp(token, 'g'), mathStr);
        }
    });

    return sanitizeGeneratedHtml(htmlContent);
}

function replaceWithFormattedContent(element, content, fallbackText) {
    const template = document.createElement('template');
    template.innerHTML = formatContent(content, fallbackText);
    element.replaceChildren(template.content.cloneNode(true));
    renderMath(element);
}


export { formatContent };
export { replaceWithFormattedContent };
export { transformPromptTags };
