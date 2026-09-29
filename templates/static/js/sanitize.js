/**
 * Untrusted-content handling. Model output is rendered as HTML, so every
 * path that touches it must go through one of these helpers first.
 */

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function sanitizeUrl(rawUrl) {
    if (!rawUrl) return '#';
    try {
        const parsed = new URL(rawUrl, window.location.origin);
        if (['http:', 'https:', 'mailto:'].includes(parsed.protocol)) {
            return parsed.href;
        }
    } catch (error) {
        console.warn('Unsafe url ignored:', rawUrl, error);
    }
    return '#';
}

function sanitizeImageUrl(rawUrl) {
    if (!rawUrl) return '#';
    try {
        const parsed = new URL(rawUrl, window.location.origin);
        if (['http:', 'https:'].includes(parsed.protocol)) {
            return parsed.href;
        }
    } catch (error) {
        console.warn('Unsafe image url ignored:', rawUrl, error);
    }
    return '#';
}

function sanitizeGeneratedHtml(html) {
    const template = document.createElement('template');
    template.innerHTML = html;
    const allowedTags = new Set(['A', 'B', 'BLOCKQUOTE', 'BR', 'CODE', 'DEL', 'DIV', 'EM', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'HR', 'I', 'IMG', 'LI', 'OL', 'P', 'PRE', 'S', 'SPAN', 'STRONG', 'TABLE', 'TBODY', 'TD', 'TH', 'THEAD', 'TR', 'UL']);
    const allowedAttributes = new Set(['alt', 'class', 'colspan', 'href', 'loading', 'rel', 'rowspan', 'src', 'target', 'title']);

    template.content.querySelectorAll('*').forEach(node => {
        if (!allowedTags.has(node.tagName)) {
            node.replaceWith(document.createTextNode(node.textContent || ''));
            return;
        }

        Array.from(node.attributes).forEach(attribute => {
            const attributeName = attribute.name.toLowerCase();
            if (!allowedAttributes.has(attributeName) || attributeName.startsWith('on') || attributeName === 'style') {
                node.removeAttribute(attribute.name);
            }
        });
    });

    template.content.querySelectorAll('a').forEach(anchor => {
        anchor.href = sanitizeUrl(anchor.getAttribute('href'));
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.referrerPolicy = 'strict-origin-when-cross-origin';
    });

    template.content.querySelectorAll('img').forEach(image => {
        const safeSrc = sanitizeImageUrl(image.getAttribute('src'));
        if (safeSrc === '#') {
            image.remove();
            return;
        }
        image.src = safeSrc;
        image.loading = 'lazy';
        image.decoding = 'async';
        image.referrerPolicy = 'strict-origin-when-cross-origin';
    });

    const container = document.createElement('div');
    container.appendChild(template.content.cloneNode(true));
    return container.innerHTML;
}

const SVG_ALLOWED_TAGS = new Set([
    'SVG', 'G', 'PATH', 'RECT', 'CIRCLE', 'ELLIPSE', 'LINE', 'POLYLINE', 'POLYGON',
    'TEXT', 'TSPAN', 'TEXTPATH', 'DEFS', 'USE', 'SYMBOL', 'MARKER', 'LINEARGRADIENT',
    'RADIALGRADIENT', 'STOP', 'CLIPPATH', 'MASK', 'PATTERN', 'FILTER', 'FEGBLEND',
    'FECOLORMATRIX', 'FEOFFSET', 'FEBLEND', 'FECOMPONENTTRANSFER', 'FEFUNCA', 'FEFUNCB',
    'FEFUNCG', 'FEFUNCR', 'FEGAUSSIANBLUR', 'TITLE', 'DESC', 'STYLE'
]);

const SVG_ALLOWED_ATTRIBUTES = new Set([
    'class', 'cx', 'cy', 'd', 'dx', 'dy', 'fill', 'fill-opacity', 'fill-rule',
    'font-family', 'font-size', 'font-style', 'font-weight', 'fx', 'fy', 'fr',
    'gradienttransform', 'gradientunits', 'height', 'id', 'offset', 'opacity',
    'orient', 'patternunits', 'points', 'preserveaspectratio', 'r', 'refx', 'refy',
    'rx', 'ry', 'spreadmethod', 'stop-color', 'stop-opacity', 'stroke',
    'stroke-dasharray', 'stroke-linecap', 'stroke-linejoin', 'stroke-miterlimit',
    'stroke-opacity', 'stroke-width', 'style', 'text-anchor', 'text-decoration',
    'transform', 'version', 'viewbox', 'width', 'x', 'x1', 'x2', 'xlink:href',
    'xmlns', 'xmlns:xlink', 'y', 'y1', 'y2'
]);

const SVG_URL_ATTRIBUTES = new Set(['xlink:href', 'href', 'src']);

/**
 * Sanitize SVG produced by Mermaid before it reaches the DOM.
 *
 * Mermaid runs with `securityLevel: 'strict'`, but the diagram source is model
 * output and the renderer is a third-party CDN bundle, so the result is treated
 * as untrusted: unknown elements are unwrapped, every attribute outside the
 * allowlist is dropped, and `url()` references are restricted to same-document
 * fragments. `<style>` is kept because Mermaid relies on it for theming, but it
 * is scanned for `@import` and `javascript:` which can leak data.
 */
function sanitizeRenderedSvg(svgMarkup) {
    const template = document.createElement('template');
    template.innerHTML = String(svgMarkup || '');

    const svgRoot = template.content.querySelector('svg');
    if (!svgRoot) return '';

    const urlPattern = /url\(\s*(['"]?)([^'")]+)\1\s*\)/gi;
    const isSafeCssValue = value => {
        if (/javascript\s*:/i.test(value) || /@import/i.test(value) || /expression\s*\(/i.test(value)) {
            return false;
        }
        let safe = true;
        // Fragment references (#gradient-1) stay inside the document; anything
        // else (http, //cdn, relative paths) is dropped.
        value.replace(urlPattern, (match, quote, target) => {
            if (!String(target).trim().startsWith('#')) safe = false;
            return match;
        });
        return safe;
    };

    Array.from(template.content.querySelectorAll('*')).forEach(node => {
        const tagName = node.tagName.toLowerCase();

        if (!SVG_ALLOWED_TAGS.has(node.tagName)) {
            if (node.tagName === 'SCRIPT' || node.tagName === 'FOREIGNOBJECT') {
                node.remove();
                return;
            }
            node.replaceWith(document.createTextNode(node.textContent || ''));
            return;
        }

        if (tagName === 'style') {
            const css = node.textContent || '';
            if (!isSafeCssValue(css)) {
                node.remove();
            }
            return;
        }

        Array.from(node.attributes).forEach(attribute => {
            const name = attribute.name.toLowerCase();
            const value = attribute.value || '';

            if (name.startsWith('on') || !SVG_ALLOWED_ATTRIBUTES.has(name)) {
                node.removeAttribute(attribute.name);
                return;
            }
            if (SVG_URL_ATTRIBUTES.has(name) && !value.trim().startsWith('#')) {
                node.removeAttribute(attribute.name);
                return;
            }
            if (name === 'style' && !isSafeCssValue(value)) {
                node.removeAttribute(attribute.name);
            }
        });
    });

    const container = document.createElement('div');
    container.appendChild(template.content.cloneNode(true));
    return container.innerHTML;
}


export { escapeHtml };
export { sanitizeGeneratedHtml };
export { sanitizeImageUrl };
export { sanitizeRenderedSvg };
export { sanitizeUrl };
