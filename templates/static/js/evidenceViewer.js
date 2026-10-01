import {
    HIGHLIGHT_ABSENT_NOTE,
    HIGHLIGHT_DRAWN_NOTE,
    NO_TARGET_MESSAGES,
    VIEW_KIND_LABELS,
    VIEW_NOT_A_RASTER_BADGE,
    VIEW_TARGET_MISSING_CODE,
    VIEW_TARGET_MISSING_MESSAGE
} from './config.js';

/**
 * The evidence view: the region a citation points at, drawn.
 *
 * This module exists because T3.3's acceptance criterion — "三种格式 citation
 * 均可视觉定位" — is about three different drawings, not one. A PDF needs a
 * rectangle in a coordinate frame, a slide needs an ordered list with one
 * element marked, and a DOCX paragraph needs its neighbours. Collapsing them
 * into a single "location" widget is how a paragraph index ends up being
 * drawn as a rectangle, so this module branches per kind and says which it is.
 *
 * **There is no page raster, and the module never pretends to one.** The
 * server cannot render a PDF page in this build (see `config.js`
 * `EVIDENCE_PAGE_IMAGE_PATH`) and the browser cannot either without a
 * dependency the project does not take. What the page map draws instead is the
 * *parse's own geometry*: every element rectangle the parser recorded for that
 * page, positioned in one frame, with the cited rectangle drawn on top from
 * the evidence's own locator. That is a schematic of the layout, not a
 * picture of the page, and `data-raster="false"` plus a visible badge say so
 * on every render.
 *
 * **The one rule everything else serves: the highlight is drawn or it is
 * absent, and it is never a guess.** The target rectangle is read from
 * `view.target` and from nowhere else — not reconstructed from the page's
 * element list, not derived from the neighbours of the cited block, not
 * copied out of the `xywh=` fragment by parsing text. On top of that the
 * client re-checks the four numbers itself in {@link isDrawableBox}: a
 * non-finite coordinate, a zero or negative extent, or a frame that cannot
 * contain the box all mean *no highlight*, and the reason is shown instead.
 * The server already refuses a relative or reversed box; this is the second
 * of two independent gates on the same failure, because a highlight over the
 * wrong lines is a wrong citation that looks completely right.
 *
 * Everything user-facing is Chinese, and every degraded state is a visible
 * element with a `data-state` — the panel never answers "could not locate
 * this" with an empty box, because an empty box reads as a load failure.
 */

/**
 * A coordinate must be a finite number and an extent must be positive.
 *
 * Written as a total predicate over data arriving from a server this client
 * does not own: the whole point is that a future build which starts sending a
 * NaN, a zero-height box or a `null` target gets *no highlight* rather than a
 * `NaN%`-positioned div that renders as nothing and looks like a page with no
 * text on it.
 */
function isDrawableBox(box) {
    if (!box || typeof box !== 'object') {
        return false;
    }
    const numbers = [box.x, box.y, box.width, box.height];
    if (!numbers.every(value => typeof value === 'number' && Number.isFinite(value))) {
        return false;
    }
    return box.width > 0 && box.height > 0;
}

/**
 * Position a rectangle inside a frame, as percentages.
 *
 * `top` flips. PDF user space measures y upward from the bottom-left of the
 * page and CSS measures it downward from the top, so a block whose lower edge
 * is at `y` is `height` units *below* the frame's top edge — getting this
 * backwards mirrors the whole page, which is the same shape of mistake as a
 * relative box and would be equally invisible without something to compare
 * against. The two are computed from the same four numbers, so the highlight
 * and the blocks it sits among cannot disagree about the flip.
 */
function placeBox(frame, box) {
    if (!isDrawableBox(frame) || !isDrawableBox(box)) {
        return null;
    }
    const left = ((box.x - frame.x) / frame.width) * 100;
    const top = ((frame.y + frame.height - (box.y + box.height)) / frame.height) * 100;
    const width = (box.width / frame.width) * 100;
    const height = (box.height / frame.height) * 100;
    return { left, top, width, height };
}

function applyPlacement(element, placement) {
    element.style.left = `${placement.left}%`;
    element.style.top = `${placement.top}%`;
    element.style.width = `${placement.width}%`;
    element.style.height = `${placement.height}%`;
}

function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
        node.className = className;
    }
    if (text !== undefined && text !== null) {
        node.textContent = String(text);
    }
    return node;
}

function paragraph(className, text) {
    return text ? element('p', className, text) : null;
}

/**
 * The sentence explaining a missing highlight.
 *
 * Prefers the reason the view shipped, falls back to the reason vocabulary
 * keyed by its code, and only then to the view-local "the block is not in the
 * region". The order is not cosmetic: the view's own reason is the one that
 * describes *this* absence, and a generic fallback would describe a different
 * one.
 */
function describeMissingHighlight(view) {
    const rejected = view?.target_rejected || {};
    const code = String(rejected.reason_code || '');
    if (rejected.message) {
        return String(rejected.message);
    }
    if (NO_TARGET_MESSAGES[code]) {
        return NO_TARGET_MESSAGES[code];
    }
    if (code === VIEW_TARGET_MISSING_CODE || code) {
        return VIEW_TARGET_MISSING_MESSAGE;
    }
    return '该证据没有可绘制的定位坐标。';
}

/**
 * The status line under the view, and the single place that says whether a
 * highlight was drawn.
 *
 * `data-highlight` is the machine-readable half and it is one of exactly two
 * values. There is no third, because a client that could not tell "the
 * server sent no box" from "the client decided not to draw one" would have to
 * trust the drawing, and this whole module exists so it does not have to.
 */
function buildHighlightStatus(view, drawn, reason) {
    const status = element('p', 'evidence-view-status');
    status.dataset.highlight = drawn ? 'drawn' : 'absent';
    status.textContent = drawn ? HIGHLIGHT_DRAWN_NOTE : `${reason} ${HIGHLIGHT_ABSENT_NOTE}`;
    if (!drawn) {
        const code = element('span', 'evidence-view-reason-code');
        code.dataset.reasonCode = String(view?.target_rejected?.reason_code || '');
        status.appendChild(code);
    }
    return status;
}

/**
 * A note about what was left out, whenever something was.
 *
 * Silent truncation is a release gate at zero, so a page map that dropped
 * blocks to fit its payload, clipped a preview or refused to place a mirrored
 * box says so. A viewer that quietly shows 40 of 400 blocks teaches the reader
 * that the page has 40 blocks.
 */
function buildTruncationNotes(view) {
    const notes = [];
    if (view.blocks_truncated) {
        notes.push(
            `本页共 ${Number(view.blocks_total || 0)} 个带坐标的元素，` +
            `布局中只画出前 ${Number(view.blocks_shown || 0)} 个，其余已省略。`
        );
    }
    const omitted = view.omitted || {};
    const relativeCount = Number(omitted.relative_bbox || 0);
    const unpositionedCount = Number(omitted.unpositioned || 0);
    if (relativeCount) {
        notes.push(`${relativeCount} 个元素是相对坐标框，无法确定位置，未画入布局。`);
    }
    if (unpositionedCount) {
        notes.push(`${unpositionedCount} 个元素没有坐标，未画入布局。`);
    }
    return notes;
}

function buildHeader(view, { badge }) {
    const head = element('div', 'evidence-view-head');
    const kind = element('span', 'evidence-view-kind', VIEW_KIND_LABELS[view.kind] || '定位视图');
    kind.dataset.kind = String(view.kind || '');
    head.appendChild(kind);
    if (view.label) {
        head.appendChild(element('span', 'evidence-view-label', view.label));
    }
    if (badge) {
        const marker = element('span', 'evidence-view-badge', VIEW_NOT_A_RASTER_BADGE);
        marker.dataset.raster = 'false';
        head.appendChild(marker);
    }
    return head;
}

/**
 * The PDF page map.
 *
 * The frame is the union rectangle the server sent, used only to turn PDF
 * points into percentages — a single uniform rescale, so every rectangle's
 * position relative to every other one is exactly the page's. A frame that
 * will not contain the target is treated as undrawable rather than clamped:
 * clamping would put the highlight somewhere the evidence never pointed.
 */
function renderPageMap(view, root) {
    root.dataset.kind = 'page_map';
    root.appendChild(buildHeader(view, { badge: view.is_raster !== true }));

    const canvas = element('div', 'evidence-page-map');
    canvas.dataset.coordinateSpace = String(view.coordinate_space || '');

    const frame = view.frame;
    if (isDrawableBox(frame)) {
        // The frame's own aspect is what stops the drawing from being
        // rescaled non-uniformly, which would turn a correct rectangle into a
        // stretched one.
        canvas.style.aspectRatio = `${Number(frame.width)} / ${Number(frame.height)}`;
    }

    const blocks = Array.isArray(view.blocks) ? view.blocks : [];
    let placed = 0;
    blocks.forEach(block => {
        const placement = isDrawableBox(frame) ? placeBox(frame, block) : null;
        if (!placement) {
            return;
        }
        placed += 1;
        const node = element('div', 'evidence-page-block', block.text_preview || '');
        node.dataset.elementId = String(block.element_id || '');
        node.dataset.role = String(block.role || '');
        applyPlacement(node, placement);
        if (block.preview_truncated) {
            node.dataset.previewTruncated = 'true';
            node.title = '（预览已截断）';
        }
        canvas.appendChild(node);
    });
    canvas.dataset.blocksPlaced = String(placed);
    root.appendChild(canvas);

    // The only read of the target rectangle in this module.
    const target = view.target;
    const drawable = isDrawableBox(frame) && isDrawableBox(target);
    const placement = drawable ? placeBox(frame, target) : null;
    if (placement) {
        const highlight = element('div', 'evidence-page-target', target.text || '');
        highlight.dataset.role = 'highlight';
        applyPlacement(highlight, placement);
        if (target.element_id) {
            highlight.dataset.elementId = String(target.element_id);
        }
        if (target.text_truncated) {
            highlight.dataset.textTruncated = 'true';
        }
        canvas.appendChild(highlight);
    }
    canvas.dataset.highlight = placement ? 'drawn' : 'absent';

    root.appendChild(buildHighlightStatus(view, Boolean(placement), describeMissingHighlight(view)));
    buildTruncationNotes(view).forEach(note => {
        root.appendChild(element('p', 'evidence-view-truncation', note));
    });
    return root;
}

/**
 * The slide view: the slide's own text, in order, with the cited element
 * marked. A slide has no pixel coordinates in the schema, so nothing here is
 * positioned and nothing is invented.
 */
function renderSlideView(view, root) {
    root.dataset.kind = 'slide_view';
    root.appendChild(buildHeader(view, { badge: false }));

    const list = element('ol', 'evidence-slide-list');
    const blocks = Array.isArray(view.blocks) ? view.blocks : [];
    blocks.forEach(block => {
        const item = element('li', 'evidence-slide-block');
        item.dataset.elementId = String(block.element_id || '');
        item.dataset.role = String(block.role || '');
        if (block.is_target) {
            item.dataset.isTarget = 'true';
        }
        if (block.text_truncated) {
            item.dataset.textTruncated = 'true';
        }
        item.appendChild(element('span', 'evidence-slide-text', block.text || '（无文本）'));
        list.appendChild(item);
    });
    list.dataset.blocksShown = String(blocks.length);
    root.appendChild(list);

    const drawable = Boolean(view.target);
    root.appendChild(buildHighlightStatus(view, drawable, describeMissingHighlight(view)));
    return root;
}

/**
 * The DOCX view: the cited paragraph between its neighbours, with the heading
 * path that is how a reader finds it in a document that shows no page numbers.
 */
function renderElementView(view, root) {
    root.dataset.kind = 'element_view';
    root.appendChild(buildHeader(view, { badge: false }));

    const path = Array.isArray(view.heading_path) ? view.heading_path.filter(Boolean) : [];
    if (path.length) {
        const trail = element('p', 'evidence-heading-path', path.join(' › '));
        trail.dataset.depth = String(path.length);
        root.appendChild(trail);
    }

    const list = element('ol', 'evidence-paragraph-list');
    const context = Array.isArray(view.context) ? view.context : [];
    context.forEach(block => {
        const item = element('li', 'evidence-paragraph-block');
        item.dataset.paragraph = String(block.paragraph ?? '');
        item.dataset.role = String(block.role || '');
        if (block.is_target) {
            item.dataset.isTarget = 'true';
        }
        if (block.text_truncated) {
            item.dataset.textTruncated = 'true';
        }
        item.appendChild(element('span', 'evidence-paragraph-number', `第 ${block.paragraph} 段`));
        item.appendChild(element('span', 'evidence-paragraph-text', block.text || '（无文本）'));
        list.appendChild(item);
    });
    list.dataset.blocksShown = String(context.length);
    root.appendChild(list);

    const drawable = Boolean(view.target);
    root.appendChild(buildHighlightStatus(view, drawable, describeMissingHighlight(view)));
    return root;
}

/**
 * Render one evidence's view into the panel.
 *
 * Returns the root element, or `null` when there is no view — which is the
 * answer for a spreadsheet range and a plain-text character offset, and for
 * any format this build does not know. `null` is not a failure: the passage
 * and the state banner are already on screen, and a client handed a view it
 * could not draw would have to invent one.
 */
function renderEvidenceView(view) {
    // The literal id, not a constant, so `tools/check_frontend.py` can
    // require this container to be declared in `index.html`.
    const container = document.getElementById('evidenceViewer');
    if (!container) {
        return null;
    }
    container.replaceChildren();
    if (!view || typeof view !== 'object') {
        container.hidden = true;
        container.dataset.kind = '';
        return null;
    }

    const root = element('div', 'evidence-view');
    root.dataset.kind = String(view.kind || '');
    root.dataset.highlight = 'absent';

    if (view.note) {
        const note = paragraph('evidence-view-note', view.note);
        // The note is the "this is a schematic" statement. It is a visible
        // element rather than a `title` tooltip because a reader who cannot
        // see it would read the layout as the page, which is the mistake the
        // badge and this note exist together to prevent.
        note.dataset.role = 'disclaimer';
        root.appendChild(note);
    }

    if (view.kind === 'page_map') {
        renderPageMap(view, root);
    } else if (view.kind === 'slide_view') {
        renderSlideView(view, root);
    } else if (view.kind === 'element_view') {
        renderElementView(view, root);
    } else {
        root.dataset.kind = 'unsupported';
        root.appendChild(paragraph('evidence-view-note', '该定位类型没有可展示的视图。'));
    }

    container.appendChild(root);
    container.hidden = false;
    return root;
}


export { describeMissingHighlight, isDrawableBox, placeBox, renderEvidenceView };
