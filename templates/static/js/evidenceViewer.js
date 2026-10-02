import {
    HIGHLIGHT_ABSENT_NOTE,
    HIGHLIGHT_DRAWN_NOTE,
    NO_TARGET_MESSAGES,
    VIEW_KIND_LABELS,
    VIEW_NOT_A_RASTER_BADGE,
    VIEW_TARGET_MISSING_CODE,
    VIEW_TARGET_MISSING_MESSAGE
} from './config.js';
import { formatNumber, onLocaleChange, t } from './i18n.js';

/**
 * The catalogue key prefix for the sentence about a reason code.
 *
 * Two vocabularies meet in {@link reasonCatalogueKey} and neither is derived
 * from the other: the server sends a `reason_code`, and `config.js` holds this
 * client's copy of the code vocabulary. The key is built *from* the code, but
 * only after the client recognises the code — a key built from a code nobody
 * recognises renders as `evidence.no_target.something_new` on the page.
 */
const REASON_CODE_KEY_PREFIX = 'evidence.no_target.';

/** The one code that means "the region resolved, the block is not in it". */
const TARGET_MISSING_KEY = 'evidence.view_target_missing';

/**
 * The floor for the placeholder that stands in for an empty block.
 *
 * Not interface copy in the catalogue sense — it stands for the *absence* of
 * the document's own text — but it is still a sentence the interface chose, so
 * it is translated like one and falls back to this rather than to a key.
 */
const NO_TEXT_PLACEHOLDER = '（无文本）';

/**
 * `t()`, with a floor under it.
 *
 * `t()` already falls back to the Chinese catalogue and then to the key
 * itself. The key is right for a reviewer reading the page source and wrong for
 * a reader looking at the page: a badge that says `evidence.not_a_raster` reads
 * as a broken button, which is the opposite of the reassurance it carries. So
 * every sentence this module migrated away from a `config.js` constant keeps
 * that constant as its floor, and a catalogue entry that goes missing degrades
 * to a sentence — never to a dotted key and never to a blank element.
 *
 * The floors are unreachable while the catalogue carries every key, and
 * `tests/test_i18n.py` asserts that it does, in both languages.
 */
function localised(key, floor) {
    const text = t(key);
    return text === key ? floor : text;
}

/** The reason code the resolver sent, resolved to a catalogue key or `null`. */
function reasonCatalogueKey(code) {
    if (code === VIEW_TARGET_MISSING_CODE) {
        return TARGET_MISSING_KEY;
    }
    // `NO_TARGET_MESSAGES` is the client's vocabulary, and membership in it is
    // what says this build can describe the code at all.
    return NO_TARGET_MESSAGES[code] ? `${REASON_CODE_KEY_PREFIX}${code}` : null;
}

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
 * **Interface copy is localised; the document's words are not.** Every sentence
 * this module puts in front of a reader — the view kind, the badge, the two
 * highlight notes, the truncation notes, the paragraph numbers, the "no text"
 * placeholders, the unsupported-kind notice — goes through `t()` and is re-read
 * on a locale change. The things it does *not* translate are the ones that are
 * not the interface's: the cited passage, a page block's preview, a heading
 * path, `view.label`, and `view.note`. Those are the document's words or the
 * server's own prose, and an English reader looking at a Chinese paper must
 * still see them in the paper's language — see `documentLocale` in `i18n.js`.
 * Getting that boundary wrong in the *other* direction is the worse error,
 * which is why the two are kept visibly apart below: interface strings are
 * looked up by key, document strings are copied byte for byte.
 *
 * Every degraded state is a visible element with a `data-state` — the panel
 * never answers "could not locate this" with an empty box, because an empty
 * box reads as a load failure.
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
 * The reason code is looked up **before** the sentence the view shipped, and
 * the order is the point. The code is the stable, language-neutral half of the
 * contract and every code this build knows has a sentence in the reader's own
 * language; `target_rejected.message` is the server's prose, whose language is
 * an assumption rather than a fact. So the code decides, and the server's
 * sentence is used only for a code this build has never heard of — which is
 * exactly the order `errorPresentation()` in `i18n.js` uses for backend
 * errors, and for the same reason: an English interface that strands a Chinese
 * sentence beside it is the mixed-language page, not a translated one.
 *
 * The last resort is the view-local "this evidence has no drawable
 * coordinates", and it is a *visible* sentence rather than an empty element.
 */
function describeMissingHighlight(view) {
    const rejected = view?.target_rejected || {};
    const code = String(rejected.reason_code || '');
    const key = reasonCatalogueKey(code);
    if (key) {
        return localised(
            key,
            code === VIEW_TARGET_MISSING_CODE
                ? VIEW_TARGET_MISSING_MESSAGE
                : NO_TARGET_MESSAGES[code]
        );
    }
    if (rejected.message) {
        return String(rejected.message);
    }
    return localised('evidence.view.no_drawable_target', '该证据没有可绘制的定位坐标。');
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
    status.textContent = drawn
        ? localised('evidence.highlight_drawn', HIGHLIGHT_DRAWN_NOTE)
        : `${reason} ${localised('evidence.highlight_absent', HIGHLIGHT_ABSENT_NOTE)}`;
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
 *
 * The counts are the reader's numbers, so they go through `formatNumber`: a
 * four-figure block count grouped the English way and not the Chinese way is
 * the sort of detail that makes an interface look translated rather than
 * localised, and this is one of the few places it is visible at all.
 */
function buildTruncationNotes(view) {
    const notes = [];
    if (view.blocks_truncated) {
        notes.push(t('evidence.view.blocks_truncated', {
            total: formatNumber(Number(view.blocks_total || 0)),
            shown: formatNumber(Number(view.blocks_shown || 0))
        }));
    }
    const omitted = view.omitted || {};
    const relativeCount = Number(omitted.relative_bbox || 0);
    const unpositionedCount = Number(omitted.unpositioned || 0);
    if (relativeCount) {
        notes.push(t('evidence.view.omitted_relative', { count: formatNumber(relativeCount) }));
    }
    if (unpositionedCount) {
        notes.push(t('evidence.view.omitted_unpositioned', { count: formatNumber(unpositionedCount) }));
    }
    return notes;
}

/**
 * The header: what this view is, what it is labelled, and whether it is a
 * rendering of the page.
 *
 * `view.kind` is a wire value this build may not know. An unknown kind gets the
 * generic "located view" sentence rather than the key it would otherwise print,
 * because the alternative is a header reading `evidence.view_kind.something_new`
 * to someone who has just clicked a citation.
 */
function buildHeader(view, { badge }) {
    const head = element('div', 'evidence-view-head');
    const kind = element('span', 'evidence-view-kind', t(
        VIEW_KIND_LABELS[view.kind] ? `evidence.view_kind.${view.kind}` : 'evidence.view_kind.fallback'
    ));
    kind.dataset.kind = String(view.kind || '');
    head.appendChild(kind);
    if (view.label) {
        // The server's own label for the region ("p3", "slide 4"). It names the
        // document's own coordinate, so it is shown as sent.
        head.appendChild(element('span', 'evidence-view-label', view.label));
    }
    if (badge) {
        const marker = element('span', 'evidence-view-badge', localised(
            'evidence.not_a_raster',
            VIEW_NOT_A_RASTER_BADGE
        ));
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
            node.title = t('evidence.view.preview_truncated');
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
        // `block.text` is the document's own words and is copied as sent; the
        // placeholder only stands in when there are none.
        item.appendChild(element(
            'span',
            'evidence-slide-text',
            block.text || localised('evidence.view.no_text', NO_TEXT_PLACEHOLDER)
        ));
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
        // The number is interface furniture around the document's words, so it
        // is the one half of this row that follows the reader's switch; the
        // text beside it never does.
        item.appendChild(element(
            'span',
            'evidence-paragraph-number',
            t('evidence.paragraph_label', { label: block.paragraph })
        ));
        item.appendChild(element(
            'span',
            'evidence-paragraph-text',
            block.text || localised('evidence.view.no_text', NO_TEXT_PLACEHOLDER)
        ));
        list.appendChild(item);
    });
    list.dataset.blocksShown = String(context.length);
    root.appendChild(list);

    const drawable = Boolean(view.target);
    root.appendChild(buildHighlightStatus(view, drawable, describeMissingHighlight(view)));
    return root;
}

/**
 * The view this container is currently showing.
 *
 * Remembered for exactly the reason `evidence.js` remembers which host it
 * rendered a claim into: this module produces its text at render time, so a
 * locale change can only be honoured by re-rendering, and a re-render needs
 * the payload the last render used. `null` means "nothing is on screen", which
 * is different from "something is on screen and it is empty" — so a repaint
 * after a deliberate hide does nothing rather than resurrecting the view.
 */
let lastRenderedView = null;

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
        lastRenderedView = null;
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
        //
        // It is also the one sentence in this panel this module does *not*
        // own: the server chose it, and it carries no code beside it to key a
        // translation on. It is shown as sent, which is the honest reading —
        // the alternative is guessing at which of the server's notes this is.
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
        root.appendChild(paragraph('evidence-view-note', t('evidence.unsupported_view')));
    }

    container.appendChild(root);
    container.hidden = false;
    lastRenderedView = view;
    return root;
}

/**
 * Repaint the open panel in the new language.
 *
 * This is the same shape as `repaintLocalizedSurfaces()` in `evidence.js`:
 * remember what was rendered, re-render it on `onLocaleChange`, and do nothing
 * when there is nothing on screen. `evidence.js` owns the panel and repaints
 * through `renderEvidencePayload`, which calls this module — so the panel-open
 * case is already covered there. This subscription covers the rest: the panel
 * is open but its payload repaint did not run, and a viewer left showing the
 * previous language beside an interface in the new one, which is worse than no
 * translation at all because the reader has no way to know which half to trust.
 *
 * Both renders write the same children into the same container and the last one
 * wins, so the two subscriptions cannot disagree.
 */
function repaintLocalizedView() {
    if (!lastRenderedView) return;
    const panel = document.getElementById('evidencePanel');
    // The panel carries the viewer's visibility. A closed panel is hidden with
    // its contents, and repainting a subtree nobody can see buys nothing and
    // risks resurrecting a view the reader has already dismissed.
    if (panel && panel.hidden) return;
    renderEvidenceView(lastRenderedView);
}

// A module is a singleton in the ES module graph, so this runs once per page
// and needs no `localeBound` flag: `evidence.js` guards its own subscription
// because it subscribes from an initialiser that may run more than once.
onLocaleChange(repaintLocalizedView);

export {
    describeMissingHighlight,
    isDrawableBox,
    placeBox,
    renderEvidenceView,
    repaintLocalizedView
};
