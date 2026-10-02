import {
    DEGRADED_ALIGNMENT_OUTCOMES,
    DIRECT_QUOTE_CLOSE_MARK,
    DIRECT_QUOTE_OPEN_MARK,
    PRESENTATION_DIRECT_QUOTE,
    PRESENTATION_PARAPHRASE,
    WRAPPING_QUOTE_PAIRS
} from './config.js';
import { t } from './i18n.js';

/**
 * How a claim may be shown to a reader — the front-end half of the release
 * gate in section 7 of the development plan: "Direct quote exact/normalized
 * alignment 100%；对不齐降级 paraphrase，不带引号".
 *
 * The gate is a rendering rule, and a rendering rule enforced only by a
 * stylesheet is not enforced. If the distinguishing signal is a background
 * colour, then a theme change, a stylesheet that fails to load, a high-contrast
 * mode, or a screen reader all erase it — and the failure they leave behind is
 * the exact failure the gate exists to prevent: a sentence the paper did not
 * say, presented as though it had. So the distinction is carried four times
 * over, in ways that fail independently:
 *
 *   1. **Different elements.** A verified claim is a `<blockquote>`, a
 *      paraphrase is a `<p>`. The tag itself says which, so the information
 *      survives every stylesheet.
 *   2. **Data attributes.** `data-presentation` and `data-quote-marks` on the
 *      claim element, readable by any script and by the test suite without a
 *      browser. This is what `tests/test_frontend_evidence.py` asserts on.
 *   3. **Real quotation marks in the markup.** A direct quote's marks are text
 *      nodes built from the two constants in `config.js`, inside a span marked
 *      `aria-hidden` so a screen reader announces the words rather than the
 *      punctuation. Not `::before { content: '"' }`, which is styling.
 *   4. **Textual access.** A prefix that survives with styles off and reads
 *      correctly when a claim is copied out of the page.
 *
 * And the negative half, which is the half the gate is actually about: the
 * paraphrase path cannot emit a quotation mark even if someone wants it to.
 * `renderParaphraseClaim` contains no mark constant and no mark-building call,
 * its text is passed through `stripWrappingQuotes` first, and it refuses to
 * attach a location chip — because a chip beside an unverified sentence is a
 * chip the reader will read as a citation, which is the mistake the gate was
 * written to prevent.
 *
 * Nothing here decides whether a claim is a quote. The backend does, with
 * string comparison, in `rag/citations.py`. This module only refuses to
 * contradict it: an outcome, flag or field this module does not recognise
 * resolves to a paraphrase, because an unrecognised value is not evidence of
 * alignment and failing toward the unverifiable side is the recoverable
 * direction.
 */

const DEGRADED_OUTCOMES = new Set(DEGRADED_ALIGNMENT_OUTCOMES);

function isDirectQuote(claim) {
    return String(claim?.presentation || '') === PRESENTATION_DIRECT_QUOTE;
}

/**
 * Resolve the presentation a claim must be rendered under.
 *
 * Read defensively and in one place, because the Q&A response shape is still
 * moving and this module is not allowed to guess wrong in the direction that
 * puts quotation marks on unverified text. Three sources are consulted, most
 * specific first:
 *
 * * an explicit `presentation` field, which is what the current contract
 *   sends and the only one a caller sets deliberately;
 * * a boolean (`uses_quotation_marks` / `quote_marks_allowed`), which is the
 *   same statement as a flag and is what older payloads carry;
 * * the alignment `outcome`, classified the way `CitationVerification`
 *   classifies it server-side.
 *
 * Anything still unrecognised — a new outcome, a missing field, `null` — is a
 * paraphrase. There is no default-to-quote branch, and that absence is the
 * point: adding a new outcome to this list requires a decision, which is the
 * same decision the backend forces by raising on an unclassified outcome.
 */
function resolvePresentation(claim) {
    if (!claim || typeof claim !== 'object') {
        return PRESENTATION_PARAPHRASE;
    }

    const explicit = String(claim.presentation || '').trim();
    if (explicit === PRESENTATION_DIRECT_QUOTE) {
        return PRESENTATION_DIRECT_QUOTE;
    }
    if (explicit === PRESENTATION_PARAPHRASE) {
        return PRESENTATION_PARAPHRASE;
    }

    if (claim.uses_quotation_marks === true || claim.quote_marks_allowed === true) {
        return PRESENTATION_DIRECT_QUOTE;
    }
    if (claim.uses_quotation_marks === false || claim.quote_marks_allowed === false) {
        return PRESENTATION_PARAPHRASE;
    }

    const outcome = String(claim.outcome || '').trim();
    if (outcome === 'aligned' || claim.aligned === true || claim.is_direct_quote === true) {
        return PRESENTATION_DIRECT_QUOTE;
    }
    if (outcome && DEGRADED_OUTCOMES.has(outcome)) {
        return PRESENTATION_PARAPHRASE;
    }

    return PRESENTATION_PARAPHRASE;
}

/**
 * Remove one matched pair of surrounding quotation marks.
 *
 * Mirrors the backend's `_strip_wrapping_quotes` deliberately rather than
 * importing a different rule, for the same reason the server writes the list
 * out longhand: a client that strips a pair the server would not, or refuses a
 * pair the server would, shows text the reader cannot match against the paper.
 * A text that is nothing but a pair is left alone — `"…"` carries no claim, and
 * stripping it to an empty string would render as nothing at all.
 */
function stripWrappingQuotes(text) {
    const stripped = String(text ?? '').trim();
    for (const [opening, closing] of WRAPPING_QUOTE_PAIRS) {
        if (
            stripped.length < opening.length + closing.length
            || !stripped.startsWith(opening)
            || !stripped.endsWith(closing)
        ) {
            continue;
        }
        const inner = stripped.slice(opening.length, stripped.length - closing.length).trim();
        if (inner) {
            return inner;
        }
    }
    return stripped;
}

/**
 * Whether a string still carries an outer pair of quotation marks.
 *
 * Exists to make the forbidden state observable. `renderParaphraseClaim` calls
 * it after stripping and reports a survivor rather than shipping it silently:
 * a paraphrase that arrives already wrapped in marks is a payload this client
 * could not clean, and that is worth knowing rather than assuming away.
 */
function hasWrappingQuotes(text) {
    const stripped = String(text ?? '').trim();
    return WRAPPING_QUOTE_PAIRS.some(([opening, closing]) => (
        stripped.length >= opening.length + closing.length
        && stripped.startsWith(opening)
        && stripped.endsWith(closing)
    ));
}

function claimText(claim) {
    return String(claim?.text ?? claim?.quote ?? claim?.rendered_text ?? '').trim();
}

function buildClaimElement(tagName, claim, presentation) {
    const element = document.createElement(tagName);
    element.className = `claim claim-${presentation === PRESENTATION_DIRECT_QUOTE ? 'direct-quote' : 'paraphrase'}`;
    element.dataset.presentation = presentation;
    element.dataset.quoteMarks = presentation === PRESENTATION_DIRECT_QUOTE ? 'allowed' : 'forbidden';
    return element;
}

function buildMarkSpan(mark) {
    const span = document.createElement('span');
    // The marks are real text, not a `::before` rule, so the distinction
    // survives a stylesheet that never loads. Hiding them from assistive
    // technology is not hiding them from the gate: the word `quote` in the
    // accessible name below is what a screen reader user actually needs, and a
    // reader announcing `“` before every claim would be worse.
    span.className = 'claim-quote-mark';
    span.setAttribute('aria-hidden', 'true');
    span.textContent = mark;
    return span;
}

/**
 * Render a claim the server verified against the source text.
 *
 * Only reached when the presentation resolves to a direct quote, so the marks
 * below are the *only* place in this module that can add them. The text goes
 * in as `textContent` — never as markup — because a verified quote is still
 * text from a document a stranger uploaded.
 */
function renderDirectQuoteClaim(claim) {
    const element = buildClaimElement('blockquote', claim, PRESENTATION_DIRECT_QUOTE);
    const text = claimText(claim);
    const mode = String(claim?.mode || '').trim();
    if (mode) {
        element.dataset.verificationMode = mode;
    }

    const prefix = document.createElement('span');
    prefix.className = 'claim-prefix';
    // The prefix is the textual carrier of the release gate: the one of the
    // four carriers that follows the language the reader chose. A gate stated in
    // a language the reader cannot read is not a gate, so the sentences that
    // distinguish a verified quotation from a paraphrase are catalogue entries
    // resolved at render time, and both locales say the same thing about the
    // same claim.
    prefix.textContent = mode === 'normalized'
        ? t('claim.prefix.normalized_quote')
        : t('claim.prefix.direct_quote');

    const body = document.createElement('p');
    body.className = 'claim-text';
    body.appendChild(buildMarkSpan(DIRECT_QUOTE_OPEN_MARK));
    body.appendChild(document.createTextNode(text));
    body.appendChild(buildMarkSpan(DIRECT_QUOTE_CLOSE_MARK));

    element.appendChild(prefix);
    element.appendChild(body);

    const message = String(claim?.message || '').trim();
    if (message) {
        const note = document.createElement('p');
        note.className = 'claim-verdict';
        note.textContent = message;
        element.appendChild(note);
    }

    if (claim?.locator) {
        element.dataset.locatorKind = String(claim.locator.kind || '');
    }
    return element;
}

/**
 * Render a claim the server could not align with the source.
 *
 * The rule this function exists to keep: **no quotation mark leaves here.**
 *
 * Not "we styled it differently" — structurally. The body of this function
 * contains no mark constant and no call that builds one, the text is stripped
 * of an outer pair first, and the element is built by `buildClaimElement` with
 * the paraphrase presentation, which is a `<p>`, not a `<blockquote>`. The
 * test suite reads this function's body and asserts that no quotation mark
 * character appears in it, so the property is checked by something that cannot
 * be satisfied by editing a stylesheet.
 *
 * Two further choices are deliberate:
 *
 * * **The wording is stated, not implied.** A reader who sees a quote-shaped
 *   sentence with no quotation marks is owed an explanation, and the backend
 *   already writes one per outcome. Showing it is the difference between a
 *   degraded claim and a broken one.
 * * **No location is attached.** A paraphrase has no verified position, and a
 *   page chip beside it would read as a citation. The absence is enforced by
 *   simply not rendering one — `locator` is deliberately not read here.
 */
function renderParaphraseClaim(claim) {
    const element = buildClaimElement('p', claim, PRESENTATION_PARAPHRASE);
    const text = stripWrappingQuotes(claimText(claim));

    const prefix = document.createElement('span');
    prefix.className = 'claim-prefix';
    prefix.textContent = t('claim.prefix.paraphrase');

    const body = document.createElement('span');
    body.className = 'claim-text';
    body.textContent = text;

    if (hasWrappingQuotes(body.textContent)) {
        // A pair survived the strip — this client could not clean the payload.
        // Reported rather than shipped, because a paraphrase rendered inside
        // quotation marks is the release-gate failure, and it is worth a
        // console entry to find out which producer sent it.
        console.error('Paraphrase text still carries wrapping quotation marks; rendering without them.', body.textContent);
    }

    element.appendChild(prefix);
    element.appendChild(body);

    const message = String(claim?.message || '').trim();
    if (message) {
        const note = document.createElement('span');
        note.className = 'claim-verdict';
        note.textContent = message;
        element.appendChild(note);
    }
    return element;
}

/**
 * Render one claim under whichever presentation the server decided.
 *
 * The single dispatch point, so there is exactly one place where a claim can
 * become a quotation and exactly one where it cannot.
 */
function renderClaim(claim) {
    return resolvePresentation(claim) === PRESENTATION_DIRECT_QUOTE
        ? renderDirectQuoteClaim(claim)
        : renderParaphraseClaim(claim);
}

/**
 * The accessible name for a citation chip, so the verdict is announced too.
 *
 * Resolved through the catalogue for the same reason the visible prefix is: the
 * verdict is the part a screen-reader user has no other way to learn, so it is
 * the part that must not be stranded in a language the reader did not ask for.
 * The label is passed as an already-delimited fragment — the two languages
 * bracket a citation label differently, and the reader only ever sees the
 * sentence this builds.
 */
function describeClaim(claim) {
    const direct = resolvePresentation(claim) === PRESENTATION_DIRECT_QUOTE;
    const label = String(claim?.citation_label || claim?.label || '').trim();
    if (!direct) {
        return t('claim.chip.paraphrase');
    }
    return t('claim.chip.direct_quote', { label: label ? `(${label})` : '' });
}


export { describeClaim };
export { hasWrappingQuotes };
export { isDirectQuote };
export { renderClaim };
export { renderDirectQuoteClaim };
export { renderParaphraseClaim };
export { resolvePresentation };
export { stripWrappingQuotes };
