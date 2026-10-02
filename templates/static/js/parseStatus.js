// The reparse warning copy and the parser-option labels are catalogue keys
// resolved with `t()` at render time, not constants imported from `config.js`:
// both are reader-facing sentences, and the constants in `config.js` hold only
// the `zh-CN` half of each pair.
import {
    CLOUD_OUTCOME_OPERATION,
    CLOUD_OUTCOME_PRESENTATIONS,
    CLOUD_OUTCOME_UNKNOWN,
    REPARSE_API_PATH,
    REPARSE_PARSER_OPTIONS
} from './config.js';
import { onLocaleChange, t } from './i18n.js';
import { state } from './state.js';
import { apiRequest } from './api.js';
import { capturePreservedFields, dismissNotice, notifyFailure, restorePreservedFields } from './runtime.js';
import { setControlDisabled, updateStatus } from './dom.js';

/**
 * Parse status and the reparse control — T3.4, whose acceptance criterion is a
 * sentence rather than a component: "用户能明确知道文档被谁解析、是否经过云端".
 *
 * That sentence is the design constraint. The backend answers it from
 * `ParseOutcome` (see `build_parse_provenance` in `analysis/orchestrator.py`),
 * and the interesting part is that it answers it as *three separate facts*
 * rather than one flag:
 *
 *   * `used_cloud` — the winning text came from the cloud. This document left
 *     the machine.
 *   * `cloud_attempted` — a cloud request was made and paid for, which may still
 *     be true of a document that was ultimately parsed locally.
 *   * `touched_network` — any attempt was allowed to reach the network.
 *
 * Collapsing them into one "cloud: yes/no" is exactly how a deployment ends up
 * quietly billing for a result it discarded, so all three are shown, each with
 * its own copy, and the difference is left to the reader to draw. The
 * alternative — showing only `used_cloud` — understates the egress, which for
 * this product is the direction that matters.
 *
 * ## `cloud_outcome`, and why it has its own row
 *
 * The fourth fact the backend sends is `cloud_outcome`: the machine-readable
 * verdict of the cloud leg (`success`, `partial`, `quota_exhausted`,
 * `timed_out`, `blocked`, `failed`). It was already travelling all the way to
 * the client and this module was throwing it away, which collapsed four
 * different outcomes into one sentence derived from `error_message`. A reader
 * whose cloud parse had run out of quota was told what a reader whose network
 * failed was told, and neither could act on it.
 *
 * So it gets its own row, and the row is *short*: it names the outcome. The
 * explanation, the hint and the retry button belong to the failure surface, and
 * the four outcomes that mean "no cloud text came back" are handed to
 * `runtime.notifyFailure` exactly like every other failure in this interface.
 * One place describes a failure; the card says what happened.
 *
 * ## Why an absent outcome is not silently a success
 *
 * An outcome this build cannot name is reported as a failure, mirroring
 * `UNKNOWN_FAILURE_KIND`. The exception is an *empty* outcome with no record of
 * a cloud leg at all: that is a purely local parse, and announcing a cloud
 * failure over one would be the loudest possible kind of wrong. An empty
 * outcome *with* `cloud_attempted` set is the case that matters — a billed cloud
 * request whose outcome did not survive — and that degrades to the generic
 * message rather than to silence.
 *
 * **On the quality verdict.** The wire payload the orchestrator currently
 * emits (`build_parse_provenance`) does not include the `ParseQualityReport`.
 * Rather than render an empty quality card and let it read as "quality passed",
 * this module checks for the field and, when it is absent, says so in the card.
 * An absent signal is reported as absent; it is never reported as a good one.
 *
 * **On reparse.** The control offers what T3.4 names — forced OCR, and a
 * different parser — and states the consequence of each next to the option
 * rather than after the fact, because a cloud parser is a decision to upload
 * someone's file to a third party and that decision is the user's to make. Each
 * option additionally names the *requirement* for it to work at all, because the
 * shipped default (`PARSER_BACKEND=legacy`) refuses every named parser with a
 * 409 and the browser is not told which parsers this deployment has; see
 * `REPARSE_PARSER_OPTIONS`. The endpoint it POSTs to is the T2.4 contract,
 * `/v1/documents/{id}/reparse`, which this repository does not currently serve.
 * So the failure path is a first-class, visible state rather than a silent
 * no-op: an unavailable endpoint says so, in the card, in the same words the
 * rest of this module uses for every other degraded parse. A control that looks
 * armed and does nothing is the failure mode this product is built against.
 *
 * **On language.** Every sentence here is produced at render time and goes
 * through `t()`. Re-rendering on a locale change is therefore not enough on its
 * own — the strings have to be *looked up* again — which is why this module
 * subscribes to the locale event and rebuilds the card, the parser menu and the
 * last reparse result together. What the card says is unchanged by this: the
 * same facts, the same tone, in whichever language the reader chose.
 */

/** How each parse status maps onto the tones the stylesheet already defines. */
const STATUS_TONES = {
    success: 'pass',
    partial: 'warn',
    failed: 'fail',
    error: 'fail'
};

function toneForStatus(status) {
    return STATUS_TONES[String(status || '').toLowerCase()] || 'pass';
}

/** Whether a `status` says the document produced no usable text at all. */
function isFailureStatus(status) {
    const value = String(status || '').toLowerCase();
    return value === 'failed' || value === 'error';
}

/**
 * Read one `cloud_outcome` and say what it is.
 *
 * Pure, and pure for a reason: this is the whole decision, and it is the kind of
 * decision that is only ever checked by reading it. An unrecognised value
 * resolves to `CLOUD_OUTCOME_UNKNOWN` rather than to `success` — an outcome
 * this build cannot name is a failure, the same direction `UNKNOWN_FAILURE_KIND`
 * fails in.
 *
 * @returns {{raw: string, name: string, known: boolean, code: string,
 *            reports: boolean, tone: string, sentenceKey: string,
 *            hasCloudLeg: boolean}}
 */
function describeCloudOutcome(parse) {
    const raw = String(parse?.cloud_outcome ?? '').trim().toLowerCase();
    const known = Object.prototype.hasOwnProperty.call(CLOUD_OUTCOME_PRESENTATIONS, raw);
    const name = known ? raw : CLOUD_OUTCOME_UNKNOWN;
    const presentation = CLOUD_OUTCOME_PRESENTATIONS[name];

    // Only a cloud leg that was actually attempted, or a document that produced
    // nothing, is entitled to a cloud verdict. A local parse failure with no
    // cloud attempt is not a cloud problem and must not be described as one.
    const cloudInvolved = Boolean(parse?.cloud_attempted || parse?.used_cloud);
    const degraded = isFailureStatus(parse?.status);

    // Two different absences, and they are not the same fact.
    //
    //   * A *non-empty* value this build cannot name is a failure: the server
    //     reported a verdict and the verdict is not one we can act on. Same
    //     direction as `UNKNOWN_FAILURE_KIND`, which is terminal.
    //   * An *empty* value is the ordinary case — `_cloud_outcome_of` reads the
    //     outcome off a *failed* attempt, so every successful cloud parse
    //     arrives with `""`. Reporting that as a failure would announce a broken
    //     cloud parser above every healthy cloud parse. It becomes a failure
    //     only when a cloud leg is on the record *and* the document produced
    //     nothing, which is the billed-but-unrecorded case, and it degrades to
    //     the generic message rather than to silence.
    const reports = known
        ? presentation.reports
        : (raw ? presentation.reports : (cloudInvolved && degraded));

    return {
        raw,
        name,
        known,
        // A code is only minted when there is something to report, so the card
        // and the notice cannot end up describing a failure that was declined.
        code: reports ? presentation.code : '',
        tone: presentation.tone,
        reports,
        // The key, not the sentence: `t()` is called at render time so a locale
        // switch re-reads the catalogue, and so this function stays free of the
        // interface's language. An *empty* outcome degrades to the generic
        // wording and an unrecognised one says it is unrecognised — those are
        // different facts and the reader can act on them differently.
        sentenceKey: `parse.cloud_outcome.${raw ? (known ? raw : 'unknown') : CLOUD_OUTCOME_UNKNOWN}`,
        // Whether the card has a cloud leg to describe at all. A parse that
        // never left the machine has no cloud outcome, and inventing a row for
        // it would put "the server reported an outcome this version does not
        // know" above every healthy local document — which is worse than the
        // blank this rule avoids.
        hasCloudLeg: Boolean(raw) || cloudInvolved
    };
}

function appendFact(container, label, value, tone) {
    const row = document.createElement('div');
    row.className = 'parse-fact';

    const key = document.createElement('span');
    key.className = 'parse-fact-label';
    key.textContent = label;

    const detail = document.createElement('span');
    detail.className = 'parse-fact-value';
    if (tone) {
        detail.dataset.tone = tone;
    }
    detail.textContent = value;

    row.append(key, detail);
    container.appendChild(row);
    return row;
}

function appendNotice(container, message, tone) {
    const notice = document.createElement('p');
    notice.className = 'parse-notice';
    notice.dataset.tone = tone;
    notice.setAttribute('role', 'note');
    notice.textContent = message;
    container.appendChild(notice);
    return notice;
}

/**
 * The notice raised for the current cloud outcome, so a re-render replaces it.
 *
 * A module-level handle rather than a marker on the card, because the card is
 * rebuilt from scratch on every render and the notice lives in the runtime's
 * container, outside it. Without this, re-parsing the same document three times
 * leaves three identical quota notices stacked above it — the reader cannot tell
 * a repeated fact from three separate events.
 */
let cloudOutcomeNoticeId = null;

/** Whether the locale subscription below has already been made. */
let cloudOutcomeLocaleBound = false;

/**
 * Hand a cloud outcome to the failure surface, or decline to.
 *
 * `notifyFailure` is the only thing in this front end that decides what a
 * failure looks like, and the branch here is deliberately not a second copy of
 * that decision: this function supplies the *code*, and the code's kind comes
 * from `ERROR_PRESENTATIONS` as it does for every other failure. So
 * `quota_exhausted` gets no button at all (terminal — a spent quota is unchanged
 * by asking again) and `timed_out` gets one (retryable — the batch may already
 * have finished on the provider's side), without either being decided here.
 *
 * The reparse is offered as the retry because it is the same request re-issued:
 * the document is unchanged and the reader chose the parser. It is not a reload
 * and it is not a fresh analysis.
 */
function reportCloudOutcome(outcome) {
    dismissCloudOutcomeNotice();
    if (!outcome.reports || !outcome.code) {
        return null;
    }
    cloudOutcomeNoticeId = notifyFailure(
        { code: outcome.code, operation: CLOUD_OUTCOME_OPERATION },
        { onRetry: () => submitReparse() }
    );
    return cloudOutcomeNoticeId;
}

function dismissCloudOutcomeNotice() {
    if (!cloudOutcomeNoticeId) {
        return false;
    }
    const dismissed = dismissNotice(cloudOutcomeNoticeId);
    cloudOutcomeNoticeId = null;
    return dismissed;
}

/**
 * Show the cloud outcome as a fact, in the card.
 *
 * Separate from the notice on purpose. The card answers "what happened to this
 * document", and that question has an answer for `success` and `partial` too —
 * neither of which is a failure and neither of which gets a notice. Folding the
 * row into the notice would mean the two successful outcomes said nothing at
 * all, which is the same silence this module was written to remove.
 */
function appendCloudOutcomeRow(facts, outcome) {
    if (!outcome.hasCloudLeg) {
        return null;
    }
    appendFact(facts, t('parse.cloud_result'), t(outcome.sentenceKey), outcome.tone);
    return facts;
}

/**
 * Render every signal the parse outcome carries, degraded or not.
 *
 * Nothing is collapsed into a single "OK". A document that was parsed
 * successfully, by a local parser, with three quality warnings is a document
 * the reader has been told less about than one that parsed perfectly, and
 * telling them less is what the section 7 gate — "截断/低质量必须是可见警告或
 * 失败态" — is written against.
 */
function renderParseStatus(parse) {
    const container = document.getElementById('parseStatus');
    if (!container) {
        return;
    }
    container.replaceChildren();

    // An empty object is what a hand-assembled result carries; the orchestrator
    // returns `{}` rather than a dict of confident-looking falses precisely so
    // this branch can say "unknown" instead of rendering "parsed by ".
    if (!parse || typeof parse !== 'object' || !Object.keys(parse).length) {
        dismissCloudOutcomeNotice();
        appendNotice(container, t('parse.unknown_notice'), 'unknown');
        container.dataset.state = 'unknown';
        return;
    }

    container.dataset.state = toneForStatus(parse.status);
    container.dataset.usedCloud = String(Boolean(parse.used_cloud));
    container.dataset.fallbackUsed = String(Boolean(parse.fallback_used));

    // The cloud leg's own verdict, resolved before anything is drawn so the row
    // and the notice cannot disagree about which outcome this was.
    const cloudOutcome = describeCloudOutcome(parse);
    container.dataset.cloudOutcome = cloudOutcome.raw;
    container.dataset.cloudOutcomeCode = cloudOutcome.code || '';

    if (parse.summary) {
        const summary = document.createElement('p');
        summary.className = 'parse-summary';
        summary.textContent = String(parse.summary);
        container.appendChild(summary);
    }

    const facts = document.createElement('div');
    facts.className = 'parse-facts';
    appendFact(facts, t('parse.field.parser'), String(parse.parsed_by || t('parse.value.not_reported')), 'neutral');
    if (parse.media_type) {
        appendFact(facts, t('parse.field.media_type'), String(parse.media_type), 'neutral');
    }
    appendFact(facts, t('parse.field.status'), String(parse.status || t('parse.value.not_reported')), toneForStatus(parse.status));
    if (parse.cache_hit) {
        appendFact(facts, t('parse.field.cache'), t('parse.value.cache_hit'), 'neutral');
    }
    appendCloudOutcomeRow(facts, cloudOutcome);
    container.appendChild(facts);

    // --- the cloud question, in three parts --------------------------------
    if (parse.used_cloud) {
        appendNotice(container, t('parse.notice.used_cloud'), 'cloud');
    } else if (parse.cloud_attempted) {
        appendNotice(container, t('parse.notice.cloud_discarded'), 'warn');
    } else if (parse.touched_network === false) {
        appendNotice(container, t('parse.notice.local_only'), 'local');
    }

    if (parse.fallback_used) {
        appendNotice(container, t('parse.notice.fallback'), 'warn');
    }

    // The row above says what happened; this says what to do about it, and
    // offers the retry only when the code's policy says one is worth offering.
    reportCloudOutcome(cloudOutcome);

    // --- quality -----------------------------------------------------------
    renderParseQuality(container, parse.quality);

    // --- warnings ----------------------------------------------------------
    const warnings = Array.isArray(parse.warnings) ? parse.warnings.filter(Boolean) : [];
    if (warnings.length) {
        const list = document.createElement('ul');
        list.className = 'parse-warning-list';
        warnings.forEach(message => {
            const item = document.createElement('li');
            item.textContent = String(message);
            list.appendChild(item);
        });
        container.appendChild(list);
    }

    if (parse.error_message) {
        appendNotice(container, t('parse.notice.failed', { message: String(parse.error_message) }), 'fail');
    }
}

/**
 * Render the quality gate's verdict, or say that it was not reported.
 *
 * The `ParseQualityReport` keeps its flags individually addressable on purpose
 * — "which check failed and on what numbers" is the only question a quality
 * report can answer — so the flags are listed one per row rather than folded
 * into a score. A score would be a threshold, and a threshold is somewhere this
 * project has promised not to put judgement.
 */
function renderParseQuality(container, quality) {
    if (!quality || typeof quality !== 'object') {
        appendNotice(container, t('parse.quality.absent'), 'unknown');
        return;
    }

    const status = String(quality.status || 'pass');
    const tone = toneForStatus(status);
    const heading = document.createElement('p');
    heading.className = 'parse-quality-heading';
    heading.dataset.tone = tone;
    heading.textContent = t('parse.quality.heading', { status });
    container.appendChild(heading);

    const flags = Array.isArray(quality.flags) ? quality.flags : [];
    if (!flags.length) {
        const clean = document.createElement('p');
        clean.className = 'parse-quality-clean';
        clean.textContent = t('parse.quality.clean');
        container.appendChild(clean);
        return;
    }

    const list = document.createElement('ul');
    list.className = 'parse-quality-flags';
    flags.forEach(flag => {
        const severity = String(flag?.severity || 'warn');
        const item = document.createElement('li');
        item.dataset.severity = severity;
        const rule = document.createElement('span');
        rule.className = 'parse-quality-rule';
        rule.textContent = String(flag?.rule_id || t('parse.quality.rule_fallback'));
        const message = document.createElement('span');
        message.className = 'parse-quality-message';
        message.textContent = String(flag?.message || '');
        item.append(rule, message);
        list.appendChild(item);
    });
    container.appendChild(list);
}

// --------------------------------------------------------------------------
// Reparse (T3.4)
// --------------------------------------------------------------------------

function buildReparseUrl(documentId) {
    return REPARSE_API_PATH.replace('{documentId}', encodeURIComponent(String(documentId || '')));
}

function renderParserOptions(select) {
    if (!select) {
        return;
    }
    const fragment = document.createDocumentFragment();
    REPARSE_PARSER_OPTIONS.forEach(option => {
        const element = document.createElement('option');
        element.value = option.value;
        // The label is a catalogue key, and it carries the *requirement* for the
        // choice as well as its name: see `REPARSE_PARSER_OPTIONS` in
        // `config.js` for why the options are labelled rather than filtered
        // against what this deployment happens to have installed.
        element.textContent = t(option.labelKey);
        element.dataset.leavesMachine = String(Boolean(option.leavesMachine));
        element.dataset.requirementKey = String(option.requirementKey || '');
        fragment.appendChild(element);
    });
    // The selected value is the reader's own choice and is restored after the
    // menu is rebuilt, so a language switch does not silently re-select the
    // first entry.
    const previous = String(select.value || '');
    select.replaceChildren(fragment);
    if (REPARSE_PARSER_OPTIONS.some(option => option.value === previous)) {
        select.value = previous;
    }
}

function currentParserOption(select) {
    const value = String(select?.value || '');
    return REPARSE_PARSER_OPTIONS.find(option => option.value === value) || REPARSE_PARSER_OPTIONS[0];
}

/**
 * Show the consequence of the option currently selected, before it is used.
 *
 * The cloud warning is the important one: it is shown the moment a cloud parser
 * is selected, not on submit, so the decision is made with the fact in front of
 * the reader rather than after it. A checkbox for forced OCR carries its own
 * cost warning for the same reason — forced OCR is slower and less accurate on
 * already-digital text, which is a trade the reader should be choosing.
 *
 * The selected option's own requirement rides along with it. Under the shipped
 * default (`PARSER_BACKEND=legacy`) every named parser answers `409
 * parser_unavailable`, and the browser is not told the parser catalogue, so the
 * only honest place for that fact is next to the choice — before the reader
 * commits to it, rather than after the server refuses it.
 */
function updateReparseWarnings() {
    const notice = document.getElementById('reparseWarning');
    const submit = document.getElementById('reparseSubmitBtn');
    const option = currentParserOption(document.getElementById('reparseParser'));
    const forceOcr = document.getElementById('reparseForceOcr')?.checked === true;

    const messages = [];
    if (option.requirementKey) {
        messages.push(t(option.requirementKey));
    }
    if (forceOcr) {
        messages.push(t('reparse.force_ocr_warning'));
    }
    if (option.leavesMachine) {
        messages.push(t('reparse.cloud_warning'));
    }

    if (notice) {
        notice.replaceChildren();
        notice.hidden = messages.length === 0;
        notice.dataset.leavesMachine = String(Boolean(option.leavesMachine));
        messages.forEach(message => {
            const paragraph = document.createElement('p');
            paragraph.className = 'reparse-warning-item';
            paragraph.textContent = message;
            notice.appendChild(paragraph);
        });
    }

    // The reparse control is only meaningful once there is a document id to
    // reparse. With none, the button is disabled rather than live-and-failing.
    const hasDocument = Boolean(state.currentDocumentId);
    setControlDisabled(submit, !hasDocument);
    if (submit) {
        submit.title = hasDocument
            ? t('reparse.hint.ready')
            : t('reparse.hint.no_document');
    }
}

/**
 * The last reparse result, kept as a key rather than as a rendered string.
 *
 * A rendered string would freeze the sentence in the language that was active
 * when the reparse ran, which is the same staleness this module's locale
 * subscription exists to remove everywhere else.
 */
let lastReparseResult = null;

function setReparseResult(key, params, tone) {
    const box = document.getElementById('reparseResult');
    if (!box) {
        return;
    }
    lastReparseResult = key ? { key, params: params || null, tone: tone || 'idle' } : null;
    box.replaceChildren();
    box.dataset.tone = tone || 'idle';
    if (!key) {
        box.hidden = true;
        return;
    }
    const paragraph = document.createElement('p');
    paragraph.textContent = t(key, params);
    box.appendChild(paragraph);
    box.hidden = false;
}

function toggleReparseControls(forceOpen) {
    const controls = document.getElementById('reparseControls');
    const button = document.getElementById('reparseBtn');
    if (!controls) {
        return;
    }
    const open = typeof forceOpen === 'boolean' ? forceOpen : controls.hidden;
    controls.hidden = !open;
    controls.setAttribute('aria-hidden', String(!open));
    button?.setAttribute('aria-expanded', String(open));
    if (open) {
        updateReparseWarnings();
    }
}

async function submitReparse() {
    const submit = document.getElementById('reparseSubmitBtn');
    const documentId = state.currentDocumentId;
    if (!documentId) {
        setReparseResult('reparse.result.no_document', null, 'warn');
        return;
    }

    if (state.reparseController) {
        state.reparseController.abort();
    }
    state.reparseController = new AbortController();
    setControlDisabled(submit, true);
    setReparseResult('reparse.result.started', null, 'idle');
    updateStatus(t('reparse.status.started'), 'idle');

    const parser = String(document.getElementById('reparseParser')?.value || '');
    const forceOcr = document.getElementById('reparseForceOcr')?.checked === true;

    try {
        // Captured before the request: the parser choice and the OCR checkbox are
        // the two things a reparse failure would most cost the reader to re-enter,
        // and a retry of a *different* reparse is not a retry.
        const preserved = capturePreservedFields();
        const payload = await apiRequest(buildReparseUrl(documentId), {
            method: 'POST',
            body: {
                parser,
                force_ocr: forceOcr,
                session_id: state.currentSessionId,
                session_token: state.currentSessionToken
            },
            cache: 'no-store',
            operation: 'reparse',
            signal: state.reparseController.signal
        });

        if (payload?.parse && typeof payload.parse === 'object' && Object.keys(payload.parse).length) {
            renderParseStatus(payload.parse);
        }
        setReparseResult('reparse.result.finished', null, 'pass');
        updateStatus(t('reparse.status.finished'), 'success');
    } catch (error) {
        if (error.name === 'AbortError' || error.code === 'client_cancelled') {
            return;
        }
        // Every failure leaves the document exactly as it was — nothing above
        // this branch writes to `state.currentParseProvenance` or re-renders the
        // status card, and nothing below it will. The message says so in words
        // rather than leaving the reader to infer it from an unchanged card.
        //
        // A 404 is named separately because it is the one a client cannot fix by
        // retrying: either the server does not serve the reparse route at all,
        // or this document id is not one it knows. We do not guess which — they
        // call for different responses — but both are reported as "nothing was
        // reparsed, the document is unchanged". A button that looks armed and
        // silently does nothing is the failure this product is built against, so
        // the route being absent is a visible outcome. The other statuses (a 409
        // while a parse is in flight, a 500) get the same honest "left
        // untouched" contract with the reason the server gave attached, so a
        // non-404 failure is never misreported as a missing route.
        restorePreservedFields(preserved);
        const reason = error.message || `${error.status || ''} ${error.code || ''}`.trim();
        if (error.status === 404) {
            setReparseResult(
                'reparse.result.no_route',
                { path: buildReparseUrl(documentId), reason },
                'fail'
            );
        } else {
            setReparseResult('reparse.result.failed', { reason }, 'fail');
        }
        // The runtime adds the notice with a retry button only if the code is
        // retryable, so a 404 (route absent) and a 409 (busy) cannot both end up
        // offering the reader the same next step.
        notifyFailure(error, { onRetry: () => submitReparse() });
        updateStatus(t('reparse.status.unavailable'), 'error');
    } finally {
        state.reparseController = null;
        setControlDisabled(submit, Boolean(state.currentDocumentId));
    }
}

function handleReparseKeydown(event) {
    if (event.key === 'Enter' && event.target?.id === 'reparseParser') {
        event.preventDefault();
        submitReparse();
    }
}

function initializeParseStatus() {
    renderParserOptions(document.getElementById('reparseParser'));
    document.getElementById('reparseBtn')?.addEventListener('click', () => toggleReparseControls());
    document.getElementById('reparseSubmitBtn')?.addEventListener('click', submitReparse);
    document.getElementById('reparseParser')?.addEventListener('change', updateReparseWarnings);
    document.getElementById('reparseForceOcr')?.addEventListener('change', updateReparseWarnings);
    document.getElementById('reparseControls')?.addEventListener('keydown', handleReparseKeydown);
    toggleReparseControls(false);
    renderParseStatus(null);
    // The card's sentences are produced at render time, so a language switch has
    // to be followed by a re-render or the reader is left looking at the other
    // language. Subscribed once, behind a flag, because
    // `initializeParseStatus` is callable more than once and a second
    // subscription would double every render.
    //
    // Re-rendering also re-raises the notice rather than leaving it in the
    // previous language: `reportCloudOutcome` dismisses the old one first, so the
    // outcome is described once, in the reader's current language, rather than
    // twice in two.
    //
    // The reparse control is part of the same card and had the same problem in a
    // second form: its option labels and its requirement notice are built once,
    // so a switch left the menu describing itself in the old language while the
    // card beside it had already repainted. Both are rebuilt here — the menu
    // preserving the reader's own selection — and the last result line is
    // re-resolved from its key rather than from the string it rendered before.
    if (!cloudOutcomeLocaleBound) {
        cloudOutcomeLocaleBound = true;
        onLocaleChange(() => {
            renderParserOptions(document.getElementById('reparseParser'));
            updateReparseWarnings();
            if (lastReparseResult) {
                setReparseResult(lastReparseResult.key, lastReparseResult.params, lastReparseResult.tone);
            }
            renderParseStatus(state.currentParseProvenance);
        });
    }
}

function resetParseStatus() {
    dismissCloudOutcomeNotice();
    toggleReparseControls(false);
    setReparseResult(null, null, 'idle');
    renderParseStatus(null);
}


export { describeCloudOutcome };
export { initializeParseStatus };
export { renderParseQuality };
export { renderParseStatus };
export { resetParseStatus };
export { submitReparse };
export { toggleReparseControls };
