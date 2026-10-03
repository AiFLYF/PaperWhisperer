import {
    ANSWER_MODE_LABELS,
    EVIDENCE_API_PATH,
    EVIDENCE_DEEP_LINK_PARAMS,
    EXPORT_FACT_FIELD_LABELS,
    EXPORT_FORMATS,
    EXPORT_PRESENTATION_LABELS,
    EXPORT_STATUS_LABELS,
    EXPORT_UNKNOWN_VALUE,
    PRESENTATION_DIRECT_QUOTE,
    PRESENTATION_PARAPHRASE
} from './config.js';
import { formatDateTime, getLocale, onLocaleChange, t } from './i18n.js';
import { state } from './state.js';
import { scheduleButtonFeedbackReset, setControlDisabled, updateStatus } from './dom.js';
import { normalizeSmartActions, normalizeSmartTextItems, resetSmartSuggestions } from './suggestions.js';
import { getCurrentSvgSource } from './mermaid.js';
import { escapeHtml } from './sanitize.js';
import { formatContent } from './format.js';

/** Session report composition, the export preview and file downloads. */

/**
 * `t()`, with a floor under it.
 *
 * The same helper `evidenceViewer.js` and `paperList.js` define, for the same
 * reason. Most of this module's copy used to be a `config.js` constant or an
 * inline literal, and every one of those is the zh-CN floor: a catalogue entry
 * that goes missing degrades to the sentence the report was written with, never
 * to a dotted key inside a file someone may open in six months. `t()` itself
 * falls back to the Chinese catalogue and then to the key, and neither of those
 * is what a reader looking at an exported table wants to read.
 */
function localised(key, floor) {
    const value = t(key);
    return value === key ? floor : value;
}

/**
 * The four prefixes this module builds keys from at run time.
 *
 * They exist because four vocabularies arrive as *values* rather than as
 * literals: a fact field named by `field_name`, a presentation named by
 * `PRESENTATION_*`, a fact status named by the payload, and an answer mode named
 * by the transcript entry. A key assembled from one of those is invisible to any
 * grep of `t('…')` calls, so `tests/test_i18n.py` asserts the vocabularies on
 * both sides — the `config.js` tables and the catalogue — are the same set.
 */
const FACT_FIELD_KEY_PREFIX = 'export.fact.';
const PRESENTATION_KEY_PREFIX = 'export.presentation.';
const STATUS_KEY_PREFIX = 'export.status.';
const ANSWER_MODE_KEY_PREFIX = 'mode.';

/**
 * Every export is one of two things: the report, and the record of what
 * produced it.
 *
 * T3.5's acceptance criterion is "导出物可追溯生成条件" — someone opening an
 * exported file six months later can tell what generated it and from which
 * version of which document. That is a different requirement from "the report
 * is complete", and it is the one that was actually missing: the report listed
 * the summary and the citations and nothing about the citation *provenance*,
 * so a file that was in a reader's downloads folder was indistinguishable from
 * one regenerated today with a different model.
 *
 * Two rules make the traceability real rather than decorative:
 *
 * 1. **Nothing is omitted for being unknown.** Every provenance field renders
 *    as `（未记录）` when the analysis did not record it, because a field that
 *    silently disappears reads as "not important" and a field that is
 *    explicitly empty reads as "this run did not record it". The second is
 *    information; the first is a gap.
 * 2. **A paraphrase never acquires quotation marks on the way out.** In the
 *    app the distinction is enforced by `claims.js` refusing to emit the mark
 *    constants; an exported file has no such guard, and a downstream reader
 *    (a co-author, a reviewer's PDF, a grep for "…") has no way to tell which
 *    of our sentences are the paper's. So every claim in the citation index
 *    carries its presentation explicitly, and the paraphrase is rendered with
 *    `presentation: paraphrase` and no quotation marks anywhere in the
 *    output. This is the T1.6 release gate at its weakest link, and the reason
 *    it is worth building here rather than trusting the pipeline to stay
 *    verified forever.
 */

function setExportEnabled(enabled) {
    setControlDisabled(document.getElementById('exportBtn'), !enabled);
}

function resetExportState() {
    state.currentAnalysisResult = null;
    state.currentSourceFileName = '';
    state.currentElapsedSeconds = null;
    state.currentOutputFile = '';
    state.currentChatTurns = [];
    state.currentSections = {};
    state.currentSessionToken = '';
    resetSmartSuggestions();
    renderExportPreview();
    setExportEnabled(false);
}

function sanitizeFileStem(name) {
    return String(name || 'paperwhisperer_session')
        .replace(/\.[^.]+$/, '')
        .replace(/[^\w\u4e00-\u9fa5-]+/g, '_')
        .replace(/^_+|_+$/g, '') || 'paperwhisperer_session';
}

/**
 * A timestamp, in the active locale's date and time convention.
 *
 * This is the exported file's own "when", so it is resolved through
 * `formatDateTime` rather than `Date.prototype.toLocaleString()`: the latter
 * follows the *browser's* preference, which is not the language the reader chose
 * for this interface and not the language the rest of the file is being written
 * in. A provenance table whose one timestamp is in a third language is exactly
 * the mixed artefact the catalogue exists to prevent.
 */
function formatExportTimestamp(value) {
    const date = value ? new Date(value) : new Date();
    if (Number.isNaN(date.getTime())) {
        return formatDateTime(new Date());
    }
    return formatDateTime(date);
}

function formatMarkdownList(items) {
    const normalizedItems = normalizeSmartTextItems(items, 12);
    return normalizedItems.length ? normalizedItems.map(item => `- ${item}`).join('\n') : t('export.list.none');
}

function formatMarkdownActions(actions) {
    const normalizedActions = normalizeSmartActions(actions, 8);
    return normalizedActions.length
        ? normalizedActions.map(action => `- **${action.label}**: ${action.prompt}`).join('\n')
        : t('export.list.none');
}

function formatWorkspaceGuidanceMarkdown(items) {
    return items.map(item => `- **${item.label}**: ${item.value} — ${item.detail}`).join('\n');
}

/**
 * The per-section status table.
 *
 * The `| --- |` separator is markdown syntax and stays here; the header and the
 * four row labels are the report's, so they are keyed. `'None'` and `'N/A'` are
 * *cell values* rather than labels, and they are resolved through the catalogue
 * for the same reason as the provenance table: an English report that says
 * `Completed | 4, None` is a report nobody can read.
 */
function formatSectionStatusTable(status) {
    const none = t('export.section_table.none');
    const na = t('export.na');
    const completed = Array.isArray(status.completed_sections) ? status.completed_sections.join(', ') : na;
    const failed = Array.isArray(status.failed_sections) && status.failed_sections.length ? status.failed_sections.join(', ') : none;
    const disabled = Array.isArray(status.disabled_sections) && status.disabled_sections.length ? status.disabled_sections.join(', ') : none;
    return [
        `| ${t('export.section_table.header')} | ${t('export.section_table.sections')} |`,
        '| --- | --- |',
        `| ${t('export.section_table.completed')} | ${completed || none} |`,
        `| ${t('export.section_table.failed')} | ${failed} |`,
        `| ${t('export.section_table.disabled')} | ${disabled} |`,
        `| ${t('export.section_table.quality')} | ${status.quality || na} |`
    ].join('\n');
}

function formatPaperTrace(items) {
    const normalizedItems = Array.isArray(items) ? items.slice(0, 8) : [];
    if (!normalizedItems.length) return t('export.list.none');
    return normalizedItems.map((item, index) => {
        const bits = [item.source, item.year, item.venue].filter(Boolean).join(' · ');
        const suffix = bits ? ` — ${bits}` : '';
        return `${index + 1}. ${item.title || t('export.paper.untitled')}${suffix}`;
    }).join('\n');
}

// --------------------------------------------------------------------------
// T3.5 — traceability: the citation index and the run provenance
// --------------------------------------------------------------------------

/**
 * The two URLs a citation carries, and why there are two.
 *
 * `api_url` is the Evidence API endpoint: machine-checkable, stable, and the
 * thing a reader can curl to confirm that the citation still resolves to the
 * passage. `deep_link` is this application with the two ids in its fragment,
 * which `evidence.js` reads on load to open the panel on that citation —
 * without it a link in an exported file points at JSON, and a reader who
 * cannot follow it has to take the provenance on trust.
 *
 * Both are built from the shared constants in `config.js` rather than written
 * out here, so the path an export advertises and the path the app calls cannot
 * be two different paths.
 */
function buildEvidenceApiUrl(documentId, evidenceId) {
    return EVIDENCE_API_PATH
        .replace('{documentId}', encodeURIComponent(String(documentId || '')))
        .replace('{evidenceId}', encodeURIComponent(String(evidenceId || '')));
}

function buildCitationDeepLink(documentId, evidenceId) {
    const params = new URLSearchParams();
    params.set(EVIDENCE_DEEP_LINK_PARAMS.evidence, String(evidenceId || ''));
    if (documentId) {
        params.set(EVIDENCE_DEEP_LINK_PARAMS.document, String(documentId));
    }
    const base = `${window.location.origin}${window.location.pathname}`;
    return `${base}#${params.toString()}`;
}

/**
 * Every claim in this analysis that carries at least one citation, with its
 * presentation stated.
 *
 * This is the export's version of the release gate. A claim with a citation is
 * either a span the verifier aligned inside a cited passage — a direct quote,
 * which is the paper's characters and may be shown as one — or a fact whose
 * `value` is our own sentence about the paper, which may not. The two arrive
 * in the payload in different places (`fact_extraction.verified_quotes` and
 * `paper_facts` respectively) and are joined here by `field_name` +
 * `evidence_id`, so a fact that aligned gets the quote's presentation and a
 * fact that did not gets the paraphrase's.
 *
 * A citation whose evidence id is not in this analysis is still emitted, with
 * the id it named. Dropping it would make the index look complete while
 * quietly omitting exactly the entries a reader needs to check.
 */
function collectExportCitations(result) {
    const payload = result || {};
    const documentId = String(payload.document_id || '');
    const verified = Array.isArray(payload.fact_extraction?.verified_quotes)
        ? payload.fact_extraction.verified_quotes
        : [];
    const verifiedByKey = new Map();
    verified.forEach(item => {
        const fieldName = String(item?.field_name || '');
        String(item?.evidence_id || '').split(',').map(value => value.trim()).filter(Boolean)
            .forEach(evidenceId => {
                verifiedByKey.set(`${fieldName}:${evidenceId}`, item);
            });
    });

    const entries = [];
    const seen = new Set();

    const push = (entry) => {
        const key = `${entry.field_name}:${entry.presentation}:${entry.text}`;
        if (seen.has(key)) {
            return;
        }
        seen.add(key);
        entries.push(entry);
    };

    // Verified quotes first: they are the paper's own words, and a reader
    // looking for the quotations in this file should find them at the top.
    verified.forEach(item => {
        const quote = String(item?.quote || '').trim();
        if (!quote) {
            return;
        }
        String(item?.evidence_id || '').split(',').map(value => value.trim()).filter(Boolean)
            .forEach(evidenceId => {
                push({
                    field_name: String(item?.field_name || ''),
                    text: quote,
                    presentation: PRESENTATION_DIRECT_QUOTE,
                    status: 'verified',
                    evidence_id: evidenceId,
                    api_url: buildEvidenceApiUrl(documentId, evidenceId),
                    deep_link: buildCitationDeepLink(documentId, evidenceId)
                });
            });
    });

    const facts = payload.paper_facts && typeof payload.paper_facts === 'object' ? payload.paper_facts : {};
    Object.entries(facts).forEach(([key, value]) => {
        const items = Array.isArray(value) ? value : (value && typeof value === 'object' ? [value] : []);
        items.forEach((fact, index) => {
            if (!fact || typeof fact !== 'object') {
                return;
            }
            const fieldName = String(fact.field_name || key);
            const text = String(fact.value || '').trim();
            const evidenceIds = Array.isArray(fact.evidence_ids) ? fact.evidence_ids : [];
            if (!text || !evidenceIds.length) {
                return;
            }
            evidenceIds.forEach(evidenceId => {
                const matched = verifiedByKey.get(`${fieldName}:${evidenceId}`);
                push({
                    field_name: fieldName,
                    text: matched ? String(matched.quote || text).trim() : text,
                    presentation: matched ? PRESENTATION_DIRECT_QUOTE : PRESENTATION_PARAPHRASE,
                    status: String(fact.extraction_status || ''),
                    evidence_id: String(evidenceId),
                    api_url: buildEvidenceApiUrl(documentId, evidenceId),
                    deep_link: buildCitationDeepLink(documentId, evidenceId),
                    order: index
                });
            });
        });
    });

    return entries;
}

/**
 * The eight fact-layer field names, as the label a reader sees.
 *
 * `field_name` is the schema's own vocabulary, so a field this build does not
 * carry a label for still exports under its raw name rather than being dropped —
 * see `EXPORT_FACT_FIELD_LABELS`. Which language that raw name is *read in* is
 * the interface's, so the catalogue is consulted first and the schema's name is
 * the floor.
 */
function factFieldLabel(fieldName) {
    const raw = String(fieldName || '');
    return localised(`${FACT_FIELD_KEY_PREFIX}${raw}`, EXPORT_FACT_FIELD_LABELS[raw] || raw);
}

/** How a claim is labelled in an export, by the presentation constant itself. */
function presentationLabel(presentation) {
    const raw = String(presentation || '');
    return localised(`${PRESENTATION_KEY_PREFIX}${raw}`, EXPORT_PRESENTATION_LABELS[raw] || raw);
}

/**
 * A fact's extraction status, as an export states it.
 *
 * A status this build has no entry for is reported under its raw name rather
 * than as a dotted key: the export is the record of what happened, and inventing
 * a sentence for a status the catalogue has never heard of would be a guess
 * about a run nobody can reconstruct.
 */
function statusLabel(status) {
    const raw = String(status || '');
    return localised(`${STATUS_KEY_PREFIX}${raw}`, EXPORT_STATUS_LABELS[raw] || raw || unknownValue());
}

/** The answer mode a transcript entry was filed under. */
function answerModeLabel(answerMode) {
    const raw = String(answerMode || '');
    return localised(
        `${ANSWER_MODE_KEY_PREFIX}${raw}`,
        ANSWER_MODE_LABELS[raw] || ANSWER_MODE_LABELS.evidence
    );
}

function formatCitationMarkdown(entry, index) {
    const label = factFieldLabel(entry.field_name) || t('export.fact.unnamed');
    const presentation = presentationLabel(entry.presentation);
    // The quotation marks live in the *text*, taken verbatim from the verifier,
    // and nowhere else. A paraphrase entry reaches this line without them,
    // because nothing in the paraphrase path may produce one.
    const text = entry.presentation === PRESENTATION_DIRECT_QUOTE ? `“${entry.text}”` : entry.text;
    const status = statusLabel(entry.status);
    return [
        `### ${index}. ${label}`,
        '',
        t('export.citation.presentation', { value: presentation }),
        t('export.citation.status', { value: status }),
        t('export.citation.evidence_id', { value: entry.evidence_id }),
        t('export.citation.links', { deep_link: entry.deep_link, api_url: entry.api_url }),
        '',
        text,
        ''
    ].join('\n');
}

function formatCitationIndexMarkdown(result) {
    const entries = collectExportCitations(result);
    if (!entries.length) {
        return Boolean(result?.paper_facts) ? t('export.citations.none') : t('export.citations.no_facts');
    }
    const lines = [
        `> ${t('export.citations.policy_note')}`,
        ''
    ];
    entries.forEach((entry, index) => {
        lines.push(formatCitationMarkdown(entry, index + 1));
    });
    return lines.join('\n');
}

/**
 * What an export writes where a run recorded nothing, rather than omitting it.
 *
 * Resolved through the catalogue so the exported file says it in the reader's
 * language, with `EXPORT_UNKNOWN_VALUE` kept as the floor beneath it for the
 * same reason every other floor here is kept.
 */
function unknownValue() {
    return localised('export.unknown_value', EXPORT_UNKNOWN_VALUE);
}

/**
 * The run provenance, as rows.
 *
 * A fixed list, and every entry is always present. A provenance table that
 * drops the fields a particular run did not record is a table whose absence
 * is ambiguous: the reader cannot tell "this run had no cost" from "this build
 * does not write cost down". `EXPORT_UNKNOWN_VALUE` makes the difference
 * visible, which is the whole point of writing the table down at all.
 *
 * **Every label is resolved at export time, not at module load.** The labels are
 * written *into* the downloaded file, so the language they arrive in has to be
 * the one the reader was reading when they pressed export — not the one that
 * happened to be active when this module was first parsed. That is why these are
 * `t(...)` calls inside the function body and not constants beside it: a
 * module-level table would freeze the first locale it ever saw.
 *
 * The *values* are the run's and are never rewritten: a model name, a token
 * count and a run id are recorded in whatever form the server sent them, and
 * translating them would be a different artefact.
 */
function buildProvenanceRows(result, exportedAt) {
    const payload = result || {};
    const run = payload.analysis_run && typeof payload.analysis_run === 'object' ? payload.analysis_run : {};
    const parse = payload.parse && typeof payload.parse === 'object' ? payload.parse : {};
    const facts = payload.paper_facts && typeof payload.paper_facts === 'object' ? payload.paper_facts : {};
    const value = (input) => {
        if (input === undefined || input === null || input === '') {
            return unknownValue();
        }
        return String(input);
    };
    const flag = (input) => (input === true
        ? t('export.value.yes')
        : (input === false ? t('export.value.no') : unknownValue()));
    const cost = typeof run.llm_cost === 'number' ? run.llm_cost.toFixed(6) : unknownValue();
    const coverage = run.fact_count
        ? `${run.grounded_fact_count ?? 0}/${run.fact_count}`
        : unknownValue();
    // Server-supplied lists joined with a localised separator. The separator is
    // the exporter's punctuation, not the server's, so it follows the switch;
    // the items themselves are passed through untouched.
    const notes = (input) => (Array.isArray(input) && input.length
        ? input.join(t('export.provenance.join'))
        : unknownValue());

    return [
        [t('export.provenance.row.exported_at'), formatExportTimestamp(exportedAt)],
        [t('export.provenance.row.source_file'), value(payload.source_filename)],
        [t('export.provenance.row.document_id'), value(payload.document_id)],
        [t('export.provenance.row.session_id'), value(payload.session_id)],
        [t('export.provenance.row.run_id'), value(run.run_id)],
        [t('export.provenance.row.model'), value(run.model || facts.extraction_model)],
        [t('export.provenance.row.prompt_version'), value(run.prompt_version || facts.prompt_version)],
        [t('export.provenance.row.evidence_version'), value(run.evidence_version || facts.evidence_version)],
        [t('export.provenance.row.llm_calls'), value(run.llm_calls)],
        [t('export.provenance.row.input_tokens'), value(run.llm_input_tokens)],
        [t('export.provenance.row.output_tokens'), value(run.llm_output_tokens)],
        [t('export.provenance.row.cost'), cost],
        [t('export.provenance.row.fact_coverage'), coverage],
        [t('export.provenance.row.parser'), value(parse.parsed_by)],
        [t('export.provenance.row.media_type'), value(parse.media_type)],
        [t('export.provenance.row.used_cloud'), flag(parse.used_cloud)],
        [t('export.provenance.row.fallback_used'), flag(parse.fallback_used)],
        [t('export.provenance.row.cache_hit'), flag(parse.cache_hit)],
        [t('export.provenance.row.degradations'), notes(run.degradations)],
        [t('export.provenance.row.warnings'), notes(parse.warnings)]
    ];
}

function formatProvenanceMarkdown(result, exportedAt) {
    const rows = buildProvenanceRows(result, exportedAt);
    return [
        `| ${t('export.provenance.table_header')} | ${t('export.provenance.value_header')} |`,
        '| --- | --- |',
        ...rows.map(([term, detail]) => `| ${term} | ${detail} |`)
    ].join('\n');
}

/**
 * The three guidance cards, shared by the on-screen panel and the exported file.
 *
 * The same three appear in both places, which is why they are built here rather
 * than written out twice: a reader who reads "2 sections ready" in the panel and
 * then finds a different sentence in the downloaded file has been told two
 * things about one run. The labels, the copy and the punctuation all follow the
 * switch; the counts and the next action's own label are the run's.
 */
function getWorkspaceGuidanceItems(status, completed) {    const failed = Array.isArray(status.failed_sections) ? status.failed_sections.length : 0;
    const disabled = Array.isArray(status.disabled_sections) ? status.disabled_sections.length : 0;
    const paperLeadCount = state.currentPaperSearchResults.length + state.currentPaperRecommendations.length;
    const nextAction = state.currentNextActions[0];
    return [
        {
            label: t('export.guidance.state.label'),
            value: failed
                ? t('export.guidance.state.ready_review', { ready: completed, review: failed })
                : t('export.guidance.state.ready', { count: completed }),
            detail: disabled
                ? t('export.guidance.state.disabled', { count: disabled })
                : t('export.guidance.state.core')
        },
        {
            label: t('export.guidance.next.label'),
            // The action's own label is the payload's words, shown as sent; only
            // the fallback for "no action was suggested" is ours.
            value: nextAction?.label || t('export.guidance.next.fallback'),
            detail: nextAction?.prompt || t('export.guidance.next.detail_fallback')
        },
        {
            label: t('export.guidance.trail.label'),
            value: state.currentReadingQueue.length
                ? t('export.guidance.trail.saved', { count: state.currentReadingQueue.length })
                : t('export.guidance.trail.leads', { count: paperLeadCount }),
            detail: state.currentReadingQueue.length
                ? t('export.guidance.trail.saved_detail')
                : t('export.guidance.trail.detail')
        }
    ];
}

function renderExportPreview() {
    const preview = document.getElementById('exportPreview');
    if (!preview) return;
    if (!state.currentAnalysisResult) {
        const empty = document.createElement('p');
        empty.className = 'empty-state';
        empty.textContent = t('export.preview_empty');
        preview.replaceChildren(empty);
        return;
    }

    const status = state.currentAnalysisResult.analysis_status || {};
    const completed = Array.isArray(status.completed_sections) ? status.completed_sections.length : Object.keys(state.currentSections || {}).length;
    const failed = Array.isArray(status.failed_sections) ? status.failed_sections.length : 0;
    const paperLeadCount = state.currentPaperSearchResults.length + state.currentPaperRecommendations.length;
    const exportPreviewItems = [
        {
            label: t('export.preview.card.analysis'),
            value: t('export.preview.card.analysis_value', { count: completed }),
            detail: failed
                ? t('export.preview.card.analysis_review', { count: failed })
                : t('export.preview.card.analysis_ok')
        },
        {
            label: t('export.preview.card.qa'),
            value: t('export.preview.card.qa_value', { count: state.currentChatTurns.length }),
            detail: state.currentChatTurns.length
                ? t('export.preview.card.qa_included')
                : t('export.preview.card.qa_empty')
        },
        {
            label: t('export.preview.card.papers'),
            value: t('export.preview.card.papers_value', { count: paperLeadCount }),
            detail: state.currentReadingQueue.length
                ? t('export.preview.card.papers_saved', { count: state.currentReadingQueue.length })
                : t('export.preview.card.papers_empty')
        },
        {
            label: t('export.preview.card.visuals'),
            value: state.currentMermaidSource
                ? t('export.preview.card.map_ready')
                : t('export.preview.card.no_map'),
            detail: state.currentMermaidSource
                ? t('export.preview.card.svg_ready')
                : t('export.preview.card.visuals_empty')
        }
    ];
    const guidanceItems = getWorkspaceGuidanceItems(status, completed);

    const grid = document.createElement('div');
    grid.className = 'export-preview-grid';
    grid.setAttribute('aria-label', t('export.preview.grid_label'));
    exportPreviewItems.forEach(item => {
        const card = document.createElement('div');
        card.className = 'export-preview-card';
        const label = document.createElement('span');
        label.textContent = item.label;
        const value = document.createElement('strong');
        value.textContent = item.value;
        const detail = document.createElement('small');
        detail.textContent = item.detail;
        card.append(label, value, detail);
        grid.appendChild(card);
    });

    const guidance = document.createElement('div');
    guidance.className = 'workspace-guidance';
    guidance.setAttribute('aria-label', t('export.preview.guidance_label'));
    guidanceItems.forEach(item => {
        const card = document.createElement('div');
        card.className = 'workspace-guidance-item';
        const label = document.createElement('span');
        label.textContent = item.label;
        const value = document.createElement('strong');
        value.textContent = item.value;
        const detail = document.createElement('small');
        detail.textContent = item.detail;
        card.append(label, value, detail);
        guidance.appendChild(card);
    });

    const note = document.createElement('p');
    note.className = 'export-preview-note';
    note.textContent = t('export.preview.note');
    preview.replaceChildren(grid, guidance, note);
}

function buildSessionMarkdown() {
    if (!state.currentAnalysisResult) return '';

    const reportTime = new Date();
    const svgSource = getCurrentSvgSource();
    const fileStem = sanitizeFileStem(state.currentSourceFileName || state.currentAnalysisResult.session_id || 'paperwhisperer_session');
    const svgFileName = `${fileStem}_visual_map.svg`;
    const status = state.currentAnalysisResult.analysis_status || {};
    const completed = Array.isArray(status.completed_sections) ? status.completed_sections.length : Object.keys(state.currentSections || {}).length;
    const guidanceItems = getWorkspaceGuidanceItems(status, completed);
    const lines = [
        `# ${t('export.report.title')}`,
        '',
        `> ${t('export.report.subtitle')}`,
        `> ${t('export.report.generated_by')}`,
        '',
        '---',
        '',
        `## ${t('export.report.overview')}`,
        '',
        `| ${t('export.report.table_header')} | ${t('export.provenance.value_header')} |`,
        '| --- | --- |',
        `| ${t('export.report.row.source_file')} | ${state.currentSourceFileName || t('export.na')} |`,
        `| ${t('export.report.row.session_id')} | ${state.currentAnalysisResult.session_id || state.currentSessionId || t('export.na')} |`,
        `| ${t('export.report.row.generated_at')} | ${formatExportTimestamp(reportTime.toISOString())} |`,
        `| ${t('export.report.row.duration')} | ${t('export.report.duration_value', { value: state.currentElapsedSeconds ?? t('export.na') })} |`,
        `| ${t('export.report.row.char_count')} | ${state.currentAnalysisResult.char_count ?? t('export.na')} |`,
        `| ${t('export.report.row.qa_turns')} | ${state.currentChatTurns.length} |`,
        `| ${t('export.report.row.saved_papers')} | ${state.currentReadingQueue.length} |`,
        '',
        `### ${t('export.report.guidance')}`,
        '',
        formatWorkspaceGuidanceMarkdown(guidanceItems),
        '',
        `### ${t('export.report.section_status')}`,
        '',
        formatSectionStatusTable(status),
        '',
        '---',
        '',
        `## ${t('export.report.section.overview')}`,
        '',
        state.currentAnalysisResult.summary || t('export.report.no_summary'),
        '',
        '---',
        '',
        `## ${t('export.report.section.citations')}`,
        '',
        state.currentAnalysisResult.quotes || t('export.report.no_citations'),
        '',
        '---',
        '',
        `## ${t('export.report.section.structure')}`,
        '',
        state.currentAnalysisResult.mindmap || t('export.report.no_structure')
    ];

    if (state.currentAnalysisResult.evaluation) {
        lines.push('', '---', '', `## ${t('export.report.section.evaluation')}`, '', state.currentAnalysisResult.evaluation);
    }

    if (state.currentAnalysisResult.research_brief) {
        lines.push('', '---', '', `## ${t('export.report.section.brief')}`, '', state.currentAnalysisResult.research_brief);
    }

    if (state.currentSuggestedQuestions.length || state.currentNextActions.length) {
        lines.push(
            '',
            '---',
            '',
            `## ${t('export.report.section.followups')}`,
            '',
            `### ${t('export.report.section.questions')}`,
            '',
            formatMarkdownList(state.currentSuggestedQuestions),
            '',
            `### ${t('export.report.section.actions')}`,
            '',
            formatMarkdownActions(state.currentNextActions)
        );
    }

    if (state.currentPaperSearchResults.length || state.currentPaperRecommendations.length || state.currentReadingQueue.length) {
        lines.push(
            '',
            '---',
            '',
            `## ${t('export.report.section.trace')}`,
            '',
            `### ${t('export.report.section.queue')}`,
            '',
            formatPaperTrace(state.currentReadingQueue),
            '',
            `### ${t('export.report.section.search')}`,
            '',
            formatPaperTrace(state.currentPaperSearchResults),
            '',
            `### ${t('export.report.section.recommendations')}`,
            '',
            formatPaperTrace(state.currentPaperRecommendations)
        );
    }

    if (state.currentMermaidSource) {
        lines.push(
            '',
            '---',
            '',
            `## ${t('export.report.section.mermaid')}`,
            '',
            '```mermaid',
            state.currentMermaidSource,
            '```'
        );

        if (svgSource) {
            lines.push(
                '',
                `### ${t('export.report.section.svg')}`,
                '',
                t('export.report.svg_companion', { file: svgFileName })
            );
        }
    }

    if (state.currentChatTurns.length) {
        lines.push('', '---', '', `## ${t('export.report.section.qa')}`, '');
        state.currentChatTurns.forEach((turn, index) => {
            const modeLabel = answerModeLabel(turn.answer_mode);
            lines.push(
                `### Q${index + 1}`,
                '',
                t('export.report.turn_mode', { mode: modeLabel }),
                '',
                turn.question || t('export.report.no_question'),
                '',
                `### A${index + 1}`,
                '',
                turn.answer || t('export.report.no_answer'),
                ''
            );
        });
    }

    lines.push(
        '',
        '---',
        '',
        `## ${t('export.citations_heading')}`,
        '',
        formatCitationIndexMarkdown(state.currentAnalysisResult),
        '',
        '---',
        '',
        `## ${t('export.provenance_heading')}`,
        '',
        `> ${t('export.provenance.note')}`,
        '',
        formatProvenanceMarkdown(state.currentAnalysisResult, reportTime.toISOString()),
        '',
        '---',
        '',
        `## ${t('export.report.metadata')}`,
        '',
        `- ${t('export.report.app')}`,
        `- ${t('export.report.export_time', { time: formatExportTimestamp(reportTime.toISOString()) })}`
    );
    return lines.join('\n');
}

/**
 * The same report as a standalone HTML file.
 *
 * Built from the same functions as the Markdown rather than by converting it,
 * because a converter would have to decide what a `## ` inside a Chinese
 * sentence is, and the one thing this file must not do is reformat the claim
 * text. Every value goes through `escapeHtml`; the only markup that is not
 * escaped is the table and the headings this function writes itself. Model
 * text is rendered as paragraphs, never as HTML.
 *
 * **`<html lang>` comes from the active locale.** It used to be hardcoded
 * `zh-CN`, which misinforms a screen reader about the phonetics of every word
 * in an English export, and a browser's own translate prompt about the document
 * as a whole. The exported file is written in the reader's language, so it has to
 * declare that language.
 */
function buildSessionHtml() {
    if (!state.currentAnalysisResult) return '';
    const reportTime = new Date();
    const result = state.currentAnalysisResult;
    const provenanceRows = buildProvenanceRows(result, reportTime.toISOString());
    const citations = collectExportCitations(result);

    const section = (title, body) => [
        `<h2>${escapeHtml(title)}</h2>`,
        body
            ? formatContent(String(body), '')
            : `<p class="empty">${escapeHtml(t('export.html.section_empty'))}</p>`
    ].join('\n');

    const parts = [
        '<!DOCTYPE html>',
        `<html lang="${escapeHtml(getLocale())}">`,
        '<head>',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        `<title>${escapeHtml(sanitizeFileStem(state.currentSourceFileName || 'paperwhisperer_session'))}${escapeHtml(t('export.html.title_suffix'))}</title>`,
        '<style>',
        'body{font-family:system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif;',
        'max-width:52rem;margin:0 auto;padding:2rem 1.25rem;line-height:1.7;color:#1c1c1e;background:#fff;}',
        'h1{font-size:1.6rem}h2{font-size:1.15rem;margin-top:2.4rem;border-bottom:1px solid #e5e5ea;padding-bottom:.3rem;}',
        'table{border-collapse:collapse;width:100%;font-size:.9rem;}th,td{border:1px solid #d1d1d6;',
        'padding:.4rem .6rem;text-align:left;vertical-align:top;}th{background:#f5f5f7;width:12rem;}',
        'blockquote{border-left:3px solid #0a84ff;margin:.5rem 0;padding:.2rem .9rem;color:#3a3a3c;}',
        '.citation{border:1px solid #e5e5ea;border-radius:.4rem;padding:.7rem .9rem;margin:.6rem 0;}',
        '.citation[data-presentation="paraphrase"]{border-left:3px solid #ff9f0a;}',
        '.citation-meta{font-size:.82rem;color:#6e6e73;margin:0 0 .4rem;}',
        '.empty{color:#8e8e93;font-style:italic;}',
        '</style>',
        '</head>',
        '<body>',
        `<h1>${escapeHtml(t('export.html.heading'))}</h1>`,
        `<p class="citation-meta">${escapeHtml(t('export.html.exported_at', { time: formatExportTimestamp(reportTime.toISOString()) }))}</p>`,
        section(t('export.report.section.overview'), result.summary),
        section(t('export.report.section.citations'), result.quotes),
        section(t('export.report.section.structure'), result.mindmap)
    ];

    if (result.evaluation) {
        parts.push(section(t('export.report.section.evaluation'), result.evaluation));
    }
    if (result.research_brief) {
        parts.push(section(t('export.report.section.brief'), result.research_brief));
    }

    parts.push(
        `<h2>${escapeHtml(t('export.citations_heading'))}</h2>`,
        `<p class="citation-meta">${escapeHtml(t('export.citations.policy_note'))}</p>`
    );
    if (!citations.length) {
        parts.push(`<p class="empty">${escapeHtml(t('export.citations.none'))}</p>`);
    }
    citations.forEach((entry, index) => {
        const label = factFieldLabel(entry.field_name) || t('export.fact.unnamed');
        const presentation = presentationLabel(entry.presentation);
        const status = statusLabel(entry.status);
        // The `<blockquote>` below is emitted for a direct quote and NOT for a
        // paraphrase, and the element itself is the marker: a paraphrase in a
        // blockquote is a paraphrase wearing quotation marks, which is the one
        // thing the T1.6 gate forbids. `data-presentation` is on the element so
        // the distinction survives a stylesheet that does not load.
        const body = entry.presentation === PRESENTATION_DIRECT_QUOTE
            ? `<blockquote>${escapeHtml(entry.text)}</blockquote>`
            : `<p>${escapeHtml(entry.text)}</p>`;
        parts.push([
            `<div class="citation" data-presentation="${escapeHtml(entry.presentation)}">`,
            `<p class="citation-meta">${index + 1}. ${escapeHtml(label)} · ${escapeHtml(presentation)} · ${escapeHtml(status)}`,
            ` · <a href="${escapeHtml(entry.deep_link)}">${escapeHtml(t('export.citation.open_link'))}</a>`,
            ` · <a href="${escapeHtml(entry.api_url)}">Evidence API</a></p>`,
            body,
            '</div>'
        ].join('\n'));
    });

    parts.push(
        `<h2>${escapeHtml(t('export.provenance_heading'))}</h2>`,
        `<p class="citation-meta">${escapeHtml(t('export.provenance.note'))}</p>`,
        '<table>',
        t('export.provenance.html_header'),
        ...provenanceRows.map(([term, detail]) => (
            `<tr><th scope="row">${escapeHtml(term)}</th><td>${escapeHtml(detail)}</td></tr>`
        )),
        '</table>',
        '</body>',
        '</html>'
    );
    return parts.join('\n');
}

function scheduleObjectUrlRevoke(url, delay = 1000) {
    setTimeout(() => URL.revokeObjectURL(url), delay);
}

function triggerTextDownload(fileName, content, mimeType = 'text/plain;charset=utf-8') {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    try {
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
    } finally {
        a.remove();
        scheduleObjectUrlRevoke(url);
    }
}

/**
 * Populate the format control from `EXPORT_FORMATS`.
 *
 * Options are generated rather than written into `index.html` so that the
 * offered formats and the formats this module can actually produce are the
 * same list. A hard-coded `<option>` for a format the exporter does not
 * implement is a button that fails for the user, and the failure looks like a
 * broken export rather than a missing feature. DOCX is absent for that
 * reason — see `EXPORT_FORMATS`.
 */
/** The catalogue key a format's `<option>` is labelled through. */
function formatOptionKey(format) {
    return `export.format_option.${String(format?.value || '')}`;
}

function initializeExportFormats() {
    const select = document.getElementById('exportFormat');
    if (!select) {
        return 0;
    }
    if (select.options.length) {
        return select.options.length;
    }
    EXPORT_FORMATS.forEach(format => {
        const option = document.createElement('option');
        option.value = format.value;
        option.textContent = localised(formatOptionKey(format), format.label);
        select.appendChild(option);
    });
    select.dataset.formats = String(EXPORT_FORMATS.length);
    return EXPORT_FORMATS.length;
}

function resolveExportFormat() {
    const selected = String(document.getElementById('exportFormat')?.value || '');
    return EXPORT_FORMATS.find(format => format.value === selected) || EXPORT_FORMATS[0];
}

function exportSessionReport() {
    const exportBtn = document.getElementById('exportBtn');
    if (!state.currentAnalysisResult) {
        updateStatus(t('export.status.needs_analysis'), 'idle');
        return;
    }

    const fileStem = sanitizeFileStem(state.currentSourceFileName || state.currentAnalysisResult.session_id || 'paperwhisperer_session');
    const format = resolveExportFormat();
    const content = format.value === 'html'
        ? buildSessionHtml()
        : buildSessionMarkdown();
    if (!content) {
        updateStatus(t('export.status.not_ready'), 'error');
        return;
    }

    triggerTextDownload(`${fileStem}_session_report.${format.extension}`, content, format.mime);

    const svgSource = getCurrentSvgSource();
    if (svgSource) {
        triggerTextDownload(`${fileStem}_visual_map.svg`, svgSource, 'image/svg+xml;charset=utf-8');
    }

    if (exportBtn) {
        const originalText = exportBtn.textContent;
        exportBtn.textContent = svgSource ? t('export.button.done_svg') : t('export.button.done');
        exportBtn.classList.add('action-success');
        exportBtn.setAttribute('aria-busy', 'true');
        setControlDisabled(exportBtn, true);
        scheduleButtonFeedbackReset(exportBtn, () => {
            exportBtn.textContent = originalText;
            exportBtn.classList.remove('action-success');
            exportBtn.removeAttribute('aria-busy');
            setControlDisabled(exportBtn, false);
        }, 1800);
    }
    updateStatus(svgSource ? t('export.status.done_svg') : t('export.status.done'), 'success');
}

function downloadMermaidSVG() {
    const svgSource = getCurrentSvgSource();
    if (!svgSource) return;
    triggerTextDownload(`paper_map_${Date.now()}.svg`, svgSource, 'image/svg+xml;charset=utf-8');
}

/**
 * Repaint the preview in the reader's language.
 *
 * The same shape `evidenceViewer.js` uses: a render-time surface has to be
 * re-rendered on a locale change, and this one is *not* re-rendered by anything
 * else — nothing re-runs an analysis on a language switch. Without this the
 * panel kept reading in whichever language it was last built in, beside an
 * interface that had just switched, which is the mixed-language case the whole
 * catalogue exists to prevent.
 *
 * It is safe to call unconditionally because `renderExportPreview` reads only
 * `state` and the DOM and rebuilds its container whole, so a second call for the
 * same switch is idempotent rather than additive. The format `<option>`s are
 * *not* repainted here: they are populated once and guarded by
 * `select.options.length`, and their labels are file-format names.
 *
 * A module is a singleton in the ES module graph, so this runs once per page
 * and needs no `localeBound` flag — the same reasoning `evidenceViewer.js`
 * records at its own subscription.
 */
onLocaleChange(renderExportPreview);

export { answerModeLabel };
export { buildCitationDeepLink };
export { buildEvidenceApiUrl };
export { buildProvenanceRows };
export { buildSessionHtml };
export { buildSessionMarkdown };
export { collectExportCitations };
export { downloadMermaidSVG };
export { exportSessionReport };
export { factFieldLabel };
export { formatCitationIndexMarkdown };
export { formatExportTimestamp };
export { formatMarkdownActions };
export { formatMarkdownList };
export { formatPaperTrace };
export { formatProvenanceMarkdown };
export { formatSectionStatusTable };
export { initializeExportFormats };
export { localised };
export { presentationLabel };
export { renderExportPreview };
export { resetExportState };
export { sanitizeFileStem };
export { setExportEnabled };
export { statusLabel };
export { unknownValue };
