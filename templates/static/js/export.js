import {
    ANSWER_MODE_LABELS,
    EVIDENCE_API_PATH,
    EVIDENCE_DEEP_LINK_PARAMS,
    EXPORT_CITATIONS_HEADING,
    EXPORT_FORMATS,
    EXPORT_PRESENTATION_LABELS,
    EXPORT_PROVENANCE_HEADING,
    EXPORT_UNKNOWN_VALUE,
    PRESENTATION_DIRECT_QUOTE,
    PRESENTATION_PARAPHRASE
} from './config.js';
import { state } from './state.js';
import { scheduleButtonFeedbackReset, setControlDisabled, updateStatus } from './dom.js';
import { normalizeSmartActions, normalizeSmartTextItems, resetSmartSuggestions } from './suggestions.js';
import { getCurrentSvgSource } from './mermaid.js';
import { escapeHtml } from './sanitize.js';
import { formatContent } from './format.js';

/** Session report composition, the export preview and file downloads. */

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

function formatExportTimestamp(value) {
    const date = value ? new Date(value) : new Date();
    if (Number.isNaN(date.getTime())) {
        return new Date().toLocaleString();
    }
    return date.toLocaleString();
}

function formatMarkdownList(items) {
    const normalizedItems = normalizeSmartTextItems(items, 12);
    return normalizedItems.length ? normalizedItems.map(item => `- ${item}`).join('\n') : '- _None._';
}

function formatMarkdownActions(actions) {
    const normalizedActions = normalizeSmartActions(actions, 8);
    return normalizedActions.length
        ? normalizedActions.map(action => `- **${action.label}**: ${action.prompt}`).join('\n')
        : '- _None._';
}

function formatWorkspaceGuidanceMarkdown(items) {
    return items.map(item => `- **${item.label}**: ${item.value} — ${item.detail}`).join('\n');
}

function formatSectionStatusTable(status) {
    const completed = Array.isArray(status.completed_sections) ? status.completed_sections.join(', ') : 'N/A';
    const failed = Array.isArray(status.failed_sections) && status.failed_sections.length ? status.failed_sections.join(', ') : 'None';
    const disabled = Array.isArray(status.disabled_sections) && status.disabled_sections.length ? status.disabled_sections.join(', ') : 'None';
    return [
        '| Status | Sections |',
        '| --- | --- |',
        `| Completed | ${completed || 'None'} |`,
        `| Failed | ${failed} |`,
        `| Disabled | ${disabled} |`,
        `| Quality | ${status.quality || 'N/A'} |`
    ].join('\n');
}

function formatPaperTrace(items) {
    const normalizedItems = Array.isArray(items) ? items.slice(0, 8) : [];
    if (!normalizedItems.length) return '- _None._';
    return normalizedItems.map((item, index) => {
        const bits = [item.source, item.year, item.venue].filter(Boolean).join(' · ');
        const suffix = bits ? ` — ${bits}` : '';
        return `${index + 1}. ${item.title || 'Untitled paper'}${suffix}`;
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

function formatCitationMarkdown(entry, index) {
    const label = EXPORT_FACT_FIELD_LABELS[entry.field_name] || entry.field_name || '未命名字段';
    const presentation = EXPORT_PRESENTATION_LABELS[entry.presentation] || entry.presentation;
    // The quotation marks live in the *text*, taken verbatim from the verifier,
    // and nowhere else. A paraphrase entry reaches this line without them,
    // because nothing in the paraphrase path may produce one.
    const text = entry.presentation === PRESENTATION_DIRECT_QUOTE ? `“${entry.text}”` : entry.text;
    const status = EXPORT_STATUS_LABELS[entry.status] || entry.status || EXPORT_UNKNOWN_VALUE;
    return [
        `### ${index}. ${label}`,
        '',
        `- 呈现方式: ${presentation}`,
        `- 字段状态: ${status}`,
        `- 证据 ID: ${entry.evidence_id}`,
        `- 证据链接: [打开该引用](${entry.deep_link}) · [Evidence API](${entry.api_url})`,
        '',
        text,
        ''
    ].join('\n');
}

function formatCitationIndexMarkdown(result) {
    const entries = collectExportCitations(result);
    if (!entries.length) {
        const hasFacts = Boolean(result?.paper_facts);
        return hasFacts
            ? '本次分析没有产出带引用的结论。'
            : '本次分析没有产出事实层，因此没有可导出的引用链接。';
    }
    const lines = [
        `> 直接引语逐字回源校验过；转述是我们的表述，不加引号。两种呈现方式在这里是分开的字段，不靠标点区分。`,
        ''
    ];
    entries.forEach((entry, index) => {
        lines.push(formatCitationMarkdown(entry, index + 1));
    });
    return lines.join('\n');
}

/**
 * The run provenance, as rows.
 *
 * A fixed list, and every entry is always present. A provenance table that
 * drops the fields a particular run did not record is a table whose absence
 * is ambiguous: the reader cannot tell "this run had no cost" from "this build
 * does not write cost down". `EXPORT_UNKNOWN_VALUE` makes the difference
 * visible, which is the whole point of writing the table down at all.
 */
function buildProvenanceRows(result, exportedAt) {
    const payload = result || {};
    const run = payload.analysis_run && typeof payload.analysis_run === 'object' ? payload.analysis_run : {};
    const parse = payload.parse && typeof payload.parse === 'object' ? payload.parse : {};
    const facts = payload.paper_facts && typeof payload.paper_facts === 'object' ? payload.paper_facts : {};
    const value = (input) => {
        if (input === undefined || input === null || input === '') {
            return EXPORT_UNKNOWN_VALUE;
        }
        return String(input);
    };
    const flag = (input) => (input === true ? '是' : input === false ? '否' : EXPORT_UNKNOWN_VALUE);
    const cost = typeof run.llm_cost === 'number' ? run.llm_cost.toFixed(6) : EXPORT_UNKNOWN_VALUE;
    const coverage = run.fact_count
        ? `${run.grounded_fact_count ?? 0}/${run.fact_count}`
        : EXPORT_UNKNOWN_VALUE;

    return [
        ['导出时间', formatExportTimestamp(exportedAt)],
        ['源文件', value(payload.source_filename)],
        ['文档 ID', value(payload.document_id)],
        ['会话 ID', value(payload.session_id)],
        ['分析运行 ID', value(run.run_id)],
        ['模型', value(run.model || facts.extraction_model)],
        ['提示词版本', value(run.prompt_version || facts.prompt_version)],
        ['证据版本', value(run.evidence_version || facts.evidence_version)],
        ['LLM 调用次数', value(run.llm_calls)],
        ['输入 tokens', value(run.llm_input_tokens)],
        ['输出 tokens', value(run.llm_output_tokens)],
        ['成本', cost],
        ['事实覆盖', coverage],
        ['解析器', value(parse.parsed_by)],
        ['媒体类型', value(parse.media_type)],
        ['是否经过云端解析', flag(parse.used_cloud)],
        ['是否发生 fallback', flag(parse.fallback_used)],
        ['是否命中解析缓存', flag(parse.cache_hit)],
        ['降级说明', Array.isArray(run.degradations) && run.degradations.length
            ? run.degradations.join('；')
            : EXPORT_UNKNOWN_VALUE],
        ['解析告警', Array.isArray(parse.warnings) && parse.warnings.length
            ? parse.warnings.join('；')
            : EXPORT_UNKNOWN_VALUE]
    ];
}

function formatProvenanceMarkdown(result, exportedAt) {
    const rows = buildProvenanceRows(result, exportedAt);
    return [
        '| 项目 | 值 |',
        '| --- | --- |',
        ...rows.map(([term, detail]) => `| ${term} | ${detail} |`)
    ].join('\n');
}

function getWorkspaceGuidanceItems(status, completed) {    const failed = Array.isArray(status.failed_sections) ? status.failed_sections.length : 0;
    const disabled = Array.isArray(status.disabled_sections) ? status.disabled_sections.length : 0;
    const paperLeadCount = state.currentPaperSearchResults.length + state.currentPaperRecommendations.length;
    const nextAction = state.currentNextActions[0];
    return [
        {
            label: 'Workspace state',
            value: failed ? `${completed} ready · ${failed} need review` : `${completed} sections ready`,
            detail: disabled ? `${disabled} optional section(s) disabled for this run.` : 'Core analysis cards are ready for review and export.'
        },
        {
            label: 'Best next step',
            value: nextAction?.label || 'Ask a grounded follow-up',
            detail: nextAction?.prompt || 'Use Evidence mode to verify claims before switching to critique or reproduce mode.'
        },
        {
            label: 'Research trail',
            value: state.currentReadingQueue.length ? `${state.currentReadingQueue.length} saved paper(s)` : `${paperLeadCount} paper lead(s)`,
            detail: state.currentReadingQueue.length ? 'Saved papers are included in the session export.' : 'Search or recommend papers, then save the strongest leads to the queue.'
        }
    ];
}

function renderExportPreview() {
    const preview = document.getElementById('exportPreview');
    if (!preview) return;
    if (!state.currentAnalysisResult) {
        const empty = document.createElement('p');
        empty.className = 'empty-state';
        empty.textContent = 'Export the current analysis, Q&A history, research trace, and Mermaid assets after a successful run.';
        preview.replaceChildren(empty);
        return;
    }

    const status = state.currentAnalysisResult.analysis_status || {};
    const completed = Array.isArray(status.completed_sections) ? status.completed_sections.length : Object.keys(state.currentSections || {}).length;
    const failed = Array.isArray(status.failed_sections) ? status.failed_sections.length : 0;
    const paperLeadCount = state.currentPaperSearchResults.length + state.currentPaperRecommendations.length;
    const exportPreviewItems = [
        { label: 'Analysis', value: `${completed} ready`, detail: failed ? `${failed} section(s) need review` : 'Core sections are ready' },
        { label: 'Q&A', value: `${state.currentChatTurns.length} turns`, detail: state.currentChatTurns.length ? 'Local conversation will be included' : 'Ask follow-ups to enrich the report' },
        { label: 'Papers', value: `${paperLeadCount} leads`, detail: state.currentReadingQueue.length ? `${state.currentReadingQueue.length} saved to queue` : 'Save strong leads before export' },
        { label: 'Visuals', value: state.currentMermaidSource ? 'Map ready' : 'No map', detail: state.currentMermaidSource ? 'Mermaid SVG can be exported' : 'Enable structure diagram for visuals' }
    ];
    const guidanceItems = getWorkspaceGuidanceItems(status, completed);

    const grid = document.createElement('div');
    grid.className = 'export-preview-grid';
    grid.setAttribute('aria-label', 'Export contents summary');
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
    guidance.setAttribute('aria-label', 'Workspace guidance');
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
    note.textContent = 'Includes analysis, research brief, reading queue, suggested follow-ups, research trace, and local Q&A history.';
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
        '# PaperWhisperer Session Report',
        '',
        '> Rich export of the current analysis session, including follow-up Q&A and visual assets.',
        '> Generated by [PaperWhisperer](https://github.com/AiFLYF/PaperWhisperer).',
        '',
        '---',
        '',
        '## Session Overview',
        '',
        '| Item | Value |',
        '| --- | --- |',
        `| Source file | ${state.currentSourceFileName || 'N/A'} |`,
        `| Session ID | ${state.currentAnalysisResult.session_id || state.currentSessionId || 'N/A'} |`,
        `| Generated at | ${formatExportTimestamp(reportTime.toISOString())} |`,
        `| Analysis duration | ${state.currentElapsedSeconds ?? 'N/A'} s |`,
        `| Character count | ${state.currentAnalysisResult.char_count ?? 'N/A'} |`,
        `| Q&A turns | ${state.currentChatTurns.length} |`,
        `| Saved papers | ${state.currentReadingQueue.length} |`,
        '',
        '### Workspace Guidance',
        '',
        formatWorkspaceGuidanceMarkdown(guidanceItems),
        '',
        '### Section Status',
        '',
        formatSectionStatusTable(status),
        '',
        '---',
        '',
        '## Overview',
        '',
        state.currentAnalysisResult.summary || '_No summary generated._',
        '',
        '---',
        '',
        '## Key Citations',
        '',
        state.currentAnalysisResult.quotes || '_No citations generated._',
        '',
        '---',
        '',
        '## Text Structure',
        '',
        state.currentAnalysisResult.mindmap || '_No text structure generated._'
    ];

    if (state.currentAnalysisResult.evaluation) {
        lines.push('', '---', '', '## Evaluation', '', state.currentAnalysisResult.evaluation);
    }

    if (state.currentAnalysisResult.research_brief) {
        lines.push('', '---', '', '## Deep Research Brief', '', state.currentAnalysisResult.research_brief);
    }

    if (state.currentSuggestedQuestions.length || state.currentNextActions.length) {
        lines.push(
            '',
            '---',
            '',
            '## Suggested Follow-ups',
            '',
            '### Questions',
            '',
            formatMarkdownList(state.currentSuggestedQuestions),
            '',
            '### Next Actions',
            '',
            formatMarkdownActions(state.currentNextActions)
        );
    }

    if (state.currentPaperSearchResults.length || state.currentPaperRecommendations.length || state.currentReadingQueue.length) {
        lines.push(
            '',
            '---',
            '',
            '## Research Trace',
            '',
            '### Reading Queue',
            '',
            formatPaperTrace(state.currentReadingQueue),
            '',
            '### Paper Search Results',
            '',
            formatPaperTrace(state.currentPaperSearchResults),
            '',
            '### Auto Recommendations',
            '',
            formatPaperTrace(state.currentPaperRecommendations)
        );
    }

    if (state.currentMermaidSource) {
        lines.push(
            '',
            '---',
            '',
            '## Mermaid Source',
            '',
            '```mermaid',
            state.currentMermaidSource,
            '```'
        );

        if (svgSource) {
            lines.push(
                '',
                '### Visual Map SVG',
                '',
                `The rendered SVG is exported as a companion file: \`${svgFileName}\`.`
            );
        }
    }

    if (state.currentChatTurns.length) {
        lines.push('', '---', '', '## Ask Questions', '');
        state.currentChatTurns.forEach((turn, index) => {
            const modeLabel = ANSWER_MODE_LABELS[turn.answer_mode] || ANSWER_MODE_LABELS.evidence;
            lines.push(
                `### Q${index + 1}`,
                '',
                `Mode: ${modeLabel}`,
                '',
                turn.question || '_No question text._',
                '',
                `### A${index + 1}`,
                '',
                turn.answer || '_No answer text._',
                ''
            );
        });
    }

    lines.push(
        '',
        '---',
        '',
        `## ${EXPORT_CITATIONS_HEADING}`,
        '',
        formatCitationIndexMarkdown(state.currentAnalysisResult),
        '',
        '---',
        '',
        `## ${EXPORT_PROVENANCE_HEADING}`,
        '',
        '> 这张表记录的是“生成这份文件的条件”。打开一个六个月前的导出件时，它是判断这份结论还能不能信的唯一依据。',
        '',
        formatProvenanceMarkdown(state.currentAnalysisResult, reportTime.toISOString()),
        '',
        '---',
        '',
        '## Export Metadata',
        '',
        '- App: [PaperWhisperer](https://github.com/AiFLYF/PaperWhisperer)',
        `- Session export time: ${formatExportTimestamp(reportTime.toISOString())}`
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
            : '<p class="empty">（本次分析没有这一节的内容）</p>'
    ].join('\n');

    const parts = [
        '<!DOCTYPE html>',
        '<html lang="zh-CN">',
        '<head>',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        `<title>${escapeHtml(sanitizeFileStem(state.currentSourceFileName || 'paperwhisperer_session'))} — PaperWhisperer 导出</title>`,
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
        '<h1>PaperWhisperer 会话报告</h1>',
        `<p class="citation-meta">导出于 ${escapeHtml(formatExportTimestamp(reportTime.toISOString()))}</p>`,
        section('Overview', result.summary),
        section('Key Citations', result.quotes),
        section('Text Structure', result.mindmap)
    ];

    if (result.evaluation) {
        parts.push(section('Evaluation', result.evaluation));
    }
    if (result.research_brief) {
        parts.push(section('Deep Research Brief', result.research_brief));
    }

    parts.push(
        `<h2>${escapeHtml(EXPORT_CITATIONS_HEADING)}</h2>`,
        '<p class="citation-meta">直接引语逐字回源校验过；转述是我们的表述，不加引号。两种呈现方式在这里是分开的字段，不靠标点区分。</p>'
    );
    if (!citations.length) {
        parts.push('<p class="empty">本次分析没有产出带引用的结论。</p>');
    }
    citations.forEach((entry, index) => {
        const label = EXPORT_FACT_FIELD_LABELS[entry.field_name] || entry.field_name || '未命名字段';
        const presentation = EXPORT_PRESENTATION_LABELS[entry.presentation] || entry.presentation;
        const status = EXPORT_STATUS_LABELS[entry.status] || entry.status || EXPORT_UNKNOWN_VALUE;
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
            ` · <a href="${escapeHtml(entry.deep_link)}">打开该引用</a>`,
            ` · <a href="${escapeHtml(entry.api_url)}">Evidence API</a></p>`,
            body,
            '</div>'
        ].join('\n'));
    });

    parts.push(
        `<h2>${escapeHtml(EXPORT_PROVENANCE_HEADING)}</h2>`,
        '<p class="citation-meta">这张表记录的是“生成这份文件的条件”。打开一个六个月前的导出件时，它是判断这份结论还能不能信的唯一依据。</p>',
        '<table>',
        '<tr><th>项目</th><th>值</th></tr>',
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
        option.textContent = format.label;
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
        updateStatus('Analyze a document before exporting', 'idle');
        return;
    }

    const fileStem = sanitizeFileStem(state.currentSourceFileName || state.currentAnalysisResult.session_id || 'paperwhisperer_session');
    const format = resolveExportFormat();
    const content = format.value === 'html'
        ? buildSessionHtml()
        : buildSessionMarkdown();
    if (!content) {
        updateStatus('Export content is not ready', 'error');
        return;
    }

    triggerTextDownload(`${fileStem}_session_report.${format.extension}`, content, format.mime);

    const svgSource = getCurrentSvgSource();
    if (svgSource) {
        triggerTextDownload(`${fileStem}_visual_map.svg`, svgSource, 'image/svg+xml;charset=utf-8');
    }

    if (exportBtn) {
        const originalText = exportBtn.textContent;
        exportBtn.textContent = svgSource ? 'Exported Report + SVG' : 'Exported Report';
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
    updateStatus(svgSource ? 'Session report and SVG exported' : 'Session report exported', 'success');
}

function downloadMermaidSVG() {
    const svgSource = getCurrentSvgSource();
    if (!svgSource) return;
    triggerTextDownload(`paper_map_${Date.now()}.svg`, svgSource, 'image/svg+xml;charset=utf-8');
}


export { buildCitationDeepLink };
export { buildEvidenceApiUrl };
export { buildProvenanceRows };
export { buildSessionHtml };
export { buildSessionMarkdown };
export { collectExportCitations };
export { downloadMermaidSVG };
export { exportSessionReport };
export { formatCitationIndexMarkdown };
export { formatExportTimestamp };
export { formatMarkdownActions };
export { formatMarkdownList };
export { formatPaperTrace };
export { formatProvenanceMarkdown };
export { formatSectionStatusTable };
export { initializeExportFormats };
export { renderExportPreview };
export { resetExportState };
export { sanitizeFileStem };
export { setExportEnabled };
