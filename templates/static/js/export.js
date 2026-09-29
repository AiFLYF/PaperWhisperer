import { ANSWER_MODE_LABELS } from './config.js';
import { state } from './state.js';
import { scheduleButtonFeedbackReset, setControlDisabled, updateStatus } from './dom.js';
import { normalizeSmartActions, normalizeSmartTextItems, resetSmartSuggestions } from './suggestions.js';
import { getCurrentSvgSource } from './mermaid.js';

/** Session report composition, the export preview and file downloads. */

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

function getWorkspaceGuidanceItems(status, completed) {
    const failed = Array.isArray(status.failed_sections) ? status.failed_sections.length : 0;
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

    lines.push('', '---', '', '## Export Metadata', '', `- App: [PaperWhisperer](https://github.com/AiFLYF/PaperWhisperer)`, `- Session export time: ${formatExportTimestamp(reportTime.toISOString())}`);
    return lines.join('\n');
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

function exportSessionReport() {
    const exportBtn = document.getElementById('exportBtn');
    if (!state.currentAnalysisResult) {
        updateStatus('Analyze a document before exporting', 'idle');
        return;
    }

    const fileStem = sanitizeFileStem(state.currentSourceFileName || state.currentAnalysisResult.session_id || 'paperwhisperer_session');
    const markdown = buildSessionMarkdown();
    if (!markdown) {
        updateStatus('Export content is not ready', 'error');
        return;
    }

    triggerTextDownload(`${fileStem}_session_report.md`, markdown, 'text/markdown;charset=utf-8');

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


export { downloadMermaidSVG };
export { exportSessionReport };
export { formatExportTimestamp };
export { formatMarkdownActions };
export { formatMarkdownList };
export { formatPaperTrace };
export { formatSectionStatusTable };
export { renderExportPreview };
export { resetExportState };
export { sanitizeFileStem };
export { setExportEnabled };
