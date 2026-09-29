import { state } from './state.js';
import { readSseStream } from './api.js';
import { focusAnalysisWorkspace, hideError, setButtonLoading, setCancelVisible, setControlDisabled, showError, updateAnalysisProgress, updateStatus } from './dom.js';
import { getUploadValidationError, setFileInfo, updateFileMeta } from './upload.js';
import { applyAnalysisResult, applyAnalysisSection, captureWorkspaceState, resetResultView, restoreWorkspaceState } from './sections.js';

/**
 * Document analysis: owns the SSE request, the request token that discards
 * stale events, and the cancel path that restores the previous workspace.
 */

async function analyze() {
    const apiKey = document.getElementById('apiKey').value.trim();
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    const generateMermaid = document.getElementById('generateMermaid').checked;
    const generateEvaluation = document.getElementById('generateEvaluation').checked;
    const generateResearchBrief = document.getElementById('generateResearchBrief').checked;
    const analyzeBtn = document.getElementById('analyzeBtn');
    const askBtn = document.getElementById('askBtn');

    const uploadError = getUploadValidationError(file);
    if (uploadError) {
        updateFileMeta(file ? uploadError : '');
        showError(uploadError);
        updateStatus('Upload needs attention', 'error');
        return;
    }

    const previousWorkspace = state.currentAnalysisResult ? captureWorkspaceState() : null;

    if (state.analyzeController) {
        state.analyzeController.abort();
    }
    state.analyzeController = new AbortController();
    state.analyzeRequestId += 1;
    const requestId = state.analyzeRequestId;

    resetResultView();
    state.currentSessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    state.pendingAnalysisSnapshot = previousWorkspace;
    setButtonLoading(analyzeBtn, 'Analyzing...', 'Analyze Document', true);
    setCancelVisible('cancelAnalyzeBtn', true);
    document.getElementById('loading').classList.add('active');
    updateAnalysisProgress('upload', 'Uploading and preparing your document...');
    hideError();
    setControlDisabled(askBtn, true);
    updateStatus('Analyzing document...', 'idle');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('session_id', state.currentSessionId);
    formData.append('generate_mermaid', String(generateMermaid));
    formData.append('generate_evaluation', String(generateEvaluation));
    formData.append('generate_research_brief', String(generateResearchBrief));
    if (apiKey) formData.append('api_key', apiKey);

    try {
        const response = await fetch('/api/analyze/stream', {
            method: 'POST',
            body: formData,
            signal: state.analyzeController.signal
        });

        await readSseStream(response, {
            start: payload => {
                state.currentSessionId = payload.session_id || state.currentSessionId;
                document.getElementById('result').classList.add('active');
                setFileInfo(file.name, '...');
                updateAnalysisProgress('analyze', 'AI is analyzing structure, citations, and research signals...');
                updateStatus('Analysis started. Streaming sections...', 'idle');
            },
            section: payload => {
                if (requestId !== state.analyzeRequestId) return;
                const sectionName = payload.name;
                const section = payload.section || {};
                state.currentSections = { ...state.currentSections, [sectionName]: section };
                applyAnalysisSection(sectionName, section, generateEvaluation, generateMermaid, generateResearchBrief);
                updateAnalysisProgress('render', `Rendering ${sectionName.replace('_', ' ')} results...`);
                updateStatus(`Streaming ${sectionName}...`, 'idle');
            },
            done: async payload => {
                if (requestId !== state.analyzeRequestId) return;
                await applyAnalysisResult(payload, file.name, generateEvaluation, generateMermaid, generateResearchBrief);
                updateAnalysisProgress('ready', 'Workspace ready. You can ask questions or export the session.');
                updateStatus('Analysis ready for follow-up questions', 'success');
                focusAnalysisWorkspace();
            },
            error: payload => {
                throw new Error(payload.error || 'Analysis failed.');
            }
        });
    } catch (error) {
        if (error.name === 'AbortError') {
            return;
        }
        if (previousWorkspace) {
            restoreWorkspaceState(previousWorkspace);
            showError(`${error.message || 'Analysis failed.'}\n\nYour previous successful workspace has been restored.`);
            focusAnalysisWorkspace({ scroll: false });
        } else {
            state.currentSessionId = '';
            resetResultView();
            showError(error.message || 'Analysis failed.');
        }
        updateStatus('Analysis failed', 'error');
    } finally {
        if (requestId === state.analyzeRequestId) {
            document.getElementById('loading').classList.remove('active');
            setButtonLoading(analyzeBtn, 'Analyzing...', 'Analyze Document', false);
            setCancelVisible('cancelAnalyzeBtn', false);
            state.analyzeController = null;
            state.pendingAnalysisSnapshot = null;
        }
    }
}

function cancelAnalyzeRequest() {
    if (!state.analyzeController) return;
    state.analyzeController.abort();
    state.analyzeRequestId += 1;
    document.getElementById('loading').classList.remove('active');
    setButtonLoading(document.getElementById('analyzeBtn'), 'Analyzing...', 'Analyze Document', false);
    setCancelVisible('cancelAnalyzeBtn', false);
    state.analyzeController = null;
    if (state.pendingAnalysisSnapshot) {
        restoreWorkspaceState(state.pendingAnalysisSnapshot);
        state.pendingAnalysisSnapshot = null;
        focusAnalysisWorkspace({ scroll: false });
    } else {
        state.currentSessionId = '';
        resetResultView();
    }
    updateStatus('Analysis canceled', 'idle');
}


export { analyze };
export { cancelAnalyzeRequest };
