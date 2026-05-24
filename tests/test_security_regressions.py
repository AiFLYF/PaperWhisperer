import asyncio
import json
import os
from email.message import Message
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import web_app


class FakeResponse:
    def __init__(self, body, content_type="application/pdf", content_length=None, url="https://example.com/paper.pdf"):
        self.body = body
        self.offset = 0
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.url = url
        self.closed = False

    def geturl(self):
        return self.url

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.body) - self.offset
        start = self.offset
        end = min(len(self.body), start + size)
        self.offset = end
        return self.body[start:end]

    def close(self):
        self.closed = True


@pytest.fixture
def public_example_urls(monkeypatch):
    monkeypatch.setattr(web_app, "is_public_http_url", lambda raw_url: str(raw_url).startswith("https://example.com"))


def patch_remote_response(monkeypatch, response):
    monkeypatch.setattr(web_app.urllib.request, "urlopen", lambda *args, **kwargs: response)


class FakeUploadFile:
    def __init__(self, body):
        self.body = body
        self.offset = 0

    async def read(self, size=-1):
        if size is None or size < 0:
            size = len(self.body) - self.offset
        start = self.offset
        end = min(len(self.body), start + size)
        self.offset = end
        return self.body[start:end]


def test_index_page_renders():
    response = TestClient(web_app.app).get("/")

    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "camera=(), microphone=(), geolocation=()" in response.headers["permissions-policy"]
    assert float(response.headers["x-process-time-ms"]) >= 0
    assert "PaperWhisperer" in response.text
    assert "aria-live=\"polite\"" in response.text
    assert 'id="themeBtn" aria-label="Switch to dark theme" aria-pressed="false"' in response.text
    assert 'stroke-width="2" aria-hidden="true" focusable="false"' in response.text
    assert 'stroke-linejoin="round" aria-hidden="true" focusable="false"' in response.text
    assert 'id="zoomInBtn" title="Zoom In" aria-label="Zoom in visual map" aria-controls="mermaidChart"' in response.text
    assert 'id="downloadMermaidBtn" title="Download SVG" aria-label="Download visual map SVG" aria-controls="mermaidChart"' in response.text
    assert "Deep Research Brief" in response.text
    assert "Reading Queue" in response.text
    assert 'id="dropZone" role="button" tabindex="0" aria-describedby="fileMeta" aria-label="Choose or drop a document file" aria-invalid="false" aria-keyshortcuts="Alt+U" aria-controls="file"' in response.text
    assert 'class="file-meta" id="fileMeta" role="status" aria-live="polite"' in response.text
    assert "Drop a paper here or browse" in response.text
    assert 'id="analysisProgressSteps"' in response.text
    assert 'class="progress-step active" data-step="upload" aria-current="step"' in response.text
    assert 'class="error is-hidden" id="error" role="alert" aria-live="assertive" aria-atomic="true" tabindex="-1"' in response.text
    assert "Upload" in response.text
    assert 'id="answerModeHint"' in response.text
    assert 'id="apiKey" placeholder="Enter your API Key" autocomplete="off" autocapitalize="off" spellcheck="false"' in response.text
    assert 'type="search" id="paperSearchInput" placeholder="Search by topic, task, method, or dataset..." aria-describedby="paperSearchMeta" autocomplete="off" autocapitalize="off" spellcheck="false" inputmode="search" aria-keyshortcuts="Alt+S"' in response.text
    assert 'id="questionInput" placeholder="Ask about evidence, methods, limitations, or reproducibility..." aria-describedby="answerModeHint" autocomplete="off" autocapitalize="off" spellcheck="false" aria-keyshortcuts="Alt+Q"' in response.text
    assert 'id="answerModeGroup" role="radiogroup" aria-label="Answer mode" aria-describedby="answerModeHint"' in response.text
    assert 'role="radio" data-mode="evidence" aria-checked="true" aria-pressed="true" tabindex="0"' in response.text
    assert 'role="radio" data-mode="reproduce" aria-checked="false" aria-pressed="false" tabindex="-1"' in response.text
    assert "Ask about evidence, methods, limitations, or reproducibility" in response.text
    assert 'id="backToTopBtn" aria-label="Back to top" aria-hidden="true" tabindex="-1"' in response.text
    assert 'class="skip-links"' in response.text
    assert 'href="#uploadWorkspace"' in response.text
    assert 'class="workspace-quick-nav"' in response.text
    assert 'aria-label="Workspace quick navigation"' in response.text
    assert 'href="#readingQueueTitle"' in response.text
    assert 'id="uploadWorkspace"' in response.text
    assert 'aria-labelledby="uploadWorkspaceTitle"' in response.text
    assert 'class="shortcut-hints"' in response.text
    assert '<kbd>Alt</kbd> + <kbd>S</kbd> Search' in response.text
    assert 'id="cancelAnalyzeBtn" hidden disabled aria-disabled="true"' in response.text
    assert 'id="cancelPaperSearchBtn" hidden disabled aria-disabled="true"' in response.text
    assert 'id="clearReadingQueueBtn" disabled aria-disabled="true"' in response.text
    assert 'id="exportBtn" disabled aria-disabled="true"' in response.text
    assert 'id="recommendBtn" disabled aria-disabled="true"' in response.text
    assert 'id="cancelRecommendBtn" hidden disabled aria-disabled="true"' in response.text
    assert 'id="askBtn" type="button" disabled aria-disabled="true"' in response.text
    assert 'id="cancelAskBtn" hidden disabled aria-disabled="true"' in response.text
    assert 'class="example-query-chips"' in response.text
    assert 'data-example-query="retrieval augmented generation evaluation"' in response.text
    assert 'RAG evaluation' in response.text
    assert 'class="question-starter-chips"' in response.text
    assert 'data-starter-question="What is the central research question and main contribution?"' in response.text
    assert 'Evidence check' in response.text
    assert 'rel="preconnect" href="https://cdn.jsdelivr.net"' in response.text
    assert 'rel="dns-prefetch" href="https://github.com"' in response.text
    assert 'href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" referrerpolicy="strict-origin-when-cross-origin"' in response.text
    assert 'src="https://cdn.jsdelivr.net/npm/marked/marked.min.js" defer referrerpolicy="strict-origin-when-cross-origin"' in response.text
    assert 'fetchpriority="high"' in response.text
    assert 'class="github-link" href="https://github.com/AiFLYF/PaperWhisperer" target="_blank" rel="noopener noreferrer" referrerpolicy="strict-origin-when-cross-origin"' in response.text
    assert 'src="https://github.com/favicon.ico" alt="" aria-hidden="true" width="18" height="18" loading="lazy" decoding="async" referrerpolicy="strict-origin-when-cross-origin"' in response.text
    assert 'class="author-link" href="https://github.com/AiFLYF" target="_blank" rel="noopener noreferrer" referrerpolicy="strict-origin-when-cross-origin"' in response.text
    assert 'aria-labelledby="paperSearchTitle"' in response.text
    assert 'aria-labelledby="readingQueueTitle"' in response.text
    assert 'role="region" aria-label="Analysis results workspace" tabindex="-1"' in response.text
    assert 'class="result-card is-hidden" id="mermaidCard" aria-labelledby="visualMapTitle" aria-hidden="true"' in response.text
    assert 'style="display: none;"' not in response.text
    assert 'aria-labelledby="askQuestionsTitle"' in response.text
    assert 'class="smart-prompts is-hidden" id="smartPrompts" role="region" aria-label="Suggested follow-up questions" aria-live="polite" aria-hidden="true"' in response.text
    assert 'class="next-actions is-hidden" id="nextActions" role="region" aria-label="Recommended next actions" aria-live="polite" aria-hidden="true"' in response.text
    assert 'class="heart" aria-hidden="true"' in response.text
    assert 'id="aiToggleBtn" aria-expanded="false" aria-controls="aiList" aria-label="Show AI collaborators"' in response.text
    assert 'class="arrow" aria-hidden="true"' in response.text
    assert 'class="ai-list" id="aiList" hidden' in response.text


def test_static_frontend_assets_are_served():
    client = TestClient(web_app.app)

    js_response = client.get("/static/js/app.js")
    css_response = client.get("/static/css/style.css")

    assert js_response.status_code == 200
    assert "setAnswerMode" in js_response.text
    assert "SUPPORTED_UPLOAD_EXTENSIONS" in js_response.text
    assert "MAX_UPLOAD_BYTES" in js_response.text
    assert "getUploadFileExtension" in js_response.text
    assert "appendFileMetaItem" in js_response.text
    assert "fileMeta.replaceChildren()" in js_response.text
    assert "fileMeta.innerHTML = ''" not in js_response.text
    assert "dropZone?.setAttribute('aria-invalid', 'false')" in js_response.text
    assert "dropZone?.setAttribute('aria-invalid', String(hasError))" in js_response.text
    assert "Ready to analyze" in js_response.text
    assert "Review needed" in js_response.text
    assert "Supported document type and size." in js_response.text
    assert "updateAnalysisProgress" in js_response.text
    assert "element.setAttribute('aria-current', 'step')" in js_response.text
    assert "element.removeAttribute('aria-current')" in js_response.text
    assert "btn.replaceChildren(createThemeIcon(isDark ? 'sun' : 'moon'))" in js_response.text
    assert "document.createElementNS(SVG_NS, 'svg')" in js_response.text
    assert "svg.setAttribute('aria-hidden', 'true')" in js_response.text
    assert "svg.setAttribute('focusable', 'false')" in js_response.text
    assert "btn.innerHTML = sunIcon" not in js_response.text
    assert "btn.innerHTML = moonIcon" not in js_response.text
    assert "btn.setAttribute('aria-pressed', String(isDark))" in js_response.text
    assert "button.setAttribute('aria-checked', selected ? 'true' : 'false')" in js_response.text
    assert "button.tabIndex = selected ? 0 : -1" in js_response.text
    assert "button.addEventListener('keydown', handleAnswerModeKeydown)" in js_response.text
    assert "function handleAnswerModeKeydown(event)" in js_response.text
    assert "const navigationKeys = ['ArrowLeft', 'ArrowUp', 'ArrowRight', 'ArrowDown', 'Home', 'End']" in js_response.text
    assert "nextChip.focus({ preventScroll: true })" in js_response.text
    assert "function readLocalStorageValue(key)" in js_response.text
    assert "function writeLocalStorageValue(key, value)" in js_response.text
    assert "Local storage read failed" in js_response.text
    assert "Local storage write failed" in js_response.text
    assert "localStorage.getItem(THEME_STORAGE_KEY)" not in js_response.text
    assert "localStorage.setItem(THEME_STORAGE_KEY, nextTheme)" not in js_response.text
    assert "Switch to light theme" in js_response.text
    assert "Switch to dark theme" in js_response.text
    assert "getPaperStateDetails" in js_response.text
    assert "getWorkspaceGuidanceItems" in js_response.text
    assert "function setOptionalCardVisible(cardId, visible)" in js_response.text
    assert "card.setAttribute('aria-hidden', String(!visible))" in js_response.text
    assert "setOptionalCardVisible('mermaidCard', visible)" in js_response.text
    assert "container.setAttribute('aria-hidden', String(!currentSuggestedQuestions.length))" in js_response.text
    assert "container.setAttribute('aria-hidden', String(!currentNextActions.length))" in js_response.text
    assert "document.getElementById('chatHistory').replaceChildren()" in js_response.text
    assert "container.replaceChildren()" in js_response.text
    assert "document.getElementById('mermaidChart').replaceChildren()" in js_response.text
    assert "mermaidDiv.replaceChildren()" in js_response.text
    assert "fallback.textContent = cleanSource" in js_response.text
    assert "document.getElementById('fileInfo').replaceChildren()" in js_response.text
    assert "fileInfo.replaceChildren()" in js_response.text
    assert "getReadingQueueKey" in js_response.text
    assert "hasReadingQueueItem" in js_response.text
    assert "container.replaceChildren(...cards)" in js_response.text
    assert "removeButton.dataset.queueRemoveIndex = String(index)" in js_response.text
    assert "titleLink.textContent = item.title" in js_response.text
    assert "stateElement.append(icon, body)" in js_response.text
    assert "button.dataset.paperAction = action" in js_response.text
    assert "addButton.setAttribute('aria-disabled', String(addButtonDisabled))" in js_response.text
    assert "abstract.textContent = item.abstract || 'No abstract available.'" in js_response.text
    assert "ANSWER_MODE_DETAILS" in js_response.text
    assert "function replaceWithFormattedContent(element, content, fallbackText)" in js_response.text
    assert "element.replaceChildren(template.content.cloneNode(true))" in js_response.text
    assert "function createSectionStateCard" in js_response.text
    assert "element.replaceChildren(createSectionStateCard('Section needs review', errorMessage, 'error'))" in js_response.text
    assert "detailElement.textContent = detail" in js_response.text
    assert "replaceWithFormattedContent(body, message, 'No content available.')" in js_response.text
    assert "replaceWithFormattedContent(shell.content, answer, 'No answer available.')" in js_response.text
    assert "body.innerHTML = formatContent(message" not in js_response.text
    assert "shell.content.innerHTML = formatContent(answer" not in js_response.text
    assert "renderSectionStateCard" not in js_response.text
    assert "function sanitizeImageUrl(rawUrl)" in js_response.text
    assert "const safeSrc = sanitizeImageUrl(image.getAttribute('src'))" in js_response.text
    assert "Unsafe image url ignored" in js_response.text
    assert "anchor.referrerPolicy = 'strict-origin-when-cross-origin'" in js_response.text
    assert "image.decoding = 'async'" in js_response.text
    assert "image.referrerPolicy = 'strict-origin-when-cross-origin'" in js_response.text
    assert "Nothing to copy yet" in js_response.text
    assert "Session report exported" in js_response.text
    assert "Exported Report + SVG" in js_response.text
    assert "function buildApiError(data, fallbackMessage)" in js_response.text
    assert "if (data.code) error.code = data.code" in js_response.text
    assert "if (data.timestamp) error.timestamp = data.timestamp" in js_response.text
    assert "throw buildApiError(data, 'Paper search failed.')" in js_response.text
    assert "throw buildApiError(data, 'Connection failed.')" in js_response.text
    assert "btnElement.textContent = 'Copied'" in js_response.text
    assert "btnElement.innerText" not in js_response.text
    assert "const buttonFeedbackTimers = new WeakMap()" in js_response.text
    assert "function scheduleButtonFeedbackReset(button, callback, delay = 1600)" in js_response.text
    assert "clearTimeout(existingTimer)" in js_response.text
    assert "buttonFeedbackTimers.delete(button)" in js_response.text
    assert "scheduleButtonFeedbackReset(btnElement, () => { btnElement.textContent = originalText; })" in js_response.text
    assert "exportBtn.textContent = svgSource ? 'Exported Report + SVG' : 'Exported Report'" in js_response.text
    assert "exportBtn.innerText" not in js_response.text
    assert "exportBtn.setAttribute('aria-busy', 'true')" in js_response.text
    assert "exportBtn.removeAttribute('aria-busy')" in js_response.text
    assert "setControlDisabled(exportBtn, true)" in js_response.text
    assert "setControlDisabled(exportBtn, false)" in js_response.text
    assert "aria-busy" in js_response.text
    assert "aria-disabled" in js_response.text
    assert "setControlDisabled" in js_response.text
    assert "element.setAttribute('aria-disabled', String(disabled))" in js_response.text
    assert "addButton.setAttribute('aria-disabled', String(addButtonDisabled))" in js_response.text
    assert "is-busy" in js_response.text
    assert "Search query needed" in js_response.text
    assert "Question needed" in js_response.text
    assert "initializeBackToTop" in js_response.text
    assert "scheduleBackToTopVisibilityUpdate" in js_response.text
    assert "backToTopFramePending" in js_response.text
    assert "button.setAttribute('aria-hidden', String(!isVisible))" in js_response.text
    assert "button.tabIndex = isVisible ? 0 : -1" in js_response.text
    assert "requestAnimationFrame(() =>" in js_response.text
    assert "window.addEventListener('scroll', scheduleBackToTopVisibilityUpdate, { passive: true })" in js_response.text
    assert "window.addEventListener('scroll', updateBackToTopVisibility" not in js_response.text
    assert "scrollToTop" in js_response.text
    assert "getMotionSafeScrollBehavior" in js_response.text
    assert "prefers-reduced-motion: reduce" in js_response.text
    assert "scrollElementIntoView" in js_response.text
    assert "scrollWindowTo" in js_response.text
    assert "behavior: getMotionSafeScrollBehavior()" in js_response.text
    assert "behavior: 'smooth'" not in js_response.text
    assert "focusAnalysisWorkspace" in js_response.text
    assert "preventScroll" in js_response.text
    assert "formatWorkspaceGuidanceMarkdown" in js_response.text
    assert "formatSectionStatusTable" in js_response.text
    assert "handleWorkspaceShortcut" in js_response.text
    assert "focusWorkspaceTarget" in js_response.text
    assert "reader.releaseLock()" in js_response.text
    assert "Upload shortcut focused document source" in js_response.text
    assert "loadMermaidRenderer" in js_response.text
    assert "mermaidReadyPromise" in js_response.text
    assert "mermaid-svg-fit" in js_response.text
    assert "mermaid-error-message" in js_response.text
    assert "setMermaidCardVisible" in js_response.text
    assert "setOptionalCardVisible" in js_response.text
    assert "mermaidHidden" in js_response.text
    assert "evaluationHidden" in js_response.text
    assert "researchBriefHidden" in js_response.text
    assert "setOptionalCardVisible('evaluationCard', generateEvaluation)" in js_response.text
    assert "setOptionalCardVisible('researchBriefCard', generateResearchBrief)" in js_response.text
    assert "mermaidCard.style.display" not in js_response.text
    assert "document.getElementById('mermaidCard').style.display" not in js_response.text
    assert "document.getElementById('evaluationCard').style.display" not in js_response.text
    assert "document.getElementById('researchBriefCard').style.display" not in js_response.text
    assert "createInlineSpinner" in js_response.text
    assert "style=\"width:16px" not in js_response.text
    assert "style=\"color:#d9480f" not in js_response.text
    assert "svgElement.style" not in js_response.text
    assert "const mermaidReady = import" not in js_response.text
    assert "cancelAnalyzeRequest" in js_response.text
    assert "cancelPaperSearchRequest" in js_response.text
    assert "setCancelVisible" in js_response.text
    assert "const shouldShow = list.hidden" in js_response.text
    assert "list.hidden = !shouldShow" in js_response.text
    assert "button.setAttribute('aria-label', shouldShow ? 'Hide AI collaborators' : 'Show AI collaborators')" in js_response.text
    assert "Analysis canceled" in js_response.text
    assert "errorEl.classList.remove('is-hidden')" in js_response.text
    assert "errorEl.classList.add('is-hidden')" in js_response.text
    assert "errorEl.style.display" not in js_response.text
    assert "clipboard-fallback-field" in js_response.text
    assert "textarea.remove()" in js_response.text
    assert "document.body.removeChild(textarea)" not in js_response.text
    assert "function triggerTextDownload(fileName, content, mimeType = 'text/plain;charset=utf-8')" in js_response.text
    assert "a.remove()" in js_response.text
    assert "document.body.removeChild(a)" not in js_response.text
    assert "textarea.style.position" not in js_response.text
    assert "textarea.style.opacity" not in js_response.text
    assert "errorEl.focus({ preventScroll: true })" in js_response.text
    assert "useExampleQuery" in js_response.text
    assert "Example query loaded" in js_response.text
    assert "useStarterQuestion" in js_response.text
    assert "Starter question loaded" in js_response.text
    assert "container.classList.toggle('is-hidden', !currentSuggestedQuestions.length)" in js_response.text
    assert "container.classList.toggle('is-hidden', !currentNextActions.length)" in js_response.text
    assert "currentSuggestedQuestions.length ? 'flex'" not in js_response.text
    assert "currentNextActions.length ? 'flex'" not in js_response.text
    assert "queue-rank" in js_response.text
    assert "link.target = '_blank'" in js_response.text
    assert "link.rel = 'noopener noreferrer'" in js_response.text
    assert "link.referrerPolicy = 'strict-origin-when-cross-origin'" in js_response.text
    assert "queue-details" in js_response.text
    assert "queue-actions" in js_response.text
    assert "handleReadingQueueClick" in js_response.text
    assert "data-queue-remove-index" in js_response.text
    assert "handlePaperResultClick" in js_response.text
    assert "data-paper-action" in js_response.text
    assert "onclick=\"" not in js_response.text
    assert "exportPreviewItems" in js_response.text
    assert "grid.setAttribute('aria-label', 'Export contents summary')" in js_response.text
    assert "preview.replaceChildren(grid, guidance, note)" in js_response.text
    assert "note.textContent = 'Includes analysis, research brief, reading queue, suggested follow-ups, research trace, and local Q&A history.'" in js_response.text
    assert "preview.innerHTML" not in js_response.text
    assert "Mermaid SVG can be exported" in js_response.text
    assert css_response.status_code == 200
    assert "prefers-reduced-motion" in css_response.text
    assert ".drop-zone.has-error" in css_response.text
    assert ".file-meta-ready" in css_response.text
    assert ".file-meta-error" in css_response.text
    assert ".file-meta-details" in css_response.text
    assert ".file-meta-status" in css_response.text
    assert ".progress-step.active" in css_response.text
    assert "--success-color" in css_response.text
    assert "--error-color" in css_response.text
    assert ".inline-spinner" in css_response.text
    assert ".mermaid-svg-fit" in css_response.text
    assert ".mermaid-fallback-message" in css_response.text
    assert ".mermaid-error-message" in css_response.text
    assert ".mermaid-raw-fallback" in css_response.text
    assert ".is-hidden" in css_response.text
    assert ".clipboard-fallback-field" in css_response.text
    assert ".paper-state" in css_response.text
    assert ".workspace-guidance" in css_response.text
    assert ".btn.is-busy" in css_response.text
    assert ".action-btn.action-success" in css_response.text
    assert ".action-btn:disabled" in css_response.text
    assert ".answer-mode-hint" in css_response.text
    assert ".section-state" in css_response.text
    assert "@media (pointer: coarse)" in css_response.text
    assert "-webkit-tap-highlight-color" in css_response.text
    assert ".back-to-top.show" in css_response.text
    assert ".skip-links" in css_response.text
    assert ".skip-links a:focus-visible" in css_response.text
    assert ".shortcut-hints" in css_response.text
    assert ".shortcut-hints kbd" in css_response.text
    assert ".request-cancel" in css_response.text
    assert ".request-actions" in css_response.text
    assert ".workspace-quick-nav" in css_response.text
    assert ".workspace-quick-nav a:focus-visible" in css_response.text
    assert ".example-query-chips" in css_response.text
    assert ".example-query-chips button:focus-visible" in css_response.text
    assert ".question-starter-chips" in css_response.text
    assert ".question-starter-chips button:focus-visible" in css_response.text
    assert ".queue-rank" in css_response.text
    assert ".queue-details" in css_response.text
    assert ".queue-actions" in css_response.text
    assert ".export-preview-card" in css_response.text
    assert ".export-preview-card small" in css_response.text


def test_standalone_landing_page_accessibility_regressions():
    html = Path(__file__).resolve().parents[1].joinpath("index.html").read_text(encoding="utf-8")

    assert '<html lang="zh-CN">' in html
    assert 'class="skip-link" href="#main-content"' in html
    assert '<main id="main-content">' in html
    assert 'aria-label="Primary navigation"' in html
    assert "@media(prefers-reduced-motion:reduce)" in html
    assert "rel=\"noopener noreferrer\"" in html
    assert 'target="_blank" rel="noopener"' not in html


def test_icon_routes_use_static_cache_headers():
    client = TestClient(web_app.app)

    for path in ("/logo.ico", "/favicon.ico"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == web_app.STATIC_ICON_CACHE_CONTROL
        assert response.headers["content-type"] == "image/x-icon"


def test_health_endpoint_reports_runtime_status():
    response = TestClient(web_app.app).get("/api/health")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app"] == "PaperWhisperer"
    assert payload["version"] == web_app.APP_VERSION
    assert web_app.app.version == web_app.APP_VERSION
    assert payload["uptime_seconds"] >= 0
    assert set(payload["folders"]) == {"uploads", "output", "context"}


def test_save_upload_file_accepts_valid_pdf(tmp_path):
    destination = tmp_path / "paper.pdf"
    body = b"%PDF-1.7\nlocal upload"

    total_bytes = asyncio.run(web_app.save_upload_file(FakeUploadFile(body), destination, web_app.MAX_CONTENT_LENGTH))

    assert total_bytes == len(body)
    assert destination.read_bytes() == body


def test_save_upload_file_rejects_disguised_html_and_cleans_up(tmp_path):
    destination = tmp_path / "paper.pdf"

    with pytest.raises(ValueError):
        asyncio.run(web_app.save_upload_file(FakeUploadFile(b"<html></html>"), destination, web_app.MAX_CONTENT_LENGTH))

    assert not destination.exists()


def test_remove_file_safely_logs_cleanup_failures(tmp_path, monkeypatch, caplog):
    destination = tmp_path / "stale.tmp"
    destination.write_text("stale", encoding="utf-8")

    def fail_remove(path):
        raise OSError("permission denied")

    monkeypatch.setattr(web_app.os, "remove", fail_remove)
    caplog.set_level("ERROR", logger=web_app.__name__)

    web_app.remove_file_safely(str(destination), "test temp file")

    assert "Failed to remove test temp file" in caplog.text
    assert str(destination) in caplog.text


def test_close_response_safely_logs_cleanup_failures(caplog):
    class BrokenCloseResponse:
        def close(self):
            raise OSError("close failed")

    caplog.set_level("ERROR", logger=web_app.__name__)

    web_app.close_response_safely(BrokenCloseResponse(), "test response")

    assert "Failed to close test response" in caplog.text


def test_close_upload_file_safely_logs_cleanup_failures(caplog):
    class BrokenUploadFile:
        async def close(self):
            raise OSError("close failed")

    caplog.set_level("ERROR", logger=web_app.__name__)

    asyncio.run(web_app.close_upload_file_safely(BrokenUploadFile(), "test upload"))

    assert "Failed to close test upload" in caplog.text


def test_public_hostname_validation_uses_short_cache(monkeypatch):
    web_app.PUBLIC_HOSTNAME_CACHE.clear()
    calls = []

    def fake_getaddrinfo(hostname, *args, **kwargs):
        calls.append(hostname)
        return [(None, None, None, None, ("93.184.216.34", 443))]

    monkeypatch.setattr(web_app.socket, "getaddrinfo", fake_getaddrinfo)

    assert web_app.resolve_public_hostname("example.com") is True
    assert web_app.resolve_public_hostname("example.com") is True
    assert calls == ["example.com"]


def test_public_hostname_validation_does_not_cache_private_ip_literals():
    web_app.PUBLIC_HOSTNAME_CACHE.clear()

    assert web_app.is_ip_literal("127.0.0.1") is True
    assert web_app.is_ip_literal("example.com") is False
    assert web_app.resolve_public_hostname("127.0.0.1") is False
    assert web_app.PUBLIC_HOSTNAME_CACHE == {}


def test_public_http_url_rejects_invalid_ports():
    assert web_app.is_public_http_url("https://example.com:99999/paper.pdf") is False
    assert web_app.is_public_http_url("https://example.com:notaport/paper.pdf") is False


def test_analyze_endpoint_missing_file_returns_structured_error():
    response = TestClient(web_app.app).post("/api/analyze")

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "Please upload a file."
    assert payload["code"] == "missing_file"
    assert "timestamp" in payload


def test_analyze_stream_missing_file_returns_structured_error():
    response = TestClient(web_app.app).post("/api/analyze/stream")

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "Please upload a file."
    assert payload["code"] == "missing_file"
    assert "timestamp" in payload


def test_import_paper_missing_url_returns_structured_error():
    response = TestClient(web_app.app).post(
        "/api/import-paper",
        content=json.dumps({}),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "A downloadable paper URL is required."
    assert payload["code"] == "missing_paper_url"
    assert "timestamp" in payload


def test_analyze_stream_emits_sections_and_done(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path / "context"))
    monkeypatch.setattr(web_app, "OUTPUT_FOLDER", str(tmp_path / "output"))
    monkeypatch.setattr(web_app, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    os.makedirs(web_app.CONTEXT_FOLDER, exist_ok=True)
    os.makedirs(web_app.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(web_app.UPLOAD_FOLDER, exist_ok=True)

    def fake_analyze_stream(self, file_path, generate_mermaid=True, generate_evaluation=True, generate_research_brief=True):
        self.document_content = "streamed document"
        sections = {
            "summary": web_app.build_section_result("success", "summary"),
            "quotes": web_app.build_section_result("success", "quotes"),
            "mindmap": web_app.build_section_result("success", "mindmap"),
            "mermaid": web_app.build_section_result("disabled"),
            "evaluation": web_app.build_section_result("disabled"),
            "research_brief": web_app.build_section_result("success", "brief"),
        }
        assert generate_research_brief is True
        yield {"type": "section", "name": "research_brief", "section": sections["research_brief"]}
        yield {
            "type": "done",
            "result": {
                "summary": "summary",
                "quotes": "quotes",
                "mindmap": "mindmap",
                "mermaid": "",
                "evaluation": "",
                "research_brief": "brief",
                "sections": sections,
                "char_count": len(self.document_content),
                "elapsed_seconds": 0.1,
                "suggested_questions": [],
                "next_actions": [],
                "analysis_status": {},
            },
        }

    monkeypatch.setattr(web_app.PaperWhisperer, "analyze_stream", fake_analyze_stream)

    response = TestClient(web_app.app).post(
        "/api/analyze/stream",
        data={"api_key": "key", "generate_research_brief": "true"},
        files={"file": ("paper.txt", b"plain text document", "text/plain")},
    )

    assert response.status_code == 200
    assert "event: start" in response.text
    assert "event: section" in response.text
    assert '"name": "research_brief"' in response.text
    assert "event: done" in response.text
    assert '"session_token"' in response.text


def test_analyze_stream_error_event_is_structured(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "UPLOAD_FOLDER", str(tmp_path / "uploads"))
    os.makedirs(web_app.UPLOAD_FOLDER, exist_ok=True)

    def fail_analyze_stream(self, file_path, generate_mermaid=True, generate_evaluation=True, generate_research_brief=True):
        raise RuntimeError("stream failed")
        yield

    monkeypatch.setattr(web_app.PaperWhisperer, "analyze_stream", fail_analyze_stream)

    response = TestClient(web_app.app).post(
        "/api/analyze/stream",
        data={"api_key": "key"},
        files={"file": ("paper.txt", b"plain text document", "text/plain")},
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert '"error": "stream failed"' in response.text
    assert '"code": "streaming_analysis_failed"' in response.text
    assert '"timestamp"' in response.text


def test_retry_delay_is_capped():
    assert web_app.get_retry_delay_seconds(0) == 2
    assert web_app.get_retry_delay_seconds(2) == 6
    assert web_app.get_retry_delay_seconds(10) == 6
    assert web_app.get_retry_delay_seconds(10, max_delay=8) == 8
    assert web_app.get_retry_delay_seconds(-1) == 2
    assert web_app.get_retry_delay_seconds("bad") == 2
    assert web_app.get_retry_delay_seconds(2, max_delay=0) == 1
    assert web_app.get_retry_delay_seconds(2, max_delay="bad") == 6


def test_remote_error_descriptions_are_consistent():
    assert web_app.describe_remote_http_error(404) == "The paper file could not be found at the remote source."
    assert web_app.describe_remote_http_error(403) == "The remote source denied access to the paper file."
    assert web_app.describe_remote_http_error(429) == "The remote source rate limited the paper download. Please retry in a moment."
    assert web_app.describe_remote_http_error(500) == "Remote paper download failed with HTTP 500."
    assert web_app.describe_remote_url_error("timeout") == "Paper download failed: timeout"


def test_remote_pdf_validation_preserves_initial_chunk(monkeypatch, public_example_urls):
    body = b"%PDF-1.7\n" + (b"x" * 5000)
    patch_remote_response(monkeypatch, FakeResponse(body))

    response, file_name, content_type, initial_chunk = web_app.stream_remote_paper(
        title="Valid PDF",
        pdf_url="https://example.com/paper.pdf",
        url="",
    )

    assert file_name.endswith(".pdf")
    assert content_type == "application/pdf"
    assert initial_chunk == body[:4096]
    assert b"".join(web_app.iter_remote_file_chunks(response, web_app.MAX_CONTENT_LENGTH, initial_chunk)) == body


def test_remote_pdf_import_uses_app_user_agent(monkeypatch, public_example_urls):
    captured = {}

    def fake_urlopen(request, *args, **kwargs):
        captured["user_agent"] = request.get_header("User-agent")
        return FakeResponse(b"%PDF-1.7\nproxied body")

    monkeypatch.setattr(web_app.urllib.request, "urlopen", fake_urlopen)

    web_app.stream_remote_paper(title="Valid PDF", pdf_url="https://example.com/paper.pdf", url="")

    assert captured["user_agent"] == web_app.APP_USER_AGENT


def test_download_remote_paper_preserves_initial_chunk(monkeypatch, public_example_urls):
    body = b"%PDF-1.7\n" + (b"x" * 5000)
    patch_remote_response(monkeypatch, FakeResponse(body))

    temp_path, file_name = web_app.download_remote_paper(
        title="Valid PDF",
        pdf_url="https://example.com/paper.pdf",
        url="",
    )
    try:
        assert file_name.endswith(".pdf")
        assert Path(temp_path).read_bytes() == body
    finally:
        web_app.remove_file_safely(temp_path, "test imported paper")


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b"<!doctype html><html></html>", "text/html"),
        (b"<html></html>", "application/octet-stream"),
        (b"not a pdf", "application/pdf"),
        (b"%PDF-1.7\n", "application/json"),
    ],
)
def test_remote_file_validation_rejects_non_documents(monkeypatch, public_example_urls, body, content_type):
    patch_remote_response(monkeypatch, FakeResponse(body, content_type=content_type))

    with pytest.raises(ValueError):
        web_app.stream_remote_paper(title="Bad PDF", pdf_url="https://example.com/paper.pdf", url="")


def test_remote_file_validation_rejects_oversized_content_length(monkeypatch, public_example_urls):
    patch_remote_response(
        monkeypatch,
        FakeResponse(b"%PDF-1.7\n", content_length=web_app.MAX_CONTENT_LENGTH + 1),
    )

    with pytest.raises(ValueError):
        web_app.stream_remote_paper(title="Large PDF", pdf_url="https://example.com/paper.pdf", url="")


def test_download_proxy_serves_validated_pdf(monkeypatch, public_example_urls):
    body = b"%PDF-1.7\nproxied body"
    patch_remote_response(monkeypatch, FakeResponse(body))

    response = TestClient(web_app.app).get(
        "/api/download-paper",
        params={"pdf_url": "https://example.com/paper.pdf", "title": "Proxy PDF"},
    )

    assert response.status_code == 200
    assert response.content == body
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_download_proxy_rejects_html_response(monkeypatch, public_example_urls):
    patch_remote_response(monkeypatch, FakeResponse(b"<html></html>", content_type="text/html"))

    response = TestClient(web_app.app).get(
        "/api/download-paper",
        params={"pdf_url": "https://example.com/paper.pdf", "title": "HTML"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert "error" in payload
    assert payload["code"] == "invalid_request"
    assert "timestamp" in payload


@pytest.mark.parametrize(
    "path",
    [
        "/api/import-paper",
        "/api/ask",
        "/api/ask/stream",
        "/api/search-papers",
        "/api/reading-queue",
        "/api/recommend-papers",
    ],
)
def test_json_api_rejects_invalid_json(path):
    response = TestClient(web_app.app).post(
        path,
        content="{not valid json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "Invalid JSON body."
    assert payload["code"] == "invalid_json"
    assert "timestamp" in payload


def test_json_api_rejects_non_object_body():
    response = TestClient(web_app.app).post(
        "/api/ask",
        content="[]",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "JSON body must be an object."
    assert payload["code"] == "json_not_object"
    assert "timestamp" in payload


def test_parse_json_object_request_logs_body_read_failure(caplog):
    class BrokenRequest:
        async def body(self):
            raise RuntimeError("client disconnected")

    data, response = asyncio.run(web_app.parse_json_object_request(BrokenRequest()))

    assert data is None
    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload["error"] == "Unable to read request body."
    assert payload["code"] == "body_read_failed"
    assert "timestamp" in payload
    assert "Failed to read JSON request body: client disconnected" in caplog.text


def test_load_session_payload_removes_corrupted_json(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    session_file = tmp_path / "broken.json"
    session_file.write_text("{not valid json", encoding="utf-8")

    assert web_app.load_session_payload("broken") is None
    assert not session_file.exists()


def test_load_session_payload_normalizes_nested_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    session_file = tmp_path / "session.json"
    session_file.write_text(
        json.dumps(
            {
                "expires_at": web_app.build_session_expiry(),
                "document_content": "full text",
                "qa_history": "invalid",
                "analysis": {"sections": "invalid"},
                "paper_search": {"last_results": "invalid", "last_recommendation": "invalid"},
                "session_auth": "invalid",
            }
        ),
        encoding="utf-8",
    )

    payload = web_app.load_session_payload("session")

    assert payload["qa_history"] == []
    assert payload["analysis"]["sections"] == {}
    assert payload["paper_search"]["last_results"] == []
    assert payload["paper_search"]["last_recommendation"] == {}
    assert payload["paper_search"]["reading_queue"] == []
    assert payload["session_auth"] == {"token_hash": ""}
    assert payload["document_excerpt"]


def test_load_session_payload_normalizes_analysis_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    session_file = tmp_path / "session.json"
    session_file.write_text(
        json.dumps(
            {
                "expires_at": web_app.build_session_expiry(),
                "analysis": {
                    "sections": {},
                    "suggested_questions": ["  问题一？  ", "问题一？", ""],
                    "next_actions": [
                        {"label": "  行动  ", "prompt": "  请解释方法。  "},
                        {"label": "重复", "prompt": "请解释方法。"},
                        {"label": "无效"},
                    ],
                    "analysis_status": "invalid",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = web_app.load_session_payload("session")

    assert payload["analysis"]["suggested_questions"] == ["问题一？"]
    assert payload["analysis"]["next_actions"] == [{"label": "行动", "prompt": "请解释方法。"}]
    assert payload["analysis"]["analysis_status"] == {}


def test_finalize_analysis_sections_adds_smart_metadata():
    whisperer = web_app.PaperWhisperer("")
    sections = {
        "summary": web_app.build_section_result("success", "summary"),
        "quotes": web_app.build_section_result("success", "quotes"),
        "mindmap": web_app.build_section_result("success", "mindmap"),
        "mermaid": web_app.build_section_result("disabled"),
        "evaluation": web_app.build_section_result("disabled"),
        "research_brief": web_app.build_section_result("success", "brief"),
    }

    result = whisperer._finalize_analysis_sections("document content", sections)

    assert result["suggested_questions"]
    assert result["next_actions"]
    assert result["analysis_status"]["quality"] == "complete"
    assert "视觉图谱" in result["analysis_status"]["disabled_sections"]


def test_load_session_payload_normalizes_reading_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    session_file = tmp_path / "session.json"
    session_file.write_text(
        json.dumps(
            {
                "expires_at": web_app.build_session_expiry(),
                "paper_search": {
                    "reading_queue": [
                        {"title": "  Paper One  ", "authors": [{"name": "Alice"}], "year": "2024", "url": "https://example.com/1"},
                        {"title": "Paper One", "authors": ["Duplicate"]},
                        {"title": ""},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    payload = web_app.load_session_payload("session")

    assert payload["paper_search"]["reading_queue"] == [
        {
            "source": "",
            "paper_id": "",
            "title": "Paper One",
            "abstract": "",
            "authors": ["Alice"],
            "year": "2024",
            "venue": "",
            "url": "https://example.com/1",
            "pdf_url": "",
            "saved_at": payload["paper_search"]["reading_queue"][0]["saved_at"],
        }
    ]


def test_reading_queue_endpoint_requires_valid_session(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))

    response = TestClient(web_app.app).post(
        "/api/reading-queue",
        content=json.dumps({"session_id": "missing", "items": []}),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "invalid_request"
    assert "timestamp" in payload


def test_reading_queue_endpoint_saves_normalized_items(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    session_id = "session"
    token = "token"
    payload = {
        "expires_at": web_app.build_session_expiry(),
        "paper_search": {},
        "session_auth": {"token_hash": web_app.hash_session_token(token)},
    }
    web_app.write_session_payload(session_id, payload)

    response = TestClient(web_app.app).post(
        "/api/reading-queue",
        content=json.dumps(
            {
                "session_id": session_id,
                "session_token": token,
                "items": [
                    {"title": "Saved Paper", "authors": ["Alice"], "url": "https://example.com/paper"},
                    {"title": "Saved Paper", "authors": ["Duplicate"]},
                ],
            }
        ),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert web_app.load_session_payload(session_id)["paper_search"]["reading_queue"][0]["title"] == "Saved Paper"


def test_normalize_answer_mode_defaults_invalid_values():
    assert web_app.normalize_answer_mode("critique") == "critique"
    assert web_app.normalize_answer_mode("invalid") == "evidence"
    assert web_app.normalize_answer_mode(None) == "evidence"


def test_ask_api_persists_answer_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    token = "token"
    session_id = "session"
    web_app.write_session_payload(
        session_id,
        {
            "expires_at": web_app.build_session_expiry(),
            "document_content": "document",
            "qa_history": [],
            "session_auth": {"token_hash": web_app.hash_session_token(token)},
        },
    )
    monkeypatch.setattr(web_app.PaperWhisperer, "answer_question", lambda self, question, history=None, answer_mode="evidence": f"{answer_mode}: answer")

    response = TestClient(web_app.app).post(
        "/api/ask",
        content=json.dumps(
            {
                "session_id": session_id,
                "session_token": token,
                "api_key": "key",
                "question": "What?",
                "answer_mode": "reproduce",
            }
        ),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    payload = web_app.load_session_payload(session_id)
    assert payload["qa_history"][0]["answer_mode"] == "reproduce"
    assert response.json()["answer"] == "reproduce: answer"


def test_ask_stream_persists_answer_mode_and_sse_events(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    token = "token"
    session_id = "session"
    web_app.write_session_payload(
        session_id,
        {
            "expires_at": web_app.build_session_expiry(),
            "document_content": "document",
            "qa_history": [],
            "session_auth": {"token_hash": web_app.hash_session_token(token)},
        },
    )

    def fake_stream_answer(self, question, history=None, answer_mode="evidence"):
        yield f"{answer_mode}: "
        yield "streamed answer"

    monkeypatch.setattr(web_app.PaperWhisperer, "stream_answer_question", fake_stream_answer)

    response = TestClient(web_app.app).post(
        "/api/ask/stream",
        content=json.dumps(
            {
                "session_id": session_id,
                "session_token": token,
                "api_key": "key",
                "question": "What?",
                "answer_mode": "critique",
            }
        ),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert "event: start" in response.text
    assert "event: delta" in response.text
    assert "critique: " in response.text
    assert "streamed answer" in response.text
    assert "event: done" in response.text
    payload = web_app.load_session_payload(session_id)
    assert payload["qa_history"] == [
        {
            "question": "What?",
            "answer": "critique: streamed answer",
            "answer_mode": "critique",
            "timestamp": payload["qa_history"][0]["timestamp"],
        }
    ]


def test_ask_stream_error_event_is_structured(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    token = "token"
    session_id = "session"
    web_app.write_session_payload(
        session_id,
        {
            "expires_at": web_app.build_session_expiry(),
            "document_content": "document",
            "qa_history": [],
            "session_auth": {"token_hash": web_app.hash_session_token(token)},
        },
    )

    def fail_stream_answer(self, question, history=None, answer_mode="evidence"):
        raise RuntimeError("qa stream failed")
        yield

    monkeypatch.setattr(web_app.PaperWhisperer, "stream_answer_question", fail_stream_answer)

    response = TestClient(web_app.app).post(
        "/api/ask/stream",
        content=json.dumps(
            {
                "session_id": session_id,
                "session_token": token,
                "api_key": "key",
                "question": "What?",
            }
        ),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert '"error": "qa stream failed"' in response.text
    assert '"code": "streaming_question_failed"' in response.text
    assert '"timestamp"' in response.text


def test_cleanup_expired_sessions_removes_malformed_json(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    monkeypatch.setattr(web_app, "LAST_SESSION_CLEANUP_AT", 0.0)
    session_file = tmp_path / "broken.json"
    session_file.write_text("{not valid json", encoding="utf-8")

    web_app.cleanup_expired_sessions(force=True)

    assert not session_file.exists()
    assert "Removed unreadable session file" in caplog.text


def test_cleanup_expired_sessions_removes_non_object_json(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    monkeypatch.setattr(web_app, "LAST_SESSION_CLEANUP_AT", 0.0)
    session_file = tmp_path / "broken.json"
    session_file.write_text("[]", encoding="utf-8")

    web_app.cleanup_expired_sessions(force=True)

    assert not session_file.exists()
    assert "session payload must be an object" in caplog.text


def test_arxiv_search_uses_app_user_agent(monkeypatch):
    captured = {}

    def fake_http_get_text(url, timeout=None, headers=None, retries=None, ssl_context=None):
        captured["headers"] = headers
        return "<?xml version='1.0'?><feed xmlns='http://www.w3.org/2005/Atom'></feed>"

    monkeypatch.setattr(web_app, "http_get_text", fake_http_get_text)

    assert web_app.search_arxiv_papers("retrieval augmented generation", 2) == []
    assert captured["headers"]["User-Agent"] == web_app.APP_USER_AGENT


def test_semantic_scholar_search_uses_app_user_agent(monkeypatch):
    captured = {}

    def fake_http_get_json(url, timeout=None, headers=None, retries=None):
        captured["headers"] = headers
        return {"data": []}

    monkeypatch.setattr(web_app, "SEMANTIC_SCHOLAR_API_KEY", "")
    monkeypatch.setattr(web_app, "http_get_json", fake_http_get_json)

    assert web_app.search_semantic_scholar_papers("retrieval augmented generation", 2) == []
    assert captured["headers"]["User-Agent"] == web_app.APP_USER_AGENT


def test_search_papers_falls_back_to_direct_search_without_api_key(monkeypatch):
    monkeypatch.setattr(web_app, "PAPER_SEARCH_ENABLE_REWRITE", True)
    monkeypatch.setattr(web_app, "search_papers", lambda query, limit: {"query": query, "items": [], "errors": []})

    response = TestClient(web_app.app).post(
        "/api/search-papers",
        content=json.dumps({"query": "graph neural networks"}),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["rewritten_query"] == "graph neural networks"
    assert payload["rewrite_model"] == ""
    assert "direct search" in payload["reason"]


def test_search_papers_uses_direct_query_when_rewrite_disabled(monkeypatch):
    captured = {}
    monkeypatch.setattr(web_app, "PAPER_SEARCH_ENABLE_REWRITE", False)

    def fake_search(query, limit):
        captured["query"] = query
        captured["limit"] = limit
        return {"query": query, "items": [], "errors": []}

    monkeypatch.setattr(web_app, "search_papers", fake_search)

    response = TestClient(web_app.app).post(
        "/api/search-papers",
        content=json.dumps({"query": "  efficient transformers  ", "api_key": "unused", "limit": 3}),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert captured == {"query": "efficient transformers", "limit": 3}
    payload = response.json()
    assert payload["rewritten_query"] == "efficient transformers"
    assert payload["rewrite_model"] == ""
    assert payload["reason"] == "Direct search without AI rewriting."


def test_search_papers_rejects_invalid_session_token(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    monkeypatch.setattr(web_app, "search_papers", lambda query, limit: {"query": query, "items": [], "errors": []})
    session_id = "session"
    web_app.write_session_payload(
        session_id,
        {
            "expires_at": web_app.build_session_expiry(),
            "paper_search": {},
            "session_auth": {"token_hash": web_app.hash_session_token("correct-token")},
        },
    )

    response = TestClient(web_app.app).post(
        "/api/search-papers",
        content=json.dumps({"query": "rag", "session_id": session_id, "session_token": "wrong-token"}),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 403
    payload = response.json()
    assert payload["code"] == "invalid_session_token"
    assert "timestamp" in payload


def test_recommend_papers_rejects_invalid_session_token(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "CONTEXT_FOLDER", str(tmp_path))
    session_id = "session"
    web_app.write_session_payload(
        session_id,
        {
            "expires_at": web_app.build_session_expiry(),
            "document_content": "paper text",
            "session_auth": {"token_hash": web_app.hash_session_token("correct-token")},
        },
    )

    response = TestClient(web_app.app).post(
        "/api/recommend-papers",
        content=json.dumps({"session_id": session_id, "session_token": "wrong-token", "api_key": "key"}),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 403
    payload = response.json()
    assert payload["code"] == "invalid_session_token"
    assert "timestamp" in payload
