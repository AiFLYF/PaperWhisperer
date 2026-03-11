import json
import logging
import os
import re
import time
import unicodedata
import uuid
import threading
import concurrent.futures
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation

from env_loader import load_project_env


load_project_env()

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
CONTEXT_FOLDER = "context"
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max limit

app = FastAPI(title="PaperWhisperer")
templates = Jinja2Templates(directory="templates")

# Ensure required folders exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, CONTEXT_FOLDER]:
    os.makedirs(folder, exist_ok=True)


SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx", ".pptx")
ALLOWED_EXTENSIONS = set(SUPPORTED_EXTENSIONS)
SUPPORTED_FILE_TYPES_TEXT = ", ".join(SUPPORTED_EXTENSIONS)


def parse_int_env(name, default, min_value=1, max_value=32):
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def parse_bool_env(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_bool_value(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


MAX_LLM_CONCURRENCY = parse_int_env("OPENAI_MAX_CONCURRENCY", default=5, min_value=1, max_value=32)
LLM_REQUEST_SEMAPHORE = threading.BoundedSemaphore(MAX_LLM_CONCURRENCY)


def resolve_api_key(explicit_key):
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    return os.getenv("OPENAI_API_KEY", "").strip()


def is_allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def secure_filename(filename):
    value = unicodedata.normalize("NFKD", str(filename)).encode("ascii", "ignore").decode("ascii")
    value = value.replace("/", " ").replace("\\", " ")
    value = "_".join(value.split())
    value = re.sub(r"[^A-Za-z0-9_.-]", "", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("._")
    if os.name == "nt" and value and value.split(".")[0].upper() in {
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }:
        value = f"_{value}"
    return value


def sanitize_identifier(raw_value, prefix):
    candidate = secure_filename((raw_value or "").strip())
    return candidate or f"{prefix}_{uuid.uuid4().hex}"


def build_safe_upload_filename(filename):
    original_ext = os.path.splitext(filename)[1].lower()
    if original_ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type. Please upload one of: {SUPPORTED_FILE_TYPES_TEXT}")

    sanitized = secure_filename(filename)
    if not sanitized or not sanitized.lower().endswith(original_ext):
        return f"file_{uuid.uuid4().hex}{original_ext}"
    return sanitized


def clean_extracted_text(text):
    """Clean extracted text: collapse excessive blank lines and trim whitespace."""
    if not text:
        return ""
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse runs of whitespace (excluding newlines) into single space
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def get_session_file_path(session_id):
    return os.path.join(CONTEXT_FOLDER, f"{session_id}.json")


def build_document_excerpt(content, limit=12000):
    return (content or "")[:limit]


def trim_text_for_log(text, limit=2000):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def build_session_payload(session_id, source_filename, document_content, analysis):
    generated_at = datetime.now().isoformat(timespec="seconds")
    return {
        "session_id": session_id,
        "source_filename": source_filename,
        "generated_at": generated_at,
        "document_content": document_content,
        "document_excerpt": build_document_excerpt(document_content),
        "qa_history": [],
        "analysis": {
            "summary": analysis.get("summary", ""),
            "quotes": analysis.get("quotes", ""),
            "mindmap": analysis.get("mindmap", ""),
            "mermaid": analysis.get("mermaid", ""),
            "evaluation": analysis.get("evaluation", ""),
            "char_count": analysis.get("char_count", 0),
            "elapsed_seconds": analysis.get("elapsed_seconds"),
            "output_file": analysis.get("output_file", ""),
        },
    }


def is_failed_llm_result(value):
    text = (value or "").strip()
    return text.startswith("生成失败，请重试")


def describe_llm_status_code(status_code):
    status_messages = {
        400: "AI 服务请求格式错误，请检查模型配置、参数或接口兼容性。",
        401: "AI 服务认证失败，请检查 API Key 是否正确或已过期。",
        403: "AI 服务拒绝访问，当前 API Key 可能无权使用该模型或接口。",
        404: "AI 服务地址或模型不存在，请检查 OPENAI_BASE_URL 和模型名称。",
        408: "AI 服务请求超时，请稍后重试。",
        409: "AI 服务请求冲突，请稍后重试。",
        415: "AI 服务不支持当前请求媒体类型，请检查供应商兼容性。",
        422: "AI 服务无法处理当前请求，请检查输入内容或参数。",
        429: "AI 服务触发限流，请稍后重试或降低并发。",
        500: "AI 服务提供商内部错误，请稍后重试。",
        502: "AI 服务网关异常，请稍后重试。",
        503: "AI 服务暂时不可用，请稍后重试。",
        504: "AI 服务网关超时，请稍后重试。",
    }
    return status_messages.get(status_code, f"AI 服务请求失败，状态码: {status_code}。")


def looks_like_html_response(value):
    text = str(value or "").lstrip().lower()
    html_markers = ("<!doctype html", "<html", "<head", "<body", "<meta ")
    return any(text.startswith(marker) for marker in html_markers)


def extract_message_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item["text"]))
            else:
                item_type = getattr(item, "type", None)
                item_text = getattr(item, "text", None)
                if item_type == "text" and item_text:
                    text_parts.append(str(item_text))
        return "\n".join(text_parts).strip()
    return str(content).strip()


def write_session_payload(session_id, payload):
    with open(get_session_file_path(session_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_session_payload(session_id):
    session_file = get_session_file_path(session_id)
    if not os.path.exists(session_file):
        return None
    with open(session_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None
    payload.setdefault("document_content", "")
    payload.setdefault("document_excerpt", build_document_excerpt(payload.get("document_content", "")))
    payload.setdefault("qa_history", [])
    payload.setdefault("analysis", {})
    payload.setdefault("source_filename", "")
    payload.setdefault("session_id", session_id)
    payload.setdefault("generated_at", datetime.now().isoformat(timespec="seconds"))
    return payload


class TextChunker:
    """文本分块器，用于处理超长文本"""
    def __init__(self, chunk_size=4000, overlap=200):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text):
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try sentence boundaries: Chinese period, English period+space, newline
            if end < text_length:
                best_pos = -1
                for sep in ('。', '. ', '\n'):
                    pos = chunk.rfind(sep)
                    if pos > best_pos:
                        best_pos = pos

                if best_pos > self.chunk_size // 2:
                    chunk = chunk[:best_pos + 1]
                    end = start + best_pos + 1

            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - self.overlap

        return chunks


class DocumentLoader:
    """文档加载器，支持 TXT/PDF/DOCX/PPTX"""
    @staticmethod
    def load_txt(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    @staticmethod
    def load_pdf(file_path):
        try:
            reader = PdfReader(file_path)
            text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()
        except Exception as e:
            raise ValueError(f"PDF 读取失败: {str(e)}")

    @staticmethod
    def load_docx(file_path):
        try:
            doc = Document(file_path)
            parts = []

            # Extract paragraphs
            for p in doc.paragraphs:
                text = p.text.strip()
                if text:
                    parts.append(text)

            # Extract tables
            for table_index, table in enumerate(doc.tables, start=1):
                rows_text = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows_text.append(" | ".join(cells))
                if rows_text:
                    parts.append(f"[Table {table_index}]\n" + "\n".join(rows_text))

            return "\n\n".join(parts).strip()
        except Exception as e:
            raise ValueError(f"DOCX 读取失败: {str(e)}")

    @staticmethod
    def load_pptx(file_path):
        try:
            presentation = Presentation(file_path)
            slides_text = []
            for slide_index, slide in enumerate(presentation.slides, start=1):
                shape_texts = []
                for shape in slide.shapes:
                    text = getattr(shape, 'text', '')
                    if text and text.strip():
                        shape_texts.append(text.strip())

                # Extract slide notes
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        shape_texts.append(f"[Notes] {notes}")

                if shape_texts:
                    slides_text.append(f"[Slide {slide_index}]\n" + "\n".join(shape_texts))
            return "\n\n".join(slides_text).strip()
        except Exception as e:
            raise ValueError(f"PPTX 读取失败: {str(e)}")

    @staticmethod
    def load(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        loaders = {
            '.txt': DocumentLoader.load_txt,
            '.pdf': DocumentLoader.load_pdf,
            '.docx': DocumentLoader.load_docx,
            '.pptx': DocumentLoader.load_pptx,
        }
        loader = loaders.get(ext)
        if not loader:
            raise ValueError(f"不支持的文件格式: {ext} (支持: {SUPPORTED_FILE_TYPES_TEXT})")
        raw_text = loader(file_path)
        return clean_extracted_text(raw_text)


class PaperWhisperer:
    """文献分析核心类"""
    def __init__(self, api_key):
        self.name = "PaperWhisperer"
        self.version = "0.6.0"
        self.api_key = resolve_api_key(api_key)
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.request_timeout = parse_int_env("OPENAI_REQUEST_TIMEOUT_SECONDS", default=60, min_value=5, max_value=600)
        self.max_retries = parse_int_env("OPENAI_MAX_RETRIES", default=3, min_value=1, max_value=10)
        self.chunker = TextChunker(4000, 200)
        self.max_concurrency = MAX_LLM_CONCURRENCY
        self.summary_chunk_workers = min(3, self.max_concurrency)
        self.analysis_workers = min(5, self.max_concurrency)
        self.document_content = ""

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        ) if self.api_key else None

    def _call_llm(self, system_prompt, user_prompt, max_retries=None):
        if not self.client:
            raise ValueError("API key is required. Provide it in request body or set OPENAI_API_KEY.")

        retries = self.max_retries if max_retries is None else max_retries

        for attempt in range(retries):
            try:
                with LLM_REQUEST_SEMAPHORE:
                    raw_response = self.client.chat.completions.with_raw_response.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=4000,
                        timeout=self.request_timeout
                    )

                status_code = getattr(raw_response, "status_code", None)
                response = raw_response.parse()

                if status_code is None:
                    raise ValueError("AI 服务未返回可识别的 HTTP 状态码。")
                if not (200 <= status_code < 300):
                    raise ValueError(describe_llm_status_code(status_code))

                if isinstance(response, str):
                    if looks_like_html_response(response):
                        raise ValueError("AI 服务返回了网页内容而不是模型结果。通常是 API Key 缺失、无效，或 OPENAI_BASE_URL 指向了网页地址。")
                    raise ValueError("AI 服务返回了字符串而不是标准响应对象，请检查供应商接口兼容性。")

                choices = getattr(response, "choices", None)
                if not choices:
                    if looks_like_html_response(response):
                        raise ValueError("AI 服务返回了网页内容而不是模型结果。通常是 API Key 缺失、无效，或 OPENAI_BASE_URL 指向了网页地址。")
                    raise ValueError("AI 服务返回成功，但响应中缺少 choices 字段。")

                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", None) if message else None
                text_content = extract_message_text(content)
                if looks_like_html_response(text_content):
                    raise ValueError("AI 服务返回了网页内容而不是模型结果。通常是 API Key 缺失、无效，或 OPENAI_BASE_URL 指向了网页地址。")
                if not text_content:
                    raise ValueError("AI 服务返回内容为空")
                return text_content
            except APIStatusError as e:
                message = describe_llm_status_code(e.status_code)
            except APITimeoutError:
                message = "AI 服务请求超时，请稍后重试。"
            except APIConnectionError:
                message = "AI 服务连接失败，请检查网络、API 地址或供应商服务状态。"
            except Exception as e:
                message = str(e)

            if attempt < retries - 1:
                time.sleep(min(2 * (attempt + 1), 8))
            else:
                logger.error(message)
                return f"生成失败，请重试 ({message})"

    def _get_worker_count(self, task_count, configured_workers):
        return max(1, min(task_count, configured_workers))

    def _generate_summary_chunk(self, content):
        system_prompt = """你是一个专业的学术文献分析助手，擅长总结论文的核心观点。请用中文回复，保持专业、简洁、准确。
【极其重要的公式格式要求】：
1. 行内公式必须且只能使用单个美元符号包裹，例如：$E = mc^2$。绝对不要使用 \\( \\) 或 ( )。
2. 独立块级公式必须且只能使用双美元符号包裹，例如：$$\\int_0^1 x^2 dx$$。绝对不要使用 \\[ \\] 或[ ]。
3. 公式内部的下划线（_）和星号（*）不要做任何 Markdown 转义，直接输出原生的 LaTeX 代码。
4. 绝对不要把公式放在普通的代码块（```）中。"""

        user_prompt = f"""请仔细阅读以下文献内容，然后：
1. 提取 3-5 个核心观点（每个观点用一句话概括）
2. 找出 2-3 个最值得引用的金句
文献内容：{content}

请按以下格式输出：
## 核心观点
1. [观点1]
2. [观点2]
3. [观点3]

## 引用片段
- "[引用1，严格遵守公式格式要求保留原文公式]"
- "[引用2，严格遵守公式格式要求保留原文公式]"
"""
        return self._call_llm(system_prompt, user_prompt)

    def _merge_summaries(self, summaries):
        if not summaries:
            return None
        if len(summaries) == 1:
            return summaries[0]

        combined = "\n\n--- 章节 ---\n\n".join(summaries)

        system_prompt = """你是一个专业的学术文献分析助手，擅长整合多个文献片段的摘要。请用中文回复，保持专业、简洁、准确。
【极其重要的公式格式要求】：
1. 行内公式必须且只能使用单个美元符号包裹，例如：$E = mc^2$。绝对不要使用 \\( \\) 或 ( )。
2. 独立块级公式必须且只能使用双美元符号包裹，例如：$$\\int_0^1 x^2 dx$$。绝对不要使用 \\[ \\] 或[ ]。
3. 公式内部的下划线（_）和星号（*）不要做任何转义，直接输出原生的 LaTeX 代码。"""

        user_prompt = f"""以下是一篇长文献不同部分的摘要内容，请整合成一份完整、连贯的摘要：
{combined}

请按以下格式输出：
## 核心观点
[整合后的核心观点列表]

## 引用片段
[整合后的引用片段列表，保留原文公式]
"""
        return self._call_llm(system_prompt, user_prompt)

    def generate_summary(self, content):
        chunks = self.chunker.chunk_text(content)

        if len(chunks) == 1:
            return self._generate_summary_chunk(content)

        worker_count = self._get_worker_count(len(chunks), self.summary_chunk_workers)
        if worker_count == 1:
            chunk_summaries = [summary for summary in map(self._generate_summary_chunk, chunks) if summary]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                chunk_summaries = list(filter(None, executor.map(self._generate_summary_chunk, chunks)))

        if len(chunk_summaries) > 1:
            return self._merge_summaries(chunk_summaries)
        return chunk_summaries[0] if chunk_summaries else "无法生成摘要"

    def extract_quotes(self, content):
        system_prompt = """你是一个专业的学术文献分析助手，擅长从文献中提取重要的引用片段。请用中文回复，精确提取文献中的原句。
【极其重要的公式格式要求】：
1. 行内公式必须且只能使用单个美元符号包裹，例如：$E = mc^2$。绝对不要使用 \\( \\) 或 ( )。
2. 独立块级公式必须且只能使用双美元符号包裹，例如：$$\\int_0^1 x^2 dx$$。绝对不要使用 \\[ \\] 或[ ]。
3. 公式内部的下划线（_）和星号（*）不要做任何转义，直接输出原生的 LaTeX 代码。"""

        user_prompt = f"""请从以下文献中提取 3-5 个最值得引用的金句或核心观点：
{content[:15000]}

请按以下格式输出：
## 引用片段
1. "[原句1，严格按要求保留公式]"
2. "[原句2，严格按要求保留公式]"
3. "[原句3，严格按要求保留公式]"
"""
        return self._call_llm(system_prompt, user_prompt)

    def generate_mindmap(self, content):
        system_prompt = """你是一个专业的学术文献分析助手，擅长分析文献结构并生成思维导图。请用中文回复。如涉及公式，请严格使用 $...$ (行内) 或 $$...$$ (块级) 包裹。"""

        user_prompt = f"""请为以下文献生成一个文本格式的思维导图：
{content[:10000]}

请按以下格式输出：
## 思维导图
[使用 ├── 和 └── 符号的层级结构]
"""
        return self._call_llm(system_prompt, user_prompt)

    def generate_mermaid_mindmap(self, content):
        system_prompt = """你是一个专业的学术文献分析助手，擅长分析文献结构并生成 Mermaid 格式的思维导图。请直接输出 Mermaid 代码，不要添加任何解释。
重要提示：
1. 必须以 "graph TD" 或 "graph LR" 开头
2. 节点ID只能包含字母、数字和下划线
3. 节点文本用方括号包裹，如：A[标题]
4. 不要使用特殊字符，中文可以正常使用
5. 保持简洁，不要超过20个节点
6. Mermaid节点文本内不要包含复杂的LaTeX公式，以免渲染崩溃，请用简短的中文概括。"""

        user_prompt = f"""请为以下文献生成 Mermaid 格式的思维导图代码。
文献内容：{content[:4000]}

请只输出 Mermaid 代码，格式如下：
graph TD
    A[论文标题]
    A --> B[章节1]
    A --> C[章节2]
    B --> B1[小节1]
    B --> B2[小节2]
"""
        result = self._call_llm(system_prompt, user_prompt)
        if result:
            # 寻找真正的 Mermaid 代码起始行，过滤掉大模型输出的开头废话
            lines = result.strip().split('\n')
            start_idx = -1
            valid_prefixes = ("graph ", "mindmap", "flowchart ", "pie", "sequenceDiagram", "stateDiagram", "classDiagram")

            for i, line in enumerate(lines):
                if any(line.strip().startswith(prefix) for prefix in valid_prefixes):
                    start_idx = i
                    break

            if start_idx != -1:
                result = '\n'.join(lines[start_idx:]).strip()
            else:
                # 极端情况下：正则回退提取
                match = re.search(r'```(?:mermaid)?\s*\n(.*?)\n```', result, re.DOTALL | re.IGNORECASE)
                if match:
                    result = match.group(1).strip()
                else:
                    # 默认添加graph TD前缀
                    result = "graph TD\n" + result

            # 确保代码有效
            if not any(result.startswith(prefix) for prefix in valid_prefixes):
                result = "graph TD\n" + result
            return result
        return None

    def generate_evaluation(self, content):
        system_prompt = """你是一个专业的学术论文评审专家，擅长对论文进行批判性评价。请用中文回复，包括论文的优点、局限性、历史地位和贡献。
【极其重要的公式格式要求】：
1. 行内公式必须且只能使用单个美元符号包裹，例如：$E = mc^2$。绝对不要使用 \\( \\) 或 ( )。
2. 独立块级公式必须且只能使用双美元符号包裹，例如：$$\\int_0^1 x^2 dx$$。绝对不要使用 \\[ \\] 或[ ]。"""

        user_prompt = f"""请对以下文献进行总结性评价，包括：
1. **论文的主要贡献**：这篇论文的核心创新点是什么？
2. **历史地位**：在相关领域的重要性如何？是否是奠基性工作？
3. **主要优点**：论文的优势和创新之处
4. **局限性**：论文存在的问题或后续工作指出的缺点
5. **值得学习的地方**：对读者有什么启发？

文献内容：{content[:15000]}

请按以下格式输出：
## 论文评价

### 主要贡献
[评价内容]

### 历史地位
[评价内容]

### 主要优点
- 优点1
- 优点2

### 局限性
- 局限性1
- 局限性2

### 值得学习的地方
- 学习点1
- 学习点2
"""
        return self._call_llm(system_prompt, user_prompt)

    def answer_question(self, question, history=None):
        if not self.document_content:
            return "没有文档内容，请先上传文档进行分析。"

        history = history or []
        document_budget = 12000
        total_history_budget = 8000
        per_turn_budget = 2400
        document_window = (self.document_content or "")[:document_budget]

        history_sections = []
        used_history_chars = 0
        for turn in reversed(history):
            question_text = trim_text_for_log(turn.get("question", ""), limit=400)
            answer_text = trim_text_for_log(turn.get("answer", ""), limit=800)
            if not question_text and not answer_text:
                continue

            section = f"Q: {question_text}\nA: {answer_text}"
            section_length = len(section)
            if section_length > per_turn_budget:
                section = section[:per_turn_budget].rstrip() + "\n...[truncated]"
                section_length = len(section)

            if used_history_chars + section_length > total_history_budget:
                break

            history_sections.append(section)
            used_history_chars += section_length

        history_sections.reverse()
        history_block = "\n\n---\n\n".join(history_sections)

        system_prompt = (
            "你是专业学术助手。请优先基于给定文档内容回答问题，"
            "若文档中没有答案，请明确说明。"
            "可以参考此前问答历史来理解上下文，但不要把历史结论当作高于文档的事实来源。"
            "避免重复复述已经确认过的文档段落。"
            "如涉及公式，请严格使用 $...$ (行内) 或 $$...$$ (块级) 包裹。"
        )

        user_prompt_parts = [
            f"文档内容（节选）:\n{document_window}",
        ]
        if history_block:
            user_prompt_parts.append(f"最近问答历史:\n{history_block}")
        user_prompt_parts.append(f"用户当前问题:\n{question}")
        user_prompt_parts.append("请给出简洁、准确的中文回答；如果是在追问，请延续上下文但不要重复长段原文。")
        user_prompt = "\n\n".join(user_prompt_parts)
        return self._call_llm(system_prompt, user_prompt)

    def analyze(self, file_path, generate_mermaid=True, generate_evaluation=True):
        """核心分析流程（已优化为并发执行）"""
        content = DocumentLoader.load(file_path)
        self.document_content = content

        result = {'char_count': len(content)}

        task_count = 3 + int(generate_mermaid) + int(generate_evaluation)
        worker_count = self._get_worker_count(task_count, self.analysis_workers)

        t_start = time.time()

        # 使用线程池并发执行大模型请求，减少整体等待时间
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_summary = executor.submit(self.generate_summary, content)
            future_quotes = executor.submit(self.extract_quotes, content)
            future_mindmap = executor.submit(self.generate_mindmap, content)

            future_mermaid = executor.submit(self.generate_mermaid_mindmap, content) if generate_mermaid else None
            future_eval = executor.submit(self.generate_evaluation, content) if generate_evaluation else None

            # 获取结果
            result['summary'] = future_summary.result()
            result['quotes'] = future_quotes.result()
            result['mindmap'] = future_mindmap.result()

            if future_mermaid:
                result['mermaid'] = future_mermaid.result()
            if future_eval:
                result['evaluation'] = future_eval.result()

        elapsed = time.time() - t_start
        result['elapsed_seconds'] = round(elapsed, 1)

        required_fields = [result.get('summary'), result.get('quotes'), result.get('mindmap')]
        failed_required_fields = [value for value in required_fields if is_failed_llm_result(value)]
        if len(failed_required_fields) == len(required_fields):
            raise RuntimeError(failed_required_fields[0])

        logger.info(f"Analysis completed in {elapsed:.1f}s for {os.path.basename(file_path)} ({len(content)} chars)")
        return result

# ================= 路由控制 =================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/analyze")
async def analyze(
    file: UploadFile | None = File(None),
    api_key: str = Form(""),
    generate_mermaid: str | None = Form(None),
    generate_evaluation: str | None = Form(None),
    session_id: str = Form(""),
):
    file_path = None

    if file is None:
        return JSONResponse(content={"error": "Please upload a file."}, status_code=400)

    if file.filename == "":
        return JSONResponse(content={"error": "Please select a file."}, status_code=400)
    if not is_allowed_file(file.filename):
        return JSONResponse(
            content={"error": f"Unsupported file type. Please upload one of: {SUPPORTED_FILE_TYPES_TEXT}"},
            status_code=400,
        )

    try:
        original_filename = build_safe_upload_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        generate_mermaid_bool = parse_bool_value(generate_mermaid, default=True)
        generate_evaluation_bool = parse_bool_value(generate_evaluation, default=True)

        resolved_api_key = resolve_api_key(api_key)
        if not resolved_api_key:
            return JSONResponse(
                content={"error": "API key is required. Provide api_key or set OPENAI_API_KEY."},
                status_code=400,
            )

        whisperer = PaperWhisperer(resolved_api_key)
        result = whisperer.analyze(file_path, generate_mermaid_bool, generate_evaluation_bool)

        safe_session_id = sanitize_identifier(session_id, "session")

        base_name = os.path.splitext(original_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_analysis_{timestamp}.md")

        md_content = f"""# PaperWhisperer 分析报告

> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 源文件: {original_filename}
> 耗时: {result.get('elapsed_seconds', 'N/A')}s

---

## AI 摘要

{result.get('summary', '')}

---

## 引用片段

{result.get('quotes', '')}

---

## 思维导图

{result.get('mindmap', '')}

---

"""

        if generate_evaluation_bool:
            md_content += f"## 论文评价\n\n{result.get('evaluation', '')}\n\n---\n"

        md_content += f"## 元信息\n\n- 版本: {whisperer.version}\n- 字符数: {result['char_count']}\n"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        result["output_file"] = output_file
        result["session_id"] = safe_session_id

        session_payload = build_session_payload(
            session_id=safe_session_id,
            source_filename=original_filename,
            document_content=whisperer.document_content,
            analysis=result,
        )
        write_session_payload(safe_session_id, session_payload)

        return JSONResponse(content=result)

    except Exception as e:
        logger.exception("Document analysis failed")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if file:
            await file.close()
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


@app.post("/api/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    data = data or {}
    if not isinstance(data, dict):
        return JSONResponse(content={"error": "JSON body must be an object."}, status_code=400)

    question = data.get("question", "").strip()
    raw_session_id = data.get("session_id", "")

    if not question:
        return JSONResponse(content={"error": "Please enter a question."}, status_code=400)

    safe_session_id = sanitize_identifier(raw_session_id, "session")

    resolved_api_key = resolve_api_key(data.get("api_key", ""))
    if not resolved_api_key:
        return JSONResponse(
            content={"error": "API key is required. Provide api_key or set OPENAI_API_KEY."},
            status_code=400,
        )

    try:
        session_payload = load_session_payload(safe_session_id)
        if not session_payload:
            return JSONResponse(
                content={"error": "Session expired or context not found. Please upload and analyze the file again."},
                status_code=400,
            )

        whisperer = PaperWhisperer(resolved_api_key)
        whisperer.document_content = session_payload.get("document_content", "")

        t_start = time.time()
        answer = whisperer.answer_question(question, history=session_payload.get("qa_history", []))
        elapsed = time.time() - t_start

        qa_history = session_payload.get("qa_history", [])
        qa_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        session_payload["qa_history"] = qa_history
        session_payload["document_excerpt"] = build_document_excerpt(session_payload.get("document_content", ""))
        write_session_payload(safe_session_id, session_payload)

        logger.info(f"Q&A completed in {elapsed:.1f}s for session {safe_session_id}")
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        logger.exception("Question answering failed")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    host = os.getenv("FASTAPI_HOST") or os.getenv("FLASK_HOST", "0.0.0.0")
    port = parse_int_env(
        "FASTAPI_PORT",
        default=parse_int_env("FLASK_PORT", default=5000, min_value=1, max_value=65535),
        min_value=1,
        max_value=65535,
    )
    reload_enabled = parse_bool_env("FASTAPI_RELOAD", default=parse_bool_env("FLASK_DEBUG", default=False))

    uvicorn.run("web_app:app", host=host, port=port, reload=reload_enabled)
