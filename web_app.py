import concurrent.futures
import hashlib
import ipaddress
import json
import logging
import mimetypes
import os
import re
import secrets
import ssl
import time
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime

try:
    import certifi
except ImportError:
    certifi = None

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
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


ARXIV_API_URL = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_TIMEOUT_SECONDS = parse_int_env("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS", default=20, min_value=5, max_value=120)
SEMANTIC_SCHOLAR_MAX_RETRIES = parse_int_env("SEMANTIC_SCHOLAR_MAX_RETRIES", default=3, min_value=1, max_value=6)
PAPER_SEARCH_RESULT_LIMIT = parse_int_env("PAPER_SEARCH_RESULT_LIMIT", default=8, min_value=1, max_value=20)
RECOMMENDATION_RESULT_LIMIT = parse_int_env("RECOMMENDATION_RESULT_LIMIT", default=6, min_value=1, max_value=20)
PAPER_SEARCH_ENABLE_REWRITE = parse_bool_env("PAPER_SEARCH_ENABLE_REWRITE", default=True)
PAPER_SEARCH_REWRITE_MODEL = os.getenv("PAPER_SEARCH_REWRITE_MODEL", "").strip()
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
REMOTE_IMPORT_TIMEOUT_SECONDS = parse_int_env("REMOTE_IMPORT_TIMEOUT_SECONDS", default=30, min_value=5, max_value=180)
SESSION_TTL_SECONDS = parse_int_env("SESSION_TTL_SECONDS", default=24 * 60 * 60, min_value=60, max_value=30 * 24 * 60 * 60)
SESSION_CLEANUP_INTERVAL_SECONDS = parse_int_env("SESSION_CLEANUP_INTERVAL_SECONDS", default=10 * 60, min_value=60, max_value=24 * 60 * 60)
SESSION_PERSIST_FULL_DOCUMENT = parse_bool_env("SESSION_PERSIST_FULL_DOCUMENT", default=True)

MAX_LLM_CONCURRENCY = parse_int_env("OPENAI_MAX_CONCURRENCY", default=5, min_value=1, max_value=32)
LLM_REQUEST_SEMAPHORE = threading.BoundedSemaphore(MAX_LLM_CONCURRENCY)
LAST_SESSION_CLEANUP_AT = 0.0


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


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def parse_iso_datetime(value):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def build_session_expiry(now=None):
    base_time = now or datetime.now()
    return datetime.fromtimestamp(base_time.timestamp() + SESSION_TTL_SECONDS).isoformat(timespec="seconds")


def generate_session_token():
    return secrets.token_urlsafe(24)


def hash_session_token(token):
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def build_document_excerpt(content, limit=12000):
    return (content or "")[:limit]


def trim_text_for_log(text, limit=2000):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def build_ssl_context():
    if certifi:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


def http_get_json(url, timeout=20, headers=None, retries=1, ssl_context=None):
    request = urllib.request.Request(url, headers=headers or {})
    last_error = None
    for attempt in range(max(1, retries)):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return json.loads(response.read().decode(charset, errors="ignore"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(min(2 * (attempt + 1), 6))
                continue
            raise
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(min(2 * (attempt + 1), 6))
                continue
            raise
    raise last_error


def http_get_text(url, timeout=20, headers=None, retries=1, ssl_context=None):
    request = urllib.request.Request(url, headers=headers or {})
    last_error = None
    for attempt in range(max(1, retries)):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl_context) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="ignore")
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(min(2 * (attempt + 1), 6))
                continue
            raise
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(min(2 * (attempt + 1), 6))
                continue
            raise
    raise last_error


def compact_text(text, limit=400):
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."


def parse_year(value):
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else ""


def normalize_author_list(authors, limit=8):
    normalized = []
    for author in authors or []:
        if isinstance(author, str):
            name = author.strip()
        elif isinstance(author, dict):
            name = str(author.get("name") or author.get("author") or "").strip()
        else:
            name = str(getattr(author, "name", "") or "").strip()
        if name:
            normalized.append(name)
        if len(normalized) >= limit:
            break
    return normalized


def normalize_paper_record(source, record):
    if source == "Semantic Scholar":
        open_access_pdf = record.get("openAccessPdf") or {}
        return {
            "source": source,
            "paper_id": str(record.get("paperId") or "").strip(),
            "title": compact_text(record.get("title") or "", limit=300),
            "abstract": compact_text(record.get("abstract") or "", limit=2000),
            "authors": normalize_author_list(record.get("authors") or []),
            "year": parse_year(record.get("year")),
            "venue": compact_text(record.get("venue") or "Semantic Scholar", limit=120),
            "url": str(record.get("url") or "").strip(),
            "pdf_url": str(open_access_pdf.get("url") or "").strip(),
        }

    return {
        "source": source,
        "paper_id": str(record.get("paper_id") or record.get("id") or "").strip(),
        "title": compact_text(record.get("title") or "", limit=300),
        "abstract": compact_text(record.get("abstract") or record.get("summary") or "", limit=2000),
        "authors": normalize_author_list(record.get("authors") or []),
        "year": parse_year(record.get("year") or record.get("published") or ""),
        "venue": compact_text(record.get("venue") or source, limit=120),
        "url": str(record.get("url") or record.get("id") or "").strip(),
        "pdf_url": str(record.get("pdf_url") or "").strip(),
    }


def deduplicate_papers(items):
    deduplicated = []
    seen_titles = set()
    for item in items or []:
        title_key = re.sub(r"\s+", " ", str(item.get("title") or "")).strip().lower()
        if not title_key or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        deduplicated.append(item)
    return deduplicated


def search_arxiv_papers(query, limit):
    encoded_query = urllib.parse.quote(query)
    url = f"{ARXIV_API_URL}?search_query=all:{encoded_query}&start=0&max_results={limit}"
    feed_text = http_get_text(
        url,
        timeout=SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
        headers={"User-Agent": "PaperWhisperer/0.7"},
        retries=2,
        ssl_context=build_ssl_context(),
    )
    root = ET.fromstring(feed_text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    items = []

    for entry in root.findall("atom:entry", namespace):
        title = compact_text(entry.findtext("atom:title", default="", namespaces=namespace), limit=300)
        summary = compact_text(entry.findtext("atom:summary", default="", namespaces=namespace), limit=2000)
        paper_id = (entry.findtext("atom:id", default="", namespaces=namespace) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=namespace) or "").strip()
        authors = [author.findtext("atom:name", default="", namespaces=namespace) for author in entry.findall("atom:author", namespace)]
        pdf_url = ""
        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "").strip()
                break
        items.append(normalize_paper_record("arXiv", {
            "paper_id": paper_id,
            "title": title,
            "abstract": summary,
            "authors": authors,
            "published": published,
            "venue": "arXiv",
            "url": paper_id,
            "pdf_url": pdf_url,
        }))
    return items


def search_semantic_scholar_papers(query, limit):
    params = urllib.parse.urlencode({
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,venue,url,authors,openAccessPdf,paperId",
    })
    url = f"{SEMANTIC_SCHOLAR_SEARCH_URL}?{params}"
    headers = {"User-Agent": "PaperWhisperer/0.7"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    payload = http_get_json(
        url,
        timeout=SEMANTIC_SCHOLAR_TIMEOUT_SECONDS,
        headers=headers,
        retries=SEMANTIC_SCHOLAR_MAX_RETRIES,
    )
    return [normalize_paper_record("Semantic Scholar", item) for item in payload.get("data", [])]


def search_papers(query, limit=None):
    clean_query = compact_text(query, limit=240)
    if not clean_query:
        raise ValueError("Please enter a search query.")

    try:
        resolved_limit = int(limit or PAPER_SEARCH_RESULT_LIMIT)
    except (TypeError, ValueError):
        resolved_limit = PAPER_SEARCH_RESULT_LIMIT
    resolved_limit = max(1, min(resolved_limit, PAPER_SEARCH_RESULT_LIMIT))
    items = []
    errors = []

    for source_name, search_fn in (
        ("Semantic Scholar", search_semantic_scholar_papers),
        ("arXiv", search_arxiv_papers),
    ):
        try:
            items.extend(search_fn(clean_query, resolved_limit))
        except urllib.error.HTTPError as exc:
            logger.warning("%s paper search failed: %s", source_name, exc)
            if exc.code == 429:
                errors.append(f"{source_name}: rate limit reached, please retry in a moment")
            else:
                errors.append(f"{source_name}: HTTP {exc.code}")
        except ssl.SSLCertVerificationError:
            logger.warning("%s paper search SSL verification failed", source_name)
            errors.append(f"{source_name}: SSL certificate verification failed")
        except urllib.error.URLError as exc:
            logger.warning("%s paper search failed: %s", source_name, exc)
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
                errors.append(f"{source_name}: SSL certificate verification failed")
            else:
                errors.append(f"{source_name}: {reason}")
        except Exception as exc:
            logger.warning("%s paper search failed: %s", source_name, exc)
            errors.append(f"{source_name}: {exc}")

    return {
        "query": clean_query,
        "items": deduplicate_papers(items)[:resolved_limit],
        "errors": errors,
    }


def build_session_payload(session_id, source_filename, document_content, analysis, session_token):
    generated_at = now_iso()
    stored_document_content = document_content if SESSION_PERSIST_FULL_DOCUMENT else ""
    return {
        "session_id": session_id,
        "source_filename": source_filename,
        "generated_at": generated_at,
        "created_at": generated_at,
        "updated_at": generated_at,
        "expires_at": build_session_expiry(),
        "document_content": stored_document_content,
        "document_excerpt": build_document_excerpt(document_content),
        "qa_history": [],
        "paper_search": {
            "last_query": "",
            "last_results": [],
            "last_recommendation": {},
        },
        "session_auth": {
            "token_hash": hash_session_token(session_token),
        },
        "analysis": {
            "summary": analysis.get("summary", ""),
            "quotes": analysis.get("quotes", ""),
            "mindmap": analysis.get("mindmap", ""),
            "mermaid": analysis.get("mermaid", ""),
            "evaluation": analysis.get("evaluation", ""),
            "sections": analysis.get("sections", {}),
            "char_count": analysis.get("char_count", 0),
            "elapsed_seconds": analysis.get("elapsed_seconds"),
            "output_file": analysis.get("output_file", ""),
        },
    }


def is_failed_llm_result(value):
    text = (value or "").strip()
    return text.startswith("生成失败，请重试")


def build_section_result(status, content="", error="", retryable=False):
    return {
        "status": status,
        "content": content or "",
        "error": error or "",
        "retryable": bool(retryable),
    }


def build_sse_event(event_name, payload):
    data = json.dumps(payload or {}, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"


def build_sse_headers():
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def is_retryable_llm_error(message):
    normalized = str(message or "").lower()
    retry_markers = (
        "超时",
        "timeout",
        "连接失败",
        "connection",
        "429",
        "限流",
        "502",
        "503",
        "504",
        "网关",
    )
    return any(marker in normalized for marker in retry_markers)


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


def extract_json_object(text):
    raw_text = str(text or "").strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        return raw_text[start:end + 1]
    return raw_text


def write_session_payload(session_id, payload):
    now_text = now_iso()
    if isinstance(payload, dict):
        payload["updated_at"] = now_text
        payload["expires_at"] = build_session_expiry()
    with open(get_session_file_path(session_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_content_type(value):
    return str(value or "").split(";", 1)[0].strip().lower()


def is_public_http_url(raw_url):
    try:
        parsed = urllib.parse.urlparse(str(raw_url or "").strip())
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False
    hostname = (parsed.hostname or "").strip()
    if not hostname:
        return False
    lowered = hostname.lower()
    if lowered in {"localhost", "localhost.localdomain"}:
        return False

    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        return True

    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def looks_like_direct_file_url(raw_url):
    path = urllib.parse.urlparse(str(raw_url or "").strip()).path.lower()
    return any(path.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def guess_extension_from_content_type(content_type):
    normalized = normalize_content_type(content_type)
    mapping = {
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    }
    return mapping.get(normalized, "")


def extract_filename_from_content_disposition(content_disposition):
    value = str(content_disposition or "")
    match = re.search(r"filename\*=UTF-8''([^;]+)", value, flags=re.IGNORECASE)
    if match:
        return urllib.parse.unquote(match.group(1)).strip('" ')
    match = re.search(r'filename="?([^";]+)"?', value, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def build_import_filename(title, source_url, content_disposition, content_type):
    disposition_name = extract_filename_from_content_disposition(content_disposition)
    url_name = os.path.basename(urllib.parse.urlparse(str(source_url or "")).path)
    title_slug = secure_filename(title or "")
    candidate_name = disposition_name or url_name or title_slug or "imported_paper"
    candidate_root, candidate_ext = os.path.splitext(candidate_name)
    inferred_ext = candidate_ext.lower() if candidate_ext.lower() in ALLOWED_EXTENSIONS else ""
    if not inferred_ext:
        inferred_ext = guess_extension_from_content_type(content_type)
    if not inferred_ext:
        inferred_ext = ".pdf"
    safe_root = secure_filename(candidate_root) or title_slug or "imported_paper"
    return build_safe_upload_filename(f"{safe_root}{inferred_ext}")


def iter_downloadable_paper_urls(pdf_url, url):
    seen = set()
    for candidate, require_direct_file in ((pdf_url, False), (url, True)):
        normalized_candidate = str(candidate or "").strip()
        if not normalized_candidate or normalized_candidate in seen:
            continue
        seen.add(normalized_candidate)
        yield normalized_candidate, require_direct_file


def stream_remote_paper(title, pdf_url, url):
    candidate_urls = list(iter_downloadable_paper_urls(pdf_url, url))
    if not candidate_urls:
        raise ValueError("No downloadable paper link found for this result.")

    ssl_context = build_ssl_context()
    last_error = None

    for source_url, require_direct_file in candidate_urls:
        if not is_public_http_url(source_url):
            last_error = ValueError("Only public http/https paper URLs are allowed.")
            continue
        if require_direct_file and not looks_like_direct_file_url(source_url):
            last_error = ValueError("This result does not provide a direct downloadable file. Please open it manually and upload the paper file.")
            continue

        request = urllib.request.Request(source_url, headers={"User-Agent": "PaperWhisperer/0.8"})
        try:
            response = urllib.request.urlopen(request, timeout=REMOTE_IMPORT_TIMEOUT_SECONDS, context=ssl_context)
            content_type = normalize_content_type(response.headers.get("Content-Type"))
            if content_type.startswith("text/html"):
                response.close()
                raise ValueError("The paper link returned an HTML page instead of a downloadable file.")

            file_name = build_import_filename(
                title=title,
                source_url=response.geturl() or source_url,
                content_disposition=response.headers.get("Content-Disposition"),
                content_type=content_type,
            )
            if not is_allowed_file(file_name):
                response.close()
                raise ValueError(f"Unsupported remote file type. Please use one of: {SUPPORTED_FILE_TYPES_TEXT}")

            return response, file_name, content_type
        except Exception as exc:
            last_error = exc

    if last_error:
        raise last_error
    raise ValueError("Paper import failed.")


async def save_upload_file(upload_file, destination_path, max_bytes):
    total_bytes = 0
    try:
        with open(destination_path, "wb") as f:
            while True:
                chunk = await upload_file.read(1024 * 64)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError(f"Uploaded file is too large. Limit: {max_bytes // (1024 * 1024)} MB")
                f.write(chunk)
        if total_bytes <= 0:
            raise ValueError("Uploaded file is empty.")
        return total_bytes
    except Exception:
        if destination_path and os.path.exists(destination_path):
            try:
                os.remove(destination_path)
            except Exception:
                pass
        raise


def iter_remote_file_chunks(response, max_bytes):
    total_bytes = 0
    try:
        while True:
            chunk = response.read(1024 * 64)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(f"Remote file is too large. Limit: {max_bytes // (1024 * 1024)} MB")
            yield chunk
        if total_bytes <= 0:
            raise ValueError("Downloaded file is empty.")
    finally:
        try:
            response.close()
        except Exception:
            pass


def cleanup_expired_sessions(force=False):
    global LAST_SESSION_CLEANUP_AT
    now_ts = time.time()
    if not force and now_ts - LAST_SESSION_CLEANUP_AT < SESSION_CLEANUP_INTERVAL_SECONDS:
        return
    LAST_SESSION_CLEANUP_AT = now_ts
    for name in os.listdir(CONTEXT_FOLDER):
        if not name.endswith(".json"):
            continue
        file_path = os.path.join(CONTEXT_FOLDER, name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            expires_at = parse_iso_datetime((payload or {}).get("expires_at"))
            if expires_at and expires_at.timestamp() < now_ts:
                os.remove(file_path)
        except Exception:
            continue


def validate_session_token(session_payload, session_token):
    expected_hash = str((session_payload.get("session_auth") or {}).get("token_hash") or "")
    provided_hash = hash_session_token(session_token)
    return bool(expected_hash and session_token and secrets.compare_digest(expected_hash, provided_hash))


def get_session_document_content(session_payload):
    content = str(session_payload.get("document_content") or "")
    if content:
        return content
    return str(session_payload.get("document_excerpt") or "")


def load_validated_session(raw_session_id, session_token, require_token=True):
    if not raw_session_id:
        raise ValueError("session_id is required.")
    safe_session_id = sanitize_identifier(raw_session_id, "session")
    session_payload = load_session_payload(safe_session_id)
    if not session_payload:
        raise ValueError("Session expired or context not found. Please upload and analyze the file again.")
    if require_token and not validate_session_token(session_payload, session_token):
        raise PermissionError("Invalid or missing session token. Please analyze the document again.")
    return safe_session_id, session_payload


def download_remote_paper(title, pdf_url, url):
    response = None
    temp_path = None
    try:
        response, file_name, _content_type = stream_remote_paper(title=title, pdf_url=pdf_url, url=url)
        temp_path = os.path.join(UPLOAD_FOLDER, file_name)
        total_bytes = 0
        with open(temp_path, "wb") as f:
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_CONTENT_LENGTH:
                    raise ValueError(f"Remote file is too large. Limit: {MAX_CONTENT_LENGTH // (1024 * 1024)} MB")
                f.write(chunk)

        if total_bytes <= 0:
            raise ValueError("Downloaded file is empty.")

        return temp_path, file_name
    except Exception:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise
    finally:
        if response:
            try:
                response.close()
            except Exception:
                pass


def finalize_analysis_result(result, whisperer, original_filename, generate_evaluation_bool, session_id):
    safe_session_id = sanitize_identifier(session_id, "session")
    session_token = generate_session_token()
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
    result["session_token"] = session_token
    result["source_filename"] = original_filename

    session_payload = build_session_payload(
        session_id=safe_session_id,
        source_filename=original_filename,
        document_content=whisperer.document_content,
        analysis=result,
        session_token=session_token,
    )
    write_session_payload(safe_session_id, session_payload)
    return result


def analyze_saved_file(file_path, original_filename, api_key, generate_mermaid_bool, generate_evaluation_bool, session_id):
    cleanup_expired_sessions()
    resolved_api_key = resolve_api_key(api_key)
    if not resolved_api_key:
        raise ValueError("API key is required. Provide api_key or set OPENAI_API_KEY.")

    whisperer = PaperWhisperer(resolved_api_key)
    result = whisperer.analyze(file_path, generate_mermaid_bool, generate_evaluation_bool)
    return finalize_analysis_result(
        result=result,
        whisperer=whisperer,
        original_filename=original_filename,
        generate_evaluation_bool=generate_evaluation_bool,
        session_id=session_id,
    )


def load_session_payload(session_id):
    session_file = get_session_file_path(session_id)
    if not os.path.exists(session_file):
        return None
    with open(session_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return None

    expires_at = parse_iso_datetime(payload.get("expires_at"))
    if expires_at and expires_at.timestamp() < time.time():
        try:
            os.remove(session_file)
        except Exception:
            pass
        return None

    payload.setdefault("document_content", "")
    payload.setdefault("document_excerpt", build_document_excerpt(payload.get("document_content", "")))
    payload.setdefault("qa_history", [])
    payload.setdefault("analysis", {})
    payload.setdefault("paper_search", {})
    payload["paper_search"].setdefault("last_query", "")
    payload["paper_search"].setdefault("last_results", [])
    payload["paper_search"].setdefault("last_recommendation", {})
    payload.setdefault("source_filename", "")
    payload.setdefault("session_id", session_id)
    payload.setdefault("generated_at", now_iso())
    payload.setdefault("created_at", payload.get("generated_at") or now_iso())
    payload.setdefault("updated_at", payload.get("generated_at") or now_iso())
    payload.setdefault("expires_at", build_session_expiry())
    payload.setdefault("session_auth", {})
    payload["session_auth"].setdefault("token_hash", "")
    payload["analysis"].setdefault("sections", {})
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
        self.version = "0.8.0"
        self.api_key = resolve_api_key(api_key)
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.search_rewrite_model = PAPER_SEARCH_REWRITE_MODEL or self.model
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

    def _call_llm(self, system_prompt, user_prompt, max_retries=None, model=None):
        if not self.client:
            raise ValueError("API key is required. Provide it in request body or set OPENAI_API_KEY.")

        retries = self.max_retries if max_retries is None else max_retries

        for attempt in range(retries):
            try:
                with LLM_REQUEST_SEMAPHORE:
                    raw_response = self.client.chat.completions.with_raw_response.create(
                        model=(model or self.model),
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
                raise RuntimeError(message)

    def _stream_llm(self, system_prompt, user_prompt, max_retries=None, model=None):
        if not self.client:
            raise ValueError("API key is required. Provide it in request body or set OPENAI_API_KEY.")

        retries = self.max_retries if max_retries is None else max_retries

        for attempt in range(retries):
            try:
                with LLM_REQUEST_SEMAPHORE:
                    with self.client.chat.completions.with_streaming_response.create(
                        model=(model or self.model),
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=4000,
                        timeout=self.request_timeout,
                        stream=True,
                    ) as response:
                        status_code = getattr(response, "status_code", None)
                        if status_code is None:
                            raise ValueError("AI 服务未返回可识别的 HTTP 状态码。")
                        if not (200 <= status_code < 300):
                            raise ValueError(describe_llm_status_code(status_code))

                        saw_text = False
                        for chunk in response.iter_lines():
                            if not chunk:
                                continue
                            decoded = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
                            line = decoded.strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            try:
                                payload = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            choices = payload.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            text = delta.get("content")
                            if not text:
                                continue
                            saw_text = True
                            yield str(text)

                        if not saw_text:
                            raise ValueError("AI 服务返回内容为空")
                        return
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
                raise RuntimeError(message)

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

    def _build_answer_prompts(self, question, history=None):
        if not self.document_content:
            raise ValueError("没有文档内容，请先上传文档进行分析。")

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
        return system_prompt, user_prompt

    def answer_question(self, question, history=None):
        system_prompt, user_prompt = self._build_answer_prompts(question, history=history)
        return self._call_llm(system_prompt, user_prompt)

    def stream_answer_question(self, question, history=None):
        system_prompt, user_prompt = self._build_answer_prompts(question, history=history)
        full_answer = []
        for chunk in self._stream_llm(system_prompt, user_prompt):
            full_answer.append(chunk)
            yield chunk
        return "".join(full_answer)

    def rewrite_search_query(self, query, context_text=""):
        clean_query = compact_text(query, limit=240)
        if not clean_query:
            raise ValueError("Please enter a search query.")

        context_excerpt = build_document_excerpt(context_text or "", limit=6000)
        system_prompt = (
            "You are an academic literature retrieval assistant. "
            "Rewrite user search requests into a precise English academic search query. "
            "If the user is clearly asking for a specific paper by nickname, alias, version name, or shorthand, resolve it to the canonical paper title instead of broadening it. "
            "Only broaden the query when the user's intent is genuinely ambiguous. "
            "Return JSON only."
        )
        user_prompt = f'''Rewrite the following paper search request into a concise English query for Semantic Scholar and arXiv.

Required JSON schema:
{{
  "original_query": "original user query",
  "rewritten_query": "better english academic query",
  "topics": ["topic 1", "topic 2", "topic 3"],
  "why": "brief reason in Chinese"
}}

Constraints:
- rewritten_query must be concise and in English
- preserve the user's research intent
- if the query likely refers to a specific known paper, prefer the canonical paper title
- examples: "yolov1的论文" should become the actual paper title "You Only Look Once: Unified, Real-Time Object Detection"
- do not broaden a specific-paper query into a vague family query like "YOLO papers"
- only expand or generalize when the user query is actually unclear or ambiguous
- topics should be short
- why should explain what was clarified or expanded in Chinese
- do not wrap JSON in markdown fences

User query:
{clean_query}

Optional context from current paper:
{context_excerpt}
'''
        raw_response = self._call_llm(system_prompt, user_prompt, model=self.search_rewrite_model)
        try:
            rewrite_meta = json.loads(extract_json_object(raw_response))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Search query rewriting failed: {exc}") from exc

        rewritten_query = compact_text(rewrite_meta.get("rewritten_query") or "", limit=240)
        if not rewritten_query:
            raise ValueError("Search query rewriting returned an empty rewritten query.")

        return {
            "original_query": clean_query,
            "rewritten_query": rewritten_query,
            "topics": rewrite_meta.get("topics") or [],
            "reason": str(rewrite_meta.get("why") or "").strip(),
            "model": self.search_rewrite_model,
        }

    def recommend_papers(self, content, limit=None):
        excerpt = build_document_excerpt(content, limit=12000)
        if not excerpt:
            raise ValueError("Current session does not contain document content.")

        resolved_limit = max(1, min(limit or RECOMMENDATION_RESULT_LIMIT, RECOMMENDATION_RESULT_LIMIT))
        rewrite_meta = self.rewrite_search_query(
            query="Find closely related follow-up papers for this paper.",
            context_text=excerpt,
        )
        search_result = search_papers(rewrite_meta.get("rewritten_query", ""), resolved_limit)
        return {
            "original_query": rewrite_meta.get("original_query", ""),
            "query": rewrite_meta.get("rewritten_query", ""),
            "topics": rewrite_meta.get("topics") or [],
            "reason": rewrite_meta.get("reason", ""),
            "rewrite_model": rewrite_meta.get("model", ""),
            "items": search_result.get("items", []),
            "errors": search_result.get("errors", []),
        }

    def _resolve_section_future(self, future, enabled=True):
        if not enabled:
            return build_section_result("disabled")
        try:
            content_value = future.result()
            if not content_value:
                return build_section_result("empty")
            return build_section_result("success", content=content_value)
        except Exception as exc:
            return build_section_result(
                "failed",
                error=str(exc),
                retryable=is_retryable_llm_error(str(exc)),
            )

    def _finalize_analysis_sections(self, content, sections):
        result = {"char_count": len(content)}
        result["sections"] = sections
        result["summary"] = sections["summary"]["content"]
        result["quotes"] = sections["quotes"]["content"]
        result["mindmap"] = sections["mindmap"]["content"]
        result["mermaid"] = sections["mermaid"]["content"]
        result["evaluation"] = sections["evaluation"]["content"]

        required_sections = [sections["summary"], sections["quotes"], sections["mindmap"]]
        if all(section["status"] == "failed" for section in required_sections):
            raise RuntimeError(required_sections[0]["error"] or "核心分析项全部失败")
        return result

    def analyze(self, file_path, generate_mermaid=True, generate_evaluation=True):
        """核心分析流程（已优化为并发执行）"""
        content = DocumentLoader.load(file_path)
        self.document_content = content

        sections = {}
        task_count = 3 + int(generate_mermaid) + int(generate_evaluation)
        worker_count = self._get_worker_count(task_count, self.analysis_workers)

        t_start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_summary = executor.submit(self.generate_summary, content)
            future_quotes = executor.submit(self.extract_quotes, content)
            future_mindmap = executor.submit(self.generate_mindmap, content)
            future_mermaid = executor.submit(self.generate_mermaid_mindmap, content) if generate_mermaid else None
            future_eval = executor.submit(self.generate_evaluation, content) if generate_evaluation else None

            sections["summary"] = self._resolve_section_future(future_summary)
            sections["quotes"] = self._resolve_section_future(future_quotes)
            sections["mindmap"] = self._resolve_section_future(future_mindmap)
            sections["mermaid"] = self._resolve_section_future(future_mermaid, enabled=generate_mermaid)
            sections["evaluation"] = self._resolve_section_future(future_eval, enabled=generate_evaluation)

        elapsed = time.time() - t_start
        result = self._finalize_analysis_sections(content, sections)
        result["elapsed_seconds"] = round(elapsed, 1)

        logger.info(f"Analysis completed in {elapsed:.1f}s for {os.path.basename(file_path)} ({len(content)} chars)")
        return result

    def analyze_stream(self, file_path, generate_mermaid=True, generate_evaluation=True):
        content = DocumentLoader.load(file_path)
        self.document_content = content

        sections = {
            "summary": build_section_result("pending"),
            "quotes": build_section_result("pending"),
            "mindmap": build_section_result("pending"),
            "mermaid": build_section_result("disabled") if not generate_mermaid else build_section_result("pending"),
            "evaluation": build_section_result("disabled") if not generate_evaluation else build_section_result("pending"),
        }
        task_count = 3 + int(generate_mermaid) + int(generate_evaluation)
        worker_count = self._get_worker_count(task_count, self.analysis_workers)
        t_start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(self.generate_summary, content): ("summary", True),
                executor.submit(self.extract_quotes, content): ("quotes", True),
                executor.submit(self.generate_mindmap, content): ("mindmap", True),
            }
            if generate_mermaid:
                future_map[executor.submit(self.generate_mermaid_mindmap, content)] = ("mermaid", True)
            if generate_evaluation:
                future_map[executor.submit(self.generate_evaluation, content)] = ("evaluation", True)

            for future in concurrent.futures.as_completed(future_map):
                section_name, enabled = future_map[future]
                section_result = self._resolve_section_future(future, enabled=enabled)
                sections[section_name] = section_result
                yield {
                    "type": "section",
                    "name": section_name,
                    "section": section_result,
                }

        elapsed = time.time() - t_start
        result = self._finalize_analysis_sections(content, sections)
        result["elapsed_seconds"] = round(elapsed, 1)
        logger.info(f"Streaming analysis completed in {elapsed:.1f}s for {os.path.basename(file_path)} ({len(content)} chars)")
        yield {
            "type": "done",
            "result": result,
        }

# ================= 路由控制 =================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/logo.ico")
async def logo_ico():
    return FileResponse("logo.ico", media_type="image/x-icon")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("logo.ico", media_type="image/x-icon")


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
        cleanup_expired_sessions()
        original_filename = build_safe_upload_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)

        await save_upload_file(file, file_path, MAX_CONTENT_LENGTH)

        generate_mermaid_bool = parse_bool_value(generate_mermaid, default=True)
        generate_evaluation_bool = parse_bool_value(generate_evaluation, default=True)
        result = analyze_saved_file(
            file_path=file_path,
            original_filename=original_filename,
            api_key=api_key,
            generate_mermaid_bool=generate_mermaid_bool,
            generate_evaluation_bool=generate_evaluation_bool,
            session_id=session_id,
        )
        return JSONResponse(content=result)

    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
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


@app.get("/api/download-paper")
async def download_paper(url: str = "", pdf_url: str = "", title: str = ""):
    try:
        response, file_name, content_type = stream_remote_paper(title=title, pdf_url=pdf_url, url=url)
        media_type = content_type or mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        quoted_name = urllib.parse.quote(file_name)
        headers = {
            "Content-Disposition": f"attachment; filename=\"{file_name}\"; filename*=UTF-8''{quoted_name}",
            "X-Content-Type-Options": "nosniff",
        }
        return StreamingResponse(iter_remote_file_chunks(response, MAX_CONTENT_LENGTH), media_type=media_type, headers=headers)
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except urllib.error.HTTPError as exc:
        logger.warning("Paper proxy download failed: %s", exc)
        if exc.code == 404:
            message = "The paper file could not be found at the remote source."
        elif exc.code == 403:
            message = "The remote source denied access to the paper file."
        elif exc.code == 429:
            message = "The remote source rate limited the paper download. Please retry in a moment."
        else:
            message = f"Remote paper download failed with HTTP {exc.code}."
        return JSONResponse(content={"error": message}, status_code=400)
    except urllib.error.URLError as exc:
        logger.warning("Paper proxy download failed: %s", exc)
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
            message = "SSL certificate verification failed while downloading the paper file."
        else:
            message = f"Paper download failed: {reason}"
        return JSONResponse(content={"error": message}, status_code=400)
    except Exception as exc:
        logger.exception("Paper proxy download failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/analyze/stream")
async def analyze_stream(
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

    cleanup_expired_sessions()
    resolved_api_key = resolve_api_key(api_key)
    if not resolved_api_key:
        return JSONResponse(content={"error": "API key is required. Provide api_key or set OPENAI_API_KEY."}, status_code=400)

    original_filename = build_safe_upload_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, original_filename)
    generate_mermaid_bool = parse_bool_value(generate_mermaid, default=True)
    generate_evaluation_bool = parse_bool_value(generate_evaluation, default=True)

    try:
        await save_upload_file(file, file_path, MAX_CONTENT_LENGTH)
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:
        logger.exception("Streaming analyze upload failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)
    finally:
        if file:
            await file.close()

    async def event_generator():
        try:
            whisperer = PaperWhisperer(resolved_api_key)
            safe_session_id = sanitize_identifier(session_id, "session")
            yield build_sse_event("start", {"session_id": safe_session_id, "source_filename": original_filename})

            final_result = None
            for event in whisperer.analyze_stream(file_path, generate_mermaid_bool, generate_evaluation_bool):
                if event["type"] == "section":
                    yield build_sse_event("section", {"name": event["name"], "section": event["section"]})
                elif event["type"] == "done":
                    final_result = event["result"]

            if final_result is None:
                raise RuntimeError("Analysis stream completed without a final result.")

            final_payload = finalize_analysis_result(
                result=final_result,
                whisperer=whisperer,
                original_filename=original_filename,
                generate_evaluation_bool=generate_evaluation_bool,
                session_id=session_id,
            )
            yield build_sse_event("done", final_payload)
        except Exception as exc:
            logger.exception("Streaming document analysis failed")
            yield build_sse_event("error", {"error": str(exc)})
        finally:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=build_sse_headers())


@app.post("/api/import-paper")
async def import_paper(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    data = data or {}
    if not isinstance(data, dict):
        return JSONResponse(content={"error": "JSON body must be an object."}, status_code=400)

    file_path = None
    try:
        cleanup_expired_sessions()
        title = str(data.get("title") or "").strip()
        url = str(data.get("url") or "").strip()
        pdf_url = str(data.get("pdf_url") or "").strip()
        if not pdf_url and not url:
            return JSONResponse(content={"error": "A downloadable paper URL is required."}, status_code=400)

        generate_mermaid_bool = parse_bool_value(data.get("generate_mermaid"), default=True)
        generate_evaluation_bool = parse_bool_value(data.get("generate_evaluation"), default=True)
        file_path, original_filename = download_remote_paper(title=title, pdf_url=pdf_url, url=url)
        result = analyze_saved_file(
            file_path=file_path,
            original_filename=original_filename,
            api_key=str(data.get("api_key") or ""),
            generate_mermaid_bool=generate_mermaid_bool,
            generate_evaluation_bool=generate_evaluation_bool,
            session_id=str(data.get("session_id") or ""),
        )
        return JSONResponse(content=result)
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except urllib.error.HTTPError as exc:
        logger.warning("Paper import download failed: %s", exc)
        if exc.code == 404:
            message = "The paper file could not be found at the remote source."
        elif exc.code == 403:
            message = "The remote source denied access to the paper file."
        elif exc.code == 429:
            message = "The remote source rate limited the paper download. Please retry in a moment."
        else:
            message = f"Remote paper download failed with HTTP {exc.code}."
        return JSONResponse(content={"error": message}, status_code=400)
    except urllib.error.URLError as exc:
        logger.warning("Paper import download failed: %s", exc)
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
            message = "SSL certificate verification failed while downloading the paper file."
        else:
            message = f"Paper download failed: {reason}"
        return JSONResponse(content={"error": message}, status_code=400)
    except Exception as exc:
        logger.exception("Paper import failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)
    finally:
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

    cleanup_expired_sessions()
    question = data.get("question", "").strip()
    raw_session_id = data.get("session_id", "")
    session_token = str(data.get("session_token") or "")

    if not question:
        return JSONResponse(content={"error": "Please enter a question."}, status_code=400)

    resolved_api_key = resolve_api_key(data.get("api_key", ""))
    if not resolved_api_key:
        return JSONResponse(
            content={"error": "API key is required. Provide api_key or set OPENAI_API_KEY."},
            status_code=400,
        )

    try:
        safe_session_id, session_payload = load_validated_session(raw_session_id, session_token, require_token=True)

        whisperer = PaperWhisperer(resolved_api_key)
        whisperer.document_content = get_session_document_content(session_payload)

        t_start = time.time()
        answer = whisperer.answer_question(question, history=session_payload.get("qa_history", []))
        elapsed = time.time() - t_start

        qa_history = session_payload.get("qa_history", [])
        qa_history.append({
            "question": question,
            "answer": answer,
            "timestamp": now_iso(),
        })
        session_payload["qa_history"] = qa_history
        session_payload["document_excerpt"] = build_document_excerpt(get_session_document_content(session_payload))
        write_session_payload(safe_session_id, session_payload)

        logger.info(f"Q&A completed in {elapsed:.1f}s for session {safe_session_id}")
        return JSONResponse(content={"answer": answer})
    except PermissionError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=403)
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as e:
        logger.exception("Question answering failed")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/ask/stream")
async def ask_question_stream(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    data = data or {}
    if not isinstance(data, dict):
        return JSONResponse(content={"error": "JSON body must be an object."}, status_code=400)

    cleanup_expired_sessions()
    question = str(data.get("question") or "").strip()
    raw_session_id = str(data.get("session_id") or "")
    session_token = str(data.get("session_token") or "")

    if not question:
        return JSONResponse(content={"error": "Please enter a question."}, status_code=400)

    resolved_api_key = resolve_api_key(data.get("api_key", ""))
    if not resolved_api_key:
        return JSONResponse(
            content={"error": "API key is required. Provide api_key or set OPENAI_API_KEY."},
            status_code=400,
        )

    try:
        safe_session_id, session_payload = load_validated_session(raw_session_id, session_token, require_token=True)
    except PermissionError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=403)
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)

    async def event_generator():
        try:
            whisperer = PaperWhisperer(resolved_api_key)
            whisperer.document_content = get_session_document_content(session_payload)
            yield build_sse_event("start", {"session_id": safe_session_id})

            full_answer_parts = []
            t_start = time.time()
            for chunk in whisperer.stream_answer_question(question, history=session_payload.get("qa_history", [])):
                full_answer_parts.append(chunk)
                yield build_sse_event("delta", {"text": chunk})

            answer = "".join(full_answer_parts)
            elapsed = time.time() - t_start

            qa_history = session_payload.get("qa_history", [])
            qa_history.append({
                "question": question,
                "answer": answer,
                "timestamp": now_iso(),
            })
            session_payload["qa_history"] = qa_history
            session_payload["document_excerpt"] = build_document_excerpt(get_session_document_content(session_payload))
            write_session_payload(safe_session_id, session_payload)

            logger.info(f"Streaming Q&A completed in {elapsed:.1f}s for session {safe_session_id}")
            yield build_sse_event("done", {"answer": answer})
        except Exception as exc:
            logger.exception("Streaming question answering failed")
            yield build_sse_event("error", {"error": str(exc)})

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=build_sse_headers())


@app.post("/api/search-papers")
async def search_papers_api(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    data = data or {}
    if not isinstance(data, dict):
        return JSONResponse(content={"error": "JSON body must be an object."}, status_code=400)

    cleanup_expired_sessions()
    query = str(data.get("query") or "").strip()
    limit = data.get("limit") or PAPER_SEARCH_RESULT_LIMIT
    raw_session_id = str(data.get("session_id") or "").strip()
    session_token = str(data.get("session_token") or "")
    context_text = str(data.get("context_text") or "")

    if not query:
        return JSONResponse(content={"error": "Please enter a search query."}, status_code=400)

    try:
        rewrite_meta = {
            "original_query": query,
            "rewritten_query": query,
            "topics": [],
            "reason": "Direct search without AI rewriting.",
            "model": "",
        }
        resolved_api_key = resolve_api_key(data.get("api_key", ""))
        session_payload = None
        safe_session_id = ""

        if raw_session_id:
            safe_session_id, session_payload = load_validated_session(raw_session_id, session_token, require_token=True)

        if PAPER_SEARCH_ENABLE_REWRITE:
            if not resolved_api_key:
                return JSONResponse(
                    content={"error": "API key is required for AI-assisted paper search. Provide api_key or set OPENAI_API_KEY."},
                    status_code=400,
                )
            whisperer = PaperWhisperer(resolved_api_key)
            rewrite_context = context_text
            if session_payload and not rewrite_context:
                rewrite_context = get_session_document_content(session_payload)
            rewrite_meta = whisperer.rewrite_search_query(query, context_text=rewrite_context)

        result = search_papers(rewrite_meta["rewritten_query"], limit)
        result["original_query"] = rewrite_meta.get("original_query", query)
        result["rewritten_query"] = rewrite_meta.get("rewritten_query", result["query"])
        result["topics"] = rewrite_meta.get("topics", [])
        result["reason"] = rewrite_meta.get("reason", "")
        result["rewrite_model"] = rewrite_meta.get("model", "")

        if session_payload:
            session_payload.setdefault("paper_search", {})
            session_payload["paper_search"]["last_query"] = result["rewritten_query"]
            session_payload["paper_search"]["last_results"] = result["items"]
            write_session_payload(safe_session_id, session_payload)
        return JSONResponse(content=result)
    except PermissionError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=403)
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:
        logger.exception("Paper search failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.post("/api/recommend-papers")
async def recommend_papers_api(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}

    data = data or {}
    if not isinstance(data, dict):
        return JSONResponse(content={"error": "JSON body must be an object."}, status_code=400)

    cleanup_expired_sessions()
    raw_session_id = str(data.get("session_id") or "").strip()
    session_token = str(data.get("session_token") or "")
    if not raw_session_id:
        return JSONResponse(content={"error": "session_id is required."}, status_code=400)

    limit = data.get("limit") or RECOMMENDATION_RESULT_LIMIT
    resolved_api_key = resolve_api_key(data.get("api_key", ""))
    if not resolved_api_key:
        return JSONResponse(
            content={"error": "API key is required. Provide api_key or set OPENAI_API_KEY."},
            status_code=400,
        )

    try:
        safe_session_id, session_payload = load_validated_session(raw_session_id, session_token, require_token=True)

        whisperer = PaperWhisperer(resolved_api_key)
        result = whisperer.recommend_papers(get_session_document_content(session_payload), limit=limit)

        session_payload.setdefault("paper_search", {})
        session_payload["paper_search"]["last_recommendation"] = {
            "original_query": result.get("original_query", ""),
            "query": result.get("query", ""),
            "topics": result.get("topics", []),
            "reason": result.get("reason", ""),
            "rewrite_model": result.get("rewrite_model", ""),
            "items": result.get("items", []),
            "errors": result.get("errors", []),
            "generated_at": now_iso(),
        }
        write_session_payload(safe_session_id, session_payload)
        return JSONResponse(content=result)
    except PermissionError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=403)
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:
        logger.exception("Paper recommendation failed")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


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
