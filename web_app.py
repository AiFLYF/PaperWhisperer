import os
import re
import time
import uuid
import threading
import concurrent.futures
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAI
from PyPDF2 import PdfReader
from env_loader import load_project_env


load_project_env()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max limit
app.config['CONTEXT_FOLDER'] = 'context'

# Ensure required folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['CONTEXT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)


ALLOWED_EXTENSIONS = {".txt", ".pdf"}


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


def sanitize_identifier(raw_value, prefix):
    candidate = secure_filename((raw_value or "").strip())
    return candidate or f"{prefix}_{uuid.uuid4().hex}"


def build_safe_upload_filename(filename):
    original_ext = os.path.splitext(filename)[1].lower()
    if original_ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")

    sanitized = secure_filename(filename)
    if not sanitized or not sanitized.lower().endswith(original_ext):
        return f"file_{uuid.uuid4().hex}{original_ext}"
    return sanitized


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
            
            # 尽量在句子结束处截断
            if end < text_length:
                last_period = chunk.rfind('。')
                last_newline = chunk.rfind('\n')
                split_pos = max(last_period, last_newline)
                
                if split_pos > self.chunk_size // 2:
                    chunk = chunk[:split_pos + 1]
                    end = start + split_pos + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - self.overlap
            
        return chunks


class DocumentLoader:
    """文档加载器，支持 TXT 和 PDF"""
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
    def load(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            return DocumentLoader.load_txt(file_path)
        elif ext == '.pdf':
            return DocumentLoader.load_pdf(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext} (请确保上传 txt 或 pdf 文件)")


class PaperWhisperer:
    """文献分析核心类"""
    def __init__(self, api_key):
        self.name = "PaperWhisperer"
        self.version = "0.5.2"
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
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.7,
                        max_tokens=4000,
                        timeout=self.request_timeout
                    )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(min(2 * (attempt + 1), 8))
                else:
                    print(f"LLM API Error: {str(e)}")
                    return f"生成失败，请重试 ({str(e)})"

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
{content[:15000]}  # 限制长度以防超 token

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
## 📳 论文评价

### 🎆 主要贡献
[评价内容]

### 📱 历史地位
[评价内容]

### ✅ 主要优点
- 优点1
- 优点2

### ♿️ 局限性
- 局限性1
- 局限性2

### 📕 值得学习的地方
- 学习点1
- 学习点2
"""
        return self._call_llm(system_prompt, user_prompt)
    
    def answer_question(self, question):
        if not self.document_content:
            return "没有文档内容，请先上传文档进行分析。"
        system_prompt = (
            "你是专业学术助手。请只基于给定文档内容回答问题，"
            "若文档中没有答案，请明确说明。"
        )
        user_prompt = (
            f"文档内容:\n{self.document_content[:12000]}\n\n"
            f"用户问题:\n{question}\n\n"
            "请给出简洁、准确的中文回答。"
        )
        return self._call_llm(system_prompt, user_prompt)
    
    def analyze(self, file_path, generate_mermaid=True, generate_evaluation=True):
        """核心分析流程（已优化为并发执行）"""
        content = DocumentLoader.load(file_path)
        self.document_content = content
        
        result = {'char_count': len(content)}

        task_count = 3 + int(generate_mermaid) + int(generate_evaluation)
        worker_count = self._get_worker_count(task_count, self.analysis_workers)

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
                
        return result

# ================= 路由控制 =================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    file_path = None

    if 'file' not in request.files:
        return jsonify({'error': 'Please upload a file.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Please select a file.'}), 400
    if not is_allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type. Please upload a .txt or .pdf file.'}), 400

    try:
        original_filename = build_safe_upload_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(file_path)

        generate_mermaid = parse_bool_value(request.form.get('generate_mermaid'), default=True)
        generate_evaluation = parse_bool_value(request.form.get('generate_evaluation'), default=True)

        api_key = resolve_api_key(request.form.get('api_key', ''))
        if not api_key:
            return jsonify({'error': 'API key is required. Provide api_key or set OPENAI_API_KEY.'}), 400

        whisperer = PaperWhisperer(api_key)
        result = whisperer.analyze(file_path, generate_mermaid, generate_evaluation)
        
        raw_session_id = request.form.get('session_id', '')
        safe_session_id = sanitize_identifier(raw_session_id, "session")
        
        context_file = os.path.join(app.config['CONTEXT_FOLDER'], f"{safe_session_id}.txt")
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(whisperer.document_content)
        
        base_name = os.path.splitext(original_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_analysis_{timestamp}.md")
        
        md_content = f"""# 📫 PaperWhisperer 分析报告

> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 源文件: {original_filename}

---

## 📒 AI 摘要

{result.get('summary', '')}

---

## 📕 引用片段

{result.get('quotes', '')}

---

## 🗥 思维导图

{result.get('mindmap', '')}

---

"""

        if generate_evaluation:
            md_content += f"## 📳 论文评价\n\n{result.get('evaluation', '')}\n\n---\n"
            
        md_content += f"## 📳 元信息\n\n- 版本: {whisperer.version}\n- 字符数: {result['char_count']}\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        result['output_file'] = output_file
        result['session_id'] = safe_session_id
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.exception("Document analysis failed")
        return jsonify({'error': str(e)}), 500
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'JSON body must be an object.'}), 400

    question = data.get('question', '').strip()
    raw_session_id = data.get('session_id', '')
    
    if not question:
        return jsonify({'error': 'Please enter a question.'}), 400

    safe_session_id = sanitize_identifier(raw_session_id, "session")
    
    api_key = resolve_api_key(data.get('api_key', ''))
    if not api_key:
        return jsonify({'error': 'API key is required. Provide api_key or set OPENAI_API_KEY.'}), 400
    
    try:
        context_file = os.path.join(app.config['CONTEXT_FOLDER'], f"{safe_session_id}.txt")
        if not os.path.exists(context_file):
            return jsonify({'error': 'Session expired or context not found. Please upload and analyze the file again.'}), 400
        
        with open(context_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        whisperer = PaperWhisperer(api_key)
        whisperer.document_content = content
        
        answer = whisperer.answer_question(question)
        
        return jsonify({'answer': answer})
    
    except Exception as e:
        app.logger.exception("Question answering failed")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=parse_int_env("FLASK_PORT", default=5000, min_value=1, max_value=65535),
        debug=parse_bool_env("FLASK_DEBUG", default=False),
    )