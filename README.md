# 📚 PaperWhisperer

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![License](https://img.shields.io/badge/License-MIT-lightgrey) ![AI](https://img.shields.io/badge/AI--Powered-OpenAI-blueviolet) ![Version](https://img.shields.io/badge/version-0.9.0-green)

一个面向论文/文献阅读场景的 AI 助手。
上传 `.txt`、`.pdf`、`.docx` 或 `.pptx` 后，可自动生成结构化分析结果、可视化结构图与批判性评价，并支持基于当前文档会话继续多轮追问。

项目地址：<https://github.com/AiFLYF/PaperWhisperer>

## ✨ 项目能做什么
- 自动生成文档摘要，快速抓住核心观点
- 提取适合引用或做笔记的关键片段
- 生成文本结构图，帮助理解论文层级与逻辑
- 可选生成 Mermaid 可视化图，便于展示与分享
- 可选输出批判性评价，辅助形成自己的判断
- 支持基于关键词/任务/方法的论文搜索（Semantic Scholar + arXiv 聚合）
- 支持在 `Paper Search` 中通过后端代理稳定下载结果论文文件名，或一键加入当前分析流程
- 支持基于当前论文内容自动生成检索主题并推荐延伸阅读
- 支持“上传后追问”问答，并保留当前文档的最近多轮会话上下文
- 支持导出完整会话结果（Markdown + Mermaid SVG）
- 自动保存基础 Markdown 分析报告到 `output/`

## 🎯 适用人群
- 需要快速读论文的学生
- 需要整理阅读笔记的研究者
- 需要将文档分析流程产品化的开发者

## 🧱 技术栈
- Python 3.10+
- FastAPI
- OpenAI Python SDK（OpenAI 兼容接口）
- PyPDF2
- python-docx
- python-pptx
- HTML / CSS / JavaScript
- KaTeX + Mermaid + marked

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
先复制一份环境变量模板：

```powershell
Copy-Item .env.example .env
```

然后按需填写 `.env`。应用启动时会自动加载项目根目录下的 `.env`，通常不需要再手动执行 `export` / `$env:`。

可参考 `.env.example`：

```bash
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
PAPER_SEARCH_ENABLE_REWRITE=true
PAPER_SEARCH_REWRITE_MODEL=gpt-4o-mini
SEMANTIC_SCHOLAR_API_KEY=
SEMANTIC_SCHOLAR_MAX_RETRIES=3
OPENAI_MAX_CONCURRENCY=5
OPENAI_REQUEST_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=3
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=5000
FASTAPI_RELOAD=false
# 兼容旧配置（可选）
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=false
```

PowerShell 示例：
```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4o-mini"
$env:PAPER_SEARCH_ENABLE_REWRITE="true"
$env:PAPER_SEARCH_REWRITE_MODEL="gpt-4o-mini"
$env:SEMANTIC_SCHOLAR_API_KEY=""
$env:SEMANTIC_SCHOLAR_MAX_RETRIES="3"
$env:OPENAI_MAX_CONCURRENCY="5"
$env:OPENAI_REQUEST_TIMEOUT_SECONDS="60"
$env:OPENAI_MAX_RETRIES="3"
$env:FASTAPI_HOST="0.0.0.0"
$env:FASTAPI_PORT="5000"
$env:FASTAPI_RELOAD="false"
# 兼容旧配置（可选）
$env:FLASK_HOST="0.0.0.0"
$env:FLASK_PORT="5000"
$env:FLASK_DEBUG="false"
```

说明：
- 项目会优先读取系统环境变量，其次再读取 `.env`
- `OPENAI_MAX_CONCURRENCY` 控制全局大模型并发上限
- `OPENAI_REQUEST_TIMEOUT_SECONDS` 控制单次请求超时
- `OPENAI_MAX_RETRIES` 控制失败重试次数
- `PAPER_SEARCH_ENABLE_REWRITE` 控制手动搜索是否先经过 AI 理解与改写
- `PAPER_SEARCH_REWRITE_MODEL` 用于指定搜索改写专用模型；未设置时回退到 `OPENAI_MODEL`
- `SEMANTIC_SCHOLAR_API_KEY` 可选，用于提升 Semantic Scholar 搜索限额
- `SEMANTIC_SCHOLAR_MAX_RETRIES` 控制遇到限流时的自动重试次数
- `FASTAPI_RELOAD` 建议仅在本地开发时开启
- 仍兼容 `FLASK_HOST` / `FLASK_PORT` / `FLASK_DEBUG` 作为回退配置

### 3. 启动 Web 应用
```bash
python web_app.py
```
浏览器访问：`http://localhost:5000`

可选：直接使用 uvicorn 启动
```bash
uvicorn web_app:app --host 0.0.0.0 --port 5000
```

### 4. （可选）运行命令行 Demo
```bash
python paper_whisperer_demo.py <你的文件.txt|pdf|docx|pptx>
```

## 🖥️ Web 使用流程
1. 打开网页并填写 API Key（或直接使用环境变量）
2. 上传 `.txt` / `.pdf` / `.docx` / `.pptx` 文档
3. 点击 `Analyze Document`
4. 可在 `Paper Search` 区域直接检索相关论文
5. 查看 `Overview`、`Key Citations`、`Text Structure`、`Evaluation`
6. 在 `Auto Recommendations` 区域基于当前论文生成延伸阅读结果
7. 在 `Paper Search` 结果中可直接点击 `Download` 下载原文，或点击 `Add` 自动导入并替换当前分析文档
8. 在 `Ask Questions` 区域继续追问（会自动带入当前文档会话的最近多轮上下文）
9. 点击 `Export Session` 导出当前分析、问答记录与 Mermaid 资源

## 🔌 API 接口说明

### `POST /api/analyze`
上传并分析文档（非流式版本）。

表单参数：
- `file`（必填）：`.txt`、`.pdf`、`.docx` 或 `.pptx`
- `api_key`（可选）：本次请求临时 API Key
- `generate_mermaid`（可选）：`true/false`
- `generate_evaluation`（可选）：`true/false`
- `session_id`（可选）：用于后续问答

说明：若不传 `api_key`，服务端会读取 `OPENAI_API_KEY`。

返回结构要点：
- 保留顶层 `summary` / `quotes` / `mindmap` / `mermaid` / `evaluation` 字段用于兼容旧前端
- 新增 `sections`，按 section 返回 `status`、`content`、`error`、`retryable`
- 新增 `session_token`，用于后续问答、推荐和带上下文的搜索鉴权

### `POST /api/analyze/stream`（SSE 流式）
上传并流式分析文档，实时返回各分析模块结果。

表单参数：与 `/api/analyze` 相同。

SSE 事件类型：
- `event: start` - 分析开始，返回 `session_id` 和 `source_filename`
- `event: section` - 单个分析模块完成，返回 `name`（模块名）和 `section`（结果）
- `event: done` - 全部分析完成，返回完整结果
- `event: error` - 分析失败，返回错误信息

前端处理示例：
```javascript
const response = await fetch('/api/analyze/stream', { method: 'POST', body: formData });
const reader = response.body.getReader();
// 解析 SSE 事件流...
```

优势：
- 实时反馈分析进度，用户体验更好
- 无需等待全部模块完成即可看到部分结果
- 支持按模块显示加载状态

### `POST /api/ask`
基于已保存的文档上下文进行追问（非流式版本）。

说明：服务端会保存 `context/<session_id>.json`，其中包含该次分析对应文档的正文、分析结果摘要与问答历史。追问时只会使用当前 `session_id` 这一个文档会话里的 Ask Questions 历史，不会跨文档、不会做全局混用；同时会优先携带文档节选，并附加最近若干轮问答历史，而不是每轮重复塞入整篇文档。

请求体示例：
```json
{
  "question": "这篇文献最重要的贡献是什么？",
  "session_id": "session_123",
  "session_token": "returned-by-analyze",
  "api_key": "optional"
}
```

### `POST /api/ask/stream`（SSE 流式）
基于已保存的文档上下文进行流式追问，实时返回答案。

请求体参数：与 `/api/ask` 相同。

SSE 事件类型：
- `event: start` - 问答开始
- `event: delta` - 增量文本片段，返回 `text` 字段
- `event: done` - 问答完成，返回完整 `answer`
- `event: error` - 问答失败，返回错误信息

优势：
- 打字机效果，实时显示答案
- 用户无需等待完整答案生成
- 更接近 ChatGPT 的交互体验

### `POST /api/search-papers`
按关键词聚合检索 Semantic Scholar 与 arXiv 论文结果。默认会先经过 AI 理解与检索词改写，再执行外部论文搜索。

请求体示例：
```json
{
  "query": "large language model reasoning",
  "limit": 8,
  "session_id": "optional",
  "session_token": "required-when-session_id-is-provided",
  "api_key": "optional"
}
```

返回结构示例：
```json
{
  "original_query": "large language model reasoning",
  "rewritten_query": "large language model reasoning chain-of-thought benchmark",
  "query": "large language model reasoning chain-of-thought benchmark",
  "topics": ["reasoning", "llm", "benchmark"],
  "reason": "将用户输入补全为更适合学术检索的英文短语。",
  "rewrite_model": "gpt-4o-mini",
  "items": [
    {
      "source": "Semantic Scholar",
      "paper_id": "...",
      "title": "...",
      "abstract": "...",
      "authors": ["..."],
      "year": "2024",
      "venue": "NeurIPS",
      "url": "https://...",
      "pdf_url": "https://..."
    }
  ],
  "errors": []
}
```

### `GET /api/download-paper`
通过后端代理下载 `Paper Search` 结果对应的论文文件，并尽量返回稳定文件名。

查询参数示例：
```text
/api/download-paper?title=Attention%20Is%20All%20You%20Need&pdf_url=https://arxiv.org/pdf/1706.03762.pdf
```

说明：
- 服务端优先使用 `pdf_url`，其次仅在 `url` 本身就是文件直链时才尝试下载
- 会复用与导入相同的远程链接校验逻辑，拒绝本地地址、HTML 落地页和不支持的文件类型
- 下载代理现在采用流式响应，避免服务端整块读入远程文件
- 返回 `Content-Disposition: attachment`，便于浏览器用稳定文件名保存

### `POST /api/import-paper`
把 `Paper Search` 中的搜索结果直接下载为临时文件，并复用现有分析链路生成当前会话结果。

请求体示例：
```json
{
  "title": "Attention Is All You Need",
  "url": "https://www.semanticscholar.org/paper/...",
  "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
  "session_id": "session_123",
  "api_key": "optional",
  "generate_mermaid": true,
  "generate_evaluation": true
}
```

说明：
- 服务端优先使用 `pdf_url` 下载原文；若没有 `pdf_url`，仅在 `url` 本身就是文件直链时才尝试导入
- 导入成功后返回结构与 `/api/analyze` 基本一致，并覆盖当前分析会话上下文
- 当前实现优先面向可直接获取的 PDF 原文

### `POST /api/recommend-papers`
基于当前分析 session 的论文内容生成检索主题，并返回一批延伸阅读结果。

请求体示例：
```json
{
  "session_id": "session_123",
  "session_token": "returned-by-analyze",
  "api_key": "optional",
  "limit": 6
}
```

返回结构示例：
```json
{
  "original_query": "Find closely related follow-up papers for this paper.",
  "query": "retrieval augmented generation benchmark",
  "topics": ["retrieval", "rag evaluation", "benchmarking"],
  "reason": "这些结果与当前论文的任务、方法或评估设置最相关。",
  "rewrite_model": "gpt-4o-mini",
  "items": [...],
  "errors": []
}
```

技术说明：
- 手动搜索在默认配置下会先由大模型理解用户意图，并改写为更适合学术搜索源的英文检索短语，再复用统一搜索链路。
- 对于明显指向单篇论文的别名或简称（如 `yolov1的论文`），改写逻辑会优先还原为正式论文标题，而不是泛化成整类论文。
- 自动推荐同样走“先理解/改写，再搜索”的两阶段流程，并优先结合当前论文内容生成主题。
- `PAPER_SEARCH_REWRITE_MODEL` 可单独指定搜索改写所用模型，便于与分析主模型分离配置。
- 搜索层对 Semantic Scholar 429 做自动重试；arXiv 请求显式使用本地 CA 证书链，降低 SSL 证书校验失败概率。

## 🗂️ 项目结构
```text
.
├── web_app.py                # FastAPI 应用与 API 路由
├── paper_whisperer_demo.py   # 命令行 Demo
├── templates/
│   └── index.html            # Web 界面
├── requirements.txt
├── .env.example
├── uploads/                  # 运行时上传目录（已忽略）
├── context/                  # 会话缓存目录（JSON，会保存文档、分析结果与问答历史，已忽略）
└── output/                   # 分析结果目录（已忽略）
```

## 🔐 安全与隐私说明
- 代码中不硬编码真实 API Key
- `uploads/`、`context/`、`output/` 默认不提交到 Git
- 临时上传文件在分析结束后会自动清理
- Web 会话上下文保存在 `context/*.json`，便于多轮追问、推荐结果缓存与完整结果导出
- session 现在带有 `expires_at` 过期时间，并通过 `session_token` 保护后续会话请求
- 默认会尽量减少落盘的全文内容；若需要持久化完整文档，可通过 `SESSION_PERSIST_FULL_DOCUMENT=true` 开启
- 支持"请求级 key"与"环境变量 key"两种模式
- 外部论文搜索依赖公开学术接口，单个来源失败时会降级返回其他来源结果
- 从搜索结果导入论文时，只允许公开 `http/https` 链接，并在下载完成后清理临时文件

### 🛡️ 安全防护机制
项目实现了多层安全防护，确保用户数据和系统安全：

**文件上传安全**
- `secure_filename()` - 净化文件名，移除特殊字符，防止路径穿越攻击
- `build_safe_upload_filename()` - 强制校验文件扩展名，拒绝危险文件类型
- `save_upload_file()` - 分块保存上传文件，服务端强制执行 16MB 大小限制
- 上传完成后自动清理临时文件

**URL 安全校验**
- `is_public_http_url()` - 仅允许公开 `http/https` 协议
- 拒绝 `localhost`、私有 IP 地址（如 `192.168.x.x`、`10.x.x.x`）、环回地址
- 防止 SSRF（服务端请求伪造）攻击

**会话安全**
- `session_token` 使用 `secrets.token_urlsafe(24)` 生成高强度随机令牌
- Token 哈希存储（SHA-256），不保存明文
- `validate_session_token()` 使用 `secrets.compare_digest()` 安全比较，防止时序攻击
- 会话自动过期清理（默认 24 小时）

**前端安全**
- `escapeHtml()` - 转义 HTML 特殊字符，防止 XSS
- `sanitizeUrl()` - URL 协议白名单校验
- `sanitizeGeneratedHtml()` - 移除 `on*` 事件属性，净化动态内容
- 所有外部链接添加 `rel="noopener noreferrer"`

**流式响应安全**
- SSE 响应设置 `X-Accel-Buffering: no`，禁用代理缓冲
- 远程文件下载采用流式传输，避免大文件占用内存

## ⚠️ 当前已知限制
- 代码中部分中文提示词仍有历史编码痕迹（不影响主流程）
- 分析质量与所选模型、提示词质量相关
- PDF 提取效果取决于原文档文本层质量
- DOCX/PPTX 主要提取文本层内容，复杂排版、图表和图片中的文字可能提取不完整
- 搜索结果中的 `Add` 优先适配可直接下载的 PDF；如果结果只有论文落地页而没有文件直链，仍需手动下载后再上传

## 🛠️ 常见 API 配置问题
- 如果提示“AI 服务返回了网页内容而不是模型结果”，通常说明 API Key 缺失/无效，或者 `OPENAI_BASE_URL` 指到了网页地址而不是 API 地址
- 如果出现 `401`，通常是 API Key 错误、为空或已过期
- 如果出现 `403`，通常是当前 Key 无权访问目标模型或供应商拒绝访问
- 如果出现 `404`，请优先检查 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`
- 如果出现 `429`，说明请求过快、并发过高或额度受限
- 如果核心分析项全部失败，`/api/analyze` 会直接返回错误，而不会再伪装成成功结果

## ✅ 项目亮点
- 面向论文/文献阅读，不是通用聊天壳，而是围绕"读、提炼、追问、导出"设计
- 同时支持 `PDF`、`TXT`、`DOCX`、`PPTX` 四类常见文档格式
- Web 端支持多轮追问，且上下文严格绑定当前文档会话
- 支持 Mermaid 可视化结构图与 SVG 导出
- 导出结果不仅包含分析内容，也可带上问答历史，便于沉淀阅读笔记
- **SSE 流式响应**：分析和问答支持实时流式输出，打字机效果更流畅
- **多层安全防护**：文件上传安全、URL 校验、会话 Token 保护、前端 XSS 防护

## 📝 更新日志

### v0.9.0
- **SSE 流式 API**：新增 `/api/analyze/stream` 和 `/api/ask/stream` 流式端点
  - 分析结果实时推送，支持 `start`/`section`/`done`/`error` 事件
  - 问答答案流式返回，支持 `start`/`delta`/`done`/`error` 事件
  - 前端实现打字机效果，用户体验更流畅
- **安全防护增强**：
  - 文件上传：`secure_filename()` 防止路径穿越，分块上传 + 大小限制
  - URL 校验：`is_public_http_url()` 拒绝私有 IP 和本地地址，防止 SSRF
  - 会话安全：Token 使用 `secrets.compare_digest()` 安全比较，防止时序攻击
  - 前端安全：XSS 防护、URL 白名单、事件属性净化
- **前端优化**：
  - CSS 样式内联到 `index.html`，移除外部 CSS 文件依赖
  - 简化项目结构，减少文件数量

### v0.8.0
- **会话安全**：分析结果新增 `session_token`，追问、推荐和带上下文搜索需要 token 才能访问当前 session
- **会话生命周期**：`context/<session>.json` 新增 `created_at` / `updated_at` / `expires_at`，并自动清理过期 session
- **上传限额**：`/api/analyze` 改为分块保存上传文件，并在服务端强制执行大小限制
- **分析契约**：新增 `sections` 结构化返回，按 section 提供状态、内容、错误与是否可重试
- **下载代理**：`/api/download-paper` 改为流式响应，减少服务端内存占用

### v0.7.0
- **论文搜索**：新增 `POST /api/search-papers`，聚合 Semantic Scholar 与 arXiv 结果并统一数据结构
- **自动推荐**：新增 `POST /api/recommend-papers`，基于当前论文内容生成检索主题并返回延伸阅读
- **研究工作台 UI**：Web 端新增 `Paper Search` 与 `Auto Recommendations` 面板
- **会话扩展**：`context/<session>.json` 新增论文搜索与推荐结果缓存字段

### v0.6.1
- **多轮问答**：Web 端会话从纯文本升级为 `context/<session>.json`，追问会携带最近多轮问答历史
- **上下文预算调整**：问答默认保留约 `12000` 字符文档窗口 + 约 `8000` 字符历史预算，单轮历史上限约 `2400` 字符
- **完整导出**：新增“下载完整结果”，可导出当前分析、问答记录、Mermaid 源码，并在可用时额外导出 SVG
- **导出版式优化**：导出的 Markdown 增加会话信息表、清晰章节与 Q/A 编号结构

### v0.6.0
- **效率优化**：CLI Demo 分析任务（摘要/引用/导图）改为并发执行，速度提升 2-3x
- **内容提取增强**：DOCX 新增表格提取，PPTX 新增备注页提取
- **文本清洗**：新增 `clean_extracted_text()` 自动清理多余空行和冗余空白
- **分块优化**：TextChunker 新增英文句号边界识别，中英文文档分块更精准
- **问答能力**：追问上下文窗口从 12000 字符扩展到 15000 字符（当时仅文档窗口）
- **请求计时**：Web 端分析和问答请求增加耗时日志，API 返回 `elapsed_seconds`
- **版本升级**：统一版本号至 `0.6.0`

### v0.5.2
- 初始支持 `.txt` / `.pdf` / `.docx` / `.pptx` 四种格式
- Web 端并发分析、Mermaid 可视化、批判性评价
- 页面 UI：GitHub 链接、AI 标签、深浅色主题

## 📄 License
MIT
