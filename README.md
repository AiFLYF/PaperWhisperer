# PaperWhisperer

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-Research%20Workspace-009688) ![License](https://img.shields.io/badge/License-MIT-lightgrey) ![AI](https://img.shields.io/badge/AI--Powered-OpenAI%20Compatible-blueviolet)

PaperWhisperer 是一个面向论文和技术文档阅读的 AI Research Workspace。它可以上传文档、抽取结构、生成可视化图谱、输出批判性评价和深度阅读简报，并在同一个会话中继续搜索论文、保存阅读队列、追问细节和导出完整研究笔记。

项目地址：<https://github.com/AiFLYF/PaperWhisperer>

## 核心能力

- 支持 `.pdf`、`.txt`、`.docx`、`.pptx` 四类文档。
- 生成 `Overview`、`Key Citations`、`Text Structure`、`Visual Map`、`Evaluation`、`Deep Research Brief`。
- 支持 SSE 流式分析，先完成的分析 section 会优先显示，并可取消正在进行的长耗时请求。
- 支持 Evidence / Explain / Critique / Reproduce 四种追问模式。
- 支持 Semantic Scholar + arXiv 论文搜索。
- 支持基于当前论文自动推荐延伸阅读。
- 支持把搜索或推荐结果保存到带编号卡片和结构化元信息的 Reading Queue。
- 支持从公开 PDF 直链导入论文并替换当前分析会话。
- 支持导出 Markdown 会话报告和 Mermaid SVG，导出预览会用分组卡片展示分析、问答、论文线索和可视化资产；Mermaid 渲染器会在需要图谱时按需加载。
- 内置上传校验、远程 URL 校验、会话 token、XSS 防护和 JSON 请求校验。
- 前端提供拖拽上传、分析进度、结果状态卡、请求取消按钮、顶部快速导航、跳转链接、快捷键、返回顶部、键盘焦点流转、移动端触控优化和资源加载预热。

## 使用场景

- 快速读懂论文的研究生、开发者和研究人员。
- 需要沉淀文献综述、组会笔记、复现计划的人。
- 想把论文搜索、阅读、追问和导出串成一个轻量工作台的人。

## 技术栈

- Python 3.10+
- FastAPI + Uvicorn
- OpenAI Python SDK（兼容 OpenAI API 协议的服务）
- pypdf / python-docx / python-pptx
- Vanilla HTML / CSS / JavaScript
- KaTeX / marked / svg-pan-zoom

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制环境模板：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

最少需要配置：

```bash
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

应用启动时会自动加载项目根目录的 `.env`。

### 3. 启动 Web 应用

```bash
python web_app.py
```

浏览器访问：

```text
http://localhost:5000
```

也可以直接用 Uvicorn：

```bash
uvicorn web_app:app --host 0.0.0.0 --port 5000
```

### 4. 运行测试

```bash
python -m py_compile web_app.py
node --check templates/static/js/app.js
python -m pytest tests/test_security_regressions.py
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 空 | OpenAI-compatible API key。 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL。 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 主分析与问答模型。 |
| `OPENAI_MAX_CONCURRENCY` | `5` | 全局 LLM 并发上限。 |
| `OPENAI_REQUEST_TIMEOUT_SECONDS` | `60` | 单次 LLM 请求超时。 |
| `OPENAI_MAX_RETRIES` | `3` | LLM 失败重试次数。 |
| `PAPERWHISPERER_VERSION` | `0.9.0` | 可选部署版本标识，会出现在 `/api/health`。 |
| `PAPER_SEARCH_ENABLE_REWRITE` | `true` | 搜索前是否用 AI 改写检索词。 |
| `PAPER_SEARCH_REWRITE_MODEL` | `OPENAI_MODEL` | 搜索改写模型。 |
| `PAPER_SEARCH_RESULT_LIMIT` | `8` | Paper Search 默认返回数量。 |
| `RECOMMENDATION_RESULT_LIMIT` | `6` | Auto Recommendations 默认返回数量。 |
| `SEMANTIC_SCHOLAR_API_KEY` | 空 | 可选，用于提升 Semantic Scholar 限额。 |
| `SEMANTIC_SCHOLAR_TIMEOUT_SECONDS` | `20` | Semantic Scholar / arXiv 检索请求超时。 |
| `SEMANTIC_SCHOLAR_MAX_RETRIES` | `3` | Semantic Scholar 限流重试次数。 |
| `REMOTE_IMPORT_TIMEOUT_SECONDS` | `30` | 公开论文导入下载超时。 |
| `SESSION_TTL_SECONDS` | `86400` | session JSON 默认有效期。 |
| `SESSION_CLEANUP_INTERVAL_SECONDS` | `600` | 过期 session 清理间隔。 |
| `SESSION_PERSIST_FULL_DOCUMENT` | `false` | 是否把完整文档内容持久化到 session JSON。 |
| `FASTAPI_HOST` | `0.0.0.0` | Web 服务监听地址。 |
| `FASTAPI_PORT` | `5000` | Web 服务端口。 |
| `FASTAPI_RELOAD` | `false` | 是否启用 Uvicorn reload。 |
| `FLASK_HOST` / `FLASK_PORT` / `FLASK_DEBUG` | 可选 | 迁移期兼容旧配置，作为 FastAPI 配置 fallback。 |

## Web 使用流程

1. 打开页面，输入 API Key 或使用服务端 `.env`。
2. 上传 PDF / TXT / DOCX / PPTX。
3. 按需开启结构图、批判性评价、深度阅读简报。
4. 点击 `Analyze Document`，等待流式 section 完成；页面会显示上传、分析、渲染、就绪进度，并在完成后聚焦到分析工作区。长耗时分析、检索、推荐和追问都可以用页面上的 Cancel 按钮中止；也可用 `Alt+U`、`Alt+S`、`Alt+Q` 快速跳到上传、论文搜索和追问输入。
5. 在 `Paper Search` 搜索相关论文，也可以点击示例查询快速开始；空搜索、加载中、无结果和部分失败都会显示可操作提示。
6. 对结果点击 `Save` 加入 Reading Queue，点击 `Add` 可导入公开 PDF 原文继续分析。
7. 在 `Auto Recommendations` 基于当前论文生成延伸阅读。
8. 在 Ask Questions 中选择追问模式并继续提问，可点击示例问题快速填充贡献、证据、局限和复现类追问，模式说明会提示答案结构差异。
9. 点击 `Export Session` 导出分析、阅读队列、搜索轨迹、问答历史和 Mermaid 资源；复制和导出按钮会显示成功或缺失内容反馈。

## API 概览

### `POST /api/analyze`

上传并分析文档，返回完整 JSON。

表单参数：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `file` | 是 | `.txt`、`.pdf`、`.docx`、`.pptx`。 |
| `api_key` | 否 | 请求级 API key，未传时读取环境变量。 |
| `generate_mermaid` | 否 | 是否生成 Mermaid 可视化图，默认 `true`。 |
| `generate_evaluation` | 否 | 是否生成批判性评价，默认 `true`。 |
| `generate_research_brief` | 否 | 是否生成深度阅读简报，默认 `true`。 |
| `session_id` | 否 | 可选自定义 session id。 |

返回要点：

- `session_id`：当前分析会话。
- `session_token`：后续 session-bound 请求必须携带。
- `sections`：按 section 返回 `status`、`content`、`error`、`retryable`。
- 顶层兼容字段：`summary`、`quotes`、`mindmap`、`mermaid`、`evaluation`、`research_brief`。
- `suggested_questions` / `next_actions` / `analysis_status`：用于智能 follow-up。

### `POST /api/analyze/stream`

上传并用 SSE 流式分析文档。表单参数与 `/api/analyze` 相同。

事件示例：

```text
event: start
data: {"session_id":"...","source_filename":"paper.pdf"}

event: section
data: {"name":"summary","section":{"status":"success","content":"...","error":"","retryable":false}}

event: done
data: {"session_id":"...","session_token":"...","sections":{...}}
```

### `POST /api/ask`

基于当前 session 文档上下文进行非流式追问。

```json
{
  "question": "这篇论文的核心贡献是什么？",
  "answer_mode": "evidence",
  "session_id": "session_123",
  "session_token": "returned-by-analyze",
  "api_key": "optional"
}
```

`answer_mode` 可选值：

- `evidence`：先给结论，再列文档证据和不确定点。
- `explain`：教学式解释概念、方法和公式。
- `critique`：从审稿视角分析贡献、假设、局限和威胁效度。
- `reproduce`：输出复现步骤、变量、依赖和风险清单。

### `POST /api/ask/stream`

基于当前 session 文档上下文进行流式追问。请求体与 `/api/ask` 相同。

事件示例：

```text
event: start
data: {"session_id":"session_123"}

event: delta
data: {"text":"增量答案片段"}

event: done
data: {"answer":"完整答案"}
```

### `POST /api/search-papers`

聚合检索 Semantic Scholar 和 arXiv。若配置允许，会先用 AI 改写检索词。

```json
{
  "query": "large language model reasoning",
  "limit": 8,
  "session_id": "optional",
  "session_token": "required-when-session_id-is-provided",
  "api_key": "optional"
}
```

### `GET /api/download-paper`

通过后端代理下载搜索结果中的公开论文文件。

```text
/api/download-paper?title=Attention%20Is%20All%20You%20Need&pdf_url=https://arxiv.org/pdf/1706.03762.pdf
```

远程下载会拒绝私有地址、本地地址、HTML 落地页、超大文件和不支持的文件类型。

### `POST /api/import-paper`

从搜索结果中的公开 PDF 直链下载论文并复用分析链路。

```json
{
  "title": "Attention Is All You Need",
  "url": "https://www.semanticscholar.org/paper/...",
  "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
  "session_id": "session_123",
  "api_key": "optional",
  "generate_mermaid": true,
  "generate_evaluation": true,
  "generate_research_brief": true
}
```

### `POST /api/reading-queue`

保存当前 session 的 Reading Queue。

```json
{
  "session_id": "session_123",
  "session_token": "returned-by-analyze",
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
  ]
}
```

服务端会归一化、去重、限制数量并写回 session JSON。

### `POST /api/recommend-papers`

基于当前 session 文档内容生成延伸阅读检索主题并返回推荐论文。

```json
{
  "session_id": "session_123",
  "session_token": "returned-by-analyze",
  "api_key": "optional",
  "limit": 6
}
```

### `GET /api/health`

返回轻量运行状态，便于本地检查、容器探活或部署平台健康检查。

```json
{
  "status": "ok",
  "app": "PaperWhisperer",
  "version": "0.9.0",
  "timestamp": "2026-05-25T00:00:00Z",
  "uptime_seconds": 12.345,
  "folders": {
    "uploads": {"exists": true, "writable": true},
    "output": {"exists": true, "writable": true},
    "context": {"exists": true, "writable": true}
  }
}
```

该接口响应包含 `Cache-Control: no-store`，探活和部署检查会读取实时运行状态。`uptime_seconds` 可用于确认进程是否刚重启。

## Session 生命周期

- 分析成功后服务端会写入 `context/<session_id>.json`。
- `session_token` 只在分析响应中返回，服务端只保存 token hash。
- `/api/ask`、`/api/ask/stream`、`/api/recommend-papers`、`/api/reading-queue` 必须携带有效 `session_token`。
- `/api/search-papers` 如果携带 `session_id`，也必须携带有效 token。
- session 默认带 `expires_at`，过期或损坏的 JSON 会被自动清理。
- 默认不持久化完整文档，除非设置 `SESSION_PERSIST_FULL_DOCUMENT=true`。

## 安全设计

- 上传文件分块保存并强制大小上限。
- 文件名使用安全净化，防止路径穿越。
- 文档签名校验会拒绝伪装成 PDF/DOCX/PPTX 的 HTML 等内容。
- 远程下载只允许公开 HTTP/HTTPS 链接，拒绝 localhost、私有 IP、环回地址等 SSRF 风险目标。
- 远程响应会校验 `Content-Type`、`Content-Length` 和文件头。
- JSON API 会统一拒绝非法 JSON 或非对象 body，并返回 `error`、`code`、`timestamp` 便于定位问题。
- 前端动态内容经过 HTML 转义和 URL 白名单处理。
- SSE 设置 `X-Accel-Buffering: no`，降低代理缓冲影响。

## 项目结构

```text
.
├── web_app.py                  # FastAPI 应用、文档处理、LLM 编排、API 路由
├── paper_whisperer_demo.py     # CLI Demo
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
├── templates/
│   ├── index.html              # Web 页面结构
│   └── static/
│       ├── css/style.css       # 页面样式
│       └── js/app.js           # 前端交互逻辑
├── tests/
│   └── test_security_regressions.py
├── uploads/                    # 运行时上传目录，默认忽略
├── context/                    # session JSON，默认忽略
└── output/                     # Markdown 分析报告，默认忽略
```

## 已知限制

- PDF 提取质量取决于原文件是否有可复制文本层。
- DOCX/PPTX 主要提取文本和表格/备注，复杂图表或图片 OCR 不在当前范围。
- Paper Search 的 `Add` 更适合公开 PDF 直链；只有落地页时可能需要手动下载。
- 分析质量与所选模型和 API 服务稳定性相关。
- 当前是单机 session 文件存储，不是多用户账号系统。

## 更新摘要

### 当前版本重点

- 新增 Deep Research Brief 深度阅读简报。
- 新增 Reading Queue 阅读队列和 `/api/reading-queue`。
- 新增 Evidence / Explain / Critique / Reproduce 追问模式。
- 增强 Session Export，包含分析、阅读队列、搜索轨迹、推荐结果、Q&A 模式和 Mermaid 资源。
- 强化前端可访问性：状态 live region、错误 alert、减少动效模式。
- 强化安全回归测试：上传、远程下载、JSON、session、阅读队列和问答模式。

## License

MIT
