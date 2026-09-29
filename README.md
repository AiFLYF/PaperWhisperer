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
- 内置上传校验、远程 URL 校验、会话 token、XSS 防护、JSON 请求校验和结构化错误响应。
- 前端提供拖拽上传、结构化文件反馈、带当前步骤语义的分析进度、结果状态卡、请求取消按钮、顶部快速导航、跳转链接、可被辅助技术识别的快捷键、返回顶部、语义化按钮反馈、追问模式键盘导航、减少动效适配、移动端触控优化、受限存储容错和资源加载预热。

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
python -m pip install -e ".[dev]"

make check          # ruff + pytest + 前端结构检查
make test           # 只跑 pytest
make lint           # 只跑 ruff
make check-frontend # 只跑前端结构检查
```

`make` 不是必需设施，等价命令是 `python -m pytest tests/ -q`、`python -m ruff check .`、
`node --experimental-vm-modules tools/check_js_syntax.mjs templates/static/js` 和
`python tools/check_frontend.py`。

前端结构检查不依赖任何参考副本，会拦下三类回归：`index.html` 引用了不存在的文件、
层叠顺序被改乱（`tokens.css` 必须最先加载）、以及模块之间出现循环依赖。

`make parity` 是另一回事：它拿 `.parity/` 里的拆分前原件做行为比对，用来验证当初那次
前端拆分确实无损。该目录不进版本库，只在本地保留。

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
2. 上传 PDF / TXT / DOCX / PPTX；选择文件后会显示名称、大小、类型、上限和是否可分析。
3. 按需开启结构图、批判性评价、深度阅读简报。
4. 点击 `Analyze Document`，等待流式 section 完成；页面会显示上传、分析、渲染、就绪进度，并在完成后聚焦到分析工作区。长耗时分析、检索、推荐和追问都可以用页面上的 Cancel 按钮中止；也可用 `Alt+U`、`Alt+S`、`Alt+Q` 快速跳到上传、论文搜索和追问输入。
5. 在 `Paper Search` 搜索相关论文，也可以点击示例查询快速开始；空搜索、加载中、无结果和部分失败都会显示可操作提示。
6. 对结果点击 `Save` 加入 Reading Queue，点击 `Add` 可导入公开 PDF 原文继续分析。
7. 在 `Auto Recommendations` 基于当前论文生成延伸阅读。
8. 在 Ask Questions 中选择追问模式并继续提问，可点击示例问题快速填充贡献、证据、局限和复现类追问，模式说明会提示答案结构差异。
9. 点击 `Export Session` 导出分析、阅读队列、搜索轨迹、问答历史和 Mermaid 资源；复制和导出按钮会显示成功或缺失内容反馈。

## API 概览

所有 JSON API 校验和异常响应都会包含 `error`、`code`、`timestamp` 字段；SSE 流式接口的 `error` 事件也使用相同结构，方便前端和脚本按错误码处理。

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

该接口响应包含 `Cache-Control: no-store`，探活和部署检查会读取实时运行状态。`uptime_seconds` 可用于确认进程是否刚重启。所有响应都会附带 `X-Process-Time-Ms`，便于本地调试或部署排查慢请求。

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
- 响应会附加 `X-Content-Type-Options`、`X-Frame-Options`、`Referrer-Policy` 和 `Permissions-Policy` 等防御性安全头。
- 前端动态内容经过 HTML 转义和 URL 白名单处理，生成内容中的图片源仅允许 HTTP/HTTPS。
- 临时上传、远程下载、session 和响应关闭路径会记录清理失败，方便排查磁盘或网络资源异常。
- SSE 设置 `X-Accel-Buffering: no`，降低代理缓冲影响。

## 项目结构

后端按职责分层，模块之间只往下依赖：

```text
.
├── web_app.py                  # 入口：装配 app 并启动服务
├── paperwhisperer/
│   ├── app.py                  # FastAPI 实例、静态资源挂载、路由注册
│   ├── config.py               # 配置读取（向后兼容 re-export）
│   ├── core/
│   │   ├── config.py           # 环境变量解析、路径与限额、密钥解析
│   │   ├── text.py             # 文本工具、会话与文件的安全清理
│   │   └── errors.py           # 统一错误响应、SSE 事件构造
│   ├── documents/              # 上传落盘、格式校验、文本抽取与分块
│   ├── llm/                    # OpenAI 客户端（重试/流式）、prompt 与模式定义
│   ├── analysis/               # 编排器、section 定义、结果持久化服务
│   ├── search/                 # Semantic Scholar / arXiv 检索与 HTTP 层
│   ├── sessions/               # session 读写、token 校验与过期清理
│   ├── remote/                 # 远程下载与 SSRF 校验
│   └── api/                    # 路由（documents / qa / papers / health）与共享依赖
├── paper_whisperer_demo.py     # CLI Demo
├── pyproject.toml              # 依赖、pytest 与 ruff 配置
├── requirements.txt            # Python 依赖
├── .env.example                # 环境变量模板
├── templates/
│   ├── index.html              # Web 页面结构，按层序引入 CSS 与入口模块
│   └── static/
│       ├── css/                # 11 个分层样式：tokens → base → layout → ...
│       └── js/                 # 20 个 ES module，main.js 为唯一入口
├── assets/                     # 落地页（根 index.html）自有资源
├── tools/                      # 前端结构检查脚本（CI 使用）
├── tests/                      # pytest 行为测试
├── uploads/                    # 运行时上传目录，默认忽略
├── context/                    # session JSON，默认忽略
└── output/                     # Markdown 分析报告，默认忽略
```

前端同样按职责拆分：`main.js` 只负责装配，各模块（`state` / `api` / `sanitize` / `mermaid` /
`chat` / `papers` / `export` 等）之间无循环依赖。

## 已知限制

- PDF 提取质量取决于原文件是否有可复制文本层。
- DOCX/PPTX 主要提取文本和表格/备注，复杂图表或图片 OCR 不在当前范围。
- Paper Search 的 `Add` 更适合公开 PDF 直链；只有落地页时可能需要手动下载。
- 分析质量与所选模型和 API 服务稳定性相关。
- 当前是单机 session 文件存储，不是多用户账号系统。

## 更新摘要

### 当前版本重点

- 后端从单文件 `web_app.py` 拆为 `paperwhisperer/` 分层包，模块只往下依赖，配置与错误处理各自收敛到单一来源。
- 前端从 2690 行单体 `app.js` 拆为 20 个 ES module，样式从单文件拆为 11 层 CSS；拆分行为由 85 项等价探针与 CSS 规则比对保证无损。
- 测试从字符串断言改为行为测试，测试替身只替换真正会触网的边界（LLM 传输层与远程下载层），运行时目录全部重定向到临时目录。
- 修复流式分析阻塞事件循环：阻塞式生成器改由工作线程驱动，SSE 消费端保持全异步。
- 修复流式重试重复吐字：探针缓冲区扣留 1 个 chunk，只有在尚未吐出任何内容时才允许重放。
- 修复关闭任一可选分析项时 `/api/analyze` 抛 `KeyError` 变成 500 的问题。
- 修复 Mermaid 闭合围栏残留在图表源码里导致渲染失败的问题。
- 修复流式路径缺失 HTML 响应检测，代理登录页不再被当作正文吐给前端。
- 修复清理函数在 `finally` 中抛出时掀翻流式响应的问题。
- 强化安全与稳定性：上传、远程下载、JSON、session、阅读队列、问答模式、临时资源清理日志、生成内容 URL 白名单、状态卡 DOM 文本节点渲染、格式化内容集中替换和前端动态内容硬化。
- 工程化设施：`pyproject.toml`、`.editorconfig`、`Makefile` 与 GitHub Actions（后端 lint + 测试、前端等价校验）。

## License

MIT
