# PaperWhisperer

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-Research%20Workspace-009688) ![License](https://img.shields.io/badge/License-MIT-lightgrey) ![AI](https://img.shields.io/badge/AI--Powered-OpenAI%20Compatible-blueviolet)

PaperWhisperer 是一个 Evidence-first 的研究工作台，面向论文和长篇技术文档的阅读。任何分析结论都要能回到论文的具体页、段落、表格或公式；证据不足时明确说不足，不补全。

主链路：

```text
上传 → Parser Gateway（legacy / docling / MinerU 云端，按格式与质量分级 fallback）
     → Canonical ParsedDocument（格式感知 SourceLocator，PDF 到页并含 bbox）
     → 结构感知分块 → 混合检索（稠密 + 词法）→ 可验证引用
     → PaperFacts 事实层 → Overview / Brief / Mindmap / Evaluation / Q&A
```

同一个会话里可以继续检索论文、维护阅读队列、按四种模式追问，并导出带引用链路的完整研究笔记。

项目地址：<https://github.com/AiFLYF/PaperWhisperer>

## 核心能力

- 四类文档（`.pdf`、`.txt`、`.docx`、`.pptx`）统一走 Parser Gateway；业务层不再直接 import pypdf / python-docx / python-pptx。
- 解析质量门控 8 条规则（页覆盖、空白页比例、乱码、碎行、页眉页脚重复、文本密度、空表格壳、定位覆盖），每条 WARN/FAIL 可配置，可按 `rule_id` 覆盖 severity；**FAIL 会透出为面向用户的 warning**，不允许"解析成功但屏幕什么都不说"。
- 结构感知分块，表格/公式/图注独立成 chunk 并带上下文；标题作 metadata 附着。
- 混合检索：稠密 + 词法 → RRF 融合 → 可选 rerank（默认关）→ 父/邻块扩展。rerank 无标准 OpenAI 接口，做成 provider 且默认关闭，由检索评测决定是否启用。
- 引用逐字校验在服务端完成：direct quote 做精确/归一化对齐，对不上自动降级为**不带引号**的 paraphrase，UI 严格区分两者。
- Evidence API：点击引用可打开 snippet、表格 HTML，以及（装了 `render` extra 时）该引用所指向的 PDF 页真实渲染图。
- PaperFacts 事实层一次低温结构化抽取，Overview / Brief / Mindmap 渲染同一份事实，跨卡片结论一致。
- 四种追问模式：Evidence / Explain / Critique / Reproduce。答案带 `claims[evidence_ids]`、`citations` 和 `confidence`（`supported` / `partially_supported` / `insufficient_evidence`）；文档答不出的问题输出 insufficient 而不是编造。
- 检索聚合 Semantic Scholar 与 arXiv，Crossref / OpenAlex 补全元数据，DOI 丢失可回查；基于当前论文生成延伸阅读推荐。
- 长任务走 durable job 系统：解析、索引、分析全部移出 API 请求生命周期，worker 重启可恢复，刷新与 SSE 断线不丢任务，可取消、可重试。
- 双档位共用同一套存储/任务抽象：个人档零 Docker（SQLite + 本地文件 + 进程内 worker），Hosted 档 `docker compose` 起 PostgreSQL + pgvector。
- 支持 SSE 流式分析，先完成的 section 优先显示，可取消长耗时请求。
- 中英双语界面与导出报告，随语言开关重绘。
- 内置上传校验、远程 URL 校验、session token、XSS 防护、SSRF 防护、JSON 请求校验和结构化错误响应。
- 七道发布门禁落成机器可判定的阈值，`make gates` 一条命令出报告；语料、标注与 bake-off 都在 `eval/`。

## 使用场景

- 快速读懂论文的研究生、开发者和研究人员。
- 需要沉淀文献综述、组会笔记、复现计划的人。
- 想把论文搜索、阅读、追问和导出串成一个轻量工作台的人。
- 需要一个结论可回源、引用可定位的本地研究环境，且不希望为此搭一套基础设施的人。

## 技术栈

- Python 3.10+
- FastAPI + Uvicorn
- OpenAI Python SDK（LLM / embedding 全部走 OpenAI 兼容协议，`base_url` 可配）
- pypdf / python-docx / python-pptx（仅 legacy adapter 使用）
- docling（`parse` extra，可选）、MinerU 云端 API（可选）
- SQLite + FTS5 + numpy，或 PostgreSQL + pgvector
- Vanilla HTML / CSS / JavaScript（27 个 ES module，13 层 CSS）
- KaTeX / marked / svg-pan-zoom

依赖分 extra，按需安装：

| Extra | 装上得到什么 |
| --- | --- |
| （无） | 个人档全功能：legacy 解析、SQLite、进程内 worker |
| `parse` | docling 进程内解析（含 torch 链，体积最大） |
| `render` | PDF 页图渲染（`pypdfium2` + `pillow`），证据面板显示真实页面 |
| `hosted` | Hosted 档：`psycopg` 驱动 + numpy |
| `local-embeddings` | 离线本地 embedding（`fastembed`，ONNX，无 torch）；默认关闭 |
| `dev` | pytest / httpx / ruff |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt        # 等价 make install；pyproject.toml 是唯一依赖来源
```

需要 docling 进程内解析时（重依赖，含 torch 链，首次运行还会下载模型）：

```bash
pip install -e ".[parse]"
```

`requirements.txt` 是指向 `-e .` 的指针文件，不是第二份依赖清单——`make check-deps` 会在你往里写包名时让构建失败。

### 2. 可复现安装

仓库带 `uv.lock`（传递依赖全 pin，含各 extra）。要精确复现 bake-off 那样的环境：

```bash
uv lock            # 改完 pyproject.toml 后刷新
uv sync --frozen   # 按锁文件安装
```

目前 CI 仍走 `pip install -e ".[dev]"`，没有 `uv sync --frozen`；也就是说**依赖锁还没进任何构建路径**，`docling>=2.0` 跨 2.x 的浮动仍是 bake-off 报告不可复现的主要原因。这一点是已知的，见 `docs/OPTIMIZATION_2026.md`。

### 3. 配置环境变量

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

`.env.example` 是配置的权威清单，每一项都带注释说明取值范围与踩过的坑；应用启动时会自动加载项目根目录的 `.env`。

### 4. 启动

```bash
python web_app.py                  # 或 make serve
```

浏览器访问 `http://localhost:5000`。

Hosted 档（PostgreSQL + pgvector，一条命令起）：

```bash
docker compose up -d --build       # 或 make compose-up
```

见 `docs/DEPLOYMENT.md`——它先讲个人档再讲 Hosted 档，顺序不是随便排的。

### 5. 运行测试

```bash
python -m pip install -e ".[dev]"

make check          # ruff + pytest + 依赖检查 + 前端结构检查 + 存储契约（不依赖 Docker）
make test           # 只跑 pytest
make lint           # 只跑 ruff
make gates          # 跑发布门禁，出 eval/reports/release_gates.{md,json}
```

`make` 不是必需设施，等价命令是 `python -m pytest tests/ -q`、`python -m ruff check .`、
`node --experimental-vm-modules tools/check_js_syntax.mjs templates/static/js` 和
`python tools/check_frontend.py`。

前端结构检查不依赖任何参考副本，会拦下三类回归：`index.html` 引用了不存在的文件、
层叠顺序被改乱（`tokens.css` 必须最先加载）、以及模块之间出现循环依赖。

`make parity` 是另一回事：它拿 `.parity/` 里的拆分前原件做行为比对，用来验证当初那次
前端拆分确实无损。该目录不进版本库，只在本地保留。

## 环境变量

常用项如下；完整清单与每项的边界校验见 `.env.example`。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 空 | OpenAI-compatible API key。 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL。 |
| `OPENAI_MODEL` | `gpt-4o-mini` | 主分析与问答模型。 |
| `OPENAI_MAX_CONCURRENCY` | `5` | 全局 LLM 并发上限。 |
| `OPENAI_REQUEST_TIMEOUT_SECONDS` | `60` | 单次 LLM 请求超时。 |
| `OPENAI_MAX_RETRIES` | `3` | LLM 失败重试次数。 |
| `OPENAI_LLM_MAX_TOKENS` | `8000` | 单次响应 token 预算，与模型推理共用；接推理模型时建议调大。 |
| `PAPERWHISPERER_VERSION` | `0.10.0` | 可选部署版本标识，会出现在 `/api/health`。 |
| `MAX_UPLOAD_BYTES` | 16 MB | 单文件上传/下载上限，也是远程导入的上限（1 MB–512 MB）。 |
| `PARSER_BACKEND` | `gateway` | 解析链路开关：`gateway` 或 `legacy`。设回 `legacy` 会让引用只定位到页、定位不到页内位置。 |
| `PARSER_PRIMARY` | `docling` | 首选解析器，内置 `legacy` / `docling` / `mineru`。 |
| `PARSER_FALLBACK_CHAIN` | 空 | 逗号分隔的 fallback 顺序；空 = 不 fallback。 |
| `DOCUMENT_PRIVACY_MODE` | `local_only` | `local_only` 禁止一切云解析；`allow_cloud` 仍需二次同意。 |
| `MINERU_API_TOKEN` | 空 | 为空即禁用云端解析。 |
| `STORAGE_BACKEND` | `sqlite` | `sqlite`（个人档）或 `postgres`（Hosted 档）。 |
| `STORAGE_PG_DSN` | 空 | Hosted 档 PostgreSQL 连接串。 |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | `text-embedding-3-small` / `1536` | 换模型即换维度，需要重建索引而非混用向量。 |
| `RERANK_ENABLED` | `false` | rerank 默认关，由检索评测决定是否启用。 |
| `JOBS_ENABLED` | `true` | durable job 系统；关掉则长任务回到请求生命周期内。 |
| `PAPER_SEARCH_RESULT_LIMIT` | `8` | Paper Search 默认返回数量。 |
| `RECOMMENDATION_RESULT_LIMIT` | `6` | Auto Recommendations 默认返回数量。 |
| `SEMANTIC_SCHOLAR_API_KEY` | 空 | 可选，用于提升 Semantic Scholar 限额。 |
| `SESSION_TTL_SECONDS` | `86400` | session JSON 默认有效期。 |
| `SESSION_PERSIST_FULL_DOCUMENT` | `true` | 是否把完整文档内容持久化到 session JSON。 |
| `FASTAPI_HOST` / `FASTAPI_PORT` / `FASTAPI_RELOAD` | `0.0.0.0` / `5000` / `false` | Web 服务监听与 reload。 |
| `FLASK_HOST` / `FLASK_PORT` / `FLASK_DEBUG` | 可选 | 迁移期兼容旧配置，作为 FastAPI 配置 fallback。 |

未识别的枚举值不会被照办：`PARSER_BACKEND` 退回 `legacy`、`STORAGE_BACKEND` 退回 `sqlite`，
并写一条日志。宁可退回一个已知可用的档，也不要让部署以为自己在另一档上跑。

## Web 使用流程

1. 打开页面，输入 API Key 或使用服务端 `.env`。
2. 上传 PDF / TXT / DOCX / PPTX；选择文件后会显示名称、大小、类型、上限和是否可分析，解析状态卡会说明由哪个 parser 解析、质量如何、是否触发过 fallback。
3. 按需开启结构图、批判性评价、深度阅读简报。
4. 点击 `Analyze Document`，等待流式 section 完成；长耗时分析、检索、推荐和追问都可以用页面上的 Cancel 按钮中止；也可用 `Alt+U`、`Alt+S`、`Alt+Q` 快速跳到上传、论文搜索和追问输入。
5. 在 `Paper Search` 搜索相关论文，对结果点击 `Save` 加入 Reading Queue，点击 `Add` 导入公开 PDF 原文继续分析。
6. 在 `Auto Recommendations` 基于当前论文生成延伸阅读。
7. 在 Ask Questions 中选择追问模式并继续提问。
8. 点击引用 chip 打开证据面板（桌面右栏并排 / 移动端 bottom sheet，打开不离开问题上下文）；Direct quote 与 Paraphrase 视觉严格区分。
9. 点击 `Export Session` 导出分析、阅读队列、搜索轨迹、问答历史和 Mermaid 资源。

## API 概览

所有 JSON API 校验和异常响应都会包含 `error`、`code`、`timestamp` 字段；SSE 流式接口的 `error` 事件也使用相同结构，方便前端和脚本按错误码处理。

### 会话面（`/api`）

`POST /api/analyze` 与 `POST /api/analyze/stream`：上传并分析文档，前者返回完整 JSON，后者走 SSE。表单参数相同：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `file` | 是 | `.txt`、`.pdf`、`.docx`、`.pptx`。 |
| `api_key` | 否 | 请求级 API key，未传时读取环境变量。 |
| `generate_mermaid` | 否 | 是否生成 Mermaid 可视化图，默认 `true`。 |
| `generate_evaluation` | 否 | 是否生成批判性评价，默认 `true`。 |
| `generate_research_brief` | 否 | 是否生成深度阅读简报，默认 `true`。 |
| `session_id` | 否 | 可选自定义 session id。 |

返回要点：`session_id`、`session_token`（后续 session-bound 请求必须携带）、
`sections`（按 section 返回 `status` / `content` / `error` / `retryable`）、
顶层兼容字段 `summary` / `quotes` / `mindmap` / `mermaid` / `evaluation` / `research_brief`，
以及 `suggested_questions` / `next_actions` / `analysis_status`。

其余会话面接口：`POST /api/ask`、`POST /api/ask/stream`（`answer_mode` 取 `evidence` /
`explain` / `critique` / `reproduce`）、`POST /api/search-papers`、`GET /api/download-paper`、
`POST /api/import-paper`、`POST /api/reading-queue`、`POST /api/recommend-papers`、
`POST /api/paper-recommendations`（多信号推荐，每行带推荐理由；该 payload 不落 session，
因为会话写入器会剥掉 `reasons` 字段）、`GET /api/health`。

### 资源面（`/v1`）

文档是持久资源，不依赖 session token：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/documents` | 上传并注册文档 |
| `POST` | `/v1/documents/register` | 注册一个已在磁盘上的文档 |
| `GET` | `/v1/documents/{id}` | 文档元信息 |
| `GET` | `/v1/documents/{id}/metadata` | 解析元信息（parser / 质量 / fallback） |
| `GET` | `/v1/documents/{id}/pages` | 分页内容 |
| `POST` | `/v1/documents/{id}/reparse` | 换 parser 重解析 |
| `DELETE` | `/v1/documents/{id}` | 删除文档 |
| `POST` | `/v1/documents/{id}/qa` | 基于该文档问答（SSE） |
| `POST` | `/v1/search/papers` | 论文检索 |
| `GET` | `/v1/documents/{id}/evidence/{evidence_id}` | 证据详情：snippet / 表格 HTML / 页图 |
| `GET` | `/v1/documents/{id}/pages/{page}/image` | 该页的渲染图 |

没装 `render` extra 时，证据路由返回 501、payload 的 `page_image.renderer` 读作 `none`，
前端继续用坐标重建布局——**不会**渲染一张坏图。

### 任务面（`/v1/jobs`）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/v1/jobs` | 提交任务（幂等键去重） |
| `GET` | `/v1/jobs/{id}` | 任务状态 |
| `GET` | `/v1/jobs/{id}/events` | 状态事件历史 |
| `GET` | `/v1/jobs/{id}/stream` | SSE 订阅状态变化 |
| `POST` | `/v1/jobs/{id}/cancel` | 取消（写入 `cancel_requested`） |
| `POST` | `/v1/jobs/{id}/retry` | 重试（重试预算耗尽后显式重投） |
| `POST` | `/api/jobs/analyze` | 提交分析任务（会话面入口） |
| `POST` | `/api/jobs/ask` | 提交问答任务（会话面入口） |

## 备份与恢复

个人档备份就是复制数据目录，但工具会做更多：打包前对 SQLite 做一致性快照，清单记录每个成员的
sha256 与大小，恢复时**先全量校验再落盘**，所以一个损坏的备份会在目标目录还没被碰过之前就被发现。

```bash
make backup                  # 备份到 backups/（TIER=personal|hosted）
make verify-backup           # 只校验每个摘要，不恢复
make restore                 # 恢复到指定目录（目标必须为空或不存在）
make restore-drill           # 备份本 checkout → 恢复到临时目录 → 逐字节对比
```

`make restore-drill` 是唯一算数的验收：**"能备份"必须验证过一次 restore 才算完成**。

## Session 生命周期

- 分析成功后服务端会写入 `context/<session_id>.json`。
- `session_token` 只在分析响应中返回，服务端只保存 token hash。
- `/api/ask`、`/api/ask/stream`、`/api/recommend-papers`、`/api/reading-queue` 必须携带有效 `session_token`。
- `/api/search-papers` 如果携带 `session_id`，也必须携带有效 token。
- session 默认带 `expires_at`，过期或损坏的 JSON 会被自动清理。

`/v1` 资源面不走 session，走文档 id；两套鉴权模型是分开的。

## 安全设计

- 上传文件分块保存并强制大小上限；文件名安全净化，防止路径穿越。
- 文档签名校验会拒绝伪装成 PDF/DOCX/PPTX 的 HTML 等内容。
- 远程下载只允许公开 HTTP/HTTPS 链接，拒绝 localhost、私有 IP、环回地址等 SSRF 风险目标；**每一个重定向目标都会在请求前单独校验**，解析出的地址与请求使用的地址之间没有时间窗。
- 远程响应会校验 `Content-Type`、`Content-Length` 和文件头。
- 沙箱帧跨进程用 JSON 而不是 pickle 传递。
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
├── paper_whisperer_demo.py     # CLI Demo
├── paperwhisperer/
│   ├── app.py                  # FastAPI 实例、静态资源挂载、路由注册
│   ├── core/                   # 环境变量解析、路径与限额、统一错误响应、可观测性、SLO
│   ├── documents/              # 上传落盘、格式校验、文本抽取与分块
│   ├── ingestion/              # 解析底座：canonical schema、Parser Gateway、质量门控、
│   │                           #   fallback 编排、docling / MinerU / legacy 三个 adapter
│   ├── llm/                    # OpenAI 客户端（重试/流式）、embedding、prompt 与模式定义
│   ├── rag/                    # 分块、混合检索、rerank、引用逐字校验、证据抽取
│   ├── analysis/               # 编排器、section 定义、PaperFacts 事实层、结果持久化
│   ├── search/                 # Semantic Scholar / arXiv 检索、身份归一、元数据补全
│   ├── sessions/               # session 读写、token 校验与过期清理
│   ├── remote/                 # 远程下载与 SSRF 校验
│   ├── infra/                  # durable job 系统、对象存储、SQLite / PostgreSQL 双实现
│   └── api/                    # 路由（documents / qa / papers / evidence / jobs / v1 / health）
├── templates/
│   ├── index.html              # Web 页面结构，按层序引入 CSS 与入口模块
│   └── static/
│       ├── css/                # 13 个分层样式：tokens → base → layout → ...
│       └── js/                 # 27 个 ES module，main.js 为唯一入口
├── assets/                     # 落地页（根 index.html）自有资源
├── tools/                      # 备份/恢复、前端结构检查、故障注入、负载探针、标注工作流
├── eval/                       # 语料清单、bake-off、标注、七道发布门禁
├── docs/                       # 计划、部署、门禁、bake-off、评测方法等 17 篇
├── tests/                      # pytest 行为测试
├── uploads/                    # 运行时上传目录，默认忽略
├── context/                    # session JSON、解析缓存与存储，默认忽略
└── output/                     # Markdown 分析报告，默认忽略
```

前端同样按职责拆分：`main.js` 只负责装配，各模块（`state` / `api` / `sanitize` / `mermaid` /
`chat` / `papers` / `export` / `evidence` 等）之间无循环依赖。

## 已知限制

- PDF 提取质量取决于原文件是否有可复制文本层；扫描件需要 OCR，`PARSER_FORCE_OCR` 可强制触发。
- DOCX/PPTX 主要提取文本和表格/备注，复杂图表或图片 OCR 不在当前范围。
- `MAX_DOCUMENT_PAGES` 目前只被 `/api/health` 报告，**解析网关还不拒绝超大文档**，不要把它当资源防护用。
- `PAGE_RENDER_ORIGINALS_DIR` 配了也没有东西可找：还没有代码把上传写进对象存储的 original 层，`/api/analyze` 也会在分析结束后删掉临时上传。证据路由在这种情况下明确报 `reason=original_unavailable`。
- `AUTH_MODE` 默认 `disabled`，`AUTH_STATIC_TOKENS` / `AUTH_TENANT_DOMAIN` / tenant scope 已实现但**尚未接线**：当前部署默认是单租户、无鉴权，不要把实例暴露到公网。`assert_startup_safe()` 会在 `AUTH_MODE=static` 时拒绝启动而不是退化成开放访问。
- 没有入站速率限制；Semantic Scholar / arXiv 的出站限速是有的。
- Paper Search 的 `Add` 更适合公开 PDF 直链；只有落地页时可能需要手动下载。
- 分析质量与所选模型和 API 服务稳定性相关。
- 依赖未锁进构建路径：CI 用 `pip install -e`，`uv.lock` 存在但没被使用，见"快速开始"第 2 节。
- 引用精度与 Recall@K 两道门禁仍是 BLOCKED：缺的不是 harness，而是一份**经人工确认**的标注集。详见 `docs/RELEASE_GATES.md`。

## 文档

| 文档 | 内容 |
| --- | --- |
| `docs/DEVELOPMENT_PLAN.md` | P0–P4 分段开发计划与六项已确认决策 |
| `docs/DEPLOYMENT.md` | 双档位部署、从个人档迁到 Hosted 档、备份恢复 |
| `docs/RELEASE_GATES.md` | 七道门禁测什么、怎么判、今天卡在哪 |
| `docs/PARSER_BAKE_OFF.md` | legacy / docling / MinerU 的对比与默认路由决策 |
| `docs/RAG_MEASUREMENT.md` | 检索与引用的评测方法 |
| `docs/OPTIMIZATION_2026.md` | 2026-10 的优化与新增功能规划，含安全审计记录 |
| `docs/FAILURE_MODES.md` | 故障模式与可观察行为 |
| `docs/SLO.md` | 性能目标 |
| `docs/LLM_TESTS.md` | 需要真实模型的测试怎么跑 |
| `docs/DOCLING_SERVE.md` | docling-serve 容器模式 |
| `docs/I18N.md` | 双语界面约定 |
| `docs/FRONTEND_DESIGN.md` | 前端分层与设计系统 |
| `docs/E2E_JOURNEY.md` / `docs/USER_WALKTHROUGH.md` | 端到端旅程与用户走查 |

## 更新摘要

### 0.10.0

- 解析底座：canonical schema、Parser Gateway、质量门控与分级 fallback 全部落地，业务层不再直接依赖 pypdf / python-docx / python-pptx；默认路由由 bake-off 数据决定（docling 优先，DOCX 走 legacy）。
- Evidence RAG：结构感知分块、混合检索、引用逐字回源校验，对不齐降级为不带引号的 paraphrase；PaperFacts 事实层统一各卡片事实。
- 长任务移出 API 请求生命周期：durable job 系统（幂等、可取消、可重试、worker 重启可恢复），`/v1` 资源化文档与任务接口。
- 双档位存储：SQLite（个人档）/ PostgreSQL + pgvector（Hosted 档）共用同一套接口，同一测试套件两档各跑一遍。
- 证据面板：引用 chip、snippet、表格 HTML、PDF 页图渲染（`render` extra）。
- 中英双语界面与导出报告。
- 安全加固：每个重定向目标单独校验、沙箱帧改 JSON、出站限速、SSRF 时间窗关闭。
- 工程化：七道发布门禁、bake-off 与标注工作流、`uv.lock`、故障注入与负载探针、备份/恢复与 restore 演练。

### 0.9.0

- 后端从单文件 `web_app.py` 拆为 `paperwhisperer/` 分层包，配置与错误处理各自收敛到单一来源。
- 前端从单体 `app.js` 拆为 ES module，样式分层。
- 测试从字符串断言改为行为测试，测试替身只替换真正会触网的边界。
- 修复流式分析阻塞事件循环、流式重试重复吐字、关闭可选分析项时的 500 等问题。

## License

MIT
