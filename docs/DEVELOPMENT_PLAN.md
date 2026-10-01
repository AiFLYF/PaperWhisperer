# PaperWhisperer 分段开发计划

依据《PaperWhisperer 产品级生产级重构规划（最终技术复核版）》（2026-09-29）制定，并结合 2026-09-29 需求确认会议的六项决策做了架构调整。规划文档是背景与理由的完整来源，本文件是可执行的任务分解；两者冲突时以本文件的"已确认决策"为准。

## 1. 定位

从"AI PDF 摘要器"升级为 Evidence-first Research Workspace：任何分析结论都能回到论文具体页、段落、表格、公式；证据不足时明确说不足，不补全。

主链路：

```
上传 → Parser Gateway（Docling 默认候选 / MinerU 云端 fallback / 本地 OCR 修复）
     → Canonical ParsedDocument（格式感知 SourceLocator）
     → 结构感知 chunk → 混合检索（dense + lexical）→ 可验证引用
     → PaperFacts 事实层 → Overview / Brief / Mindmap / Evaluation / Q&A
```

## 2. 已确认决策（2026-09-29）

| # | 决策 | 内容 |
|---|---|---|
| D1 | 计划范围 | 完整 P0-P4；P0/P1 拆到可直接开工的任务粒度，P2-P4 完整但粒度递粗 |
| D2 | 部署档位 | 双档位。个人档为默认：零 Docker（SQLite + 本地文件 + 进程内任务）；Hosted 档可选：docker compose 起 PostgreSQL+pgvector。两档共用同一套 storage/job 抽象接口 |
| D3 | 模型运行方式 | LLM / embedding / rerank 全部走 OpenAI 兼容 API（base_url 可配）。不做本地 BGE-M3。rerank 无标准 OAI 接口，做成可选 provider，默认关闭 |
| D4 | MinerU 云端 | P0 即接入云端精准解析 API 全链路（签名上传、轮询、下载、normalize），含 privacy 同意流程与审计。MinerU self-host 本地版只留接口占位，实现延后 |
| D5 | 评测语料 | 定分类标准与配额（见第 6 节），由项目所有者按清单投喂文档 |
| D6 | 本计划 | markdown 入仓，随开发勾进度 |

### 与规划文档的差异说明

1. 文档以 Hosted（PostgreSQL/Celery/Redis/S3/多实例）为生产主架构；本计划改为个人档默认零 Docker，Hosted 组件（Celery、Redis、S3、鉴权 tenant）推迟到 P4 且仅当公开部署时实施。
2. 文档推荐本地 BGE-M3 embedding 与 bge-reranker；本计划全部改为 API provider，选型由检索评测决定，不写死模型名。
3. MinerULocalAdapter（self-host MinerU 4.x）从 P0 移除，接口保留。
4. 目录结构采用渐进改造（在现有 `paperwhisperer/` 包上扩展），不做一步到位的大搬迁。

## 3. 目标架构与档位

| 层 | 个人档（默认） | Hosted 档（compose） |
|---|---|---|
| 元数据/任务/会话 | SQLite（WAL） | PostgreSQL |
| 向量检索 | SQLite BLOB + numpy 暴力检索（个人量级足够） | pgvector |
| 词法检索 | normalized substring + FTS5（构建时探测，可用则启用） | PostgreSQL FTS + pg_trgm |
| 对象存储 | 本地文件系统（数据目录） | MinIO / S3（P4） |
| 后台任务 | 进程内 worker + DB job 表驱动 | 同左；多实例时升级 Celery+Redis（P4） |
| LLM/embedding | OpenAI 兼容 API | 同左 |
| 解析 | Docling 进程内（或 docling-serve 容器）+ MinerU 云端 fallback | 同左 |

两档必须通过同一套接口访问：`MetadataStore` / `VectorStore` / `LexicalRetriever` / `ObjectStore` / `JobSystem`。行为一致性由同一测试套件在两档各跑一遍保证。

## 4. 目录结构规划（渐进）

现有包基础上新增三个顶层模块，`search/` 到 P3 再演进为 `scholarly/`，`documents/` 中的 uploads/validation 保留：

```
paperwhisperer/
  ingestion/            # P0 新增
    schema.py           # Canonical ParsedDocument / Element / SourceLocator / ...
    gateway.py          # DocumentParser Protocol + ParserRouter
    quality.py          # ParseQualityReport
    orchestrator.py     # fallback 编排 + 隐私策略
    adapters/
      docling_adapter.py
      mineru_cloud.py
      legacy_loader.py  # 旧 DocumentLoader 包一层，仅回滚用
  rag/                  # P1 新增
    chunking.py
    embeddings.py
    indexing.py
    retrieve.py
    rerank.py           # 可选 provider
    citations.py        # quote 回源校验
  infra/                # P1 起逐步新增
    stores/             # sqlite_impl/ pg_impl/
    object_store.py
    jobs.py
  analysis/             # 已有；P1 增加 paper_facts.py
  scholarly/            # P3 由 search/ 演进
  api/ core/ llm/ sessions/ remote/   # 已有
eval/
  corpus/               # bake-off 语料（gitignore，本地投喂）
  manifest.yaml         # 语料分类清单
  run_parser_eval.py
```

## 5. 分阶段计划

### Phase 0 — 解析底座（对应规划文档 P0 / PW-001~006）

目标：业务层统一走 Parser Gateway；解析结果带格式感知 SourceLocator；质量门控与分级 fallback 可用；默认路由由 bake-off 数据决定。

| 任务 | 名称 | 内容 | 验收标准 | 依赖 | 规模 |
|---|---|---|---|---|---|
| T0.1 | 评测语料标准与收集 | 按第 6 节分类标准建 `eval/corpus/` + `manifest.yaml`；先放 3-5 份样例 fixture 供开发用；所有者按清单投喂至 50+ 份 | manifest 覆盖全部类别与配额；bake-off（T0.9）前 50 份就位 | 无 | S |
| T0.2 | Canonical Schema（PW-001） | `ingestion/schema.py`：ParsedDocument / SourceUnit / Element / Table / Formula / Figure / SourceLocator / Chunk / Evidence / PaperFacts，全部 Pydantic + schema_version。SourceLocator 按格式：pdf_page(page,bbox)、slide、sheet_range、docx_element、txt_offset | 每种格式一份最小 fixture 通过校验；locator 序列化/反解单测；schema 有版本字段 | 无 | M |
| T0.3 | Parser Gateway（PW-002） | `ingestion/gateway.py`：DocumentParser Protocol、ParseResult(status/quality/raw_artifact_uri/warnings/duration_ms)、ParserRouter；统一 timeout/error 映射；记录 parser_name/version/options_hash；`PARSER_BACKEND=legacy\|gateway` feature flag | gateway 模式下四格式全走新链路；legacy 开关可整体回滚；mock parser 单测 | T0.2 | M |
| T0.4 | Docling Adapter（PW-003） | 进程内与 docling-serve 两种模式；DoclingDocument 到 canonical 的 normalize（保留 page/bbox provenance、表格二维结构、公式 LaTeX）；raw Docling JSON 落盘 | 三格式 fixture 产出合法 ParsedDocument；PDF 元素带 page+bbox；表格保留行列结构；serve 模式 mock 测试不依赖网络。注意：docling 为重依赖（含 torch），requirements 拆 base / parse extras，首次运行下载模型 | T0.3 | L |
| T0.5 | MinerU Cloud Adapter（PW-004） | `file-urls/batch` 签名 → PUT 上传 → 轮询 → 下载 → normalize；timeout/429/额度降级重试；cloud consent 检查（privacy_mode=allow_cloud 且显式同意）；每次调用写审计（操作者、时间、文件、触发原因） | mock 全链路单测；真实 token smoke 测试（手动触发，CI 可选）；local_only 模式集成测试证明零云调用。需要提供 MINERU_API_TOKEN | T0.3 | M |
| T0.6 | Parse Quality Gate（PW-005） | `ingestion/quality.py`：页覆盖、空页比例、乱码/控制字符、极端碎行、header/footer 重复、文本密度、表格空壳、locator 覆盖率；每条规则 WARN/FAIL 可配置，保留 flags 不压缩成黑盒分数 | 语料中已知坏文档（扫描、乱码）触发正确级别；每条规则独立单测 | T0.2 | M |
| T0.7 | Fallback 编排 + 隐私策略 | `ingestion/orchestrator.py`：主 route → 本地修复重试（force OCR）→ 云 fallback（策略允许时）；整篇择优，不按页混合拼接；解析缓存按 (sha256, parser, version, options_hash) 幂等 | 编排全路径单测；local_only 集成测试零出网；缓存命中不重复付费解析 | T0.4, T0.5, T0.6 | M |
| T0.8 | 配置系统重构 | core/config.py 扩展 PARSER_* / DOCUMENT_PRIVACY_MODE / MINERU_* / MAX_UPLOAD_BYTES / MAX_DOCUMENT_PAGES 等；迁移 pyproject.toml + lockfile；.env.example 同步 | 所有外部策略可配置且有安全默认值；依赖版本锁定 | 无 | S |
| T0.9 | Parser Eval Runner + bake-off（PW-006） | `eval/run_parser_eval.py`：同一 corpus 批跑 legacy / Docling / MinerU cloud，输出对比报告（page coverage、阅读顺序、表格、公式、locator、耗时、失败率）；跑首轮 bake-off 并落档路由决策 | 报告可复现（corpus 版本固定）；默认 parser 与 fallback 规则写入决策记录 | T0.1, T0.4, T0.5 | M |

P0 退出标准：

- 常见 PDF/DOCX/PPTX/TXT 只走统一 Parser Gateway；扫描/复杂文档自动 fallback（本地 OCR 必有，云端按策略）
- 成功解析文档的关键正文元素 100% 有可解析 SourceLocator（PDF 到 page 可用时含 bbox，PPTX 到 slide/element，DOCX 到 heading/paragraph）
- 业务层不再 import pypdf / python-docx / python-pptx（旧分支仅存于 legacy adapter）
- bake-off 报告与默认路由决策已落档

### Phase 1 — Evidence RAG 与可信引用（对应 PW-007~012）

目标：全文检索取代前 N 字符 excerpt；direct quote 程序级 100% 回源；PaperFacts 统一事实层。

| 任务 | 名称 | 内容 | 验收标准 | 依赖 | 规模 |
|---|---|---|---|---|---|
| T1.1 | 存储抽象双档 | `infra/stores/`：MetadataStore、VectorStore（SQLite BLOB+numpy / pgvector）、LexicalStore（substring+FTS5 探测 / FTS+pg_trgm）；接口定义与两实现 | 同一接口各自单测；个人档零外部服务；compose 档一键起 PG 且同套测试通过 | 无（可与 P0 并行） | L |
| T1.2 | 结构感知 Chunking（PW-008 前半） | `rag/chunking.py`：按 Element/章节切分；标题作 metadata 附着；表格/公式/图注独立 chunk 带上下文；token-aware 合并切分；index_version 编入 embedding provider/model/dim | 长论文 fixture 无跨语义段拆断；表格不混入正文；索引可幂等重建 | T0.2 | M |
| T1.3 | Embedding Gateway | llm/ 扩展：OAI 兼容 /embeddings，provider 可配；批量与并发限制；按 (model, content_hash) 缓存；成本记录进 analysis_run | provider 切换无需改业务代码；缓存命中零重复调用；失败重试 | T0.8 | M |
| T1.4 | Hybrid Retriever（PW-009） | `rag/retrieve.py`：dense + lexical + metadata filter → RRF 融合 → 可选 rerank（默认关）→ parent/neighbor 扩展；按任务类型的 Retrieval Plan（问答/概念/贡献/实验/复现，见规划文档 6.4）；每次检索记录候选与最终 evidence | 中英混合 query 均可召回；专有名词/DOI/数据集名精确查询稳定；检索记录可查 | T1.1, T1.2, T1.3 | L |
| T1.5 | Evidence API（PW-007） | GET /v1/documents/{id}/evidence/{evidence_id}：locator 解析 → snippet / 页图 / 表格 HTML | citation 点击能打开证据；locator 100% 可解析或明确标记无视觉定位能力 | T1.1 | M |
| T1.6 | Citation Verifier（PW-010） | `rag/citations.py`：direct quote 服务端 exact / normalized 对齐（程序校验，非提示词）；失败自动降级为 paraphrase，UI 严格区分 | 100% deterministic 单测；eval 中 direct quote exact-match = 100% | T0.2 | M |
| T1.7 | Q&A 迁移（PW-011） | 移除 head excerpt 作为唯一知识源；AnswerResult 输出 answer + claims[evidence_ids] + citations + confidence(supported / partially_supported / insufficient_evidence) | 命中后文（实验/限制章节）的问题可正确回答；无证据问题输出 insufficient 而非编造 | T1.4, T1.5, T1.6 | M |
| T1.8 | PaperFacts + Analysis DAG（PW-012） | `analysis/paper_facts.py`：一次低温度结构化抽取（research_question / contributions / method / datasets / metrics / main_results / limitations / reproducibility，每字段带 evidence_ids 与 extraction_status）；Overview/Brief/Mindmap 改为渲染事实层；Evaluation = 事实层 + 专项检索；Mermaid 第一层由 section tree 直接生成；analysis_run 记录 model/prompt_version/evidence_version/cost | 跨卡片事实一致（同一结论在不同卡片相同）；LLM 调用成本较旧路径下降且有对比数据 | T1.4 | L |
| T1.9 | 前端 Evidence 体验 | citation chips；evidence panel（桌面右栏并排 / 移动 bottom sheet，打开不离开问题上下文）；parse status 展示（parser / quality / fallback 警告 / reparse 按钮）；Direct quote 与 Paraphrase 视觉严格区分 | 上述交互可用；degraded 状态对用户可见 | T1.5 | M |

P1 退出标准：

- 任何 Q&A / Quote / 关键分析不再依赖"文档前 N 字符"
- Direct quote 逐字回源 100%；citation 可定位到 source unit / element
- 中文与中英混合检索有独立回归指标（不只有英文 FTS 数字）

### Phase 2 — 任务系统与持久化（规划文档 P2 的双档位重构）

目标：长任务移出 API 请求生命周期；状态持久化、可恢复、可取消；Session 迁数据库。

| 任务 | 名称 | 内容 | 验收标准 | 依赖 | 规模 |
|---|---|---|---|---|---|
| T2.1 | Durable Job 系统 | `infra/jobs.py`：job 表（SQLite/PG 双实现）+ 进程内 worker；状态机 queued → parsing → quality_check → fallback? → normalizing → indexing → analyzing → completed / partial / failed / cancelled；幂等（idempotency key）；取消写入 cancel_requested，worker 在可取消边界检查；SSE 只订阅状态 | worker 重启后任务恢复或进入可重试终态；刷新/断线不丢任务；重复提交去重 | P0 | L |
| T2.2 | 资产与数据模型 | ObjectStore 接口（文件系统实现；S3/MinIO 实现留 P4）；original / parsed / canonical / derived 分层存储；表：documents / document_versions / jobs / chunks / conversations / messages / reading_queue / analysis_runs；乐观锁防 lost update；Session JSON 一次性迁移（兼容读取，原文件保留） | 迁移脚本幂等；并发读写无 lost update（并发测试）；旧 session 首次读取自动迁移 | T2.1 | L |
| T2.3 | 长任务出 API 进程 | 解析/索引/分析全部经 job 系统；API 只提交与查询 | 负载测试下 API worker 不被长任务占用 | T2.1, T2.2 | M |
| T2.4 | API v1 资源化 | /v1/documents（upload/register/metadata/pages/reparse/delete）、/v1/jobs/{id}、/v1/documents/{id}/qa（SSE）、/v1/search/papers；对外契约冻结 + OpenAPI 文档 | 现有前端无破坏性切换；OpenAPI 生成 | T2.3 | M |
| T2.5 | Compose 档 | docker-compose.yml：app + PostgreSQL(pgvector)，可选 profile 挂 Redis/MinIO；CI 验证 compose up 冒烟 | compose up 一键可用；同一测试套件两档均通过 | T1.1 | S |
| T2.6 | 备份与恢复 | SQLite 档：数据目录说明 + 备份命令；PG 档：pg_dump 定期 + 至少一次真实 restore 演练并记录 | "能备份"必须验证过 restore 才算完成 | T2.2 | S |

P2 退出标准：API 多实例运行状态一致；worker 重启任务可恢复；浏览器刷新/SSE 断开不丢任务；旧 session 无损迁移。

### Phase 3 — 学术发现与产品体验

| 任务 | 内容 | 验收标准 | 规模 |
|---|---|---|---|
| T3.1 PaperIdentity 与元数据补全 | 统一身份优先级 DOI > arXiv > S2 paperId > OpenAlex > normalized(title, year, first_author)；Crossref/OpenAlex enrichment；search/ 演进为 scholarly/ | 预印本/正式版去重正确；DOI 丢失可回查 | M |
| T3.2 多信号推荐 | S2 recommendations + OpenAlex related/citations + LLM query expansion → 统一 dedupe → 分信号展示 | UI 有"为什么推荐"（被引/引用/related/主题相似/同数据集） | M |
| T3.3 Evidence viewer 完整化 | PDF 跳页 + bbox 高亮（需页图渲染）；PPTX 跳 slide；DOCX 定位 element；移动端 bottom sheet | 三种格式 citation 均可视觉定位 | L |
| T3.4 Reparse 与 quality UI | 手动 force OCR / 换 parser 重解析；parser route 与 quality 对用户可见 | 用户能明确知道文档被谁解析、是否经过云端 | S |
| T3.5 导出升级 | Markdown/HTML/DOCX 导出保留引用链接、document_version、analysis 元数据 | 导出物可追溯生成条件 | M |

### Phase 4 — 运营、安全与规模化

| 任务 | 内容 | 验收标准 | 规模 |
|---|---|---|---|
| T4.1 可观测性 | structured logging；request_id/job_id/document_id/analysis_run_id 贯穿全链路；指标按规划文档 13.2 裁剪（parsing / RAG / LLM / grounding / jobs / cost）；Hosted 档可选 OTel+Prometheus | 任一失败可从 document_id/job_id 追到 parser/fallback/LLM/外部 API | L |
| T4.2 安全加固 | parser 资源限制（CPU/内存/时长/页数）；zip bomb 与嵌套压缩防护；文档 prompt injection 测试集（文档内容不改变系统策略、不触发外部工具）；SBOM + 依赖扫描 + lockfile | 注入测试集通过；secrets 不进日志 | M |
| T4.3 Eval 常态化 | release gates 进 CI；corpus 扩到 150-200 份并固定版本；parser/模型升级必须跑 regression 后上线 | 升级有回归记录；gate 不通过阻塞发布 | M |
| T4.4 负载与故障注入 | k6/Locust；worker 重启、DB 连接耗尽、MinerU 429/超时、SSE 断线重连、用户取消、重复提交 | 故障场景均有明确可观察行为，无静默丢失 | M |
| T4.5 Hosted 档强化（仅当公开部署） | 鉴权 / tenant scope / 配额；Celery+Redis 多 worker；S3/MinIO；MinerULocalAdapter（self-host）按需实现 | 跨租户访问为零（安全测试）；云调用全部有审计 | L |
| T4.6 性能 SLO | 按页数/格式建立 P95 时延目标；queue backlog 与 provider 限速告警 | SLO 有数据支撑且可告警 | S |

## 6. 评测语料标准（T0.1 投喂清单）

目标 50-55 份，中英文均覆盖。放入 `eval/corpus/` 并在 `manifest.yaml` 登记类别。

| 类别 | 配额 | 关注点 |
|---|---|---|
| 数字原生 PDF，单栏 | 5 | 基线质量 |
| 双栏论文 | 10 | 阅读顺序恢复 |
| 三栏 / 复杂版式 | 2 | 阅读顺序极限 |
| 公式密集 | 5 | LaTeX 保留与检索 |
| 表格密集 / 跨页表格 | 5 | 二维结构保留 |
| 扫描 PDF，清晰 | 5 | OCR 触发 |
| 扫描 PDF，低质/倾斜 | 3 | OCR 极限与降级 |
| 混合扫描 + 文本层 | 3 | 页级质量判断 |
| 中英混排 | 4 | 多语言检索 |
| 长论文（references 很长） | 3 | 全文覆盖与 excerpt 缺陷复现 |
| DOCX（标题层级/表格/图片/公式） | 4 | element locator |
| PPTX（notes/图表文字） | 3 | slide locator |
| 异常文件（损坏/加密/超大/伪后缀） | 3 | 安全与错误路径 |

注：zip bomb 样本单独放 `eval/corpus/adversarial/`，不进常规批跑。

## 7. Release Gates（公开 beta 前门槛）

| Gate | 要求 |
|---|---|
| Direct quote exact/normalized alignment | 100%；对不齐降级 paraphrase，不带引号 |
| Citation locator resolvable | 100%；或明确标记无视觉定位能力 |
| Local-only cloud egress | 0 次（集成测试验证） |
| Silent truncation on eval corpus | 0；截断/低质量必须是可见警告或失败态 |
| Citation precision | ≥95%（beta 目标，按标注集校准） |
| Retrieval Recall@K | ≥90% 起步，中英、表格/公式/正文分开统计 |
| Task recovery | worker/API 重启后任务可恢复或明确进入可重试终态 |

## 8. 风险与开放问题

| 风险 | 应对 |
|---|---|
| docling 重依赖（torch 链）安装体积与首次模型下载 | requirements 拆 extras；文档注明磁盘/内存要求；docling-serve 容器作为替代 |
| SQLite 单写者并发（WAL 下仍限单 writer） | 个人档 job worker 与 API 同进程内协调写；Hosted 档用 PG |
| OAI 兼容 embedding 各家维度/模型不同 | index_version 编码 provider+model+dim，换模型重建索引而非混用 |
| MinerU 额度与政策变化 | 全部配置化；调用审计；额度降级有明确用户提示 |
| rerank 无标准 OAI 接口 | provider 抽象，默认关闭，由检索评测决定是否启用 |
| Vanilla JS 前端复杂度随 Evidence viewer 增长 | 保持现有栈，但 API 契约先稳定（T2.4）；框架迁移仅在成为维护瓶颈时再议 |
| OpenAlex/Crossref/S2 限速 | 统一 http client 限速与缓存（现有 search/http.py 扩展） |

## 9. 开工顺序（前 10 步）

1. T0.2 Canonical Schema（fixture 先行，不写 UI 不接向量库）
2. T0.8 配置系统重构（顺手完成，后续任务都依赖它）
3. T0.3 Parser Gateway 骨架 + legacy feature flag
4. T0.1 语料标准清单发出，所有者开始投喂（与开发并行）
5. T0.4 Docling Adapter
6. T0.6 Parse Quality Gate
7. T0.5 MinerU Cloud Adapter（token 到位后）
8. T0.7 Fallback 编排 + 隐私策略
9. T0.9 bake-off → 默认路由决策落档
10. P0 验收，进入 P1（T1.1 存储双档可提前并行）

每个任务合并条件：unit test + fixture + migration note（若涉及）+ 可观测字段。不接受"功能能跑但无法回归"的合并。
