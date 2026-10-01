# 部署与备份恢复

本文件对应计划里的 **T2.5（Compose 档）** 与 **T2.6（备份与恢复）**，阅读顺序请先看第 1 节。

两件事在动手前先说清楚，因为它们决定了后面所有内容的取舍：

1. **个人档是默认档，零 Docker。** 这是 `DEVELOPMENT_PLAN.md` 第 2 节的决策 D2。SQLite + 本地文件 + 进程内 worker，一个用户真正会跑的就是它。**本文件第 2 节讲个人档，第 3 节才讲 Hosted 档**，顺序不是随便排的。
2. **任何东西都不能把 Docker 变成单用户部署的前置条件。** 因此 `make check` 里没有 compose 目标：`make check` 必须在一台没装 Docker 的机器上是绿的。Hosted 档的 compose 检查是独立目标（`make check-compose`），由 CI 单独跑。

---

## 1. 选哪一档

| | 个人档（默认） | Hosted 档（可选） |
|---|---|---|
| 前置条件 | Python + 本项目依赖 | Docker + Docker Compose |
| 启动 | `make serve` | `docker compose up -d --build` |
| 元数据 / 任务 / 会话 | SQLite（WAL） | PostgreSQL 16 + pgvector |
| 向量检索 | SQLite BLOB + numpy 暴力检索 | pgvector（默认精确检索） |
| 词法检索 | FTS5（构建时探测，可用则启用） | PostgreSQL FTS + pg_trgm |
| 对象文件 | 本地文件系统 | 本地文件系统（MinIO/S3 是 P4） |
| 后台任务 | 进程内 worker + 同库 job 表 | 同左 |
| 备份 | 复制数据目录（`tools/backup.py`） | `pg_dump` + 对象文件 |

**换档不是「换个环境变量就行」。** 两档共用同一套 `MetadataStore` / `VectorStore` / `LexicalRetriever` / `JobStore` 接口，业务代码不知道自己在哪一档；但**数据不会自动迁移**。迁移步骤见 [第 4 节](#4-从个人档迁移到-hosted-档)。

---

## 2. 个人档（默认，零 Docker）

### 2.1 安装与启动

```bash
python -m pip install -r requirements.txt   # 等价于 make install
make serve                                  # http://127.0.0.1:5000
```

需要 Docling 进程内解析时（重依赖，含 torch 链）：

```bash
make install-parse      # pip install -e ".[parse]"
```

不装也可以运行：解析走 legacy 路径，Docling 缺失只会让 parse extra 不可用，不会让应用起不来。

### 2.2 数据在哪里

**个人档的全部状态就是项目根目录下的三个文件夹**，没有别的地方：

| 路径 | 内容 | 丢了会怎样 |
|---|---|---|
| `context/stores/paperwhisperer.sqlite3` | **唯一的数据库**：`documents` / `chunks` / `paper_facts` / `vectors` / `lexical_chunks` / `jobs` | 全丢 |
| `context/` 其余部分 | 会话 JSON（`session_*.json`）、解析缓存（`parse_cache/`）、解析器产物（`parser_artifacts/`）、MinerU 审计日志（`mineru_audit.jsonl`） | 会话与缓存丢失，可重建；审计日志不可重建 |
| `uploads/` | 上传的原始文件 | 原始文件丢失，已解析的文档还在但无法重新解析 |
| `output/` | 导出的分析结果（Markdown 等） | 导出物丢失，可重新生成 |

`jobs` 表**和 documents 在同一个 SQLite 文件里**。这是 `paperwhisperer/infra/jobs.py` 的明确设计：T2.6 要的是「备份一个数据目录」，而 job 表和产出它的 document 行如果分属两个数据库，就既不能原子写也不能一次备份。所以「备份个人档」不需要协调两个东西。

数据库路径由 `STORAGE_SQLITE_PATH` 决定，默认 `context/stores/paperwhisperer.sqlite3`。

> **关于 `-wal` / `-shm`：** SQLite 以 WAL 模式运行时，**主文件本身不是数据库**——它只保存最后一次 checkpoint 的状态，之后每个已提交的事务都在 `-wal` 里。所以「直接 `cp` 一个正在运行的 `.sqlite3`」会得到一个**一致但过期**的副本，静默丢掉最近若干次写入。下面的备份工具专门处理这件事，见 [2.3](#23-备份)。

### 2.3 备份

```bash
make backup                                  # 写到 ./backups/
python tools/backup.py --out ~/backups       # 写到别处（推荐）
python tools/backup.py --tier hosted --dsn postgresql://...   # Hosted 档
```

**备份必须写到与被备份数据不同的磁盘上。** `make backup` 在 `BACKUP_DIR` 落在项目目录内时会打印警告，因为同盘备份恰好挡不住备份唯一要挡的那些事故（磁盘损坏、误删、勒索软件）。设置：

```bash
make backup BACKUP_DIR=/mnt/backup-disk/paperwhisperer
```

备份内容：

- `uploads/`、`output/`、`context/` 下的全部普通文件
- `context/stores/paperwhisperer.sqlite3` 的**一致性快照**（见下）
- 一个 `manifest.json`，记录每个文件的 SHA-256 与大小

`__pycache__`、`*.pyc`、`.git`、`backups` 默认排除；`--exclude PATTERN` 可叠加（按路径分量匹配，任意深度生效）。

#### SQLite 快照：为什么不能直接复制

备份时应用通常正在运行，`.sqlite3` 正被写。工具的两种做法，都会记录在 `manifest.json` 的 `files[].method` 里：

| method | 什么时候 | 一致性 | 逐字节相同 |
|---|---|---|---|
| `quiesced-copy` | 拿得到立即失败的 `BEGIN EXCLUSIVE`（应用没在写） | 是 | **是** |
| `online-backup` | 别的连接正持有写锁 | 是 | 否（SQLite 会重写页） |

`quiesced-copy` 路径做三件事：立即取 `BEGIN EXCLUSIVE`（**不等待**——备份绝不能阻塞应用）、checkpoint、然后**主文件与 `-wal` 一起复制**。最后一步是要点：WAL 模式下只复制主文件会得到一个一致但陈旧的数据库，复制这一对才是一致且当前的。

`online-backup` 路径退回 SQLite 自带的在线备份 API。它同样是一致安全的（`BEGIN EXCLUSIVE` 超时为 0 立即失败，不阻塞应用），但**结果与磁盘上的字节不逐字节相同**。清单里记下用的是哪一种，所以「本次备份是逐字节的」这句话是归档文件携带的事实，不是假设。

### 2.4 恢复

```bash
make verify-backup                       # 先只校验，不写任何东西
make restore RESTORE_DIR=/srv/paperwhisperer
python tools/restore.py --list  backups/paperwhisperer-20260930T101500Z.tar.gz
```

恢复的三条硬规则，都是有意为之：

1. **校验全部通过之前，一个字节都不写。** 清单里的 SHA-256 先全查一遍，再动目标目录。恢复 900 个文件后在第 901 个失败是最糟的结果：留下一个看起来像部署目录的半成品。
2. **目标目录非空就拒绝**（除非显式 `--force`）。规则是「里面有任何东西」而不是「有会被覆盖的文件」——真正要防的是把另一个部署的会话文件悄悄合并进来，产出一个半旧半新、且不会告诉你的目录。
3. **路径逃逸一律拒绝。** 成员名解析后必须落在目标目录内。`data/../../.ssh/authorized_keys` 是任何人拿到一份归档后第一个会试的东西，而这个检查在**创建目标目录之前**完成，所以拒绝时不会留下空目录。

恢复成功后，每个 `.sqlite3` 还会做三件事：checkpoint 把 `-wal` 合并进主文件（恢复出来的目录必须自洽，否则下一次用会跳过 `-wal` 的工具复制它，最近的写入就无声消失了）、`PRAGMA integrity_check`、以及统计各表行数。**「文件有 41 MB」不是恢复报告，「documents 有 12 行、jobs 有 3 行」才是。**

> 恢复出来的 WAL 模式数据库，被任何进程读一次就会重新生成一个空的 `-wal`/`-shm` 对——这是 SQLite 的正常行为，不是恢复没清干净。

### 2.5 定时备份

个人档的备份是纯文件操作，可以直接放进 crontab：

```bash
# 每天 03:17 备份到独立磁盘。不用 03:00：整点是所有人的整点。
17 3 * * * cd /srv/paperwhisperer && make backup BACKUP_DIR=/mnt/backup-disk/pw PYTHON=/srv/paperwhisperer/.venv/bin/python >> /var/log/pw-backup.log 2>&1
```

退出码：`0` 成功、`1` 备份没做成、`2` 参数用错。**备份命令的退出码是它对 cron 的全部承诺**，所以工具在任何一步失败时都返回非 0，而不是警告一下然后返回 0。

建议加一条**恢复演练**的定期任务——第 2.6 节的整条判据就是「没恢复过的备份只是一个假设」：

```bash
# 每周日 04:23 做一次完整演练，恢复到临时目录后删掉
23 4 * * 0 cd /srv/paperwhisperer && make restore-drill PYTHON=/srv/paperwhisperer/.venv/bin/python >> /var/log/pw-drill.log 2>&1
```

`make restore-drill` 走的是 CLI 全流程：备份 → 校验 → 恢复到 `mktemp -d` 出来的空目录 → 逐字节 `diff -r`。它比对三个运行时目录而不是仓库根目录，因为那才是备份捕获的东西；比对仓库根会让每个源文件都以「只在一侧存在」的形式报出来，教会读报告的人忽略输出。

---

## 3. Hosted 档（可选，Docker）

### 3.1 起

```bash
docker compose up -d --build --wait     # 或 make compose-up
```

一条命令起两个服务：`app` 与 `db`（`pgvector/pgvector:pg16`）。

**Redis 与 MinIO 不在默认档位里。** 它们在 `p4` profile 后面（`docker compose --profile p4 up -d`），对应计划里 P4 的「Celery+Redis 多 worker」和「S3/MinIO」。计划第 3 节的档位表里这两行都标着 (P4)，把它们放进来但不置灰会让 Hosted 档变成「app + 四个基本闲置的服务」。

`db` 的健康检查不止是 `pg_isready`：服务器在初始化脚本跑完之前就已经接受连接了，只探活意味着 app 会连上来、发现没有 `vector` 类型、然后在第一条查询上失败。所以检查里额外断言了 `pg_extension` 里有 `vector` 这一行。

### 3.2 镜像

`Dockerfile` 有两个 target：

| target | 内容 | 什么时候用 |
|---|---|---|
| `runtime`（默认） | 项目本体 | 大多数情况 |
| `parse` | 额外装 `docling>=2.0`（torch 链，数 GB） | 需要进程内 Docling，且机器扛得住 |

```bash
PAPER_IMAGE_TARGET=parse docker compose up -d --build
```

**默认 target 不含 docling，这是硬要求不是偏好。** 计划的风险表把 docling 的 torch 链列为本项目最重的安装，T0.8 拆 `[parse]` extra 的全部意义就是让个人档不付这个代价。默认镜像带上 torch 会在实践中让那个决定无法挽回——下载才是贵的部分，而每次 `docker compose up --build` 都要再付一次。

用 **build target** 而不是 Compose profile，是因为被开关的是一次**安装**而不是一个**服务**：两个 target 共享同一个文件和同一个 entrypoint，只差一个依赖。另一个选项是 docling-serve 容器（计划里对放不下 torch 的机器的推荐答案），本文件不阻止它。

### 3.3 pg-init 的位置

扩展初始化 SQL 在 **`docker/pg/init/10-paperwhisperer.sql`**，通过只读 bind mount 挂到 `/docker-entrypoint-initdb.d`。

> **这是一次相对 T1.1 的改动。** 原先它用 Compose 的 `configs:` 内联写在 `docker-compose.yml` 里。功能等价，但 `configs` 的 `content` 字段需要 **Compose ≥ 2.23**，更老的版本上这是一个直接报字段名的硬错误——把一个能跑的部署文件变成在老 Compose 上无法解析，而换来的好处（少一个文件）并不值得。文件形式在任意 Compose 版本上都能用，而且 SQL 在 diff 里可评审，mount 所有人都认识。

注意 postgres 镜像**只在 PGDATA 为空时**跑 `/docker-entrypoint-initdb.d`。所以「推倒重来」是 `docker compose down -v`，而其他所有情况（包括从别人的 dump 恢复）都靠 store 自己的 `initialize()` 里的 `CREATE EXTENSION IF NOT EXISTS` 兜底——两条语句都是幂等的。

### 3.4 备份与恢复

```bash
# 备份：pg_dump（自定义格式）+ 对象文件
python tools/backup.py --tier hosted --dsn postgresql://paperwhisperer:pw@localhost:5432/paperwhisperer
# 定期（宿主机 crontab；需要 postgresql-client）
17 3 * * * cd /srv/paperwhisperer && make backup BACKUP_DIR=/mnt/backup-disk/pw TIER=hosted

# 恢复数据库
python tools/restore.py --tier hosted --dsn postgresql://... --dump backups/paperwhisperer-hosted-....dump
# 恢复对象文件（uploads/ output/ context/，与个人档同一套工具）
python tools/restore.py --to /srv/paperwhisperer/runtime backups/paperwhisperer-hosted-....tar.gz
```

`--tier hosted` 的备份产出**两个文件**：`<name>.tar.gz`（对象文件 + 清单）和 `<name>.dump`（`pg_dump -Fc`）。`-Fc` 是自定义格式，因为 `pg_restore` 能从中单独恢复一张表——当恢复是抢救而不是重建时，这个粒度就是几分钟和一次故障的区别。它不塞进 tar 里的原因很实际：`pg_restore` 需要可 seek 的文件，把一个嵌在 gzip 流里的文件取出来意味着先解压整个归档。

`pg_dump` 用 `--no-owner --no-acl`：带着原集群角色名和授权的 dump，无法恢复到一个角色名不同的库里，除非先手工改 dump，而抢救时改 dump 是做 SQL 考古的最差时机。密码走 `PGPASSWORD` 环境变量而非命令行，否则同机任何用户的 `ps` 都能看到。

`pg_restore` 默认**不删**已有对象（`--clean` 才删）。恢复到已有 schema 的库会在第一个冲突对象上大声失败，那应该是操作员的决定，不是工具替他做的。

> **`postgresql-client` 的主版本必须与服务端一致。** 用比服务端新的 `pg_dump` 打出来的归档，服务端可能拒绝恢复：17 给 16 的库打出来的归档第一条语句是 `SET transaction_timeout = 0`，而 16 不认识这个参数。**报错写的是「unrecognized configuration parameter」，读起来像归档损坏，但归档完全正常**——恢复时很容易因此误判备份不可用。备份输出里会带上客户端版本（`pg_dump (PostgreSQL) 17.11 …`）就是为了让这个诊断一眼完成。演练过程见 [5.4](#54-演练-dhosted-档的-pg_dump--pg_restore-真实往返)。

> **compose 的命名卷也要备份。** 上传的原文件与解析缓存在 `uploads` / `output` / `context` 三个命名卷里，不在数据库里。`docker compose down -v` 会连它们一起删掉。
>
> ```bash
> # 直接从卷里取出对象文件做备份（数据库仍用上面的 pg_dump）
> docker run --rm -v paperwhisperer_uploads:/src:ro -v /mnt/backup-disk:/dst alpine \
>   tar czf /dst/uploads.tar.gz -C /src .
> ```
>
> 等 P4 的 MinIO/S3 落地后，这一步会变成一次 bucket 同步，届时「只备份数据库」就不再充分了。

---

## 4. 从个人档迁移到 Hosted 档

数据不会自动迁移，顺序也不能反（先切档再导数据，会得到一个空库）。

1. **在个人档上做一次备份**（第 2.3 节），并**当场校验**（`make verify-backup`）。
2. **起 Hosted 档**（第 3.1 节）。
3. **导出数据目录**——`pg_dump` 不适用于 SQLite，所以直接把归档解到 compose 挂载的三个卷所在的项目目录：
   ```bash
   docker compose down
   python tools/restore.py --to . backups/paperwhisperer-personal-....tar.gz
   docker compose up -d
   ```
4. 恢复后 `make restore-drill` 式的逐字节比对依然应该通过（对个人档数据而言），**并且** Hosted 档的 store 会在首次连接时自建表。

反向（Hosted → 个人）没有对应工具：PostgreSQL 的数据不能直接变成 SQLite 文件。要回退就再在 Hosted 档上做一次 `pg_dump`，或保留一份个人档的 `tools/backup.py` 归档。需要的话这可以做成一个 `tools/migrate.py`，但现在没有需求支撑它，所以没有做。

---

## 5. 恢复演练记录（T2.6 验收标准）

T2.6 的验收标准原文：**「能备份」必须验证过 restore 才算完成**。下面记录**实际执行过的**演练及其真实输出。两次都是在本仓库真实的数据目录上跑的，不是造出来的样本。

### 5.1 演练 A：本仓库的个人档真实数据

```console
$ make restore-drill PYTHON=/home/ubuntu/miniconda3/bin/python3
scratch: /tmp/tmp.GRdWN5VkYz
=== 1. backup ===
备份完成：/tmp/tmp.GRdWN5VkYz/backup/drill.tar.gz
  档位 personal；59 个文件，共 1200507 字节
  内容摘要 sha256:4b172c2a4bdd399c4608c340008515c88f68ff34867bfeee018a9a5c058a64bc
  数据库 /home/ubuntu/PaperWhisperer_project/context/stores/paperwhisperer.sqlite3（快照方式 quiesced-copy）
=== 2. verify (no writes) ===
校验通过：59 个文件，1200507 字节，内容摘要 sha256:4b172c2a4bdd399c4608c340008515c88f68ff34867bfeee018a9a5c058a64bc
=== 3. restore into an empty directory ===
已恢复 59 个文件（1200507 字节，校验 59 个） → /tmp/tmp.GRdWN5VkYz/restored；数据库 paperwhisperer.sqlite3: ok（16 张表 / 275 行）
  数据库 .../paperwhisperer.sqlite3：analysis_runs, chunks, conversations, document_versions,
    documents, lexical_chunks, lexical_fts*, messages, paper_facts, reading_queue, vectors
=== 4. byte-for-byte comparison ===
  uploads: identical
  output: identical
  context: identical
=== drill passed: every captured file came back with the bytes it went in with ===
EXIT=0
```

| 项 | 值 |
|---|---|
| 备份了什么 | 真实 `uploads/`、`output/`、`context/`（含解析缓存、解析器产物、会话 JSON、MinerU 审计日志，以及本仓库真实 store 里的文档、chunk、词法索引与会话表） |
| 文件数 / 字节数 | 59 个文件 / 1,200,507 字节 |
| 快照方式 | `quiesced-copy`（无并发写者，因此逐字节一致） |
| 恢复到哪里 | `mktemp -d` 出的空目录 |
| 怎么验证的 | ① 清单 SHA-256 全量校验（59/59）② `diff -r` 逐字节比对三个目录 ③ `PRAGMA integrity_check` = `ok` ④ 逐表行数统计 |
| 恢复后的表 | 16 张表 / 275 行，含 T2.2 的 `document_versions` / `conversations` / `messages` / `analysis_runs` / `reading_queue` |

> 文件数会随你继续解析文档而增长（上一次演练是 56 个文件，那之后又解析了几篇）。演练断言的是**逐字节相同**，不是某个固定的文件数，所以数字变了而演练依然有效。

**幂等性：** 紧接着重跑一次，内容摘要不变（`sha256:4b172c2a…`），文件数与字节数也不变。

> 这个 store 里**没有 `jobs` 表**——本 checkout 还没有提交过任何任务，`build_job_store` 尚未在它上面跑过。`jobs` 表的往返由演练 B 覆盖。

### 5.2 演练 B：带任务队列的 store

为了覆盖「恢复必须把 `jobs` 表一起带回来」这条（T2.1 明确把 job 表放在与 documents 同一个库里，正是为了让「备份一个数据目录」成立），把**演练 A 的真实 store 文件**复制到一个人档数据目录，初始化 job 表并提交两个真实任务，然后走同一套 CLI 全流程。

```console
真实 store 中的文档已带入: ['doc_06c8fab93df64786', 'doc_6f1044751a3142e1', 'doc_21e251f136ac7d86']
提交的任务: [('3c5f742a', 'completed', 'drill-0'), ('a2310a6e', 'queued', 'drill-1')]

=== B1: 备份 ===
备份完成：/tmp/drillB2/out/drillB.tar.gz
  档位 personal；2 个文件，共 963668 字节
  内容摘要 sha256:d1ae1c0f91b352be99fead627d4818ad43dd3e9d2639198f8d3b9d4e3fd00449
  数据库 /tmp/drillB2/src/context/stores/paperwhisperer.sqlite3（快照方式 quiesced-copy）
=== B2: 校验（不写入） ===
校验通过：2 个文件，963668 字节，内容摘要 sha256:d1ae1c0f91b352be99fead627d4818ad43dd3e9d2639198f8d3b9d4e3fd00449
=== B3: 恢复到空目录 ===
已恢复 2 个文件（963668 字节，校验 2 个） → /tmp/drillB2/restored；数据库 paperwhisperer.sqlite3: ok（17 张表 / 277 行）
=== B4: 逐字节比对 ===
  uploads: identical
  output: identical
  context: identical
=== B5: 用真实 store API 打开恢复后的库 ===
恢复后 documents: [('doc_06c8fab93df64786', 'paper.pdf'), ('doc_6f1044751a3142e1', 'hinton_distillation.pdf'), ('doc_21e251f136ac7d86', 'session_document')]
  job 3c5f742a kind=document_analysis state=completed idem=drill-0 events=7 seq=7
  job a2310a6e kind=document_analysis state=queued   idem=drill-1 events=1 seq=1
恢复后按状态计数: {'queued': 1, 'completed': 1}
幂等键复用: True 3c5f742a
```

| 项 | 值 |
|---|---|
| 备份了什么 | 真实 store（3 篇文档、全部 chunk、词法索引、会话/分析表）+ 2 个任务（1 个走完 7 段事件链到 `completed`，1 个停在 `queued`） |
| 恢复后 | 17 张表 / 277 行，比演练 A 多一张 `jobs`——**这正是这个演练存在的理由** |
| 三个文档的 `source_filename` | 完好（`paper.pdf`、`hinton_distillation.pdf`、`session_document`） |
| 事件链 | `completed` 任务带 7 条事件、`sequence` 连续到 7——断线重连的客户端靠它 resync，丢了事件但保住状态的任务等于坏了 |
| 幂等键 | 恢复后重新提交同一个 `idempotency_key` 返回 `deduplicated=True` 并复用原 `job_id`。**丢了幂等键是行为变化，不是数据变化**——它意味着重复点一次「分析」会再付一次云端解析的钱 |

### 5.3 演练 C：应用正在写入时

停掉应用做的演练只能证明「对一台停着的机器做备份是忠实的」。真正难的是运行中的备份，因此单独演练，并记录两种快照方式：

| 场景 | 构造 | `method` | 结果 |
|---|---|---|---|
| 有写事务未提交 | 另一个连接 `BEGIN IMMEDIATE` 后插入一行 | `online-backup` | 51 行已提交数据全在，**未提交的那行不在**（备份不允许把未提交的写提升为已提交） |
| 有读事务挡住 checkpoint | 另一个连接停在读事务里 | `quiesced-copy`，且归档内含 `-wal` | 500 行全在 |

第二行是 WAL 那个陷阱的直接演示：读事务让 `wal_checkpoint(TRUNCATE)` 落空，于是 `-wal` 必须与主文件一起走。**如果这条路径把 sidecar 丢了，备份会是「一致但陈旧」的——500 行全没、没有一处报错。** 这正是 2.2 节那条警告要防的事。

这两个场景连同演练 A、B，以及下面 5.5 节的全部拒绝路径，都固化成了 `tests/test_backup_restore.py` 里的断言——共 20 个测试，不碰网络也不碰 Docker。

### 5.4 演练 D：Hosted 档的 `pg_dump` / `pg_restore` 真实往返

这一节原本打算写「未验证」。实际在本机**跑通了**，因为 compose 起来之后有了一台真的 PostgreSQL，而 `psql` 客户端可以装进 app 镜像里（`postgresql-client` 17.11），工具本身是纯 Python + shell out 到 `pg_dump`，因此可以在容器里原样运行。

**备份。** 对着运行中的数据库跑真实工具：

```console
$ docker run --rm --user root --network paperwhisperer_default \
    -v $PWD:/src:ro -v /tmp/hosted-bk:/out paperwhisperer:smoke sh -c '
      apt-get update -qq; apt-get install -y -qq --no-install-recommends postgresql-client
      cd /src && PYTHONPATH=/src python tools/backup.py --tier hosted \
        --dsn postgresql://paperwhisperer:paperwhisperer@db:5432/paperwhisperer \
        --out /out --name hosted.tar.gz'
备份完成：/out/hosted.tar.gz
  档位 hosted；59 个文件，共 1200507 字节
  内容摘要 sha256:4b172c2a4bdd399c4608c340008515c88f68ff34867bfeee018a9a5c058a64bc
  pg_dump /out/hosted.dump（9206 字节，pg_dump (PostgreSQL) 17.11 (Debian 17.11-0+eb13u1)）
```

`pg_restore -l` 确认这是一个真的 `-Fc` 自定义格式归档，29 条 TOC：

```
; Archive created at 2026-09-30 02:54:04 UTC
;     dbname: paperwhisperer
;     TOC Entries: 29
;     Format: CUSTOM
;     Dumped from database version: 16.15 (Debian 16.15-1.pgdg12+2)
;     Dumped by pg_dump version: 17.11 (Debian 17.11-0+deb13u1)
3; 3079 16713 EXTENSION - pg_trgm
2; 3079 16385 EXTENSION - vector
218; 1259 16806 TABLE public store_chunks
217; 1259 16798 TABLE public store_documents
221; 1259 16831 TABLE public store_lexical
219; 1259 16814 TABLE public store_paper_facts
220; 1259 16821 TABLE public store_vectors
3672; 0 16806 TABLE DATA public store_chunks
3671; 0 16798 TABLE DATA public store_documents
... (数据 / 约束 / 索引各 5 组)
```

**恢复。** 停掉 app（否则 app 自己的 healthcheck 会在 DROP 与 pg_restore 之间把表建回来——这正是它第一次尝试时发生的事），删掉全部表，pg_restore，再验证：

```console
$ docker compose stop app
$ psql -c "DROP TABLE IF EXISTS store_documents, store_chunks, store_lexical, store_paper_facts, store_vectors CASCADE"
0   # 剩下的 public 表数量

$ python tools/restore.py --tier hosted --dsn postgresql://... --dump /in/hosted.dump
pg_restore 退出码 1（耗时 0.088s）
  stderr: pg_restore: error: could not execute query: ERROR:  unrecognized configuration parameter "transaction_timeout"
  pg_restore: warning: errors ignored on restore: 1
  pg_restore 未成功，请检查上面的 stderr。
```

**工具报的是失败，而它是对的**——退出码确实是 1。但数据其实全回来了，只有一条与数据无关的语句没跑过：

| 检查 | 结果 |
|---|---|
| `vector` / `pg_trgm` 扩展 | 都在 |
| public 表数量 | 5（`store_documents` / `store_chunks` / `store_lexical` / `store_paper_facts` / `store_vectors`） |
| `store_documents` / `store_chunks` / `store_lexical` | 各 3 行，与备份前一致 |
| 文档 payload | `hosted-doc-0→paper0.pdf`、`hosted-doc-1→paper1.pdf`、`hosted-doc-2→paper2.pdf` |
| app 通过真实 store API 读回 | 3 个文档 + 词法检索命中 3 个 chunk |

#### 演练 D 暴露的一个真问题：客户端比服务端新

上面那个 `transaction_timeout` 不是数据损坏，是**版本不匹配**：`pg_dump` 17 给 PostgreSQL 16 的库打出来的归档，第一条语句是 `SET transaction_timeout = 0`，而 16 不认识这个参数（它在 17 才引入）。

它危险的地方在于**症状与病因无关**：报错写的是「unrecognized configuration parameter」，读起来像归档坏了，但归档完全正常。恢复时很容易因此以为备份不可用。

所以 `tools/backup.py` 现在把客户端版本写进输出（上面那行 `pg_dump (PostgreSQL) 17.11`），让这个诊断从「一个下午」变成「一眼」。**部署要求：让 `postgresql-client` 的主版本与服务端一致**（服务端 16 就装 `postgresql-client-16`）。工具不自动装版本——那是一个需要 root 的包管理决定，不该由一个备份命令替操作员做。

### 5.5 拒绝路径的演练

| 场景 | 结果 |
|---|---|
| 目标目录非空 | 退出码 1，`恢复目标非空：…（3 项，例如 context, output, uploads）`，哨兵文件逐字节未变，未写入任何 `.sqlite3` |
| 成员被改一个字节 | 退出码 1，`备份已损坏：data/uploads/paper.pdf 的校验和不匹配（清单 …，实际 …）。拒绝恢复。`，**目标目录根本没有被创建** |
| 归档被截断一半 | 退出码 1，`无法打开备份 …：Compressed file ended before the end-of-stream marker was reached`（不是 traceback） |
| 不是归档的文件 | 退出码 1，四种格式探测全部失败并说明原因 |
| 清单格式版本不认识 | 退出码 1，消息里带版本号，不猜测结构 |
| 成员名 `data/../../tmp/x` | 退出码 1，`备份中的成员会写到目标目录之外，已拒绝`，`/tmp/x` 不存在，目标目录也不存在 |

> 截断是最可能发生的那种损坏（复制中断、磁盘满），而它**不总是**表现为 `TarError`：gzip 流被切断是从压缩层抛出的 `EOFError`。只捕获前者的工具，恰好会在操作员最需要一句话的那天打印一个 traceback。

---

## 6. 尚未验证的部分

写在这里而不是省略，因为 T2.6 的验收标准要求演练被记录，而一份只写了做过的部分的记录等于宣称了没做过的部分。

| 事项 | 状态 | 证据 |
|---|---|---|
| 个人档备份 / 校验 / 恢复 / 逐字节比对 | **已实测** | 5.1 |
| 运行中应用的一致性快照（两种 `method`） | **已实测** | 5.3 |
| `jobs` 表往返（个人档） | **已实测** | 5.2 |
| 损坏 / 截断 / 路径逃逸 / 非空目标的拒绝 | **已实测** | 5.5 |
| **`docker compose up -d --build --wait`** | **已实测，两个服务都 healthy** | 6.1 |
| **镜像构建 + 容器内 app 启动 + `/api/health`** | **已实测** | 6.1 |
| **`pg_dump -Fc` 产出与 `pg_restore` 恢复** | **已实测**（带下述一处版本不匹配） | 5.4 |
| **`PAPER_IMAGE_TARGET=parse`（docling 镜像）** | **未实测** | 见下 |
| **`make restore-drill` 在有大量数据时** | 已实测（本仓库 56–59 个文件 / ~1.2 MB） | 5.1 |
| 两档跑同一套测试（T1.1 验收标准） | **不属于本任务** | 由存储双档那条线负责 |

### 6.1 Docker

本机 Docker 29.5.3 / Compose v5.1.4，实测记录：

```console
$ make check-compose
docker compose config --quiet && docker compose config --profiles
p4

$ docker build --target runtime -t paperwhisperer:smoke .
BUILD EXIT=0                       # 378 MB 镜像 / 87.9 MB 内容

$ docker run -d --name pwsmoke -p 127.0.0.1:8099:8000 paperwhisperer:smoke
INFO:     Application startup complete.
$ curl -fsS http://127.0.0.1:8099/api/health
{"status":"ok","app":"PaperWhisperer",...,"folders":{"uploads":{"exists":true,"writable":true},...}}
$ docker inspect --format '{{.State.Health.Status}}' pwsmoke
healthy

$ make compose-smoke
--- db: the vector extension is present
pg_trgm|1.6
vector|0.8.6
--- app: the store reports the Hosted tier
{'tier': 'hosted', 'backend': 'postgres', 'vector_engine': 'matrix', 'lexical_engine': 'fts', ...}
--- smoke passed
MAKE EXIT=0
```

`vector` 与 `pg_trgm` 扩展确认由 `docker/pg/init/10-paperwhisperer.sql` 装上（容器内 `/docker-entrypoint-initdb.d/` 里能看到该文件），app 建的表落在 PostgreSQL：`store_documents` / `store_chunks` / `store_lexical` / `store_paper_facts` / `store_vectors`。

**未实测：`PAPER_IMAGE_TARGET=parse`。** 它会拉 torch 链（数 GB），本机没有在合理时间内下完，因此 docling 那个 target 只有写法、没有构建记录。

### 6.2 Hosted 档的 `pg_dump` 往返

**已实测**，见 5.4。唯一未解决的是演练中暴露的**客户端/服务端主版本不匹配**（pg_dump 17 → PG 16），已在 5.4 写明成因、影响与处置要求（装与服务端同主版本的 `postgresql-client`）。

### 6.3 与本任务无关但被它发现的问题

这些在别处，不在本任务的路径内，**未修改**，列在这里供对应负责人处理：

1. **`paperwhisperer/infra/jobs.py` 的 PostgreSQL job 表建不起来。** `_POSTGRES_SCHEMA` 用同一个 `{table}` 占位符同时充当**表名**和**索引名**：

   ```python
   "CREATE INDEX IF NOT EXISTS {table}_state ON {table} (state, created_at, job_id)",
   ```

   而 `self.table` 是 schema 全限定的（`public.store_jobs`），PostgreSQL 不允许在 `CREATE INDEX` 里限定索引名：

   ```
   ERROR:  syntax error at or near "."
   LINE 1: CREATE INDEX IF NOT EXISTS public.store_jobs_state ON public...
   ```

   同一文件里的 `postgres_backend.py` 是对的：索引名不带 schema、表名带。所以 Hosted 档目前**只有 job 表建不起来**，一调用 `build_job_store()` 就抛 `psycopg.errors.SyntaxError`。修法是拆成两个占位符（`{table}` 用于 `ON` 之后，`{index}` 不带 schema）。**注意这个 bug 在没有 PostgreSQL 的环境下不可能被发现**——这也说明 T2.1 那条「PG 实现是 SQLite 实现的翻译」在没有服务器时是未经检验的断言。

2. **`pyproject.toml` 没有 `psycopg`，所以 Hosted 档的 app 启动后无法打开自己的数据库。** `paperwhisperer/infra/stores/postgres_backend.py` 的 `load_driver()` 在两个驱动都缺失时抛出：

   ```
   RuntimeError: Hosted 档位需要 PostgreSQL 驱动：请安装 psycopg[binary]（推荐）或 psycopg2-binary
   ```

   本文件在镜像里显式装了 `psycopg[binary]>=3.11` 绕过它——**这是权宜之计，不是修复**：正确做法是在 pyproject 里加一个 `hosted` extra，并让 `docker/pg/init` 或镜像从它安装。

   这个问题特别难查，因为它有一个假象：容器是 healthy 的、端口有响应、`/api/health` 返回 200，**然后第一条读文档的请求才炸**。`/api/health` 不碰 store 也不碰数据库。

   因此 compose 的 app healthcheck 现在额外做了一次 `build_stores()` 并断言 `tier == 'hosted' and backend == 'postgres'`（`docker-compose.yml` 里有注释说明原因）。这也是 CI job 里那一步断言存在的理由——只 curl `/api/health` 的冒烟测试**证明不了这一档是通的**。

3. **`env_loader` 是一个顶层模块，不在 wheel 里。** `paperwhisperer/core/config.py` 在模块作用域 `from env_loader import load_project_env`，而 `env_loader.py` 在仓库根目录，`pyproject.toml` 的 `[tool.setuptools] packages` 只列 package。后果：**任何非 editable 安装（`pip install .`、构建 wheel、容器镜像）都会在第一次 import 时挂掉**：

   ```
   ModuleNotFoundError: No module named 'env_loader'
   ```

   本文件的镜像里显式 `COPY env_loader.py` 绕过。正确修复是在 pyproject 加 `py-modules = ["env_loader"]`。日常的 `make dev` / `pip install -e .` 察觉不到这个问题，因为 editable 安装把仓库根放进了 `sys.path`。

4. **`numpy` 不在基础依赖里，向量检索因此退化。** 容器内 `build_stores().describe()` 的实测输出：

   ```
   'vector_engine': 'python'
   'degradations': ['向量检索退化为纯 Python 暴力比对：未安装 numpy，单次查询耗时约为矩阵运算的百倍量级。']
   ```

   退化本身是被支持的、有文档的（`sqlite_backend.py` 明确说 numpy 缺席是「有记录的悬崖」而非损坏），但计划第 3 节的档位表把个人档的向量检索写作「SQLite BLOB + **numpy** 暴力检索」，所以按计划它是预期存在的。要么把 numpy 加进基础依赖，要么在部署文档里写明个人档在容器里跑时的这个代价（这是**宿主**档不受影响的，宿主档走 pgvector）。

5. **`OBSERVABILITY_LOG_LEVEL='info'` 会被拒绝。** app 容器每次启动都打一行：

   ```
   OBSERVABILITY_LOG_LEVEL='info' is not a valid value; expected one of DEBUG, INFO, WARNING, ERROR. Falling back to 'INFO'.
   ```

   小写 `'info'` 不在允许值里。容器的环境变量默认值不对——这行来自本机的 `.env`（`docker compose` 会自动读取它），所以**它可能是本机配置的问题而不是代码的问题**，但 compose 文件传入的任何 env 都值得检查一遍大小写。

6. **仓库没有 `.dockerignore`，构建上下文是整个仓库（约 104 MB）。** 每次 `docker compose up --build` 都会把 `.git`、`eval/corpus/`、`context/parse_cache/`、`uploads/` 一起发给 Docker daemon。本机上还发生过一次 `python:3.11-slim` 拉取反复 `unexpected EOF`，上下文大不是原因，但它是「本机 Docker 不可靠」时多出来的一个变量。

   建议新增 `.dockerignore`（本任务没有该路径，未创建）：

   ```
   .git
   .parity
   __pycache__/
   *.py[cod]
   .pytest_cache/
   .ruff_cache/
   # 运行时数据：镜像里由 Dockerfile 自己建空目录，绝不该从宿主机带进去
   uploads/
   output/
   context/
   # 评测语料有几十 MB 的二进制，且 compose 构建用不到
   eval/corpus/
   eval/_assets/
   ```

   > 注意 `docker-compose.yml` 的 app 服务 bind-mount 整个仓库，所以即使加了 `.dockerignore`，**镜像内仍然会有源码**（`COPY` 明确列了要哪些文件）。`.dockerignore` 砍掉的是**构建上下文**的体积，不是最终镜像的内容。




---

## 7. CI 需要加的 job

**本任务没有改 `.github/workflows/ci.yml`**（不在本任务的路径内）。以下是 T2.5「CI 验证 compose up 冒烟」需要的增量，追加到现有 workflow 的 `jobs:` 下：

```yaml
  compose-smoke:
    name: Hosted tier (docker compose smoke)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Compose v2 is preinstalled on ubuntu-latest, and a floating `latest`
      # here is how a compose file that works today silently stops parsing.
      - name: Check the compose file resolves (no pull, no build)
        run: docker compose config --quiet

      # The actual criterion: fail if the stack cannot come up. `--wait` blocks
      # on both healthchecks, so a database that never gets the `vector`
      # extension fails the job here rather than at the first user query. The
      # explicit timeout is because the first run also builds the image and
      # initialises a cold volume, and `--wait`'s default is not generous enough
      # for a cold build on a loaded runner.
      - name: Build and start the stack, waiting for health
        run: docker compose up -d --build --wait --wait-timeout 420

      - name: The vector extension is present
        run: |
          docker compose exec -T db psql -U paperwhisperer -d paperwhisperer \
            -tAc "SELECT extname FROM pg_extension WHERE extname IN ('vector','pg_trgm') ORDER BY extname" \
            | tee /dev/stderr | grep -q vector

      # The point of the tier, and the step that is easy to leave out. The app
      # being *up* proves very little: `/api/health` reports liveness and folder
      # writability and touches neither the store nor the database, so an image
      # missing its PostgreSQL driver is healthy, serves 200, and raises on the
      # first request that reads a document. `build_stores` also *raises* rather
      # than falling back to SQLite, so this cannot pass by accident.
      - name: The app resolved the Hosted store tier
        run: |
          docker compose exec -T app python -c \
            "from paperwhisperer.infra.stores import build_stores; d=build_stores().describe(); print(d); assert d['tier']=='hosted' and d['backend']=='postgres', d"

      # The app's own store created its tables in PostgreSQL, which is the only
      # evidence that the tier is actually being used rather than merely
      # configured.
      - name: The store tables are in PostgreSQL
        run: |
          docker compose exec -T db psql -U paperwhisperer -d paperwhisperer \
            -tAc "SELECT count(*) FROM pg_tables WHERE schemaname='public'" \
            | tee /dev/stderr | grep -qE '^[1-9]'

      - name: The API answers
        run: |
          for i in $(seq 1 30); do
            if curl -fsS http://127.0.0.1:8000/api/health; then exit 0; fi
            sleep 2
          done
          echo "the app never became healthy" >&2
          docker compose logs app
          exit 1

      # Always, including on failure: a CI runner is torn down anyway, but a
      # failed job with no logs is a job nobody can debug.
      - name: Logs and teardown
        if: always()
        run: |
          docker compose logs --no-color
          docker compose down -v
```

两点说明：

- **它不进 `make check` 的聚合目标。** 个人档必须保持零 Docker 可跑，所以 compose 冒烟是独立 job，不是本地聚合的一部分。
- **它不跑完整测试套件。** 「同一套测试两档均通过」是 T1.1 的验收标准，由那条线负责；这个 job 只需要在 compose 起不来时失败。
