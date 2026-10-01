# 故障模式登记册（T4.4）

本文件对应计划里的 **T4.4 负载与故障注入**，验收标准是
「**故障场景均有明确可观察行为，无静默丢失**」。

阅读顺序：第 1 节是这张表，第 2 节是五处**已确认的静默丢失**，第 3 节说明
什么算「可观察」，第 4 节是负载探针，第 5 节是怎么复现的，第 6 节说明本任务
与已有测试的分界。

## 0. 一句话结论

T4.4 一行写了六个名字，展开成 **9 个可复现场景**：worker 重启有三个不同的死亡
点；SSE 断线要分成「客户端回来」和「客户端离开」两件事，它们的失败原因完全
不同；另加一个「恢复有没有发出声音」。

**6 个场景的失败完全可观察，3 个有已定位的静默丢失，合计 5 处。** 全部由真实
故障驱动：真的 `os._exit`、真的 HTTP 429/402/永不就绪、真的 `BEGIN EXCLUSIVE`
写锁、真的六个操作系统进程抢一个幂等键。全部跑完约 5 秒，
`tools/fault_inject.py` 在存在静默时返回**非零退出码**。

按性质分：

| # | 是什么 | 位置 |
|---|---|---|
| #1 | **代码缺陷**：claim 与首个 step 之间反复崩溃 → 任务永不进入终态 | `infra/jobs.py` `_recover_stale` |
| #2 | **代码缺陷**：死任务与活任务的状态投影逐字段相同，用户分辨不出 | `infra/jobs.py` `Job.to_state_dict` |
| #3 | **outcome 被丢弃**：三种 MinerU 失败塌缩成同一个 `status` | `ingestion/orchestrator.py` + `analysis/orchestrator.py` |
| #4 | **尚未存在的东西**：任务系统没有任何 HTTP 接口，重同步机制没接线 | 属 T2.3/T2.4，本任务不拥有 |
| #5 | **整族指标没有调用点**，且启动路径上没有 `configure()` | `core/observability.py` |

---

## 1. 场景登记册

「已验证」一栏指 `tests/test_fault_injection.py` 里有对应用例；所有场景的
可复现实现是 `tools/fault_inject.py`，测试直接断言它的输出，因此 CI 里跑的
和运维手上跑的是同一份代码。

| # | 场景 | 触发 | 预期可观察行为 | 实测 | 状态 |
|---|---|---|---|---|---|
| 1 | **worker 重启**（步骤内崩溃） | worker 进程在 stage 中途 `os._exit(9)` | 任务重新排队，trail 记录死掉的 worker，下一个 worker 跑到 `completed`，`attempts=2` | 完全符合 | ✅ 已验证 |
| 1b | **worker 重启**（claim 与首个 step 之间崩溃） | 领取后立即死亡，任务仍是 `queued` 但带租约 | 租约被释放、任务可再领取 | 租约确实被释放，**但重试预算不被检查 → 无限循环** | ❌ **静默 #1** |
| 1c | **worker 重启**（持有远期租约崩溃） | 租约还剩 300s 时 worker 死亡 | 到期前不可领取；到期后可恢复 | 到期前不动作（符合设计），**但这 300 秒里用户无法分辨死活** | ❌ **静默 #2** |
| 1d | **两个 worker 竞争同一任务** | 6 个独立进程同时 `claim_next` | 恰好一个领取成功，`attempts` 只 +1 | 12 次领取 0 次重复，`attempts` 全为 1 | ✅ 已验证 |
| 2 | **DB 连接耗尽**（并发写） | 16 线程 × 20 次提交打同一个 SQLite 文件 | 无写入丢失，提交数 = 读回数 | 320 次提交、0 失败、表内行数一致 | ✅ 已验证 |
| 2b | **DB 连接耗尽**（写锁被独占） | 另一条连接持有 `BEGIN EXCLUSIVE` | 写入阻塞后在 `busy_timeout` 后**抛出明确错误**，绝不静默丢弃 | 抛出 `sqlite3.OperationalError: database is locked`，锁释放后存储立即可用 | ✅ 已验证 |
| 2c | **连接计数** | 40 个线程同时各开一条连接 | — | **40 条连接在线程结束后仍保留，只有 `close()` 才释放** | ⚠️ 记录在案 |
| 3 | **MinerU 429** | 真实 HTTP 服务返回 429 + `Retry-After: 1` | 退避重试、遵守 `Retry-After`、**不得报成额度问题** | 2 次调用后退避 `[1.0]`，`code=mineru_http_error`，文案说 429 不说额度 | ✅ 已验证 |
| 4 | **MinerU 额度耗尽** | 真实 HTTP 402，或 HTTP 200 + `code=1002` | **不重试**（只 1 次调用）、独立 code、提示充值 | 402 与 1002 都只调 1 次，`code=mineru_quota_exhausted` | ✅ 适配器层已验证 |
| 4b | **额度在用户侧是否可分辨** | 同上 | 界面能按「充值 / 重试 / 稍后再看」分支 | **三种失败塌缩成同一个 `status="failed"`**，用户可见载荷不含该字段 | ❌ **静默 #3** |
| 5 | **MinerU 超时** | 服务永远返回 `pending` | `poll_batch` 返回 `None` 而不是抛异常；能捡回就降级 `partial` | 约 1s 后返回 `None`；补查捡回时降级 partial，捡不回时报 `timed_out` | ✅ 已验证 |
| 6 | **SSE 断线重连** | 客户端读到第 1 条后断线，期间任务走完 7 次状态变更 | 凭 `since_sequence` 补齐**错过的全部**（含终态），无丢失无重复 | 补发 10 条，以终态结束，两页序号拼接 == 完整 trail | ✅ 底层机制已验证 |
| 6b | **SSE 断线（客户端离开）** | `aclose()` 掉正在推流的流 | cleanup 必定执行；正在进行的分析不被中断、仍落库 | cleanup 执行 1 次，producer 跑完并落库 | ✅ 已验证 |
| 6c | **SSE 重连的线上可达性** | 浏览器刷新 / 断线 | 客户端能凭游标重连 | **`JobSubscription` 没有任何 HTTP 路由暴露；SSE 帧无 `id:`；前端无重连** | ❌ **静默 #4** |
| 7 | **用户取消** | 三种时机 | 如实记录，不谎称取消 | 排队中→`cancelled` 且不跑任何步骤；运行中→下一边界停；最后一步→`completed` + `cancel_requested=True`；对已完成任务再取消→`accepted=false` | ✅ 已验证 |
| 8 | **重复提交** | 6 个独立进程并发提交同一幂等键 | 只有一个任务、只被领取一次、**重复响应可区分** | 1 个 job_id、1 行、`attempts=1`、第二次领取返回 `None`；重复响应带 `deduplicated=true` + 中文提示 | ✅ 已验证 |
| 9 | **恢复有没有发出声音** | 一次真实的崩溃→恢复→重跑→完成 | `jobs.recovered_total` 等指标 +1 | **三个计数器全部纹丝不动（0/0/0）** | ❌ **静默 #5** |

---

## 2. 已确认的静默丢失（头条结论）

五处，按影响从大到小。每处给出触发条件、实测证据、修法方向，
以及它与 release gate / 退出标准的关系。

### 静默丢失 #1：claim 与首个 step 之间反复崩溃 → 任务永不进入终态

`paperwhisperer/infra/jobs.py` 的 `_JobStoreBase._recover_stale` 有两条分支：

- **运行态**（`parsing` 等）：`attempts >= max_attempts` 时进 `failed` +
  `retryable`，并结束。正确。
- **排队态**（被领取但还没进任何 step）：只释放租约，**完全不检查
  `attempts` 与 `max_attempts`**；而 `claim_next` 也从不拒绝超预算的任务。

于是：一个 worker 反复死在「领取之后、进入第一个 step 之前」这个窗口里，
任务就被**无限次重新领取**。实测 8 轮后：

```
attempts=9   max_attempts=2   state=queued   terminal=False
```

release gate 说的两个出口（「可恢复」**或**「明确进入可重试终态」）在这里
**都不成立**。任务没丢——每轮都写了 `lease_expired:<worker>` 事件——但它也
永远不会结束，而用户界面上它就是一个永远停在「等待 worker 领取」的任务。

**修法**：让排队态分支复用运行态的预算检查，`attempts >= max_attempts` 时
走同一条 `failed` + `retryable` 终态。

### 静默丢失 #2：worker 死了，用户看到的是「正在解析」

`Job.to_state_dict()` 投影给订阅者的字段是：

```
attempts, cancel_requested, created_at, job_id, kind, max_attempts,
message, partial, retryable, sequence, state, terminal, updated_at
```

**没有 `lease_owner`，没有 `lease_expires_at`**（这两项在 `Job.to_dict()` 里
有，只是不进状态投影）。实测把一个被死 worker 持有的任务和一个正被活 worker
执行的任务做字段级对比，**除了 `job_id` 逐字段完全相同**：

```
{'job_id': [...], 'state': ['parsing', 'parsing'], 'message': ['正在解析文档。', '正在解析文档。']}
```

租约是这套机制里**唯一**能区分「活着」和「死了」的东西，而它恰恰是对订阅者
隐藏的那一个。于是在租约到期之前（默认 300 秒），一个孤儿任务和一个正在解析
的任务在用户眼里是同一个东西。这正是验收标准里点名的情形：*「用户无法区分
慢请求和任务已丢」*。

**修法**：把租约投影进 `to_state_dict()`，或者投影一个派生量
（`stalled: bool` / `lease_owner` / `lease_expires_at` 都够）。

### 静默丢失 #3：三种 MinerU 失败在用户侧不可分辨

适配器层分得很清楚——实测三种失败各自有独立的 `code`、独立的调用次数、独立的
中文文案。但穿过编排层的 `_CloudParserBridge._normalize` 之后：

| 适配器 outcome | 用户可见 `status` | `error_message` | `diagnostics` |
|---|---|---|---|
| `quota_exhausted` | `failed` | 「MinerU 额度不足（HTTP 402）…」 | `[{"cloud_outcome": "quota_exhausted"}]` |
| `failed`（429） | `failed` | 「请求过于频繁（HTTP 429）…」 | `[{"cloud_outcome": "failed"}]` |
| `timed_out` | `failed` | 「解析在 N 秒内未完成…」 | `[{"cloud_outcome": "timed_out"}]` |
| `blocked` | `failed` | 隐私门控文案 | `[{"cloud_outcome": "blocked"}]` |

而 `build_parse_provenance` 构造的用户可见 `parse` 溯源块**既不含
`cloud_outcome`，也不含 `error_message`**：

```
['cache_hit','cloud_attempted','fallback_used','media_type','parsed_by',
 'status','summary','touched_network','used_cloud','warnings']
```

界面因此无法按「需要充值」/「可重试」/「稍后再看」分支处理，只能显示一段中文
散文——用户要靠读懂「额度不足」四个字才知道发生了什么。

顺带一处文档与实现不符：`mineru_cloud.py` 的 `_http_error_to_mineru_error`
注释写着「编排层据此决定提示用户充值而不是静默换 parser，而它只认 `code`」。
编排层并不读 `code`，也没有任何分支。

**修法**：把 `cloud_outcome` 提升为 `ParseResult` / `ParseOutcome` 的顶层
字段，并由 `build_parse_provenance` 透出。

### 静默丢失 #4：「SSE 断线重连」在端到端上不存在

`JobSubscription` 是一个**正确且完整**的重同步原语：它带 `since_sequence`
游标、可中断、只推状态不带内容、能在 worker 死掉之后继续工作
（它轮询的是表，不是 worker 的内存）。这些都由本任务的测试证明了。

问题在于**没有任何东西把它接到 HTTP 上**：

- `paperwhisperer/app.py` 只注册了 5 个 router：health / documents / qa /
  evidence / papers。**没有 jobs router**。
- 全仓 `paperwhisperer/api/` 没有任何一处 import `JobSubscription`、
  `JobStore`、`JobWorker` 或 `job_id`。
- 两个真实 SSE 端点（`POST /api/analyze/stream`、`POST /api/ask/stream`）
  是**一次性内容流**，`build_sse_event` 只发 `event:` 和 `data:`，
  **不发 `id:`**，所以线上根本不存在可恢复的游标。
- 前端 `templates/static/js/api.js` 的 `readSseStream` 读一次就结束，
  没有重连、没有退避、不解析 `id:`；`state.currentSessionId` 只存在内存里，
  刷新即丢。

结论：Phase 2 退出标准「浏览器刷新/SSE 断开不丢任务」**在端到端上尚未成立**。
分析本身不会丢（断开后 producer 仍跑完并落库），但**用户拿不到结果**。

**修法**：这是 T2.4 `/v1/jobs/{id}` 那一档的活——把 `JobSubscription` 接到
一个 SSE 路由上（带 `id:` 或 `since_sequence`），并让前端在 `onerror` 时带游标
重连。本任务不拥有那些文件，故只记录不修改。

### 静默丢失 #5：恢复发生了，但没有任何指标动过

`paperwhisperer/core/observability.py` 声明了完整的 `jobs` 指标族：

```
jobs.submitted_total   jobs.transitions_total   jobs.terminal_total
jobs.recovered_total   jobs.cancelled_total     jobs.queue_depth
jobs.duration_ms
```

其中 `jobs_recovered` 的文档字符串写着「应该是这一族里最响的数字」。实测一次
完整的「崩溃 → 恢复 → 重跑 → 完成」之后：

```
before: {'recovered': 0, 'transitions': 0, 'terminal': 0}
after:  {'recovered': 0, 'transitions': 0, 'terminal': 0}
```

两个原因：

1. **没有调用点。** `record_job_transition` / `record_job_submitted` /
   `record_job_queue` / `record_job_duration` 在 production 代码里**一次都没被
   调用**——全仓只有 `tests/test_observability.py` 调用它们。
   `paperwhisperer/infra/jobs.py` 完全不 import observability。
2. **默认关闭且从不开启。** `observability.enabled` 默认 `False`，而
   `web_app.py` 和 `paperwhisperer/app.py` 里**没有任何 `configure()` 调用**。

所以看板上会永远显示 0 次恢复，而这与「从未发生过恢复」在数据上无法区分。
恢复本身是可靠的（第 1 行已验证），只是没人会知道它发生过。

**修法**：在 `JobWorker` 的 transition / recovery / terminal 处写入对应指标，
并在应用 lifespan 里 `configure()`。本任务不拥有那些文件，故只记录不修改。

---

## 3. 什么算「可观察」

一个场景只有同时满足三条才算通过，三条都写进了 `tools/fault_inject.py` 的
`ScenarioResult.observable`：

1. **故障真的被触发。** `reached` 检查它确实走到了那一步。看起来失败、
   实际没发生的场景，会让它之后每一条断言都因为错误的原因通过——这是最危险的
   一种绿。
2. **结果有人能看到。** job 行、SSE 事件、API 响应、错误文案。写进一个没人
   打开的日志文件叫记录，不叫可观察。
3. **结果与健康状态可区分。** 一个和快请求长得一模一样的降级，不是可观察的
   降级，是隐形降级。

因此 `tools/fault_inject.py` 在有任何场景不通过时**返回非零退出码**——
一个永远返回 0 的工具等于学会了对一切说「没事」。

---

## 4. 负载探针

`python tools/load_probe.py`（纯标准库，无 k6 / 无 Locust——两者都装不了，
而且对这个部署也不对：k6 和 Locust 驱动 HTTP 端点，而个人档真正的问题不是
「HTTP 层多快」，而是**单写者 SQLite 在哪里跟不上**）。

### 这些数字**是什么**

在**回环、单进程、临时目录里的真实 SQLite 文件**上，用真实 `JobStore` /
`JobWorker` / WAL / `busy_timeout` 压测。测的是**争用**，不是容量。

### 这些数字**不**是什么

- **不含网络。** 每一毫秒都是磁盘和锁。真实磁盘比 tmpfs 慢，NVMe 比机械盘快。
- **不含模型。** 解析和分析是真实管线里长两个数量级的长杆，且由第三方配额
  决定。这里推出的「任务/秒」对「文档/分钟」没有任何说明力。
- **不是容量计划。** 单进程一个数字，Hosted 档是多进程 + PostgreSQL，
  与本表无关。

### 实测（本机，回环，单进程）

（本机实测，回环、单进程、每线程 40 次操作；`--jobs 40`，档位 1…32）

| 场景 | 并发 1 吞吐 | 并发 32 吞吐 | p50（c=1 → c=32） | p95（c=1 → c=32） | 说明 |
|---|---|---|---|---|---|
| `submit` | 6518 次/秒 | 1085 次/秒 | 0.13 → 0.61 ms | 0.15 → **80.5** ms | 写串行化；并发不增吞吐，只增排队 |
| `read` | 8339 次/秒 | 4761 次/秒 | 0.07 → 3.12 ms | 0.12 → 18.9 ms | WAL 生效：读不排在写后面，但 CPU 仍会争 |
| `claim` | 2536 次/秒 | 608 次/秒 | 0.40 → 0.61 ms | 0.52 → 3.4 ms（p99 **747** ms） | 每次 claim 都是 `BEGIN IMMEDIATE` |
| `run` | 323 任务/秒 | 73 任务/秒 | 2.58 → 303 ms | 5.0 → **1120** ms | 整条管线走到终态 |
| `contended-write` | — | — | 252 ms | 252 ms | 写锁被独占：**2520 次全部抛 `OperationalError`，0 次静默丢弃** |

**天花板在哪：并发 1 附近。** 这是单写者存储的必然结果，也是值得直说的结论：
在个人档上，**提高并发只会让每个人的 p95/p99 变差，不会让吞吐变好**。p50 几乎
不动，p95/p99 涨一个数量级——争用不是均匀变慢，是一个倒霉的写者排在整批
写者后面。

**连接数**（这才是 SQLite 档「连接耗尽」的真实形态）：40 个一次性线程 →
线程全部结束后仍保留 40 条连接，只有 `close()` 才释放。也就是说连接数随
「**曾经跑过的线程总数**」增长，不随「当前存活线程数」增长。线程池复用线程
则连接有界；每个工作单元起一个线程则无界。

**顺带发现的一处冷启动竞态**（观察到的，罕见，未写成测试）：把一个**全新**的
SQLite 文件转成 WAL 需要一个短暂的排他锁，而 SQLite 对 journal mode 变更
**不调用 busy handler**；`SqliteDatabase._configure` 又把
`PRAGMA busy_timeout` 放在 `PRAGMA journal_mode=WAL` **之后**。于是在一个进程
内第一次有多个线程同时建连接时，有观察到一个
`sqlite3.OperationalError: database is locked` 从 `_configure` 直接抛出——此时
`busy_timeout` 还没设上。该竞态在压测中偶发（一次 40 线程的突发里出现一次），
单独构造无法稳定复现，因此**没有写成测试**：一个会随机失败的测试只会训练人
忽略它。探针因此先用主线程预热一次文件，再测稳态连接数，并把「开连接失败的
线程数」作为一个测量项输出而不是让它崩掉。

---

## 5. 怎么复现

```bash
# 全部故障场景（约 5 秒），有静默丢失时退出码非 0
python tools/fault_inject.py --all

# 单个场景
python tools/fault_inject.py worker-restart
python tools/fault_inject.py mineru-failures --json

# 负载探针
python tools/load_probe.py --concurrency 1,4,8,16,32 --jobs 40

# 测试
python -m pytest tests/test_fault_injection.py -q
```

故障是真的，不是 mock 的：

- worker 崩溃 = 真实子进程 `os._exit(9)`，跑真实的 `JobWorker` 打真实
  SQLite 文件；
- MinerU 失败 = 真实 `ThreadingHTTPServer` 在真实 socket 上返回 429/402/
  永不就绪；
- DB 耗尽 = 第二条真实连接持 `BEGIN EXCLUSIVE`；
- 竞争与重复提交 = 6 个真实操作系统进程；
- 重连 = 真实断开后凭真实游标恢复。

**用 mock 掉故障的故障注入测试，测的是那个 mock。**

## 6. 与已有测试的关系

`tests/test_jobs.py::TestRecoveryFromAKilledWorker` 已经覆盖了「步骤内崩溃」
这一个死亡点（做法是 `os._exit` + 真实子进程，本任务沿用并扩展）。本任务补上
的是另外两个死亡点（claim 与首个 step 之间、持有远期租约），以及把
`TestIdempotencyUnderLoad` 的**线程**竞争升级成**进程**竞争。
`tests/test_ingestion_mineru.py::TestRetryAndBackoff` 已覆盖适配器层的
429/402/业务码；本任务补上的是**穿过编排层之后**用户侧是否还能分辨。

已有的、这里只做了证明而没有改动的正确行为：worker 崩溃后的恢复与重试预算、
跨进程幂等去重、协作式取消的三种时机及其诚实性、`JobSubscription` 的重同步
语义。
