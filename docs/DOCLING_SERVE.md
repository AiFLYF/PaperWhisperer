# docling-serve: 实测对照与 reparse 语义

本文档记录两件此前**从未对真实服务跑过**的功能的实测结果：

1. `PARSER_DOCLING_MODE=serve` —— 向 docling-serve 发送文档；
2. `POST /v1/documents/{id}/reparse` —— 重新解析，支持强制 OCR 与解析器选择。

两者此前都只有 mock 覆盖。mock 无法与它所代表的东西产生分歧，**除非它是从
真实产物生成的**——这正是 `tests/test_docling_real.py` 记录过的教训：手写 fixture
"在同一个方向上一致地错，于是每条断言都通过，而真实文档产出 96 个元素、其中
没有一个带 locator"。

本文档中的每个数字都来自本机一次真实运行，对应的可执行断言在
`tests/test_docling_serve_real.py`（默认跳过，`RUN_DOCLING_SERVE=1` 才运行），
叙述性工具是 `tools/docling_serve_smoke.py`。

---

## 1. 结论速览

| 问题 | 结论 |
|---|---|
| docling-serve 能装上并跑起来吗？ | **能。** 1.35.0，docling 2.131.0，GPU，默认端口 5001。 |
| serve 产出的文档和 in-process 一样吗？ | **一样。** 逐元素比对，4 份文档、617～3077 个字段全部一致。 |
| 改 `PARSER_DOCLING_MODE=serve` 会让解析变差吗？ | **不会**（在两侧 docling 版本相同的前提下，见 §3.2）。 |
| 强制 OCR 有效果吗？ | **没有。** 在 docling 2.131.0 上是 no-op，见 §4。 |
| 修了哪些缺陷？ | **3 个**（2 个在 serve 传输，1 个在 reparse 解析器命名），见 §5。 |
| `local_only` 下的云端 reparse 显示什么？ | 403 `privacy_mode_local_only`，且**在读取任何字节之前**。此前显示的是 409 "请安装依赖"。 |

---

## 2. 装与跑

### 2.1 隔离安装（重要）

`docling-serve` 的依赖里有两条与本项目运行环境**冲突**：

```
fastapi[standard]<0.137.0     # 本项目装的是 0.138.0
docling-slim[...]>=2.129.0    # 会牵动 docling 本身
```

直接装进 conda 环境会**降级项目正在用的 FastAPI**。因此装在独立 venv 里：

```bash
python3 -m venv --system-site-packages ~/.venvs/docling-serve
~/.venvs/docling-serve/bin/pip install docling-serve
```

`--system-site-packages` 让 torch 复用系统那份（不重复下载几个 GB），而 venv 自己的
site-packages 优先级更高，所以 docling-serve 拿到它要的 fastapi 0.136.3，项目
仍然是 0.138.0。装完核对过：

```
项目环境:  fastapi 0.138.0   docling 2.131.0   torch 2.12.1+cu130
venv 环境: fastapi 0.136.3   docling 2.131.0   docling-serve 1.35.0
```

**两侧 docling 都是 2.131.0**，这一点对 §3 的结论至关重要：变量只有传输方式，
没有解析器版本。

### 2.2 启动

```bash
HF_ENDPOINT=https://hf-mirror.com \
  ~/.venvs/docling-serve/bin/docling-serve run \
    --host 127.0.0.1 --port 5001 --no-enable-ui
```

默认端口 5001 与适配器的默认值一致，无需改任何配置。布局模型走 GPU
（`Accelerator device: 'cuda:0'`），模型已在本机 HF 缓存中。OCR 自动选到
**RapidOCR + onnxruntime（CPU）**——它拿不到 CUDA provider，会打印一串
`onnxruntime-gpu` 提示，属噪音，不影响结果（§3、§4 都是同样的噪音）。

---

## 3. serve 与 in-process 的对照

### 3.1 怎么做才算数

要回答的不是"能不能用"——mock 已经回答了这个问题。要回答的是
**"serve 是不是产出同一份文档"**。

serve 模式的存在理由，正是"不想装 torch 那条依赖链"的部署。对这些用户来说，
如果两种模式产出的文档有差异，**这个交易是看不见的**：一份稍差的解析，
照样报 success、照样是满页覆盖、照样每个元素都有 locator。

所以 `tools/docling_serve_smoke.py` 逐元素比对，而不是比对文档级汇总数字。
先前的 bake-off 记录过汇总数字相同的两次运行，恰恰因为**文档级数字会把差异平均掉**。
比对字段：元素集合（order + role + 文本前 80 字符配对）、页码、bbox 四个坐标、
表格行列结构、公式 LaTeX、标题层级。

### 3.2 结果

| 文档 | 类型 | 比对字段数 | 结果 |
|---|---|---|---|
| `digital_single_col/hinton_distillation.pdf` | 9 页 / 5 表 / 6 公式 | 617 | **完全一致** |
| `table_dense/glue_benchmark.pdf` | 表格密集 | 3077 | **完全一致** |
| `scanned_clear/report_en_tables.pdf` | 扫描件（触发 OCR） | 59 | **完全一致** |
| `scanned_clear/paper_en_basic.pdf` | 扫描件（强制 OCR 对照） | — | 文本逐字节一致 |

`hinton_distillation` 的汇总：两种模式都是 **102 个元素 / 9 页 / 33403 字符 /
locator 覆盖 1.0000**。

**结论：serve 没有产出更差的文档。** 4 份文档、覆盖文本/表格/公式/扫描 OCR
四条路径，零差异。

一个必要的限定：**这只在两侧 docling 版本相同时成立。** 若 server 端的
docling 版本不同，差异就是解析器的，不是传输的。`tools/docling_serve_smoke.py`
因此会把两侧版本都打印出来，让结论带着限定条件一起被读到。生产部署里
app 与 server 各自独立升级，版本漂移是迟早的事——届时应当重跑这个对照。

### 3.3 顺带的一个观察：serve 明显更快

| 文档 | in-process | serve |
|---|---|---|
| `hinton_distillation.pdf` | 295.4 s | 16.1 s |
| `glue_benchmark.pdf` | 294.3 s | 20.1 s |
| `report_en_tables.pdf` | 9.2 s | 4.0 s |

**但这个数字不要引用。** in-process 侧每次都在一个冷进程里重建模型、重新初始化
pipeline，而 serve 侧常驻、模型常热——这不是解析速度的差异，是模型加载的差异。
真实部署里 app 进程也是常驻的，两者会接近得多。这里记下来只是为了说明
"serve 更慢"并不成立，**不是**一个可以写进决策依据的性能结论。

---

## 4. 强制 OCR 在 docling 2.131.0 上是 no-op

reparse 界面上有一个"用 OCR 重新通读全文"的开关。它在承诺一件事发生。
在 docling 2.131.0 上它不发生。

**机制**：`PdfPipelineOptions.do_ocr` 默认为 `True`，所以不强制也在 OCR。
`force_ocr` 只是把这个默认值再显式写一遍。读回验证：

```
force_ocr=False  读回 do_ocr = True
force_ocr=True   读回 do_ocr = True
```

**实测**（4 份扫描件，`force_ocr` 关 / 开）：

| 文档 | 字符数 | 元素数 | 全文逐字节相同 |
|---|---|---|---|
| `paper_en_basic.pdf` | 2331 / 2331 | 14 / 14 | 是 |
| `paper_en_formulas.pdf` | 1292 / 1292 | 13 / 13 | 是 |
| `scan_lowq_blurry.pdf` | 270 / 270 | 5 / 5 | 是 |
| `hybrid_en_tables.pdf` | 1152 / 1152 | 43 / 43 | 是 |

这与 `PARSER_BAKE_OFF.md` §4.4 记录的 11 份扫描件结论一致，本次是在适配器层
独立复核的。

**这对运营者意味着什么。** 开关仍然可用、不会报错、解析照常成功——它只是不改变
任何东西。`force_ocr=False` **不是**"关掉 OCR"的承诺。适配器自己的注释
（`docling_adapter.py` 的 `_build_converter` docstring）已经写明了这一点，
UI 措辞没有跟上。

**是否应当移除这个开关**：一个什么都不做的开关比没有更糟，因为它让用户相信
"这份扫描件我还没 OCR 过"。建议在 docling 版本变化（某个版本把 `do_ocr`
默认关掉）之前，把 UI 上的标签从"强制 OCR"改为说明当前版本 OCR 恒开。
本仓库中已有一处 docstring 记录该行为，改 UI 措辞超出本次范围，未实施。

---

## 5. 修复的缺陷

### 缺陷 1 —— 端点路径是虚构的（serve 模式完全不可用）

`docling_adapter.py` 的 `_serve_endpoint` 一直在向 `/v1/parse/file` 发请求。
**这个端点从来不存在。** 真实端点是 `/v1/convert/file`——服务按"把一个文件
转成 N 种格式"命名，与平行的 `/v1/chunk/*` 一致。

实测：

```
POST /v1/parse/file    -> 404
POST /v1/convert/file  -> 200
```

`docling-serve 1.35.0` 的 `/openapi.json` 里共 24 条路由，转换端点只有
`/v1/convert/{file,source}` 与 `/v1/convert/*/async`，没有 `parse`。

**为什么从没被发现**：mock 的 `urlopen` 会响应任何被请求的路径，而全部
serve 测试都是 mock。404 属于不可重试的 4xx，错误最终以
`docling 解析失败：HTTP Error 404` 呈现——一条**归咎于文档**的解析错误，
而实际上文件从未到达任何解析器。运营者会去查损坏的上传文件。

**修复**：路径改为 `/v1/convert/file`；另外对 404 单独给出诊断信息，
说明这是地址配置问题、点名实际请求的 URL、并指出只应配置到主机与端口
——因为一个操作员完全可能把完整端点填进 `PARSER_DOCLING_SERVE_URL`。

### 缺陷 2 —— 响应信封被当成文档

docling-serve 的响应是**按格式分信封的**：

```json
{"status": "success",
 "document": {"filename": "...", "md_content": "...", "html_content": "...",
              "json_content": { ...真正的 DoclingDocument... }},
 "errors": [], "processing_time": 17.8}
```

`payload["document"]` **是一个 dict，但它不是文档**。原来的判断是
"`document` 是 dict 就返回它"，于是把一个没有 `texts`、没有 `pages`、
没有 `body` 的对象交给了 normalize。

**这个不会抛异常。** 它产出一份**零元素、页覆盖率"完美"的文档**——
在一张空页面上报告成功。这是所有可用失败形态里最坏的一种。

**修复**：识别 DoclingDocument 的特征键（`texts` / `tables` / `pictures` /
`pages` / `body`）而不是判断"是不是 dict"，逐层下探到 `json_content`；
`json_content` 是 JSON 字符串时也解析（异步路由与较旧版本是字符串）。
无法识别的信封**抛异常**，而不是返回空——因为调用方无法区分
"服务返回了我不认识的东西"和"这份文档没有内容"，而两者都会变成一次
成功的空解析。

### 缺陷 3 —— reparse 菜单里的解析器名与注册表不一致（云端 reparse 永远无法运行）

`routes_v1.py` 的 `REPARSE_PARSERS` 是一张手写表，而注册表在运行时由
已安装的适配器构建。没有任何东西检查两者拼写是否一致，而它们并不一致：

| | 名字 |
|---|---|
| 表格里写的（`parser_name`） | `mineru` |
| 适配器实际注册的（`MineruCloudParser.name`） | `mineru_cloud` |

查表因此落空。实测三种配置，**全部** 409 `parser_unavailable`：

| privacy_mode | consent | 结果 |
|---|---|---|
| `local_only` | false | 409 `parser_unavailable`（"请安装可选依赖"） |
| `allow_cloud` | false | 409 `parser_unavailable` |
| `allow_cloud` | true | 409 `parser_unavailable` |

也就是说，**即使隐私模式打开、用户已同意、token 有效，云端 reparse 也永远
无法运行——它只能被拒绝。**

**修复**：`parser_name` 改为 `mineru_cloud`，与注册表一致。新增一条 ungated
测试 `test_the_reparse_table_names_a_parser_the_registry_actually_registers`
守住这个契约——它不花钱、不需要服务器，纯粹比较本仓库里的两个字面量。

### 附带修复：策略拒绝应当先于可用性解析

修好缺陷 3 之后，`local_only` 下的云端 reparse 仍然报 409
`parser_unavailable`，并建议"请安装对应的可选依赖或配置其凭据"。

**这条建议是错的。** `local_only` 下云端适配器根本不注册（这正是"不携带一个
只能失败的解析器"的设计），所以目录里确实没有它——**但原因是管理员禁止出网，
安装依赖不会让这个请求变得合法。** 用户被送去装东西，而真正需要改的是
`DOCUMENT_PRIVACY_MODE`。

已改为：若所选解析器在表中标记为 `leaves_machine`，**先**问隐私门，**再**解析
可用性。`leaves_machine` 只需查表、不需要适配器，所以顺序可以换。未列出的选择
不是云端，仍然照旧返回 `unknown_parser`。

---

## 6. 云端 reparse 在 `local_only` 下显示什么

### 6.1 修复后的行为

```
POST /v1/documents/{id}/reparse   {"parser": "mineru_cloud"}

HTTP 403
{
  "error": "云端解析未启用：当前文档隐私模式为 local_only，禁止任何文档出网。"
           "如需启用请联系管理员将 DOCUMENT_PRIVACY_MODE 设置为 allow_cloud。",
  "code": "privacy_mode_local_only"
}
```

在**任何字节被读取或发送之前**返回。用户在界面上看到的是"请联系管理员改
`DOCUMENT_PRIVACY_MODE`"——正确的归因、正确的修复动作。

### 6.2 失败模式是否可区分

三种拒绝必须可区分，因为它们把人送往不同的地方：

| 情况 | 状态码 | code | 用户该做什么 |
|---|---|---|---|
| `local_only` | 403 | `privacy_mode_local_only` | 找**管理员**改设置 |
| `allow_cloud` 但未同意 | 403 | `consent_not_granted` | 自己开**同意开关** |
| 解析器名拼错 | 400 | `unknown_parser` | 开发者的 bug |

三种都已在真实适配器在场的情况下实测通过。

修复前，**前两种都塌缩成了 409 "请安装可选依赖"**——这正是"把管理员的策略
问题报成用户的依赖问题"。

### 6.3 上传腿失败会显示什么（未实测，但代码路径已确认）

`PARSER_BAKE_OFF.md` §3.3 记录了真实 token 下 MinerU 上传腿被 provider 拒绝
（Aliyun OSS `SignatureDoesNotMatch`，客户端侧已排除）。**本次没有重跑这一段**：
那需要真实凭据，会把文档出网并产生费用，超出本次授权范围。

从代码可以确认的是**失败如何呈现**：`mineru_cloud.py` 把 HTTP 错误映射为
`mineru_http_error` / `mineru_quota_exhausted` / `mineru_task_failed`，
402 与配额码（`code=1002`）另有专门文案；超时、429、402 是三个**可区分**的
结果。reparse 侧，超过质量门的失败走 `_unhappy_status` →
`_parse_failure_response`，最终是 422 `document_parse_failed`（或对应状态码）。

**因此：上传腿失败与"没有权限"在响应上可区分**——权限问题是 **403 且带
`privacy_mode_local_only` / `consent_not_granted`**，上传失败是 **4xx/5xx 且带
`mineru_*` 原因码**，两者状态码族都不重叠。但这一段是**读代码得出的，不是实测**，
且真实上传失败的文案未经端到端验证。

---

## 7. 远程 docling-serve：策略是否自洽

`resolve_serve_url` 拒绝任何非 loopback 主机，理由是远程 docling-serve 属于出网。
**这个 guard 不应被削弱**——默认拒绝远程出网是对的。

但它与一种合法部署是不自洽的：**私有网络里另一台机器上的 docling-serve**。
把解析器与 web 应用分机部署、把 GPU 集中在解析节点，是这个模式存在的全部理由；
而这种部署恰恰会被当前的 guard 拒绝。同一台机器上另一个容器用
`172.17.0.2`（docker bridge 网关）也一样被拒——这甚至不是"远程机器"，
只是不在 loopback 上。

合理的做法是**让"远程"成为需要显式声明的事实**，而不是靠主机名推断：

- 增加一个显式开关（形如 `PARSER_DOCLING_SERVE_ALLOW_REMOTE=1`），
  与 `MINERU_*` 的四道门同样处理：**默认关，开启需要知道自己在开什么**。
- 开启后仍应记录审计（`mineru_audit.jsonl` 已有同款做法），因为此时文档
  确实离开了 app 进程所在的机器。
- 拒绝时保留现有文案，但要补一句"如需连接私有网络内的 docling-serve，
  请设置 ..."，否则运营者只会看到一个针对自己合法部署的"这是出网"的断言。

**本次未实施**：这是一个策略变更而非缺陷修复，且它会打开一条当前关闭的出网路径，
应当由运营者显式决定。记录于此供决策。

---

## 8. 复现

```bash
# 1) 起服务（见 §2.2）
HF_ENDPOINT=https://hf-mirror.com \
  ~/.venvs/docling-serve/bin/docling-serve run --host 127.0.0.1 --port 5001

# 2) 对照工具：逐元素 diff
RUN_DOCLING_SERVE=1 python3 tools/docling_serve_smoke.py
RUN_DOCLING_SERVE=1 python3 tools/docling_serve_smoke.py \
    --file eval/corpus/table_dense/glue_benchmark.pdf

# 3) 断言
RUN_DOCLING_SERVE=1 pytest tests/test_docling_serve_real.py -v
```

不带 `RUN_DOCLING_SERVE=1` 时，测试文件里 12 条中 **3 条照常运行**（纯契约，
不花钱、不需要服务器，约 0.3 s），其余 9 条带原因跳过。干净 clone 不会因为
缺少 docling-serve 而失败。全套 gated 运行约 15 分钟。

工具退出码：`0` 完全一致 / `1` 有差异 / `2` 跳过（无服务或服务端点不兼容）。

### 8.1 顺带发现：一处 mock 钉死了不存在的端点

`tests/test_ingestion_docling.py` 里有两条断言把 `/v1/parse/file` 这个**虚构
端点**当成了契约：

| 行 | 测试 |
|---|---|
| 1256 | `test_the_request_targets_the_parse_endpoint_with_a_multipart_body` |
| 1474–1476 | `test_the_endpoint_path_is_appended_to_the_base_url` |

它们**钉住了缺陷本身**。因为 mock 的 `urlopen` 会响应任何被请求的路径，
"请求发到了 `/v1/parse/file`" 在 mock 下永远成立，于是这个断言在加固一条
不存在的契约，而不是在描述真实服务。修正端点后这两条会失败，需要把期望值
改成 `/v1/convert/file`（**本次未改：不在本次可改文件范围内**）。

（同文件 1270、1299 两处也出现该路径，但它们只是构造 `HTTPError` 时用的 URL
字面量，不参与断言，无需改动。）

这正是本文档反复出现的那条教训的另一个实例：**一个 mock 固定住的东西，会
变成下一个维护者眼中的"规格"。**

### 8.2 顺带发现：`infra/jobs.py` 的并发与长解析

reparse 测试最初上传 9 页论文做真实解析（每次约 2 分钟），稳定复现上传 500
`document_upload_failed`，根因是
`sqlite3.OperationalError: cannot start a transaction within a transaction`：
`jobs.py` 的后台 worker 循环（`_loop` → `tick` → `run_once` → `claim_next`）
与请求线程里的 `drain()` 抢同一条 SQLite 连接。这只在解析**足够久**、
后台循环有机会与 drain 重叠时出现。

**这不是本次两个功能的缺陷，也未修复**（`paperwhisperer/infra/**` 不在范围内）。
仅记录，因为任何"上传一个真实大文档"的端到端测试都会撞上它，看起来像产品坏了。
reparse 的三条测试改用小文本上传（legacy 适配器毫秒级返回）即可绕开——那些
测试关心的是解析器**选择**与隐私拒绝，发生在任何解析之前。


**成本**：一次对照 = 两遍完整解析，9 页论文约 5 分钟（其中约 4.5 分钟是
in-process 侧的模型冷启动）。跑整个测试文件约 10 分钟、峰值占用一份模型的显存。
GPU 共享，请保持单条串行执行。
