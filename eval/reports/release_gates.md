# Release gates report / 发布门禁报告

- gates_version: `1.0`
- generated_at: `2026-10-12T17:00:15+00:00`
- corpus: 52/52 在场，`corpus_version` `sha256:0dfe319761a7cc9d0…`

| Gate | 要求 | 阈值 | 实测 | 状态 | 依据 |
|---|---|---|---|---|---|
| `direct_quote_alignment` | 100%；对不齐降级 paraphrase，不带引号 | `>= 100%` | 1.0000 (3654/3654) | **PASS / 通过** | 逐字引语 1827/1827 对齐；伪造引语 1827/1827 被拒且降级为 paraphrase；重排引语 124/137 对齐（13 条因换行连字符降级，属预期） |
| `citation_locator_resolvable` | 100%；或明确标记无视觉定位能力 | `>= 100%` | 0.9231 (48/52) | **FAIL / 失败** | 文档级 locator 覆盖 44/52 份（每份需全部正文元素可解析）；元素级 locator 覆盖 1237/1237（1.0000，仅供对照，不参与判定）；应有 bbox 的 1158 个元素中，未声明缺口 0 处；零正文元素文档 8 份，按未解析计入分母（paper_en_basic.pdf、paper_en_formulas.pdf、paper_en_twocol.pdf、paper_zh_cn.pdf、report_en_tables.pdf）：产不出可引用内容即视为不可解析；解析文档 52 份，解析器 legacy |
| `local_only_cloud_egress` | 0 次（集成测试验证） | `== 0` | 未测量 | **SKIP / 跳过** | 指定的 node id 没有收集到任何用例；未能收集的模块：tests/test_ingestion_security.py（该平台无法导入，非产品行为）。输出尾部：>     import resource E   ModuleNotFoundError: No module named 'resource' =========================== short test summary info =========================== ERROR tests/test_ingestion_security.py 1 error in 3.86s ERROR: found no collectors for C:\Users\FLYF\Desktop\PaperWhisperer_project\PaperWhisperer_project\tests\test_ingestion_security.py::TestHarnessAssertions::test_a_compromised_model_is_caught |
| `silent_truncation` | 0；截断/低质量必须是可见警告或失败态 | `== 0` | 0 violations / 52 checked | **PASS / 通过** | 解析 52 份文档（解析器 legacy），9 份存在未被解释的静默截断；其中 9 份有可见 warning，静默截断 0 份；已按已登记缺陷忽略的误报规则：page_coverage、text_density（3 份文档；修复 paperwhisperer/ingestion/quality.py:TextDensityRule.evaluate、paperwhisperer/ingestion/quality.py:_page_of 后本门禁会重新变红，届时请删除 eval/gates.py:KNOWN_DEFECTS 中的对应条目） |
| `citation_precision` | ≥95%（beta 目标，按标注集校准） | `>= 95%` | 未测量 | **BLOCK / 阻塞（缺标注集）** | C:\Users\FLYF\Desktop\PaperWhisperer_project\PaperWhisperer_project\eval\labels\retrieval_labels.yaml 不存在。门禁已实现（标注格式、加载校验、打分与分类别分解均在本模块内并有单测），但没有标注集就没有可计算的真值，因此不产生任何数字。绝不用系统自身输出充当真值——那样测的是系统在自己眼中的正确率，不是正确率。 |
| `retrieval_recall_at_k` | ≥90% 起步，中英、表格/公式/正文分开统计 | `>= 90%` | 未测量 | **BLOCK / 阻塞（缺标注集）** | C:\Users\FLYF\Desktop\PaperWhisperer_project\PaperWhisperer_project\eval\labels\retrieval_labels.yaml 不存在。门禁已实现（标注格式、加载校验、打分与分类别分解均在本模块内并有单测），但没有标注集就没有可计算的真值，因此不产生任何数字。绝不用系统自身输出充当真值——那样测的是系统在自己眼中的正确率，不是正确率。 |
| `task_recovery` | worker/API 重启后任务可恢复或明确进入可重试终态 | `>= 100%` | 1.0000 (5/5) | **PASS / 通过** | 5/5 个任务恢复测试通过；已跑档位：personal(SQLite) |

通过 3 / 失败 1 / 跳过 1 / 阻塞 2
