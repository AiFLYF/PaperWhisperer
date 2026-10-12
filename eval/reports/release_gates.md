# Release gates report / 发布门禁报告

- gates_version: `1.0`
- generated_at: `2026-10-12T10:50:01+00:00`
- corpus: 52/52 在场，`corpus_version` `sha256:0dfe319761a7cc9d0…`

| Gate | 要求 | 阈值 | 实测 | 状态 | 依据 |
|---|---|---|---|---|---|
| `silent_truncation` | 0；截断/低质量必须是可见警告或失败态 | `== 0` | 1 violations / 52 checked | **FAIL / 失败** | 解析 52 份文档（解析器 legacy），9 份存在未被解释的静默截断；其中 8 份有可见 warning，静默截断 1 份；已按已登记缺陷忽略的误报规则：page_coverage、text_density（3 份文档；修复 paperwhisperer/ingestion/quality.py:TextDensityRule.evaluate、paperwhisperer/ingestion/quality.py:_page_of 后本门禁会重新变红，届时请删除 eval/gates.py:KNOWN_DEFECTS 中的对应条目） |

通过 0 / 失败 1 / 跳过 0 / 阻塞 0
