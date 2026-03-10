# 📚 PaperWhisperer

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![License](https://img.shields.io/badge/License-TIM-lightgrey) ![AI](https://img.shields.io/badge/AI--Powered-OpenAI-blueviolet) ![Version](https://img.shields.io/badge/version-0.6.1-green)

一个面向论文/文献阅读场景的 AI 助手。  
上传 `.txt`、`.pdf`、`.docx` 或 `.pptx` 后，可自动生成结构化分析结果，并支持基于文档上下文继续追问。

## ✨ 项目能做什么
- 自动生成文档摘要（核心观点）
- 提取可引用片段
- 生成文本结构图（层级思维导图）
- 可选生成 Mermaid 可视化图
- 可选输出批判性评价
- 支持“上传后追问”问答，并保留最近多轮会话上下文
- 支持导出完整会话结果（Markdown + Mermaid SVG）
- 自动保存 Markdown 分析报告到 `output/`

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
- HTML/CSS/JavaScript（KaTeX + Mermaid + marked）

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
1. 打开网页并填写 API Key（或使用环境变量）
2. 上传 `.txt` / `.pdf` / `.docx` / `.pptx` 文档
3. 点击 `Analyze Document`
4. 查看摘要、引用、结构图、评价
5. 在 “Ask Questions” 区域继续追问（会自动带入最近多轮问答上下文）
6. 点击“下载完整结果”导出当前分析、问答记录与 Mermaid 资源

## 🔌 API 接口说明

### `POST /api/analyze`
上传并分析文档。

表单参数：
- `file`（必填）：`.txt`、`.pdf`、`.docx` 或 `.pptx`
- `api_key`（可选）：本次请求临时 API Key
- `generate_mermaid`（可选）：`true/false`
- `generate_evaluation`（可选）：`true/false`
- `session_id`（可选）：用于后续问答

说明：若不传 `api_key`，服务端会读取 `OPENAI_API_KEY`。

### `POST /api/ask`
基于已保存的文档上下文进行追问。

说明：服务端会保存 `context/<session_id>.json`，其中包含该次分析对应文档的正文、分析结果摘要与问答历史。追问时只会使用当前 `session_id` 这一个文档会话里的 Ask Questions 历史，不会跨文档、不会做全局混用；同时会优先携带文档节选，并附加最近若干轮问答历史，而不是每轮重复塞入整篇文档。

请求体示例：
```json
{
  "question": "这篇文献最重要的贡献是什么？",
  "session_id": "session_123",
  "api_key": "optional"
}
```

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
- 代码中不再硬编码真实 API Key
- `uploads/`、`context/`、`output/` 默认不提交到 Git
- 临时上传文件在分析结束后会清理
- Web 会话上下文保存在 `context/*.json`，便于多轮追问与完整结果导出
- 支持“请求级 key”与“环境变量 key”两种模式

## ⚠️ 当前已知限制
- 代码中部分中文提示词仍有历史编码痕迹（不影响主流程）
- 分析质量与所选模型、提示词质量相关
- PDF 提取效果取决于原文档文本层质量
- DOCX/PPTX 主要提取文本层内容，复杂排版、图表和图片中的文字可能提取不完整

## 📝 更新日志

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
