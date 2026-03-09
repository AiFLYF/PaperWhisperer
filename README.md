# 📄 PaperWhisperer · 文献私语者

> 让文献开口说话 — 10 页 PDF，3 句话读懂。

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python) ![License](https://img.shields.io/badge/License-Course%20Project-lightgrey) ![AI](https://img.shields.io/badge/AI--Powered-OpenAI-blueviolet)

------

## 🎯 项目简介

**PaperWhisperer** 是一个 AI 驱动的文献阅读助手，帮助学生和研究者快速理解文献内容、提取核心观点、生成结构化笔记。告别逐字精读，让 AI 成为你的私人文献助理。

------

## ✨ 核心功能

| 功能           | 描述                                  |
| -------------- | ------------------------------------- |
| 📝 AI 摘要生成  | 自动生成 3 句话精简摘要，抓住文献核心 |
| 💡 引用片段提取 | 智能标记可直接引用的金句              |
| 🧠 思维导图生成 | 可视化展示文献结构与逻辑关系          |
| 📄 多格式支持   | 支持 PDF、Word、TXT 等主流格式        |

------

## 📦 项目结构

```
PaperWhisperer/
├── README.md                  # 项目说明文档（本文件）
├── ONE_PAGER.md               # 产品概念单页
├── MENTOR_PERSONA.md          # 导师招募画像
├── UI_PROTOTYPE.html          # 产品交互式 UI 原型
├── paper_whisperer_demo.py    # 核心功能 Demo
├── read_docx.py               # 文档读取工具
└── AI项目创意征集令.docx      # 原始需求文档
```

------

## 🚀 快速开始

### 环境要求

- Python 3.10+
- OpenAI SDK

### 安装依赖

```bash
pip install openai
```

### 配置 API Key

**Windows PowerShell：**

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Windows CMD：**

```cmd
set OPENAI_API_KEY=your-api-key-here
```

> 💡 若未配置 API Key，程序将自动切换至**模拟模式**运行，无需真实密钥即可体验功能。

### 运行 Demo

```bash
python paper_whisperer_demo.py
```

### 查看 UI 原型

直接用浏览器打开 `UI_PROTOTYPE.html` 即可，无需任何额外配置。

------

## 📋 交付物清单

| #    | 文件                      | 内容                                                     |
| ---- | ------------------------- | -------------------------------------------------------- |
| ✅    | `ONE_PAGER.md`            | 产品名称与 Slogan、用户痛点分析、核心功能说明、AI 参与度 |
| ✅    | `UI_PROTOTYPE.html`       | AI 辅助设计的交互式 UI 原型                              |
| ✅    | `paper_whisperer_demo.py` | 可运行的命令行 Demo，演示核心功能流程                    |
| ✅    | `MENTOR_PERSONA.md`       | 导师需求画像与招募理由                                   |

------

## 🤖 AI 参与度

| 环节     | AI 参与程度       |
| -------- | ----------------- |
| 创意发散 | ████████░░░░ 60%  |
| 产品命名 | ███████████░ 90%  |
| 功能设计 | ██████████░░ 80%  |
| UI 设计  | ████████████ 100% |
| 代码框架 | ██████████░░ 80%  |
| 文档撰写 | █████████░░░ 70%  |

------

## 👥 团队

- **创意与设计**：AI + 人类协作
- **代码开发**：AI + 人类协作
- **文档撰写**：AI + 人类协作

------

## 📄 许可证

本项目为课程作业项目，仅供学习交流使用。

------

*让我们一起把 PaperWhisperer 变成同学们写论文的必备工具吧！* 🚀
