# 📄 PaperWhisperer - 文献私语者

> 让文献开口说话，10 页 PDF 变 3 句话。

---

## 🎯 项目简介

**PaperWhisperer** 是一个 AI 驱动的文献阅读助手，帮助学生和研究者快速理解文献内容、提取核心观点、生成结构化笔记。

### ✨ 核心功能
- 📝 **AI 摘要生成** - 自动生成 3 句话精简摘要
- 💡 **引用片段提取** - 智能标记可直接引用的金句
- 🧠 **思维导图生成** - 可视化展示文献结构
- 📄 **多格式支持** - 支持 PDF、Word、TXT 等格式

---

## 📦 项目结构

```
PaperWhisperer/
├── README.md              # 项目说明文档
├── ONE_PAGER.md           # 产品概念单页
├── MENTOR_PERSONA.md      # 导师招募画像
├── UI_PROTOTYPE.html      # 产品 UI 原型
├── paper_whisperer_demo.py # 代码 Demo
├── read_docx.py           # 文档读取工具
└── AI项目创意征集令.docx  # 原始需求文档
```

---

## 🚀 快速开始

### 环境要求
- Python 3.7+
- OpenAI SDK

安装依赖：
```bash
pip install openai
```

### 配置 API Key
设置环境变量（Windows PowerShell）：
```bash
$env:SILICONFLOW_API_KEY="your-api-key-here"
```

或者使用命令行临时设置：
```bash
set SILICONFLOW_API_KEY=your-api-key-here
```

### 运行代码 Demo
```bash
python paper_whisperer_demo.py
```

如果没有设置 API Key，程序会自动使用模拟模式运行。

### 查看 UI 原型
直接在浏览器中打开 `UI_PROTOTYPE.html` 文件即可。

---

## 📋 交付物清单

根据 **AI 项目创意征集令** 的要求，本项目包含以下交付物：

1. ✅ **产品概念单页** (`ONE_PAGER.md`)
   - 产品名称、Slogan
   - 用户痛点分析
   - 解决方案与核心功能
   - AI 参与度说明

2. ✅ **可视化原型** (`UI_PROTOTYPE.html`)
   - AI 生成的产品概念图/UI 图
   - 交互式原型展示

3. ✅ **代码 Demo** (`paper_whisperer_demo.py`)
   - 可运行的命令行版本
   - 演示核心功能流程

4. ✅ **导师招募画像** (`MENTOR_PERSONA.md`)
   - 详细的导师需求描述
   - 为什么需要这样的导师

---

## 🤖 AI 参与度

| 环节 | AI 参与程度 |
|------|------------|
| 创意发散 | 100% |
| 产品命名 | 90% |
| 功能设计 | 80% |
| UI 设计 | 100% |
| 代码框架 | 80% |
| 文档撰写 | 70% |

---

## 📝 GitHub 提交指南

详细的提交步骤请参考 [GITHUB_GUIDE.md](./GITHUB_GUIDE.md)

---

## 👥 团队

- **创意与设计**: AI + 人类
- **代码开发**: AI + 人类
- **文档撰写**: AI + 人类

---

## 📄 许可证

本项目为课程作业项目。

---

**让我们一起把 PaperWhisperer 变成同学们写作业/论文的必备工具吧！🚀**
