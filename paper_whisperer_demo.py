import os
import time
import json
from datetime import datetime
from openai import OpenAI
from PyPDF2 import PdfReader


class TextChunker:
    def __init__(self, chunk_size=4000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text):
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                last_period = chunk.rfind('。')
                last_newline = chunk.rfind('\n')
                split_pos = max(last_period, last_newline)
                if split_pos > start + self.chunk_size // 2:
                    chunk = chunk[:split_pos + 1]
                    end = start + split_pos + 1
            
            chunks.append(chunk.strip())
            start = end - self.overlap
        
        return [c for c in chunks if c]


class DocumentLoader:
    @staticmethod
    def load_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_pdf(file_path):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    
    @staticmethod
    def load(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            return DocumentLoader.load_txt(file_path)
        elif ext == '.pdf':
            return DocumentLoader.load_pdf(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")


class PaperWhisperer:
    def __init__(self, use_api=True, chunk_size=4000, overlap=200):
        self.name = "PaperWhisperer"
        self.version = "0.3.0"
        self.use_api = use_api
        self.chunker = TextChunker(chunk_size, overlap)
        
        if self.use_api:
            self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
            self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
            
            if self.api_key:
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
                print(f"✅ API 已配置: {self.model}")
            else:
                print("⚠️  未设置 OPENAI_API_KEY，将使用模拟模式")
                self.use_api = False
    
    def _call_llm(self, system_prompt, user_prompt, max_retries=3):
        if not self.use_api:
            return None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️  API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print("❌ 达到最大重试次数")
                    return None
    
    def _generate_summary_chunk(self, content):
        system_prompt = """你是一个专业的学术文献分析助手，擅长总结论文的核心观点。
请用中文回复，保持专业、简洁、准确。"""
        
        user_prompt = f"""请仔细阅读以下文献内容，然后：
1. 提取 1-3 个核心观点（每个观点用一句话概括）
2. 找出 1-2 个最值得引用的金句

文献内容：
{content}

请按以下格式输出：
## 核心观点
1. [观点1]
2. [观点2]
3. [观点3]

## 引用片段
- "[引用1]"
- "[引用2]"
"""
        
        return self._call_llm(system_prompt, user_prompt)
    
    def _merge_summaries(self, summaries):
        if not summaries or len(summaries) == 0:
            return None
        
        if len(summaries) == 1:
            return summaries[0]
        
        combined = "\n\n--- 章节 ---\n\n".join(summaries)
        
        system_prompt = """你是一个专业的学术文献分析助手，擅长整合多个文献片段的摘要。
请用中文回复，保持专业、简洁、准确。"""
        
        user_prompt = f"""以下是一篇长文献不同部分的摘要内容，请整合成一份完整、连贯的摘要：

{combined}

请按以下格式输出：
## 核心观点
[整合后的核心观点列表]

## 引用片段
[整合后的引用片段列表]
"""
        
        return self._call_llm(system_prompt, user_prompt)
    
    def generate_summary(self, content, use_chunking=True):
        print("\n🤖 正在生成 AI 摘要...")
        
        chunks = self.chunker.chunk_text(content)
        
        if len(chunks) == 1 or not self.use_api:
            if self.use_api:
                result = self._generate_summary_chunk(content)
                if result:
                    print("✅ 摘要生成完成！\n")
                    print(result)
                    return result
            
            return self._fallback_summary(content)
        
        print(f"📑 检测到长文本，自动分块处理（共 {len(chunks)} 个块）...")
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  📝 处理第 {i+1}/{len(chunks)} 个分块...")
            summary = self._generate_summary_chunk(chunk)
            if summary:
                chunk_summaries.append(summary)
        
        if len(chunk_summaries) > 1:
            print("🔄 合并分块摘要...")
            final_summary = self._merge_summaries(chunk_summaries)
        else:
            final_summary = chunk_summaries[0] if chunk_summaries else None
        
        if final_summary:
            print("✅ 摘要生成完成！\n")
            print(final_summary)
            return final_summary
        
        return self._fallback_summary(content)
    
    def _fallback_summary(self, content):
        summaries = [
            "核心观点 1：本文系统综述了深度学习在 NLP 领域的最新进展，包括 Transformer 架构的演进。",
            "核心观点 2：研究表明，预训练语言模型在多项下游任务中取得了 SOTA 效果，但仍存在可解释性问题。",
            "核心观点 3：作者提出了一种新的轻量化模型，在保持性能的同时将参数量减少了 60%。"
        ]
        
        print("✅ 摘要生成完成（模拟模式）！\n")
        for s in summaries:
            print(f"📝 {s}")
        
        return "\n".join(summaries)
    
    def extract_quotes(self, content):
        print("\n💡 正在提取可引用片段...")
        
        if self.use_api:
            system_prompt = """你是一个专业的学术文献分析助手，擅长从文献中提取重要的引用片段。
请用中文回复，精确提取文献中的原句。"""
            
            user_prompt = f"""请从以下文献中提取 3-5 个最值得引用的金句或核心观点：

{content}

请按以下格式输出：
## 引用片段
1. "[原句1]"
2. "[原句2]"
3. "[原句3]"
"""
            
            result = self._call_llm(system_prompt, user_prompt)
            if result:
                print("✅ 引用片段提取完成！\n")
                print(result)
                return result
        
        quotes = [
            "\"Transformer 架构的出现彻底改变了 NLP 领域...\"",
            "\"实验结果表明，我们的方法在 GLUE 基准上提升了 3.2%\"",
            "\"未来研究方向包括多模态融合和小样本学习\""
        ]
        
        print("✅ 引用片段提取完成（模拟模式）！\n")
        for quote in quotes:
            print(f"📌 {quote}")
        
        return "\n".join(quotes)
    
    def generate_mindmap(self, content):
        print("\n🧠 正在生成思维导图...")
        
        if self.use_api:
            system_prompt = """你是一个专业的学术文献分析助手，擅长分析文献结构并生成思维导图。
请用中文回复，使用层级缩进格式。"""
            
            user_prompt = f"""请为以下文献生成一个文本格式的思维导图，使用层级缩进展示结构：

{content}

请按以下格式输出：
## 思维导图
[使用 ├── 和 └── 符号的层级结构]
"""
            
            result = self._call_llm(system_prompt, user_prompt)
            if result:
                print("✅ 思维导图生成完成！\n")
                print(result)
                return result
        
        mindmap = """
📚 深度学习在 NLP 中的应用
├── 1. 研究背景
│   └── NLP 是人工智能的重要分支
├── 2. 相关工作
│   ├── 传统方法
│   └── 深度学习方法
├── 3. 方法
│   ├── 3.1 模型架构
│   └── 3.2 训练策略
├── 4. 实验
│   ├── 实验设置
│   └── 实验结果
└── 5. 结论
    └── 未来研究方向
"""
        
        print("✅ 思维导图生成完成（模拟模式）！\n")
        print(mindmap)
        return mindmap
    
    def save_results(self, file_path, summary, quotes, mindmap):
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{base_name}_analysis_{timestamp}.md")
        
        content = f"""# 📄 {self.name} 分析报告

> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 源文件: {file_path}

---

## 📝 AI 摘要

{summary}

---

## 💡 引用片段

{quotes}

---

## 🧠 思维导图

{mindmap}

---

## 📊 元信息

- 版本: {self.version}
- API 模式: {"启用" if self.use_api else "模拟"}
- 模型: {self.model if self.use_api else "N/A"}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n💾 分析结果已保存至: {output_file}")
        return output_file
    
    def load_document(self, file_path):
        print(f"📄 正在加载文档: {file_path}...")
        
        if not os.path.exists(file_path):
            print("⚠️  文件不存在，使用示例内容")
            return self._get_sample_content()
        
        try:
            content = DocumentLoader.load(file_path)
            print(f"✅ 文档加载成功！({len(content)} 字符)")
            return content
        except Exception as e:
            print(f"⚠️  文档加载失败: {e}")
            print("使用示例内容")
            return self._get_sample_content()
    
    def _get_sample_content(self):
        return """
        深度学习在自然语言处理中的应用
        
        摘要：本文系统综述了深度学习在 NLP 领域的最新进展，包括 Transformer 架构的演进。
        研究表明，预训练语言模型在多项下游任务中取得了 SOTA 效果，但仍存在可解释性问题。
        作者提出了一种新的轻量化模型，在保持性能的同时将参数量减少了 60%。
        
        1. 引言
        自然语言处理（NLP）是人工智能领域的重要分支。近年来，深度学习技术的快速发展为 NLP 
        带来了革命性的变化。从早期的词袋模型到如今的 Transformer 架构，NLP 任务的性能有了
        显著提升。本文系统综述了深度学习在 NLP 领域的最新进展，旨在为研究人员提供全面的
        技术overview。
        
        2. 相关工作
        早期的 NLP 方法主要依赖规则和统计模型，如 n-gram 语言模型和隐马尔可夫模型。
        随着深度学习的兴起，循环神经网络（RNN）及其变体 LSTM 和 GRU 成为 NLP 任务的主流。
        近年来，注意力机制和 Transformer 架构的提出进一步推动了 NLP 的发展。
        
        3. 方法
        3.1 模型架构
        我们提出的模型基于 Transformer 架构，但进行了以下改进：采用稀疏注意力机制减少计算
        复杂度；引入位置编码的改进版本；使用层级归一化提升训练稳定性。
        
        3.2 训练策略
        采用两阶段训练方法：预训练和微调。预训练阶段使用大规模无标注数据进行自监督学习；
        微调阶段针对特定任务进行有监督训练。
        
        4. 实验
        实验结果表明，我们的方法在 GLUE 基准上提升了 3.2%，在 SuperGLUE 上提升了 2.8%。
        同时，模型的推理速度相比基线方法提升了 40%，参数量减少了 60%。
        
        5. 结论
        本文提出了一种高效的深度学习模型，在多个 NLP 基准上取得了 SOTA 效果。未来研究
        方向包括多模态融合、小样本学习和模型可解释性研究。Transformer 架构的出现彻底改变
        了 NLP 领域，为通用人工智能的实现奠定了基础。
        """
    
    def run(self, file_path, save_output=True):
        print("=" * 60)
        print(f"📄 {self.name} v{self.version} - 文献私语者")
        print("=" * 60)
        
        content = self.load_document(file_path)
        summary = self.generate_summary(content)
        quotes = self.extract_quotes(content)
        mindmap = self.generate_mindmap(content)
        
        if save_output:
            self.save_results(file_path, summary, quotes, mindmap)
        
        print("\n" + "=" * 60)
        print("🎉 分析完成！感谢使用 PaperWhisperer！")
        print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "sample.txt"
    
    app = PaperWhisperer(use_api=True)
    app.run(file_path)
