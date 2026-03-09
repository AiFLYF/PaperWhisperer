import os
import re
import time
import uuid
import concurrent.futures
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAI
from PyPDF2 import PdfReader

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max limit
app.config['CONTEXT_FOLDER'] = 'context'

# Ensure required folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['CONTEXT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)


def resolve_api_key(explicit_key):
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    return os.getenv("OPENAI_API_KEY", "").strip()


class TextChunker:
    """鏂囨湰鍒嗗潡鍣紝鐢ㄤ簬澶勭悊瓒呴暱鏂囨湰"""
    def __init__(self, chunk_size=4000, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text):
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # 灏介噺鍦ㄥ彞瀛愮粨鏉熷鎴柇
            if end < text_length:
                last_period = chunk.rfind('。')
                last_newline = chunk.rfind('\n')
                split_pos = max(last_period, last_newline)
                
                if split_pos > self.chunk_size // 2:
                    chunk = chunk[:split_pos + 1]
                    end = start + split_pos + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - self.overlap
            
        return chunks


class DocumentLoader:
    """鏂囨。鍔犺浇鍣紝鏀寔 TXT 鍜?PDF"""
    @staticmethod
    def load_txt(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    @staticmethod
    def load_pdf(file_path):
        try:
            reader = PdfReader(file_path)
            text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            return text.strip()
        except Exception as e:
            raise ValueError(f"PDF 璇诲彇澶辫触: {str(e)}")
    
    @staticmethod
    def load(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.txt':
            return DocumentLoader.load_txt(file_path)
        elif ext == '.pdf':
            return DocumentLoader.load_pdf(file_path)
        else:
            raise ValueError(f"涓嶆敮鎸佺殑鏂囦欢鏍煎紡: {ext} (璇风‘淇濅笂浼?txt鎴?pdf鏂囦欢)")


class PaperWhisperer:
    """鏂囩尞鍒嗘瀽鏍稿績绫?"""
    def __init__(self, api_key):
        self.name = "PaperWhisperer"
        self.version = "0.5.0"
        self.api_key = resolve_api_key(api_key)
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.chunker = TextChunker(4000, 200)
        self.document_content = ""
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        ) if self.api_key else None
    
    def _call_llm(self, system_prompt, user_prompt, max_retries=3):
        if not self.client:
            raise ValueError("API key is required. Provide it in request body or set OPENAI_API_KEY.")
            
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
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"LLM API Error: {str(e)}")
                    return f"鐢熸垚澶辫触锛岃閲嶈瘯 ({str(e)})"
    
    def _generate_summary_chunk(self, content):
        system_prompt = """浣犳槸涓€涓笓涓氱殑瀛︽湳鏂囩尞鍒嗘瀽鍔╂墜锛屾搮闀挎€荤粨璁烘枃鐨勬牳蹇冭鐐广€?璇风敤涓枃鍥炲锛屼繚鎸佷笓涓氥€佺畝娲併€佸噯纭€?
銆愭瀬鍏堕噸瑕佺殑鍏紡鏍煎紡瑕佹眰銆戯細
1. 琛屽唴鍏紡蹇呴』涓斿彧鑳戒娇鐢ㄥ崟涓編鍏冪鍙峰寘瑁癸紝渚嬪锛?E = mc^2$銆傜粷瀵逛笉瑕佷娇鐢?\\( \\) 鎴?( )銆?2. 鐙珛鍧楃骇鍏紡蹇呴』涓斿彧鑳戒娇鐢ㄥ弻缇庡厓绗﹀彿鍖呰９锛屼緥濡傦細$$\\int_0^1 x^2 dx$$銆傜粷瀵逛笉瑕佷娇鐢?\\[ \\] 鎴?[ ]銆?3. 鍏紡鍐呴儴鐨勪笅鍒掔嚎锛坃锛夊拰鏄熷彿锛?锛変笉瑕佸仛浠讳綍 Markdown 杞箟锛岀洿鎺ヨ緭鍑哄師鐢熺殑 LaTeX 浠ｇ爜銆?4. 缁濆涓嶈鎶婂叕寮忔斁鍦ㄦ櫘閫氱殑浠ｇ爜鍧楋紙```锛変腑銆?"""
        
        user_prompt = f"""璇蜂粩缁嗛槄璇讳互涓嬫枃鐚唴瀹癸紝鐒跺悗锛?1. 鎻愬彇 3-5 涓牳蹇冭鐐癸紙姣忎釜瑙傜偣鐢ㄤ竴鍙ヨ瘽姒傛嫭锛?2. 鎵惧嚭 2-3 涓渶鍊煎緱寮曠敤鐨勯噾鍙?
鏂囩尞鍐呭锛?{content}

璇锋寜浠ヤ笅鏍煎紡杈撳嚭锛?## 鏍稿績瑙傜偣
1. [瑙傜偣1]
2. [瑙傜偣2]
3. [瑙傜偣3]

## 寮曠敤鐗囨
- "[寮曠敤1锛屼弗鏍奸伒瀹堝叕寮忔牸寮忚姹備繚鐣欏師鏂囧叕寮廬"
- "[寮曠敤2锛屼弗鏍奸伒瀹堝叕寮忔牸寮忚姹備繚鐣欏師鏂囧叕寮廬"
"""
        return self._call_llm(system_prompt, user_prompt)
    
    def _merge_summaries(self, summaries):
        if not summaries:
            return None
        if len(summaries) == 1:
            return summaries[0]
        
        combined = "\n\n--- 绔犺妭 ---\n\n".join(summaries)
        
        system_prompt = """浣犳槸涓€涓笓涓氱殑瀛︽湳鏂囩尞鍒嗘瀽鍔╂墜锛屾搮闀挎暣鍚堝涓枃鐚墖娈电殑鎽樿銆?璇风敤涓枃鍥炲锛屼繚鎸佷笓涓氥€佺畝娲併€佸噯纭€?
銆愭瀬鍏堕噸瑕佺殑鍏紡鏍煎紡瑕佹眰銆戯細
1. 琛屽唴鍏紡蹇呴』涓斿彧鑳戒娇鐢ㄥ崟涓編鍏冪鍙峰寘瑁癸紝渚嬪锛?E = mc^2$銆傜粷瀵逛笉瑕佷娇鐢?\\( \\) 鎴?( )銆?2. 鐙珛鍧楃骇鍏紡蹇呴』涓斿彧鑳戒娇鐢ㄥ弻缇庡厓绗﹀彿鍖呰９锛屼緥濡傦細$$\\int_0^1 x^2 dx$$銆傜粷瀵逛笉瑕佷娇鐢?\\[ \\] 鎴?[ ]銆?3. 鍏紡鍐呴儴鐨勪笅鍒掔嚎锛坃锛夊拰鏄熷彿锛?锛変笉瑕佸仛浠讳綍杞箟锛岀洿鎺ヨ緭鍑哄師鐢熺殑 LaTeX 浠ｇ爜銆?"""
        
        user_prompt = f"""浠ヤ笅鏄竴绡囬暱鏂囩尞涓嶅悓閮ㄥ垎鐨勬憳瑕佸唴瀹癸紝璇锋暣鍚堟垚涓€浠藉畬鏁淬€佽繛璐殑鎽樿锛?
{combined}

璇锋寜浠ヤ笅鏍煎紡杈撳嚭锛?## 鏍稿績瑙傜偣
[鏁村悎鍚庣殑鏍稿績瑙傜偣鍒楄〃]

## 寮曠敤鐗囨
[鏁村悎鍚庣殑寮曠敤鐗囨鍒楄〃锛屼繚鐣欏師鏂囧叕寮廬
"""
        return self._call_llm(system_prompt, user_prompt)
    
    def generate_summary(self, content):
        chunks = self.chunker.chunk_text(content)
        
        if len(chunks) == 1:
            return self._generate_summary_chunk(content)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            chunk_summaries = list(filter(None, executor.map(self._generate_summary_chunk, chunks)))
            
        if len(chunk_summaries) > 1:
            return self._merge_summaries(chunk_summaries)
        return chunk_summaries[0] if chunk_summaries else "鏃犳硶鐢熸垚鎽樿"
    
    def extract_quotes(self, content):
        system_prompt = """浣犳槸涓€涓笓涓氱殑瀛︽湳鏂囩尞鍒嗘瀽鍔╂墜锛屾搮闀夸粠鏂囩尞涓彁鍙栭噸瑕佺殑寮曠敤鐗囨銆?璇风敤涓枃鍥炲锛岀簿纭彁鍙栨枃鐚腑鐨勫師鍙ャ€?
銆愭瀬鍏堕噸瑕佺殑鍏紡鏍煎紡瑕佹眰銆戯細
1. 琛屽唴鍏紡蹇呴』涓斿彧鑳戒娇鐢ㄥ崟涓編鍏冪鍙峰寘瑁癸紝渚嬪锛?E = mc^2$銆傜粷瀵逛笉瑕佷娇鐢?\\( \\) 鎴?( )銆?2. 鐙珛鍧楃骇鍏紡蹇呴』涓斿彧鑳戒娇鐢ㄥ弻缇庡厓绗﹀彿鍖呰９锛屼緥濡傦細$$\\int_0^1 x^2 dx$$銆傜粷瀵逛笉瑕佷娇鐢?\\[ \\] 鎴?[ ]銆?3. 鍏紡鍐呴儴鐨勪笅鍒掔嚎锛坃锛夊拰鏄熷彿锛?锛変笉瑕佸仛浠讳綍杞箟锛岀洿鎺ヨ緭鍑哄師鐢熺殑 LaTeX 浠ｇ爜銆?"""
        
        user_prompt = f"""璇蜂粠浠ヤ笅鏂囩尞涓彁鍙?3-5 涓渶鍊煎緱寮曠敤鐨勯噾鍙ユ垨鏍稿績瑙傜偣锛?
{content[:15000]}  # 闄愬埗闀垮害浠ラ槻瓒匱oken

璇锋寜浠ヤ笅鏍煎紡杈撳嚭锛?## 寮曠敤鐗囨
1. "[鍘熷彞1锛屼弗鏍兼寜瑕佹眰淇濈暀鍏紡]"
2. "[鍘熷彞2锛屼弗鏍兼寜瑕佹眰淇濈暀鍏紡]"
3. "[鍘熷彞3锛屼弗鏍兼寜瑕佹眰淇濈暀鍏紡]"
"""
        return self._call_llm(system_prompt, user_prompt)
    
    def generate_mindmap(self, content):
        system_prompt = """浣犳槸涓€涓笓涓氱殑瀛︽湳鏂囩尞鍒嗘瀽鍔╂墜锛屾搮闀垮垎鏋愭枃鐚粨鏋勫苟鐢熸垚鎬濈淮瀵煎浘銆?璇风敤涓枃鍥炲銆傚鏋滄秹鍙婂叕寮忥紝璇蜂弗鏍间娇鐢?$...$ (琛屽唴) 鎴?$$...$$ (鍧楃骇) 鍖呰９銆?"""
        
        user_prompt = f"""璇蜂负浠ヤ笅鏂囩尞鐢熸垚涓€涓枃鏈牸寮忕殑鎬濈淮瀵煎浘锛?
{content[:10000]}

璇锋寜浠ヤ笅鏍煎紡杈撳嚭锛?## 鎬濈淮瀵煎浘
[浣跨敤 鈹溾攢鈹€ 鍜?鈹斺攢鈹€ 绗﹀彿鐨勫眰绾х粨鏋刔
"""
        return self._call_llm(system_prompt, user_prompt)
    
    def generate_mermaid_mindmap(self, content):
        system_prompt = """浣犳槸涓€涓笓涓氱殑瀛︽湳鏂囩尞鍒嗘瀽鍔╂墜锛屾搮闀垮垎鏋愭枃鐚粨鏋勫苟鐢熸垚 Mermaid 鏍煎紡鐨勬€濈淮瀵煎浘銆?璇风洿鎺ヨ緭鍑?Mermaid 浠ｇ爜锛屼笉瑕佹坊鍔犱换浣曡В閲娿€?
閲嶈鎻愮ず锛?1. 蹇呴』浠?"graph TD" 鎴?"graph LR" 寮€澶?2. 鑺傜偣ID鍙兘鍖呭惈瀛楁瘝銆佹暟瀛楀拰涓嬪垝绾?3. 鑺傜偣鏂囨湰鐢ㄦ柟鎷彿鍖呰９锛屽锛欰[鏍囬]
4. 涓嶈浣跨敤鐗规畩瀛楃锛屼腑鏂囧彲浠ユ甯镐娇鐢?5. 淇濇寔绠€娲侊紝涓嶈瓒呰繃20涓妭鐐?6. Mermaid鑺傜偣鏂囨湰鍐呬笉瑕佸寘鍚鏉傜殑LaTeX鍏紡锛屼互鍏嶆覆鏌撳穿婧冿紝璇风敤绠€鐭殑涓枃姒傛嫭銆?"""
        
        user_prompt = f"""璇蜂负浠ヤ笅鏂囩尞鐢熸垚 Mermaid 鏍煎紡鐨勬€濈淮瀵煎浘浠ｇ爜銆?
鏂囩尞鍐呭锛?{content[:4000]}

璇峰彧杈撳嚭 Mermaid 浠ｇ爜锛屾牸寮忓涓嬶細
graph TD
    A[璁烘枃鏍囬]
    A --> B[绔犺妭1]
    A --> C[绔犺妭2]
    B --> B1[灏忚妭1]
    B --> B2[灏忚妭2]
"""
        result = self._call_llm(system_prompt, user_prompt)
        if result:
            # 瀵绘壘鐪熸鐨?Mermaid 浠ｇ爜璧峰琛岋紝杩囨护鎺夊ぇ妯″瀷杈撳嚭鐨勫紑澶村簾璇?            lines = result.strip().split('\n')
            start_idx = -1
            valid_prefixes = ("graph ", "mindmap", "flowchart ", "pie", "sequenceDiagram", "stateDiagram", "classDiagram")
            
            for i, line in enumerate(lines):
                if any(line.strip().startswith(prefix) for prefix in valid_prefixes):
                    start_idx = i
                    break
            
            if start_idx != -1:
                result = '\n'.join(lines[start_idx:]).strip()
            else:
                # 鏋佺鎯呭喌锛氭鍒欏洖閫€鎻愬彇
                match = re.search(r'```(?:mermaid)?\s*\n(.*?)\n```', result, re.DOTALL | re.IGNORECASE)
                if match:
                    result = match.group(1).strip()
                else:
                    # 榛樿娣诲姞graph TD鍓嶇紑
                    result = "graph TD\n" + result
                
            # 纭繚浠ｇ爜鏈夋晥
            if not any(result.startswith(prefix) for prefix in valid_prefixes):
                result = "graph TD\n" + result
            return result
        return None
    
    def generate_evaluation(self, content):
        system_prompt = """浣犳槸涓€涓笓涓氱殑瀛︽湳璁烘枃璇勫涓撳锛屾搮闀垮璁烘枃杩涜鎵瑰垽鎬ц瘎浠枫€?璇风敤涓枃鍥炲锛屽寘鎷鏂囩殑浼樼偣銆佸眬闄愭€с€佸巻鍙插湴浣嶅拰璐＄尞銆?
銆愭瀬鍏堕噸瑕佺殑鍏紡鏍煎紡瑕佹眰銆戯細
1. 琛屽唴鍏紡蹇呴』涓斿彧鑳戒娇鐢ㄥ崟涓編鍏冪鍙峰寘瑁癸紝渚嬪锛?E = mc^2$銆傜粷瀵逛笉瑕佷娇鐢?\\( \\) 鎴?( )銆?2. 鐙珛鍧楃骇鍏紡蹇呴』涓斿彧鑳戒娇鐢ㄥ弻缇庡厓绗﹀彿鍖呰９锛屼緥濡傦細$$\\int_0^1 x^2 dx$$銆傜粷瀵逛笉瑕佷娇鐢?\\[ \\] 鎴?[ ]銆?"""
        
        user_prompt = f"""璇峰浠ヤ笅鏂囩尞杩涜鎬荤粨鎬ц瘎浠凤紝鍖呮嫭锛?
1. **璁烘枃鐨勪富瑕佽础鐚?*锛氳繖绡囪鏂囩殑鏍稿績鍒涙柊鐐规槸浠€涔堬紵
2. **鍘嗗彶鍦颁綅**锛氬湪鐩稿叧棰嗗煙鐨勯噸瑕佹€у浣曪紵鏄惁鏄鍩烘€у伐浣滐紵
3. **涓昏浼樼偣**锛氳鏂囩殑浼樺娍鍜屽垱鏂颁箣澶?4. **灞€闄愭€?*锛氳鏂囧瓨鍦ㄧ殑闂鎴栧悗缁伐浣滄寚鍑虹殑缂虹偣
5. **鍊煎緱瀛︿範鐨勫湴鏂?*锛氬璇昏€呮湁浠€涔堝惎鍙戯紵

鏂囩尞鍐呭锛?{content[:15000]}

璇锋寜浠ヤ笅鏍煎紡杈撳嚭锛?## 馃搳 璁烘枃璇勪环

### 馃幆 涓昏璐＄尞
[璇勪环鍐呭]

### 馃搱 鍘嗗彶鍦颁綅
[璇勪环鍐呭]

### 鉁?涓昏浼樼偣
- 浼樼偣1
- 浼樼偣2

### 鈿狅笍 灞€闄愭€?- 灞€闄愭€?
- 灞€闄愭€?

### 馃挕 鍊煎緱瀛︿範鐨勫湴鏂?- 瀛︿範鐐?
- 瀛︿範鐐?
"""
        return self._call_llm(system_prompt, user_prompt)
    
    def answer_question(self, question):
        if not self.document_content:
            return "没有文档内容，请先上传文档进行分析。"
        system_prompt = (
            "你是专业学术助手。请只基于给定文档内容回答问题，"
            "若文档中没有答案，请明确说明。"
        )
        user_prompt = (
            f"文档内容:\n{self.document_content[:12000]}\n\n"
            f"用户问题:\n{question}\n\n"
            "请给出简洁、准确的中文回答。"
        )
        return self._call_llm(system_prompt, user_prompt)
    
    def analyze(self, file_path, generate_mermaid=True, generate_evaluation=True):
        """鏍稿績鍒嗘瀽娴佺▼锛堝凡浼樺寲涓哄苟鍙戞墽琛岋級"""
        content = DocumentLoader.load(file_path)
        self.document_content = content
        
        result = {'char_count': len(content)}
        
        # 使用线程池并发执行大模型请求，减少整体等待时间
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_summary = executor.submit(self.generate_summary, content)
            future_quotes = executor.submit(self.extract_quotes, content)
            future_mindmap = executor.submit(self.generate_mindmap, content)
            
            future_mermaid = executor.submit(self.generate_mermaid_mindmap, content) if generate_mermaid else None
            future_eval = executor.submit(self.generate_evaluation, content) if generate_evaluation else None
            
            # 鑾峰彇缁撴灉
            result['summary'] = future_summary.result()
            result['quotes'] = future_quotes.result()
            result['mindmap'] = future_mindmap.result()
            
            if future_mermaid:
                result['mermaid'] = future_mermaid.result()
            if future_eval:
                result['evaluation'] = future_eval.result()
                
        return result

# ================= 璺敱鎺у埗 =================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Please upload a file.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '璇烽€夋嫨鏂囦欢'}), 400
    
    # 淇鏂囦欢瀹夊叏闂
    # 淇鏂囦欢瀹夊叏闂锛氭彁鍙栧師濮嬪悗缂€
    original_ext = os.path.splitext(file.filename)[1]
    original_filename = secure_filename(file.filename)
    
    # 妫€鏌?secure_filename 鏄惁鎶婁腑鏂囧悕鐮村潖鎴愪簡绫讳技 "txt" 杩欐牱娌℃湁鍚庣紑鐨勫瓧绗︿覆
    # 濡傛灉澶勭悊鍚庣殑鍚嶅瓧涓虹┖锛屾垨鑰呬笉鍖呭惈鍘熸潵鐨勫悗缂€鍚嶏紝鍒欑洿鎺ヤ娇鐢?UUID 閲嶆柊鍛藉悕
    if not original_filename or not original_filename.endswith(original_ext):
        original_filename = f"file_{uuid.uuid4().hex}{original_ext}"
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    file.save(file_path)
    
    generate_mermaid = request.form.get('generate_mermaid', 'true').lower() == 'true'
    generate_evaluation = request.form.get('generate_evaluation', 'true').lower() == 'true'
    
    # 浣跨敤棰勮鐨?API KEY
    api_key = resolve_api_key(request.form.get('api_key', ''))
    if not api_key:
        return jsonify({'error': 'API key is required. Provide api_key or set OPENAI_API_KEY.'}), 400
    
    try:
        whisperer = PaperWhisperer(api_key)
        result = whisperer.analyze(file_path, generate_mermaid, generate_evaluation)
        
        # 鎻愬彇骞惰繃婊?session_id 闃叉璺緞绌胯秺婕忔礊
        raw_session_id = request.form.get('session_id', f'session_{uuid.uuid4().hex}')
        safe_session_id = secure_filename(raw_session_id)
        
        # 淇濆瓨涓婁笅鏂囦緵杩介棶浣跨敤
        context_file = os.path.join(app.config['CONTEXT_FOLDER'], f"{safe_session_id}.txt")
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(whisperer.document_content)
        
        # 鐢熸垚 Markdown 鍒嗘瀽鎶ュ憡
        base_name = os.path.splitext(original_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_name}_analysis_{timestamp}.md")
        
        md_content = f"""# 馃搫 PaperWhisperer 鍒嗘瀽鎶ュ憡

> 鐢熸垚鏃堕棿: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 婧愭枃浠? {original_filename}

---

## 馃摑 AI 鎽樿

{result.get('summary', '')}

---

## 馃挕 寮曠敤鐗囨

{result.get('quotes', '')}

---

## 馃 鎬濈淮瀵煎浘

{result.get('mindmap', '')}

---

"""

        if generate_evaluation:
            md_content += f"## 馃搳 璁烘枃璇勪环\n\n{result.get('evaluation', '')}\n\n---\n"
            
        md_content += f"## 馃搳 鍏冧俊鎭痋n\n- 鐗堟湰: {whisperer.version}\n- 瀛楃鏁? {result['char_count']}\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        result['output_file'] = output_file
        result['session_id'] = safe_session_id
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # 娓呯悊涓婁紶鐨勪复鏃舵枃浠讹紙鍙€夛級
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.route('/api/ask', methods=['POST'])
def ask_question():
    # 增加 or {} 防御空数据崩溃
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    raw_session_id = data.get('session_id', 'default')
    
    if not question:
        return jsonify({'error': 'Please enter a question.'}), 400
        
    safe_session_id = secure_filename(raw_session_id)
    
    api_key = resolve_api_key(data.get('api_key', ''))
    if not api_key:
        return jsonify({'error': 'API key is required. Provide api_key or set OPENAI_API_KEY.'}), 400
    
    try:
        context_file = os.path.join(app.config['CONTEXT_FOLDER'], f"{safe_session_id}.txt")
        if not os.path.exists(context_file):
            return jsonify({'error': 'Session expired or context not found. Please upload and analyze the file again.'}), 400
        
        with open(context_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        whisperer = PaperWhisperer(api_key)
        whisperer.document_content = content
        
        answer = whisperer.answer_question(question)
        
        return jsonify({'answer': answer})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


