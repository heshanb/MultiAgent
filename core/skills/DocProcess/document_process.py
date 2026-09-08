import os
import uuid
import re
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from settings.Define import Params, PathConfig
from settings.logger_manager import get_logger

logger = get_logger(__name__)

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False


def _ensure_ocr():
    """懒加载 OCR 相关依赖（pytesseract / PIL / pdf2image），并配置 Tesseract 路径。"""
    _state = getattr(_ensure_ocr, "_state", None)
    if _state is not None:
        return _state
    try:
        import pytesseract
        from PIL import Image
        import pdf2image

        import platform as _platform
        if _platform.system() == 'Windows':
            possible_tesseract_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Tesseract-OCR\tesseract.exe',
            ]
            for path in possible_tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    print(f"设置 Tesseract 路径: {path}")
                    break

        _ensure_ocr._state = (True, pytesseract, Image, pdf2image)
    except ImportError:
        _ensure_ocr._state = (False, None, None, None)
    return _ensure_ocr._state

class DocumentProcessor:
    """文档处理工具类，支持多种文档格式的读取、修改和保存"""
    SUPPORTED_FORMATS = Params.DOC_SUPPORTED_FORMATS | {'.md', '.pdf'}
    _BASE_DIR = PathConfig.BASE_DIR
    UPLOAD_DIR = PathConfig.UPLOADS_DIR
    OUTPUT_DIR = PathConfig.OUTPUTS_DIR
    
    def __init__(self):
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> tuple:
        """保存上传的文件并返回 (文件路径, 保存的文件名)"""
        saved_filename = filename
        file_path = self.UPLOAD_DIR / saved_filename
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return str(file_path), saved_filename
    
    async def read_document(self, file_path: str) -> str:
        """读取文档内容并返回文本（统一转换为MD格式）"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.docx':
            async for chunk_text in self._read_docx_with_paragraphs(file_path):
                yield chunk_text
        elif ext == '.txt':
            async for chunk_text in self._read_txt(file_path):
                yield chunk_text
        if ext in ['.xlsx', '.xls']:
            async for chunk_text in self._read_excel(file_path):
                yield chunk_text
        elif ext in ['.pptx', '.ppt']:
            async for chunk_text in self._read_pptx(file_path):
                yield chunk_text
        elif ext in ['.html', '.htm']:
            async for chunk_text in self._read_html(file_path):
                yield chunk_text
        elif ext in ['.py', '.java', '.cpp', '.c', '.h', '.js', '.ts', '.jsx', '.tsx', '.css', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.cs', '.sh', '.bash', '.sql', '.yaml', '.yml', '.json', '.xml', '.md', '.ini', '.cfg', '.conf', '.toml']:
            async for chunk_text in self._read_txt(file_path):
                yield chunk_text
        elif ext == '.doc':
            async for chunk_text in self._read_doc(file_path):
                yield chunk_text
        elif ext == '.pdf':
            async for chunk_text in self._read_pdf(file_path):
                yield chunk_text
        elif ext == '.md':
            async for chunk_text in self._read_md(file_path):
                yield chunk_text
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    async def process_document_modification(self, file_path: str, modification_request: str,
                                     conversation_history: List[Dict[str, str]], llm) -> Tuple[str, Optional[str]]:
        """处理文档修改/整理/格式转换请求（异步）"""
        ext = Path(file_path).suffix.lower()
        output_ext = self.detect_output_format(modification_request, conversation_history)
        is_format_conversion = (output_ext != ext)
        
        try:
            text_only = ""
            if ext == '.docx':
                async for chunk_text in self._read_docx_with_paragraphs(file_path):
                    text_only += chunk_text
            elif ext == '.txt':
                async for chunk_text in self._read_txt(file_path):
                    text_only += chunk_text
            elif ext in ['.xlsx', '.xls']:
                async for chunk_text in self._read_excel(file_path):
                    text_only += chunk_text
            elif ext in ['.pptx', '.ppt']:
                async for chunk_text in self._read_pptx(file_path):
                    text_only += chunk_text
            elif ext == '.doc':
                async for chunk_text in self._read_doc(file_path):
                    text_only += chunk_text
            elif ext == '.pdf':
                async for chunk_text in self.read_document(file_path):
                    text_only += chunk_text
            elif ext == '.md':
                async for chunk_text in self._read_md(file_path):
                    text_only += chunk_text
            else:
                raise ValueError(f"不支持的文件格式: {ext}")
            
            system_prompt = self._build_system_prompt(modification_request, ext, output_ext)
            
            prompt_list = [
                {"role": "system", "content": system_prompt},
            ] + conversation_history + [
                {"role": "user", "content": f"用户需求：{modification_request}\n\n文档内容：\n{text_only}"},
            ]
            
            from core.agent import _stream_by_prompt
            
            content = ""
            for chunk_text in _stream_by_prompt(prompt_list, model=Params.DEFAULT_TEXT_TOOL_MODEL):
                content += chunk_text
            content = content.strip()
            
            if not content:
                return "抱歉，暂时无法处理文档请求。", None
            
            is_typo_request = any(kw in modification_request for kw in ['错别字', '错字', '纠错'])
            no_change_markers = ['没有发现错别字', '未发现错别字', '没有错别字']
            
            if is_typo_request and any(m in content for m in no_change_markers):
                return "文档检查完成，没有发现错别字。", None
            
            original_filename = Path(file_path).stem
            output_filename = f"{original_filename}_{uuid.uuid4().hex[:8]}{output_ext}"
            output_path = self.OUTPUT_DIR / output_filename
            
            if is_format_conversion:
                logger.info(f"格式转换: {ext} → {output_ext}, request={modification_request[:80]}")
                self._save_content_to_format(content, str(output_path), output_ext)
            else:
                if ext == '.docx':
                    text_paragraphs_info = text_only
                    self._modify_docx_with_format(file_path, content, modification_request, str(output_path), text_paragraphs_info)
                elif ext == '.txt':
                    original_content = text_only
                    self._modify_txt_with_format(content, str(output_path), modification_request, original_content)
                elif ext in ['.xlsx', '.xls']:
                    self._modify_excel(file_path, content, str(output_path))
                elif ext in ['.pptx', '.ppt']:
                    self._modify_pptx(file_path, content, str(output_path))
                elif ext == '.doc':
                    self._modify_doc(content, str(output_path), original_filename)
                elif ext == '.pdf':
                    await self.aysnc_modify_pdf(file_path, content, str(output_path), modification_request, text_only)
                elif ext == '.md':
                    self._modify_txt_with_format(content, str(output_path), modification_request, text_only)
            
            if output_path.exists():
                download_url = f"http://127.0.0.1:5001/download/{output_filename}"
                
                if is_format_conversion:
                    result_content = f"文档已完成处理并保存为 {output_ext} 格式！\n\n[下载处理后的文档]({download_url})"
                elif is_typo_request:
                    result_content = f"错别字已修正完毕！\n\n[下载修正后的文档]({download_url})"
                else:
                    result_content = f"文档修改完成！\n\n[下载修改后的文档]({download_url})"
                
                return result_content, str(output_path)
            else:
                return "抱歉，文档处理失败，请重试。", None
                
        except Exception as e:
            logger.error(f"文档处理失败: {str(e)[:200]}")
            return f"抱歉，文档处理失败：{str(e)[:100]}", None
    
    def _save_content_to_format(self, content: str, output_path: str, output_ext: str):
        """将内容保存为指定格式（用于格式转换场景）"""
        if output_ext == '.pdf':
            self._create_pdf(content, output_path)
        elif output_ext in ['.docx', '.doc']:
            self._create_docx(content, output_path)
        elif output_ext == '.txt':
            self._create_txt(content, output_path)
        elif output_ext in ['.xlsx', '.xls']:
            self._create_excel(content, output_path)
        elif output_ext in ['.pptx', '.ppt']:
            self._create_pptx(content, output_path)
        elif output_ext == '.md':
            self._create_md(content, output_path)
        elif output_ext == '.json':
            Path(output_path).write_text(content, encoding="utf-8")
        else:
            self._create_txt(content, output_path)
    
    def create_document(self, creation_request: str, conversation_history: List[Dict[str, str]], 
                       llm, output_ext: str = '.docx') -> Tuple[str, Optional[str]]:
        """创建新文档"""
        system_prompt = f"""你是一个专业的文档创作助手。你的任务是根据用户的要求创建一份完整的文档。

要求：
1. 根据用户需求，生成结构完整、内容丰富的文档
2. 生成结构化文字、分段、标题、列表、表格内容；
3. 调整格式、排版、润色、纠错、逻辑梳理。
4. 内容要专业、准确、有条理
5. 直接返回文档内容，不要任何解释、说明或额外文字
6. 保持清晰的换行结构，便于后续保存为文档文件
7. 输出格式为 {output_ext}

注意：你的回复将直接保存为文档文件，所以只能包含文档内容本身！"""
        
        try:
            prompt_list = [
                {"role": "system", "content": system_prompt},
            ] + conversation_history + [
                {"role": "user", "content": f"创建要求：{creation_request}"},
            ]
            
            response = llm.invoke(prompt_list)
            content = response.content if hasattr(response, 'content') else str(response)
            
            if not content or content.strip() == "":
                return "抱歉，暂时无法创建文档。", None
            
            markers = ['文档内容如下', '以下是文档内容', '创建好的文档', '文档已创建']
            extracted_content = None
            for marker in markers:
                if marker in content:
                    idx = content.index(marker)
                    newline_idx = content.find('\n', idx)
                    if newline_idx != -1:
                        extracted_content = content[newline_idx + 1:].strip()
                        break
            
            final_content = extracted_content if extracted_content else content
            
            doc_type_keywords = {
                '报告': 'report', '总结': 'summary', '计划': 'plan', '方案': 'proposal',
                '会议': 'meeting', '记录': 'record', '通知': 'notice', '公告': 'announcement',
                '合同': 'contract', '协议': 'agreement', '简历': 'resume', '论文': 'paper',
                '文章': 'article', '演讲稿': 'speech', '策划': 'proposal'
            }
            
            doc_type = 'document'
            for keyword, type_name in doc_type_keywords.items():
                if keyword in creation_request:
                    doc_type = type_name
                    break
            
            output_filename = f"{doc_type}_{uuid.uuid4().hex[:8]}{output_ext}"
            output_path = self.OUTPUT_DIR / output_filename
            
            if output_ext in ['.docx', '.doc']:
                self._create_docx(final_content, str(output_path))
            elif output_ext == '.txt':
                self._create_txt(final_content, str(output_path))
            elif output_ext in ['.xlsx', '.xls']:
                self._create_excel(final_content, str(output_path))
            elif output_ext in ['.pptx', '.ppt']:
                self._create_pptx(final_content, str(output_path))
            
            if output_path.exists():
                download_url = f"http://127.0.0.1:5001/download/{output_filename}"
                result_content = f"""文档创建完成！

<document-preview filename="{output_filename}" download-url="{download_url}">
{final_content}
</document-preview>"""
                return result_content, str(output_path)
            else:
                return "文档创建失败，请重试。", None
                
        except Exception as e:
            return f"抱歉，文档创建失败：{str(e)}", None
    
    def _build_system_prompt(self, modification_request: str, file_ext: str, 
                            output_ext: str = None) -> str:
        """根据用户需求动态构建系统提示词，覆盖修改、整理、转换、汇总等场景"""
        if output_ext is None:
            output_ext = self.detect_output_format(modification_request)
        
        is_format_conversion = (output_ext and output_ext != file_ext)
        is_organize = any(kw in modification_request for kw in [
            '整理', '汇总', '归纳', '提炼', '梳理', '归类', '整合',
            '总结', '摘要', '概括', '简述', '精简', '浓缩',
            '分析', '报告', '对比', '评估', '研究',
            '提取', '筛选', '查找', '找出', '列出',
            '转成', '转换成', '导出', '写入', '保存为', '另存为',
        ])
        
        if '错别字' in modification_request or '错字' in modification_request or '纠错' in modification_request:
            return """你是一个专业的文档校对助手。

任务：检查并修正文档中的错别字和词语搭配错误。

要求：
1. 只返回修改后的文档内容本身，不要任何解释、说明、标注或额外文字
2. 保持原文档的换行结构和格式完全一致
3. 如果没有发现错别字，直接回复"没有发现错别字"

注意：你的回复将直接保存为文档文件。"""

        if is_format_conversion:
            return f"""你是一个专业的文档处理助手，擅长将一种格式的文档内容转换为另一种格式。

源文件格式：{file_ext}
目标输出格式：{output_ext}

任务：根据用户需求，将下方提供的{file_ext}文件内容进行相应处理后，以{output_ext}格式输出。

要求：
1. 完整理解用户需求，对源文件内容进行整理/汇总/分析/改写等操作
2. 输出高质量、结构化的内容，适合保存为{output_ext}文件
3. 如果是表格类内容，请用 Markdown 表格格式输出（| 列1 | 列2 | ... |）
4. 如果是文档类内容，请合理使用 Markdown 标题（# / ## / ###）、列表、表格等格式
5. 只返回最终内容本身，不要任何解释、开头语或结束语

注意：你的回复将直接被保存为{output_ext}文件！"""

        if is_organize:
            return f"""你是一个专业的文档分析与整理助手。

任务：根据用户需求，对下方提供的文档内容进行整理、汇总、分析或改写。

用户需求理解：{modification_request}

要求：
1. 准确把握用户的核心意图（整理结构 / 汇总数据 / 提炼要点 / 分析内容 / 格式转换等）
2. 输出结构化、条理清晰、重点突出的内容
3. 使用 Markdown 格式组织内容（标题用 # / ## / ###，表格用 | 分隔，列表用 - 或数字编号）
4. 保留对用户有用的所有重要信息，不要遗漏关键数据
5. 只返回最终整理好的内容本身，不要任何解释、开头语或结束语

注意：你的回复将直接被保存为文档文件！"""

        return f"""你是一个专业的文档修改助手。

任务：根据用户的具体要求，对下方提供的文档内容进行针对性的修改。

用户的修改要求：{modification_request}

要求：
1. 严格按照用户要求进行修改（增删内容 / 调整格式 / 润色语言 / 修正错误等）
2. 只修改用户指定的部分，未提及的内容保持原样
3. 保留原文档的结构和换行格式
4. 直接返回修改后的完整内容，不要解释、不要说明改了什么

注意：你的回复将直接保存为文档文件。"""
    
    def _read_docx(self, file_path: str) -> str:
        """读取 .docx 文件"""
        try:
            from docx import Document
            doc = Document(file_path)
            content = []
            for para in doc.paragraphs:
                content.append(para.text)
            return '\n'.join(content)
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx")
    
    def _read_docx_to_md(self, file_path: str) -> str:
        """读取 .docx 文件并转换为 Markdown 格式"""
        try:
            from docx import Document
            doc = Document(file_path)
            md_lines = []
            
            for para in doc.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                
                # 根据段落样式确定标题级别
                style_name = para.style.name.lower()
                if style_name.startswith('heading 1'):
                    md_lines.append(f"# {text}")
                elif style_name.startswith('heading 2'):
                    md_lines.append(f"## {text}")
                elif style_name.startswith('heading 3'):
                    md_lines.append(f"### {text}")
                elif style_name.startswith('heading 4'):
                    md_lines.append(f"#### {text}")
                elif style_name.startswith('heading 5'):
                    md_lines.append(f"##### {text}")
                elif style_name.startswith('heading 6'):
                    md_lines.append(f"###### {text}")
                else:
                    md_lines.append(text)
            
            return '\n\n'.join(md_lines)
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx")
    
    async def _read_docx_with_paragraphs(self, file_path: str):
        """读取 .docx 文件，逐段 yield 文本"""
        from docx import Document
        doc = Document(file_path)
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                yield para.text + '\n'
    
    async def _read_txt(self, file_path: str):
        """读取 .txt 文件，逐行 yield"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line
    
    def _read_html(self, file_path: str) -> str:
        """读取 HTML 文件，提取纯文本内容并转换为 Markdown 格式"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # 使用正则表达式去除HTML标签，提取纯文本
            import re
            
            # 去除脚本和样式标签及其内容
            html_content = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', html_content, flags=re.IGNORECASE)
            
            # 去除HTML标签，保留文本内容
            text = re.sub(r'<[^>]+>', '', html_content)
            
            # 去除多余的空白字符
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 处理HTML实体
            text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&quot;', '"').replace('&apos;', "'")
            
            return text
        except Exception as e:
            # 如果HTML解析失败，尝试作为普通文本读取
            print(f"HTML解析失败，尝试作为文本文件读取: {str(e)}")
            return self._read_txt(file_path)
    
    async def _read_excel(self, file_path: str) -> str:
        """读取 Excel 文件并转换为 Markdown 格式（支持 .xls 和 .xlsx）"""
        ext = Path(file_path).suffix.lower()
        
        # 处理旧版 .xls 格式
        if ext == '.xls':
            async for chunk_text in self._read_excel_xls(file_path):
                yield chunk_text
        
        # 处理 .xlsx 格式
        elif ext == '.xlsx':
            async for chunk_text in self._read_excel_xlsx(file_path):
                yield chunk_text
        else:
            raise ValueError(f"不支持的 Excel 格式: {ext}")
    
    async def _read_excel_xls(self, file_path: str) -> str:
        """读取 .xls 格式 Excel 文件"""
        if not XLRD_AVAILABLE:
            raise ImportError("请安装 xlrd: pip install xlrd")
        
        try:
            wb = xlrd.open_workbook(file_path)
            content = []
            for sheet_name in wb.sheet_names():
                ws = wb.sheet_by_name(sheet_name)
                content.append(f"## {sheet_name}")
                
                # 获取所有行数据
                rows = []
                for row_idx in range(ws.nrows):
                    row_data = []
                    for col_idx in range(ws.ncols):
                        cell_value = ws.cell_value(row_idx, col_idx)
                        # 处理 xlrd 的特殊值
                        if isinstance(cell_value, float) and cell_value == int(cell_value):
                            row_data.append(str(int(cell_value)))
                        else:
                            row_data.append(str(cell_value))
                    rows.append(row_data)
                
                if rows:
                    # 转换为MD表格
                    header = '| ' + ' | '.join(rows[0]) + ' |'
                    content.append(header)
                    separator = '| ' + ' | '.join(['---'] * len(rows[0])) + ' |'
                    content.append(separator)
                    for row in rows[1:]:
                        content.append('| ' + ' | '.join(row) + ' |')
                
                content.append('')  # 空行分隔不同sheet
                # 每个sheet的内容作为一块文本
                yield '\n'.join(content)

            # return '\n'.join(content)
        except Exception as e:
            # # xlrd读取失败，尝试作为文本文件读取（可能是CSV或其他文本格式）
            # print(f"xlrd 读取失败，尝试作为文本文件读取: {str(e)}")
            # try:
            #     with open(file_path, 'r', encoding='utf-8') as f:
            #         content = f.read()
            #     # 尝试解析为CSV格式
            #     yield self._parse_text_as_csv(content, file_path)
            # except Exception as text_e:
            raise ValueError(f"无法读取文件 '{file_path}'。文件可能不是有效的 Excel 格式，或已损坏。尝试的错误: xlrd: {str(e)}")
    
    def _parse_text_as_csv(self, content: str, file_path: str) -> str:
        """将文本内容解析为CSV格式并转换为MD表格"""
        lines = content.strip().split('\n')
        if not lines:
            return f"## {Path(file_path).name}\n\n文件内容为空或无法解析"
        
        # 尝试识别分隔符
        separators = [',', '\t', ';', '|']
        best_separator = ','
        max_count = 0
        
        for sep in separators:
            count = lines[0].count(sep)
            if count > max_count:
                max_count = count
                best_separator = sep
        
        # 解析CSV
        rows = []
        for line in lines:
            # 跳过空行和BOM
            line = line.strip().replace('\ufeff', '')
            if line:
                rows.append([cell.strip() for cell in line.split(best_separator)])
        
        if rows:
            content = [f"## {Path(file_path).name}"]
            # 转换为MD表格
            header = '| ' + ' | '.join(rows[0]) + ' |'
            content.append(header)
            separator = '| ' + ' | '.join(['---'] * len(rows[0])) + ' |'
            content.append(separator)
            for row in rows[1:]:
                content.append('| ' + ' | '.join(row) + ' |')
            return '\n'.join(content)
        
        return f"## {Path(file_path).name}\n\n{content}"
    
    async def _read_excel_xlsx(self, file_path: str) -> str:
        """读取 .xlsx 格式 Excel 文件"""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path)
            content = []
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                content.append(f"## {sheet}")
                
                # 获取所有行数据
                rows = []
                for row in ws.iter_rows(values_only=True):
                    rows.append([str(cell) if cell is not None else '' for cell in row])
                
                if rows:
                    # 转换为MD表格
                    header = '| ' + ' | '.join(rows[0]) + ' |'
                    content.append(header)
                    separator = '| ' + ' | '.join(['---'] * len(rows[0])) + ' |'
                    content.append(separator)
                    for row in rows[1:]:
                        content.append('| ' + ' | '.join(row) + ' |')
                
                content.append('')  # 空行分隔不同sheet

                # 每个sheet的内容作为一块文本
                yield '\n'.join(content)

            # return '\n'.join(content)
        except ImportError:
            raise ImportError("请安装 openpyxl: pip install openpyxl")
    
    async def _read_pptx(self, file_path: str):
        """读取 PowerPoint 文件，逐页 yield Markdown 内容"""
        try:
            from pptx import Presentation
            prs = Presentation(file_path)

            for i, slide in enumerate(prs.slides, 1):
                parts = [f"## 幻灯片 {i}\n"]
                
                slide_texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                        if shape.text_frame and hasattr(shape.text_frame, 'text'):
                            text_content = shape.text_frame.text.strip()
                            if text_content:
                                slide_texts.append(text_content)
                
                if slide_texts:
                    parts.append(f"### {slide_texts[0]}\n")
                    for text in slide_texts[1:]:
                        parts.append(text + '\n')
                
                parts.append('')
                yield '\n'.join(parts)
        except ImportError:
            raise ImportError("请安装 python-pptx: pip install python-pptx")

    async def _read_md(self, file_path: str):
        """异步分片读取markdown原始文本，按chunk_size流式yield文本片段"""
        import aiofiles
        chunk_size = 1024
        try:
            async with aiofiles.open(file_path, 'r', encoding="utf-8") as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except FileNotFoundError:
            raise FileNotFoundError(f"md文件不存在: {file_path}")
        except PermissionError:
            raise PermissionError(f"无权限读取文件: {file_path}")
        except Exception as e:
            raise RuntimeError(f"读取md文件失败：{str(e)}") from e
    async def _read_doc(self, file_path: str):
        """读取旧版 .doc 文件，一次性 yield"""
        try:
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc_obj = word.Documents.Open(file_path)
            text_only = doc_obj.Content.Text
            doc_obj.Close()
            word.Quit()
            yield text_only
        except ImportError:
            raise ValueError("暂不支持直接读取 .doc 文件，请先转换为 .docx 格式")
    
    def _read_pdf(self, file_path: str) -> str:
        """读取 PDF 文件内容（支持扫描件OCR识别，并转换为MD格式）"""
        try:
            from unstructured.partition.pdf import partition_pdf
            from unstructured.staging.base import elements_to_markdown
            _unstructured_ok = True
        except ImportError:
            _unstructured_ok = False

        if _unstructured_ok:
            try:
                print(f"使用 unstructured 解析 PDF: {file_path}")
                elements = partition_pdf(file_path)
                
                if not elements:
                    print("unstructured 未解析到任何内容")
                else:
                    print(f"unstructured 解析到 {len(elements)} 个元素")
                
                text = elements_to_markdown(elements)
                print(f"unstructured 提取总文本长度: {len(text)}")
                
                if text.strip():
                    return text.strip()
            except Exception as e:
                print(f"unstructured 解析失败，尝试使用 pypdf: {str(e)}")
        
        # 回退到 pypdf
        if PdfReader is not None:
            try:
                with open(file_path, 'rb') as f:
                    reader = PdfReader(f)
                    text = ""
                    num_pages = len(reader.pages)
                    print(f"PDF 文件页数: {num_pages}")
                    
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            print(f"第 {i+1} 页提取到文本长度: {len(page_text)}")
                            text += page_text + "\n\n"
                        else:
                            print(f"第 {i+1} 页未提取到文本")
                    
                    print(f"pypdf 提取总文本长度: {len(text)}")
                    if text.strip():
                        return text.strip()
            except Exception as e:
                print(f"pypdf 读取错误: {str(e)}")
        
        # 如果以上方法都失败，尝试OCR（处理扫描件）
        ocr_ok, pytesseract, Image, pdf2image = _ensure_ocr()
        if ocr_ok:
            try:
                print(f"尝试使用 OCR 解析扫描件 PDF: {file_path}")
                
                # 设置poppler路径（支持Windows和Linux）
                import platform
                poppler_path = None
                if platform.system() == 'Windows':
                    # Windows系统常见的poppler路径
                    possible_paths = [
                        r'C:\Program Files\poppler-24.08.0\Library\bin',
                        r'C:\Program Files\poppler\bin',
                        r'C:\poppler\Library\bin',
                        r'C:\Program Files (x86)\poppler\bin',
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            poppler_path = path
                            print(f"找到 poppler 路径: {poppler_path}")
                            break
                
                # 将PDF转换为图像
                if poppler_path:
                    images = pdf2image.convert_from_path(file_path, poppler_path=poppler_path)
                else:
                    images = pdf2image.convert_from_path(file_path)
                
                print(f"PDF 转换为 {len(images)} 张图片")
                
                text = ""
                for i, image in enumerate(images):
                    # 使用Tesseract进行OCR识别
                    page_text = pytesseract.image_to_string(image, lang='chi_sim')
                    print(f"OCR 第 {i+1} 页提取到文本长度: {len(page_text)}")
                    if page_text:
                        text += page_text + "\n\n"
                
                print(f"OCR 提取总文本长度: {len(text)}")
                if text.strip():
                    return text.strip()
            except Exception as e:
                print(f"OCR 解析失败: {str(e)}")
        
        # 如果所有方法都失败
        raise ValueError(f"无法从 PDF 文件中提取文本，可能是扫描件或加密文件")
    
    def _modify_docx_with_format(self, original_path: str, modified_content: str, 
                                instructions: str, output_path: str, paragraph_info: List[Tuple[int, str]]):
        """修改 .docx 文件并保持格式"""
        from docx import Document
        doc = Document(original_path)
        modified_lines = modified_content.split('\n')
        
        for idx, (para_index, original_text) in enumerate(paragraph_info):
            if idx < len(modified_lines):
                para = doc.paragraphs[para_index]
                new_text = modified_lines[idx]
                for run in para.runs:
                    run.text = ""
                if new_text:
                    para.add_run(new_text)
        
        doc.save(output_path)
    
    def _modify_txt_with_format(self, modified_content: str, output_path: str, 
                               instructions: str, original_content: str):
        """修改 .txt 文件并处理格式"""
        is_typo_request = '错别字' in instructions or '错字' in instructions or '错误字' in instructions
        
        if is_typo_request:
            no_change_markers = ['没有发现', '未发现', '没有错别字', '无需修改', '原文档内容', '文档内容正确']
            is_no_change = any(marker in modified_content for marker in no_change_markers)
            
            if is_no_change:
                return
            
            markers = ['修改后的内容', '修改后内容', '以下是修改后的', '修改后的文档', '文档内容如下', '修改后的文本']
            extracted_content = None
            for marker in markers:
                if marker in modified_content:
                    idx = modified_content.index(marker)
                    newline_idx = modified_content.find('\n', idx)
                    if newline_idx != -1:
                        extracted_content = modified_content[newline_idx + 1:].strip()
                        break
            
            if extracted_content is None:
                if modified_content.startswith('好的') or modified_content.startswith('已经') or modified_content.startswith('我'):
                    paragraphs = modified_content.split('\n\n')
                    if len(paragraphs) > 1:
                        extracted_content = '\n\n'.join(paragraphs[1:]).strip()
            
            final_content = extracted_content if extracted_content else modified_content
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
        else:
            if '对齐' in instructions or '格式' in instructions:
                lines = modified_content.split('\n')
                processed_lines = []
                
                for line in lines:
                    if '左对齐' in instructions or '左' in instructions:
                        processed_lines.append(line.lstrip())
                    elif '居中' in instructions:
                        stripped_line = line.strip()
                        if stripped_line:
                            total_spaces = max(0, (80 - len(stripped_line)) // 2)
                            processed_lines.append(' ' * total_spaces + stripped_line)
                        else:
                            processed_lines.append('')
                    elif '右对齐' in instructions:
                        stripped_line = line.strip()
                        if stripped_line:
                            total_spaces = max(0, 80 - len(stripped_line))
                            processed_lines.append(' ' * total_spaces + stripped_line)
                        else:
                            processed_lines.append('')
                    else:
                        processed_lines.append(line.lstrip())
                
                modified_content = '\n'.join(processed_lines)
            
            markers = ['修改后的内容', '修改后内容', '以下是修改后的', '修改后的文档', '文档内容如下', '修改后的文本']
            extracted_content = None
            for marker in markers:
                if marker in modified_content:
                    idx = modified_content.index(marker)
                    newline_idx = modified_content.find('\n', idx)
                    if newline_idx != -1:
                        extracted_content = modified_content[newline_idx + 1:].strip()
                        break
            
            if extracted_content is None:
                if modified_content.startswith('好的') or modified_content.startswith('已经') or modified_content.startswith('我'):
                    paragraphs = modified_content.split('\n\n')
                    if len(paragraphs) > 1:
                        extracted_content = '\n\n'.join(paragraphs[1:]).strip()
            
            final_content = extracted_content if extracted_content else modified_content
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
    
    def _modify_excel(self, original_path: str, modified_content: str, output_path: str):
        """修改 Excel 文件"""
        import openpyxl
        wb = openpyxl.load_workbook(original_path)
        modified_lines = modified_content.split('\n')
        current_sheet = None
        row_idx = 0
        
        for line in modified_lines:
            if line.startswith('=== ') and line.endswith(' ==='):
                sheet_name = line[4:-4]
                if sheet_name in wb.sheetnames:
                    current_sheet = wb[sheet_name]
                    row_idx = 1
                else:
                    current_sheet = wb.create_sheet(sheet_name)
                    row_idx = 1
            elif current_sheet and line.strip():
                cells = line.split('\t')
                for col_idx, cell_value in enumerate(cells, 1):
                    current_sheet.cell(row=row_idx, column=col_idx, value=cell_value)
                row_idx += 1
        
        wb.save(output_path)
    
    def _modify_pptx(self, original_path: str, modified_content: str, output_path: str):
        """修改 PowerPoint 文件"""
        from pptx import Presentation
        prs = Presentation(original_path)
        
        pptx_content = modified_content
        
        markers = ['修改后的内容', '修改后内容', '以下是修改后的', '修改后的文档']
        for marker in markers:
            if marker in modified_content:
                idx = modified_content.index(marker)
                newline_idx = modified_content.find('\n', idx)
                if newline_idx != -1:
                    pptx_content = modified_content[newline_idx + 1:]
                    break
        
        modified_lines = pptx_content.split('\n')
        current_slide_idx = 0
        current_text_idx = 0
        
        for line in modified_lines:
            if line.startswith('=== 幻灯片 ') and ' ===' in line:
                slide_num = int(line.split('幻灯片 ')[1].split(' ===')[0])
                if slide_num <= len(prs.slides):
                    current_slide_idx = slide_num - 1
                    current_text_idx = 0
            elif line.strip() and current_slide_idx < len(prs.slides):
                slide = prs.slides[current_slide_idx]
                text_shapes = [shape for shape in slide.shapes
                               if hasattr(shape, "has_text_frame") and shape.has_text_frame
                               and shape.text_frame.text.strip()]
                if current_text_idx < len(text_shapes):
                    text_shapes[current_text_idx].text_frame.text = line
                    current_text_idx += 1
        
        prs.save(output_path)

    async def aysnc_modify_pdf(self, original_path: str, modified_content: str, 
                               output_path: str, instructions: str, original_content: str):
        """异步修改pdf文件内容"""
        import asyncio
        
        def _extract_modified_text(text: str) -> str:
            markers = [
                '修改后的内容', '修改后内容', '以下是修改后的', 
                '修改后的文档', '修改后的PDF', '文档内容如下',
                '以下是修改后的内容', '以下是修改后'
            ]
            for marker in markers:
                if marker in text:
                    idx = text.index(marker)
                    newline_idx = text.find('\n', idx)
                    if newline_idx != -1:
                        return text[newline_idx + 1:].strip()
            if text.startswith('好的') or text.startswith('已经') or text.startswith('我') or text.startswith('以下'):
                paragraphs = text.split('\n\n')
                if len(paragraphs) > 1:
                    return '\n\n'.join(paragraphs[1:]).strip()
            return text.strip()
        
        def _run_pdf_edit():
            import fitz
            
            is_typo_request = any(kw in instructions for kw in ['错别字', '错字', '纠错', '拼写错误', '语法错误'])
            is_format_request = any(kw in instructions for kw in ['对齐', '格式', '排版', '字体', '行距', '缩进', '布局'])
            is_rewrite_request = any(kw in instructions for kw in ['重写', '改写', '重新写', '润色', '扩写', '缩写'])
            
            extracted = _extract_modified_text(modified_content)
            
            no_change_markers = ['没有发现', '未发现', '没有错别字', '无需修改', 
                                 '原文档内容', '文档内容正确', '内容保持不变']
            is_no_change_reply = any(marker in modified_content for marker in no_change_markers)
            
            if is_typo_request and is_no_change_reply:
                logger.info("错别字检查未发现问题，直接保存原 PDF")
                try:
                    import shutil
                    shutil.copy2(original_path, output_path)
                    return True
                except Exception:
                    pass
            
            if is_no_change_reply:
                extracted = original_content.strip()
            
            try:
                doc = fitz.open(original_path)
            except Exception as e:
                logger.warning(f"无法打开原始 PDF，将重新创建: {e}")
                doc = None
            
            if is_format_request or is_rewrite_request:
                if doc is not None:
                    try:
                        doc.close()
                    except Exception:
                        pass
                logger.info(f"指令涉及布局变更 (format={is_format_request}, rewrite={is_rewrite_request})，重建 PDF")
                self._create_pdf(extracted, output_path)
                return True
            
            if doc is not None and len(doc) > 0:
                try:
                    extracted_clean = re.sub(r'\s+', '', extracted)
                    original_clean = re.sub(r'\s+', '', original_content)
                    
                    if len(extracted_clean) > 0 and len(original_clean) > 0:
                        try:
                            from difflib import SequenceMatcher
                            ratio = SequenceMatcher(None, original_clean, extracted_clean).ratio()
                            
                            no_change_threshold = 0.97 if is_typo_request else 0.95
                            if ratio > no_change_threshold:
                                doc.save(output_path)
                                doc.close()
                                logger.info("PDF 内容无实质性变化，直接保存原文件")
                                return True
                            
                            if ratio < 0.5:
                                logger.info(f"内容相似度较低 ({ratio:.2f})，文本改动幅度大，重建 PDF")
                                doc.close()
                                self._create_pdf(extracted, output_path)
                                return True
                            
                            lower_bound = 0.55 if is_typo_request else 0.4
                            changed_blocks = []
                            orig_blocks = original_content.split('\n')
                            mod_blocks = extracted.split('\n')
                            
                            for orig_line in orig_blocks:
                                orig_stripped = orig_line.strip()
                                if not orig_stripped or len(orig_stripped) < 2:
                                    continue
                                for mod_line in mod_blocks:
                                    mod_stripped = mod_line.strip()
                                    if mod_stripped and mod_stripped != orig_stripped:
                                        try:
                                            line_ratio = SequenceMatcher(None, orig_stripped, mod_stripped).ratio()
                                            if lower_bound < line_ratio < no_change_threshold and orig_stripped not in [b[0] for b in changed_blocks]:
                                                changed_blocks.append((orig_stripped, mod_stripped))
                                        except Exception:
                                            pass
                            
                            replaced = 0
                            if changed_blocks:
                                for page in doc:
                                    for orig_text, new_text in changed_blocks:
                                        try:
                                            rects = page.search_for(orig_text)
                                            for rect in rects:
                                                page.add_redact_annot(rect, text=new_text, fill=(1, 1, 1))
                                                replaced += 1
                                        except Exception:
                                            continue
                                
                                if replaced > 0:
                                    doc.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                                    doc.save(output_path)
                                    doc.close()
                                    logger.info(f"PDF 就地修改完成，替换了 {replaced} 处文本 (typo={is_typo_request})")
                                    return True
                        except Exception as e:
                            logger.warning(f"就地编辑失败，尝试重新创建: {e}")
                    
                    doc.close()
                except Exception as e:
                    logger.warning(f"PDF 就地编辑异常: {e}")
                    try:
                        doc.close()
                    except Exception:
                        pass
            
            self._create_pdf(extracted, output_path)
            return True
        
        await asyncio.to_thread(_run_pdf_edit)
    
    def _modify_doc(self, modified_content: str, output_path: str, original_filename: str):
        """修改 .doc 文件（转换为 .docx）"""
        docx_output_path = self.OUTPUT_DIR / f"{original_filename}_{uuid.uuid4().hex[:8]}.docx"
        
        from docx import Document
        doc = Document()
        
        modified_lines = modified_content.split('\n')
        for line in modified_lines:
            if line.strip():
                doc.add_paragraph(line)
        
        doc.save(str(docx_output_path))
        
        try:
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc_obj = word.Documents.Open(str(docx_output_path))
            doc_obj.SaveAs(str(output_path), FileFormat=0)
            doc_obj.Close()
            word.Quit()
            docx_output_path.unlink()
        except:
            os.replace(str(docx_output_path), str(output_path))
    
    def _create_docx(self, content: str, output_path: str):
        """创建 .docx 文档"""
        from docx import Document
        from docx.shared import Pt
        
        doc = Document()
        
        def clean_md_inline(text):
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            text = re.sub(r'~~(.+?)~~', r'\1', text)
            text = re.sub(r'`(.+?)`', r'\1', text)
            return text.strip()
        
        def is_table_row(line):
            s = line.strip()
            return s.startswith('|') and s.endswith('|') and '|' in s[1:-1]
        
        def is_table_separator(line):
            s = line.strip()
            cells = s.strip('|').split('|')
            return all(re.match(r'^[\s\-:]+$', cell) for cell in cells if cell.strip())
        
        def parse_table_rows(lines):
            if len(lines) < 3:
                return None, []
            header = [cell.strip() for cell in lines[0].strip().strip('|').split('|')]
            data_rows = []
            for line in lines[2:]:
                if is_table_row(line) and not is_table_separator(line):
                    row = [cell.strip() for cell in line.strip().strip('|').split('|')]
                    data_rows.append(row)
            return header, data_rows
        
        paragraphs = content.split('\n')
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]
            
            if is_table_row(para) and not is_table_separator(para):
                table_lines = [para]
                j = i + 1
                while j < len(paragraphs):
                    if is_table_row(paragraphs[j]) or is_table_separator(paragraphs[j]):
                        table_lines.append(paragraphs[j])
                        j += 1
                    else:
                        break
                
                header, data_rows = parse_table_rows(table_lines)
                if header and data_rows:
                    table = doc.add_table(rows=len(data_rows) + 1, cols=len(header))
                    table.style = 'Table Grid'
                    header_cells = table.rows[0].cells
                    for col_idx, cell_text in enumerate(header):
                        cell = header_cells[col_idx]
                        cell.text = clean_md_inline(cell_text)
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                run.font.bold = True
                                run.font.size = Pt(10)
                    for row_idx, row_data in enumerate(data_rows):
                        row_cells = table.rows[row_idx + 1].cells
                        for col_idx, cell_text in enumerate(row_data):
                            if col_idx < len(row_cells):
                                row_cells[col_idx].text = clean_md_inline(cell_text)
                                for paragraph in row_cells[col_idx].paragraphs:
                                    for run in paragraph.runs:
                                        run.font.size = Pt(10)
                    i = j
                    continue
                else:
                    doc.add_paragraph(para.strip())
                    i = j
                    continue
            
            stripped = para.strip()
            if not stripped:
                i += 1
                continue
            
            heading_level = None
            title_text = None
            
            if stripped.startswith('#'):
                title_text = stripped.lstrip('#').strip()
                if stripped.startswith('######'):
                    heading_level = 6
                elif stripped.startswith('#####'):
                    heading_level = 5
                elif stripped.startswith('####'):
                    heading_level = 4
                elif stripped.startswith('###'):
                    heading_level = 3
                elif stripped.startswith('##'):
                    heading_level = 2
                else:
                    heading_level = 1
            elif re.match(r'^[一二三四五六七八九十百]+[、.]', stripped):
                title_text = stripped
                heading_level = 2
            elif re.match(r'^\d+[.、]', stripped):
                title_text = stripped
                heading_level = 3
            elif re.match(r'^[\(（]?\d+[\)）]', stripped) or re.match(r'^[①③④⑤⑧⑨⑩]', stripped):
                title_text = stripped
                heading_level = 4
            elif re.match(r'^[A-Z][.、]', stripped) or re.match(r'^[a-z][.、]', stripped):
                title_text = stripped
                heading_level = 4
            elif len(stripped) < 50 and not stripped.endswith(('。', '！', '？', '，', '；', ':', '：')):
                title_text = stripped
                heading_level = 2
            
            if heading_level and title_text:
                doc.add_heading(title_text, level=heading_level)
            else:
                doc.add_paragraph(stripped)
            
            i += 1
        
        doc.save(output_path)
    
    def _create_txt(self, content: str, output_path: str):
        """创建 .txt 文档"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def _create_md(self, content: str, output_path: str):
        """创建 .md 文档"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _create_excel(self, content: str, output_path: str):
        """创建 Excel 文档"""
        import openpyxl
        from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
        
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        
        lines = content.split('\n')
        parsed_rows = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if all(c in '-|: ' for c in line):
                continue
            
            cells = []
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            elif '\t' in line:
                cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
            elif ',' in line and line.count(',') > 1:
                cells = [cell.strip() for cell in line.split(',')]
            else:
                cells = [line]
            
            if cells:
                parsed_rows.append(cells)
        
        for row_idx, row_data in enumerate(parsed_rows, 1):
            for col_idx, cell_value in enumerate(row_data, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=cell_value)
                
                if row_idx == 1:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                    cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if cell.value:
                        cell_length = len(str(cell.value))
                        if cell_length > max_length:
                            max_length = cell_length
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
        
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
            for cell in row:
                cell.border = thin_border
        
        wb.save(output_path)
    
    def _create_pptx(self, content: str, output_path: str):
        """创建 PowerPoint 文档"""
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
        
        def clean_md(text):
            text = re.sub(r'#{1,6}\s*', '', text)
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            text = re.sub(r'~~(.+?)~~', r'\1', text)
            text = re.sub(r'`(.+?)`', r'\1', text)
            text = re.sub(r'^[-*]\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'^\d+[.、]\s*', '', text, flags=re.MULTILINE)
            return text.strip()
        
        def set_font(paragraph, size=18, bold=False, color=None):
            for run in paragraph.runs:
                run.font.size = Pt(size)
                run.font.bold = bold
                if color:
                    run.font.color.rgb = color
        
        slide_blocks = re.split(r'\n#{1,3}\s*幻灯片\s*\d+[:：]\s*', content)
        slide_blocks = [b.strip() for b in slide_blocks if b.strip()]
        
        for idx, block in enumerate(slide_blocks):
            lines = block.split('\n')
            lines = [l.strip() for l in lines if l.strip() and l.strip() != '---']
            
            if not lines:
                continue
            
            title_text = clean_md(lines[0])
            title_text = re.sub(r'^幻灯片\s*\d+[:：]\s*', '', title_text)
            
            if idx == 0:
                slide_layout = prs.slide_layouts[6]
                slide = prs.slides.add_slide(slide_layout)
                
                left = Inches(1)
                top = Inches(2.5)
                width = Inches(11.333)
                height = Inches(1)
                txBox = slide.shapes.add_textbox(left, top, width, height)
                tf = txBox.text_frame
                p = tf.paragraphs[0]
                p.text = title_text
                p.alignment = 1
                set_font(p, size=40, bold=True)
                
                if len(lines) > 1:
                    top2 = Inches(3.8)
                    txBox2 = slide.shapes.add_textbox(left, top2, width, Inches(2))
                    tf2 = txBox2.text_frame
                    tf2.word_wrap = True
                    for line in lines[1:]:
                        if line.strip() and line.strip() != '---':
                            p = tf2.add_paragraph()
                            p.text = clean_md(line)
                            p.alignment = 1
                            set_font(p, size=24)
            else:
                slide_layout = prs.slide_layouts[1]
                slide = prs.slides.add_slide(slide_layout)
                slide.shapes.title.text = title_text
                
                title_shape = slide.shapes.title
                for para in title_shape.text_frame.paragraphs:
                    set_font(para, size=32, bold=True)
                
                if len(lines) > 1:
                    body_shape = slide.shapes.placeholders[1]
                    tf = body_shape.text_frame
                    tf.clear()
                    
                    first_para = True
                    for line in lines[1:]:
                        if not line.strip() or line.strip() == '---':
                            continue
                        
                        cleaned = clean_md(line)
                        if not cleaned:
                            continue
                        
                        if first_para:
                            p = tf.paragraphs[0]
                            first_para = False
                        else:
                            p = tf.add_paragraph()
                        
                        indent_level = 0
                        if cleaned.startswith('- ') or cleaned.startswith('• '):
                            cleaned = cleaned[2:]
                            indent_level = 0
                        elif cleaned.startswith('  - ') or cleaned.startswith('    • '):
                            cleaned = cleaned.lstrip()
                            if cleaned.startswith('- ') or cleaned.startswith('• '):
                                cleaned = cleaned[2:]
                            indent_level = 1
                        
                        p.text = cleaned
                        p.level = indent_level
                        set_font(p, size=20 if indent_level == 0 else 18)
        
        prs.save(output_path)
    
    def _create_pdf(self, content: str, output_path: str):
        """创建 PDF 文档（优先使用 reportlab，自动回退到 docx→pdf 转换）"""
        from docx import Document
        from docx.shared import Pt
        
        doc = Document()
        
        def clean_md_inline(text):
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            text = re.sub(r'~~(.+?)~~', r'\1', text)
            text = re.sub(r'`(.+?)`', r'\1', text)
            return text.strip()
        
        def is_table_row(line):
            s = line.strip()
            return s.startswith('|') and s.endswith('|') and '|' in s[1:-1]
        
        def is_table_separator(line):
            s = line.strip()
            cells = s.strip('|').split('|')
            return all(re.match(r'^[\s\-:]+$', cell) for cell in cells if cell.strip())
        
        def parse_table_rows(lines):
            if len(lines) < 3:
                return None, []
            header = [cell.strip() for cell in lines[0].strip().strip('|').split('|')]
            data_rows = []
            for line in lines[2:]:
                if is_table_row(line) and not is_table_separator(line):
                    row = [cell.strip() for cell in line.strip().strip('|').split('|')]
                    data_rows.append(row)
            return header, data_rows
        
        paragraphs = content.split('\n')
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]
            
            if is_table_row(para) and not is_table_separator(para):
                table_lines = [para]
                j = i + 1
                while j < len(paragraphs):
                    if is_table_row(paragraphs[j]) or is_table_separator(paragraphs[j]):
                        table_lines.append(paragraphs[j])
                        j += 1
                    else:
                        break
                
                header, data_rows = parse_table_rows(table_lines)
                if header and data_rows:
                    table = doc.add_table(rows=len(data_rows) + 1, cols=len(header))
                    table.style = 'Table Grid'
                    header_cells = table.rows[0].cells
                    for col_idx, cell_text in enumerate(header):
                        cell = header_cells[col_idx]
                        cell.text = clean_md_inline(cell_text)
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                run.font.bold = True
                                run.font.size = Pt(10)
                    for row_idx, row_data in enumerate(data_rows):
                        row_cells = table.rows[row_idx + 1].cells
                        for col_idx, cell_text in enumerate(row_data):
                            if col_idx < len(row_cells):
                                row_cells[col_idx].text = clean_md_inline(cell_text)
                                for paragraph in row_cells[col_idx].paragraphs:
                                    for run in paragraph.runs:
                                        run.font.size = Pt(10)
                    i = j
                    continue
                else:
                    doc.add_paragraph(para.strip())
                    i = j
                    continue
            
            stripped = para.strip()
            if not stripped:
                i += 1
                continue
            
            if stripped.startswith('#'):
                title_text = stripped.lstrip('#').strip()
                if stripped.startswith('######'):
                    heading_level = 6
                elif stripped.startswith('#####'):
                    heading_level = 5
                elif stripped.startswith('####'):
                    heading_level = 4
                elif stripped.startswith('###'):
                    heading_level = 3
                elif stripped.startswith('##'):
                    heading_level = 2
                else:
                    heading_level = 1
                doc.add_heading(title_text, level=heading_level)
            else:
                doc.add_paragraph(stripped)
            
            i += 1
        
        temp_docx = self.OUTPUT_DIR / f"_temp_{uuid.uuid4().hex[:8]}.docx"
        doc.save(str(temp_docx))
        
        try:
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc_obj = word.Documents.Open(str(temp_docx))
            doc_obj.SaveAs(str(output_path), FileFormat=17)
            doc_obj.Close()
            word.Quit()
        except Exception:
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.pdfbase import pdfmetrics
                from reportlab.pdfbase.ttfonts import TTFont
                
                font_paths = [
                    r'C:\Windows\Fonts\msyh.ttc',
                    r'C:\Windows\Fonts\simhei.ttf',
                    r'C:\Windows\Fonts\simsun.ttc',
                ]
                font_name = 'Helvetica'
                for fp in font_paths:
                    if os.path.exists(fp):
                        try:
                            pdfmetrics.registerFont(TTFont('CNFont', fp))
                            font_name = 'CNFont'
                            break
                        except Exception:
                            continue
                
                styles = getSampleStyleSheet()
                styles['Normal'].fontName = font_name
                styles['Normal'].fontSize = 11
                styles['Title'].fontName = font_name
                styles['Heading1'].fontName = font_name
                styles['Heading2'].fontName = font_name
                
                story = []
                for line in paragraphs:
                    stripped = line.strip()
                    if not stripped:
                        story.append(Spacer(1, 6))
                        continue
                    story.append(Paragraph(clean_md_inline(stripped), styles['Normal']))
                
                doc_pdf = SimpleDocTemplate(str(output_path), pagesize=A4)
                doc_pdf.build(story)
            except Exception:
                import shutil
                shutil.copy(str(temp_docx), str(output_path))
        finally:
            try:
                temp_docx.unlink()
            except Exception:
                pass
    
    def detect_output_format(self, creation_request: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """检测用户期望的输出文件格式"""
        format_keywords = {
            'pdf': '.pdf', 'PDF': '.pdf',
            'word': '.docx', 'docx': '.docx', 'doc': '.docx', '文档': '.docx',
            'txt': '.txt', '文本': '.txt', '纯文本': '.txt',
            'excel': '.xlsx', 'xlsx': '.xlsx', 'xls': '.xlsx', '表格': '.xlsx', '工作表': '.xlsx',
            'ppt': '.pptx', 'pptx': '.pptx', '幻灯片': '.pptx', '演示': '.pptx', 'powerpoint': '.pptx', '演示文稿': '.pptx',
            'md': '.md', 'markdown': '.md', 'Markdown': '.md'
        }
        
        for keyword, ext in format_keywords.items():
            if keyword in creation_request:
                return ext
        
        if conversation_history:
            for msg in reversed(conversation_history):
                msg_content = msg.get('content', '') if isinstance(msg, dict) else str(msg)
                for keyword, ext in format_keywords.items():
                    if keyword in msg_content:
                        return ext
        
        return '.docx'
    
    def get_download_url(self, file_path: str) -> str:
        """生成下载链接"""
        return f"/download/{Path(file_path).name}"


doc_processor = DocumentProcessor()