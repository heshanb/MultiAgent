import os
import uuid
from typing import Optional, Dict, Any
from pathlib import Path
from settings.Define import Params, PathConfig

class DocumentProcessor:
    """文档处理工具类，支持多种文档格式的读取、修改和保存"""
    SUPPORTED_FORMATS = Params.DOC_SUPPORTED_FORMATS
    # 使用绝对路径，基于当前文件所在目录
    _BASE_DIR = PathConfig.BASE_DIR
    UPLOAD_DIR = PathConfig.UPLOADS_DIR
    OUTPUT_DIR = PathConfig.OUTPUTS_DIR
    
    def __init__(self):
        self.UPLOAD_DIR.mkdir(exist_ok=True)
        self.OUTPUT_DIR.mkdir(exist_ok=True)
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> tuple:
        """保存上传的文件并返回 (文件路径, 保存的文件名)"""
        # file_id = str(uuid.uuid4())[:8]
        # ext = Path(filename).suffix.lower()
        saved_filename = filename
        file_path = self.UPLOAD_DIR / saved_filename
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return str(file_path), saved_filename
    
    def read_document(self, file_path: str) -> str:
        """读取文档内容并返回文本"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.docx':
            return self._read_docx(file_path)
        elif ext == '.txt':
            return self._read_txt(file_path)
        elif ext in ['.xlsx', '.xls']:
            return self._read_excel(file_path)
        elif ext in ['.pptx', '.ppt']:
            return self._read_pptx(file_path)
        elif ext in ['.py', '.java', '.cpp', '.c', '.h', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.cs', '.sh', '.bash', '.sql', '.yaml', '.yml', '.json', '.xml', '.md', '.ini', '.cfg', '.conf', '.toml']:
            return self._read_txt(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    def modify_and_save(self, original_path: str, modified_content: str, instructions: str) -> str:
        """根据修改指令修改文档并保存"""
        ext = Path(original_path).suffix.lower()
        file_id = str(uuid.uuid4())[:8]
        output_filename = f"modified_{file_id}{ext}"
        output_path = self.OUTPUT_DIR / output_filename
        
        if ext == '.docx':
            return self._modify_docx(original_path, modified_content, instructions, str(output_path))
        elif ext == '.txt':
            return self._modify_txt(modified_content, str(output_path))
        else:
            raise ValueError(f"暂不支持修改此格式: {ext}")
    
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
    
    def _read_txt(self, file_path: str) -> str:
        """读取 .txt 文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _read_excel(self, file_path: str) -> str:
        """读取 Excel 文件"""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path)
            content = []
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                content.append(f"=== {sheet} ===")
                for row in ws.iter_rows(values_only=True):
                    content.append('\t'.join(str(cell) for cell in row if cell is not None))
            return '\n'.join(content)
        except ImportError:
            raise ImportError("请安装 openpyxl: pip install openpyxl")
    
    def _read_pptx(self, file_path: str) -> str:
        """读取 PowerPoint 文件"""
        try:
            from pptx import Presentation
            prs = Presentation(file_path)
            content_parts = []

            for i, slide in enumerate(prs.slides, 1):
                content_parts.append(f"=== 幻灯片 {i} ===")
                for shape in slide.shapes:
                    # 使用正确的属性
                    if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                        if shape.text_frame and hasattr(shape.text_frame, 'text'):
                            text_content = shape.text_frame.text.strip()
                            if text_content:
                                content_parts.append(text_content)

            return '\n'.join(content_parts)
        except ImportError:
            raise ImportError("请安装 python-pptx: pip install python-pptx")
    
    def _modify_docx(self, original_path: str, modified_content: str, instructions: str, output_path: str) -> str:
        """修改 .docx 文件"""
        try:
            from docx import Document
            from docx.shared import Pt, RGBColor
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            
            doc = Document(original_path)
            
            paragraphs = modified_content.split('\n')
            for i, para_text in enumerate(paragraphs):
                if i < len(doc.paragraphs):
                    doc.paragraphs[i].text = para_text
                else:
                    doc.add_paragraph(para_text)
            
            doc.save(output_path)
            return output_path
        except Exception as e:
            raise Exception(f"修改 docx 文件失败: {str(e)}")
    
    def _modify_txt(self, modified_content: str, output_path: str) -> str:
        """修改 .txt 文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        return output_path
    
    def get_download_url(self, file_path: str) -> str:
        """生成下载链接（实际应用中应该返回完整的 URL）"""
        return f"/download/{Path(file_path).name}"


doc_processor = DocumentProcessor()