"""
文件上传安全校验模块
- 文件大小限制
- 文件类型校验
- 恶意文件检测
"""
import os
from typing import Set, Tuple
from fastapi import HTTPException, UploadFile
import magic
import mimetypes

# 允许的文件扩展名
ALLOWED_EXTENSIONS: Set[str] = {
    # 文档类
    '.txt', '.md', '.doc', '.docx', '.pdf', '.xls', '.xlsx', '.ppt', '.pptx',
    # 代码类
    '.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.php', '.rb', '.swift', '.kt',
    '.html', '.css', '.json', '.xml', '.yaml', '.yml',
    # 图片类
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg',
    # 工程类
    '.dxf', '.dwg',
    # 数据类
    '.csv', '.tsv',
}

# 允许的文件类型（MIME类型）
ALLOWED_MIME_TYPES: Set[str] = {
    'text/plain', 'text/markdown', 'text/csv',
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'image/png', 'image/jpeg', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp', 'image/svg+xml',
    'application/json', 'application/xml', 'text/xml',
    'application/x-yaml', 'text/yaml',
    'application/x-dxf', 'application/dxf',
}

# 最大文件大小：500MB
MAX_FILE_SIZE: int = 500 * 1024 * 1024

# 危险文件扩展名（禁止上传）
DANGEROUS_EXTENSIONS: Set[str] = {
    '.exe', '.bat', '.cmd', '.com', '.msi', '.scr', '.pif', '.vbs', '.jse',
    '.ws', '.wsf', '.wsc', '.wsh', '.ps1', '.ps1xml', '.ps2', '.ps2xml', '.psc1', '.psc2',
    '.msh', '.msh1', '.msh2', '.mshxml', '.msh1xml', '.msh2xml', '.reg', '.inf',
    '.sh', '.bash', '.zsh', '.csh', '.ksh', '.tcsh',
}


class FileValidator:
    """文件上传安全校验器"""

    def __init__(
        self,
        max_size: int = MAX_FILE_SIZE,
        allowed_extensions: Set[str] = ALLOWED_EXTENSIONS,
        allowed_mime_types: Set[str] = ALLOWED_MIME_TYPES,
    ):
        self.max_size = max_size
        self.allowed_extensions = allowed_extensions
        self.allowed_mime_types = allowed_mime_types

    def validate(self, file: UploadFile) -> Tuple[bool, str]:
        """
        校验文件安全性
        返回: (是否安全, 错误信息)
        """
        # 1. 检查文件名
        if not file.filename:
            return False, "文件名不能为空"

        # 2. 检查文件扩展名
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in DANGEROUS_EXTENSIONS:
            return False, f"禁止上传的文件类型: {ext}"

        if ext not in self.allowed_extensions:
            return False, f"不支持的文件类型: {ext}，支持的类型: {', '.join(sorted(self.allowed_extensions))}"

        # 3. 检查文件大小
        if file.size and file.size > self.max_size:
            return False, f"文件大小超过限制: {file.size / 1024 / 1024:.2f}MB > {self.max_size / 1024 / 1024:.2f}MB"

        # 4. 检查文件名长度
        if len(file.filename) > 255:
            return False, "文件名过长"

        # 5. 检查文件名是否包含危险字符
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
        if any(c in file.filename for c in dangerous_chars):
            return False, "文件名包含非法字符"

        return True, ""

    async def validate_with_content(self, file: UploadFile) -> Tuple[bool, str]:
        """
        校验文件安全性（包含内容检查）
        返回: (是否安全, 错误信息)
        """
        # 先进行基础校验
        is_safe, error_msg = self.validate(file)
        if not is_safe:
            return False, error_msg

        # 读取文件内容检查大小
        content = await file.read()
        await file.seek(0)  # 重置文件指针

        if len(content) > self.max_size:
            return False, f"文件内容大小超过限制: {len(content) / 1024 / 1024:.2f}MB"

        # 检查文件内容是否为空
        if len(content) == 0:
            return False, "文件内容为空"

        # 检测文件MIME类型
        try:
            mime = magic.Magic(mime=True)
            detected_mime = mime.from_buffer(content[:2048])  # 只检测文件头
            
            # 对于文本文件，放宽MIME类型检查
            ext = os.path.splitext(file.filename)[1].lower()
            text_extensions = {'.txt', '.md', '.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', 
                             '.go', '.php', '.rb', '.swift', '.kt', '.html', '.css', '.json', 
                             '.xml', '.yaml', '.yml', '.csv', '.tsv'}
            
            # docx、xlsx、pptx 文件本质是ZIP格式，需要特殊处理
            office_extensions = {'.docx', '.xlsx', '.pptx'}
            
            if ext in office_extensions:
                # Office文件允许是ZIP格式
                if detected_mime not in self.allowed_mime_types and detected_mime != 'application/zip':
                    return False, f"文件类型不匹配: 检测到 {detected_mime}"
            elif ext not in text_extensions and detected_mime not in self.allowed_mime_types:
                # 对于非文本文件和非Office文件，严格检查MIME类型
                return False, f"文件类型不匹配: 检测到 {detected_mime}"
        except Exception as e:
            # magic库不可用时，跳过MIME类型检查
            pass

        return True, ""


# 全局文件校验器实例
file_validator = FileValidator()