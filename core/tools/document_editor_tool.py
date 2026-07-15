"""
文档编辑工具类
提供文档创建、修改、格式化等操作的工具接口
"""

from typing import Any, Dict, Optional, List, Union
from pydantic import BaseModel, Field
from core.DocProcess.cloud_document_editor import DocumentEditorFactory, CloudDocumentEditor


class DocumentEditorTool:
    """文档编辑工具"""
    
    def __init__(self):
        self.editor = DocumentEditorFactory.create_editor()
    
    def create_document(self, content: str, output_path: str) -> Dict[str, Any]:
        """
        创建新的docx文档
        
        Args:
            content: 文档内容
            output_path: 输出文件路径
        
        Returns:
            操作结果，包含success和message字段
        """
        try:
            result = self.editor.create_document(content, output_path)
            if result:
                return {
                    "success": True,
                    "message": f"文档创建成功: {output_path}",
                    "file_path": output_path
                }
            else:
                return {
                    "success": False,
                    "message": "文档创建失败"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"文档创建失败: {str(e)}"
            }
    
    def update_document(self, file_path: str, changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新docx文档内容
        
        Args:
            file_path: 文档路径
            changes: 修改内容，支持以下操作：
                - replace_text: 字典，key为要替换的文本，value为新文本
                - add_paragraph: 列表，每个元素包含content和可选的position
                - delete_paragraph: 列表，包含要删除的段落索引
                - add_table: 列表，包含表格数据
        
        Returns:
            操作结果
        """
        try:
            result = self.editor.update_document(file_path, changes)
            if result:
                return {
                    "success": True,
                    "message": f"文档更新成功: {file_path}",
                    "file_path": file_path
                }
            else:
                return {
                    "success": False,
                    "message": "文档更新失败"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"文档更新失败: {str(e)}"
            }
    
    def format_document(self, file_path: str, formatting: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化docx文档
        
        Args:
            file_path: 文档路径
            formatting: 格式设置，支持以下选项：
                - font_size: 字体大小（磅）
                - font_name: 字体名称
                - font_color: 字体颜色（十六进制，如#FF0000）
                - alignment: 对齐方式（left/center/right/justify）
                - line_spacing: 行距（倍）
                - margin: 边距设置，包含top/bottom/left/right（毫米）
        
        Returns:
            操作结果
        """
        try:
            result = self.editor.format_document(file_path, formatting)
            if result:
                return {
                    "success": True,
                    "message": f"文档格式化成功: {file_path}",
                    "file_path": file_path
                }
            else:
                return {
                    "success": False,
                    "message": "文档格式化失败"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"文档格式化失败: {str(e)}"
            }
    
    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """
        提取文档文本内容
        
        Args:
            file_path: 文档路径
        
        Returns:
            操作结果，包含文本内容
        """
        try:
            text = self.editor.extract_text(file_path)
            if text:
                return {
                    "success": True,
                    "message": "文本提取成功",
                    "content": text,
                    "length": len(text)
                }
            else:
                return {
                    "success": False,
                    "message": "文档为空或提取失败"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"文本提取失败: {str(e)}"
            }
    
    def reformat_document(self, file_path: str, instructions: str) -> Dict[str, Any]:
        """
        根据自然语言指令重新格式化文档
        
        Args:
            file_path: 文档路径
            instructions: 格式化指令（自然语言）
        
        Returns:
            操作结果
        """
        try:
            # 解析自然语言指令，转换为格式化参数
            formatting = self._parse_instructions(instructions)
            
            if formatting:
                result = self.format_document(file_path, formatting)
                if result["success"]:
                    result["message"] = f"根据指令格式化成功: {instructions}"
                return result
            else:
                return {
                    "success": False,
                    "message": "无法解析格式化指令"
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"格式化失败: {str(e)}"
            }
    
    def _parse_instructions(self, instructions: str) -> Optional[Dict[str, Any]]:
        """
        解析自然语言格式化指令
        
        Args:
            instructions: 自然语言指令
        
        Returns:
            格式化参数字典
        """
        formatting = {}
        
        # 解析字体大小
        import re
        
        # 字体大小
        size_match = re.search(r'字体大小[\u4e00-\u9fa5]*(\d+)', instructions)
        if size_match:
            formatting['font_size'] = int(size_match.group(1))
        
        # 字体名称
        font_name_match = re.search(r'(宋体|黑体|微软雅黑|仿宋|楷体|Arial|Times New Roman)', instructions)
        if font_name_match:
            formatting['font_name'] = font_name_match.group(1)
        
        # 对齐方式
        if '居中' in instructions or '居中对齐' in instructions:
            formatting['alignment'] = 'center'
        elif '右对齐' in instructions:
            formatting['alignment'] = 'right'
        elif '左对齐' in instructions:
            formatting['alignment'] = 'left'
        elif '两端对齐' in instructions:
            formatting['alignment'] = 'justify'
        
        # 行距
        spacing_match = re.search(r'行距[\u4e00-\u9fa5]*([\d.]+)', instructions)
        if spacing_match:
            formatting['line_spacing'] = float(spacing_match.group(1))
        
        # 字体颜色
        color_match = re.search(r'字体颜色[\u4e00-\u9fa5]*([#\w]+)', instructions)
        if color_match:
            color = color_match.group(1)
            if not color.startswith('#'):
                # 颜色名称映射
                color_map = {
                    '红色': '#FF0000',
                    '蓝色': '#0000FF',
                    '绿色': '#00FF00',
                    '黑色': '#000000',
                    '灰色': '#808080',
                    '橙色': '#FFA500',
                    '紫色': '#800080'
                }
                color = color_map.get(color, '#000000')
            formatting['font_color'] = color
        
        # 边距
        margin_match = re.search(r'边距[\u4e00-\u9fa5]*(\d+)', instructions)
        if margin_match:
            margin_value = float(margin_match.group(1))
            formatting['margin'] = {
                'top': margin_value,
                'bottom': margin_value,
                'left': margin_value,
                'right': margin_value
            }
        
        return formatting if formatting else None


# 创建全局工具实例
document_editor_tool = DocumentEditorTool()


if __name__ == "__main__":
    # 测试工具类
    tool = DocumentEditorTool()
    
    # 测试创建文档
    test_content = """文档编辑工具测试

这是一个测试文档，用于验证文档编辑工具的功能。

主要功能：
1. 创建新文档
2. 更新文档内容
3. 格式化文档
4. 提取文本内容
"""
    
    test_file = "tool_test.docx"
    
    print("测试创建文档...")
    result = tool.create_document(test_content, test_file)
    print(result)
    
    print("\n测试提取文本...")
    result = tool.extract_text(test_file)
    print(f"文本长度: {result.get('length', 0)}")
    
    print("\n测试更新文档...")
    changes = {
        'replace_text': {'测试文档': '示例文档'},
        'add_paragraph': [{'content': '这是通过工具新增的段落'}]
    }
    result = tool.update_document(test_file, changes)
    print(result)
    
    print("\n测试格式化文档...")
    formatting = {
        'font_size': 14,
        'font_name': '微软雅黑',
        'alignment': 'justify',
        'line_spacing': 1.5
    }
    result = tool.format_document(test_file, formatting)
    print(result)
    
    print("\n测试自然语言格式化...")
    result = tool.reformat_document(test_file, "将字体大小设置为12号，使用宋体，居中对齐，行距1.5倍")
    print(result)
    
    print("\n所有测试完成！")