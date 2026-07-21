import os
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from settings.Define import Params
from settings.logger_manager import get_logger

logger = get_logger(__name__)


但是
class IssueAnalyze:
    """问题分析类：封装多模态识图、问题分析、错误诊断等能力"""

    def __init__(self):
        self._init_llm()

    def _init_llm(self):
        """初始化多模态识图模型"""
        self.llm_vl = ChatOpenAI(
            model=Params.DEFAULT_MULTIMODAL_MODEL,
            temperature=0.05,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
        )

    @staticmethod
    def preprocess_image(pil_img: Image.Image) -> Image.Image:
        """图片预处理：提高多模态模型的识别准确率
        
        处理步骤：
        1. 转换为灰度图
        2. 高斯滤波去噪
        3. USM锐化增强边缘
        """
        # PIL 图片 → NumPy 数组，RGB通道 → BGR通道（OpenCV 默认通道顺序）
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # 转成灰度图
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # 高斯滤波：用 3×3 的核大小做平滑处理
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # USM 锐化：锐化 = 原图 × 1.5 + 模糊图 × (-0.5) + 0
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        # 转换回PIL格式
        return Image.fromarray(sharp)

    @staticmethod
    def base64_to_image(base64_data: str) -> Image.Image:
        """将base64数据转换为PIL图片"""
        if base64_data.startswith('data:image/'):
            # 去掉 data:image/xxx;base64, 前缀
            base64_data = base64_data.split(',')[1]
        
        image_bytes = base64.b64decode(base64_data)
        return Image.open(BytesIO(image_bytes))

    @staticmethod
    def image_to_base64(pil_img: Image.Image) -> str:
        """将PIL图片转换为base64字符串"""
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def analyze_issue(self, user_query: str, images_data: list = None) -> str:
        """分析用户问题（支持图片输入）
        
        Args:
            user_query: 用户问题文本
            images_data: 图片数据列表，每个元素包含 'data' 字段（base64格式）
        
        Returns:
            分析结果文本
        """
        logger.info(f"开始分析问题: {user_query[:100]}")
        logger.info(f"图片数量: {len(images_data) if images_data else 0}")
        
        has_images = images_data and len(images_data) > 0
        
        if has_images:
            # 构建多模态消息
            message_content = []
            
            # 添加文字内容
            if user_query.strip():
                message_content.append({
                    "type": "text", 
                    "text": f"{user_query}\n\n请分析图片中的内容。"
                })
            else:
                message_content.append({
                    "type": "text", 
                    "text": "请分析图片中的内容。"
                })
            
            # 处理并添加图片
            for img_data in images_data:
                if img_data.get('data'):
                    try:
                        # 转换为PIL图片
                        pil_img = self.base64_to_image(img_data['data'])
                        # 预处理图片
                        processed_img = self.preprocess_image(pil_img)
                        # 转换回base64
                        processed_base64 = self.image_to_base64(processed_img)
                        
                        # 使用 image_url 格式（模型支持的格式）
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{processed_base64}"
                            }
                        })
                        logger.info("图片预处理完成")
                    except Exception as e:
                        logger.error(f"图片处理失败: {str(e)}")
            
            logger.info(f"构建多模态消息，包含 {len(message_content)} 个元素")
            
            # 调用多模态模型
            response = self.llm_vl.invoke([HumanMessage(content=message_content)])
        else:
            # 纯文本分析
            prompt = f"""你是一个专业的问题解答助手，擅长分析各种问题并提供解决方案。

用户问题：{user_query}

分析和解答要求：
1. 仔细理解用户的问题，分析问题的核心需求
2. 提供清晰、有条理的分析步骤
3. 给出具体的解决方案或建议
4. 如果是技术问题，提供详细的排查步骤和解决方法
5. 如果是错误信息，分析可能的原因并给出修复建议
6. 回答要专业、准确，使用自然友好的语言
7. 最后给出总结性的结论"""
            
            response = self.llm_vl.invoke([{"role": "user", "content": prompt}])
        
        content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"分析完成，结果长度: {len(content)}")
        
        return content