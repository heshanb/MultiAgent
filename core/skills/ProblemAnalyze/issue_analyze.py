import os
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from settings.Define import Params
from settings.logger_manager import get_logger

logger = get_logger(__name__)


class IssueAnalyze:
    """问题分析类：封装多模态识图、问题分析、错误诊断等能力"""

    def __init__(self):
        self._init_llm()

    def _init_llm(self):
        """初始化分析模型（支持流式）"""
        self.llm = ChatOpenAI(
            model=Params.DEFAULT_CHAT_MODEL,
            temperature=0.05,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
            streaming=True,
        )

        self.llm_vl = ChatOpenAI(
            model=Params.DEFAULT_MULTIMODAL_MODEL,
            temperature=0.05,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
            streaming=True,
        )

        self.client = OpenAI(
            # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
        )


    @staticmethod
    def preprocess_image(pil_img: Image.Image) -> Image.Image:
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        return Image.fromarray(sharp)

    @staticmethod
    def base64_to_image(base64_data: str) -> Image.Image:
        if base64_data.startswith('data:image/'):
            base64_data = base64_data.split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        return Image.open(BytesIO(image_bytes))

    @staticmethod
    def image_to_base64(pil_img: Image.Image) -> str:
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _build_prompt(self, user_query: str) -> str:
        return f"""你是一个专业的问题解答助手，擅长分析各种问题并提供解决方案。

用户问题：{user_query}

分析和解答要求：
1. 仔细理解用户的问题，分析问题的核心需求
2. 提供清晰、有条理的分析步骤
3. 给出具体的解决方案或建议
4. 如果是技术问题，提供详细的排查步骤和解决方法
5. 如果是错误信息，分析可能的原因并给出修复建议
6. 回答要专业、准确，使用自然友好的语言
7. 最后给出总结性的结论"""

    def stream_issue(self, user_query: str, images_data: list = None):
        """流式分析问题（generator），逐字 yield LLM 输出"""
        logger.info(f"stream_issue 开始: {user_query[:80]}")
        has_images = images_data and len(images_data) > 0

        if has_images:
            vl_content = []
            if user_query.strip():
                vl_content.append({"type": "text", "text": f"{user_query}\n\n请分析图片中的内容。"})
            else:
                vl_content.append({"type": "text", "text": "请分析图片中的内容。"})

            for img_data in images_data:
                if img_data.get('data'):
                    try:
                        pil_img = self.base64_to_image(img_data['data'])
                        processed_img = self.preprocess_image(pil_img)
                        processed_base64 = self.image_to_base64(processed_img)
                        vl_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{processed_base64}"}
                        })
                    except Exception as e:
                        logger.error(f"图片处理失败: {str(e)}")

            for chunk in self.llm_vl.stream([HumanMessage(content=vl_content)]):
                if chunk.content:
                    yield chunk.content
        else:
            prompt = self._build_prompt(user_query)
            messages = [{"role": "user", "content": prompt}]
            completion = self.client.chat.completions.create(
                model=Params.DEFAULT_CHAT_MODEL,
                messages=messages,
                extra_body={"enable_thinking": False},
                stream=True,
            )
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content


    def analyze_issue(self, user_query: str, images_data: list = None) -> str:
        """完整分析问题（内部流式收集后返回完整字符串）"""
        logger.info(f"analyze_issue 开始: {user_query[:80]}")
        full = ""
        for chunk in self.stream_issue(user_query, images_data):
            full += chunk
        logger.info(f"分析完成，结果长度: {len(full)}")
        return full