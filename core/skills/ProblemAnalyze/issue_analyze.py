import os
import base64
from io import BytesIO

# 懒加载重型库
_cv2 = None
_np = None
_Image = None
_OpenAI = None
_ChatOpenAI = None
_HumanMessage = None

def _get_heavy_libs():
    """懒加载重型库"""
    global _cv2, _np, _Image, _OpenAI, _ChatOpenAI, _HumanMessage
    if _cv2 is None:
        import cv2
        import numpy as np
        from PIL import Image
        from openai import OpenAI
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        _cv2 = cv2
        _np = np
        _Image = Image
        _OpenAI = OpenAI
        _ChatOpenAI = ChatOpenAI
        _HumanMessage = HumanMessage
    return _cv2, _np, _Image, _OpenAI, _ChatOpenAI, _HumanMessage

from settings.Define import Params
from settings.logger_manager import get_logger

logger = get_logger(__name__)


class IssueAnalyze:
    """问题分析类：封装多模态识图、问题分析、错误诊断等能力"""

    def __init__(self):
        self._init_llm()

    def _init_llm(self):
        """初始化分析模型（支持流式）"""
        _, _, _, OpenAI, ChatOpenAI, _ = _get_heavy_libs()
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
    def preprocess_image(pil_img):
        # PIL（Pillow）库读取的图片默认是 RGB 格式，而 OpenCV 默认使用 BGR 格式。需要将RGB通道转换为BGR通道，以便后续使用 OpenCV 函数处理时颜色不会错乱
        cv2, np, Image, _, _, _ = _get_heavy_libs()
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # 转换为灰度图
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # 使用 3x3 的内核对灰度图进行高斯模糊，这通常用于去除噪声
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # 非锐化掩蔽（Unsharp Masking） 的一种变体，用于图像锐化（Sharpening），增强图像边缘细节，使图片看起来更清晰
        # 公式: sharp = 1.5 * gray - 0.5 * blur
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        # 将处理后的 numpy 数组转回 PIL Image 对象
        return Image.fromarray(sharp)

    @staticmethod
    def base64_to_image(base64_data: str):
        _, _, Image, _, _, _ = _get_heavy_libs()
        if base64_data.startswith('data:image/'):
            base64_data = base64_data.split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        return Image.open(BytesIO(image_bytes))

    @staticmethod
    def image_to_base64(pil_img) -> str:
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
                vl_content.append({"type": "text", "text": f"{user_query}\n\n请仔细分析图片中的所有内容，包括文字、图标、按钮、颜色标记（如红色框、高亮区域等）、界面布局等视觉元素。详细描述你看到的内容，并根据用户的问题给出专业的分析和解答。"})
            else:
                vl_content.append({"type": "text", "text": "请仔细分析图片中的所有内容，包括文字、图标、按钮、颜色标记（如红色框、高亮区域等）、界面布局等视觉元素。详细描述你看到的内容。"})

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

            _, _, _, _, _, HumanMessage = _get_heavy_libs()
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