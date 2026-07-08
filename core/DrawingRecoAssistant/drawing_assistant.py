import os
import base64
import shutil
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import ezdxf
from PIL import Image
from pdf2image import convert_from_path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings.Define import Params, PathConfig
from settings.logger_manager import get_logger

logger = get_logger(__name__)


class DrawingAssistant:
    """图纸识别助手：封装多模态识图、DXF 解析、尺寸校核、国标检索等能力"""

    def __init__(self):
        self._init_llms()
        self._init_vector_store()
        self._init_tools()
        self._init_graph()

    # ─────────────── 初始化 ───────────────
    def _init_llms(self):
        self.llm_vl = ChatOpenAI(
            model=Params.DEFAULT_MULTIMODAL_MODEL,
            temperature=0.05,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
        )
        self.llm_text = ChatOpenAI(
            model=Params.DEFAULT_TEXT_TOOL_MODEL,
            temperature=0,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
        )

    def _init_vector_store(self):
        self.embedding = DashScopeEmbeddings(model=Params.DEFAULT_EMBEDDING_MODEL)
        try:
            self.vector_store = Chroma(
                collection_name=Params.COLLECTION_NAME,
                embedding_function=self.embedding,
                persist_directory=PathConfig.KNOWLEDGE_DB_DIR,
            )
            count = self.vector_store._collection.count()
            if count > 0:
                self.vector_store.similarity_search_with_score("test", k=1)
        except Exception as e:
            error_msg = str(e)
            if "dimension" in error_msg.lower() or "expecting embedding" in error_msg.lower():
                logger.warning(f"向量库维度不匹配，删除旧库并重建: {error_msg}")
                if Path(PathConfig.KNOWLEDGE_DB_DIR).exists():
                    shutil.rmtree(str(PathConfig.KNOWLEDGE_DB_DIR), ignore_errors=True)
                self.vector_store = Chroma(
                    collection_name=Params.COLLECTION_NAME,
                    embedding_function=self.embedding,
                    persist_directory=PathConfig.KNOWLEDGE_DB_DIR,
                )
            else:
                raise
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)

    def _init_tools(self):
        retriever = self.retriever

        @tool
        def unit_convert(value: float, unit_from: str, unit_to: str) -> str:
            """图纸单位换算 mm/cm/m/inch"""
            unit_base = {"mm": 1, "cm": 10, "m": 1000, "inch": 25.4}
            uf = unit_from.lower().strip()
            ut = unit_to.lower().strip()
            base_mm = value * unit_base[uf]
            res = base_mm / unit_base[ut]
            return f"换算结果：{value} {unit_from} = {res:.4f} {unit_to}"

        @tool
        def search_engineering_standard(query: str) -> str:
            """检索机械/建筑国标、螺纹、公差、标准件知识库"""
            docs = retriever.invoke(query)
            context = "\n==== 知识库条目 ====\n".join([d.page_content for d in docs])
            return f"国标检索内容：\n{context}"

        @tool
        def parse_dxf_file(dxf_path: str) -> str:
            """解析DXF矢量图纸，提取图层、尺寸标注、图框、标题栏信息"""
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            info = []
            info.append(f"DXF版本：{doc.dxfversion}")
            info.append(f"图纸单位：{doc.units}")
            dim_texts = []
            for dim in msp.query("DIMENSION"):
                if dim.dxf.text:
                    dim_texts.append(dim.dxf.text)
            info.append(f"提取尺寸标注列表：{list(set(dim_texts))}")
            layers = [layer.dxf.name for layer in doc.layers]
            info.append(f"图纸图层：{layers}")
            return "\n".join(info)

        self.tools = [unit_convert, search_engineering_standard, parse_dxf_file]
        self.llm_with_tools = self.llm_text.bind_tools(self.tools)

    # ─────────────── 图片预处理 ───────────────
    @staticmethod
    def preprocess_image(pil_img: Image.Image) -> Image.Image:
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        return Image.fromarray(sharp)

    @staticmethod
    def img_to_base64(pil_img: Image.Image) -> str:
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def pdf_to_base64_list(pdf_path: str) -> list[str]:
        pages = convert_from_path(pdf_path, dpi=300)
        b64_list = []
        for page in pages:
            page = DrawingAssistant.preprocess_image(page)
            b64_list.append(DrawingAssistant.img_to_base64(page))
        return b64_list

    # ─────────────── 状态定义 ───────────────
    class DrawingState(MessagesState):
        drawing_base64: str | None
        dxf_path: str | None
        file_type: str
        drawing_full_info: str | None
        user_query: str
        is_new_drawing: bool

    # ─────────────── 节点定义 ───────────────
    def _reset_old_drawing_node(self, state: DrawingState) -> DrawingState:
        if state["is_new_drawing"]:
            state["drawing_full_info"] = None
            state["messages"] = []
            logger.info("检测到新图纸，清空历史图纸数据")
        return state

    def _parse_drawing_node(self, state: DrawingState) -> DrawingState:
        file_type = state["file_type"]
        full_info = ""

        if file_type == "dxf" and state["dxf_path"]:
            tool_map = {t.name: t for t in self.tools}
            dxf_res = tool_map["parse_dxf_file"].invoke({"dxf_path": state["dxf_path"]})
            full_info += f"===== DXF矢量解析结果 =====\n{dxf_res}\n"

        if state["drawing_base64"]:
            vision_prompt = (
                "你是资深机械制图工程师，分析工程图纸，结构化提取：\n"
                "1. 标题栏：图号、零件名称、比例、材料、单位、制图日期\n"
                "2. 所有尺寸：线性、直径Φ、半径R、公差±、螺纹规格Mxx\n"
                "3. 视图类型：主/俯/左/剖视图、剖面符号\n"
                "4. 工艺要求：表面粗糙度、热处理、装配间隙\n"
                "5. 标注符号：孔、倒角、沉孔、基准符号\n"
                "输出条理清晰，不要遗漏尺寸数据。"
            )
            msg = HumanMessage(content=[
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{state['drawing_base64']}"}},
            ])
            vision_resp = self.llm_vl.invoke([msg])
            full_info += f"\n===== 视觉识图解析结果 =====\n{vision_resp.content}"

        state["drawing_full_info"] = full_info
        if full_info.strip():
            chunks = self.text_splitter.split_text(full_info)
            if chunks:
                self.vector_store.add_texts(chunks)
        return state

    def _dimension_check_node(self, state: DrawingState) -> DrawingState:
        draw_info = state["drawing_full_info"]
        prompt = (
            f"图纸原始信息：\n{draw_info}\n"
            "任务：尺寸校核工程师\n"
            "1. 检查尺寸链是否闭合、有无漏标/重复尺寸\n"
            "2. 核对公差数值是否符合国标常规区间\n"
            "3. 检查单位混用冲突（mm/inch）\n"
            f"用户问题：{state['user_query']}\n"
            "输出校核风险点与优化建议。"
        )
        resp = self.llm_text.invoke([SystemMessage(content=prompt)])
        state["messages"].append(("assistant", f"【尺寸校核报告】\n{resp.content}"))
        return state

    def _tool_call_node(self, state: DrawingState) -> DrawingState:
        sys_msg = SystemMessage(content=f"图纸基础数据：{state['drawing_full_info']}")
        all_messages = [sys_msg] + state["messages"] + [HumanMessage(state["user_query"])]
        resp = self.llm_with_tools.invoke(all_messages)
        state["messages"].append(resp)
        if resp.tool_calls:
            tool_map = {t.name: t for t in self.tools}
            for call in resp.tool_calls:
                tool_res = tool_map[call["name"]].invoke(call["args"])
                state["messages"].append(ToolMessage(content=tool_res, tool_call_id=call["id"]))
        return state

    def _final_answer_node(self, state: DrawingState) -> DrawingState:
        info = state["drawing_full_info"]
        history = state["messages"]
        query = state["user_query"]
        prompt = (
            f"# 图纸完整解析数据\n{info}\n\n"
            f"# 历史分析记录\n{history}\n\n"
            f"# 用户提问\n{query}\n\n"
            "以机械工程师身份输出完整回答，分模块：图纸信息、尺寸分析、国标依据、加工建议；简洁专业。"
        )
        ans = self.llm_text.invoke([HumanMessage(prompt)])
        state["messages"] = [("assistant", ans.content)]
        return state

    # ─────────────── 路由 ───────────────
    def _route_condition(self, state: DrawingState) -> str:
        if state["is_new_drawing"]:
            return Params.RESET_DRAWING_NODE
        if not state["drawing_full_info"]:
            return Params.PARSE_DRAWING_NODE
        q = state["user_query"].lower()
        if any(k in q for k in ["尺寸", "公差", "校核", "漏标", "尺寸链"]):
            return Params.CHECK_DIMENSION_NODE
        if any(k in q for k in ["换算", "mm转", "英寸", "国标", "标准件", "螺纹", "公差标准"]):
            return Params.CALL_TOOLS_NODE
        return Params.GEN_FINAL_ANSWER_NODE

    # ─────────────── 构建 Graph ───────────────
    def _init_graph(self):
        self.memory = MemorySaver()
        builder = StateGraph(self.DrawingState)

        builder.add_node(Params.RESET_DRAWING_NODE, self._reset_old_drawing_node)
        builder.add_node(Params.PARSE_DRAWING_NODE, self._parse_drawing_node)
        builder.add_node(Params.CHECK_DIMENSION_NODE, self._dimension_check_node)
        builder.add_node(Params.CALL_TOOLS_NODE, self._tool_call_node)
        builder.add_node(Params.GEN_FINAL_ANSWER_NODE, self._final_answer_node)

        builder.set_conditional_entry_point(
            self._route_condition,
            {
                Params.RESET_DRAWING_NODE: Params.RESET_DRAWING_NODE,
                Params.PARSE_DRAWING_NODE: Params.PARSE_DRAWING_NODE,
                Params.CHECK_DIMENSION_NODE: Params.CHECK_DIMENSION_NODE,
                Params.CALL_TOOLS_NODE: Params.CALL_TOOLS_NODE,
                Params.GEN_FINAL_ANSWER_NODE: Params.GEN_FINAL_ANSWER_NODE,
            },
        )

        builder.add_edge(Params.RESET_DRAWING_NODE, Params.PARSE_DRAWING_NODE)
        builder.add_edge(Params.PARSE_DRAWING_NODE, Params.GEN_FINAL_ANSWER_NODE)
        builder.add_edge(Params.CHECK_DIMENSION_NODE, Params.GEN_FINAL_ANSWER_NODE)
        builder.add_edge(Params.CALL_TOOLS_NODE, Params.GEN_FINAL_ANSWER_NODE)
        builder.add_edge(Params.GEN_FINAL_ANSWER_NODE, END)

        self.graph = builder.compile(checkpointer=self.memory)

    # ─────────────── 对外 API ───────────────
    def invoke(
        self,
        user_query: str,
        img_path: str | None = None,
        pdf_path: str | None = None,
        dxf_path: str | None = None,
        thread_id: str = "draw_chat_001",
    ) -> str:
        """
        调用图纸助手
        Args:
            user_query: 用户问题
            img_path: 图片路径
            pdf_path: PDF 路径
            dxf_path: DXF 路径
            thread_id: 会话 ID（同一图纸多轮对话使用相同 ID）
        Returns:
            AI 回答文本
        """
        input_state = {
            "user_query": user_query,
            "drawing_base64": None,
            "dxf_path": None,
            "file_type": "",
            "messages": [],
            "is_new_drawing": True,
        }

        if img_path:
            img = Image.open(img_path)
            img = self.preprocess_image(img)
            input_state["drawing_base64"] = self.img_to_base64(img)
            input_state["file_type"] = "image"
            input_state["drawing_full_info"] = None
        elif pdf_path:
            b64_list = self.pdf_to_base64_list(pdf_path)
            input_state["drawing_base64"] = b64_list[0]
            input_state["file_type"] = "pdf"
            input_state["drawing_full_info"] = None
        elif dxf_path:
            input_state["dxf_path"] = dxf_path
            input_state["file_type"] = "dxf"
            input_state["drawing_full_info"] = None
        else:
            input_state["is_new_drawing"] = False

        config = {"configurable": {"thread_id": thread_id}}
        output = self.graph.invoke(input_state, config=config)
        return output["messages"][-1].content


# ─────────────── 模块级便捷入口（兼容旧调用方式） ───────────────
_assistant_instance: DrawingAssistant | None = None


def get_drawing_assistant() -> DrawingAssistant:
    """获取 DrawingAssistant 单例"""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = DrawingAssistant()
    return _assistant_instance


def run_drawing_assistant(
    user_query: str,
    img_path: str = None,
    pdf_path: str = None,
    dxf_path: str = None,
    thread_id: str = "draw_chat_001",
) -> str:
    """兼容旧 API 的便捷函数"""
    return get_drawing_assistant().invoke(user_query, img_path, pdf_path, dxf_path, thread_id)


# ─────────────── 测试入口 ───────────────
if __name__ == "__main__":
    assistant = DrawingAssistant()

    res = assistant.invoke(
        user_query="解析DXF图层，检查尺寸链是否冲突",
        dxf_path=os.path.join(Params.DXF_FOLDER, "part.dxf"),
        thread_id="session_02",
    )
    print("=== DXF图纸回答 ===")
    print(res)