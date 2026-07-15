import os
import re
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
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.retrievers import EnsembleRetriever
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
        # 多模态识图模型
        self.llm_vl = ChatOpenAI(
            model=Params.DEFAULT_MULTIMODAL_MODEL,
            temperature=0.05,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
        )
        # 文本工具模型
        self.llm_text = ChatOpenAI(
            model=Params.DEFAULT_TEXT_TOOL_MODEL,
            temperature=0,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
        )
        self.last_sources: list[dict] = []  # {source_file, content, score}

    # ─────────────── PDF 数据清洗（通用） ───────────────
    # 国标中常见的结构性关键词（用于判断内容是否为纯表头/框架）
    _STRUCTURAL_KEYWORDS = frozenset({
        "表", "图", "类", "组", "系", "代号", "名称", "设备名称", "序号", "类别",
        "主参数", "折算系数", "第二主参数", "附录", "前言", "范围", "规范性引用文件",
        "术语", "定义", "发布", "实施", "单位", "起草", "归口",
    })

    @staticmethod
    def _clean_page_content(page_content: str) -> str | None:
        """清洗单页 PDF 内容，返回清洗后的文本；若页面无有效内容则返回 None

        通用策略（不依赖特定 PDF 结构）：
        1. 过滤空页/纯空白页
        2. 过滤无中文字符的页面（乱码 PDF / 英文扫描版）
        3. 过滤乱码占比过高的页面
        4. 移除孤立的页码/标准号行
        """
        if not page_content or not page_content.strip():
            return None

        cleaned = page_content.strip()
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        if not lines:
            return None

        # 统计中文字符数量
        all_text = "".join(lines)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', all_text)
        if len(chinese_chars) < 5:
            return None

        # 统计乱码字符（不可打印字符 + 高频特殊符号）
        garbled = sum(1 for c in all_text if ord(c) < 32 or ord(c) in {0x3000})
        total_chars = len(all_text)
        if total_chars > 0 and garbled / total_chars > 0.3:
            return None

        # 移除孤立的页码/标准号行（如 "GB/T18003—2024"、"Ⅰ" 等）
        meaningful_lines = [
            line for line in lines
            if not re.match(
                r'^(GB/T\s*\d+[—\-]\d{4}|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+'
                r'|ICS\s*\d|CCS\s*\w+|犐犆犛.*|犌犅.*)$',
                line,
            )
        ]
        if not meaningful_lines:
            return None

        return "\n".join(meaningful_lines)

    @staticmethod
    def _is_meaningful_chunk(chunk_content: str) -> bool:
        """判断分块后的文本是否包含有效信息

        通用策略（不依赖特定 PDF 结构）：
        1. 最小长度阈值
        2. 最少中文字符数
        3. 排除纯数字块
        4. 排除数字行占比过高的块
        5. 排除结构性关键词占比过高的块（纯表头/框架无实质内容）
        """
        if not chunk_content or not chunk_content.strip():
            return False

        text = chunk_content.strip()
        if len(text) < 30:
            return False

        # 必须包含足够的中文字符
        chinese_chars = set(re.findall(r'[\u4e00-\u9fff]', text))
        if len(chinese_chars) < 5:
            return False

        # 过滤纯数字/空白块
        non_space = text.replace("\n", "").replace(" ", "")
        if re.match(r'^[\d]+$', non_space):
            return False

        # 过滤数字行占比过高的块（如 "0\n1\n2\n...\n9"）
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines:
            numeric_lines = sum(1 for l in lines if re.match(r'^\d{1,4}$', l))
            if numeric_lines / len(lines) > 0.5:
                return False

        # 过滤结构性关键词占比过高的块（纯表头/框架，无实质内容）
        meaningful_keywords = chinese_chars - DrawingAssistant._STRUCTURAL_KEYWORDS
        if len(meaningful_keywords) < 3:
            return False

        return True

    def _load_pdfs_to_chroma(self) -> int:
        """从 enginerring_standard_pdf 目录加载所有 PDF 到向量库，返回导入的文档数"""
        pdf_folder = Params.PDF_FOLDER
        if not os.path.isdir(pdf_folder):
            logger.warning(f"PDF 知识库目录不存在: {pdf_folder}")
            return 0

        pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Params.CHUNK_SIZE,
            chunk_overlap=Params.CHUNK_OVERLAP,
            separators=Params.SEPARATORS,
        )
        all_docs = []
        total_pages = 0
        cleaned_pages = 0
        total_chunks = 0
        meaningful_chunks = 0

        for file in os.listdir(pdf_folder):
            if not file.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(pdf_folder, file)
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                total_pages += len(pages)

                clean_pages = []
                for page in pages:
                    cleaned = self._clean_page_content(page.page_content)
                    if cleaned is None:
                        continue
                    page.page_content = cleaned
                    page.metadata["source_file"] = file
                    page.metadata["doc_type"] = "机械制图国标"
                    clean_pages.append(page)

                cleaned_pages += len(clean_pages)
                skipped_pages = len(pages) - len(clean_pages)

                if not clean_pages:
                    logger.warning(f"  跳过 {file}: 无有效内容 ({len(pages)} 页全部为空)")
                    continue

                split_docs = pdf_splitter.split_documents(clean_pages)
                total_chunks += len(split_docs)

                # 分块后二次过滤：移除无意义的块
                before_filter = len(split_docs)
                split_docs = [d for d in split_docs if self._is_meaningful_chunk(d.page_content)]
                after_filter = len(split_docs)
                meaningful_chunks += after_filter

                all_docs.extend(split_docs)
                logger.info(
                    f"  已加载: {file} (页: {len(clean_pages)}/{len(pages)}, "
                    f"块: {after_filter}/{before_filter}, "
                    f"跳过空页: {skipped_pages})"
                )
            except Exception as e:
                logger.warning(f"  加载失败 {file}: {e}")

        if not all_docs:
            logger.warning("未找到任何 PDF 文件，知识库为空")
            return 0

        self.vector_store = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embedding,
            persist_directory=str(PathConfig.KNOWLEDGE_DB_DIR),
            collection_name=Params.COLLECTION_NAME,
        )
        logger.info(
            f"知识库构建完成：{meaningful_chunks} 个有效块（原始 {total_chunks} 个），"
            f"有效页 {cleaned_pages}/{total_pages}"
        )
        return len(all_docs)

    def _init_vector_store(self):
        self.embedding = DashScopeEmbeddings(model=Params.DEFAULT_EMBEDDING_MODEL)
        need_rebuild = False

        try:
            self.vector_store = Chroma(
                collection_name=Params.COLLECTION_NAME,
                embedding_function=self.embedding,
                persist_directory=PathConfig.KNOWLEDGE_DB_DIR,
            )
            count = self.vector_store._collection.count()
            if count > 0:
                self.vector_store.similarity_search_with_score("test", k=1)
                # 检查知识库是否包含 PDF 元数据（有 source_file 才算已导入）
                sample = self.vector_store._collection.get(limit=3, include=["metadatas"])
                has_pdf_meta = any(
                    m and isinstance(m, dict) and m.get("source_file")
                    for m in sample.get("metadatas", [])
                )
                if not has_pdf_meta:
                    logger.info("向量库中缺少 PDF 知识库文档，自动重建...")
                    need_rebuild = True
            else:
                logger.info("向量库为空，自动构建知识库...")
                need_rebuild = True
        except Exception as e:
            error_msg = str(e)
            if "dimension" in error_msg.lower() or "expecting embedding" in error_msg.lower():
                logger.warning(f"向量库维度不匹配，删除旧库并重建: {error_msg}")
                need_rebuild = True
            else:
                raise

        if need_rebuild:
            if Path(PathConfig.KNOWLEDGE_DB_DIR).exists():
                shutil.rmtree(str(PathConfig.KNOWLEDGE_DB_DIR), ignore_errors=True)
            self._load_pdfs_to_chroma()

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)

        # 构建 BM25 关键词检索器（稀疏检索），从向量库中提取所有已存储的文档
        self.bm25_retriever = None
        try:
            all_data = self.vector_store._collection.get(include=["documents", "metadatas"])
            raw_docs = all_data.get("documents", [])
            raw_metas = all_data.get("metadatas", [])
            bm25_docs = []
            for i, text in enumerate(raw_docs):
                if text and text.strip():
                    meta = raw_metas[i] if i < len(raw_metas) else None
                    if meta is None or not isinstance(meta, dict):
                        meta = {}
                    bm25_docs.append(Document(page_content=text, metadata=meta))
            if bm25_docs:
                self.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
                self.bm25_retriever.k = 4
                logger.info(f"BM25 关键词检索器构建完成，文档数: {len(bm25_docs)}")
            else:
                logger.warning("向量库中文档为空，跳过 BM25 构建")
        except Exception as e:
            logger.warning(f"BM25 构建失败，回退到纯向量检索: {e}")

        # 混合检索器：向量语义检索 + BM25 关键词检索
        vector_ret = self.vector_store.as_retriever(search_kwargs={"k": 4})
        if self.bm25_retriever:
            retrievers = [vector_ret, self.bm25_retriever]
            weights = [0.5, 0.5]
        else:
            retrievers = [vector_ret]
            weights = [1.0]
        self.retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)

    def _do_search_standard(self, query: str) -> str:
        """检索知识库并记录来源"""
        docs = self.retriever.invoke(query)
        context_parts = []
        detailed_sources: list[dict] = []
        seen_files: set[str] = set()
        for d in docs:
            src = d.metadata.get("source_file", "") if d.metadata else ""
            context_parts.append(d.page_content)
            if src and src not in seen_files:
                seen_files.add(src)
                score = getattr(d, "score", None)
                detailed_sources.append({
                    "source_file": src,
                    "content": d.page_content[:500],
                    "score": round(float(score), 4) if score is not None else None,
                })
        self.last_sources = detailed_sources
        context = "\n==== 知识库条目 ====\n".join(context_parts)
        return f"国标检索内容：\n{context}"

    def _init_tools(self):
        _this = self
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
            return _this._do_search_standard(query)

        @tool
        def parse_dxf_file(dxf_path: str) -> str:
            """解析DXF矢量图纸，提取图层、尺寸标注、图框、标题栏信息"""
            doc = ezdxf.readfile(dxf_path)  # 读取 DXF 文件
            msp = doc.modelspace()  # 获取模型空间
            info = []
            info.append(f"DXF版本：{doc.dxfversion}")
            info.append(f"图纸单位：{doc.units}")
            dim_texts = []
            for dim in msp.query("DIMENSION"): # 查询所有尺寸标注实体
                if dim.dxf.text:  # 过滤掉空的尺寸
                    dim_texts.append(dim.dxf.text)
            info.append(f"提取尺寸标注列表：{list(set(dim_texts))}")
            # 图层：工程图纸按功能分层，如 0（默认层）、DIMENSIONS（标注层）、HIDDEN（虚线层）、TITLE_BLOCK（标题栏层）。
            layers = [layer.dxf.name for layer in doc.layers]  # 遍历所有图层
            info.append(f"图纸图层：{layers}")
            # 拼接成文本返回给LLM
            return "\n".join(info)

        self.tools = [unit_convert, search_engineering_standard, parse_dxf_file]
        # llm_with_tools — 让 LLM 学会调用工具
        self.llm_with_tools = self.llm_text.bind_tools(self.tools)

    # ─────────────── 图片预处理 ───────────────
    @staticmethod
    def preprocess_image(pil_img: Image.Image) -> Image.Image:
        # 让工程图纸的线条和标注更清晰，提高多模态模型的识别准确率。

        # PIL 图片 → NumPy 数组，RGB通道 → BGR通道（OpenCV 默认通道顺序）
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # 转成灰度图
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        # 高斯滤波：用 3×3 的核大小做平滑处理，核越大越模糊，3×3 是轻量模糊，只去噪不变形。
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # USM 锐化：锐化 = 原图 × 1.5 + 模糊图 × (-0.5) + 0，把模糊掉的边缘信息"加回来"，让线条更锐利
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        # 转换回PIL：下游 LangChain 多模态模型接受 PIL Image 格式
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
        sources: list[dict]

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
        # 强制调用知识库检索工具
        query = state["user_query"]
        tool_res = self._do_search_standard(query)
        state["messages"].append(ToolMessage(content=tool_res, tool_call_id="forced_tool_call"))
        return state

    def _final_answer_node(self, state: DrawingState) -> DrawingState:
        info = state["drawing_full_info"]
        history = state["messages"]
        query = state["user_query"]
        # 从历史消息中提取知识库检索结果
        kb_content = ""
        for msg in history:
            if isinstance(msg, ToolMessage) and "国标检索内容" in msg.content:
                kb_content = msg.content
                break
        if info and info.strip():
            prompt = (
                f"# 图纸完整解析数据\n{info}\n\n"
                f"# 知识库检索结果\n{kb_content}\n\n"
                f"# 用户提问\n{query}\n\n"
                "请严格基于知识库检索结果回答，不要自由发挥。以机械工程师身份输出完整回答，分模块：图纸信息、尺寸分析、国标依据、加工建议；简洁专业。"
            )
        else:
            prompt = (
                f"# 知识库检索结果\n{kb_content}\n\n"
                f"# 用户提问\n{query}\n\n"
                "请严格基于知识库检索结果回答，不要自由发挥。以机械工程师身份直接回答用户问题，简洁专业。"
            )
        ans = self.llm_text.invoke([HumanMessage(prompt)])
        state["messages"] = [("assistant", ans.content)]
        state["sources"] = list(self.last_sources)
        self.last_sources = []
        return state

    # ─────────────── 路由 ───────────────
    def _route_condition(self, state: DrawingState) -> str:
        if state["is_new_drawing"]:
            return Params.RESET_DRAWING_NODE
        # 没有图纸文件时，跳过解析节点，直接走工具或回答
        if not state["drawing_full_info"] and state.get("file_type", ""):
            return Params.PARSE_DRAWING_NODE
        q = state["user_query"].lower()
        if any(k in q for k in ["尺寸", "公差", "校核", "漏标", "尺寸链"]):
            return Params.CHECK_DIMENSION_NODE
        tools_kw = ["换算", "mm转", "英寸", "国标", "标准件", "螺纹", "公差标准", "标准", "gb",
                    "剖面", "表示法", "视图", "剖视", "断面", "图纸规范", "制图"]
        if any(k in q for k in tools_kw):
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
    ) -> dict:
        """
        调用图纸助手
        Args:
            user_query: 用户问题
            img_path: 图片路径
            pdf_path: PDF 路径
            dxf_path: DXF 路径
            thread_id: 会话 ID（同一图纸多轮对话使用相同 ID）
        Returns:
            {"answer": "AI 回答文本", "sources": ["source_file1.pdf", ...]}
        """
        input_state = {
            "user_query": user_query,
            "drawing_base64": None,
            "dxf_path": None,
            "file_type": "",
            "drawing_full_info": "",
            "messages": [],
            "is_new_drawing": True,
            "sources": [],
        }

        if img_path:
            img = Image.open(img_path)
            img = self.preprocess_image(img)
            input_state["drawing_base64"] = self.img_to_base64(img)
            input_state["file_type"] = "image"
            input_state["drawing_full_info"] = ""
        elif pdf_path:
            b64_list = self.pdf_to_base64_list(pdf_path)
            input_state["drawing_base64"] = b64_list[0]
            input_state["file_type"] = "pdf"
            input_state["drawing_full_info"] = ""
        elif dxf_path:
            input_state["dxf_path"] = dxf_path
            input_state["file_type"] = "dxf"
            input_state["drawing_full_info"] = ""
        else:
            input_state["is_new_drawing"] = False

        config = {"configurable": {"thread_id": thread_id}}
        output = self.graph.invoke(input_state, config=config)
        sources = output.get("sources", [])
        return {"answer": output["messages"][-1].content, "sources": sources}


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
    """兼容旧 API 的便捷函数，只返回回答文本"""
    result = get_drawing_assistant().invoke(user_query, img_path, pdf_path, dxf_path, thread_id)
    return result["answer"] if isinstance(result, dict) else result


# ─────────────── 测试入口 ───────────────
if __name__ == "__main__":
    assistant = DrawingAssistant()

    res = assistant.invoke(
        user_query="解析DXF图层，检查尺寸链是否冲突",
        dxf_path=os.path.join(Params.DXF_FOLDER, "part.dxf"),
        thread_id="session_02",
    )
    print("=== DXF图纸回答 ===")
    print(res["answer"] if isinstance(res, dict) else res)
    if isinstance(res, dict) and res.get("sources"):
        print(f"引用来源: {res['sources']}")