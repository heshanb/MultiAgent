from pathlib import Path
import os
import logging

class PathConfig:
    """
    路径配置定义文件
    统一管理所有项目中的路径，避免硬编码相对路径
    """
    # 基础目录（基于当前脚本所在位置）
    # Define.py 在 settings 目录下，所以需要向上两级到 MultiAgent 目录
    BASE_DIR = Path(__file__).parent.parent.resolve()
    # 各子目录路径
    TEMPLATES_DIR = BASE_DIR / "templates"
    UPLOADS_DIR = BASE_DIR / "uploads"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    RESOURCE_DIR = BASE_DIR / "resource"
    DB_DIR = BASE_DIR / "chroma_db"
    KNOWLEDGE_DB_DIR = BASE_DIR / "drawing_knowledge_db"
    LOG_DIR = BASE_DIR / "logs"
    # 知识库目录
    COMMON_KNOWLEDGE_DIR = RESOURCE_DIR / "common_knowledge"
    # 初始化时自动创建目录（取消注释以启用）
    COUPLET_FILE = os.path.join(RESOURCE_DIR, "couplettest.csv")
    # 验证目录是否存在（可选）
    @classmethod
    def ensure_directories(cls):
        """确保所有必需的目录存在"""
        directories = [
            cls.TEMPLATES_DIR, 
            cls.UPLOADS_DIR,
            cls.OUTPUTS_DIR,
            cls.RESOURCE_DIR,
            cls.DB_DIR,
            cls.LOG_DIR,
            cls.COMMON_KNOWLEDGE_DIR
        ]
        for dir_path in directories:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logging.getLogger(__name__).info(f"创建目录: {dir_path}")
                

class Params:
    DEFAULT_CHAT_MODEL="qwen3.7-plus"
    DEFAULT_EMBEDDING_MODEL="text-embedding-v4"
    # 多模态识图：Qwen - VL - Max
    DEFAULT_MULTIMODAL_MODEL="qwen-vl-max"
    # 文本工具模型：Qwen-Turbo
    DEFAULT_TEXT_TOOL_MODEL="qwen-turbo"
    API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

    SUPERVISOR_NODE = "supervisor_node"
    TRAVEL_NODE = "travel_node"
    JOKE_NODE = "joke_node"
    DOCUMENT_NODE = "document_node"
    COUPLET_NODE = "couplet_node"
    CODE_NODE = "code_node"
    OTHER_NODE = "other_node"
    DRAWING_NODE = "drawing_node"
    PROBLEM_ANALYSIS_NODE = "problem_analysis_node"
    REFLECTION_NODE = "reflection_node"

    NODE_LIST =  ["supervisor", "travel", "joke", "couplet", "document", "code", "drawing", "problem_analysis", "other"]
    MAPPING_NODE = {
        SUPERVISOR_NODE: "supervisor",
        TRAVEL_NODE: "travel",
        JOKE_NODE: "joke",
        COUPLET_NODE: "couplet",
        DOCUMENT_NODE: "document",
        CODE_NODE: "code",
        DRAWING_NODE: "drawing",
        PROBLEM_ANALYSIS_NODE: "problem_analysis",
        OTHER_NODE: "other"
    }

    DOC_SUPPORTED_FORMATS = {'.docx', '.doc', '.txt', '.xlsx', '.xls', '.pptx', '.ppt', '.html', '.htm'}

    # ===================== 图纸助手节点名称 =====================
    RESET_DRAWING_NODE = "reset_drawing"
    PARSE_DRAWING_NODE = "parse_drawing"
    CHECK_DIMENSION_NODE = "check_dimension"
    CALL_TOOLS_NODE = "call_tools"
    GEN_FINAL_ANSWER_NODE = "gen_final_answer"

    # PDF图纸存放路径
    PDF_FOLDER = os.path.join(PathConfig.RESOURCE_DIR, "enginerring_standard_pdf")
    # 矢量图纸存放路径
    DXF_FOLDER = os.path.join(PathConfig.RESOURCE_DIR, "enginerring_drawing_CAD")
    # 图片图纸存放路径
    PIC_FOLDER = os.path.join(PathConfig.RESOURCE_DIR, "enginerring_drawing_pic")
    COLLECTION_NAME = "engineering_standard"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    SEPARATORS = ["\n\n", "\n", "。", "、", " "]


    # ===================== 代码助手节点名称 =====================
    MAPPING_MODEL = {
        "Qwen3.7-Plus": "qwen3.7-plus",
        "DeepSeek-V4-Pro": "deepseek-v4-pro",
        "GLM-5.2-Fast-Preview": "glm-5.2-fast-preview",
    }

    # ===================== Redis 缓存配置 =====================
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_PASSWORD = ""

PathConfig.ensure_directories()