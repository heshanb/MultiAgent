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
    LOG_DIR = BASE_DIR / "logs"
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
            cls.LOG_DIR
        ]
        for dir_path in directories:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logging.getLogger(__name__).info(f"创建目录: {dir_path}")
                

class Params:
    DEFAULT_CHAT_MODEL="qwen3.7-max"
    DEFAULT_EMBEDDING_MODEL="text-embedding-v1"
    API_KEY="sk-25cd912ecdf3486785dff572b13c1da1"
    API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"

    SUPERVISOR_NODE = "supervisor_node"
    TRAVEL_NODE = "travel_node"
    JOKE_NODE = "joke_node"
    DOCUMENT_NODE = "document_node"
    COUPLET_NODE = "couplet_node"
    CODE_NODE = "code_node"
    OTHER_NODE = "other_node"
    REFLECTION_NODE = "reflection_node"

    NODE_LIST =  ["supervisor", "travel", "joke", "couplet", "document", "code", "other"]
    MAPPING_NODE = {
        SUPERVISOR_NODE: "supervisor",
        TRAVEL_NODE: "travel",
        JOKE_NODE: "joke",
        COUPLET_NODE: "couplet",
        DOCUMENT_NODE: "document",
        CODE_NODE: "code",
        OTHER_NODE: "other"
    }

    DOC_SUPPORTED_FORMATS = {'.docx', '.doc', '.txt', '.xlsx', '.xls', '.pptx', '.ppt'}


PathConfig.ensure_directories()