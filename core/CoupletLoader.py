# 将对联文本加载到向量数据库中。

import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from settings.Define import PathConfig, Params
from settings.logger_manager import get_logger

logger = get_logger(__name__)

# 初始化嵌入模型
embedding_model = DashScopeEmbeddings(model=Params.DEFAULT_EMBEDDING_MODEL)

# 使用 Chroma 作为向量数据库（本地存储，无需额外服务）
vector_store = Chroma(
    collection_name="couplet",
    embedding_function=embedding_model,
    persist_directory=PathConfig.DB_DIR  # 数据持久化目录
)

lines = []
with open(PathConfig.COUPLET_FILE, "r", encoding="utf-8") as file:
    for line in file:
        logger.debug(line)
        lines.append(line)
vector_store.add_texts(lines)

logger.info("向量数据加载完成！")