import os
import shutil
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from settings.Define import PathConfig, Params
from settings.logger_manager import get_logger

logger = get_logger(__name__)


def _get_or_create_vector_store(embedding_model, collection_name, persist_dir):
    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_dir,
        )
        count = store._collection.count()
        if count > 0:
            store.similarity_search_with_score("test", k=1)
        return store
    except Exception as e:
        error_msg = str(e)
        if "dimension" in error_msg.lower() or "expecting embedding" in error_msg.lower():
            logger.warning(f"向量库维度不匹配，删除旧库并重建: {error_msg}")
            db_path = PathConfig.DB_DIR
            if db_path.exists():
                shutil.rmtree(str(db_path), ignore_errors=True)
            return Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=persist_dir,
            )
        raise


class Couplet:
    def __init__(self):
        self.couplet_file = PathConfig.COUPLET_FILE
        self.embedding_model = DashScopeEmbeddings(model=Params.DEFAULT_EMBEDDING_MODEL)
        self.collection_name = Params.MAPPING_NODE[Params.COUPLET_NODE]
        self.vector_store = _get_or_create_vector_store(
            self.embedding_model, self.collection_name, PathConfig.DB_DIR
        )

    def load_vector(self) -> bool:
        try:
            lines = []
            with open(self.couplet_file, "r", encoding="utf-8") as file:
                for line in file:
                    logger.debug(line)
                    lines.append(line)
            self.vector_store.add_texts(lines)

            logger.info("向量数据加载完成！")
            return True
        except Exception as e:
            logger.error(f"向量数据库连接失败: {str(e)}")
            return False

    def similarity_search(self, query: str):
        from core.cache.rag_cache import get_rag_cache
        cache = get_rag_cache()

        def embed_func(text):
            return self.embedding_model.embed_query(text)

        # 缓存查询
        cached = cache.search(query, embed_func)
        if cached:
            return cached

        samples = []

        try:
            # RAG检索
            scored_results = self.vector_store.similarity_search_with_score(query, k=10)
            logger.info(f"检索到 {len(scored_results)} 条结果")
            for doc, score in scored_results:
                samples.append(doc.page_content)

            logger.info(f"参考样本数量: {len(samples)}")
            prompt_template = ChatPromptTemplate.from_messages([
                ("system",
                 "你是一个专业的对联大师，你的任务是根据用户给出的上联，设计一个下联。回答时，可以参考下面的参考对联。参考对联：{samples}请用中文回答问题，注意事项：直接给出下联是什么就行，不需要做过多的解释"),
                ("user", "{text}")
            ])

            # RAG增强
            prompt = prompt_template.invoke({"samples": samples, "text": query})
            logger.info(prompt)

            # 保存缓存
            cache.save(query, str(prompt), embed_func)

            return prompt
        except Exception as e:
            logger.error(f"Prompt 构建失败: {str(e)}")
            return None