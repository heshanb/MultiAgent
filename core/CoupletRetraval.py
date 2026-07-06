# 对联数据RAG
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from settings.Define import PathConfig, Params
from settings.logger_manager import get_logger

logger = get_logger(__name__)

query = "帮我对个对联，上联：瑞雪兆丰年"

embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

# 使用 Chroma 作为向量数据库（本地存储，无需额外服务）
vector_store = Chroma(
    collection_name="couplet",
    embedding_function=embedding_model,
    persist_directory=PathConfig.DB_DIR  # 数据持久化目录
)

samples=[]

scored_results = vector_store.similarity_search_with_score(query, k=10)
for doc, score in scored_results:
    samples.append(doc.page_content)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
        你是一个专业的对联大师，你的任务是根据用户给出的上联，设计一个下联。
        回答时，可以参考下面的参考对联。
        参考对联：
            {samples}
        请用中文回答问题
    """),
    ("user", "{text}")
])
prompt = prompt_template.invoke({"samples": samples, "text": query})
logger.info(prompt)

llm = ChatTongyi(
    model=Params.DEFAULT_CHAT_MODEL,
)

response = llm.invoke(prompt)
logger.info(response.content)