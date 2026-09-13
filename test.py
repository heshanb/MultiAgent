"""
LangChain + Qdrant: 文档切片并存入向量库（混合检索版）
混合检索 = Dense(语义) + Sparse(关键词) 双路召回 + RRF 融合排序
"""
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

from torch.distributed.checkpoint import logger

# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters.character import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams
# from loguru import logger
# # ===================== 配置区 =====================
# # 如果你用的是硅基流动/阿里云百炼等兼容 OpenAI 接口的平台，需要设置 BASE_URL
# os.environ["OPENAI_API_KEY"] = "sk-25cd912ecdf3486785dff572b13c1da1"
# os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # ← 改成你实际的平台地址
#
# QDRANT_URL = "http://localhost:6333"
# COLLECTION_NAME = "my_documents"
# EMBEDDING_MODEL = "text-embedding-v4"  # 阿里云百炼支持，维度 1024
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50
# DOC_PATH = "./cache/sample.txt"
# # ==================================================
#
#
# def init_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
#     """初始化 Qdrant Collection（如已存在则跳过）"""
#     collections = [c.name for c in client.get_collections().collections]
#     if collection_name not in collections:
#         client.create_collection(
#             collection_name=collection_name,
#             vectors_config={
#                 # Dense 向量（语义检索）
#                 "dense": VectorParams(size=vector_size, distance=Distance.COSINE),
#                 # Sparse 向量（关键词检索）—— Qdrant 原生支持，无需指定维度
#             },
#             sparse_vectors_config={
#                 "sparse": {},  # 使用 Qdrant 内置稀疏向量格式
#             },
#         )
#         logger.info(f"✅ Collection '{collection_name}' 创建成功 (dense_dim={vector_size} + sparse)")
#     else:
#         logger.info(f"ℹ️  Collection '{collection_name}' 已存在，跳过创建")
#
#
# def ingest_document(doc_path: str):
#     """主流程：加载 → 切片 → 向量化(Dense+Sparse) → 存入 Qdrant"""
#
#     # Step 1: 加载文档
#     loader = TextLoader(doc_path, encoding="utf-8")
#     documents = loader.load()
#     logger.info(f"📄 加载文档: {len(documents)} 个")
#
#     # Step 2: 文本切片
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", "。", ".", " ", ""],
#     )
#     chunks = splitter.split_documents(documents)
#     logger.info(f"✂️  切片完成: {len(chunks)} 个 chunks")
#
#     # Step 3: 初始化 Dense Embedding（语义向量）
#     dense_embeddings = OpenAIEmbeddings(
#         model=EMBEDDING_MODEL,
#         api_key=os.environ["OPENAI_API_KEY"],
#         check_embedding_ctx_length=False,  # ← 加这个！禁止 tokenize，直接发原文
#     )
#     logger.info("已初始化 Dense Embedding（语义向量）")
#
#     # Step 4: 初始化 Sparse Embedding（关键词向量，本地运行，无需 API）
#     # FastEmbedSparse 使用 BM25/SPLADE 等算法在本地生成稀疏向量
#     sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
#
#     logger.info("初始化 QdrantClient...")
#     # Step 5: 初始化 Qdrant 并确保 Collection 存在
#     client = QdrantClient(url=QDRANT_URL, timeout=120)
#
#     logger.info("初始化 Qdrant Collection...")
#     init_qdrant_collection(client, COLLECTION_NAME, vector_size=1024)
#
#     # Step 6: 创建混合检索 VectorStore
#     vectorstore = QdrantVectorStore(
#         client=client,
#         collection_name=COLLECTION_NAME,
#         embedding=dense_embeddings,          # Dense: 语义检索
#         sparse_embedding=sparse_embeddings,   # Sparse: 关键词检索
#         retrieval_mode=RetrievalMode.HYBRID,  # 混合模式 = Dense + Sparse + RRF 融合
#         vector_name="dense",
#         sparse_vector_name="sparse",
#     )
#
#     # Step 7: 批量写入 Qdrant
#     ids = vectorstore.add_documents(documents=chunks, batch_size=100)
#     logger.info(f"✅ 成功写入 {len(ids)} 条向量到 Qdrant (Dense + Sparse)")
#
#     return vectorstore
#
#
# def verify_search(vectorstore, query: str, k: int = 3):
#     """验证混合检索效果"""
#     results = vectorstore.similarity_search_with_score(query, k=k)  # ← 修复：补全方法名
#     logger.info(f"\n🔍 查询: '{query}'")
#     logger.info("-" * 60)
#     for i, (doc, score) in enumerate(results, 1):
#         logger.info(f"[{i}] Score: {score:.4f}")
#         logger.info(f"    Content: {doc.page_content[:150]}...")
#         logger.info(f"    Metadata: {doc.metadata}")
#
# def verify_hybrid(vectorstore, query: str, k: int = 3):
#     """验证混合检索是否真正生效"""
#     # 1. 检查写入的点是否同时有 dense + sparse 向量
#     points, _ = vectorstore.client.scroll(
#         collection_name=vectorstore.collection_name,
#         limit=1,
#         with_vectors=True,  # ← 关键：返回向量内容
#     )
#     if not points:
#         logger.error("❌ Collection 为空，无法验证")
#         return
#
#     vec_names = list(points[0].vector.keys()) if isinstance(points[0].vector, dict) else ["default"]
#     logger.info(f"📊 存储的向量类型: {vec_names}")
#
#     has_dense = "dense" in vec_names
#     has_sparse = "sparse" in vec_names
#
#     if has_dense and has_sparse:
#         logger.info("✅ 数据层确认：Dense + Sparse 双路向量均已写入")
#     elif has_dense and not has_sparse:
#         logger.warning("⚠️  仅有 Dense 向量！Sparse 写入失败，当前实际是纯语义检索")
#     else:
#         logger.error(f"❌ 异常向量配置: {vec_names}")
#
#     # 2. 执行混合检索并观察日志
#     logger.info(f"\n🔍 混合检索查询: '{query}'")
#     results = vectorstore.similarity_search_with_score(query, k=k)
#     for i, (doc, score) in enumerate(results, 1):
#         logger.info(f"[{i}] RRF Score: {score:.4f} | {doc.page_content[:100]}...")

if __name__ == "__main__":
    # ⚠️ 首次运行前，先删除旧的 Collection（因为向量配置变了）
    # 在终端执行: curl -X DELETE http://localhost:6333/collections/my_documents

    # # 执行入库
    # vs = ingest_document(DOC_PATH)
    #
    # # 验证混合检索
    # # verify_search(vs, query="Qdrant在100万向量下的P99延迟是多少")
    # verify_hybrid(vs, query="Qdrant在100万向量下的P99延迟是多少")

    print('开始导入 CrossEncoder')
    from sentence_transformers import CrossEncoder
    print('CrossEncoder 导入完成')
    print('开始初始化 CrossEncoder')
    reranker = CrossEncoder(
        'D:/Python_project/MultiAgent/models/bge-reranker-large',
        max_length=512,
        device='cpu'
    )
    print('CrossEncoder 初始化完成')

    # 使用 reranker 预测
    scores = reranker.predict([['query', 'doc1'], ['query', 'doc2']])
    print(scores)
