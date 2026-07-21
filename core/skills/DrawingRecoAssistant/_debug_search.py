"""诊断脚本：检查知识库检索质量"""
import sys
sys.path.insert(0, "/")

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
from settings.Define import PathConfig, Params

embedding = DashScopeEmbeddings(model=Params.DEFAULT_EMBEDDING_MODEL)
vs = Chroma(
    collection_name=Params.COLLECTION_NAME,
    embedding_function=embedding,
    persist_directory=str(PathConfig.KNOWLEDGE_DB_DIR),
)

print(f"=== 向量库状态 ===")
print(f"文档总数: {vs._collection.count()}")

queries = [
    "GB/T 18003 人造板机械设备型号编制方法",
    "GBT 18003",
    "18003",
    "人造板机械",
]

for q in queries:
    print(f"\n=== 查询: {q} ===")
    results = vs.similarity_search_with_score(q, k=3)
    for i, (doc, score) in enumerate(results):
        src = doc.metadata.get("source_file", "?")
        print(f"  [{i+1}] score={score:.4f} | source={src}")
        print(f"       content: {doc.page_content[:150]}...")

# 检查所有文档的 source_file
print(f"\n=== 所有文档来源 ===")
all_data = vs._collection.get(include=["metadatas"])
sources = set()
for m in all_data.get("metadatas", []):
    if m and isinstance(m, dict):
        sources.add(m.get("source_file", "?"))
    else:
        sources.add(f"NoneType:{m}")
for s in sorted(sources):
    print(f"  {s}")