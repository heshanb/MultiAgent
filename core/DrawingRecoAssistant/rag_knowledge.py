"""
国标 PDF 批量导入向量库代码（RAG 知识库构建）
"""

import os

from langchain_classic import text_splitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma  import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from settings.Define import Params, PathConfig


# 1.初始化嵌入模型
Embeddings = DashScopeEmbeddings(
    model=Params.DEFAULT_EMBEDDING_MODEL
)

# 2.文本分割器
Text_splitters = RecursiveCharacterTextSplitter(
    chunk_size = Params.CHUNK_SIZE,  # 分割后的文本段的最大长度
    chunk_overlap = Params.CHUNK_OVERLAP,
    separators = Params.SEPARATORS
)


def load_standard_pdfs():
    all_docs = []
    # 遍历文件夹内的所有PDF文件
    for file in os.listdir(Params.PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(Params.PDF_FOLDER, file)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            # 给文档增加来源元数据
            for page in pages:
                page.metadata["source_file"] = file
                page.metadata["doc_type"] = "机械制图国标"
            split_docs = Text_splitters.split_documents(pages)
            all_docs.extend(split_docs)

    # 写入向量库并持久化
    vector_db = Chroma.from_documents(
        documents=all_docs,
        embedding=Embeddings,
        persist_directory=PathConfig.KNOWLEDGE_DB_DIR,
        collection_name=Params.COLLECTION_NAME,
    )
    return vector_db


if __name__ == "__main__":
    # 加载pdf文档，并创建向量检索器
    vector_db = load_standard_pdfs()
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    # 问题
    test_query = "机件上斜面实形的表示法"
    # 向量检索，检索相关的片段
    res = retriever.invoke(test_query)
    print(f"检索测试结果：", res[0].page_content[:300])