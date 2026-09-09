import os
import json
import uuid
import shutil
import time
import asyncio
import functools
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from core.auth.auth import get_current_user, security
from core.auth.database import get_db, User
from settings.Define import PathConfig, Params
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import QdrantClient
    from langchain.embeddings import DashScopeEmbeddings
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import QdrantClient
    from langchain_community.embeddings import DashScopeEmbeddings

from core.skills.DocProcess.document_process import doc_processor
from settings.logger_manager import get_logger
from fastembed import SparseTextEmbedding
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

EMBEDDING_DATABASE = os.getenv("EMBEDDING_DATABASE", "chroma").strip().lower()
print(f"EMBEDDING_DATABASE={EMBEDDING_DATABASE}")
QDRANT_URL = "http://127.0.0.1:6333"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1


def convert_vector_score(raw_score: float, vector_store, db_type: str = "chroma") -> float:
    if db_type == "qdrant":
        return raw_score
    try:
        meta = vector_store._collection.metadata or {}
        metric = meta.get("hnsw:space", "l2")
    except Exception:
        metric = "l2"
    if metric == "cosine":
        return 1.0 - raw_score
    elif metric == "ip":
        return raw_score
    else:
        return 1.0 / (1.0 + raw_score)


def _create_qdrant_client():
    return QdrantClient(url=QDRANT_URL, prefer_grpc=False)


def _get_chroma_persist_dir(username: str) -> str:
    persist_dir = PathConfig.DB_DIR / "knowledge" / username
    persist_dir.mkdir(parents=True, exist_ok=True)
    return str(persist_dir)


def retry_qdrant(max_retries=MAX_RETRIES, delay=RETRY_DELAY_BASE):
    if EMBEDDING_DATABASE != "qdrant":
        def decorator(func):
            return func
        return decorator

    def decorator(func):
        import inspect
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(1, max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        msg = str(e)
                        is_retryable = (
                            "10051" in msg
                            or "10053" in msg
                            or "connection" in msg.lower()
                            or "qdrant" in msg.lower()
                            or "unreachable" in msg.lower()
                        )
                        if is_retryable and attempt < max_retries:
                            logger.warning(
                                f"{func.__name__} 第 {attempt}/{max_retries} 次失败，"
                                f"将使用全新 Qdrant 连接重试: {e}"
                            )
                            await asyncio.sleep(delay * attempt)
                        else:
                            raise
                raise last_exception
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(1, max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        msg = str(e)
                        is_retryable = (
                            "10051" in msg
                            or "10053" in msg
                            or "connection" in msg.lower()
                            or "qdrant" in msg.lower()
                            or "unreachable" in msg.lower()
                        )
                        if is_retryable and attempt < max_retries:
                            logger.warning(
                                f"{func.__name__} 第 {attempt}/{max_retries} 次失败，"
                                f"将使用全新 Qdrant 连接重试: {e}"
                            )
                            time.sleep(delay * attempt)
                        else:
                            raise
                raise last_exception
            return sync_wrapper
    return decorator


def get_user_knowledge_dir(username: str) -> Path:
    user_dir = PathConfig.COMMON_KNOWLEDGE_DIR / username
    if not user_dir.exists():
        user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_kb_subdir(username: str, kb_id: str) -> Path:
    subdir = get_user_knowledge_dir(username) / kb_id
    if not subdir.exists():
        subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def get_kb_metadata_path(username: str) -> Path:
    return get_user_knowledge_dir(username) / "kb_metadata.json"


def load_kb_metadata(username: str) -> list:
    meta_path = get_kb_metadata_path(username)
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_kb_metadata(username: str, meta_list: list):
    meta_path = get_kb_metadata_path(username)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)


_embeddings_cache = {}
_sparse_embeddings_cache = {}


def _get_embeddings(username: str):
    if username not in _embeddings_cache:
        embeddings = DashScopeEmbeddings(model=Params.DEFAULT_EMBEDDING_MODEL)
        logger.info("DashScope 向量模型已初始化")
        _embeddings_cache[username] = embeddings
    return _embeddings_cache[username]


def _get_sparse_embeddings(username: str):
    if username not in _sparse_embeddings_cache:
        sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("BM25 稀疏向量模型已初始化")
        _sparse_embeddings_cache[username] = sparse
    return _sparse_embeddings_cache[username]


def _ensure_collection_exists(username: str):
    from qdrant_client.http import models as qdrant_models

    collection_name = f"knowledge_{username}"
    temp_client = _create_qdrant_client()
    try:
        collections = temp_client.get_collections()
        existing_names = [c.name for c in collections.collections]
        if collection_name not in existing_names:
            temp_client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=1024,
                    distance=qdrant_models.Distance.COSINE,
                ),
                sparse_vectors_config={
                    "langchain-sparse": qdrant_models.SparseVectorParams(),
                },
            )
            logger.info(f"已创建 Qdrant 集合: {collection_name}")
    finally:
        temp_client.close()


def _build_vector_store(username: str):
    if EMBEDDING_DATABASE == "qdrant":
        return _build_qdrant_vector_store(username)
    else:
        return _build_chroma_vector_store(username)


def _build_chroma_vector_store(username: str):
    embeddings = _get_embeddings(username)
    persist_dir = _get_chroma_persist_dir(username)
    return Chroma(
        collection_name=f"knowledge_{username}",
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _build_qdrant_vector_store(username: str):
    _ensure_collection_exists(username)
    client = _create_qdrant_client()
    embeddings = _get_embeddings(username)
    sparse_embeddings = _get_sparse_embeddings(username)
    return QdrantVectorStore(
        client=client,
        collection_name=f"knowledge_{username}",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        validate_collection_config=False,
    )


def get_vector_store(username: str):
    return _build_vector_store(username)


@retry_qdrant()
def get_vector_count(username: str) -> int:
    try:
        vector_store = get_vector_store(username)
        return vector_store._collection.count()
    except Exception as e:
        logger.error(f"获取向量数量失败: {str(e)}")
        return 0


def clean_document_content(content: str) -> str:
    import re

    if not content:
        return ""

    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r' {2,}', ' ', content)
    content = re.sub(r'\t{2,}', '\t', content)
    content = content.strip()

    content = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\n\r\t。，！？；：、""\'\'（）()【】[]{}《》<>·…—–\-\s]', '', content)

    content = re.sub(r'([。，！？；：、]){2,}', r'\1', content)

    content = content.replace('\ufeff', '')

    lines = content.split('\n')
    cleaned_lines = []
    prev_line = None
    for line in lines:
        if line != prev_line:
            cleaned_lines.append(line)
            prev_line = line
    content = '\n'.join(cleaned_lines)

    paragraphs = content.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    content = '\n\n'.join(paragraphs)

    return content


@retry_qdrant()
async def chunk_and_store_document(file_path: str, username: str, kb_id: str):
    try:
        logger.info(f"开始处理文档: {file_path}, 用户: {username}, 知识库: {kb_id}")

        content = ""
        async for chunk in doc_processor.read_document(file_path):
            content += chunk

        logger.info(f"文档读取完成，内容长度: {len(content) if content else 0}")

        if not content:
            logger.warning(f"文档内容为空: {file_path}")
            raise ValueError("文档内容为空，无法进行向量化")

        content = clean_document_content(content)
        logger.info(f"文档清洗完成，内容长度: {len(content) if content else 0}")

        if not content:
            logger.warning(f"文档清洗后内容为空: {file_path}")
            return

        is_md = False
        if file_path.endswith(".md"):
            from langchain_text_splitters import MarkdownHeaderTextSplitter

            is_md = True
            # 按 Markdown 标题层级切分
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "H1"),
                    ("##", "H2"),
                    ("###", "H3")
                ]
            )

            sections = md_splitter.split_text(content)

        # 对超长章节进行递归精切
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Params.CHUNK_SIZE, # 根据 Embedding 模型上限调整
            chunk_overlap=Params.CHUNK_OVERLAP, # 保持跨 chunk 的上下文连贯
            separators=Params.SEPARATORS,
        )

        vector_store = get_vector_store(username)
        filename = os.path.basename(file_path)

        if is_md:
            chunks = text_splitter.split_documents(sections)
            # 批量提取文本和元数据，避免循环调用 add_texts 导致网络请求频繁
            texts_to_add = [chunk.page_content for chunk in chunks]
            metadatas_to_add = [
                {**chunk.metadata, "source": filename, "kb_id": kb_id}
                for chunk in chunks
            ]
            if texts_to_add:
                vector_store.add_texts(texts=texts_to_add, metadatas=metadatas_to_add)
        else:
            chunks = text_splitter.split_text(content)
            vector_store.add_texts(
                texts=chunks,
                metadatas=[{"source": filename, "kb_id": kb_id}] * len(chunks)
            )

        for chunk in chunks:
            print("=="*20)
            print(chunk)

        if not chunks:
            logger.warning(f"文档切块结果为空: {file_path}")
            return

        logger.info(f"文档 {file_path} 已成功切块并存储，共 {len(chunks)} 个段落")

    except Exception as e:
        logger.error(f"文档切块存储失败: {str(e)}")
        raise


@retry_qdrant()
def delete_document_vectors(filename: str, username: str, kb_id: str = None):
    try:
        vector_store = get_vector_store(username)

        try:
            all_docs = vector_store._collection.get()
            if all_docs and 'ids' in all_docs and 'metadatas' in all_docs:
                ids_to_delete = []
                for idx, metadata in enumerate(all_docs['metadatas']):
                    if metadata and metadata.get('source') == filename:
                        if kb_id is None or metadata.get('kb_id') == kb_id:
                            ids_to_delete.append(all_docs['ids'][idx])

                if ids_to_delete:
                    vector_store._collection.delete(ids=ids_to_delete)
                    logger.info(f"已删除文档 {filename} 对应的 {len(ids_to_delete)} 个向量")
                else:
                    logger.info(f"未找到文档 {filename} 对应的向量")
            else:
                logger.info("向量数据库为空")
        except Exception as get_error:
            logger.warning(f"使用 _collection.get() 失败: {str(get_error)}")

    except Exception as e:
        logger.error(f"删除向量失败: {str(e)}")

@retry_qdrant()
def delete_kb_vectors(username: str, kb_id: str):
    try:
        vector_store = get_vector_store(username)
        try:
            if EMBEDDING_DATABASE == "qdrant":
                is_deleted = vector_store._collection.delete(
                    filter={
                        "must": [
                            {
                                "key": "kb_id",
                                "match": {
                                    "value": kb_id
                                }
                            }
                        ]
                    }
                )
            else:
                vector_store._collection.delete(where={"kb_id": kb_id})
                is_deleted = True
            if is_deleted:
                logger.info(f"已删除知识库 {kb_id} 对应的所有向量")
            else:
                logger.warning(f"未找到知识库 {kb_id} 对应的向量，或无需删除")
            # all_docs = vector_store._collection.get()
            # if all_docs and 'ids' in all_docs and 'metadatas' in all_docs:
            #     ids_to_delete = []
            #     for idx, metadata in enumerate(all_docs['metadatas']):
            #         if metadata and metadata.get('kb_id') == kb_id:
            #             ids_to_delete.append(all_docs['ids'][idx])
            #
            #     if ids_to_delete:
            #         vector_store._collection.delete(ids=ids_to_delete)
            #         logger.info(f"已删除知识库 {kb_id} 对应的 {len(ids_to_delete)} 个向量")
        except Exception as e:
            logger.warning(f"删除知识库向量失败: {str(e)}")
    except Exception as e:
        logger.error(f"删除知识库向量异常: {str(e)}")


@router.get("/bases")
async def list_knowledge_bases(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_dir = get_user_knowledge_dir(current_user.username)

    needs_migration = False
    if not load_kb_metadata(current_user.username):
        for item in user_dir.iterdir():
            if item.is_file() and not item.name.startswith('.') and item.name != 'kb_metadata.json':
                needs_migration = True
                break

    if needs_migration:
        migrator = OldDataMigrator()
        await migrator.migrate(current_user.username, user_dir)

    meta_list = load_kb_metadata(current_user.username)

    result = []
    for kb in meta_list:
        kb_id = kb['id']
        subdir = get_user_knowledge_dir(current_user.username) / kb_id
        doc_count = 0
        if subdir.exists():
            doc_count = sum(
                1 for f in subdir.iterdir()
                if f.is_file() and not f.name.startswith('~$') and not f.name.startswith('.')
            )
        result.append({
            "id": kb_id,
            "name": kb.get('name', '未命名知识库'),
            "description": kb.get('description', ''),
            "created_at": kb.get('created_at', 0),
            "doc_count": doc_count
        })

    return {"bases": result}


class OldDataMigrator:
    async def migrate(self, username: str, user_dir: Path):
        meta_list = load_kb_metadata(username)
        if meta_list:
            return

        old_files = []
        for item in user_dir.iterdir():
            if item.is_file() and not item.name.startswith('.') and item.name != 'kb_metadata.json':
                old_files.append(item)

        if not old_files:
            return

        kb_id = f"kb_{uuid.uuid4().hex[:12]}"
        kb_name = '我的知识库'

        existing_meta = load_kb_metadata(username)
        old_kb_meta_path = user_dir / 'kb_meta_legacy.json'
        if old_kb_meta_path.exists():
            try:
                with open(old_kb_meta_path, 'r', encoding='utf-8') as f:
                    legacy = json.load(f)
                    if legacy.get('name'):
                        kb_name = legacy['name']
                    legacy_desc = legacy.get('description', '')
            except Exception:
                legacy_desc = ''
        else:
            legacy_desc = ''

        existing_meta.append({
            "id": kb_id,
            "name": kb_name,
            "description": legacy_desc,
            "created_at": time.time()
        })
        save_kb_metadata(username, existing_meta)

        subdir = get_kb_subdir(username, kb_id)

        for file_path in old_files:
            try:
                dest = subdir / file_path.name
                if dest.exists():
                    dest.unlink()
                shutil.move(str(file_path), str(dest))
            except Exception as e:
                logger.error(f"迁移文件失败 {file_path.name}: {str(e)}")
                continue

        try:
            for file_path in old_files:
                dest = subdir / file_path.name
                if dest.exists():
                    try:
                        await chunk_and_store_document(str(dest), username, kb_id)
                    except Exception as e:
                        logger.warning(f"重新向量化失败 {file_path.name}: {str(e)}")
        except Exception as e:
            logger.error(f"迁移时重新向量化失败: {str(e)}")

        logger.info(f"已将 {len(old_files)} 个旧文件迁移到知识库 {kb_id}")


@router.post("/bases")
async def create_knowledge_base(
    name: str = Query(..., description="知识库名称"),
    description: str = Query("", description="知识库描述"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="知识库名称不能为空")

    meta_list = load_kb_metadata(current_user.username)
    for kb in meta_list:
        if kb['name'] == name.strip():
            raise HTTPException(status_code=400, detail="已存在同名知识库")

    kb_id = f"kb_{uuid.uuid4().hex[:12]}"
    meta_list.append({
        "id": kb_id,
        "name": name.strip(),
        "description": description.strip(),
        "created_at": time.time()
    })
    save_kb_metadata(current_user.username, meta_list)

    get_kb_subdir(current_user.username, kb_id)

    return {"id": kb_id, "name": name.strip(), "description": description.strip()}


@router.put("/bases/{kb_id}")
async def update_knowledge_base(
    kb_id: str,
    name: str = Query(None),
    description: str = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    meta_list = load_kb_metadata(current_user.username)
    found = False
    for kb in meta_list:
        if kb['id'] == kb_id:
            if name is not None:
                if not name.strip():
                    raise HTTPException(status_code=400, detail="知识库名称不能为空")
                kb['name'] = name.strip()
            if description is not None:
                kb['description'] = description.strip()
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="知识库不存在")

    save_kb_metadata(current_user.username, meta_list)
    return {"message": "更新成功"}


@router.delete("/bases/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    meta_list = load_kb_metadata(current_user.username)
    meta_list = [kb for kb in meta_list if kb['id'] != kb_id]
    save_kb_metadata(current_user.username, meta_list)

    delete_kb_vectors(current_user.username, kb_id)

    subdir = get_user_knowledge_dir(current_user.username) / kb_id
    if subdir.exists():
        shutil.rmtree(subdir, ignore_errors=True)
        logger.info(f"已删除知识库目录: {subdir}")

    return {"message": "知识库已删除"}


@router.get("/documents")
async def list_documents(
    kb_id: str = Query(..., description="知识库ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subdir = get_user_knowledge_dir(current_user.username) / kb_id
    if not subdir.exists():
        return {"documents": [], "kb_id": kb_id}

    documents = []
    for file_path in subdir.iterdir():
        if file_path.is_file():
            filename = file_path.name
            if not (filename.startswith('~$') or filename.startswith('.') or filename == 'Thumbs.db'):
                documents.append({
                    "filename": filename,
                    "size": file_path.stat().st_size,
                    "modified_at": file_path.stat().st_mtime
                })

    return {"documents": documents, "kb_id": kb_id}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Query(..., description="知识库ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subdir = get_kb_subdir(current_user.username, kb_id)

    allowed_extensions = Params.DOC_SUPPORTED_FORMATS | {'.md', '.pdf'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式，支持的格式: {', '.join(allowed_extensions)}")

    file_path = subdir / file.filename
    try:
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)

        logger.info(f"用户 {current_user.username} 上传文档到知识库 {kb_id}: {file.filename}")

        try:
            await chunk_and_store_document(str(file_path), current_user.username, kb_id)
            return {"status": "success", "message": "文档上传成功", "filename": file.filename}
        except Exception as chunk_error:
            logger.warning(f"文档向量化失败，但文件已保存: {str(chunk_error)}")
            return {
                "status": "warning",
                "message": "文档上传成功，但无法提取文本内容（可能是扫描件或加密文件），该文档暂无法用于智能问答",
                "filename": file.filename,
                "warning": "无法提取文本内容，文档暂无法用于智能问答"
            }

    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@router.delete("/delete-doc")
async def delete_document(
    kb_id: str = Query(..., description="知识库ID"),
    filename: str = Query(..., description="文件名"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subdir = get_user_knowledge_dir(current_user.username) / kb_id
    file_path = subdir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文档不存在")

    try:
        file_path.unlink()
        delete_document_vectors(filename, current_user.username, kb_id)
        logger.info(f"用户 {current_user.username} 从知识库 {kb_id} 删除文档: {filename}")
        return {"message": "文档删除成功", "filename": filename}
    except Exception as e:
        logger.error(f"文件删除失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")


@router.get("/files/{kb_id}/{filename}")
async def get_knowledge_file(
    kb_id: str,
    filename: str,
    token: str = None,
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    from fastapi.responses import FileResponse

    current_user = None

    if token:
        from core.auth.auth import decode_access_token
        payload = decode_access_token(token)
        if payload:
            username = payload.get("sub")
            if username:
                current_user = db.query(User).filter(User.username == username).first()

    if not current_user and credentials:
        from core.auth.auth import decode_access_token
        payload = decode_access_token(credentials.credentials)
        if payload:
            username = payload.get("sub")
            if username:
                current_user = db.query(User).filter(User.username == username).first()

    if not current_user:
        raise HTTPException(status_code=401, detail="需要登录")

    subdir = get_user_knowledge_dir(current_user.username) / kb_id
    file_path = subdir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    ext = file_path.suffix.lower()
    content_type = "application/octet-stream"
    inline_filename = None

    if ext in (".html", ".htm"):
        content_type = "text/html; charset=utf-8"
        inline_filename = filename
    elif ext == ".txt":
        content_type = "text/plain; charset=utf-8"
        inline_filename = filename
    elif ext == ".md":
        content_type = "text/markdown; charset=utf-8"
        inline_filename = filename

    if inline_filename:
        from urllib.parse import quote
        encoded_filename = quote(inline_filename, encoding='utf-8')
        return FileResponse(
            path=str(file_path),
            media_type=content_type,
            headers={"Content-Disposition": f"inline; filename=\"{encoded_filename}\"; filename*=UTF-8''{encoded_filename}"}
        )
    else:
        return FileResponse(
            path=str(file_path),
            media_type=content_type,
            filename=filename
        )


@retry_qdrant()
@router.get("/search")
async def search_knowledge(
    query: str,
    kb_id: str = Query(None, description="知识库ID，不传则搜索全部"),
    k: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        vector_store = get_vector_store(current_user.username)

        logger.info(f"用户：{current_user.username} 召回测试提问: {query}  知识库ID：{kb_id}")
        if kb_id:
            raw_results = vector_store.similarity_search_with_score(
                query, k=k,
                filter={"kb_id": kb_id},
            )
        else:
            raw_results = vector_store.similarity_search_with_score(query, k=k)

        documents = []
        for doc, raw in raw_results:
            score = convert_vector_score(raw, vector_store, EMBEDDING_DATABASE)
            if score < 0.2:
                continue
            documents.append({
                "content": doc.page_content,
                "source": doc.metadata.get('source', ''),
                "kb_id": doc.metadata.get('kb_id', ''),
                "score": float(score)
            })

        documents.sort(key=lambda x: x['score'], reverse=True)
        for i, doc in enumerate(documents):
            doc['rank'] = i + 1

        return {"results": documents}

    except Exception as e:
        logger.error(f"知识库搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@router.post("/reindex/{kb_id}")
async def reindex_knowledge_base(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subdir = get_user_knowledge_dir(current_user.username) / kb_id
    if not subdir.exists():
        return {"message": "知识库目录不存在"}

    allowed_extensions = Params.DOC_SUPPORTED_FORMATS | {'.md', '.pdf'}
    document_files = []

    for file_path in subdir.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            if file_ext in allowed_extensions and not file_path.name.startswith('.'):
                document_files.append(file_path)

    if not document_files:
        return {"message": "未找到需要重新索引的文档"}

    delete_kb_vectors(current_user.username, kb_id)

    success_count = 0
    fail_count = 0

    for file_path in document_files:
        try:
            await chunk_and_store_document(str(file_path), current_user.username, kb_id)
            success_count += 1
        except Exception as e:
            logger.error(f"重新索引文档 {file_path.name} 失败: {str(e)}")
            fail_count += 1

    logger.info(f"知识库 {kb_id} 重新索引完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    return {"message": "重新索引完成", "success": success_count, "failed": fail_count}


def get_all_users() -> list:
    try:
        from core.auth.database import get_db
        db = next(get_db())
        users = db.query(User).all()
        return [user.username for user in users]
    except Exception as e:
        logger.error(f"获取所有用户失败: {str(e)}")
        return []