import os
import json
import uuid
import shutil
import time
import asyncio
import functools
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Body
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from core.auth.auth import get_current_user, security
from core.auth.database import get_db, User, SessionLocal, Knowledge
from settings.Define import PathConfig, Params
# 懒加载 doc_processor
_doc_processor = None

def get_doc_processor():
    global _doc_processor
    if _doc_processor is None:
        from core.skills.DocProcess.document_process import DocumentProcessor
        _doc_processor = DocumentProcessor()
    return _doc_processor

from settings.logger_manager import get_logger
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

EMBEDDING_DATABASE = os.getenv("EMBEDDING_DATABASE", "chroma").strip().lower()
print(f"EMBEDDING_DATABASE={EMBEDDING_DATABASE}")
QDRANT_URL = "http://127.0.0.1:6333"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1

# 懒加载重型库
_TextSplitter = None
_Chroma = None
_QdrantVectorStore = None
_RetrievalMode = None
_QdrantClient = None
_DashScopeEmbeddings = None
_SparseTextEmbedding = None


def _get_text_splitter():
    global _TextSplitter
    if _TextSplitter is None:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            _TextSplitter = RecursiveCharacterTextSplitter
        except ImportError:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            _TextSplitter = RecursiveCharacterTextSplitter
    return _TextSplitter


def _get_chroma():
    global _Chroma
    if _Chroma is None:
        try:
            from langchain.vectorstores import Chroma
            _Chroma = Chroma
        except ImportError:
            from langchain_chroma import Chroma
            _Chroma = Chroma
    return _Chroma


def _get_qdrant():
    global _QdrantVectorStore, _RetrievalMode, _QdrantClient
    if _QdrantVectorStore is None:
        from langchain_qdrant import QdrantVectorStore, RetrievalMode
        from qdrant_client import QdrantClient
        _QdrantVectorStore = QdrantVectorStore
        _RetrievalMode = RetrievalMode
        _QdrantClient = QdrantClient
    return _QdrantVectorStore, _RetrievalMode, _QdrantClient


def _get_embeddings():
    global _DashScopeEmbeddings
    if _DashScopeEmbeddings is None:
        try:
            from langchain.embeddings import DashScopeEmbeddings
            _DashScopeEmbeddings = DashScopeEmbeddings
        except ImportError:
            from langchain_community.embeddings import DashScopeEmbeddings
            _DashScopeEmbeddings = DashScopeEmbeddings
    return _DashScopeEmbeddings


def _get_sparse_embedding():
    global _SparseTextEmbedding
    if _SparseTextEmbedding is None:
        from fastembed import SparseTextEmbedding
        _SparseTextEmbedding = SparseTextEmbedding
    return _SparseTextEmbedding


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
    _, _, QdrantClient = _get_qdrant()
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


def _extract_kb_id(knowledge_id_val) -> str:
    """从 knowledge_id 字段中提取 kb_id，兼容新旧两种格式"""
    if not isinstance(knowledge_id_val, dict):
        return ''
    if 'kb_id' in knowledge_id_val:
        return knowledge_id_val['kb_id']
    for k in knowledge_id_val:
        if isinstance(k, str) and k.startswith('kb_'):
            return k
    return ''


def _normalize_knowledge_id(knowledge_id_val) -> dict:
    """把 knowledge_id 统一为 {'kb_id': '...', ...} 格式"""
    if not isinstance(knowledge_id_val, dict):
        return {}
    if 'kb_id' in knowledge_id_val:
        return dict(knowledge_id_val)
    for k, v in list(knowledge_id_val.items()):
        if isinstance(k, str) and k.startswith('kb_') and isinstance(v, dict):
            result = {'kb_id': k}
            result.update(v)
            return result
    return dict(knowledge_id_val)


_kb_docs_synced_users = set()


def _sync_document_paths(username: str):
    """扫描知识库目录，把实际文件列表同步到 knowledges 表的 document_paths"""
    if username in _kb_docs_synced_users:
        return

    user_dir = get_user_knowledge_dir(username)
    if not user_dir.exists():
        _kb_docs_synced_users.add(username)
        return

    actual_docs = {}
    for subdir in user_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith('kb_'):
            files = []
            for fp in subdir.iterdir():
                if fp.is_file() and not fp.name.startswith('~$') and not fp.name.startswith('.') and fp.name != 'Thumbs.db':
                    files.append(fp.name)
            files.sort()
            actual_docs[subdir.name] = files

    if not actual_docs:
        _kb_docs_synced_users.add(username)
        return

    db = SessionLocal()
    try:
        changed = 0
        for row in db.query(Knowledge).filter(Knowledge.user_name == username).all():
            kb_id = _extract_kb_id(row.knowledge_id)
            if not kb_id or kb_id not in actual_docs:
                continue
            norm = _normalize_knowledge_id(row.knowledge_id)
            existing_paths = norm.get('document_paths', [])
            target_paths = actual_docs[kb_id]
            if not target_paths and isinstance(row.chunk_retrieve_setting, dict) and row.chunk_retrieve_setting:
                target_paths = list(row.chunk_retrieve_setting.keys())
            if existing_paths != target_paths:
                norm['document_paths'] = target_paths
                row.knowledge_id = norm
                changed += 1

        if changed > 0:
            db.commit()
            logger.info(f"已为用户 {username} 的 {changed} 个知识库同步 document_paths")
    except Exception as e:
        logger.error(f"同步 document_paths 失败: {e}")
        db.rollback()
    finally:
        db.close()
        _kb_docs_synced_users.add(username)


_CHUNK_CONFIG_KEYS = {
    'chunk_size', 'chunk_overlap', 'separators',
    'parent_chunk_size', 'parent_chunk_overlap', 'parent_separators',
    'child_chunk_size', 'child_chunk_overlap', 'child_separators',
}


def _extract_chunk_fields_from_item(item: dict) -> tuple:
    """从前端传入的 item dict 中提取 chunk 配置，返回 (chunk_strategy_dict, other_extra_dict)"""
    cs_raw = item.get('chunk_strategy', 'standard')
    other = {}

    chunk_fields_found = {k: v for k, v in item.items() if k in _CHUNK_CONFIG_KEYS and v is not None}

    if isinstance(cs_raw, dict) and cs_raw:
        cs_dict = dict(cs_raw)
        if chunk_fields_found:
            cs_dict.update(chunk_fields_found)
    else:
        strategy = cs_raw if isinstance(cs_raw, str) else 'standard'
        strategy = strategy if strategy in VALID_CHUNK_STRATEGIES else 'standard'
        cs_dict = _normalize_chunk_strategy(strategy)
        if chunk_fields_found:
            cs_dict.update(chunk_fields_found)

    for k, v in item.items():
        if k in ('id', 'name', 'description', 'chunk_strategy', 'created_at', 'file_chunk_configs') or k in _CHUNK_CONFIG_KEYS:
            continue
        other[k] = v

    return cs_dict, other


def load_kb_metadata(username: str) -> list:
    _sync_document_paths(username)

    db = SessionLocal()
    try:
        rows = db.query(Knowledge).filter(Knowledge.user_name == username).all()
        result = []
        for row in rows:
            cs_list = _db_chunk_retrieve_to_list(row.chunk_retrieve_setting)
            # 取第一个文件的配置作为知识库级别默认值（兼容其他逻辑）
            first_cs = cs_list[0][1] if cs_list else _default_chunk_strategy()

            extra = _normalize_knowledge_id(row.knowledge_id)
            kb_id = extra.get('kb_id', str(row.id))

            entry = {
                'id': kb_id,
                'name': row.knowledge_name,
                'description': row.knowledge_description or '',
                'chunk_strategy': first_cs.get('strategy', 'standard'),
                'created_at': row.created_at.timestamp() if row.created_at else time.time(),
                'file_chunk_configs': {fname: cs for fname, cs in cs_list if fname is not None},
            }

            for k, v in first_cs.items():
                if k != 'strategy':
                    entry[k] = v

            for k, v in extra.items():
                if k == 'kb_id':
                    continue
                if k in _CHUNK_CONFIG_KEYS and k not in entry:
                    entry[k] = v
                elif k not in _CHUNK_CONFIG_KEYS:
                    entry[k] = v

            result.append(entry)
        return result
    except Exception as e:
        logger.error(f"从数据库加载知识库元数据失败: {e}")
        return []
    finally:
        db.close()

def get_kb_file_rag_configs(username: str, kb_id: str) -> dict:
    """获取知识库中每个文档的 RAG 配置。
    返回 {filename: {embedding_model_name, rag_retrieve_method, rerank_model_setting}}
    """
    db = SessionLocal()
    try:
        rows = db.query(Knowledge).filter(Knowledge.user_name == username).all()
        for row in rows:
            if _extract_kb_id(row.knowledge_id) == kb_id:
                raw = row.chunk_retrieve_setting
                result = {}
                if isinstance(raw, dict) and raw:
                    for fname, config in raw.items():
                        if isinstance(config, dict):
                            result[fname] = {
                                "embedding_model_name": config.get("embedding_model_name"),
                                "rag_retrieve_method": config.get("rag_retrieve_method", "similarity"),
                                "rerank_model_setting": config.get("rerank_model_setting", {"rerank_enable": False}),
                            }
                return result
        return {}
    except Exception as e:
        logger.error(f"get_kb_file_rag_configs failed: {e}")
        return {}
    finally:
        db.close()

def save_kb_metadata(username: str, meta_list: list):
    db = SessionLocal()
    try:
        existing = db.query(Knowledge).filter(Knowledge.user_name == username).all()
        existing_map = {}
        for row in existing:
            kb_id = _extract_kb_id(row.knowledge_id)
            if kb_id:
                existing_map[kb_id] = row

        incoming_ids = set()
        for item in meta_list:
            kb_id = item.get('id', '')
            if not kb_id:
                continue
            incoming_ids.add(kb_id)

            cs_dict, extra_config = _extract_chunk_fields_from_item(item)

            if kb_id in existing_map:
                row = existing_map[kb_id]
                row.knowledge_name = item.get('name', '未命名知识库')
                row.knowledge_description = item.get('description', '')
                row.chunk_retrieve_setting = cs_dict
                existing_extra = _normalize_knowledge_id(row.knowledge_id)
                merged_extra = {**existing_extra, **extra_config, 'kb_id': kb_id}
                for k in _CHUNK_CONFIG_KEYS:
                    merged_extra.pop(k, None)
                if 'chunk_strategy' in merged_extra and not isinstance(merged_extra['chunk_strategy'], (dict, list)):
                    pass
                else:
                    merged_extra.pop('chunk_strategy', None)
                # 移除多余的 file_chunk_configs 字段
                merged_extra.pop('file_chunk_configs', None)
                row.knowledge_id = merged_extra
            else:
                kb = Knowledge(
                    knowledge_name=item.get('name', '未命名知识库'),
                    knowledge_description=item.get('description', ''),
                    user_name=username,
                    chunk_retrieve_setting=cs_dict,
                    knowledge_id={'kb_id': kb_id, **extra_config},
                )
                db.add(kb)

        for kb_id, row in existing_map.items():
            if kb_id not in incoming_ids:
                db.delete(row)

        db.commit()
    except Exception as e:
        logger.error(f"保存知识库元数据到数据库失败: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def _kb_add_document(username: str, kb_id: str, filename: str):
    db = SessionLocal()
    try:
        for row in db.query(Knowledge).filter(Knowledge.user_name == username).all():
            kid = _extract_kb_id(row.knowledge_id)
            if kid == kb_id:
                norm = _normalize_knowledge_id(row.knowledge_id)
                paths = norm.get('document_paths', [])
                if filename not in paths:
                    paths = list(paths) + [filename]
                    norm['document_paths'] = paths
                    row.knowledge_id = norm
                    db.commit()
                break
    except Exception as e:
        logger.error(f"更新知识库 document_paths 失败: {e}")
        db.rollback()
    finally:
        db.close()


def _kb_remove_document(username: str, kb_id: str, filename: str):
    db = SessionLocal()
    try:
        for row in db.query(Knowledge).filter(Knowledge.user_name == username).all():
            kid = _extract_kb_id(row.knowledge_id)
            if kid == kb_id:
                norm = _normalize_knowledge_id(row.knowledge_id)
                paths = norm.get('document_paths', [])
                if filename in paths:
                    paths = [p for p in paths if p != filename]
                    norm['document_paths'] = paths
                    row.knowledge_id = norm
                    db.commit()
                break
    except Exception as e:
        logger.error(f"更新知识库 document_paths 失败: {e}")
        db.rollback()
    finally:
        db.close()


def _kb_remove_chunk_config(username: str, kb_id: str, filename: str):
    db = SessionLocal()
    try:
        for row in db.query(Knowledge).filter(Knowledge.user_name == username).all():
            kid = _extract_kb_id(row.knowledge_id)
            if kid == kb_id:
                raw = row.chunk_retrieve_setting
                if isinstance(raw, dict) and filename in raw:
                    raw.pop(filename, None)
                    row.chunk_retrieve_setting = raw
                    db.commit()
                break
    except Exception as e:
        logger.error(f"移除文档切片配置失败: {e}")
        db.rollback()
    finally:
        db.close()


_embeddings_cache = {}
_sparse_embeddings_cache = {}


def _get_embeddings(username: str, model_name: str = None):
    cache_key = f"{username}:{model_name or 'default'}"
    if cache_key not in _embeddings_cache:
        model = model_name or Params.DEFAULT_EMBEDDING_MODEL
        DashScopeEmbeddings = _get_embeddings()
        embeddings = DashScopeEmbeddings(model=model)
        logger.info(f"DashScope 向量模型已初始化: {model}")
        _embeddings_cache[cache_key] = embeddings
    return _embeddings_cache[cache_key]


def _get_sparse_embeddings(username: str):
    if username not in _sparse_embeddings_cache:
        SparseTextEmbedding = _get_sparse_embedding()
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


def _build_vector_store(username: str, model_name: str = None, retrieval_mode: str = "similarity"):
    if EMBEDDING_DATABASE == "qdrant":
        return _build_qdrant_vector_store(username, model_name, retrieval_mode)
    else:
        return _build_chroma_vector_store(username, model_name, retrieval_mode)

def _build_chroma_vector_store(username: str, model_name: str = None, retrieval_mode: str = "similarity"):
    if retrieval_mode == "hybrid":
        logger.warning("Chroma 不支持混合检索，已降级为向量检索。如需混合检索请使用 Qdrant。")
    embeddings = _get_embeddings(username, model_name)
    persist_dir = _get_chroma_persist_dir(username)
    Chroma = _get_chroma()
    return Chroma(
        collection_name=f"knowledge_{username}",
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )

def _build_qdrant_vector_store(username: str, model_name: str = None, retrieval_mode: str = "similarity"):
    _ensure_collection_exists(username)
    client = _create_qdrant_client()
    embeddings = _get_embeddings(username, model_name)
    sparse_embeddings = _get_sparse_embeddings(username)
    QdrantVectorStore, RetrievalMode, _ = _get_qdrant()
    mode = RetrievalMode.HYBRID if retrieval_mode == "hybrid" else RetrievalMode.DENSE
    return QdrantVectorStore(
        client=client,
        collection_name=f"knowledge_{username}",
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=mode,
        validate_collection_config=False,
    )

def get_vector_store(username: str, model_name: str = None, retrieval_mode: str = "similarity"):
    return _build_vector_store(username, model_name, retrieval_mode)


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


def _split_by_qa_pattern(content: str) -> list:
    """
    QA 分段模式：将文档按问答对切分。
    支持以下格式：
      - JSON 数组格式：[{"question": "...", "answer": "..."}, ...]
      - Q: ... A: ... / Q：... A：...
      - 问：... 答：...
      - 问题：... 答案：...
    若未检测到问答格式则返回空列表，由调用方回退到标准分段。
    """
    import re
    import json

    qa_pairs = []

    # 1. 尝试解析 JSON 格式的 QA 数据
    stripped = content.strip()
    if stripped.startswith('[') or stripped.startswith('{'):
        try:
            data = json.loads(stripped)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    q = item.get('question') or item.get('Question') or item.get('Q') or item.get('q')
                    a = item.get('answer') or item.get('Answer') or item.get('A') or item.get('a')
                    if q and a:
                        qa_pairs.append("问：{}\n答：{}".format(str(q).strip(), str(a).strip()))
            if qa_pairs:
                return qa_pairs
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. 正则匹配文本格式的 QA
    combined_pat = re.compile(
        r'(?:^|\n)\s*(?:Q|问|问题)\s*[:：]\s*(.+?)\s*\n\s*(?:A|答|答案)\s*[:：]\s*(.+?)(?=\n\s*(?:Q|问|问题|A|答|答案)\s*[:：]|\Z)',
        re.DOTALL,
    )

    for m in combined_pat.finditer(content):
        q = m.group(1).strip()
        a = m.group(2).strip()
        if q and a:
            qa_pairs.append("问：{}\n答：{}".format(q, a))

    if qa_pairs:
        return qa_pairs

    # 3. 逐行解析文本格式的 QA
    lines = content.split('\n')
    current_q = None
    current_a_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_q and current_a_lines:
                qa_pairs.append("问：{}\n答：{}".format(current_q, ''.join(current_a_lines).strip()))
                current_q = None
                current_a_lines = []
            continue
        if re.match(r'^(Q|问|问题)\s*[:：]', stripped):
            if current_q and current_a_lines:
                qa_pairs.append("问：{}\n答：{}".format(current_q, ''.join(current_a_lines).strip()))
            current_q = re.sub(r'^(Q|问|问题)\s*[:：]\s*', '', stripped)
            current_a_lines = []
        elif re.match(r'^(A|答|答案)\s*[:：]', stripped) and current_q:
            current_a_lines.append(re.sub(r'^(A|答|答案)\s*[:：]\s*', '', stripped))
        elif current_q:
            current_a_lines.append(line)
    if current_q and current_a_lines:
        qa_pairs.append("问：{}\n答：{}".format(current_q, ''.join(current_a_lines).strip()))

    return qa_pairs


def _chunk_with_strategy(
    content: str,
    strategy: str,
    filename: str,
    kb_id: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    separators: list = None,
    parent_chunk_size: int = None,
    parent_chunk_overlap: int = None,
    parent_separators: list = None,
    child_chunk_size: int = None,
    child_chunk_overlap: int = None,
    child_separators: list = None,
) -> tuple:
    """
    根据策略对文本进行切片，返回 (texts, metadatas)。
    strategy: standard | parent_child | qa

    standard 策略: 使用 chunk_size / chunk_overlap / separators。
    parent_child 策略:
        父段参数优先使用 parent_chunk_size / parent_chunk_overlap / parent_separators；
        若未传，则从基础参数推导（chunk_size*2、chunk_overlap*2）。
        子段参数优先使用 child_chunk_size / child_chunk_overlap / child_separators；
        若未传，则使用 chunk_size / chunk_overlap / separators。
    qa 策略: 不使用上述参数。
    所有参数不传则使用全局默认值。
    """
    texts_to_add = []
    metadatas_to_add = []

    base_size = chunk_size if chunk_size and chunk_size > 0 else Params.CHUNK_SIZE
    base_overlap = chunk_overlap if chunk_overlap is not None and chunk_overlap >= 0 else Params.CHUNK_OVERLAP
    base_separators = separators if separators else Params.SEPARATORS

    if strategy == "qa":
        qa_pairs = _split_by_qa_pattern(content)
        if qa_pairs:
            for pair in qa_pairs:
                texts_to_add.append(pair)
                metadatas_to_add.append({"source": filename, "kb_id": kb_id, "chunk_strategy": "qa"})
            return texts_to_add, metadatas_to_add
        logger.warning("未检测到问答格式")
        return [], []

    if strategy == "parent_child":
        p_size = parent_chunk_size if parent_chunk_size and parent_chunk_size > 0 else max(base_size * 2, 1200)
        p_overlap = parent_chunk_overlap if parent_chunk_overlap is not None and parent_chunk_overlap >= 0 else min(base_overlap * 2, 300)
        p_separators = parent_separators if parent_separators else base_separators
        if p_overlap >= p_size:
            p_overlap = max(0, p_size // 4)

        c_size = child_chunk_size if child_chunk_size and child_chunk_size > 0 else base_size
        c_overlap = child_chunk_overlap if child_chunk_overlap is not None and child_chunk_overlap >= 0 else base_overlap
        c_separators = child_separators if child_separators else base_separators
        if c_overlap >= c_size:
            c_overlap = max(0, c_size // 4)

        RecursiveCharacterTextSplitter = _get_text_splitter()
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=p_size,
            chunk_overlap=p_overlap,
            separators=p_separators,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            separators=c_separators,
        )

        parents = parent_splitter.split_text(content)
        for idx, parent_text in enumerate(parents):
            children = child_splitter.split_text(parent_text)
            if not children:
                texts_to_add.append(parent_text)
                metadatas_to_add.append({
                    "source": filename, "kb_id": kb_id,
                    "chunk_strategy": "parent_child",
                    "parent_id": f"{filename}_{idx}",
                    "parent_content": parent_text,
                })
                continue
            for child_text in children:
                texts_to_add.append(child_text)
                metadatas_to_add.append({
                    "source": filename, "kb_id": kb_id,
                    "chunk_strategy": "parent_child",
                    "parent_id": f"{filename}_{idx}",
                    "parent_content": parent_text,
                })
        return texts_to_add, metadatas_to_add

    text_splitter = _get_text_splitter()(
        chunk_size=base_size,
        chunk_overlap=base_overlap,
        separators=base_separators,
    )
    chunks = text_splitter.split_text(content)
    for chunk in chunks:
        texts_to_add.append(chunk)
        metadatas_to_add.append({"source": filename, "kb_id": kb_id, "chunk_strategy": "standard"})
    return texts_to_add, metadatas_to_add


@retry_qdrant()
async def chunk_and_store_document(
    file_path: str,
    username: str,
    kb_id: str,
    chunk_strategy: str = "standard",
    chunk_size: int = None,
    chunk_overlap: int = None,
    separators: list = None,
    parent_chunk_size: int = None,
    parent_chunk_overlap: int = None,
    parent_separators: list = None,
    child_chunk_size: int = None,
    child_chunk_overlap: int = None,
    child_separators: list = None,
    embedding_model: str = None,
    retrieval_mode: str = "hybrid",
):
    try:
        logger.info(f"开始处理文档: {file_path}, 用户: {username}, 知识库: {kb_id}, 切片策略: {chunk_strategy}")

        content = ""
        async for chunk in get_doc_processor().read_document(file_path):
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

        vector_store = get_vector_store(username, embedding_model, retrieval_mode)
        filename = os.path.basename(file_path)

        chunk_kwargs = {
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "separators": separators,
            "parent_chunk_size": parent_chunk_size, "parent_chunk_overlap": parent_chunk_overlap, "parent_separators": parent_separators,
            "child_chunk_size": child_chunk_size, "child_chunk_overlap": child_chunk_overlap, "child_separators": child_separators,
        }

        # Markdown 标题切分逻辑暂时注释，预览和存储逻辑保持一致
        # if file_path.endswith(".md") and chunk_strategy != "qa":
        #     from langchain_text_splitters import MarkdownHeaderTextSplitter
        #     md_splitter = MarkdownHeaderTextSplitter(
        #         headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")]
        #     )
        #     sections = md_splitter.split_text(content)
        #     all_texts = []
        #     all_metas = []
        #     for sec in sections:
        #         st, sm = _chunk_with_strategy(sec.page_content, chunk_strategy, filename, kb_id, **chunk_kwargs)
        #         for i in range(len(st)):
        #             merged_meta = {**sec.metadata, **sm[i]}
        #             all_metas.append(merged_meta)
        #         all_texts.extend(st)
        #     if all_texts:
        #         vector_store.add_texts(texts=all_texts, metadatas=all_metas)
        #         logger.info(f"文档 {file_path} 已成功切块并存储，共 {len(all_texts)} 个段落 (策略: {chunk_strategy})")
        #     else:
        #         logger.warning(f"文档切块结果为空: {file_path}")
        #     return

        texts_to_add, metadatas_to_add = _chunk_with_strategy(content, chunk_strategy, filename, kb_id, **chunk_kwargs)

        if texts_to_add:
            vector_store.add_texts(texts=texts_to_add, metadatas=metadatas_to_add)
            logger.info(f"文档 {file_path} 已成功切块并存储，共 {len(texts_to_add)} 个段落 (策略: {chunk_strategy})")
        else:
            logger.warning(f"文档切块结果为空: {file_path}")

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
            "chunk_strategy": kb.get('chunk_strategy', 'standard'),
            "chunk_size": kb.get('chunk_size', Params.CHUNK_SIZE),
            "chunk_overlap": kb.get('chunk_overlap', Params.CHUNK_OVERLAP),
            "separators": kb.get('separators', Params.SEPARATORS),
            "parent_chunk_size": kb.get('parent_chunk_size'),
            "parent_chunk_overlap": kb.get('parent_chunk_overlap'),
            "parent_separators": kb.get('parent_separators'),
            "child_chunk_size": kb.get('child_chunk_size'),
            "child_chunk_overlap": kb.get('child_chunk_overlap'),
            "child_separators": kb.get('child_separators'),
            "created_at": kb.get('created_at', 0),
            "doc_count": doc_count
        })

    return {"bases": result}

VALID_CHUNK_STRATEGIES = {"standard", "parent_child", "qa"}


def _default_chunk_strategy() -> dict:
    return {
        "strategy": "standard",
        "chunk_size": Params.CHUNK_SIZE,
        "chunk_overlap": Params.CHUNK_OVERLAP,
        "separators": Params.SEPARATORS,
    }


def _db_chunk_retrieve_to_list(raw) -> list:
    """解析数据库 chunk_retrieve_setting 字段，返回 [(filename, cs_dict), ...] 列表。

    支持三种格式:
    1. 单文件: {"chunk_strategy": {...}, "rag_retrieve_method": "..."}  -> [(None, cs_dict)]
    2. 多文件: {"a.md": {"chunk_strategy": {...}}, "b.xlsx": {...}}   -> [("a.md", cs), ("b.xlsx", cs)]
    3. 旧格式: 字符串或扁平 dict                                       -> [(None, _normalize_chunk_strategy(raw))]
    """
    if not raw:
        return [(None, _default_chunk_strategy())]

    if isinstance(raw, dict) and raw:
        # 数据库多文件格式: {"file1.md": {"chunk_strategy": {...}}, "file2.xlsx": {...}}
        if any(isinstance(v, dict) and "chunk_strategy" in v for v in raw.values()):
            result = []
            for fname, config in raw.items():
                if isinstance(config, dict) and "chunk_strategy" in config:
                    result.append((fname, _normalize_inner_cs(config["chunk_strategy"])))
            return result
        
        # 如果是字典但不包含 chunk_strategy，尝试规范化处理
        return [(None, _normalize_chunk_strategy(raw))]

    if isinstance(raw, dict):
        # 顶层是 filename -> {chunk_strategy: ...} -> 多文件格式
        result = []
        for filename, config in raw.items():
            if isinstance(config, dict) and "chunk_strategy" in config:
                cs = config["chunk_strategy"]
                result.append((filename, _normalize_inner_cs(cs)))
            elif isinstance(config, dict):
                result.append((filename, _normalize_inner_cs(config)))
            else:
                result.append((filename, _default_chunk_strategy()))
        if result:
            return result

        # 兜底: 旧格式扁平 dict
        return [(None, _normalize_chunk_strategy(raw))]

    return [(None, _default_chunk_strategy())]

def _normalize_inner_cs(cs) -> dict:
    """把 chunk_strategy 子字段规范化为统一的扁平 dict"""
    if not isinstance(cs, dict) or not cs:
        return _default_chunk_strategy()

    strategy = cs.get("strategy", "standard")
    if strategy == "parent_child":
        ps = cs.get("parent_setting", {})
        chs = cs.get("child_setting", {})
        return {
            "strategy": "parent_child",
            "parent_chunk_size": ps.get("chunk_size", max(Params.CHUNK_SIZE * 2, 1200)),
            "parent_chunk_overlap": ps.get("chunk_overlap", min(Params.CHUNK_OVERLAP * 2, 300)),
            "parent_separators": ps.get("separators", Params.SEPARATORS),
            "child_chunk_size": chs.get("chunk_size", Params.CHUNK_SIZE),
            "child_chunk_overlap": chs.get("chunk_overlap", Params.CHUNK_OVERLAP),
            "child_separators": chs.get("separators", Params.SEPARATORS),
        }
    elif strategy in ("qa", "question_answer"):
        return {"strategy": "qa"}
    else:
        return {
            "strategy": strategy if strategy in VALID_CHUNK_STRATEGIES else "standard",
            "chunk_size": cs.get("chunk_size", Params.CHUNK_SIZE),
            "chunk_overlap": cs.get("chunk_overlap", Params.CHUNK_OVERLAP),
            "separators": cs.get("separators", Params.SEPARATORS),
        }

def _build_db_chunk_strategy(strategy: str, cs_dict: dict) -> dict:
    """构建数据库存储格式的 chunk_strategy（嵌套结构）"""
    if strategy == "parent_child":
        return {
            "strategy": "parent_child",
            "parent_setting": {
                "chunk_size": cs_dict.get("parent_chunk_size", 1600),
                "chunk_overlap": cs_dict.get("parent_chunk_overlap", 300),
                "separators": cs_dict.get("parent_separators", ["\n\n", "\n", "。", "；", "，", " ", ""]),
            },
            "child_setting": {
                "chunk_size": cs_dict.get("child_chunk_size", 800),
                "chunk_overlap": cs_dict.get("child_chunk_overlap", 150),
                "separators": cs_dict.get("child_separators", ["\n\n", "\n", "。", "；", "，", " ", ""]),
            },
        }
    elif strategy in ("qa", "question_answer"):
        return {"strategy": "question_answer"}
    else:
        return {
            "strategy": "standard",
            "chunk_size": cs_dict.get("chunk_size", 800),
            "chunk_overlap": cs_dict.get("chunk_overlap", 150),
            "separators": cs_dict.get("separators", ["\n\n", "\n", "。", "；", "，", " ", ""]),
        }

def _normalize_chunk_strategy(raw) -> dict:
    """把数据库里存的 chunk_strategy 统一解析为 dict 格式"""
    if isinstance(raw, dict) and raw:
        return raw
    if isinstance(raw, str):
        strategy = raw
    else:
        strategy = "standard"
    strategy = strategy if strategy in VALID_CHUNK_STRATEGIES else "standard"
    if strategy == "parent_child":
        return {
            "strategy": "parent_child",
            "parent_chunk_size": max(Params.CHUNK_SIZE * 2, 1200),
            "parent_chunk_overlap": min(Params.CHUNK_OVERLAP * 2, 300),
            "parent_separators": Params.SEPARATORS,
            "child_chunk_size": Params.CHUNK_SIZE,
            "child_chunk_overlap": Params.CHUNK_OVERLAP,
            "child_separators": Params.SEPARATORS,
        }
    elif strategy == "qa":
        return {"strategy": "qa"}
    else:
        return {
            "strategy": "standard",
            "chunk_size": Params.CHUNK_SIZE,
            "chunk_overlap": Params.CHUNK_OVERLAP,
            "separators": Params.SEPARATORS,
        }


def _chunk_strategy_to_kwargs(cs_dict: dict) -> dict:
    """从 chunk_strategy JSON 中提取 chunk_and_store_document 需要的 kwargs"""
    if not cs_dict:
        cs_dict = _default_chunk_strategy()
    strategy = cs_dict.get("strategy", "standard")
    if strategy == "parent_child":
        return {
            "chunk_strategy": "parent_child",
            "parent_chunk_size": cs_dict.get("parent_chunk_size"),
            "parent_chunk_overlap": cs_dict.get("parent_chunk_overlap"),
            "parent_separators": _parse_separators(cs_dict.get("parent_separators")),
            "child_chunk_size": cs_dict.get("child_chunk_size"),
            "child_chunk_overlap": cs_dict.get("child_chunk_overlap"),
            "child_separators": _parse_separators(cs_dict.get("child_separators")),
        }
    elif strategy == "qa":
        return {"chunk_strategy": "qa"}
    else:
        return {
            "chunk_strategy": "standard",
            "chunk_size": cs_dict.get("chunk_size"),
            "chunk_overlap": cs_dict.get("chunk_overlap"),
            "separators": _parse_separators(cs_dict.get("separators")),
        }


def _parse_separators(separators_raw: str) -> list:
    """
    解析前端传入的分隔符字符串，支持两种格式：
      1. JSON 数组字符串: '["\\n\\n","\\n","。"]'
      2. 逗号分隔字符串: '\\n\\n,\\n,。,；,，, ,'
    字面量转义符 \\n \\t \\r 会被还原为真实控制字符。
    返回分隔符列表。
    """
    if not separators_raw:
        return Params.SEPARATORS
    # 如果已经是列表，直接处理
    if isinstance(separators_raw, list):
        decoded = []
        for s in separators_raw:
            s = str(s).replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
            decoded.append(s)
        return decoded if decoded else Params.SEPARATORS
    separators_raw = separators_raw.strip()
    if not separators_raw:
        return Params.SEPARATORS

    parts = None
    if separators_raw.startswith("["):
        try:
            parsed = json.loads(separators_raw)
            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                parts = parsed
        except (json.JSONDecodeError, TypeError):
            pass

    if parts is None:
        parts = [p for p in separators_raw.split(",")]

    decoded = []
    for s in parts:
        s = s.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        decoded.append(s)
    return decoded


def _validate_chunk_params(chunk_size, chunk_overlap, label: str = "chunk"):
    """校验分片参数，返回 (chunk_size, chunk_overlap) 或抛出 HTTPException。label 用于错误提示。"""
    cs, co = None, None
    if chunk_size is not None:
        try:
            cs = int(chunk_size)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{label}_size 必须是整数")
        if cs <= 0:
            raise HTTPException(status_code=400, detail=f"{label}_size 必须大于 0")
    if chunk_overlap is not None:
        try:
            co = int(chunk_overlap)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"{label}_overlap 必须是整数")
        if co < 0:
            raise HTTPException(status_code=400, detail=f"{label}_overlap 不能小于 0")
    if cs is not None and co is not None and co >= cs:
        raise HTTPException(status_code=400, detail=f"{label}_overlap 必须小于 {label}_size")
    return cs, co


@router.post("/bases")
async def create_knowledge_base(
    name: str = Query(..., description="知识库名称"),
    description: str = Query("", description="知识库描述"),
    chunk_strategy: str = Query("standard", description="切片策略: standard/parent_child/qa"),
    chunk_size: int = Query(None, description="最大 chunk 长度（字符数），仅 standard/parent_child 生效"),
    chunk_overlap: int = Query(None, description="chunk 重叠长度，仅 standard/parent_child 生效"),
    separators: str = Query(None, description="分隔符列表（JSON 数组或逗号分隔字符串）"),
    parent_chunk_size: int = Query(None, description="父段最大 chunk 长度，仅 parent_child 生效"),
    parent_chunk_overlap: int = Query(None, description="父段 chunk 重叠长度，仅 parent_child 生效"),
    parent_separators: str = Query(None, description="父段分隔符列表，仅 parent_child 生效"),
    child_chunk_size: int = Query(None, description="子段最大 chunk 长度，仅 parent_child 生效"),
    child_chunk_overlap: int = Query(None, description="子段 chunk 重叠长度，仅 parent_child 生效"),
    child_separators: str = Query(None, description="子段分隔符列表，仅 parent_child 生效"),
    retrieval_mode: str = Query("hybrid", description="检索方式: vector/hybrid"),
    embedding_model: str = Query(None, description="向量模型"),
    rerank_enabled: bool = Query(False, description="是否启用Rerank"),
    rerank_model: str = Query(None, description="Rerank模型"),
    rerank_top_k: int = Query(3, description="Rerank Top K"),
    file_rag_configs: str = Query(None, description="各文档的RAG配置(JSON)"),
    kb_id: str = Query(None, description="知识库ID（前端预先生成，用于保持文件目录与数据库记录一致）"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="知识库名称不能为空")

    if chunk_strategy not in VALID_CHUNK_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"无效的切片策略，支持: {', '.join(VALID_CHUNK_STRATEGIES)}")

    cs, co = _validate_chunk_params(chunk_size, chunk_overlap, "chunk")
    sep_list = _parse_separators(separators) if separators else None
    pcs, pco = _validate_chunk_params(parent_chunk_size, parent_chunk_overlap, "parent_chunk")
    p_sep = _parse_separators(parent_separators) if parent_separators else None
    ccs, cco = _validate_chunk_params(child_chunk_size, child_chunk_overlap, "child_chunk")
    c_sep = _parse_separators(child_separators) if child_separators else None

    cs_dict = _normalize_chunk_strategy(chunk_strategy)
    if cs is not None:
        cs_dict["chunk_size"] = cs
    if co is not None:
        cs_dict["chunk_overlap"] = co
    if sep_list is not None:
        cs_dict["separators"] = sep_list
    if pcs is not None:
        cs_dict["parent_chunk_size"] = pcs
    if pco is not None:
        cs_dict["parent_chunk_overlap"] = pco
    if p_sep is not None:
        cs_dict["parent_separators"] = p_sep
    if ccs is not None:
        cs_dict["child_chunk_size"] = ccs
    if cco is not None:
        cs_dict["child_chunk_overlap"] = cco
    if c_sep is not None:
        cs_dict["child_separators"] = c_sep

    meta_list = load_kb_metadata(current_user.username)
    for kb in meta_list:
        if kb['name'] == name.strip():
            raise HTTPException(status_code=400, detail="已存在同名知识库")

    kb_id = kb_id if kb_id else f"kb_{uuid.uuid4().hex[:12]}"

    file_rag = {}
    if file_rag_configs:
        try:
            file_rag = json.loads(file_rag_configs)
        except Exception:
            file_rag = {}

    # 构建 chunk_retrieve_setting: {"文件名": {"chunk_strategy": {...}, rag...}}
    chunk_retrieve_setting = {}
    for fname, rc in file_rag.items():
        # 优先使用文件自己的 chunk_strategy，其次用顶层参数
        file_cs = rc.get("chunk_strategy")
        if file_cs:
            file_strategy = file_cs.get("strategy", chunk_strategy)
            db_cs = _build_db_chunk_strategy(file_strategy, file_cs)
        else:
            db_cs = _build_db_chunk_strategy(chunk_strategy, cs_dict)
        chunk_retrieve_setting[fname] = {
            "chunk_strategy": db_cs,
            "rag_retrieve_method": rc.get("rag_retrieve_method", "similarity"),
            "embedding_model_name": rc.get("embedding_model_name"),
            "rerank_model_setting": rc.get("rerank_model_setting", {"rerank_enable": False}),
        }

    # 构建 document_paths
    document_paths = list(file_rag.keys())

    # 直接写入数据库，完全控制格式
    kb = Knowledge(
        knowledge_name=name.strip(),
        knowledge_description=description.strip(),
        user_name=current_user.username,
        chunk_retrieve_setting=chunk_retrieve_setting if chunk_retrieve_setting else cs_dict,
        knowledge_id={"kb_id": kb_id, "document_paths": document_paths},
    )
    db.add(kb)
    db.commit()

    get_kb_subdir(current_user.username, kb_id)

    resp = {"id": kb_id, "name": name.strip(), "description": description.strip(), "chunk_strategy": cs_dict}
    return resp


@router.put("/bases/{kb_id}")
async def update_knowledge_base(
    kb_id: str,
    name: str = Query(None),
    description: str = Query(None),
    chunk_strategy: str = Query(None, description="切片策略: standard/parent_child/qa"),
    chunk_size: int = Query(None, description="最大 chunk 长度"),
    chunk_overlap: int = Query(None, description="chunk 重叠长度"),
    separators: str = Query(None, description="分隔符列表"),
    parent_chunk_size: int = Query(None, description="父段最大 chunk 长度"),
    parent_chunk_overlap: int = Query(None, description="父段 chunk 重叠长度"),
    parent_separators: str = Query(None, description="父段分隔符列表"),
    child_chunk_size: int = Query(None, description="子段最大 chunk 长度"),
    child_chunk_overlap: int = Query(None, description="子段 chunk 重叠长度"),
    child_separators: str = Query(None, description="子段分隔符列表"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if chunk_strategy is not None and chunk_strategy not in VALID_CHUNK_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"无效的切片策略，支持: {', '.join(VALID_CHUNK_STRATEGIES)}")

    cs, co = _validate_chunk_params(chunk_size, chunk_overlap, "chunk")
    sep_list = _parse_separators(separators) if separators else None
    pcs, pco = _validate_chunk_params(parent_chunk_size, parent_chunk_overlap, "parent_chunk")
    p_sep = _parse_separators(parent_separators) if parent_separators else None
    ccs, cco = _validate_chunk_params(child_chunk_size, child_chunk_overlap, "child_chunk")
    c_sep = _parse_separators(child_separators) if child_separators else None

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

            existing_cs = kb.get('chunk_strategy')
            if isinstance(existing_cs, dict):
                new_cs = dict(existing_cs)
            else:
                new_cs = _normalize_chunk_strategy(existing_cs or 'standard')

            if chunk_strategy is not None and chunk_strategy != new_cs.get('strategy'):
                new_cs = _normalize_chunk_strategy(chunk_strategy)

            if cs is not None:
                new_cs['chunk_size'] = cs
            if co is not None:
                new_cs['chunk_overlap'] = co
            if sep_list is not None:
                new_cs['separators'] = sep_list
            if pcs is not None:
                new_cs['parent_chunk_size'] = pcs
            if pco is not None:
                new_cs['parent_chunk_overlap'] = pco
            if p_sep is not None:
                new_cs['parent_separators'] = p_sep
            if ccs is not None:
                new_cs['child_chunk_size'] = ccs
            if cco is not None:
                new_cs['child_chunk_overlap'] = cco
            if c_sep is not None:
                new_cs['child_separators'] = c_sep

            kb['chunk_strategy'] = new_cs
            for k in _CHUNK_CONFIG_KEYS:
                kb.pop(k, None)

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

    kb_row = None
    try:
        kb_rows = db.query(Knowledge).filter(Knowledge.user_name == current_user.username).all()
        for row in kb_rows:
            if _extract_kb_id(row.knowledge_id) == kb_id:
                kb_row = row
                break
    except Exception:
        kb_row = None

    chunk_configs = {}
    rag_configs = {}
    if kb_row and kb_row.chunk_retrieve_setting and isinstance(kb_row.chunk_retrieve_setting, dict):
        raw = kb_row.chunk_retrieve_setting
        for fname, config in raw.items():
            if not isinstance(config, dict):
                continue
            cs = config.get("chunk_strategy", {})
            if cs:
                chunk_configs[fname] = _normalize_inner_cs(cs)
            rag_configs[fname] = {
                "embedding_model_name": config.get("embedding_model_name", "text-embedding-v4"),
                "rag_retrieve_method": config.get("rag_retrieve_method", "similarity"),
                "rerank_model_setting": config.get("rerank_model_setting", {"rerank_enable": False}),
            }

    seen_filenames = set()
    documents = []

    if subdir.exists():
        for file_path in subdir.iterdir():
            if file_path.is_file():
                filename = file_path.name
                if not (filename.startswith('~$') or filename.startswith('.') or filename == 'Thumbs.db'):
                    seen_filenames.add(filename)
                    has_config = filename in chunk_configs
                    cs = chunk_configs.get(filename, _default_chunk_strategy())
                    rag = rag_configs.get(filename, {
                        "embedding_model_name": "text-embedding-v4",
                        "rag_retrieve_method": "similarity",
                        "rerank_model_setting": {"rerank_enable": False},
                    })
                    documents.append({
                        "id": filename,
                        "filename": filename,
                        "name": filename,
                        "size": file_path.stat().st_size,
                        "modified_at": file_path.stat().st_mtime,
                        "chunk_strategy": cs.get("strategy", "standard"),
                        "chunk_size": cs.get("chunk_size", ""),
                        "chunk_overlap": cs.get("chunk_overlap", ""),
                        "separators": cs.get("separators", []),
                        "parent_chunk_size": cs.get("parent_chunk_size", ""),
                        "parent_chunk_overlap": cs.get("parent_chunk_overlap", ""),
                        "parent_separators": cs.get("parent_separators", []),
                        "child_chunk_size": cs.get("child_chunk_size", ""),
                        "child_chunk_overlap": cs.get("child_chunk_overlap", ""),
                        "child_separators": cs.get("child_separators", []),
                        "rag_config": rag,
                        "configured": has_config,
                        "on_disk": True,
                    })

    stale_files = [fname for fname in chunk_configs if fname not in seen_filenames]
    if stale_files and kb_row:
        try:
            raw = kb_row.chunk_retrieve_setting
            if isinstance(raw, dict):
                for fname in stale_files:
                    raw.pop(fname, None)
                kb_row.chunk_retrieve_setting = raw
            db.commit()
        except Exception as e:
            logger.warning(f"清理缺失文档的切片配置失败: {e}")
            db.rollback()

    return {"documents": documents, "kb_id": kb_id}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Query(..., description="知识库ID"),
    chunk_strategy: str = Query(None, description="切片策略，不传则使用知识库默认策略"),
    chunk_size: int = Query(None, description="最大Chunk长度，仅standard策略生效"),
    chunk_overlap: int = Query(None, description="Chunk重叠长度，仅standard策略生效"),
    separators: str = Query(None, description="分隔符列表(JSON数组或逗号分隔)，仅standard策略生效"),
    parent_chunk_size: int = Query(None, description="父段Chunk最大长度，仅parent_child策略生效"),
    parent_chunk_overlap: int = Query(None, description="父段Chunk重叠长度，仅parent_child策略生效"),
    parent_separators: str = Query(None, description="父段分隔符列表，仅parent_child策略生效"),
    child_chunk_size: int = Query(None, description="子段Chunk最大长度，仅parent_child策略生效"),
    child_chunk_overlap: int = Query(None, description="子段Chunk重叠长度，仅parent_child策略生效"),
    child_separators: str = Query(None, description="子段分隔符列表，仅parent_child策略生效"),
    embedding_model: str = Query(None, description="向量模型名称"),
    retrieval_mode: str = Query("hybrid", description="检索方式: vector/hybrid"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subdir = get_kb_subdir(current_user.username, kb_id)

    allowed_extensions = Params.DOC_SUPPORTED_FORMATS | {'.md', '.pdf'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式，支持的格式: {', '.join(allowed_extensions)}")

    kb_meta = None
    meta_list = load_kb_metadata(current_user.username)
    for kb in meta_list:
        if kb['id'] == kb_id:
            kb_meta = kb
            break

    cs_dict = kb_meta.get('chunk_strategy') if kb_meta else None
    if isinstance(cs_dict, str):
        cs_dict = _normalize_chunk_strategy(cs_dict)
    elif not cs_dict:
        cs_dict = _default_chunk_strategy()

    if chunk_strategy is not None and chunk_strategy != cs_dict.get('strategy'):
        cs_dict = _normalize_chunk_strategy(chunk_strategy)

    if cs_dict.get('strategy') == 'standard':
        if chunk_size is not None:
            cs_dict['chunk_size'] = chunk_size
        if chunk_overlap is not None:
            cs_dict['chunk_overlap'] = chunk_overlap
        if separators is not None:
            cs_dict['separators'] = _parse_separators(separators)
    elif cs_dict.get('strategy') == 'parent_child':
        if parent_chunk_size is not None:
            cs_dict['parent_chunk_size'] = parent_chunk_size
        if parent_chunk_overlap is not None:
            cs_dict['parent_chunk_overlap'] = parent_chunk_overlap
        if parent_separators is not None:
            cs_dict['parent_separators'] = _parse_separators(parent_separators)
        if child_chunk_size is not None:
            cs_dict['child_chunk_size'] = child_chunk_size
        if child_chunk_overlap is not None:
            cs_dict['child_chunk_overlap'] = child_chunk_overlap
        if child_separators is not None:
            cs_dict['child_separators'] = _parse_separators(child_separators)

    chunk_kwargs = _chunk_strategy_to_kwargs(cs_dict)

    file_path = subdir / file.filename
    try:
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)

        logger.info(f"用户 {current_user.username} 上传文档到知识库 {kb_id}: {file.filename}")

        _kb_add_document(current_user.username, kb_id, file.filename)

        # 只保存文件，不做向量化处理，等待用户配置切片策略后点击"设置"按钮再处理
        return {"status": "success", "message": "文档上传成功，请配置切片策略", "filename": file.filename, "document_id": file.filename}

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
    file_on_disk = file_path.exists()

    try:
        if file_on_disk:
            file_path.unlink()
            delete_document_vectors(filename, current_user.username, kb_id)
        _kb_remove_document(current_user.username, kb_id, filename)
        _kb_remove_chunk_config(current_user.username, kb_id, filename)
        logger.info(f"用户 {current_user.username} 从知识库 {kb_id} 删除文档: {filename}")
        return {"message": "文档删除成功", "filename": filename}
    except Exception as e:
        logger.error(f"文件删除失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")


@router.delete("/documents/{doc_id}")
async def delete_document_v2(
    doc_id: str,
    kb_id: str = Query(..., description="知识库ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subdir = get_user_knowledge_dir(current_user.username) / kb_id
    file_path = subdir / doc_id
    file_on_disk = file_path.exists()

    try:
        # 查找知识库记录
        kb_row = None
        all_kb_rows = db.query(Knowledge).filter(Knowledge.user_name == current_user.username).all()
        for row in all_kb_rows:
            if _extract_kb_id(row.knowledge_id) == kb_id:
                kb_row = row
                break

        if not kb_row:
            raise HTTPException(status_code=404, detail=f"知识库 {kb_id} 不存在")

        # 删除本地文件
        if file_on_disk:
            file_path.unlink()
            delete_document_vectors(doc_id, current_user.username, kb_id)

        # 深拷贝避免修改影响其他知识库记录
        import copy
        raw = copy.deepcopy(kb_row.chunk_retrieve_setting) if isinstance(kb_row.chunk_retrieve_setting, dict) else {}
        raw.pop(doc_id, None)

        # 更新 knowledge_id.document_paths
        norm = copy.deepcopy(_normalize_knowledge_id(kb_row.knowledge_id))
        norm['document_paths'] = [p for p in norm.get('document_paths', []) if p != doc_id]
        norm.pop('file_chunk_configs', None)

        from datetime import datetime
        # 使用 UPDATE 语句只更新当前知识库记录
        update_data = {
            Knowledge.chunk_retrieve_setting: raw,
            Knowledge.knowledge_id: norm,
            Knowledge.update_at: datetime.utcnow(),
        }
        db.query(Knowledge).filter(Knowledge.knowledge_name == kb_row.knowledge_name).update(update_data)
        db.commit()

        logger.info(f"用户 {current_user.username} 从知识库 {kb_id} 删除文档: {doc_id}")
        return {"message": "文档删除成功", "filename": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件删除失败: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")


@router.put("/documents/{doc_id}")
async def replace_document(
    doc_id: str,
    kb_id: str = Query(..., description="知识库ID"),
    file: UploadFile = File(...),
    chunk_strategy: str = Query(None),
    chunk_size: int = Query(None),
    chunk_overlap: int = Query(None),
    separators: str = Query(None),
    parent_chunk_size: int = Query(None),
    parent_chunk_overlap: int = Query(None),
    parent_separators: str = Query(None),
    child_chunk_size: int = Query(None),
    child_chunk_overlap: int = Query(None),
    child_separators: str = Query(None),
    embedding_model: str = Query(None),
    retrieval_mode: str = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subdir = get_user_knowledge_dir(current_user.username) / kb_id
    old_path = subdir / doc_id
    file_existed = old_path.exists()
    if file_existed:
        old_path.unlink()

    allowed_extensions = Params.DOC_SUPPORTED_FORMATS | {'.md', '.pdf'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式")

    chunk_kwargs = {}
    try:
        content = await file.read()
        new_path = subdir / file.filename
        with open(new_path, 'wb') as f:
            f.write(content)

        if file_existed:
            delete_document_vectors(doc_id, current_user.username, kb_id)
        _kb_remove_document(current_user.username, kb_id, doc_id)
        _kb_remove_chunk_config(current_user.username, kb_id, doc_id)
        _kb_add_document(current_user.username, kb_id, file.filename)

        cs_dict = None
        if chunk_strategy:
            cs_dict = _normalize_chunk_strategy(chunk_strategy)
        else:
            cs_dict = _default_chunk_strategy()

        if cs_dict.get('strategy') == 'standard':
            if chunk_size is not None: cs_dict['chunk_size'] = chunk_size
            if chunk_overlap is not None: cs_dict['chunk_overlap'] = chunk_overlap
            if separators is not None: cs_dict['separators'] = _parse_separators(separators)
        elif cs_dict.get('strategy') == 'parent_child':
            if parent_chunk_size is not None: cs_dict['parent_chunk_size'] = parent_chunk_size
            if parent_chunk_overlap is not None: cs_dict['parent_chunk_overlap'] = parent_chunk_overlap
            if parent_separators is not None: cs_dict['parent_separators'] = _parse_separators(parent_separators)
            if child_chunk_size is not None: cs_dict['child_chunk_size'] = child_chunk_size
            if child_chunk_overlap is not None: cs_dict['child_chunk_overlap'] = child_chunk_overlap
            if child_separators is not None: cs_dict['child_separators'] = _parse_separators(child_separators)

        chunk_kwargs = _chunk_strategy_to_kwargs(cs_dict)

        # 只保存文件，不做向量化处理，等待用户配置切片策略后点击"设置"按钮再处理
        logger.info(f"用户 {current_user.username} 替换知识库 {kb_id} 文档: {doc_id} -> {file.filename}")
        return {"status": "success", "message": "文档替换成功，请配置切片策略", "filename": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档替换失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档替换失败: {str(e)}")


@router.patch("/documents/{doc_id}")
async def update_document_settings(
    doc_id: str,
    kb_id: str = Query(..., description="知识库ID"),
    body: dict = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subdir = get_user_knowledge_dir(current_user.username) / kb_id
    file_path = subdir / doc_id
    file_on_disk = file_path.exists()

    try:
        chunk_strategy = body.get('chunk_strategy', 'standard')
        cs_dict = _normalize_chunk_strategy(chunk_strategy)

        if chunk_strategy == 'standard':
            if body.get('chunk_size') is not None: cs_dict['chunk_size'] = body['chunk_size']
            if body.get('chunk_overlap') is not None: cs_dict['chunk_overlap'] = body['chunk_overlap']
            if body.get('separators'): cs_dict['separators'] = body['separators']
        elif chunk_strategy == 'parent_child':
            if body.get('parent_chunk_size') is not None: cs_dict['parent_chunk_size'] = body['parent_chunk_size']
            if body.get('parent_chunk_overlap') is not None: cs_dict['parent_chunk_overlap'] = body['parent_chunk_overlap']
            if body.get('parent_separators'): cs_dict['parent_separators'] = body['parent_separators']
            if body.get('child_chunk_size') is not None: cs_dict['child_chunk_size'] = body['child_chunk_size']
            if body.get('child_chunk_overlap') is not None: cs_dict['child_chunk_overlap'] = body['child_chunk_overlap']
            if body.get('child_separators'): cs_dict['child_separators'] = body['child_separators']

        rag_config = body.get('rag_config') or {}
        embedding_model_name = rag_config.get('embedding_model_name', 'text-embedding-v4')
        rag_retrieve_method = rag_config.get('rag_retrieve_method', 'similarity')
        rerank_model_setting = rag_config.get('rerank_model_setting', {'rerank_enable': False})

        db_chunk_strategy = _build_db_chunk_strategy(chunk_strategy, cs_dict)

        kb_row = None
        all_kb_rows = db.query(Knowledge).filter(Knowledge.user_name == current_user.username).all()
        for row in all_kb_rows:
            if _extract_kb_id(row.knowledge_id) == kb_id:
                kb_row = row
                break

        if not kb_row:
            raise HTTPException(status_code=404, detail=f"知识库 {kb_id} 不存在")

        # 深拷贝避免修改影响其他知识库记录
        import copy
        raw = copy.deepcopy(kb_row.chunk_retrieve_setting) if isinstance(kb_row.chunk_retrieve_setting, dict) else {}
        raw[doc_id] = {
            "chunk_strategy": db_chunk_strategy,
            "embedding_model_name": embedding_model_name,
            "rag_retrieve_method": rag_retrieve_method,
            "rerank_model_setting": rerank_model_setting,
        }

        # 更新知识库名称和描述（如果前端传了）
        name = body.get('name')
        description = body.get('description')

        logger.info(f"更新知识库名称：{name} 描述：{description}")
        if name is not None:
            if isinstance(name, list):
                name = name[0] if name else ''
            if not str(name).strip():
                raise HTTPException(status_code=400, detail="知识库名称不能为空")
            name = str(name).strip()
        if description is not None:
            if isinstance(description, list):
                description = description[0] if description else ''
            description = str(description).strip()

        # 同步 knowledge_id.document_paths 与 chunk_retrieve_setting 的键
        norm = copy.deepcopy(_normalize_knowledge_id(kb_row.knowledge_id))
        norm['document_paths'] = list(raw.keys())
        # 移除多余的 file_chunk_configs 字段
        norm.pop('file_chunk_configs', None)

        from datetime import datetime
        # 使用 UPDATE 语句只更新当前知识库记录，避免影响其他记录
        update_data = {
            Knowledge.chunk_retrieve_setting: raw,
            Knowledge.knowledge_id: norm,
            Knowledge.update_at: datetime.utcnow(),
        }
        if name is not None:
            update_data[Knowledge.knowledge_name] = name
        if description is not None:
            update_data[Knowledge.knowledge_description] = description

        db.query(Knowledge).filter(Knowledge.knowledge_name == kb_row.knowledge_name).update(update_data)
        db.commit()

        if file_on_disk:
            chunk_kwargs = _chunk_strategy_to_kwargs(cs_dict)
            delete_document_vectors(doc_id, current_user.username, kb_id)
            await chunk_and_store_document(
                str(file_path), current_user.username, kb_id,
                chunk_strategy=chunk_kwargs['chunk_strategy'],
                chunk_size=chunk_kwargs.get('chunk_size'),
                chunk_overlap=chunk_kwargs.get('chunk_overlap'),
                separators=chunk_kwargs.get('separators'),
                parent_chunk_size=chunk_kwargs.get('parent_chunk_size'),
                parent_chunk_overlap=chunk_kwargs.get('parent_chunk_overlap'),
                parent_separators=chunk_kwargs.get('parent_separators'),
                child_chunk_size=chunk_kwargs.get('child_chunk_size'),
                child_chunk_overlap=chunk_kwargs.get('child_chunk_overlap'),
                child_separators=chunk_kwargs.get('child_separators'),
                embedding_model=embedding_model_name,
                retrieval_mode=rag_retrieve_method,
            )
            logger.info(f"用户 {current_user.username} 更新知识库 {kb_id} 文档 {doc_id} 的设置并重新向量化")
            return {"status": "success", "message": "文档设置保存成功，已重新向量化", "filename": doc_id}
        else:
            logger.info(f"用户 {current_user.username} 更新知识库 {kb_id} 文档 {doc_id} 的设置（文件不在磁盘，仅保存配置）")
            return {"status": "success", "message": "设置已保存，但源文件不在磁盘，需重新上传文档以触发向量化", "filename": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存文档设置失败: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"保存失败: {str(e)}")


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
        seen_parent_ids = set()
        for doc, raw in raw_results:
            score = convert_vector_score(raw, vector_store, EMBEDDING_DATABASE)
            if score < 0.2:
                continue
            chunk_strategy = doc.metadata.get('chunk_strategy', 'standard')
            parent_content = doc.metadata.get('parent_content')
            parent_id = doc.metadata.get('parent_id')

            if chunk_strategy == 'parent_child' and parent_id:
                if parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)
                display_content = parent_content if parent_content else doc.page_content
            else:
                display_content = doc.page_content

            documents.append({
                "content": display_content,
                "source": doc.metadata.get('source', ''),
                "kb_id": doc.metadata.get('kb_id', ''),
                "score": float(score),
                "chunk_strategy": chunk_strategy
            })

        documents.sort(key=lambda x: x['score'], reverse=True)
        for i, doc in enumerate(documents):
            doc['rank'] = i + 1

        return {"results": documents}

    except Exception as e:
        logger.error(f"知识库搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.post("/chunk_preview")
async def chunk_preview(
    file: UploadFile = File(...),
    chunk_strategy: str = Query("standard"),
    chunk_size: int = Query(None),
    chunk_overlap: int = Query(None),
    separators: str = Query(None),
    parent_chunk_size: int = Query(None),
    parent_chunk_overlap: int = Query(None),
    parent_separators: str = Query(None),
    child_chunk_size: int = Query(None),
    child_chunk_overlap: int = Query(None),
    child_separators: str = Query(None),
    current_user: User = Depends(get_current_user),
):
    try:
        import tempfile
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content_bytes = await file.read()
            tmp.write(content_bytes)
            tmp_path = tmp.name
        try:
            content = ""
            async for chunk in get_doc_processor().read_document(tmp_path):
                content += chunk
            if not content:
                return {"chunks": [], "message": "文档内容为空"}
            content = clean_document_content(content)
            if not content:
                return {"chunks": [], "message": "文档清洗后内容为空"}
            sep_list = _parse_separators(separators) if separators else None
            p_sep = _parse_separators(parent_separators) if parent_separators else None
            c_sep = _parse_separators(child_separators) if child_separators else None

            if chunk_strategy not in VALID_CHUNK_STRATEGIES:
                return {"chunks": [], "message": f"无效的切片策略: {chunk_strategy}"}

            chunk_kwargs = {}
            if chunk_strategy == "standard":
                chunk_kwargs = {
                    "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "separators": sep_list,
                }
            elif chunk_strategy == "parent_child":
                chunk_kwargs = {
                    "parent_chunk_size": parent_chunk_size, "parent_chunk_overlap": parent_chunk_overlap, "parent_separators": p_sep,
                    "child_chunk_size": child_chunk_size, "child_chunk_overlap": child_chunk_overlap, "child_separators": c_sep,
                }

            texts, metadatas = _chunk_with_strategy(content, chunk_strategy, file.filename, "preview", **chunk_kwargs)

            chunks = []
            for i, (text, meta) in enumerate(zip(texts, metadatas)):
                chunks.append({"index": i, "content": text, "length": len(text), "chunk_strategy": meta.get("chunk_strategy", chunk_strategy), "parent_id": meta.get("parent_id")})
            return {"chunks": chunks, "total": len(chunks)}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        logger.error(f"切片预览失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"切片预览失败: {str(e)}")

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
            if file_ext in allowed_extensions and not file_path.name.startswith('.') and file_path.name != 'Thumbs.db':
                document_files.append(file_path)

    if not document_files:
        return {"message": "未找到需要重新索引的文档"}

    meta_list = load_kb_metadata(current_user.username)
    cs_dict = _default_chunk_strategy()
    for kb in meta_list:
        if kb['id'] == kb_id:
            raw = kb.get('chunk_strategy')
            cs_dict = _normalize_chunk_strategy(raw)
            break

    chunk_kwargs = _chunk_strategy_to_kwargs(cs_dict)
    delete_kb_vectors(current_user.username, kb_id)

    success_count = 0
    fail_count = 0

    for file_path in document_files:
        try:
            await chunk_and_store_document(
                str(file_path), current_user.username, kb_id,
                chunk_strategy=chunk_kwargs['chunk_strategy'],
                chunk_size=chunk_kwargs.get('chunk_size'),
                chunk_overlap=chunk_kwargs.get('chunk_overlap'),
                separators=chunk_kwargs.get('separators'),
                parent_chunk_size=chunk_kwargs.get('parent_chunk_size'),
                parent_chunk_overlap=chunk_kwargs.get('parent_chunk_overlap'),
                parent_separators=chunk_kwargs.get('parent_separators'),
                child_chunk_size=chunk_kwargs.get('child_chunk_size'),
                child_chunk_overlap=chunk_kwargs.get('child_chunk_overlap'),
                child_separators=chunk_kwargs.get('child_separators'),
            )
            success_count += 1
        except Exception as e:
            logger.error(f"重新索引文档 {file_path.name} 失败: {str(e)}")
            fail_count += 1

    strategy_name = chunk_kwargs['chunk_strategy']
    logger.info(f"知识库 {kb_id} 重新索引完成: 成功 {success_count} 个, 失败 {fail_count} 个, 策略: {strategy_name}")
    return {"message": "重新索引完成", "success": success_count, "failed": fail_count, "chunk_strategy": cs_dict}


def get_all_users() -> list:
    try:
        from core.auth.database import get_db
        db = next(get_db())
        users = db.query(User).all()
        return [user.username for user in users]
    except Exception as e:
        logger.error(f"获取所有用户失败: {str(e)}")
        return []