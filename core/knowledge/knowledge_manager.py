import os
import time as _t

# 强制禁用 HuggingFace 离线模式，确保 fastembed 可以下载 BM25 模型
os.environ["HF_HUB_OFFLINE"] = "0"

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
_t0 = _t.time()

import json
import uuid
import shutil
import time
import asyncio
import functools
from pathlib import Path
print(f"[启动耗时] 基础import完成: {_t.time() - _t0:.2f}s")

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Body
from fastapi.security import HTTPAuthorizationCredentials
print(f"[启动耗时] fastapi import完成: {_t.time() - _t0:.2f}s")

from sqlalchemy.orm import Session
print(f"[启动耗时] sqlalchemy import完成: {_t.time() - _t0:.2f}s")

from core.auth.auth import get_current_user, security
print(f"[启动耗时] auth import完成: {_t.time() - _t0:.2f}s")

from core.auth.database import get_db, User, SessionLocal, Knowledge
print(f"[启动耗时] database import完成: {_t.time() - _t0:.2f}s")

from settings.Define import PathConfig, Params
print(f"[启动耗时] Define import完成: {_t.time() - _t0:.2f}s")
# 懒加载 qdrant_client（避免启动时加载）
_qdrant_models = None

def _get_qdrant_models():
    global _qdrant_models
    if _qdrant_models is None:
        from qdrant_client import models as qdrant_models
        _qdrant_models = qdrant_models
    return _qdrant_models
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
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

logger = get_logger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

import time as _time
_start_time = _time.time()
EMBEDDING_DATABASE = os.getenv("EMBEDDING_DATABASE", "chroma").strip().lower()
print(f"EMBEDDING_DATABASE={EMBEDDING_DATABASE} (耗时: {_time.time() - _start_time:.2f}s)")
QDRANT_URL = "http://127.0.0.1:6333"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 1

# HNSW 索引配置（全局默认值）
HNSW_CONFIG = {
    "M": 16,
    "ef_construct": 100,
    "ef": 128,
    "full_scan_threshold": 10000,
    "max_indexing_threads": 4,# 显式限制
}

# 懒加载重型库
_TextSplitter = None
_Chroma = None
_QdrantVectorStore = None
_RetrievalMode = None
_QdrantClient = None
_DashScopeEmbeddings = None
_SparseTextEmbedding = None

class RecurseCharacterTextSplitter:
    """自定义递归字符文本分割器（单例模式）。
    
    优势：
    1. 单例模式：确保全局只有一个实例
    2. 零依赖：不依赖 langchain_text_splitters，避免 164 秒的加载时间
    3. 高性能：纯 Python 实现，轻量快速
    """
    # 单例模式
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, *args, **kwargs):
        # 单例模式下，__init__ 可能被多次调用，需要防止重复初始化
        if not hasattr(self, "_initialized"):
            self._initialized = True
    
    def split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100, 
                   separators: list[str] = None) -> list[str]:
        """递归字符文本分割。
        
        按照分隔符优先级递归分割文本，确保每个块不超过 chunk_size。
        
        Args:
            text: 要分割的文本
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            separators: 分隔符列表（按优先级排序），默认使用 Params.SEPARATORS
        
        Returns:
            分割后的文本块列表
        """
        if separators is None:
            separators = Params.SEPARATORS
        
        # 如果文本长度小于 chunk_size，直接返回
        if len(text) <= chunk_size:
            return [text] if text else []
        
        # 尝试使用每个分隔符进行分割
        for i, separator in enumerate(separators):
            # 使用当前分隔符分割文本
            splits = text.split(separator)
            
            # 如果分割后有任何块超过 chunk_size，递归处理
            if any(len(s) > chunk_size for s in splits):
                # 继续用下一个分隔符递归分割
                result = []
                for s in splits:
                    if len(s) > chunk_size:
                        # 递归分割
                        remaining_separators = separators[i+1:] if i+1 < len(separators) else []
                        result.extend(self.split_text(s, chunk_size, chunk_overlap, remaining_separators))
                    else:
                        result.append(s)
                return self._merge_splits(result, separator, chunk_size, chunk_overlap)
            else:
                # 所有块都符合要求，合并并返回
                return self._merge_splits(splits, separator, chunk_size, chunk_overlap)
        
        # 如果所有分隔符都尝试过了，按字符强制分割
        return self._force_split(text, chunk_size, chunk_overlap)
    
    def _merge_splits(self, splits: list[str], separator: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """合并分割后的文本块，确保不超过 chunk_size 并保持重叠。
        
        关键改进：确保每个 chunk 都从自然断点开始，不会在句子中间切断。
        
        Args:
            splits: 分割后的文本块列表
            separator: 分隔符
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
        
        Returns:
            合并后的文本块列表
        """
        if not splits:
            return []
        
        merged = []
        current_chunk = splits[0]
        
        for i in range(1, len(splits)):
            next_split = splits[i]
            
            # 如果当前块加上下一个块不超过 chunk_size，合并
            if len(current_chunk) + len(separator) + len(next_split) <= chunk_size:
                current_chunk += separator + next_split
            else:
                # 保存当前块
                merged.append(current_chunk)
                
                # 关键修复：新块应该从当前块的末尾重叠部分开始
                # 但要确保从自然断点开始
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    # 找到当前块末尾的自然断点位置
                    overlap_start = self._find_natural_break_point(current_chunk, chunk_overlap)
                    overlap_text = current_chunk[overlap_start:]
                    
                    # 清理重叠文本开头的不合适字符（逗号、分号等中间停顿符号）
                    overlap_text = self._clean_chunk_start(overlap_text)
                    
                    current_chunk = overlap_text + separator + next_split
                else:
                    current_chunk = next_split
        
        # 添加最后一个块
        if current_chunk:
            merged.append(current_chunk)
        
        return merged
    
    def _clean_chunk_start(self, text: str) -> str:
        """清理 chunk 开头的不合适字符。
        
        移除不应该作为 chunk 开头的标点符号（逗号、分号等中间停顿符号）。
        
        Args:
            text: 原始文本
        
        Returns:
            清理后的文本
        """
        # 不应该作为 chunk 开头的字符（中间停顿符号）
        invalid_start_chars = ['，', '；', ',', ';', '、', '：', ':']
        
        # 跳过开头的无效字符
        start = 0
        while start < len(text) and text[start] in invalid_start_chars:
            start += 1
        
        # 同时跳过可能的空格
        while start < len(text) and text[start] in [' ', '\t']:
            start += 1
        
        return text[start:] if start < len(text) else text
    
    def _find_natural_break_point(self, text: str, max_overlap: int) -> int:
        """在文本末尾找到自然断点位置，确保新 chunk 从句子/段落开头开始。
        
        策略：优先在 max_overlap 范围内查找，如果没找到则适当扩大搜索范围。
        宁可稍微超过 max_overlap，也不要切断句子。
        
        Args:
            text: 完整文本
            max_overlap: 最大重叠字符数
        
        Returns:
            自然断点的起始位置索引
        """
        if len(text) <= max_overlap:
            return 0
        
        # 第一优先级：在 max_overlap 范围内查找换行符
        search_start = max(0, len(text) - max_overlap)
        for i in range(len(text) - 1, search_start - 1, -1):
            if text[i] == '\n':
                return i + 1
        
        # 第二优先级：在 max_overlap 范围内查找标题标记
        for i in range(len(text) - 1, search_start - 1, -1):
            if text[i] == '#' and (i == 0 or text[i-1] == '\n'):
                return i
        
        # 第三优先级：在 max_overlap 范围内查找句子结束符
        for i in range(len(text) - 1, search_start - 1, -1):
            if text[i] in ['。', '！', '？', '\r']:
                j = i + 1
                while j < len(text) and text[j] in [' ', '\t']:
                    j += 1
                return j
        
        # 如果 max_overlap 范围内没找到，扩大搜索范围到 2 倍
        extended_search_start = max(0, len(text) - max_overlap * 2)
        
        # 第四优先级：在扩展范围内查找换行符
        for i in range(len(text) - 1, extended_search_start - 1, -1):
            if text[i] == '\n':
                return i + 1
        
        # 第五优先级：在扩展范围内查找句子结束符
        for i in range(len(text) - 1, extended_search_start - 1, -1):
            if text[i] in ['。', '！', '？', '\r']:
                j = i + 1
                while j < len(text) and text[j] in [' ', '\t']:
                    j += 1
                return j
        
        # 最后：如果还是没找到，返回默认位置（硬切）
        return search_start
    
    def _force_split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        """强制按字符分割文本（最后手段）。
        
        Args:
            text: 要分割的文本
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
        
        Returns:
            分割后的文本块列表
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # 移动到下一个块，保留重叠
            start = end - chunk_overlap if chunk_overlap < end else end
        
        return chunks



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


def _get_embeddings_class():
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
        # from fastembed import SparseTextEmbedding
        from langchain_qdrant import FastEmbedSparse
        _SparseTextEmbedding = FastEmbedSparse
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


# Qdrant 客户端单例（全局共享，避免重复创建）
_qdrant_client = None
_qdrant_client_lock = None  # 用于线程安全

def _create_qdrant_client(timeout: int = 30):
    """获取 Qdrant 客户端单例（线程安全），默认超时30秒"""
    global _qdrant_client, _qdrant_client_lock
    
    if _qdrant_client is None:
        # 初始化锁（如果还没有）
        if _qdrant_client_lock is None:
            import threading
            _qdrant_client_lock = threading.Lock()
        
        # 双重检查锁定模式
        with _qdrant_client_lock:
            if _qdrant_client is None:
                _, _, QdrantClient = _get_qdrant()
                logger.info(f"创建全局 Qdrant 客户端单例 (timeout={timeout}s)")
                _qdrant_client = QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=timeout)
    
    return _qdrant_client


def _create_temp_qdrant_client(timeout: int = 10):
    """创建临时 Qdrant 客户端（短超时，用完即关）"""
    _, _, QdrantClient = _get_qdrant()
    return QdrantClient(url=QDRANT_URL, prefer_grpc=False, timeout=timeout, check_compatibility=False)


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
                        # 初始化阶段明确不可达的错误不重试，直接抛出让用户看到
                        is_retryable = (
                            "10051" in msg
                            or "10053" in msg
                            or "connection" in msg.lower()
                            or "qdrant" in msg.lower()
                            or "unreachable" in msg.lower()
                        ) and "不可达" not in msg and "无法初始化" not in msg
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
                        # 初始化阶段明确不可达的错误不重试，直接抛出让用户看到
                        is_retryable = (
                            "10051" in msg
                            or "10053" in msg
                            or "connection" in msg.lower()
                            or "qdrant" in msg.lower()
                            or "unreachable" in msg.lower()
                        ) and "不可达" not in msg and "无法初始化" not in msg
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
    返回 {filename: {embedding_model_name, rag_retrieve_method, score_thresh_setting, rerank_model_setting, rrf_setting}}
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
                                "score_thresh_setting": config.get("score_thresh_setting", {"enable": False}),
                                "rerank_model_setting": config.get("rerank_model_setting", {"rerank_enable": False}),
                                "rrf_setting": config.get("rrf_setting", {"vector_weight": 0.7, "rrf_k": 60}),
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
_vector_store_cache = {}


def _get_dense_embeddings(username: str, model_name: str = None):
    cache_key = f"{username}:{model_name or 'default'}"
    if cache_key not in _embeddings_cache:
        model = model_name or Params.DEFAULT_EMBEDDING_MODEL
        # DashScopeEmbeddings = _get_embeddings_class()
        # embeddings = DashScopeEmbeddings(model=model)
        # 初始化 Dense Embedding（语义向量）
        from langchain_openai import OpenAIEmbeddings
        import httpx
        
        # 配置 HTTP 客户端超时
        http_client = httpx.Client(
            timeout=httpx.Timeout(300.0, connect=30.0, read=300.0),  # 5分钟超时
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        
        dense_embeddings = OpenAIEmbeddings(
            model=model,
            api_key=DASHSCOPE_API_KEY,
            base_url=Params.API_BASE,  # ← 关键！设置DashScope兼容接口地址
            check_embedding_ctx_length=False,
            request_timeout=300,  # 增加到 5 分钟
            max_retries=2,  # 添加重试机制
            http_client=http_client,  # 使用自定义 HTTP 客户端
        )
        logger.info(f"DashScope 向量模型已初始化: {model}, 超时: 300s, base_url: https://dashscope.aliyuncs.com/compatible-mode/v1")
        _embeddings_cache[cache_key] = dense_embeddings
    return _embeddings_cache[cache_key]


def _get_sparse_embeddings(username: str):
    if username not in _sparse_embeddings_cache:
        from langchain_qdrant import FastEmbedSparse
        SparseTextEmbedding = _get_sparse_embedding()
        # 使用本地已下载的 BM25 模型，避免 HuggingFace 下载问题
        # fastembed 期望的目录结构: cache_dir/models--Qdrant--bm25/snapshots/xxx/
        models_dir = Path(__file__).parent.parent.parent / "models"
        sparse = FastEmbedSparse(
            model_name="Qdrant/bm25",
            cache_dir=str(models_dir),
        )
        logger.info(f"BM25 稀疏向量模型已初始化，缓存目录: {models_dir}")
        _sparse_embeddings_cache[username] = sparse
    return _sparse_embeddings_cache[username]


def _ensure_collection_exists(username: str):
    qdrant_models = _get_qdrant_models()

    collection_name = f"knowledge_{username}"
    temp_client = _create_temp_qdrant_client(timeout=5)
    try:
        collections = temp_client.get_collections()
        existing_names = [c.name for c in collections.collections]
        if collection_name not in existing_names:
            temp_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    # Dense 向量（语义检索）
                    "dense": qdrant_models.VectorParams(size=1024, distance=qdrant_models.Distance.COSINE,),
                    # Sparse 向量（关键词检索）—— Qdrant 原生支持，无需指定维度
                },
                sparse_vectors_config={
                    "sparse": qdrant_models.SparseVectorParams(),  # 统一使用"sparse"名称
                },
                # HNSW 索引配置
                hnsw_config=qdrant_models.HnswConfigDiff(
                    m=HNSW_CONFIG["M"],
                    ef_construct=HNSW_CONFIG["ef_construct"],
                    full_scan_threshold=HNSW_CONFIG["full_scan_threshold"],
                    max_indexing_threads=HNSW_CONFIG["max_indexing_threads"],
                ),
            )
            logger.info(f"已创建 Qdrant 集合: {collection_name} (H, HNSW: M={HNSW_CONFIG['M']}, ef_construct={HNSW_CONFIG['ef_construct']})")
    except Exception as e:
        logger.warning(f"⚠️ Qdrant 连接失败，跳过集合创建: {str(e)}")
    finally:
        temp_client.close()


def _init_qdrant_collection(client, collection_name: str, vector_size: int):
    """初始化 Qdrant Collection（如已存在则跳过），使用临时客户端避免阻塞"""
    qdrant_models = _get_qdrant_models()
    temp_client = _create_temp_qdrant_client(timeout=5)
    try:
        collections = [c.name for c in temp_client.get_collections().collections]
        if collection_name not in collections:
            temp_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": qdrant_models.VectorParams(size=vector_size, distance=qdrant_models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": {},
                },
                # HNSW 索引配置
                hnsw_config=qdrant_models.HnswConfigDiff(
                    m=HNSW_CONFIG["M"],
                    ef_construct=HNSW_CONFIG["ef_construct"],
                    full_scan_threshold=HNSW_CONFIG["full_scan_threshold"],
                    max_indexing_threads=HNSW_CONFIG["max_indexing_threads"],
                ),
            )
            logger.info(f"✅ Collection '{collection_name}' 创建成功 (dense_dim={vector_size} + sparse, HNSW: M={HNSW_CONFIG['M']}, ef_construct={HNSW_CONFIG['ef_construct']})")
        else:
            logger.info(f"ℹ️  Collection '{collection_name}' 已存在，跳过创建")
        return True
    except Exception as e:
        logger.error(f"❌ Qdrant 不可达，Collection 初始化失败: {str(e)}")
        return False
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
    embeddings = _get_dense_embeddings(username, model_name)
    persist_dir = _get_chroma_persist_dir(username)
    Chroma = _get_chroma()
    return Chroma(
        collection_name=f"knowledge_{username}",
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )

# QdrantVectorStore 缓存（按 username + model_name + retrieval_mode 缓存）
_qdrant_vector_store_cache = {}

def _build_qdrant_vector_store(username: str, model_name: str = None, retrieval_mode: str = "similarity"):
    # 使用缓存 key 避免重复创建
    cache_key = f"{username}:{model_name or 'default'}:{retrieval_mode}"
    if cache_key in _qdrant_vector_store_cache:
        logger.info(f"使用缓存的 QdrantVectorStore: {cache_key}")
        return _qdrant_vector_store_cache[cache_key]
    
    logger.info(f"创建新的 QdrantClient: {cache_key}")
    client = _create_qdrant_client()
    logger.info("QdrantClient已创建，准备初始化 Qdrant Collection")
    embeddings = _get_dense_embeddings(username, model_name)
    sparse_embeddings = _get_sparse_embeddings(username)
    collection_name = f"knowledge_{username}"
    init_ok = _init_qdrant_collection(client, collection_name, vector_size=1024)
    if not init_ok:
        raise RuntimeError(
            f"Qdrant 不可达（地址: {QDRANT_URL}），无法初始化 Collection '{collection_name}'。"
            f"请检查 Qdrant 服务是否正常运行，或在 .env 中将 EMBEDDING_DATABASE 改为 chroma。"
        )

    QdrantVectorStore, RetrievalMode, _ = _get_qdrant()
    mode = RetrievalMode.HYBRID if retrieval_mode == "hybrid" else RetrievalMode.DENSE
    logger.info("开始初始化 QdrantVectorStore")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=mode,
        validate_collection_config=False,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    logger.info("QdrantVectorStore 初始化完成")

    # 存入缓存
    _qdrant_vector_store_cache[cache_key] = vector_store
    return vector_store

def get_vector_store(username: str, model_name: str = None, retrieval_mode: str = "similarity"):
    cache_key = f"{username}:{model_name or 'default'}:{retrieval_mode}"
    if cache_key not in _vector_store_cache:
        logger.info(f"创建新的 VectorStore 缓存: {cache_key}")
        _vector_store_cache[cache_key] = _build_vector_store(username, model_name, retrieval_mode)
    else:
        logger.info(f"使用缓存的 VectorStore: {cache_key} (当前缓存数: {len(_vector_store_cache)})")
    return _vector_store_cache[cache_key]


@retry_qdrant()
def get_vector_count(username: str) -> int:
    try:
        vector_store = get_vector_store(username)
        collection_name = f"knowledge_{username}"
        count_response = vector_store._client.count(collection_name=collection_name)
        return count_response.count
    except Exception as e:
        logger.error(f"获取向量数量失败: {str(e)}")
        return 0

# 数据清洗
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
    # 1. 去除 Markdown 分割线 --- 或 ***
    content = re.sub(r'\n-{3,}\n', '', content)
    content = re.sub(r'\n\*{3,}\n', '', content)
    # 2. (可选) 去除多余的连续空行，保持最多两个换行
    content = re.sub(r'\n{3,}', '\n\n', content)

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
      - Markdown 表格格式：| 问题 | 答案 |  （Excel转换后的格式）
      - Q: ... A: ... / Q：... A：...
      - 问：... 答：...
      - 问题：... 答案：...
    若未检测到问答格式则返回空列表，由调用方回退到标准分段。
    """
    import re
    import json

    qa_pairs = []

    print(f"_split_by_qa_pattern: {content[:140]}")

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

    # 2. 尝试解析 Markdown 表格格式的 QA 数据（Excel转换后的格式）
    # 匹配格式：| 问题 | 答案 | 或 | Question | Answer | 等
    lines = content.split('\n')
    header_found = False
    q_col_idx = None
    a_col_idx = None
    total_cols = None
    
    for line in lines:
        line_stripped = line.strip()
        # 跳过空行和分隔符行（如 | --- | --- |）
        if not line_stripped or line_stripped.startswith('|---') or re.match(r'^\|[\s\-:|]+\|$', line_stripped):
            continue
        
        # 检查是否是表头行（包含"问题"或"question"和"答案"或"answer"）
        if not header_found and '|' in line_stripped:
            # 使用更智能的方法解析表头：按 | 分割，但考虑转义的 \|
            # 先替换转义的 | 为占位符
            temp_line = line_stripped.replace('\\|', '%%ESCAPED_PIPE%%')
            header_cells = [cell.strip().replace('%%ESCAPED_PIPE%%', '|') for cell in temp_line.split('|') if cell.strip()]
            
            is_header = False
            temp_q_idx = None
            temp_a_idx = None
            
            for idx, cell in enumerate(header_cells):
                cell_lower = cell.lower()
                if '问题' in cell or 'question' in cell_lower or cell_lower in ['q', '问']:
                    temp_q_idx = idx
                    is_header = True
                elif '答案' in cell or 'answer' in cell_lower or cell_lower in ['a', '答']:
                    temp_a_idx = idx
                    is_header = True
            
            if is_header and temp_q_idx is not None and temp_a_idx is not None:
                header_found = True
                q_col_idx = temp_q_idx
                a_col_idx = temp_a_idx
                total_cols = len(header_cells)
                continue
            elif is_header and len(header_cells) >= 2:
                # 如果找到表头但无法明确识别列，默认第一列是问题，第二列是答案
                header_found = True
                q_col_idx = 0
                a_col_idx = 1
                total_cols = len(header_cells)
                continue
        
        # 解析数据行
        if header_found and '|' in line_stripped:
            # 使用同样的方法解析数据行：考虑转义的 \|
            temp_line = line_stripped.replace('\\|', '%%ESCAPED_PIPE%%')
            cells = [cell.strip().replace('%%ESCAPED_PIPE%%', '|') for cell in temp_line.split('|') if cell.strip()]
            
            # 如果单元格数量与表头不匹配，尝试修复
            if len(cells) != total_cols and total_cols is not None:
                # 可能是某些单元格为空，需要补齐
                while len(cells) < total_cols:
                    cells.append('')
                # 或者某些单元格被错误分割，需要合并
                if len(cells) > total_cols:
                    # 合并多余的单元格到最后一列
                    cells = cells[:total_cols-1] + [' | '.join(cells[total_cols-1:])]
            
            if len(cells) > max(q_col_idx, a_col_idx):
                q = cells[q_col_idx]
                a = cells[a_col_idx]
                # 跳过分隔符行（如 ---）
                if q and a and q != '---' and a != '---':
                    qa_pairs.append("问：{}\n答：{}".format(q, a))
    
    if qa_pairs:
        return qa_pairs

    # 3. 正则匹配文本格式的 QA
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

    # 4. 逐行解析文本格式的 QA
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
    file_path: str = None,
    metadata_tags: list = None,
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
    import time
    chunk_start = time.time()
    logger.info(f"开始文档切块: 策略={strategy}, 内容长度={len(content)}")
    
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
            logger.info(f"QA切块完成: {len(qa_pairs)} 个问答对, 耗时: {time.time() - chunk_start:.2f}s")
            return texts_to_add, metadatas_to_add
        logger.warning("未检测到问答格式")
        return [], []
    elif strategy == "parent_child":
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

        logger.info(f"父子切块参数: 父段(size={p_size}, overlap={p_overlap}), 子段(size={c_size}, overlap={c_overlap})")
        
        logger.info("创建父段分割器...")
        parent_splitter = RecurseCharacterTextSplitter()
        logger.info("创建子段分割器...")
        child_splitter = RecurseCharacterTextSplitter()

        logger.info("开始父段切分...")
        parents = parent_splitter.split_text(content, chunk_size=p_size, chunk_overlap=p_overlap, separators=p_separators)
        logger.info(f"父段切分完成: {len(parents)} 个父段")
        
        for idx, parent_text in enumerate(parents):
            if idx % 10 == 0:
                logger.info(f"处理父段 {idx}/{len(parents)}...")
            children = child_splitter.split_text(parent_text,
                                                 chunk_size=c_size,
                                                 chunk_overlap=c_overlap,
                                                 separators=c_separators)
            if not children:
                texts_to_add.append(parent_text)
                metadatas_to_add.append({
                    "source": filename, "kb_id": kb_id,
                    "chunk_strategy": "parent_child",
                    "parent_id": f"{filename}_{idx}",
                    "parent_content": parent_text,
                })
                continue
            
            # 过滤掉过小的纯重叠 chunk（小于目标大小的 30%）
            min_child_size = int(c_size * 0.3)
            valid_children = [child for child in children if len(child) >= min_child_size]
            
            # 如果过滤后没有有效子段，保留父段本身
            if not valid_children:
                texts_to_add.append(parent_text)
                metadatas_to_add.append({
                    "source": filename, "kb_id": kb_id,
                    "chunk_strategy": "parent_child",
                    "parent_id": f"{filename}_{idx}",
                    "parent_content": parent_text,
                })
                continue
            
            # 记录被过滤的 chunk 数量
            filtered_count = len(children) - len(valid_children)
            if filtered_count > 0:
                logger.debug(f"父段 {idx}: 过滤掉 {filtered_count} 个过小的重叠 chunk")
            
            for child_text in valid_children:
                texts_to_add.append(child_text)
                metadatas_to_add.append({
                    "source": filename, "kb_id": kb_id,
                    "chunk_strategy": "parent_child",
                    "parent_id": f"{filename}_{idx}",
                    "parent_content": parent_text,
                })
        
        logger.info(f"父子切块完成: {len(texts_to_add)} 个子段, 耗时: {time.time() - chunk_start:.2f}s")
        return texts_to_add, metadatas_to_add
    elif strategy == "standard":
        logger.info(f"标准切块参数: size={base_size}, overlap={base_overlap}")
        text_splitter = RecurseCharacterTextSplitter()
        logger.info("开始标准切分...")
        chunks = text_splitter.split_text(content, chunk_size=base_size, chunk_overlap=base_overlap,
                                          separators=base_separators)
        logger.info(f"标准切块完成: {len(chunks)} 个块, 耗时: {time.time() - chunk_start:.2f}s")
        for chunk in chunks:
            texts_to_add.append(chunk)
            metadatas_to_add.append({"source": filename, "kb_id": kb_id, "chunk_strategy": "standard"})
        return texts_to_add, metadatas_to_add
    elif strategy == "table":
        # 表格切片模式：使用实际文件路径读取Excel/CSV
        from core.knowledge.table_chunk import TableChunk
        actual_file_path = file_path if file_path else filename
        table_chunk = TableChunk(actual_file_path)
        table_chunk.set_metaflag(metadata_tags if metadata_tags else [])
        chunks = table_chunk.chunk_table()
        for chunk in chunks:
            texts_to_add.append(chunk["chunk_text"])
            metadata = chunk["metadata"]
            metadata["source"] = filename
            metadata["kb_id"] = kb_id
            metadata["chunk_strategy"] = "table"
            metadatas_to_add.append(metadata)

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
    metadata_tags: list = None,
):
    try:
        import time
        start_time = time.time()
        logger.info(f"开始处理文档: {file_path}, 用户: {username}, 知识库: {kb_id}, 切片策略: {chunk_strategy}, 检索模式: {retrieval_mode}")

        # 1. 读取文档
        read_start = time.time()
        content = ""
        async for chunk in get_doc_processor().read_document(file_path):
            content += chunk
        logger.info(f"文档读取完成: {time.time() - read_start:.2f}s, 内容长度: {len(content) if content else 0}")

        if not content:
            logger.warning(f"文档内容为空: {file_path}")
            raise ValueError("文档内容为空，无法进行向量化")

        # 2. 清洗文档
        clean_start = time.time()
        content = clean_document_content(content)
        logger.info(f"文档清洗完成: {time.time() - clean_start:.2f}s, 内容长度: {len(content) if content else 0}")

        if not content:
            logger.warning(f"文档清洗后内容为空: {file_path}")
            return

        # 3. 获取向量存储
        store_start = time.time()
        vector_store = get_vector_store(username, embedding_model, retrieval_mode)
        logger.info(f"获取向量存储完成: {time.time() - store_start:.2f}s")
        
        filename = os.path.basename(file_path)

        chunk_kwargs = {
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "separators": separators,
            "parent_chunk_size": parent_chunk_size, "parent_chunk_overlap": parent_chunk_overlap, "parent_separators": parent_separators,
            "child_chunk_size": child_chunk_size, "child_chunk_overlap": child_chunk_overlap, "child_separators": child_separators,
        }
        
        # 表格切片：添加元数据标签和文件路径
        if chunk_strategy == "table":
            chunk_kwargs["file_path"] = file_path
            chunk_kwargs["metadata_tags"] = metadata_tags

        # 4. 文档切块
        chunk_start = time.time()
        texts_to_add, metadatas_to_add = _chunk_with_strategy(content, chunk_strategy, filename, kb_id, **chunk_kwargs)
        logger.info(f"文档切块完成: {time.time() - chunk_start:.2f}s, 共 {len(texts_to_add)} 个块")

        if texts_to_add:
            from langchain_core.documents import Document
            
            embed_start = time.time()
            logger.info(f"开始向量化: {len(texts_to_add)} 个文本块")
            
            # 构建 Document 对象列表
            documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(texts_to_add, metadatas_to_add)
            ]
            
            # DashScope API 限制 batch_size 最大为 10
            batch_size = 10  # DashScope API 硬性限制
            try:
                # 批量写入（同步方式）
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    vector_store.add_documents(documents=batch, batch_size=batch_size)
                
                embed_elapsed = time.time() - embed_start
                logger.info(f"向量化完成: {len(documents)} 段, 耗时 {embed_elapsed:.2f}s, 吞吐 {len(documents) / embed_elapsed:.0f} docs/s")
            except Exception as e:
                embed_elapsed = time.time() - embed_start
                err_msg = str(e)
                # 区分错误来源：嵌入模型API（如欠费、鉴权失败） vs 向量库（Qdrant/Chroma）
                if any(kw in err_msg for kw in ("Arrearage", "Access denied", "401", "403", "Unauthorized",
                                                "api_key", "API key", "invalid_api_key", "overdue",
                                                "rate limit", "RateLimit", "insufficient")):
                    source = "嵌入模型API"
                else:
                    source = "向量库(Qdrant/Chroma)"
                logger.warning(f"⚠️ 向量化失败 (耗时 {embed_elapsed:.2f}s, 来源: {source}): {err_msg}")
                # 重新抛出，让调用方（后台任务）知道向量化失败
                raise

        else:
            logger.warning(f"文档切块结果为空: {file_path}")

        logger.info(f"文档处理总耗时: {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.warning(f"⚠️ 文档处理失败 (共耗时 {time.time() - start_time:.2f}s): {str(e)}")


@retry_qdrant()
def delete_document_vectors(filename: str, username: str, kb_id: str = None, model_name: str = None, retrieval_mode: str = "similarity"):
    """删除文档对应的向量（直接使用 Qdrant 客户端，不依赖向量模型）"""
    # 创建一个临时客户端，使用较短的超时时间（5秒）
    temp_client = _create_temp_qdrant_client(timeout=5)
    collection_name = f"knowledge_{username}"

    try:
        from qdrant_client.http import models as qdrant_models
        
        filter_conditions = [
            qdrant_models.FieldCondition(
                key="metadata.source",
                match=qdrant_models.MatchValue(value=filename)
            )
        ]
        
        if kb_id is not None:
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="metadata.kb_id",
                    match=qdrant_models.MatchValue(value=kb_id)
                )
            )
        
        temp_client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.Filter(must=filter_conditions)
        )
        logger.info(f"已删除文档 {filename} 对应的向量")
    finally:
        # 关闭临时客户端
        try:
            temp_client.close()
        except:
            pass

@retry_qdrant()
def delete_kb_vectors(username: str, kb_id: str):
    try:
        vector_store = get_vector_store(username)
        collection_name = f"knowledge_{username}"
        
        try:
            if EMBEDDING_DATABASE == "qdrant":
                from qdrant_client.http import models as qdrant_models
                vector_store._client.delete(
                    collection_name=collection_name,
                    points_selector=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="metadata.kb_id",
                                match=qdrant_models.MatchValue(value=kb_id)
                            )
                        ]
                    )
                )
            else:
                vector_store._collection.delete(where={"kb_id": kb_id})
            logger.info(f"已删除知识库 {kb_id} 对应的所有向量")
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

VALID_CHUNK_STRATEGIES = {"standard", "parent_child", "qa", "table"}


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
    elif strategy == "table":
        return {"strategy": "table", "metadata_tags": cs.get("metadata_tags", [])}
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
    elif strategy == "table":
        result = {"strategy": "table"}
        if isinstance(cs_dict, dict) and cs_dict.get("metadata_tags"):
            result["metadata_tags"] = cs_dict["metadata_tags"]
        return result
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
    elif strategy == "table":
        return {"strategy": "table"}
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
    elif strategy == "table":
        return {"chunk_strategy": "table"}
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
    
    注意：自动移除不适合作为分隔符的 `\n`（单换行），因为它会切断句子。
    """
    if not separators_raw:
        return Params.SEPARATORS
    # 如果已经是列表，直接处理
    if isinstance(separators_raw, list):
        decoded = []
        for s in separators_raw:
            s = str(s).replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
            decoded.append(s)
        # 过滤掉单换行符（会切断句子）
        return [s for s in decoded if s != "\n"] if decoded else Params.SEPARATORS
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
    
    # 过滤掉单换行符（会切断句子）
    return [s for s in decoded if s != "\n"]


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
            "score_thresh_setting": rc.get("score_thresh_setting", {"enable": False}),
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
    chunk_strategy: str = Query(None, description="切片策略: standard/parent_child/qa/table"),
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

    kb_row = None
    all_kb_rows = db.query(Knowledge).filter(Knowledge.user_name == current_user.username).all()
    for row in all_kb_rows:
        if _extract_kb_id(row.knowledge_id) == kb_id:
            kb_row = row
            break

    if not kb_row:
        raise HTTPException(status_code=404, detail="知识库不存在")

    import copy
    from datetime import datetime

    update_data = {Knowledge.update_at: datetime.now()}

    if name is not None:
        if not name.strip():
            raise HTTPException(status_code=400, detail="知识库名称不能为空")
        update_data[Knowledge.knowledge_name] = name.strip()

    if description is not None:
        update_data[Knowledge.knowledge_description] = description.strip()

    if chunk_strategy is not None:
        raw = copy.deepcopy(kb_row.chunk_retrieve_setting) if isinstance(kb_row.chunk_retrieve_setting, dict) else {}

        for fname in list(raw.keys()):
            if isinstance(raw[fname], dict) and "chunk_strategy" in raw[fname]:
                file_cs_dict = raw[fname]["chunk_strategy"]
                old_file_strategy = file_cs_dict.get("strategy", "standard")

                new_file_cs = _normalize_chunk_strategy(chunk_strategy)
                if cs is not None: new_file_cs['chunk_size'] = cs
                if co is not None: new_file_cs['chunk_overlap'] = co
                if sep_list is not None: new_file_cs['separators'] = sep_list
                if pcs is not None: new_file_cs['parent_chunk_size'] = pcs
                if pco is not None: new_file_cs['parent_chunk_overlap'] = pco
                if p_sep is not None: new_file_cs['parent_separators'] = p_sep
                if ccs is not None: new_file_cs['child_chunk_size'] = ccs
                if cco is not None: new_file_cs['child_chunk_overlap'] = cco
                if c_sep is not None: new_file_cs['child_separators'] = c_sep

                raw[fname]["chunk_strategy"] = _build_db_chunk_strategy(chunk_strategy, new_file_cs)

        update_data[Knowledge.chunk_retrieve_setting] = raw

    if update_data:
        db.query(Knowledge).filter(Knowledge.id == kb_row.id).update(update_data)
        db.commit()
        logger.info(f"已更新知识库: {kb_id}, 更新字段: {list(update_data.keys())}")

    meta_list = load_kb_metadata(current_user.username)
    found = False
    for kb in meta_list:
        if kb['id'] == kb_id:
            if name is not None:
                kb['name'] = name.strip()
            if description is not None:
                kb['description'] = description.strip()
            if chunk_strategy is not None:
                kb['chunk_strategy'] = _normalize_chunk_strategy(chunk_strategy)
            found = True
            break
    if found:
        save_kb_metadata(current_user.username, meta_list)

    return {"message": "更新成功"}


@router.delete("/bases/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    from sqlalchemy import or_

    meta_list = load_kb_metadata(current_user.username)
    meta_list = [kb for kb in meta_list if kb['id'] != kb_id]
    save_kb_metadata(current_user.username, meta_list)

    kb_row = None
    all_kb_rows = db.query(Knowledge).filter(Knowledge.user_name == current_user.username).all()
    for row in all_kb_rows:
        if _extract_kb_id(row.knowledge_id) == kb_id:
            kb_row = row
            break

    if kb_row:
        try:
            db.delete(kb_row)
            db.commit()
            logger.info(f"已从数据库删除知识库记录: {kb_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"删除数据库记录失败: {e}")

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
    metadata_tags_map = {}
    if kb_row and kb_row.chunk_retrieve_setting and isinstance(kb_row.chunk_retrieve_setting, dict):
        raw = kb_row.chunk_retrieve_setting
        for fname, config in raw.items():
            if not isinstance(config, dict):
                continue
            cs = config.get("chunk_strategy", {})
            if cs:
                chunk_configs[fname] = _normalize_inner_cs(cs)
                mt_list = cs.get("metadata_tags", []) if isinstance(cs, dict) else []
                if isinstance(mt_list, list):
                    mt_dict = {}
                    for item in mt_list:
                        if isinstance(item, dict):
                            tags = item.get("tags", [])
                            if not isinstance(tags, list):
                                continue
                            sheet = item.get("sheet")
                            if sheet:
                                mt_dict[sheet] = {tag: True for tag in tags}
                            else:
                                mt_dict['_csv_default'] = {tag: True for tag in tags}
                    if mt_dict:
                        metadata_tags_map[fname] = mt_dict
            rag_configs[fname] = {
                "embedding_model_name": config.get("embedding_model_name", "text-embedding-v4"),
                "rag_retrieve_method": config.get("rag_retrieve_method", "similarity"),
                "score_thresh_setting": config.get("score_thresh_setting", {"enable": False}),
                "rerank_model_setting": config.get("rerank_model_setting", {"rerank_enable": False}),
                "rrf_setting": config.get("rrf_setting", {"vector_weight": 0.7, "rrf_k": 60}),
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
                        "score_thresh_setting": {"enable": False},
                        "rerank_model_setting": {"rerank_enable": False},
                        "rrf_setting": {"vector_weight": 0.7, "rrf_k": 60},
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
                        "metadata_tags": metadata_tags_map.get(filename, {}),
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


@router.delete("/documents/{doc_id:path}")
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


@router.put("/{kb_id}/documents/{doc_id}")
async def replace_document(
    doc_id: str,
    kb_id: str,
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


@router.patch("/documents/{doc_id:path}")
async def update_document_settings(
    doc_id: str,
    kb_id: str = Query(..., description="知识库ID"),
    body: dict = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    import time
    start_time = time.time()
    
    subdir = get_user_knowledge_dir(current_user.username) / kb_id
    file_path = subdir / doc_id
    file_on_disk = file_path.exists()
    
    logger.info(f"[性能追踪] 开始保存设置 - 文档: {doc_id}, 用户: {current_user.username}")

    try:
        chunk_strategy = body.get('chunk_strategy', 'standard')
        cs_dict = _normalize_chunk_strategy(chunk_strategy)

        if chunk_strategy == 'standard':
            if body.get('chunk_size') is not None: cs_dict['chunk_size'] = body['chunk_size']
            if body.get('chunk_overlap') is not None: cs_dict['chunk_overlap'] = body['chunk_overlap']
            if body.get('separators'): cs_dict['separators'] = _parse_separators(body['separators'])
        elif chunk_strategy == 'parent_child':
            if body.get('parent_chunk_size') is not None: cs_dict['parent_chunk_size'] = body['parent_chunk_size']
            if body.get('parent_chunk_overlap') is not None: cs_dict['parent_chunk_overlap'] = body['parent_chunk_overlap']
            if body.get('parent_separators'): cs_dict['parent_separators'] = _parse_separators(body['parent_separators'])
            if body.get('child_chunk_size') is not None: cs_dict['child_chunk_size'] = body['child_chunk_size']
            if body.get('child_chunk_overlap') is not None: cs_dict['child_chunk_overlap'] = body['child_chunk_overlap']
            if body.get('child_separators'): cs_dict['child_separators'] = _parse_separators(body['child_separators'])

        print(f"body: {body}")
        rag_config = body.get('rag_config') or {}
        embedding_model_name = rag_config.get('embedding_model_name', 'text-embedding-v4')
        rag_retrieve_method = rag_config.get('rag_retrieve_method', 'similarity')
        rerank_model_setting = rag_config.get('rerank_model_setting', {'rerank_enable': False})
        score_thresh_setting = rag_config.get('score_thresh_setting', {'enable': False})
        rrf_setting = rag_config.get('rrf_setting', {'vector_weight': 0.7, 'rrf_k': 60})

        db_chunk_strategy = _build_db_chunk_strategy(chunk_strategy, cs_dict)

        if chunk_strategy == 'table':
            metadata_tags_raw = body.get('metadata_tags', {}) or {}
            metadata_tags_list = []
            if isinstance(metadata_tags_raw, dict):
                if isinstance(metadata_tags_raw.get('_csv_default'), dict):
                    cols_dict = metadata_tags_raw['_csv_default']
                    checked_tags = [col for col, checked in cols_dict.items() if checked]
                    if checked_tags:
                        metadata_tags_list.append({
                            "source": doc_id,
                            "tags": checked_tags
                        })
                else:
                    for sheet_name, cols_dict in metadata_tags_raw.items():
                        if not isinstance(cols_dict, dict):
                            continue
                        checked_tags = [col for col, checked in cols_dict.items() if checked]
                        if checked_tags:
                            metadata_tags_list.append({
                                "source": doc_id,
                                "sheet": sheet_name,
                                "tags": checked_tags
                            })
            if metadata_tags_list:
                db_chunk_strategy["metadata_tags"] = metadata_tags_list

        logger.info(f"[性能追踪] 构建切片策略完成: {time.time() - start_time:.2f}s")

        kb_row = None
        all_kb_rows = db.query(Knowledge).filter(Knowledge.user_name == current_user.username).all()
        for row in all_kb_rows:
            if _extract_kb_id(row.knowledge_id) == kb_id:
                kb_row = row
                break

        if not kb_row:
            raise HTTPException(status_code=404, detail=f"知识库 {kb_id} 不存在")
        
        logger.info(f"[性能追踪] 查询知识库完成: {time.time() - start_time:.2f}s")

        # 深拷贝避免修改影响其他知识库记录
        import copy
        raw = copy.deepcopy(kb_row.chunk_retrieve_setting) if isinstance(kb_row.chunk_retrieve_setting, dict) else {}
        raw[doc_id] = {
            "chunk_strategy": db_chunk_strategy,
            "embedding_model_name": embedding_model_name,
            "rag_retrieve_method": rag_retrieve_method,
            "score_thresh_setting": score_thresh_setting,
            "rerank_model_setting": rerank_model_setting,
            "rrf_setting": rrf_setting,
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
            Knowledge.update_at: datetime.now(),
        }
        if name is not None:
            update_data[Knowledge.knowledge_name] = name
        if description is not None:
            update_data[Knowledge.knowledge_description] = description

        db.query(Knowledge).filter(Knowledge.knowledge_name == kb_row.knowledge_name).update(update_data)
        db.commit()
        logger.info(f"[性能追踪] 数据库更新完成: {time.time() - start_time:.2f}s")

        if file_on_disk:
            chunk_kwargs = _chunk_strategy_to_kwargs(cs_dict)
            
            # 提取元数据标签（表格切片模式），转换为 TableChunk 需要的 list 格式
            metadata_tags_raw = body.get('metadata_tags') or {}
            metaflag_for_chunk = None
            if chunk_strategy == 'table' and isinstance(metadata_tags_raw, dict):
                if isinstance(metadata_tags_raw.get('_csv_default'), dict):
                    cols_dict = metadata_tags_raw['_csv_default']
                    checked = [col for col, c in cols_dict.items() if c]
                    if checked:
                        metaflag_for_chunk = ['source'] + checked
                else:
                    for sheet_name, cols_dict in metadata_tags_raw.items():
                        if not isinstance(cols_dict, dict):
                            continue
                        checked = [col for col, c in cols_dict.items() if c]
                        if checked:
                            metaflag_for_chunk = ['source', 'sheet'] + checked
                            break
            
            # 将所有耗时操作（删除向量+向量化）都改为后台异步任务，不阻塞用户请求
            async def background_vectorize():
                try:
                    # 先删除旧向量（传递 model_name 和 retrieval_mode 确保使用正确的 VectorStore）
                    try:
                        delete_document_vectors(doc_id, current_user.username, kb_id, 
                                               model_name=embedding_model_name, 
                                               retrieval_mode=rag_retrieve_method)
                        logger.info(f"已删除文档 {doc_id} 的旧向量")
                    except Exception as e:
                        logger.warning(f"删除旧向量失败（继续执行）: {str(e)}")
                    
                    # 再执行向量化
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
                        metadata_tags=metaflag_for_chunk,
                    )
                    logger.info(f"后台向量化完成: 用户 {current_user.username} 知识库 {kb_id} 文档 {doc_id}")
                except Exception as e:
                    logger.warning(f"⚠️ 后台向量化失败（文档 {doc_id} 实际未生成向量，请排查嵌入模型API或向量库后重试）: {str(e)}")
            
            # 启动后台任务，不等待完成
            import asyncio
            asyncio.create_task(background_vectorize())
            
            logger.info(f"[性能追踪] 后台任务已启动: {time.time() - start_time:.2f}s")
            logger.info(f"用户 {current_user.username} 更新知识库 {kb_id} 文档 {doc_id} 的设置（向量化在后台执行）")
            return {"status": "success", "message": "文档设置保存成功，向量化正在后台执行", "filename": doc_id}
        else:
            logger.info(f"[性能追踪] 保存完成（无文件）: {time.time() - start_time:.2f}s")
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

# 召回测试接口
@retry_qdrant()
@router.get("/search")
async def search_knowledge(
    query: str,
    kb_id: str = Query(None, description="知识库ID，不传则搜索全部"),
    k: int = 10,
    retrieval_mode: str = Query("similarity", description="检索方式: similarity(向量检索)/hybrid(混合检索)"),
    rrf_setting_str: str = Query(None, description="加权RRF设置(JSON字符串)"),
    score_thresh_setting_str: str = Query(None, description="Score阈值设置(JSON字符串)"),
    bm25_thresh_setting_str: str = Query(None, description="BM25阈值设置(JSON字符串)"),
    detail_mode: bool = Query(False, description="是否返回详细分数信息"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        import time
        import json
        start_time = time.time()
        
        # 解析RRF设置
        rrf_setting = {}
        if rrf_setting_str:
            try:
                rrf_setting = json.loads(rrf_setting_str)
            except Exception as e:
                logger.warning(f"解析rrf_setting失败: {e}")
        
        # 解析Score阈值设置
        score_thresh_setting = {"enable": False, "score": Params.DEFAULT_SCORE_THRESHOLD}
        if score_thresh_setting_str:
            try:
                score_thresh_setting = json.loads(score_thresh_setting_str)
            except Exception as e:
                logger.warning(f"解析score_thresh_setting失败: {e}")
        
        # 解析BM25阈值设置
        bm25_thresh_setting = {"enable": False, "score": 0.5}
        if bm25_thresh_setting_str:
            try:
                bm25_thresh_setting = json.loads(bm25_thresh_setting_str)
            except Exception as e:
                logger.warning(f"解析bm25_thresh_setting失败: {e}")
        
        t0 = time.time()
        # 根据前端传入的检索方式选择对应的 vector_store
        vector_store = get_vector_store(current_user.username, retrieval_mode=retrieval_mode)
        logger.info(f"[耗时] 获取 vector_store: {(time.time()-t0)*1000:.0f}ms, 检索模式: {retrieval_mode} RRF设置: {rrf_setting}")

        logger.info(f"用户：{current_user.username} 召回测试提问: {query}  知识库ID：{kb_id}  检索方式: {retrieval_mode}")

        t1 = time.time()
        
        # 混合检索时使用加权RRF
        if retrieval_mode == "hybrid" and EMBEDDING_DATABASE == "qdrant":
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            qdrant_filter = Filter(must=[FieldCondition(key="metadata.kb_id", match=MatchValue(value=kb_id))]) if kb_id else None
            
            # 获取底层的Qdrant client
            client = vector_store._client
            collection_name = f"knowledge_{current_user.username}"
            
            # 1. 生成查询向量
            query_vector = vector_store._embeddings.embed_query(query)
            
            # 2. 生成sparse向量（BM25）
            sparse_vector = None
            if hasattr(vector_store, '_sparse_embeddings') and vector_store._sparse_embeddings:
                try:
                    sparse_vector = vector_store._sparse_embeddings.embed_query(query)
                except Exception as e:
                    logger.warning(f"生成sparse向量失败: {e}")
            
            vector_results = []
            bm25_results = []
            
            # 获取Score阈值配置（只作用于向量召回支路）
            score_threshold_enabled = score_thresh_setting.get('enable', False)
            score_threshold = score_thresh_setting.get('score', Params.DEFAULT_SCORE_THRESHOLD)
            
            # 获取BM25阈值配置（只作用于BM25召回支路）
            bm25_threshold_enabled = bm25_thresh_setting.get('enable', False)
            bm25_threshold = bm25_thresh_setting.get('score', 0.5)
            
            logger.info(f"[混合检索] BM25阈值配置: enable={bm25_threshold_enabled}, score={bm25_threshold}")
            
            # 3. 向量检索
            try:
                vector_search_result = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using='dense',
                    query_filter=qdrant_filter,
                    limit=k * 2,  # 获取更多结果用于融合
                    with_payload=True,
                    with_vectors=False
                )
                
                for point in vector_search_result.points:
                    # Score阈值只作用于向量召回支路
                    if score_threshold_enabled and point.score < score_threshold:
                        continue  # 向量相似度低于阈值，直接过滤，不参与RRF融合
                    
                    doc_content = point.payload.get('page_content', '')
                    metadata = point.payload.get('metadata', {})
                    from langchain_core.documents import Document
                    doc = Document(page_content=doc_content, metadata=metadata)
                    vector_results.append((doc, point.score))
            except Exception as e:
                logger.error(f"向量检索失败: {e}")
            
            # 4. BM25检索
            if sparse_vector:
                try:
                    # 将SparseVector转换为Qdrant需要的格式
                    from qdrant_client.http import models as qdrant_models
                    sparse_query = qdrant_models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values
                    )
                    
                    bm25_search_result = client.query_points(
                        collection_name=collection_name,
                        query=sparse_query,
                        using='sparse',
                        query_filter=qdrant_filter,
                        limit=k * 2,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    for point in bm25_search_result.points:
                        # BM25阈值只作用于BM25召回支路
                        if bm25_threshold_enabled and point.score < bm25_threshold:
                            continue  # BM25分数低于阈值，直接过滤，不参与RRF融合
                        
                        doc_content = point.payload.get('page_content', '')
                        metadata = point.payload.get('metadata', {})
                        from langchain_core.documents import Document
                        doc = Document(page_content=doc_content, metadata=metadata)
                        bm25_results.append((doc, point.score))
                except Exception as e:
                    logger.error(f"BM25检索失败: {e}")
            
            # 5. 加权RRF融合（使用默认参数）
            vector_weight = (rrf_setting.get('vector_weight') if rrf_setting and rrf_setting.get('vector_weight') else Params.DEFAULT_RRF_WEIGHT)
            rrf_k = (rrf_setting.get('rrf_k') if rrf_setting and rrf_setting.get('rrf_k') else Params.DEFAULT_RRF_K)
            bm25_weight = 1.0 - vector_weight
            
            doc_scores = {}
            
            # 处理向量检索结果
            for rank, (doc, raw_score) in enumerate(vector_results or [], 1):
                doc_id = doc.metadata.get('source', '') + '_' + str(doc.metadata.get('chunk_index', rank))
                vector_weight_score = vector_weight * (1.0 / (rrf_k + rank))
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'vector_rank': rank,
                        'bm25_rank': None,
                        'rrf_score': vector_weight_score,
                        'vector_score': raw_score,  # 保存向量原始分数
                        'bm25_score': None  # BM25原始分数
                    }
                else:
                    doc_scores[doc_id]['vector_rank'] = rank
                    doc_scores[doc_id]['rrf_score'] += vector_weight_score
                    doc_scores[doc_id]['vector_score'] = raw_score  # 更新向量原始分数
            
            # 处理BM25检索结果
            for rank, (doc, raw_score) in enumerate(bm25_results or [], 1):
                doc_id = doc.metadata.get('source', '') + '_' + str(doc.metadata.get('chunk_index', rank))
                bm25_weight_score = bm25_weight * (1.0 / (rrf_k + rank))
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'vector_rank': None,
                        'bm25_rank': rank,
                        'rrf_score': bm25_weight_score,
                        'vector_score': None,  # 向量原始分数
                        'bm25_score': raw_score  # 保存BM25原始分数
                    }
                else:
                    doc_scores[doc_id]['bm25_rank'] = rank
                    doc_scores[doc_id]['rrf_score'] += bm25_weight_score
                    doc_scores[doc_id]['bm25_score'] = raw_score  # 更新BM25原始分数
            
            # 按RRF分数排序
            sorted_docs = sorted(doc_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
            raw_results = [(item['doc'], item['rrf_score']) for item in sorted_docs[:k]]
            
            logger.info(f"[耗时] 混合检索RRF融合: {(time.time()-t1)*1000:.0f}ms, 返回 {len(raw_results)} 条结果")
        
        else:
            # 普通向量检索或Chroma
            if kb_id:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                qdrant_filter = Filter(must=[FieldCondition(key="metadata.kb_id", match=MatchValue(value=kb_id))])
                raw_results = vector_store.similarity_search_with_score(
                    query, k=k,
                    filter=qdrant_filter,
                )
            else:
                raw_results = vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(f"[耗时] 向量检索: {(time.time()-t1)*1000:.0f}ms, 返回 {len(raw_results)} 条原始结果")

        documents = []
        seen_parent_ids = set()

        # 获取Score阈值配置（只用于普通向量检索，不用于混合检索）
        score_threshold_enabled = score_thresh_setting.get('enable', False)
        score_threshold = score_thresh_setting.get('score', Params.DEFAULT_SCORE_THRESHOLD)

        for doc, raw in raw_results:
            # 混合检索时，raw已经是RRF分数，不需要再应用Score阈值过滤
            # 普通向量检索时，raw是向量相似度分数，需要应用Score阈值过滤
            if retrieval_mode != "hybrid":
                score = convert_vector_score(raw, vector_store, EMBEDDING_DATABASE)
                # 如果启用了Score阈值，使用配置的阈值；否则使用默认阈值
                final_threshold = score_threshold if score_threshold_enabled else Params.DEFAULT_SCORE_THRESHOLD
                if score < final_threshold:
                    continue
            else:
                # 混合检索直接使用RRF分数
                score = raw
            chunk_strategy = doc.metadata.get('chunk_strategy', 'standard')
            page_content = doc.page_content  #doc.metadata.get('parent_content')
            parent_id = doc.metadata.get('parent_id')

            if chunk_strategy == 'parent_child' and parent_id:
                if parent_id in seen_parent_ids:
                    continue
                seen_parent_ids.add(parent_id)

            doc_data = {
                "content": page_content,
                "source": doc.metadata.get('source', ''),
                "kb_id": doc.metadata.get('kb_id', ''),
                "score": float(score),
                "chunk_strategy": chunk_strategy
            }
            if chunk_strategy == 'parent_child':
                doc_data['parent_id'] = parent_id
            if chunk_strategy == 'table':
                doc_data['metadata'] = {k: v for k, v in doc.metadata.items() if k not in ('source', 'kb_id', 'chunk_strategy')}
            
            # 详细模式：返回各支路分数
            if detail_mode and retrieval_mode == "hybrid":
                doc_id = doc.metadata.get('source', '') + '_' + str(doc.metadata.get('chunk_index', 0))
                if doc_id in doc_scores:
                    item = doc_scores[doc_id]
                    doc_data['vector_score'] = item.get('vector_score')  # 向量原始分数
                    doc_data['bm25_score'] = item.get('bm25_score')  # BM25原始分数
                    doc_data['rrf_score'] = item.get('rrf_score')  # RRF融合分数
                    doc_data['vector_rank'] = item.get('vector_rank')  # 向量排名
                    doc_data['bm25_rank'] = item.get('bm25_rank')  # BM25排名

            documents.append(doc_data)

        documents.sort(key=lambda x: x['score'], reverse=True)
        for i, doc in enumerate(documents):
            doc['rank'] = i + 1

        elapsed = round((time.time() - start_time) * 1000, 2)
        logger.debug(f"知识库搜索耗时: {elapsed}ms")
        return {"results": documents, "elapsed_ms": elapsed}

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
    metadata_tags: str = Query(None),
    current_user: User = Depends(get_current_user),
):
    import time
    start_time = time.time()
    try:
        logger.info(f"开始切片预览: {file.filename}, 策略: {chunk_strategy}")
        import tempfile
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content_bytes = await file.read()
            tmp.write(content_bytes)
            tmp_path = tmp.name
        logger.info(f"文件读取完成: {file.filename}, 大小: {len(content_bytes)} bytes")
        try:
            # 表格切片模式：直接使用文件路径，不需要解析为文本内容
            if chunk_strategy == "table":
                content = ""  # 表格切片不需要content参数
            else:
                content = ""
                async for chunk in get_doc_processor().read_document(tmp_path):
                    content += chunk
                logger.info(f"文档解析完成: {file.filename}, 内容长度: {len(content)}")
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
            elif chunk_strategy == "table":
                # 表格切片：解析元数据标签
                meta_tags_list = None
                if metadata_tags:
                    try:
                        meta_tags_list = json.loads(metadata_tags)
                    except json.JSONDecodeError:
                        logger.warning(f"metadata_tags JSON解析失败: {metadata_tags}")
                chunk_kwargs = {
                    "metadata_tags": meta_tags_list,
                }

            texts, metadatas = _chunk_with_strategy(content, chunk_strategy, file.filename, "preview", file_path=tmp_path, **chunk_kwargs)
            logger.info(f"文档切块完成: {len(texts)} 个块")
            logger.info(f"切块结果: texts={len(texts)}, metadatas={len(metadatas)}")
            if texts:
                logger.info(f"第一个块内容: {texts[0][:200]}")
                logger.info(f"第一个块元数据: {metadatas[0]}")

            chunks = []
            for i, (text, meta) in enumerate(zip(texts, metadatas)):
                chunks.append({"index": i, "content": text, "length": len(text), "chunk_strategy": meta.get("chunk_strategy", chunk_strategy), "parent_id": meta.get("parent_id")})
            
            elapsed = time.time() - start_time
            logger.info(f"切片预览完成: {file.filename}, 耗时: {elapsed:.2f}s, 块数: {len(chunks)}")
            return {"chunks": chunks, "total": len(chunks)}
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"切片预览失败: {str(e)}, 耗时: {elapsed:.2f}s")
        raise HTTPException(status_code=500, detail=f"切片预览失败: {str(e)}")


@router.post("/chunk_preview_from_disk")
async def chunk_preview_from_disk(
    kb_id: str = Query(..., description="知识库ID"),
    filename: str = Query(..., description="文件名"),
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
    metadata_tags: str = Query(None),
    current_user: User = Depends(get_current_user),
):
    """直接从磁盘读取文件进行切片预览，避免下载-上传的网络开销"""
    import time
    start_time = time.time()
    try:
        logger.info(f"开始磁盘切片预览: {filename}, 策略: {chunk_strategy}")
        
        subdir = get_user_knowledge_dir(current_user.username) / kb_id
        file_path = subdir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        logger.info(f"文件路径: {file_path}")
        content = ""
        async for chunk in get_doc_processor().read_document(str(file_path)):
            content += chunk
        logger.info(f"文档解析完成: {filename}, 内容长度: {len(content)}")
        
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
        elif chunk_strategy == "table":
            # 表格切片：解析元数据标签
            meta_tags_list = None
            if metadata_tags:
                try:
                    meta_tags_list = json.loads(metadata_tags)
                except json.JSONDecodeError:
                    logger.warning(f"metadata_tags JSON解析失败: {metadata_tags}")
            chunk_kwargs = {
                "metadata_tags": meta_tags_list,
            }

        texts, metadatas = _chunk_with_strategy(content, chunk_strategy, filename, "preview", file_path=str(file_path), **chunk_kwargs)
        logger.info(f"文档切块完成: {len(texts)} 个块")

        chunks = []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            chunks.append({"index": i, "content": text, "length": len(text), "chunk_strategy": meta.get("chunk_strategy", chunk_strategy), "parent_id": meta.get("parent_id")})
        
        elapsed = time.time() - start_time
        logger.info(f"磁盘切片预览完成: {filename}, 耗时: {elapsed:.2f}s, 块数: {len(chunks)}")
        return {"chunks": chunks, "total": len(chunks)}
    except HTTPException:
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"磁盘切片预览失败: {str(e)}, 耗时: {elapsed:.2f}s")
        raise HTTPException(status_code=500, detail=f"切片预览失败: {str(e)}")

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


def preload_knowledge_components():
    """预加载知识库核心组件，避免首次请求时的长延迟"""
    try:
        start = time.time()
        logger.info("🔄 开始预加载知识库组件...")

        # 1. 预加载 Qdrant 客户端
        if EMBEDDING_DATABASE == "qdrant":
            _create_qdrant_client()
            logger.info("  ✅ Qdrant 客户端已预加载")

        # 2. 预加载 Dense Embeddings（为所有已存在用户预加载默认模型）
        users = get_all_users()
        if users:
            for user in users:
                _get_dense_embeddings(user, Params.DEFAULT_EMBEDDING_MODEL)
                logger.info(f"✅ Dense Embeddings 已预加载 ({user}, 模型: {Params.DEFAULT_EMBEDDING_MODEL})")
                # 3. 预加载 Sparse Embeddings (BM25)
                _get_sparse_embeddings(user)
                logger.info(f"✅ Sparse Embeddings (BM25) 已预加载 ({user})")

        else:
            # 无用户时预加载一个通用实例
            _get_dense_embeddings("default", Params.DEFAULT_EMBEDDING_MODEL)
            logger.info(f"✅ Dense Embeddings 已预加载 (通用实例, 模型: {Params.DEFAULT_EMBEDDING_MODEL})")
            _get_sparse_embeddings("default")
            logger.info("✅ Sparse Embeddings (BM25) 已预加载")

        elapsed = time.time() - start
        logger.info(f"✅ 知识库组件预加载完成 (耗时: {elapsed:.2f}s)")
    except Exception as e:
        logger.warning(f"知识库组件预加载失败（将延迟到首次请求时加载）: {e}")