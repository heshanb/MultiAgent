"""
RAG 三层缓存：结果缓存 + 语义缓存 + Embedding缓存
依赖 Redis Stack（需要 redis 包 + RediSearch 模块）
"""
import hashlib
import json
import time
import re
import numpy as np
from typing import Optional
import redis

from settings.Define import Params
from settings.logger_manager import get_logger

logger = get_logger(__name__)


class RAGCache:
    RESULT_TTL = 3600
    EMBEDDING_TTL = 86400
    SEMANTIC_THRESHOLD = 0.90

    def __init__(self):
        self._redis = None
        self._index_created = False

    # ─────────── Redis 连接 ───────────
    @property
    def client(self):
        if self._redis is None:
            try:
                self._redis = redis.Redis(
                    host=Params.REDIS_HOST,
                    port=Params.REDIS_PORT,
                    db=Params.REDIS_DB,
                    password=Params.REDIS_PASSWORD or None,
                    decode_responses=False,
                    socket_connect_timeout=3,
                    socket_timeout=3,
                )
                self._redis.ping()
                logger.info("Redis 连接成功")
            except Exception as e:
                logger.warning(f"Redis 不可用，缓存关闭: {e}")
                self._redis = None
        return self._redis

    def _available(self) -> bool:
        return self.client is not None

    # ─────────── 查询归一化 ───────────
    @staticmethod
    def _normalize(query: str) -> str:
        q = query.strip().lower()
        q = re.sub(r'\s+', '', q)
        q = re.sub(r'[，。！？、；：""''（）【】《》]', '', q)
        return q

    # ==================== 1. 结果缓存 ====================
    def _result_key(self, query: str) -> str:
        return f"rag:result:{hashlib.md5(self._normalize(query).encode()).hexdigest()}"

    def get_result(self, query: str) -> Optional[str]:
        if not self._available():
            return None
        try:
            data = self.client.get(self._result_key(query))
            if data:
                logger.info(f"结果缓存命中: {query[:50]}")
                return data.decode('utf-8')
        except Exception as e:
            logger.warning(f"结果缓存读取失败: {e}")
        return None

    def set_result(self, query: str, result: str):
        if not self._available():
            return
        try:
            self.client.setex(self._result_key(query), self.RESULT_TTL, result)
            logger.info(f"结果缓存已保存: {query[:50]}")
        except Exception as e:
            logger.warning(f"结果缓存保存失败: {e}")

    # ==================== 2. 语义缓存（Redis Stack 向量相似度） ====================
    SEMANTIC_IDX = "rag_semantic_idx"
    SEMANTIC_PFX = "rag:semantic:"

    def _ensure_index(self, dim: int):
        if self._index_created or not self._available():
            return
        try:
            self.client.execute_command(f"FT.INFO {self.SEMANTIC_IDX}")
            self._index_created = True
        except Exception:
            try:
                self.client.execute_command(
                    "FT.CREATE", self.SEMANTIC_IDX,
                    "ON", "HASH", "PREFIX", "1", self.SEMANTIC_PFX,
                    "SCHEMA",
                    "embedding", "VECTOR", "FLAT", "6",
                    "TYPE", "FLOAT32",
                    "DIM", str(dim),
                    "DISTANCE_METRIC", "COSINE",
                    "result", "TEXT",
                    "query_text", "TEXT",
                    "timestamp", "NUMERIC"
                )
                self._index_created = True
                logger.info(f"语义缓存向量索引创建成功，维度={dim}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    self._index_created = True
                else:
                    logger.warning(f"语义缓存索引创建失败: {e}")

    def get_semantic(self, query_embedding: list) -> Optional[str]:
        if not self._available():
            return None
        try:
            self._ensure_index(len(query_embedding))
            emb_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            result = self.client.execute_command(
                "FT.SEARCH", self.SEMANTIC_IDX,
                f"@embedding:[VECTOR_RANGE {self.SEMANTIC_THRESHOLD} $vec]=>{{$yield_distance_as: dist}}",
                "PARAMS", "2", "vec", emb_bytes,
                "SORTBY", "dist",
                "LIMIT", "0", "1",
                "RETURN", "2", "result", "query_text",
                "DIALECT", "2"
            )
            if result and len(result) > 1:
                doc = result[1]
                cached_result = doc[3] if len(doc) > 3 else None
                cached_query = doc[5] if len(doc) > 5 else b""
                if cached_result:
                    q = cached_query.decode() if isinstance(cached_query, bytes) else cached_query
                    logger.info(f"语义缓存命中，相似查询: {q}")
                    return cached_result.decode() if isinstance(cached_result, bytes) else cached_result
        except Exception as e:
            logger.warning(f"语义缓存搜索失败: {e}")
        return None

    def set_semantic(self, query: str, query_embedding: list, result: str):
        if not self._available():
            return
        try:
            self._ensure_index(len(query_embedding))
            key = f"{self.SEMANTIC_PFX}{hashlib.md5(query.encode()).hexdigest()}"
            emb_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            self.client.hset(key, mapping={
                "embedding": emb_bytes,
                "result": result,
                "query_text": query,
                "timestamp": int(time.time())
            })
            self.client.expire(key, self.RESULT_TTL)
            logger.info(f"语义缓存已保存: {query[:50]}")
        except Exception as e:
            logger.warning(f"语义缓存保存失败: {e}")

    # ==================== 3. Embedding 缓存 ====================
    def _emb_key(self, text: str) -> str:
        return f"rag:emb:{hashlib.md5(text.encode()).hexdigest()}"

    def get_embedding(self, text: str) -> Optional[list]:
        if not self._available():
            return None
        try:
            data = self.client.get(self._emb_key(text))
            if data:
                logger.info(f"Embedding缓存命中: {text[:50]}")
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Embedding缓存读取失败: {e}")
        return None

    def set_embedding(self, text: str, embedding: list):
        if not self._available():
            return
        try:
            self.client.setex(self._emb_key(text), self.EMBEDDING_TTL, json.dumps(embedding))
            logger.info(f"Embedding缓存已保存: {text[:50]}")
        except Exception as e:
            logger.warning(f"Embedding缓存保存失败: {e}")

    # ==================== 统一入口 ====================
    def search(self, query: str, embedding_func) -> Optional[str]:
        """三层缓存串联：结果 → 语义 → 全部miss返回None"""
        # 第1层：结果缓存
        result = self.get_result(query)
        if result:
            return result

        # 第2层：语义缓存
        if embedding_func:
            emb = self.get_embedding(query)
            if emb is None:
                try:
                    emb = embedding_func(query)
                    self.set_embedding(query, emb)
                except Exception as e:
                    logger.warning(f"Embedding计算失败: {e}")
                    return None

            result = self.get_semantic(emb)
            if result:
                self.set_result(query, result)
                return result

        return None

    def save(self, query: str, result: str, embedding_func=None):
        """保存所有缓存"""
        self.set_result(query, result)
        if embedding_func:
            try:
                emb = self.get_embedding(query)
                if emb is None:
                    emb = embedding_func(query)
                    self.set_embedding(query, emb)
                self.set_semantic(query, emb, result)
            except Exception as e:
                logger.warning(f"保存语义缓存失败: {e}")


_rag_cache = None


def get_rag_cache() -> RAGCache:
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = RAGCache()
    return _rag_cache