import json
import time
import threading
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, field_validator, model_validator

from core.auth.auth import get_current_user
from core.auth.database import get_db, User, Agent, PublishHistory, CustomTool
from core.knowledge.knowledge_manager import get_user_knowledge_dir, get_vector_store, EMBEDDING_DATABASE, \
    convert_vector_score
from settings.Define import Params, PathConfig
from settings.logger_manager import get_logger
from slowapi import Limiter
from slowapi.util import get_remote_address
import pybreaker

logger = get_logger(__name__)

# 官方渠道名称（商店展示与收费标记依赖该常量）
OFFICIAL_CHANNEL_NAME = "小智官方"

# 渠道名称映射：前端渠道ID -> 平台名称
CHANNEL_NAME_MAP = {
    "store": OFFICIAL_CHANNEL_NAME,
    "wechat": "微信",
    "wechat_miniapp": "微信小程序",
    "wechat_official": "微信公众号",
    "dingtalk": "钉钉",
    "feishu": "飞书",
    "web": "Web",
    "api": "API",
}

# 懒加载 FlagEmbedding（避免启动时导入卡住）
_FlagReranker = None

MEMORY_MAX_TURNS = 10

# 懒加载 LangChain 库
_InMemoryChatMessageHistory = None
_HumanMessage = None
_AIMessage = None
_SystemMessage = None

# 会话记忆存储
_chat_memories = {}
_memories_lock = threading.Lock()

_reranker_cache = {}
_reranker_cache_lock = threading.Lock()

RERANKER_BASE = None
RERANKER_LARGE = None

class _LightReranker:
    """用 transformers 直连加载 rerank 模型，接口兼容 CrossEncoder，
    避免 sentence_transformers 整包导入导致的 ~2 分钟启动开销。"""

    def __init__(self, model_name_or_path, max_length=512, device='cpu',
                 trust_remote_code=True):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)
        self.model.eval().to(device)
        self.max_length = max_length
        self.device = device

    def rank(self, query, passages, top_k=10):
        """与 CrossEncoder.rank 同签名，返回 [(passage, score)]，按分数降序。"""
        import torch
        inputs = self.tokenizer(
            [[query, p] for p in passages],
            padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt", return_token_type_ids=True,
        ).to(self.device)
        with torch.no_grad():
            scores = self.model(**inputs, return_dict=True).logits.view(-1).tolist()
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def predict(self, pairs):
        """兼容 CrossEncoder.predict，返回每条 pair 的分数列表。"""
        import torch
        inputs = self.tokenizer(
            [list(p) for p in pairs],
            padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt", return_token_type_ids=True,
        ).to(self.device)
        with torch.no_grad():
            return self.model(**inputs, return_dict=True).logits.view(-1).tolist()

def _get_langchain_classes():
    global _InMemoryChatMessageHistory, _HumanMessage, _AIMessage, _SystemMessage
    if _InMemoryChatMessageHistory is None:
        from langchain_core.chat_history import InMemoryChatMessageHistory
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        _InMemoryChatMessageHistory = InMemoryChatMessageHistory
        _HumanMessage = HumanMessage
        _AIMessage = AIMessage
        _SystemMessage = SystemMessage
    return _InMemoryChatMessageHistory, _HumanMessage, _AIMessage, _SystemMessage


def _get_chat_history(session_key: str) -> "InMemoryChatMessageHistory":
    InMemoryChatMessageHistory, _, _, _ = _get_langchain_classes()
    with _memories_lock:
        if session_key not in _chat_memories:
            _chat_memories[session_key] = InMemoryChatMessageHistory()
        return _chat_memories[session_key]


def _clear_chat_history(session_key: str):
    with _memories_lock:
        _chat_memories.pop(session_key, None)


def load_reranker():
    """预加载 reranker 模型（仅当本地已有模型时加载）

    安装版不随包分发模型权重（体积数 GB），且目标机器可能无网络，
    因此本地无模型时直接跳过，改由实际检索时按需加载，避免阻塞启动。
    """
    global RERANKER_BASE, RERANKER_LARGE

    for model_name, current in (("bge-reranker-base", RERANKER_BASE), ("bge-reranker-large", RERANKER_LARGE)):
        if current is not None:
            continue
        if not (PathConfig.MODEL_DIR / model_name).exists():
            logger.info(f"本地无 Reranker 模型 {model_name}，跳过预加载（使用时按需加载）")
            continue
        try:
            _get_reranker(model_name)
        except Exception as e:
            logger.warning(f"Reranker 预加载失败 {model_name}: {e}")


# def _get_reranker(model_name: str):
#     """获取或创建 reranker 模型（带缓存机制）"""
#     cache_key = f"{model_name}_crossEncoder"
#
#     with _reranker_cache_lock:
#         if cache_key in _reranker_cache:
#             return _reranker_cache[cache_key]
#
#     # 尝试从本地加载
#     local_model_path = PathConfig.MODEL_DIR / model_name
#     if local_model_path.exists():
#         model_path = str(local_model_path.as_posix())
#         logger.info(f"从本地加载 Reranker 模型: {model_path}")
#     else:
#         model_path = model_name
#         logger.info(f"从 HuggingFace 加载 Reranker 模型: {model_path}")
#
#     logger.info(f" 开始加载Reranker模型: {model_name}")
#
#     import os
#     os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
#     # 显式清除 HF_HUB_OFFLINE，确保 fastembed 可以下载 BM25 模型
#     os.environ.pop("HF_HUB_OFFLINE", None)
#     # TRANSFORMERS_OFFLINE 仅影响 transformers 库，不影响 fastembed
#     os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
#
#     try:
#         from sentence_transformers import CrossEncoder
#         logger.info(f" 正在加载模型权重（CPU 模式约需 30-60 秒，请耐心等待...）")
#         start_time = time.time()
#         # 临时禁用 tqdm 进度条，避免输出到 stderr 导致日志混乱
#         import sys
#         from contextlib import redirect_stderr
#         import io
#         dummy_stderr = io.StringIO()
#         with redirect_stderr(dummy_stderr):
#             reranker = CrossEncoder(model_path,
#                                      max_length=512,
#                                      device='cpu',
#                                      trust_remote_code=True
#                                      )
#         load_time = time.time() - start_time
#         logger.info(f" Reranker模型加载完成: {model_name} (耗时: {load_time:.2f}s)")
#     except Exception as e:
#         logger.error(f" Reranker模型加载失败: {model_name}, 错误: {e}")
#         raise
#
#     with _reranker_cache_lock:
#         _reranker_cache[cache_key] = reranker
#
#     return reranker

def _get_reranker(model_name: str):
    """获取或创建 reranker 模型（带缓存机制）"""
    cache_key = f"{model_name}_crossEncoder"

    with _reranker_cache_lock:
        if cache_key in _reranker_cache:
            return _reranker_cache[cache_key]

    # 尝试从本地加载
    local_model_path = PathConfig.MODEL_DIR / model_name
    if local_model_path.exists():
        model_path = str(local_model_path.as_posix())
        logger.info(f"从本地加载 Reranker 模型: {model_path}")
        use_offline = True
    elif PathConfig.FROZEN:
        # 安装版未分发模型权重，禁止联网下载数 GB 权重阻塞请求，直接跳过 rerank
        raise FileNotFoundError(f"本地缺少 Reranker 模型: {local_model_path}")
    else:
        model_path = model_name
        logger.info(f"从 HuggingFace 加载 Reranker 模型: {model_path}")
        use_offline = False   # 要走下载，不能再设 OFFLINE

    logger.info(f" 开始加载Reranker模型: {model_name}")

    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # 显式清除 HF_HUB_OFFLINE，确保 fastembed 可以下载 BM25 模型
    os.environ.pop("HF_HUB_OFFLINE", None)
    if use_offline:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    else:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    try:
        logger.info(f" 正在加载模型权重（CPU 模式约需 10-30 秒，请耐心等待...）")
        start_time = time.time()
        import io
        from contextlib import redirect_stderr
        dummy_stderr = io.StringIO()
        with redirect_stderr(dummy_stderr):
            reranker = _LightReranker(model_path, max_length=512, device='cpu',
                                      trust_remote_code=True)
        load_time = time.time() - start_time
        logger.info(f" Reranker模型加载完成: {model_name} (耗时: {load_time:.2f}s)")
    except Exception as e:
        logger.error(f"Reranker模型加载失败: {model_name}, 错误: {e}")
        raise

    with _reranker_cache_lock:
        _reranker_cache[cache_key] = reranker

    return reranker


# ==================== 工具调用相关 ====================

def _tools_to_openai_format(tools: list) -> list:
    """将自定义工具转换为 OpenAI function calling 格式，并增强描述以帮助 LLM 正确判断调用时机"""
    # 工具调用条件规则：明确说明什么场景下应该/不应该调用该工具
    tool_call_rules = {
        "天气查询": "️ 调用条件：仅当用户明确要求查询天气时调用（如'北京天气怎么样'、'今天下雨吗'）。如果用户只是在对话中提到城市名（如回答'在哪个城市'的问题），不要调用此工具。",
    }
    
    # 认证相关参数名（后端自动处理，不需要 LLM 生成）
    auth_param_names = {"salt", "sign", "appid", "appId", "appKey", "app_key", "api_key", "apikey", "token", "access_token"}
    
    functions = []
    for tool in tools:
        properties = {}
        required = []
        auth_type = tool.get("auth_type", "none")
        
        for param in tool.get("parameters", []):
            param_name = param["name"]
            # 过滤掉认证相关参数，不让 LLM 看到
            if param_name.lower() in auth_param_names:
                continue
            properties[param_name] = {
                "type": param.get("type", "string"),
                "description": param.get("description", "")
            }
            if param.get("required"):
                required.append(param_name)
        
        # 增强工具描述：追加调用条件说明
        original_desc = tool.get("description", "")
        tool_name = tool["name"]
        if tool_name in tool_call_rules:
            enhanced_desc = f"{original_desc}\n\n{tool_call_rules[tool_name]}"
        else:
            enhanced_desc = original_desc
        
        functions.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": enhanced_desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    return functions


async def _execute_tool(tool_config: dict, arguments: dict) -> str:
    """执行自定义 API 工具调用，返回结果字符串"""
    import httpx
    import hashlib
    import random
    # 兼容前端字段名 url 和后端字段名 api_url
    url = tool_config.get("api_url") or tool_config.get("url", "")
    if not url:
        return json.dumps({"error": "工具配置缺少 API 地址"}, ensure_ascii=False)
    method = tool_config.get("method", "GET").upper()
    headers = dict(tool_config.get("headers", {}))
    timeout = tool_config.get("timeout", 10)

    # 替换 URL 中的路径参数 {param}
    path_params = set()
    for key, value in arguments.items():
        placeholder = "{" + key + "}"
        if placeholder in url:
            url = url.replace(placeholder, str(value))
            path_params.add(key)

    # 处理认证（兼容前端 auth_value 和后端 auth_config）
    auth_type = tool_config.get("auth_type", "none")
    auth_config = tool_config.get("auth_config") or {}
    # 如果 auth_config 为空但 auth_value 有值，尝试解析
    if not auth_config and tool_config.get("auth_value"):
        try:
            auth_config = json.loads(tool_config["auth_value"])
        except:
            auth_config = {"token": tool_config["auth_value"]}
    
    if auth_type == "bearer":
        token = auth_config.get("token", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    elif auth_type == "api_key":
        api_key = auth_config.get("key", "")
        if api_key:
            headers["X-API-Key"] = api_key

    # 特殊处理：需要 appId 和 appKey 的工具（如百度翻译）
    tool_name = tool_config.get("name", "")
    logger.info(f"[TOOL] {tool_name} 调用前，认证类型: {auth_type}")
    if auth_type == "api_id_key":
        # 从 auth_config 中获取统一的 appId 和 appKey
        app_id = auth_config.get("appId", "")
        app_key = auth_config.get("appKey", "")

        logger.info(f"[TOOL] {tool_name} 调用前，认证配置: appId={app_id}, appKey={app_key}")
        
        if app_id and app_key:
            # 百度翻译特殊处理：自动生成 salt 和 sign
            if tool_name == "百度翻译":
                salt = str(random.randint(32768, 65536))
                sign_str = f"{app_id}{arguments.get('q', '')}{salt}{app_key}"
                sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest()
                
                # 添加到请求参数中
                arguments["appid"] = app_id
                arguments["salt"] = salt
                arguments["sign"] = sign
                logger.info(f"[TOOL] 百度翻译自动填充认证参数: appid={app_id}, salt={salt}")
            else:
                # 其他工具：将 appId 和 appKey 添加到参数中
                arguments["appId"] = app_id
                arguments["appKey"] = app_key
                logger.info(f"[TOOL] {tool_name} 自动填充认证参数: appId={app_id}")
        else:
            logger.warning(f"[TOOL] {tool_name} 缺少认证配置，请在工具配置中设置 API ID 和 API Key")

    # 分离路径参数和请求体/查询参数
    body_params = {k: v for k, v in arguments.items() if k not in path_params}

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            if method == "GET":
                response = await client.get(url, headers=headers, params=body_params)
            elif method == "POST":
                # 百度翻译等 API 需要 form-urlencoded 格式
                content_type_header = headers.get("Content-Type", "")
                if "application/x-www-form-urlencoded" in content_type_header:
                    response = await client.post(url, headers=headers, data=body_params)
                else:
                    response = await client.post(url, headers=headers, json=body_params)
            elif method == "PUT":
                response = await client.put(url, headers=headers, json=body_params)
            elif method == "DELETE":
                response = await client.delete(url, headers=headers, params=body_params)
            else:
                return json.dumps({"error": f"不支持的请求方法: {method}"}, ensure_ascii=False)

            response.raise_for_status()
            # 尝试解析 JSON，如果失败则返回纯文本
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    result = response.json()
                    return json.dumps(result, ensure_ascii=False)
                except json.JSONDecodeError:
                    return json.dumps({"raw_text": response.text[:2000]}, ensure_ascii=False)
            else:
                # 非 JSON 响应，直接返回文本
                return json.dumps({"raw_text": response.text[:2000]}, ensure_ascii=False)
    except httpx.TimeoutException:
        return json.dumps({"error": "工具调用超时"}, ensure_ascii=False)
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def _trim_history(history, max_turns: int = MEMORY_MAX_TURNS):
    messages = history.messages
    if len(messages) > max_turns * 2:
        keep_from = len(messages) - max_turns * 2
        history.clear()
        for m in messages[keep_from:]:
            history.add_message(m)


router = APIRouter(prefix="/api/agents", tags=["agents"])

# 接口限流：令牌桶，按客户端IP限流
limiter = Limiter(key_func=get_remote_address)


# ---------------------- 2、熔断器：保护下游大模型服务 ----------------------
# 连续失败3次触发熔断，熔断等待30s进入半开状态
llm_breaker = pybreaker.CircuitBreaker(
    fail_max=3,
    reset_timeout=30
)


class LLMRemoteError(Exception):
    """大模型远程服务异常"""
    pass

class AgentCreate(BaseModel):
    """创建智能体的请求体。

    用户首次新建智能体时提交，只需提供基础信息（名称必填，其他可选）。
    其他配置项（system_prompt / model / knowledge_base 等）通过 AgentUpdate 补全。
    """
    name: str
    description: Optional[str] = None
    avatar: Optional[str] = ""


class AgentUpdate(BaseModel):
    """更新智能体的请求体。

    所有字段均可选，前端只传需要修改的字段，后端做部分更新。
    用于编辑器里修改 system_prompt、模型、知识库、变量、开场白、工具等全部配置。
    """
    name: Optional[str] = None
    description: Optional[str] = None
    avatar: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    variables: Optional[list] = None
    knowledge_base: Optional[list] = None
    tools: Optional[list] = None
    opening: Optional[str] = None
    presets: Optional[list] = None
    is_published: Optional[bool] = None
    deep_think: Optional[bool] = None
    struct_output: Optional[dict] = None
    temperature: Optional[float] = None


class AgentResponse(BaseModel):
    """智能体详情的响应体（从 ORM Model 序列化而来）。

    GET /api/agents、列表查询等接口的返回结构。
    Config.from_attributes = True 允许直接用 Agent ORM 对象构造。
    """
    id: int
    name: str
    description: Optional[str] = None
    avatar: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    variables: Optional[list] = None
    knowledge_base: Optional[list] = None
    tools: Optional[list] = None
    opening: Optional[str] = None
    presets: Optional[list] = None
    is_published: bool
    category: Optional[str] = None
    publish_version: Optional[str] = None
    publish_desc: Optional[str] = None
    publish_channels: Optional[str] = None
    deep_think: Optional[bool] = False
    struct_output: Optional[dict] = None
    temperature: Optional[float] = 0.7
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_official: bool = False
    is_free: bool = False
    source_channel: Optional[str] = None

    class Config:
        from_attributes = True

    @model_validator(mode="after")
    def derive_channel_flags(self):
        """根据 publish_channels 推导商店展示所需的平台/收费标记"""
        channel = (self.publish_channels or "").strip()
        if channel.startswith("["):
            try:
                parsed = json.loads(channel)
                if isinstance(parsed, list) and parsed:
                    channel = CHANNEL_NAME_MAP.get(parsed[0], parsed[0])
            except (json.JSONDecodeError, TypeError):
                pass
        if not channel:
            channel = OFFICIAL_CHANNEL_NAME
        self.is_official = channel == OFFICIAL_CHANNEL_NAME
        self.is_free = self.is_official
        self.source_channel = None if self.is_official else channel
        return self

    @field_validator('struct_output', mode='before')
    @classmethod
    def parse_struct_output(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return {"enable": False, "fields": []}
        return v

    @field_validator("variables", "knowledge_base", "presets", "tools", mode="before")
    @classmethod
    def parse_json_fields(cls, v):
        """将数据库中的 JSON 字符串解析为列表"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return []
        return v


class AgentChatRequest(BaseModel):
    """智能体对话的请求体（编辑器测试 + 商店公开使用共用）。

    核心字段：message（用户提问）。
    可选字段：
      - variables: 本轮传入的变量值（会替换 system_prompt 中的 {{key}}）
      - system_prompt: 编辑器测试时可临时覆盖 system_prompt
      - session_id: 会话标识，用于后端内存 history 隔离
      - deep_think: 是否开启深度思考模式
      - history: 前端维护的完整对话历史 [{role, content}, ...]。
                 传了则后端优先用它组装 messages，不再依赖内存 history；
                 没传则回退到后端 InMemoryChatMessageHistory。
      - model: 本轮使用的模型（编辑器测试时可临时覆盖 agent.model）
    """
    message: str
    variables: Optional[dict] = None
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None
    deep_think: Optional[bool] = False
    history: Optional[list] = None
    model: Optional[str] = None
    struct_output: Optional[bool] = False


class PromptOptimizeRequest(BaseModel):
    """Prompt 优化请求体。

    前端把当前 prompt 发给 AI 改写/润色/扩写，
    instruction 可选，让 AI 按特定方向优化（如"更简洁"、"加入更多细节"）。
    """
    prompt: str
    instruction: Optional[str] = None
    mode: Optional[str] = "auto"
    knowledge_bases: Optional[list] = None


def _deserialize_vars(raw: Optional[str]) -> list:
    """将数据库中存储的 JSON 字符串反序列化为 Python 列表。

    用于读取 Agent.variables / Agent.knowledge_base 等字段时，
    将 DB 里的 '[{"key":"hospital",...}]' 还原为 [{"key":"hospital",...}]。

    Args:
        raw: 数据库中取出的字符串值，可能为 None 或无效 JSON。

    Returns:
        解析成功返回 list；为 None 或解析失败时返回空列表 []。
    """
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


def _serialize_vars(data: Optional[list]) -> Optional[str]:
    """将 Python 列表序列化为 JSON 字符串，便于存入数据库。

    用于写入 Agent.variables / Agent.knowledge_base 等字段时，
    将 [{"key":"hospital",...}] 转为 '[{"key":"hospital",...}]'。

    Args:
        data: 待序列化的列表，可为 None。

    Returns:
        data 为 None 时返回 None；否则返回 ensure_ascii=False 的 JSON 字符串，
        中文字符保持原样（不转义为 \\uXXXX）。
    """
    if data is None:
        return None
    return json.dumps(data, ensure_ascii=False)


# RAG 判断缓存
_rag_search_cache = {}
_rag_search_cache_lock = threading.Lock()

# 缓存LLM实例，避免重复创建
_rag_judge_llm = None
_rag_judge_llm_lock = threading.Lock()

# 意图分类缓存
_intent_cache = {}
_intent_cache_lock = threading.Lock()

# 查询改写缓存
_query_rewrite_cache = {}
_query_rewrite_cache_lock = threading.Lock()


def need_rag_search(query: str) -> bool:
    """判断用户问题是否需要走RAG检索。

    优化：前置规则预检，闲聊/问候直接返回False，不调LLM，大幅降低延迟。

    Args:
        query: 用户问题。

    Returns:
        返回True：需要走RAG检索。返回False：不需要走RAG检索。
    """
    # 检查缓存
    cache_key = query.strip().lower()
    with _rag_search_cache_lock:
        if cache_key in _rag_search_cache:
            return _rag_search_cache[cache_key]

    query_stripped = query.strip()
    query_lower = query_stripped.lower()

    #  快速规则预检：闲聊/问候直接返回False，不调LLM
    chat_patterns = [
        # 问候
        "你好", "您好", "嗨", "嗨喽", "嗨咯", "hello", "hi", "hey", "哈喽",
        "早上好", "下午好", "晚上好", "早安", "晚安",
        # 告别
        "再见", "拜拜", "bye", "goodbye", "走了", "先走了",
        # 感谢
        "谢谢", "感谢", "thanks", "thank you", "多谢", "辛苦了",
        # 确认/回应
        "不客气", "没关系", "好的", "知道了", "明白", "收到",
        "嗯", "哦", "啊", "是的", "不是", "对", "不对",
        "在吗", "在不在", "有人吗", "在嘛",
        "嗯呢", "哦哦", "哈哈", "嘿嘿", "呵呵",
        # 身份询问
        "你是谁", "你叫什么", "你能做什么", "介绍一下自己", "你是什么",
        # 时间/天气
        "今天天气", "几点了", "现在时间", "今天星期",
        # 纯语气词（短查询）
        "嗯嗯", "啊啊", "哦哦", "好好",
    ]
    # 规则1：完全匹配或包含闲聊关键词
    for pattern in chat_patterns:
        if query_lower == pattern or query_lower.startswith(pattern) or query_lower.endswith(pattern):
            with _rag_search_cache_lock:
                _rag_search_cache[cache_key] = False
            logger.info(f"RAG检索判断(规则预检): '{query_stripped}' -> 不需要")
            return False

    # 规则2：超短查询（<=3字）且不含疑问词，大概率是闲聊
    if len(query_stripped) <= 3:
        question_words = ["怎么", "如何", "为什么", "什么", "哪里", "谁", "多少", "吗", "呢", "哪"]
        if not any(w in query_stripped for w in question_words):
            with _rag_search_cache_lock:
                _rag_search_cache[cache_key] = False
            logger.info(f"RAG检索判断(规则预检): '{query_stripped}' -> 不需要")
            return False

    # 规则3：纯英文问候/告别
    english_chat = ["hi", "hello", "hey", "bye", "goodbye", "thanks", "thank", "ok", "yes", "no"]
    if query_lower in english_chat:
        with _rag_search_cache_lock:
            _rag_search_cache[cache_key] = False
        logger.info(f"RAG检索判断(规则预检): '{query_stripped}' -> 不需要")
        return False

    # 规则预检未命中，使用 LLM 进行语义判断
    try:
        from core.agent_langgraph import get_llm_by_model

        # 使用缓存的LLM实例，避免重复创建
        global _rag_judge_llm
        with _rag_judge_llm_lock:
            if _rag_judge_llm is None:
                _rag_judge_llm = get_llm_by_model(Params.DEFAULT_TEXT_TOOL_MODEL, temperature=0.1, deep_think=False)

        llm = _rag_judge_llm

        judge_prompt = f"""你是一个意图分类器，请判断以下用户问题是否需要查询知识库来回答。

用户问题：{query_stripped}

# 判断标准：
【需要知识库】：问题涉及具体业务咨询、专业知识、事实查询、操作指南、政策规定、产品说明、技术问题、症状描述、健康建议等需要查阅资料才能准确回答的内容
【不需要知识库】：寒暄问候、简单确认、情感表达、闲聊、纯语气词、打招呼、感谢等日常交流

# 输出格式：
请只回答"是"或"否"，不要添加任何其他内容。"""

        logger.info("开始判断RAG检索...")
        response = llm.invoke([{"role": "user", "content": judge_prompt}])

        # 提取响应内容
        if hasattr(response, 'content'):
            answer = response.content.strip()
        elif isinstance(response, str):
            answer = response.strip()
        else:
            answer = str(response).strip()

        # 判断结果
        result = True if answer == "是" else False

        # 缓存结果
        with _rag_search_cache_lock:
            _rag_search_cache[cache_key] = result

        logger.info(f"RAG检索判断(LLM): '{query_stripped}' -> {'需要' if result else '不需要'}")
        return result

    except Exception as e:
        logger.warning(f"RAG检索判断失败，默认需要检索: {e}")
        # 失败时默认需要检索，保证功能正常
        return True

def is_query_relevant_to_agent(query: str, agent_name: str, system_prompt: str) -> bool:
    """判断用户问题是否与智能体角色/名称相关。
    
    通过提取智能体名称和系统提示词中的核心关键词，
    判断用户问题是否包含这些关键词或相关概念。
    
    Args:
        query: 用户问题
        agent_name: 智能体名称
        system_prompt: 系统提示词
        
    Returns:
        bool: True 表示相关，False 表示不相关
    """
    import re
    
    if not query or not query.strip():
        return False
    
    query_lower = query.lower().strip()
    logger.info(f"[相关性判断] query='{query}', agent_name='{agent_name}'")
    
    # 1. 检查是否包含智能体名称（或其核心部分）
    if agent_name:
        agent_name_lower = agent_name.lower().strip()
        # 提取智能体名称中的关键词（去除"助手"、"智能体"等通用词）
        generic_words = ["助手", "智能体", "机器人", "bot", "assistant", "agent"]
        agent_keywords = agent_name_lower
        for word in generic_words:
            agent_keywords = agent_keywords.replace(word, "")
        agent_keywords = agent_keywords.strip()
        
        logger.info(f"[相关性判断] agent_keywords='{agent_keywords}'")
        # 如果用户问题包含智能体名称或核心关键词，认为相关
        if agent_name_lower in query_lower or (agent_keywords and agent_keywords in query_lower):
            logger.info(f"[相关性判断] -> 相关（名称匹配）")
            return True
    
    # 2. 从系统提示词中提取核心角色关键词
    if system_prompt:
        role_keywords = []
        
        # 匹配 "你是..." 或 "你是一位..." 模式
        role_patterns = [
            r'你是[一位个]?(.+?)[,，。]',
            r'你作为(.+?)[,，。]',
            r'角色[：:](.+?)[\n。]',
        ]
        
        for pattern in role_patterns:
            matches = re.findall(pattern, system_prompt)
            for match in matches:
                # 提取关键名词短语
                keywords = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]{2,}', match)
                role_keywords.extend(keywords)
        
        logger.info(f"[相关性判断] role_keywords={role_keywords}")
        
        # 如果没有匹配到角色定义，尝试从系统提示词中提取高频关键词
        if not role_keywords:
            # 提取所有中文词汇（2个字符以上）
            all_keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', system_prompt[:500])
            # 过滤掉常见停用词
            stop_words = ["如果", "那么", "但是", "因为", "所以", "可以", "需要", "应该", "用户", "问题", "回答", "提供"]
            role_keywords = [kw for kw in all_keywords if kw not in stop_words][:10]
            logger.info(f"[相关性判断] fallback role_keywords={role_keywords}")
        
        # 检查用户问题是否包含这些角色关键词（精确匹配）
        if role_keywords:
            for keyword in role_keywords:
                if keyword.lower() in query_lower:
                    logger.info(f"[相关性判断] -> 相关（关键词'{keyword}'匹配）")
                    return True
        
        # 3. 宽松匹配：将长词拆分为 2-4 字的滑动窗口，检查重叠
        if role_keywords and len(query_lower) >= 2:
            # 从用户问题中提取所有 2-4 字的子串
            query_ngrams = set()
            clean_query = re.sub(r'[\u4e00-\u9fa5a-zA-Z0-9]', '', query_lower)
            for i in range(len(clean_query)):
                for size in range(2, 5):
                    if i + size <= len(clean_query):
                        query_ngrams.add(clean_query[i:i+size])
            
            # 从角色关键词中提取所有 2-4 字的子串
            keyword_ngrams = set()
            for kw in role_keywords:
                clean_kw = re.sub(r'[\u4e00-\u9fa5a-zA-Z0-9]', '', kw.lower())
                for i in range(len(clean_kw)):
                    for size in range(2, 5):
                        if i + size <= len(clean_kw):
                            keyword_ngrams.add(clean_kw[i:i+size])
            
            overlap = query_ngrams & keyword_ngrams
            logger.info(f"[相关性判断] query_ngrams_sample={list(query_ngrams)[:10]}, keyword_ngrams_sample={list(keyword_ngrams)[:10]}, overlap={overlap}")
            
            # 如果有 meaningful 的重叠（至少一个 3 字以上的词，或 2 个 2 字词），认为相关
            long_overlap = [w for w in overlap if len(w) >= 3]
            short_overlap = [w for w in overlap if len(w) == 2]
            if long_overlap or len(short_overlap) >= 2:
                logger.info(f"[相关性判断] -> 相关（长词重叠{len(long_overlap)}个，短词重叠{len(short_overlap)}个）")
                return True
    
    # 4. 如果以上都没匹配到，认为不相关
    logger.info(f"[相关性判断] -> 不相关")
    return False


def classify_intent(query: str) -> str:
    """意图路由分类：判断查询是否需要改写。

    返回:
        "rewrite" - 需要改写（复杂/模糊/专业查询）
        "direct" - 直接检索（清晰明确的查询）
        "skip" - 跳过检索（闲聊/问候）
    """
    cache_key = query.strip().lower()
    with _intent_cache_lock:
        if cache_key in _intent_cache:
            return _intent_cache[cache_key]

    # 规则1：闲聊/问候/否定回复类直接跳过
    chat_patterns = [
        "你好", "您好", "嗨", "hello", "hi", "在吗", "在不在",
        "谢谢", "感谢", "再见", "拜拜", "bye",
        "你是谁", "你叫什么", "你能做什么", "介绍一下",
        "哈哈", "呵呵", "嗯嗯", "好的", "知道了",
        "今天天气", "几点了", "现在时间",
        # 否定性回复（对话式回答，不是搜索意图）
        "没有", "不用", "不需要", "不清楚", "不知道",
        "没兴趣", "不想", "不要", "算了", "随便", "都行"
    ]
    
    # 先检查是否包含区域名关键词（如城市名、区名等），如果有则不跳过
    location_keywords = ["北京", "上海", "广州", "深圳", "海淀", "朝阳", "通州", "顺义", "昌平", "丰台", "东城", "西城", "区", "市", "县"]
    has_location = any(kw in query for kw in location_keywords)
    
    if not has_location and any(p in query.lower() for p in chat_patterns):
        result = "skip"
        with _intent_cache_lock:
            _intent_cache[cache_key] = result
        return result

    # 规则2：短查询且包含明确关键词，直接检索
    if len(query.strip()) <= 10:
        has_keyword = any(c.isalnum() for c in query)
        if has_keyword and not any(p in query for p in ["怎么", "如何", "为什么", "什么是", "请问"]):
            result = "direct"
            with _intent_cache_lock:
                _intent_cache[cache_key] = result
            return result

    # 规则3：长查询或复杂查询需要改写
    result = "rewrite"
    with _intent_cache_lock:
        _intent_cache[cache_key] = result
    return result

def rewrite_query(query: str, max_retries: int = 2) -> tuple:
    """使用 SLM 小模型优化用户查询，提高向量检索效果。

    优化1（Latency）：使用轻量级SLM模型，减少TTFT
    优化3（Cost）：前置意图路由，闲聊跳过改写

    Args:
        query: 用户原始查询
        max_retries: 最大重试次数

    Returns:
        (rewritten_query, intent_type): 优化后的查询字符串和意图类型
    """
    # 检查缓存
    cache_key = query.strip().lower()
    with _query_rewrite_cache_lock:
        if cache_key in _query_rewrite_cache:
            cached = _query_rewrite_cache[cache_key]
            return cached.get('rewritten', query), cached.get('intent', 'direct')

    # 优化3：意图路由分类
    intent = classify_intent(query)
    logger.info(f"[意图路由] query='{query[:50]}...', intent={intent}")

    # 闲聊类：跳过检索
    if intent == "skip":
        result = {'rewritten': query, 'intent': 'skip'}
        with _query_rewrite_cache_lock:
            _query_rewrite_cache[cache_key] = result
        return query, 'skip'

    # 清晰查询：直接检索，不改写
    if intent == "direct":
        result = {'rewritten': query, 'intent': 'direct'}
        with _query_rewrite_cache_lock:
            _query_rewrite_cache[cache_key] = result
        return query, 'direct'

    # 过短的查询不需要改写
    if len(query.strip()) < 5:
        result = {'rewritten': query, 'intent': 'direct'}
        with _query_rewrite_cache_lock:
            _query_rewrite_cache[cache_key] = result
        return query, 'direct'

    # 优化1：使用SLM小模型进行改写（降低延迟）
    try:
        from core.agent_langgraph import get_llm_by_model

        llm = get_llm_by_model(Params.DEFAULT_TEXT_TOOL_MODEL, temperature=0.1, deep_think=False)

        rewrite_prompt = f"""你是一个查询优化专家。将用户查询转换为更适合向量检索的格式。

原始查询：{query}

规则：
1. 纠正错别字，提取关键信息
2. 去除礼貌用语和填充词
3. 保持核心意图不变
4. 输出控制在30字以内

只输出优化后的查询，不要解释。"""

        response = llm.invoke([{"role": "user", "content": rewrite_prompt}])

        if hasattr(response, 'content'):
            rewritten = response.content.strip()
        elif isinstance(response, str):
            rewritten = response.strip()
        else:
            rewritten = str(response).strip()

        for prefix in ["优化后的查询：", "优化查询：", "改写后：", "查询："]:
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        if not rewritten or len(rewritten) < 3:
            logger.warning(f"查询改写结果为空，使用原始查询")
            rewritten = query

        result = {'rewritten': rewritten, 'intent': 'rewrite'}
        with _query_rewrite_cache_lock:
            _query_rewrite_cache[cache_key] = result

        logger.info(f"[查询改写] '{query[:30]}...' -> '{rewritten[:30]}...'")
        return rewritten, 'rewrite'

    except Exception as e:
        logger.warning(f"查询改写失败，使用原始查询: {e}")
        result = {'rewritten': query, 'intent': 'direct'}
        with _query_rewrite_cache_lock:
            _query_rewrite_cache[cache_key] = result
        return query, 'direct'

def _merge_results(results_a, results_b, is_chroma=True):
    """合并两次检索结果，去重并保留最高分（优化2：语义漂移防护）"""
    merged = {}
    for doc, score in (results_a or []):
        key = doc.page_content[:100]
        if key not in merged or score > merged[key][1]:
            merged[key] = (doc, score)
    for doc, score in (results_b or []):
        key = doc.page_content[:100]
        if key not in merged or score > merged[key][1]:
            merged[key] = (doc, score)
    sorted_results = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return sorted_results[:10]

def _build_rag_context(agent: Agent, user_username: str, query: str, log_tag: str = "") -> tuple:
    """构建 RAG 上下文，返回 (kb_content_text, citation_data)"""
    knowledge_base = _deserialize_vars(agent.knowledge_base)
    if not knowledge_base:
        return "", []

    logger.info(f"用户问题: {query}")
    kb_content_text = ""
    citation_data = []
    try:
        from core.knowledge.knowledge_manager import get_kb_file_rag_configs

        # 获取该知识库每个文档的 RAG 配置
        file_rag_configs = get_kb_file_rag_configs(user_username, knowledge_base[0])

        # 实现查询改写：使用LLM纠正用户问题中的错别字、对于长文本内容提取关键信息，转成标准化的查询格式
        # 优化1+3：查询改写 + 意图路由
        new_query, intent_type = rewrite_query(query)
        logger.info(f"[{log_tag}] 意图类型: {intent_type}, 改写查询: '{new_query}'")

        # 闲聊类：跳过知识库检索
        if intent_type == "skip":
            logger.info(f"[{log_tag}] 闲聊查询，跳过知识库检索")
            return "", []


        documents = []

        # 优化2：语义漂移防护 - 计算改写查询与原始查询的语义相似度（只计算一次）
        rewrite_similarity = 1.0
        if intent_type == "rewrite" and new_query != query:
            try:
                from difflib import SequenceMatcher
                rewrite_similarity = SequenceMatcher(None, query, new_query).ratio()
                logger.debug(f"[{log_tag}] 改写相似度: {rewrite_similarity:.2f}")
            except:
                pass

        is_chroma = EMBEDDING_DATABASE == "chroma"

        def _do_search(vs, kb_id, top_k, fname, use_filter=True, search_query=None):
            """执行向量检索，支持指定查询词"""
            sq = search_query or new_query
            if EMBEDDING_DATABASE == "chroma":
                kwargs = {"k": top_k}
                if use_filter:
                    kwargs["filter"] = {"kb_id": kb_id, "source": fname}
                return vs.similarity_search_with_relevance_scores(sq, **kwargs)
            else:
                if kb_id:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    qdrant_filter = Filter(
                        must=[FieldCondition(key="metadata.kb_id", match=MatchValue(value=kb_id))])
                    raw_results = vs.similarity_search_with_score(
                        sq, k=top_k,
                        filter=qdrant_filter,
                    )
                else:
                    raw_results = vs.similarity_search_with_score(sq, k=top_k)
                return raw_results

        def _do_hybrid_search(vs, kb_id, top_k, fname, vector_weight=0.7, rrf_k=60, score_thresh_setting=None):
            """执行加权RRF混合检索"""
            sq = new_query
            
            if EMBEDDING_DATABASE == "chroma":
                # Chroma不支持混合检索，降级为向量检索
                kwargs = {"k": top_k}
                if kb_id:
                    kwargs["filter"] = {"kb_id": kb_id, "source": fname}
                return vs.similarity_search_with_relevance_scores(sq, **kwargs)
            
            # Qdrant混合检索：分别获取向量和BM25结果，然后加权RRF融合
            from qdrant_client.models import Filter, FieldCondition, MatchValue, QueryRequest, Prefetch
            from qdrant_client.http import models as qdrant_models
            
            qdrant_filter = Filter(
                must=[FieldCondition(key="metadata.kb_id", match=MatchValue(value=kb_id))])
            
            vector_results = []
            bm25_results = []
            
            # 获取Score阈值配置（只作用于向量召回支路）
            score_threshold_enabled = False
            score_threshold = Params.DEFAULT_SCORE_THRESHOLD
            if score_thresh_setting and isinstance(score_thresh_setting, dict):
                score_threshold_enabled = score_thresh_setting.get('enable', False)
                score_threshold = score_thresh_setting.get('score', Params.DEFAULT_SCORE_THRESHOLD)
            
            try:
                # 获取底层的Qdrant client
                client = vs._client
                collection_name = f"knowledge_{user_username}"
                
                # 1. 生成查询向量
                query_vector = vs._embeddings.embed_query(sq)
                
                # 2. 生成sparse向量（BM25）
                sparse_vector = None
                if hasattr(vs, '_sparse_embeddings') and vs._sparse_embeddings:
                    try:
                        sparse_vector = vs._sparse_embeddings.embed_query(sq)
                    except Exception as e:
                        logger.warning(f"[{log_tag}] 生成sparse向量失败: {e}")
                
                # 3. 向量检索
                vector_search_result = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    using='dense',
                    query_filter=qdrant_filter,
                    limit=top_k * 2,
                    with_payload=True,
                    with_vectors=False
                )
                
                # 转换为标准格式（Score阈值只作用于向量召回支路）
                for point in vector_search_result.points:
                    # Score阈值只作用于向量召回支路
                    if score_threshold_enabled and point.score < score_threshold:
                        continue  # 向量相似度低于阈值，直接过滤，不参与RRF融合
                    
                    doc_content = point.payload.get('page_content', '')
                    metadata = point.payload.get('metadata', {})
                    from langchain_core.documents import Document
                    doc = Document(page_content=doc_content, metadata=metadata)
                    vector_results.append((doc, point.score))
                
                # 4. BM25检索（如果有sparse向量）
                if sparse_vector:
                    # 将SparseVector转换为Qdrant需要的格式
                    sparse_query = qdrant_models.SparseVector(
                        indices=sparse_vector.indices,
                        values=sparse_vector.values
                    )
                    
                    bm25_search_result = client.query_points(
                        collection_name=collection_name,
                        query=sparse_query,
                        using='sparse',
                        query_filter=qdrant_filter,
                        limit=top_k * 2,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    # 转换为标准格式（BM25结果不受Score阈值影响）
                    for point in bm25_search_result.points:
                        doc_content = point.payload.get('page_content', '')
                        metadata = point.payload.get('metadata', {})
                        from langchain_core.documents import Document
                        doc = Document(page_content=doc_content, metadata=metadata)
                        bm25_results.append((doc, point.score))
                
            except Exception as e:
                logger.error(f"[{log_tag}] 混合检索失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # 降级为普通向量检索
                try:
                    vector_results = vs.similarity_search_with_score(
                        sq, k=top_k,
                        filter=qdrant_filter,
                    )
                except Exception as e2:
                    logger.error(f"[{log_tag}] 降级检索也失败: {e2}")
                    return []
            
            # 5. 加权RRF融合
            if not vector_results and not bm25_results:
                return []
            
            # 构建文档ID到结果的映射
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
                        'rrf_score': vector_weight_score
                    }
                else:
                    doc_scores[doc_id]['vector_rank'] = rank
                    doc_scores[doc_id]['rrf_score'] += vector_weight_score
            
            # 处理BM25检索结果
            bm25_weight = 1.0 - vector_weight
            for rank, (doc, raw_score) in enumerate(bm25_results or [], 1):
                doc_id = doc.metadata.get('source', '') + '_' + str(doc.metadata.get('chunk_index', rank))
                bm25_weight_score = bm25_weight * (1.0 / (rrf_k + rank))
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'doc': doc,
                        'vector_rank': None,
                        'bm25_rank': rank,
                        'rrf_score': bm25_weight_score
                    }
                else:
                    doc_scores[doc_id]['bm25_rank'] = rank
                    doc_scores[doc_id]['rrf_score'] += bm25_weight_score
            
            # 按RRF分数排序
            sorted_docs = sorted(doc_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
            
            # 转换为标准格式
            result = []
            for item in sorted_docs[:top_k]:
                result.append((item['doc'], item['rrf_score']))
            
            return result

        def _retrieve_one_doc(filename, rc):
            """检索单个文档，返回过滤后的文档列表"""
            embedding_model = rc.get('embedding_model_name')
            rag_method = rc.get('rag_retrieve_method', 'similarity')
            retrieval_mode = "hybrid" if rag_method == "hybrid" else "similarity"

            vector_store = get_vector_store(user_username, embedding_model, retrieval_mode)
            
            # 获取RRF配置
            rrf_cfg = rc.get('rrf_setting', {})
            vector_weight = rrf_cfg.get('vector_weight', 0.7) if isinstance(rrf_cfg, dict) else 0.7
            rrf_k = rrf_cfg.get('rrf_k', 60) if isinstance(rrf_cfg, dict) else 60
            
            # 获取Score阈值配置
            score_thresh_setting = rc.get('score_thresh_setting', {})
            
            # 根据检索模式执行不同的检索
            if retrieval_mode == "hybrid":
                retrieved = _do_hybrid_search(vector_store, knowledge_base[0], 5, filename, vector_weight, rrf_k, score_thresh_setting)
            else:
                retrieved = _do_search(vector_store, knowledge_base[0], 5, filename, use_filter=True)

            st_cfg = rc.get('score_thresh_setting', {})
            st_enable = st_cfg.get('enable', False) if isinstance(st_cfg, dict) else False
            st_score = st_cfg.get('score', 0.5) if isinstance(st_cfg, dict) else 0.5
            default_threshold = 0.3 if retrieval_mode == "hybrid" else Params.DEFAULT_SCORE_THRESHOLD

            base_threshold = st_score if st_enable else default_threshold
            dynamic_threshold = base_threshold
            
            # 混合检索的RRF分数只反映排名，不反映实际相关性
            # 用户配置的阈值如果过低（<0.2），会导致不相关文档通过
            # 对于混合检索，设置最低阈值保障
            if retrieval_mode == "hybrid" and st_enable and st_score < 0.2:
                logger.warning(f"[{log_tag}] 混合检索阈值过低({st_score})，已调整为0.2以避免不相关文档通过")
                dynamic_threshold = max(dynamic_threshold, 0.2)
            
            if intent_type == "rewrite" and rewrite_similarity < 0.7:
                # 改写相似度低时，限制阈值上限为0.8（避免过低）
                dynamic_threshold = min(dynamic_threshold, 0.8)

            result_docs = []
            for doc, raw in retrieved or []:
                score = raw if is_chroma else convert_vector_score(raw, vector_store, EMBEDDING_DATABASE)
                final_threshold = dynamic_threshold if st_enable else default_threshold
                # 调试日志：打印每个文档的原始分数和转换后分数
                logger.debug(f"[{log_tag}] 文档分数 - raw={raw}, converted={score}, threshold={final_threshold}, pass={score >= final_threshold}")
                if score < final_threshold:
                    continue
                result_docs.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', ''),
                    "kb_id": doc.metadata.get('kb_id', ''),
                    "score": float(score)
                })
            return result_docs

        # 并行检索所有文档
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(len(file_rag_configs), 5)) as executor:
            futures = {executor.submit(_retrieve_one_doc, fn, rc): fn for fn, rc in file_rag_configs.items()}
            for future in as_completed(futures):
                try:
                    docs = future.result()
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"[{log_tag}] 文档检索失败 {futures[future]}: {e}")

        logger.info(f"[{log_tag}] 多文档向量检索结果数={len(documents)}")

        if documents:
            documents.sort(key=lambda x: x['score'], reverse=True)
            # 取所有文档中第一个启用 rerank 的配置
            rerank_cfg = None
            for rc in file_rag_configs.values():
                rs = rc.get('rerank_model_setting', {})
                if rs.get('rerank_enable'):
                    rerank_cfg = rc
                    break

            if rerank_cfg:
                try:
                    rerank_model = rerank_cfg.get('rerank_model_setting', {}).get('rerank_model', 'bge-reranker-large')

                    # 使用缓存的 reranker 模型，避免重复加载
                    reranker = _get_reranker(rerank_model)

                    start_time = time.time()
                    # 优化：截断文档内容到 300 字，减少 rerank 推理耗时
                    MAX_RERANK_CONTENT_LEN = 300
                    pairs = [[query, d['content'][:MAX_RERANK_CONTENT_LEN]] for d in documents]
                    scores = reranker.predict(pairs)
                    compute_time = time.time() - start_time
                    logger.info(f"[{log_tag}] Rerank 推理耗时: {compute_time:.2f}s (截断长度={MAX_RERANK_CONTENT_LEN})")

                    if not isinstance(scores, list):
                        if hasattr(scores, 'tolist'):
                            scores = scores.tolist()
                        else:
                            scores = [scores]
                    # 保留原始检索分数，rerank 分数仅用于排序
                    for d, s in zip(documents, scores):
                        d['rerank_score'] = float(s)
                    documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
                    top_k = rerank_cfg.get('rerank_model_setting', {}).get('rerank_top_k', 3)
                    documents = documents[:top_k]
                    logger.info(f"[{log_tag}] Rerank 结果数={len(documents)}")
                    for i, doc in enumerate(documents):
                        logger.info(f"[{log_tag}] Rerank文档[{i+1}] 来源={doc.get('source', '')} 分数={doc.get('rerank_score', 0):.4f} 内容={doc.get('content', '')[:200]}")
                except Exception as rerank_err:
                    logger.warning(f"[{log_tag}] Rerank 失败，使用原始排序: {rerank_err}")

            content_parts = []
            for i, doc in enumerate(documents):
                doc['rank'] = i + 1
                source = doc.get('source', '')
                content_parts.append(f"[{doc['rank']}] 来源: {source}\n{doc['content']}")
            kb_content_text = "\n---参考知识---\n" + "\n".join(content_parts)

            # 构建引用数据（用于前端展示）
            citation_data = [
                {
                    "rank": doc.get('rank', i + 1),
                    "source": doc.get('source', ''),
                    "content": doc.get('content', ''),
                    "score": round(doc.get('score', 0), 4)
                }
                for i, doc in enumerate(documents)
            ]
    except Exception as e:
        logger.warning(f"[{log_tag}] 知识库检索失败: {str(e)}")

    return kb_content_text, citation_data


def _build_messages(agent: Agent, data: AgentChatRequest, kb_content_text: str) -> list:
    _, HumanMessage, AIMessage, SystemMessage = _get_langchain_classes()

    system_prompt = data.system_prompt or agent.system_prompt or ""
    if data.variables:
        for k, v in data.variables.items():
            system_prompt = system_prompt.replace("{{" + k + "}}", str(v))
            system_prompt = system_prompt.replace("{" + k + "}", str(v))

    if kb_content_text:
        system_prompt = system_prompt.replace("{{knowledge_content}}", kb_content_text)
        system_prompt = system_prompt.replace("{knowledge_content}", kb_content_text)

    messages = [SystemMessage(content=system_prompt)]

    use_memory_history = not (data.history and len(data.history) > 0)
    if use_memory_history:
        session_key = f"agent_{agent.id}_{data.session_id or 'default'}"
        history = _get_chat_history(session_key)
        _trim_history(history)
        if not history.messages and agent.opening and agent.opening.strip():
            history.add_ai_message(agent.opening.strip())
        messages.extend(history.messages)
    else:
        if data.history:
            for item in data.history:
                role = item.get('role', '')
                content = item.get('content', '')
                if role == 'user':
                    messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=data.message))
    return messages

@llm_breaker
async def _establish_llm_connection(model_name: str, messages, enable_thinking: bool = False, tools: list = None):
    """建立与大模型的连接（带熔断保护）。
    
    只负责建立连接和验证响应状态，不负责流式读取。
    异常: 连接失败、HTTP错误等会触发熔断器计数
    返回: httpx stream 上下文管理器（需用 async with 进入）
    """
    from core.agent_langgraph import get_http_client, _to_openai_messages
    import os
    # 根据模型名称选择正确的 API 端点和 Key，并映射为 API 实际支持的模型名
    model_lower = model_name.lower()
    if "deepseek" in model_lower:
        # DeepSeek 模型通过 DashScope 兼容接口调用
        base_url = Params.API_BASE
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        # 映射为 DashScope 实际支持的模型名
        if "deepseek-v4-pro" in model_lower:
            model_name = "deepseek-v4-pro"
        elif "deepseek-reasoner" in model_lower:
            model_name = "deepseek-reasoner"
        elif "deepseek-chat" in model_lower:
            model_name = "deepseek-chat"
    elif "glm" in model_lower:
        base_url = "https://open.bigmodel.cn/api/paas/v4"
        api_key = os.getenv("GLM_API_KEY", "")
        if "glm-5.2-fast-preview" in model_lower:
            model_name = "glm-5.2-fast-preview"
        elif "glm-4-plus" in model_lower:
            model_name = "glm-4-plus"
    else:
        base_url = Params.API_BASE
        api_key = os.getenv("DASHSCOPE_API_KEY", "")

    payload = {
        "model": model_name,
        "messages": _to_openai_messages(messages),
        "temperature": 0.2,
        "stream": True,
        "enable_thinking": enable_thinking,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    url = f"{base_url.rstrip('/')}/chat/completions"
    
    _t1 = time.time()
    # stream() 返回异步上下文管理器，不能 await
    client = get_http_client()
    headers = {"Authorization": f"Bearer {api_key}"}
    response_cm = client.stream("POST", url, json=payload, headers=headers)
    logger.info(f"[PERF] LLM连接建立耗时 {time.time() - _t1:.3f}s, model={model_name}, base_url={base_url}")
    return response_cm


async def _make_chat_generator(model_name: str, messages, agent_id: int, session_id: str, http_request,
                               log_tag: str = "", citation_data: list = None, request_start_time: float = None, 
                               enable_thinking: bool = False, data = None, tools_config: list = None,
                               history = None):
    """构造一个 async generator，用于 StreamingResponse 流式输出 AI 回复。
    流式协议：
        每个 chunk: {"text": full_text, "status": "streaming"}
        结束时:     {"text": full_text, "status": "done", "citation": [...], "response_time_s": xxx}
        被中断:     {"text": full_text, "status": "stop"}
        出错:       {"text": str(e), "status": "error"}
    
    熔断保护：连接建立阶段的异常会触发熔断器计数，流式输出阶段的异常不会触发熔断。
    支持工具调用：如果配置了 tools_config，会先进行工具调用，再流式输出最终回复。
    """
    import asyncio
    import json as _json
    _, HumanMessage, AIMessage, _ = _get_langchain_classes()

    # 检查熔断器状态
    if llm_breaker.current_state == 'open':
        raise HTTPException(status_code=503, detail="大模型服务暂时不可用，请稍后重试")

    # 将工具配置转换为 OpenAI 格式
    openai_tools = _tools_to_openai_format(tools_config) if tools_config else None
    if openai_tools:
        logger.info(f"[{log_tag}] 工具 schema: {json.dumps(openai_tools, ensure_ascii=False)}")
        # 调试：打印原始工具配置
        logger.info(f"[{log_tag}] 原始工具配置: {json.dumps(tools_config, ensure_ascii=False)}")

    async def generate():
        nonlocal openai_tools, messages
        full_text = ""
        response = None
        tool_call_rounds = 0
        max_tool_rounds = 3  # 最多 3 轮工具调用
        final_response_attempts = 0  # 记录尝试生成最终回复的次数
        max_final_attempts = 2  # 最多尝试2次生成最终回复
        executed_tool_signatures = set()  # 记录已执行的工具调用签名（工具名+参数）
        try:
            logger.info(f"[{log_tag}] 开始流式输出, tools={len(tools_config) if tools_config else 0}")
            
            while tool_call_rounds <= max_tool_rounds:
                # 在每次LLM调用之前，清理消息历史中的工具调用信息
                # 避免LLM在回复中引用之前的工具名称、参数和历史问题
                if tool_call_rounds > 0:
                    logger.info(f"[{log_tag}] 清理消息历史中的工具调用信息（第{tool_call_rounds}轮）")
                    cleaned_messages = []
                    for msg in messages:
                        # 处理字典类型的消息
                        if isinstance(msg, dict):
                            role = msg.get("role")
                            if role == "tool":
                                # 保留工具结果，但转换为普通消息格式，让LLM能看到工具返回的内容
                                cleaned_messages.append({
                                    "role": "user",
                                    "content": f"[工具 {msg.get('name', '')} 返回结果]\n{msg.get('content', '')}"
                                })
                            elif role == "assistant" and "tool_calls" in msg:
                                # 将包含tool_calls的assistant消息转换为普通文本消息
                                cleaned_messages.append({
                                    "role": "assistant",
                                    "content": msg.get("content", "")
                                })
                            else:
                                cleaned_messages.append(msg)
                        # 处理LangChain消息对象
                        else:
                            # 保留系统消息、用户消息等非工具相关消息
                            if hasattr(msg, 'type') and msg.type != "tool":
                                cleaned_messages.append(msg)
                    messages = cleaned_messages
                
                # 建立连接（带熔断保护）
                response_cm = await _establish_llm_connection(model_name, messages, enable_thinking, openai_tools)
                
                # 收集本次响应的完整内容（用于检测工具调用）
                collected_chunks = []
                tool_calls_buffer = []
                tool_calls_map = {}  # 按 index 聚合 tool_calls
                tool_calls_map_logged = False  # 标记是否已打印工具调用信号日志
                
                async with response_cm as response:
                    response.raise_for_status()
                    _t2 = time.time()
                    _chunk_idx = 0
                    _line_count = 0
                    async for raw_line in response.aiter_lines():
                        _line_count += 1
                        if await http_request.is_disconnected():
                            logger.warning(f"[{log_tag}] 客户端断开连接，已读取 {_line_count} 行")
                            return
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            response_data = json.loads(data_str)
                        except json.JSONDecodeError as e:
                            logger.warning(f"[{log_tag}] JSON解析失败 #{_line_count}: {e}")
                            continue
                        
                        # 收集 tool_calls（可能分布在多个 chunk 中）
                        choices_for_tool = response_data.get("choices", [])
                        if not choices_for_tool:
                            continue
                        msg = choices_for_tool[0].get("delta", {})
                        if msg.get("tool_calls"):
                            for tc in msg["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_map:
                                    tool_calls_map[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                                existing = tool_calls_map[idx]
                                if tc.get("id"):
                                    existing["id"] = tc["id"]
                                if tc.get("type"):
                                    existing["type"] = tc["type"]
                                func = tc.get("function", {})
                                if func.get("name"):
                                    existing["function"]["name"] = func["name"]
                                if func.get("arguments"):
                                    existing["function"]["arguments"] += func["arguments"]
                                # 调试：打印每个 chunk 的 tool_call 片段
                                logger.debug(f"[{log_tag}] tool_call chunk idx={idx}, name='{func.get('name','')}', args='{func.get('arguments','')}'")
                            # 只在第一次检测到 tool_calls 时记录一次
                            if not tool_calls_map_logged:
                                tool_calls_map_logged = True
                                logger.info(f"[{log_tag}] 检测到工具调用信号")
                        
                        choices = response_data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            _chunk_idx += 1
                            collected_chunks.append(content)
                            if _chunk_idx == 1:
                                logger.info(f"[PERF] {log_tag}: 首字到达耗时 {time.time() - _t2:.3f}s")
                            full_text += content
                            yield f"data: {_json.dumps({'text': full_text, 'status': 'streaming'}, ensure_ascii=False)}\n\n"
                    logger.info(f"[{log_tag}] 流式读取完成，共读取 {_line_count} 行，{_chunk_idx} 个有效chunk")
                
                # 将聚合后的 tool_calls 转为列表
                tool_calls_buffer = list(tool_calls_map.values())
                
                # 调试：打印聚合后的 tool_calls
                if tool_calls_buffer:
                    logger.info(f"[{log_tag}] 聚合后的 tool_calls: {json.dumps(tool_calls_buffer, ensure_ascii=False)}")

                # 只获取和用户问题相关的 tool_calls_buffer，在此添加逻辑
                # 过滤掉与当前用户问题无关的工具调用（避免历史对话中的工具调用被重复执行）
                current_question = data.message if data else ""
                relevant_tool_calls = []
                for tc in tool_calls_buffer:
                    tool_name = tc.get("function", {}).get("name", "")
                    tool_args_str = tc.get("function", {}).get("arguments", "{}")
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    # 检查工具调用是否已经执行过（避免重复调用同一工具）
                    tool_signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                    if tool_signature in executed_tool_signatures:
                        logger.info(f"[{log_tag}] 过滤掉重复的工具调用: {tool_name} (参数: {tool_args_str})")
                        continue
                    
                    # 检查工具调用的参数值是否在当前用户问题中出现
                    is_relevant = False
                    for param_value in tool_args.values():
                        if isinstance(param_value, str) and param_value and param_value in current_question:
                            is_relevant = True
                            break
                    
                    if is_relevant:
                        relevant_tool_calls.append(tc)
                    else:
                        logger.info(f"[{log_tag}] 过滤掉不相关的工具调用: {tool_name} (参数: {tool_args_str})")
                
                tool_calls_buffer = relevant_tool_calls
                logger.info(f"[{log_tag}] 过滤后保留 {len(tool_calls_buffer)} 个相关工具调用")


                # 检查是否有工具调用
                if tool_calls_buffer and openai_tools:
                    tool_call_rounds += 1
                    logger.info(f"[{log_tag}] 检测到 {len(tool_calls_buffer)} 个工具调用，第 {tool_call_rounds} 轮")
                    
                    # 收集实际执行的工具调用（用于添加到消息历史）
                    executed_tool_calls = []
                    
                    # 执行工具调用
                    for tc in tool_calls_buffer:
                        tool_name = tc.get("function", {}).get("name", "")
                        tool_args_str = tc.get("function", {}).get("arguments", "{}")
                        
                        # 调试：打印原始参数
                        logger.info(f"[{log_tag}] 工具 {tool_name} 原始参数字符串: {tool_args_str}")
                        
                        # 跳过名称为空的工具调用
                        if not tool_name:
                            logger.warning(f"[{log_tag}] 跳过名称为空的工具调用: {tc}")
                            continue
                        
                        try:
                            tool_args = json.loads(tool_args_str)
                        except json.JSONDecodeError as e:
                            logger.warning(f"[{log_tag}] 参数 JSON 解析失败: {e}, 原始: {tool_args_str}")
                            tool_args = {}
                        
                        # 找到对应的工具配置
                        tool_config = next((t for t in tools_config if t["name"] == tool_name), None)
                        if tool_config:
                            logger.info(f"[{log_tag}] 执行工具: {tool_name}, 参数: {tool_args}")
                            result = await _execute_tool(tool_config, tool_args)
                            logger.info(f"[{log_tag}] 工具 {tool_name} 返回: {result[:200]}")
                            
                            # 记录工具调用签名，防止重复调用
                            tool_signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                            executed_tool_signatures.add(tool_signature)
                            
                            # 记录实际执行的工具调用
                            executed_tool_calls.append(tc)
                            
                            # 将工具结果添加到消息历史
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id", ""),
                                "name": tool_name,
                                "content": result
                            })
                        else:
                            logger.warning(f"[{log_tag}] 未找到工具配置: {tool_name} (可用工具: {[t['name'] for t in tools_config]})")
                    
                    # 只添加实际执行的工具调用到 assistant 消息中（避免 LLM 在下一轮看到未执行的 tool_calls）
                    if executed_tool_calls:
                        messages.append({
                            "role": "assistant",
                            "tool_calls": executed_tool_calls
                        })
                    
                    # 清空 full_text，准备接收下一轮回复
                    full_text = ""
                    tool_calls_buffer = []
                    continue  # 继续下一轮 LLM 调用
                
                # 没有工具调用（或所有工具调用都被过滤），结束循环
                # 如果之前执行过工具但被过滤了，需要让 LLM 生成最终回复
                if not tool_calls_buffer and tool_call_rounds > 0:
                    # 所有工具调用都被过滤，但之前执行过工具
                    # 让LLM基于已有的工具结果生成最终回复
                    if full_text.strip():
                        # LLM已经生成了有效回复（有有效chunk），直接使用，不再重试
                        logger.info(f"[{log_tag}] LLM已生成有效回复（{len(full_text)}字符），直接使用，不再重试")
                        break
                    if openai_tools is None:
                        # 已经设置过 openai_tools = None，但LLM仍然输出工具调用
                        # 说明LLM受消息历史影响继续输出工具调用格式，直接结束
                        logger.warning(f"[{log_tag}] 已禁用工具调用但LLM仍输出工具调用信号，强制结束循环")
                        break
                    if final_response_attempts >= max_final_attempts:
                        # 超过最大重试次数，直接结束
                        logger.warning(f"[{log_tag}] 尝试生成最终回复超过{max_final_attempts}次，LLM仍返回工具调用，强制结束")
                        break
                    final_response_attempts += 1
                    logger.info(f"[{log_tag}] 所有工具调用被过滤，让LLM基于工具结果生成最终回复（第{final_response_attempts}次尝试）")
                    openai_tools = None  # 不传tools，让LLM生成文本回复
                    # 清理消息历史中的工具调用信息，避免LLM在回复中引用之前的工具名称和参数
                    cleaned_messages = []
                    for msg in messages:
                        # 处理字典类型的消息
                        if isinstance(msg, dict):
                            role = msg.get("role")
                            if role == "tool":
                                continue  # 跳过工具结果消息
                            elif role == "assistant" and "tool_calls" in msg:
                                # 将包含tool_calls的assistant消息转换为普通文本消息
                                cleaned_messages.append({
                                    "role": "assistant",
                                    "content": msg.get("content", "")
                                })
                            else:
                                cleaned_messages.append(msg)
                        # 处理LangChain消息对象
                        else:
                            # 保留系统消息、用户消息等非工具相关消息
                            if hasattr(msg, 'type') and msg.type != "tool":
                                cleaned_messages.append(msg)
                    messages = cleaned_messages
                    continue  # 继续下一轮LLM调用
                break
            
            # 工具调用循环结束后，清理消息历史中的工具调用信息
            # 避免LLM在后续对话中引用之前的工具名称和参数
            if tool_call_rounds > 0:
                logger.info(f"[{log_tag}] 工具调用循环结束，清理消息历史中的工具调用信息")
                cleaned_messages = []
                for msg in messages:
                    # 处理字典类型的消息
                    if isinstance(msg, dict):
                        role = msg.get("role")
                        if role == "tool":
                            continue  # 跳过工具结果消息
                        elif role == "assistant" and "tool_calls" in msg:
                            # 将包含tool_calls的assistant消息转换为普通文本消息
                            cleaned_messages.append({
                                "role": "assistant",
                                "content": msg.get("content", "")
                            })
                        else:
                            cleaned_messages.append(msg)
                    # 处理LangChain消息对象
                    else:
                        # 保留系统消息、用户消息等非工具相关消息
                        if hasattr(msg, 'type') and msg.type != "tool":
                            cleaned_messages.append(msg)
                messages = cleaned_messages

            if data and not data.history:
                # 如果传入了 history 对象，直接使用它保存
                if history is not None:
                    history.add_message(HumanMessage(content=data.message))
                    history.add_message(AIMessage(content=full_text))
                    _trim_history(history)
                else:
                    # 否则使用原来的逻辑（向后兼容）
                    session_key = f"agent_{agent_id}_{session_id or 'default'}"
                    hist = _get_chat_history(session_key)
                    hist.add_message(HumanMessage(content=data.message))
                    hist.add_message(AIMessage(content=full_text))
                    _trim_history(hist)
            meta = {"citation": citation_data or []}
            if request_start_time is not None:
                meta["response_time_s"] = round(time.time() - request_start_time, 3)
            yield f"data: {_json.dumps({'text': full_text, 'status': 'done', **meta}, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            yield f"data: {_json.dumps({'text': full_text, 'status': 'stop'}, ensure_ascii=False)}\n\n"
        except pybreaker.CircuitBreakerError:
            logger.error(f"[{log_tag}] 熔断器已打开，拒绝请求")
            yield f"data: {_json.dumps({'text': '大模型服务异常，请稍后重试', 'status': 'error'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"[{log_tag}] error: {e}")
            yield f"data: {_json.dumps({'text': str(e), 'status': 'error'}, ensure_ascii=False)}\n\n"

    return generate()


# 获取用户的所有智能体列表
@router.get("")
async def list_agents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取当前用户的所有智能体列表"""
    agents = db.query(Agent).filter(Agent.user_name == current_user.username).order_by(Agent.updated_at.desc()).all()
    return [AgentResponse.model_validate(a) for a in agents]


# 创建智能体
@router.post("")
async def create_agent(
    body: AgentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """创建新智能体"""
    agent = Agent(
        user_id=current_user.id,
        user_name=current_user.username,
        name=body.name,
        description=body.description,
        avatar=body.avatar or "",
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return AgentResponse.model_validate(agent)


# 获取可用模型列表（必须在 /{agent_id} 之前定义）
@router.get("/models")
async def list_models():
    """获取可用的模型列表"""
    from settings.Define import Params
    return {
        "models": [
            {"id": "glm", "name": "GLM-5.2 Fast Preview"},
            {"id": "qwen-turbo", "name": "Qwen Turbo"},
            {"id": "qwen-vl-max", "name": "Qwen VL Max"},
            {"id": "default", "name": "Qwen3.8-Flash"},
        ]
    }


# Prompt 优化接口
@router.post("/prompt-optimize")
async def optimize_prompt(
    data: PromptOptimizeRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
):
    is_targeted = data.mode == "targeted"
    if is_targeted:
        base_prompt = (
            "你是一个资深 AI 提示词调优顾问。用户已经有了一段可用的系统提示词，"
            "现在希望在保留原有意图、角色和主体结构的前提下，朝某个特定方向进行调整和强化。\n\n"
            "调优原则：\n"
            "1. 保留原有角色身份、核心任务和行为准则的主体内容，不要推倒重来\n"
            "2. 围绕用户指定的目标，有针对性地优化语气、风格、侧重点、约束条件或输出方式\n"
            "3. 如果用户指定的方向与现有内容冲突，以用户新目标为准进行局部调整\n"
            "4. 如果用户输入里有 {{变量名}} 形式的占位符，必须完整保留，不要修改它\n"
            "5. 直接输出调优后的提示词，不要说\"好的，以下是优化后的提示词\"之类的前缀"
        )
    else:
        base_prompt = (
            "你是一个世界顶级的 Prompt 工程师。你的任务是把用户提供的提示词，"
            "重写成一段结构清晰、角色鲜明、指令明确、边界清晰的高质量系统提示词。\n\n"
            "改写要求：\n"
            "1. 明确角色与身份：说明 AI 扮演什么角色\n"
            "2. 给出具体目标：AI 需要完成什么任务\n"
            "3. 列出行为准则：AI 应该 / 不应该做什么\n"
            "4. 添加输出格式：建议以什么格式回答\n"
            "5. 保留用户原始意图，不要添加无关功能\n"
            "6. 如果用户输入里有 {{变量名}} 形式的占位符，必须完整保留，不要修改它\n\n"
            "直接输出改写后的提示词，不要说\"好的，以下是优化后的提示词\"之类的前缀。"
        )

    if data.instruction and data.instruction.strip():
        base_prompt += f"\n\n{'调优目标' if is_targeted else '用户额外的优化要求'}：{data.instruction.strip()}"

    kb_names_for_prompt = []
    if data.knowledge_bases:
        try:
            from core.knowledge.knowledge_manager import load_kb_metadata
            meta_list = load_kb_metadata(current_user.username)
            kb_id_map = {kb['id']: kb.get('name', kb['id']) for kb in meta_list}
            for kb_id in data.knowledge_bases:
                name = kb_id_map.get(kb_id, str(kb_id))
                kb_names_for_prompt.append(name)
        except Exception:
            for kb_id in data.knowledge_bases:
                kb_names_for_prompt.append(str(kb_id))

    if kb_names_for_prompt:
        kb_names_str = "、".join(kb_names_for_prompt)
        base_prompt += (
            f"\n\n【知识库强制规则】\n"
            f"用户已为该智能体绑定以下知识库：{kb_names_str}。\n"
            f"你生成的系统提示词中必须遵循以下规则：\n"
            f"1. 每一处提到\"知识库\"的地方，必须紧跟具体的知识库名称，格式为：知识库具体名称\n"
            f"   例如：将\"结合知识库检索结果\"改为\"结合知识库{kb_names_for_prompt[0]}检索结果\"\n"
            f"2. 每一处提到知识库\"内容\"或\"检索结果\"的地方，必须在后面紧跟 {{{{knowledge_content}}}} 内置变量\n"
            f"   例如：将\"引用知识库中的内容\"改为\"引用知识库{kb_names_for_prompt[0]}中的内容{{{{knowledge_content}}}}\"\n"
            f"3. 这两条规则不可省略，必须严格执行"
        )

    user_msg_prefix = "调优以下提示词" if is_targeted else "优化以下提示词"
    if is_targeted and data.instruction and data.instruction.strip():
        user_msg_prefix += f"，使其更贴近目标：{data.instruction.strip()}"
    user_msg = f"{user_msg_prefix}\n\n---\n{data.prompt or '(空，帮我从零开始写一个好的系统提示词)'}\n---"
    logger.debug(f"[mode={data.mode}] user_msg: {user_msg}")

    async def generate():
        from langchain_openai import ChatOpenAI
        import os
        _, HumanMessage, _, SystemMessage = _get_langchain_classes()
        llm = ChatOpenAI(
            model=Params.DEFAULT_CHAT_MODEL,
            temperature=0.6,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
            streaming=True,
        )

        full_text = ""
        try:
            async for chunk in llm.astream([
                SystemMessage(content=base_prompt),
                HumanMessage(content=user_msg),
            ]):
                if await http_request.is_disconnected():
                    break
                token = chunk.content or ""
                if token:
                    full_text += token
                    yield f"data: {json.dumps({'text': token, 'status': 'streaming'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'text': '', 'status': 'done', 'full_text': full_text}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"优化提示词失败: {e}")
            yield f"data: {json.dumps({'text': str(e), 'status': 'error'}, ensure_ascii=False)}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")


# 获取当前用户的知识库列表（必须在 /{agent_id} 之前定义）
@router.get("/knowledge")
async def list_user_knowledge(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
):
    """获取当前用户的所有知识库列表"""
    from core.knowledge.knowledge_manager import load_kb_metadata
    meta_list = load_kb_metadata(current_user.username)
    documents = []
    for kb in meta_list:
        documents.append({
            "id": kb.get("id"),
            "name": kb.get("name") or kb.get("id") or "未命名",
            "description": kb.get("description", ""),
            "chunk_strategy": kb.get("chunk_strategy", "standard"),
        })
    return {"documents": documents}


# ==================== 自定义工具管理 API ====================
# 注意：必须在 /{agent_id} 路由之前定义，否则会被错误匹配

class CustomToolCreate(BaseModel):
    """创建自定义工具请求体"""
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    url: str
    method: str = "GET"
    timeout: int = 10
    headers: Optional[dict] = None
    parameters: Optional[list] = None
    auth_type: str = "none"
    auth_value: Optional[str] = None


class CustomToolUpdate(BaseModel):
    """更新自定义工具请求体"""
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    url: Optional[str] = None
    method: Optional[str] = None
    timeout: Optional[int] = None
    headers: Optional[dict] = None
    parameters: Optional[list] = None
    auth_type: Optional[str] = None
    auth_value: Optional[str] = None


class CustomToolResponse(BaseModel):
    """自定义工具响应体"""
    id: int
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    url: str
    method: str
    timeout: int
    headers: Optional[dict] = None
    parameters: Optional[list] = None
    auth_type: str
    auth_value: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

    @field_validator("headers", "parameters", mode="before")
    @classmethod
    def parse_json_fields(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return None
        return v


@router.get("/tools")
async def list_custom_tools(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取当前用户的所有自定义工具"""
    tools = db.query(CustomTool).filter(
        CustomTool.user_name == current_user.username
    ).order_by(CustomTool.created_at.desc()).all()

    return {
        "ok": True,
        "tools": [
            {
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "category": t.category,
                "url": t.url,
                "method": t.method,
                "timeout": t.timeout,
                "headers": json.loads(t.headers) if t.headers else None,
                "parameters": json.loads(t.parameters) if t.parameters else None,
                "auth_type": t.auth_type,
                "auth_value": t.auth_value,
                "created_at": t.created_at.strftime("%Y-%m-%d %H:%M:%S") if t.created_at else None,
            }
            for t in tools
        ]
    }


@router.post("/tools")
async def create_custom_tool(
    body: CustomToolCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """创建自定义工具"""
    # 检查同名工具
    existing = db.query(CustomTool).filter(
        CustomTool.user_name == current_user.username,
        CustomTool.name == body.name
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="已存在同名工具")

    tool = CustomTool(
        user_id=current_user.id,
        user_name=current_user.username,
        name=body.name,
        description=body.description,
        category=body.category,
        url=body.url,
        method=body.method.upper(),
        timeout=body.timeout,
        headers=json.dumps(body.headers, ensure_ascii=False) if body.headers else None,
        parameters=json.dumps(body.parameters, ensure_ascii=False) if body.parameters else None,
        auth_type=body.auth_type,
        auth_value=body.auth_value,
    )
    db.add(tool)
    db.commit()
    db.refresh(tool)

    return {
        "ok": True,
        "tool": {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "url": tool.url,
            "method": tool.method,
            "timeout": tool.timeout,
            "headers": json.loads(tool.headers) if tool.headers else None,
            "parameters": json.loads(tool.parameters) if tool.parameters else None,
            "auth_type": tool.auth_type,
        }
    }


@router.put("/tools/{tool_id}")
async def update_custom_tool(
    tool_id: int,
    body: CustomToolUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """更新自定义工具"""
    tool = db.query(CustomTool).filter(
        CustomTool.id == tool_id,
        CustomTool.user_name == current_user.username
    ).first()
    if not tool:
        raise HTTPException(status_code=404, detail="工具不存在或无权操作")

    # 如果修改了名称，检查是否与其他工具重名
    if body.name and body.name != tool.name:
        existing = db.query(CustomTool).filter(
            CustomTool.user_name == current_user.username,
            CustomTool.name == body.name,
            CustomTool.id != tool_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="已存在同名工具")

    if body.name is not None:
        tool.name = body.name
    if body.description is not None:
        tool.description = body.description
    if body.category is not None:
        tool.category = body.category
    if body.url is not None:
        tool.url = body.url
    if body.method is not None:
        tool.method = body.method.upper()
    if body.timeout is not None:
        tool.timeout = body.timeout
    if body.headers is not None:
        tool.headers = json.dumps(body.headers, ensure_ascii=False)
    if body.parameters is not None:
        tool.parameters = json.dumps(body.parameters, ensure_ascii=False)
    if body.auth_type is not None:
        tool.auth_type = body.auth_type
    if body.auth_value is not None:
        tool.auth_value = body.auth_value

    db.commit()
    db.refresh(tool)

    return {
        "ok": True,
        "tool": {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "url": tool.url,
            "method": tool.method,
            "timeout": tool.timeout,
            "headers": json.loads(tool.headers) if tool.headers else None,
            "parameters": json.loads(tool.parameters) if tool.parameters else None,
            "auth_type": tool.auth_type,
        }
    }


@router.delete("/tools/{tool_id}")
async def delete_custom_tool(
    tool_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除自定义工具"""
    tool = db.query(CustomTool).filter(
        CustomTool.id == tool_id,
        CustomTool.user_name == current_user.username
    ).first()
    if not tool:
        raise HTTPException(status_code=404, detail="工具不存在或无权操作")

    db.delete(tool)
    db.commit()

    return {"ok": True, "message": "工具已删除"}


# 获取单个智能体详情（用于编辑器）
@router.get("/{agent_id}")
async def get_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取单个智能体的详情（用于编辑器）"""
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_name == current_user.username).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在或无权访问")
    return AgentResponse.model_validate(agent)


# 更新智能体
@router.put("/{agent_id}")
async def update_agent(
    agent_id: int,
    body: AgentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """更新智能体（仅创建者可更新）"""
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_name == current_user.username).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在或无权更新")
    if body.name is not None:
        agent.name = body.name
    if body.description is not None:
        agent.description = body.description
    if body.avatar is not None:
        agent.avatar = body.avatar
    if body.system_prompt is not None:
        agent.system_prompt = body.system_prompt
    if body.model is not None:
        agent.model = body.model
    if body.variables is not None:
        agent.variables = _serialize_vars(body.variables)
    if body.knowledge_base is not None:
        agent.knowledge_base = _serialize_vars(body.knowledge_base)
    if body.tools is not None:
        agent.tools = _serialize_vars(body.tools)
    if body.opening is not None:
        agent.opening = body.opening
    if body.presets is not None:
        agent.presets = _serialize_vars(body.presets)
    if body.is_published is not None:
        agent.is_published = body.is_published
    if body.deep_think is not None:
        agent.deep_think = body.deep_think
    if body.struct_output is not None:
        import json
        agent.struct_output = json.dumps(body.struct_output, ensure_ascii=False)
    if body.temperature is not None:
        agent.temperature = body.temperature
    agent.updated_at = datetime.now()
    db.commit()
    db.refresh(agent)
    return AgentResponse.model_validate(agent)


# 删除智能体
@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除智能体（仅创建者可删除）"""
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_name == current_user.username).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在或无权删除")
    db.delete(agent)
    db.commit()
    return {"ok": True, "message": "删除成功"}


# 获取智能体商店列表（已发布的智能体，公开接口无需认证）
@router.get("/store/list")
async def list_store_agents(
    db: Session = Depends(get_db),
):
    """获取所有已发布的智能体列表（智能体商店）"""
    agents = db.query(Agent).filter(Agent.is_published == True).order_by(Agent.updated_at.desc()).all()
    return [AgentResponse.model_validate(a) for a in agents]


# 获取单个智能体详情（公开接口无需认证）
@router.get("/store/{agent_id}")
async def get_store_agent(
    agent_id: int,
    db: Session = Depends(get_db),
):
    """获取单个已发布智能体的详情"""
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.is_published == True).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在或未发布")
    return AgentResponse.model_validate(agent)


# agent-user页面，用户与智能体进行对话
@router.post("/store/{agent_id}/chat")
@limiter.limit("8/minute")   # 每个IP每分钟最多8次请求
async def store_chat_with_agent(
        request: Request,
        agent_id: int,
        data: AgentChatRequest,
        db: Session = Depends(get_db),
):
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(status_code=404, detail="智能体不存在")
        if not agent.is_published:
            raise HTTPException(status_code=404, detail="该智能体未发布")
        if not data.message or not data.message.strip():
            raise HTTPException(status_code=400, detail="消息内容不能为空")

        owner_user = db.query(User).filter(User.id == agent.user_id).first()

        # 记录请求开始时间（用于计算响应耗时）
        request_start_time = time.time()
        
        # 优化：判断用户问题是否与智能体角色相关，不相关则跳过 RAG 检索
        system_prompt = agent.system_prompt or ""
        if need_rag_search(data.message):
            kb_text, citation_data = _build_rag_context(agent, owner_user.username if owner_user else "", data.message,
                                                        log_tag="store_chat")
            logger.info(f"[store_chat] 问题与智能体相关，执行 RAG 检索, kb_len={len(kb_text)}")
        else:
            kb_text = ""
            citation_data = []
            logger.info(f"[store_chat] 问题与智能体不相关，跳过 RAG 检索")
        messages = _build_messages(agent, data, kb_text)

        logger.info(f"[store_chat] messages_count={len(messages)}, kb_len={len(kb_text)}")

        from fastapi.responses import StreamingResponse
        tools_config = json.loads(agent.tools) if agent.tools else []
        gen = await _make_chat_generator(agent.model, messages, agent_id, data.session_id, request, log_tag="store",
                                         citation_data=citation_data, request_start_time=request_start_time,
                                         enable_thinking=data.deep_think, data=data, tools_config=tools_config)
        return StreamingResponse(gen, media_type="text/event-stream")
    except pybreaker.CircuitBreakerError as e:
        raise HTTPException(status_code=503, detail="大模型服务异常，请稍后重试")
    except LLMRemoteError:
        raise HTTPException(status_code=500, detail="调用大模型服务失败，请重试")


@router.get("/{agent_id}/knowledge")
async def list_agent_knowledge(
        agent_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问")

    from core.knowledge.knowledge_manager import load_kb_metadata
    user_dir = get_user_knowledge_dir(current_user.username)
    meta_list = load_kb_metadata(current_user.username)
    kb_name_map = {kb['id']: kb['name'] for kb in meta_list}

    documents = []
    for subdir in user_dir.iterdir() if user_dir.exists() else []:
        if subdir.is_dir() and subdir.name.startswith('kb_'):
            for file_path in subdir.iterdir():
                if file_path.is_file():
                    filename = file_path.name
                    if not (filename.startswith('~$') or filename.startswith('.') or filename == 'Thumbs.db'):
                        kb_name = kb_name_map.get(subdir.name, subdir.name)
                        documents.append({
                            "filename": filename,
                            "size": file_path.stat().st_size,
                            "modified_at": file_path.stat().st_mtime,
                            "kb_id": subdir.name,
                            "kb_name": kb_name,
                        })

    selected = set(_deserialize_vars(agent.knowledge_base))
    for doc in documents:
        doc["selected"] = doc["filename"] in selected

    return {"documents": documents}


def _keyword_fallback(vector_store, query, kb_id=None, existing_ids=None, top_k=3):
    if EMBEDDING_DATABASE != "chroma":
        return []
    if not query.strip():
        return []

    kwargs = {}
    if kb_id:
        kwargs["where"] = {"kb_id": kb_id}

    try:
        results = vector_store.get(**kwargs)
    except Exception as e:
        logger.warning(f"关键词补充: Chroma get 失败 {e}")
        return []

    if not results or not results.get("documents"):
        return []

    import re
    chars = [c for c in re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', query) if len(c) >= 2]
    if not chars:
        chars = [c for c in query if '\u4e00' <= c <= '\u9fa5']
    if not chars:
        return []

    scored = []
    for i, doc_text in enumerate(results["documents"]):
        if existing_ids and doc_text in existing_ids:
            continue
        hit = sum(1 for c in chars if c in doc_text)
        score = hit / max(len(chars), 1)
        if score > 0.1:
            scored.append((doc_text, results["metadatas"][i] if results["metadatas"] else {}, score))

    scored.sort(key=lambda x: x[2], reverse=True)

    fallback = []
    for text, meta, score in scored[:top_k]:
        fallback.append({
            "content": text,
            "source": meta.get('source', ''),
            "kb_id": meta.get('kb_id', ''),
            "score": 0.3 + score * 0.4
        })

    return fallback


@router.post("/{agent_id}/chat")
async def chat_with_agent(
        agent_id: int,
        data: AgentChatRequest,
        http_request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
        kb_id: str = Query(None, description="知识库ID，不传则搜索全部"),
):
    logger.info(f"[chat_with_agent] kb_id={kb_id}")
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权使用该智能体")
    if not data.message or not data.message.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    import asyncio
    import json as _json
    from core.agent_langgraph import get_llm_by_model
    _, HumanMessage, AIMessage, SystemMessage = _get_langchain_classes()

    # 优先使用请求中的 model，其次使用 agent.model，最后使用系统默认模型
    raw_model = (data.model or agent.model or "").strip()
    model_name = Params.DEFAULT_CHAT_MODEL if (not raw_model or raw_model == "default") else raw_model
    logger.info(f"[chat_with_agent] 使用模型: {model_name} (原始值: {raw_model or '空'})")

    # 修复：始终使用数据库中的 system_prompt，忽略前端传的
    # 这样可以确保所有 prompt 修改都生效
    system_prompt = agent.system_prompt or ""
    logger.info(f"[chat_with_agent] 使用数据库中的 system_prompt，长度={len(system_prompt)}")
    if data.system_prompt is None or data.system_prompt == "":
        variables = _deserialize_vars(agent.variables)
        if data.variables:
            for var in variables:
                key = var.get("key")
                if key and key in data.variables:
                    val = str(data.variables[key])
                    system_prompt = system_prompt.replace(f"{{{{{key}}}}}", val)
                    system_prompt = system_prompt.replace(f"{{{key}}}", val)

    # # 优化：判断用户问题是否与智能体角色相关，不相关则跳过 RAG 检索
    # is_relevant = is_query_relevant_to_agent(data.message, agent.name, system_prompt)
    # if is_relevant:
    #     kb_content_text, _ = _build_rag_context(agent, current_user.username, data.message, log_tag="chat")
    #     logger.info(f"[chat_with_agent] 问题与智能体相关，执行 RAG 检索, kb_len={len(kb_content_text)}")
    # else:
    #     kb_content_text = ""
    #     logger.info(f"[chat_with_agent] 问题与智能体不相关，跳过 RAG 检索")

    kb_content_text, intent_type = _build_rag_context(agent, current_user.username, data.message, log_tag="chat")
    if kb_content_text:
        system_prompt = system_prompt.replace("{{knowledge_content}}", kb_content_text)
        system_prompt = system_prompt.replace("{knowledge_content}", kb_content_text)
    
    # 闲聊模式下禁用工具调用
    if intent_type == "skip":
        logger.info(f"[chat_with_agent] 闲聊模式，禁用工具调用")
        tools_config = []
    else:
        # 加载工具配置
        tools_config = json.loads(agent.tools) if agent.tools else []
    
    # 调试日志：打印最终 system_prompt 中是否包含门店地址回复规则
    if "门店地址回复规则" in system_prompt:
        logger.info(f"[chat_with_agent]  system_prompt 包含门店地址回复规则")
    else:
        logger.warning(f"[chat_with_agent]  system_prompt 不包含门店地址回复规则")
    
    # 打印 system_prompt 中知识库内容部分（前500字符）
    kb_idx = system_prompt.find("##【知识库内容与回复规则】")
    if kb_idx >= 0:
        logger.info(f"[chat_with_agent] system_prompt 知识库部分: {system_prompt[kb_idx:kb_idx+500]}")
    
    # 调试日志：打印 data.system_prompt 是否为空
    if data.system_prompt:
        logger.warning(f"[chat_with_agent] ️ 前端传了自定义 system_prompt，长度={len(data.system_prompt)}")
    else:
        logger.info(f"[chat_with_agent]  前端未传自定义 system_prompt，使用数据库中的")

    if data.history and len(data.history) > 0:
        messages = [SystemMessage(content=system_prompt)]
        for item in data.history:
            role = item.get('role', '')
            content = item.get('content', '')
            # 只保留助手回复（清理后的），不保留历史用户消息
            # 避免LLM看到历史用户消息后误以为需要处理历史问题
            if role == 'assistant':
                # 清理助手回复中的工具调用描述，避免LLM在后续对话中引用
                import re
                # 移除"问题类型判断"段落（包括其后的所有内容直到实际答案开始）
                content = re.sub(r'问题类型判断[：:\s].*?(?=\n\n|\n#|\n\*\*|$)', '', content, flags=re.DOTALL)
                # 移除"关于'xxx'"段落（引用历史问题）
                content = re.sub(r'\n\d+\.\s*关于["\']["\']+["\'].*?(?=\n\d+\.|\n\n|$)', '', content, flags=re.DOTALL)
                # 移除"正在为您调用xxx工具"等工具调用描述
                content = re.sub(r'正在为您调用.*?工具.*?(?=\n\n|$)', '', content, flags=re.DOTALL)
                content = re.sub(r'我将(分别)?调用.*?工具.*?(?=\n\n|$)', '', content, flags=re.DOTALL)
                content = re.sub(r'已识别该问题为.*?类问题.*?(?=\n\n|$)', '', content, flags=re.DOTALL)
                content = re.sub(r'工具返回结果[：:\s].*?(?=\n\n|$)', '', content, flags=re.DOTALL)
                content = re.sub(r'下一步建议[：:\s].*?(?=\n\n|$)', '', content, flags=re.DOTALL)
                # 移除"用户提出了X个问题"等元描述
                content = re.sub(r'用户提出了?\d+?个问题.*?(?=\n\n|$)', '', content, flags=re.DOTALL)
                content = re.sub(r'均不属于.*?相关问题.*?(?=\n\n|$)', '', content, flags=re.DOTALL)
                # 清理多余的空行
                content = re.sub(r'\n{3,}', '\n\n', content).strip()
                if content:
                    messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=data.message))
        logger.info(f"[chat_with_agent] 使用前端 history={len(data.history)}条, kb_len={len(kb_content_text)}")

        try:
            from fastapi.responses import StreamingResponse
            tools_config = json.loads(agent.tools) if agent.tools else []
            gen = await _make_chat_generator(model_name, messages, agent_id, data.session_id, http_request, log_tag="chat", 
                                             enable_thinking=data.deep_think, data=data, tools_config=tools_config)
            return StreamingResponse(gen, media_type="text/event-stream")
        except pybreaker.CircuitBreakerError as e:
            raise HTTPException(status_code=503, detail="大模型服务异常，请稍后重试")
        except LLMRemoteError:
            raise HTTPException(status_code=500, detail="调用大模型服务失败，请重试")

    session_key = data.session_id or f"user_{current_user.id}_agent_{agent.id}"
    history = _get_chat_history(session_key)
    _trim_history(history)
    
    # 过滤掉孤立的用户消息（没有助手回复跟进的用户消息）
    # 这会导致LLM看到多个连续的用户消息，误以为用户提出了多个问题
    if history.messages:
        filtered_messages = []
        i = 0
        while i < len(history.messages):
            msg = history.messages[i]
            # 如果是用户消息，检查下一条是否是助手回复
            if hasattr(msg, 'role') and msg.role == 'user':
                if i + 1 < len(history.messages):
                    next_msg = history.messages[i + 1]
                    # 如果下一条是助手回复，保留这对消息
                    if hasattr(next_msg, 'role') and next_msg.role == 'assistant':
                        filtered_messages.append(msg)
                        filtered_messages.append(next_msg)
                        i += 2
                        continue
                # 否则是孤立的用户消息，跳过
                logger.info(f"[chat_with_agent] 过滤掉孤立的用户消息: {msg.content[:50]}...")
                i += 1
            else:
                filtered_messages.append(msg)
                i += 1
        history.messages = filtered_messages

    if not history.messages and agent.opening and agent.opening.strip():
        history.add_ai_message(agent.opening.strip())
        logger.info(f"[chat_with_agent] 新会话，注入开场白到记忆: {agent.opening[:50]}...")

    logger.info(f"[chat_with_agent] 会话 {session_key} 历史轮数={len(history.messages) // 2}")

    # tools_config 已在上面根据 intent_type 设置，此处不再重复加载

    async def generate():
        try:
            if data.deep_think:
                logger.info(f"[chat_with_agent] 深度思考模式已开启")

            history.add_user_message(data.message)

            langchain_msgs = [SystemMessage(content=system_prompt)]
            # 过滤掉历史消息中的工具调用相关消息（tool角色和包含tool_calls的assistant消息）
            # 只保留用户和助手的纯文本对话内容，避免LLM在后续对话中引用历史工具结果
            for msg in history.messages:
                # 跳过tool角色的消息
                if hasattr(msg, 'type') and msg.type == 'tool':
                    continue
                # 跳过包含tool_calls的assistant消息
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    continue
                # 对于assistant消息，确保只添加纯文本内容
                if hasattr(msg, 'role') and msg.role == 'assistant':
                    # 创建新的AIMessage，只包含content，清除所有工具相关属性
                    if hasattr(msg, 'content') and msg.content:
                        clean_msg = AIMessage(content=msg.content)
                        langchain_msgs.append(clean_msg)
                    continue
                langchain_msgs.append(msg)

            # 如果有工具配置，使用 _make_chat_generator 支持工具调用
            if tools_config:
                logger.info(f"[chat_with_agent] 使用工具调用模式, tools={len(tools_config)}")
                gen = await _make_chat_generator(
                    model_name, langchain_msgs, agent_id, data.session_id, http_request,
                    log_tag="chat", enable_thinking=data.deep_think, data=data, tools_config=tools_config,
                    history=history  # 传入 history 对象，让 _make_chat_generator 直接保存到正确的 session_key
                )
                async for chunk in gen:
                    if await http_request.is_disconnected():
                        break
                    yield chunk
                # 对话历史已在 _make_chat_generator 内部保存（通过传入的 history 参数）
            else:
                # 没有工具配置，使用普通流式输出
                llm = get_llm_by_model(model_name, 0.3, deep_think=data.deep_think)

                full_response = ""
                try:
                    async for chunk in llm.astream(langchain_msgs):
                        if await http_request.is_disconnected():
                            break
                        text = chunk.content if hasattr(chunk, "content") else str(chunk)
                        if text:
                            full_response += text
                            yield f"data: {_json.dumps({'text': full_response, 'status': 'streaming'}, ensure_ascii=False)}\n\n"
                    yield f"data: {_json.dumps({'text': full_response, 'status': 'done'}, ensure_ascii=False)}\n\n"
                    if full_response:
                        history.add_ai_message(full_response)
                    else:
                        if history.messages and isinstance(history.messages[-1], HumanMessage):
                            history.messages.pop()
                except asyncio.CancelledError:
                    if history.messages and isinstance(history.messages[-1], HumanMessage):
                        history.messages.pop()
                    yield f"data: {_json.dumps({'text': full_response, 'status': 'stop'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            if history.messages and isinstance(history.messages[-1], HumanMessage):
                history.messages.pop()
            logger.error(f"[chat_with_agent] 智能体对话失败: {str(e)}")
            yield f"data: {_json.dumps({'text': f'智能体执行出错: {str(e)}', 'status': 'error'}, ensure_ascii=False)}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.delete("/{agent_id}/memory")
async def clear_agent_memory(
        agent_id: int,
        session_id: Optional[str] = Query(None),
        current_user: User = Depends(get_current_user),
):
    session_key = session_id or f"user_{current_user.id}_agent_{agent_id}"
    _clear_chat_history(session_key)
    return {"ok": True, "session_key": session_key}


@router.post("/{agent_id}/conversation/save")
async def save_conversation(
        agent_id: int,
        data: dict,
        db: Session = Depends(get_db),
):
    """保存对话历史到agents_conversations表（公开访问，无需认证）"""
    from sqlalchemy import text
    import json

    messages = data.get("messages", [])
    sources = data.get("sources", [])
    session_id = data.get("session_id", "")

    # 获取智能体信息
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")

    # 将 session_id 简化为 use_{agent_id} 格式
    # 例如: use_4_1790334838446 -> use_4
    simplified_session_id = f"use_{agent_id}"

    # 从 agents 表获取创建该智能体的用户名
    user_name = agent.user_name or ""
    variable_list = agent.variables or []
    if isinstance(variable_list, str):
        variable_list = json.loads(variable_list) if variable_list else []

    # 前端传递的自定义变量的值，格式: {"hospital": "北京清华长庚医院"}
    custom_variables = data.get("variables", {})

    # 更新 variable_list 中每个变量的 default 值
    for variable in variable_list:
        key = variable.get("key", "")
        for var_key, var_value in custom_variables.items():
            if var_key == key:
                variable["default"] = var_value
                break

    chat_messages_json = json.dumps(messages, ensure_ascii=False)
    sources_json = json.dumps(sources if sources else [], ensure_ascii=False)
    variables_json = json.dumps(variable_list, ensure_ascii=False)

    logger.info(f"[save_conversation] session_id={session_id} -> {simplified_session_id}, user_name={user_name}, variables_json={variables_json}")

    # 查询是否已存在该智能体的对话记录（每个智能体只保留一条记录）
    result = db.execute(
        text("SELECT id FROM agents_conversations WHERE agent_name = :agent_name AND session_id = :session_id"),
        {"agent_name": agent.name, "session_id": simplified_session_id}
    ).fetchone()

    if result:
        # 更新现有记录
        db.execute(
            text("""
                UPDATE agents_conversations 
                SET user_name = :user_name,
                    chat_messages = :chat_messages, 
                    source = :source,
                    variables = :variables,
                    last_save_at = NOW()
                WHERE agent_name = :agent_name AND session_id = :session_id
            """),
            {
                "user_name": user_name,
                "chat_messages": chat_messages_json,
                "source": sources_json,
                "variables": variables_json,
                "agent_name": agent.name,
                "session_id": simplified_session_id
            }
        )
    else:
        # 插入新记录
        db.execute(
            text("""
                INSERT INTO agents_conversations (session_id, user_name, agent_name, chat_messages, source, variables, last_save_at)
                VALUES (:session_id, :user_name, :agent_name, :chat_messages, :source, :variables, NOW())
            """),
            {
                "session_id": simplified_session_id,
                "user_name": user_name,
                "agent_name": agent.name,
                "chat_messages": chat_messages_json,
                "source": sources_json,
                "variables": variables_json
            }
        )

    db.commit()
    return {"ok": True}


@router.get("/{agent_id}/conversation/load")
async def load_conversation(
        agent_id: int,
        session_id: str = Query(...),
        db: Session = Depends(get_db),
):
    """加载对话历史（公开访问，无需认证）"""
    from sqlalchemy import text
    import json

    # 获取智能体信息
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")

    # 将 session_id 简化为 use_{agent_id} 格式
    simplified_session_id = f"use_{agent_id}"

    # 查询该智能体的对话记录
    result = db.execute(
        text(
            "SELECT chat_messages, source, variables FROM agents_conversations WHERE agent_name = :agent_name AND session_id = :session_id"),
        {"agent_name": agent.name, "session_id": simplified_session_id}
    ).fetchone()

    if not result:
        return {"messages": [], "sources": [], "variables": {}}

    messages = json.loads(result[0]) if result[0] else []
    sources = json.loads(result[1]) if result[1] else []
    variables = json.loads(result[2]) if result[2] else []

    # 将 variables 列表转换为前端需要的格式: {"hospital": "北京清华长庚医院"}
    variables_dict = {}
    for v in variables:
        key = v.get("key", "")
        default = v.get("default", "")
        if key:
            variables_dict[key] = default

    return {
        "opening": agent.opening,
        "messages": messages,
        "sources": sources,
        "variables": variables_dict
    }


@router.delete("/{agent_id}/conversation/delete")
async def delete_conversation(
        agent_id: int,
        session_id: str = Query(...),
        db: Session = Depends(get_db),
):
    """删除对话（公开访问，无需认证）"""
    from sqlalchemy import text

    # 获取智能体信息
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")

    # 将 session_id 简化为 use_{agent_id} 格式
    simplified_session_id = f"use_{agent_id}"

    db.execute(
        text("DELETE FROM agents_conversations WHERE agent_name = :agent_name AND session_id = :session_id"),
        {"agent_name": agent.name, "session_id": simplified_session_id}
    )

    db.commit()
    return {"ok": True}


class PublishRequest(BaseModel):
    """发布智能体的请求体"""
    version: str
    publish_desc: Optional[str] = None
    category: Optional[str] = None
    channels: List[str] = []


@router.post("/{agent_id}/publish")
async def publish_agent(
    agent_id: int,
    request: PublishRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """发布智能体到指定渠道"""
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_name == current_user.username).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在或无权操作")

    # 将渠道ID映射为平台名称，取第一个作为主要发布平台
    platform_name = CHANNEL_NAME_MAP.get(request.channels[0], request.channels[0]) if request.channels else OFFICIAL_CHANNEL_NAME
    
    # 更新智能体发布状态（publish_channels 存储纯字符串）
    agent.is_published = True
    agent.publish_version = request.version
    agent.publish_desc = request.publish_desc
    agent.category = request.category
    agent.publish_channels = platform_name
    agent.updated_at = datetime.now()

    # 记录发布历史
    history = PublishHistory(
        agent_id=agent_id,
        user_name=current_user.username,
        version=request.version,
        description=request.publish_desc,
        channels=platform_name,
        published_at=datetime.now()
    )
    db.add(history)
    db.commit()
    db.refresh(agent)

    return {
        "ok": True,
        "id": agent.id,
        "is_published": True,
        "publish_version": agent.publish_version,
        "publish_channels": agent.publish_channels,
    }


@router.get("/{agent_id}/publish-history")
async def get_publish_history(
    agent_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取智能体的发布历史记录"""
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_name == current_user.username).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在或无权操作")

    histories = db.query(PublishHistory).filter(
        PublishHistory.agent_id == agent_id,
        PublishHistory.user_name == current_user.username
    ).order_by(PublishHistory.published_at.desc()).all()

    return {
        "ok": True,
        "history": [
            {
                "id": h.id,
                "version": h.version,
                "description": h.description,
                "channels": h.channels,
                "published_at": h.published_at.strftime("%Y-%m-%d %H:%M:%S") if h.published_at else None
            }
            for h in histories
        ]
    }