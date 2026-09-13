"""
LangGraph 智能体框架实现 - 支持快速流式响应

基于 agent.py-bak 中的框架代码，所有智能体节点都使用 LangGraph 框架实现，
并保持 LLM 快速流式响应输出到前端展示的能力。

核心特性：
1. 使用 LangGraph StateGraph 构建智能体图
2. 所有节点支持流式输出（通过 _stream_by_prompt_async）
3. 保留反思机制进行质量评估
4. 支持前端实时看到 LLM 生成的内容
"""
from __future__ import annotations
import os
import json
import asyncio
import time as _time
import re
import uuid
from typing import Annotated, TypedDict
from operator import add
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import END, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from settings.Define import Params
from settings.logger_manager import get_logger
from langgraph.config import get_stream_writer
from langgraph.types import StreamWriter
from datetime import datetime

logger = get_logger(__name__)

# 懒加载重型库
_ChatOpenAI = None
_OpenAI = None
_HumanMessage = None
_AIMessage = None
_SystemMessage = None
_ToolMessage = None

# 懒加载全局变量
_openai_client = None
_http_client = None
_llm = None
_doc_tools = None
_llm_with_tools = None

_SESSION_MEMORIES: dict = {}
_SESSION_CONTEXTS: dict = {}
_MAX_SESSIONS = 200
_WINDOW_SIZE = 5



load_dotenv()

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None

class AgentState(TypedDict):
    """LangGraph 智能体状态定义"""
    messages: Annotated[list, add]
    type: str
    file_path: str
    file_paths: list[str]
    previous_node: str
    skill: str
    sources: str
    user: str
    images: list
    project_context: dict
    thread_id: str


class SessionMemoryManager:
    """持久化会话记忆管理器（使用数据库存储，服务重启后数据不丢失）。

    每个 thread_id 对应数据库中的一条记录，
    chat_msg 字段存储整个对话历史的 JSON 数组，
    context 字段存储元信息（upload_files, output_file 等）。
    """

    @staticmethod
    def get_db_session():
        """获取数据库会话"""
        from core.auth.database import SessionLocal
        return SessionLocal()

    @staticmethod
    def get_or_create(thread_id):
        """从数据库加载历史消息"""
        if not thread_id:
            return []

        db = SessionMemoryManager.get_db_session()
        try:
            from core.auth.database import SessionMemory
            # 查询该 thread_id 的记录
            memory = db.query(SessionMemory).filter(
                SessionMemory.thread_id == int(thread_id)
            ).first()

            if not memory or not memory.chat_msg:
                return []

            # chat_msg 已经是 Python 列表（SQLAlchemy 自动反序列化 JSON）
            chat_history = memory.chat_msg if isinstance(memory.chat_msg, list) else []

            # 转换为 LangChain 消息格式
            history = []
            _, HumanMessage, AIMessage, _, _ = _get_langchain_models()
            for msg in chat_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        history.append(HumanMessage(content=content))
                    elif role == "assistant":
                        history.append(AIMessage(content=content))

            # 只保留最近 10 条（5 轮对话）
            if len(history) > _WINDOW_SIZE * 2:
                history = history[-_WINDOW_SIZE * 2:]
                # 同时更新数据库中的 chat_msg
                memory.chat_msg = chat_history[-_WINDOW_SIZE * 2:]
                memory.update_at = datetime.now()
                db.commit()

            return history
        finally:
            db.close()

    @staticmethod
    def save(thread_id, user_msg: str, assistant_msg: str):
        """保存一轮对话到数据库"""
        if not thread_id or not user_msg:
            return

        db = SessionMemoryManager.get_db_session()
        try:
            from core.auth.database import SessionMemory
            import json

            thread_id_int = int(thread_id)
            # 查找或创建记录
            memory = db.query(SessionMemory).filter(
                SessionMemory.thread_id == thread_id_int
            ).first()

            if memory:
                # 更新已有记录
                # 创建新列表副本，避免 SQLAlchemy 变更检测问题
                chat_history = list(memory.chat_msg) if isinstance(memory.chat_msg, list) else []
                chat_history.append({"role": "user", "content": user_msg})
                chat_history.append({"role": "assistant", "content": assistant_msg or ""})

                # 只保留最近 10 条（5 轮对话）
                if len(chat_history) > _WINDOW_SIZE * 2:
                    chat_history = chat_history[-_WINDOW_SIZE * 2:]

                memory.chat_msg = chat_history
                memory.update_at = datetime.now()

                print(f"memory: {memory}")
            else:
                # 创建新记录
                new_memory = SessionMemory(
                    thread_id=thread_id_int,
                    chat_msg=[
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg or ""}
                    ],
                    context=None,
                    update_at=datetime.now(),
                )
                db.add(new_memory)

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"[Memory] 保存会话失败 thread_id={thread_id}: {e}")
        finally:
            db.close()

    @staticmethod
    def load(thread_id) -> list:
        """加载历史消息"""
        return SessionMemoryManager.get_or_create(thread_id)

    @staticmethod
    def set_context(thread_id, **kwargs):
        """保存会话元信息（如 upload_files, output_file 等）"""
        if not thread_id:
            return

        db = SessionMemoryManager.get_db_session()
        try:
            from core.auth.database import SessionMemory

            thread_id_int = int(thread_id)
            memory = db.query(SessionMemory).filter(
                SessionMemory.thread_id == thread_id_int
            ).first()

            if memory:
                # 更新 context
                # 关键修复：创建新字典副本，避免 SQLAlchemy 变更检测问题
                ctx = dict(memory.context) if isinstance(memory.context, dict) else {}
                ctx.update({k: v for k, v in kwargs.items() if v is not None})
                memory.context = ctx
                memory.update_at = datetime.now()
                db.commit()
            else:
                # 创建新记录（只有 context，没有对话）
                new_memory = SessionMemory(
                    thread_id=thread_id_int,
                    chat_msg=[],
                    context={k: v for k, v in kwargs.items() if v is not None},
                )
                db.add(new_memory)
                db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"[Memory] 保存上下文失败 thread_id={thread_id}: {e}")
        finally:
            db.close()

    @staticmethod
    def get_context(thread_id, key=None):
        """读取会话元信息。key=None 时返回整个 dict"""
        if not thread_id:
            return {} if key is None else None

        db = SessionMemoryManager.get_db_session()
        try:
            from core.auth.database import SessionMemory

            memory = db.query(SessionMemory).filter(
                SessionMemory.thread_id == int(thread_id)
            ).first()

            if memory and memory.context:
                ctx = memory.context if isinstance(memory.context, dict) else {}
                if key is None:
                    return ctx
                return ctx.get(key)
            return {} if key is None else None
        finally:
            db.close()

    @staticmethod
    def clear(thread_id: str):
        """清除会话记忆"""
        if not thread_id:
            return

        db = SessionMemoryManager.get_db_session()
        try:
            from core.auth.database import SessionMemory
            db.query(SessionMemory).filter(
                SessionMemory.thread_id == thread_id
            ).delete()
            db.commit()
            logger.info(f"[Memory] 清除会话 thread_id={thread_id}")
        except Exception as e:
            db.rollback()
            logger.error(f"[Memory] 清除会话失败 thread_id={thread_id}: {e}")
        finally:
            db.close()

    @staticmethod
    def has_memory(thread_id) -> bool:
        """检查会话是否有记忆"""
        if not thread_id:
            return False

        db = SessionMemoryManager.get_db_session()
        try:
            from core.auth.database import SessionMemory
            count = db.query(SessionMemory).filter(
                SessionMemory.thread_id == int(thread_id)
            ).count()
            return count > 0
        finally:
            db.close()

    @staticmethod
    def session_count() -> int:
        """获取总会话数"""
        db = SessionMemoryManager.get_db_session()
        try:
            from core.auth.database import SessionMemory
            count = db.query(SessionMemory).count()
            return count or 0
        finally:
            db.close()

def get_openai_client():
    """懒加载 OpenAI 客户端"""
    return _get_openai_client()

def _get_openai_client():
    """懒加载原生 OpenAI 客户端实例"""
    global _OpenAI
    if _OpenAI is None:
        _t0 = _time.time()
        from openai import OpenAI as _O
        _OpenAI = _O(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
        )
        logger.info(f"[PERF] openai.OpenAI 初始化耗时: {_time.time()-_t0:.3f}s")
    return _OpenAI

def _get_langchain_messages_full():
    """只加载消息类，不加载重型聊天模型，用于文档处理等场景"""
    global _HumanMessage, _AIMessage, _SystemMessage, _ToolMessage
    if _HumanMessage is None:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        _HumanMessage = HumanMessage
        _AIMessage = AIMessage
        _SystemMessage = SystemMessage
        _ToolMessage = ToolMessage
    return _HumanMessage, _AIMessage, _SystemMessage, _ToolMessage

def get_http_client():
    """懒加载 HTTP 客户端"""
    global _http_client
    if _http_client is None:
        import httpx
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers={"Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY', '')}"},
        )
    return _http_client

def _get_langchain_models():
    global _ChatOpenAI, _HumanMessage, _AIMessage, _SystemMessage, _ToolMessage
    if _ChatOpenAI is None:
        from openai import OpenAI
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        _ChatOpenAI = OpenAI
        _HumanMessage = HumanMessage
        _AIMessage = AIMessage
        _SystemMessage = SystemMessage
        _ToolMessage = ToolMessage
    return _ChatOpenAI, _HumanMessage, _AIMessage, _SystemMessage, _ToolMessage

def get_llm():
    """懒加载 LLM 客户端"""
    global _llm
    if _llm is None:
        OpenAI, _, _, _, _ = _get_langchain_models()
        _llm = OpenAI(
            model=Params.DEFAULT_CHAT_MODEL,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
            timeout=60,
            streaming=True
        )
    return _llm

def get_llm_by_model(model_id: str, temperature: float = 0.0, deep_think: bool = False):
    """根据前端选择的模型ID创建对应的LLM实例"""
    model_id = model_id.lower()
    logger.info(f"根据模型ID创建LLM实例: {model_id}, deep_think={deep_think}")

    # 懒加载 ChatOpenAI
    ChatOpenAI, _, _, _, _ = _get_langchain_models()

    # "return_reasoning": True  # 建议同时开启，强制返回完整的推理链内容
    thinking_kwargs =  {"enable_thinking": True, "return_reasoning": True} if deep_think else {}

    if model_id == "deepseek" or model_id == "deepseek-v4-pro":
        model_name = "deepseek-reasoner" if deep_think else "deepseek-v4-pro"
        base = dict(
            model=model_name,
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com/v1",
            timeout=120,
            streaming=True,
            extra_body=thinking_kwargs
        )
        if temperature is not None and temperature != 0.0:
            base["temperature"] = temperature
        return ChatOpenAI(**base)

    elif model_id == "glm" or model_id == "glm-5.2-fast-preview":
        model_name = "glm-4-plus" if deep_think else "glm-5.2-fast-preview"
        base = dict(
            model=model_name,
            api_key=os.getenv("GLM_API_KEY", ""),
            base_url="https://open.bigmodel.cn/api/paas/v4",
            timeout=120,
            streaming=True,
            extra_body=thinking_kwargs
        )
        if temperature is not None and temperature != 0.0:
            base["temperature"] = temperature
        return ChatOpenAI(**base)

    elif model_id == "qwen_vl" or model_id == "qwen-vl-max":
        model_name = "qwen-plus" if deep_think else "qwen-vl-max"
        base = dict(
            model=model_name,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
            timeout=120,
            extra_body=thinking_kwargs
        )
        if temperature is not None and temperature != 0.0:
            base["temperature"] = temperature
        return ChatOpenAI(**base)

    elif model_id == "qwen_turbo":
        model_name = Params.DEFAULT_TEXT_TOOL_MODEL
        base = dict(
            model=model_name,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
            timeout=120,
            extra_body=thinking_kwargs
        )
        if temperature is not None and temperature != 0.0:
            base["temperature"] = temperature
        return ChatOpenAI(**base)

    else:
        logger.warning(f"未知模型ID: {model_id}，使用默认模型 {Params.DEFAULT_CHAT_MODEL}")
        model_name = Params.DEFAULT_CHAT_MODEL
        base = dict(
            model=model_name,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
            timeout=120,
            streaming=True,
            extra_body=thinking_kwargs
        )
        if temperature is not None and temperature != 0.0:
            base["temperature"] = temperature
        return ChatOpenAI(**base)

def get_llm_with_tools():
    """懒加载原生 OpenAI 客户端（用于 tool calling）"""
    global _llm_with_tools
    if _llm_with_tools is None:
        _t_start = _time.time()
        _llm_with_tools = _get_openai_client()
        logger.info(f"[PERF] get_llm_with_tools: 总耗时 {_time.time()-_t_start:.3f}s")
    return _llm_with_tools

# 向后兼容：使用模块级别的 __getattr__ 实现懒加载
def __getattr__(name, thread_id:str=""):
    if name == "openai_client":
        return get_openai_client()
    elif name == "http_client":
        return get_http_client()
    elif name == "llm":
        return get_llm()
    elif name == "doc_tools":
        from core.tools.doc_tools import create_doc_tools
        global _doc_tools
        if _doc_tools is None:
            _doc_tools = create_doc_tools(thread_id=thread_id)
        return _doc_tools
    elif name == "llm_with_tools":
        return get_llm_with_tools()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



def _build_prompt_list(messages_list, system_prompt: str, user_query: str, thread_id=None) -> list:
    """通用：构建带 system + 历史对话 + 用户消息的 prompt_list

    优先从 SessionMemoryManager 读取历史，自动维护滑动窗口（最近 5 轮对话）。
    无 memory 时 fallback 到前端传来的 messages_list。
    """
    conversation_history = []

    if thread_id and SessionMemoryManager.has_memory(thread_id):
        conversation_history = SessionMemoryManager.load(thread_id)
        logger.info(f"[Memory] thread_id={thread_id} 注入历史消息 {len(conversation_history)} 条")
    else:
        for msg in messages_list[:-1] if len(messages_list) > 1 else []:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                conversation_history.append({"role": "assistant", "content": content})
            elif isinstance(msg, dict) and msg.get("role") == "user":
                conversation_history.append({"role": "user", "content": content})
            else:
                conversation_history.append({"role": "user", "content": content})
        conversation_history = conversation_history[-5:]

    return (
            [{"role": "system", "content": system_prompt}]
            + conversation_history
            + [{"role": "user", "content": user_query}]
    )

def _to_openai_messages(prompt_list) -> list:
    """把各种格式的 prompt 统一转成原生 OpenAI SDK 需要的 [{role, content}] dict 列表

    正确处理 LangChain 的 ToolMessage（role=tool, 带 tool_call_id）
    和带 tool_calls 的 AIMessage（role=assistant, content=None, 带 tool_calls 字段）
    """
    result = []
    for item in prompt_list:
        if isinstance(item, dict):
            result.append(item)
            continue
        if hasattr(item, 'type'):
            msg_type = item.type
            content = item.content if hasattr(item, 'content') else str(item)

            if msg_type == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": getattr(item, "tool_call_id", ""),
                    "content": content,
                })
            elif msg_type == "ai" or msg_type == "assistant":
                entry = {"role": "assistant", "content": content}
                tool_calls = getattr(item, "tool_calls", None)
                if tool_calls:
                    entry["tool_calls"] = tool_calls
                result.append(entry)
            elif msg_type == "system":
                result.append({"role": "system", "content": content})
            elif msg_type in ("human", "user"):
                result.append({"role": "user", "content": content})
            else:
                role_map = {"human": "user", "ai": "assistant", "assistant": "assistant", "user": "user"}
                role = role_map.get(msg_type, "user")
                result.append({"role": role, "content": content})
        else:
            result.append({"role": "user", "content": str(item)})
    return result



async def _stream_by_prompt_async(prompt_list, model: str = None, temperature: float = 0.3):
    """异步流式 yield chunk 文本（使用全局 httpx.AsyncClient 连接池）"""
    _t0 = _time.time()
    openai_msgs = _to_openai_messages(prompt_list)
    logger.info(f"[PERF] _stream_by_prompt: 转换 msgs 耗时 {_time.time()-_t0:.3f}s, 消息数={len(openai_msgs)}")

    use_model = model or Params.DEFAULT_CHAT_MODEL
    payload = {
        "model": use_model,
        "messages": openai_msgs,
        "temperature": temperature,
        "stream": True,
        "enable_thinking": False,
    }
    url = f"{Params.API_BASE.rstrip('/')}/chat/completions"

    _t1 = _time.time()
    async with get_http_client().stream("POST", url, json=payload) as response:
        logger.info(f"[PERF] _stream_by_prompt: HTTP 连接建立耗时 {_time.time()-_t1:.3f}s, status={response.status_code}, model={use_model}")
        response.raise_for_status()

        _t2 = _time.time()
        _chunk_idx = 0
        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                _chunk_idx += 1
                if _chunk_idx == 1:
                    logger.info(f"[PERF] _stream_by_prompt: 首字到达耗时 {_time.time()-_t2:.3f}s, 首字={repr(content[:20])}")
                yield content
    logger.info(f"[PERF] _stream_by_prompt: 总耗时 {_time.time()-_t0:.3f}s, 有效 chunk 数={_chunk_idx}")

async def _async_iter_sync_generator(sync_generator):
    """将同步生成器包装为异步迭代器，使用哨兵值避免 StopIteration 异常"""
    loop = asyncio.get_event_loop()
    iterator = iter(sync_generator)
    _sentinel = object()
    while True:
        #try:
        chunk = await loop.run_in_executor(None, next, iterator, _sentinel)
        if chunk is _sentinel:
            return
        yield chunk
        # except StopIteration :
        #     return


def _clean_messages_for_chat_reply(messages) -> list:
    """把带 tool_calls / ToolMessage 的 LangChain 消息列表，
    转换成只含 system/user/assistant 的干净聊天历史，
    供不带 tools 的普通 LLM 做最终自然语言回复。

    策略：
    - AIMessage.tool_calls 转成 "[调用了工具 name(args)]" 自然语言
    - ToolMessage.content 转成 "[工具返回：xxx]"，追加到最近一条 assistant
    - 最后加 system 提示"请用自然聊天语气回复用户"
    """
    result = []

    def _append_to_last_assistant(text: str):
        if result and result[-1]["role"] == "assistant":
            result[-1]["content"] = (result[-1]["content"] + " " + text).strip()
        else:
            result.append({"role": "assistant", "content": text})

    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "tool":
                content = _strip_download_url_if_no_modification(content)
                _append_to_last_assistant(f"[工具返回：{content}]")
            elif role == "assistant":
                text = content or ""
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    parts = []
                    for tc in tool_calls:
                        name = tc.get("name", "")
                        args_str = ", ".join(f"{k}={v}" for k, v in tc.get("args", {}).items())
                        parts.append(f"[调用了工具 {name}({args_str})]")
                    text = (text + " " + " ".join(parts)).strip()
                if text:
                    result.append({"role": "assistant", "content": text})
            elif role in ("system", "user"):
                result.append({"role": role, "content": content})
            continue

        if not hasattr(msg, 'type'):
            continue

        t = msg.type
        content = msg.content if hasattr(msg, 'content') else str(msg)

        if t == "tool":
            content = _strip_download_url_if_no_modification(content)
            _append_to_last_assistant(f"[工具返回：{content}]")
        elif t in ("ai", "assistant"):
            text = content or ""
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                parts = []
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args_str = ", ".join(f"{k}={v}" for k, v in tc.get("args", {}).items())
                    parts.append(f"[调用了工具 {name}({args_str})]")
                text = (text + " " + " ".join(parts)).strip()
            if text:
                result.append({"role": "assistant", "content": text})
        elif t == "system":
            result.append({"role": "system", "content": content})
        elif t in ("human", "user"):
            result.append({"role": "user", "content": content})

    result.append({
        "role": "system",
        "content": (
            "（现在请你根据以上对话和工具执行结果，用自然、友好的聊天语气回复用户，"
            "总结执行结果并告诉用户，不要重复工具调用过程。"
            "重要规则：\n"
            "1. 如果工具返回内容里提到「没有发现错别字」「无需修改」「内容正确」等，"
            "不要编造下载链接，直接告诉用户检查完毕即可\n"
            "2. 下载链接只在工具返回里明确给了 [下载文档](url) 时才提，否则不要自行生成\n"
            "3. 用户要求的格式转换/文档整理这类有实际改动的，正常告知结果"
        )
    })
    return result


def _strip_download_url_if_no_modification(content: str) -> str:
    """如果内容里包含「没有发现错别字/无需修改」等语义，移除其中的下载链接。

    规则：
    - 检测 typo 无发现关键词 → 移除 [下载文档](url) markdown 链接
    - 移除裸的 http(s)://... 下载 URL
    """
    _typo_no_keywords = [
        "没有发现", "未发现", "没有错别字", "无错别字", "没有错字",
        "无错字", "无需修改", "不需要修改", "内容正确", "没有问题",
        "检查完毕", "一切正常", "全部正确", "没有拼写",
    ]
    if not any(kw in content for kw in _typo_no_keywords):
        return content

    import re as _re
    content = _re.sub(r'\[下载文档\]\((https?://[^)]+)\)', '', content)
    content = _re.sub(r'https?://[^\s)\]>,，。！？]+', '', content)
    content = _re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


def _create_streaming_node(skill_type: str):
    """创建一个支持流式输出的 LangGraph 节点工厂函数
    
    每个节点都使用 _stream_by_prompt_async() 实现快速流式响应，
    而不是使用同步的 llm.invoke()。
    """
    async def streaming_node_func(state: AgentState, writer: StreamWriter = None):
        logger.info(f"{skill_type}_node 开始流式处理")
        
        messages_list = state.get("messages", [])
        user_query = ""
        if messages_list:
            last = messages_list[-1]
            user_query = last.content if hasattr(last, 'content') else str(last)
        
        thread_id = state.get("thread_id")
        file_path = state.get("file_path")
        file_paths = state.get("file_paths") or ([file_path] if file_path else [])
        images_data = state.get("images")
        current_user = state.get("user", "")
        project_context = state.get("project_context")
        
        full_response = ""
        final_sources = []
        
        if skill_type == "problem_analysis":
            from core.skills.ProblemAnalyze.issue_analyze import IssueAnalyze
            analyzer = IssueAnalyze()
            async for chunk in _async_iter_sync_generator(analyzer.stream_issue(user_query, images_data)):
                full_response += chunk
                if chunk:
                    writer({"text": chunk, "sources": [], "status": "streaming"})
            return {
                "messages": [AIMessage(content=full_response)],
                "type": "problem_analysis",
                "previous_node": "problem_analysis_node",
                "sources": [],
            }
        
        elif skill_type == "other":
            system_prompt = """你是一个专业的智能回答助手，需要根据用户问题来做出相应回应，回答要专业、清晰，使用自然友好的语言。

            # 自我介绍
            当用户问到以下这类问题时，请按照要求回答：

            (1)小智是谁
            - 小智是由 **MiviAgent 团队** 开发的智能助手平台
            - 开发日期：**2026 年 9 月**
            - 小智是一个**智能体创建与管理平台**，用户可以自主创建属于自己的智能体

            (2)小智能做什么
            小智的核心能力：

            1. **自主创建智能体**：用户可以根据自己的需求，创建专属的智能体，定义智能体的名称、角色、技能和行为
            2. **智能体管理**：编辑、配置、管理已创建的智能体
            3. **知识库管理**：为智能体配置专属知识库，提升回答准确性
            4. **内置技能**：
               - 文档处理：阅读、整理、合并、转换各类文档（PPT、Word、Excel、PDF、TXT 等）
               - 图纸识别：识别机械工程图纸，进行尺寸校核、公差分析、国标检索
               - 代码生成：根据需求生成编程代码
               - 旅行规划：生成旅游路线规划
               - 问题解答：回答各类知识性问题
               - 趣味互动：讲笑话、对对联等

            ## 注意事项
            1. 回答要简洁明了，不要过度冗长
            2. 使用自然友好的语气，像朋友一样交流
            3. 如果用户的问题超出你的能力范围，请礼貌告知并建议其他解决方案
            4. 不要编造不存在的信息
            """
            prompt_list = _build_prompt_list(messages_list, system_prompt, user_query, thread_id)
            
            async for chunk_text in _stream_by_prompt_async(prompt_list):
                full_response += chunk_text
                if chunk_text:
                    writer({"text": chunk_text, "sources": [], "status": "streaming"})

            if not full_response:
                full_response = "抱歉，您咨询的问题不在我的能力范围内。"
            
            return {
                "messages": [AIMessage(content=full_response)],
                "type": "other",
                "previous_node": "other_node",
                "sources": [],
            }
        
        elif skill_type == "travel":
            system_prompt = """你是一个专业的旅行规划助手，根据用户的问题，生成一个旅游路线规划。请用中文回答，内容详实但不超过300字。"""
            prompt_list = _build_prompt_list(messages_list, system_prompt, user_query, thread_id=thread_id)
            
            try:
                if MultiServerMCPClient:
                    from langchain_classic.agents import create_react_agent
                    amap_api_key = os.getenv("AMAP_API_KEY")
                    client = MultiServerMCPClient(
                        {"amap-maps-sse": {"url": f"https://mcp.amap.com/sse?key={amap_api_key}",
                                           "transport": "streamable_http"}}
                    )

                    tools = await client.get_tools()
                    agent = create_react_agent(llm=get_llm(), tools=tools)

                    # 将字典格式的消息转换为 LangChain 消息对象
                    _get_langchain_messages_full()
                    langchain_messages = []
                    for msg in prompt_list:
                        if isinstance(msg, dict):
                            role = msg.get("role", "user")
                            content_msg = msg.get("content", "")
                            if role == "system":
                                langchain_messages.append(_SystemMessage(content=content_msg))
                            elif role == "assistant":
                                langchain_messages.append(_AIMessage(content=content_msg))
                            else:
                                langchain_messages.append(_HumanMessage(content=content_msg))
                        else:
                            langchain_messages.append(msg)

                    async for chunk in agent.astream({"messages": langchain_messages}):
                        if hasattr(chunk, 'content'):
                            full_response += chunk.content
                            if chunk.content:
                                writer({"text": chunk.content, "sources": [], "status": "streaming"})
                        elif isinstance(chunk, dict) and 'content' in chunk:
                            full_response += chunk['content']
                            if chunk['content']:
                                writer({"text": chunk['content'], "sources": [], "status": "streaming"})
                else:
                    raise ImportError("MultiServerMCPClient not available")
            except Exception as e:
                logger.warning(f"MCP 服务不可用，使用本地规划: {str(e)[:50]}")
                async for chunk_text in _stream_by_prompt_async(prompt_list):
                    full_response += chunk_text
                    if chunk_text:
                        writer({"text": chunk_text, "sources": [], "status": "streaming"})

            return {
                "messages": [AIMessage(content=full_response)],
                "type": "travel",
                "previous_node": "travel_node",
                "sources": [],
            }
        
        elif skill_type == "joke":
            system_prompt = """你是一个专业的笑话大师，根据用户的问题，写一个有趣的中文笑话。不超过150字。"""
            prompt_list = _build_prompt_list(messages_list, system_prompt, user_query, thread_id=thread_id)
            
            async for chunk_text in _stream_by_prompt_async(prompt_list, model=Params.DEFAULT_TEXT_TOOL_MODEL):
                full_response += chunk_text
                if chunk_text:
                    writer({"text": chunk_text, "sources": [], "status": "streaming"})

            return {
                "messages": [AIMessage(content=full_response)],
                "type": "joke",
                "previous_node": "joke_node",
                "sources": [],
            }
        
        elif skill_type == "couplet":
            system_prompt = """你是一个专业的对联大师，精通中国古典文学和对联艺术。请根据用户的要求创作对联。
            要求：
            1. 上下联字数相等
            2. 上下联词性相对（名词对名词，动词对动词，形容词对形容词等）
            3. 上下联平仄相对（仄起平收）
            4. 上下联意境相关或相对
            5. 如果用户给出上联，请对出下联；如果用户给出下联，请对出上联；如果用户要求整副对联，则自行创作上下联
            6. 直接给出对联，不需要解释或说明"""
            
            prompt_list = _build_prompt_list(messages_list, system_prompt, user_query, thread_id=thread_id)
            
            async for chunk_text in _stream_by_prompt_async(prompt_list, model=Params.DEFAULT_TEXT_TOOL_MODEL):
                full_response += chunk_text
                if chunk_text:
                    writer({"text": chunk_text, "sources": [], "status": "streaming"})
            
            return {
                "messages": [AIMessage(content=full_response)],
                "type": "couplet",
                "previous_node": "couplet_node",
                "sources": [],
            }
        
        elif skill_type == "document":
            file_paths = state.get("file_paths") or (
                [state.get("file_path")] if state.get("file_path") else [])
            file_paths = [p for p in file_paths if p]

            # 如果当前轮没有文件，但历史对话中可能有已生成的文档，尝试从历史消息中提取
            if not file_paths and thread_id and SessionMemoryManager.has_memory(thread_id):
                # 尝试从 context 中恢复 upload_files
                context = SessionMemoryManager.get_context(thread_id)
                if context and "upload_files" in context:
                    upload_files = context["upload_files"]
                    if isinstance(upload_files, list) and upload_files:
                        file_paths = [f.get("file_path") for f in upload_files if f.get("file_path")]
                        logger.info(f"[Memory] 从 context 中恢复文件路径: {file_paths}")

                # 如果还是没有，尝试从历史消息中查找包含文档路径的信息
                if not file_paths:
                    history_msgs = SessionMemoryManager.load(thread_id)
                    for hist_msg in reversed(history_msgs):  # 从最新的消息往前找
                        # 兼容 LangChain 消息对象和字典
                        if hasattr(hist_msg, 'content'):
                            content = hist_msg.content
                        elif isinstance(hist_msg, dict):
                            content = hist_msg.get("content", "")
                        else:
                            continue

                        if isinstance(content, str):
                            # 查找包含 "生成文档：" 或 "已保存到：" 的消息
                            if "生成文档：" in content or "已保存到：" in content:
                                import re as _re_memory
                                # 匹配 "生成文档：D:\...\xxx.docx" 或 "已保存到：D:\...\xxx.docx"
                                doc_pattern = r'(?:生成文档：|已保存到：)([^\n\r]+?\.(?:docx|pdf|xlsx|pptx|txt|md|csv))'
                                matches = _re_memory.findall(doc_pattern, content)
                                if matches:
                                    file_paths = [match.strip() for match in matches]
                                    logger.info(f"[Memory] 从历史消息中恢复文档路径: {file_paths}")
                                    break

            # 如果当前轮没有文件，但历史对话中可能有已生成的文档，尝试从历史消息中提取
            if not file_paths and thread_id and SessionMemoryManager.has_memory(thread_id):
                history_msgs = SessionMemoryManager.load(thread_id)
                # 从历史消息中查找包含文档路径的信息
                for hist_msg in reversed(history_msgs):  # 从最新的消息往前找
                    # 兼容 LangChain 消息对象和字典
                    if hasattr(hist_msg, 'content'):
                        content = hist_msg.content
                    elif isinstance(hist_msg, dict):
                        content = hist_msg.get("content", "")
                    else:
                        continue

                    if isinstance(content, str):
                        # 查找包含 "生成文档：" 或 "已保存到：" 的消息
                        if "生成文档：" in content or "已保存到：" in content:
                            import re as _re_memory
                            # 匹配 "生成文档：D:\...\xxx.docx" 或 "已保存到：D:\...\xxx.docx"
                            doc_pattern = r'(?:生成文档：|已保存到：)([^\n\r]+?\.(?:docx|pdf|xlsx|pptx|txt|md|csv))'
                            matches = _re_memory.findall(doc_pattern, content)
                            if matches:
                                file_paths = [match.strip() for match in matches]
                                logger.info(f"[Memory] 从历史消息中恢复文档路径: {file_paths}")
                                break

            logger.info(f" [stream_skill] file_paths={file_paths}")
            _t_import = _time.time()
            from core.tools.doc_tools import get_doc_system_prompt, create_doc_tools
            logger.info(f"[PERF] doc_tools import 耗时: {_time.time() - _t_import:.3f}s")

            _t_models = _time.time()
            logger.info(f"[PERF] _get_langchain_messages_full 耗时: {_time.time() - _t_models:.3f}s")

            _tool_tasks = []

            try:
                clean_user_request = user_query
                if '[用户要求]' in user_query:
                    clean_user_request = user_query.split('[用户要求]')[-1].strip()
                for meta_tag in ['[原始文件格式]', '[文档内容]', '[原始文件]']:
                    if meta_tag in clean_user_request:
                        clean_user_request = clean_user_request.split(meta_tag)[0].strip()
                # 用 LLM 做语义推理：判断用户是否在引用"上一轮生成的文档"
                # 只有在已有对话历史（存在 assistant 消息）时才检查
                _is_referring_generated = False
                _has_previous_turn = any(
                    (hasattr(m, 'type') and m.type in ("ai", "assistant")) or
                    (isinstance(m, dict) and m.get("role") in ("assistant", "ai"))
                    for m in messages_list[:-1]
                )

                print(f"len(messages_list)={len(messages_list)}, _has_previous_turn={_has_previous_turn}")
                if _has_previous_turn:
                    _intent_prompt = (
                        '你是一个意图判断助手。请判断用户的最新消息是否在引用**上一轮对话中生成的文档**。\n'
                        '用户可能的表达方式包括但不限于：\n'
                        '- 明确提到"生成的文档"、"刚才生成的"、"合并后的文档"、"这个文档"等\n'
                        '- 说"对于刚才那个"、"把刚才的"、"刚才那个文件"等指代上一轮输出的内容\n'
                        '- 说"优化下格式"、"调整下排版"、"再改改"等，且上下文中有上一轮生成文档的历史\n'
                        '\n'
                        '请只回答 YES 或 NO，不要解释。\n'
                        '\n'
                        '对话历史（最近 3 条）：\n'
                    )
                    # 取最近 3 条消息作为上下文
                    _recent_msgs = messages_list[-4:-1] if len(messages_list) >= 4 else messages_list[:-1]
                    for _m in _recent_msgs:
                        _role = "用户" if (hasattr(_m, 'type') and _m.type in ("human", "user")) or (
                                    isinstance(_m, dict) and _m.get("role") == "user") else "助手"
                        _content = _m.content if hasattr(_m, 'content') else (
                            _m.get("content", "") if isinstance(_m, dict) else str(_m))
                        _intent_prompt += f"{_role}：{_content[:200]}\n"
                    _intent_prompt += f"\n用户最新消息：{clean_user_request}\n\n请回答 YES 或 NO："

                    try:
                        _intent_client = _get_openai_client()
                        _intent_response = _intent_client.chat.completions.create(
                            model=Params.DEFAULT_TEXT_TOOL_MODEL,
                            messages=[{"role": "user", "content": _intent_prompt}],
                            temperature=0.1,
                            max_tokens=20,
                        )
                        _intent_result = _intent_response.choices[0].message.content.strip().upper()
                        _is_referring_generated = (_intent_result == "YES")
                        logger.info(
                            f"[INTENT_CHECK] 用户是否引用生成文档: {_is_referring_generated} (LLM 回答: {_intent_result})")
                    except Exception as e:
                        logger.warning(f"[INTENT_CHECK] 意图判断失败，回退到关键词匹配: {e}")
                        # 回退到关键词匹配
                        _generated_doc_keywords = ["生成的这个文档", "生成的文档", "刚才生成的", "合并后的文档",
                                                   "合并后的文件", "生成的文件", "这个文档", "刚才那个文档",
                                                   "刚才那个文件", "优化下格式", "调整下排版", "再改改"]
                        _is_referring_generated = any(kw in clean_user_request for kw in _generated_doc_keywords)

                # 如果是引用生成的文档，从对话历史中提取下载链接并自动下载到 cache 目录
                if _is_referring_generated and len(messages_list) > 1:
                    _generated_file_path = None
                    for msg in reversed(messages_list[:-1]):
                        msg_type = getattr(msg, 'type', None) or (msg.get("role") if isinstance(msg, dict) else None)
                        content = msg.content if hasattr(msg, 'content') else (
                            msg.get("content", "") if isinstance(msg, dict) else str(msg))

                        logger.debug(f"[GENERATED_DOC_DEBUG] msg_type={msg_type}, content[:300]={content[:300]}")

                        # 从下载链接中提取 URL 和文件名（支持多种格式）
                        _dl_match = re.search(
                            r'\[下载.*?文档.*?\]\((https?://[^)]+/([^/)]+\.(?:docx|pdf|txt|xlsx|pptx)))\)', content)
                        if not _dl_match:
                            # 尝试更宽松的正则
                            _dl_match = re.search(r'(https?://[^)]+/([^/)]+\.(?:docx|pdf|txt|xlsx|pptx)))', content)

                        if _dl_match:
                            _download_url = _dl_match.group(1)
                            _generated_filename = _dl_match.group(2)

                            # 确保 cache 目录存在
                            _cache_dir = Path(__file__).parent.parent / "cache"
                            _cache_dir.mkdir(parents=True, exist_ok=True)

                            _cache_file = _cache_dir / _generated_filename

                            # 如果文件不存在，自动下载（异步执行，不阻塞主流程）
                            if not _cache_file.exists():
                                try:
                                    import urllib.request
                                    import urllib.parse
                                    # 对 URL 中的中文字符进行编码
                                    _encoded_url = urllib.parse.quote(_download_url, safe=':/?=&')
                                    logger.info(f"[GENERATED_DOC] 正在下载文件: {_download_url}")

                                    # 定义同步下载函数
                                    def _download_file():
                                        _req = urllib.request.Request(_encoded_url)
                                        _req.add_header('User-Agent', 'Mozilla/5.0')
                                        with urllib.request.urlopen(_req, timeout=10) as _response:
                                            with open(_cache_file, 'wb') as _f:
                                                _f.write(_response.read())

                                    # 在线程池中异步执行下载，不阻塞事件循环
                                    await asyncio.to_thread(_download_file)
                                    logger.info(f"[GENERATED_DOC] 下载成功: {_cache_file}")
                                except Exception as e:
                                    logger.error(f"[GENERATED_DOC] 下载失败: {e}")
                                    continue

                            if _cache_file.exists():
                                _generated_file_path = str(_cache_file)
                                logger.info(f"[GENERATED_DOC] 使用缓存文件: {_generated_file_path}")
                                break

                    if _generated_file_path:
                        file_paths = [_generated_file_path]
                        logger.info(f"[GENERATED_DOC] 替换 file_paths 为生成的文档: {file_paths}")
                    else:
                        logger.warning(f"[GENERATED_DOC] 未能从对话历史中提取生成文档路径")

                _t0 = _time.time()

                # 导入 OpenAI 格式的工具定义（纯 JSON，不依赖 langchain）
                from core.tools.doc_tools import DOC_TOOLS_SCHEMA, create_doc_tools
                tools_schema = DOC_TOOLS_SCHEMA

                # 缓存 langchain 工具（用于实际执行）
                global _doc_tools
                if _doc_tools is None:
                    _t_create = _time.time()
                    _doc_tools = create_doc_tools(thread_id=thread_id)
                    logger.info(f"[PERF] create_doc_tools 耗时: {_time.time() - _t_create:.3f}s")
                tools_map = {t.name: t for t in _doc_tools}

                system_prompt = get_doc_system_prompt()
                user_msg_parts = []
                if file_paths:
                    if len(file_paths) == 1:
                        _fp = file_paths[0]
                        _file_ext = Path(_fp).suffix.lower()
                        user_msg_parts.append(f"附件文件路径：{_fp}")
                        user_msg_parts.append(f"附件文件格式：{_file_ext}")
                    else:
                        user_msg_parts.append(f"附件文件数量：{len(file_paths)}")
                        for i, _fp in enumerate(file_paths):
                            _file_ext = Path(_fp).suffix.lower()
                            _fname = Path(_fp).name
                            user_msg_parts.append(f"附件{i + 1}：{_fname}（路径：{_fp}，格式：{_file_ext}）")
                    user_msg_parts.append(
                        "⚠️ 重要提示：你现在**没有任何一个文件的实际内容**，只有路径。你必须调用工具（比如 read_document_and_answer）才能读取文件内容并回答用户的问题。绝对不要在没调工具的情况下直接回复用户！")
                user_msg_parts.append(f"用户请求：{clean_user_request}")

                logger.info(f"用户请求：{user_msg_parts}")

                # 使用原生 OpenAI API 格式的消息
                # 先加入对话历史（不含最后一条用户消息，因为下面会单独构造）
                history_messages = []
                if len(messages_list) > 1:
                    for msg in messages_list[:-1]:
                        if isinstance(msg, dict):
                            history_messages.append(msg)
                        elif hasattr(msg, 'type'):
                            msg_type = msg.type
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            if msg_type in ("ai", "assistant"):
                                entry = {"role": "assistant", "content": content}
                                tool_calls = getattr(msg, "tool_calls", None)
                                if tool_calls:
                                    entry["tool_calls"] = tool_calls
                                history_messages.append(entry)
                            elif msg_type in ("human", "user"):
                                history_messages.append({"role": "user", "content": content})
                            elif msg_type == "system":
                                history_messages.append({"role": "system", "content": content})
                            elif msg_type == "tool":
                                history_messages.append({
                                    "role": "tool",
                                    "tool_call_id": getattr(msg, "tool_call_id", ""),
                                    "content": content,
                                })

                messages = [
                               {"role": "system", "content": system_prompt},
                           ] + history_messages + [
                               {"role": "user", "content": "\n\n".join(user_msg_parts)},
                           ]

                logger.info("开始LLM推理（静默判断是否需要调工具）")
                _t2 = _time.time()

                # 使用原生 OpenAI API 调用（带 tools 参数）
                client = get_llm_with_tools()
                response = client.chat.completions.create(
                    model=Params.DEFAULT_TEXT_TOOL_MODEL,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=0.2,
                )
                logger.info(f"[PERF] document(tool): LLM 工具选择耗时 {_time.time() - _t2:.3f}s")

                choice = response.choices[0]
                message = choice.message
                content_text = message.content or ""

                # 解析 tool_calls
                tool_calls = []
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tc in message.tool_calls:
                        tool_calls.append({
                            "name": tc.function.name,
                            "args": json.loads(tc.function.arguments),
                            "id": tc.id,
                            "type": "tool_call",
                        })

                if not tool_calls and file_paths:
                    _DOC_KEYWORDS = [
                        "说说", "讲讲", "介绍", "讲了什么", "总结", "摘要", "概括", "分析",
                        "看看", "检查", "提取", "整理", "修改", "翻译", "转成", "保存",
                        "文档", "文件", "内容", "讲解", "论述", "说明", "简述", "要点",
                        "核心", "重点", "什么", "如何", "有没有", "对吗", "错别字",
                    ]
                    _has_doc_keyword = any(kw in clean_user_request for kw in _DOC_KEYWORDS)
                    if _has_doc_keyword:
                        _SAVE_KEYWORDS = [
                            "保存", "输出", "导出", "另存为", "生成", "创建", "下载",
                            "word", "docx", "excel", "xlsx", "pdf", "markdown",
                            "txt", "ppt", "pptx", "格式",
                        ]
                        _has_save_intent = any(kw in clean_user_request.lower() for kw in _SAVE_KEYWORDS)

                        if _has_save_intent:
                            logger.warning(
                                f"[PERF] document(tool): LLM 未调工具但有保存意图，强制走 modify_existing_document 兜底")
                        else:
                            logger.warning(
                                f"[PERF] document(tool): LLM 未调工具但有文件+文档关键词，强制走 read_document_and_answer 兜底")
                        async def _read_one_for_fallback(_fp: str) -> str:
                            from core.skills.DocProcess.document_process import doc_processor
                            _ext = Path(_fp).suffix.lower()
                            _content = ""
                            if _ext in ['.xlsx', '.xls']:
                                async for chunk_text in doc_processor._read_excel(_fp):
                                    _content += chunk_text
                            elif _ext == '.docx':
                                from docx import Document as DocxDoc
                                _doc = DocxDoc(_fp)
                                for pi, para in enumerate(_doc.paragraphs):
                                    if para.text.strip():
                                        _content += para.text + "\n"
                            elif _ext == '.txt':
                                async for chunk_text in doc_processor._read_txt(_fp):
                                    _content += chunk_text
                            elif _ext in ['.pptx', '.ppt']:
                                async for chunk_text in doc_processor._read_pptx(_fp):
                                    _content += chunk_text
                            elif _ext == '.doc':
                                async for chunk_text in doc_processor._read_doc(_fp):
                                    _content += chunk_text
                            elif _ext == '.pdf':
                                async for chunk_text in doc_processor.read_document(_fp):
                                    _content += chunk_text
                            elif _ext == '.json':
                                import json
                                try:
                                    with open(_fp, "r", encoding="utf-8") as f:
                                        _obj = json.load(f)
                                    _content = json.dumps(_obj, ensure_ascii=False, indent=2)
                                except json.JSONDecodeError:
                                    _content = Path(_fp).read_text(encoding="utf-8")
                            else:
                                raise ValueError(f"不支持的文件格式: {_ext}")
                            return _content

                        _fallback_content = ""
                        try:
                            _fallback_content = await _read_one_for_fallback(file_paths[0])
                        except Exception as e:
                            logger.error(f"[FALLBACK] 读取文件内容失败: {e}")
                            _fallback_content = f"[读取失败：{e}]"

                        if _has_save_intent:
                            from core.tools.doc_tools import _infer_output_format
                            _output_format = _infer_output_format(clean_user_request, ".docx")
                            tool_calls = [{
                                "name": "modify_existing_document",
                                "args": {
                                    "file_path": file_paths[0],
                                    "file_content": _fallback_content,
                                    "user_request": clean_user_request,
                                    "output_format": _output_format,
                                },
                                "id": f"fallback_{uuid.uuid4().hex[:8]}",
                                "type": "tool_call",
                            }]
                        else:
                            tool_calls = [{
                                "name": "read_document_and_answer",
                                "args": {
                                    "file_path": file_paths[0],
                                    "file_content": _fallback_content,
                                    "user_request": clean_user_request,
                                },
                                "id": f"fallback_{uuid.uuid4().hex[:8]}",
                                "type": "tool_call",
                            }]

                if not tool_calls:
                    logger.info(f"[PERF] document(tool): 闲聊/非文档需求，直接回复, 耗时 {_time.time() - _t2:.3f}s")
                    if writer:
                        writer({"text": content_text, "sources": [], "status": "streaming"})
                    full_response = content_text
                    logger.info(f"[PERF] document(tool): 总耗时 {_time.time() - _t0:.3f}s")
                    return {
                        "messages": [AIMessage(content=full_response)],
                        "type": "document",
                        "previous_node": "document_node",
                        "sources": [],
                    }

                if writer:
                    writer({"text": "理解用户需求", "sources": [], "status": "step", "step": {"title": "理解用户需求", "status": "running"}})
                if writer:
                    writer({"text": f"用户请求：{clean_user_request[:100]}", "sources": [], "status": "step_detail", "step_detail": f"用户请求：{clean_user_request[:100]}"})
                if len(file_paths) == 1:
                    if writer:
                        writer({"text": f"附带文件：{Path(file_paths[0]).name}", "sources": [], "status": "step_detail", "step_detail": f"附带文件：{Path(file_paths[0]).name}"})
                elif len(file_paths) > 1:
                    _names = [Path(p).name for p in file_paths]
                    if writer:
                        writer({"text": f"附带 {len(file_paths)} 个文件：{', '.join(_names)}", "sources": [], "status": "step_detail", "step_detail": f"附带 {len(file_paths)} 个文件：{', '.join(_names)}"})
                if writer:
                    writer({"text": "理解用户需求", "sources": [], "status": "step", "step": {"title": "理解用户需求", "status": "done"}})

                async def _read_one(_fp: str) -> str:
                    from core.skills.DocProcess.document_process import doc_processor
                    _ext = Path(_fp).suffix.lower()
                    _content = ""
                    if _ext in ['.xlsx', '.xls']:
                        async for chunk_text in doc_processor._read_excel(_fp):
                            _content += chunk_text
                    elif _ext == '.docx':
                        from docx import Document as DocxDoc
                        _doc = DocxDoc(_fp)
                        for pi, para in enumerate(_doc.paragraphs):
                            if para.text.strip():
                                _content += para.text + "\n"
                    elif _ext == '.txt':
                        async for chunk_text in doc_processor._read_txt(_fp):
                            _content += chunk_text
                    elif _ext in ['.pptx', '.ppt']:
                        async for chunk_text in doc_processor._read_pptx(_fp):
                            _content += chunk_text
                    elif _ext == '.doc':
                        async for chunk_text in doc_processor._read_doc(_fp):
                            _content += chunk_text
                    elif _ext == '.pdf':
                        async for chunk_text in doc_processor.read_document(_fp):
                            _content += chunk_text
                    elif _ext == '.json':
                        import json
                        try:
                            with open(_fp, "r", encoding="utf-8") as f:
                                _obj = json.load(f)
                            _content = json.dumps(_obj, ensure_ascii=False, indent=2)
                        except json.JSONDecodeError:
                            _content = Path(_fp).read_text(encoding="utf-8")
                    else:
                        raise ValueError(f"不支持的文件格式: {_ext}")
                    return _content

                file_content = ""
                file_path_for_tool = file_paths[0] if file_paths else None
                if file_paths:
                    _t1 = _time.time()
                    _file_parts = []
                    logger.info(f"[FILE_READ] 开始读取 {len(file_paths)} 个文件: {[Path(p).name for p in file_paths]}")
                    for idx, _fp in enumerate(file_paths):
                        _fname = Path(_fp).name
                        _ext = Path(_fp).suffix.lower()
                        if writer:
                            writer({"text": f"读取文件 ({idx + 1}/{len(file_paths)})", "sources": [], "status": "step", "step": {"title": f"读取文件 ({idx + 1}/{len(file_paths)})", "status": "running"}})
                        try:
                            _c = await _read_one(_fp)
                            if not _c:
                                logger.warning(f"[FILE_READ] 文件 {_fname} 读取内容为空！")
                            _file_parts.append(f"===== 文件{idx + 1}：{_fname} =====\n{_c}")
                            if writer:
                                writer({"text": f"{_fname}：{len(_c)} 字", "sources": [], "status": "step_detail", "step_detail": f"{_fname}：{len(_c)} 字"})
                        except Exception as e:
                            logger.error(f"[FILE_READ] 读取文件失败 {_fp}: {e}")
                            import traceback
                            logger.error(f"[FILE_READ] 异常堆栈: {traceback.format_exc()}")
                            _file_parts.append(f"===== 文件{idx + 1}：{_fname} =====\n[读取失败：{e}]")
                        if writer:
                            writer({"text": f"读取文件 ({idx + 1}/{len(file_paths)})", "sources": [], "status": "step", "step": {"title": f"读取文件 ({idx + 1}/{len(file_paths)})", "status": "done"}})
                    file_content = "\n\n".join(_file_parts)
                    logger.info(
                        f"[PERF] document(tool): 读取 {len(file_paths)} 个文件耗时 {_time.time() - _t1:.3f}s, chars={len(file_content)}")
                    if writer:
                        writer({"text": f"共读取 {len(file_content)} 字", "sources": [], "status": "step_detail", "step_detail": f"共读取 {len(file_content)} 字"})

                if writer:
                    writer({"text": "选择文档处理工具", "sources": [], "status": "step", "step": {"title": "选择文档处理工具", "status": "running"}})
                if writer:
                    writer({"text": "选择文档处理工具", "sources": [], "status": "step", "step": {"title": "选择文档处理工具", "status": "done"}})

                logger.info("工具选择已完成，准备执行工具")

                from core.tools.doc_tools import set_progress_queue

                _final_download_url = ""

                for tc in tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {})

                    # 强制覆盖 LLM 生成的 file_content，使用我们实际读取的内容
                    if tool_name in ("modify_existing_document", "read_document_and_answer", "convert_file_format"):
                        if file_path_for_tool:
                            tool_args["file_path"] = file_path_for_tool
                        if file_content:
                            tool_args["file_content"] = file_content
                            logger.info(f"[TOOL_ARG] {tool_name}: file_content 已强制覆盖，chars={len(file_content)}")
                        else:
                            logger.warning(f"[TOOL_ARG] {tool_name}: file_content 为空，无法覆盖！")

                    if writer:
                        writer({"text": f"调用工具：{tool_name}", "sources": [], "status": "step", "step": {"title": f"调用工具：{tool_name}", "status": "running"}})

                    _arg_preview = {}
                    for k, v in tool_args.items():
                        if k in ("file_content", "content_description"):
                            _arg_preview[k] = f"<{len(str(v))} 字>"
                        elif k == "file_path":
                            _arg_preview[k] = Path(str(v)).name if v else ""
                        else:
                            _arg_preview[k] = str(v)[:80]
                    if writer:
                        writer({"text": f"工具入参：{_arg_preview}", "sources": [], "status": "step_detail", "step_detail": f"工具入参：{_arg_preview}"})

                    _progress_q = asyncio.Queue()
                    set_progress_queue(_progress_q)

                    logger.info(f"[PERF] document(tool): 调用 {tool_name}, args keys={list(tool_args.keys())}")
                    _t3 = _time.time()
                    selected_tool = tools_map.get(tool_name)
                    _tool_task = asyncio.create_task(selected_tool.ainvoke(tool_args))
                    _tool_tasks.append(_tool_task)

                    while not _tool_task.done():
                        try:
                            _msg = await asyncio.wait_for(_progress_q.get(), timeout=0.2)
                            if writer:
                                writer({"text": _msg, "sources": [], "status": "step_detail", "step_detail": _msg})
                        except asyncio.TimeoutError:
                            pass

                    tool_result = await _tool_task
                    _elapsed = _time.time() - _t3

                    while not _progress_q.empty():
                        try:
                            _msg = _progress_q.get_nowait()
                            if writer:
                                writer({"text": _msg, "sources": [], "status": "step_detail", "step_detail": _msg})
                        except asyncio.QueueEmpty:
                            break

                    set_progress_queue(None)
                    logger.info(f"[PERF] document(tool): {tool_name} 执行耗时 {_elapsed:.3f}s")

                    _result_text = str(tool_result)

                    _url_match = re.search(r'\[下载文档\]\((https?://[^)]+)\)', _result_text)
                    if _url_match:
                        _final_download_url = _url_match.group(1)

                    if writer:
                        writer({"text": f"工具执行完成，耗时 {_elapsed:.1f}s", "sources": [], "status": "step_detail", "step_detail": f"工具执行完成，耗时 {_elapsed:.1f}s"})
                    if _result_text:
                        _result_display = _result_text[:200] + ("..." if len(_result_text) > 200 else "")
                        if writer:
                            writer({"text": f"工具返回：{_result_display}", "sources": [], "status": "step_detail", "step_detail": f"工具返回：{_result_display}"})

                    messages.append({"role": "assistant", "content": content_text, "tool_calls": [
                        {"id": tc.get("id", ""), "type": "function", "name": tc.get("name", ""),
                         "args": tc.get("args", {})}
                    ]})
                    messages.append({"role": "tool", "content": _result_text, "tool_call_id": tc.get("id", "")})

                if writer:
                    writer({"text": f"调用工具：{tool_name}", "sources": [], "status": "step", "step": {"title": f"调用工具：{tool_name}", "status": "done"}})
                if writer:
                    writer({"text": "生成最终回复", "sources": [], "status": "step", "step": {"title": "生成最终回复", "status": "running"}})

                _t_final = _time.time()
                logger.info(f"开始 LLM 生成最终聊天回复, messages 数={len(messages)}")

                _cleaned = _clean_messages_for_chat_reply(messages)
                logger.info(f"[PERF] 清理后聊天消息数={len(_cleaned)}")

                _yielded_any = False
                full_response = ""
                async for chunk_text in _stream_by_prompt_async(
                        _cleaned, model=Params.DEFAULT_CHAT_MODEL, temperature=0.5
                ):
                    if writer:
                        writer({"text": chunk_text, "sources": [], "status": "streaming"})
                    full_response += chunk_text
                    _yielded_any = True

                # 如果有下载链接，且 LLM 回复中没有包含下载链接，则追加
                if _final_download_url:
                    # 检查 LLM 回复中是否已包含下载链接
                    _has_download_link = any(
                        kw in full_response for kw in ["[下载", "download", "下载文档", "下载链接"])
                    if not _has_download_link:
                        if writer:
                            writer({"text": f"\n\n [下载文档]({_final_download_url})", "sources": [],
                                    "status": "streaming"})

                if not _yielded_any:
                    if _final_download_url:
                        _final_text = f"✅ 文档已处理完成，点击下方链接下载：\n\n [下载处理后的文档]({_final_download_url})"
                    else:
                        _final_text = _result_text or "工具执行完成，但未返回结果。"
                    if writer:
                        writer({"text": _final_text, "sources": [], "status": "streaming"})

                if thread_id:
                    context = {}
                    if _final_download_url:
                        context["output_file"] = _final_download_url
                    if file_paths:
                        context["upload_files"] = []
                        for file_path in file_paths:
                            ext = os.path.splitext(file_path)[1]
                            file_name = os.path.basename(file_path)
                            file_info = {"ext": ext, "file_name": file_name, "file_path": file_path}

                            logger.info(f"[Memory] 保存文件信息: {file_info}")
                            context["upload_files"].append(file_info)

                    SessionMemoryManager.set_context(thread_id, **context)

                if writer:
                    writer({"text": "生成最终回复", "sources": [], "status": "step", "step": {"title": "生成最终回复", "status": "done"}})
                logger.info(f"[PERF] document(tool): LLM 生成回复耗时 {_time.time() - _t_final:.3f}s, 总耗时 {_time.time() - _t0:.3f}s, has_download={bool(_final_download_url)}")


            except asyncio.CancelledError:
                for _t in _tool_tasks:
                    if not _t.done():
                        _t.cancel()

                logger.info("[STOP] document skill 被取消，已中止所有工具任务")
                raise

            except Exception as e:
                logger.error(f"document 处理失败: {str(e)[:80]}")
                if writer:
                    writer({"text": f"文档处理失败: {str(e)[:100]}", "sources": [], "status": "streaming"})
                full_response = f"文档处理失败: {str(e)[:100]}"

            # logger.info(f"[Memory] 保存会话记忆: {user_query} -> {full_response}")
            # if thread_id and user_query and full_response:
            #     SessionMemoryManager.save(thread_id, user_query, full_response)

            return {
                "messages": [AIMessage(content=full_response)],
                "type": "document",
                "previous_node": "document_node",
                "sources": [],
            }

        elif skill_type == "drawing":
            try:
                from core.skills.DrawingRecoAssistant.drawing_assistant import get_drawing_assistant
                assistant = get_drawing_assistant()
                
                img_path = pdf_path = dxf_path = None
                if file_path:
                    ext = Path(file_path).suffix.lower()
                    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
                        img_path = file_path
                    elif ext == ".pdf":
                        pdf_path = file_path
                    elif ext == ".dxf":
                        dxf_path = file_path

                # 使用异步迭代器包装同步生成器
                async for chunk in _async_iter_sync_generator(
                        assistant.stream_invoke(
                            user_query=user_query,
                            img_path=img_path,
                            pdf_path=pdf_path,
                            dxf_path=dxf_path,
                            thread_id="agent_draw_session",
                        )
                ):
                    if isinstance(chunk, dict) and "sources" in chunk:
                        final_sources = chunk["sources"]
                    elif isinstance(chunk, str):
                        full_response += chunk
                        if writer:
                            writer({"text": chunk, "sources": [], "status": "streaming"})

                
            except Exception as e:
                logger.error(f"drawing 处理失败: {str(e)[:80]}")
                full_response = f"图纸识别处理失败: {str(e)[:100]}"
            
            return {
                "messages": [AIMessage(content=full_response)],
                "type": "drawing",
                "previous_node": "drawing_node",
                "sources": json.dumps(final_sources, ensure_ascii=False),
            }
        
        elif skill_type == "code":
            skill_hint = state.get("skill", "")
            
            if skill_hint == "project_code":
                from core.skills.AICode.project_programming import Project_Programming
                project_programming = Project_Programming(file_path, messages_list, project_context)
                message = messages_list[-1]

                async for chunk in _async_iter_sync_generator(project_programming.get_model_response_stream(message)):
                    if chunk:
                        full_response += chunk
                        # 使用StreamWriter 实时输出chunk到前端
                        if writer:
                            writer({"text": chunk, "sources": [], "status": "streaming"})

            else:
                from core.skills.AICode.general_programming import General_Programming
                general_programming = General_Programming(file_path, messages_list)
                message = messages_list[-1]
                async for chunk in _async_iter_sync_generator(general_programming.get_model_response_stream(message)):
                    if chunk:
                        full_response += chunk
                        # 使用StreamWriter 实时输出chunk到前端
                        if writer:
                            writer({"text": chunk, "sources": [], "status": "streaming"})
            
            return {
                "messages": [AIMessage(content=full_response)],
                "type": "code",
                "previous_node": "code_node",
                "sources": [],
            }

        return {
            "messages": [AIMessage(content=full_response)],
            "type": skill_type,
            "previous_node": f"{skill_type}_node",
            "sources": final_sources,
        }
    
    return streaming_node_func


def _create_supervisor_node():
    """创建路由监督节点"""
    async def supervisor_node(state: AgentState):
        logger.info("supervisor_node")
        prompts = """你是一个专业的客服助手，需要根据用户的问题进行任务分类，并将任务分给对应的Agent来执行。

分类规则：
1. 如果用户的问题是和旅游路线规划相关的，那就返回travel。
2. 如果用户的问题是和讲笑话相关的，那就返回joke。
3. 如果用户的问题是和对联相关的，那就返回couplet。
4. 如果用户的问题是和文档相关的（包含修改已有文档、新建文档并写入内容、排版、修正格式、修改对齐方式，查看文档中的错别字等等），那就返回document。
5. 如果用户的问题是和编程相关的，那就返回code。
6. 如果用户的问题是和工程图纸识别、尺寸提取、公差校核、DXF解析、国标查询、单位换算等相关的，那就返回drawing。
7. 如果是其它问题，那就返回other。

重要：如果用户的问题是追问、确认、或者对之前回答的反馈（例如"确认下是否写得对？"、"再优化一下"、"不对，应该是..."），请根据对话历史判断之前讨论的主题，并返回对应的分类。
【强制要求】你只能返回以下9个选项中的一个：supervisor, travel, joke, couplet, document, code, drawing, problem_analysis, other
【禁止】不要返回任何解释、说明、行程规划、笑话内容或其他额外文本。
【格式】只返回一个单词，例如：travel
"""
        message = state["messages"][-1]
        if hasattr(message, 'content'):
            message_content = message.content
        else:
            message_content = str(message)
        logger.info(f"用户消息: {message_content[:30]}...")

        conversation_history = []
        for msg in state["messages"][:-1]:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            if isinstance(msg, dict) and msg.get("role") == "assistant":
                conversation_history.append({"role": "assistant", "content": content})
            elif isinstance(msg, dict) and msg.get("role") == "user":
                conversation_history.append({"role": "user", "content": content})
            else:
                msg_type = str(type(msg))
                if "HumanMessage" in msg_type:
                    conversation_history.append({"role": "user", "content": content})
                elif "AIMessage" in msg_type:
                    conversation_history.append({"role": "assistant", "content": content})

        prompt_list = [
            {"role": "system", "content": prompts},
        ] + conversation_history + [
            {"role": "user", "content": message_content},
        ]
        
        if state.get("type") == END:
            logger.info("已有类型 END，返回 END")
            return {"type": END}
        elif state.get("skill", "").strip():
            skill_type = state["skill"].strip()
            logger.info(f"前端指定技能: {skill_type}")

            if skill_type in ['project_code', 'general_code']:
                logger.info("路由到 code_node")
                return {"type": "code"}
            elif skill_type in Params.NODE_LIST:
                logger.info(f"直接路由到 {skill_type}")
                return {"type": skill_type}
            else:
                logger.warning(f"未知技能类型: {skill_type}，尝试通过LLM分类")
        
        try:
            response = ""
            async for chunk_text in _stream_by_prompt_async(prompt_list, model=Params.DEFAULT_CHAT_MODEL):
                response += chunk_text
            
            typeRes = response.strip()
            logger.debug(f"分类结果原始值: '{typeRes}' (长度: {len(typeRes)})")

            skill_required_nodes = {'code', 'document', 'drawing'}

            if typeRes.strip() in Params.NODE_LIST:
                if typeRes.strip() in skill_required_nodes and not state.get("skill", "").strip():
                    logger.warning(f"分类结果为 {typeRes.strip()}，但未选择技能，默认路由到 other")
                    return {"type": "other"}
                logger.info(f"分类成功: {typeRes.strip()}")
                return {"type": typeRes.strip()}
            else:
                logger.warning(f"分类失败: '{typeRes}' 不在列表 {Params.NODE_LIST} 中")
                logger.info("默认返回 other")
                return {"type": "other"}
        except Exception as e:
            logger.error(f"supervisor_node LLM 调用失败: {str(e)[:50]}")
            return {"type": "other"}
    
    return supervisor_node


def _create_reflection_node():
    """创建反思节点，评估输出质量"""
    async def reflection_node(state: AgentState):
        logger.info("reflection_node 开始反思")
        
        previous_node = state.get("previous_node", "")
        last_message = state["messages"][-1]
        user_message = state["messages"][-2] if len(state["messages"]) >= 2 else state["messages"][0]
        
        if hasattr(last_message, 'content'):
            ai_response = last_message.content
        else:
            ai_response = str(last_message)
        
        if hasattr(user_message, 'content'):
            user_query = user_message.content
        else:
            user_query = str(user_message)
        
        reflection_prompts = {
            "travel_node": """你是一个专业的旅行规划质量评估专家。请评估以下旅行路线规划的质量：

评估标准：
1. 是否包含具体的景点、交通、住宿等实用信息
2. 路线安排是否合理、可行
3. 是否满足用户的具体需求
4. 信息是否准确、详细

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",
            
            "joke_node": """你是一个专业的笑话质量评估专家。请评估以下笑话的质量：

评估标准：
1. 是否有趣、有笑点
2. 是否符合用户要求的主题
3. 语言是否流畅自然
4. 长度是否合适（不超过100字）

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",
            
            "couplet_node": """你是一个专业的对联质量评估专家。请评估以下对联的质量：

评估标准：
1. 上下联是否对仗工整（词性、平仄、意境）
2. 是否符合用户给出的上联主题
3. 是否有文化内涵和文学美感

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",
            
            "document_node": """你是一个专业的文档修改质量评估专家。请评估以下文档修改的质量：

评估标准：
1. 是否按照用户要求进行了修改
2. 修改后的内容是否准确、通顺
3. 是否保留了原文档的核心内容
4. 格式是否正确

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",
            
            "code_node": """你是一个专业的代码质量评估专家。请评估以下代码的质量：

评估标准：
1. 代码是否能正确运行（语法正确）
2. 代码逻辑是否合理、无Bug
3. 是否有必要的注释和说明
4. 是否满足用户的具体需求
5. 代码风格是否规范

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",
            
            "other_node": """你是一个专业的回答质量评估专家。请评估以下回答的质量：

评估标准：
1. 回答是否礼貌、得体
2. 是否清晰表达了无法处理的原因

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",
            
            "drawing_node": """你是一个专业的工程图纸分析质量评估专家。请评估以下图纸分析回答的质量：

评估标准：
1. 是否准确提取了图纸中的尺寸、标注、图层等信息
2. 尺寸分析是否合理、尺寸链是否闭合
3. 国标引用是否准确
4. 加工建议是否专业、实用
5. 回答结构是否清晰（图纸信息、尺寸分析、国标依据、加工建议）

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",
        }
        
        reflection_prompt = reflection_prompts.get(previous_node, reflection_prompts["other_node"])
        
        try:
            response = ""
            async for chunk_text in _stream_by_prompt_async([
                {"role": "user", "content": reflection_prompt.format(user_query=user_query, ai_response=ai_response)}
            ]):
                response += chunk_text
            
            reflection_result = response.strip()
        except Exception as e:
            logger.error(f"reflection_node LLM 调用失败: {str(e)[:50]}")
            reflection_result = "PASS"
        
        logger.info(f"反思结果: {reflection_result[:50]}...")
        
        if reflection_result.startswith("PASS"):
            logger.info("反思通过，质量合格")
            return {"type": END}
        else:
            logger.info("反思未通过，需要重新生成")
            feedback = reflection_result.replace("FAIL:", "").strip()
            feedback_message = f"之前的回答质量不够好，请根据以下反馈重新生成：{feedback}"
            return {
                "messages": [HumanMessage(content=feedback_message)],
                "type": previous_node.replace("_node", "")
            }
    
    return reflection_node


def _routing_func(state: AgentState):
    """路由函数：根据 state 中的 type 字段决定下一步走向"""
    current_type = state["type"]
    logger.info(f"routing_func: type = '{current_type}'")
    
    if current_type == Params.MAPPING_NODE.get(Params.TRAVEL_NODE):
        logger.info(f"路由到 {Params.TRAVEL_NODE}")
        return Params.TRAVEL_NODE
    elif current_type == Params.MAPPING_NODE.get(Params.JOKE_NODE):
        logger.info(f"路由到 {Params.JOKE_NODE}")
        return Params.JOKE_NODE
    elif current_type == Params.MAPPING_NODE.get(Params.COUPLET_NODE):
        logger.info(f"路由到 {Params.COUPLET_NODE}")
        return Params.COUPLET_NODE
    elif current_type == Params.MAPPING_NODE.get(Params.DOCUMENT_NODE):
        logger.info(f"路由到 {Params.DOCUMENT_NODE}")
        return Params.DOCUMENT_NODE
    elif current_type == Params.MAPPING_NODE.get(Params.CODE_NODE):
        logger.info(f"路由到 {Params.CODE_NODE}")
        return Params.CODE_NODE
    elif current_type == Params.MAPPING_NODE.get(Params.DRAWING_NODE):
        logger.info(f"路由到 {Params.DRAWING_NODE}")
        return Params.DRAWING_NODE
    elif current_type == Params.MAPPING_NODE.get(Params.PROBLEM_ANALYSIS_NODE):
        logger.info(f"路由到 {Params.PROBLEM_ANALYSIS_NODE}")
        return Params.PROBLEM_ANALYSIS_NODE
    elif current_type == END:
        logger.info("路由到 END")
        return END
    else:
        logger.info(f"路由到 {Params.OTHER_NODE} (未知类型: {current_type})")
        return Params.OTHER_NODE


def _reflection_routing_func(state: AgentState):
    """反思节点的路由：PASS则结束，FAIL则回到原节点重新生成"""
    current_type = state["type"]
    logger.info(f"reflection_routing_func: type = '{current_type}'")
    if current_type == END:
        logger.info("反思通过，路由到 END")
        return END
    else:
        logger.info(f"反思未通过，路由回 {current_type}_node")
        return f"{current_type}_node"

def create_agent_graph():
    """创建基于 LangGraph 的智能体图，支持流式输出
    
    使用 _create_streaming_node 工厂函数创建所有节点，
    每个节点都支持快速流式响应。
    
    图结构：
    START -> supervisor_node -> [skill nodes] -> END
    """
    builder = StateGraph(AgentState)
    
    # 添加所有流式节点
    builder.add_node(Params.SUPERVISOR_NODE, _create_supervisor_node())
    builder.add_node(Params.TRAVEL_NODE, _create_streaming_node("travel"))
    builder.add_node(Params.JOKE_NODE, _create_streaming_node("joke"))
    builder.add_node(Params.COUPLET_NODE, _create_streaming_node("couplet"))
    builder.add_node(Params.DOCUMENT_NODE, _create_streaming_node("document"))
    builder.add_node(Params.CODE_NODE, _create_streaming_node("code"))
    builder.add_node(Params.DRAWING_NODE, _create_streaming_node("drawing"))
    builder.add_node(Params.PROBLEM_ANALYSIS_NODE, _create_streaming_node("problem_analysis"))
    builder.add_node(Params.OTHER_NODE, _create_streaming_node("other"))
    
    # 添加边
    builder.add_edge(START, Params.SUPERVISOR_NODE)
    builder.add_conditional_edges(
        Params.SUPERVISOR_NODE,
        _routing_func,
        [Params.TRAVEL_NODE, Params.JOKE_NODE, Params.COUPLET_NODE, 
         Params.DOCUMENT_NODE, Params.CODE_NODE, Params.DRAWING_NODE, 
         Params.PROBLEM_ANALYSIS_NODE, Params.OTHER_NODE, END]
    )
    
    # 各节点处理完后到 END
    builder.add_edge(Params.TRAVEL_NODE, END)
    builder.add_edge(Params.JOKE_NODE, END)
    builder.add_edge(Params.COUPLET_NODE, END)
    builder.add_edge(Params.DOCUMENT_NODE, END)
    builder.add_edge(Params.CODE_NODE, END)
    builder.add_edge(Params.DRAWING_NODE, END)
    builder.add_edge(Params.PROBLEM_ANALYSIS_NODE, END)
    builder.add_edge(Params.OTHER_NODE, END)
    
    return builder.compile(checkpointer=InMemorySaver())


def create_streaming_agent_graph_with_reflection():
    """创建支持流式输出和反思机制的 LangGraph 智能体图
    
    与 create_agent_graph() 的区别：
    - 添加了反思节点进行质量评估
    - 反思通过则结束，未通过则重新生成
    - 所有节点都使用流式输出（通过 _stream_by_prompt_async）
    
    图结构：
    START -> supervisor_node -> [skill nodes] -> reflection_node -> [END or skill nodes]
    """
    builder = StateGraph(AgentState)
    
    # 添加所有流式节点
    builder.add_node(Params.SUPERVISOR_NODE, _create_supervisor_node())
    builder.add_node(Params.TRAVEL_NODE, _create_streaming_node("travel"))
    builder.add_node(Params.JOKE_NODE, _create_streaming_node("joke"))
    builder.add_node(Params.COUPLET_NODE, _create_streaming_node("couplet"))
    builder.add_node(Params.DOCUMENT_NODE, _create_streaming_node("document"))
    builder.add_node(Params.CODE_NODE, _create_streaming_node("code"))
    builder.add_node(Params.DRAWING_NODE, _create_streaming_node("drawing"))
    builder.add_node(Params.PROBLEM_ANALYSIS_NODE, _create_streaming_node("problem_analysis"))
    builder.add_node(Params.OTHER_NODE, _create_streaming_node("other"))
    builder.add_node(Params.REFLECTION_NODE, _create_reflection_node())
    
    # 添加边
    builder.add_edge(START, Params.SUPERVISOR_NODE)
    builder.add_conditional_edges(
        Params.SUPERVISOR_NODE,
        _routing_func,
        [Params.TRAVEL_NODE, Params.JOKE_NODE, Params.COUPLET_NODE, 
         Params.DOCUMENT_NODE, Params.CODE_NODE, Params.DRAWING_NODE, 
         Params.PROBLEM_ANALYSIS_NODE, Params.OTHER_NODE, END]
    )
    
    # 各节点处理完后到反思节点
    builder.add_edge(Params.TRAVEL_NODE, Params.REFLECTION_NODE)
    builder.add_edge(Params.JOKE_NODE, Params.REFLECTION_NODE)
    builder.add_edge(Params.COUPLET_NODE, Params.REFLECTION_NODE)
    builder.add_edge(Params.DOCUMENT_NODE, Params.REFLECTION_NODE)
    builder.add_edge(Params.CODE_NODE, Params.REFLECTION_NODE)
    builder.add_edge(Params.DRAWING_NODE, Params.REFLECTION_NODE)
    builder.add_edge(Params.PROBLEM_ANALYSIS_NODE, Params.REFLECTION_NODE)
    builder.add_edge(Params.OTHER_NODE, Params.REFLECTION_NODE)
    
    # 反思节点的路由
    builder.add_conditional_edges(
        Params.REFLECTION_NODE,
        _reflection_routing_func,
        [END, Params.TRAVEL_NODE, Params.JOKE_NODE, Params.COUPLET_NODE,
         Params.DOCUMENT_NODE, Params.CODE_NODE, Params.DRAWING_NODE,
         Params.PROBLEM_ANALYSIS_NODE, Params.OTHER_NODE]
    )
    
    return builder.compile(checkpointer=InMemorySaver())