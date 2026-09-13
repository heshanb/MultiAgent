from __future__ import annotations
import os
import re
import uuid


os.environ.setdefault("DASHSCOPE_API_KEY", "")

import asyncio
import json
import time as _time
from typing import Annotated, TypedDict
from operator import add
from pathlib import Path
from settings.Define import Params
from settings.logger_manager import get_logger
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None

logger = get_logger(__name__)

# 懒加载重型库
_ChatOpenAI = None
_OpenAI = None


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
_HumanMessage = None
_AIMessage = None
_SystemMessage = None
_ToolMessage = None
_doc_tools = None
_llm_with_tools = None


def _get_langchain_models():
    global _ChatOpenAI, _HumanMessage, _AIMessage, _SystemMessage, _ToolMessage
    if _ChatOpenAI is None:
        # from langchain_openai import ChatOpenAI
        from openai import OpenAI
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
        _ChatOpenAI = OpenAI
        _HumanMessage = HumanMessage
        _AIMessage = AIMessage
        _SystemMessage = SystemMessage
        _ToolMessage = ToolMessage
    return _ChatOpenAI, _HumanMessage, _AIMessage, _SystemMessage, _ToolMessage


def _get_langchain_messages():
    global _HumanMessage, _AIMessage
    if _HumanMessage is None:
        from langchain_core.messages import HumanMessage, AIMessage
        _HumanMessage = HumanMessage
        _AIMessage = AIMessage
    return _HumanMessage, _AIMessage


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

# 懒加载全局变量
_openai_client = None
_http_client = None
_llm = None
_doc_tools = None
_llm_with_tools = None


def get_openai_client():
    """懒加载 OpenAI 客户端"""
    return _get_openai_client()


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


def get_llm():
    """懒加载 LLM 客户端"""
    global _llm
    if _llm is None:
        OpenAI, _, _, _, _ = _get_langchain_models()
        # _llm = ChatOpenAI(
        #     model=Params.DEFAULT_CHAT_MODEL,
        #     api_key=os.getenv("DASHSCOPE_API_KEY"),
        #     base_url=Params.API_BASE,
        #     timeout=60,
        #     streaming=True
        # )
        _llm = OpenAI(
            model=Params.DEFAULT_CHAT_MODEL,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
            timeout=60,
            streaming=True
        )
    return _llm


def get_llm_with_tools():
    """懒加载原生 OpenAI 客户端（用于 tool calling）"""
    global _llm_with_tools
    if _llm_with_tools is None:
        _t_start = _time.time()
        _llm_with_tools = _get_openai_client()
        logger.info(f"[PERF] get_llm_with_tools: 总耗时 {_time.time()-_t_start:.3f}s")
    return _llm_with_tools


# 向后兼容：使用模块级别的 __getattr__ 实现懒加载
def __getattr__(name):
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
            _doc_tools = create_doc_tools()
        return _doc_tools
    elif name == "llm_with_tools":
        return get_llm_with_tools()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_SESSION_MEMORIES: dict = {}
_SESSION_CONTEXTS: dict = {}
_MAX_SESSIONS = 200
_WINDOW_SIZE = 5


class SessionMemoryManager:
    """轻量级会话短期记忆管理器（手写实现，避免 LangChain 0.3+ 的 DeprecationWarning）。
    
    每个 thread_id 对应一个独立的消息列表 [HumanMessage, AIMessage, ...]，
    自动维护滑动窗口（保留最近 5 轮对话 = 10 条消息），防止内存泄漏。
    同时维护一份 session context（文件路径等元信息），让后续追问能
    自动恢复上次对话关联的文件上下文。
    """

    @staticmethod
    def get_or_create(thread_id):
        if thread_id not in _SESSION_MEMORIES:
            if len(_SESSION_MEMORIES) >= _MAX_SESSIONS:
                old_key = next(iter(_SESSION_MEMORIES))
                _SESSION_MEMORIES.pop(old_key, None)
                _SESSION_CONTEXTS.pop(old_key, None)
            _SESSION_MEMORIES[thread_id] = []
            _SESSION_CONTEXTS[thread_id] = {}
            logger.info(f"[Memory] 创建新会话记忆 thread_id={thread_id}, 总会话数={len(_SESSION_MEMORIES)}")
        return _SESSION_MEMORIES[thread_id]

    @staticmethod
    def save(thread_id, user_msg: str, assistant_msg: str):
        if not thread_id or not user_msg:
            return
        history = SessionMemoryManager.get_or_create(thread_id)
        _, HumanMessage, AIMessage, _, _ = _get_langchain_models()
        history.append(HumanMessage(content=user_msg))
        history.append(AIMessage(content=assistant_msg or ""))
        if len(history) > _WINDOW_SIZE * 2:
            del history[:len(history) - _WINDOW_SIZE * 2]

    @staticmethod
    def load(thread_id) -> list:
        if not thread_id or thread_id not in _SESSION_MEMORIES:
            return []
        return list(_SESSION_MEMORIES[thread_id])

    @staticmethod
    def set_context(thread_id, **kwargs):
        """保存会话元信息（如 file_path, file_name, ext 等）"""
        if not thread_id:
            return
        SessionMemoryManager.get_or_create(thread_id)
        ctx = _SESSION_CONTEXTS.setdefault(thread_id, {})
        ctx.update({k: v for k, v in kwargs.items() if v is not None})

    @staticmethod
    def get_context(thread_id, key=None):
        """读取会话元信息。key=None 时返回整个 dict"""
        if not thread_id:
            return {} if key is None else None
        ctx = _SESSION_CONTEXTS.get(thread_id, {})
        if key is None:
            return dict(ctx)
        return ctx.get(key)

    @staticmethod
    def clear(thread_id):
        if thread_id in _SESSION_MEMORIES:
            del _SESSION_MEMORIES[thread_id]
            _SESSION_CONTEXTS.pop(thread_id, None)
            logger.info(f"[Memory] 清除会话 thread_id={thread_id}")

    @staticmethod
    def has_memory(thread_id) -> bool:
        return thread_id in _SESSION_MEMORIES

    @staticmethod
    def session_count() -> int:
        return len(_SESSION_MEMORIES)


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

class State(TypedDict):
    messages: Annotated[list, add]
    type: str
    file_path: str  # 添加文件路径字段
    file_paths: list[str]  # 添加多文件路径字段，用于处理多文件场景
    previous_node: str  # 记录上一个执行的节点类型，用于反思后路由
    skill: str  # 前端选择的技能类型：travel/joke/couplet/document/code/drawing/other
    sources: str  # 引用来源文件列表(JSON字符串)
    user: str  # 当前登录用户（用于知识库检索）
    images: list  # 图片数据列表，用于问题分析
    project_context: dict  # 项目上下文，包含项目名称和文件列表


async def classify_skill(messages_list, skill_hint: str = "", thread_id=None) -> str:
    """快速路由：前端指定 skill 直接映射，否则用 LLM 分类

    返回值: "travel" / "joke" / "couplet" / "document" / "code" / "drawing" / "problem_analysis" / "other"
    """
    nodes = Params.NODE_LIST

    if skill_hint and skill_hint.strip():
        skill = skill_hint.strip()
        if skill in ('project_code', 'general_code'):
            return 'code'
        if skill in nodes:
            return skill
        logger.warning(f"未知技能类型: {skill}，尝试通过 LLM 分类")

    prompts = """你是一个专业的客服助手，需要根据用户的问题进行任务分类。

分类规则：
1. 如果用户的问题是和旅游路线规划相关的，那就返回travel。
2. 如果用户的问题是和讲笑话相关的，那就返回joke。
3. 如果用户的问题是和对联相关的，那就返回couplet。
4. 如果用户的问题是和文档相关的（包含修改已有文档、新建文档并写入内容、排版、修正格式、修改对齐方式，查看文档中的错别字等等），那就返回document。
5. 如果用户的问题是和编程相关的，那就返回code。
6. 如果用户的问题是和工程图纸识别、尺寸提取、公差校核、DXF解析、国标查询、单位换算等相关的，那就返回drawing。
7. 如果是其它问题，那就返回other。
8. 如果用户是来咨询问题、排查报错、分析现象，返回problem_analysis。

只返回以上选项之一，不要返回任何其它内容。"""

    last_msg = messages_list[-1]
    if hasattr(last_msg, 'content'):
        message_content = last_msg.content
    else:
        message_content = str(last_msg)

    prompt_list = _build_prompt_list(messages_list, prompts, message_content, thread_id=thread_id)

    skill_required = {'code', 'document', 'drawing'}
    try:
        response = ""
        async for chunk_text in _stream_by_prompt_async(prompt_list, model=Params.DEFAULT_TEXT_TOOL_MODEL):
            response += chunk_text

        type_res = response.strip()
        if type_res in nodes:
            if type_res in skill_required and not skill_hint:
                return 'other'
            logger.info(f"LLM 分类结果: {type_res}")
            return type_res
    except Exception as e:
        logger.error(f"LLM 分类失败: {str(e)[:80]}")

    return 'other'


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


def _stream_by_prompt(prompt_list, model: str = None, temperature: float = 0.3):
    """同步包装：有 running loop 时在新线程跑，否则直接 asyncio.run"""
    import concurrent.futures

    async def _collect():
        chunks = []
        async for c in _stream_by_prompt_async(prompt_list, model=model, temperature=temperature):
            chunks.append(c)
        return chunks

    def _run_in_thread():
        return asyncio.run(_collect())

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_in_thread)
            chunks = future.result()
    except RuntimeError:
        chunks = asyncio.run(_collect())

    for c in chunks:
        yield c


async def stream_skill(skill_type: str, state_input: dict):
    """根据 skill 类型返回异步流式 generator，逐字 yield LLM 输出"""
    messages_list = state_input.get("messages", [])
    user_query = ""
    if messages_list:
        last = messages_list[-1]
        user_query = last.content if hasattr(last, 'content') else str(last)

    images_data = state_input.get("images")
    file_path = state_input.get("file_path")
    current_user = state_input.get("user", "")
    thread_id = state_input.get("thread_id")

    if skill_type == "problem_analysis":
        from core.skills.ProblemAnalyze.issue_analyze import IssueAnalyze
        analyzer = IssueAnalyze()
        for chunk in analyzer.stream_issue(user_query, images_data):
            yield {"chunk": chunk, "sources": []}
        return

    elif skill_type == "other":
        # 智能回答助手
        system_prompt = f"""你是一个专业的智能回答助手，需要根据用户问题来做出相应回应，回答要专业、清晰，使用自然友好的语言"""
        prompt_list = _build_prompt_list(messages_list, system_prompt, user_query, thread_id)
        full_text = ""
        async for chunk_text in _stream_by_prompt_async(prompt_list):
            full_text += chunk_text
            yield {"chunk": chunk_text, "sources": []}
        if not full_text:
            yield {"chunk": "抱歉，您咨询的问题不在我的能力范围内。", "sources": []}

        return

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
                agent = create_react_agent(model=get_llm(), tools=tools)
                async for chunk in agent.astream({"messages": prompt_list}):
                    yield {"chunk": chunk, "sources": []}
            else:
                raise ImportError("MultiServerMCPClient not available")
        except Exception as e:
            logger.warning(f"MCP 服务不可用，使用本地规划: {str(e)[:50]}")
            async for chunk_text in _stream_by_prompt_async(prompt_list):
                yield {"chunk": chunk_text, "sources": []}
        return

    elif skill_type == "joke":
        system_prompt = """你是一个专业的笑话大师，根据用户的问题，写一个有趣的中文笑话。不超过150字。"""
        prompt_list = _build_prompt_list(messages_list, system_prompt, user_query, thread_id=thread_id)
        async for chunk_text in _stream_by_prompt_async(prompt_list, model=Params.DEFAULT_TEXT_TOOL_MODEL):
            yield {"chunk": chunk_text, "sources": []}
        return

    elif skill_type == "couplet":
        _t0 = _time.time()
        logger.info(f"[PERF] couplet: 开始, user_query={user_query[:30]}")
        try:
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
                yield {"chunk": chunk_text, "sources": []}

        except Exception as e:
            logger.error(f"couplet 流式失败: {str(e)[:80]}")
            yield {"chunk": f"对联生成失败: {str(e)[:100]}", "sources": []}
        logger.info(f"[PERF] couplet: 总耗时 {_time.time()-_t0:.3f}s")
        return

    elif skill_type == "document":
        file_paths = state_input.get("file_paths") or ([state_input.get("file_path")] if state_input.get("file_path") else [])
        file_paths = [p for p in file_paths if p]

        logger.info(f" [stream_skill] file_paths={file_paths}")
        _t_import = _time.time()
        from core.tools.doc_tools import get_doc_system_prompt, create_doc_tools
        logger.info(f"[PERF] doc_tools import 耗时: {_time.time()-_t_import:.3f}s")
        
        _t_models = _time.time()
        # HumanMessage, AIMessage, SystemMessage, ToolMessage = _get_langchain_messages_full()
        logger.info(f"[PERF] _get_langchain_messages_full 耗时: {_time.time()-_t_models:.3f}s")

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
                    _role = "用户" if (hasattr(_m, 'type') and _m.type in ("human", "user")) or (isinstance(_m, dict) and _m.get("role") == "user") else "助手"
                    _content = _m.content if hasattr(_m, 'content') else (_m.get("content", "") if isinstance(_m, dict) else str(_m))
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
                    logger.info(f"[INTENT_CHECK] 用户是否引用生成文档: {_is_referring_generated} (LLM 回答: {_intent_result})")
                except Exception as e:
                    logger.warning(f"[INTENT_CHECK] 意图判断失败，回退到关键词匹配: {e}")
                    # 回退到关键词匹配
                    _generated_doc_keywords = ["生成的这个文档", "生成的文档", "刚才生成的", "合并后的文档", "合并后的文件", "生成的文件", "这个文档", "刚才那个文档", "刚才那个文件", "优化下格式", "调整下排版", "再改改"]
                    _is_referring_generated = any(kw in clean_user_request for kw in _generated_doc_keywords)
            
            # 如果是引用生成的文档，从对话历史中提取下载链接并自动下载到 cache 目录
            if _is_referring_generated and len(messages_list) > 1:
                _generated_file_path = None
                for msg in reversed(messages_list[:-1]):
                    msg_type = getattr(msg, 'type', None) or (msg.get("role") if isinstance(msg, dict) else None)
                    content = msg.content if hasattr(msg, 'content') else (msg.get("content", "") if isinstance(msg, dict) else str(msg))
                    
                    logger.debug(f"[GENERATED_DOC_DEBUG] msg_type={msg_type}, content[:300]={content[:300]}")
                    
                    # 从下载链接中提取 URL 和文件名（支持多种格式）
                    _dl_match = re.search(r'\[下载.*?文档.*?\]\((https?://[^)]+/([^/)]+\.(?:docx|pdf|txt|xlsx|pptx)))\)', content)
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
                _doc_tools = create_doc_tools()
                logger.info(f"[PERF] create_doc_tools 耗时: {_time.time()-_t_create:.3f}s")
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
                        user_msg_parts.append(f"附件{i+1}：{_fname}（路径：{_fp}，格式：{_file_ext}）")
                user_msg_parts.append("⚠️ 重要提示：你现在**没有任何一个文件的实际内容**，只有路径。你必须调用工具（比如 read_document_and_answer）才能读取文件内容并回答用户的问题。绝对不要在没调工具的情况下直接回复用户！")
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
            logger.info(f"[PERF] document(tool): LLM 工具选择耗时 {_time.time()-_t2:.3f}s")

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
                    logger.warning(f"[PERF] document(tool): LLM 未调工具但有文件+文档关键词，强制走 read_document_and_answer 兜底")
                    
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
                        logger.info(f"[FALLBACK] 读取文件内容成功，chars={len(_fallback_content)}")
                    except Exception as e:
                        logger.error(f"[FALLBACK] 读取文件内容失败: {e}")
                        _fallback_content = f"[读取失败：{e}]"
                    
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
                logger.info(f"[PERF] document(tool): 闲聊/非文档需求，直接回复, 耗时 {_time.time()-_t2:.3f}s")
                yield {"chunk": content_text}
                logger.info(f"[PERF] document(tool): 总耗时 {_time.time()-_t0:.3f}s")
                return

            yield {"step": {"title": "理解用户需求", "status": "running"}}
            yield {"step_detail": f"用户请求：{clean_user_request[:100]}"}
            if len(file_paths) == 1:
                yield {"step_detail": f"附带文件：{Path(file_paths[0]).name}"}
            elif len(file_paths) > 1:
                _names = [Path(p).name for p in file_paths]
                yield {"step_detail": f"附带 {len(file_paths)} 个文件：{', '.join(_names)}"}
            yield {"step": {"title": "理解用户需求", "status": "done"}}

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
                    yield {"step": {"title": f"读取文件 ({idx+1}/{len(file_paths)})", "status": "running"}}
                    try:
                        _c = await _read_one(_fp)
                        if not _c:
                            logger.warning(f"[FILE_READ] 文件 {_fname} 读取内容为空！")
                        _file_parts.append(f"===== 文件{idx+1}：{_fname} =====\n{_c}")
                        yield {"step_detail": f"{_fname}：{len(_c)} 字"}
                    except Exception as e:
                        logger.error(f"[FILE_READ] 读取文件失败 {_fp}: {e}")
                        import traceback
                        logger.error(f"[FILE_READ] 异常堆栈: {traceback.format_exc()}")
                        _file_parts.append(f"===== 文件{idx+1}：{_fname} =====\n[读取失败：{e}]")
                    yield {"step": {"title": f"读取文件 ({idx+1}/{len(file_paths)})", "status": "done"}}
                file_content = "\n\n".join(_file_parts)
                logger.info(f"[PERF] document(tool): 读取 {len(file_paths)} 个文件耗时 {_time.time()-_t1:.3f}s, chars={len(file_content)}")
                yield {"step_detail": f"共读取 {len(file_content)} 字"}

            yield {"step": {"title": "选择文档处理工具", "status": "running"}}
            yield {"step": {"title": "选择文档处理工具", "status": "done"}}

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

                yield {"step": {"title": f"调用工具：{tool_name}", "status": "running"}}

                _arg_preview = {}
                for k, v in tool_args.items():
                    if k in ("file_content", "content_description"):
                        _arg_preview[k] = f"<{len(str(v))} 字>"
                    elif k == "file_path":
                        _arg_preview[k] = Path(str(v)).name if v else ""
                    else:
                        _arg_preview[k] = str(v)[:80]
                yield {"step_detail": f"工具入参：{_arg_preview}"}

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
                        yield {"step_detail": _msg}
                    except asyncio.TimeoutError:
                        pass

                tool_result = await _tool_task
                _elapsed = _time.time() - _t3

                while not _progress_q.empty():
                    try:
                        _msg = _progress_q.get_nowait()
                        yield {"step_detail": _msg}
                    except asyncio.QueueEmpty:
                        break

                set_progress_queue(None)
                logger.info(f"[PERF] document(tool): {tool_name} 执行耗时 {_elapsed:.3f}s")

                _result_text = str(tool_result)

                _url_match = re.search(r'\[下载文档\]\((https?://[^)]+)\)', _result_text)
                if _url_match:
                    _final_download_url = _url_match.group(1)

                yield {"step_detail": f"工具执行完成，耗时 {_elapsed:.1f}s"}
                if _result_text:
                    _result_display = _result_text[:200] + ("..." if len(_result_text) > 200 else "")
                    yield {"step_detail": f"工具返回：{_result_display}"}

                messages.append({"role": "assistant", "content": content_text, "tool_calls": [
                    {"id": tc.get("id", ""), "type": "function", "name": tc.get("name", ""), "args": tc.get("args", {})}
                ]})
                messages.append({"role": "tool", "content": _result_text, "tool_call_id": tc.get("id", "")})

            yield {"step": {"title": f"调用工具：{tool_name}", "status": "done"}}
            yield {"step": {"title": "生成最终回复", "status": "running"}}

            _t_final = _time.time()
            logger.info(f"开始 LLM 生成最终聊天回复, messages 数={len(messages)}")

            _cleaned = _clean_messages_for_chat_reply(messages)
            logger.info(f"[PERF] 清理后聊天消息数={len(_cleaned)}")

            _yielded_any = False
            _full_response = ""
            async for chunk_text in _stream_by_prompt_async(
                _cleaned, model=Params.DEFAULT_CHAT_MODEL, temperature=0.5
            ):
                yield {"chunk": chunk_text}
                _full_response += chunk_text
                _yielded_any = True

            # 如果有下载链接，且 LLM 回复中没有包含下载链接，则追加
            if _final_download_url:
                # 检查 LLM 回复中是否已包含下载链接
                _has_download_link = any(kw in _full_response for kw in ["[下载", "download", "下载文档", "下载链接"])
                if not _has_download_link:
                    yield {"chunk": f"\n\n [下载文档]({_final_download_url})"}

            if not _yielded_any:
                if _final_download_url:
                    _final_text = f"✅ 文档已处理完成，点击下方链接下载：\n\n📥 [下载处理后的文档]({_final_download_url})"
                else:
                    _final_text = _result_text or "工具执行完成，但未返回结果。"
                yield {"chunk": _final_text}

            yield {"step": {"title": "生成最终回复", "status": "done"}}
            logger.info(f"[PERF] document(tool): LLM 生成回复耗时 {_time.time()-_t_final:.3f}s, 总耗时 {_time.time()-_t0:.3f}s, has_download={bool(_final_download_url)}")
        except asyncio.CancelledError:
            for _t in _tool_tasks:
                if not _t.done():
                    _t.cancel()
            logger.info("[STOP] document skill 被取消，已中止所有工具任务")
            raise
        except Exception as e:
            logger.error(f"document 处理失败: {str(e)[:80]}")
            yield {"chunk": f"文档处理失败: {str(e)[:100]}", "sources": []}
        return

    elif skill_type == "drawing":
        try:
            from core.skills.DrawingRecoAssistant.drawing_assistant import get_drawing_assistant
            file_path = state_input.get("file_path", "")
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
            result = assistant.invoke(
                user_query=user_query,
                img_path=img_path,
                pdf_path=pdf_path,
                dxf_path=dxf_path,
                thread_id="agent_draw_session",
            )
            answer = result["answer"] if isinstance(result, dict) else result
            sources = result.get("sources", []) if isinstance(result, dict) else []
            if answer:
                chunk_size = 20
                for i in range(0, len(answer), chunk_size):
                    yield {"chunk": answer[i:i + chunk_size], "sources": sources}
        except Exception as e:
            logger.error(f"drawing 处理失败: {str(e)[:80]}")
            yield {"chunk": f"图纸识别处理失败: {str(e)[:100]}", "sources": []}
        return

    elif skill_type == "code":
        logger.info("code skill: 直接处理")
        skill_hint = state_input.get("skill", "")
        project_context = state_input.get("project_context")
        messages = state_input.get("messages", [])
        file_path = state_input.get("file_path")

        if skill_hint == "project_code":
            from core.skills.AICode.project_programming import Project_Programming
            project_programming = Project_Programming(file_path, messages, project_context)
            message = messages[-1]
            full_content = ""
            chunk_count = 0
            for chunk in project_programming.get_model_response_stream(message, get_llm()):
                if chunk:
                    full_content += chunk
                    chunk_count += 1
                    logger.info(f"code stream chunk #{chunk_count}, len={len(chunk)}")
                    yield {"chunk": chunk, "sources": []}
            logger.info(f"project_code 完成, content_len={len(full_content)}, chunks={chunk_count}")
        else:
            from core.skills.AICode.general_programming import General_Programming
            general_programming = General_Programming(file_path, messages)
            message = messages[-1]
            response = general_programming.get_model_response(message, get_llm())
            content = general_programming.check_save_code(response, file_path)
            chunk_size = 20
            for i in range(0, len(content), chunk_size):
                yield {"chunk": content[i:i + chunk_size], "sources": []}
        return


