import os
import re
import uuid


os.environ.setdefault("DASHSCOPE_API_KEY", "")

import asyncio
import json
import httpx
import time as _time
from typing import Annotated, TypedDict
from operator import add
from pathlib import Path
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from settings.Define import Params, PathConfig
from settings.logger_manager import get_logger
from openai import OpenAI

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None

logger = get_logger(__name__)

openai_client = OpenAI(
    # 如果没有配置环境变量，请用阿里云百炼API Key替换：api_key="sk-xxx"
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=Params.API_BASE,
)

http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(120.0, connect=10.0),
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    headers={"Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY', '')}"},
)

llm = ChatTongyi(
    model=Params.DEFAULT_CHAT_MODEL,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=Params.API_BASE,
    timeout=60,  # 设置LLM调用超时时间为60秒
    streaming=True
)

from core.tools.doc_tools import create_doc_tools

logger.info(f"开始创建绑定工具的大模型客户端")
doc_tools = create_doc_tools()
llm_with_tools = ChatOpenAI(
    model=Params.DEFAULT_TEXT_TOOL_MODEL,
    temperature=0.2,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=Params.API_BASE,
).bind_tools(doc_tools)
logger.info(f"创建完毕")

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


def get_llm_by_model(model_id: str):
    """根据前端选择的模型ID创建对应的LLM实例"""
    model_id = model_id.lower()
    logger.info(f"根据模型ID创建LLM实例: {model_id}")
    
    if model_id == "deepseek" or model_id == "deepseek-v4-pro":
        return ChatOpenAI(
            model="deepseek-v4-pro",
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com/v1",
            timeout=120,
            streaming=True
        )
    elif model_id == "glm" or model_id == "glm-5.2-fast-preview":
        return ChatOpenAI(
            model="glm-5.2-fast-preview",
            api_key=os.getenv("GLM_API_KEY", ""),
            base_url="https://open.bigmodel.cn/api/paas/v4",
            timeout=120,
            streaming=True
        )
    elif model_id == "qwen_vl" or model_id == "qwen-vl-max":
        return ChatOpenAI(
            model="qwen-vl-max",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
            timeout=120
        )
    else:
        logger.warning(f"未知模型ID: {model_id}，使用默认模型 {Params.DEFAULT_CHAT_MODEL}")
        return ChatOpenAI(
            model=Params.DEFAULT_CHAT_MODEL,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
            timeout=120,
            streaming=True
        )

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str
    file_path: str  # 添加文件路径字段
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
    async with http_client.stream("POST", url, json=payload) as response:
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
        # sources = []
        # knowledge_context = ""
        # try:
        #     from core.knowledge.knowledge_manager import get_vector_store, get_all_users
        #     logger.info(f"[DEBUG] stream_skill other: 开始知识库检索, user_query={user_query[:60]}, current_user={current_user}")
        #     all_users = get_all_users()
        #     logger.info(f"[DEBUG] stream_skill other: all_users={all_users}")
        #     for user in all_users:
        #         try:
        #             logger.info(f"[DEBUG] stream_skill other: 检索用户 {user}")
        #             vs = get_vector_store(user)
        #             logger.info(f"[DEBUG] stream_skill other: vs={vs}")
        #             results = vs.similarity_search(user_query, k=2)
        #             logger.info(f"[DEBUG] stream_skill other: 用户 {user} 检索到 {len(results)} 条")
        #             sources.extend(results)
        #         except Exception as e:
        #             logger.warning(f"[DEBUG] stream_skill other: 用户 {user} 检索失败: {str(e)[:120]}")
        #             continue
        #     if current_user and current_user not in all_users:
        #         try:
        #             logger.info(f"[DEBUG] stream_skill other: 检索当前用户 {current_user}")
        #             vs = get_vector_store(current_user)
        #             results = vs.similarity_search(user_query, k=3)
        #             logger.info(f"[DEBUG] stream_skill other: 当前用户 {current_user} 检索到 {len(results)} 条")
        #             sources.extend(results)
        #         except Exception as e:
        #             logger.warning(f"[DEBUG] stream_skill other: 当前用户 {current_user} 检索失败: {str(e)[:120]}")
        #             pass
        #     logger.info(f"[DEBUG] stream_skill other: 汇总 sources 共 {len(sources)} 条")
        #     if sources:
        #         knowledge_context = "\n\n".join([doc.page_content for doc in sources])
        #         logger.info(f"[DEBUG] stream_skill other: knowledge_context 长度={len(knowledge_context)}")
        # except Exception as e:
        #     logger.warning(f"知识库检索失败: {str(e)[:80]}")
        #
        # logger.info(f"[DEBUG] stream_skill other: knowledge_context={knowledge_context}...")
#         if knowledge_context:
#             system_prompt = f"""你是一个专业的知识问答助手。请根据提供的参考资料回答用户的问题：
#
# 参考资料：
# {knowledge_context}
#
# 用户问题：{user_query}
#
# 要求：
# 1. 仔细阅读并分析参考资料，寻找与用户问题相关的信息
# 2. 如果找到相关信息，请基于这些信息进行专业、准确的回答
# 3. 如果参考资料中没有相关信息，请说明"抱歉，我无法回答这个问题"
# 4. 回答要简洁、清晰，使用自然友好的语言，不要提及"知识库"或"来源"等字样"""
#
#             full_sources = []
#             for doc in sources:
#                 full_sources.append({
#                     "source": doc.metadata.get('source', '未知'),
#                     "content": doc.page_content,
#                 })
#
#             prompt_list = _build_prompt_list(messages_list, system_prompt, user_query, thread_id=thread_id)
#             async for chunk_text in _stream_by_prompt_async(prompt_list):
#                 yield {"chunk": chunk_text, "sources": full_sources}
#         else:
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
        file_path = state_input.get("file_path")
        from core.tools.doc_tools import get_doc_system_prompt
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

        _tool_tasks = []

        try:
            clean_user_request = user_query
            if '[用户要求]' in user_query:
                clean_user_request = user_query.split('[用户要求]')[-1].strip()
            for meta_tag in ['[原始文件格式]', '[文档内容]', '[原始文件]']:
                if meta_tag in clean_user_request:
                    clean_user_request = clean_user_request.split(meta_tag)[0].strip()

            _t0 = _time.time()
            tools_map = {t.name: t for t in doc_tools}

            system_prompt = get_doc_system_prompt()
            user_msg_parts = []
            if file_path:
                _file_ext = Path(file_path).suffix.lower()
                user_msg_parts.append(f"附件文件路径：{file_path}")
                user_msg_parts.append(f"附件文件格式：{_file_ext}")
            user_msg_parts.append(f"用户请求：{clean_user_request}")

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="\n\n".join(user_msg_parts)),
            ]

            logger.info("开始LLM推理（静默判断是否需要调工具）")
            _t2 = _time.time()

            ai_response = await llm_with_tools.ainvoke(messages)
            logger.info(f"[PERF] document(tool): LLM 工具选择耗时 {_time.time()-_t2:.3f}s")

            tool_calls = getattr(ai_response, "tool_calls", None) or []
            content_text = getattr(ai_response, "content", "") or ""

            if not tool_calls:
                logger.info(f"[PERF] document(tool): 闲聊/非文档需求，直接回复, 耗时 {_time.time()-_t2:.3f}s")
                yield {"chunk": content_text}
                logger.info(f"[PERF] document(tool): 总耗时 {_time.time()-_t0:.3f}s")
                return

            yield {"step": {"title": "理解用户需求", "status": "running"}}
            yield {"step_detail": f"用户请求：{clean_user_request[:100]}"}
            if file_path:
                _fname = Path(file_path).name
                yield {"step_detail": f"附带文件：{_fname}"}
            yield {"step": {"title": "理解用户需求", "status": "done"}}

            file_content = ""
            ext = ""
            if file_path:
                from core.skills.DocProcess.document_process import doc_processor

                ext = Path(file_path).suffix.lower()
                _filename = Path(file_path).name

                yield {"step": {"title": f"读取 {ext} 文件内容", "status": "running"}}
                _t1 = _time.time()
                if ext in ['.xlsx', '.xls']:
                    async for chunk_text in doc_processor._read_excel(file_path):
                        file_content += chunk_text
                elif ext == '.docx':
                    docx_paragraph_info = []
                    from docx import Document as DocxDoc
                    _doc = DocxDoc(file_path)
                    for pi, para in enumerate(_doc.paragraphs):
                        if para.text.strip():
                            docx_paragraph_info.append((pi, para.text))
                    file_content = '\n'.join(t for _, t in docx_paragraph_info)
                elif ext == '.txt':
                    async for chunk_text in doc_processor._read_txt(file_path):
                        file_content += chunk_text
                elif ext in ['.pptx', '.ppt']:
                    async for chunk_text in doc_processor._read_pptx(file_path):
                        file_content += chunk_text
                elif ext == '.doc':
                    async for chunk_text in doc_processor._read_doc(file_path):
                        file_content += chunk_text
                elif ext == '.pdf':
                    async for chunk_text in doc_processor.read_document(file_path):
                        file_content += chunk_text
                elif ext == '.json':
                    import json
                    try:
                        file_content = ""
                        with open(file_path, "r", encoding="utf-8") as f:
                            file_content = json.load(f)

                        file_content = json.dumps(file_content, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        file_content = Path(file_path).read_text(encoding="utf-8")
                else:
                    raise ValueError(f"不支持的文件格式: {ext}")
                logger.info(f"[PERF] document(tool): 读取文件耗时 {_time.time()-_t1:.3f}s, ext={ext}, chars={len(file_content)}")
                yield {"step_detail": f"已读取 {len(file_content)} 字"}
                yield {"step": {"title": f"读取 {ext} 文件内容", "status": "done"}}

            yield {"step": {"title": "选择文档处理工具", "status": "running"}}
            yield {"step": {"title": "选择文档处理工具", "status": "done"}}

            logger.info("工具选择已完成，准备执行工具")

            from core.tools.doc_tools import set_progress_queue

            _final_download_url = ""

            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})

                if tool_name == "modify_existing_document":
                    if file_path:
                        tool_args.setdefault("file_path", file_path)
                    if file_content:
                        tool_args["file_content"] = file_content
                elif tool_name == "convert_file_format":
                    if file_path:
                        tool_args.setdefault("file_path", file_path)
                    if file_content:
                        tool_args["file_content"] = file_content

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

                messages.append(ai_response)
                messages.append(ToolMessage(content=_result_text, tool_call_id=tc.get("id", "")))

            yield {"step": {"title": f"调用工具：{tool_name}", "status": "done"}}
            yield {"step": {"title": "生成最终回复", "status": "running"}}

            _t_final = _time.time()
            logger.info(f"开始 LLM 生成最终聊天回复, messages 数={len(messages)}")

            _cleaned = _clean_messages_for_chat_reply(messages)
            logger.info(f"[PERF] 清理后聊天消息数={len(_cleaned)}")

            _yielded_any = False
            async for chunk_text in _stream_by_prompt_async(
                _cleaned, model=Params.DEFAULT_CHAT_MODEL, temperature=0.5
            ):
                yield {"chunk": chunk_text}
                _yielded_any = True

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
            for chunk in project_programming.get_model_response_stream(message, llm):
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
            response = general_programming.get_model_response(message, llm)
            content = general_programming.check_save_code(response, file_path)
            chunk_size = 20
            for i in range(0, len(content), chunk_size):
                yield {"chunk": content[i:i + chunk_size], "sources": []}
        return