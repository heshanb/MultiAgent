import json
import threading
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from core.auth.auth import get_current_user
from core.auth.database import get_db, User, Agent
from core.knowledge.knowledge_manager import get_user_knowledge_dir, get_vector_store, EMBEDDING_DATABASE, convert_vector_score
from settings.logger_manager import get_logger

logger = get_logger(__name__)

MEMORY_MAX_TURNS = 10

# 懒加载 LangChain 库
_InMemoryChatMessageHistory = None
_HumanMessage = None
_AIMessage = None
_SystemMessage = None

# 会话记忆存储
_chat_memories = {}
_memories_lock = threading.Lock()


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


def _trim_history(history, max_turns: int = MEMORY_MAX_TURNS):
    messages = history.messages
    if len(messages) > max_turns * 2:
        keep_from = len(messages) - max_turns * 2
        history.clear()
        for m in messages[keep_from:]:
            history.add_message(m)

router = APIRouter(prefix="/api/agents", tags=["agents"])


class AgentCreate(BaseModel):
    """创建智能体的请求体。

    用户首次新建智能体时提交，只需提供基础信息（名称必填，其他可选）。
    其他配置项（system_prompt / model / knowledge_base 等）通过 AgentUpdate 补全。
    """
    name: str
    description: Optional[str] = None
    avatar: Optional[str] = "🤖"


class AgentUpdate(BaseModel):
    """更新智能体的请求体。

    所有字段均可选，前端只传需要修改的字段，后端做部分更新。
    用于编辑器里修改 system_prompt、模型、知识库、变量、开场白等全部配置。
    """
    name: Optional[str] = None
    description: Optional[str] = None
    avatar: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    variables: Optional[list] = None
    knowledge_base: Optional[list] = None
    opening: Optional[str] = None
    presets: Optional[list] = None
    is_published: Optional[bool] = None
    deep_think: Optional[bool] = None


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
    opening: Optional[str] = None
    presets: Optional[list] = None
    is_published: bool
    category: Optional[str] = None
    publish_version: Optional[str] = None
    publish_desc: Optional[str] = None
    publish_channels: Optional[str] = None
    deep_think: Optional[bool] = False
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


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
    """
    message: str
    variables: Optional[dict] = None
    system_prompt: Optional[str] = None
    session_id: Optional[str] = None
    deep_think: Optional[bool] = False
    history: Optional[list] = None


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


@router.post("", response_model=AgentResponse)
async def create_agent(
    data: AgentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not data.name or not data.name.strip():
        raise HTTPException(status_code=400, detail="智能体名称不能为空")
    if len(data.name.strip()) > 100:
        raise HTTPException(status_code=400, detail="智能体名称不能超过100字符")

    name_stripped = data.name.strip()
    existing = db.query(Agent).filter(
        Agent.user_id == current_user.id,
        Agent.name == name_stripped
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="已存在同名的智能体，请更换名称")

    agent = Agent(
        user_id=current_user.id,
        name=name_stripped,
        description=data.description,
        avatar=data.avatar or "🤖",
        system_prompt="",
        variables="[]",
        knowledge_base="[]",
        is_published=False,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)

    logger.info(f"用户 {current_user.username} 创建智能体: {agent.name} (id={agent.id})")
    return AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        avatar=agent.avatar,
        system_prompt=agent.system_prompt,
        variables=_deserialize_vars(agent.variables),
        knowledge_base=_deserialize_vars(agent.knowledge_base),
        opening=agent.opening,
        presets=_deserialize_vars(agent.presets),
        is_published=agent.is_published,
        category=agent.category,
        publish_version=agent.publish_version,
        publish_desc=agent.publish_desc,
        publish_channels=agent.publish_channels,
        deep_think=agent.deep_think,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


@router.get("", response_model=List[AgentResponse])
async def list_agents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    agents = db.query(Agent).filter(Agent.user_id == current_user.id).order_by(
        Agent.updated_at.desc()
    ).all()

    result = []
    for a in agents:
        result.append(AgentResponse(
            id=a.id,
            name=a.name,
            description=a.description,
            avatar=a.avatar,
            system_prompt=a.system_prompt,
            variables=_deserialize_vars(a.variables),
            knowledge_base=_deserialize_vars(a.knowledge_base),
            opening=a.opening,
            presets=_deserialize_vars(a.presets),
            is_published=a.is_published,
            category=a.category,
            publish_version=a.publish_version,
            publish_desc=a.publish_desc,
            publish_channels=a.publish_channels,
            deep_think=a.deep_think,
            created_at=a.created_at,
            updated_at=a.updated_at,
        ))
    return result


AVAILABLE_MODELS = [
    {"id": "glm", "name": "GLM-5.2 Fast Preview", "desc": "智谱开源，速度快"},
    {"id": "qwen_vl", "name": "Qwen VL Max", "desc": "阿里百炼，多模态"},
    {"id": "qwen_turbo", "name": "Qwen turbo", "desc": "稳定通用，均衡表现"},
    {"id": "default", "name": "Deepseek V4 Pro-0813", "desc": "深度思考，能力最强"},
]


@router.get("/models")
async def get_available_models():
    return {"models": AVAILABLE_MODELS}


@router.post("/prompt-optimize")
async def optimize_prompt(
    data: PromptOptimizeRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
):
    import asyncio
    from langchain_openai import ChatOpenAI
    _, _, _, SystemMessage, HumanMessage = _get_langchain_classes()
    from settings.Define import Params
    import os

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
            f"1. 每一处提到\"知识库\"的地方，必须紧跟具体的知识库名称，格式为：知识库`具体名称`\n"
            f"   例如：将\"结合知识库检索结果\"改为\"结合知识库`{kb_names_for_prompt[0]}`检索结果\"\n"
            f"2. 每一处提到知识库\"内容\"或\"检索结果\"的地方，必须在后面紧跟 {{{{knowledge_content}}}} 内置变量\n"
            f"   例如：将\"引用知识库中的内容\"改为\"引用知识库`{kb_names_for_prompt[0]}`中的内容{{{{knowledge_content}}}}\"\n"
            f"3. 这两条规则不可省略，必须严格执行"
        )

    user_msg_prefix = "调优以下提示词" if is_targeted else "优化以下提示词"
    if is_targeted and data.instruction and data.instruction.strip():
        user_msg_prefix += f"，使其更贴近目标：{data.instruction.strip()}"
    user_msg = f"{user_msg_prefix}\n\n---\n{data.prompt or '(空，帮我从零开始写一个好的系统提示词)'}\n---"
    logger.debug(f"[mode={data.mode}] user_msg: {user_msg}")

    async def generate():
        from langchain_openai import ChatOpenAI
        _, _, _, SystemMessage, HumanMessage = _get_langchain_classes()
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


@router.get("/knowledge")
async def list_knowledge_documents(
    current_user: User = Depends(get_current_user),
):
    from core.knowledge.knowledge_manager import load_kb_metadata
    meta_list = load_kb_metadata(current_user.username)
    bases = []
    for kb in meta_list:
        bases.append({
            "id": kb['id'],
            "name": kb.get('name', '未命名知识库'),
        })
    return {"documents": bases}


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权访问该智能体")

    return AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        avatar=agent.avatar,
        system_prompt=agent.system_prompt,
        variables=_deserialize_vars(agent.variables),
        knowledge_base=_deserialize_vars(agent.knowledge_base),
        opening=agent.opening,
        presets=_deserialize_vars(agent.presets),
        is_published=agent.is_published,
        category=agent.category,
        publish_version=agent.publish_version,
        publish_desc=agent.publish_desc,
        publish_channels=agent.publish_channels,
        deep_think=agent.deep_think,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: int,
    data: AgentUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权修改该智能体")

    update_fields = data.model_dump(exclude_unset=True)
    if "name" in update_fields and update_fields["name"] is not None:
        new_name = update_fields["name"].strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="智能体名称不能为空")
        if len(new_name) > 100:
            raise HTTPException(status_code=400, detail="智能体名称不能超过100字符")
        duplicate = db.query(Agent).filter(
            Agent.user_id == current_user.id,
            Agent.name == new_name,
            Agent.id != agent_id
        ).first()
        if duplicate:
            raise HTTPException(status_code=400, detail="已存在同名的智能体，请更换名称")
        update_fields["name"] = new_name
    if "variables" in update_fields:
        update_fields["variables"] = _serialize_vars(update_fields["variables"])
    if "knowledge_base" in update_fields:
        update_fields["knowledge_base"] = _serialize_vars(update_fields["knowledge_base"])
    if "presets" in update_fields:
        update_fields["presets"] = _serialize_vars(update_fields["presets"])

    for k, v in update_fields.items():
        if k == "is_published":
            agent.is_published = v
        else:
            setattr(agent, k, v)

    agent.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(agent)

    logger.info(f"用户 {current_user.username} 更新智能体: {agent.name} (id={agent.id})")

    return AgentResponse(
        id=agent.id,
        name=agent.name,
        description=agent.description,
        avatar=agent.avatar,
        system_prompt=agent.system_prompt,
        variables=_deserialize_vars(agent.variables),
        knowledge_base=_deserialize_vars(agent.knowledge_base),
        opening=agent.opening,
        presets=_deserialize_vars(agent.presets),
        is_published=agent.is_published,
        category=agent.category,
        publish_version=agent.publish_version,
        publish_desc=agent.publish_desc,
        publish_channels=agent.publish_channels,
        deep_think=agent.deep_think,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权删除该智能体")

    name = agent.name
    db.delete(agent)
    db.commit()

    logger.info(f"用户 {current_user.username} 删除智能体: {name} (id={agent_id})")
    return {"message": "删除成功"}


class PublishRequest(BaseModel):
    """智能体发布请求体。

    用于 publish_agent 接口。
    version: 发布版本号（如 v1.0.0），前端填写后写入 Agent.publish_version
    publish_desc: 本次发布的说明文字
    channels: 发布渠道列表，如 ["wechat_mini", "wechat_kf", "xiaozhi_store"]
    """
    version: Optional[str] = None
    publish_desc: Optional[str] = None
    category: Optional[str] = None
    channels: Optional[list] = None


@router.post("/{agent_id}/publish")
async def publish_agent(
    agent_id: int,
    data: PublishRequest = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权发布该智能体")

    if not agent.name:
        raise HTTPException(status_code=400, detail="请先填写智能体名称")

    agent.is_published = True
    if data:
        if data.version:
            agent.publish_version = data.version.strip()[:50]
        if data.publish_desc:
            agent.publish_desc = data.publish_desc.strip()
        if data.category:
            agent.category = data.category.strip()[:50]
        if data.channels:
            agent.publish_channels = json.dumps(data.channels, ensure_ascii=False)
    agent.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(agent)

    logger.info(f"用户 {current_user.username} 发布智能体: {agent.name} (id={agent.id}) version={agent.publish_version}")
    return {"message": "发布成功", "is_published": True, "publish_version": agent.publish_version, "publish_desc": agent.publish_desc}


@router.post("/{agent_id}/unpublish")
async def unpublish_agent(
    agent_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权操作该智能体")

    agent.is_published = False
    agent.updated_at = datetime.utcnow()
    db.commit()

    logger.info(f"用户 {current_user.username} 取消发布智能体: {agent.name} (id={agent_id})")
    return {"message": "已取消发布", "is_published": False}


@router.get("/store/list")
async def store_list(
    db: Session = Depends(get_db),
):
    """小智商店 - 列出所有已发布的智能体"""
    agents = db.query(Agent).filter(
        Agent.is_published == True,
        Agent.name != None,
        Agent.name != ""
    ).order_by(Agent.updated_at.desc()).all()

    user_ids = {a.user_id for a in agents if a.user_id}
    users_map = {}
    if user_ids:
        users = db.query(User).filter(User.id.in_(user_ids)).all()
        users_map = {u.id: u.username for u in users}

    result = []
    for a in agents:
        channels = []
        if a.publish_channels:
            try:
                channels = json.loads(a.publish_channels)
            except Exception:
                channels = []
        author_name = users_map.get(a.user_id, "匿名用户")
        result.append({
            "id": a.id,
            "name": a.name,
            "description": a.description or "",
            "avatar": a.avatar or "🤖",
            "author": author_name,
            "category": a.category or "other",
            "publish_version": a.publish_version or "",
            "publish_desc": a.publish_desc or "",
            "publish_channels": channels,
            "is_official": True,
            "is_free": True,
            "updated_at": a.updated_at.isoformat() if a.updated_at else "",
        })
    return result


@router.get("/store/{agent_id}")
async def store_agent_detail(
    agent_id: int,
    db: Session = Depends(get_db),
):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")
    if not agent.is_published:
        raise HTTPException(status_code=404, detail="该智能体未发布")

    from core.auth.database import User
    owner = db.query(User).filter(User.id == agent.user_id).first()
    owner_name = owner.username if owner else "匿名"

    channels = []
    if agent.publish_channels:
        try:
            channels = json.loads(agent.publish_channels)
        except Exception:
            channels = []

    return {
        "id": agent.id,
        "name": agent.name,
        "description": agent.description or "",
        "avatar": agent.avatar or "🤖",
        "system_prompt": agent.system_prompt or "",
        "model": agent.model or "default",
        "variables": _deserialize_vars(agent.variables),
        "opening": agent.opening or "",
        "publish_version": agent.publish_version or "",
        "publish_desc": agent.publish_desc or "",
        "publish_channels": channels,
        "deep_think": agent.deep_think or False,
        "category": agent.category or "other",
        "owner_name": owner_name,
        "created_at": agent.created_at.strftime("%Y-%m-%d %H:%M") if agent.created_at else "",
        "updated_at": agent.updated_at.isoformat() if agent.updated_at else "",
    }


def _build_rag_context(agent: Agent, user_username: str, query: str, log_tag: str = "") -> str:
    knowledge_base = _deserialize_vars(agent.knowledge_base)
    if not knowledge_base:
        return ""

    print(f"_build_rag_context: knowledge_base={knowledge_base}")
    kb_content_text = ""
    try:
        from core.knowledge.knowledge_manager import get_kb_file_rag_configs

        # 获取该知识库每个文档的 RAG 配置
        file_rag_configs = get_kb_file_rag_configs(user_username, knowledge_base[0])

        documents = []
        # 对每个文档用各自的向量模型和检索方式分别检索
        for filename, rc in file_rag_configs.items():
            embedding_model = rc.get('embedding_model_name')
            rag_method = rc.get('rag_retrieve_method', 'similarity')
            # rag_retrieve_method: "similarity" -> 向量检索(DENSE), "hybrid" -> 混合检索
            retrieval_mode = "hybrid" if rag_method == "hybrid" else "vector"
            vector_store = get_vector_store(user_username, embedding_model, retrieval_mode)

            def _do_search(vs, kb_id, fname, use_filter=True):
                kwargs = {"k": 5}
                if use_filter:
                    kwargs["filter"] = {"kb_id": kb_id, "source": fname}
                if EMBEDDING_DATABASE == "chroma":
                    return vs.similarity_search_with_relevance_scores(query, **kwargs)
                else:
                    return vs.similarity_search_with_score(query, **kwargs)

            retrieved = _do_search(vector_store, knowledge_base[0], filename, use_filter=True)
            if not retrieved:
                # 降级：只按 kb_id 过滤
                kwargs = {"k": 5, "filter": {"kb_id": knowledge_base[0]}}
                if EMBEDDING_DATABASE == "chroma":
                    retrieved = vector_store.similarity_search_with_relevance_scores(query, **kwargs)
                else:
                    retrieved = vector_store.similarity_search_with_score(query, **kwargs)

            is_chroma = EMBEDDING_DATABASE == "chroma"
            for doc, raw in retrieved or []:
                score = raw if is_chroma else convert_vector_score(raw, vector_store, EMBEDDING_DATABASE)
                if score < 0.2:
                    continue
                documents.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', ''),
                    "kb_id": doc.metadata.get('kb_id', ''),
                    "score": float(score)
                })

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
                    from FlagEmbedding import FlagReranker
                    rerank_model = rerank_cfg.get('rerank_model_setting', {}).get('rerank_model', 'bge-reranker-large')
                    reranker = FlagReranker(rerank_model, use_fp16=True)
                    pairs = [[query, d['content']] for d in documents]
                    scores = reranker.compute_score(pairs)
                    if not isinstance(scores, list):
                        scores = [scores]
                    for d, s in zip(documents, scores):
                        d['score'] = float(s)
                    documents.sort(key=lambda x: x['score'], reverse=True)
                    top_k = rerank_cfg.get('rerank_model_setting', {}).get('rerank_top_k', 3)
                    documents = documents[:top_k]
                except Exception as rerank_err:
                    logger.warning(f"[{log_tag}] Rerank 失败，使用原始排序: {rerank_err}")

            content_parts = []
            for i, doc in enumerate(documents):
                doc['rank'] = i + 1
                source = doc.get('source', '')
                content_parts.append(f"[{doc['rank']}] 来源: {source}\n{doc['content']}")
            kb_content_text = "\n---参考知识---\n" + "\n".join(content_parts)
    except Exception as e:
        logger.warning(f"[{log_tag}] 知识库检索失败: {str(e)}")

    return kb_content_text

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


async def _make_chat_generator(llm, messages, data: AgentChatRequest, agent_id: int, session_id: str, http_request, log_tag: str = ""):
    """构造一个 async generator，用于 StreamingResponse 流式输出 AI 回复。

    调用方式（外层 async def 需 await 拿到内部 generate 函数）：
        gen_func = await _make_chat_generator(...)
        return StreamingResponse(gen_func(), media_type="text/event-stream")

    流式协议：
        每个 chunk: {"text": full_text, "status": "streaming"}
        结束时:     {"text": full_text, "status": "done"}
        被中断:     {"text": full_text, "status": "stop"}
        出错:       {"text": str(e), "status": "error"}
    """
    import asyncio
    import json as _json
    _, HumanMessage, AIMessage, _ = _get_langchain_classes()

    async def generate():
        full_text = ""
        try:
            async for chunk in llm.astream(messages):
                if await http_request.is_disconnected():
                    break
                token = chunk.content or ""
                if token:
                    full_text += token
                    yield f"data: {_json.dumps({'text': full_text, 'status': 'streaming'}, ensure_ascii=False)}\n\n"
            if not data.history:
                session_key = f"agent_{agent_id}_{session_id or 'default'}"
                history = _get_chat_history(session_key)
                history.add_message(HumanMessage(content=data.message))
                history.add_message(AIMessage(content=full_text))
                _trim_history(history)
            yield f"data: {_json.dumps({'text': full_text, 'status': 'done'}, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            yield f"data: {_json.dumps({'text': full_text, 'status': 'stop'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"[{log_tag}] error: {e}")
            yield f"data: {_json.dumps({'text': str(e), 'status': 'error'}, ensure_ascii=False)}\n\n"

    return generate()


@router.post("/store/{agent_id}/chat")
async def store_chat_with_agent(
    agent_id: int,
    data: AgentChatRequest,
    http_request: Request,
    db: Session = Depends(get_db),
):
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")
    if not agent.is_published:
        raise HTTPException(status_code=404, detail="该智能体未发布")
    if not data.message or not data.message.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    owner_user = db.query(User).filter(User.id == agent.user_id).first()

    kb_text = _build_rag_context(agent, owner_user.username if owner_user else "", data.message, log_tag="store_chat")
    messages = _build_messages(agent, data, kb_text)

    from core.agent import get_llm_by_model
    llm = get_llm_by_model(agent.model or "default", deep_think=data.deep_think)

    logger.info(f"[store_chat] messages_count={len(messages)}, kb_len={len(kb_text)}")

    from fastapi.responses import StreamingResponse
    gen = await _make_chat_generator(llm, messages, data, agent_id, data.session_id, http_request, log_tag="store")
    return StreamingResponse(gen, media_type="text/event-stream")


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
    logger.info(f"chat_with_agent kb_id={kb_id}")
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="智能体不存在")
    if agent.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="无权使用该智能体")
    if not data.message or not data.message.strip():
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    import asyncio
    import json as _json
    from core.agent import get_llm_by_model
    _, HumanMessage, AIMessage, SystemMessage = _get_langchain_classes()

    system_prompt = data.system_prompt if (data.system_prompt is not None and data.system_prompt != "") else (agent.system_prompt or "")
    if data.system_prompt is None or data.system_prompt == "":
        variables = _deserialize_vars(agent.variables)
        if data.variables:
            for var in variables:
                key = var.get("key")
                if key and key in data.variables:
                    val = str(data.variables[key])
                    system_prompt = system_prompt.replace(f"{{{{{key}}}}}", val)
                    system_prompt = system_prompt.replace(f"{{{key}}}", val)

    kb_content_text = _build_rag_context(agent, current_user.username, data.message, log_tag="chat")
    if kb_content_text:
        system_prompt = system_prompt.replace("{{knowledge_content}}", kb_content_text)
        system_prompt = system_prompt.replace("{knowledge_content}", kb_content_text)

    if data.history and len(data.history) > 0:
        messages = [SystemMessage(content=system_prompt)]
        for item in data.history:
            role = item.get('role', '')
            content = item.get('content', '')
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=data.message))
        llm = get_llm_by_model(agent.model or "default", deep_think=data.deep_think)
        logger.info(f"[chat_with_agent] 使用前端 history={len(data.history)}条, kb_len={len(kb_content_text)}")

        from fastapi.responses import StreamingResponse
        gen = await _make_chat_generator(llm, messages, data, agent_id, data.session_id, http_request, log_tag="chat")
        return StreamingResponse(gen, media_type="text/event-stream")

    session_key = data.session_id or f"user_{current_user.id}_agent_{agent.id}"
    history = _get_chat_history(session_key)
    _trim_history(history)

    if not history.messages and agent.opening and agent.opening.strip():
        history.add_ai_message(agent.opening.strip())
        logger.info(f"新会话，注入开场白到记忆: {agent.opening[:50]}...")

    logger.info(f"会话 {session_key} 历史轮数={len(history.messages)//2}")

    async def generate():
        try:
            if data.deep_think:
                logger.info("深度思考模式已开启")
            llm = get_llm_by_model(agent.model or "default", 0.3, deep_think=data.deep_think)

            history.add_user_message(data.message)

            langchain_msgs = [SystemMessage(content=system_prompt)]
            langchain_msgs.extend(history.messages)

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
            logger.error(f"智能体对话失败: {str(e)}")
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