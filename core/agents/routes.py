import json
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from core.auth.auth import get_current_user
from core.auth.database import get_db, User, Agent
from core.knowledge.knowledge_manager import get_user_knowledge_dir, get_vector_store
from settings.logger_manager import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])


class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    avatar: Optional[str] = "🤖"


class AgentUpdate(BaseModel):
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


class AgentResponse(BaseModel):
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
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AgentChatRequest(BaseModel):
    message: str
    variables: Optional[dict] = None


class PromptOptimizeRequest(BaseModel):
    prompt: str
    instruction: Optional[str] = None


def _deserialize_vars(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


def _serialize_vars(data: Optional[list]) -> Optional[str]:
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
        is_published=agent.is_published,
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
            is_published=a.is_published,
            created_at=a.created_at,
            updated_at=a.updated_at,
        ))
    return result


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
        is_published=agent.is_published,
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
        is_published=agent.is_published,
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


@router.post("/{agent_id}/publish")
async def publish_agent(
    agent_id: int,
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
    agent.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(agent)

    logger.info(f"用户 {current_user.username} 发布智能体: {agent.name} (id={agent.id})")
    return {"message": "发布成功", "is_published": True}


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


AVAILABLE_MODELS = [
    {"id": "deepseek", "name": "DeepSeek V4 Pro", "desc": "深度思考，能力最强"},
    {"id": "glm", "name": "GLM-5.2 Fast Preview", "desc": "智谱开源，速度快"},
    {"id": "qwen_vl", "name": "Qwen VL Max", "desc": "阿里百炼，多模态"},
    {"id": "default", "name": "通义千问 (默认)", "desc": "稳定通用，均衡表现"},
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
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.callbacks import AsyncIteratorCallbackHandler
    from settings.Define import Params
    import os

    base_prompt = (
        "你是一个世界顶级的 Prompt 工程师。你的任务是把用户提供的简单提示词，"
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
        base_prompt += f"\n\n用户额外的优化要求：{data.instruction.strip()}"

    user_msg = f"请优化以下提示词：\n\n---\n{data.prompt or '(空，帮我从零开始写一个好的系统提示词)'}\n---"

    async def generate():
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
                if http_request.is_disconnected():
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
    user_dir = get_user_knowledge_dir(current_user.username)
    meta_list = load_kb_metadata(current_user.username)
    kb_name_map = {kb['id']: kb['name'] for kb in meta_list}

    documents = []
    idx = 0
    for subdir in user_dir.iterdir() if user_dir.exists() else []:
        if subdir.is_dir() and subdir.name.startswith('kb_'):
            for file_path in subdir.iterdir():
                if file_path.is_file():
                    filename = file_path.name
                    if not (filename.startswith('~$') or filename.startswith('.') or filename == 'Thumbs.db'):
                        idx += 1
                        ext = file_path.suffix.lower().lstrip('.')
                        kb_name = kb_name_map.get(subdir.name, subdir.name)
                        documents.append({
                            "id": idx,
                            "filename": filename,
                            "file_type": ext,
                            "size": file_path.stat().st_size,
                            "modified_at": file_path.stat().st_mtime,
                            "kb_id": subdir.name,
                            "kb_name": kb_name,
                        })
    return {"documents": documents}


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


@router.post("/{agent_id}/chat")
async def chat_with_agent(
    agent_id: int,
    data: AgentChatRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
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
    from core.skills.DocProcess.document_process import doc_processor

    system_prompt = agent.system_prompt or ""
    variables = _deserialize_vars(agent.variables)
    knowledge_base = _deserialize_vars(agent.knowledge_base)

    if data.variables:
        for var in variables:
            key = var.get("key")
            if key and key in data.variables:
                system_prompt = system_prompt.replace(f"{{{{{key}}}}}", str(data.variables[key]))

    kb_context = ""
    if knowledge_base:
        try:
            vector_store = get_vector_store(current_user.username)
            retrieved = vector_store.similarity_search(data.message, k=3)
            if retrieved:
                kb_context = "\n\n---参考知识---\n"
                for i, doc in enumerate(retrieved, 1):
                    source = doc.metadata.get("source", "")
                    kb_context += f"[{i}] 来源: {source}\n{doc.page_content}\n\n"
        except Exception as e:
            logger.warning(f"知识库检索失败: {str(e)}")

    full_prompt = ""
    if system_prompt:
        full_prompt += f"[系统角色设定]\n{system_prompt}\n\n"
    if kb_context:
        full_prompt += f"{kb_context}\n"
    full_prompt += f"[用户问题]\n{data.message}"

    async def generate():
        try:
            llm = get_llm_by_model(agent.model or "")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if kb_context:
                messages.append({
                    "role": "system",
                    "content": f"参考以下知识库内容来回答用户问题：\n{kb_context}"
                })
            messages.append({"role": "user", "content": data.message})

            full_response = ""
            try:
                for chunk in llm.stream(messages):
                    if await http_request.is_disconnected():
                        break
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if text:
                        full_response += text
                        yield f"data: {_json.dumps({'text': text, 'status': 'streaming'})}\n\n"
                yield f"data: {_json.dumps({'text': '', 'status': 'done'})}\n\n"
            except asyncio.CancelledError:
                yield f"data: {_json.dumps({'text': '', 'status': 'stop'})}\n\n"
        except Exception as e:
            logger.error(f"智能体对话失败: {str(e)}")
            yield f"data: {_json.dumps({'text': f'智能体执行出错: {str(e)}', 'status': 'error'})}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(generate(), media_type="text/event-stream")