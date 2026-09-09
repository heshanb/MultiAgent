import json
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
    system_prompt: Optional[str] = None


class PromptOptimizeRequest(BaseModel):
    prompt: str
    instruction: Optional[str] = None
    mode: Optional[str] = "auto"
    knowledge_bases: Optional[list] = None


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
        opening=agent.opening,
        presets=_deserialize_vars(agent.presets),
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
            opening=a.opening,
            presets=_deserialize_vars(a.presets),
            is_published=a.is_published,
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
    from langchain_core.messages import SystemMessage, HumanMessage
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
    version: Optional[str] = None
    publish_desc: Optional[str] = None


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

    variables = _deserialize_vars(agent.variables)
    knowledge_base = _deserialize_vars(agent.knowledge_base)

    # print(f"agent.system_prompt={agent.system_prompt}")
    # print(f"variables={variables}")
    # print(f"agent.variables={agent.variables}")
    # print(f"knowledge_base={knowledge_base}")
    # print(f"data.variables={data.variables}")
    # print(f"kb_id={kb_id}")
    # print(f"data.system_prompt={data.system_prompt}")

    if data.system_prompt is not None and data.system_prompt != "":
        system_prompt = data.system_prompt
    else:
        system_prompt = agent.system_prompt or ""
        if data.variables:
            for var in variables:
                key = var.get("key")
                if key and key in data.variables:
                    val = str(data.variables[key])
                    system_prompt = system_prompt.replace(f"{{{{{key}}}}}", val)
                    system_prompt = system_prompt.replace(f"{{{key}}}", val)

    kb_content_text = ""
    if knowledge_base:
        try:
            vector_store = get_vector_store(current_user.username)

            if kb_id:
                retrieved = vector_store.similarity_search_with_score(
                    data.message,
                    k=3,
                    filter={"kb_id": kb_id},
                )
            else:
                retrieved = vector_store.similarity_search_with_score(data.message, k=3)

            logger.info(f"相似度查询结果={retrieved}")
            if not retrieved:
                retrieved = vector_store.hybrid_search(data.message)
                logger.info(f"混合检索结果={retrieved}")

            if retrieved:
                kb_context = "\n---参考知识---\n"
                content_parts = []
                documents = []
                for doc, raw in retrieved:
                    score = convert_vector_score(raw, vector_store, EMBEDDING_DATABASE)
                    logger.info(f"raw_score={raw} converted_score={score}")
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
                    source = doc.get('source', ''),
                    kb_context += f"[{doc['rank']}] 来源: {source}\n{doc['content']}\n"
                    content_parts.append(kb_context)

                kb_content_text = "\n".join(content_parts)
        except Exception as e:
            logger.warning(f"知识库检索失败: {str(e)}")

    logger.info(f"kb_content_text={kb_content_text}")
    system_prompt = system_prompt.replace("{{knowledge_content}}", kb_content_text)
    system_prompt = system_prompt.replace("{knowledge_content}", kb_content_text)

    async def generate():
        try:
            llm = get_llm_by_model(agent.model or "")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

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