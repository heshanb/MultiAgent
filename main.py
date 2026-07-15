import asyncio
import json
import random
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import Optional
from core.DocProcess.document_process import doc_processor
from settings.Define import PathConfig
from core.agent import create_agent_graph
from core.auth import init_db, get_current_user_optional, User
from core.security import SecurityMiddleware, file_validator, get_audit_logger
from settings.logger_manager import get_logger

logger = get_logger(__name__)


class ChatRequest(BaseModel):
    message: str
    thread_id: int | None = None
    file_content: str | None = None
    file_name: str | None = None
    original_file_ext: str | None = None
    history: list | None = None
    skill: str | None = None


graph_cache = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("lifespan - 初始化数据库")
    init_db()
    graph = create_agent_graph()
    graph_cache["graph"] = graph
    yield


app = FastAPI(title="MultiAgent Chat API")

# 安全中间件（速率限制、安全响应头）
app.add_middleware(SecurityMiddleware)

# 改进CORS配置 - 生产环境应该限制白名单
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5001,http://127.0.0.1:5001").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 注册认证路由
from core.auth import router as auth_router
app.include_router(auth_router)

# 注册知识库管理路由
from core.knowledge import router as knowledge_router
app.include_router(knowledge_router)


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = os.path.join(PathConfig.TEMPLATES_DIR , "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/knowledge", response_class=HTMLResponse)
async def knowledge_page():
    """知识库文档管理页面"""
    knowledge_file = os.path.join(PathConfig.TEMPLATES_DIR, "knowledge.html")
    with open(knowledge_file, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """处理文件上传，使用 doc_processor 保存文件"""
    audit_logger = get_audit_logger()
    client_ip = request.client.host if request.client else "unknown"
    username = current_user.username if current_user else "anonymous"
    
    if not file.filename:
        audit_logger.log_file_upload_failed(
            username=username,
            filename="unknown",
            reason="No file provided",
            ip_address=client_ip,
        )
        raise HTTPException(status_code=400, detail="No file provided")

    # 文件安全校验
    is_safe, error_msg = await file_validator.validate_with_content(file)
    if not is_safe:
        audit_logger.log_file_upload_failed(
            username=username,
            filename=file.filename,
            reason=error_msg,
            ip_address=client_ip,
        )
        raise HTTPException(status_code=400, detail=error_msg)

    content = await file.read()
    file_path, saved_filename = doc_processor.save_uploaded_file(content, file.filename)
    
    # 审计日志
    audit_logger.log_file_upload(
        username=username,
        filename=saved_filename,
        file_size=len(content),
        ip_address=client_ip,
    )
    
    user_info = f" (用户: {current_user.username})" if current_user else " (未登录用户)"
    logger.info(f"文件上传成功{user_info}: {saved_filename}, 路径: {file_path}")

    return {"file_id": saved_filename, "file_path": file_path, "filename": saved_filename}


@app.get("/download/{filename}")
async def download_file(filename: str):
    """提供文件下载"""
    file_path = os.path.join(PathConfig.OUTPUTS_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)


@app.post("/chat")
async def chat(
    request: ChatRequest,
    http_request: Request,
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    # 审计日志
    audit_logger = get_audit_logger()
    client_ip = http_request.client.host if http_request.client else "unknown"
    username = current_user.username if current_user else "anonymous"
    audit_logger.log_chat(
        username=username,
        message_length=len(request.message),
        ip_address=client_ip,
    )
    
    logger.info("chat" + (f" - 用户: {current_user.username}" if current_user else " - 未登录"))
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    thread_id = request.thread_id or random.randint(1, 100000)

    async def generate():
        g = create_agent_graph()
        if not g:
            yield f"data: {json.dumps('Graph not initialized')}\n\n"
            return

        config = {"configurable": {"thread_id": thread_id}}
        try:
            file_path_for_state = None
            message_content = request.message

            if request.file_name:
                logger.info(f"检测到文件: {request.file_name}，正在读取内容...")
                file_path = doc_processor.UPLOAD_DIR / request.file_name
                file_ext = Path(request.file_name).suffix.lower()
                # 图纸类文件：图片、DXF、PDF，不读取文本内容，只传路径
                DRAWING_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf', '.dxf'}
                is_drawing_file = file_ext in DRAWING_EXTS or request.skill == 'drawing'

                if not file_path.exists() and request.file_content:
                    logger.info(f"文件不存在，但检测到 file_content，正在保存文件...")
                    try:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(request.file_content)
                        logger.info(f"文件保存成功: {file_path}")
                    except Exception as e:
                        logger.error(f"保存文件失败: {str(e)}")

                if file_path.exists():
                    file_path_for_state = str(file_path)
                    if is_drawing_file:
                        # 图纸文件：不读取文本内容，由 drawing_node 直接处理
                        message_content = request.message
                        logger.info(f"图纸文件上传，跳过文本读取: {file_path}")
                    elif not request.file_content:
                        try:
                            file_content = doc_processor.read_document(str(file_path))
                            original_ext = Path(request.file_name).suffix.lower()
                            message_content = f"[文档内容]\n{file_content}\n\n[用户要求]\n{request.message}\n\n[原始文件格式]\n{original_ext}"
                            logger.info(f"文档内容读取成功，长度: {len(file_content)}，格式: {original_ext}")
                        except Exception as e:
                            logger.error(f"读取文档失败: {str(e)}")
                            message_content = f"[用户要求]\n{request.message}\n\n注意：文档读取失败，请检查文件格式。"
                    else:
                        message_content = f"[文档内容]\n{request.file_content}\n\n[用户要求]\n{request.message}"
                else:
                    logger.warning(f"文件不存在: {file_path}")
                    message_content = f"[用户要求]\n{request.message}\n\n注意：未找到上传的文件。"
            elif request.file_content:
                message_content = f"[文档内容]\n{request.file_content}\n\n[用户要求]\n{request.message}"

            # Build messages list with history
            messages_list = []
            if request.history:
                for msg in request.history:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == 'user':
                        messages_list.append(HumanMessage(content=content))
                    elif role == 'assistant':
                        messages_list.append({"role": "assistant", "content": content})
            
            # Add current message
            messages_list.append(HumanMessage(content=message_content))

            state_input = {"messages": messages_list}
            if file_path_for_state:
                state_input["file_path"] = file_path_for_state
            if request.skill:
                state_input["skill"] = request.skill
            if current_user:
                state_input["user"] = current_user.username

            # 先发送一个空响应，表示开始处理
            yield f"data: {json.dumps({'text': ''})}\n\n"
            await asyncio.sleep(0.1)
            
            result = None
            response = "暂无响应"
            
            try:
                # 使用线程池执行图，避免阻塞事件循环
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, g.invoke, state_input, config)
                
                messages = result.get("messages", [])

                logger.info(f"messages: {messages}")
                if messages:
                    last_msg = messages[-1]
                    logger.debug(f"last_msg: {last_msg}")
                    response = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                else:
                    response = "暂无响应"
                    
                logger.info(f"响应生成成功: {response[:50]}...")
                    
            except asyncio.TimeoutError:
                logger.error("Agent graph execution timeout")
                response = "请求超时，请稍后重试或简化您的问题"
            except Exception as e:
                logger.error(f"Agent graph execution error: {str(e)}")
                response = f"服务执行出错: {str(e)}"

            # 获取sources数据（已经是列表格式，不需要JSON解析）
            sources = result.get("sources", []) if result else []
            # 确保sources是列表格式
            if not isinstance(sources, list):
                sources = []

            # 发送最终响应
            yield f"data: {json.dumps({'text': response, 'sources': sources})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'text': f'服务错误: {str(e)}'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5001
    )