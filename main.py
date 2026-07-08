import asyncio
import json
import random
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from core.DocProcess.document_process import doc_processor
from settings.Define import PathConfig
from core.agent import create_agent_graph
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
    logger.info("lifespan")
    graph = create_agent_graph()
    graph_cache["graph"] = graph
    yield


app = FastAPI(title="MultiAgent Chat API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
async def index():
    index_file = os.path.join(PathConfig.TEMPLATES_DIR , "index.html")
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """处理文件上传，使用 doc_processor 保存文件"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    file_path, saved_filename = doc_processor.save_uploaded_file(content, file.filename)
    logger.info(f"文件上传成功: {saved_filename}, 路径: {file_path}")

    return {"file_id": saved_filename, "file_path": file_path, "filename": saved_filename}


@app.get("/download/{filename}")
async def download_file(filename: str):
    """提供文件下载"""
    file_path = os.path.join(PathConfig.OUTPUTS_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)


@app.post("/chat")
async def chat(request: ChatRequest):
    logger.info("chat")
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

            result = g.invoke(state_input, config)
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                response = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
            else:
                response = "暂无响应"

            for i in range(1, len(response) + 1):
                yield f"data: {json.dumps(response[:i])}\n\n"
                await asyncio.sleep(0.03)
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps(str(e))}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5001
    )