import asyncio
import json
import random
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from langchain_community.tools import sleep
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# 导入 llm 和 get_llm_by_model
from core.agent import get_llm_by_model
from typing import Optional
from core.skills.DocProcess.document_process import doc_processor
from settings.Define import PathConfig, Params
from core.agent import create_agent_graph
from core.auth import init_db, get_current_user_optional, User
from core.security import SecurityMiddleware, file_validator, get_audit_logger
from settings.logger_manager import get_logger

logger = get_logger(__name__)

# 线程池配置 - 用于执行CPU密集型任务
THREAD_POOL_SIZE = int(os.getenv("THREAD_POOL_SIZE", 8))
EXECUTOR = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE, thread_name_prefix="agent-worker-")

# 会话状态缓存（每个会话独立）
session_graph_cache = {}
cache_lock = threading.Lock()

# 全局图缓存（兼容旧代码）
graph_cache = {}

# 全局：维护运行中的进程
_running_procs = {}
_proc_lock = None

class ChatRequest(BaseModel):
    message: str
    thread_id: int | None = None
    file_content: str | None = None
    file_name: str | None = None
    original_file_ext: str | None = None
    history: list | None = None
    skill: str | None = None
    images: list | None = None
    project_context: dict | None = None
    model: str | None = None  # 用户选择的模型ID


def get_graph_for_session(thread_id: int):
    """获取或创建会话专属的图实例"""
    with cache_lock:
        if thread_id not in session_graph_cache:
            session_graph_cache[thread_id] = create_agent_graph()
            logger.info(f"为会话 {thread_id} 创建新的图实例")
        return session_graph_cache[thread_id]

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


@app.get("/project-code", response_class=HTMLResponse)
async def project_code_page():
    """项目编程页面 - AI辅助项目开发"""
    project_code_file = os.path.join(PathConfig.TEMPLATES_DIR, "project-code.html")
    with open(project_code_file, "r", encoding="utf-8") as f:
        return f.read()


class ScanRequest(BaseModel):
    path: str


class ReadFileRequest(BaseModel):
    path: str
    project_root: Optional[str] = None


@app.post("/read-file")
async def read_file(request: ReadFileRequest):
    """读取文件内容"""
    file_path = request.path
    project_root = request.project_root
    logger.info(f"读取文件: {file_path}, 项目根路径: {project_root}")
    
    try:
        # 如果已经是绝对路径，直接使用
        if os.path.isabs(file_path):
            full_path = file_path
        elif project_root:
            # 标准化路径分隔符
            file_path_normalized = file_path.replace('/', os.sep).replace('\\', os.sep)
            project_root_normalized = project_root.replace('/', os.sep).replace('\\', os.sep)
            # 如果 file_path 已经以 project_root 开头，避免重复拼接
            if file_path_normalized.startswith(project_root_normalized + os.sep):
                full_path = file_path_normalized
            elif file_path_normalized == project_root_normalized:
                full_path = file_path_normalized
            else:
                full_path = os.path.join(project_root_normalized, file_path_normalized)
        else:
            # 回退到当前工作目录
            full_path = os.path.join(os.getcwd(), file_path)
        
        if not os.path.exists(full_path):
            logger.warning(f"文件不存在: {full_path}")
            return {"content": f"[错误] 文件不存在: {full_path}"}
        
        if not os.path.isfile(full_path):
            logger.warning(f"路径不是文件: {full_path}")
            return {"content": f"[错误] 不是文件: {full_path}"}
        
        # 检查文件大小（限制为10MB）
        file_size = os.path.getsize(full_path)
        if file_size > 10 * 1024 * 1024:
            return {"content": f"[提示] 文件过大 ({file_size // 1024}KB)，已跳过显示"}
        
        # 尝试检测编码并读取
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
        content = None
        
        for encoding in encodings:
            try:
                with open(full_path, 'r', encoding=encoding) as f:
                    content = f.read()
                logger.info(f"使用编码 {encoding} 读取成功，长度: {len(content)}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"读取文件失败 ({encoding}): {str(e)}")
        
        if content is None:
            # 最后尝试以二进制方式读取前一部分
            with open(full_path, 'rb') as f:
                raw = f.read(4096)
            content = f"[提示] 无法识别文本编码，可能是二进制文件\n\n{repr(raw[:200])}"
        
        return {"content": content}
    
    except Exception as e:
        logger.error(f"读取文件异常: {str(e)}")
        return {"content": f"[读取失败] {str(e)}"}


@app.post("/scan-directory")
async def scan_directory(request: ScanRequest):
    """扫描目录结构"""
    folder_path = request.path
    logger.info(f"扫描目录: {folder_path}")
    
    def scan_dir(path, parent_path=""):
        result = []
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.name.startswith('.'):
                        continue
                    relative_path = os.path.join(parent_path, entry.name)
                    if entry.is_dir(follow_symlinks=False):
                        children = scan_dir(entry.path, relative_path)
                        result.append({
                            'name': entry.name,
                            'path': relative_path.replace('\\', '/'),
                            'type': 'folder',
                            'children': children
                        })
                    else:
                        result.append({
                            'name': entry.name,
                            'path': relative_path.replace('\\', '/'),
                            'type': 'file'
                        })
            # 按类型排序，文件夹在前
            result.sort(key=lambda x: (x['type'], x['name']))
        except Exception as e:
            logger.error(f"扫描目录失败: {str(e)}")
            result = []
        
        return result
    
    try:
        # 获取当前工作目录
        current_dir = os.getcwd()
        logger.info(f"当前工作目录: {current_dir}")
        
        # 如果路径只是目录名，尝试在当前目录下查找
        if not os.path.isabs(folder_path) and not folder_path.startswith('\\'):
            # 首先尝试直接扫描当前工作目录
            full_path = current_dir
            if os.path.isdir(full_path):
                folder_path = full_path
            else:
                # 尝试查找常见的项目目录
                common_paths = ['src', 'app', 'project', 'backend']
                for common in common_paths:
                    test_path = os.path.join(current_dir, common)
                    if os.path.isdir(test_path):
                        folder_path = test_path
                        break
        
        if not os.path.isdir(folder_path):
            # 如果路径不是有效目录，返回模拟数据
            logger.warning(f"无效目录路径: {folder_path}，返回模拟数据")
            return generate_mock_tree()
        
        tree_data = scan_dir(folder_path)
        logger.info(f"目录扫描完成，共 {len(tree_data)} 个条目")
        
        # 包装成包含根目录的结构
        root_name = os.path.basename(folder_path)
        if not root_name:  # 如果是根目录
            root_name = 'project'
        
        def add_expanded_flag(items):
            """为目录节点添加expanded标志，所有子文件夹都不展开"""
            for item in items:
                if item.get('type') == 'folder' and item.get('children'):
                    item['expanded'] = False
                    add_expanded_flag(item['children'])
            return items
        
        tree_data = add_expanded_flag(tree_data)
        
        return [{
            'name': root_name,
            'path': folder_path,
            'type': 'folder',
            'children': tree_data,
            'expanded': True
        }]
    except Exception as e:
        logger.error(f"目录扫描异常: {str(e)}")
        return generate_mock_tree()


def generate_mock_tree():
    """生成模拟目录树数据"""
    return [
        {
            'name': 'src',
            'path': '/project/src',
            'type': 'folder',
            'children': [
                {
                    'name': 'main.py',
                    'path': '/project/src/main.py',
                    'type': 'file'
                },
                {
                    'name': 'utils.py',
                    'path': '/project/src/utils.py',
                    'type': 'file'
                },
                {
                    'name': 'models',
                    'path': '/project/src/models',
                    'type': 'folder',
                    'children': [
                        {'name': 'user.py', 'path': '/project/src/models/user.py', 'type': 'file'},
                        {'name': 'product.py', 'path': '/project/src/models/product.py', 'type': 'file'}
                    ]
                }
            ]
        },
        {
            'name': 'tests',
            'path': '/project/tests',
            'type': 'folder',
            'children': [
                {'name': 'test_main.py', 'path': '/project/tests/test_main.py', 'type': 'file'}
            ]
        },
        {
            'name': 'README.md',
            'path': '/project/README.md',
            'type': 'file'
        },
        {
            'name': 'requirements.txt',
            'path': '/project/requirements.txt',
            'type': 'file'
        }
    ]


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
    logger.info(f"请求消息长度: {len(request.message)}")
    logger.info(f"请求技能: {request.skill}")
    logger.info(f"请求图片数量: {len(request.images) if request.images else 0}")
    if request.images:
        logger.info(f"第一张图片名称: {request.images[0].get('name') if isinstance(request.images[0], dict) else '未知'}")
        data_len = len(request.images[0].get('data', '')) if isinstance(request.images[0], dict) else 0
        logger.info(f"第一张图片数据长度: {data_len}")
    
    if not request.message.strip() and not request.images:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    thread_id = request.thread_id or random.randint(1, 100000)

    async def generate():
        # 获取会话专属的图实例
        g = get_graph_for_session(thread_id)
        if not g:
            yield f"data: {json.dumps('Graph not initialized')}\n\n"
            return

        config = {"configurable": {"thread_id": thread_id}}
        try:
            file_path_for_state = None
            message_content = request.message

            if request.file_name:
                logger.info(f"检测到文件: {request.file_name}，正在处理...")
                file_path = doc_processor.UPLOAD_DIR / request.file_name
                file_ext = Path(request.file_name).suffix.lower()
                # 图纸类文件：图片、DXF、PDF，不读取文本内容，只传路径
                DRAWING_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf', '.dxf'}
                is_drawing_file = file_ext in DRAWING_EXTS or request.skill == 'drawing'

                if file_path.exists():
                    file_path_for_state = str(file_path)
                    if is_drawing_file:
                        # 图纸文件：不读取文本内容，由 drawing_node 直接处理
                        message_content = request.message
                        logger.info(f"图纸文件，跳过文本读取: {file_path}")
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
                        # 使用前端传来的文件内容（不自动保存）
                        message_content = f"[文档内容]\n{request.file_content}\n\n[用户要求]\n{request.message}"
                else:
                    # 文件不存在，使用前端传来的内容（不自动保存）
                    if request.file_content:
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
            if request.images:
                state_input["images"] = request.images
                logger.info(f"接收到 {len(request.images)} 张图片")
            else:
                logger.info("未接收到图片数据")
            if request.project_context:
                state_input["project_context"] = request.project_context
                logger.info(f"接收到项目上下文: {request.project_context.get('project_name')}")

            # 先发送一个空响应，表示开始处理
            yield f"data: {json.dumps({'text': '', 'status': 'thinking'})}\n\n"
            await asyncio.sleep(0.1)
            
            result = None
            response = "暂无响应"
            file_changes = []
            sources = []
            
            try:
                # 如果是项目编程技能，直接使用流式响应
                if request.skill == "project_code":
                    from core.skills.AICode.project_programming import Project_Programming
                    
                    # 根据用户选择的模型创建LLM实例
                    selected_model = request.model or "qwen"
                    # 如果有图片，强制使用多模态模型
                    if request.images and len(request.images) > 0:
                        selected_model = Params.DEFAULT_MULTIMODAL_MODEL
                        logger.info(f"检测到图片输入，自动切换为多模态模型: {selected_model}")
                    logger.info(f"选择的模型: {selected_model}")
                    model_llm = get_llm_by_model(selected_model)
                    
                    logger.info("正在创建 Project_Programming 实例")
                    project_programming = Project_Programming(
                        file_path_for_state,
                        messages_list,
                        request.project_context,
                        images=request.images if request.images else None
                    )
                    logger.info("Project_Programming 实例创建成功")
                    
                    message = messages_list[-1]
                    logger.info("开始流式获取模型响应")
                    
                    # 直接流式响应
                    full_response = ""
                    chunk_count = 0
                    for chunk in project_programming.get_model_response_stream(message, model_llm):
                        if chunk:
                            full_response += chunk
                            chunk_count += 1
                            logger.info(f"发送第 {chunk_count} 个 chunk，长度: {len(chunk)}")
                            yield f"data: {json.dumps({'text': chunk, 'sources': sources, 'status': 'streaming'})}\n\n"
                            await asyncio.sleep(0.05)
                    
                    response = full_response
                    logger.info(f"流式响应完成，内容长度: {len(response)}, chunk数量: {chunk_count}")
                else:
                    # 使用专用线程池执行图，避免阻塞事件循环
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(EXECUTOR, g.invoke, state_input, config)
                    
                    messages = result.get("messages", [])
                    logger.info(f"messages: {messages}")
                    if messages:
                        last_msg = messages[-1]
                        logger.debug(f"last_msg: {last_msg}")
                        response = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
                    else:
                        response = "暂无响应"
                    
                    logger.info(f"响应生成成功: {response[:50]}...")
                        
                    # 获取sources数据（已经是列表格式，不需要JSON解析）
                    sources = result.get("sources", []) if result else []
                    # 确保sources是列表格式
                    if not isinstance(sources, list):
                        sources = []
                
                # 提取文件变更信息（project_code 技能时都提取，包括多模态识图）
                if request.skill == "project_code":
                    # 获取项目根路径和文件列表，优先使用完整路径
                    project_root = ""
                    project_files = []
                    if request.project_context:
                        project_root = request.project_context.get("project_path", "") or request.project_context.get("project_name", "")
                        project_files = request.project_context.get("files", [])
                    print(f"[DEBUG] extract_file_changes 调用参数: project_root={project_root}, project_files={len(project_files)}")
                    file_changes = extract_file_changes(response, request.file_name, project_root)
                
            except asyncio.TimeoutError:
                logger.error("Agent graph execution timeout")
                response = "请求超时，请稍后重试或简化您的问题"
            except Exception as e:
                logger.error(f"Agent graph execution error: {str(e)}")
                response = f"服务执行出错: {str(e)}"

            # 发送最终响应，包含文件变更信息
            yield f"data: {json.dumps({'text': '', 'sources': sources, 'file_changes': file_changes, 'status': 'done'})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'text': f'服务错误: {str(e)}', 'status': 'error'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _filter_file_changes(changes: list, project_root: str) -> list:
    """过滤文件变更列表：移除幻觉文件、不存在文件的修改、运行示例输出"""
    if not changes:
        return changes
    
    import os
    
    filtered = []
    hallucinated_names = ['new_file', '未知文件', 'output', 'result', '示例', '测试', 'output.txt', 'result.txt']
    
    for c in changes:
        file_name = c.get('file', '')
        change_type = c.get('type', '')
        content = c.get('content', '') or ''
        
        name_only = file_name.split('/')[-1].split('\\')[-1].lower()
        
        # 1. 过滤典型幻觉文件名
        if any(n in name_only for n in hallucinated_names) or name_only == '':
            print(f"[过滤] 跳过典型幻觉文件名: {file_name}")
            continue
        
        # 2. modified 类型但文件不存在于项目中 → 跳过
        if change_type == 'modified':
            file_exists = False
            try:
                if project_root:
                    check_path = os.path.join(project_root, file_name)
                    file_exists = os.path.isfile(check_path)
                if not file_exists:
                    # 支持路径分隔符的两种格式
                    check_path2 = file_name.replace('/', '\\')
                    if project_root:
                        check_path2 = os.path.join(project_root, file_name.replace('\\', os.sep).replace('/', os.sep))
                    file_exists = os.path.isfile(check_path2)
            except Exception:
                file_exists = False
            if not file_exists:
                print(f"[过滤] 跳过不存在的文件（修改）: {file_name}")
                continue
        
        # 3. added 类型且内容是运行示例输出（不是真实源代码） → 跳过
        if change_type == 'added':
            is_real_source = ('def ' in content or 'class ' in content or 'import ' in content or
                              'function ' in content or 'const ' in content or 'let ' in content or
                              'package ' in content or '#include' in content)
            is_sample_output = (
                ('[' in content and ']' in content and 'def ' not in content and 'class ' not in content) or
                bool(re.search(r'^[\[\{\'\"]', content, re.MULTILINE)) or
                bool(re.search(r'^print\(', content, re.MULTILINE)) or
                bool(re.search(r'^console\.', content, re.MULTILINE)) or
                bool(re.search(r'^={3,}$', content, re.MULTILINE)) or
                bool(re.search(r'^-+.*测试.*-+$', content, re.MULTILINE))
            )
            if is_sample_output and not is_real_source:
                print(f"[过滤] 跳过运行示例输出（非源代码）: {file_name}")
                continue
        
        filtered.append(c)
    
    print(f"[过滤] 原始 {len(changes)} 个，过滤后 {len(filtered)} 个")
    return filtered


def extract_file_changes(content: str, current_file: str = None, project_root: str = None) -> list:
    """从AI回复中提取文件变更信息"""
    import re
    changes = []
    
    # 全局清理：移除思考标记（防止LLM的思考文本混入代码块或文件路径）
    global_thinking_patterns = [
        r'<!--THINKING_START-->[\s\S]*?<!--THINKING_END-->',
        r'🤔\s*让我思考一下[.。\s]*',
        r'🤔\s*Let me think[.。\s]*',
        r'📝\s*继续分析中[.。\s]*',
        r'🔍\s*深入研究中[.。\s]*',
        r'💡\s*发现关键点[.。\s]*',
        r'⚡\s*正在处理[.。\s]*',
        r'让我思考一下[.。\s]*',
        r'Let me think[.。\s]*',
        r'继续分析中[.。\s]*',
        r'深入研究中[.。\s]*',
        r'发现关键点[.。\s]*',
        r'正在处理[.。\s]*',
        r'<thinking>[\s\S]*?</thinking>',
        r'<thought>[\s\S]*?</thought>',
    ]
    for pattern in global_thinking_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    # 查找所有代码块
    code_blocks = re.findall(r'```(\w+)?\n([\s\S]*?)```', content)
    
    for idx, (lang, code) in enumerate(code_blocks):
        code = code.strip()
        
        # 跳过空代码块
        if not code:
            continue
            
        # 跳过目录结构（tree命令输出）
        if '├──' in code or '└──' in code or '│' in code:
            continue
            
        # 跳过测试输出和运行示例（加强过滤）
        is_test_output = re.search(r'^={3,}$', code, re.MULTILINE) or \
                        '测试通过' in code or \
                        '测试结果' in code or \
                        '运行示例' in code or \
                        '控制台输出' in code or \
                        '输出示例' in code or \
                        '>>> ' in code or \
                        re.search(r'^\d+! = \d+', code) or \
                        re.search(r'^---.*阶乘.*---$', code, re.MULTILINE) or \
                        re.search(r'^---.*测试.*---$', code, re.MULTILINE) or \
                        (re.search(r'! = \d+', code) and not 'def ' in code and not 'class ' in code)
        if is_test_output:
            logger.info(f"跳过测试输出代码块: {code[:50]}...")
            continue
        
        # 清理代码中的思考标记（LLM可能将思考文本混入代码块）
        thinking_patterns = [
            r'🤔\s*让我思考一下[.。\s]*',
            r'🤔\s*Let me think[.。\s]*',
            r'📝\s*继续分析中[.。\s]*',
            r'🔍\s*深入研究中[.。\s]*',
            r'💡\s*发现关键点[.。\s]*',
            r'⚡\s*正在处理[.。\s]*',
            r'让我思考一下[.。\s]*',
            r'Let me think[.。\s]*',
            r'继续分析中[.。\s]*',
            r'深入研究中[.。\s]*',
            r'发现关键点[.。\s]*',
            r'正在处理[.。\s]*',
            r'<thinking>[\s\S]*?</thinking>',
            r'<thought>[\s\S]*?</thought>',
        ]
        for pattern in thinking_patterns:
            code = re.sub(pattern, '', code, flags=re.IGNORECASE)
        code = code.strip()
        
        # 清理后如果为空则跳过
        if not code:
            continue
            
        # 查找代码块前面的内容
        # 找到所有代码块的位置
        code_block_positions = []
        pos = 0
        while True:
            start = content.find('```', pos)
            if start == -1:
                break
            end = content.find('```', start + 3)
            if end == -1:
                end = len(content)
            code_block_positions.append((start, end))
            pos = end + 3
        
        if idx < len(code_block_positions):
            current_start = code_block_positions[idx][0]
            if idx == 0:
                before_text = content[:current_start]
            else:
                prev_end = code_block_positions[idx-1][1]
                before_text = content[prev_end+3:current_start]
        else:
            before_text = ""
        
        # 清理 before_text 中的思考标记（防止干扰文件路径提取）
        for pattern in thinking_patterns:
            before_text = re.sub(pattern, '', before_text, flags=re.IGNORECASE)
        before_text = before_text.strip()
        
        # 方法1：查找"文件路径:"标记
        file_path_match = re.search(r'文件路径[：:]\s*([^\s\n]+)', before_text, re.IGNORECASE)
        # 方法1b：查找"### 文件：xxx"标记（新格式）
        if not file_path_match:
            file_path_match = re.search(r'###\s*文件[：:]\s*([^\s\n]+)', before_text, re.IGNORECASE)
        
        # 方法1.5：查找"xxx.py"在"修改后的完整代码"或"修复后的完整代码"之前的模式
        file_name_from_context = None
        if '修改后的完整代码' in before_text or '修复后的完整代码' in before_text:
            # 查找这些关键词之前提到的文件名
            context_before_code = before_text.split('修改后的完整代码')[0].split('修复后的完整代码')[0]
            
            # 方法1.5.1：直接查找文件名
            file_in_context = re.search(r'([a-zA-Z0-9_\u4e00-\u9fff\/-]+\.(py|js|ts|html|css|java|cpp|go|rs|json|md))', context_before_code, re.IGNORECASE)
            if file_in_context:
                file_name_from_context = file_in_context.group(1).strip()
                print(f"[DEBUG] 从上下文提取文件名: {file_name_from_context}")
            else:
                # 方法1.5.2：查找"修复 xxx.py"或"修改 xxx.py"模式
                fix_match = re.search(r'(修复|修改)\s*([a-zA-Z0-9_\u4e00-\u9fff\/-]+\.(py|js|ts|html|css|java|cpp|go|rs|json|md))', context_before_code, re.IGNORECASE)
                if fix_match:
                    file_name_from_context = fix_match.group(2).strip()
                    print(f"[DEBUG] 从修复/修改模式提取文件名: {file_name_from_context}")
                else:
                    # 方法1.5.3：查找"第n处：xxx.py"模式
                    nth_match = re.search(r'第\d+处[：:]?\s*([a-zA-Z0-9_\u4e00-\u9fff\/-]+\.(py|js|ts|html|css|java|cpp|go|rs|json|md))', context_before_code, re.IGNORECASE)
                    if nth_match:
                        file_name_from_context = nth_match.group(1).strip()
                        print(f"[DEBUG] 从第n处模式提取文件名: {file_name_from_context}")
        
        # 判断是否明确要求创建新文件
        has_new_file_keyword = any(keyword in before_text for keyword in ['新增文件', '新建文件', '创建文件', '创建新文件', '新增目录'])
        
        # 如果用户上传了文件且没有明确要求创建新文件，则优先使用当前文件
        use_current_file = current_file and not has_new_file_keyword
        
        if use_current_file:
            file_name = current_file
            print(f"[DEBUG] 用户上传了文件 {current_file}，且没有明确要求创建新文件，使用当前文件")
        elif file_path_match:
            file_name = file_path_match.group(1).strip()
            # 移除反引号
            file_name = file_name.replace('`', '')
            # 修复重复扩展名
            file_name = re.sub(r'\.py\.py$', '.py', file_name, flags=re.IGNORECASE)
            file_name = re.sub(r'\.js\.js$', '.js', file_name, flags=re.IGNORECASE)
        elif file_name_from_context:
            file_name = file_name_from_context
            print(f"[DEBUG] 使用上下文提取的文件名: {file_name}")
        else:
            # 方法2：查找"新增文件:"或"新建文件:"标记
            new_file_match = re.search(r'(新增文件|新建文件)[：:]\s*([^\s\n]+)', before_text, re.IGNORECASE)
            if new_file_match:
                file_name = new_file_match.group(2).strip()
                # 移除反引号
                file_name = file_name.replace('`', '')
            else:
                # 方法3：查找带目录路径的文件名
                file_match = re.search(r'([a-zA-Z0-9_\u4e00-\u9fff\/-]+\.(py|js|ts|html|css|java|cpp|go|rs|json|md|txt))', before_text, re.IGNORECASE)
                if file_match:
                    file_name = file_match.group(1).strip()
                else:
                    # 方法4：从代码内容推断
                    if lang:
                        ext_map = {
                            'python': '.py', 'javascript': '.js', 'typescript': '.ts',
                            'html': '.html', 'css': '.css', 'java': '.java',
                            'cpp': '.cpp', 'go': '.go', 'rust': '.rs',
                            'json': '.json', 'markdown': '.md', 'text': '.txt'
                        }
                        ext = ext_map.get(lang.lower(), '.py')
                        func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
                        if func_match:
                            file_name = func_match.group(1) + ext
                        else:
                            file_name = 'new_file' + ext
                    else:
                        file_name = current_file or "未知文件"
        
        # 判断变更类型
        change_type = 'modified'
        # 检查是否为新增文件（更广泛的关键词匹配）
        added_keywords = ['新增文件', '新建文件', '新增目录', '创建文件', '创建新文件', 
                         '创建以下文件', '生成文件', '新增以下文件', '创建', '新增',
                         '以下是', '以下代码', '文件内容', '完整代码', '实现代码',
                         '新增功能', '实现一个', '来实现', '脚本', '编写', '写一个',
                         '创建一个', '新文件', '新的']
        deleted_keywords = ['删除文件', '删除以下文件', '删除']
        
        # 方法1：检查 before_text 中的关键词
        has_added_keyword = any(keyword in before_text for keyword in added_keywords)
        has_deleted_keyword = any(keyword in before_text for keyword in deleted_keywords)
        
        # 方法2：检查文件是否已存在（支持子目录搜索）
        file_exists = False
        file_full_path = None
        try:
            # 先尝试直接拼接路径
            direct_path = os.path.join(project_root, file_name) if project_root else file_name
            if os.path.exists(direct_path):
                file_exists = True
                file_full_path = direct_path
            elif project_root:
                # 如果直接路径不存在，递归搜索项目目录
                basename = os.path.basename(file_name)
                for root, dirs, files in os.walk(project_root):
                    # 跳过 .venv 和 __pycache__ 等目录
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                    if basename in files:
                        file_exists = True
                        file_full_path = os.path.join(root, basename)
                        # 更新 file_name 为相对路径
                        rel_path = os.path.relpath(file_full_path, project_root)
                        file_name = rel_path.replace('\\', '/')
                        break
            print(f"[DEBUG] 检查文件是否存在: {file_full_path or direct_path}, 存在={file_exists}")
        except Exception as e:
            print(f"[DEBUG] 检查文件存在性失败: {e}")
        
        # 方法3：如果 current_file 与 file_name 不同，可能是新增文件
        is_different_from_current = current_file and current_file != file_name
        
        # 综合判断
        # 如果用户上传了文件且没有明确要求创建新文件，则视为修改现有文件
        if use_current_file:
            change_type = 'modified'
        elif file_exists:
            # 文件已存在，应该视为修改
            change_type = 'modified'
        elif has_added_keyword:
            change_type = 'added'
        elif has_deleted_keyword and '文件' in before_text:
            change_type = 'deleted'
        elif is_different_from_current:
            change_type = 'added'
        else:
            change_type = 'added'
        
        print(f"[DEBUG] 文件 {file_name}: type={change_type}, 存在={file_exists}, 新增关键词={has_added_keyword}, 使用当前文件={use_current_file}, before_text前100字={before_text[:100]}")
        
        # 计算行数
        lines = code.split('\n')
        changes_count = f"+{len(lines)}" if change_type == 'added' else '+1 -1'
        
        # 添加到变更列表（去重）
        existing_idx = next((i for i, c in enumerate(changes) if c['file'] == file_name), -1)
        if existing_idx >= 0:
            changes[existing_idx]['content'] = code
            changes[existing_idx]['changes'] = changes_count
        else:
            changes.append({
                "type": change_type,
                "file": file_name,
                "language": lang or (file_name.split('.')[-1] if '.' in file_name else "text"),
                "changes": changes_count,
                "content": code
            })
    
    # 添加调试日志
    print(f"[DEBUG] extract_file_changes - 检测到 {len(changes)} 个文件变更: {[c['file'] for c in changes]}")
    print(f"[DEBUG] 每个文件的类型: {[c['type'] for c in changes]}")
    
    # 最终过滤：移除幻觉文件
    changes = _filter_file_changes(changes, project_root)
    
    return changes


def extract_code_diffs(content: str, current_file: str = None) -> list:
    """从AI回复中提取代码差异，用于实时显示"""
    import re
    diffs = []
    
    # 查找所有代码块
    code_blocks = re.findall(r'```(\w+)?\n([\s\S]*?)```', content)
    
    for lang, code in code_blocks:
        # 检查代码块前面是否有文件说明
        lines = content.split('\n')
        file_name = current_file or "未知文件"
        
        # 查找代码块前面的文件名
        for i, line in enumerate(lines):
            if f'```{lang}' in line or (lang == '' and '```' in line):
                # 向前查找文件名
                for j in range(i-1, max(0, i-5), -1):
                    if any(ext in lines[j] for ext in ['.py', '.js', '.html', '.css', '.java', '.cpp', '.go', '.ts']):
                        file_name = lines[j].strip().rstrip(':').rstrip('：')
                        break
                break
        
        # 计算行数变化
        line_count = len(code.strip().split('\n'))
        
        diffs.append({
            "file": file_name,
            "language": lang or "text",
            "content": code,
            "lines": line_count,
            "type": "modified" if "修改" in content else "added"
        })
    
    return diffs



@app.get("/detect-python")
async def detect_python():
    """检测服务器上的 Python 解释器"""
    import subprocess
    import platform

    interpreters = []

    # 1. 检测系统默认 Python
    try:
        result = subprocess.run(['python', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            # 获取完整路径
            where_result = subprocess.run(['where', 'python'] if os.name == 'nt' else ['which', 'python'], capture_output=True, text=True, timeout=5)
            path = where_result.stdout.strip().split('\n')[0] if where_result.returncode == 0 else 'python'
            interpreters.append({
                'name': f'Python ({version})',
                'path': path,
                'is_default': True
            })
    except:
        pass

    # 2. 检测 python3
    try:
        result = subprocess.run(['python3', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            where_result = subprocess.run(['where', 'python3'] if os.name == 'nt' else ['which', 'python3'], capture_output=True, text=True, timeout=5)
            path = where_result.stdout.strip().split('\n')[0] if where_result.returncode == 0 else 'python3'
            interpreters.append({
                'name': f'Python3 ({version})',
                'path': path,
                'is_default': False
            })
    except:
        pass

    # 3. 检测当前虚拟环境
    venv_python = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(os.getcwd(), '.venv', 'bin', 'python')
    if os.path.exists(venv_python):
        try:
            result = subprocess.run([venv_python, '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                interpreters.append({
                    'name': f'虚拟环境 Python ({version})',
                    'path': venv_python,
                    'is_default': False,
                    'is_venv': True
                })
        except:
            pass

    # 4. 检测常见 Python 安装路径 (Windows)
    if os.name == 'nt':
        common_paths = [
            r'C:\Python313\python.exe',
            r'C:\Python312\python.exe',
            r'C:\Python311\python.exe',
            r'C:\Python310\python.exe',
            r'C:\Python39\python.exe',
            r'C:\Program Files\Python313\python.exe',
            r'C:\Program Files\Python312\python.exe',
            r'C:\Program Files\Python311\python.exe',
            r'C:\Program Files\Python310\python.exe',
            r'C:\Program Files\Python39\python.exe',
        ]
        for p in common_paths:
            if os.path.exists(p) and not any(i['path'] == p for i in interpreters):
                try:
                    result = subprocess.run([p, '--version'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        interpreters.append({
                            'name': f'Python ({version})',
                            'path': p,
                            'is_default': False
                        })
                except:
                    pass

    # 按优先级排序：虚拟环境 > 系统默认 > 其他
    interpreters.sort(key=lambda x: (not x.get('is_venv', False), not x.get('is_default', False)))

    return {'success': True, 'interpreters': interpreters}

@app.get("/browse-directory")
async def browse_directory(path: str = "/"):
    """浏览服务器目录结构，用于选择文件"""
    import platform
    try:
        target_path = os.path.abspath(path)
        if not os.path.exists(target_path):
            target_path = os.path.expanduser("~")

        if not os.path.isdir(target_path):
            target_path = os.path.dirname(target_path)

        items = []
        # 父目录
        parent = os.path.dirname(target_path)
        if parent != target_path:
            items.append({
                "name": "..",
                "path": parent,
                "type": "dir",
                "size": 0
            })

        # 列出当前目录内容
        for entry in sorted(os.scandir(target_path), key=lambda e: (not e.is_dir(), e.name.lower())):
            # 跳过隐藏文件（.开头）
            if entry.name.startswith('.'):
                continue
            try:
                items.append({
                    "name": entry.name,
                    "path": os.path.abspath(entry.path),
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if not entry.is_dir() else 0
                })
            except PermissionError:
                continue

        # Windows 下返回盘符列表
        if platform.system() == "Windows" and target_path == "/":
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({
                        "name": f"{letter}:",
                        "path": drive,
                        "type": "dir",
                        "size": 0
                    })
            items = drives

        return {
            "success": True,
            "currentPath": target_path,
            "separator": os.sep,
            "items": items
        }
    except Exception as e:
        return {"success": False, "error": str(e)}



@app.get("/detect-python")
async def detect_python():
    """检测服务器上的 Python 解释器"""
    import subprocess
    import platform

    interpreters = []

    # 1. 检测系统默认 Python
    try:
        result = subprocess.run(['python', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            # 获取完整路径
            where_result = subprocess.run(['where', 'python'] if os.name == 'nt' else ['which', 'python'], capture_output=True, text=True, timeout=5)
            path = where_result.stdout.strip().split('\n')[0] if where_result.returncode == 0 else 'python'
            interpreters.append({
                'name': f'Python ({version})',
                'path': path,
                'is_default': True
            })
    except:
        pass

    # 2. 检测 python3
    try:
        result = subprocess.run(['python3', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            where_result = subprocess.run(['where', 'python3'] if os.name == 'nt' else ['which', 'python3'], capture_output=True, text=True, timeout=5)
            path = where_result.stdout.strip().split('\n')[0] if where_result.returncode == 0 else 'python3'
            interpreters.append({
                'name': f'Python3 ({version})',
                'path': path,
                'is_default': False
            })
    except:
        pass

    # 3. 检测当前虚拟环境
    venv_python = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(os.getcwd(), '.venv', 'bin', 'python')
    if os.path.exists(venv_python):
        try:
            result = subprocess.run([venv_python, '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                interpreters.append({
                    'name': f'虚拟环境 Python ({version})',
                    'path': venv_python,
                    'is_default': False,
                    'is_venv': True
                })
        except:
            pass

    # 4. 检测常见 Python 安装路径 (Windows)
    if os.name == 'nt':
        common_paths = [
            r'C:\Python313\python.exe',
            r'C:\Python312\python.exe',
            r'C:\Python311\python.exe',
            r'C:\Python310\python.exe',
            r'C:\Python39\python.exe',
            r'C:\Program Files\Python313\python.exe',
            r'C:\Program Files\Python312\python.exe',
            r'C:\Program Files\Python311\python.exe',
            r'C:\Program Files\Python310\python.exe',
            r'C:\Program Files\Python39\python.exe',
        ]
        for p in common_paths:
            if os.path.exists(p) and not any(i['path'] == p for i in interpreters):
                try:
                    result = subprocess.run([p, '--version'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        interpreters.append({
                            'name': f'Python ({version})',
                            'path': p,
                            'is_default': False
                        })
                except:
                    pass

    # 按优先级排序：虚拟环境 > 系统默认 > 其他
    interpreters.sort(key=lambda x: (not x.get('is_venv', False), not x.get('is_default', False)))

    return {'success': True, 'interpreters': interpreters}

@app.get("/browse-directory")
async def browse_directory(path: str = "/"):
    """浏览服务器目录结构，用于选择文件"""
    import platform
    try:
        target_path = os.path.abspath(path)
        if not os.path.exists(target_path):
            target_path = os.path.expanduser("~")

        if not os.path.isdir(target_path):
            target_path = os.path.dirname(target_path)

        items = []
        # 父目录
        parent = os.path.dirname(target_path)
        if parent != target_path:
            items.append({
                "name": "..",
                "path": parent,
                "type": "dir",
                "size": 0
            })

        # 列出当前目录内容
        for entry in sorted(os.scandir(target_path), key=lambda e: (not e.is_dir(), e.name.lower())):
            # 跳过隐藏文件（.开头）
            if entry.name.startswith('.'):
                continue
            try:
                items.append({
                    "name": entry.name,
                    "path": os.path.abspath(entry.path),
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if not entry.is_dir() else 0
                })
            except PermissionError:
                continue

        # Windows 下返回盘符列表
        if platform.system() == "Windows" and target_path == "/":
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({
                        "name": f"{letter}:",
                        "path": drive,
                        "type": "dir",
                        "size": 0
                    })
            items = drives

        return {
            "success": True,
            "currentPath": target_path,
            "separator": os.sep,
            "items": items
        }
    except Exception as e:
        return {"success": False, "error": str(e)}



@app.get("/detect-python")
async def detect_python():
    """检测服务器上的 Python 解释器"""
    import subprocess
    import platform

    interpreters = []

    # 1. 检测系统默认 Python
    try:
        result = subprocess.run(['python', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            # 获取完整路径
            where_result = subprocess.run(['where', 'python'] if os.name == 'nt' else ['which', 'python'], capture_output=True, text=True, timeout=5)
            path = where_result.stdout.strip().split('\n')[0] if where_result.returncode == 0 else 'python'
            interpreters.append({
                'name': f'Python ({version})',
                'path': path,
                'is_default': True
            })
    except:
        pass

    # 2. 检测 python3
    try:
        result = subprocess.run(['python3', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            where_result = subprocess.run(['where', 'python3'] if os.name == 'nt' else ['which', 'python3'], capture_output=True, text=True, timeout=5)
            path = where_result.stdout.strip().split('\n')[0] if where_result.returncode == 0 else 'python3'
            interpreters.append({
                'name': f'Python3 ({version})',
                'path': path,
                'is_default': False
            })
    except:
        pass

    # 3. 检测当前虚拟环境
    venv_python = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(os.getcwd(), '.venv', 'bin', 'python')
    if os.path.exists(venv_python):
        try:
            result = subprocess.run([venv_python, '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                interpreters.append({
                    'name': f'虚拟环境 Python ({version})',
                    'path': venv_python,
                    'is_default': False,
                    'is_venv': True
                })
        except:
            pass

    # 4. 检测常见 Python 安装路径 (Windows)
    if os.name == 'nt':
        common_paths = [
            r'C:\Python313\python.exe',
            r'C:\Python312\python.exe',
            r'C:\Python311\python.exe',
            r'C:\Python310\python.exe',
            r'C:\Python39\python.exe',
            r'C:\Program Files\Python313\python.exe',
            r'C:\Program Files\Python312\python.exe',
            r'C:\Program Files\Python311\python.exe',
            r'C:\Program Files\Python310\python.exe',
            r'C:\Program Files\Python39\python.exe',
        ]
        for p in common_paths:
            if os.path.exists(p) and not any(i['path'] == p for i in interpreters):
                try:
                    result = subprocess.run([p, '--version'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        interpreters.append({
                            'name': f'Python ({version})',
                            'path': p,
                            'is_default': False
                        })
                except:
                    pass

    # 按优先级排序：虚拟环境 > 系统默认 > 其他
    interpreters.sort(key=lambda x: (not x.get('is_venv', False), not x.get('is_default', False)))

    return {'success': True, 'interpreters': interpreters}

@app.get("/browse-directory")
async def browse_directory(path: str = "/"):
    """浏览服务器目录结构，用于选择文件"""
    import platform
    try:
        target_path = os.path.abspath(path)
        if not os.path.exists(target_path):
            target_path = os.path.expanduser("~")

        if not os.path.isdir(target_path):
            target_path = os.path.dirname(target_path)

        items = []
        # 父目录
        parent = os.path.dirname(target_path)
        if parent != target_path:
            items.append({
                "name": "..",
                "path": parent,
                "type": "dir",
                "size": 0
            })

        # 列出当前目录内容
        for entry in sorted(os.scandir(target_path), key=lambda e: (not e.is_dir(), e.name.lower())):
            # 跳过隐藏文件（.开头）
            if entry.name.startswith('.'):
                continue
            try:
                items.append({
                    "name": entry.name,
                    "path": os.path.abspath(entry.path),
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if not entry.is_dir() else 0
                })
            except PermissionError:
                continue

        # Windows 下返回盘符列表
        if platform.system() == "Windows" and target_path == "/":
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({
                        "name": f"{letter}:",
                        "path": drive,
                        "type": "dir",
                        "size": 0
                    })
            items = drives

        return {
            "success": True,
            "currentPath": target_path,
            "separator": os.sep,
            "items": items
        }
    except Exception as e:
        return {"success": False, "error": str(e)}



@app.get("/detect-python")
async def detect_python():
    """检测服务器上的 Python 解释器"""
    import subprocess
    import platform

    interpreters = []

    # 1. 检测系统默认 Python
    try:
        result = subprocess.run(['python', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            # 获取完整路径
            where_result = subprocess.run(['where', 'python'] if os.name == 'nt' else ['which', 'python'], capture_output=True, text=True, timeout=5)
            path = where_result.stdout.strip().split('\n')[0] if where_result.returncode == 0 else 'python'
            interpreters.append({
                'name': f'Python ({version})',
                'path': path,
                'is_default': True
            })
    except:
        pass

    # 2. 检测 python3
    try:
        result = subprocess.run(['python3', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            where_result = subprocess.run(['where', 'python3'] if os.name == 'nt' else ['which', 'python3'], capture_output=True, text=True, timeout=5)
            path = where_result.stdout.strip().split('\n')[0] if where_result.returncode == 0 else 'python3'
            interpreters.append({
                'name': f'Python3 ({version})',
                'path': path,
                'is_default': False
            })
    except:
        pass

    # 3. 检测当前虚拟环境
    venv_python = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe') if os.name == 'nt' else os.path.join(os.getcwd(), '.venv', 'bin', 'python')
    if os.path.exists(venv_python):
        try:
            result = subprocess.run([venv_python, '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version = result.stdout.strip()
                interpreters.append({
                    'name': f'虚拟环境 Python ({version})',
                    'path': venv_python,
                    'is_default': False,
                    'is_venv': True
                })
        except:
            pass

    # 4. 检测常见 Python 安装路径 (Windows)
    if os.name == 'nt':
        common_paths = [
            r'C:\Python313\python.exe',
            r'C:\Python312\python.exe',
            r'C:\Python311\python.exe',
            r'C:\Python310\python.exe',
            r'C:\Python39\python.exe',
            r'C:\Program Files\Python313\python.exe',
            r'C:\Program Files\Python312\python.exe',
            r'C:\Program Files\Python311\python.exe',
            r'C:\Program Files\Python310\python.exe',
            r'C:\Program Files\Python39\python.exe',
        ]
        for p in common_paths:
            if os.path.exists(p) and not any(i['path'] == p for i in interpreters):
                try:
                    result = subprocess.run([p, '--version'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        interpreters.append({
                            'name': f'Python ({version})',
                            'path': p,
                            'is_default': False
                        })
                except:
                    pass

    # 按优先级排序：虚拟环境 > 系统默认 > 其他
    interpreters.sort(key=lambda x: (not x.get('is_venv', False), not x.get('is_default', False)))

    return {'success': True, 'interpreters': interpreters}

@app.get("/browse-directory")
async def browse_directory(path: str = "/"):
    """浏览服务器目录结构，用于选择文件"""
    import platform
    try:
        target_path = os.path.abspath(path)
        if not os.path.exists(target_path):
            target_path = os.path.expanduser("~")

        if not os.path.isdir(target_path):
            target_path = os.path.dirname(target_path)

        items = []
        # 父目录
        parent = os.path.dirname(target_path)
        if parent != target_path:
            items.append({
                "name": "..",
                "path": parent,
                "type": "dir",
                "size": 0
            })

        # 列出当前目录内容
        for entry in sorted(os.scandir(target_path), key=lambda e: (not e.is_dir(), e.name.lower())):
            # 跳过隐藏文件（.开头）
            if entry.name.startswith('.'):
                continue
            try:
                items.append({
                    "name": entry.name,
                    "path": os.path.abspath(entry.path),
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if not entry.is_dir() else 0
                })
            except PermissionError:
                continue

        # Windows 下返回盘符列表
        if platform.system() == "Windows" and target_path == "/":
            import string
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({
                        "name": f"{letter}:",
                        "path": drive,
                        "type": "dir",
                        "size": 0
                    })
            items = drives

        return {
            "success": True,
            "currentPath": target_path,
            "separator": os.sep,
            "items": items
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

class RunPythonRequest(BaseModel):
    code: str
    filename: str = 'script.py'
    pythonCommand: str = 'python'
    venvPath: str = ''
    projectRoot: str = ''
    requirementsPath: str = ''

@app.post("/run-python")
async def run_python(req: RunPythonRequest):
    """在服务器上运行 Python 代码"""
    import subprocess
    import tempfile
    import os
    import re
    global _running_procs, _proc_lock
    if _proc_lock is None:
        import threading
        _proc_lock = threading.Lock()

    logger.info(f"收到运行请求: {req}")
    logger.debug(f"req.filename={req.filename}")
    if not req.code:
        return {'success': False, 'error': '代码为空'}

    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(req.code)
            temp_file = f.name

        temp_file = os.path.join(req.projectRoot, req.filename)
        # 构建 Python 命令
        if req.venvPath:
            if os.name == 'nt':
                python_exe = os.path.join(req.venvPath, 'Scripts', 'python.exe')
            else:
                python_exe = os.path.join(req.venvPath, 'bin', 'python')

            if not os.path.exists(python_exe):
                python_exe = req.pythonCommand
        else:
            python_exe = req.pythonCommand



        def install_module(module_name: str):
            """安装指定模块"""
            install_process = subprocess.run(
                [python_exe, '-m', 'pip', 'install', module_name],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=req.projectRoot or os.getcwd()
            )
            logger.info(f"安装模块 {module_name} 结果: {install_process.stdout}")
            return install_process

        def run_code_nonblock():
            import time
            logger.info(f"正在执行: {python_exe} {temp_file}")

            # 禁用 Flask debug reloader
            try:
                with open(temp_file, 'r', encoding='utf-8') as f:
                    code_lines = f.readlines()
                modified = False
                for i, line in enumerate(code_lines):
                    if 'app.run(' in line:
                        code_lines[i] = '    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)\n'
                        modified = True
                        logger.info("已自动禁用 Flask debug reloader")
                        break
                if modified:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        f.writelines(code_lines)
            except Exception as _e:
                logger.warning(f"修改 Flask 代码失败: {_e}")

            # 输出到临时文件
            stdout_file = temp_file + '.out'
            stderr_file = temp_file + '.err'
            out_f = open(stdout_file, 'w', encoding='utf-8', buffering=1)
            err_f = open(stderr_file, 'w', encoding='utf-8', buffering=1)

            # Windows 上用 CREATE_NEW_PROCESS_GROUP + CREATE_NO_WINDOW
            creationflags = 0
            if os.name == 'nt':
                creationflags = 0x00000008 | 0x08000000

            p = subprocess.Popen(
                [python_exe, '-u', temp_file],
                stdout=out_f,
                stderr=err_f,
                cwd=req.projectRoot or os.getcwd(),
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                creationflags=creationflags
            )

            # 等 2 秒给 Flask 启动
            time.sleep(2)

            # 读取初始输出
            init_stdout = ''
            init_stderr = ''
            try:
                with open(stdout_file, 'r', encoding='utf-8', errors='replace') as f:
                    init_stdout = f.read()
            except:
                pass
            try:
                with open(stderr_file, 'r', encoding='utf-8', errors='replace') as f:
                    init_stderr = f.read()
            except:
                pass

            # 只存关键信息，不依赖 Python 的 Popen 对象（它可能失效）
            with _proc_lock:
                _running_procs[p.pid] = {
                    'pid': p.pid,
                    'stdout_file': stdout_file,
                    'stderr_file': stderr_file,
                    'temp_file': temp_file,
                    'out_f': out_f,
                    'err_f': err_f,
                    'stdout_pos': len(init_stdout),
                    'stderr_pos': len(init_stderr),
                    'ended': False,
                    'start_time': time.time(),
                }

            logger.info(f"Flask 进程已启动 PID={p.pid}")
            return {
                'pid': p.pid,
                'stdout': init_stdout,
                'stderr': init_stderr,
            }

        req_path = os.path.normpath(req.requirementsPath)
        if os.path.exists(req_path):
            with open(req_path, 'r') as f:
                for line in f:
                    module_name = line.strip()
                    if module_name:
                        if os.path.exists(module_name):
                            logger.info(f"正在安装模块: {module_name}")
                            install_module(module_name.strip())
        else:
            logger.info(f"requirements 文件不存在: {req_path}")

        run_info  = run_code_nonblock()

        # 立即返回 pid 和初始输出，后续通过 /run-output 轮询获取更多输出
        return {
            'success': True,
            'pid': run_info['pid'],
            'stdout': run_info['stdout'],
            'stderr': run_info['stderr'],
            'message': '程序已启动',
            'exitCode': None
        }

    # except subprocess.TimeoutExpired:
    #     return {'success': False, 'error': '执行超时（15秒）'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.post("/run-output")
async def run_output(req: dict = None):
    import subprocess
    if req is None:
        return {'success': False, 'error': '参数为空'}
    pid = req.get('pid')
    if not pid:
        return {'success': False, 'error': '缺少 pid'}

    try:
        pid = int(pid)
        with _proc_lock:
            info = _running_procs.get(pid)
            if not info:
                return {'success': True, 'stdout': '', 'stderr': '', 'ended': True, 'exitCode': None}

        # ========= 核心改动：用 tasklist 检测进程是否还在 =========
        ended = False
        exit_code = None
        try:
            if os.name == 'nt':
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                    capture_output=True, text=True, timeout=3
                )
                # 如果输出里找不到这个 PID，进程结束了
                if str(pid) not in result.stdout:
                    ended = True
            else:
                result = subprocess.run(
                    ['ps', '-p', str(pid), '-o', 'pid='],
                    capture_output=True, text=True, timeout=3
                )
                if str(pid) not in result.stdout:
                    ended = True
        except:
            # 检测失败，假设还在跑
            ended = False

        # 读取新增输出
        new_stdout = ''
        new_stderr = ''
        try:
            with open(info['stdout_file'], 'r', encoding='utf-8', errors='replace') as f:
                f.seek(info['stdout_pos'])
                new_stdout = f.read()
                info['stdout_pos'] = f.tell()
        except:
            pass
        try:
            with open(info['stderr_file'], 'r', encoding='utf-8', errors='replace') as f:
                f.seek(info['stderr_pos'])
                new_stderr = f.read()
                info['stderr_pos'] = f.tell()
        except:
            pass

        if ended and not info['ended']:
            info['ended'] = True
            try:
                info['out_f'].close(); info['err_f'].close();
            except:
                pass
            logger.info(f"进程 PID={pid} 已结束")

        return {
            'success': True,
            'stdout': new_stdout,
            'stderr': new_stderr,
            'ended': ended,
            'exitCode': exit_code
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.post("/run-stop")
async def run_stop(req: dict = None):
    import subprocess
    if req is None:
        return {'success': False, 'error': '参数为空'}
    pid = req.get('pid')
    if not pid:
        return {'success': False, 'error': '缺少 pid'}

    try:
        pid = int(pid)
        with _proc_lock:
            info = _running_procs.get(pid)
            if not info:
                return {'success': True, 'message': '进程不存在'}

        # Windows 上强制 kill 进程树
        if os.name == 'nt':
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(pid)],
                capture_output=True, timeout=5
            )
        else:
            subprocess.run(['kill', '-9', str(pid)], capture_output=True, timeout=3)

        # 清理
        try: info['out_f'].close(); info['err_f'].close();
        except: pass
        try: os.remove(info['temp_file']);
        except: pass

        return {'success': True, 'message': '已终止'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5001
    )