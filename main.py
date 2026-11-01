import asyncio
import json
import random
import os
import re
import sys
import time
import threading
import webview
from contextlib import asynccontextmanager
from pathlib import Path

_t0 = time.time()

# 打包后需显式从资源目录加载 .env，避免工作目录变化导致配置缺失
_ENV_FILE = Path(getattr(sys, '_MEIPASS', Path(__file__).parent)) / '.env'
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)

# 设置 HuggingFace 镜像，解决国内网络访问问题（必须在所有 import 之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'
# 显式清除离线模式，确保 fastembed 可以下载 BM25 模型
os.environ.pop('HF_HUB_OFFLINE', None)

print(f"[启动耗时] 环境变量设置: {time.time() - _t0:.2f}s")

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
print(f"[启动耗时] fastapi/uvicorn/pydantic: {time.time() - _t0:.2f}s")

# 懒加载 LangGraph 智能体框架（首次使用时才 import，避免阻塞启动）
SessionMemoryManager = None
create_agent_graph = None
AgentState = None

def _ensure_agent_langgraph():
    global SessionMemoryManager, create_agent_graph, AgentState
    if create_agent_graph is None:
        from core.agent_langgraph import SessionMemoryManager as _SMM, create_agent_graph as _CAG, AgentState as _AS
        SessionMemoryManager = _SMM
        create_agent_graph = _CAG
        AgentState = _AS
        print(f"[启动耗时] agent_langgraph (懒加载): {time.time() - _t0:.2f}s")

from typing import Optional
from datetime import datetime
from core.auth.database import SessionLocal, SessionMemory
print(f"[启动耗时] auth.database: {time.time() - _t0:.2f}s")

from settings.Define import PathConfig, GoEnvConfig, JavaEnvConfig, Params
print(f"[启动耗时] settings.Define: {time.time() - _t0:.2f}s")

from core.auth import init_db, get_current_user_optional, User
print(f"[启动耗时] core.auth: {time.time() - _t0:.2f}s")

from core.security import SecurityMiddleware, file_validator, get_audit_logger
print(f"[启动耗时] core.security: {time.time() - _t0:.2f}s")

from settings.logger_manager import get_logger
print(f"[启动耗时] logger_manager: {time.time() - _t0:.2f}s")

import version_config
print(f"[启动耗时] version_config: {time.time() - _t0:.2f}s")

# 懒加载 doc_processor
_doc_processor = None

def get_doc_processor():
    global _doc_processor
    if _doc_processor is None:
        from core.skills.DocProcess.document_process import DocumentProcessor
        _doc_processor = DocumentProcessor()
    return _doc_processor

logger = get_logger(__name__)

_running_procs = {}
_proc_lock = None
_shell_sessions = {}

agent_graph_dict = {}

class ChatRequest(BaseModel):
    message: str
    thread_id: int | None = None
    file_content: str | None = None
    file_name: str | None = None
    file_paths: list[str] | None = None
    original_file_ext: str | None = None
    history: list | None = None
    skill: str | None = None
    images: list | None = None
    project_context: dict | None = None
    model: str | None = None  # 用户选择的模型ID


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("lifespan - 开始初始化")
    print(" 正在初始化数据库...")
    init_db()
    print(" 数据库初始化完成")
    logger.info("lifespan - 数据库初始化完成")

    _load_project_path_index()

    def _preload_heavy_components():
        """在后台线程预加载重型组件，避免阻塞服务启动和窗口渲染"""
        try:
            logger.info("后台预加载 Reranker 模型...")
            from core.agents.routes import load_reranker
            load_reranker()
        except Exception as e:
            logger.warning(f"Reranker 预加载失败: {e}")

        try:
            logger.info("后台预加载知识库组件...")
            from core.knowledge.knowledge_manager import preload_knowledge_components
            preload_knowledge_components()
        except Exception as e:
            logger.warning(f"知识库组件预加载失败: {e}")

        try:
            logger.info("后台创建智能体图...")
            _ensure_agent_langgraph()
            agent_graph_dict["default"] = create_agent_graph()
            logger.info(" 智能体图创建完成")
        except Exception as e:
            logger.warning(f"智能体图创建失败: {e}")

    threading.Thread(target=_preload_heavy_components, daemon=True).start()

    yield

APP_TITLE = "小智"
app = FastAPI(title=APP_TITLE, lifespan=lifespan)

# 静态文件服务
app.mount("/static", StaticFiles(directory=str(PathConfig.BASE_DIR / "static")), name="static")

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

# 注册智能体管理路由
from core.agents import router as agents_router
app.include_router(agents_router)


@app.get("/", response_class=HTMLResponse)
async def index():
    """首页 - 官网首页"""
    index_file = os.path.join(PathConfig.TEMPLATES_DIR, "index.html")
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


@app.get("/agents", response_class=HTMLResponse)
async def agents_page():
    """智能体管理页面"""
    agent_file = os.path.join(PathConfig.TEMPLATES_DIR, "agents.html")
    with open(agent_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/agent-editor", response_class=HTMLResponse)
async def agent_editor_page():
    """智能体编辑页面"""
    editor_file = os.path.join(PathConfig.TEMPLATES_DIR, "agent-editor.html")
    with open(editor_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/publish", response_class=HTMLResponse)
async def publish_page():
    """智能体发布页面"""
    publish_file = os.path.join(PathConfig.TEMPLATES_DIR, "publish.html")
    with open(publish_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/agent-use", response_class=HTMLResponse)
async def agent_use_page():
    """智能体使用页面"""
    use_file = os.path.join(PathConfig.TEMPLATES_DIR, "agent-use.html")
    with open(use_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/products", response_class=HTMLResponse)
async def products_page():
    """产品下载页面"""
    products_file = os.path.join(PathConfig.TEMPLATES_DIR, "products.html")
    with open(products_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/products/online", response_class=HTMLResponse)
async def products_online_page():
    """在线使用页面 - 智能体聊天界面（原始主页）"""
    online_file = os.path.join(PathConfig.TEMPLATES_DIR, "online.html")
    with open(online_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/store", response_class=HTMLResponse)
async def store_page():
    """小智商店页面"""
    store_file = os.path.join(PathConfig.TEMPLATES_DIR, "store.html")
    with open(store_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/company/aboutus", response_class=HTMLResponse)
async def aboutus_page():
    """关于我们页面"""
    aboutus_file = os.path.join(PathConfig.TEMPLATES_DIR, "aboutus.html")
    with open(aboutus_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/download", response_class=HTMLResponse)
async def download_page():
    """下载页面 - 提供不同版本的下载选项"""
    download_file = os.path.join(PathConfig.TEMPLATES_DIR, "download.html")
    with open(download_file, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/release")
async def get_release_info():
    """返回当前正式版本的发布信息，供下载页面动态渲染"""
    release = version_config.RELEASE
    platforms = {
        key: {
            "label": platform["label"],
            "arch": platform["arch"],
            "artifacts": {
                artifact_key: {"filename": artifact["filename"]}
                for artifact_key, artifact in platform["artifacts"].items()
            },
        }
        for key, platform in release["platforms"].items()
    }
    return {
        "version": release["version"],
        "display_name": release["display_name"],
        "generation": release["generation"],
        "generation_label": release["generation_label"],
        "release_date": release["release_date"],
        "platforms": platforms,
    }


@app.get("/api/download/{version}")
async def download_version(version: str):
    """处理版本下载请求"""
    if version not in version_config.DOWNLOADS:
        raise HTTPException(status_code=404, detail="版本不存在")

    config = version_config.DOWNLOADS[version]
    file_path = config['path']

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在，请先构建安装包")

    return FileResponse(
        path=file_path,
        filename=config['filename'],
        media_type=config['content_type']
    )


class ScanRequest(BaseModel):
    path: str


class ReadFileRequest(BaseModel):
    path: str
    project_root: Optional[str] = None


class WriteFileRequest(BaseModel):
    path: str
    content: str
    project_root: Optional[str] = None


class DeleteFileRequest(BaseModel):
    path: str
    project_root: Optional[str] = None


class RenameFileRequest(BaseModel):
    path: str
    new_name: str
    project_root: Optional[str] = None
    search_references: bool = True
    apply_references: bool = True


class RenamePreviewRequest(BaseModel):
    path: str
    new_name: str
    project_root: Optional[str] = None
    search_references: bool = True


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
            # 如果 file_path 已经以 project_root 开头（带或不带分隔符），避免重复拼接
            if file_path_normalized.startswith(project_root_normalized + os.sep) or file_path_normalized.startswith(project_root_normalized):
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


@app.post("/write-file")
async def write_file(request: WriteFileRequest):
    """写入文件内容"""
    file_path = request.path
    project_root = request.project_root
    content = request.content
    logger.info(f"写入文件: {file_path}, 项目根路径: {project_root}, 内容长度: {len(content)}")
    
    try:
        # 如果已经是绝对路径，直接使用
        if os.path.isabs(file_path):
            full_path = file_path
        elif project_root:
            # 标准化路径分隔符
            file_path_normalized = file_path.replace('/', os.sep).replace('\\', os.sep)
            project_root_normalized = project_root.replace('/', os.sep).replace('\\', os.sep)
            # 如果 file_path 已经以 project_root 开头，避免重复拼接
            if file_path_normalized.startswith(project_root_normalized + os.sep) or file_path_normalized.startswith(project_root_normalized):
                full_path = file_path_normalized
            elif file_path_normalized == project_root_normalized:
                full_path = file_path_normalized
            else:
                full_path = os.path.join(project_root_normalized, file_path_normalized)
        else:
            full_path = os.path.join(os.getcwd(), file_path)
        
        # 确保目录存在
        dir_path = os.path.dirname(full_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"创建目录: {dir_path}")
        
        # 写入文件
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"写入文件成功: {full_path}")
        return {"success": True, "path": full_path}
    
    except Exception as e:
        logger.error(f"写入文件失败: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.post("/delete-file")
async def delete_file(request: DeleteFileRequest):
    """删除文件"""
    file_path = request.path
    project_root = request.project_root
    logger.info(f"删除文件: {file_path}, 项目根路径: {project_root}")
    
    try:
        # 如果已经是绝对路径，直接使用
        if os.path.isabs(file_path):
            full_path = file_path
        elif project_root:
            # 标准化路径分隔符
            file_path_normalized = file_path.replace('/', os.sep).replace('\\', os.sep)
            project_root_normalized = project_root.replace('/', os.sep).replace('\\', os.sep)
            # 如果 file_path 已经以 project_root 开头，避免重复拼接
            if file_path_normalized.startswith(project_root_normalized + os.sep) or file_path_normalized.startswith(project_root_normalized):
                full_path = file_path_normalized
            elif file_path_normalized == project_root_normalized:
                full_path = file_path_normalized
            else:
                full_path = os.path.join(project_root_normalized, file_path_normalized)
        else:
            full_path = os.path.join(os.getcwd(), file_path)
        
        if not os.path.exists(full_path):
            logger.warning(f"文件不存在: {full_path}")
            return {"success": False, "error": "文件不存在"}
        
        if os.path.isfile(full_path):
            os.remove(full_path)
            logger.info(f"删除文件成功: {full_path}")
        elif os.path.isdir(full_path):
            import shutil
            shutil.rmtree(full_path)
            logger.info(f"删除目录成功: {full_path}")
        
        return {"success": True, "path": full_path}
    
    except Exception as e:
        logger.error(f"删除文件失败: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}


_REFERENCE_CODE_EXTS = {
    '.py', '.pyi', '.pyw', '.ipynb',
    '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.vue', '.svelte',
    '.java', '.kt', '.kts', '.scala', '.groovy', '.gradle',
    '.c', '.h', '.cc', '.cpp', '.cxx', '.hpp', '.hh', '.hxx', '.m', '.mm',
    '.cs', '.fs', '.go', '.rs', '.rb', '.php', '.swift', '.lua', '.dart', '.r',
    '.sh', '.bash', '.bat', '.cmd', '.ps1',
    '.html', '.htm', '.css', '.scss', '.sass', '.less',
    '.json', '.xml', '.yml', '.yaml', '.toml', '.ini', '.cfg', '.conf',
    '.md', '.txt', '.sql', '.cmake', '.mk', '.make',
}

_REFERENCE_SKIP_DIRS = {
    '.git', '.svn', '.hg', '.idea', '.vs', '.vscode',
    'node_modules', '__pycache__', 'venv', '.venv', 'env',
    'build', 'dist', 'out', 'obj', 'bin', 'target',
    'chroma_db', 'uploads', 'outputs', 'logs', 'cache',
}

_REFERENCE_MAX_FILE_SIZE = 5 * 1024 * 1024


def _skip_reference_dir(name):
    """引用搜索时需要跳过的目录（隐藏目录、依赖目录与构建产物）"""
    low = name.lower()
    return low.startswith('.') or low in _REFERENCE_SKIP_DIRS


def _resolve_full_path(file_path, project_root):
    """把相对路径结合项目根目录解析为绝对路径"""
    if not file_path:
        return ''
    if os.path.isabs(file_path):
        return os.path.normpath(file_path)
    if project_root:
        fp = file_path.replace('/', os.sep).replace('\\', os.sep)
        pr = project_root.replace('/', os.sep).replace('\\', os.sep)
        if fp.startswith(pr + os.sep) or fp == pr:
            return os.path.normpath(fp)
        return os.path.normpath(os.path.join(pr, fp))
    return os.path.normpath(os.path.join(os.getcwd(), file_path))


def _read_text_with_encoding(path):
    """读取文本文件并返回 (内容, 编码)，保持原编码以便回写"""
    for enc in ('utf-8', 'utf-8-sig', 'gbk', 'gb2312'):
        try:
            with open(path, 'r', encoding=enc, newline='') as f:
                return f.read(), enc
        except (UnicodeDecodeError, LookupError):
            continue
        except OSError:
            return None, None
    return None, None


_IMPORT_LINE_HINT = re.compile(
    r'\b(import|from|require|include|using|extern|package|module|export|load)\b',
    re.IGNORECASE
)
_PATH_LINE_HINT = re.compile(r'[/\\]')


class _RenameRules:
    """重命名引用替换规则：完整文件名全局替换，主体名仅在引用语句或路径中出现时替换"""

    def __init__(self, old_name, new_name, is_dir):
        self.old_name = old_name
        self.new_name = new_name
        if is_dir:
            self.stem_old, self.stem_new = old_name, new_name
            self.full_re = None
        else:
            self.stem_old = os.path.splitext(old_name)[0]
            self.stem_new = os.path.splitext(new_name)[0]
            self.full_re = (
                re.compile(r'(?<![A-Za-z0-9_])' + re.escape(old_name) + r'(?![A-Za-z0-9_])')
                if old_name != self.stem_old else None
            )
        self.stem_re = (
            re.compile(r'(?<![A-Za-z0-9_])' + re.escape(self.stem_old) + r'(?![A-Za-z0-9_])')
            if self.stem_old and self.stem_old != self.stem_new else None
        )

    @property
    def active(self):
        return bool(self.full_re or self.stem_re)

    def apply_line(self, line):
        out = line
        if self.full_re:
            out = self.full_re.sub(self.new_name, out)
        if self.stem_re and (_IMPORT_LINE_HINT.search(out) or _PATH_LINE_HINT.search(out)):
            out = self.stem_re.sub(self.stem_new, out)
        return out

    def apply_text(self, text):
        """返回 (新文本, 发生变化的行号列表)"""
        lines = text.splitlines(keepends=True)
        changed = []
        for idx, line in enumerate(lines):
            new_line = self.apply_line(line)
            if new_line != line:
                lines[idx] = new_line
                changed.append(idx + 1)
        return ''.join(lines), changed


def _scan_references(root, rules, apply=False):
    """在项目内代码文件中查找（并可写入）引用替换，返回 (匹配列表, 替换总数)"""
    matches = []
    total = 0
    root_abs = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root_abs):
        dirnames[:] = [d for d in dirnames if not _skip_reference_dir(d)]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in _REFERENCE_CODE_EXTS:
                continue
            path = os.path.join(dirpath, fn)
            try:
                if os.path.getsize(path) > _REFERENCE_MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            content, enc = _read_text_with_encoding(path)
            if content is None:
                continue

            new_content, changed_lines = rules.apply_text(content)
            if not changed_lines:
                continue

            matches.append({
                'path': os.path.relpath(path, root_abs).replace('\\', '/'),
                'lines': changed_lines[:30],
                'count': len(changed_lines),
            })
            total += len(changed_lines)

            if apply:
                try:
                    with open(path, 'w', encoding=enc or 'utf-8', newline='') as f:
                        f.write(new_content)
                except OSError as e:
                    logger.warning(f"[Rename] 写入引用失败: {path} -> {e}")
    return matches, total


def _rename_reference_root(full_path, project_root):
    """确定引用搜索范围：优先项目根目录，否则用目标所在目录"""
    if project_root:
        pr = os.path.abspath(project_root.replace('/', os.sep).replace('\\', os.sep))
        if os.path.isdir(pr):
            return pr
    return os.path.dirname(full_path)


@app.post("/rename-preview")
async def rename_preview(request: RenamePreviewRequest):
    """预览重命名将影响到的引用位置（不修改任何文件）"""
    try:
        full_path = _resolve_full_path(request.path, request.project_root)
        if not os.path.exists(full_path):
            return {'success': False, 'error': '文件或目录不存在'}

        new_name = (request.new_name or '').strip()
        if not new_name:
            return {'success': False, 'error': '新名称不能为空'}

        is_dir = os.path.isdir(full_path)
        if not request.search_references:
            return {'success': True, 'matches': [], 'total': 0, 'is_dir': is_dir}

        rules = _RenameRules(os.path.basename(full_path), new_name, is_dir)
        if not rules.active:
            return {'success': True, 'matches': [], 'total': 0, 'is_dir': is_dir}

        root = _rename_reference_root(full_path, request.project_root)
        matches, total = _scan_references(root, rules, apply=False)
        return {'success': True, 'matches': matches, 'total': total, 'is_dir': is_dir}
    except Exception as e:
        logger.error(f"[RenamePreview] 失败: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


@app.post("/rename-file")
async def rename_file(request: RenameFileRequest):
    """重命名文件/目录，可选同步修改项目内对该名称的引用"""
    try:
        full_path = _resolve_full_path(request.path, request.project_root)
        if not os.path.exists(full_path):
            return {'success': False, 'error': '文件或目录不存在'}

        new_name = (request.new_name or '').strip()
        if not new_name:
            return {'success': False, 'error': '新名称不能为空'}
        if '/' in new_name or '\\' in new_name:
            return {'success': False, 'error': '新名称不能包含路径分隔符'}

        old_name = os.path.basename(full_path)
        is_dir = os.path.isdir(full_path)
        new_full_path = os.path.join(os.path.dirname(full_path), new_name)

        if os.path.abspath(new_full_path) == os.path.abspath(full_path):
            return {'success': True, 'path': new_full_path, 'new_name': new_name,
                    'is_dir': is_dir, 'updated_files': [], 'total': 0}

        if os.path.exists(new_full_path):
            return {'success': False, 'error': f'目标名称已存在: {new_name}'}

        os.rename(full_path, new_full_path)
        logger.info(f"[Rename] 重命名成功: {full_path} -> {new_full_path}")

        updated_files = []
        total = 0
        if request.search_references and request.apply_references:
            rules = _RenameRules(old_name, new_name, is_dir)
            if rules.active:
                root = _rename_reference_root(new_full_path, request.project_root)
                updated_files, total = _scan_references(root, rules, apply=True)
                if total:
                    logger.info(f"[Rename] {old_name} -> {new_name} 同步修改引用 {total} 处")

        return {
            'success': True,
            'path': new_full_path,
            'new_name': new_name,
            'is_dir': is_dir,
            'updated_files': updated_files,
            'total': total,
        }
    except Exception as e:
        logger.error(f"[Rename] 重命名失败: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


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
                    if entry.name.endswith('.py.out') or entry.name.endswith('.py.err'):
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
    file_path, saved_filename = get_doc_processor().save_uploaded_file(content, file.filename)
    
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


@app.api_route("/download/{filename}", methods=["GET", "HEAD"])
async def download_file(filename: str):
    """提供文件下载"""
    file_path = os.path.join(PathConfig.OUTPUTS_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)


@app.post("/api/clear_chat")
async def clear_chat_msg(request: Request):
    """清空聊天记录，并删除缓存文件"""
    try:
        _ensure_agent_langgraph()
        body = await request.json()
        filename = body.get("filename", "")
        thread_id = body.get("thread_id", "")

        if not thread_id:
            return {"success": False, "message": "会话ID不能为空"}

        logger.info(f"开始清空对话记录: {thread_id}")

        # 清空session_memories表当前会话ID的chat_msg字段内容
        db = SessionLocal()
        try:
            db.query(SessionMemory).filter(SessionMemory.thread_id == thread_id).update({SessionMemory.chat_msg: [], SessionMemory.context: {},
                                                                                         SessionMemory.update_at: datetime.now()})
            db.commit()
        finally:
            db.close()

        # 删除缓存文件（如果有）
        if filename and os.path.exists(os.path.join(PathConfig.CACHE_DIR, filename)):
            cache_dir = PathConfig.CACHE_DIR
            file_path = cache_dir / filename

            if str(file_path.resolve()).startswith(str(cache_dir.resolve())) and file_path.exists():
                file_path.unlink()
                logger.info(f"[CACHE] 已删除缓存文件: {filename}")

        return {"success": True, "message": "对话已清空"}
    except Exception as e:
        logger.error(f"[Memory] 清空对话失败: {e}")
        return {"success": False, "message": str(e)}


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

    _ensure_agent_langgraph()

    # 记录请求开始时间（用于计算后端总处理耗时）
    request_start_time = time.time()

    async def generate():
        async def _check_disconnected():
            if await http_request.is_disconnected():
                logger.info("[STOP] 客户端已断开连接，中止生成")
                raise asyncio.CancelledError("客户端断开")

        try:
            file_path_for_state = None
            all_file_paths_for_state = []
            message_content = request.message

            # 处理多文件上传
            if request.file_paths and len(request.file_paths) > 0:
                logger.info(f"检测到 {len(request.file_paths)} 个文件: {request.file_name}，正在处理...")
                
                # 收集所有文件路径
                all_file_paths = []
                all_file_contents = []
                all_images_data = []
                
                DRAWING_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf', '.dxf'}
                is_drawing_file = request.skill == 'drawing'
                
                for idx, fpath in enumerate(request.file_paths):
                    file_path = Path(fpath)
                    file_ext = file_path.suffix.lower()
                    
                    if not file_path.exists():
                        logger.warning(f"文件不存在: {file_path}")
                        continue
                    
                    all_file_paths.append(str(file_path))
                    all_file_paths_for_state.append(str(file_path))
                    
                    # 如果是图纸文件，不读取文本内容
                    if file_ext in DRAWING_EXTS:
                        is_drawing_file = True
                        continue
                    
                    # 读取文档内容
                    try:
                        file_content = ""
                        async for chunk_text in get_doc_processor().read_document(str(file_path)):
                            file_content += chunk_text
                        all_file_contents.append({
                            'name': request.file_name.split(', ')[idx] if request.file_name else f'file_{idx}',
                            'ext': file_ext,
                            'content': file_content
                        })
                        logger.info(f"文档内容读取成功: {file_path.name}, 长度: {len(file_content)}")
                    except Exception as e:
                        logger.error(f"读取文档失败 {file_path}: {str(e)}")
                
                # 构建消息内容
                if is_drawing_file:
                    # 图纸文件：不读取文本内容
                    message_content = request.message
                    logger.info("图纸文件，跳过文本读取")
                elif all_file_contents:
                    # 组合所有文档内容
                    content_parts = []
                    for fc in all_file_contents:
                        content_parts.append(f"--- {fc['name']} ---\n{fc['content']}")
                    
                    combined_content = "\n\n".join(content_parts)
                    message_content = f"[文档内容]\n{combined_content}\n\n[用户要求]\n{request.message}"
                    logger.info(f"多文档内容组合成功，总长度: {len(combined_content)}")
                else:
                    message_content = request.message
            
            elif request.file_name:
                logger.info(f"检测到文件: {request.file_name}，正在处理...")
                file_path = get_doc_processor().UPLOAD_DIR / request.file_name
                file_ext = Path(request.file_name).suffix.lower()
                # 图纸类文件：图片、DXF、PDF，不读取文本内容，只传路径
                DRAWING_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.pdf', '.dxf'}
                is_drawing_file = file_ext in DRAWING_EXTS or request.skill == 'drawing'

                if file_path.exists():
                    file_path_for_state = str(file_path)
                    all_file_paths_for_state = [str(file_path)]
                    # 存进会话记忆的 context，后续追问可自动恢复
                    SessionMemoryManager.set_context(
                        thread_id,
                        upload_files=[{
                            "file_path": file_path_for_state,
                            "file_name": request.file_name,
                            "ext": file_ext,
                        }],
                    )

                    logger.info(f"[Memory] 保存文件上下文: file_path={file_path_for_state}")

                    if is_drawing_file:
                        # 图纸文件：不读取文本内容，由 drawing_node 直接处理
                        message_content = request.message
                        logger.info(f"图纸文件，跳过文本读取: {file_path}")
                    elif not request.file_content:
                        try:
                            file_content = ""
                            async for chunk_text in get_doc_processor().read_document(str(file_path)):
                                file_content += chunk_text

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

            # 没上传文件时，尝试从会话记忆恢复上一次的文件路径
            if not file_path_for_state and SessionMemoryManager.has_memory(thread_id):
                _mem_file_path = SessionMemoryManager.get_context(thread_id, "file_path")
                if _mem_file_path and Path(_mem_file_path).exists():
                    file_path_for_state = _mem_file_path
                    logger.info(f"[Memory] 从会话记忆恢复文件路径: {_mem_file_path}")

            # Build messages list with history
            from langchain_core.messages import HumanMessage
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

            state_input = {"messages": messages_list, "thread_id": thread_id}
            if file_path_for_state:
                state_input["file_path"] = file_path_for_state
            if all_file_paths_for_state:
                state_input["file_paths"] = all_file_paths_for_state
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
            await _check_disconnected()
            yield f"data: {json.dumps({'text': '', 'status': 'thinking'})}\n\n"
            await asyncio.sleep(0.1)
            
            result = None
            response = "暂无响应"
            file_changes = []
            sources = []
            
            # 记录后端处理开始时间
            backend_start_time = time.time()
            
            try:
                from langchain_core.messages import HumanMessage

                # 构建状态输入
                state_input = {
                    "messages": messages_list,
                    "type": "",  # 空字符串，让 supervisor_node 自动分类
                    "file_path": file_path_for_state or "",
                    "file_paths": all_file_paths_for_state or [],
                    "previous_node": "",
                    "skill": request.skill or "",
                    "sources": "",
                    "user": current_user.username if current_user else "",
                    "images": request.images or [],
                    "project_context": request.project_context or {},
                    "thread_id": thread_id or "",
                    "model_name": Params.resolve_model_name(request.model)
                }

                logger.info(f"开始执行 LangGraph 智能体图，skill={request.skill}")

                # 创建智能体图
                agent_graph = agent_graph_dict.get(thread_id or "default", create_agent_graph())
                if thread_id:
                    agent_graph_dict[str(thread_id)] = agent_graph

                # 配置（用于 checkpointer 的 thread_id）
                config = {"configurable": {"thread_id": thread_id or "default"}}

                full_response = ""
                chunk_count = 0
                last_sources = []

                # 使用 astream 实现真正的流式输出
                async for event in agent_graph.astream(state_input, config=config, stream_mode="custom"):
                    print(f"event: {event}")
                    await _check_disconnected()

                    # event 格式: (node_name, custom_data)
                    if isinstance(event, tuple) and len(event) == 2:
                        node_name, custom_data = event
                    else:
                        # 兼容旧格式
                        node_name = "unknown"
                        custom_data = event

                    logger.info(f"节点 {node_name} 流式输出")

                    # 从自定义数据中提取内容，保留原始事件的完整字段（status/step/step_detail）
                    if isinstance(custom_data, dict):
                        event_status = custom_data.get("status", "streaming")

                        # 只有 streaming 状态的文本才累积到完整响应
                        if "text" in custom_data and event_status == "streaming":
                            full_response += custom_data["text"]
                            chunk_count += 1

                        # 提取 sources
                        if "sources" in custom_data:
                            last_sources = custom_data["sources"]

                        # 转发完整事件数据到前端，保留 status/step/step_detail 等字段
                        event_payload = dict(custom_data)
                        if "sources" not in event_payload:
                            event_payload["sources"] = last_sources

                        yield f"data: {json.dumps(event_payload, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.01)

                response = full_response
                sources = last_sources
                logger.info(f"LangGraph 流式响应完成，内容长度={len(response)}，chunk数量={chunk_count}")
                # 提取文件变更信息（project_code 技能时都提取，包括多模态识图）
                if request.skill == "project_code":
                    # 获取项目根路径和文件列表，优先使用完整路径
                    project_root = ""
                    project_files = []
                    if request.project_context:
                        project_root = request.project_context.get("project_path", "") or request.project_context.get("project_name", "")
                        project_files = request.project_context.get("files", [])
                    logger.debug(f"extract_file_changes 调用参数: project_root={project_root}, project_files={len(project_files)}")
                    file_changes = extract_file_changes(response, request.file_name, project_root)
                    logger.debug(f"extract_file_changes 返回结果: {file_changes}")
            except asyncio.TimeoutError:
                logger.error("Agent graph execution timeout")
                response = "请求超时，请稍后重试或简化您的问题"
            except Exception as e:
                err_msg = str(e)
                logger.error(f"Agent graph execution error: {err_msg}")

                if '403' in err_msg and ('Free quota exhausted' in err_msg or 'AllocationQuota' in err_msg):
                    response = " API 免费额度已用完，请在阿里云 DashScope 控制台充值后再试，或切换其他模型。"
                elif '401' in err_msg or 'Invalid API key' in err_msg or 'authentication' in err_msg.lower():
                    response = " API Key 无效，请检查环境变量 DASHSCOPE_API_KEY 是否正确配置。"
                elif '429' in err_msg or 'rate limit' in err_msg.lower() or 'Too Many Requests' in err_msg:
                    response = " 请求频率过高，请稍后再试。"
                elif 'timed out' in err_msg.lower() or 'TimeoutError' in err_msg:
                    response = " 模型服务响应超时，请稍后重试。"
                else:
                    response = f" 服务执行出错: {err_msg[:200]}"

            # 保存本轮对话到 LangChain 会话短期记忆
            try:
                SessionMemoryManager.save(thread_id, message_content, response)
                logger.info(f"[Memory] 保存会话 thread_id={thread_id}, user_len={len(message_content)}, ai_len={len(response)}, total_sessions={SessionMemoryManager.session_count()}")
            except Exception as _mem_err:
                logger.warning(f"[Memory] 保存失败: {str(_mem_err)[:80]}")

            # 所有分支都通过 streaming 逐块发送过了，done 只做结束信号，text 为空
            backend_elapsed = round(time.time() - request_start_time, 3)
            logger.info(f"[耗时统计] 后端处理总耗时: {backend_elapsed}s")
            yield f"data: {json.dumps({'text': '', 'sources': sources, 'file_changes': file_changes, 'status': 'done', 'response_time_s': backend_elapsed})}\n\n"
        except asyncio.CancelledError:
            partial = ""
            try:
                partial = full_response
            except UnboundLocalError:
                partial = ""
            logger.info(f"[STOP] 生成被中止，保存已生成内容，长度={len(partial)}")
            try:
                yield f"data: {json.dumps({'text': '', 'status': 'stop'})}\n\n"
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'text': f'服务错误: {str(e)}', 'status': 'error'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/memory/clear/{thread_id}")
async def delete_memory(thread_id: str):
    # 删除指定会话记忆中的所有数据
    if not thread_id:
        return {"message": "会话ID不能为空"}

    try:
        _ensure_agent_langgraph()
        SessionMemoryManager.clear(thread_id)
        return {"message": f"会话 {thread_id} 已删除"}
    except Exception as e:
        logger.error(f"删除会话 {thread_id} 失败: {str(e)}")
        return {"message": f"删除会话 {thread_id} 失败: {str(e)}"}


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
                bool(re.search(r'[\[\{\'\"]', content, re.MULTILINE)) or
                bool(re.search(r'print\(', content, re.MULTILINE)) or
                bool(re.search(r'console\.', content, re.MULTILINE)) or
                bool(re.search(r'={3,}$', content, re.MULTILINE)) or
                bool(re.search(r'-+.*测试.*-+$', content, re.MULTILINE))
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
        r'\s*让我思考一下[.。\s]*',
        r'\s*Let me think[.。\s]*',
        r'\s*继续分析中[.。\s]*',
        r'\s*深入研究中[.。\s]*',
        r'\s*发现关键点[.。\s]*',
        r'\s*正在处理[.。\s]*',
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
    code_blocks = re.findall(r'(\w+)?\n([\s\S]*?)', content)
    
    for idx, (lang, code) in enumerate(code_blocks):
        code = code.strip()
        
        # 跳过空代码块
        if not code:
            continue
            
        # 跳过目录结构（tree命令输出）
        if '' in code or '' in code or '' in code:
            continue
            
        # 跳过测试输出和运行示例（加强过滤）
        is_test_output = re.search(r'={3,}$', code, re.MULTILINE) or \
                        '测试通过' in code or \
                        '测试结果' in code or \
                        '运行示例' in code or \
                        '控制台输出' in code or \
                        '输出示例' in code or \
                        '>>> ' in code or \
                        re.search(r'\d+! = \d+', code) or \
                        re.search(r'---.*阶乘.*---$', code, re.MULTILINE) or \
                        re.search(r'---.*测试.*---$', code, re.MULTILINE) or \
                        (re.search(r'! = \d+', code) and not 'def ' in code and not 'class ' in code)
        if is_test_output:
            logger.info(f"跳过测试输出代码块: {code[:50]}...")
            continue
        
        # 清理代码中的思考标记（LLM可能将思考文本混入代码块）
        thinking_patterns = [
            r'\s*让我思考一下[.。\s]*',
            r'\s*Let me think[.。\s]*',
            r'\s*继续分析中[.。\s]*',
            r'\s*深入研究中[.。\s]*',
            r'\s*发现关键点[.。\s]*',
            r'\s*正在处理[.。\s]*',
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
            start = content.find('', pos)
            if start == -1:
                break
            end = content.find('', start + 3)
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
        file_path_match = re.search(r'文件路径[：:]\s*([\s\n]+)', before_text, re.IGNORECASE)
        # 方法1b：查找"### 文件：xxx"标记（新格式）
        if not file_path_match:
            file_path_match = re.search(r'###\s*文件[：:]\s*([\s\n]+)', before_text, re.IGNORECASE)
        
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
            file_name = file_name.replace('', '')
            # 修复重复扩展名
            file_name = re.sub(r'\.py\.py$', '.py', file_name, flags=re.IGNORECASE)
            file_name = re.sub(r'\.js\.js$', '.js', file_name, flags=re.IGNORECASE)
        elif file_name_from_context:
            file_name = file_name_from_context
            print(f"[DEBUG] 使用上下文提取的文件名: {file_name}")
        else:
            # 方法2：查找"新增文件:"或"新建文件:"标记
            new_file_match = re.search(r'(新增文件|新建文件)[：:]\s*([\s\n]+)', before_text, re.IGNORECASE)
            if new_file_match:
                file_name = new_file_match.group(2).strip()
                # 移除反引号
                file_name = file_name.replace('', '')
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
    code_blocks = re.findall(r'(\w+)?\n([\s\S]*?)', content)
    
    for lang, code in code_blocks:
        # 检查代码块前面是否有文件说明
        lines = content.split('\n')
        file_name = current_file or "未知文件"
        
        # 查找代码块前面的文件名
        for i, line in enumerate(lines):
            if f'{lang}' in line or (lang == '' and '' in line):
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

    # 4. 从系统环境变量 PATH 中检测 Python 安装路径 (Windows)
    if os.name == 'nt':
        path_env = os.environ.get('PATH', '')
        path_dirs = path_env.split(';')
        
        for path_dir in path_dirs:
            path_dir = path_dir.strip()
            if not path_dir:
                continue
            
            # 检查该目录下是否有 python.exe
            python_exe_path = os.path.join(path_dir, 'python.exe')
            
            # 跳过已检测过的路径
            if os.path.exists(python_exe_path) and not any(i['path'] == python_exe_path for i in interpreters):
                try:
                    result = subprocess.run([python_exe_path, '--version'], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        version = result.stdout.strip()
                        record = {
                            'name': f'Python ({version})',
                            'path': python_exe_path,
                            'is_default': False
                        }
                        if record not in interpreters:
                            interpreters.append(record)
                except:
                    pass

    # 按优先级排序：虚拟环境 > 系统默认 > 其他
    interpreters.sort(key=lambda x: (not x.get('is_venv', False), not x.get('is_default', False)))

    # 去重：基于 path 字段去重，保留优先级高的（虚拟环境优先，其次是系统默认）
    seen_paths = set()
    unique_interpreters = []
    for interpreter in interpreters:
        if interpreter.get('path') in seen_paths:
            continue
        seen_paths.add(interpreter.get('path'))
        logger.info(f"添加解释器: {interpreter.get('name')}")
        unique_interpreters.append(interpreter)

    interpreters = unique_interpreters

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

# 项目名 -> 绝对路径 的内存索引，供"最近的项目"直接定位目录
_PROJECT_PATH_INDEX = {}
_PROJECT_PATH_INDEX_LOCK = threading.Lock()
_PROJECT_PATH_CACHE_FILE = os.path.join(os.path.expanduser('~'), '.xiaozhi_project_paths.json')


def _load_project_path_index():
    """启动时从本地缓存加载项目路径索引"""
    global _PROJECT_PATH_INDEX
    try:
        if os.path.exists(_PROJECT_PATH_CACHE_FILE):
            with open(_PROJECT_PATH_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                with _PROJECT_PATH_INDEX_LOCK:
                    _PROJECT_PATH_INDEX = data
                logger.info(f"[ProjectIndex] 已加载 {len(_PROJECT_PATH_INDEX)} 条项目路径记录")
    except Exception as e:
        logger.warning(f"[ProjectIndex] 加载缓存失败: {e}")


def _save_project_path_index():
    """将项目路径索引持久化到本地缓存"""
    try:
        with _PROJECT_PATH_INDEX_LOCK:
            data = dict(_PROJECT_PATH_INDEX)
        with open(_PROJECT_PATH_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[ProjectIndex] 保存缓存失败: {e}")


def _remember_project_path(project_path):
    """记录项目名与真实绝对路径的映射"""
    if not project_path or not os.path.isdir(project_path):
        return
    project_name = os.path.basename(os.path.normpath(project_path))
    if not project_name:
        return
    with _PROJECT_PATH_INDEX_LOCK:
        if _PROJECT_PATH_INDEX.get(project_name) == project_path:
            return
        _PROJECT_PATH_INDEX[project_name] = project_path
    logger.info(f"[ProjectIndex] 记录项目路径: {project_name} -> {project_path}")
    _save_project_path_index()


def _lookup_project_path(project_name):
    """按项目名查找已记录的绝对路径（目录不存在时返回 None）"""
    if not project_name:
        return None
    with _PROJECT_PATH_INDEX_LOCK:
        cached = _PROJECT_PATH_INDEX.get(project_name)
    if cached and os.path.isdir(cached):
        return cached
    return None


@app.get("/find-project-path")
async def find_project_path(project_name: str):
    """在服务器上搜索项目的真实路径"""
    try:
        import glob

        # 优先命中历史导入记录（项目名 -> 绝对路径）
        cached_path = _lookup_project_path(project_name)
        if cached_path:
            logger.info(f"[FindProject] 命中历史记录: {project_name} -> {cached_path}")
            return {"success": True, "path": cached_path}

        # 定义搜索目录列表（按优先级）
        search_dirs = [
            os.path.dirname(os.getcwd()),  # 当前工作目录的父目录 (D:\Python_project)
            os.path.expanduser('~'),       # 用户主目录
            os.getcwd(),                   # 当前工作目录
        ]

        # 补充已知项目的父目录（便于同盘其他项目被快速命中）
        with _PROJECT_PATH_INDEX_LOCK:
            known_paths = list(_PROJECT_PATH_INDEX.values())
        search_dirs.extend(os.path.dirname(p) for p in known_paths if p)
        
        # 去重
        search_dirs = list(dict.fromkeys(d for d in search_dirs if d))
        
        logger.info(f"[FindProject] 开始搜索项目: {project_name}, 搜索目录: {search_dirs}")
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
                
            search_pattern = os.path.join(search_dir, '**', project_name)
            matches = glob.glob(search_pattern, recursive=True)
            
            # 过滤出目录
            dirs = [m for m in matches if os.path.isdir(m)]
            
            if dirs:
                # 返回第一个匹配的路径
                logger.info(f"[FindProject] 找到项目: {dirs[0]}")
                _remember_project_path(dirs[0])
                return {"success": True, "path": dirs[0]}
        
        logger.warning(f"[FindProject] 未找到项目: {project_name}")
        return {"success": False, "error": f"未找到项目: {project_name}"}
    except Exception as e:
        logger.error(f"[FindProject] 搜索失败: {e}")
        return {"success": False, "error": str(e)}

@app.get("/scan-project")
async def scan_project(project_path: str):
    """扫描项目目录，返回文件树结构（用于 PyWebView 环境）"""
    try:
        if not os.path.exists(project_path):
            return {"success": False, "error": f"路径不存在: {project_path}"}

        # 记录项目名与真实路径的映射，便于"最近的项目"直接定位
        _remember_project_path(project_path)
        
        def scan_directory(path, relative_path=''):
            items = []
            try:
                for entry in os.scandir(path):
                    entry_relative = os.path.join(relative_path, entry.name).replace('\\', '/')
                    if entry.is_file():
                        items.append({
                            'name': entry.name,
                            'path': entry_relative,
                            'type': 'file'
                        })
                    elif entry.is_dir() and not entry.name.startswith('.'):
                        children = scan_directory(entry.path, entry_relative)
                        items.append({
                            'name': entry.name,
                            'path': entry_relative,
                            'type': 'folder',
                            'children': children,
                            'expanded': False
                        })
            except PermissionError:
                pass
            return items
        
        tree = scan_directory(project_path)
        return {"success": True, "tree": tree}
    except Exception as e:
        logger.error(f"[ScanProject] 扫描失败: {e}")
        return {"success": False, "error": str(e)}

@app.get("/project-signature")
async def project_signature(project_path: str):
    """返回项目目录结构的指纹，用于前端轮询判断目录树是否需要刷新"""
    try:
        if not os.path.exists(project_path):
            return {"success": False, "error": f"路径不存在: {project_path}"}

        import hashlib

        parts = []
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', '__pycache__')]
            relative_root = os.path.relpath(root, project_path).replace('\\', '/')
            for name in sorted(files):
                full_path = os.path.join(root, name)
                try:
                    stat = os.stat(full_path)
                except OSError:
                    continue
                parts.append(f"{relative_root}/{name}:{stat.st_mtime_ns}:{stat.st_size}")
            for name in sorted(dirs):
                parts.append(f"{relative_root}/{name}/")

        signature = hashlib.md5('|'.join(parts).encode('utf-8')).hexdigest()
        return {"success": True, "signature": signature, "count": len(parts)}
    except Exception as e:
        logger.error(f"[ProjectSignature] 计算失败: {e}")
        return {"success": False, "error": str(e)}


class FindSymbolRequest(BaseModel):
    project_root: str
    file_path: str = ''
    symbol: str
    mode: str = 'references'
    current_content: Optional[str] = None


CODE_FILE_EXTENSIONS = {
    '.py', '.pyw', '.pyi', '.c', '.h', '.cpp', '.cc', '.cxx', '.hpp', '.hh', '.hxx',
    '.inl', '.ipp', '.cs', '.java', '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx',
    '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.kts', '.m', '.mm', '.scala',
    '.lua', '.pl', '.pm', '.r', '.jl', '.dart', '.vue', '.svelte', '.sql', '.sh',
    '.ps1', '.psm1', '.bat', '.cmd', '.vb', '.asm', '.s', '.f90', '.f', '.for',
    '.hs', '.clj', '.ex', '.exs', '.erl', '.hrl', '.ml', '.mli', '.nim', '.zig',
    '.groovy', '.gradle', '.toml', '.json', '.yml', '.yaml', '.xml', '.html', '.css', '.scss'
}

SKIP_SCAN_DIRS = {
    'node_modules', '.git', '.svn', '.hg', '__pycache__', '.venv', 'venv', 'env',
    'build', 'dist', 'out', 'target', 'obj', '.idea', '.vs', '.vscode', 'packages',
    'debug', 'release', 'x64', 'x86', 'bin', '.pytest_cache', '.mypy_cache', 'coverage',
    'site-packages', '.next', '.nuxt', 'vendor'
}

SYMBOL_TYPE_WORDS = {
    'int', 'void', 'char', 'bool', 'boolean', 'float', 'double', 'long', 'short',
    'unsigned', 'signed', 'byte', 'string', 'String', 'str', 'var', 'auto', 'dynamic',
    'object', 'any', 'number', 'uint', 'ulong', 'usize', 'isize', 'size_t', 'ssize_t',
    'int8_t', 'int16_t', 'int32_t', 'int64_t', 'uint8_t', 'uint16_t', 'uint32_t',
    'uint64_t', 'float32', 'float64', 'Decimal', 'int64', 'List', 'Dict', 'Set',
    'Map', 'Array', 'Optional', 'Promise', 'Future', 'HashMap', 'ArrayList', 'Task',
    'void*', 'byte[]', 'char*', 'wchar_t', 'BOOLEAN', 'DWORD', 'UINT', 'INT', 'LPCSTR'
}

SYMBOL_DECL_WORDS = {
    'class', 'struct', 'interface', 'enum', 'union', 'namespace', 'trait', 'record',
    'type', 'module', 'object', 'protocol', 'extension', 'actor', 'package'
}

SYMBOL_FUNC_WORDS = {
    'def', 'function', 'func', 'fn', 'fun', 'sub', 'procedure', 'async'
}

SYMBOL_MODIFIER_WORDS = {
    'const', 'let', 'var', 'val', 'final', 'static', 'extern', 'inline', 'virtual',
    'public', 'private', 'protected', 'internal', 'readonly', 'mutable', 'abstract',
    'override', 'export', 'default', 'register', 'volatile', 'mutable', 'decltype',
    'typename', 'using', 'typedef', 'new', 'sealed', 'partial', 'unsafe', 'fixed',
    'synchronized', 'native', 'transient', 'strictfp', 'lazy', 'weak', 'unowned'
}

SYMBOL_CONTROL_WORDS = {
    'if', 'else', 'elif', 'while', 'for', 'foreach', 'switch', 'case', 'catch',
    'return', 'throw', 'delete', 'new', 'sizeof', 'typeof', 'await', 'yield',
    'print', 'echo', 'cout', 'cin', 'printf', 'assert', 'using', 'lock', 'try',
    'with', 'do', 'goto', 'break', 'continue', 'raise', 'pass', 'in', 'of', 'is',
    'and', 'or', 'not', 'when', 'match', 'unless', 'until', 'loop'
}

_SYMBOL_ASSIGN_BEFORE_RE = re.compile(r'(=|==|!=|<=|>=|&&|\|\||<<|>>|\+|-|\*|/|%|&|\||\^|!|~|:|;|,|\(|\[|\{)\s*$')
_SYMBOL_PREV_WORD_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*$')


def _read_code_text(path: str, max_bytes: int = 2 * 1024 * 1024):
    try:
        if os.path.getsize(path) > max_bytes:
            return None
    except OSError:
        return None
    for enc in ('utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def _classify_symbol_occurrence(line: str, start: int, end: int, symbol: str, ext: str) -> str:
    """根据上下文把一处符号出现判定为 definition / declaration / reference"""
    before = line[:start]
    after = line[end:]
    before_trim = before.rstrip()
    after_trim = after.lstrip()
    stripped = line.strip()

    prev_word = ''
    pm = _SYMBOL_PREV_WORD_RE.search(before_trim)
    if pm:
        prev_word = pm.group(1)

    line_ends_semicolon = stripped.endswith(';')
    line_has_brace = '{' in after

    if stripped.startswith('#'):
        if re.match(r'#\s*define\s+' + re.escape(symbol) + r'\b', stripped):
            return 'definition'
        if re.match(r'#\s*(?:include|import|pragma|ifdef|ifndef|undef|error|line)', stripped):
            return 'reference'

    if ext in ('.py', '.pyw', '.pyi'):
        if re.match(r'(?:async\s+)?def\s+' + re.escape(symbol) + r'\b', stripped):
            return 'definition'
        if re.match(r'class\s+' + re.escape(symbol) + r'\b', stripped):
            return 'definition'
        if re.match(re.escape(symbol) + r'\s*(?::[^=]*)?=', stripped) and not before.strip():
            return 'definition'
        if re.match(re.escape(symbol) + r'\s*:', stripped) and not before.strip():
            return 'declaration'
        if prev_word in SYMBOL_DECL_WORDS or prev_word in SYMBOL_FUNC_WORDS:
            return 'definition'
        if _SYMBOL_ASSIGN_BEFORE_RE.search(before_trim) and not before_trim.endswith(':'):
            return 'reference'
        return 'reference'

    if prev_word in SYMBOL_DECL_WORDS:
        if line_ends_semicolon and not line_has_brace and after_trim in ('', ';'):
            return 'declaration'
        return 'definition'

    if prev_word in SYMBOL_FUNC_WORDS:
        return 'definition'

    if '{' in before:
        return 'reference'

    if _SYMBOL_ASSIGN_BEFORE_RE.search(before_trim):
        return 'reference'

    if after_trim.startswith('(') or after_trim.startswith('<('):
        if line_ends_semicolon and not line_has_brace:
            if prev_word and prev_word not in SYMBOL_CONTROL_WORDS:
                return 'declaration'
            return 'reference'
        if prev_word in SYMBOL_CONTROL_WORDS:
            return 'reference'
        if prev_word in SYMBOL_TYPE_WORDS or prev_word in SYMBOL_MODIFIER_WORDS:
            return 'definition'
        if line_has_brace or stripped.endswith(':'):
            return 'definition'
        if prev_word:
            return 'definition'
        return 'reference'

    if prev_word in SYMBOL_TYPE_WORDS or prev_word in SYMBOL_MODIFIER_WORDS:
        if line_ends_semicolon and not after_trim.startswith('=') and not line_has_brace:
            return 'declaration'
        return 'definition'

    if after_trim.startswith(':='):
        return 'definition'

    if after_trim.startswith('=') and not after_trim.startswith('==') and not after_trim.startswith('=>'):
        if before_trim.endswith('.') or before_trim.endswith('->') or before_trim.endswith('::'):
            return 'reference'
        if ext in ('.py', '.pyw', '.pyi'):
            return 'definition'
        if before_trim:
            return 'definition'
        return 'reference'

    if after_trim.startswith(':') and not after_trim.startswith('::'):
        if ext in ('.py', '.pyw', '.pyi') and not before_trim:
            return 'definition'
        return 'reference'

    if before_trim and after_trim in ('', ';') and line_ends_semicolon:
        if prev_word and prev_word not in SYMBOL_CONTROL_WORDS:
            return 'declaration'
        return 'reference'

    if before_trim and after_trim in ('', ',', ')'):
        if prev_word and prev_word not in SYMBOL_CONTROL_WORDS:
            return 'declaration'

    return 'reference'


@app.post("/find-symbol")
async def find_symbol(request: FindSymbolRequest):
    """在项目内查找符号的定义 / 声明 / 所有引用（语言无关的通用实现）"""
    symbol = (request.symbol or '').strip()
    if not symbol:
        return {"success": False, "error": "符号名为空"}

    root = request.project_root or ''
    if not root or not os.path.isdir(root):
        return {"success": False, "error": f"项目路径无效: {root}"}

    mode = (request.mode or 'references').lower()
    try:
        word_re = re.compile(r'(?<![A-Za-z0-9_$])' + re.escape(symbol) + r'(?![A-Za-z0-9_$])')

        current_abs = ''
        if request.file_path:
            fp = request.file_path.replace('/', os.sep).replace('\\', os.sep)
            if os.path.isabs(fp):
                current_abs = os.path.normcase(os.path.abspath(fp))
            else:
                current_abs = os.path.normcase(os.path.abspath(os.path.join(root, fp)))

        max_files = 3000
        max_total_bytes = 30 * 1024 * 1024
        scanned = 0
        total_bytes = 0
        truncated = False
        results = []

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith('.') and d.lower() not in SKIP_SCAN_DIRS
            ]
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext not in CODE_FILE_EXTENSIONS:
                    continue
                if scanned >= max_files or total_bytes >= max_total_bytes:
                    truncated = True
                    break
                full_path = os.path.join(dirpath, name)
                norm_path = os.path.normcase(os.path.abspath(full_path))

                content = None
                if current_abs and norm_path == current_abs and request.current_content is not None:
                    content = request.current_content
                else:
                    content = _read_code_text(full_path)
                if content is None:
                    continue

                scanned += 1
                total_bytes += len(content)

                if symbol not in content:
                    continue

                rel_path = os.path.relpath(full_path, root).replace('\\', '/')
                lines = content.split('\n')
                for idx, line in enumerate(lines):
                    for m in word_re.finditer(line):
                        stripped_line = line.strip()
                        if not stripped_line:
                            continue
                        if ext in ('.py', '.pyw', '.pyi'):
                            if stripped_line.startswith('#'):
                                continue
                        elif stripped_line.startswith('//') or stripped_line.startswith('/*') or stripped_line.startswith('*'):
                            continue

                        kind = _classify_symbol_occurrence(line, m.start(), m.end(), symbol, ext)
                        results.append({
                            'file': full_path.replace('\\', '/'),
                            'rel_path': rel_path,
                            'name': name,
                            'line': idx + 1,
                            'column': m.start() + 1,
                            'text': line.strip(),
                            'kind': kind,
                        })
            if truncated:
                break

        if mode in ('definition', 'declaration'):
            defs = [r for r in results if r['kind'] == 'definition']
            decls = [r for r in results if r['kind'] == 'declaration']
            if mode == 'definition':
                results = defs or decls or results
            else:
                results = decls or defs or results

        results.sort(key=lambda r: (r['rel_path'], r['line'], r['column']))
        if len(results) > 2000:
            results = results[:2000]
            truncated = True

        logger.info(f"[FindSymbol] {symbol} mode={mode} 命中 {len(results)} 处，扫描 {scanned} 个文件")
        return {
            "success": True,
            "symbol": symbol,
            "mode": mode,
            "results": results,
            "files_scanned": scanned,
            "truncated": truncated,
        }
    except Exception as e:
        logger.error(f"[FindSymbol] 查找失败: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


class RunPythonRequest(BaseModel):
    code: str
    filename: str = 'script.py'
    pythonCommand: str = 'python'
    venvPath: str = ''
    projectRoot: str = ''
    requirementsPath: str = ''

class LanguageRunRequest(BaseModel):
    language: str
    code: str
    filename: str = ''
    compilerPath: str = ''
    projectRoot: str = ''
    configuration: str = ''
    platform: str = ''


@app.post("/run-python")
async def run_python(req: RunPythonRequest):
    """在服务器上运行 Python 代码（支持交互式 input()）"""
    import subprocess
    import os
    import threading
    import time
    global _running_procs, _proc_lock
    if _proc_lock is None:
        _proc_lock = threading.Lock()

    logger.info(f"收到运行请求: {req}")
    if not req.code:
        return {'success': False, 'error': '代码为空'}

    try:
        # 确保 cwd 是绝对路径
        if req.projectRoot:
            cwd = os.path.abspath(req.projectRoot)
            # 如果 projectRoot 目录不存在，使用文件所在的目录
            if not os.path.isdir(cwd):
                cwd = os.path.dirname(os.path.abspath(os.path.join(req.projectRoot, req.filename)))
        else:
            cwd = os.getcwd()
        
        # temp_file 使用绝对路径
        temp_file = os.path.join(cwd, req.filename)

        if req.venvPath:
            if os.name == 'nt':
                python_exe = os.path.join(req.venvPath, 'Scripts', 'python.exe')
            else:
                python_exe = os.path.join(req.venvPath, 'bin', 'python')
            if not os.path.exists(python_exe):
                python_exe = req.pythonCommand
        else:
            python_exe = req.pythonCommand

        logger.info(f"正在执行: {python_exe} {temp_file}")
        logger.info(f"工作目录: {cwd}")

        p = subprocess.Popen(
            [python_exe, '-u', temp_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )

        info = {
            'pid': p.pid,
            'proc': p,
            'stdout_buf': '',
            'stderr_buf': '',
            'buf_lock': threading.Lock(),
            'ended': False,
            'exitCode': None,
            'start_time': time.time(),
        }

        def _reader(stream, buf_key):
            """后台线程：逐字节读取管道，正确处理 UTF-8 多字节字符"""
            try:
                utf8_buffer = bytearray()
                while True:
                    byte = stream.read(1)
                    if not byte:
                        break
                    
                    utf8_buffer.append(byte[0])
                    
                    # 检查是否构成完整的 UTF-8 字符
                    try:
                        text = utf8_buffer.decode('utf-8')
                        with info['buf_lock']:
                            info[buf_key] += text
                        utf8_buffer.clear()
                    except UnicodeDecodeError:
                        # UTF-8 字符不完整，继续读取下一个字节
                        if len(utf8_buffer) >= 4:
                            # 超过最大 UTF-8 字节数，强制解码（避免卡死）
                            text = utf8_buffer.decode('utf-8', errors='replace')
                            with info['buf_lock']:
                                info[buf_key] += text
                            utf8_buffer.clear()
            except Exception as _e:
                logger.warning(f"管道读取线程异常: {_e}")
            finally:
                try: stream.close()
                except: pass

        threading.Thread(target=_reader, args=(p.stdout, 'stdout_buf'), daemon=True).start()
        threading.Thread(target=_reader, args=(p.stderr, 'stderr_buf'), daemon=True).start()

        def _waiter():
            """等待进程结束，更新状态"""
            try:
                ret = p.wait()
                info['exitCode'] = ret
            # except subprocess.TimeoutExpired:
            #     p.kill()
            #     info['exitCode'] = -1
            #     with info['buf_lock']:
            #         info['stderr_buf'] += '\n\n[执行超时（10分钟），进程已被终止]\n'
            except Exception as _e:
                info['exitCode'] = -1
                logger.warning(f"等待进程异常: {_e}")
            finally:
                info['ended'] = True
                try:
                    if p.stdin: p.stdin.close()
                except: pass

        threading.Thread(target=_waiter, daemon=True).start()

        with _proc_lock:
            _running_procs[p.pid] = info

        time.sleep(0.3)
        with info['buf_lock']:
            init_stdout = info['stdout_buf']
            init_stderr = info['stderr_buf']
            info['stdout_buf'] = ''
            info['stderr_buf'] = ''

        return {
            'success': True,
            'pid': p.pid,
            'stdout': init_stdout,
            'stderr': init_stderr,
            'exitCode': None,
            'runCommand': _format_command([python_exe, '-u', temp_file]),
            'message': '程序已启动'
        }

    except Exception as e:
        logger.error(f"执行失败: {e}")
        return {'success': False, 'error': str(e)}


def _spawn_tracked_process(cmd, cwd, env=None, merge_stderr=False):
    """启动受管进程并复用统一的进程管理框架（_running_procs）"""
    import subprocess
    global _running_procs, _proc_lock
    if _proc_lock is None:
        _proc_lock = threading.Lock()

    p = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        cwd=cwd,
        env=env or os.environ.copy()
    )

    info = {
        'pid': p.pid,
        'proc': p,
        'stdout_buf': '',
        'stderr_buf': '',
        'buf_lock': threading.Lock(),
        'ended': False,
        'exitCode': None,
        'start_time': time.time(),
    }

    def _reader(stream, buf_key):
        try:
            utf8_buffer = bytearray()
            while True:
                byte = stream.read(1)
                if not byte:
                    break
                utf8_buffer.append(byte[0])
                try:
                    text = utf8_buffer.decode('utf-8')
                    with info['buf_lock']:
                        info[buf_key] += text
                    utf8_buffer.clear()
                except UnicodeDecodeError:
                    if len(utf8_buffer) >= 4:
                        text = utf8_buffer.decode('utf-8', errors='replace')
                        with info['buf_lock']:
                            info[buf_key] += text
                        utf8_buffer.clear()
        except Exception as _e:
            logger.warning(f"管道读取线程异常: {_e}")
        finally:
            try: stream.close()
            except: pass

    threading.Thread(target=_reader, args=(p.stdout, 'stdout_buf'), daemon=True).start()
    if p.stderr:
        threading.Thread(target=_reader, args=(p.stderr, 'stderr_buf'), daemon=True).start()

    def _waiter():
        try:
            info['exitCode'] = p.wait()
        except Exception as _e:
            info['exitCode'] = -1
            logger.warning(f"等待进程异常: {_e}")
        finally:
            info['ended'] = True
            try:
                if p.stdin: p.stdin.close()
            except: pass

    threading.Thread(target=_waiter, daemon=True).start()

    with _proc_lock:
        _running_procs[p.pid] = info

    return info


_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[()][A-Z0-9]|\x1b[=>]')


def _clean_pty_output(text):
    """清理 PTY 输出中的 ANSI 转义序列与 winpty 内部占位标记"""
    if not text:
        return ''
    text = _ANSI_ESCAPE_RE.sub('', text)
    text = text.replace('0011Ignore', '')
    text = text.replace('\x00', '')
    text = text.replace('\r\n', '\n')
    # 进度条刷新用的孤立 \r 直接丢弃，避免产生大量覆盖行
    text = text.replace('\r', '')
    return text


def _focus_java_gui_windows(pid, info):
    """把 Java 进程新出现的 GUI 窗口（JOptionPane 等）提到前台

    子进程不是系统前台进程，Windows 的前台锁会让它创建的窗口
    只出现在主程序窗口后面，用户看不到弹窗，以为程序卡住或没执行。
    这里检测到新窗口就主动置前，保证弹窗可见。
    """
    if os.name != 'nt':
        return
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return

    user32 = ctypes.windll.user32
    seen = set()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _on_window(hwnd, lparam):
        wpid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
        if wpid.value != pid or not user32.IsWindowVisible(hwnd):
            return True
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, 256)
        name = cls.value
        # 只处理 AWT/Swing 窗口，跳过 winpty 的隐藏控制台窗口
        if not name.startswith(('SunAwt', 'JDialog', 'javax.swing')):
            return True
        if hwnd in seen:
            return True
        seen.add(hwnd)
        try:
            user32.AllowSetForegroundWindow(pid)
            user32.ShowWindow(hwnd, 5)
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        except Exception:
            pass
        return True

    while not info.get('ended'):
        time.sleep(0.3)
        try:
            user32.EnumWindows(_on_window, 0)
        except Exception:
            break


def _spawn_pty_process(cmd, cwd, env=None, focus_gui=False):
    """在伪终端(PTY)中启动进程，让 System.console() / isatty() 等特性可用

    普通管道(stdin/stdout=PIPE)下 Java 判定自己不在终端中，
    System.console() 会返回 null，导致密码输入等能力缺失。
    PTY 让子进程看到一个真实的控制台设备，行为与本地终端一致。
    """
    global _running_procs, _proc_lock
    if _proc_lock is None:
        _proc_lock = threading.Lock()

    try:
        from winpty import PtyProcess
    except ImportError:
        raise RuntimeError('未安装 pywinpty，无法创建伪终端')

    p = PtyProcess.spawn(list(cmd), cwd=cwd, env=env or os.environ.copy(), dimensions=(40, 120))

    info = {
        'pid': p.pid,
        'proc': p,
        'pty': p,
        'is_pty': True,
        'stdout_buf': '',
        'stderr_buf': '',
        'buf_lock': threading.Lock(),
        'ended': False,
        'exitCode': None,
        'start_time': time.time(),
        'echo_pending': '',
    }

    def _strip_echo(text):
        """吞掉终端回显的输入内容，避免与前端 ❯ 提示重复"""
        pending = info.get('echo_pending') or ''
        if not pending:
            return text
        i = 0
        while i < len(text) and i < len(pending) and text[i] == pending[i]:
            i += 1
        if i == 0:
            # 终端未回显输入内容（密码模式），仅回显了一个换行，吞掉它
            info['echo_pending'] = ''
            return text[1:] if text.startswith('\n') else text
        text = text[i:]
        pending = pending[i:]
        if not pending:
            if text.startswith('\n'):
                text = text[1:]
        info['echo_pending'] = pending
        return text

    def _reader():
        try:
            while True:
                try:
                    chunk = p.read()
                except EOFError:
                    break
                except Exception:
                    break
                if not chunk:
                    continue
                text = _clean_pty_output(chunk)
                if text:
                    with info['buf_lock']:
                        text = _strip_echo(text)
                        if text:
                            info['stdout_buf'] += text
        except Exception as _e:
            logger.warning(f"PTY 读取线程异常: {_e}")

    def _waiter():
        try:
            while p.isalive():
                time.sleep(0.1)
            info['exitCode'] = p.exitstatus
        except Exception as _e:
            info['exitCode'] = -1
            logger.warning(f"PTY 等待进程异常: {_e}")
        finally:
            # 等待读取线程把终端残余输出读完，避免前端收到 ended 时丢失结尾内容
            reader_thread.join(timeout=0.5)
            info['ended_at'] = time.time()
            info['ended'] = True

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    threading.Thread(target=_waiter, daemon=True).start()

    if focus_gui:
        threading.Thread(target=_focus_java_gui_windows, args=(p.pid, info), daemon=True).start()

    with _proc_lock:
        _running_procs[p.pid] = info

    return info


# 各语言的候选编译器/工具链
_COMPILER_CANDIDATES = {
    'c': ['gcc', 'clang', 'cc'],
    'cpp': ['g++', 'clang++', 'c++'],
    'java': ['javac'],
    'csharp': ['csc'],
    'go': ['go'],
}

_COMPILER_VERSION_ARGS = {
    'gcc': ['--version'],
    'clang': ['--version'],
    'cc': ['--version'],
    'g++': ['--version'],
    'clang++': ['--version'],
    'c++': ['--version'],
    'javac': ['-version'],
    'csc': ['/help'],
    'dotnet': ['--version'],
    'go': ['version'],
}

# Windows 下常见的安装目录（补充 PATH 之外的检测）
_WINDOWS_COMPILER_DIRS = {
    'c': [
        r'C:\mingw64\bin', r'C:\mingw32\bin', r'C:\MinGW\bin',
        r'C:\msys64\mingw64\bin', r'C:\msys64\ucrt64\bin',
        r'C:\TDM-GCC-64\bin', r'C:\Program Files\mingw-w64\mingw64\bin',
    ],
    'cpp': [
        r'C:\mingw64\bin', r'C:\mingw32\bin', r'C:\MinGW\bin',
        r'C:\msys64\mingw64\bin', r'C:\msys64\ucrt64\bin',
        r'C:\TDM-GCC-64\bin', r'C:\Program Files\mingw-w64\mingw64\bin',
    ],
    'java': [
        r'C:\Program Files\Java', r'C:\Program Files\Eclipse Adoptium',
        r'C:\Program Files\Microsoft\jdk',
    ],
    'csharp': [
        r'C:\Windows\Microsoft.NET\Framework64\v4.0.30319',
        r'C:\Windows\Microsoft.NET\Framework\v4.0.30319',
        r'C:\Program Files\dotnet',
    ],
    'go': [
        r'C:\Program Files\Go\bin', r'C:\Go\bin',
    ],
}


def _find_vcvars_for_cl(cl_path, platform=''):
    """根据 cl.exe 路径反推同属 Visual Studio 安装目录下的 vcvars 脚本

    platform 为 Win32 时使用 32 位环境（vcvars32.bat），否则使用 64 位环境。
    """
    bat_name = 'vcvars32.bat' if str(platform).lower() in ('win32', 'x86') else 'vcvars64.bat'
    d = os.path.dirname(os.path.abspath(cl_path))
    for _ in range(8):
        candidate = os.path.join(d, 'Auxiliary', 'Build', bat_name)
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _find_vs_install_dirs():
    """查找 Visual Studio 安装根目录列表"""
    if os.name != 'nt':
        return []

    vs_dirs = []
    vswhere = r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if os.path.exists(vswhere):
        try:
            import subprocess
            out = subprocess.run(
                [vswhere, '-latest', '-products', '*', '-property', 'installationPath'],
                capture_output=True, text=True, timeout=15,
                encoding='utf-8', errors='ignore'
            ).stdout.strip()
            if out:
                vs_dirs.append(out)
        except Exception:
            pass

    if not vs_dirs:
        for base in (r'C:\Program Files\Microsoft Visual Studio',
                     r'C:\Program Files (x86)\Microsoft Visual Studio'):
            if not os.path.isdir(base):
                continue
            for year in os.listdir(base):
                ydir = os.path.join(base, year)
                if not os.path.isdir(ydir):
                    continue
                for edition in os.listdir(ydir):
                    edir = os.path.join(ydir, edition)
                    if os.path.isdir(os.path.join(edir, 'VC', 'Tools', 'MSVC')):
                        vs_dirs.append(edir)
    return vs_dirs


def _find_msbuild_for_cl(cl_path):
    """根据 cl.exe 路径定位同一 Visual Studio 安装下的 MSBuild.exe"""
    if os.name != 'nt':
        return ''
    d = os.path.dirname(os.path.abspath(cl_path))
    for _ in range(8):
        candidate = os.path.join(d, 'MSBuild', 'Current', 'Bin', 'MSBuild.exe')
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent

    for vs in _find_vs_install_dirs():
        candidate = os.path.join(vs, 'MSBuild', 'Current', 'Bin', 'MSBuild.exe')
        if os.path.exists(candidate):
            return candidate
    return ''


def _find_project_file(cwd):
    """在项目根目录查找 Visual Studio 工程文件，返回 (类型, 路径)"""
    try:
        entries = os.listdir(cwd)
    except OSError:
        return None, ''
    for name in entries:
        if name.lower().endswith('.vcxproj'):
            return 'vcxproj', os.path.join(cwd, name)
    for name in entries:
        if name.lower().endswith('.sln'):
            return 'sln', os.path.join(cwd, name)
    return None, ''


_BUILD_DIR_NAMES = {
    'debug', 'release', 'x64', 'x86', 'obj', 'bin', 'build', 'out', 'ipch',
    '.vs', '.git', 'node_modules', 'test', 'tests', 'unittest',
    'jsontestrunner', 'test_lib_json',
}


def _skip_build_dir(name):
    """判断目录是否属于构建产物或测试目录，遍历工程源码时需要跳过"""
    low = name.lower()
    return low in _BUILD_DIR_NAMES or low.endswith('.tlog') or low.endswith('.dir')


def _normalize_platform(plat):
    """归一化工程平台名称，x86 与 Win32 视为同一目标平台"""
    p = str(plat).strip().lower()
    if p in ('win32', 'x86'):
        return 'win32'
    if p in ('x64', 'x86_64', 'amd64'):
        return 'x64'
    return p


def _parse_project_configs(proj_path):
    """解析 vcxproj 中声明的所有 配置|平台 组合"""
    try:
        with open(proj_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except OSError:
        return []
    pairs = re.findall(r'<ProjectConfiguration\s+Include="([^"|]+)\|([^"]+)"', content)
    configs = []
    for conf, plat in pairs:
        item = (conf.strip(), plat.strip())
        if item not in configs:
            configs.append(item)
    return configs


def _find_output_exe(cwd, stem, since=0):
    """在工程目录中查找构建产出的可执行文件（取最新的一个，可限定修改时间下限）"""
    target = (stem + '.exe').lower()
    best, best_mtime = '', -1
    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if d.lower() not in ('ipch', '.vs', '.git', 'node_modules')]
        for name in files:
            if name.lower() != target:
                continue
            path = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime < since:
                continue
            if mtime > best_mtime:
                best, best_mtime = path, mtime
    return best


def _run_msbuild(msbuild, proj_path, conf, plat, cwd, timeout=300):
    """调用 MSBuild 构建 Visual Studio 工程，返回 (returncode, 输出, 开始时间)"""
    import subprocess
    import time

    started = time.time()
    cmd = [msbuild, proj_path, '/nologo', '/v:minimal',
           '/p:Configuration=' + conf, '/p:Platform=' + plat]
    try:
        r = subprocess.run(
            cmd, capture_output=True, cwd=cwd, timeout=timeout,
            **_no_window_kwargs()
        )
    except subprocess.TimeoutExpired:
        return -1, f'工程构建超时（超过 {timeout} 秒）', started
    except Exception as e:
        return -1, f'工程构建失败: {e}', started
    return r.returncode, _decode_console_bytes((r.stdout or b'') + (r.stderr or b'')), started


def _strip_cpp_comments(text):
    """去掉 C/C++ 源码中的注释，避免注释里的 main() 等字样干扰判断"""
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text = re.sub(r'//[^\n]*', ' ', text)
    return text


def _has_main_function(content):
    """判断源文件是否定义了 main 函数（忽略注释中的同名文字）"""
    code = _strip_cpp_comments(content)
    return bool(re.search(r'(?:^|[;}\s])(?:int|void|auto)\s+main\s*\(', code, re.M))


def _collect_extra_sources(cwd, source_path, lang):
    """收集工程内其它可参与编译的源文件，排除构建产物、测试目录和自带 main 的入口文件"""
    target_abs = os.path.abspath(source_path)
    exts = ('.cpp', '.cc', '.cxx') if lang == 'cpp' else ('.c',)
    found = []
    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if not _skip_build_dir(d)]
        for name in files:
            if not name.lower().endswith(exts):
                continue
            path = os.path.join(root, name)
            if os.path.abspath(path) == target_abs:
                continue
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except OSError:
                continue
            if _has_main_function(content):
                continue
            found.append(path)
    return sorted(found)


def _collect_include_dirs(cwd, limit=20):
    """收集工程内所有包含头文件的目录及其父目录，作为自动附加的包含路径"""
    dirs = []
    for root, subs, files in os.walk(cwd):
        subs[:] = [d for d in subs if not _skip_build_dir(d)]
        if any(f.lower().endswith(('.h', '.hpp', '.hh', '.hxx')) for f in files):
            # 父目录用于支持 <json/reader.h> 这类带目录前缀的包含写法
            for candidate in (root, os.path.dirname(root)):
                if candidate and os.path.abspath(candidate) != os.path.abspath(cwd):
                    if candidate not in dirs:
                        dirs.append(candidate)
            if len(dirs) >= limit:
                break
    return dirs


def _find_msvc_cl():
    """查找 Visual Studio 自带的 MSVC 编译器 cl.exe，返回路径列表（新版本优先）"""
    vs_dirs = _find_vs_install_dirs()

    results = []
    for vs in vs_dirs:
        msvc_root = os.path.join(vs, 'VC', 'Tools', 'MSVC')
        if not os.path.isdir(msvc_root):
            continue
        try:
            versions = sorted(os.listdir(msvc_root), reverse=True)
        except OSError:
            continue
        for ver in versions:
            cl = os.path.join(msvc_root, ver, 'bin', 'Hostx64', 'x64', 'cl.exe')
            if os.path.exists(cl):
                results.append(cl)
                break
    return results


def _decode_console_bytes(raw):
    """按控制台代码页解码命令输出（中文 Windows 下为 GBK），失败时回退 UTF-8"""
    import locale
    preferred = locale.getpreferredencoding(False) or 'utf-8'
    for enc in (preferred, 'gbk', 'utf-8'):
        try:
            return raw.decode(enc).strip()
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode('utf-8', errors='ignore').strip()


def _no_window_kwargs():
    """Windows 下隐藏子进程控制台窗口，减少安全软件对构建链的启发式拦截"""
    import subprocess
    if os.name != 'nt':
        return {}
    return {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)}


def _run_vcvars_batch(vcvars, command, cwd=None, timeout=120):
    """在 vcvars 环境初始化后的 cmd 中执行命令，返回 (returncode, 输出)"""
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp(prefix='multagent_msvc_')
    bat = os.path.join(tmp, 'run.bat')
    content = (
        '@echo off\r\n'
        'call "{vcvars}" >nul 2>&1\r\n'
        '{cmd}\r\n'
        'exit /b %ERRORLEVEL%\r\n'
    ).format(vcvars=vcvars, cmd=command)
    # cmd 按 ANSI 读取批处理文件，用 GBK 写入以兼容含中文的路径
    with open(bat, 'w', encoding='gbk', errors='ignore') as f:
        f.write(content)

    try:
        r = subprocess.run(
            [bat], capture_output=True, cwd=cwd or tmp, timeout=timeout,
            **_no_window_kwargs()
        )
    except subprocess.TimeoutExpired:
        return -1, f'执行超时（超过 {timeout} 秒）'
    except Exception as e:
        return -1, f'执行失败: {e}'

    raw = (r.stdout or b'') + (r.stderr or b'')
    return r.returncode, _decode_console_bytes(raw)


def _msvc_version(cl_path):
    """获取 MSVC 编译器版本行（需在 vcvars 环境下运行 cl）"""
    vcvars = _find_vcvars_for_cl(cl_path)
    if not vcvars:
        return ''
    rc, out = _run_vcvars_batch(vcvars, '"{0}"'.format(cl_path), timeout=30)
    for line in out.splitlines():
        line = line.strip()
        if line and ('C/C++' in line or 'Compiler' in line or '编译器' in line):
            return line[:120]
    return ''


def _find_compiler_executable(names, lang=''):
    """在 PATH 与常见安装目录中查找可执行文件，返回 (path, base_name)"""
    import shutil
    for name in names:
        found = shutil.which(name)
        if found:
            return found, name

    if os.name == 'nt':
        # Go / Java 工具链目录由配置自动探测，优先于常见安装目录
        search_dirs = _WINDOWS_COMPILER_DIRS.get(lang, [])
        if lang == 'go':
            search_dirs = GoEnvConfig.path_entries() + list(search_dirs)
        elif lang == 'java':
            search_dirs = JavaEnvConfig.path_entries() + list(search_dirs)
        exts = ['.exe', '.cmd', '.bat']
        for name in names:
            for base_dir in search_dirs:
                if not os.path.isdir(base_dir):
                    continue
                for ext in exts:
                    candidate = os.path.join(base_dir, name + ext)
                    if os.path.exists(candidate):
                        return candidate, name
                # JDK 安装目录可能是子目录（如 jdk-17\bin）
                try:
                    for sub in os.listdir(base_dir):
                        sub_bin = os.path.join(base_dir, sub, 'bin')
                        for ext in exts:
                            candidate = os.path.join(sub_bin, name + ext)
                            if os.path.exists(candidate):
                                return candidate, name
                except OSError:
                    pass
    return None, None


def _compiler_version(path):
    """获取编译器版本信息（首行）"""
    import subprocess
    try:
        result = subprocess.run(
            [path] + _COMPILER_VERSION_ARGS.get(os.path.basename(path).split('.')[0], ['--version']),
            capture_output=True, text=True, timeout=10,
            encoding='utf-8', errors='ignore'
        )
        text = (result.stdout or '') + (result.stderr or '')
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:120]
    except Exception:
        pass
    return ''


@app.get("/detect-compiler")
async def detect_compiler(language: str = ''):
    """检测系统中指定语言的编译器/工具链"""
    lang = (language or '').lower()
    if lang not in _COMPILER_CANDIDATES:
        return {'success': False, 'error': f'暂不支持的语言: {language}', 'compilers': []}

    compilers = []
    seen = set()

    for name in _COMPILER_CANDIDATES[lang]:
        path, base = _find_compiler_executable([name], lang)
        if not path or path in seen:
            continue
        seen.add(path)
        version = _compiler_version(path)
        compilers.append({
            'name': f"{base} {version}".strip(),
            'path': path,
            'version': version,
            'base': base,
            'is_default': len(compilers) == 0
        })

    # C/C++ 未找到 GCC 系编译器时，回退到 Visual Studio 自带的 MSVC
    if lang in ('c', 'cpp') and not compilers:
        for cl_path in _find_msvc_cl():
            if cl_path in seen:
                continue
            seen.add(cl_path)
            version = _msvc_version(cl_path)
            compilers.append({
                'name': ('MSVC ' + version).strip(),
                'path': cl_path,
                'version': version,
                'base': 'cl.exe',
                'toolchain': 'msvc',
                'is_default': len(compilers) == 0
            })

    return {'success': True, 'language': lang, 'compilers': compilers}


def _format_command(cmd):
    """把命令列表/字符串格式化成可直接复制执行的命令行文本"""
    if isinstance(cmd, str):
        return cmd
    parts = []
    for item in cmd:
        item = str(item)
        parts.append(f'"{item}"' if ' ' in item else item)
    return ' '.join(parts)


def _compile_with_msbuild(compiler, proj_type, proj_path, cwd, stem, timeout=300,
                          configuration='', platform=''):
    """用 MSBuild 构建整个 Visual Studio 工程。

    返回 (是否成功, 输出, 产物路径, 命令)；未找到 MSBuild 时返回 None。
    工程文件能正确携带依赖库、预编译头和包含路径，是 vcxproj 项目的首选构建方式。
    """
    msbuild = _find_msbuild_for_cl(compiler)
    if not msbuild:
        return None

    if proj_type == 'vcxproj':
        configs = _parse_project_configs(proj_path)
    else:
        configs = []
    if not configs:
        configs = [('Release', 'x64'), ('Debug', 'x64'),
                   ('Release', 'Win32'), ('Debug', 'Win32')]

    if configuration or platform:
        # 用户在界面指定了配置与平台，按该组合构建
        wanted = (configuration or 'Release', platform or 'x64')
        matched = [c for c in configs
                   if c[0].lower() == wanted[0].lower()
                   and _normalize_platform(c[1]) == _normalize_platform(wanted[1])]
        if matched:
            configs = matched
        elif proj_type == 'vcxproj' and configs:
            # 工程只支持其声明的 配置|平台 组合，指定组合不存在时明确报错
            available = '、'.join(f'{c[0]}|{c[1]}' for c in configs)
            return (False,
                    f'工程未提供 {wanted[0]}|{wanted[1]} 生成配置，'
                    f'可用组合: {available}',
                    '', '')
        else:
            configs = [wanted]
    else:
        # Release 构建更快且产物即最终版本，优先尝试
        configs.sort(key=lambda item: 0 if item[0].lower().startswith('release') else 1)

    proj_name = os.path.splitext(os.path.basename(proj_path))[0]
    failures = []
    for conf, plat in configs:
        cmd_text = _format_command([msbuild, proj_path,
                                    '/p:Configuration=' + conf,
                                    '/p:Platform=' + plat])
        rc, output, started = _run_msbuild(msbuild, proj_path, conf, plat, cwd, timeout=timeout)
        if rc == 0:
            # 优先取本次构建新生成的产物；工程已是最新时 MSBuild 不会重新链接，
            # 此时放宽为查找任意已有产物
            artifact = (_find_output_exe(cwd, stem, since=started)
                        or _find_output_exe(cwd, proj_name, since=started)
                        or _find_output_exe(cwd, stem)
                        or _find_output_exe(cwd, proj_name))
            if artifact:
                logger.info(f'[run-code] 工程构建成功: {cmd_text}')
                return True, _filter_charset_warnings(output), artifact, cmd_text
            failures.append(f'$ {cmd_text}\n构建成功但未找到输出的 {stem}.exe')
            continue
        logger.info(f'[run-code] 工程构建失败 ({conf}|{plat})，尝试其它配置')
        failures.append(f'$ {cmd_text}\n{_filter_charset_warnings(output)}')
        # 链接期未解析符号说明工程未把依赖项纳入编译，换配置也无法解决，直接回退
        if re.search(r'(?:LNK2001|LNK2019|LNK1120)', output):
            logger.info('[run-code] 工程缺少依赖项导致链接失败，跳过其它配置')
            break

    return False, '\n\n'.join(failures), '', ''


def _filter_charset_warnings(output):
    """过滤 C4819/C4828 字符集警告。

    这些警告来自工程内 GBK 编码的历史文件，而编译统一按 UTF-8 读取源码所产生，
    与代码正确性无关，属于噪声输出。
    """
    kept = [ln for ln in output.splitlines()
            if not re.search(r'warning\s+C4819|warning\s+C4828', ln)]
    return '\n'.join(kept).strip()


def _is_utf8_source(path):
    """判断源文件是否为 UTF-8 编码（含非 ASCII 字符时才需要 /utf-8 开关）"""
    try:
        with open(path, 'rb') as f:
            raw = f.read()
    except OSError:
        return False
    if not any(b >= 0x80 for b in raw):
        return False
    try:
        raw.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def _msvc_cl_compile(vcvars, lang, source_path, cwd, stem, extra_sources=None, include_dirs=None):
    """直接调用 cl.exe 编译（可附带工程内其它源文件），返回 (是否成功, 输出, 产物路径, 命令)"""
    extra_sources = extra_sources or []
    include_dirs = include_dirs or []

    def rel(path):
        return os.path.relpath(path, cwd).replace('/', '\\')

    out_name = stem + '.exe'
    if lang == 'c':
        flags = '/nologo /TC'
    elif lang == 'cpp':
        flags = '/nologo /TP /EHsc /std:c++17'
    else:
        return False, '', f'暂不支持编译的语言: {lang}', ''
    # 主源文件由编辑器以 UTF-8 保存，需显式告知编译器，避免中文按 GBK 解析出错
    if _is_utf8_source(source_path):
        flags += ' /utf-8'

    parts = ['cl', flags, f'/Fe:"{out_name}"']
    for directory in include_dirs:
        parts.append(f'/I"{rel(directory)}"')
    parts.append(f'"{rel(source_path)}"')
    for extra in extra_sources:
        parts.append(f'"{rel(extra)}"')
    # 使用相对路径，避免中文目录在 cmd 中编码错乱
    command = ' '.join(parts)

    logger.info(f'[run-code] 编译命令 (MSVC, cwd={cwd}): {command}')
    rc, output = _run_vcvars_batch(vcvars, command, cwd=cwd, timeout=300)
    out_path = os.path.join(cwd, out_name)
    if rc != 0 or not os.path.exists(out_path):
        return False, output, '', command

    # MSVC 编译成功时会回显源文件名，过滤掉这类无意义输出
    echoed = {os.path.basename(p) for p in [source_path] + list(extra_sources)}
    cleaned = '\n'.join(
        ln for ln in output.splitlines()
        if ln.strip() and ln.strip() not in echoed
    ).strip()
    return True, _filter_charset_warnings(cleaned), out_path, command


def _compile_or_fail(lang, compiler, source_path, cwd, configuration='', platform=''):
    """执行编译，返回 (是否成功, 编译输出, 产物路径或错误文本, 实际执行的命令)"""
    import subprocess

    stem = os.path.splitext(os.path.basename(source_path))[0]
    is_msvc = os.path.basename(compiler).lower() == 'cl.exe'

    if is_msvc:
        # MSVC 必须先在 vcvars 环境初始化后才能找到标准库头文件与链接器
        vcvars = _find_vcvars_for_cl(compiler, platform)
        if not vcvars:
            bat = 'vcvars32.bat' if str(platform).lower() in ('win32', 'x86') else 'vcvars64.bat'
            return False, '', f'未找到 {bat}，无法初始化 MSVC 编译环境', ''
        if lang not in ('c', 'cpp'):
            return False, '', f'暂不支持编译的语言: {lang}', ''

        logger.info(f'[run-code] 环境初始化: {vcvars}')

        # 项目根目录存在 vcxproj/sln 时，优先交给 MSBuild 构建整个工程
        proj_type, proj_path = _find_project_file(cwd)
        msbuild_output = ''
        if proj_type:
            logger.info(f'[run-code] 检测到工程文件: {proj_path}')
            result = _compile_with_msbuild(compiler, proj_type, proj_path, cwd, stem,
                                           configuration=configuration, platform=platform)
            if result is not None:
                ok, output, artifact, command = result
                if ok:
                    return True, output, artifact, command
                msbuild_output = output

        # 工程构建不可用时，回退为直接调用 cl 编译；同时把工程内其它源文件一起编译，
        # 以解决「源文件依赖同目录/子目录实现代码，但工程未把依赖加入编译项」导致的链接失败
        extra_sources = _collect_extra_sources(cwd, source_path, lang) if proj_type else []
        include_dirs = _collect_include_dirs(cwd) if extra_sources else []
        if extra_sources:
            logger.info(f'[run-code] 回退编译，附带 {len(extra_sources)} 个源文件、'
                        f'{len(include_dirs)} 个包含目录')

        ok, output, artifact, command = _msvc_cl_compile(
            vcvars, lang, source_path, cwd, stem, extra_sources, include_dirs
        )
        if ok:
            return True, output, artifact, command

        merged = '\n\n'.join(part for part in (msbuild_output, output) if part.strip())
        if msbuild_output:
            merged = '工程构建未成功，已回退到直接编译源文件：\n\n' + merged
        return False, merged, '编译失败，请检查上面的错误信息', command

    if lang == 'c':
        out_path = os.path.join(cwd, stem + ('.exe' if os.name == 'nt' else ''))
        cmd = [compiler, source_path, '-o', out_path]
    elif lang == 'cpp':
        out_path = os.path.join(cwd, stem + ('.exe' if os.name == 'nt' else ''))
        cmd = [compiler, '-std=c++17', source_path, '-o', out_path]
    elif lang == 'java':
        out_path = None
        # 中文 Windows 下 javac 的诊断信息按 GBK 输出，而下方统一按 UTF-8 解码，
        # 会导致报错提示乱码，因此强制 javac 使用英文输出
        cmd = [compiler, '-J-Duser.language=en', '-J-Duser.country=US',
               '-encoding', 'UTF-8', source_path]
    elif lang == 'csharp':
        out_path = os.path.join(cwd, stem + '.exe')
        cmd = [compiler, '/nologo', f'/out:{out_path}', source_path]
    else:
        return False, '', f'暂不支持编译的语言: {lang}', ''

    cmd_text = _format_command(cmd)
    logger.info(f'[run-code] 编译命令 ({lang}, cwd={cwd}): {cmd_text}')

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=cwd, timeout=120,
            encoding='utf-8', errors='ignore'
        )
    except FileNotFoundError:
        return False, '', f'找不到编译器: {compiler}', cmd_text
    except subprocess.TimeoutExpired:
        return False, '', '编译超时（超过 120 秒）', cmd_text

    output = ((result.stdout or '') + (result.stderr or '')).strip()
    if result.returncode != 0:
        return False, output, '编译失败，请检查上面的错误信息', cmd_text

    return True, output, out_path, cmd_text


def _find_java_launcher(javac_path):
    """根据 javac 路径推导同目录下的 java 启动器"""
    import shutil
    directory = os.path.dirname(javac_path)
    for name in ('java.exe', 'java') if os.name == 'nt' else ('java',):
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    return shutil.which('java') or 'java'


def _detect_java_class_name(code, fallback):
    """从 Java 源码中提取 public class 名称，作为源文件名"""
    match = re.search(r'\bpublic\s+(?:final\s+|abstract\s+)?class\s+(\w+)', code)
    if not match:
        match = re.search(r'\bclass\s+(\w+)', code)
    return match.group(1) if match else fallback


@app.post("/run-code")
def run_code(req: LanguageRunRequest):
    """编译并运行 C / C++ / Java / C# / Go 代码（复用统一进程管理框架）

    使用同步函数（非 async），由 FastAPI 放入线程池执行：
    编译/构建是秒级阻塞操作，若写成 async 会卡死 uvicorn 事件循环，
    导致整个服务（含界面请求）在构建期间完全无响应。
    """
    import shutil

    lang = (req.language or '').lower()

    logger.info(f'编译 {lang} 代码')
    if not req.code:
        return {'success': False, 'error': '代码为空'}
    if lang not in _COMPILER_CANDIDATES:
        return {'success': False, 'error': f'暂不支持的语言: {lang}'}

    default_names = {
        'c': 'main.c', 'cpp': 'main.cpp', 'java': 'Main.java',
        'csharp': 'Program.cs', 'go': 'main.go'
    }
    filename = req.filename or default_names[lang]

    # Java 的源文件名必须与 public class 名称一致
    if lang == 'java':
        stem_default = os.path.splitext(filename)[0]
        class_name = _detect_java_class_name(req.code, stem_default)
        filename = class_name + '.java'

    if req.projectRoot and os.path.isdir(os.path.abspath(req.projectRoot)):
        cwd = os.path.abspath(req.projectRoot)
    else:
        cwd = os.getcwd()

    source_path = os.path.join(cwd, filename)
    try:
        with open(source_path, 'w', encoding='utf-8') as f:
            f.write(req.code)
    except Exception as e:
        return {'success': False, 'error': f'写入源文件失败: {e}'}

    compiler = req.compilerPath
    if not compiler:
        compiler, _ = _find_compiler_executable(_COMPILER_CANDIDATES[lang], lang)
    if not compiler and lang in ('c', 'cpp'):
        # 未找到 GCC 系编译器时，回退到 Visual Studio 自带的 MSVC
        candidates = _find_msvc_cl()
        compiler = candidates[0] if candidates else ''
    if not compiler:
        return {'success': False, 'error': f'未找到 {lang} 的编译器，请在设置中选择编译器路径'}

    logger.info(f'[run-code] 开始运行 language={lang} compiler={compiler} cwd={cwd} file={filename}')

    stem = os.path.splitext(filename)[0]
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'LANG': 'zh_CN.UTF-8'}
    # 自动补齐 Go / JDK 工具链路径，避免应用启动后才安装导致命令找不到
    GoEnvConfig.apply(env)
    JavaEnvConfig.apply(env)

    try:
        if lang == 'go':
            is_test_file = filename.endswith('_test.go')
            # 目录中无 go.mod 时，go run 会因模块模式报错，改用临时目录运行避免污染项目
            if os.path.exists(os.path.join(cwd, 'go.mod')):
                run_cwd = cwd
                run_source = source_path
            else:
                import tempfile
                tmp_dir = tempfile.mkdtemp(prefix='multagent_go_')
                if is_test_file:
                    # 测试文件依赖同包其它源码，需一并复制到临时目录
                    for name in os.listdir(cwd):
                        if name.endswith('.go') and name != filename:
                            try:
                                shutil.copy2(os.path.join(cwd, name), os.path.join(tmp_dir, name))
                            except OSError:
                                pass
                run_source = os.path.join(tmp_dir, os.path.basename(source_path))
                shutil.copy2(source_path, run_source)
                with open(os.path.join(tmp_dir, 'go.mod'), 'w', encoding='utf-8') as f:
                    f.write('module sandbox\n\ngo 1.20\n')
                run_cwd = tmp_dir
            if is_test_file:
                # go run 不支持 *_test.go，必须用 go test 编译并执行测试
                run_cmd = [compiler, 'test', '-v', '-count=1', '.']
            else:
                run_cmd = [compiler, 'run', run_source]
            logger.info(f'[run-code] 运行命令 (go, cwd={run_cwd}): {_format_command(run_cmd)}')
            info = _spawn_tracked_process(run_cmd, run_cwd, env)
            compile_output = ''
            compile_command = _format_command(run_cmd)
            run_command = compile_command
        else:
            ok, compile_output, artifact, compile_command = _compile_or_fail(
                lang, compiler, source_path, cwd,
                configuration=req.configuration, platform=req.platform
            )
            if not ok:
                return {
                    'success': False, 'error': artifact, 'stdout': compile_output,
                    'compileOutput': compile_output, 'compileCommand': compile_command
                }

            if lang == 'java':
                cmd = [_find_java_launcher(compiler), '-Dfile.encoding=UTF-8', '-cp', cwd, stem]
                run_command = _format_command(cmd)
                logger.info(f'[run-code] 运行命令 (java/PTY, cwd={cwd}): {run_command}')
                # Java 走伪终端，使 System.console() 可用（密码输入、isatty 等）
                try:
                    info = _spawn_pty_process(cmd, cwd, env, focus_gui=True)
                except Exception as pty_err:
                    # PTY 不可用时降级为普通管道，功能可用但 System.console() 为 null
                    logger.warning(f'[run-code] PTY 启动失败，回退普通管道: {pty_err}')
                    info = _spawn_tracked_process(cmd, cwd, env)
            else:
                cmd = [artifact]
                run_command = _format_command(cmd)
                logger.info(f'[run-code] 运行命令 ({lang}, cwd={cwd}): {run_command}')
                info = _spawn_tracked_process(cmd, cwd, env)

        time.sleep(0.3)
        with info['buf_lock']:
            init_stdout = info['stdout_buf']
            init_stderr = info['stderr_buf']
            info['stdout_buf'] = ''
            info['stderr_buf'] = ''

        if compile_output:
            init_stdout = compile_output + ('\n' if init_stdout else '') + init_stdout

        return {
            'success': True,
            'pid': info['pid'],
            'stdout': init_stdout,
            'stderr': init_stderr,
            'exitCode': None,
            'compileCommand': compile_command,
            'runCommand': run_command,
            'message': '程序已启动'
        }

    except FileNotFoundError as e:
        return {'success': False, 'error': f'找不到可执行程序: {e}'}
    except Exception as e:
        logger.error(f"运行 {lang} 代码失败: {e}")
        return {'success': False, 'error': str(e)}


@app.post("/run-input")
async def run_input(req: dict = None):
    """向运行中的进程 stdin 写入输入（用于 input() 交互）"""
    import threading
    global _running_procs, _proc_lock
    if _proc_lock is None:
        _proc_lock = threading.Lock()

    if req is None:
        return {'success': False, 'error': '参数为空'}
    pid = req.get('pid')
    text = req.get('text', '')
    newline = req.get('newline', '\n')
    if not pid:
        return {'success': False, 'error': '缺少 pid'}

    try:
        pid = int(pid)
        with _proc_lock:
            info = _running_procs.get(pid)

        if not info:
            return {'success': False, 'error': '进程不存在或已结束'}
        if info['ended']:
            return {'success': False, 'error': '进程已结束，无法再输入'}

        proc = info['proc']

        if info.get('is_pty'):
            try:
                nl = newline or '\r\n'
                if nl == '\n':
                    nl = '\r\n'
                with info['buf_lock']:
                    info['echo_pending'] = text + '\n'
                proc.write(text + nl)
                return {'success': True}
            except EOFError:
                return {'success': False, 'error': '进程已断开（PTY 已关闭）'}
            except Exception as e:
                return {'success': False, 'error': str(e)}

        if not proc.stdin:
            return {'success': False, 'error': '进程没有打开 stdin'}

        try:
            data = (text + newline).encode('utf-8')
            proc.stdin.write(data)
            proc.stdin.flush()
            return {'success': True}
        except BrokenPipeError:
            return {'success': False, 'error': '进程已断开（BrokenPipe）'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.post("/shell-start")
async def shell_start(req: dict = None):
    """创建终端会话（记录工作目录，命令逐条在会话目录中执行）"""
    import uuid
    if req is None:
        req = {}
    project_root = (req.get('projectRoot') or '').strip()
    cwd = os.getcwd()
    if project_root:
        abs_root = os.path.abspath(project_root)
        if os.path.isdir(abs_root):
            cwd = abs_root

    session_id = uuid.uuid4().hex[:12]
    _shell_sessions[session_id] = {'cwd': cwd}
    return {'success': True, 'sessionId': session_id, 'cwd': cwd}


@app.post("/shell-exec")
async def shell_exec(req: dict = None):
    """在终端会话中执行一条命令，返回进程 pid 供前端轮询输出"""
    import base64
    if req is None:
        return {'success': False, 'error': '参数为空'}
    session_id = req.get('sessionId')
    command = req.get('command', '')
    if not session_id or session_id not in _shell_sessions:
        return {'success': False, 'error': '终端会话不存在'}
    if not command.strip():
        return {'success': True, 'skipped': True}

    cwd = _shell_sessions[session_id]['cwd']

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env.pop('VIRTUAL_ENV_PROMPT', None)
    env.pop('VIRTUAL_ENV', None)
    # 自动补齐 Go / JDK 工具链路径，避免应用启动后才安装导致命令找不到
    GoEnvConfig.apply(env)
    JavaEnvConfig.apply(env)

    if os.name == 'nt':
        cmd_b64 = base64.b64encode(command.encode('utf-8')).decode('ascii')
        script = (
            "$ProgressPreference='SilentlyContinue'\n"
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8\n"
            "$OutputEncoding=[System.Text.Encoding]::UTF8\n"
            f"$__cmd=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{cmd_b64}'))\n"
            "try { Invoke-Expression $__cmd } catch { Write-Output $_.Exception.Message }\n"
            "Write-Output \"__CWD__$($PWD.Path)\"\n"
        )
        encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
        cmd = ['powershell.exe', '-NoLogo', '-NoProfile', '-EncodedCommand', encoded]
    else:
        cmd = ['/bin/bash', '-lc', command]

    try:
        info = _spawn_tracked_process(cmd, cwd, env)
        return {'success': True, 'pid': info['pid'], 'sessionId': session_id}
    except Exception as e:
        logger.error(f"终端命令执行失败: {e}")
        return {'success': False, 'error': str(e)}


@app.post("/shell-sync-cwd")
async def shell_sync_cwd(req: dict = None):
    """命令执行结束后同步会话工作目录"""
    if req is None:
        return {'success': False, 'error': '参数为空'}
    session_id = req.get('sessionId')
    new_cwd = (req.get('cwd') or '').strip()
    if not session_id or session_id not in _shell_sessions:
        return {'success': False, 'error': '终端会话不存在'}
    if new_cwd and os.path.isdir(new_cwd):
        _shell_sessions[session_id]['cwd'] = new_cwd
    return {'success': True, 'cwd': _shell_sessions[session_id]['cwd']}


@app.post("/shell-close")
async def shell_close(req: dict = None):
    """关闭终端会话"""
    if req is None:
        return {'success': False, 'error': '参数为空'}
    session_id = req.get('sessionId')
    _shell_sessions.pop(session_id, None)
    return {'success': True}


@app.post("/run-output")
async def run_output(req: dict = None):
    import threading
    global _running_procs, _proc_lock
    if _proc_lock is None:
        _proc_lock = threading.Lock()
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

        new_stdout = ''
        new_stderr = ''
        with info['buf_lock']:
            new_stdout = info['stdout_buf']
            new_stderr = info['stderr_buf']
            info['stdout_buf'] = ''
            info['stderr_buf'] = ''

        ended = info['ended']
        exit_code = info['exitCode']

        # PTY 退出后读取线程可能还有残余输出，留出短暂的收尾时间再清理
        grace = 0.3 if info.get('is_pty') else 0.0
        ended_at = info.get('ended_at')
        settled = ended and ended_at is not None and (time.time() - ended_at) >= grace

        if settled and not new_stdout and not new_stderr:
            with _proc_lock:
                _running_procs.pop(pid, None)

        return {
            'success': True,
            'stdout': new_stdout,
            'stderr': new_stderr,
            'ended': ended,
            'exitCode': exit_code
        }
    except Exception as e:
        logger.error(f"run_output 异常: {e}")
        return {'success': False, 'error': str(e)}


@app.post("/run-stop")
async def run_stop(req: dict = None):
    import subprocess
    import threading
    global _running_procs, _proc_lock
    if _proc_lock is None:
        _proc_lock = threading.Lock()
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

        if info.get('is_pty'):
            try:
                info['pty'].terminate(force=True)
            except Exception:
                pass
            try:
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(pid)],
                    capture_output=True, timeout=5
                )
            except Exception:
                pass
        elif os.name == 'nt':
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(pid)],
                capture_output=True, timeout=5
            )
        else:
            try:
                info['proc'].kill()
            except:
                subprocess.run(['kill', '-9', str(pid)], capture_output=True, timeout=3)

        info['ended'] = True
        if info['exitCode'] is None:
            info['exitCode'] = -1

        with _proc_lock:
            _running_procs.pop(pid, None)

        return {'success': True, 'message': '已终止'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


class InstallPackageRequest(BaseModel):
    package: str
    pythonCommand: str = 'python'
    venvPath: str = ''
    projectRoot: str = ''


@app.post("/install-package")
async def install_package(req: InstallPackageRequest):
    """安装 Python 包"""
    import subprocess
    import sys

    if not req.package or not req.package.strip():
        return {'success': False, 'error': '包名不能为空'}

    try:
        # 确定 Python 解释器路径
        if req.venvPath:
            if os.name == 'nt':
                python_exe = os.path.join(req.venvPath, 'Scripts', 'python.exe')
            else:
                python_exe = os.path.join(req.venvPath, 'bin', 'python')
            if not os.path.exists(python_exe):
                python_exe = req.pythonCommand
        else:
            python_exe = req.pythonCommand

        # 运行 pip install
        cmd = [python_exe, '-m', 'pip', 'install', req.package.strip()]
        logger.info(f'执行安装命令: {cmd}')

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=req.projectRoot or os.getcwd(),
            timeout=300
        )

        return {
            'success': True,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }

    except subprocess.TimeoutExpired:
        return {'success': False, 'error': '安装超时（5分钟）'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


class SyntaxCheckRequest(BaseModel):
    filePath: str


@app.post("/check-syntax")
async def check_syntax(req: SyntaxCheckRequest):
    """检查 Python 文件的语法错误"""
    import ast
    
    if not req.filePath:
        return {'success': False, 'error': '文件路径不能为空', 'errorCount': 0}
    
    try:
        # 读取文件内容
        with open(req.filePath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 尝试解析语法
        try:
            ast.parse(code)
            return {'success': True, 'errorCount': 0, 'errors': []}
        except SyntaxError as e:
            return {
                'success': True,
                'errorCount': 1,
                'errors': [{
                    'line': e.lineno,
                    'column': e.offset,
                    'message': str(e.msg)
                }]
            }
    except FileNotFoundError:
        return {'success': False, 'error': '文件不存在', 'errorCount': 0}
    except Exception as e:
        return {'success': False, 'error': str(e), 'errorCount': 0}

class AICodeChatRequest(BaseModel):
    """AI 辅助编程助手请求模型"""
    query: str
    current_file_path: str = ""
    current_file_code: str = ""
    project_root: str = ""
    command: str = ""  # /explain, /fix, /refactor, /new
    model: str = None
    images: list = None  # 截图数据


@app.post("/api/ai-code/chat")
async def ai_code_chat(request: AICodeChatRequest):
    """AI 辅助编程助手 - 流式接口（统一版）"""
    from core.skills.AICode.project_programming import Project_Programming
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")
    
    # 获取项目文件列表
    project_files = []
    if request.project_root:
        try:
            import os
            for root, dirs, files in os.walk(request.project_root):
                # 忽略隐藏目录和常见忽略目录
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', '.git', 'venv', 'env']]
                for file in files:
                    if not file.startswith('.') and file not in ['.DS_Store']:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, request.project_root)
                        project_files.append(rel_path)
        except Exception as e:
            logger.warning(f"获取项目文件列表失败: {e}")
    
    # 创建 Project_Programming 实例
    project_context = {
        "project_name": request.project_root,
        "project_path": request.project_root,
        "files": project_files
    }
    
    programming_assistant = Project_Programming(
        file_path=request.current_file_path,
        state_list=[],
        project_context=project_context,
        images=request.images  # 传递截图数据
    )
    
    async def generate():
        async for chunk in programming_assistant.stream_ai_code_chat(
            query=request.query,
            current_file_path=request.current_file_path,
            current_file_code=request.current_file_code,
            command=request.command,
            model_name=Params.resolve_model_name(request.model),
        ):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/event-stream")


class AICodeWriteFileRequest(BaseModel):
    """AI 辅助编程助手 - 文件写入请求"""
    file_path: str
    content: str
    project_root: str = ""


@app.post("/api/ai-code/write-file")
async def ai_code_write_file(request: AICodeWriteFileRequest):
    """AI 辅助编程助手 - 写入文件（V2 工具调用）"""
    from core.skills.AICode.project_programming import Project_Programming
    
    # 创建 Project_Programming 实例
    project_context = {
        "project_name": request.project_root,
        "project_path": request.project_root,
        "files": []
    }
    
    programming_assistant = Project_Programming(
        file_path=request.file_path,
        state_list=[],
        project_context=project_context,
        images=None
    )
    
    result = programming_assistant._write_file_tool(request.file_path, request.content)
    return result


# ========= PyWebView 暴露给前端JS的函数 =========
def select_project_folder():
    """弹出系统文件夹选择框，返回真实完整本地路径"""
    logger.info("========== select_project_folder 被调用 ==========")
    try:
        result = webview.windows[0].create_file_dialog(
            webview.FileDialog.FOLDER,
        )
        logger.info(f"create_file_dialog 返回结果: {result}")
        if result and len(result) > 0:
            logger.info(f"用户选择的文件夹: {result[0]}")
            return result[0]
        logger.warning("用户取消选择或返回空结果")
        return None
    except Exception as e:
        logger.error(f"select_project_folder 执行异常: {e}", exc_info=True)
        return None

def start_fastapi():
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
    )

def wait_for_server(url, timeout=120):
    """等待本地服务端口就绪，避免窗口先加载导致白屏"""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


if __name__ == "__main__":
    APP_URL = "http://127.0.0.1:5001/products/online"

    # 崩溃与退出兜底：进程被原生崩溃或静默终止前，把最后的现场写入日志
    import atexit
    import faulthandler
    import signal

    def _log_process_exit():
        logger.info(f"进程正常退出 (pid={os.getpid()})")

    def _on_signal(signum, _frame):
        logger.warning(f"进程收到信号 {signum}，即将退出")
        raise SystemExit(0)

    atexit.register(_log_process_exit)
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _on_signal)
        except (ValueError, OSError):
            pass

    try:
        PathConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
        _fault_file = open(PathConfig.LOG_DIR / "fault.log", "a", encoding="utf-8")
        faulthandler.enable(file=_fault_file, all_threads=True)
        logger.info(f"faulthandler 已启用 (pid={os.getpid()})")
    except Exception as _e:
        logger.warning(f"faulthandler 启用失败: {_e}")

    try:
        # 启动 FastAPI 后台进程
        t = threading.Thread(target=start_fastapi, daemon=True)
        t.start()

        # 等待服务就绪后再打开窗口，确保首个界面即为聊天界面
        if not wait_for_server(APP_URL):
            logger.warning("服务启动超时，仍尝试打开窗口")

        # 创建webview窗口，加载你的前端页面地址
        window = webview.create_window(
            title=APP_TITLE,
            width=1400,
            height=900,
            url=APP_URL,
            maximized=True,  # 启动直接最大化铺满屏幕
            # fullscreen=True,  # 开启真正全屏
        )

        window.expose(select_project_folder)
        # 关闭隐私模式并指定存储目录，持久化 localStorage/Cookie（登录态、历史对话）
        webview_storage_dir = PathConfig.DATA_DIR / "webview_storage"
        webview_storage_dir.mkdir(parents=True, exist_ok=True)
        webview.start(private_mode=False, storage_path=str(webview_storage_dir))
    except BaseException as _e:
        logger.critical(f"主入口异常退出: {type(_e).__name__}: {_e}", exc_info=True)
        raise