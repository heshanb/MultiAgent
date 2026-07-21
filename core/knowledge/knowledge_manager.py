import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from core.auth.auth import get_current_user, security
from core.auth.database import get_db, User
from settings.Define import PathConfig, Params
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma
    from langchain.embeddings import DashScopeEmbeddings
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_community.embeddings import DashScopeEmbeddings
from core.skills.DocProcess.document_process import doc_processor
from settings.logger_manager import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# 全局向量存储缓存
vector_stores = {}


def get_user_knowledge_dir(username: str) -> Path:
    """获取用户知识库目录"""
    user_dir = PathConfig.COMMON_KNOWLEDGE_DIR / username
    if not user_dir.exists():
        user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_vector_store(username: str):
    """获取或创建用户的向量存储"""
    if username not in vector_stores:
        persist_dir = PathConfig.COMMON_KNOWLEDGE_DIR / username / "vector_db"
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings = DashScopeEmbeddings(model=Params.DEFAULT_EMBEDDING_MODEL)
        vector_stores[username] = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=f"knowledge_{username}"
        )
    
    return vector_stores[username]


def get_vector_count(username: str) -> int:
    """获取用户向量数据库中的记录数量"""
    try:
        vector_store = get_vector_store(username)
        # 获取集合中的所有文档数量
        return vector_store._collection.count()
    except Exception as e:
        logger.error(f"获取向量数量失败: {str(e)}")
        return 0


def clean_document_content(content: str) -> str:
    """
    文档内容数据清洗函数
    处理步骤：
    1. 去除多余的空白字符（多个换行、空格、制表符等）
    2. 去除特殊字符和控制字符
    3. 统一文本格式
    4. 去除重复内容
    """
    import re
    
    if not content:
        return ""
    
    # 1. 去除多余的空白字符
    # 替换多个换行符为单个换行符
    content = re.sub(r'\n{3,}', '\n\n', content)
    # 替换多个空格为单个空格
    content = re.sub(r' {2,}', ' ', content)
    # 替换多个制表符为单个制表符
    content = re.sub(r'\t{2,}', '\t', content)
    # 去除首尾空白
    content = content.strip()
    
    # 2. 去除特殊字符和控制字符
    # 保留基本的标点符号和中文、英文、数字
    content = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\n\r\t。，！？；：、""\'\'（）()【】[]{}《》<>·…—–\-\s]', '', content)
    
    # 3. 去除连续重复的标点符号
    content = re.sub(r'([。，！？；：、]){2,}', r'\1', content)
    
    # 4. 处理编码问题，去除BOM标记
    content = content.replace('\ufeff', '')
    
    # 5. 去除重复的段落（简单处理：连续相同的段落只保留一个）
    lines = content.split('\n')
    cleaned_lines = []
    prev_line = None
    for line in lines:
        if line != prev_line:
            cleaned_lines.append(line)
            prev_line = line
    content = '\n'.join(cleaned_lines)
    
    # 6. 去除空白段落
    paragraphs = content.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    content = '\n\n'.join(paragraphs)
    
    return content


def chunk_and_store_document(file_path: str, username: str):
    """将文档切块并存储到向量数据库"""
    try:
        logger.info(f"开始处理文档: {file_path}, 用户: {username}")
        
        # 读取文档内容
        content = doc_processor.read_document(file_path)
        logger.info(f"文档读取完成，内容长度: {len(content) if content else 0}")
        
        if not content:
            logger.warning(f"文档内容为空: {file_path}")
            raise ValueError("文档内容为空，无法进行向量化")
        
        # 数据清洗
        content = clean_document_content(content)
        logger.info(f"文档清洗完成，内容长度: {len(content) if content else 0}")
        
        if not content:
            logger.warning(f"文档清洗后内容为空: {file_path}")
            return
        
        # 文档切块
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Params.CHUNK_SIZE,
            chunk_overlap=Params.CHUNK_OVERLAP,
            separators=Params.SEPARATORS,
        )
        chunks = text_splitter.split_text(content)
        logger.info(f"文档切块完成，共 {len(chunks)} 个块")
        
        if not chunks:
            logger.warning(f"文档切块结果为空: {file_path}")
            return
        
        # 存储到向量数据库
        vector_store = get_vector_store(username)
        vector_store.add_texts(
            texts=chunks,
            metadatas=[{"source": os.path.basename(file_path)}] * len(chunks)
        )
        
        logger.info(f"文档 {file_path} 已成功切块并存储，共 {len(chunks)} 个段落")
        
    except Exception as e:
        logger.error(f"文档切块存储失败: {str(e)}")
        raise


def delete_document_vectors(filename: str, username: str):
    """删除文档对应的向量"""
    try:
        vector_store = get_vector_store(username)
        
        # 获取集合中的所有文档
        try:
            all_docs = vector_store._collection.get()
            if all_docs and 'ids' in all_docs and 'metadatas' in all_docs:
                ids_to_delete = []
                for idx, metadata in enumerate(all_docs['metadatas']):
                    if metadata and metadata.get('source') == filename:
                        ids_to_delete.append(all_docs['ids'][idx])
                
                if ids_to_delete:
                    vector_store._collection.delete(ids=ids_to_delete)
                    logger.info(f"已删除文档 {filename} 对应的 {len(ids_to_delete)} 个向量")
                else:
                    logger.info(f"未找到文档 {filename} 对应的向量")
            else:
                logger.info("向量数据库为空")
        except Exception as get_error:
            # 如果使用 _collection.get() 失败，尝试其他方法
            logger.warning(f"使用 _collection.get() 失败，尝试 similarity_search: {str(get_error)}")
            results = vector_store.similarity_search(f"source:{filename}", k=100)
            
            if results:
                ids_to_delete = [doc.metadata.get('id') for doc in results if doc.metadata.get('id')]
                if ids_to_delete:
                    vector_store._collection.delete(ids=ids_to_delete)
                    logger.info(f"已删除文档 {filename} 对应的 {len(ids_to_delete)} 个向量")
            else:
                logger.info(f"未找到文档 {filename} 对应的向量")
        
    except Exception as e:
        logger.error(f"删除向量失败: {str(e)}")
        # 如果删除向量失败，继续执行更新操作，不中断流程


@router.get("/documents")
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取用户的所有知识库文档"""
    user_dir = get_user_knowledge_dir(current_user.username)
    
    documents = []
    if user_dir.exists():
        for file_path in user_dir.iterdir():
            if file_path.is_file():
                # 过滤掉系统临时文件和隐藏文件
                filename = file_path.name
                # 排除以 ~$ 开头的Office临时文件、以 . 开头的隐藏文件、Thumbs.db等系统文件
                if not (filename.startswith('~$') or filename.startswith('.') or filename == 'Thumbs.db'):
                    documents.append({
                        "filename": filename,
                        "size": file_path.stat().st_size,
                        "modified_at": file_path.stat().st_mtime
                    })
    
    return {"documents": documents}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """上传知识库文档"""
    user_dir = get_user_knowledge_dir(current_user.username)
    
    # 安全检查：只允许支持的文档格式
    allowed_extensions = Params.DOC_SUPPORTED_FORMATS | {'.md', '.pdf'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式，支持的格式: {', '.join(allowed_extensions)}")
    
    # 保存文件
    file_path = user_dir / file.filename
    try:
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        logger.info(f"用户 {current_user.username} 上传文档: {file.filename}")
        
        # 尝试触发RAG文档切块（失败不影响文件保存）
        try:
            chunk_and_store_document(str(file_path), current_user.username)
            return {"status": "success", "message": "文档上传成功", "filename": file.filename}
        except Exception as chunk_error:
            logger.warning(f"文档向量化失败，但文件已保存: {str(chunk_error)}")
            return {
                "status": "warning",
                "message": "文档上传成功，但无法提取文本内容（可能是扫描件或加密文件），该文档暂无法用于智能问答",
                "filename": file.filename,
                "warning": "无法提取文本内容，文档暂无法用于智能问答"
            }
        
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@router.put("/update/{filename}")
async def update_document(
    filename: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """更新知识库文档（支持替换为不同名称的文件）"""
    user_dir = get_user_knowledge_dir(current_user.username)
    old_file_path = user_dir / filename
    
    if not old_file_path.exists():
        raise HTTPException(status_code=404, detail="文档不存在")
    
    # 获取新文件名
    new_filename = file.filename
    new_file_path = user_dir / new_filename
    
    # 保存新文件
    try:
        content = await file.read()
        with open(new_file_path, 'wb') as f:
            f.write(content)
        
        # 删除旧文件（如果文件名不同）
        if new_filename != filename and old_file_path.exists():
            os.remove(old_file_path)
        
        # 删除旧文档的向量
        delete_document_vectors(filename, current_user.username)
        
        # 重新触发RAG文档切块
        try:
            chunk_and_store_document(str(new_file_path), current_user.username)
        except Exception as chunk_error:
            logger.warning(f"文档切块失败，文件可能已损坏: {str(chunk_error)}")
        
        logger.info(f"用户 {current_user.username} 更新文档: {filename} -> {new_filename}")
        return {"message": "文档更新成功", "filename": new_filename}
        
    except Exception as e:
        logger.error(f"文件更新失败: {str(e)}")
        # 如果新文件已创建，清理它
        if new_file_path.exists():
            os.remove(new_file_path)
        raise HTTPException(status_code=500, detail=f"文件更新失败: {str(e)}")


@router.get("/files/{filename}")
async def get_knowledge_file(
    filename: str,
    token: str = None,
    db: Session = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """获取知识库文件（用于预览HTML等文件）"""
    from fastapi.responses import FileResponse
    
    current_user = None
    
    # 优先使用URL参数中的token
    if token:
        from core.auth.auth import decode_access_token
        payload = decode_access_token(token)
        if payload:
            username = payload.get("sub")
            if username:
                current_user = db.query(User).filter(User.username == username).first()
    
    # 如果URL参数token无效，尝试从请求头获取认证
    if not current_user and credentials:
        from core.auth.auth import decode_access_token
        payload = decode_access_token(credentials.credentials)
        if payload:
            username = payload.get("sub")
            if username:
                current_user = db.query(User).filter(User.username == username).first()
    
    if not current_user:
        raise HTTPException(status_code=401, detail="需要登录")
    
    user_dir = get_user_knowledge_dir(current_user.username)
    file_path = user_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 获取文件扩展名
    ext = file_path.suffix.lower()
    
    # 根据文件类型设置正确的content-type
    content_type = "application/octet-stream"
    inline_filename = None
    
    if ext == ".html":
        content_type = "text/html; charset=utf-8"
        # 对于HTML文件，设置inline模式让浏览器直接打开
        inline_filename = filename
    elif ext == ".htm":
        content_type = "text/html; charset=utf-8"
        inline_filename = filename
    elif ext == ".txt":
        content_type = "text/plain; charset=utf-8"
        inline_filename = filename
    elif ext == ".md":
        content_type = "text/markdown; charset=utf-8"
        inline_filename = filename
    
    # 使用 FileResponse，对于可预览的文件不设置filename参数（避免下载）
    if inline_filename:
        # 对文件名进行URL编码以支持中文
        from urllib.parse import quote
        encoded_filename = quote(inline_filename, encoding='utf-8')
        return FileResponse(
            path=str(file_path),
            media_type=content_type,
            headers={"Content-Disposition": f"inline; filename=\"{encoded_filename}\"; filename*=UTF-8''{encoded_filename}"}
        )
    else:
        return FileResponse(
            path=str(file_path),
            media_type=content_type,
            filename=filename
        )


@router.delete("/delete/{filename}")
async def delete_document(
    filename: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除知识库文档"""
    user_dir = get_user_knowledge_dir(current_user.username)
    file_path = user_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文档不存在")
    
    try:
        # 删除文件
        file_path.unlink()
        
        # 删除对应的向量
        delete_document_vectors(filename, current_user.username)
        
        logger.info(f"用户 {current_user.username} 删除文档: {filename}")
        return {"message": "文档删除成功", "filename": filename}
        
    except Exception as e:
        logger.error(f"文件删除失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")


@router.get("/search")
async def search_knowledge(
    query: str,
    k: int = 3,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """搜索知识库中的相关文档"""
    try:
        vector_store = get_vector_store(current_user.username)
        results = vector_store.similarity_search(query, k=k)
        
        documents = []
        for doc in results:
            documents.append({
                "content": doc.page_content,
                "source": doc.metadata.get('source', '')
            })
        
        return {"results": documents}
        
    except Exception as e:
        logger.error(f"知识库搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@router.post("/reindex")
async def reindex_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """重新索引用户的所有文档（用于修复文档未正确向量化的问题）"""
    user_dir = get_user_knowledge_dir(current_user.username)
    
    if not user_dir.exists():
        return {"message": "用户目录不存在"}
    
    # 获取用户的所有文档文件
    allowed_extensions = Params.DOC_SUPPORTED_FORMATS | {'.md', '.pdf'}
    document_files = []
    
    for file_path in user_dir.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            if file_ext in allowed_extensions and not file_path.name.startswith('.'):
                document_files.append(file_path)
    
    if not document_files:
        return {"message": "未找到需要重新索引的文档"}
    
    # 先清空现有的向量数据库
    try:
        vector_store = get_vector_store(current_user.username)
        # 删除所有向量
        vector_store.delete_collection()
        # 重新创建向量存储
        persist_dir = PathConfig.COMMON_KNOWLEDGE_DIR / current_user.username / "vector_db"
        embeddings = DashScopeEmbeddings(model=Params.DEFAULT_EMBEDDING_MODEL)
        vector_stores[current_user.username] = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name=f"knowledge_{current_user.username}"
        )
        logger.info(f"已清空用户 {current_user.username} 的向量数据库")
    except Exception as e:
        logger.warning(f"清空向量数据库失败: {str(e)}")
    
    # 重新向量化所有文档
    success_count = 0
    fail_count = 0
    
    for file_path in document_files:
        try:
            chunk_and_store_document(str(file_path), current_user.username)
            success_count += 1
        except Exception as e:
            logger.error(f"重新索引文档 {file_path.name} 失败: {str(e)}")
            fail_count += 1
    
    logger.info(f"用户 {current_user.username} 重新索引完成: 成功 {success_count} 个, 失败 {fail_count} 个")
    return {"message": f"重新索引完成", "success": success_count, "failed": fail_count}


def get_all_users() -> list:
    """获取所有注册用户的用户名列表"""
    try:
        from core.auth.database import get_db
        db = next(get_db())
        users = db.query(User).all()
        return [user.username for user in users]
    except Exception as e:
        logger.error(f"获取所有用户失败: {str(e)}")
        return []