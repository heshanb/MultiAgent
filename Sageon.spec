# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

block_cipher = None

# 娣诲姞椤圭洰鏍圭洰褰曞埌璺緞
project_root = Path(r'D:\Python_project\MultiAgent').resolve()
sys.path.insert(0, str(project_root))

a = Analysis(
    ['main.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
        ('resource', 'resource'),
        ('settings', 'settings'),
        ('core', 'core'),
        ('.env', '.'),
        ('models/models--Qdrant--bm25', 'models/models--Qdrant--bm25'),
    ],
    hiddenimports=[
        'langchain',
        'langchain_openai',
        'langchain_core',
        'langchain_community',
        'langchain_chroma',
        'langchain_classic',
        'langchain_text_splitters',
        'langchain_mcp_adapters',
        'langgraph',
        'pydantic',
        'openai',
        'paddlepaddle',
        'paddleocr',
        'chromadb',
        'pypdf',
        'pillow',
        'ezdxf',
        'numpy',
        'pdf2image',
        'python_dotenv',
        'sqlalchemy',
        'aiosqlite',
        'pymysql',
        'cryptography',
        'bcrypt',
        'redis',
        'reportlab',
        'pymupdf',
        'langchain_qdrant',
        'qdrant_client',
        'fastembed',
        'FlagEmbedding',
        'onnxruntime',
        'slowapi',
        'pybreaker',
        'tenacity',
        'pywebview',
        'webview',
        'uvicorn',
        'fastapi',
        'sse_starlette',
        'jose',
        'passlib',
        'magic',
        'xlrd',
        'openpyxl',
        'docx',
        'pptx',
        'portalocker',
        'msvcrt',
        'qdrant_client.local',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Sageon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resource/sageon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Sageon',
)