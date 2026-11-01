"""
Sageon - Windows 打包脚本
支持生成安装包和便携版
"""
import os
import sys
import shutil
import subprocess
import importlib.util
from pathlib import Path

from version_config import (
    APP_NAME,
    VERSION,
    build_filename,
    get_display_name,
    get_file_version,
)

BASE_DIR = Path(__file__).parent.resolve()

def check_requirements():
    """检查打包所需的工具是否已安装"""
    print(" 检查打包环境...")

    required_packages = {
        'pyinstaller': 'PyInstaller',
        'pywebview': 'webview',
    }
    missing = []

    for package, module in required_packages.items():
        if importlib.util.find_spec(module) is not None:
            print(f"  ✓ {package}")
        else:
            missing.append(package)
            print(f"  ✗ {package} (未安装)")

    if missing:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing)}")
        print("请运行: pip install " + " ".join(missing))
        return False

    return True

def create_spec_file():
    """创建 PyInstaller spec 文件"""
    print("\n📝 创建打包配置文件...")
    
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

block_cipher = None

# 添加项目根目录到路径
project_root = Path(r'{BASE_DIR}').resolve()
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
    hooksconfig={{}},
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
    name='{APP_NAME}',
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
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{APP_NAME}',
)
'''
    
    spec_file = BASE_DIR / f'{APP_NAME}.spec'
    with open(spec_file, 'w', encoding='utf-8') as f:
        f.write(spec_content)
    
    print(f"  ✓ 已创建: {spec_file}")
    return spec_file

def build_with_pyinstaller():
    """使用 PyInstaller 打包"""
    print("\n🔨 开始打包 (这可能需要几分钟)...")
    
    spec_file = BASE_DIR / f'{APP_NAME}.spec'
    
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--clean', '--noconfirm', '-y',
        str(spec_file)
    ]
    
    try:
        result = subprocess.run(cmd, cwd=str(BASE_DIR), check=True)
        print("  ✓ 打包完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ 打包失败: {e}")
        return False

def find_chinese_isl():
    """查找 Inno Setup 中文语言文件，官方安装包默认不附带该文件"""
    candidates = [
        BASE_DIR / "installer" / "ChineseSimplified.isl",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "Languages" / "ChineseSimplified.isl",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "Languages" / "ChineseSimplified.isl",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "Languages" / "ChineseSimplified.isl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_iscc():
    """查找 Inno Setup 编译器"""
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Inno Setup 6" / "ISCC.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def create_installer_with_inno():
    """使用 Inno Setup 创建安装包"""
    print("\n📋 创建 Inno Setup 安装脚本...")

    chinese_isl = find_chinese_isl()
    if chinese_isl:
        languages_section = f'Name: "chinesesimplified"; MessagesFile: "{chinese_isl}"'
    else:
        languages_section = '; 未检测到 ChineseSimplified.isl，安装向导使用默认语言'
        print("  ⚠️  未找到 ChineseSimplified.isl，安装向导将使用默认语言")

    iss_content = f'''[Setup]
AppName={get_display_name()}
AppVersion={VERSION}
AppVerName={get_display_name()}
AppPublisher=Sageon Team
AppPublisherURL=https://example.com
AppSupportURL=https://example.com
AppUpdatesURL=https://example.com
VersionInfoVersion={get_file_version()}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DefaultDirName={{autopf}}\\{APP_NAME}
DisableDirPage=no
DefaultGroupName={get_display_name()}
OutputDir={BASE_DIR}\\dist
OutputBaseFilename={build_filename().rsplit('.', 1)[0]}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Languages]
{languages_section}

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked

[Files]
Source: "{BASE_DIR}\\dist\\{APP_NAME}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.exe"
Name: "{{group}}\\{{cm:UninstallProgram,{APP_NAME}}}"; Filename: "{{uninstallexe}}"
Name: "{{autodesktop}}\\{APP_NAME}"; Filename: "{{app}}\\{APP_NAME}.exe"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{APP_NAME}.exe"; Description: "{{cm:LaunchProgram,{APP_NAME}}}"; Flags: nowait postinstall skipifsilent
'''
    
    iss_file = BASE_DIR / f'{APP_NAME}_installer.iss'
    with open(iss_file, 'w', encoding='utf-8') as f:
        f.write(iss_content)
    
    print(f"  ✓ 安装脚本已创建: {iss_file}")

    iscc = find_iscc()
    if not iscc:
        print("\n⚠️  未找到 Inno Setup 编译器 ISCC.exe")
        print("   请下载并安装 Inno Setup: https://jrsoftware.org/isdl.php")
        print(f"   安装后打开 {iss_file} 并点击 Build -> Compile")
        return iss_file

    print(f"\n🔨 正在编译安装包 (ISCC: {iscc})...")
    try:
        subprocess.run([str(iscc), str(iss_file)], cwd=str(BASE_DIR), check=True)
        installer = BASE_DIR / 'dist' / build_filename()
        print(f"  ✓ 安装包已生成: {installer}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ 安装包编译失败: {e}")

    return iss_file

def main():
    """主函数"""
    print("=" * 60)
    print(f"  {get_display_name()} - Windows 打包工具")
    print("=" * 60)
    
    if not check_requirements():
        return
    
    print("\n选择打包方式:")
    print("  1. 仅打包成可执行文件")
    print("  2. 打包 + 创建安装包脚本 (需要 Inno Setup)")
    print("  3. 全部打包")
    
    choice = input("\n请选择 (1-3): ").strip()
    
    if choice == '1':
        create_spec_file()
        build_with_pyinstaller()
    elif choice == '2':
        create_spec_file()
        if build_with_pyinstaller():
            create_installer_with_inno()
    elif choice == '3':
        create_spec_file()
        if build_with_pyinstaller():
            create_installer_with_inno()
    else:
        print("无效的选择")
        return
    
    print("\n" + "=" * 60)
    print("✅ 打包完成!")
    print(f"📁 输出目录: {BASE_DIR / 'dist'}")
    print("=" * 60)

if __name__ == '__main__':
    main()