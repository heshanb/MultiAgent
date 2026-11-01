from pathlib import Path
import os
import sys
import re
import logging

class PathConfig:
    """
    路径配置定义文件
    统一管理所有项目中的路径，避免硬编码相对路径
    """
    # 是否为打包后的可执行程序
    FROZEN = getattr(sys, 'frozen', False)

    if FROZEN:
        # 打包后只读资源位于 _internal 目录（PyInstaller 6.x 的 _MEIPASS）
        BASE_DIR = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)).resolve()
        # 用户数据目录，安装目录不可写，日志/数据库/上传等写入此处
        DATA_DIR = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / "Sageon"
    else:
        # 基础目录（基于当前脚本所在位置）
        # Define.py 在 settings 目录下，所以需要向上两级到 MultiAgent 目录
        BASE_DIR = Path(__file__).parent.parent.resolve()
        DATA_DIR = BASE_DIR

    # 各子目录路径
    TEMPLATES_DIR = BASE_DIR / "templates"
    UPLOADS_DIR = DATA_DIR / "uploads"
    OUTPUTS_DIR = DATA_DIR / "outputs"
    CACHE_DIR = DATA_DIR / "cache"
    RESOURCE_DIR = BASE_DIR / "resource"
    DB_DIR = DATA_DIR / "chroma_db"
    KNOWLEDGE_DB_DIR = DATA_DIR / "drawing_knowledge_db"
    LOG_DIR = DATA_DIR / "logs"
    # 知识库目录
    COMMON_KNOWLEDGE_DIR = DATA_DIR / "resource" / "common_knowledge"
    # 模型目录
    MODEL_DIR = DATA_DIR / "models"

    # 验证目录是否存在（可选）
    @classmethod
    def ensure_directories(cls):
        """确保所有必需的目录存在"""
        directories = [
            cls.TEMPLATES_DIR, 
            cls.UPLOADS_DIR,
            cls.OUTPUTS_DIR,
            cls.CACHE_DIR,
            cls.RESOURCE_DIR,
            cls.DB_DIR,
            cls.LOG_DIR,
            cls.COMMON_KNOWLEDGE_DIR
        ]
        for dir_path in directories:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logging.getLogger(__name__).info(f"创建目录: {dir_path}")

class GoEnvConfig:
    """
    Go 工具链环境配置

    应用自身进程启动时会继承一份系统环境变量快照，若 Go 是在此之后才加入 PATH，
    则本进程内执行 go 命令会提示「不是内部或外部命令」。
    该类用于在运行时自动补齐 Go 相关路径，无需重启应用或重启电脑。

    所有路径均在运行时探测，不写死用户名与安装位置，便于迁移到其它服务器：
      1. 已配置的 PATH 中能否直接找到 go
      2. GOROOT 环境变量
      3. 系统/用户注册表 PATH 中包含 Go 的目录
      4. 常见安装位置（Program Files、LOCALAPPDATA、C:\\Go 等）
      5. GOPATH 下的 bin 目录（go install 后的可执行文件）

    若需强制指定，可在下方 GO_BIN_DIR / EXTRA_PATH_DIRS 中填写。
    """

    # 是否启用自动补齐
    ENABLED = True

    # 手动指定 go 可执行文件所在目录（留空则自动探测）
    GO_BIN_DIR = ""

    # 额外追加到 PATH 的目录（在自动探测结果之后追加）
    EXTRA_PATH_DIRS = []

    # 可执行文件名（区分平台）
    GO_EXECUTABLE = "go.exe" if os.name == "nt" else "go"

    # 匹配「以 go 结尾的 bin 目录」的正则，用于从注册表 PATH 中筛选
    _GO_DIR_PATTERN = re.compile(r"[\\/]go[\\/]bin[\\/]?$", re.IGNORECASE)

    # 探测结果缓存，避免重复扫描文件系统与注册表
    _go_bin_dir_cache = None
    _gopath_bin_dir_cache = None
    _resolved = False

    @classmethod
    def _registry_path_values(cls):
        """读取注册表中系统级与用户级的 PATH 原始值"""
        if os.name != "nt":
            return []
        try:
            import winreg
        except ImportError:
            return []

        values = []
        targets = [
            (winreg.HKEY_LOCAL_MACHINE,
             r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        ]
        for root, subkey in targets:
            try:
                with winreg.OpenKey(root, subkey) as key:
                    value, _ = winreg.QueryValueEx(key, "Path")
                    if value:
                        values.append(value)
            except OSError:
                continue
        return values

    @classmethod
    def _split_path(cls, value):
        """拆分 PATH 字符串并展开其中的 %VAR% 形式变量"""
        if not value:
            return []
        result = []
        for item in value.split(os.pathsep):
            item = item.strip().strip('"')
            if not item:
                continue
            expanded = os.path.expandvars(item)
            # 变量无法展开时保留原样，避免丢弃系统原有的路径项
            result.append(expanded if "%" not in expanded else item)
        return result

    @classmethod
    def _candidate_dirs(cls):
        """按优先级收集所有可能的 Go bin 目录"""
        candidates = []

        if cls.GO_BIN_DIR:
            candidates.append(os.path.expandvars(cls.GO_BIN_DIR))

        goroot = os.environ.get("GOROOT")
        if goroot:
            candidates.append(os.path.join(os.path.expandvars(goroot), "bin"))

        for value in cls._registry_path_values():
            for item in cls._split_path(value):
                if cls._GO_DIR_PATTERN.search(item):
                    candidates.append(item)

        for item in cls._split_path(os.environ.get("PATH") or os.environ.get("Path")):
            if cls._GO_DIR_PATTERN.search(item):
                candidates.append(item)

        if os.name == "nt":
            program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
            program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            candidates.append(os.path.join(program_files, "Go", "bin"))
            candidates.append(os.path.join(program_files_x86, "Go", "bin"))
            if local_app_data:
                candidates.append(os.path.join(local_app_data, "Programs", "Go", "bin"))
            candidates.append(r"C:\Go\bin")
        else:
            candidates.extend(["/usr/local/go/bin", "/usr/lib/go/bin", "/opt/go/bin"])

        # 去重并保持顺序
        seen = set()
        unique = []
        for path in candidates:
            normalized = os.path.normpath(path)
            key = normalized.lower() if os.name == "nt" else normalized
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique

    @classmethod
    def find_go_bin_dir(cls):
        """返回首个真实存在 go 可执行文件的目录，未找到时返回空字符串"""
        if cls._resolved:
            return cls._go_bin_dir_cache or ""

        for directory in cls._candidate_dirs():
            if os.path.isfile(os.path.join(directory, cls.GO_EXECUTABLE)):
                cls._go_bin_dir_cache = directory
                break

        cls._resolved = True
        return cls._go_bin_dir_cache or ""

    @classmethod
    def find_gopath_bin_dir(cls):
        """返回 GOPATH 下的 bin 目录（go install 生成的程序所在位置）"""
        if cls._gopath_bin_dir_cache is not None:
            return cls._gopath_bin_dir_cache

        gopath_dirs = cls._split_path(os.environ.get("GOPATH"))
        if not gopath_dirs:
            go_bin = cls.find_go_bin_dir()
            if go_bin:
                # 调用 go env GOPATH 获取真实值（可能来自 go env -w 配置）
                try:
                    import subprocess
                    go_exe = os.path.join(go_bin, cls.GO_EXECUTABLE)
                    completed = subprocess.run(
                        [go_exe, "env", "GOPATH"],
                        capture_output=True, text=True, timeout=10,
                        encoding="utf-8", errors="ignore"
                    )
                    if completed.returncode == 0:
                        gopath_dirs = cls._split_path(completed.stdout.strip())
                except Exception:
                    gopath_dirs = []
            if not gopath_dirs:
                gopath_dirs = [os.path.join(os.path.expanduser("~"), "go")]

        cls._gopath_bin_dir_cache = os.path.join(gopath_dirs[0], "bin")
        return cls._gopath_bin_dir_cache

    @classmethod
    def path_entries(cls):
        """返回需要补充到 PATH 的目录列表（仅保留真实存在的目录）"""
        if not cls.ENABLED:
            return []

        entries = []

        go_bin = cls.find_go_bin_dir()
        if go_bin:
            entries.append(go_bin)

        gopath_bin = cls.find_gopath_bin_dir()
        if gopath_bin and os.path.isdir(gopath_bin):
            entries.append(gopath_bin)

        for extra in cls.EXTRA_PATH_DIRS:
            expanded = os.path.expandvars(extra)
            if expanded and os.path.isdir(expanded):
                entries.append(expanded)

        seen = set()
        unique = []
        for path in entries:
            normalized = os.path.normpath(path)
            key = normalized.lower() if os.name == "nt" else normalized
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique

    @classmethod
    def apply(cls, env):
        """将 Go 相关目录前置到 env 的 PATH 中，返回同一个 env 对象"""
        entries = cls.path_entries()
        if not entries or not env:
            return env

        # Windows 下环境变量名不区分大小写，需先找出实际使用的键名
        path_key = None
        for key in env:
            if key.upper() == "PATH":
                path_key = key
                break
        if path_key is None:
            path_key = "Path" if os.name == "nt" else "PATH"

        current = cls._split_path(env.get(path_key, ""))
        current_keys = set()
        for item in current:
            normalized = os.path.normpath(item)
            current_keys.add(normalized.lower() if os.name == "nt" else normalized)

        missing = []
        for entry in entries:
            normalized = os.path.normpath(entry)
            key = normalized.lower() if os.name == "nt" else normalized
            if key not in current_keys:
                missing.append(entry)

        if missing:
            env[path_key] = os.pathsep.join(missing + current) if current else os.pathsep.join(missing)
        return env


class JavaEnvConfig:
    """
    JDK 工具链环境配置

    与 GoEnvConfig 同理：应用进程启动时会继承一份系统环境变量快照，
    若 JDK 是在此之后才安装的，则进程内 PATH 与 JAVA_HOME 仍是旧值，
    执行 javac / java 会提示找不到命令，必须重启应用甚至重启电脑。

    该类在运行时重新探测 JDK，使新装的 JDK 无需重启即可生效：
      1. 已配置的 JAVA_HOME
      2. 注册表中系统级与用户级的 JAVA_HOME、PATH
      3. 常见安装位置（Program Files 下的 Java / Microsoft / Adoptium 等，
         含 jdk-17.x 这类带版本号的子目录）
    """

    # 是否启用自动补齐
    ENABLED = True

    # 手动指定 JDK 根目录（留空则自动探测）
    JDK_HOME = ""

    # 额外追加到 PATH 的目录（在自动探测结果之后追加）
    EXTRA_PATH_DIRS = []

    # 可执行文件名（区分平台）
    JAVAC_EXECUTABLE = "javac.exe" if os.name == "nt" else "javac"

    # 匹配 JDK 根目录的正则，用于从注册表 PATH 中筛选
    _JDK_DIR_PATTERN = re.compile(r"[\\/](?:jdk|jre|java)[^\\/]*[\\/]bin[\\/]?$", re.IGNORECASE)

    # 探测结果缓存，避免重复扫描文件系统与注册表
    _jdk_bin_dir_cache = None
    _resolved = False

    @classmethod
    def _registry_values(cls):
        """读取注册表中系统级与用户级的环境变量值"""
        if os.name != "nt":
            return []
        try:
            import winreg
        except ImportError:
            return []

        values = []
        targets = [
            (winreg.HKEY_LOCAL_MACHINE,
             r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        ]
        for root, subkey in targets:
            try:
                with winreg.OpenKey(root, subkey) as key:
                    for name in ("JAVA_HOME", "Path"):
                        try:
                            value, _ = winreg.QueryValueEx(key, name)
                        except OSError:
                            continue
                        if value:
                            values.append((name, value))
            except OSError:
                continue
        return values

    @classmethod
    def _split_path(cls, value):
        """拆分 PATH 字符串并展开其中的 %VAR% 形式变量"""
        if not value:
            return []
        result = []
        for item in value.split(os.pathsep):
            item = item.strip().strip('"')
            if not item:
                continue
            expanded = os.path.expandvars(item)
            result.append(expanded if "%" not in expanded else item)
        return result

    @classmethod
    def _version_sub_dirs(cls, base_dir):
        """返回常见 JDK 安装根目录下的候选 JDK 目录（含带版本号的子目录）"""
        if not os.path.isdir(base_dir):
            return []
        found = []
        try:
            for sub in os.listdir(base_dir):
                sub_path = os.path.join(base_dir, sub)
                if os.path.isdir(sub_path):
                    found.append(sub_path)
        except OSError:
            return []
        # 版本号越新越优先（目录名倒序，jdk-21 排在 jdk-8 之前）
        return sorted(found, reverse=True)

    @classmethod
    def _candidate_homes(cls):
        """按优先级收集所有可能的 JDK 根目录"""
        candidates = []

        if cls.JDK_HOME:
            candidates.append(os.path.expandvars(cls.JDK_HOME))

        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            candidates.append(os.path.expandvars(java_home))

        # 注册表里的 JAVA_HOME 才是 JDK 刚安装后的最新值，进程环境变量可能已过期
        for name, value in cls._registry_values():
            if name == "JAVA_HOME":
                candidates.append(os.path.expandvars(value))
            else:
                for item in cls._split_path(value):
                    if cls._JDK_DIR_PATTERN.search(item):
                        candidates.append(os.path.dirname(item))

        for item in cls._split_path(os.environ.get("PATH") or os.environ.get("Path")):
            if cls._JDK_DIR_PATTERN.search(item):
                candidates.append(os.path.dirname(item))

        if os.name == "nt":
            program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
            program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
            local_app_data = os.environ.get("LOCALAPPDATA", "")
            bases = [
                os.path.join(program_files, "Java"),
                os.path.join(program_files, "Microsoft"),
                os.path.join(program_files, "Eclipse Adoptium"),
                os.path.join(program_files, "Amazon Corretto"),
                os.path.join(program_files, "Zulu"),
                os.path.join(program_files, "BellSoft"),
                os.path.join(program_files, "Semeru"),
                os.path.join(program_files_x86, "Java"),
            ]
            if local_app_data:
                bases.append(os.path.join(local_app_data, "Programs", "Eclipse Adoptium"))
            for base in bases:
                if os.path.basename(base).lower() == "microsoft":
                    # Microsoft 目录下混有其它产品，只取 jdk 开头的子目录
                    for sub_path in cls._version_sub_dirs(base):
                        if os.path.basename(sub_path).lower().startswith("jdk"):
                            candidates.append(sub_path)
                else:
                    candidates.extend(cls._version_sub_dirs(base))

        # 去重并保持顺序
        seen = set()
        unique = []
        for path in candidates:
            normalized = os.path.normpath(path)
            key = normalized.lower() if os.name == "nt" else normalized
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique

    @classmethod
    def find_jdk_bin_dir(cls):
        """返回首个真实存在 javac 的 bin 目录，未找到时返回空字符串"""
        if cls._resolved:
            return cls._jdk_bin_dir_cache or ""

        for home in cls._candidate_homes():
            bin_dir = os.path.join(home, "bin")
            if os.path.isfile(os.path.join(bin_dir, cls.JAVAC_EXECUTABLE)):
                cls._jdk_bin_dir_cache = bin_dir
                break

        cls._resolved = True
        return cls._jdk_bin_dir_cache or ""

    @classmethod
    def find_jdk_home(cls):
        """返回 JDK 根目录（bin 的上一级），未找到时返回空字符串"""
        bin_dir = cls.find_jdk_bin_dir()
        return os.path.dirname(bin_dir) if bin_dir else ""

    @classmethod
    def path_entries(cls):
        """返回需要补充到 PATH 的目录列表（仅保留真实存在的目录）"""
        if not cls.ENABLED:
            return []

        entries = []

        bin_dir = cls.find_jdk_bin_dir()
        if bin_dir:
            entries.append(bin_dir)

        for extra in cls.EXTRA_PATH_DIRS:
            expanded = os.path.expandvars(extra)
            if expanded and os.path.isdir(expanded):
                entries.append(expanded)

        seen = set()
        unique = []
        for path in entries:
            normalized = os.path.normpath(path)
            key = normalized.lower() if os.name == "nt" else normalized
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique

    @classmethod
    def apply(cls, env):
        """将 JDK 目录前置到 env 的 PATH 并补齐 JAVA_HOME，返回同一个 env 对象"""
        if not env:
            return env

        entries = cls.path_entries()

        # Windows 下环境变量名不区分大小写，需先找出实际使用的键名
        path_key = None
        for key in env:
            if key.upper() == "PATH":
                path_key = key
                break
        if path_key is None:
            path_key = "Path" if os.name == "nt" else "PATH"

        if entries:
            current = cls._split_path(env.get(path_key, ""))
            current_keys = set()
            for item in current:
                normalized = os.path.normpath(item)
                current_keys.add(normalized.lower() if os.name == "nt" else normalized)

            missing = []
            for entry in entries:
                normalized = os.path.normpath(entry)
                key = normalized.lower() if os.name == "nt" else normalized
                if key not in current_keys:
                    missing.append(entry)

            if missing:
                env[path_key] = os.pathsep.join(missing + current) if current else os.pathsep.join(missing)

        # JAVA_HOME 缺失或指向已失效的路径时，用探测到的 JDK 根目录补上
        home = cls.find_jdk_home()
        if home:
            current_home = env.get("JAVA_HOME", "")
            if not current_home or not os.path.isfile(
                    os.path.join(current_home, "bin", cls.JAVAC_EXECUTABLE)):
                env["JAVA_HOME"] = home
        return env


class Params:
    DEFAULT_CHAT_MODEL="qwen3.8-flash"
    DEFAULT_EMBEDDING_MODEL="text-embedding-v4"
    DEFAULT_MULTIMODAL_MODEL="qwen3-vl-plus"
    DEFAULT_TEXT_TOOL_MODEL="qwen-turbo" # 轻量级模型
    API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_SCORE_THRESHOLD=0.2
    DEFAULT_RRF_WEIGHT=0.7
    DEFAULT_RRF_K=60

    SUPERVISOR_NODE = "supervisor_node"
    TRAVEL_NODE = "travel_node"
    JOKE_NODE = "joke_node"
    COUPLET_NODE = "couplet_node"
    DOCUMENT_NODE = "document_node"
    CODE_NODE = "code_node"
    OTHER_NODE = "other_node"
    DRAWING_NODE = "drawing_node"
    PROBLEM_ANALYSIS_NODE = "problem_analysis_node"
    REFLECTION_NODE = "reflection_node"

    NODE_LIST =  ["supervisor", "travel", "joke", "couplet", "document", "code", "drawing", "problem_analysis", "other"]
    MAPPING_NODE = {
        SUPERVISOR_NODE: "supervisor",
        TRAVEL_NODE: "travel",
        JOKE_NODE: "joke",
        COUPLET_NODE: "couplet",
        DOCUMENT_NODE: "document",
        CODE_NODE: "code",
        DRAWING_NODE: "drawing",
        PROBLEM_ANALYSIS_NODE: "problem_analysis",
        OTHER_NODE: "other"
    }

    DOC_SUPPORTED_FORMATS = {'.docx', '.doc', '.txt', '.xlsx', '.xls', '.pptx', '.ppt', '.html', '.json', '.pdf', '.md'}

    # ===================== 图纸助手节点名称 =====================
    RESET_DRAWING_NODE = "reset_drawing"
    PARSE_DRAWING_NODE = "parse_drawing"
    CHECK_DIMENSION_NODE = "check_dimension"
    CALL_TOOLS_NODE = "call_tools"
    GEN_FINAL_ANSWER_NODE = "gen_final_answer"

    # PDF图纸存放路径
    PDF_FOLDER = os.path.join(PathConfig.RESOURCE_DIR, "enginerring_standard_pdf")
    # 矢量图纸存放路径
    DXF_FOLDER = os.path.join(PathConfig.RESOURCE_DIR, "enginerring_drawing_CAD")
    # 图片图纸存放路径
    PIC_FOLDER = os.path.join(PathConfig.RESOURCE_DIR, "enginerring_drawing_pic")
    COLLECTION_NAME = "engineering_standard"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]


    # ===================== 代码助手节点名称 =====================
    MAPPING_MODEL = {
        "Qwen3.8-Flash": "qwen3.8-flash",
        "DeepSeek-V4-Pro": "deepseek-v4-pro",
        "GLM-5.2-Fast-Preview": "glm-5.2-fast-preview",
    }

    # 前端下拉框的短 ID 别名，同样需要还原成接口实际模型名
    MODEL_ALIASES = {
        "qwen": "qwen3.8-flash",
        "default": "qwen3.8-flash",
        "deepseek": "deepseek-v4-pro",
        "glm": "glm-5.2-fast-preview",
    }

    @classmethod
    def resolve_model_name(cls, name):
        """把前端传来的展示名/短 ID 统一还原为接口实际模型名

        界面显示的是 Qwen3.8-Flash 这类名字，而百炼接口只认
        qwen3.8-flash，直接把展示名发过去会返回 model_not_found。
        """
        key = str(name or "").strip()
        if not key:
            return cls.DEFAULT_CHAT_MODEL

        if key in cls.MAPPING_MODEL:
            return cls.MAPPING_MODEL[key]
        if key in cls.MODEL_ALIASES:
            return cls.MODEL_ALIASES[key]

        lowered = key.lower()
        for display, model_id in cls.MAPPING_MODEL.items():
            if display.lower() == lowered:
                return model_id
        for alias, model_id in cls.MODEL_ALIASES.items():
            if alias.lower() == lowered:
                return model_id

        return key

    # ===================== Redis 缓存配置 =====================
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_PASSWORD = ""

PathConfig.ensure_directories()