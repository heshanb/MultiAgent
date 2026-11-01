"""版本管理配置。

应用英文名：Sageon，寓意「运行的智者」。
版本命名规则：1.0-1.9 为第 1 代版本，2.0-2.9 为第 2 代版本，以此类推。
对外只发布唯一的正式商业版本，命名形如 Sageon 1.1。
"""

from settings.Define import PathConfig
from datetime import datetime

APP_NAME = "Sageon"
APP_NAME_CN = "小智"
APP_NAME_MEANING = "运行的智者"

VERSION = "1.0"
RELEASE_DATE = datetime.now().strftime("%Y-%m-%d")

MINOR_PER_GENERATION = 10

DIST_DIR = PathConfig.BASE_DIR / "dist"


def parse_version(version=None):
    """将 "1.1" 解析为 (1, 1)。"""
    raw = (version or VERSION).strip()
    parts = raw.split(".")
    if len(parts) < 2:
        raise ValueError(f"版本号格式应为「主版本.次版本」，当前为: {version or VERSION}")
    return int(parts[0]), int(parts[1])


def validate_version(version=None):
    """校验并规范化版本号，返回 "主版本.次版本" 形式。"""
    major, minor = parse_version(version)
    if not 0 <= minor < MINOR_PER_GENERATION:
        raise ValueError(
            f"次版本号需在 0-{MINOR_PER_GENERATION - 1} 之间，当前为: {version or VERSION}"
        )
    return f"{major}.{minor}"


def get_generation(version=None):
    """返回版本所属的代数，1.0-1.9 为第 1 代，2.0-2.9 为第 2 代。"""
    major, _ = parse_version(version)
    return major


def get_generation_label(version=None):
    return f"第 {get_generation(version)} 代"


def get_display_name(version=None):
    """商业发布名称，如 Sageon 1.1。"""
    return f"{APP_NAME} {validate_version(version)}"


def get_file_version(version=None):
    """Windows 可执行文件需要的四段式版本号。"""
    major, minor = parse_version(version)
    return f"{major}.{minor}.0.0"

def get_release_date(version=None):
    """返回当前正式版本的发布日期。"""
    return RELEASE["release_date"]


def build_filename(version=None, arch="x64"):
    """生成安装包文件名，如 Sageon-1.1-Windows-x64.exe。"""
    return f"{APP_NAME}-{validate_version(version)}-Windows-{arch}.exe"


def build_artifact(version=None, arch="x64"):
    filename = build_filename(version, arch=arch)
    return {
        "filename": filename,
        "path": str(DIST_DIR / filename),
        "content_type": "application/octet-stream",
    }


RELEASE = {
    "version": VERSION,
    "display_name": get_display_name(),
    "generation": get_generation(),
    "generation_label": get_generation_label(),
    "release_date": RELEASE_DATE,
    "platforms": {
        "windows": {
            "label": "Windows",
            "arch": "64 bit",
            "artifacts": {
                "windows-x64": build_artifact(VERSION, "x64"),
            },
        },
    },
}

DOWNLOADS = {
    key: dict(artifact, version=RELEASE["version"], display_name=RELEASE["display_name"])
    for platform in RELEASE["platforms"].values()
    for key, artifact in platform["artifacts"].items()
}

CURRENT_RELEASE = RELEASE