"""
敏感操作日志审计模块
记录用户的关键操作，用于安全审计
"""
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
from settings.Define import PathConfig
import logging

# 审计日志配置
AUDIT_LOG_DIR = PathConfig.BASE_DIR / "logs" / "audit"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "audit.log"

# 确保日志目录存在
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)

# 审计日志级别
class AuditLevel:
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, log_file: Path = AUDIT_LOG_FILE):
        self.log_file = log_file
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log(
        self,
        action: str,
        user: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        level: str = AuditLevel.INFO,
        resource: Optional[str] = None,
    ):
        """
        记录审计日志
        
        Args:
            action: 操作类型（如：LOGIN, LOGOUT, FILE_UPLOAD, USER_CREATE等）
            user: 用户名
            ip_address: IP地址
            details: 详细信息
            level: 日志级别
            resource: 操作的资源
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user or "anonymous",
            "ip_address": ip_address or "unknown",
            "resource": resource or "",
            "details": details or {},
            "level": level,
        }
        
        log_message = json.dumps(log_entry, ensure_ascii=False)
        
        if level == AuditLevel.INFO:
            self.logger.info(log_message)
        elif level == AuditLevel.WARNING:
            self.logger.warning(log_message)
        elif level == AuditLevel.ERROR:
            self.logger.error(log_message)
        elif level == AuditLevel.CRITICAL:
            self.logger.critical(log_message)

    def log_login(self, username: str, ip_address: str, success: bool):
        """记录登录操作"""
        self.log(
            action="USER_LOGIN",
            user=username,
            ip_address=ip_address,
            details={"success": success},
            level=AuditLevel.INFO if success else AuditLevel.WARNING,
        )

    def log_logout(self, username: str, ip_address: str):
        """记录登出操作"""
        self.log(
            action="USER_LOGOUT",
            user=username,
            ip_address=ip_address,
            level=AuditLevel.INFO,
        )

    def log_register(self, username: str, email: str, ip_address: str):
        """记录注册操作"""
        self.log(
            action="USER_REGISTER",
            user=username,
            ip_address=ip_address,
            details={"email": email},
            level=AuditLevel.INFO,
        )

    def log_file_upload(self, username: str, filename: str, file_size: int, ip_address: str):
        """记录文件上传操作"""
        self.log(
            action="FILE_UPLOAD",
            user=username,
            ip_address=ip_address,
            details={
                "filename": filename,
                "file_size": file_size,
                "file_size_mb": round(file_size / 1024 / 1024, 2),
            },
            resource=filename,
            level=AuditLevel.INFO,
        )

    def log_file_upload_failed(self, username: str, filename: str, reason: str, ip_address: str):
        """记录文件上传失败操作"""
        self.log(
            action="FILE_UPLOAD_FAILED",
            user=username,
            ip_address=ip_address,
            details={
                "filename": filename,
                "reason": reason,
            },
            resource=filename,
            level=AuditLevel.WARNING,
        )

    def log_chat(self, username: str, message_length: int, ip_address: str):
        """记录聊天操作"""
        self.log(
            action="CHAT_MESSAGE",
            user=username,
            ip_address=ip_address,
            details={"message_length": message_length},
            level=AuditLevel.INFO,
        )

    def log_password_change(self, username: str, ip_address: str):
        """记录密码修改操作"""
        self.log(
            action="PASSWORD_CHANGE",
            user=username,
            ip_address=ip_address,
            level=AuditLevel.INFO,
        )

    def log_role_change(self, admin_user: str, target_user: str, new_role: str, ip_address: str):
        """记录角色变更操作"""
        self.log(
            action="ROLE_CHANGE",
            user=admin_user,
            ip_address=ip_address,
            details={"target_user": target_user, "new_role": new_role},
            level=AuditLevel.INFO,
        )

    def log_security_event(self, event_type: str, details: Dict[str, Any], ip_address: str):
        """记录安全事件"""
        self.log(
            action=f"SECURITY_{event_type}",
            ip_address=ip_address,
            details=details,
            level=AuditLevel.WARNING,
        )


# 全局审计日志实例
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """获取审计日志实例"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger