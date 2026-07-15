"""
安全中间件
- 请求头安全检查
- 敏感信息过滤
- 请求频率限制
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import time
from collections import defaultdict
from .audit_logger import get_audit_logger


class RateLimiter:
    """简单的内存速率限制器"""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """检查是否允许请求"""
        now = time.time()
        # 清理过期记录
        self.requests[key] = [
            t for t in self.requests[key] if now - t < self.window_seconds
        ]
        
        if len(self.requests[key]) >= self.max_requests:
            return False
        
        self.requests[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """获取剩余请求数"""
        now = time.time()
        self.requests[key] = [
            t for t in self.requests[key] if now - t < self.window_seconds
        ]
        return max(0, self.max_requests - len(self.requests[key]))


# 全局速率限制器实例
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


class SecurityMiddleware(BaseHTTPMiddleware):
    """安全中间件"""

    async def dispatch(self, request: Request, call_next):
        # 获取客户端IP
        client_ip = request.client.host if request.client else "unknown"
        
        # 速率限制检查
        rate_key = f"ip:{client_ip}"
        if not rate_limiter.is_allowed(rate_key):
            audit_logger = get_audit_logger()
            audit_logger.log_security_event(
                "RATE_LIMIT_EXCEEDED",
                {"ip": client_ip, "path": request.url.path},
                client_ip,
            )
            return Response(
                content='{"detail": "请求过于频繁，请稍后再试"}',
                status_code=429,
                media_type="application/json",
            )

        # 添加安全响应头
        response = await call_next(request)
        
        # 安全响应头
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # 添加速率限制响应头
        remaining = rate_limiter.get_remaining(rate_key)
        response.headers["X-RateLimit-Limit"] = str(rate_limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(rate_limiter.window_seconds)
        
        return response