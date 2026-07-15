from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
import random
from typing import Optional

from .database import get_db, User
from .auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_active_user,
    require_role,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from core.security import get_audit_logger

router = APIRouter(prefix="/api/auth", tags=["认证"])


class UserCreate(BaseModel):
    """用户注册请求模型"""
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None
    password: str
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """用户响应模型"""
    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """登录响应模型"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


@router.post("/register", response_model=TokenResponse)
async def register(
    request: Request,
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """用户注册"""
    # 检查用户名是否已存在
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 检查邮箱是否已存在（如果提供了邮箱）
    if user_data.email and db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="邮箱已被注册")
    # 检查手机号是否已存在（如果提供了手机号）
    if user_data.phone and db.query(User).filter(User.phone == user_data.phone).first():
        raise HTTPException(status_code=400, detail="手机号已被注册")
    
    # 创建新用户
    new_user = User(
        username=user_data.username,
        email=user_data.email if user_data.email else (user_data.username + "@local.dev"),
        phone=user_data.phone,
        hashed_password=get_password_hash(user_data.password),
        full_name=user_data.full_name,
        role="user",
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # 生成token
    access_token = create_access_token(
        data={"sub": new_user.username, "role": new_user.role}
    )
    
    # 审计日志
    client_ip = request.client.host if request.client else "unknown"
    audit_logger = get_audit_logger()
    audit_logger.log_register(
        username=new_user.username,
        email=new_user.email,
        ip_address=client_ip,
    )
    
    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(new_user),
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """用户登录"""
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        # 审计日志 - 登录失败
        client_ip = request.client.host if request.client else "unknown"
        audit_logger = get_audit_logger()
        audit_logger.log_login(
            username=form_data.username,
            ip_address=client_ip,
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="用户已被禁用")
    
    # 更新最后登录时间
    user.last_login = datetime.utcnow()
    db.commit()
    
    # 生成token
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}
    )
    
    # 审计日志 - 登录成功
    client_ip = request.client.host if request.client else "unknown"
    audit_logger = get_audit_logger()
    audit_logger.log_login(
        username=user.username,
        ip_address=client_ip,
        success=True,
    )
    
    return TokenResponse(
        access_token=access_token,
        user=UserResponse.model_validate(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """获取当前用户信息"""
    return current_user


@router.put("/me", response_model=UserResponse)
async def update_user_info(
    full_name: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """更新用户信息"""
    if full_name is not None:
        current_user.full_name = full_name
    current_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user)
    return current_user


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    """获取用户列表（仅管理员）"""
    users = db.query(User).all()
    return users


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    role: str,
    request: Request,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    """更新用户角色（仅管理员）"""
    if role not in ["user", "vip", "admin"]:
        raise HTTPException(status_code=400, detail="无效的角色")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    
    old_role = user.role
    user.role = role
    db.commit()
    
    # 审计日志
    client_ip = request.client.host if request.client else "unknown"
    audit_logger = get_audit_logger()
    audit_logger.log_role_change(
        admin_user=current_user.username,
        target_user=user.username,
        new_role=role,
        ip_address=client_ip,
    )
    
    return {"message": "角色更新成功", "old_role": old_role, "new_role": role}

# ================ 验证码和密码找回 ================
verification_codes = {}

class SendCodeRequest(BaseModel):
    username: str
    phone: str

class ResetPasswordRequest(BaseModel):
    username: str
    phone: str
    code: str
    new_password: str
    confirm_password: str

@router.post("/send-reset-code")
async def send_reset_code(request: Request, code_data: SendCodeRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == code_data.username).first()
    if not user: raise HTTPException(status_code=400, detail="用户名不存在")
    if not user.phone or user.phone != code_data.phone: raise HTTPException(status_code=400, detail="手机号与注册时不匹配")
    code = str(random.randint(100000, 999999))
    verification_codes[code_data.phone] = (code, datetime.utcnow() + timedelta(minutes=5))
    print("\n" + "="*50 + "\n【密码重置验证码】用户: " + code_data.username + ", 手机号: " + code_data.phone + "\n验证码: " + code + " (5分钟内有效)\n" + "="*50 + "\n")
    return {"message": "验证码已发送（演示环境：请查看服务器控制台）", "code": code}

@router.post("/reset-password")
async def reset_password(request: Request, reset_data: ResetPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == reset_data.username).first()
    if not user: raise HTTPException(status_code=400, detail="用户名不存在")
    if not user.phone or user.phone != reset_data.phone: raise HTTPException(status_code=400, detail="手机号不匹配")
    stored = verification_codes.get(reset_data.phone)
    if not stored: raise HTTPException(status_code=400, detail="请先获取验证码")
    code, expire_time = stored
    if datetime.utcnow() > expire_time: verification_codes.pop(reset_data.phone, None); raise HTTPException(status_code=400, detail="验证码已过期")
    if code != reset_data.code: raise HTTPException(status_code=400, detail="验证码错误")
    if reset_data.new_password != reset_data.confirm_password: raise HTTPException(status_code=400, detail="两次输入的密码不一致")
    if len(reset_data.new_password) < 8: raise HTTPException(status_code=400, detail="密码长度至少8位")
    user.hashed_password = get_password_hash(reset_data.new_password)
    user.updated_at = datetime.utcnow()
    db.commit()
    verification_codes.pop(reset_data.phone, None)
    return {"message": "密码重置成功，请使用新密码登录"}