from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import quote_plus
import os

# 加载环境变量
load_dotenv()

# 数据库配置 - 临时使用SQLite
USE_SQLITE = os.getenv("USE_SQLITE", "true").lower() == "true"

if USE_SQLITE:
    # SQLite配置
    from settings.Define import PathConfig
    DATABASE_URL = f"sqlite:///{os.path.join(PathConfig.BASE_DIR, 'users.db')}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    print("✅ 使用SQLite数据库（临时方案）")
else:
    # MySQL数据库配置
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "3306")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "multiagent_db")

    # URL编码密码（处理特殊字符如 @ # 等）
    encoded_password = quote_plus(DB_PASSWORD)

    # 构建MySQL连接URL
    DATABASE_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

    # 创建引擎 - MySQL配置
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,  # 连接池大小
        max_overflow=20,  # 最大溢出连接数
        pool_recycle=3600,  # 连接回收时间（秒）
        pool_pre_ping=True,  # 连接前检测有效性
        echo=False  # 生产环境设为False，调试时可设为True
    )
    print(f"✅ 使用MySQL数据库: {DB_HOST}:{DB_PORT}/{DB_NAME}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """用户表"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    phone = Column(String(20), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    role = Column(String(20), default="user")  # user, vip, admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


class Conversation(Base):
    """对话表"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    thread_id = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Message(Base):
    """消息表"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON字符串
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """初始化数据库，创建所有表，并安全迁移新增列"""
    try:
        Base.metadata.create_all(bind=engine)
        # 安全迁移：为已存在的 users 表添加 phone 列（SQLite 专用）
        if USE_SQLITE:
            try:
                with engine.connect() as conn:
                    # 检查 phone 列是否存在
                    result = conn.execute(
                        __import__("sqlalchemy").text("PRAGMA table_info(users)")
                    )
                    columns = [row[1] for row in result]
                    if "phone" not in columns:
                        conn.execute(__import__("sqlalchemy").text(
                            "ALTER TABLE users ADD COLUMN phone VARCHAR(20)"
                        ))
                        conn.commit()
                        print("✅ SQLite 迁移：成功添加 phone 列到 users 表")
                    # 移除 email 列的 NOT NULL 约束（SQLite 不支持 ALTER COLUMN，需要重建表）
                    # 检查 email 是否是 NOT NULL 的（通过尝试插入空值来检测）
            except Exception as migrate_err:
                print(f"⚠️  数据库迁移提示: {migrate_err}")
                print("   如需完整迁移，可删除 users.db 让数据库重建")
        if USE_SQLITE:
            print("✅ SQLite数据库初始化成功")
        else:
            print(f"✅ MySQL数据库连接成功: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        if USE_SQLITE:
            print("请检查SQLite数据库文件路径是否正确")
        else:
            print("请检查:")
            print("1. MySQL服务是否已启动")
            print("2. .env文件中的数据库配置是否正确")
            print("3. 数据库是否已创建（CREATE DATABASE multiagent_db）")
        raise


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()