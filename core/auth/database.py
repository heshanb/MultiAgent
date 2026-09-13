from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, BigInteger, JSON
from sqlalchemy.dialects.mysql import JSON as MySQLJSON
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
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
    created_at = Column(DateTime, default=datetime.now())
    updated_at = Column(DateTime, default=datetime.now(), onupdate=datetime.now())
    last_login = Column(DateTime, nullable=True)


class Conversation(Base):
    """对话表"""
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    thread_id = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.now())
    updated_at = Column(DateTime, default=datetime.now(), onupdate=datetime.now())


class Message(Base):
    """消息表"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON字符串
    created_at = Column(DateTime, default=datetime.now())

class SessionMemory(Base):
    """会话记忆表（用于长期记忆，服务重启后数据不丢失）
    每个 thread_id 只有一条记录，chat_msg 存储整个对话历史的 JSON 数组
    """
    __tablename__ = "session_memories"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    thread_id = Column(BigInteger, primary_key=True, index=True, nullable=False)
    chat_msg = Column(JSON, nullable=True)  # JSON 数组，存储对话历史
    context = Column(JSON, nullable=True)   # JSON 对象，存储元信息（upload_files, output_file 等）
    create_at = Column(DateTime, default=datetime.now())
    update_at = Column(DateTime, default=datetime.now(), onupdate=datetime.now())
class Agent(Base):
    """智能体表"""
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    avatar = Column(String(20), nullable=True, default="🤖")
    system_prompt = Column(Text, nullable=True)
    model = Column(String(50), nullable=True, default="")
    variables = Column(Text, nullable=True)
    knowledge_base = Column(Text, nullable=True)
    opening = Column(Text, nullable=True)
    presets = Column(Text, nullable=True)
    is_published = Column(Boolean, default=False)
    category = Column(String(50), nullable=True)
    publish_version = Column(String(50), nullable=True)
    publish_desc = Column(Text, nullable=True)
    publish_channels = Column(Text, nullable=True)
    deep_think = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now())
    updated_at = Column(DateTime, default=datetime.now(), onupdate=datetime.now())


if USE_SQLITE:
    _JSON = SQLiteJSON
else:
    _JSON = MySQLJSON


class Knowledge(Base):
    """知识库元数据表（knowledges）"""
    __tablename__ = "knowledges"

    id = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    knowledge_name = Column(String(50), nullable=False)
    knowledge_description = Column(String(255), nullable=True)
    user_name = Column(String(20), nullable=False, index=True)
    chunk_retrieve_setting = Column(_JSON, nullable=False, default=dict)
    knowledge_id = Column(_JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.now())
    update_at = Column(DateTime, default=datetime.now(), onupdate=datetime.now())


def init_db():
    """初始化数据库，创建所有表，并安全迁移新增列"""
    try:
        Base.metadata.create_all(bind=engine)
        SessionMemory.__table__.create(bind=engine, checkfirst=True)
        _ensure_agents_table()
        _ensure_knowledges_table()

        if USE_SQLITE:
            try:
                with engine.connect() as conn:
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


def _ensure_agents_table():
    """确保 agents 表存在（兼容已存在的旧数据库）"""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if not inspector.has_table("agents"):
        print("📌 检测到 agents 表不存在，正在创建...")
        Agent.__table__.create(bind=engine, checkfirst=True)
        print("✅ agents 表创建成功")
    else:
        existing_cols = {c['name'] for c in inspector.get_columns("agents")}
        missing = [c for c in Agent.__table__.columns.keys() if c not in existing_cols]
        for col_name in missing:
            col_obj = Agent.__table__.c[col_name]
            col_type = str(col_obj.type)
            try:
                with engine.connect() as conn:
                    conn.execute(text(f"ALTER TABLE agents ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                print(f"📌 agents 表新增列: {col_name} ({col_type})")
            except Exception as e:
                print(f"⚠️ agents 表新增列 {col_name} 失败: {e}")


def _ensure_knowledges_table():
    """确保 knowledges 表存在（兼容已存在的旧数据库）"""
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if not inspector.has_table("knowledges"):
        print("📌 检测到 knowledges 表不存在，正在创建...")
        Knowledge.__table__.create(bind=engine, checkfirst=True)
        print("✅ knowledges 表创建成功")
    else:
        existing_cols = {c['name'] for c in inspector.get_columns("knowledges")}
        missing = [c for c in Knowledge.__table__.columns.keys() if c not in existing_cols]
        for col_name in missing:
            col_obj = Knowledge.__table__.c[col_name]
            col_type = str(col_obj.type)
            try:
                with engine.connect() as conn:
                    conn.execute(text(f"ALTER TABLE knowledges ADD COLUMN {col_name} {col_type}"))
                    conn.commit()
                print(f"📌 knowledges 表新增列: {col_name} ({col_type})")
            except Exception as e:
                print(f"⚠️ knowledges 表新增列 {col_name} 失败: {e}")


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()