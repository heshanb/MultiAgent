import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from logging import getLogger

logger = getLogger(__name__)


class ShortTermMemory:
    """短期记忆 - 存储当前会话的对话历史和上下文信息"""
    
    def __init__(self, state_list: list = None):
        """
        初始化短期记忆
        
        Args:
            state_list: 对话状态列表，包含历史消息
        """
        self.state_list = state_list or []
        self.context = {}  # 当前会话的临时上下文
        self.recent_actions = []  # 最近的操作记录
        self.current_task = None  # 当前正在执行的任务
        self.session_start_time = time.time()
        
    def add_action(self, action_type: str, details: dict = None):
        """
        记录用户操作
        
        Args:
            action_type: 操作类型（如 'file_open', 'code_edit', 'task_start'）
            details: 操作详情
        """
        action = {
            'timestamp': time.time(),
            'type': action_type,
            'details': details or {}
        }
        self.recent_actions.append(action)
        # 只保留最近20个操作，防止内存过大
        if len(self.recent_actions) > 20:
            self.recent_actions = self.recent_actions[-20:]
        
    def set_current_task(self, task: str):
        """
        设置当前任务
        
        Args:
            task: 任务描述
        """
        self.current_task = task
        self.add_action('task_start', {'task': task})
        
    def update_context(self, key: str, value):
        """
        更新上下文信息
        
        Args:
            key: 上下文键
            value: 上下文值
        """
        self.context[key] = value
        
    def get_context(self, key: str, default=None):
        """
        获取上下文信息
        
        Args:
            key: 上下文键
            default: 默认值
            
        Returns:
            上下文值
        """
        return self.context.get(key, default)
    
    def get_context_summary(self) -> str:
        """
        获取短期记忆摘要，用于提示词
        
        Returns:
            格式化的上下文摘要字符串
        """
        summary = ""
        
        if self.current_task:
            summary += f"当前任务: {self.current_task}\n"
        
        if self.recent_actions:
            summary += "最近操作: "
            recent_types = [action['type'] for action in self.recent_actions[-5:]]
            summary += ", ".join(recent_types) + "\n"
        
        if self.context:
            summary += "当前上下文: "
            context_items = [f"{k}: {v}" for k, v in self.context.items()]
            summary += ", ".join(context_items) + "\n"
        
        return summary.strip()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        构建对话历史
        
        Returns:
            格式化的对话历史列表
        """
        history = []
        
        for msg in self.state_list[:-1]:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            msg_type = str(type(msg))
            is_assistant = False
            is_user = False
            
            if isinstance(msg, dict):
                role = msg.get("role", "")
                if role == "assistant" or role == "system":
                    is_assistant = True
                elif role == "user":
                    is_user = True
            else:
                if "AIMessage" in msg_type:
                    is_assistant = True
                elif "HumanMessage" in msg_type:
                    is_user = True
            
            if is_assistant:
                history.append({"role": "assistant", "content": content})
            elif is_user:
                history.append({"role": "user", "content": content})
        
        return history
    
    def get_session_duration(self) -> float:
        """
        获取会话持续时间（秒）
        
        Returns:
            会话持续时间
        """
        return time.time() - self.session_start_time
    
    def clear(self):
        """清空短期记忆"""
        self.context = {}
        self.recent_actions = []
        self.current_task = None
        self.session_start_time = time.time()


class LongTermMemory:
    """长期记忆 - 跨会话持久化存储项目知识和用户偏好"""
    
    def __init__(self, memory_file: Optional[str] = None):
        """
        初始化长期记忆
        
        Args:
            memory_file: 记忆存储文件路径
        """
        self.memory_file = Path(memory_file) if memory_file else None
        self.memory_store = {
            'project_info': {},           # 项目基本信息
            'code_knowledge': {},         # 代码知识（函数、类、依赖关系）
            'user_preferences': {},       # 用户偏好设置
            'issue_history': [],          # 历史问题记录
            'refactor_history': [],       # 重构历史
            'learned_patterns': [],       # 学习到的模式和最佳实践
            'file_analysis_cache': {}     # 文件分析缓存
        }
        
    def load_memory(self):
        """从磁盘加载长期记忆"""
        if self.memory_file and self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.memory_store.update(loaded)
                logger.info(f"成功加载长期记忆: {self.memory_file}")
            except Exception as e:
                logger.warning(f"加载长期记忆失败: {str(e)}")
    
    def save_memory(self):
        """保存长期记忆到磁盘"""
        if self.memory_file:
            try:
                # 确保父目录存在
                self.memory_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.memory_file, 'w', encoding='utf-8') as f:
                    json.dump(self.memory_store, f, ensure_ascii=False, indent=2)
                logger.info(f"成功保存长期记忆: {self.memory_file}")
            except Exception as e:
                logger.warning(f"保存长期记忆失败: {str(e)}")
    
    def add_project_info(self, key: str, value):
        """
        添加项目信息
        
        Args:
            key: 信息键
            value: 信息值
        """
        self.memory_store['project_info'][key] = value
        self.save_memory()
    
    def get_project_info(self, key: str, default=None):
        """
        获取项目信息
        
        Args:
            key: 信息键
            default: 默认值
            
        Returns:
            项目信息值
        """
        return self.memory_store['project_info'].get(key, default)
    
    def add_code_knowledge(self, file_name: str, knowledge: dict):
        """
        添加代码知识（函数、类、依赖等）
        
        Args:
            file_name: 文件名
            knowledge: 代码知识字典
        """
        if file_name not in self.memory_store['code_knowledge']:
            self.memory_store['code_knowledge'][file_name] = {}
        self.memory_store['code_knowledge'][file_name].update(knowledge)
        self.save_memory()
    
    def get_code_knowledge(self, file_name: str) -> dict:
        """
        获取指定文件的代码知识
        
        Args:
            file_name: 文件名
            
        Returns:
            代码知识字典
        """
        return self.memory_store['code_knowledge'].get(file_name, {})
    
    def add_issue(self, issue: dict):
        """
        记录发现的问题
        
        Args:
            issue: 问题字典，包含 'description', 'file', 'severity', 'line' 等字段
        """
        issue['timestamp'] = time.time()
        self.memory_store['issue_history'].append(issue)
        # 只保留最近100个问题
        if len(self.memory_store['issue_history']) > 100:
            self.memory_store['issue_history'] = self.memory_store['issue_history'][-100:]
        self.save_memory()
    
    def get_recent_issues(self, limit: int = 10) -> List[dict]:
        """
        获取最近的问题记录
        
        Args:
            limit: 返回数量限制
            
        Returns:
            问题列表
        """
        return self.memory_store['issue_history'][-limit:]
    
    def add_refactor_record(self, record: dict):
        """
        记录重构操作
        
        Args:
            record: 重构记录字典
        """
        record['timestamp'] = time.time()
        self.memory_store['refactor_history'].append(record)
        self.save_memory()
    
    def add_learned_pattern(self, pattern: dict):
        """
        记录学习到的模式
        
        Args:
            pattern: 模式字典，包含 'name', 'description', 'examples' 等字段
        """
        self.memory_store['learned_patterns'].append(pattern)
        self.save_memory()
    
    def get_learned_patterns(self) -> List[dict]:
        """
        获取所有学习到的模式
        
        Returns:
            模式列表
        """
        return self.memory_store['learned_patterns']
    
    def update_file_analysis(self, file_name: str, analysis: dict):
        """
        更新文件分析缓存
        
        Args:
            file_name: 文件名
            analysis: 分析结果字典
        """
        self.memory_store['file_analysis_cache'][file_name] = {
            'timestamp': time.time(),
            'analysis': analysis
        }
        self.save_memory()
    
    def get_file_analysis(self, file_name: str) -> Optional[dict]:
        """
        获取文件分析缓存
        
        Args:
            file_name: 文件名
            
        Returns:
            分析结果字典，如果不存在返回None
        """
        cache = self.memory_store['file_analysis_cache'].get(file_name)
        if cache:
            # 检查缓存是否过期（超过7天）
            if time.time() - cache['timestamp'] < 7 * 24 * 60 * 60:
                return cache['analysis']
            else:
                # 清除过期缓存
                del self.memory_store['file_analysis_cache'][file_name]
                self.save_memory()
        return None
    
    def set_user_preference(self, key: str, value):
        """
        设置用户偏好
        
        Args:
            key: 偏好键
            value: 偏好值
        """
        self.memory_store['user_preferences'][key] = value
        self.save_memory()
    
    def get_user_preference(self, key: str, default=None):
        """
        获取用户偏好
        
        Args:
            key: 偏好键
            default: 默认值
            
        Returns:
            偏好值
        """
        return self.memory_store['user_preferences'].get(key, default)
    
    def get_memory_summary(self) -> str:
        """
        获取长期记忆摘要，用于提示词
        
        Returns:
            格式化的记忆摘要字符串
        """
        summary = ""
        
        # 项目信息
        if self.memory_store['project_info']:
            summary += "**项目记忆：**\n"
            for key, value in self.memory_store['project_info'].items():
                summary += f"- {key}: {value}\n"
        
        # 历史问题摘要
        if self.memory_store['issue_history']:
            recent_issues = self.memory_store['issue_history'][-5:]
            summary += f"\n**最近发现的问题（{len(recent_issues)}个）：**\n"
            for issue in recent_issues:
                desc = issue.get('description', '')[:50]
                file = issue.get('file', '')
                summary += f"- {desc}... ({file})\n"
        
        # 学习到的模式
        if self.memory_store['learned_patterns']:
            summary += f"\n**已学习的模式（{len(self.memory_store['learned_patterns'])}个）：**\n"
            for pattern in self.memory_store['learned_patterns'][:3]:
                summary += f"- {pattern.get('name', '')}\n"
        
        return summary.strip()
    
    def clear(self):
        """清空所有长期记忆"""
        self.memory_store = {
            'project_info': {},
            'code_knowledge': {},
            'user_preferences': {},
            'issue_history': [],
            'refactor_history': [],
            'learned_patterns': [],
            'file_analysis_cache': {}
        }
        self.save_memory()


class MemoryManager:
    """记忆管理器 - 统一管理短期记忆和长期记忆"""
    
    def __init__(self, state_list: list = None, memory_file: str = None):
        """
        初始化记忆管理器
        
        Args:
            state_list: 对话状态列表
            memory_file: 长期记忆存储文件路径
        """
        self.short_term = ShortTermMemory(state_list)
        self.long_term = LongTermMemory(memory_file)
        self.long_term.load_memory()
    
    def get_combined_summary(self) -> str:
        """
        获取短期记忆和长期记忆的组合摘要
        
        Returns:
            格式化的组合摘要字符串
        """
        short_summary = self.short_term.get_context_summary()
        long_summary = self.long_term.get_memory_summary()
        
        summary = ""
        if short_summary:
            summary += f"**当前上下文：**\n{short_summary}\n\n"
        if long_summary:
            summary += long_summary
        
        return summary.strip()
    
    def save(self):
        """保存长期记忆"""
        self.long_term.save_memory()
    
    def close(self):
        """关闭记忆管理器，确保保存"""
        self.save()


# 全局记忆管理器实例（可选单例模式）
_global_memory_manager = None


def get_memory_manager(state_list: list = None, memory_file: str = None) -> MemoryManager:
    """
    获取记忆管理器实例
    
    Args:
        state_list: 对话状态列表
        memory_file: 长期记忆存储文件路径
        
    Returns:
        MemoryManager实例
    """
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager(state_list, memory_file)
    
    return _global_memory_manager