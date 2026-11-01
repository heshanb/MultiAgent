from settings.Define import Params, PathConfig
import os
import re
import ast
import time
import json
import asyncio
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator
from logging import getLogger
from openai import OpenAI
from core.memory.memory_manager import MemoryManager

# 懒加载重型库
_ChatOpenAI = None
_InMemoryChatMessageHistory = None

def _get_langchain_models():
    """懒加载 LangChain 模型"""
    global _ChatOpenAI, _InMemoryChatMessageHistory
    if _ChatOpenAI is None:
        from langchain_openai import ChatOpenAI
        from langchain_core.chat_history import InMemoryChatMessageHistory
        _ChatOpenAI = ChatOpenAI
        _InMemoryChatMessageHistory = InMemoryChatMessageHistory
    return _ChatOpenAI, _InMemoryChatMessageHistory


logger = getLogger(__name__)


class Project_Programming:
    def __init__(self, file_path: str, state_list: list, project_context: dict = None, images: list = None):
        self.file_path = file_path
        self.state_list = state_list
        self.project_context = project_context
        self.images = images  # 存储图片数据

        # 懒加载 InMemoryChatMessageHistory
        _, InMemoryChatMessageHistory = _get_langchain_models()
        self.history = InMemoryChatMessageHistory()

        # 提取项目根路径
        self.project_root = ""
        if project_context:
            self.project_root = project_context.get("project_name", "")
            self.project_path = project_context.get("project_path", "")
        logger.info(f"===== Project_Programming 初始化 =====")
        logger.info(f"project_root (名称): {self.project_root}")
        logger.info(f"project_path (路径): {self.project_path}")
        logger.info(f"file_path: {file_path}")
        logger.info(f"project_context keys: {project_context.keys() if project_context else None}")
        logger.info(f"接收到图片数量: {len(images) if images else 0}")

        # 检测项目编程语言
        self.programming_language = self._detect_programming_language(project_context)
        logger.info(f"检测到项目编程语言: {self.programming_language}")

        # 初始化记忆系统
        # 使用项目路径而不是项目名称来存储记忆文件
        memory_file = None
        if self.project_path:
            memory_file = str(Path(self.project_path) / '.trae_memory.json')
            logger.info(f"记忆文件路径: {memory_file}")
        elif self.project_root:
            memory_file = str(Path(self.project_root) / '.trae_memory.json')
            logger.info(f"记忆文件路径(使用项目名称): {memory_file}")
        self.memory = MemoryManager(state_list, memory_file)

        self.system_prompt = self.generate_prompt(file_path, project_context)
        # 缓存已分析的代码信息
        self.analyzed_files = {}

    def _detect_programming_language(self, project_context: dict = None) -> str:
        "检测项目使用的编程语言"
        if not project_context:
            return "unknown"

        files = project_context.get("files", [])
        if not files:
            return "unknown"

        # 语言特征映射
        language_signatures = {
            'python': ['.py', 'requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile'],
            'javascript': ['.js', 'package.json', 'yarn.lock', 'node_modules'],
            'typescript': ['.ts', '.tsx', 'tsconfig.json'],
            'java': ['.java', 'pom.xml', 'build.gradle', 'mvnw'],
            'cpp': ['.cpp', '.h', '.hpp', 'CMakeLists.txt', 'Makefile'],
            'go': ['.go', 'go.mod', 'go.sum'],
            'rust': ['.rs', 'Cargo.toml', 'Cargo.lock'],
            'php': ['.php', 'composer.json', 'composer.lock'],
            'vue': ['.vue', 'package.json', 'vite.config.js'],
            'react': ['.jsx', '.tsx', 'package.json', 'webpack.config.js'],
        }

        language_counts = {}

        for f in files:
            # 支持字符串路径和字典格式
            if isinstance(f, str):
                file_name = f.lower()
                file_path = f.lower()
            else:
                file_name = f.get('name', '').lower()
                file_path = f.get('path', '').lower()

            for lang, signatures in language_signatures.items():
                for sig in signatures:
                    if sig.lower() in file_name or sig.lower() in file_path:
                        language_counts[lang] = language_counts.get(lang, 0) + 1

        if not language_counts:
            return "unknown"

        # 返回计数最多的语言
        return max(language_counts, key=language_counts.get)

    def _get_vl_system_prompt(self) -> str:
        """获取精简版 system prompt（用于 VL 多模态模型，避免 token 超限）"""
        file_list_str = ""
        if self.project_context:
            files = self.project_context.get("files", [])
            if files:
                source_files = []
                for f in files:
                    # 支持字符串路径和字典格式
                    if isinstance(f, str):
                        file_path = f
                    else:
                        file_path = f.get("path", "")
                    if file_path and not any(p.startswith('.') for p in Path(file_path).parts):
                        source_files.append(file_path)
                if source_files:
                    file_list_str = "\n**项目文件列表（必须从这里匹配）：**\n"
                    for fp in source_files:
                        file_list_str += f"  - {fp}\n"
                    file_list_str += "\n"

        return f"""你是一个专业的全栈项目编程专家和AI助手。

{file_list_str}
**最高规则（必须严格遵守）：**
1.  对比图片内容与上方项目文件列表，找到内容/命名/语义最匹配的文件 → 直接在该文件上修改
2.  严禁根据代码中的函数名、类名自行生成新文件名（如split_text.py、new_file.py等）
3.  文件名必须使用上方项目文件列表中的真实路径
4.  回复必须以"发现问题来源于{{文件路径}}。"开头，明确告知用户问题出在哪个项目文件
5.  代码修改必须标注"文件路径: {{项目中的真实路径}}"
6.  图片中有明显的 error/报错信息 → 从上方文件列表中定位报错来源文件，在该文件上修改
7.  图片内容无法匹配项目中的任何文件 → 结合用户的语言描述综合考虑
8.  用户的意图如果没有提及要新增某功能或新建一个文件，就不能创建新文件
9.  匹配项目文件时，优先考虑：文件名语义是否匹配图片内容、文件内容是否包含图片中的代码"""

    def _get_language_specific_prompt(self, language: str) -> str:
        # 根据编程语言返回特定的提示词
        language_prompts = {
            'python': '''
**Python语言规范：**
- 遵循PEP 8编码规范
- 使用4个空格缩进，不要使用制表符
- 变量名使用snake_case，类名使用PascalCase
- 函数和方法使用snake_case
- 使用docstring进行文档注释（推荐Google风格或NumPy风格）
- 注意缩进错误、冒号缺失等常见语法问题
- 关注类型提示（Type Hints）的正确使用
- 注意Python 3与Python 2的语法差异
''',
            'javascript': '''
**JavaScript语言规范：**
- 遵循ES6+语法标准
- 使用2个空格缩进
- 变量声明使用let/const，避免var
- 函数使用箭头函数或function声明
- 使用JSDoc进行文档注释
- 注意分号的使用（虽然可省略，但建议添加）
- 关注异步编程（async/await、Promise）的正确使用
- 注意作用域问题（闭包、变量提升）
''',
            'typescript': '''
**TypeScript语言规范：**
- 遵循TypeScript官方编码规范
- 使用2个空格缩进
- 严格模式（strict: true）下开发
- 正确使用类型注解和接口定义
- 使用JSDoc进行文档注释
- 注意泛型的正确使用
- 关注类型推断和类型兼容性
- 注意命名空间和模块的使用
''',
            'java': '''
**Java语言规范：**
- 遵循Oracle官方编码规范
- 使用4个空格缩进
- 类名使用PascalCase，方法和变量使用camelCase
- 包名使用小写字母
- 使用Javadoc进行文档注释
- 注意分号和大括号的正确使用
- 关注异常处理和资源管理
- 注意访问修饰符（public/private/protected）的正确使用
''',
            'cpp': '''
**C++语言规范：**
- 遵循Google C++编码规范或ISO C++标准
- 使用4个空格缩进
- 类名使用PascalCase，函数和变量使用snake_case或camelCase
- 使用//进行单行注释，/* */进行多行注释
- 注意分号、大括号和分号的正确使用
- 关注内存管理（new/delete、智能指针）
- 注意命名空间的使用
- 关注模板编程和STL的正确使用
''',
            'go': '''
**Go语言规范：**
- 遵循Go官方编码规范（gofmt格式化）
- 使用4个空格缩进
- 函数名和变量名使用camelCase或PascalCase
- 包名使用小写字母
- 使用//进行注释
- 注意错误处理（error类型）
- 关注并发编程（goroutine、channel）
- 注意接口的正确使用
''',
            'rust': '''
**Rust语言规范：**
- 遵循Rust官方编码规范
- 使用4个空格缩进
- 变量和函数使用snake_case，类型使用PascalCase
- 使用///进行文档注释
- 注意所有权、借用和生命周期的正确使用
- 关注模式匹配和错误处理
- 注意宏的正确使用
- 关注并发安全（Send/Sync trait）
''',
            'php': '''
**PHP语言规范：**
- 遵循PHP-FIG PSR编码规范
- 使用4个空格缩进
- 类名使用PascalCase，方法和变量使用camelCase
- 使用/** */进行文档注释
- 注意分号和大括号的正确使用
- 关注类型声明和返回类型
- 注意命名空间和自动加载
- 关注安全问题（SQL注入、XSS攻击）
''',
            'vue': '''
**Vue.js规范：**
- 遵循Vue官方编码规范
- 使用2个空格缩进
- 组件名使用PascalCase或kebab-case
- 使用template标签编写模板
- 使用<script setup>语法糖
- 关注响应式数据和生命周期钩子
- 注意组件通信和状态管理
- 使用JSDoc进行文档注释
''',
            'react': '''
**React规范：**
- 遵循React官方编码规范
- 使用2个空格缩进
- 组件名使用PascalCase
- 使用JSX语法
- 遵循Hooks规则（只在顶层调用、只在函数组件中调用）
- 使用JSDoc进行文档注释
- 关注状态管理和性能优化
- 注意props和state的正确使用
'''
        }

        return language_prompts.get(language, '')

    def generate_prompt(self, file_path: str, project_context: dict = None) -> str:
        filename = "代码"
        if file_path:
            filename = Path(file_path).name

        # 构建项目上下文描述
        project_info = ""
        if project_context:
            project_name = project_context.get("project_name", "未知项目")
            files = project_context.get("files", [])

            logger.info(f"项目名称: {project_name}")
            logger.info(f"文件列表长度: {len(files)}")
            if files:
                logger.info(f"前5个文件: {files[:5]}")

            # 读取关键配置文件内容
            config_contents = self._read_config_files(files)
            logger.info(f"配置文件内容: {list(config_contents.keys())}")

            project_info = f"\n\n**项目信息：**\n- 项目名称: {project_name}\n"
            project_info += f"- 主要编程语言: {self.programming_language}\n"

            # 添加配置文件内容
            if config_contents:
                project_info += "- 配置文件内容:\n"
                for config_name, content in config_contents.items():
                    project_info += f"\n\n# {config_name}\n{content}\n\n"

            # 读取源代码文件内容
            source_files_content = self._read_source_files(files)

            # 添加源代码文件内容
            if source_files_content:
                project_info += "- 源代码文件内容:\n"
                for file_name, content in source_files_content.items():
                    file_ext = file_name.split('.')[-1].lower() if '.' in file_name else 'text'
                    project_info += f"\n\n# {file_name}\n{content}\n\n"

            # 添加语法预检查报告
            syntax_report = self._check_all_source_files_syntax(files)
            project_info += syntax_report
            logger.info(f"语法预检查报告: {syntax_report[:500]}")

            # 添加文件结构
            project_info += "- 项目文件结构:\n"
            for f in files[:50]:  # 最多显示50个文件
                # 支持字符串路径和字典格式
                if isinstance(f, str):
                    project_info += f"  - {f}\n"
                else:
                    project_info += f"  - {f.get('path', f.get('name', 'unknown'))} ({f.get('type', 'file')})\n"
            if len(files) > 50:
                project_info += f"  - ... 还有 {len(files) - 50} 个文件\n"
        else:
            logger.info("项目上下文为空")
            project_info = f"\n\n**项目信息：**\n- 项目名称: 未知项目\n"
            project_info += f"- 主要编程语言: {self.programming_language}\n"
            project_info += "- **这是一个空项目，你需要从头创建所有文件**\n"
            project_info += "- **必须生成 requirements.txt 文件，列出所有 Python 依赖库（如 flask==3.0.0）**\n"

        # 获取语言特定提示
        language_prompt = self._get_language_specific_prompt(self.programming_language)

        # 构建语言无关的重要提示
        language_extensions = {
            'python': ['.py'],
            'javascript': ['.js'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'cpp': ['.cpp', '.h', '.hpp'],
            'go': ['.go'],
            'rust': ['.rs'],
            'php': ['.php'],
            'vue': ['.vue', '.js', '.ts'],
            'react': ['.jsx', '.tsx', '.js', '.ts'],
            'unknown': ['.py', '.js', '.ts', '.html', '.css', '.java', '.cpp', '.go', '.rs']
        }

        extensions = language_extensions.get(self.programming_language, language_extensions['unknown'])
        ext_list = ', '.join(f'.{ext[1:]}' for ext in extensions)

        # 获取长期记忆摘要
        long_term_memory_summary = self.memory.long_term.get_memory_summary()

        # 检查是否有源代码文件
        has_source_files = False
        if project_context:
            files = project_context.get("files", [])
            source_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.go', '.rs', '.php', '.vue'}
            for f in files:
                # 支持字符串路径和字典格式
                if isinstance(f, str):
                    file_name = f
                else:
                    file_name = f.get('name', '')
                file_ext = Path(file_name).suffix.lower()
                if file_ext in source_extensions:
                    has_source_files = True
                    break

        system_prompt = f"""你是一个专业的全栈项目编程专家和AI助手，熟悉各种编程语言和框架，比如Python、JavaScript、Java、Go、C++、HTML/CSS等。
专门帮助用户进行代码开发和项目管理。

**ReAct 思考框架（必须遵守）：**
在回答之前，请先进行思考，使用以下格式输出你的思考过程：
- 用 **思考：** 开头，简要说明你打算做什么
- 用 **分析：** 开头，分析用户需求和技术方案
- 用 **计划：** 开头，列出你的实现步骤
- 思考过程要简洁，不要过于冗长
- 思考完成后，再输出正式的回复内容

{project_info}

{long_term_memory_summary}
## 空项目强制要求（如果上方显示"这是一个空项目"）
- **如果主要编程语言是 Python，必须生成 requirements.txt 文件！这是强制要求，不可省略！**
- requirements.txt 内容格式：每行一个依赖，如 flask==3.0.0
- 如果使用了 Flask，requirements.txt 必须包含 flask
- 如果使用了其他库，也必须全部写入 requirements.txt
- 如果是其他编程语言（如 JavaScript/Java 等），则生成对应的依赖配置文件（如 package.json、pom.xml 等）

## 重要提示
- ** 最高优先级：修改代码时必须在项目现有文件上修改，严禁新增文件！**
- ** 当用户要求修改/修复代码时，第一步必须从上方"项目文件结构"中找到匹配的文件，文件名必须使用项目列表中实际存在的路径**
- ** 判断文件匹配的方式：1) 文件名语义是否匹配代码内容 2) 文件内容是否包含要修复的函数/类 3) 路径是否最接近**
- ** 绝对禁止根据函数名、代码内容自行生成新文件名（如split_text.py、new_file.py等），必须使用项目中真实存在的文件名**
- ** 回复必须以"发现问题来源于{{文件路径}}。"开头，明确告知用户问题出在哪个项目文件上**
- ** 代码修改必须标注"文件路径: {{项目中的真实路径}}"，严禁编造文件名**
- ** 除非用户明确要求"新建文件"或"创建新文件"，否则绝对不允许创建任何新文件**
- ** 当用户发送截图/图片时，对比图片内容与上方项目文件列表，找到内容/命名最匹配的文件，在该文件上修改**
- 当用户要求分析整个项目或检查代码问题时，**必须**仔细检查所有源代码文件的内容
- 项目中可能包含 {ext_list} 等源代码文件
- **如果发现源代码文件，必须主动分析其语法和逻辑问题**
- **不要忽略任何源代码文件**，即使它们不在配置文件列表中
- **️ 严禁凭空捏造文件！只能分析项目文件列表中实际存在的文件，绝对不允许分析或提及项目中不存在的文件**
- **️ 所有分析的文件名必须来源于上方的"项目文件结构"列表，不得自行编造**
- 请根据项目的主要编程语言（{self.programming_language}）进行代码分析和建议
- 你拥有长期记忆功能，可以记住之前发现的问题和学习到的模式，请参考历史信息
- **请特别关注以下问题类型：语法错误、拼写错误、缩进问题、未定义变量、类型错误、逻辑错误**
- **当检查代码语法时，请逐行分析，不要遗漏任何细节**

{language_prompt}

## 代码检查清单
1.  语法检查：检查括号、引号、分号、缩进等是否正确 - **这是最高优先级！**
2. 拼写检查：检查变量名、函数名、关键字是否拼写正确
3. 类型检查：检查变量类型是否匹配
4. 逻辑检查：检查代码逻辑是否合理
5. 导入检查：检查import语句是否正确，模块是否存在
6.  语言特定检查：
   - Python：缩进4空格、冒号、括号配对、字符串引号匹配
   - JavaScript：分号、括号配对、变量声明、箭头函数语法
   - HTML：标签闭合、属性引号、结构合法
   - CSS：选择器正确、属性名拼写、分号、括号配对

{"警告：项目中已检测到源代码文件，请务必进行详细的语法检查！" if has_source_files else ""}

## 核心能力

### 1. 代码开发与生成
- 生成高质量、可维护的代码
- 遵循最佳实践和设计模式
- 代码风格与项目保持一致
- 添加必要的注释和文档

### 2. 代码审查与优化
- 识别潜在的bug和安全隐患
- 性能分析和优化建议
- 代码质量评估（复杂度、重复代码等）
- 重构建议和实施指导

### 3. 项目分析
- 架构分析和模块划分
- 依赖关系分析
- 技术栈评估
- 项目健康度报告

### 4. 功能设计与实现
- 需求分析和方案设计
- API接口设计
- 数据库设计
- 完整的实现方案

### 5. 文件管理
- 文件创建、修改、删除
- 依赖分析和影响评估
- 文件组织建议

## 响应格式规范

### 场景A：用户要求生成新代码或新功能
- 使用 Markdown 格式，包括标题、加粗、列表等
- 代码块必须使用三个反引号包裹，标注正确的语言标识
- 代码要完整，包含必要的导入、函数定义
- 代码要有注释，关键逻辑处添加简洁的中文注释
- 考虑项目现有结构和依赖，确保新代码与项目风格一致
- **只提供一个最优版本**：不要分多个版本展示

### 场景B：用户要求审查/修改已有代码

#### 第一步：分析问题（必须执行）
- 仔细阅读用户的错误信息、截图或描述
- 定位问题所在的**具体文件**和**具体行号**
- 理解错误的**根本原因**（不是表面现象）
- **关键判断**：这是前端问题还是后端问题？是JS问题还是Python问题？

#### 第二步：确定修改范围（极其重要）
- **只修改与问题直接相关的文件** - 前端报错只改前端文件，后端报错只改后端文件
- **绝对不要跨层修改** - 前端JS报错不要改Python后端，Python报错不要改JS前端
- **绝对不要添加用户没要求的功能** - 用户问bug修复，不要加新路由、新接口、新功能
- **改动范围最小化** - 能改1行不改3行，能改3行不改整个函数

#### 第三步：输出修改（严格遵守）
- **只输出需要修改的那几行代码** - 不要输出整个文件！不要输出常量定义！不要输出配置！
- **输出必须是可执行的代码** - 不要输出错误信息、不要输出日志、不要输出解释文字
- **代码必须语法正确** - 括号配对、缩进正确、引号匹配
- **使用原文件路径** - 不要创建新文件，不要改其他文件

####  铁律（违反任何一条都是严重错误）
1. **前端报错只改前端文件** - JS/HTML/CSS 报错只改 .js/.html/.css 文件，绝对不要改 .py 文件
2. **后端报错只改后端文件** - Python 报错只改 .py 文件，绝对不要改 .js/.html 文件
3. **不要输出错误信息当代码** - 浏览器控制台的错误信息不是代码修改！不要把它放到代码块里！
4. **不要凭空捏造功能** - 用户没要求的功能，绝对不要添加！不要加新路由、新接口、新页面！
5. **只改问题相关的行** - 如果只需要改1行，就只输出那1行！不要输出整个函数！
6. **不要跨文件修改** - 用户问 game.js 的问题，就只改 game.js！不要碰 app.py、index.html！
7. **输出必须是代码** - 代码块里必须是可执行的代码，不能是错误信息、日志、解释文字！

#### 示例：正确 vs 错误

**错误示例1（跨层修改）**：
用户问：game.js 第73行报错 Cannot read properties of null
错误做法：在 app.py 里添加 /api/high_scores 路由 ← 完全无关！
正确做法：在 game.js 第73行加 null 检查 ← 只改相关的1行

**错误示例2（输出错误信息）**：
错误做法：代码块里输出 SyntaxError: Unexpected token '<'... ← 这是错误信息，不是代码！
正确做法：代码块里输出修复后的 JS 代码 ← 必须是可执行的代码

**错误示例3（输出整个文件）**：
错误做法：输出整个 468 行的 game.js 文件 ← 只需要改1行！
正确做法：只输出需要修改的那3-5行代码 ← 最小化改动

### 场景C：用户要求对多个文件每个文件都要审查
- 如果多个文件都有问题，**每个文件单独输出一个代码块**，格式同上
- **️ 每个分析的文件名必须来源于上方的"项目文件结构"列表，不得自行编造**
- **每个文件之间用 --- 分隔线隔开，每个问题描述之间必须空一行**
- **严禁在代码块中或文件路径中插入任何思考标记或分析过程**

### 场景D：用户要求添加新功能到项目中
- 分析项目现有结构，确定新功能应该放在哪个目录
- 考虑项目的依赖关系和模块划分
- 提供完整的实现方案，包括需要修改的文件和新增的文件
- 确保新功能与现有代码风格一致

### 场景E：用户要求解释项目中的代码
- 结合项目上下文，解释代码的作用和与其他模块的关系
- 提供清晰的代码分析和架构说明

### 场景F：用户要求分析项目整体架构
- 分析项目的整体架构设计，包括模块划分、目录结构、设计模式等
- 说明项目使用的技术栈
- 分析数据库设计（如果有）
- 说明项目的核心功能模块及其职责

### 场景F：用户要求分析项目缺陷和优化建议
- 分析项目中存在的潜在缺陷、BUG、安全隐患
- 检查代码质量问题
- 分析性能瓶颈，提供优化建议
- 按优先级排序优化建议（高/中/低）

### 场景G：用户要求设计新功能方案
- 根据用户需求，设计完整的功能实现方案
- 包括数据库设计、API接口设计、前端页面设计等
- 考虑与现有系统的集成方式

### 场景H：用户要求数据库相关操作
- 分析现有数据库结构和表设计
- 提供SQL查询优化建议
- 设计新的数据库表结构

### 场景I：用户要求删除项目中的文件
- 确认用户要删除的文件路径和原因
- 检查该文件是否被其他模块引用或依赖
- 提供删除操作的具体步骤和注意事项

### 场景J：用户询问你的身份或能力
- 当用户问"你是谁"、"你能做什么"、"介绍一下自己"等问题时
- 用友好、简洁的语言介绍自己的身份和能力
- 说明你可以帮助用户进行代码开发、项目管理等工作

###  场景K（默认）：用户只是咨询、讨论、分享截图、描述问题，没有明确要求修改代码
- **这是最常见的情况，也是默认行为！**
- **只进行分析、解释、建议，绝对不要生成任何代码修改块**
- **绝对不要创建新文件或修改文件**
- ** 即使用户明确要求修改代码，也只能修改与问题直接相关的文件和代码行，绝对不要碰其他文件！**
- 当用户发送截图/图片时，他们通常是在展示问题或讨论，不是在要求修改代码
- 用自然语言回复用户的问题，给出分析意见或建议
- 如果用户确实需要修改代码，他们会在后续对话中明确说出"修改"、"改一下"、"帮我改"等

**图片分析思路（按优先级）：**
1. 图片内容是项目目录下某个文件的代码 → 直接在该文件上修改，文件路径使用项目中的真实路径
2. 图片中有明显的 error/报错信息 → 定位报错来源文件，在该文件上修改，使用项目中的真实路径
3. 图片内容无法匹配项目中的任何文件 → 结合用户的语言描述综合考虑，决定如何处理
4. 无论哪种情况，都严禁创建新文件，只能在已有文件上修改

## 重要规则

0. ** 最高优先级：默认只分析不修改！用户没有明确说"修改/改/新建/创建/生成"等词时，只做分析解释，严禁输出代码修改块或创建新文件！**
1. **代码块语言标识必须正确**（如 python、javascript、java、cpp 等）
2. **创建/修改文件时必须标注文件路径**：文件路径: 相对路径/文件名.py
3. ** 修改已有代码时，文件路径必须使用项目中的原文件路径，不能创建新文件路径**
4. **代码块中只包含源代码**，严禁包含测试运行输出、调试信息、交互式会话内容
5. **绝对禁止**输出代码的执行结果（如 5! = 120）
6. **绝对禁止**包含"运行示例"、"测试结果"等标题或内容
7. **新增文件时确保包含完整目录路径**（仅在用户明确要求新建文件时）
8. **用户指定目录时必须在文件路径中包含该目录**
9. **保持对话上下文**：记住之前的对话历史，能够回答追问和确认类问题

## 输出格式要求


{{问题分析}}

{{解决方案}}

{{代码实现}}


请根据用户的具体需求，提供专业、详细、可执行的解决方案。"""

        return system_prompt

    def _read_config_files(self, files: list) -> dict:
        """读取项目中的关键配置文件内容"""
        config_files = ['requirements.txt', 'package.json', 'pom.xml', 'go.mod', 'Cargo.toml',
                       'setup.py', 'pyproject.toml', 'README.md', 'appsettings.json',
                       'application.yml', 'application.yaml', 'docker-compose.yml']

        contents = {}
        for f in files:
            # 支持字符串路径和字典格式
            if isinstance(f, str):
                file_path = f
                file_name = Path(f).name
            else:
                file_path = f.get('path', '')
                file_name = f.get('name', '')

            if file_name in config_files:
                if 'content' in f and f['content']:
                    contents[file_name] = f['content'][:2000]
                    logger.info(f"从项目上下文读取配置文件: {file_name}")
                    continue

                try:
                    # 优先使用项目路径，其次使用项目名称
                    base_path = ""
                    if hasattr(self, 'project_path') and self.project_path:
                        base_path = self.project_path
                    elif hasattr(self, 'project_root') and self.project_root:
                        base_path = self.project_root

                    if base_path:
                        full_path = Path(base_path) / file_path
                    else:
                        full_path = Path(file_path)

                    if full_path.exists() and full_path.is_file():
                        content = full_path.read_text(encoding='utf-8', errors='ignore')
                        contents[file_name] = content[:2000]
                except Exception as e:
                    logger.debug(f"读取配置文件失败 {file_path}: {str(e)}")

        return contents

    def _read_source_files(self, files: list) -> dict:
        """读取项目中的源代码文件内容"""
        source_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
                           '.java', '.cpp', '.go', '.rs', '.php', '.vue', '.json'}

        contents = {}
        logger.info(f"_read_source_files 开始 - project_root: {self.project_root}")
        logger.info(f"文件列表长度: {len(files)}")

        for f in files:
            # 支持字符串路径和字典格式
            if isinstance(f, str):
                file_path = f
                file_name = Path(f).name
            else:
                file_path = f.get('path', '')
                file_name = f.get('name', '')
            file_ext = Path(file_name).suffix.lower()

            logger.debug(f"处理文件: {file_name}, path: {file_path}, ext: {file_ext}")

            if file_ext in source_extensions:
                if isinstance(f, dict) and 'content' in f and f['content']:
                    contents[file_name] = f['content'][:3000]
                    logger.info(f"从项目上下文读取源代码文件: {file_name}")
                    continue

                try:
                    # 优先使用项目路径，其次使用项目名称
                    base_path = ""
                    if hasattr(self, 'project_path') and self.project_path:
                        base_path = self.project_path
                    elif hasattr(self, 'project_root') and self.project_root:
                        base_path = self.project_root

                    if base_path:
                        full_path = Path(base_path) / file_path
                    else:
                        full_path = Path(file_path)

                    logger.debug(f"尝试读取文件 - base_path: {base_path}, file_path: {file_path}, full_path: {full_path}")

                    if full_path.exists() and full_path.is_file():
                        content = full_path.read_text(encoding='utf-8', errors='ignore')
                        contents[file_name] = content[:3000]
                        logger.info(f"从磁盘读取源代码文件: {file_name}, 内容长度: {len(content)}")
                    else:
                        logger.warning(f"文件不存在或不是文件: {full_path}")
                except Exception as e:
                    logger.error(f"读取源代码文件失败 {file_path}: {str(e)}")

        logger.info(f"_read_source_files 完成 - 成功读取 {len(contents)} 个文件")
        return contents

    def _analyze_python_code(self, code: str) -> Dict[str, Any]:
        """分析Python代码，提取关键信息"""
        analysis = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'complexity': 0,
            'issues': []
        }

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    analysis['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'complexity': complexity
                    })
                    analysis['complexity'] += complexity
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append({
                        'name': node.name,
                        'line': node.lineno
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    analysis['imports'].append(f"{node.module}.{node.names[0].name}")

            # 检查常见问题
            analysis['issues'] = self._detect_code_issues(tree, code)

        except SyntaxError as e:
            analysis['issues'].append(f"语法错误: {e.msg} 第 {e.lineno} 行")

        return analysis

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """计算函数的圈复杂度"""
        complexity = 1  # 基础复杂度

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or,
                               ast.Try, ast.IfExp)):
                complexity += 1

        return complexity

    def _detect_code_issues(self, tree: ast.AST, code: str) -> List[str]:
        """检测代码中的常见问题"""
        issues = []
        lines = code.split('\n')

        for node in ast.walk(tree):
            # 检查未使用的变量
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                # 简单检查：赋值但可能未使用
                pass

            # 检查print语句（应该使用logging）
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
                issues.append(f"第 {node.lineno} 行: 使用了 print 语句，建议使用 logging 模块")

            # 检查魔法数字
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                if node.value not in (0, 1, -1):
                    issues.append(f"第 {node.lineno} 行: 使用了魔法数字 {node.value}")

        return issues

    def _check_all_source_files_syntax(self, files: list) -> str:
        """检查所有源代码文件的语法，并返回检查报告"""
        report = "\n**代码语法预检查报告：**\n"
        issues_found = []
        source_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.go', '.rs'}

        for f in files:
            # 支持字符串路径和字典格式
            if isinstance(f, str):
                file_name = Path(f).name
                file_path = f
            else:
                file_name = f.get('name', '')
                file_path = f.get('path', '')
            file_ext = Path(file_name).suffix.lower()

            if file_ext not in source_extensions:
                continue

            # 获取文件内容
            content = ""
            if isinstance(f, dict) and 'content' in f and f['content']:
                content = f['content']
            else:
                try:
                    # 优先使用项目路径，其次使用项目名称
                    base_path = ""
                    if hasattr(self, 'project_path') and self.project_path:
                        base_path = self.project_path
                    elif hasattr(self, 'project_root') and self.project_root:
                        base_path = self.project_root

                    if base_path:
                        full_path = Path(base_path) / file_path
                    else:
                        full_path = Path(file_path)

                    logger.debug(f"语法检查读取文件 - base_path: {base_path}, file_path: {file_path}, full_path: {full_path}")

                    if full_path.exists() and full_path.is_file():
                        content = full_path.read_text(encoding='utf-8', errors='ignore')
                    else:
                        logger.warning(f"语法检查 - 文件不存在: {full_path}")
                except Exception as e:
                    logger.error(f"语法检查 - 读取文件失败 {file_path}: {str(e)}")
                    continue

            if not content.strip():
                issues_found.append(f"- **{file_name}**: 文件为空")
                continue

            # Python语法检查
            if file_ext == '.py':
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    issues_found.append(f"- **{file_name}**: 语法错误 - {e.msg} (第 {e.lineno} 行)")
                except Exception as e:
                    issues_found.append(f"- **{file_name}**: 解析错误 - {str(e)[:50]}")

            # 检查内容是否包含乱码或非代码内容
            if self._contains_garbage_content(content):
                issues_found.append(f"- **{file_name}**: 检测到可能的乱码或非代码内容")

            # 检查文件大小是否合理
            if len(content) < 10:
                issues_found.append(f"- **{file_name}**: 文件内容过短，可能不完整")

        if issues_found:
            report += "\n".join(issues_found) + "\n"
            report += f"\n**总计发现 {len(issues_found)} 个潜在问题**\n"
        else:
            report += "未发现明显的语法问题\n"

        return report

    def _contains_garbage_content(self, content: str) -> bool:
        """检查内容是否包含乱码或非代码内容"""
        # 检查是否包含中文字符（代码中不应有大量无意义中文）
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', content)
        if chinese_chars:
            # 如果中文内容不是注释或字符串，可能是乱码
            for chars in chinese_chars:
                if len(chars) > 5:
                    # 检查是否在字符串或注释中
                    lines = content.split('\n')
                    for line in lines:
                        if chars in line:
                            # 简单检查：如果不是以#开头的注释，可能是乱码
                            stripped = line.strip()
                            if not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                                return True

        # 检查是否包含连续的重复字符（可能是乱码）
        if re.search(r'(.)\1{5,}', content):
            return True

        # 检查是否包含明显的乱码模式
        garbage_patterns = [
            r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]',  # 控制字符
            r'[\xFF\xFE\xFD\xFC]',  # 无效UTF-8
        ]
        for pattern in garbage_patterns:
            if re.search(pattern, content):
                return True

        return False

    def _find_dependencies(self, file_path: str) -> List[str]:
        """查找文件的依赖关系"""
        dependencies = []

        try:
            if hasattr(self, 'project_root') and self.project_root:
                full_path = Path(self.project_root) / file_path
            else:
                full_path = Path(file_path)

            if full_path.exists():
                content = full_path.read_text(encoding='utf-8', errors='ignore')

                # 查找import语句
                import_patterns = [
                    r'from\s+([a-zA-Z0-9_.]+)\s+import',
                    r'import\s+([a-zA-Z0-9_.]+)'
                ]

                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        # 检查是否是项目内部模块
                        if '.' in match:
                            module_path = match.replace('.', '/') + '.py'
                            dependencies.append(module_path)

        except Exception as e:
            logger.debug(f"分析依赖失败 {file_path}: {str(e)}")

        return dependencies

    def _generate_file_list(self) -> str:
        """生成项目文件列表"""
        if not self.project_context or not self.project_context.get('files'):
            return ""

        files = self.project_context['files']
        file_list = "\n项目文件结构:\n"

        # 按目录组织文件
        dir_structure = {}
        for f in files:
            # 支持字符串路径和字典格式
            if isinstance(f, str):
                path = f
                file_type = 'file'
            else:
                path = f.get('path', f.get('name', 'unknown'))
                file_type = f.get('type', 'file')
            parts = path.split('/')
            current = dir_structure

            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {'__type': file_type if i == len(parts)-1 else 'dir', '__children': {}}
                elif i == len(parts)-1:
                    current[part]['__type'] = file_type
                current = current[part]['__children']

        def format_structure(structure, prefix=""):
            result = ""
            items = sorted(structure.keys())
            for i, name in enumerate(items):
                item = structure[name]
                is_last = i == len(items) - 1
                connector = " " if is_last else " "

                if item['__type'] == 'dir':
                    result += f"{prefix}{connector}{name}/\n"
                    new_prefix = prefix + ("    " if is_last else "   ")
                    result += format_structure(item['__children'], new_prefix)
                else:
                    result += f"{prefix}{connector}{name}\n"

            return result

        file_list += format_structure(dir_structure)
        return file_list

    def get_model_response(self, message, llm) -> str:
        """获取模型响应（非流式）"""
        if hasattr(message, 'content'):
            message_content = message.content
        else:
            message_content = str(message)

        # 使用短期记忆获取对话历史
        conversation_history = self.memory.short_term.get_conversation_history()

        # 记录用户输入到短期记忆
        self.memory.short_term.add_action('user_message', {'content': message_content[:100]})

        # 构建用户消息内容（支持多模态：文本 + 图片）
        user_message_content = self._build_multimodal_message(message_content)

        prompt_list = [
                          {"role": "system", "content": self.system_prompt},
                      ] + conversation_history + [
                          {"role": "user", "content": user_message_content},
                      ]

        try:
            is_multimodal = isinstance(user_message_content, list)
            if is_multimodal:
                # VL多模态模型：不支持 system 角色，使用精简版 system prompt 避免 token 超限
                system_prefix = f"[系统指令]\n{self._get_vl_system_prompt()}\n\n[用户消息]\n"
                vl_user_content = []
                for part in user_message_content:
                    if part.get("type") == "text":
                        vl_user_content.append({"type": "text", "text": system_prefix + part["text"]})
                    else:
                        vl_user_content.append(part)
                prompt_list = conversation_history + [
                    {"role": "user", "content": vl_user_content},
                ]
            response = llm.invoke(prompt_list)
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)

            # 将响应保存到长期记忆
            self._update_memory_from_response(message_content, response_content)

            return response_content
        except Exception as e:
            logger.error(f"project_code_node LLM 调用失败: {str(e)}")
            return f"** 错误：** 抱歉，当前服务暂时不可用，请稍后重试。"

    def _update_memory_from_response(self, user_message: str, response: str):
        """从响应中提取信息并更新记忆"""
        # 检测并记录问题
        issues = self._extract_issues_from_response(response)
        for issue in issues:
            self.memory.long_term.add_issue(issue)
            logger.info(f"记录问题到长期记忆: {issue.get('description', '')}")

        # 检测并记录学习到的模式
        patterns = self._extract_patterns_from_response(response)
        for pattern in patterns:
            self.memory.long_term.add_learned_pattern(pattern)
            logger.info(f"记录模式到长期记忆: {pattern.get('name', '')}")

        # 记录代码知识
        code_knowledge = self._extract_code_knowledge(user_message, response)
        if code_knowledge:
            for file_name, knowledge in code_knowledge.items():
                self.memory.long_term.add_code_knowledge(file_name, knowledge)
                logger.info(f"记录代码知识到长期记忆: {file_name}")

        # 保存长期记忆
        self.memory.save()

    def _extract_issues_from_response(self, response: str) -> List[dict]:
        """从响应中提取问题信息"""
        issues = []

        # 查找问题描述模式
        issue_patterns = [
            r'发现(\d+)处问题',
            r'第(\d+)处：(.+)',
            r'问题：(.+)',
            r'错误：(.+)',
            r'BUG：(.+)',
            r'缺陷：(.+)',
            r'问题描述：(.+)',
            r'行(\d+)：(.+)',
        ]

        for pattern in issue_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        issues.append({
                            'description': match[1],
                            'line': match[0],
                            'severity': 'medium'
                        })
                else:
                    issues.append({
                        'description': match,
                        'severity': 'medium'
                    })

        return issues

    def _extract_patterns_from_response(self, response: str) -> List[dict]:
        """从响应中提取学习到的模式"""
        patterns = []

        # 查找设计模式或最佳实践描述
        pattern_keywords = ['模式', '最佳实践', '设计模式', '建议', '优化']
        for keyword in pattern_keywords:
            if keyword in response:
                # 简单提取包含关键字的句子
                sentences = response.split('。')
                for sentence in sentences:
                    if keyword in sentence and len(sentence) > 10:
                        patterns.append({
                            'name': keyword + ': ' + sentence[:30],
                            'description': sentence.strip()
                        })

        return patterns

    def _extract_code_knowledge(self, user_message: str, response: str) -> Dict[str, dict]:
        """从响应中提取代码知识"""
        knowledge = {}

        # 查找代码块
        code_blocks = re.findall(r'(\w+)?\n([\s\S]*?)', response)
        for lang, code in code_blocks:
            if lang and code.strip():
                # 尝试从代码中提取函数和类信息
                if lang.lower() == 'python':
                    try:
                        tree = ast.parse(code)
                        file_knowledge = {'functions': [], 'classes': []}
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                file_knowledge['functions'].append(node.name)
                            elif isinstance(node, ast.ClassDef):
                                file_knowledge['classes'].append(node.name)
                        if file_knowledge['functions'] or file_knowledge['classes']:
                            knowledge['extracted_code'] = file_knowledge
                    except:
                        pass

        return knowledge

    def _build_multimodal_message(self, text_content: str):
        """构建多模态消息（文本 + 图片）"""
        if not self.images or len(self.images) == 0:
            return text_content

        # 构建多模态内容列表
        content_parts = []

        # 多张图片时，在文本前添加图片编号说明，方便LLM区分"第1张图""第2张图"
        img_count = len(self.images)
        if img_count > 1:
            prefix = f"[用户上传了{img_count}张图片，按顺序为："
            prefix += "、".join([f"图片{i+1}" for i in range(img_count)])
            prefix += "]\n\n"
            text_content = prefix + text_content

        # 添加文本部分
        if text_content:
            content_parts.append({
                "type": "text",
                "text": text_content
            })

        # 添加所有图片
        for img in self.images:
            img_data = img.get('data', '')
            img_name = img.get('name', 'image')

            # 提取 base64 数据（去掉 data:image/xxx;base64, 前缀）
            base64_data = img_data
            if ',' in img_data:
                base64_data = img_data.split(',', 1)[1]

            # 检测图片类型
            mime_type = 'image/png'  # 默认
            if img_data.startswith('data:'):
                mime_type = img_data.split(':')[1].split(';')[0]

            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": img_data
                }
            })

            logger.info(f"添加图片到消息: {img_name}, 类型: {mime_type}")

        logger.info(f"构建多模态消息: {len(content_parts)} 个部分（1个文本 + {len(self.images)}张图片）")
        return content_parts

    def get_model_response_stream(self, message, model_name: str = ""):
        """获取模型响应（流式）- 使用原生 OpenAI API"""
        from openai import OpenAI

        model_name = Params.resolve_model_name(model_name)

        if hasattr(message, 'content'):
            message_content = message.content
        else:
            message_content = str(message)

        # 识别用户意图
        intent_result = self._recognize_intent(message_content)

        # 使用短期记忆获取对话历史
        conversation_history = self.memory.short_term.get_conversation_history()

        # 记录用户输入到短期记忆
        self.memory.short_term.add_action('user_message', {'content': message_content})

        # 构建用户消息内容（支持多模态：文本 + 图片）
        is_multimodal = False
        user_message_content = self._build_multimodal_message(message_content)
        for part in user_message_content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                is_multimodal = True
                break

        prompt_list = [
                          {"role": "system", "content": self.system_prompt},
                      ] + conversation_history + [
                          {"role": "user", "content": user_message_content},
                      ]

        try:
            if is_multimodal:
                # VL多模态模型：不支持 system 角色，使用精简版 system prompt 避免 token 超限
                system_prefix = f"[系统指令]\n{self._get_vl_system_prompt()}\n\n[用户消息]\n"
                vl_user_content = []
                for part in user_message_content:
                    if part.get("type") == "text":
                        vl_user_content.append({"type": "text", "text": system_prefix + part["text"]})
                    else:
                        vl_user_content.append(part)
                vl_prompt_list = [
                    {"role": "user", "content": vl_user_content},
                ]

                # 使用原生 OpenAI API 调用多模态模型
                client = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url=Params.API_BASE,
                )
                response = client.chat.completions.create(
                    model=Params.DEFAULT_MULTIMODAL_MODEL,
                    messages=vl_prompt_list,
                    stream=True,
                )

                total_text = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        response_text = chunk.choices[0].delta.content
                        self._update_memory_from_response(message_content, response_text)
                        total_text += response_text
                        yield response_text
                        time.sleep(0.03)

                # 多模态路径完成，直接返回
                return

            thinking_count = 0
            paragraph_count = 0
            in_code_block = False
            code_block_count = 0
            full_response = ""

            # === ReAct 思考阶段：用专门的思考模型先输出思考过程 ===
            react_prompt = f"""你是一个专业的代码分析助手。请对用户的问题进行思考分析，使用以下格式：
    **思考：** 简要说明你打算做什么
    **分析：** 分析用户需求和技术方案  
    **计划：** 列出你的实现步骤

    **重要规则（必须严格遵守）：**
    -  只输出思考、分析、计划三个部分！
    -  回复过程中严禁输出任何代码块或代码片段！
    -  回复过程中严禁提及要修改哪些具体文件！
    -  回复过程中严禁输出完整的实现代码！
    -  不要列出依赖安装命令！
    -  保持简洁，每个部分 1-2 句话即可！

    用户问题：{message_content}

    请开始你的思考分析（保持简洁）："""

            # 使用原生 OpenAI API 进行思考阶段
            try:
                client = OpenAI(
                    api_key=os.getenv("DASHSCOPE_API_KEY"),
                    base_url=Params.API_BASE,
                )

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": react_prompt}],
                    stream=True,
                )

                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        yield content
                        time.sleep(0.03)
            except Exception as e:
                # 如果思考模型不可用，跳过思考阶段
                logger.warning(f"ReAct 思考模型调用失败，跳过思考阶段: {e}")

            # 思考阶段结束标记
            yield "\n\n---\n\n"

            # === 正式回复阶段：用主 LLM 输出正式内容 ===
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=Params.API_BASE,
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=prompt_list,
                stream=True,
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content  # 累积完整响应
                    code_block_count += content.count('')
                    in_code_block = code_block_count % 2 == 1

                    if intent_result and not in_code_block:
                        thinking_count += len(content)
                        paragraph_count += content.count('\n')

                        if not in_code_block and (thinking_count > 1000 or (paragraph_count > 10 and thinking_count > 400)):
                            thinking_count = 0
                            paragraph_count = 0
                            thinking_phrases = [
                                "\n\n<!--THINKING_START--><span style='color:#666;font-style:italic'> 继续分析中...</span><!--THINKING_END-->\n\n",
                                "\n\n<!--THINKING_START--><span style='color:#666;font-style:italic'> 深入研究中...</span><!--THINKING_END-->\n\n",
                                "\n\n<!--THINKING_START--><span style='color:#666;font-style:italic'> 发现关键点...</span><!--THINKING_END-->\n\n",
                                "\n\n<!--THINKING_START--><span style='color:#666;font-style:italic'> 正在处理...</span><!--THINKING_END-->\n\n",
                            ]
                            yield thinking_phrases[len(conversation_history) % len(thinking_phrases)]
                            time.sleep(0.3)
                    yield content

            # 流式响应完成后，更新长期记忆
            self.history.add_ai_message(full_response)
            self._update_memory_from_response(message_content, full_response)

        except Exception as e:
            logger.error(f"project_code_node LLM 流式调用失败: {str(e)}")
            yield "\n\n** 错误：** 抱歉，当前服务暂时不可用，请稍后重试。"

    def _recognize_intent(self, message: str) -> str:
        """识别用户意图"""
        intent_keywords = {
            '生成新代码': ['生成', '创建', '新建', '写一个', '写一段', '实现', '编写', '开发', '构建'],
            '审查/修改代码': ['审查', '修改', '优化', '重构', '改进', '调整', '修复', 'bug', '错误', '缺陷', '性能', '优化建议'],
            '添加新功能': ['添加', '新增', '增加', '扩展', '功能', '模块', 'feature'],
            '解释代码': ['解释', '说明', '什么意思', '什么作用', '如何工作', '原理', '理解', '解析'],
            '分析项目架构': ['分析', '架构', '结构', '设计', '技术栈', '整体', '项目', '框架', '组成'],
            '设计方案': ['设计', '方案', '规划', '架构设计', '方案设计', '技术方案'],
            '数据库操作': ['数据库', 'SQL', '表', '数据', 'MySQL', 'PostgreSQL', 'MongoDB', '数据结构'],
            '删除文件': ['删除', '移除', '清理', '删除文件', '删除代码'],
            '代码审查': ['审查', '检查', 'review', '审核', '代码检查'],
            '项目文档': ['文档', 'README', '说明文档', '文档编写', '文档更新'],
            '调试排错': ['调试', '报错', '异常', '崩溃', 'stack', 'trace', '错误日志'],
            '性能优化': ['性能', '优化', '速度', '效率', '慢', '卡顿'],
        }

        self_introduction = {
            '介绍自己': ['介绍', '自己', '身份', '你是', '你是谁', '谁', '你能做什么', '你能干什么', '你能干啥', '你能做啥', '你的功能'],
        }

        for intent, keywords in self_introduction.items():
            for keyword in keywords:
                if keyword in message:
                    return ""

        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in message:
                    logger.info(f"识别意图: {intent}")
                    return intent

        return "其他"

    # ==================== AI 编程助手统一功能 ====================

    # Prompt 模板（统一版）
    SYSTEM_PROMPT = """你是 AI 编程助手，具备工具调用能力，可以读写项目文件。

## 工作流程
1. 分析用户需求，确定需要修改哪些文件
2. 如果需要读取文件，调用 read_file 读取内容
3. 如果需要搜索代码，调用 search_code 检索
4. 基于读取的内容，生成修改后的完整代码
5. 调用 write_file 写入修改后的代码（会自动生成 diff 供用户确认）
6. 所有文件修改完成后，用自然语言总结修改内容

## 可用工具
1. read_file(file_path): 读取项目内指定文件内容
2. write_file(file_path, content): 写入/覆盖文件（会自动生成 diff）
3. search_code(keyword): 项目全局代码检索

## 重要规则（必须遵守）
1. 参考【当前打开文件】和【项目相关代码片段】回答，不要编造项目不存在的逻辑。
2. 如果用户没有打开任何文件，你必须先调用 search_code 或 read_file 工具读取相关文件！不能直接输出模板回复！
3. 修改文件前，必须先调用 read_file 读取原文件内容。
4. 工具调用必须使用 JSON 格式，每次只调用一个工具。
5. 当用户要求修改代码时，你必须在读取文件后立即调用 write_file 工具！不能只用文字描述修改方案！
6. write_file 的 content 参数必须是完整的修改后的代码，不能只写修改的部分！
7. 【代码类型匹配规则】（非常重要）：
   - Python 文件(.py)：只能写入 Python 代码，绝对不能写入 HTML/CSS/JavaScript 代码
   - HTML 文件(.html/.htm)：只能写入 HTML 代码，可以包含 Jinja2 模板语法，但不能写入纯 Python 代码
   - JavaScript 文件(.js)：只能写入 JavaScript 代码，不能写入 HTML 文档结构
   - CSS 文件(.css)：只能写入 CSS 样式代码
   - 如果你不确定代码类型，请先读取原文件查看其内容格式
8. 当所有工具调用完成后，在最终回复中：
   - 用自然语言总结修改内容
   - 对于每个修改的文件，输出完整的修改后代码，使用正确的代码块格式（python 或 html 或 javascript）
   - 绝对不要输出任何 JSON 格式的工具调用！
   - 绝对不要输出 {"tool": ...} 格式的内容！
9. 使用中文回复，代码注释使用中文。

## 工具调用格式（JSON）
{"tool": "工具名", "args": {"参数名": "参数值"}}

示例：
{"tool": "read_file", "args": {"file_path": "src/main.py"}}
{"tool": "write_file", "args": {"file_path": "src/main.py", "content": "完整代码内容"}}
{"tool": "search_code", "args": {"keyword": "function_name"}}"""

    def _retrieve_project_context(self, query: str, top_k: int = 5) -> str:
        """从项目代码库检索相关代码片段"""
        try:
            from core.knowledge.knowledge_manager import get_vector_store
            
            vector_store = get_vector_store("default_user", retrieval_mode="hybrid")
            
            results = vector_store.similarity_search_with_score(query, k=top_k)
            
            chunks_text = []
            for doc, score in results:
                source = doc.metadata.get('source', 'unknown')
                chunk_text = f"【文件: {source}】(相似度: {score:.3f})\n{doc.page_content}\n"
                chunks_text.append(chunk_text)
            
            return '\n'.join(chunks_text) if chunks_text else "未检索到相关代码片段。"
        except Exception as e:
            logger.warning(f"RAG 检索失败: {e}")
            return "RAG 检索失败，将仅基于当前文件回答。"

    def _read_file_tool(self, file_path: str) -> Dict[str, Any]:
        """工具：读取项目内指定文件"""
        try:
            if os.path.isabs(file_path):
                full_path = file_path
            else:
                full_path = os.path.join(self.project_path or self.project_root, file_path)
            
            if not os.path.exists(full_path):
                return {"success": False, "error": f"文件不存在: {full_path}"}
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            return {"success": True, "content": content, "path": full_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _write_file_tool(self, file_path: str, content: str) -> Dict[str, Any]:
        """工具：写入/覆盖文件，返回 diff 对比"""
        try:
            if os.path.isabs(file_path):
                full_path = file_path
            else:
                full_path = os.path.join(self.project_path or self.project_root, file_path)
            
            old_content = ""
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    old_content = f.read()
            
            # 代码类型验证：检测内容是否与文件类型匹配
            validation_result = self._validate_code_type(file_path, content)
            if not validation_result['valid']:
                logger.warning(f"代码类型验证失败: {file_path} - {validation_result['reason']}")
                return {
                    "success": False, 
                    "error": f"代码类型不匹配: {validation_result['reason']}",
                    "validation_failed": True
                }
            
            dir_path = os.path.dirname(full_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            diff = self._generate_diff(old_content, content, file_path)
            
            return {"success": True, "path": full_path, "diff": diff}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _validate_code_type(self, file_path: str, content: str) -> Dict[str, Any]:
        """验证代码内容是否与文件类型匹配"""
        ext = os.path.splitext(file_path)[1].lower()
        
        # Python 文件验证
        if ext == '.py':
            # 检测是否包含 HTML 标签
            html_patterns = [
                r'<!DOCTYPE\s+html>',
                r'<html[>]*>',
                r'<head[>]*>',
                r'<body[>]*>',
                r'<div[>]*>',
                r'<script[>]*>',
                r'<style[>]*>',
                r'<meta[>]*>',
                r'<link[>]*rel=',
            ]
            
            for pattern in html_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return {
                        'valid': False,
                        'reason': f'Python 文件(.py)中检测到 HTML 代码（匹配模式: {pattern}），请检查是否将 HTML 代码错误地写入了 Python 文件'
                    }
            
            # 检测是否包含大量 HTML 特征
            html_tag_count = len(re.findall(r'<[a-zA-Z][>]*>', content))
            if html_tag_count > 5:
                return {
                    'valid': False,
                    'reason': f'Python 文件(.py)中包含 {html_tag_count} 个 HTML 标签，疑似将 HTML 代码错误地写入了 Python 文件'
                }
        
        # HTML 文件验证
        elif ext in ['.html', '.htm']:
            # 检测是否包含 Python 代码特征（但不是 Flask/Jinja2 模板语法）
            python_patterns = [
                r'import\s+\w+',
                r'from\s+\w+\s+import',
                r'def\s+\w+\s*\(',
                r'class\s+\w+',
            ]
            
            # 检查是否有多行 Python 代码（排除 Jinja2 模板中的 Python 表达式）
            python_code_lines = 0
            for line in content.split('\n'):
                line_stripped = line.strip()
                # 跳过 Jinja2 模板语法
                if line_stripped.startswith('{%') or line_stripped.startswith('{{') or line_stripped.startswith('{#'):
                    continue
                # 跳过 HTML 注释
                if line_stripped.startswith('<!--'):
                    continue
                # 检查 Python 代码特征
                for pattern in python_patterns:
                    if re.match(pattern, line_stripped):
                        python_code_lines += 1
                        break
            
            if python_code_lines > 3:
                return {
                    'valid': False,
                    'reason': f'HTML 文件(.html)中包含 {python_code_lines} 行 Python 代码，疑似将 Python 代码错误地写入了 HTML 文件'
                }
        
        # JavaScript 文件验证
        elif ext == '.js':
            # 检测是否包含 HTML 标签
            html_patterns = [
                r'<!DOCTYPE\s+html>',
                r'<html[>]*>',
                r'<head[>]*>',
                r'<body[>]*>',
            ]
            
            for pattern in html_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return {
                        'valid': False,
                        'reason': f'JavaScript 文件(.js)中检测到 HTML 文档结构，请检查是否将 HTML 代码错误地写入了 JS 文件'
                    }
        
        return {'valid': True, 'reason': ''}

    def _generate_diff(self, old_content: str, new_content: str, file_path: str) -> str:
        """生成文本 diff"""
        # 统一行分隔符为 \n，避免 \r\n 导致前端解析问题
        old_content = old_content.replace('\r\n', '\n').replace('\r', '\n')
        new_content = new_content.replace('\r\n', '\n').replace('\r', '\n')
        
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=3,
        )
        
        return ''.join(diff)

    def _search_code_tool(self, keyword: str, max_results: int = 10) -> Dict[str, Any]:
        """工具：项目全局代码检索"""
        try:
            results = []
            project_root = self.project_path or self.project_root
            for root, dirs, files in os.walk(project_root):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules')]
                
                for file in files:
                    if file.startswith('.') or file.endswith(('.pyc', '.pyo', '.pyd')):
                        continue
                    
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if keyword.lower() in content.lower():
                            rel_path = os.path.relpath(file_path, project_root)
                            lines = content.split('\n')
                            match_lines = [i+1 for i, line in enumerate(lines) if keyword.lower() in line.lower()]
                            
                            results.append({
                                'file': rel_path,
                                'match_lines': match_lines[:5],
                            })
                            
                            if len(results) >= max_results:
                                break
                    except:
                        continue
            
            return {"success": True, "results": results, "count": len(results)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """从 LLM 输出中解析工具调用"""
        pattern = r'\{"tool":\s*"(\w+)",\s*"args":\s*\{[}]*\}\}'
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None

    def _generate_proposed_diff(self, file_path: str, new_content: str) -> Dict[str, Any]:
        """生成提议的 diff（不实际写入文件）"""
        try:
            if os.path.isabs(file_path):
                full_path = file_path
            else:
                full_path = os.path.join(self.project_path or self.project_root, file_path)
            
            old_content = ""
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    old_content = f.read()
            
            diff = self._generate_diff(old_content, new_content, file_path)
            
            return {"success": True, "diff": diff, "path": full_path, "new_content": new_content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def stream_ai_code_chat(
        self,
        query: str,
        current_file_path: str,
        current_file_code: str,
        command: str = "",
        model_name: str = None,
        max_tool_calls: int = 5,
    ) -> AsyncGenerator[str, None]:
        """统一版：流式 AI 编程助手（支持工具调用和 diff 生成）"""
        model_name = Params.resolve_model_name(model_name)
        
        # 构建用户消息内容（支持图片识别）
        user_content = f"""【当前打开文件路径】：{current_file_path}
【当前打开文件代码】：

{current_file_code[:5000] if current_file_code else "未打开任何文件"}

【项目文件列表】：
{chr(10).join(['- ' + f for f in (self.project_context.get('files', [])[:50] if self.project_context else [])]) or "未提供项目文件列表"}

用户需求：{query}"""
        
        # 如果有图片，构建多模态消息
        if self.images and len(self.images) > 0:
            user_content = self._build_multimodal_message(user_content)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        try:
            yield " **正在连接 AI 模型...**\n\n"
            await asyncio.sleep(0.1)
            
            client = OpenAI(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=Params.API_BASE,
                timeout=60.0,  # 添加超时设置
            )
            
            yield " **AI 模型已连接，开始分析...**\n\n"
            await asyncio.sleep(0.1)
            
            for iteration in range(max_tool_calls):
                yield f" **Agent 迭代 {iteration + 1}/{max_tool_calls}...**\n\n"
                await asyncio.sleep(0.1)
                
                try:
                    # 添加进度提示
                    yield f" **正在等待 AI 模型响应...**\n\n"
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        stream=False,
                        timeout=60.0,  # 减少超时时间到 60 秒
                    )
                except Exception as e:
                    yield f" **AI 模型调用失败：** {str(e)}\n\n"
                    logger.error(f"AI 模型调用失败: {e}")
                    break
                
                assistant_msg = response.choices[0].message.content
                messages.append({"role": "assistant", "content": assistant_msg})
                
                tool_call = self._parse_tool_call(assistant_msg)
                
                # 如果没有检测到工具调用，检查是否有代码块输出
                if not tool_call:
                    # 检测是否有代码块（LLM 可能用代码块展示修改方案）
                    code_block_match = re.search(r'(?:html|javascript|python|js|py)?\s*\n([\s\S]*?)\n', assistant_msg)
                    if code_block_match and iteration > 0:
                        # LLM 输出了代码块，提取并自动调用 write_file
                        code_content = code_block_match.group(1).strip()
                        # 尝试从上下文推断文件路径
                        file_path = current_file_path if current_file_path else "unknown_file"
                        
                        yield f"\n\n **检测到代码块，自动调用 write_file：** {file_path}\n\n"
                        
                        result = self._generate_proposed_diff(file_path, code_content)
                        
                        if result["success"]:
                            yield f" **工具调用成功**\n\n"
                            yield f"diff\n{result['diff']}\n\n\n"
                            if "new_content" in result and result["new_content"]:
                                yield f"<!--AI_CODE_CHANGE_START-->\n{result['new_content']}\n<!--AI_CODE_CHANGE_END-->\n\n"
                        else:
                            yield f" **工具调用失败：** {result.get('error', '未知错误')}\n\n"
                        
                        messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                        continue
                    else:
                        # 没有工具调用也没有代码块，结束循环
                        break
                
                tool_name = tool_call["tool"]
                tool_args = tool_call["args"]
                
                yield f"\n\n **调用工具：** {tool_name}\n\n"
                
                if tool_name == "read_file":
                    result = self._read_file_tool(tool_args["file_path"])
                elif tool_name == "write_file":
                    # V2: 不直接写入，而是生成 diff 预览
                    result = self._generate_proposed_diff(tool_args["file_path"], tool_args["content"])
                elif tool_name == "search_code":
                    result = self._search_code_tool(tool_args["keyword"])
                else:
                    result = {"success": False, "error": f"未知工具: {tool_name}"}
                
                if result["success"]:
                    yield f" **工具调用成功**\n\n"
                    if "diff" in result:
                        # V2: 结构化输出 diff 和 new_content，方便前端解析
                        yield f"diff\n{result['diff']}\n\n\n"
                        # 使用特殊标记包裹 new_content，前端可提取
                        if "new_content" in result and result["new_content"]:
                            yield f"<!--AI_CODE_CHANGE_START-->\n{result['new_content']}\n<!--AI_CODE_CHANGE_END-->\n\n"
                    if "content" in result:
                        yield f"\n{result['content'][:1000]}\n\n\n"
                else:
                    yield f" **工具调用失败：** {result.get('error', '未知错误')}\n\n"
                
                messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
            
            yield f"\n\n **Agent 循环结束，生成最终回复：**\n\n"
            
            # 添加一个明确的指令，要求 LLM 用自然语言总结，并输出修改后的完整代码
            messages.append({
                "role": "user",
                "content": """所有工具调用已完成。请按以下格式回复：

1. 用自然语言总结你的修改内容
2. ️ 对于每个修改的文件，你必须输出完整的修改后代码，使用代码块格式：
   html
   完整的HTML代码
   
   或
   javascript
   完整的JS代码
   
3. ️ 绝对不要输出任何 JSON 格式的工具调用！
4. ️ 绝对不要输出 {"tool": ...} 格式的内容！
5. 代码块必须包含完整的文件内容，不能只写修改的部分。"""
            })
            
            # 先收集完整的最终回复（保留原始内容用于检测）
            final_content = ""
            final_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
            )
            
            for chunk in final_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if content.strip():
                        final_content += content
                        # 过滤掉 JSON 工具调用格式后再输出给前端
                        filtered_content = re.sub(r'\{"tool":\s*"["]+",\s*"args":\s*\{[}]+\}\}', '', content)
                        if filtered_content.strip():
                            yield filtered_content
                        await asyncio.sleep(0.01)
            
            # 检测最终回复中的代码块，自动生成 diff
            code_blocks = list(re.finditer(r'(?:html|javascript|python|js|py|css)?\s*\n([\s\S]*?)\n', final_content))
            for match in code_blocks:
                code_content = match.group(1).strip()
                if len(code_content) > 50:  # 只处理有意义的代码块
                    # 尝试推断文件路径
                    file_path = current_file_path if current_file_path else None
                    if not file_path:
                        path_match = re.search(r'["](["]*\.(?:html|js|py|css|vue|jsx|tsx))["]', final_content)
                        if path_match:
                            file_path = path_match.group(1)
                    if not file_path:
                        file_path = "modified_file"
                    
                    yield f"\n\n **检测到修改代码，生成 diff：** {file_path}\n\n"
                    
                    result = self._generate_proposed_diff(file_path, code_content)
                    
                    if result["success"]:
                        yield f"diff\n{result['diff']}\n\n\n"
                        if "new_content" in result and result["new_content"]:
                            yield f"<!--AI_CODE_CHANGE_START-->\n{result['new_content']}\n<!--AI_CODE_CHANGE_END-->\n\n"
            
            # 如果没有检测到代码块，尝试从 JSON 工具调用中提取（LLM 可能还是输出了 JSON）
            if not code_blocks:
                json_tool_match = re.search(r'\{"tool":\s*"write_file",\s*"args":\s*\{"file_path":\s*"(["]+)",\s*"content":\s*"((?:["\\]|\\.)*)"\}\}', final_content)
                if json_tool_match:
                    file_path = json_tool_match.group(1)
                    # 解码转义字符
                    code_content = json_tool_match.group(2).encode().decode('unicode_escape')
                    
                    yield f"\n\n **检测到 JSON 工具调用，自动生成 diff：** {file_path}\n\n"
                    
                    result = self._generate_proposed_diff(file_path, code_content)
                    
                    if result["success"]:
                        yield f"diff\n{result['diff']}\n\n\n"
                        if "new_content" in result and result["new_content"]:
                            yield f"<!--AI_CODE_CHANGE_START-->\n{result['new_content']}\n<!--AI_CODE_CHANGE_END-->\n\n"
            
            # 注意：此时 SSE 流还在继续，前端会收到这些 diff 内容
            
        except Exception as e:
            logger.error(f"AI 编程助手 Agent 调用失败: {e}")
            yield f"\n\n**错误：** AI 编程助手 Agent 调用失败: {str(e)}"