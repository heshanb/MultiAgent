from settings.Define import Params, PathConfig
import os
import re
import ast
import time
from logging import getLogger
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.memory.memory_manager import MemoryManager
from langchain_openai import ChatOpenAI

logger = getLogger(__name__)


class Project_Programming:
    def __init__(self, file_path: str, state_list: list, project_context: dict = None, images: list = None):
        self.file_path = file_path
        self.state_list = state_list
        self.project_context = project_context
        self.images = images  # 存储图片数据
        
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
                source_files = [f for f in files if f.get("path") and not any(p.startswith('.') for p in Path(f['path']).parts)]
                if source_files:
                    file_list_str = "\n**项目文件列表（必须从这里匹配）：**\n"
                    for f in source_files:
                        file_list_str += f"  - {f['path']}\n"
                    file_list_str += "\n"

        return f"""你是一个专业的全栈项目编程专家和AI助手。

{file_list_str}
**最高规则（必须严格遵守）：**
1. 🔴 对比图片内容与上方项目文件列表，找到内容/命名/语义最匹配的文件 → 直接在该文件上修改
2. 🔴 严禁根据代码中的函数名、类名自行生成新文件名（如split_text.py、new_file.py等）
3. 🔴 文件名必须使用上方项目文件列表中的真实路径
4. 🔴 回复必须以"发现问题来源于{{文件路径}}。"开头，明确告知用户问题出在哪个项目文件
5. 🔴 代码修改必须标注"文件路径: {{项目中的真实路径}}"
6. 🔴 图片中有明显的 error/报错信息 → 从上方文件列表中定位报错来源文件，在该文件上修改
7. 🔴 图片内容无法匹配项目中的任何文件 → 结合用户的语言描述综合考虑
8. 🔴 用户的意图如果没有提及要新增某功能或新建一个文件，就不能创建新文件
9. 🔴 匹配项目文件时，优先考虑：文件名语义是否匹配图片内容、文件内容是否包含图片中的代码"""

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
                    project_info += f"\n```\n# {config_name}\n{content}\n```\n"
            
            # 读取源代码文件内容
            source_files_content = self._read_source_files(files)
            
            # 添加源代码文件内容
            if source_files_content:
                project_info += "- 源代码文件内容:\n"
                for file_name, content in source_files_content.items():
                    file_ext = file_name.split('.')[-1].lower() if '.' in file_name else 'text'
                    project_info += f"\n```\n# {file_name}\n{content}\n```\n"
            
            # 添加语法预检查报告
            syntax_report = self._check_all_source_files_syntax(files)
            project_info += syntax_report
            logger.info(f"语法预检查报告: {syntax_report[:500]}")
            
            # 添加文件结构
            project_info += "- 项目文件结构:\n"
            for f in files[:50]:  # 最多显示50个文件
                project_info += f"  - {f['path']} ({f['type']})\n"
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
        ext_list = ', '.join(f'`.{ext[1:]}`' for ext in extensions)

        # 获取长期记忆摘要
        long_term_memory_summary = self.memory.long_term.get_memory_summary()
        
        # 检查是否有源代码文件
        has_source_files = False
        if project_context:
            files = project_context.get("files", [])
            source_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.go', '.rs', '.php', '.vue'}
            for f in files:
                file_name = f.get('name', '')
                file_ext = Path(file_name).suffix.lower()
                if file_ext in source_extensions:
                    has_source_files = True
                    break
        
        system_prompt = f"""你是一个专业的全栈项目编程专家和AI助手，熟悉各种编程语言和框架，比如Python、JavaScript、Java、Go、C++、HTML/CSS等。
专门帮助用户进行代码开发和项目管理。

**ReAct 思考框架（必须遵守）：**
在回答之前，请先进行思考，使用以下格式输出你的思考过程：
- 用 `**思考：**` 开头，简要说明你打算做什么
- 用 `**分析：**` 开头，分析用户需求和技术方案
- 用 `**计划：**` 开头，列出你的实现步骤
- 思考过程要简洁，不要过于冗长
- 思考完成后，再输出正式的回复内容

{project_info}

{long_term_memory_summary}
## 空项目强制要求（如果上方显示"这是一个空项目"）
- **如果主要编程语言是 Python，必须生成 requirements.txt 文件！这是强制要求，不可省略！**
- requirements.txt 内容格式：每行一个依赖，如 `flask==3.0.0`
- 如果使用了 Flask，requirements.txt 必须包含 `flask`
- 如果使用了其他库，也必须全部写入 requirements.txt
- 如果是其他编程语言（如 JavaScript/Java 等），则生成对应的依赖配置文件（如 package.json、pom.xml 等）

## 重要提示
- **🔴 最高优先级：修改代码时必须在项目现有文件上修改，严禁新增文件！**
- **🔴 当用户要求修改/修复代码时，第一步必须从上方"项目文件结构"中找到匹配的文件，文件名必须使用项目列表中实际存在的路径**
- **🔴 判断文件匹配的方式：1) 文件名语义是否匹配代码内容 2) 文件内容是否包含要修复的函数/类 3) 路径是否最接近**
- **🔴 绝对禁止根据函数名、代码内容自行生成新文件名（如split_text.py、new_file.py等），必须使用项目中真实存在的文件名**
- **🔴 回复必须以"发现问题来源于{{文件路径}}。"开头，明确告知用户问题出在哪个项目文件上**
- **🔴 代码修改必须标注"文件路径: {{项目中的真实路径}}"，严禁编造文件名**
- **🔴 除非用户明确要求"新建文件"或"创建新文件"，否则绝对不允许创建任何新文件**
- **🔴 当用户发送截图/图片时，对比图片内容与上方项目文件列表，找到内容/命名最匹配的文件，在该文件上修改**
- 当用户要求分析整个项目或检查代码问题时，**必须**仔细检查所有源代码文件的内容
- 项目中可能包含 {ext_list} 等源代码文件
- **如果发现源代码文件，必须主动分析其语法和逻辑问题**
- **不要忽略任何源代码文件**，即使它们不在配置文件列表中
- **⚠️ 严禁凭空捏造文件！只能分析项目文件列表中实际存在的文件，绝对不允许分析或提及项目中不存在的文件**
- **⚠️ 所有分析的文件名必须来源于上方的"项目文件结构"列表，不得自行编造**
- 请根据项目的主要编程语言（{self.programming_language}）进行代码分析和建议
- 你拥有长期记忆功能，可以记住之前发现的问题和学习到的模式，请参考历史信息
- **请特别关注以下问题类型：语法错误、拼写错误、缩进问题、未定义变量、类型错误、逻辑错误**
- **当检查代码语法时，请逐行分析，不要遗漏任何细节**

{language_prompt}

## 代码检查清单
1. 语法检查：检查括号、引号、分号、缩进等是否正确
2. 拼写检查：检查变量名、函数名、关键字是否拼写正确
3. 类型检查：检查变量类型是否匹配
4. 逻辑检查：检查代码逻辑是否合理
5. 导入检查：检查import语句是否正确，模块是否存在

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
- 仔细检查代码中的语法问题、逻辑问题、BUG问题等
- 考虑项目上下文，检查代码与其他模块的兼容性
- **⚠️ 只能分析项目文件结构中实际存在的文件，严禁凭空捏造文件名**
- **🔴 修改代码时，必须使用原文件路径，在原文件上输出修改后的完整代码**
- **🔴 严禁创建新文件来替代修改，除非用户明确说"新建文件"**
- 如果没有发现任何问题，回复："{filename}代码文件不需要修改，没有发现任何明显性错误"
- 如果发现问题，**严格按照以下格式回复，不要添加任何额外内容（包括复杂度分析、详细解释等）**：

发现{{n}}处问题，涉及{{m}}个文件：

---

### 文件：{{文件路径}}
- **第1处**：{{问题描述}}
- **第2处**：{{问题描述}}

修改后的完整代码：
```{{language}}
{{修改后的完整代码}}
```

---

### 文件：{{文件路径}}
- **第1处**：{{问题描述}}
- **第2处**：{{问题描述}}

修改后的完整代码：
```{{language}}
{{修改后的完整代码}}
```

---

### 场景C：用户要求对多个文件每个文件都要审查
- 如果多个文件都有问题，**每个文件单独输出一个代码块**，格式同上
- **⚠️ 每个分析的文件名必须来源于上方的"项目文件结构"列表，不得自行编造**
- **每个文件之间用 `---` 分隔线隔开，每个问题描述之间必须空一行**
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

### 🔴 场景K（默认）：用户只是咨询、讨论、分享截图、描述问题，没有明确要求修改代码
- **这是最常见的情况，也是默认行为！**
- **只进行分析、解释、建议，绝对不要生成任何代码修改块**
- **绝对不要创建新文件或修改文件**
- 当用户发送截图/图片时，他们通常是在展示问题或讨论，不是在要求修改代码
- 用自然语言回复用户的问题，给出分析意见或建议
- 如果用户确实需要修改代码，他们会在后续对话中明确说出"修改"、"改一下"、"帮我改"等

**图片分析思路（按优先级）：**
1. 图片内容是项目目录下某个文件的代码 → 直接在该文件上修改，文件路径使用项目中的真实路径
2. 图片中有明显的 error/报错信息 → 定位报错来源文件，在该文件上修改，使用项目中的真实路径
3. 图片内容无法匹配项目中的任何文件 → 结合用户的语言描述综合考虑，决定如何处理
4. 无论哪种情况，都严禁创建新文件，只能在已有文件上修改

## 重要规则

0. **🔴 最高优先级：默认只分析不修改！用户没有明确说"修改/改/新建/创建/生成"等词时，只做分析解释，严禁输出代码修改块或创建新文件！**
1. **代码块语言标识必须正确**（如 python、javascript、java、cpp 等）
2. **创建/修改文件时必须标注文件路径**：`文件路径: 相对路径/文件名.py`
3. **🔴 修改已有代码时，文件路径必须使用项目中的原文件路径，不能创建新文件路径**
4. **代码块中只包含源代码**，严禁包含测试运行输出、调试信息、交互式会话内容
5. **绝对禁止**输出代码的执行结果（如 `5! = 120`）
6. **绝对禁止**包含"运行示例"、"测试结果"等标题或内容
7. **新增文件时确保包含完整目录路径**（仅在用户明确要求新建文件时）
8. **用户指定目录时必须在文件路径中包含该目录**
9. **保持对话上下文**：记住之前的对话历史，能够回答追问和确认类问题

## 输出格式要求

```
{{问题分析}}

{{解决方案}}

{{代码实现}}
```

请根据用户的具体需求，提供专业、详细、可执行的解决方案。"""

        return system_prompt

    def _read_config_files(self, files: list) -> dict:
        """读取项目中的关键配置文件内容"""
        config_files = ['requirements.txt', 'package.json', 'pom.xml', 'go.mod', 'Cargo.toml', 
                       'setup.py', 'pyproject.toml', 'README.md', 'appsettings.json',
                       'application.yml', 'application.yaml', 'docker-compose.yml']
        
        contents = {}
        for f in files:
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
            file_path = f.get('path', '')
            file_name = f.get('name', '')
            file_ext = Path(file_name).suffix.lower()
            
            logger.debug(f"处理文件: {file_name}, path: {file_path}, ext: {file_ext}")
            
            if file_ext in source_extensions:
                if 'content' in f and f['content']:
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
            file_name = f.get('name', '')
            file_path = f.get('path', '')
            file_ext = Path(file_name).suffix.lower()
            
            if file_ext not in source_extensions:
                continue
            
            # 获取文件内容
            content = ""
            if 'content' in f and f['content']:
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
            path = f['path']
            parts = path.split('/')
            current = dir_structure
            
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {'__type': f.get('type', 'file') if i == len(parts)-1 else 'dir', '__children': {}}
                elif i == len(parts)-1:
                    current[part]['__type'] = f.get('type', 'file')
                current = current[part]['__children']
        
        def format_structure(structure, prefix=""):
            result = ""
            items = sorted(structure.keys())
            for i, name in enumerate(items):
                item = structure[name]
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                
                if item['__type'] == 'dir':
                    result += f"{prefix}{connector}{name}/\n"
                    new_prefix = prefix + ("    " if is_last else "│   ")
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
            return f"**❌ 错误：** 抱歉，当前服务暂时不可用，请稍后重试。"
    
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
        code_blocks = re.findall(r'```(\w+)?\n([\s\S]*?)```', response)
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

    def get_model_response_stream(self, message, llm):
        """获取模型响应（流式）"""
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
        user_message_content = self._build_multimodal_message(message_content)
        is_multimodal = isinstance(user_message_content, list)

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
                response = llm.invoke(vl_prompt_list)
                response_text = response.content if hasattr(response, 'content') else str(response)
                self._update_memory_from_response(message_content, response_text)
                # 分块输出，模拟流式效果
                chunk_size = 80
                for i in range(0, len(response_text), chunk_size):
                    yield response_text[i:i + chunk_size]
                    time.sleep(0.03)
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
    - 🔴 回复过程中严禁输出任何代码块或代码片段！
    - 🔴 回复过程中严禁提及要修改哪些具体文件！
    - 🔴 回复过程中严禁输出完整的实现代码！
    - 🔴 不要列出依赖安装命令！
    - 🔴 保持简洁，每个部分 1-2 句话即可！

    用户问题：{message_content}

    请开始你的思考分析（保持简洁）："""

            # 创建思考专用的 LLM 实例（使用更快的模型）
            llm_other = ChatOpenAI(
                model=Params.DEFAULT_CHAT_MODEL,  # 使用更快的模型进行思考
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url=Params.API_BASE,
                timeout=30,
                streaming=True
            )

            # 使用思考模型（需要你在 agent.py 中创建 llm_other 实例）
            try:
                for chunk in llm_other.stream([{"role": "user", "content": react_prompt}]):
                    if hasattr(chunk, 'content') and chunk.content:
                        content = chunk.content
                        yield content
                        time.sleep(0.03)
            except Exception as e:
                # 如果思考模型不可用，跳过思考阶段
                logger.warning(f"ReAct 思考模型调用失败，跳过思考阶段: {e}")

            # 思考阶段结束标记
            yield "\n\n---\n\n"

            # 主 LLM 处理中提示
            yield "<span class='typing-indicator'> 正在生成回复</span>\n\n"

            # === 正式回复阶段：用主 LLM 输出正式内容 ===
            for chunk in llm.stream(prompt_list):
                if hasattr(chunk, 'content') and chunk.content:
                    content = chunk.content
                    full_response += content  # 累积完整响应
                    code_block_count += content.count('```')
                    in_code_block = code_block_count % 2 == 1
                    
                    if intent_result and not in_code_block:
                        thinking_count += len(content)
                        paragraph_count += content.count('\n')

                        if not in_code_block and (thinking_count > 1000 or (paragraph_count > 10 and thinking_count > 400)):
                            thinking_count = 0
                            paragraph_count = 0
                            thinking_phrases = [
                                "\n\n<!--THINKING_START--><span style='color:#666;font-style:italic'>📝 继续分析中...</span><!--THINKING_END-->\n\n",
                                "\n\n<!--THINKING_START--><span style='color:#666;font-style:italic'>🔍 深入研究中...</span><!--THINKING_END-->\n\n",
                                "\n\n<!--THINKING_START--><span style='color:#666;font-style:italic'>💡 发现关键点...</span><!--THINKING_END-->\n\n",
                                "\n\n<!--THINKING_START--><span style='color:#666;font-style:italic'>⚡ 正在处理...</span><!--THINKING_END-->\n\n",
                            ]
                            yield thinking_phrases[len(conversation_history) % len(thinking_phrases)]
                            time.sleep(0.3)
                    yield content
            
            # 流式响应完成后，更新长期记忆
            self._update_memory_from_response(message_content, full_response)
            
        except Exception as e:
            logger.error(f"project_code_node LLM 流式调用失败: {str(e)}")
            yield "\n\n**❌ 错误：** 抱歉，当前服务暂时不可用，请稍后重试。"

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