# MultiAgent

基于 LangGraph 的多智能体协作系统，支持多种专业技能的 AI 对话助手。

## 功能特性

| 技能 | 图标 | 说明 |
|------|------|------|
| 旅游规划 | 🏙️ | 智能规划旅行路线，如"北京三日游" |
| 笑话 | 😂 | 讲各类笑话，如程序员笑话 |
| 对联 | 📝 | 根据上联智能生成下联 |
| 文档处理 | 📄 | 上传文档，创建、修改、排版和优化文档内容 |
| 编程 | 💻 | 编写代码，修改上传的代码文件 |
| 图纸识别 | 📐 | 上传工程图纸（图片/PDF/DXF），提取尺寸、校核公差、查询国标、单位换算 |

## 快速开始

### 环境要求

- Python 3.10+
- 虚拟环境（推荐）

### 安装

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd MultiAgent

# 2. 创建虚拟环境
python -m venv .venv

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 配置环境变量（创建 .env 文件）
echo DASHSCOPE_API_KEY=your_api_key_here > .env
```

### 启动

```bash
python main.py
```

启动后访问 http://localhost:8000 即可使用。

## 使用指南

### 基本对话

在输入框中直接输入问题，按 Enter 发送。系统会自动识别意图并路由到对应的智能体处理。

### 选择技能

1. 点击输入框下方的 **🛠️ 技能** 按钮
2. 在弹出的面板中选择需要的技能（文档处理 / 编程 / 图纸识别）
3. 选中后输入框上方会显示技能标签，系统将直接使用对应技能处理

### 上传文件

选择技能后，**➕ 上传文件** 按钮会激活：

- **文档处理**：支持 `.docx` `.doc` `.txt` `.ppt` `.pptx` `.xlsx` `.xls`
- **编程**：支持 `.docx` `.doc` `.txt` 等代码文件
- **图纸识别**：支持 `.png` `.jpg` `.jpeg` `.bmp` `.tiff` `.pdf` `.dxf`

### 示例问题

首页提供了快速示例，点击即可发送：

- 🏙️ 北京三日游
- 😂 程序员笑话
- 📝 对联：春回大地

### 历史对话

左侧边栏可以查看和管理历史对话，点击历史记录可以切换会话。

## 项目结构

```
MultiAgent/
├── main.py                  # FastAPI 入口，Web 服务
├── core/
│   ├── agent.py             # 主智能体图，Supervisor 路由 + 反思评估
│   ├── Couplet/             # 对联智能体（RAG 检索）
│   ├── DocProcess/          # 文档处理智能体
│   ├── DrawingRecoAssistant/ # 图纸识别智能体（多模态 + DXF + 国标检索）
│   └── ...
├── templates/
│   └── index.html           # 前端页面
├── settings/
│   ├── Define.py            # 配置参数和路径
│   └── logger_manager.py    # 日志管理
└── .env                     # 环境变量（API Key 等）
```

## 技术栈

- **框架**：FastAPI + LangGraph
- **LLM**：DashScope（通义千问）
- **向量数据库**：Chroma
- **多模态**：支持图片/PDF 图纸识别
- **CAD 解析**：ezdxf（DXF 矢量图纸）