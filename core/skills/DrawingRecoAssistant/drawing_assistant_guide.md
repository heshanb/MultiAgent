# DrawingAssistant 图纸识别助手 — 技术文档

## 一、概述

`drawing_assistant.py` 封装了图纸识别助手的全部功能，以 `DrawingAssistant` 类的形式对外提供 API。支持多模态识图（图片/PDF）、DXF 矢量解析、尺寸校核、国标检索、单位换算等能力，内部基于 LangGraph 构建多节点工作流。

---

## 二、工作流程总览

```
用户输入: 问题 + 图纸文件(图片/PDF/DXF)
              │
              ▼
┌─────────────────────────────────────────────────┐
│                 invoke() 入口                     │
│  1. 图片 → 预处理(锐化) → base64                  │
│  2. 判断 file_type: img / pdf / dxf              │
│  3. 调用 graph.invoke()                          │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  条件入口路由    │
              │ _route_condition│
              └───┬───┬───┬───┬┘
                  │   │   │   │
      ┌───────────┘   │   │   └──────────────┐
      ▼               ▼   ▼                  ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ ① reset  │  │ ② parse      │  │ ③ check      │  │ ④ call_tools     │
│ 重置节点  │  │ 解析节点      │  │ 尺寸校核      │  │ 工具调用          │
│          │  │              │  │              │  │                  │
│ 新图纸→  │  │ DXF: 提取    │  │ 尺寸链闭合   │  │ 单位换算          │
│ 清空历史  │  │  版本/图层/  │  │ 公差校验     │  │ 国标检索          │
│          │  │  尺寸标注    │  │ 单位冲突     │  │ DXF解析           │
│          │  │ 图片: 视觉   │  │              │  │                  │
│          │  │  LLM识图     │  │              │  │                  │
│          │  │ → 存入向量库 │  │              │  │                  │
└────┬─────┘  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
     │               │                │                   │
     └───────────────┴────────────────┴───────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ ⑤ final_answer   │
                    │ 最终回答节点       │
                    │                  │
                    │ 汇总所有解析结果   │
                    │ 输出结构化报告:    │
                    │  • 图纸信息       │
                    │  • 尺寸分析       │
                    │  • 国标依据       │
                    │  • 加工建议       │
                    └────────┬─────────┘
                             │
                             ▼
                           END
```

### 五个节点职责

| 节点 | 职责 | 何时触发 |
|------|------|---------|
| ① reset | 新图纸时清空历史数据 | `is_new_drawing=True` |
| ② parse | DXF解析 + 多模态识图 → 存入向量库 | 没有历史解析数据时 |
| ③ check | 尺寸链闭合检查、公差校验、单位冲突 | 用户问"尺寸/公差/校核" |
| ④ call_tools | LLM 自主调用换算/检索/DXF解析工具 | 用户问"换算/国标/螺纹" |
| ⑤ final_answer | 汇总所有信息，输出结构化报告 | 默认路径或①②③④之后的终点 |

### 路由逻辑

- 新图纸 → 重置 → 解析 → 最终回答
- 追问尺寸 → 校核 → 最终回答
- 追问国标/换算 → 工具调用 → 最终回答
- 其他追问 → 直接最终回答

### 核心设计思想

**解析一次，多轮复用**：图纸解析结果 `drawing_full_info` 保存在 LangGraph 状态中，同一会话内多次追问不会重复解析，直接从状态中取数据，通过检查点 `MemorySaver` 实现多轮对话记忆。

---

## 三、关键代码详解

### 3.1 `RecursiveCharacterTextSplitter` 参数

```python
self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
```

#### `chunk_size=600` — 每个文本块的最大字符数

把长文本切分成若干个小块，**每块不超过 600 个字符**。

切分优先级：先按段落 `\n\n` 切 → 再按换行 `\n` 切 → 再按空格切 → 最后按字符硬切。所以实际每块可能略小于 600，但不会超过。

#### `chunk_overlap=120` — 相邻块之间的重叠字符数

相邻两个 chunk 之间会**共享 120 个字符**。

```
原文:  [AAAAAABBBBBBCCCCCCDDDDDDEEEEEE]
                   ↓ 切分
chunk1: [AAAAAABBBBBBCCCC]           ← 600字符
chunk2:         [BBBBBBCCCCCCDDDDDD] ← 600字符，与chunk1重叠120
chunk3:                 [CCCCCCDDDDDDEEEEEE]
```

#### 为什么需要 overlap？

| 没有 overlap 的问题 | 有 overlap 的好处 |
|---|---|
| 关键信息可能刚好被切断在边界上 | 相邻块共享上下文，检索时不会漏掉边界信息 |
| 比如"螺纹公差等级为 **6g**" 刚好被切成两段 | 重叠确保边界附近的语义完整 |

#### 参数选择建议

| 场景 | chunk_size | chunk_overlap |
|------|-----------|---------------|
| 短文/FAQ | 200~400 | 40~80 |
| 国标/规范文档 | 500~800 | 100~200 |
| 长文档/论文 | 800~1500 | 150~300 |

当前设置 `600/120` 适合国标知识库：每条国标条目通常几百字，20% 的重叠率（120/600）是常见实践。

---

### 3.2 `unit_convert` — 单位换算工具

```python
@tool
def unit_convert(value: float, unit_from: str, unit_to: str) -> str:
    """图纸单位换算 mm/cm/m/inch"""
    unit_base = {"mm": 1, "cm": 10, "m": 1000, "inch": 25.4}
    uf = unit_from.lower().strip()
    ut = unit_to.lower().strip()
    base_mm = value * unit_base[uf]
    res = base_mm / unit_base[ut]
    return f"换算结果：{value} {unit_from} = {res:.4f} {unit_to}"
```

#### 逐行解释

**`@tool`** 装饰器会把这个函数注册为 LangChain 工具，LLM 看到 description 后就知道什么时候该调用它。

**基准换算表**：所有单位先统一换算成毫米（mm），因为毫米是工程图纸的最小常用单位。

| 单位 | 相对 mm 的值 | 含义 |
|------|-------------|------|
| `mm` | 1 | 基准单位 |
| `cm` | 10 | 1cm = 10mm |
| `m` | 1000 | 1m = 1000mm |
| `inch` | 25.4 | 1英寸 = 25.4mm |

**归一化输入**：用户或 LLM 可能传入 `"MM"`、`" Inch "` 等，统一转小写去空格，确保字典键能匹配。

**两步换算**：以 mm 为中间桥梁。

```
输入值 × 输入单位(mm系数) → 毫米数 → ÷ 目标单位(mm系数) → 输出值
```

举例：`unit_convert(5, "cm", "inch")`
```
5 cm × 10 = 50 mm → 50 ÷ 25.4 = 1.9685 inch
```

**`{res:.4f}`** 保留 4 位小数，工程精度要求。

#### 整个调用链路

```
用户: "把15mm换算成英寸"
  ↓
LLM 看到 @tool description → 决定调用 unit_convert
  ↓
LLM 生成: unit_convert(value=15, unit_from="mm", unit_to="inch")
  ↓
函数执行 → 返回 "换算结果：15 mm = 0.5906 inch"
  ↓
LLM 把结果整合进最终回答
```

#### 设计要点

| 设计 | 原因 |
|------|------|
| 以 mm 为基准 | 只需一张表，不用维护 4×4=16 种两两换算 |
| `lower().strip()` | 防御性编程，容错 LLM 传参不规范 |
| `:.4f` | 工程图纸通常精确到 0.0001 |
| `@tool` 而非纯函数 | 让 LLM 自主决定何时调用，而不是硬编码调用时机 |

---

### 3.3 `parse_dxf_file` — DXF 图纸解析工具

```python
@tool
def parse_dxf_file(dxf_path: str) -> str:
    """解析DXF矢量图纸，提取图层、尺寸标注、图框、标题栏信息"""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    info = []
    info.append(f"DXF版本：{doc.dxfversion}")
    info.append(f"图纸单位：{doc.units}")
    dim_texts = []
    for dim in msp.query("DIMENSION"):
        if dim.dxf.text:
            dim_texts.append(dim.dxf.text)
    info.append(f"提取尺寸标注列表：{list(set(dim_texts))}")
    layers = [layer.dxf.name for layer in doc.layers]
    info.append(f"图纸图层：{layers}")
    return "\n".join(info)
```

#### 逐行解释

| 代码 | 说明 |
|------|------|
| `ezdxf.readfile(dxf_path)` | 读取 DXF 文件 |
| `doc.modelspace()` | 获取模型空间（图纸主体），区别于 paperspace（打印布局） |
| `doc.dxfversion` | DXF 版本号，如 `"R2018"` |
| `doc.units` | 返回整数代码（INSUNITS），如 `4=mm`、`1=inch` |
| `msp.query("DIMENSION")` | 查询所有尺寸标注实体（如 ⌀10、M8、50±0.1） |
| `set(dim_texts)` | 去重，同一尺寸可能标注多次 |
| `doc.layers` | 遍历所有图层（如 `0`、`DIMENSIONS`、`HIDDEN`、`TITLE_BLOCK`） |

#### 返回示例

```
DXF版本：R2018
图纸单位：4
提取尺寸标注列表：['50', '30', '⌀10', 'M8']
图纸图层：['0', 'DIMENSIONS', 'HIDDEN']
```

---

### 3.4 `bind_tools` — 让 LLM 学会调用工具

```python
self.tools = [unit_convert, search_engineering_standard, parse_dxf_file]
self.llm_with_tools = self.llm_text.bind_tools(self.tools)
```

#### `bind_tools()` 做了什么？

把三个 Python 函数的签名和 description 注入到 LLM 的调用上下文里，LLM 就能**自主决定**何时调用哪个工具。

```
bind_tools 之前：
  LLM 只知道如何输出文本

bind_tools 之后：
  LLM 知道有三个工具可用，需要时可以输出 function_call 而不是文本
```

#### 实际效果

LLM 看到用户问"解析这张 DXF 的尺寸"，它不会自己瞎编，而是：

```
LLM 输出: { "tool_call": "parse_dxf_file", "args": { "dxf_path": "/path/to/file.dxf" } }
  ↓
框架拦截这个 tool_call → 真正执行 parse_dxf_file() → 拿到结果
  ↓
LLM 看到结果后 → 用自然语言组织最终回答
```

#### 三个工具的分工

| 工具 | 触发场景 | LLM 判断依据 |
|------|---------|-------------|
| `unit_convert` | 用户问"15mm 等于多少 inch" | 看到 `mm/cm/m/inch` 关键词 |
| `search_engineering_standard` | 用户问"M8 螺纹国标是什么" | 看到"国标""标准""规范"关键词 |
| `parse_dxf_file` | 用户上传了 DXF 文件 | 看到 `dxf_path` 参数可用 |

**`bind_tools` 的核心价值**：不用写 `if "换算" in query: call unit_convert()` 这种硬编码判断，LLM 自己理解语义后决定调用哪个工具——这就是 **Tool Calling / Function Calling** 的核心机制。

---

### 3.5 `preprocess_image` — 图片预处理（图像增强）

```python
@staticmethod
def preprocess_image(pil_img: Image.Image) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return Image.fromarray(sharp)
```

#### 处理管道

```
原始图纸(PIL) → RGB→BGR → 灰度化 → 高斯模糊 → USM锐化 → 返回PIL
     ↓                          ↓                    ↓
  可能有噪点、                 去除噪点、           线条更清晰、
  颜色偏暗、                  统一亮度、           标注更易读
  线条模糊                    保留结构
```

#### 逐行解释

| 步骤 | 代码 | 说明 |
|------|------|------|
| 格式转换 | `cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)` | PIL(RGB) → OpenCV(BGR) |
| 灰度化 | `cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)` | 工程图纸颜色无意义，转灰度提速 |
| 高斯模糊 | `cv2.GaussianBlur(gray, (3, 3), 0)` | 3×3 核轻量去噪，只去噪不变形 |
| USM锐化 | `cv2.addWeighted(gray, 1.5, blur, -0.5, 0)` | 锐化 = 原图×1.5 + 模糊图×(-0.5) |
| 格式转回 | `Image.fromarray(sharp)` | OpenCV → PIL，下游 LangChain 接受 PIL 格式 |

#### USM 锐化公式

```
锐化 = 原图 × 1.5 + 模糊图 × (-0.5) + 0

等价于：原图 + (原图 - 模糊图) × 0.5
         ↑       ↑──────────────↑
      原始信号    高频细节（边缘）
```

把模糊掉的边缘信息"加回来"，让线条更锐利。

#### 为什么需要预处理？

| 不做预处理 | 做了预处理 |
|-----------|-----------|
| 拍照图纸有噪点、阴影，LLM 可能误读 | 去噪后线条干净，LLM 识别率更高 |
| 铅笔线条灰度浅，和背景对比度低 | 锐化后线条加粗加黑，对比度提升 |
| 彩色水印/底纹干扰识别 | 灰度化后统一处理，忽略颜色干扰 |

简而言之：**把一张"拍得不太好的图纸照片"变成"扫描仪级别的清晰图"**，让多模态模型更容易读取尺寸和标注。

---

## 四、初始化流程

```python
class DrawingAssistant:
    def __init__(self):
        self._init_llms()          # 初始化视觉LLM + 文本LLM
        self._init_vector_store()  # 初始化 Chroma 向量库 + 文本分割器
        self._init_tools()         # 定义工具 + bind_tools
        self._init_graph()         # 构建 LangGraph 工作流

    @staticmethod
    def get_instance():
        """单例模式：全局唯一实例，避免重复初始化 LLM 和向量库"""
        if DrawingAssistant._instance is None:
            DrawingAssistant._instance = DrawingAssistant()
        return DrawingAssistant._instance
```

**单例模式**：`get_instance()` 确保全局只有一个 `DrawingAssistant` 实例，避免重复初始化 `ChatOpenAI`（LLM 连接）和 `Chroma`（向量库连接）。

---

## 五、对外 API

```python
def invoke(
    self,
    user_query: str,
    img_path: str | None = None,
    pdf_path: str | None = None,
    dxf_path: str | None = None,
    thread_id: str = "draw_chat_001",
) -> str:
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `user_query` | `str` | 用户问题（必填） |
| `img_path` | `str \| None` | 图片文件路径 |
| `pdf_path` | `str \| None` | PDF 文件路径 |
| `dxf_path` | `str \| None` | DXF 文件路径 |
| `thread_id` | `str` | 会话 ID，同一图纸多轮对话使用相同 ID |

### 调用示例

```python
from core.skills.DrawingRecoAssistant.drawing_assistant import DrawingAssistant

assistant = DrawingAssistant.get_instance()

# 首次调用：上传图纸
result = assistant.invoke(
    user_query="分析这张图纸的尺寸",
    dxf_path="uploads/part.dxf",
    thread_id="session_001",
)

# 追问：同一 thread_id，复用解析结果
result = assistant.invoke(
    user_query="M8螺纹的公差是多少？",
    thread_id="session_001",
)
```

---

## 六、与 agent.py 的集成

`agent.py` 中的 `drawing_node` 委托调用 `DrawingAssistant`：

```python
def drawing_node(state: State) -> State:
    assistant = DrawingAssistant.get_instance()
    result = assistant.invoke(
        user_query=state["messages"][-1].content,
        img_path=img_path,
        pdf_path=pdf_path,
        dxf_path=dxf_path,
        thread_id="agent_draw_session",
    )
    return {"messages": [AIMessage(content=result)]}
```

### 集成数据流

```
用户选择"图纸识别"技能 → 上传图纸文件 → 输入问题
    ↓
supervisor_node → skill="drawing" → 直接路由到 drawing_node
    ↓
drawing_node → DrawingAssistant.invoke()
    ↓
reflection_node → 评估回答质量 → PASS 或 FAIL 重试
    ↓
返回图纸分析结果
```