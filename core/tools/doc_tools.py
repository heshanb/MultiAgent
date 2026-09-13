import os
import uuid
import shutil
import asyncio
import contextvars
from pathlib import Path
from typing import Optional

from settings.Define import Params
from settings.logger_manager import get_logger

# 懒加载重型库，避免模块导入时阻塞
_langchain_tool = None
_ChatOpenAI = None


def _get_langchain_tool():
    global _langchain_tool
    if _langchain_tool is None:
        from langchain_core.tools import tool as _tool
        _langchain_tool = _tool
    return _langchain_tool


logger = get_logger(__name__)

_tool_progress_queue: contextvars.ContextVar[Optional[asyncio.Queue]] = contextvars.ContextVar(
    "_tool_progress_queue", default=None
)


def set_progress_queue(q: asyncio.Queue | None):
    _tool_progress_queue.set(q)


async def _report_progress(msg: str):
    q = _tool_progress_queue.get()
    if q is not None:
        try:
            await q.put(msg)
        except Exception:
            pass

ALLOWED_OUTPUT_FORMATS = {".docx", ".pdf", ".md", ".xlsx", ".pptx", ".txt", ".json"}

_MULTI_FILE_MARKER = "===== 文件"


def _detect_multi_file(content: str) -> tuple[bool, int]:
    count = content.count(_MULTI_FILE_MARKER)
    return count >= 2, count


def _collect_multi_file_names(content: str) -> list[str]:
    import re
    return re.findall(r"===== 文件\d+：(.+?) =====", content)


_SYSTEM_PROMPT = """你是一个文档处理助手，需要帮用户选择合适的文档处理工具，并最终用自然友好的聊天语气回复用户。

# 多文件说明

用户可能同时上传多个文件（如同时上传 .docx 和 .xlsx），这种情况仍然走同一个工具，系统会把所有文件的内容拼接在一起传给工具。你在参数里只需要传递第一个文件的路径（系统会自动处理全部文件），file_content 参数会自动包含所有文件拼接后的完整内容。

# 上下文引用处理（重要）

用户可能在多轮对话中引用上一轮生成的文档，例如：
- "对于生成的这个文档，能不能优化调整下格式"
- "刚才生成的文件，再帮我改改"
- "合并后的文档，排版能更好看吗"

**当你看到这类引用时：**
1. 对话历史中会包含上一轮 LLM 的回复（包含"下载文档"链接）和工具执行结果（包含生成的文件路径）
2. 你需要从对话历史中识别出**上一轮生成的文件路径**
3. 调用工具时，`file_path` 参数填**生成的文件路径**，`file_content` 参数填**生成的文件内容**（从对话历史的 tool 消息中获取）
4. 如果对话历史中有多个文件路径，优先使用**最近一次工具执行结果中的输出文件路径**

**示例：**
- 第一轮：用户上传 A.docx + B.xlsx → 调用 modify_existing_document 合并 → 生成 C_merged.docx
- 第二轮：用户说"对于生成的这个文档，优化下格式" → 你应该用 C_merged.docx 作为 file_path，而不是 A.docx 或 B.xlsx

# 核心规则

你需要先做**语义判断**：用户这条消息的核心意图是什么？

- **围绕文档的处理/创建/查询需求** → 必须调工具
  - 明显信号词：错别字、整理、修改、润色、分析、汇总、总结、统计、重写、归纳、提取、翻译、生成、创建、写一份、文档
  - 只要提到了**/文档/文件/格式/保存/输出/导出/另存为**，哪怕内容描述短或模糊，也是文档需求
  - 例："保存到excel"、"转成markdown"、"输出成PDF"、"汇总存成表格"、"文档内容" → 都是文档需求
  - 例："说说这些文档讲了什么"、"用通俗的话说说"、"介绍一下这个文件"、"讲了什么" → 都是文档查询需求 → 必须调 read_document_and_answer
- **和文档无关的闲聊** → 不调工具，直接聊天回复
  - 只包含：夸奖、感谢、打招呼、闲聊话题、"好的"/"行"/"嗯嗯"这类纯应答、纯追问语义（"你说的对"、"同意"、"可以"）

⚠️ **强制规则：有文件路径 + 用户请求里出现了"说说/讲讲/介绍/讲了什么/总结/摘要/概括/分析/看看/检查/提取/整理/修改/翻译/转成/保存"中任何一个词 → 100% 必须调工具，绝对不能直接回复！**

只有同时满足以下两个条件才允许不调工具：
1. 没有文件路径，或者用户请求里完全没提到任何文档/文件相关的意图
2. 用户说的是纯闲聊（谢谢/好的/你好/夸奖/闲聊话题）

反例（这些**必须**调工具，不能直接回复）：
❌ 有文件 + "说说讲了什么" → 直接回复"好的我来介绍"（错误！你没有文件内容，调 read_document_and_answer）
❌ 有文件 + "总结一下" → 直接回复"以下是总结..."（错误！你没有文件内容，调 read_document_and_answer）
❌ 有文件 + "看看有没有错别字" → 直接回复"好的我来检查"（错误！调 modify_existing_document）

# 如何选择工具

```
第一步：用户消息的语义是不是围绕文档处理？
（文档处理 = 错别字/整理/修改/润色/分析/汇总/重写/排版/格式转换/新建文档/文档 等）
├─ 是 → 继续判断用哪个工具
└─ 否 → 闲聊，不调任何工具，直接聊天回复

第二步：选哪个工具（只有用户有文档处理需求时才走这里）
    有附件文件路径吗？
    ├─ 有 →
    │   ├─ 用户需求是"读取/讲解/总结/摘要/介绍/说说文档内容"（不要求生成新文件）
    │   │   → read_document_and_answer（只读内容，聊天回复，不保存文件）
    │   │   · 关键词：总结、摘要、概括、讲解、说说、介绍、讲了什么、内容、含义、分析一下、解读、看懂、简述、简要、大概
    │   │   · 如果用户只问"XX文档讲了什么"、"总结下这个文件"、"用通俗的话说说" → 这个工具
    │   │   · 如果用户顺带要"保存/导出/输出成XX格式" → 走 modify_existing_document 或 convert_file_format
    │   │
    ├─ 用户只说格式转换（"把/转成/另存为/导出为 X 格式"），且不提任何内容处理需求（没有整理/修改/纠错/润色等）
    │   ├─ 如果只有 1 个文件 → convert_file_format（纯格式转换，不动内容）
    │   └─ 如果有多个文件 → modify_existing_document（多文件需要先合并再转换）
    │   │
    │   └─ 其他一切情况 → modify_existing_document，包括但不限于：
    │       · 有错别字/错字/纠错/拼写检查
    │       · 要整理/梳理/归纳/汇总/统计（且可能要求保存）
    │       · 要修改/改动/调整/重写/改写/重构
    │       · 要润色/优化/美化/排版/格式调整
    │       · 要翻译/补充/完善/扩展
    │       · 要生成图表/提取数据
    │       · 再检查/确认下/对吗/有没有问题
    │       · 同时提到了内容处理 + 格式转换（如"整理下内容，输出成 markdown"）
    │       （如果用户顺带提了格式，output_format 参数填目标格式即可）
    └─ 没有 → create_new_document（用户要你从零写一份新文档）
```

# 三种"文档内容问答"场景的判定（read_document_and_answer）

当用户上传了文件，且消息意图是以下任何一种时，**必须用 read_document_and_answer**，不要用 modify_existing_document：

1. **讲解/介绍型**：想快速了解文档是什么、讲了什么
   - "说说这些文档讲了什么"
   - "介绍一下这个文件"
   - "用通俗的话说说"
   - "这个文档主要讲什么"

2. **总结/摘要型**：想要浓缩后的简短内容
   - "详细总结这些文档内容"
   - "生成一个简短摘要"
   - "帮我概括一下"
   - "简要说明重点"
   - "提炼核心要点"

3. **分析/解读型**：想理解特定方面
   - "分析一下这份数据"
   - "解读这份报告"
   - "从这份文档里能看出什么"
   - "帮我看懂这个文件"

**关键区分**：用户如果只想要"说一下/讲一下/总结一下" → `read_document_and_answer`
                用户如果明确要求"整理成/保存成/导出成/输出成 XX 文件" → `modify_existing_document`
# 多文件场景的特殊规则（非常重要！）

当用户上传了**多个文件**时：

- **任何涉及"合并/整理/都/全部/一起"的请求** → 必须用 `modify_existing_document`
  - 例："都整理写到pdf文件保存吧" → modify_existing_document
  - 例："把这几个文件合并成一个word" → modify_existing_document
  - 例："全部转成pdf" → modify_existing_document
  - 例："整理下这些内容，合并保存到一个word文档中" → modify_existing_document

- **纯格式转换且只有1个文件** → 才用 `convert_file_format`
  - 例："把这个ppt转成pdf"（只有1个文件） → convert_file_format

⚠️ **多文件场景下，绝对不要用 convert_file_format！这个工具只能处理单个文件！**
# 示例

✅ 有文件路径 + "详细总结这些文档内容" → 总结，不要求保存 → 调 read_document_and_answer
✅ 有文件路径 + "用通俗的话说说讲了什么" → 讲解，不要求保存 → 调 read_document_and_answer
✅ 有文件路径 + "生成一个简短摘要" → 摘要，不要求保存 → 调 read_document_and_answer
✅ 有文件路径 + "帮我提炼一下要点" → 总结，不要求保存 → 调 read_document_and_answer

✅ 有文件路径 + "看看有没有错别字" → 有错别字/纠错 → 调 modify_existing_document
✅ 有文件路径 + "对吗，再检查下" → 纠错 → 调 modify_existing_document
✅ 有文件路径 + "整理成 markdown 保存" → 明确要求保存 → 调 modify_existing_document
✅ 有文件路径 + "转换成 pdf" → 纯格式转换 → 调 convert_file_format
✅ 有文件路径 + "汇总存成表格文件" → 明确要求保存 → 调 modify_existing_document
✅ 有文件路径 + "你好牛啊" → 语义是夸奖闲聊 → 不调工具，直接回复
✅ 有文件路径 + "谢谢" → 语义是感谢闲聊 → 不调工具，直接回复

✅ 没有文件路径 + "帮我写会议纪要" → 语义围绕文档 → 调 create_new_document
✅ 没有文件路径 + "你太厉害了" → 闲聊 → 不调工具
✅ 没有文件路径 + "汇总保存到excel文件中" → 提到了"保存"+"excel文件" → 文档需求 → 调 create_new_document

 有文件路径 + "总结一下" → 调 modify_existing_document（错误！总结不保存，应该用 read_document_and_answer）
❌ 有文件路径 + "讲了什么" → 调 modify_existing_document（错误！讲解不保存，应该用 read_document_and_answer）
❌ 有文件路径 + "看看有没有错别字" → 直接回复"好的我来帮你检查"（错误，语义围绕文档，必须调工具！）
❌ "汇总保存到excel" → "请提供具体内容"（错误！这是文档需求，应该调 create_new_document 让工具自己生成内容）

# 关于最终回复的语气（非常重要）

- **像人跟人说话一样**，不要用"执行成功"、"执行完毕"这类技术报告式措辞
- 正确示例："好的，我帮你检查了一遍这份文档，没有发现错别字哦 "
- 正确示例："已帮你重新检查了一下，发现了一些小问题，帮你修正了，你可以下载查看"
- 错误示例："执行成功：检查完毕，未发现错别字"
- 错误示例："✅ 工具执行完成，耗时 9.1s"

# 参数填写注意事项

## read_document_and_answer 的 user_request 参数

原样传递用户原话，工具内部会让 LLM 直接基于文档内容回答。
- 用户说"详细总结这些文档内容" → user_request 就填 "详细总结这些文档内容"
- 用户说"用通俗的话说说讲了什么" → user_request 就填 "用通俗的话说说讲了什么"

## modify_existing_document 的 user_request 参数（重要）

这个参数决定工具内部怎么处理，**必须原样传递用户的原话，不要自作聪明改写**：
- 用户说"帮我校对下有没有错别字" → user_request 就填 "帮我校对下有没有错别字"
- 用户说"检查下这个 Excel 里的拼写错误" → user_request 就填 "检查下这个 Excel 里的拼写错误"
- 用户说"整理下文档内容，输出到 markdown 文件中保存" → user_request 就填原话

工具会自动识别"错别字"、"错字"、"纠错"、"拼写错误"这些关键词来决定是否做错别字检查。如果你把用户的原话改成模糊表述，工具就无法正确判断了。

## output_format 参数

用户明确说了目标格式就填对应值，没说就保持原格式。

# 格式识别速查

.xlsx/.xls/表格/Excel → Excel
.docx/.doc/word → Word
.pdf → PDF
.md/markdown → Markdown
.pptx/.ppt/幻灯片/PowerPoint → PowerPoint
.txt/纯文本 → 纯文本
.json → JSON
"""


def _get_format_guidance(output_ext: str) -> str:
    """根据目标输出格式返回对应的 LLM 输出指引（用于 tool 内部的 prompt 构建）"""
    if output_ext in {".docx", ".pdf", ".md", ".txt"}:
        return (
            "2. 使用 Markdown 结构组织内容（标题用 # / ## / ###，列表用 - 或 1.）\n"
            "3. 表格类数据用 Markdown 表格：第一行写有实际含义的表头（如 | 门店名称 | 地址 |），第二行是对齐分隔线（| --- | --- |），之后每行为数据"
        )
    if output_ext in {".xlsx"}:
        return (
            "2. 每一行代表 Excel 中的一行数据，用 | 分隔各列\n"
            "3. 第一行必须是实际列名作为表头"
        )
    if output_ext in {".pptx"}:
        return (
            "2. 用 ## 表示每一页幻灯片的标题，其下方是该页内容\n"
            "3. 每页内容用 Markdown 格式组织（列表、表格等）\n"
            "4. 保持页面数量精简，重点突出"
        )
    if output_ext in {".json"}:
        return (
            "2. 输出合法的 JSON 格式内容（不要用 Markdown 包裹，也不要加 ```json``` 代码块标记）\n"
            "3. 所有键和字符串值必须用双引号，整个输出必须是完整可解析的 JSON\n"
            "4. 根据数据特征选择合适的 JSON 结构（对象 {} 或数组 []）"
        )
    return "2. 使用清晰的文本结构组织内容"


def _infer_output_format(output_hint: str, fallback: str = ".docx") -> str:
    hint = (output_hint or "").lower().strip()
    mapping = {
        "pdf": ".pdf", "docx": ".docx", "doc": ".docx", "word": ".docx",
        "md": ".md", "markdown": ".md",
        "xlsx": ".xlsx", "xls": ".xlsx", "excel": ".xlsx", "表格": ".xlsx",
        "pptx": ".pptx", "ppt": ".pptx", "powerpoint": ".pptx", "幻灯片": ".pptx",
        "txt": ".txt", "文本": ".txt", "纯文本": ".txt",
        "json": ".json",
    }
    if hint in mapping:
        return mapping[hint]
    for fmt in ALLOWED_OUTPUT_FORMATS:
        if hint.endswith(fmt):
            return fmt
    return fallback


def _make_result(ok: bool, message: str, download_url: str = "") -> str:
    if ok and download_url:
        base = message.strip() if message else "执行成功"
        return f"{base}\n\n[下载文档]({download_url})"
    if ok:
        return f"执行成功：{message}"
    return f"执行失败：{message}"


def get_doc_system_prompt() -> str:
    return _SYSTEM_PROMPT


def create_doc_tools(thread_id: str = ""):
    from core.skills.DocProcess.document_process import doc_processor
    tool = _get_langchain_tool()

    @tool
    async def create_new_document(content_description: str, doc_type_hint: str, output_format: str) -> str:
        """根据用户描述创建一份全新的文档文件。用于用户没有上传文件、只说"帮我写一份XX"的场景。

        Args:
            content_description: 详细描述文档应该写什么内容，比如"一份季度销售总结报告，包含各区域业绩对比"
            doc_type_hint: 文档类型提示，比如"报告"、"会议纪要"、"合同"、"简历"、"演讲稿"
            output_format: 期望输出的文件格式，可选值: pdf, docx, md, xlsx, pptx, txt, json
        """
        try:
            output_ext = _infer_output_format(output_format, ".docx")
            fmt_guidance = _get_format_guidance(output_ext)

            system_prompt = (
                f"你是专业的文档创作助手。\n"
                f"文档类型：{doc_type_hint}\n"
                f"输出格式：{output_ext}\n"
                f"要求：\n"
                f"1. 根据描述生成结构完整、内容丰富、专业准确的文档\n"
                f"{fmt_guidance}\n"
                f"你生成的内容将直接被保存为 {output_ext} 文件"
            )
            user_prompt = f"文档创作描述：{content_description}"

            await _report_progress(f"准备创建 {output_ext} 文档，类型：{doc_type_hint}")
            full_content = await _llm_generate(system_prompt, user_prompt)
            if not full_content or not full_content.strip():
                return _make_result(False, "AI 无法生成有效的文档内容")

            original_filename = f"{doc_type_hint or 'document'}_{uuid.uuid4().hex[:8]}"
            output_filename = f"{original_filename}{output_ext}"
            output_path = doc_processor.OUTPUT_DIR / output_filename

            await _report_progress(f"正在保存文件：{output_filename}")
            await asyncio.to_thread(doc_processor._save_content_to_format, full_content, str(output_path), output_ext)

            if not output_path.exists():
                return _make_result(False, "文件保存失败")

            download_url = f"http://127.0.0.1:5001/download/{output_filename}"
            logger.info(f"[doc_tool] create_new_document -> {output_filename}")
            return _make_result(True, "", download_url=download_url)

        except Exception as e:
            logger.error(f"[doc_tool] create_new_document error: {e}")
            return _make_result(False, str(e)[:200])

    @tool
    async def modify_existing_document(
        file_path: str,
        file_content: str,
        user_request: str,
        output_format: str,
    ) -> str:
        """对用户上传的文档内容进行修改、整理、汇总、分析、纠错、润色或重写。

        Args:
            file_path: 原始文件的完整路径，用于识别文件扩展名和保留部分格式
            file_content: 已提取的原始文档文本内容（Markdown 格式）
            user_request: 用户具体的修改/整理/分析/纠错要求
            output_format: 期望输出的文件格式，可选值: pdf, docx, md, xlsx, pptx, txt, json。填原格式表示只改不改格式
            thread_id: 会话 ID，用于保存生成文档的下载链接到会话记忆
        """
        try:
            from pathlib import Path as _P

            original_ext = _P(file_path).suffix.lower() if file_path else ".txt"
            output_ext = _infer_output_format(output_format, original_ext)
            original_filename = _P(file_path).stem if file_path else "document"

            is_multi, multi_count = _detect_multi_file(file_content)
            multi_file_names = _collect_multi_file_names(file_content) if is_multi else []

            if is_multi:
                original_filename = "多文档合并"
                original_ext = "多格式"

            is_format_conversion = (output_ext != original_ext) or is_multi
            is_typo = any(kw in user_request for kw in ["错别字", "错字", "纠错", "拼写错误"])

            system_prompt_parts = [
                f"你是专业的文档处理助手，源文件格式：{original_ext}",
                f"目标输出格式：{output_ext}",
                "",
                "注意：",
                "- 你的输出会直接保存为目标文件，只能包含文档实际内容，不能包含任何解释、说明、用户需求描述",
                "- 不要把用户的请求（如'整理下文档内容'、'转换成md格式'）当作标题或正文写进文档",
                f"",
                "用户需求：",
                user_request,
                "",
                "要求：",
            ]

            # 只有 word、md、PDF 格式才需要 Markdown 格式要求
            if output_ext in ['.docx', '.md', '.pdf']:
                system_prompt_parts.extend([
                    "",
                    "## 输出格式要求（非常重要！必须严格遵守）",
                    "你必须输出标准 Markdown 格式，这样生成的 Word 文档才会有清晰的层级和排版：",
                    "",
                    "### 1. 标题层级（必须用 # 符号）",
                    "- 一级标题（最大）：`# 标题文字`（用于文档大标题、'第一部分'等）",
                    "- 二级标题：`## 标题文字`（用于'一、二、三'等章节标题）",
                    "- 三级标题：`### 标题文字`（用于小节标题）",
                    "- 四级标题：`#### 标题文字`（用于更细分的小节）",
                    "- 示例：`## 一、什么是爬虫`（标题中的核心词要加粗）",
                    "",
                    "### 2. 加粗强调（必须用 `文字`）",
                    "- 标题中的核心词要加粗，如：`## 一、什么是爬虫`",
                    "- 列表项中的关键词要加粗，如：`1. 发起请求（Request）：爬虫向目标网站...`",
                    "- 段落中的关键概念要加粗，如：`请求网站并提取数据的自动化程序`",
                    "",
                    "### 3. 列表格式",
                    "- 有序列表：使用 `1. 2. 3. 4.` 数字加点号，后面必须有空格",
                    "- 无序列表：使用 `-` 或 `*` 开头，后面必须有空格",
                    "",
                    "### 4. 表格格式",
                    "- 使用标准 Markdown 表格语法，如：",
                    "  ```",
                    "  | 列1 | 列2 |",
                    "  |------|------|",
                    "  | 值1 | 值2 |",
                    "  ```",
                    "",
                    "### 5. 分隔线",
                    "- 不同部分之间使用 `---` 分隔",
                    "",
                    "### 6. 段落",
                    "- 普通段落直接写文字，段落之间用空行分隔",
                    "- 段落中的关键概念直接加粗",
                    "",
                    "## 格式示例（请严格参照此格式输出）",
                    "```markdown",
                    "# 第一部分：爬虫基本原理",
                    "",
                    "## 一、什么是爬虫",
                    "",
                    "请求网站并提取数据的自动化程序。简单来说，就是让程序自动去网站上抓取我们需要的信息。",
                    "",
                    "## 二、爬虫基本流程",
                    "",
                    "爬虫的基本工作流程如下：",
                    "",
                    "1. 发起请求（Request）：爬虫向目标网站发送请求，就像浏览器访问网页一样。",
                    "2. 获取响应（Response）：服务器收到请求后，返回相应的数据。",
                    "3. 解析数据：从返回的数据中提取出需要的内容。",
                    "4. 保存数据：将提取到的数据存储起来，方便后续使用。",
                    "",
                    "## 三、Request 中包含什么",
                    "",
                    "Request 中主要包含以下内容：",
                    "",
                    "- 请求方式：如 GET、POST 等",
                    "- 请求 URL：要访问的网址",
                    "- 请求头：如 User-Agent、Cookie 等",
                    "- 请求体：POST 请求时携带的数据",
                    "",
                    "---",
                    "",
                    "# 第二部分：全国省市统计",
                    "",
                    "## 全国行政区划统计表",
                    "",
                    "| 行政区类别 | 数量 | 包含范围 |",
                    "|------------|------|----------|",
                    "| 直辖市 | 4 个 | 北京市、天津市、上海市、重庆市 |",
                    "| 省 | 23 个 | 河北省、山西省、辽宁省... |",
                    "```",
                    "",
                    "## 注意事项",
                    "1. 标题必须用 `#` 符号，不要用纯文本",
                    "2. 加粗必须用 `文字`，不要用其他方式",
                    "3. 列表项的数字或符号后面必须有空格",
                    "4. 表格必须用 `|` 符号对齐",
                    "5. 不同部分之间用 `---` 分隔",
                    "6. 输出内容要准确、完整，不要遗漏原文档的重要信息",
                ])

            if is_multi:
                system_prompt_parts.insert(3, f"- 本次是 {multi_count} 个文件的合并内容：{', '.join(multi_file_names)}")
                system_prompt_parts.insert(4, "- 需要你处理整合后的全部内容，不要输出任何关于文件来源的说明")
                system_prompt_parts.insert(5, "- 重要：必须包含所有文件的内容，不能遗漏任何一个文件的内容！")
                system_prompt_parts.insert(6, f"- 每个文件的内容都要在输出中体现，共 {multi_count} 个文件：{', '.join(multi_file_names)}")
            if is_typo:
                system_prompt_parts.append("1. 检查并修正错别字和词语搭配错误")
                system_prompt_parts.append("2. 保持原文结构和格式一致")
                system_prompt_parts.append("3. 若无错别字直接回复'没有发现错别字'")
            elif is_format_conversion:
                fmt_guidance = _get_format_guidance(output_ext)
                system_prompt_parts.append("1. 根据需求整理/转换内容为目标格式")
                system_prompt_parts.append(fmt_guidance)
            else:
                system_prompt_parts.append("1. 严格按用户要求修改（增删/调整/润色/分析/整理）")
                system_prompt_parts.append("2. 只修改用户指定部分，未提及内容保持原样")

            if is_multi:
                await _report_progress(f"处理 {multi_count} 个文件的合并内容：{', '.join(multi_file_names)}")
            else:
                await _report_progress(f"处理文档：{original_filename}{_P(file_path).suffix.lower() if file_path else ''}")

            full_content = await _llm_generate(
                "\n".join(system_prompt_parts),
                f"源文件内容：\n{file_content}",
            )
            if not full_content or not full_content.strip():
                return _make_result(False, "AI 无法生成有效的文档内容")

            _typo_no_found = False
            if is_typo:
                _typo_no_keywords = [
                    "没有发现", "未发现", "没有错别字", "无错别字", "没有错字",
                    "无错字", "没有拼写", "无需修改", "不需要修改", "内容正确",
                    "检查完毕", "一切正常", "全部正确", "没有问题",
                ]
                if any(kw in full_content for kw in _typo_no_keywords):
                    _typo_no_found = True
                elif full_content.strip() == file_content.strip():
                    _typo_no_found = True

            if _typo_no_found:
                await _report_progress("检查完毕，未发现错别字")
                logger.info(f"[doc_tool] typo check done, no typos found")
                return _make_result(True, "文档中没有发现错别字")

            output_filename = f"{original_filename}_{uuid.uuid4().hex[:8]}{output_ext}"
            output_path = doc_processor.OUTPUT_DIR / output_filename

            await _report_progress(f"正在保存文件：{output_filename}")

            if is_format_conversion or is_multi:
                await asyncio.to_thread(doc_processor._save_content_to_format, full_content, str(output_path), output_ext)
            else:
                await asyncio.to_thread(
                    _save_modified_with_processor,
                    doc_processor, _P(file_path).suffix.lower() if file_path else ".txt", output_ext,
                    file_path, full_content, str(output_path),
                    user_request, file_content,
                )

            if not output_path.exists():
                return _make_result(False, "文件保存失败")

            download_url = f"http://127.0.0.1:5001/download/{output_filename}"
            logger.info(
                f"[doc_tool] modify_existing_document "
                f"({original_ext}→{output_ext}, typo={is_typo}, convert={is_format_conversion})"
            )

            # # 保存 output_file 到会话记忆
            # if thread_id:
            #     try:
            #         from core.agent_langgraph import SessionMemoryManager
            #         output_file = {"output_file": download_url}
            #         SessionMemoryManager.set_context(thread_id, **output_file)
            #         logger.info(f"[Memory] 保存 output_file 到会话记忆: {download_url}")
            #     except Exception as e:
            #         logger.error(f"[Memory] 保存 output_file 失败: {e}")

            return _make_result(True, "", download_url=download_url)

        except Exception as e:
            logger.error(f"[doc_tool] modify_existing_document error: {e}")
            return _make_result(False, str(e)[:200])

    @tool
    async def read_document_and_answer(
        file_path: str,
        file_content: str,
        user_request: str,
    ) -> str:
        """读取用户上传的文档内容，并基于内容回答用户的问题。适合"总结/摘要/讲解/说说讲了什么/介绍一下/分析一下"这类只需要阅读理解、不需要生成新文件的场景。

        Args:
            file_path: 原始文件的完整路径，用于识别文件扩展名
            file_content: 已提取的原始文档文本内容（Markdown 格式）
            user_request: 用户具体想了解什么，原样传递用户原话即可，比如"详细总结这些文档内容"、"用通俗的话说说讲了什么"
        """
        try:
            from pathlib import Path as _P
            original_filename = _P(file_path).stem if file_path else "document"
            original_ext = _P(file_path).suffix.lower() if file_path else ""

            logger.info(f"[doc_tool] read_document_and_answer 收到参数: file_path={file_path}, file_content长度={len(file_content)}, user_request={user_request[:50]}")
            logger.info(f"[doc_tool] file_content前100字: {file_content[:100]}")
            
            is_multi, multi_count = _detect_multi_file(file_content)
            multi_file_names = _collect_multi_file_names(file_content) if is_multi else []

            if is_multi:
                original_filename = f"{multi_count}个文件"

            system_prompt_lines = [
                f"你是一个专业的文档阅读助手。",
            ]
            if is_multi:
                system_prompt_lines.append(f"用户上传了 {multi_count} 个文件：{', '.join(multi_file_names)}，现在想了解它们的内容。")
            else:
                system_prompt_lines.append(f"用户上传了一份文档，现在想了解它的内容。")
                system_prompt_lines.append(f"文档名：{original_filename}{original_ext}")
            system_prompt_lines.extend([
                "",
                "请根据用户的问题，基于文档内容给出自然友好的回答：",
                "- 语气要像在跟人聊天，亲切、自然",
                "- 如果文档内容很长，可以分点说明，但不要用生硬的编号",
                "- 回答要完整、有条理，必要时可以适当展开",
                "- 直接输出回答内容，不要说「以下是总结」之类的引导语",
            ])
            system_prompt = "\n".join(system_prompt_lines)

            if is_multi:
                await _report_progress(f"正在阅读 {multi_count} 个文件：{', '.join(multi_file_names)}")
            else:
                await _report_progress(f"正在阅读文档：{original_filename}{original_ext}")
            
            user_prompt = f"用户问题：{user_request}\n\n文档内容：\n{file_content}"
            logger.info(f"[doc_tool] 传递给LLM的user_prompt长度={len(user_prompt)}")
            logger.info(f"[doc_tool] user_prompt前200字: {user_prompt[:200]}")
            
            answer = await _llm_generate(
                system_prompt,
                user_prompt,
            )
            if not answer or not answer.strip():
                return _make_result(False, "AI 无法从文档中提取有效信息")

            logger.info(f"[doc_tool] read_document_and_answer -> answered for '{original_filename}'")
            return _make_result(True, answer)

        except Exception as e:
            logger.error(f"[doc_tool] read_document_and_answer error: {e}")
            return _make_result(False, str(e)[:200])

    @tool
    async def convert_file_format(file_path: str, file_content: str, target_format: str) -> str:
        """将用户上传的文件转换成另一种格式并保存。比如 xlsx → md、pdf → docx、txt → pdf 等。

        Args:
            file_path: 原始文件的完整路径，用于识别原文件格式
            file_content: 已提取的原始文档文本内容（Markdown 格式）
            target_format: 目标输出格式，可选值: pdf, docx, md, xlsx, pptx, txt, json
        """
        try:
            from pathlib import Path as _P

            original_ext = _P(file_path).suffix.lower() if file_path else ".txt"
            output_ext = _infer_output_format(target_format, ".md")
            original_filename = _P(file_path).stem if file_path else "document"

            is_multi, multi_count = _detect_multi_file(file_content)
            multi_file_names = _collect_multi_file_names(file_content) if is_multi else []

            if is_multi:
                original_filename = "多文档合并"
                original_ext = "多格式"

            if output_ext == original_ext:
                output_ext = ".md"
                logger.warning(f"[doc_tool] convert_file_format: output same as original ({original_ext}), override to .md")

            fmt_guidance = _get_format_guidance(output_ext)

            system_prompt = (
                f"你是专业的文档格式转换助手。\n"
                f"原文件格式：{original_ext}\n"
                f"目标输出格式：{output_ext}\n"
            )
            if is_multi:
                system_prompt += f"本次是 {multi_count} 个文件的合并内容：{', '.join(multi_file_names)}\n"
            system_prompt += (
                f"要求：\n"
                f"1. 理解原文内容，转换为适合 {output_ext} 的结构化内容\n"
                f"{fmt_guidance}\n"
                f"只返回最终内容本身"
            )

            if is_multi:
                await _report_progress(f"格式转换：{multi_count} 个文件 → {output_ext}")
            else:
                await _report_progress(f"格式转换：{original_ext} → {output_ext}")
            full_content = await _llm_generate(
                system_prompt,
                f"原文件内容：\n{file_content}",
            )
            if not full_content or not full_content.strip():
                return _make_result(False, "AI 无法生成有效的文档内容")

            output_filename = f"{original_filename}_{uuid.uuid4().hex[:8]}{output_ext}"
            output_path = doc_processor.OUTPUT_DIR / output_filename

            await _report_progress(f"正在保存文件：{output_filename}")
            await asyncio.to_thread(doc_processor._save_content_to_format, full_content, str(output_path), output_ext)

            if not output_path.exists():
                return _make_result(False, "文件保存失败")

            download_url = f"http://127.0.0.1:5001/download/{output_filename}"

            # # 保存 output_file 到会话记忆
            # if thread_id:
            #     try:
            #         output_file = {"output_file": download_url}
            #         from core.agent_langgraph import SessionMemoryManager
            #         SessionMemoryManager.set_context(thread_id, **output_file)
            #         logger.info(f"[Memory] 保存 output_file 到会话记忆: {download_url}")
            #     except Exception as e:
            #         logger.error(f"[Memory] 保存 output_file 失败: {e}")

            logger.info(f"[doc_tool] convert_file_format ({original_ext}→{output_ext}) -> {output_filename}")
            return _make_result(True, "", download_url=download_url)

        except Exception as e:
            logger.error(f"[doc_tool] convert_file_format error: {e}")
            return _make_result(False, str(e)[:200])

    return [create_new_document, modify_existing_document, read_document_and_answer, convert_file_format]


# OpenAI 原生 API 工具定义（纯 JSON，不依赖 langchain）
DOC_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "create_new_document",
            "description": "根据用户描述创建一份全新的文档文件。用于用户没有上传文件、只说'帮我写一份XX'的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "content_description": {
                        "type": "string",
                        "description": "详细描述文档应该写什么内容，比如'一份季度销售总结报告，包含各区域业绩对比'"
                    },
                    "doc_type_hint": {
                        "type": "string",
                        "description": "文档类型提示，比如'报告'、'会议纪要'、'合同'、'简历'、'演讲稿'"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "期望输出的文件格式，可选值: pdf, docx, md, xlsx, pptx, txt, json"
                    }
                },
                "required": ["content_description", "doc_type_hint", "output_format"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_existing_document",
            "description": "对用户上传的文档内容进行修改、整理、汇总、分析、纠错、润色或重写。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "原始文件的完整路径，用于识别文件扩展名和保留部分格式"
                    },
                    "file_content": {
                        "type": "string",
                        "description": "已提取的原始文档文本内容（Markdown 格式）"
                    },
                    "user_request": {
                        "type": "string",
                        "description": "用户具体的修改/整理/分析/纠错要求"
                    },
                    "output_format": {
                        "type": "string",
                        "description": "期望输出的文件格式，可选值: pdf, docx, md, xlsx, pptx, txt, json。填原格式表示只改不改格式"
                    }
                },
                "required": ["file_path", "file_content", "user_request", "output_format"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_document_and_answer",
            "description": "读取用户上传的文档内容，并基于内容回答用户的问题。适合'总结/摘要/讲解/说说讲了什么/介绍一下/分析一下'这类只需要阅读理解、不需要生成新文件的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "原始文件的完整路径，用于识别文件扩展名"
                    },
                    "file_content": {
                        "type": "string",
                        "description": "已提取的原始文档文本内容（Markdown 格式）"
                    },
                    "user_request": {
                        "type": "string",
                        "description": "用户具体想了解什么，原样传递用户原话即可"
                    }
                },
                "required": ["file_path", "file_content", "user_request"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_file_format",
            "description": "将用户上传的文件转换成另一种格式并保存。比如 xlsx → md、pdf → docx、txt → pdf 等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "原始文件的完整路径，用于识别原文件格式"
                    },
                    "file_content": {
                        "type": "string",
                        "description": "已提取的原始文档文本内容（Markdown 格式）"
                    },
                    "target_format": {
                        "type": "string",
                        "description": "目标输出格式，可选值: pdf, docx, md, xlsx, pptx, txt, json"
                    }
                },
                "required": ["file_path", "file_content", "target_format"]
            }
        }
    }
]


_llm_cache = None


def _get_llm():
    """懒加载原生 OpenAI 客户端"""
    global _llm_cache
    if _llm_cache is None:
        from openai import OpenAI
        _llm_cache = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
        )
    return _llm_cache


async def _llm_generate(system_prompt: str, user_prompt: str) -> str:
    """异步 LLM 流式生成，内部自动推送进度到 context queue"""
    import time as _time
    import concurrent.futures
    _t0 = _time.time()
    client = _get_llm()
    
    await _report_progress("AI 正在生成文档内容...")
    full = ""
    last_yield = 0
    _t_first = None
    
    # 使用队列在线程和异步循环之间传递流式数据
    chunk_queue = asyncio.Queue()
    
    def _stream_worker():
        """在线程中执行 OpenAI 流式请求，将 chunk 放入队列"""
        try:
            response = client.chat.completions.create(
                model=Params.DEFAULT_TEXT_TOOL_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                temperature=0.1,
            )
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_queue.put_nowait(chunk.choices[0].delta.content)
        except Exception as e:
            chunk_queue.put_nowait(None)  # 用 None 标记异常
            logger.error(f"[doc_tool] _llm_generate 流式请求失败: {e}")
        finally:
            chunk_queue.put_nowait(None)  # 用 None 标记结束
    
    # 在线程池中启动流式请求
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = loop.run_in_executor(executor, _stream_worker)
        
        while True:
            delta = await chunk_queue.get()
            if delta is None:
                break
            full += delta
            if _t_first is None and len(full) > 0:
                _t_first = _time.time()
            if len(full) - last_yield >= 500:
                last_yield = len(full)
                await _report_progress(f"AI 生成中... 已输出 {len(full)} 字")
        
        await future  # 等待线程完成
    
    _t1 = _time.time()
    _ttfb = (_t_first or _t1) - _t0
    logger.info(
        f"[doc_tool] _llm_generate: "
        f"input={len(system_prompt)+len(user_prompt)}字, "
        f"output={len(full)}字, "
        f"ttfb={_ttfb:.2f}s, "
        f"total={(_t1-_t0):.2f}s"
    )
    await _report_progress(f"AI 生成完成，共 {len(full)} 字")
    return full.strip()


def _save_modified_with_processor(
    doc_processor, original_ext: str, output_ext: str,
    file_path: str, modified_content: str, output_path: str,
    user_request: str, original_content: str,
):
    """就地修改场景的分发（ext == output_ext），复用 document_process.py 的底层逻辑"""
    # 先尝试就地修改（保留格式），失败则走重建
    try:
        if output_ext == ".docx":
            doc_processor._modify_docx_with_format(file_path, modified_content, user_request, str(output_path), None)
        elif output_ext == ".txt" or output_ext == ".md":
            doc_processor._modify_txt_with_format(modified_content, str(output_path), user_request, original_content)
        elif output_ext in [".xlsx", ".xls"]:
            doc_processor._modify_excel(file_path, modified_content, str(output_path))
        elif output_ext in [".pptx", ".ppt"]:
            doc_processor._modify_pptx(file_path, modified_content, str(output_path))
        elif output_ext == ".doc":
            doc_processor._modify_doc(modified_content, str(output_path), Path(file_path).stem)
        elif output_ext == ".pdf":
            # tool 是 sync 的，这里不能 await，所以直接用重建兜底
            doc_processor._create_pdf(modified_content, str(output_path))
        else:
            doc_processor._save_content_to_format(modified_content, str(output_path), output_ext)
    except Exception as e:
        logger.warning(f"[doc_tool] 就地修改失败，回退到重建: {e}")
        doc_processor._save_content_to_format(modified_content, str(output_path), output_ext)