import os
import uuid
import shutil
import asyncio
import contextvars
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from settings.Define import Params
from settings.logger_manager import get_logger

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

_SYSTEM_PROMPT = """你是一个文档处理助手，需要帮用户选择合适的文档处理工具，并最终用自然友好的聊天语气回复用户。

# 核心规则

你需要先做**语义判断**：用户这条消息的核心意图是什么？

- **围绕文档的处理/创建需求** → 必须调工具
  - 明显信号词：错别字、整理、修改、润色、分析、汇总、总结、统计、重写、归纳、提取、翻译、生成、创建、写一份、做一个
  - 只要提到了**文件/格式/保存/输出/导出/另存为**，哪怕内容描述短或模糊，也是文档需求
  - 例："保存到excel"、"转成markdown"、"输出成PDF"、"汇总存成表格" → 都是文档需求
- **和文档无关的闲聊** → 不调工具，直接聊天回复
  - 只包含：夸奖、感谢、打招呼、闲聊话题、"好的"/"行"/"嗯嗯"这类纯应答、纯追问语义（"你说的对"、"同意"、"可以"）

注意：**有没有附件文件路径不能作为判断依据**。即使用户上传了文件，如果消息是"你太厉害了"——还是闲聊，不调工具。
即使用户没上传文件，但说"帮我写一份会议纪要"——有文档需求，调 create_new_document。

# 如何选择工具

```
第一步：用户消息的语义是不是围绕文档处理？
（文档处理 = 错别字/整理/修改/润色/分析/汇总/重写/排版/格式转换/新建文档 等）
├─ 是 → 继续判断用哪个工具
└─ 否 → 闲聊，不调任何工具，直接聊天回复

第二步：选哪个工具（只有用户有文档处理需求时才走这里）
    有附件文件路径吗？
    ├─ 有 →
    │   ├─ 用户只说格式转换（"把/转成/另存为/导出为 X 格式"），且不提任何内容处理需求（没有整理/修改/纠错/润色等）
    │   │   → convert_file_format（纯格式转换，不动内容）
    │   └─ 其他一切情况 → modify_existing_document，包括但不限于：
    │       · 有错别字/错字/纠错/拼写检查
    │       · 要整理/梳理/归纳/汇总/统计/总结
    │       · 要修改/改动/调整/重写/改写/重构
    │       · 要润色/优化/美化/排版/格式调整
    │       · 要分析/解读/提炼/摘要/浓缩
    │       · 要翻译/补充/完善/扩展
    │       · 要生成图表/提取数据
    │       · 再检查/确认下/对吗/有没有问题
    │       · 同时提到了内容处理 + 格式转换（如"整理下内容，输出成 markdown"）
    │       （如果用户顺带提了格式，output_format 参数填目标格式即可）
    └─ 没有 → create_new_document（用户要你从零写一份新文档）
```

# 示例

✅ 有文件路径 + "看看有没有错别字" → 语义围绕文档 → 调 modify_existing_document
✅ 有文件路径 + "对吗，再检查下" → 语义围绕文档 → 调 modify_existing_document
✅ 有文件路径 + "整理成 markdown" → 语义围绕文档 → 调 modify_existing_document
✅ 有文件路径 + "你好牛啊" → 语义是夸奖闲聊 → 不调工具，直接回复
✅ 有文件路径 + "谢谢" → 语义是感谢闲聊 → 不调工具，直接回复
✅ 没有文件路径 + "帮我写会议纪要" → 语义围绕文档 → 调 create_new_document
✅ 没有文件路径 + "你太厉害了" → 闲聊 → 不调工具
✅ 没有文件路径 + "汇总保存到excel文件中" → 提到了"保存"+"excel文件" → 文档需求 → 调 create_new_document
✅ 没有文件路径 + "转成markdown" → 提到了格式 → 文档需求 → 调 create_new_document

❌ 有文件路径 + "看看有没有错别字" → 直接回复"好的我来帮你检查"（错误，语义围绕文档，必须调工具！）
❌ "汇总保存到excel" → "请提供具体内容"（错误！这是文档需求，应该调 create_new_document 让工具自己生成内容）

# 关于最终回复的语气（非常重要）

- **像人跟人说话一样**，不要用"执行成功"、"执行完毕"这类技术报告式措辞
- 正确示例："好的，我帮你检查了一遍这份文档，没有发现错别字哦 😊"
- 正确示例："已帮你重新检查了一下，发现了一些小问题，帮你修正了，你可以下载查看"
- 错误示例："执行成功：检查完毕，未发现错别字"
- 错误示例："✅ 工具执行完成，耗时 9.1s"

# 参数填写注意事项

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


def create_doc_tools():
    from core.skills.DocProcess.document_process import doc_processor

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
        """
        try:
            from pathlib import Path as _P

            original_ext = _P(file_path).suffix.lower() if file_path else ".txt"
            output_ext = _infer_output_format(output_format, original_ext)
            original_filename = _P(file_path).stem if file_path else "document"

            is_format_conversion = (output_ext != original_ext)
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

            await _report_progress(f"处理文档：{original_filename}{original_ext}")
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

            if is_format_conversion:
                await asyncio.to_thread(doc_processor._save_content_to_format, full_content, str(output_path), output_ext)
            else:
                await asyncio.to_thread(
                    _save_modified_with_processor,
                    doc_processor, original_ext, output_ext,
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
            return _make_result(True, "", download_url=download_url)

        except Exception as e:
            logger.error(f"[doc_tool] modify_existing_document error: {e}")
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

            if output_ext == original_ext:
                output_ext = ".md"
                logger.warning(f"[doc_tool] convert_file_format: output same as original ({original_ext}), override to .md")

            fmt_guidance = _get_format_guidance(output_ext)

            system_prompt = (
                f"你是专业的文档格式转换助手。\n"
                f"原文件格式：{original_ext}\n"
                f"目标输出格式：{output_ext}\n"
                f"要求：\n"
                f"1. 理解原文内容，转换为适合 {output_ext} 的结构化内容\n"
                f"{fmt_guidance}\n"
                f"只返回最终内容本身"
            )

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
            logger.info(f"[doc_tool] convert_file_format ({original_ext}→{output_ext}) -> {output_filename}")
            return _make_result(True, "", download_url=download_url)

        except Exception as e:
            logger.error(f"[doc_tool] convert_file_format error: {e}")
            return _make_result(False, str(e)[:200])

    return [create_new_document, modify_existing_document, convert_file_format]


_llm_cache: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = ChatOpenAI(
            model=Params.DEFAULT_CHAT_MODEL,
            temperature=0.1,
            max_tokens=8192,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=Params.API_BASE,
        )
    return _llm_cache


async def _llm_generate(system_prompt: str, user_prompt: str) -> str:
    """异步 LLM 流式生成，内部自动推送进度到 context queue"""
    import time as _time
    _t0 = _time.time()
    llm = _get_llm()
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    await _report_progress("AI 正在生成文档内容...")
    full = ""
    last_yield = 0
    _t_first = None
    async for chunk in llm.astream(msgs):
        delta = getattr(chunk, "content", "") or ""
        full += delta
        if _t_first is None and len(full) > 0:
            _t_first = _time.time()
        if len(full) - last_yield >= 500:
            last_yield = len(full)
            await _report_progress(f"AI 生成中... 已输出 {len(full)} 字")
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