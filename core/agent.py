import os

os.environ.setdefault("DASHSCOPE_API_KEY", "")

import asyncio
import json
from typing import Annotated, TypedDict
from operator import add
from pathlib import Path
from langchain_classic.agents import create_react_agent
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.constants import END, START
from core.DocProcess.document_process import doc_processor
from core.DrawingRecoAssistant.drawing_assistant import get_drawing_assistant
from settings.Define import Params, PathConfig
from settings.logger_manager import get_logger

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None

logger = get_logger(__name__)

llm = ChatTongyi(
    model=Params.DEFAULT_CHAT_MODEL,
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    api_base=Params.API_BASE,
    timeout=60  # 设置LLM调用超时时间为60秒
)

embedding_model = DashScopeEmbeddings(model=Params.DEFAULT_EMBEDDING_MODEL)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str
    file_path: str  # 添加文件路径字段
    previous_node: str  # 记录上一个执行的节点类型，用于反思后路由
    skill: str  # 前端选择的技能类型：travel/joke/couplet/document/code/drawing/other
    sources: str  # 引用来源文件列表(JSON字符串)
    user: str  # 当前登录用户（用于知识库检索）


def create_agent_graph():
    nodes = Params.NODE_LIST

    def other_node(state: State):
        logger.info("other_node 开始处理")
        
        message = state["messages"][-1]
        if hasattr(message, 'content'):
            user_query = message.content
        else:
            user_query = str(message)
        
        # 获取当前用户
        current_user = state.get("user", "")
        
        # 对所有用户的文档进行RAG检索（不依赖登录状态）
        try:
            from core.knowledge.knowledge_manager import get_vector_store, get_all_users, get_vector_count
            
            # 获取所有用户
            all_users = get_all_users()
            logger.info(f"检索到的用户列表: {all_users}")
            
            # 检查每个用户的向量数据库状态
            for user in all_users:
                vector_count = get_vector_count(user)
                logger.info(f"用户 {user} 的向量数据库中有 {vector_count} 条记录")
            
            # 收集所有检索结果
            all_results = []
            
            # 遍历所有用户的知识库进行检索
            for user in all_users:
                try:
                    vector_store = get_vector_store(user)
                    # 检查向量数据库中的记录数量
                    vector_count = get_vector_count(user)
                    logger.info(f"用户 {user} 的向量数据库中有 {vector_count} 条记录")
                    
                    results = vector_store.similarity_search(user_query, k=2)
                    logger.info(f"用户 {user} 的知识库检索到 {len(results)} 条结果")
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"检索用户 {user} 的知识库失败: {str(e)}")
                    continue
            
            # 如果当前用户也有知识库，也进行检索（避免遗漏）
            if current_user and current_user not in all_users:
                try:
                    vector_store = get_vector_store(current_user)
                    results = vector_store.similarity_search(user_query, k=3)
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"检索当前用户 {current_user} 的知识库失败: {str(e)}")
            
            # 对结果进行去重和排序（按相关性）
            if all_results:
                # 打印检索到的详细内容
                for i, doc in enumerate(all_results):
                    source = doc.metadata.get('source', '未知')
                    content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    logger.info(f"检索结果 {i+1}: 来源={source}, 内容预览={content}")
                
                # 构建知识库内容（不包含来源标签，避免LLM在回答中提及）
                knowledge_context = "\n\n".join([doc.page_content for doc in all_results])
                
                # 构建提示词，让LLM根据知识库内容回答，但不提及知识库和来源
                system_prompt = f"""你是一个专业的知识问答助手。请根据提供的参考资料回答用户的问题：

参考资料：
{knowledge_context}

用户问题：{user_query}

要求：
1. 仔细阅读并分析参考资料，寻找与用户问题相关的信息
2. 如果找到相关信息，请基于这些信息进行专业、准确的回答
3. 如果参考资料中没有相关信息，请说明"抱歉，我无法回答这个问题"
4. 回答要简洁、清晰，使用自然友好的语言，不要提及"知识库"或"来源"等字样"""
                
                response = llm.invoke([{"role": "user", "content": system_prompt}])
                content = response.content if hasattr(response, 'content') else str(response)
                
                # 准备sources数据
                sources = []
                for doc in all_results:
                    source = {
                        "source": doc.metadata.get('source', '未知'),
                        "content": doc.page_content
                    }
                    sources.append(source)
                
                logger.info(f"知识库检索并回答成功，检索到 {len(all_results)} 条结果")
                return {"messages": [HumanMessage(content=content)],
                        "type": "other", "previous_node": "other_node",
                        "sources": sources}
            else:
                logger.info("知识库中未找到相关信息")
                return {"messages": [HumanMessage(content="抱歉，您咨询的问题不在我的知识库中。")],
                        "type": "other", "previous_node": "other_node",
                        "sources": []}
        except Exception as e:
            logger.error(f"知识库检索失败: {str(e)}")
            return {"messages": [HumanMessage(content=f"知识库查询失败：{str(e)}")],
                    "type": "other", "previous_node": "other_node",
                    "sources": []}

    def supervisor_node(state: State):
        logger.info("supervisor_node")
        prompts = """你是一个专业的客服助手，需要根据用户的问题进行任务分类，并将任务分给对应的Agent来执行。

分类规则：
1. 如果用户的问题是和旅游路线规划相关的，那就返回travel。
2. 如果用户的问题是和讲笑话相关的，那就返回joke。
3. 如果用户的问题是和对联相关的，那就返回couplet。
4. 如果用户的问题是和文档相关的（包含修改已有文档、新建文档并写入内容、排版、修正格式、修改对齐方式，查看文档中的错别字等等），那就返回document。
5. 如果用户的问题是和编程相关的，那就返回code。
6. 如果用户的问题是和工程图纸识别、尺寸提取、公差校核、DXF解析、国标查询、单位换算等相关的，那就返回drawing。
7. 如果是其它问题，那就返回other。

重要：如果用户的问题是追问、确认、或者对之前回答的反馈（例如"确认下是否写得对？"、"再优化一下"、"不对，应该是..."），请根据对话历史判断之前讨论的主题，并返回对应的分类。
除了这几个选项外，不要返回任何其它的内容。
"""

        message = state["messages"][-1]
        if hasattr(message, 'content'):
            message_content = message.content
        else:
            message_content = str(message)
        logger.info(f"用户消息: {message_content[:30]}...")

        # Build conversation history for context
        conversation_history = []
        for msg in state["messages"][:-1]:
            if hasattr(msg, 'content'):
                content = msg.content
                msg_type = type(msg).__name__
            else:
                content = str(msg)
                msg_type = "dict"

            if isinstance(msg, dict) and msg.get("role") == "assistant":
                conversation_history.append({"role": "assistant", "content": msg["content"]})
            elif "HumanMessage" in str(type(msg)) or (isinstance(msg, dict) and msg.get("role") == "user"):
                conversation_history.append({"role": "user", "content": content})

        prompt_list = [
            {"role": "system", "content": prompts},
        ] + conversation_history + [
            {"role": "user", "content": message_content},
        ]
        if "type" in state:
            logger.info("已有类型，返回 END")
            return {"type": END}
        elif state.get("skill", "").strip() in nodes:
            skill_type = state["skill"].strip()
            logger.info(f"前端指定技能: {skill_type}，直接路由")
            return {"type": skill_type}
        else:
            try:
                response = llm.invoke(prompt_list)
                typeRes = response.content.strip()
                logger.debug(f"分类结果原始值: '{typeRes}' (长度: {len(typeRes)})")
                logger.debug(f"分类结果.strip(): '{typeRes.strip()}'")
                
                # 定义需要强制选择技能的节点
                skill_required_nodes = ['code', 'document', 'drawing']
                
                if typeRes.strip() in nodes:
                    # 如果没有选择技能，但分类结果是需要技能的节点，默认路由到 other
                    if typeRes.strip() in skill_required_nodes and not state.get("skill", "").strip():
                        logger.warning(f"分类结果为 {typeRes.strip()}，但未选择技能，默认路由到 other")
                        return {"type": "other"}
                    logger.info(f"分类成功: {typeRes.strip()}")
                    return {"type": typeRes.strip()}
                else:
                    logger.warning(f"分类失败: '{typeRes}' 不在列表 {nodes} 中")
                    logger.info("默认返回 other")
                    return {"type": "other"}
            except Exception as e:
                logger.error(f"supervisor_node LLM 调用失败: {str(e)[:50]}")
                return {"type": "other"}

    def travel_node(state: State):
        system_prompt = """你是一个专业的旅行规划助手，根据用户的问题，生成一个旅游路线规划。请用中文回答，并返回一个不超过100字的规划结果"""

        # Get the last message and build conversation history
        message = state["messages"][-1]
        if hasattr(message, 'content'):
            message_content = message.content
        else:
            message_content = str(message)

        conversation_history = []
        for msg in state["messages"][:-1]:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            if isinstance(msg, dict) and msg.get("role") == "assistant":
                conversation_history.append({"role": "assistant", "content": content})
            elif "HumanMessage" in str(type(msg)) or (isinstance(msg, dict) and msg.get("role") == "user"):
                conversation_history.append({"role": "user", "content": content})

        prompt_list = [
            {"role": "system", "content": system_prompt},
        ] + conversation_history + [
            {"role": "user", "content": message_content},
        ]
        try:
            if MultiServerMCPClient:
                client = MultiServerMCPClient(
                    {"amap-maps-sse": {"url": "https://mcp.amap.com/sse?key=77067076328eb3b90559c0cb1222ff81",
                                       "transport": "streamable_http"}}
                )
                tools = asyncio.run(client.get_tools())
                agent = create_react_agent(model=llm, tools=tools)
                response = agent.invoke({"messages": prompt_list})
            else:
                raise ImportError("MultiServerMCPClient not available")
        except Exception as e:
            logger.warning(f"MCP 服务不可用，使用本地规划: {str(e)[:50]}")
            response = llm.invoke(prompt_list)
        content = response.content if hasattr(response, 'content') else str(response)
        if not content or content.strip() == "":
            content = "抱歉，暂时无法生成旅游路线规划。"
        return {"messages": [HumanMessage(content=content)], "type": "travel", "previous_node": "travel_node"}

    def joke_node(state: State):
        logger.info("joke_node")
        system_prompt = """你是一个专业的笑话大师，根据用户的问题，写一个不超过100个字的笑话。"""

        message = state["messages"][-1]
        if hasattr(message, 'content'):
            message_content = message.content
        else:
            message_content = str(message)

        conversation_history = []
        for msg in state["messages"][:-1]:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            if isinstance(msg, dict) and msg.get("role") == "assistant":
                conversation_history.append({"role": "assistant", "content": content})
            elif "HumanMessage" in str(type(msg)) or (isinstance(msg, dict) and msg.get("role") == "user"):
                conversation_history.append({"role": "user", "content": content})

        prompt_list = [
            {"role": "system", "content": system_prompt},
        ] + conversation_history + [
            {"role": "user", "content": message_content},
        ]
        try:
            response = llm.invoke(prompt_list)
            content = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"joke_node LLM 调用失败: {str(e)[:50]}")
            content = "抱歉，当前服务暂时不可用，请稍后重试。"
        return {"messages": [HumanMessage(content=content)], "type": "joke", "previous_node": "joke_node"}

    def code_node(state: State):
        logger.info("code_node")

        # 获取文件名
        file_path = state.get("file_path")
        filename = "代码"
        if file_path:
            from pathlib import Path
            filename = Path(file_path).name

        system_prompt = f"""你是一个专业的代码编程专家，熟悉各种语言的编程，比如C、C++、HTML、Go、Java、PHP、Python等。

请根据用户的具体需求进行回答：

**场景A：用户要求生成新代码**
- 使用 Markdown 格式，包括标题（# ## ###）、加粗（**文本**）、列表（- 或 1.）等
- 代码块必须独占成段，使用三个反引号包裹
- 代码要完整，包含必要的导入、函数定义、测试用例和运行示例
- 代码要有注释，关键逻辑处添加简洁的中文注释
- 分版本展示：如果适用，用二级标题（##）分隔不同版本，如"## 1. 基础版"和"## 2. 优化版"
- 复杂度分析：在代码后用列表形式简要说明时间复杂度和空间复杂度
- 运行示例：给出代码的运行输出示例

回答结构示例（生成新代码）：
## 标题

简要说明文字。

### 1. 基础版

核心逻辑说明。

```python
# 完整代码
```

### 2. 优化版

优化说明。

```python
# 完整代码
```

### 复杂度分析

- **时间复杂度**：说明
- **空间复杂度**：说明

### 运行示例

```
输出结果
```

**场景B：用户要求审查/修改已有代码**
- 仔细检查代码中的语法问题、逻辑问题、BUG问题等
- 如果没有发现任何问题，必须回复："{filename}代码文件不需要修改，没有发现任何明显性错误"
- 如果发现问题，必须按照以下格式回复：

发现{{N}}处问题：
第1处：{{问题描述}}
第2处：{{问题描述}}
...

修改后的完整代码：
```{{language}}
{{修改后的完整代码}}
```

修改后的文件下载路径：http://127.0.0.1:5001/download/{{output_filename}}

注意：代码块的语言标识必须正确（如 python、javascript、java、cpp 等），以便前端正确高亮显示。"""

        message = state["messages"][-1]
        if hasattr(message, 'content'):
            message_content = message.content
        else:
            message_content = str(message)

        conversation_history = []
        for msg in state["messages"][:-1]:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            if isinstance(msg, dict) and msg.get("role") == "assistant":
                conversation_history.append({"role": "assistant", "content": content})
            elif "HumanMessage" in str(type(msg)) or (isinstance(msg, dict) and msg.get("role") == "user"):
                conversation_history.append({"role": "user", "content": content})

        prompt_list = [
            {"role": "system", "content": system_prompt},
        ] + conversation_history + [
            {"role": "user", "content": message_content},
        ]
        try:
            response = llm.invoke(prompt_list)
            content = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"code_node LLM 调用失败: {str(e)[:50]}")
            content = "抱歉，当前服务暂时不可用，请稍后重试。"

        # 检查是否需要保存修改后的代码文件
        if file_path and "修改后的完整代码" in content:
            try:
                from pathlib import Path
                import uuid

                # 提取代码块内容
                import re
                code_match = re.search(r'```(?:\w+)?\n([\s\S]*?)```', content)
                if code_match:
                    modified_code = code_match.group(1).strip()

                    # 获取原始文件扩展名和文件名
                    original_ext = Path(file_path).suffix
                    original_name = Path(file_path).stem

                    # 生成输出文件名
                    output_filename = f"modified_{original_name}_{str(uuid.uuid4())[:8]}{original_ext}"
                    output_path = PathConfig.OUTPUTS_DIR / output_filename

                    # 保存修改后的代码
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(modified_code)

                    # 生成下载链接
                    download_url = f"http://127.0.0.1:5001/download/{output_filename}"

                    # 替换内容中的下载路径（支持占位符或LLM生成的原始文件名链接）
                    content = content.replace("http://127.0.0.1:5001/download/{output_filename}", download_url)
                    # 也替换LLM可能生成的原始文件名链接
                    original_download_pattern = rf"http://127\.0\.0\.1:5001/download/{re.escape(Path(file_path).name)}"
                    content = re.sub(original_download_pattern, download_url, content)

                    logger.info(f"代码文件保存成功: {output_path}")
            except Exception as e:
                logger.error(f"保存代码文件失败: {str(e)}")

        return {"messages": [HumanMessage(content=content)], "type": "code", "previous_node": "code_node"}

    def couplet_node(state: State):
        logger.info("couplet_node 开始处理")

        from core.Couplet.couplet import Couplet

        couplet = Couplet()
        if not couplet.load_vector():
            return {"messages": [HumanMessage(content="抱歉，向量数据库加载失败。")], "type": "couplet"}


        query = state["messages"][-1]
        if hasattr(query, 'content'):
            query_text = query.content
        else:
            query_text = str(query)
        logger.info(f"查询文本: {query_text}")

        prompt = couplet.similarity_search(query_text)
        if not prompt:
            return {"messages": [HumanMessage(content="抱歉，Prompt 构建失败。")], "type": "couplet"}

        try:
            # RAG生成
            response = llm.invoke(prompt)
            logger.info("LLM 调用成功")
        except Exception as e:
            logger.error(f"LLM 调用失败: {str(e)}")
            return {"messages": [HumanMessage(content="抱歉，LLM 调用失败。")], "type": "couplet"}

        content = response.content if hasattr(response, 'content') else str(response)
        if not content or content.strip() == "":
            content = "抱歉，暂时无法生成对联。"

        logger.info(f"对联结果: {content[:30]}...")
        return {"messages": [HumanMessage(content=content)], "type": "couplet", "previous_node": "couplet_node"}

    def document_node(state: State):
        logger.info("document_node 开始处理")
        message = state["messages"][-1]
        if hasattr(message, 'content'):
            message_content = message.content
        else:
            message_content = str(message)

        # Build conversation history for context
        conversation_history = []
        for msg in state["messages"][:-1]:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            if isinstance(msg, dict) and msg.get("role") == "assistant":
                conversation_history.append({"role": "assistant", "content": content})
            elif "HumanMessage" in str(type(msg)) or (isinstance(msg, dict) and msg.get("role") == "user"):
                conversation_history.append({"role": "user", "content": content})

        logger.info(f"用户输入内容长度: {len(message_content)}")

        file_path = state.get("file_path")

        # 如果没有文件路径，说明是新建文档
        if not file_path:
            # 解析创建要求
            if '[用户要求]' in message_content:
                parts = message_content.split('[用户要求]')
                creation_request = parts[-1].strip() if len(parts) > 1 else "创建文档"
            else:
                lines = message_content.split('\n')
                creation_request = lines[0].strip() if lines else "创建文档"

            # 检测输出格式
            output_ext = doc_processor.detect_output_format(creation_request, conversation_history)
            
            # 调用 document_process 的创建文档接口
            result_content, _ = doc_processor.create_document(
                creation_request=creation_request,
                conversation_history=conversation_history,
                llm=llm,
                output_ext=output_ext
            )
            
            return {"messages": [HumanMessage(content=result_content)], "type": "document", "previous_node": "document_node"}

        logger.info(f"文件路径: {file_path}")

        # 解析用户的修改要求
        if '[用户要求]' in message_content:
            parts = message_content.split('[用户要求]')
            modification_request = parts[-1].strip() if len(parts) > 1 else "修改文档"
        else:
            lines = message_content.split('\n')
            modification_request = lines[0].strip() if lines else "修改文档"

        logger.info(f"修改要求: {modification_request}")

        # 调用 document_process 的修改文档接口
        result_content, _ = doc_processor.process_document_modification(
            file_path=file_path,
            modification_request=modification_request,
            conversation_history=conversation_history,
            llm=llm
        )

        return {"messages": [HumanMessage(content=result_content)], "type": "document", "previous_node": "document_node"}



    def drawing_node(state: State):
        """图纸识别节点：委托 DrawingAssistant 处理工程图纸相关任务"""
        logger.info("drawing_node 开始处理")

        message = state["messages"][-1]
        if hasattr(message, 'content'):
            user_query = message.content
        else:
            user_query = str(message)

        file_path = state.get("file_path", "")
        assistant = get_drawing_assistant()

        # 根据文件扩展名判断图纸类型
        img_path = None
        pdf_path = None
        dxf_path = None

        if file_path:
            ext = Path(file_path).suffix.lower()
            if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
                img_path = file_path
            elif ext == ".pdf":
                pdf_path = file_path
            elif ext == ".dxf":
                dxf_path = file_path

        try:
            result = assistant.invoke(
                user_query=user_query,
                img_path=img_path,
                pdf_path=pdf_path,
                dxf_path=dxf_path,
                thread_id="agent_draw_session",
            )
            answer = result["answer"] if isinstance(result, dict) else result
            sources = result.get("sources", []) if isinstance(result, dict) else []
        except Exception as e:
            logger.error(f"drawing_node 处理失败: {str(e)}")
            answer = f"图纸识别处理失败：{str(e)}"
            sources = []

        return {
            "messages": [HumanMessage(content=answer)],
            "type": "drawing",
            "previous_node": "drawing_node",
            "sources": json.dumps(sources, ensure_ascii=False),
        }

    def reflection_node(state: State):
        """反思节点：评估各任务节点的输出质量，决定是否需要重新生成"""
        logger.info("reflection_node 开始反思")

        previous_node = state.get("previous_node", "")
        last_message = state["messages"][-1]
        user_message = state["messages"][-2] if len(state["messages"]) >= 2 else state["messages"][0]

        if hasattr(last_message, 'content'):
            ai_response = last_message.content
        else:
            ai_response = str(last_message)

        if hasattr(user_message, 'content'):
            user_query = user_message.content
        else:
            user_query = str(user_message)

        # 根据上一个节点类型，使用不同的反思标准
        reflection_prompts = {
            "travel_node": """你是一个专业的旅行规划质量评估专家。请评估以下旅行路线规划的质量：

评估标准：
1. 是否包含具体的景点、交通、住宿等实用信息
2. 路线安排是否合理、可行
3. 是否满足用户的具体需求
4. 信息是否准确、详细

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",

            "joke_node": """你是一个专业的笑话质量评估专家。请评估以下笑话的质量：

评估标准：
1. 是否有趣、有笑点
2. 是否符合用户要求的主题
3. 语言是否流畅自然
4. 长度是否合适（不超过100字）

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",

            "couplet_node": """你是一个专业的对联质量评估专家。请评估以下对联的质量：

评估标准：
1. 上下联是否对仗工整（词性、平仄、意境）
2. 是否符合用户给出的上联主题
3. 是否有文化内涵和文学美感

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",

            "document_node": """你是一个专业的文档修改质量评估专家。请评估以下文档修改的质量：

评估标准：
1. 是否按照用户要求进行了修改
2. 修改后的内容是否准确、通顺
3. 是否保留了原文档的核心内容
4. 格式是否正确

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",

            "code_node": """你是一个专业的代码质量评估专家。请评估以下代码的质量：

评估标准：
1. 代码是否能正确运行（语法正确）
2. 代码逻辑是否合理、无Bug
3. 是否有必要的注释和说明
4. 是否满足用户的具体需求
5. 代码风格是否规范

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",

            "other_node": """你是一个专业的回答质量评估专家。请评估以下回答的质量：

评估标准：
1. 回答是否礼貌、得体
2. 是否清晰表达了无法处理的原因

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",

            "drawing_node": """你是一个专业的工程图纸分析质量评估专家。请评估以下图纸分析回答的质量：

评估标准：
1. 是否准确提取了图纸中的尺寸、标注、图层等信息
2. 尺寸分析是否合理、尺寸链是否闭合
3. 国标引用是否准确
4. 加工建议是否专业、实用
5. 回答结构是否清晰（图纸信息、尺寸分析、国标依据、加工建议）

用户问题：{user_query}
AI回答：{ai_response}

请给出评估结果，只能返回以下两种之一：
- "PASS"：质量合格，无需修改
- "FAIL: {具体问题描述和改进建议}"：质量不合格，需要重新生成""",
        }

        reflection_prompt = reflection_prompts.get(previous_node, reflection_prompts["other_node"])

        try:
            response = llm.invoke([
                {"role": "user", "content": reflection_prompt.format(user_query=user_query, ai_response=ai_response)}
            ])
            reflection_result = response.content.strip()
        except Exception as e:
            logger.error(f"reflection_node LLM 调用失败: {str(e)[:50]}")
            reflection_result = "PASS"

        logger.info(f"反思结果: {reflection_result[:50]}...")

        if reflection_result.startswith("PASS"):
            logger.info("反思通过，质量合格")
            return {"type": END}
        else:
            logger.info("反思未通过，需要重新生成")
            feedback = reflection_result.replace("FAIL:", "").strip()
            feedback_message = f"之前的回答质量不够好，请根据以下反馈重新生成：{feedback}"
            return {"messages": [HumanMessage(content=feedback_message)], "type": previous_node.replace("_node", "")}

    def routing_func(state: State):
        current_type = state["type"]
        logger.info(f"routing_func: type = '{current_type}'")
        if current_type == Params.MAPPING_NODE.get(Params.TRAVEL_NODE):
            logger.info(f"路由到 {Params.TRAVEL_NODE}")
            return Params.TRAVEL_NODE
        elif current_type == Params.MAPPING_NODE.get(Params.JOKE_NODE):
            logger.info(f"路由到 {Params.JOKE_NODE}")
            return Params.JOKE_NODE
        elif current_type == Params.MAPPING_NODE.get(Params.COUPLET_NODE):
            logger.info(f"路由到 {Params.COUPLET_NODE}")
            return Params.COUPLET_NODE
        elif current_type == Params.MAPPING_NODE.get(Params.DOCUMENT_NODE):
            logger.info(f"路由到 {Params.DOCUMENT_NODE}")
            return Params.DOCUMENT_NODE
        elif current_type == Params.MAPPING_NODE.get(Params.CODE_NODE):
            logger.info(f"路由到 {Params.CODE_NODE}")
            return Params.CODE_NODE
        elif current_type == Params.MAPPING_NODE.get(Params.DRAWING_NODE):
            logger.info(f"路由到 {Params.DRAWING_NODE}")
            return Params.DRAWING_NODE
        elif current_type == END:
            logger.info("路由到 END")
            return END
        else:
            logger.info(f"路由到 {Params.OTHER_NODE} (未知类型: {current_type})")
            return Params.OTHER_NODE

    def reflection_routing_func(state: State):
        """反思节点的路由：PASS则结束，FAIL则回到原节点重新生成"""
        current_type = state["type"]
        logger.info(f"reflection_routing_func: type = '{current_type}'")
        if current_type == END:
            logger.info("反思通过，路由到 END")
            return END
        else:
            logger.info(f"反思未通过，路由回 {current_type}_node")
            return f"{current_type}_node"

    builder = StateGraph(State)
    builder.add_node(Params.SUPERVISOR_NODE, supervisor_node)
    builder.add_node(Params.TRAVEL_NODE, travel_node)
    builder.add_node(Params.JOKE_NODE, joke_node)
    builder.add_node(Params.COUPLET_NODE, couplet_node)
    builder.add_node(Params.CODE_NODE, code_node)
    builder.add_node(Params.DOCUMENT_NODE, document_node)
    builder.add_node(Params.DRAWING_NODE, drawing_node)
    builder.add_node(Params.OTHER_NODE, other_node)
    builder.add_node(Params.REFLECTION_NODE, reflection_node)
    builder.add_edge(START, Params.SUPERVISOR_NODE)
    builder.add_conditional_edges(Params.SUPERVISOR_NODE, routing_func,
                                  [Params.TRAVEL_NODE, Params.JOKE_NODE, Params.COUPLET_NODE, Params.DOCUMENT_NODE, Params.CODE_NODE, Params.DRAWING_NODE, Params.OTHER_NODE, END])
    builder.add_edge(Params.TRAVEL_NODE, END)
    builder.add_edge(Params.JOKE_NODE, END)
    builder.add_edge(Params.COUPLET_NODE, END)
    builder.add_edge(Params.DOCUMENT_NODE, END)
    builder.add_edge(Params.CODE_NODE, END)
    builder.add_edge(Params.DRAWING_NODE, END)
    builder.add_edge(Params.OTHER_NODE, END)
    return builder.compile(checkpointer=InMemorySaver())