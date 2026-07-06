from typing import TypedDict, Annotated
from operator import add
import random
import asyncio
from langchain_classic.agents import create_react_agent
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph
from langgraph.constants import START, END
from openai.types.responses import response
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from settings.logger_manager import get_logger

logger = get_logger(__name__)

nodes = [
    "supervisor",
    "travel",
    "joke",
    "couplet",
    "other"
]

llm = ChatTongyi(
    model="qwen3.7-max",
    api_key="sk-25cd912ecdf3486785dff572b13c1da1"  # 百炼兼容模式
)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str


def other_node(state: State):
    logger.info("other_node")
    writer = get_stream_writer()
    writer({"node": ">>>> other_node"})

    return {"messages": [HumanMessage(content="我暂时无法回答这个问题")], "type": "other"}


def supervisor_node(state: State):
    logger.info("supervisor_node")
    writer = get_stream_writer()
    writer({"node": ">>>> supervisor_node"})

    prompts = """你是一个专业的客服助手，需要根据用户的问题进行任务分类，并将任务分给对应的Agent来执行。
    如果用户的问题是和旅游路线规划相关的，那就返回travel。
    如果用户的问题是和讲笑话相关的，那就返回joke。
    如果用户的问题是和对对联相关的，那就返回couplet。
    如果是其它问题，那就返回other。
    除了这几个选项外，不要返回任何其它的内容。
    """

    prompt_list = [
        {"role": "system", "content": prompts},
        {"role": "user", "content": state["messages"][0]},
    ]

    # 该问题已经有相应的type属性，交由其它节点处理完成了，可直接返回
    if "type" in state:
        writer({"supervisor_step": f"已获得{state['type']} 智能体处理结果"})
        return {"type": END}
    else:
        response = llm.invoke(prompt_list)
        typeRes = response.content
        writer({"supervisor_step": f"问题分类结果：{typeRes}"})
        if typeRes in nodes:
            return {"type": typeRes}
        else:
            raise ValueError(f"type is not in {nodes}")


def travel_node(state: State):
    logger.info("travel_node")
    writer = get_stream_writer()
    writer({"node": ">>>> travel_node"})

    system_prompt = """你是一个专业的旅行规划助手，根据用户的问题，生成一个旅游路线规划。请用中文回答，并返回一个不超过100字的规划结果"""

    prompt_list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]},
    ]
    
    try:
        client = MultiServerMCPClient(
            {
                "amap-maps-sse": {
                    "url": "https://mcp.amap.com/sse?key=77067076328eb3b90559c0cb1222ff81",
                    "transport": "streamable_http"
                },
            }
        )
        tools = asyncio.run(client.get_tools())
        agent = create_react_agent(
            model=llm,
            tools=tools
        )
        response = agent.invoke({"messages": prompt_list})
    except Exception as e:
        logger.warning(f"MCP 连接失败，使用备用方案: {e}")
        response = llm.invoke(prompt_list)

    writer({"travel_result": response.content if hasattr(response, 'content') else str(response)})

    return {"messages": [HumanMessage(content=response.content if hasattr(response, 'content') else str(response))],
            "type": "travel"}


def joke_node(state: State):
    logger.info("joke_node")
    writer = get_stream_writer()
    writer({"node": ">>>> joke_node"})

    system_prompt = """你是一个专业的笑话大师，根据用户的问题，写一个不超过100个字的笑话。"""

    prompt_list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["messages"][0]},
    ]

    response = llm.invoke(prompt_list)
    writer({"joke_result": response.content})

    return {"messages": [HumanMessage(content=response.content)], "type": "joke"}


def couplet_node(state: State):
    logger.info("couplet_node")
    writer = get_stream_writer()
    writer({"node": ">>>> couplet_node"})

    # 创建系统提示词模板
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
            你是一个专业的对联大师，你的任务是根据用户给出的上联，设计一个下联。
            回答时，可以参考下面的参考对联。
            参考对联：
                {samples}
            请用中文回答问题
        """),
        ("user", "{text}")
    ])

    query = state["messages"][0]

    # 构建向量模型
    embedding_model = DashScopeEmbeddings(model="text-embedding-v1")

    # 使用 Chroma 作为向量数据库（本地存储，无需额外服务）
    vector_store = Chroma(
        collection_name="couplet",
        embedding_function=embedding_model,
        persist_directory="./chroma_db"  # 数据持久化目录
    )

    samples = []

    # 相似度检索
    scored_results = vector_store.similarity_search_with_score(query, k=10)
    for doc, score in scored_results:
        samples.append(doc.page_content)

    prompt = prompt_template.invoke({"samples": samples, "text": query})
    writer({"couplet_result": prompt})

    response = llm.invoke(prompt)
    writer({"couplet_result": response.content})

    return {"messages": [HumanMessage(content=response.content)], "type": "couplet"}


def routing_func(state: State):
    if state["type"] == "travel":
        return "travel_node"
    elif state["type"] == "joke":
        return "joke_node"
    elif state["type"] == "couplet":
        return "couplet_node"
    elif state["type"] == END:
        return END

    return "other_node"


builder = StateGraph(State)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node)
builder.add_node("joke_node", joke_node)
builder.add_node("couplet_node", couplet_node)
builder.add_node("other_node", other_node)
# 添加边
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", routing_func,
                              ["travel_node", "joke_node", "couplet_node", "other_node", END])
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("couplet_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")

# 构建graph
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# if __name__ == "__main__":
#     query = "给我讲一个郭德纲的笑话"
#
#     config = {
#         "configurable": {
#             "thread_id": random.randint(1, 100000)
#         }
#     }
#
#     res = graph.invoke({"messages": [query]}, config, stream_mode="values")
#
#     logger.info(res["messages"][-1].content)