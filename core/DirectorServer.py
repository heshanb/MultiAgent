import random
from Director import graph
from settings.logger_manager import get_logger

logger = get_logger(__name__)

query = "给我对一个对联，上联是：金榜题名时" #input("请输入您的问题：")

config = {
    "configurable": {
        "thread_id": random.randint(1, 100000)
    }
}

res = graph.invoke({"messages": [query]}, config, stream_mode="values")
logger.info(res)
logger.info(res["messages"][-1].content)