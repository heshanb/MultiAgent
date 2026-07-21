from langchain_community.chat_models import ChatTongyi
from settings.Define import Params, PathConfig
import os
from logging import getLogger
logger = getLogger(__name__)

class General_Programming:
    def __init__(self, file_path: str, state_list: list):
        self.system_prompt = self.generate_prompt(file_path)
        self.state_list = state_list

    def generate_prompt(self, file_path: str) -> str:
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

        return system_prompt

    def get_model_response(self, message, llm) -> str:
        if hasattr(message, 'content'):
            message_content = message.content
        else:
            message_content = str(message)

        conversation_history = []
        for msg in self.state_list[:-1]:
            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            if isinstance(msg, dict) and msg.get("role") == "assistant":
                conversation_history.append({"role": "assistant", "content": content})
            elif "HumanMessage" in str(type(msg)) or (isinstance(msg, dict) and msg.get("role") == "user"):
                conversation_history.append({"role": "user", "content": content})

        prompt_list = [
                          {"role": "system", "content": self.system_prompt},
                      ] + conversation_history + [
                          {"role": "user", "content": message_content},
                      ]
        try:
            response = llm.invoke(prompt_list)
            content = response.content if hasattr(response, 'content') else str(response)
            return content
        except Exception as e:
            logger.error(f"code_node LLM 调用失败: {str(e)[:50]}")
            content = "抱歉，当前服务暂时不可用，请稍后重试。"

            return content

    # 检查是否需要保存修改后的代码文件
    def check_save_code(self, content: str, file_path: str) -> str:
        original_content = content
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
                    return content
            except Exception as e:
                logger.error(f"保存代码文件失败: {str(e)}")

        return original_content

