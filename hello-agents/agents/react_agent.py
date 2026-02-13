# my_react_agent.py
import re
from typing import Optional, List, Tuple
from hello_agents import ReActAgent, HelloAgentsLLM, Config, Message, ToolRegistry
from dotenv import load_dotenv
from text.index import MY_REACT_PROMPT
from utils.search import search

class MyReActAgent(ReActAgent):
    """
    重写的ReAct Agent - 推理与行动结合的智能体
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            tool_registry: ToolRegistry,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            max_steps: int = 5,
            custom_prompt: Optional[str] = None
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt else MY_REACT_PROMPT
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

    def run(self, input_text: str, **kwargs) -> str:
        """运行ReAct Agent"""
        self.current_history = []
        current_step = 0

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 1. 构建提示词
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(
                tools=tools_desc,
                question=input_text,
                history=history_str
            )

            # 2. 调用LLM
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)

            # 3. 解析输出
            thought, action = self._parse_output(response_text)

            # 4. 检查完成条件
            if action and action.startswith("Finish"):
                final_answer = self._parse_action_input(action)
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(final_answer, "assistant"))
                return final_answer

            # 5. 执行工具调用
            if action:
                tool_name, tool_input = self._parse_action(action)
                observation = self.tool_registry.execute_tool(tool_name, tool_input)
                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")

        # 达到最大步数
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer


if __name__ == "__main__":
    # 加载环境变量（例如 OPENAI_API_KEY）
    load_dotenv()

    # 初始化 LLM（假设 HelloAgentsLLM 默认使用 OpenAI）
    llm = HelloAgentsLLM()

    # 创建工具注册表并注册一个模拟的搜索工具
    tool_registry = ToolRegistry()

    def search_web(query: str) -> str:
        """模拟网页搜索，返回固定结果。实际使用时可替换为真实的搜索引擎 API"""
        # 这里简单模拟根据关键词返回不同答案
        if "生命" in query or "意义" in query:
            return "根据网络搜索结果：生命的意义是一个哲学问题，不同文化有不同解读。例如，存在主义认为意义由个体赋予。"
        else:
            return f"未找到关于“{query}”的明确信息，请尝试其他关键词。"

    # 创建 ReAct Agent 实例
    react_agent = MyReActAgent(
        name="ReActAgent",
        llm=llm,
        tool_registry=tool_registry,
        system_prompt=MY_REACT_PROMPT,   # 使用预定义的 ReAct 提示词模板
        max_steps=5                      # 最大推理步数
    )

    # 运行智能体
    response = react_agent.run(
        "生命的意义是什么？如果需要，可以搜索网页，可能需要多次搜索才能得到全面答案。"
    )
    print("\n最终答案：")
    print(response)
