import re
from typing import Optional, List, Tuple
from text.index import REACT_PROMPT_TEMPLATE
from dotenv import load_dotenv
load_dotenv()
from LLM import HelloAgentsLLM
from ReAct.ToolExecutor import ToolExecutor
from utils.search import search

class ReActAgent:
    """
    ReAct 智能体核心实现
    """
    def __init__(self, llm_client: 'HelloAgentsLLM', tool_executor: 'ToolExecutor', max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history: List[str] = []   # 存储完整的 Thought / Action / Observation

    def run(self, question: str) -> Optional[str]:
        """
        运行 ReAct 循环，返回最终答案或 None
        """
        self.history = []      # 每次调用重置历史
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第 {current_step} 步 ---")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()   # 返回字符串描述
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )

            # 2. 调用 LLM 生成回复
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("❌ 错误：LLM 未返回有效响应")
                break

            # 3. 解析 LLM 输出 -> Thought 和 Action
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"💭 思考: {thought}")
                self.history.append(f"Thought: {thought}")   # 可选记录思考过程

            if not action:
                print("⚠️ 警告：未能解析出 Action，流程终止")
                break

            # 4. 处理结束指令
            if action.startswith("Finish"):
                match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                if match:
                    final_answer = match.group(1).strip()
                    print(f"🎉 最终答案: {final_answer}")
                    return final_answer
                else:
                    print("⚠️ 警告：Finish 格式错误，忽略")
                    continue

            # 5. 解析 Action 字符串 -> 工具名 + 输入
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print(f"⚠️ 警告：无法解析 Action 格式: {action}")
                self.history.append(f"Action: {action}")
                self.history.append("Observation: 无效的 Action 格式，请使用 工具名[输入]")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            self.history.append(f"Action: {action}")   # 保留原始 action 字符串

            # 6. 执行工具
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误：未找到工具 '{tool_name}'"
            else:
                try:
                    observation = tool_function(tool_input)   # 调用真实工具
                except Exception as e:
                    observation = f"工具执行异常: {e}"

            print(f"👀 观察: {observation}")
            self.history.append(f"Observation: {observation}")

        # 步数用尽或意外退出
        print("⏹️ 已达到最大步数或流程终止，无最终答案")
        return None

    # ---------- 解析辅助方法 ----------
    def _parse_output(self, text: str):
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # 关键修复：移除 $，使用贪婪匹配，捕获 Action: 之后的所有内容
        action_match = re.search(r"Action:\s*(.*)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        解析 "工具名[输入]" 格式，返回 (工具名, 输入)
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()
        return None, None


if __name__ == "__main__":
    # 4.1 初始化 LLM 客户端（从环境变量读取配置）
    llm = HelloAgentsLLM()
    # 4.2 初始化工具执行器，并注册工具
    executor = ToolExecutor()

    # 注册搜索工具（函数名与工具名对应）
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    executor.registerTool("Search", search_description, search)

    # 4.3 创建 ReAct 智能体
    agent = ReActAgent(
        llm_client=llm,
        tool_executor=executor,
        max_steps=5
    )

    # 4.4 运行一个测试问题
    question = "搜索https://datawhalechina.github.io/hello-agents这个网页"
    answer = agent.run(question)

    # 4.5 输出最终答案
    if answer:
        print("\n✅ 智能体最终返回：")
        print(answer)
    else:
        print("\n❌ 未能获得有效答案")
