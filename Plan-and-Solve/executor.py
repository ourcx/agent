from typing import List

from LLM import HelloAgentsLLM
from text.index import EXECUTOR_PROMPT_TEMPLATE


class Executor:
    def __init__(self, llm_client):
        self.llm_client = llm_client
    def init(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def execute(self, question: str, plan: List[str]) -> str:
        """
        根据计划，逐步执行并解决问题。
        """
        history = ""  # 累积历史步骤与结果
        final_answer = ""

        print("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages) or ""

            # 更新历史记录（为下一步提供上下文）
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
            print(f"✅ 步骤 {i+1} 结果: {response_text}")

            final_answer = response_text  # 最后一步的结果作为最终答案

        return final_answer