# 假设 llm_client.py 和 memory.py 已定义
from LLM import HelloAgentsLLM
# from memory import Memory
from text.index import INITIAL_PROMPT_TEMPLATE, REFINE_PROMPT_TEMPLATE, REFLECT_PROMPT_TEMPLATE
from Memory import Memory

class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        print(f"\n--- 开始处理任务 ---\n任务: {task}")

        # --- 1. 初始执行 ---
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        # --- 2. 迭代循环:反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i + 1}/{self.max_iterations} 轮迭代 ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            # b. 检查是否需要停止
            if "无需改进" in feedback:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)

        final_code = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        """一个辅助方法，用于调用LLM并获取完整的流式响应。"""
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages=messages) or ""
        return response_text


if __name__ == "__main__":
    from LLM import HelloAgentsLLM
    from Memory import Memory

    # 1. 初始化大模型客户端
    llm = HelloAgentsLLM()

    # 2. 创建反思代理（最多迭代2轮，节省时间）
    agent = ReflectionAgent(llm_client=llm, max_iterations=2)

    # 3. 定义测试任务
    task = "写一个Python函数，接收两个数字并返回它们的和"

    # 4. 运行代理
    print(f"\n🚀 开始测试任务: {task}\n")
    final_code = agent.run(task)

    # 5. 输出最终代码
    print("\n" + "=" * 50)
    print("✅ 最终生成的代码：")
    print("=" * 50)
    print(final_code)