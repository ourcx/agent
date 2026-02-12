
from Planner import Planner
from executor import Executor
from LLM import HelloAgentsLLM

class Agent:
    """整合规划器与执行器的智能代理"""

    def __init__(self, llm_client):
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)

    def run(self, question: str) -> str:
        """完整流程：规划 → 执行 → 返回最终答案"""
        # 1. 生成计划
        plan = self.planner.plan(question)
        if not plan:
            return "无法生成有效的执行计划，请稍后重试。"

        # 2. 执行计划
        answer = self.executor.execute(question, plan)
        return answer

# ========== 5. 使用示例 ==========
if __name__ == "__main__":
    # 假设你已经有一个符合接口的 LLM 客户端
    llm = HelloAgentsLLM()
    agent = Agent(llm)
    result = agent.run("什么是Python装饰器？")
    print(f"\n🎯 最终答案：\n{result}")
