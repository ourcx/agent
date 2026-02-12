import ast
import re
from typing import List

from text.index import PLANNER_PROMPT_TEMPLATE


class Planner:
    # ... __init__ 保持不变 ...
    def __init__(self,llm_client):
        self.llm_client = llm_client

    def plan(self, question: str) -> List[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        print("--- 正在生成计划 ---")
        response_text = self.llm_client.think(messages=messages) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        plan = self._parse_plan(response_text)
        if not plan:
            print(f"❌ 解析计划失败，原始响应: {response_text}")
            return []
        return plan

    def _parse_plan(self, text: str) -> List[str]:
        """从 LLM 响应中安全提取计划列表"""
        # 1. 优先提取 ```python ... ``` 中的内容
        match = re.search(r"```python\s*([\s\S]*?)\s*```", text)
        if match:
            code = match.group(1).strip()
            try:
                parsed = ast.literal_eval(code)
                if isinstance(parsed, list):
                    return parsed
            except:
                pass

        # 2. 尝试将整个响应解析为列表
        try:
            parsed = ast.literal_eval(text.strip())
            if isinstance(parsed, list):
                return parsed
        except:
            pass

        # 3. 降级方案：按行拆分（适用于带序号或项目符号的列表）
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            # 去除常见的前缀符号：-, *, 数字点
            clean = re.sub(r'^[\-\*\d\.\s]+', '', line).strip()
            if clean and not clean.startswith(('步骤', '计划')):
                lines.append(clean)
        return lines if lines else []