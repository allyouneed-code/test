import json
import os
from typing import List, Dict, Any, Optional
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
import textwrap # 确保导入 textwrap


class DemandElements(BaseModel):
    """对需求中涉及的核心要素的分类提取。"""
    subject_elements: List[str] = Field(description="需求涉及的实体，如：用户、系统、设备。")
    behavior_elements: List[str] = Field(description="需求描述的动作或操作，如：创建、查询、更新、计算。")
    object_elements: List[str] = Field(description="行为作用的目标，如：数据、资源、服务。")
    property_elements: List[str] = Field(description="实体或行为的特征描述，如：性能指标、安全等级、输入类型。")
    condition_elements: List[str] = Field(description="需求成立的前置条件或限制，如：时间约束、环境要求、异常条件（如除以零）。")

class LogicalRelationships(BaseModel):
    """描述不同需求要素之间存在的逻辑关联。"""
    temporal_relationships: List[str] = Field(description="需求执行的先后顺序，例如：'登录后才能下单'。")
    dependency_relationships: List[str] = Field(description="需求间的功能依赖，例如：'支付模块依赖用户认证'。")
    constraint_relationships: List[str] = Field(description="需求必须满足的规则或限制，例如：'输入必须是数字' 或 '响应时间≤200ms'。")
    hierarchical_relationships: List[str] = Field(description="需求的抽象与具体层级关系，例如：'核心功能→子功能→操作步骤'。")

class StructuredRequirement(BaseModel):
    """
    用户测试需求的结构化分析（根模型）。
    此模型专注于提取抽象的需求要素和逻辑关系，而非具体用例。
    """
    demand_elements: DemandElements = Field(description="对需求进行解构，识别出的核心要素。")
    logical_relationships: LogicalRelationships = Field(description="各要素之间存在的内在逻辑关系。")

# --------------------------------

class RequirementAnalyzer:
    """
    分析自然语言需求，并根据用户预定义的抽象模板，
    将其转换为结构化的要素和逻辑关系。
    """

    def __init__(self):
        """使用能够进行结构化输出的LLM初始化需求分析器。"""
        self.llm = init_chat_model("openai:gpt-3.5-turbo-1106", temperature=0.0)
        self.structured_llm = self.llm.with_structured_output(StructuredRequirement)

    def analyze(self, requirement: str, code_context: str) -> Dict[str, Any]:
        """
        使用LLM分析用户需求，并强制其填充您预定义的抽象结构模板。

        Args:
            requirement: 来自用户的自然语言测试需求。
            code_context: 用于为分析提供上下文的源代码。

        Returns:
            一个严格遵循 `StructuredRequirement` 模板的字典。
        """
        # --- 这是核心改进：一个更智能、更具引导性的Prompt ---
        prompt = f"""
        **目标:** 你是一个顶级的需求分析师。你的任务是深度解析用户需求和源代码，然后严格按照预定义的JSON结构，提取高层次的“需求要素”和“逻辑关系”。

        **用户需求:**
        {requirement}

        **源代码上下文:**
        ```python
        {code_context}
        ```

        **分析指南 (请严格遵守):**

        1.  **思维链 (Chain of Thought):**
            * 首先，通读代码和需求，理解其核心功能是根据 'operation' 参数执行不同的数学计算。
            * 其次，识别出代码中的主要逻辑分支（if/elif/else）和前置检查（isinstance）。
            * 然后，将这些代码逻辑与用户需求中的关键词（如“所有操作”、“除以零”）进行映射。
            * 最后，根据下面的详细定义，将分析结果填入JSON模板。

        2.  **字段填充详解:**

            * `demand_elements`:
                * `subject_elements`: 识别核心主体。对于这段代码，主体就是 `'calculate' 函数` 本身。
                * `behavior_elements`: **必须列出代码中所有具体、互斥的行为**。例如：`'执行加法运算'`, `'执行除法运算'`, `'校验输入类型'`。**不要使用** `'测试'` 或 `'操作'` 这种模糊的词。
                * `object_elements`: 描述行为作用的对象。
                * `property_elements`: 识别影响行为的关键属性。
                * `condition_elements`: 明确列出导致特殊行为的**条件**，例如：`'当除数为零时'`。

            * `logical_relationships`:
                * `temporal_relationships`: **只描述严格的执行顺序** (例如：必须先A后B)。如果代码中没有，**此字段必须为空数组 `[]`**。
                * `dependency_relationships`: **只描述功能依赖** (例如：函数A调用了函数B)。代码中的 `if...elif` 分支是**互斥逻辑**，**不是依赖关系**。如果不存在函数调用等依赖，**此字段必须为空数组 `[]`**。
                * `constraint_relationships`: 描述**必须遵守的规则**。例如，从 `isinstance` 检查可以得出输入类型约束。从 `if/elif` 结构可以得出操作是互斥的。
                * `hierarchical_relationships`: **必须将实际代码结构映射到层级关系上**。例如，`'calculate (核心功能) -> 输入校验 (前置检查)'`。**不要**只是重复模板示例。

        请现在开始分析，并给出结果。
        """
        try:
            structured_response = self.structured_llm.invoke(prompt)
            return structured_response.model_dump()
        except Exception as e:
            print(f"需求分析过程中出错: {e}")
            return {
                "demand_elements": {
                    "subject_elements": [], "behavior_elements": [], "object_elements": [],
                    "property_elements": [f"解析错误: {e}"], "condition_elements": []
                },
                "logical_relationships": {
                    "temporal_relationships": [], "dependency_relationships": [],
                    "constraint_relationships": [], "hierarchical_relationships": []
                }
            }

if __name__ == '__main__':
        # ------------------------- 环境配置 -------------------------
    os.environ["OPENAI_API_KEY"] = "sk-OjjN3nmNeSZxEE8c2QJz985fdY3b9XegsKi7lTcl8z6Sr2de"
    os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"

    sample_requirement = "为 'calculate' 函数编写一个全面的测试套件。确保测试所有操作（'add', 'subtract', 'multiply', 'divide'）。同时，专门测试除以零和无效输入类型的边缘情况。"
    
    sample_code = textwrap.dedent("""
        def calculate(a, b, operation):
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise TypeError("Inputs must be numeric")
            if operation == 'add':
                return a + b
            if operation == 'subtract':
                return a - b
            if operation == 'multiply':
                return a * b
            if operation == 'divide':
                if b == 0:
                    return "Error: Division by zero"
                return a / b
            return None
    """)

    analyzer = RequirementAnalyzer()
    structured_req = analyzer.analyze(requirement=sample_requirement, code_context=sample_code)
    
    print("--- 需求分析结果 ---")
    print(json.dumps(structured_req, indent=2, ensure_ascii=False))