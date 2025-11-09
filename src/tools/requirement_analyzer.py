import json
import os
from typing import List, Dict, Any, Optional, Literal
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
import textwrap 
from ..llm.client import get_llm_client

#1 U: 被测单元
class UnitUnderTest(BaseModel):
    """
    定义了规约的主体 (Unit Under Test, UUT)。
    """
    identifier: str = Field(description="唯一标识符")
    desc: str = Field(description="能力描述")
    source: Optional[List[str]] = Field(default=None, description="需求溯源")
    
#2 I: 接口规约
class InputParameter(BaseModel):
    name: str = Field(description="输入信息的标识符")
    desc: str = Field(description="输入信息描述")
    constraints: List[str] = Field(description="对该信息的语义限制列表") # 保持 List[str] 以匹配示例 [cite: 150]

class OutputParameter(BaseModel):
    name: str = Field(description="输出信息的标识符")
    constraints: List[str] = Field(description="对返回值的预期约束")
    semantics: str = Field(description="对返回值的业务含义的自然语言描述")


class ExternalDependency(BaseModel):
    """定义需要被 Mock 的外部依赖。"""
    name: str = Field(description="依赖组件的名称。")
    info_needed: List[str] = Field(description="所需外部信息或方法")
    contract: str = Field(description="对这些被调用方法行为的语义描述")

class InterfaceSpecification(BaseModel):
    """
    定义 UUT 的静态边界。
    """
    input_parameters: List[InputParameter] = Field(description="输入信息的集合 。")
    output_specification: List[OutputParameter] = Field(description="输出信息的集合。")
    external_dependencies: List[ExternalDependency] = Field(description="外部依赖的集合")

#3 B: 行为模型
class FunctionalScenario(BaseModel):
    """定义单个功能场景。"""
    id: str = Field(description="场景的唯一标识符。")
    description: str = Field(description="场景的简短自然语言描述。")
    pre_conditions: List[str] = Field(description="此场景发生必须满足的逻辑谓词")
    post_conditions: List[str] = Field(description="执行后必须为真的逻辑谓词")

class ErrorScenario(BaseModel):
    """定义单个异常/错误场景。"""
    id: str = Field(description="场景的唯一标识符")
    description: str = Field(description="错误场景的简短描述。")
    error_conditions: List[str] = Field(description="触发错误的条件")
    expected_outcome: str = Field(description="特定的、受控的错误响应")

class BehavioralModel(BaseModel):
    """
    定义 UUT 的动态逻辑和业务规则。
    """
    functional_scenarios: List[FunctionalScenario] = Field(description="功能场景的集合。")
    error_scenarios: List[ErrorScenario] = Field(description="错误场景的集合。")

# --- 联合的中间模型 ---
class StaticSpecification(BaseModel):
    """包含 (U) 和 (I) 的静态规约。"""
    unit_under_test: UnitUnderTest = Field(description="模型的锚点，标识测试对象 (U)。")
    interface_specification: InterfaceSpecification = Field(description="UUT 的静态签名和边界 (I)。")


#4 C: 约束向量
class Partition_valid(BaseModel):
    """有效等价类分区。"""
    description: str = Field(description="当前参数param_name对应的有效等价类的描述")
    range_or_set: str = Field(description="分区的取值范围或枚举集合的字符串表示")
    derived_values: List[str] = Field(
        description="从该分区派生出的边界值列表"
    )
class Partition_invalid(BaseModel):
    """无效等价类分区。"""
    description: str = Field(description="当前参数param_name对应的无效等价类的描述")
    range_or_set: str = Field(description="分区的取值范围、枚举集合或语义表示")
    derived_values: List[str] = Field(
        description="从该分区派生出的边界值列表"
    )

class ParamConstraints(BaseModel):
    """特定参数的完整等价类划分。"""
    param_name: str = Field(description="关联的参数名")
    P_valid: List[Partition_valid] = Field(description="有效等价类集，只能包含合法值")
    P_invalid: List[Partition_invalid] = Field(description="无效等价类集，包含非法值")

class ConstraintVectors(BaseModel):
    """约束向量 (C) 的根模型。"""
    param_constraints: List[ParamConstraints] = Field(description="等价类划分集 (E)")


class FullTestModel(BaseModel):
    """
    M=(U, I, B, C) 模型。
    """
    unit_under_test: UnitUnderTest = Field(description="包含 (U) 的语义规约")
    interface_specification: InterfaceSpecification = Field(description="包含 (I) 的语义规约")
    behavioral_model: BehavioralModel = Field(description="包含 (B) 的语义规约")
    constraints: ConstraintVectors = Field(description="包含 (C) 的约束向量")

# --------------------------------

class RequirementAnalyzer:
    """
    分析自然语言需求，并根据用户预定义的抽象模板，
    将其转换为结构化的要素和逻辑关系。
    """

    def __init__(self):
        """使用能够进行结构化输出的LLM初始化需求分析器。"""
        self.llm = get_llm_client(temperature=0.0)
        # 专家 1: 提取 U 和 I
        self.static_extractor = self.llm.with_structured_output(
            StaticSpecification,
            method="function_calling"
        )
        # 专家 2: 提取 B
        self.behavior_extractor = self.llm.with_structured_output(
            BehavioralModel,
            method="function_calling"
        )

    def analyze_static(self, requirement: str, code_context: str) -> StaticSpecification:
        """
        分析需求和代码，以填充形式化的 M=(U, I) 模型。

        Args:
            requirement: 自然语言测试需求。
            code_context: 作为上下文的源代码。

        Returns:
            一个严格遵循 `StaticSpecification` 的字典。
        """
        
        # --- 深度优化的 PROMPT：针对 U, I模型 ---
        prompt = f"""
        **目标:** 你是一名顶级的软件测试分析师。你的任务是深度解析用户需求，并将它们严格映射到形式化的语义模型 M=(U, I) 中，输出 Pydantic 模式。

        **用户需求:**
        {requirement}

        **分析指南 (严格遵守):**

        1.  **unit_under_test (U)**:
            * `identifier`: 需求的核心功能点名称唯一标识符。
            * `desc`: 对该业务能力的简短、无歧义的自然语言描述。
            * `source`: 列出需求中提到的任何溯源信息（如章节、条款等）。如果未提及，必须返回 `null`。

        2.  **interface_specification (I)**:
            * `input_parameters `: 识别需求中提到的所有输入参数，并为每个参数提供：
                * `name`: 参数名称。
                * `desc`: 参数的自然语言描述。
                * `constraints`: 参数的语义限制列表。
            * `output_specification (P_out)`: 描述需求的预期输出，包括：
                * `name`: 输出名称。
                * `constraints`: 对返回值的预期约束。
                * `semantics`: 对返回值的业务含义的自然语言描述。
            * `external_dependencies (D_ext)`: 识别需求中提到的任何外部依赖。如果未提及，必须返回一个空列表 `[]`。

        请立即开始分析，并返回符合 `StaticSpecification` 的 JSON 对象。
        """
        try:
            structured_response = self.static_extractor.invoke(prompt)
            return structured_response
        except Exception as e:
            print(f"需求分析过程中出错: {e}")
            raise e
        
    def analyze_behavior(self, requirement: str, code_context: str) -> BehavioralModel:
        """
        分析需求和代码，以填充形式化的需求行为模型。

        Args:
            requirement: 自然语言测试需求。
            code_context: 作为上下文的源代码。

        Returns:
            一个严格遵循 `BehavioralModel` 的字典。
        """
        
        # --- 深度优化的 PROMPT：针对 B 模型 ---
        prompt = f"""
        **目标:** 你是一名顶级的软件测试分析师。你的任务是深度解析用户需求，并将它们严格映射到形式化的语义模型BehavioralModel中。

        **用户需求:**
        {requirement}


        **分析指南 (严格遵守):**


        **`behavioral_model (B)`:**

            * **`functional_scenarios (S_func)`:** 为每个有效的功能创建一个场景。

                * `id`:必须提供一个唯一的场景ID。

                * `description`: 功能场景描述。

                * `pre_conditions`: 前置条件的简短精炼描述。

                * `post_conditions`: 预期的结果的简短精炼描述。

            * **`error_scenarios (S_err)`:** 为需求中提到的每个错误情况、异常或边缘情况创建一个场景。

                * `id`:必须提供一个唯一的场景ID。

                * `description`: 错误场景描述。

                * `error_conditions`: 触发条件的简短精炼描述。

                * `expected_outcome`: 确切的错误响应。

        请立即开始分析，并返回符合 `BehavioralModel` 的 JSON 对象。
        """
        try:
            structured_response = self.behavior_extractor.invoke(prompt)
            return structured_response
        except Exception as e:
            print(f"需求分析过程中出错: {e}")
            raise e

class ConstraintDeriver:
    """
    从 M=(U, I, B) 派生 C=(E, V)。
    """
    def __init__(self):
        self.llm = get_llm_client(temperature=0.0)
        # 准备两个结构化输出 LLM
        self.llm_E_extractor = self.llm.with_structured_output(ConstraintVectors)
    
    def derive(self, i_spec: InterfaceSpecification, requirement: str) -> List[ConstraintVectors]:
        """派生参数的 E (等价类)和V(边界值)"""
        print("--- 正在派生 E (等价类)和V(边界值)... ---")
        
        # 1. 为所有参数构建上下文
        param_contexts = []
        for param in i_spec.input_parameters:
            param_contexts.append(
                f"  - 参数: {param.name}\n"
                f"    描述: {param.desc}\n"
                f"    约束: {param.constraints}\n"
            )
        
        context_str = "\n".join(param_contexts)

        prompt = f"""
        **目标:** 你是一个顶级的测试分析专家。你的任务是根据用户需求和参数信息输入为每个参数生成“等价类划分集——ConstraintVectors”。

        ```
        **用户需求:**
        {requirement}

        **所有参数信息:**
        {context_str}

        **指南 (严格遵守):**
        1.  **遍历每个参数**，为其创建一个 `EquivalenceClassSet` 对象。
        2.  **分析 `P_invalid` (无效等价类):**
        3.  **分析 `P_valid` (有效等价类):**
        4.  **填充 `range_or_set`:**
        5.  **派生 `derived_values`:**
            * 对于每个分区，生成具体的测试值 (V)，并标明其 BVA 属性 (OnPoint, OffPoint_Invalid, OffPoint_Valid, Special, Nominal)。
        
        请严格按照 `ConstraintVectors` Pydantic 模型返回,其值为一个列表。
        """
        try:
            response = self.llm_E_extractor.invoke(prompt)
            return response
        except Exception as e:
            print(f"批量派生 E 时出错: {e}")
            return []

class TestModelAssembler:
    """
    (外观类) 封装了构建 M(U,I,B,C) 的所有步骤。
    """
    def __init__(self):
        """初始化所有需要的服务"""
        self.analyzer = RequirementAnalyzer()
        self.deriver = ConstraintDeriver()


    def build(self, requirement: str, code_context: str) -> FullTestModel:
        """
        执行 M(U,I,B,C) 的完整构建流程。
        """
        
        # 步骤 1: 分析 U 和 I
        print("---  分析M(U,I,B,C) ---")
        static_spec = self.analyzer.analyze_static(requirement, code_context)
        # structured_req = static_spec.model_dump()
        # print(json.dumps(structured_req, indent=2, ensure_ascii=False))

        # 分析 B
        behavior_spec = self.analyzer.analyze_behavior(requirement, code_context)
        # structured_req = behavior_spec.model_dump()
        # print(json.dumps(structured_req, indent=2, ensure_ascii=False))
        
        # 派生 C
        constraint_vectors = self.deriver.derive(
            i_spec=static_spec.interface_specification,
            requirement=requirement
        )
        # constraint_vectors = self.deriver._derive_all_V(static_spec.interface_specification, requirement)
        # structured_req = constraint_vectors.model_dump()
        # print(json.dumps(structured_req, indent=2, ensure_ascii=False))
        
        # 步骤 4: 组装最终的扁平化 M(U,I,B,C)
        full_model = FullTestModel(
            unit_under_test=static_spec.unit_under_test,
            interface_specification=static_spec.interface_specification,
            behavioral_model=behavior_spec,
            constraints=constraint_vectors
        )
        structured_req = full_model.model_dump()
        print(json.dumps(structured_req, indent=2, ensure_ascii=False))
        
        return full_model

if __name__ == '__main__':
        # ------------------------- 环境配置 -------------------------
    os.environ["OPENAI_API_KEY"] = "sk-OjjN3nmNeSZxEE8c2QJz985fdY3b9XegsKi7lTcl8z6Sr2de"
    os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"

    sample_requirement = """
    实现一个 'calculate' 函数，它接受两个数字（a, b）和一个操作名（operation）。
    1. 它必须支持 'add'（加）、'subtract'（减）、'multiply'（乘）和 'divide'（除）。
    2. 函数必须首先校验 a 和 b 必须为数字，如果任一输入不是数字，则应抛出 TypeError。
    3. 在执行 'divide' 操作时，如果 b 为 0，函数必须返回一个特定的错误字符串 "Error: Division by zero"。
    4. 如果 'operation' 参数不是上述四种有效操作之一，函数应返回 None。
    """

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

    
    # print("--- 需求分析结果 ---")
    # print(json.dumps(structured_req, indent=2, ensure_ascii=False))

    assembler = TestModelAssembler()
        
    full_test_model = assembler.build(
        requirement=sample_requirement, 
        code_context=sample_code
    )
    structured_req = full_test_model.model_dump()
    
    #打印最终的完整结果
    print("\n\n" + "*"*50)
    print("--- 最终 M(U,I,B,C) 完整模型 [扁平化] JSON ---")
    print("*"*50)
    print(json.dumps(structured_req, indent=2, ensure_ascii=False))