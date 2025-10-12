import json
import os
import time
from typing import TypedDict, Literal, Dict, Any, List
from langchain_community.callbacks import get_openai_callback
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# ------------------------- 导入自定义工具 -------------------------
from ..tools.code_inspector import CodeAnalyzer
from ..tools.code_executer import execute_tests_and_get_report
from ..tools.requirement_analyzer import RequirementAnalyzer
from ..tools.mutation_tester import run_mutation_test

# ------------------------- 导入配置 -------------------------
def load_config():
    """
    加载配置的函数。
    优先级顺序: 环境变量 > config.json > 默认值
    """
    # 1. 设置默认值
    config = {
        "max_retries": 3,
        "coverage_threshold": 0.8,
        "mutation_threshold": 0.9,
        "logic_filename": "logic_module.py",
        "test_filename": "test_script.py"
    }

    # 2. 尝试从 config.json 文件加载
    try:
        # 假设 config.json 在项目的根目录
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
            config.update(file_config.get("workflow_settings", {}))
            config.update(file_config.get("code_executer_settings", {})) 
    except (FileNotFoundError, json.JSONDecodeError):
        print("--- config.json not found or invalid, using default settings. ---")
        pass # 文件不存在或格式错误则忽略
  
    return config

# ------------------------- 环境配置 -------------------------
# 建议使用环境变量管理 API Keys，而不是硬编码
os.environ["OPENAI_API_KEY"] = "sk-OjjN3nmNeSZxEE8c2QJz985fdY3b9XegsKi7lTcl8z6Sr2de"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"

# ------------------------- 状态与模型定义 -------------------------
class IterationLog(TypedDict):
    """记录单次迭代的关键信息"""
    iteration: int
    test_code: str
    pass_rate: float
    coverage: float
    feedback: str

class WorkflowState(TypedDict):
    """定义工作流中传递的状态"""
    # 核心数据
    code: str                  # 待测试的源代码
    requirement: str           # 用户的测试需求
    analysis_report: str       # 代码静态分析报告 (JSON 字符串)
    structured_requirement: str    # 结构化需求报告 (JSON 字符串)
    generation_prompt: str     # 用于生成测试用例的最终 Prompt
    test_code: str             # 生成的测试用例代码
    
    # 结果与迭代
    coverage: float            # 测试覆盖率
    pass_rate: float           # 测试通过率
    execution_feedback: str    # 来自测试执行器的反馈 (例如，未覆盖的行)
    evaluation_result: str     # 评估者的最终决定 ("pass" or "not pass")
    evaluation_feedback: str   # 评估者给出的改进建议
    retry_count: int           # 重试次数
    iteration_history: List[IterationLog] # 迭代日志
    
    # 成本与效率指标
    start_time: float               # 工作流开始时间戳
    total_execution_time: float     # 总执行时间
    total_prompt_tokens: int        # 累计 Prompt Token 消耗
    total_completion_tokens: int    # 累计 Completion Token 消耗
    total_tokens: int               # 累计总 Token 消耗

    # 标记和记录变异测试工具错误
    mutation_test_has_error: bool
    mutation_error_details: str
    mutation_details: str
    mutation_score: float           # 变异测试得分

class TestEvaluation(BaseModel):
    """用于评估器结构化输出的 Schema"""
    grade: Literal["pass", "not pass"] = Field(
        description="根据测试结果（通过率、覆盖率）和反馈，决定测试用例是否达标。"
    )
    feedback: str = Field(
        description="如果测试用例不达标 (not pass)，提供清晰、具体的反馈，指导如何改进测试用例。"
    )

# ------------------------- 工作流构建器 -------------------------

class TestGenerationWorkflow:
    """
    基于图的自动化测试生成工作流。
    集成了代码分析、测试生成、代码执行和循环评估。
    """
    def __init__(self, config: dict):
        self.llm = init_chat_model("openai:gpt-3.5-turbo-1106", temperature=0.2)
        self.structured_evaluator = self.llm.with_structured_output(
            TestEvaluation,
            method="function_calling"
        )
        # 从传入的配置字典中获取参数
        self.coverage_threshold = config["coverage_threshold"]
        self.max_retries = config["max_retries"]
        self.logic_filename = config["logic_filename"]
        self.test_filename = config["test_filename"]
        self.mutation_threshold = config["mutation_threshold"]

    # ------------------------- 流程中的节点 (Nodes) -------------------------

    def code_analyzer_node(self, state: WorkflowState) -> dict:
        """
        节点1: 代码分析
        使用 CodeAnalyzer 对源代码进行深度静态分析。
        """
        print("--- Step 1: Analyzing Source Code ---")
        analyzer = CodeAnalyzer(state["code"])
        analysis_result = analyzer.analyze()
        report_str = json.dumps(analysis_result, indent=2, ensure_ascii=False)
        return {"analysis_report": report_str}

    def requirement_analyzer_node(self, state: WorkflowState) -> dict:
        """
        节点2: 需求分析
        使用 RequirementAnalyzer 对用户需求进行结构化分析。
        """
        print("--- Step 2: Analyzing Requirement ---")
        try:
            # 使用 get_openai_callback 上下文管理器来追踪 Token
            with get_openai_callback() as cb:
                analyzer = RequirementAnalyzer()
                # 将需求和代码都传递给分析器以获得更好的上下文
                analysis_result = analyzer.analyze(state["requirement"], state["code"])
                report_str = json.dumps(analysis_result, indent=2, ensure_ascii=False)           
                print(f"  -> LLM Call Tokens (Requirement Analysis): {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")

            # 将捕获到的 Token 累加到总数中
            return {
                "structured_requirement": report_str,
                "total_prompt_tokens": state["total_prompt_tokens"] + cb.prompt_tokens,
                "total_completion_tokens": state["total_completion_tokens"] + cb.completion_tokens,
                "total_tokens": state["total_tokens"] + cb.total_tokens,
            }
            
        except Exception as e:
            error_msg = f"Requirement analysis failed: {e}"
            print(f"ERROR: {error_msg}")
            # 即使失败，也返回错误信息以便调试，但不增加Token
            return {"structured_requirement": json.dumps({"error": error_msg})}

    def prompt_organizer_node(self, state: WorkflowState) -> dict:
        """
        节点3: Prompt 组织
        整合用户需求和代码分析报告，构建一个高质量的 Prompt。
        """
        print("--- Step 3: Organizing Prompt ---")
        module_name = os.path.splitext(self.logic_filename)[0]
        prompt_template = f"""
        **目标:** 生成一个全面的 pytest 测试套件。

        **源代码模块名:** `{module_name}`

        **用户原始需求:**
        {state['requirement']}

        **结构化需求分析 (您设计测试用例的主要指南):**
        这个结构化分析将用户的请求分解为具体的、可操作的测试场景。
        ```json
        {state['structured_requirement']}
        ```

        **待测试源代码:**
        ```python
        {state['code']}
        ```

        **静态代码分析报告:**
        这份报告提供了对代码结构、复杂性和潜在问题的深入洞察。用它来进一步完善您的测试用例。
        ```json
        {state['analysis_report']}
        ```

        **任务:**
        基于以上所有信息，，编写一个 pytest 测试套件。
        - **至关重要，您必须从 `{module_name}` 模块导入待测试的函数。** 例如: `from {module_name} import your_function_name`。
        - 测试代码必须是完整且可运行的。
        - 确保您生成的测试覆盖了结构化分析中概述的所有场景。
        - **只在 markdown 的代码块 (```python ... ```) 中输出测试套件的 Python 代码。** 不要包含任何其他文本或解释。
        """
        return {"generation_prompt": prompt_template}
    
    def test_generator_node(self, state: WorkflowState) -> dict:
        """
        节点4: 测试用例生成
        调用 LLM，根据 Prompt 生成测试代码。
        如果存在上一轮的反馈，会一并考虑。
        """
        print("--- Step 4: Generating Test Cases ---")
        
        # 检查是否存在反馈。如果存在，说明是迭代优化阶段。
        if feedback := state.get("evaluation_feedback"):
            print("  -> Incorporating feedback from previous run.")
            # 创建一个专注于“修复”的、全新的Prompt
            prompt = f"""
            **任务: 修复并改进现有的测试用例**

            你是一名资深的测试工程师。上一轮生成的测试用例不够健壮，未能通过质量检测。
            请根据下面提供的“改进建议”，对“上一轮的测试代码”进行修改，以解决所有已知问题。

            **上一轮的测试代码:**
            ```python
            {state['test_code']}
            ```

            **改进建议/上一轮运行结果的问题:**
            ---
            {feedback}
            ---

            **待测试源代码 (供参考):**
            ```python
            {state['code']}
            ```

            **输出要求:**
            - **只在 markdown 的代码块 (```python ... ```) 中输出修复后的、完整的 Python 测试代码。**
            - 不要包含任何额外的解释或评论。
            - 确保新的测试用例能够覆盖改进建议中提到的所有盲点。
            """
        else:
            # 如果没有反馈，说明是第一次生成，使用原始的、最详细的Prompt
            prompt = state["generation_prompt"]

        
        # 使用 get_openai_callback 上下文管理器来追踪 Token
        with get_openai_callback() as cb:
            response = self.llm.invoke(prompt)
            print(f"  -> LLM Call Tokens: {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")

        # 查找被 ```python 和 ``` 包围的代码块
        try:
            # 通常 LLM 的返回格式是 ```python\n...code...\n```
            test_code = response.content.split("```python")[1].split("```")[0].strip()
        except IndexError:
            # 如果格式不是 ```python，尝试通用的 ```
            try:
                test_code = response.content.split("```")[1].split("```")[0].strip()
            except IndexError:
                # 如果连 ``` 都没有，就认为整个返回都是代码（作为最后的备用方案）
                print("  -> WARNING: Could not find markdown fences. Assuming entire response is code.")
                test_code = response.content.strip()
        # ==================== 修改结束 ====================
        
        print("  -> Extracted Test Code:\n", test_code) # 增加一行打印，方便调试
        return {
            "test_code": test_code,
            "total_prompt_tokens": state["total_prompt_tokens"] + cb.prompt_tokens,
            "total_completion_tokens": state["total_completion_tokens"] + cb.completion_tokens,
            "total_tokens": state["total_tokens"] + cb.total_tokens,
        }

    def test_executor_node(self, state: WorkflowState) -> dict:
        """
        节点5: 测试执行与评估
        使用 CodeExecutor 真实地运行测试并获取覆盖率等指标。
        """
        print("--- Step 5: Executing Tests and Gathering Metrics ---")

        report = execute_tests_and_get_report(
            state["code"],
            state["test_code"],
            logic_filename=self.logic_filename,
            test_filename=self.test_filename
        )

        execution_feedback_parts = []
        if "error" in report and "Test execution failed" in report["error"]:
            raise RuntimeError(f"Test execution failed critically: {report['error']}")
        else:
            test_exec = report.get('test_execution', {})
            cov_metrics = report.get('coverage_metrics', {})
            
            coverage = cov_metrics.get('covered_percentage', 0.0) / 100.0
            pass_rate = test_exec.get('pass_rate', 0.0)
            
            execution_feedback_parts.append(f"Coverage: {coverage:.2%}, Pass Rate: {pass_rate:.2%}.")
            if missing_lines := cov_metrics.get('missing_lines'):
                execution_feedback_parts.append(f"Missing lines: {missing_lines}.")
            
            # 将详细的失败日志添加到反馈中
            if failed_cases := test_exec.get('failed_cases_details'):
                failed_cases_str = "\n".join(failed_cases)
                execution_feedback_parts.append(f"\n--- FAILED CASE DETAILS ---\n{failed_cases_str}\n---------------------------")

        feedback = " ".join(execution_feedback_parts)
        print(f"  -> Execution Result: {feedback}")
        
        return {
            "coverage": coverage,
            "pass_rate": pass_rate,
            "execution_feedback": feedback
        }

    def result_evaluator_node(self, state: WorkflowState) -> dict:
        """
        节点6: 结果评估 
        优先使用客观指标进行判断，只有在不达标时才让 LLM 生成反馈。
        """
        print("--- Step 6: Evaluating Results ---")
        current_retries = state.get('retry_count', 0)
        
        # 核心修改：将客观判断放在首位
        if state["pass_rate"] >= 1.0 and state["coverage"] >= self.coverage_threshold:
            print("  -> Objective metrics met. Test suite accepted.")
            grade = "pass"
            feedback = "The test suite meets all quality standards."
        else:
            print("  -> Objective metrics not met. Using raw execution feedback for the next run.")
            grade = "not pass"
            # 直接将执行器节点的原始输出作为反馈
            feedback = state['execution_feedback']

        # --- 打印迭代总结 ---
        print("\n" + "="*20 + f" Iteration #{current_retries + 1} Summary " + "="*20)
        if grade == "not pass":
            print(f"  - Feedback for Next Round:\n{feedback}")
        print("="*61)

        history = state.get("iteration_history", [])
        history.append({
            "iteration": current_retries + 1,
            "test_code": state["test_code"],
            "pass_rate": state["pass_rate"],
            "coverage": state["coverage"],
            "feedback": feedback
        })

        print(f"  -> Evaluation Grade: {grade}")

        return_data = {
            "evaluation_result": grade,
            "evaluation_feedback": feedback,
            "retry_count": current_retries + 1,
            "iteration_history": history
        }

        return return_data
    
        # --- 节点 7: 变异测试检测 ---
    def mutation_test_node(self, state: WorkflowState) -> dict:
        """
        执行变异测试，并能区分工具失败和得分低两种情况。
        """
        print("--- Step 7: Deep Quality Check (Mutation Testing) ---")
        result = run_mutation_test(
            source_code=state["code"],
            test_code=state["test_code"],
            logic_filename=self.logic_filename,
            test_filename=self.test_filename
        )
        
        if details := result.get("survived_details"):
            print("\n" + "-"*20 + " Mutation Test Details " + "-"*20)
            # 打印 mutpy 的核心输出部分
            if "[*] Start mutants generation and execution:" in details:
                print(details.split("[*] Start mutants generation and execution:")[1].strip())
            else:
                print(details) # 如果格式有变，则打印全部
            print("-" * 67 + "\n")

        if "error" in result and result["error"]:
            print(f"  -> CRITICAL ERROR: Mutation testing tool failed: {result['error']}")
            # 如果变异测试工具本身失败，设置错误标记并记录详情
            return {
                "mutation_test_has_error": True,
                "mutation_error_details": result['error'],
                "mutation_details": result.get("details", "No details available.")
            }
        
        score = result.get("mutation_score", 0.0)
        if score == 0.0: 
            print("  -> WARNING: Mutation testing tool returned a score of 0, indicating a possible failure.")
            return {
                "mutation_test_has_error": True,
                "mutation_error_details": "Mutation testing tool returned a score of 0, indicating a possible failure."
            }
        if score < self.mutation_threshold:
            print(f"  -> QA Failed. Mutation score ({score:.2%}) is too low.")
            feedback_intro = (
                f"**关于测试强度的反馈 (来自变异测试):\n** "
                f"测试用例的健壮性不足。它的变异测试得分仅为 {score:.2%}，低于 {self.mutation_threshold:.2%} 的标准。\n"
                f"**具体弱点:** 您的测试未能发现以下 {result.get('survived_count', 0)} 个潜在的bug：\n\n"
            )
            
            survived_details = result.get("survived_details", [])
            feedback_details = ""
            for i, mutant in enumerate(survived_details, 1):
                feedback_details += (
                    f"  **{i}. 在第 {mutant['original_line_no']} 行:**\n"
                    f"     - 原始代码是: `{mutant['original_code']}`\n"
                    f"     - 当它被改成: `{mutant['mutated_code']}` 时, 您的测试依然通过了。\n"
                    f"     - **改进建议:** 请补充一个能区分这两种情况的测试用例。\n\n"
                )

            qa_feedback = feedback_intro + feedback_details
            return {
                "mutation_score": score,
                "mutation_test_has_error": False,
                "evaluation_feedback": qa_feedback,  
                "evaluation_result": "not pass",   
                "mutation_details": result.get("details", "No details available.")
            }

        # 如果工具正常运行，则正常返回分数
        return {
            "mutation_score": result.get("mutation_score", 0.0),
            "mutation_test_has_error": False,
            "mutation_details": result.get("details", "No details available.")
        }

    # ------------------------- 决策逻辑 -------------------------

    def route_after_initial_evaluation(self, state: WorkflowState) -> Literal["mutation_test", "retry", "end"]:
        print("--- Step 6a: Initial Decision ---")
        if state["evaluation_result"] == "pass":
            print("  -> Decision: Initial tests passed. Proceeding to Mutation Test.")
            return "mutation_test"
        if state["retry_count"] >= self.max_retries:
            return "end"
        return "retry"

    def route_after_mutation_test(self, state: WorkflowState) -> Literal["end", "retry", "critical_error"]:
        """
        在变异测试之后做决策，逻辑变得更简单。
        """
        print("--- Step 7a: Final Decision ---")
        if state.get("mutation_test_has_error"):
            return "critical_error"
        
        # 只需检查分数是否达标即可，状态更新已在上一节点完成
        print(f"  -> Mutation Test Score: {state.get('mutation_score', 0.0):.2%}")
        if state.get("mutation_score", 0.0) >= self.mutation_threshold:
            print("  -> Decision: Mutation test passed. Ending workflow.")
            return "end"
        else:
            # 检查是否达到最大重试次数
            if state["retry_count"] >= self.max_retries:
                return "end"
            print("  -> Decision: Mutation test failed. Retrying test generation.")
            return "retry"

    # ------------------------- 构建计算图 -------------------------

    def build(self):
        """
        将所有节点和逻辑边组装成一个可执行的 LangGraph 计算图。
        """
        graph = StateGraph(WorkflowState)

        # 注册所有节点
        graph.add_node("code_analyzer", self.code_analyzer_node)
        graph.add_node("requirement_analyzer", self.requirement_analyzer_node)
        graph.add_node("prompt_organizer", self.prompt_organizer_node)
        graph.add_node("test_generator", self.test_generator_node)
        graph.add_node("test_executor", self.test_executor_node)
        graph.add_node("result_evaluator", self.result_evaluator_node)
        graph.add_node("mutation_test_node", self.mutation_test_node)

        # 定义工作流的执行路径
        graph.set_entry_point("code_analyzer")
        graph.add_edge("code_analyzer", "requirement_analyzer") 
        graph.add_edge("requirement_analyzer", "prompt_organizer")
        graph.add_edge("prompt_organizer", "test_generator")
        graph.add_edge("test_generator", "test_executor")
        graph.add_edge("test_executor", "result_evaluator")

        # 添加条件分支
        graph.add_conditional_edges(
            "result_evaluator",
            self.route_after_initial_evaluation,
            {"mutation_test": "mutation_test_node", "retry": "test_generator", "end": END},
        )
        
        # **核心修改**：为变异测试之后的决策添加新的路由
        graph.add_conditional_edges(
            "mutation_test_node",
            self.route_after_mutation_test,
            {
                "retry": "test_generator", 
                "end": END,
                "critical_error": END
            }
        )

        return graph.compile()
    
# ========================= 成本报告生成类 =========================
class WorkflowReporter:
    """
    一个专门用于生成工作流结果报告的类。
    """
    def __init__(self, final_state: WorkflowState, config: Dict[str, Any]):
        """
        初始化报告器。

        Args:
            final_state: 工作流执行完毕后的最终状态。
            config: 本次运行所使用的配置。
        """
        self.state = final_state
        self.config = config

    def _calculate_metrics(self):
        """内部方法，计算并更新最终指标。"""
        # 计算总执行时间
        self.state["total_execution_time"] = time.time() - self.state["start_time"]
        
    def generate_report(self):
        """
        生成并打印完整的成本与效率报告。
        """
        self._calculate_metrics()
        
        final_result = self.state.get('evaluation_result', 'N/A').upper()
        
        print("\n" + "="*50 + "\n            WORKFLOW COMPLETED - FINAL REPORT\n" + "="*50)

        if self.state.get("mutation_test_has_error"):
            print("\n[!!!] WORKFLOW HALTED DUE TO A CRITICAL ERROR.")
            print("  -> The mutation testing tool failed to execute.")
            print(f"  -> Error Details: {self.state.get('mutation_error_details')}")
        
        if self.state.get('retry_count', 0) >= self.config["max_retries"] and final_result == 'NOT PASS':
            print(f"\n[WARNING] Workflow stopped due to reaching the maximum retry limit ({self.config['max_retries']}).")

        print(f"\nFinal Evaluation Result: {final_result}")
        print("\n--- Generated Test Code ---")
        print(self.state.get('test_code', 'No code generated.'))
        
        print("\n" + "-"*20 + " Cost & Efficiency Metrics " + "-"*20)
        print(f"  - Total Execution Time: {self.state['total_execution_time']:.2f} seconds")
        print(f"  - Total Iterations: {self.state.get('retry_count', 0)}")
        print(f"  - Token Consumption:")
        print(f"    - Total Prompt Tokens:    {self.state.get('total_prompt_tokens', 0)}")
        print(f"    - Total Completion Tokens:  {self.state.get('total_completion_tokens', 0)}")
        print(f"    - Grand Total Tokens:     {self.state.get('total_tokens', 0)}")
        
        print("\n" + "-"*20 + " Quality Metrics " + "-"*26)
        print(f"  - Final Coverage: {self.state.get('coverage', 0.0):.2%}")
        print(f"  - Final Pass Rate: {self.state.get('pass_rate', 0.0):.2%}")
        print("="*67)

# ------------------------- 示例运行 -------------------------

if __name__ == "__main__":
    # 1. 加载配置
    app_config = load_config()

    # 2. 定义待测试的业务逻辑代码
    sample_logic_code = """
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
"""

    # 3. 用户的测试需求
    user_requirement = "Write a comprehensive test suite for the 'calculate' function. Ensure all operations ('add', 'subtract', 'multiply', 'divide') are tested. Also, specifically test the edge case of division by zero and invalid input types."

    # 4. 初始化并构建工作流
    workflow_builder = TestGenerationWorkflow(config=app_config)
    app = workflow_builder.build()
    

    # 5. 执行工作流
    print("\n" + "="*50)
    print("           STARTING TEST GENERATION WORKFLOW           ")
    print("="*50 + "\n")
    
    initial_state = {
        "code": sample_logic_code, "requirement": user_requirement, "retry_count": 0,
        "start_time": time.time(), "total_prompt_tokens": 0,
        "total_completion_tokens": 0, "total_tokens": 0,
        "iteration_history": [],
        "mutation_test_has_error": False, 
        "mutation_error_details": ""      
    }
    final_state = app.invoke(initial_state)

    # 6. 打印最终结果
    reporter = WorkflowReporter(final_state, app_config)
    reporter.generate_report()
