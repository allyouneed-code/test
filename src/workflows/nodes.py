# src/workflows/nodes.py

import json
import os
from langchain_community.callbacks import get_openai_callback

# 导入工具和 schemas
from .schemas import WorkflowState
from ..tools.code_inspector import CodeAnalyzer
from ..tools.requirement_analyzer import RequirementAnalyzer
from ..tools.code_executer import execute_tests_and_get_report
from ..tools.mutation_tester import run_mutation_test
from ..llm.client import get_llm_client


def code_analyzer_node(state: WorkflowState) -> dict:
    """节点1: 代码分析"""
    print("--- 步骤1：代码分析 ---")
    analyzer = CodeAnalyzer(state["code"])
    final_model = analyzer.process()
    report_str = json.dumps(final_model, indent=2, ensure_ascii=False)
    return {"analysis_report": report_str}

def requirement_analyzer_node(state: WorkflowState) -> dict:
    """节点2: 需求分析"""
    print("--- 步骤2：需求分析 ---")
    try:
        with get_openai_callback() as cb:
            analyzer = RequirementAnalyzer()
            analysis_result = analyzer.analyze(state["requirement"], state["code"])
            report_str = json.dumps(analysis_result, indent=2, ensure_ascii=False)           
            print(f"  -> LLM Call Tokens (Requirement Analysis): {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")
        return {
            "structured_requirement": report_str,
            "total_prompt_tokens": state["total_prompt_tokens"] + cb.prompt_tokens,
            "total_completion_tokens": state["total_completion_tokens"] + cb.completion_tokens,
            "total_tokens": state["total_tokens"] + cb.total_tokens,
        }
    except Exception as e:
        error_msg = f"Requirement analysis failed: {e}"
        print(f"ERROR: {error_msg}")
        return {"structured_requirement": json.dumps({"error": error_msg})}


def prompt_organizer_node(state: WorkflowState, logic_filename: str) -> dict:
    """
    节点3: Prompt 组织
    整合用户需求和代码分析报告，构建一个高质量的 Prompt。
    """
    print("--- 步骤3：prompt组织 ---")
    module_name = os.path.splitext(logic_filename)[0]
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

def test_generator_node(state: WorkflowState) -> dict:
    """
    节点4: 测试用例生成
    调用 LLM，根据 Prompt 生成测试代码。
    如果存在上一轮的反馈，会一并考虑。
    """
    print("--- 步骤4：生成用例 ---")
    llm = get_llm_client(temperature=0.2)
    # 检查是否存在反馈。如果存在，说明是迭代优化阶段。
    if feedback := state.get("evaluation_feedback"):
        print("  -> 检测到反馈，进行测试用例修复和改进...")
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
        response = llm.invoke(prompt)
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


def test_executor_node(state: WorkflowState, logic_filename: str, test_filename: str) -> dict:
    """
    节点5: 测试执行与评估
    使用 CodeExecutor 真实地运行测试并获取覆盖率等指标。
    """
    print("--- 步骤5：执行测试 ---")

    report = execute_tests_and_get_report(
        state["code"],
        state["test_code"],
        logic_filename=logic_filename,
        test_filename=test_filename
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

def result_evaluator_node(state: WorkflowState, coverage_threshold: float) -> dict:
    """
    节点6: 结果评估 
    优先使用客观指标进行判断，只有在不达标时才让 LLM 生成反馈。
    """
    print("--- 步骤6：评估结果 ---")
    current_retries = state.get('retry_count', 0)
    
    if state["pass_rate"] >= 1.0 and state["coverage"] >= coverage_threshold:
        print("  -> Objective metrics met. Test suite accepted.")
        grade = "pass"
        feedback = "The test suite meets all quality standards."
    else:
        print("  -> Objective metrics not met. Using raw execution feedback for the next run.")
        grade = "not pass"
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

def mutation_test_node(state: WorkflowState, app_config: dict) -> dict:
    """
    执行变异测试，并能区分工具失败和得分低两种情况。
    """
    print("--- 步骤7：变异测试 ---")
    result = run_mutation_test(
        source_code=state["code"], test_code=state["test_code"],
        logic_filename=app_config["logic_filename"], test_filename=app_config["test_filename"],
    )
    mutation_threshold = app_config.get("mutation_threshold", 0.9)
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
    if score < mutation_threshold:
        print(f"  -> QA Failed. Mutation score ({score:.2%}) is too low.")
        feedback_intro = (
            f"**关于测试强度的反馈 (来自变异测试):\n** "
            f"测试用例的健壮性不足。它的变异测试得分仅为 {score:.2%}，低于 {mutation_threshold:.2%} 的标准。\n"
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