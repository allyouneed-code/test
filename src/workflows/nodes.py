# src/workflows/nodes.py

import json
import os
from langchain_community.callbacks import get_openai_callback

# 导入工具和 schemas
from .schemas import WorkflowState
from ..tools.code_inspector import CodeAnalyzer
from ..tools.requirement_analyzer import RequirementAnalyzer
from ..tools.requirement_analyzer import TestModelAssembler, FullTestModel
from ..tools.validator import StaticDifferentialValidator, BatchValidationReport
from ..tools.code_executer import execute_tests_and_get_report
from ..tools.mutation_tester import run_mutation_test
from ..llm.client import get_llm_client


def _compress_and_clean_report(report_json_str: str) -> str:
    """
    清洗并压缩代码分析报告：
    1. 剔除 CFG 节点中冗余的源代码副本 ('statements' 字段)。
    2. 移除 JSON 中的空格和换行。
    """
    if not report_json_str:
        return "{}"
    
    try:
        data = json.loads(report_json_str)
        
        # 1. 针对性清洗：移除 Code Inspector 报告中 G (CFG) 节点的 statements
        # 原因：这些是源码的拷贝，LLM 对照源码即可，无需在 Prompt 中重复
        if isinstance(data, dict) and "G" in data and "Nodes" in data["G"]:
            for node in data["G"]["Nodes"]:
                if "statements" in node:
                    del node["statements"]
        
        # 2. 格式压缩：去除空格和换行 (separators)
        return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        
    except Exception as e:
        print(f"Warning: Report compression failed: {e}")
        return report_json_str

def _compress_json(json_str: str) -> str:
    """通用 JSON 压缩，仅去除空格"""
    try:
        data = json.loads(json_str)
        return json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    except:
        return json_str

def code_analyzer_node(state: WorkflowState) -> dict:
    """代码分析"""
    print("--- 步骤1：代码分析 ---")
    analyzer = CodeAnalyzer(state["code"])
    final_model = analyzer.process()
    report_str = json.dumps(final_model, indent=2, ensure_ascii=False)
    return {"analysis_report": report_str,
             "analysis_model": final_model
           }

def requirement_analyzer_node(state: WorkflowState) -> dict:
    """需求分析"""
    print("--- 步骤2：需求分析 ---")
    try:
        with get_openai_callback() as cb:
            assembler = TestModelAssembler()
            analysis_result: FullTestModel = assembler.build(state["requirement"], state["code"])
            analysis_result_dict = analysis_result.model_dump()
            report_str = json.dumps(analysis_result_dict, indent=2, ensure_ascii=False)                    
            
        return {
            "structured_requirement": report_str,
            "requirement_model": analysis_result,
            "total_prompt_tokens": state["total_prompt_tokens"] + cb.prompt_tokens,
            "total_completion_tokens": state["total_completion_tokens"] + cb.completion_tokens,
            "total_tokens": state["total_tokens"] + cb.total_tokens,
        }
    except Exception as e:
        error_msg = f"Requirement analysis failed: {e}"
        print(f"ERROR: {error_msg}")
        return {"structured_requirement": json.dumps({"error": error_msg}),
                "requirement_model": None}

def validator_node(state: WorkflowState) -> dict:
    """
    节点: 静态差分验证
    对比 M_req (需求模型) 和 M_code (代码模型)。
    """
    print("--- 静态差分验证 ---")
    
    m_req: FullTestModel = state.get("requirement_model")
    m_code: dict = state.get("analysis_model")

    if not m_req or not m_code:
        error_msg = "Skipping validation: Requirement model or Code model is missing."
        print(f"  -> WARNING: {error_msg}")
        return {"validation_report": json.dumps({"error": error_msg})}

    try:
        llm = get_llm_client(temperature=0.0)
        
        validator = StaticDifferentialValidator(m_req=m_req, m_code=m_code, llm_client=llm)
        
        report = validator.validate()
        report_str = json.dumps(report, indent=2, ensure_ascii=False)
        
        print("  -> Validation Report Generated.")
        return {"validation_report": report_str}
        
    except Exception as e:
        error_msg = f"Static validation failed: {e}"
        print(f"ERROR: {error_msg}")
        return {"validation_report": json.dumps({"error": error_msg})}


def prompt_organizer_node(state: WorkflowState, logic_filename: str) -> dict:
    """
    节点3: Prompt 组织
    整合用户需求和代码分析报告，构建一个高质量的 Prompt。
    """
    print("--- 步骤3：prompt组织 (Optimized) ---")
    module_name = os.path.splitext(logic_filename)[0]
    
    # 1. 数据准备与压缩
    # 清洗分析报告 (Token 降幅最大)
    clean_analysis_report = _compress_and_clean_report(state.get('analysis_report', '{}'))
    # 压缩需求结构 (去除格式空格)
    clean_struct_req = _compress_json(state.get('structured_requirement', '{}'))
    req_model = state.get("requirement_model") # 这是一个 FullTestModel 对象 (Pydantic)
    req_ids = []
    if req_model:
        # 提取功能场景 ID
        if req_model.behavioral_model and req_model.behavioral_model.functional_scenarios:
            for s in req_model.behavioral_model.functional_scenarios:
                req_ids.append(f"- {s.id}: {s.description}")
        # 提取异常场景 ID
        if req_model.behavioral_model and req_model.behavioral_model.error_scenarios:
            for s in req_model.behavioral_model.error_scenarios:
                req_ids.append(f"- {s.id}: {s.description}")
    req_id_list_str = "\n".join(req_ids)
    prompt_template = f"""
    **目标:** 生成一个全面的 pytest 测试套件。

    **源代码模块名:** `{module_name}`

    **用户原始需求:**
    {state['requirement']}

    **需求覆盖项列表 (Traceability IDs):**
    请确保你生成的测试用例能够覆盖以下所有需求项。
    {req_id_list_str}

    **待测试源代码:**
    ```python
    {state['code']}
    ```

    **静态代码分析报告:**
    这份报告提供了代码结构的信息。用它来进一步完善您的测试用例。
    ```json
    {clean_analysis_report}
    ```

    **任务:**
    基于以上所有信息，，编写一个 pytest 测试套件。
    - **至关重要，您必须从 `{module_name}` 模块导入待测试的函数。** 例如: `from {module_name} import your_function_name`。
    - 测试代码必须是完整且可运行的。
    - 确保您生成的测试覆盖了结构化分析中概述的所有场景。
    - 除非源代码中有明确的raise语句，否则不要编写预期值错误、类型错误或任何异常的测试。如果代码通过返回None、0.0或隐式处理无效输入（如Python的默认TypeError）来处理无效输入，则测试必须断言该特定行为，而不是你发明的异常。
    - **使用装饰器标记需求:** 对于每一个测试函数，你必须使用 `@pytest.mark.requirement("ID")` 装饰器来指明它覆盖了哪个需求 ID。
    - **只在 markdown 的代码块 (```python ... ```) 中输出测试套件的 Python 代码。** 不要包含任何其他文本或解释。
    """
    return {"generation_prompt": prompt_template}

def test_creator_node(state: WorkflowState) -> dict:
    """
    节点4.1: 测试用例创造者 (LLM 1)
    从0到1，根据完整的Prompt“创造”测试代码的第一个版本。
    (此节点不再处理反馈)
    """
    print("--- 步骤4.1：生成测试用例草稿 (Creator) ---")
    llm = get_llm_client(temperature=0.2)
    prompt = state["generation_prompt"]

    with get_openai_callback() as cb:
        response = llm.invoke(prompt)
        print(f"  -> LLM Call Tokens (Creator): {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")

    try:
        test_code = response.content.split("```python")[1].split("```")[0].strip()
    except IndexError:
        print("  -> WARNING: Could not find markdown fences. Assuming entire response is code.")
        test_code = response.content.strip()
    
    print("  -> Extracted Test Code (Draft):\n", test_code)
    
    return {
        "test_code": test_code, # 输出到 test_code 供 Reviewer 审查
        "total_prompt_tokens": state["total_prompt_tokens"] + cb.prompt_tokens,
        "total_completion_tokens": state["total_completion_tokens"] + cb.completion_tokens,
        "total_tokens": state["total_tokens"] + cb.total_tokens,
    }

def test_reviewer_node(state: WorkflowState) -> dict:
    """
    节点: 测试用例审查者 (LLM 2)
    在执行前，对测试代码草稿进行“逻辑审查”，防止明显的 F-Case。
    """
    print("--- 审查测试用例逻辑 (Reviewer) ---")
    llm = get_llm_client(temperature=0.0) 
    
    draft_test_code = state["test_code"]
    
    prompt = f"""
    **任务: 审查测试用例的逻辑正确性**

    你是一名资深的QA专家。请审查以下测试代码草稿，确保它的断言（asserts）在逻辑上是正确的，并且与源代码的行为一致。

    **待测试源代码:**
    ```python
    {state['code']}
    ```

    **用户原始需求:**
    {state['requirement']}

    **测试代码草稿:**
    ```python
    {draft_test_code}
    ```

    **审查指南:**
    1.  **检查断言逻辑:** * `？
        * 断言中的error逻辑是否和代码逻辑相符`？
        * 代码没实现这个功能
        * 用例的输入和预期的输出是否正确对应？
        * `assert` 的预期值是否符合代码逻辑？
    2.  **检查导入:** 导入语句是否正确？
    3.  **不要质疑需求**：假设源代码和需求是正确的，只审查测试代码是否*同时*符合这两者。

    **输出要求:**
    - **如果测试代码草稿在逻辑上是完美的**：原封不动地返回该代码。
    - **如果测试代码草稿有逻辑错误**：返回修复了逻辑错误后的*完整*代码。
    - **只在 markdown 的代码块 (```python ... ```) 中输出最终的 Python 测试代码。** 不要包含任何解释。
    """

    with get_openai_callback() as cb:
        response = llm.invoke(prompt)
        print(f"  -> LLM Call Tokens (Reviewer): {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")

    try:
        validated_test_code = response.content.split("```python")[1].split("```")[0].strip()
    except IndexError:
        print("  -> WARNING: Reviewer did not use markdown. Using full response.")
        validated_test_code = response.content.strip()
    
    print("  -> Validated Test Code:\n", validated_test_code)
    
    return {
        "test_code": validated_test_code, # 覆盖原始草稿，准备执行
        "total_prompt_tokens": state["total_prompt_tokens"] + cb.prompt_tokens,
        "total_completion_tokens": state["total_completion_tokens"] + cb.completion_tokens,
        "total_tokens": state["total_tokens"] + cb.total_tokens,
    }

def test_refiner_node(state: WorkflowState) -> dict:
    """
    节点4.3: 测试用例修复者 (LLM 3)
    在“质量循环”中被调用，
    只修复技术错误（E-Cases）或覆盖率不足（Coverage）。
    """
    print("--- 步骤4.3：修复技术/覆盖率问题 (Refiner) ---")
    llm = get_llm_client(temperature=0.1) # 使用低温 "修复者"
    
    feedback = state["evaluation_feedback"] # 这个反馈只包含 E-Case 或 Coverage
    req_model = state.get("requirement_model")
    req_ids = []
    if req_model:
        if req_model.behavioral_model and req_model.behavioral_model.functional_scenarios:
            for s in req_model.behavioral_model.functional_scenarios:
                req_ids.append(f"- {s.id}: {s.description}")
        if req_model.behavioral_model and req_model.behavioral_model.error_scenarios:
            for s in req_model.behavioral_model.error_scenarios:
                req_ids.append(f"- {s.id}: {s.description}")
    req_id_list_str = "\n".join(req_ids)
    prompt = f"""
    **任务: 修复测试套件**

    你是一名专业的Python测试工程师。上一轮的测试代码在执行时遇到了技术错误或覆盖率不足。
    请根据“执行反馈”，修复“上一轮的测试代码”。

    **上一轮的测试代码:**
    ```python
    {state['test_code']}
    ```

    **执行反馈 (需要修复的问题):**
    ---
    {feedback}
    ---
    
    **待测试源代码 (供参考):**
    ```python
    {state['code']}
    ```

    **需求覆盖项列表 (供参考):**
    {req_id_list_str}

    **修复指南:**
    - 如果反馈是程序运行报错`Error`，请修复 Python 语法。
    - 如果反馈是 `Missing lines: ...`，请添加*新的*测试用例来覆盖这些缺失的行。
    - 如果反馈包含“**关于测试强度的反馈 (来自变异测试)**”，请仔细阅读“具体弱点”，并补充*新的*、*更强*的测试用例来杀死（kill）那些存活的变异体。
    - **不要**修改那些*已经通过*的测试用例的 `assert` 逻辑（除非变异测试反馈明确指出了一个逻辑弱点）。
    - **不要**修改那些*已经通过*的测试用例的 `assert` 逻辑。
    - **[重要] 维护追溯性:**
       - **绝对不要删除** 现有的 `@pytest.mark.requirement(...)` 装饰器，除非你删除了对应的测试函数。
       - **为新用例添加标记:** 如果你添加了*新*的测试函数（例如为了覆盖率），**必须**从上面的“需求覆盖项列表”中选择一个最相关的 ID，并添加 `@pytest.mark.requirement("ID")` 装饰器。
       - 如果新用例是为了覆盖具体的代码实现细节且难以对应到具体需求，可以使用 `@pytest.mark.requirement("impl_detail")`，但尽量对应到原始需求 ID。

    **输出要求:**
    - **只在 markdown 的代码块 (```python ... ```) 中输出修复后的、完整的 Python 测试代码。**
    - 不要包含任何额外的解释或评论。
    """
    
    with get_openai_callback() as cb:
        response = llm.invoke(prompt)
        print(f"  -> LLM Call Tokens (Refiner): {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")

    try:
        refined_test_code = response.content.split("```python")[1].split("```")[0].strip()
    except IndexError:
        print("  -> WARNING: Refiner did not use markdown. Using full response.")
        refined_test_code = response.content.strip()
    
    print("  -> Refined Test Code:\n", refined_test_code)
    
    return {
        "test_code": refined_test_code, # 覆盖上一轮代码，准备重新执行
        "retry_count": state.get('retry_count', 0) + 1, # 增加重试计数
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
        test_failures = test_exec.get('failed', 0)
        test_errors = test_exec.get('error', 0)
        
        execution_feedback_parts.append(f"Coverage: {coverage:.2%}, Pass Rate: {pass_rate:.2%}.")
        execution_feedback_parts.append(f"Failures: {test_failures}, Errors: {test_errors}.")
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
        "test_failures": test_failures,
        "test_errors": test_errors,
        "execution_feedback": feedback
    }

def result_evaluator_node(state: WorkflowState, coverage_threshold: float) -> dict:
    """
    节点6: 结果评估 
    逻辑：
    1. F-Case > 0 -> FAIL_F_CASE (硬失败，不重试)
    2. E-Case > 0 -> RETRY (尝试自动修复语法/环境错误)
    3. Coverage < Threshold -> RETRY (尝试补充测试用例)
    4. All Good -> QUALITY_MET (进入变异测试)
    """
    print("--- 步骤6：评估结果 ---")
    
    # 1. 优先检查 F-Case (测试逻辑失败)
    # 如果断言失败，通常意味着测试代码对业务逻辑理解有误(幻觉)，或者是业务代码本身有Bug。
    # 此时不应盲目重试，而应停止并报告。
    test_failures = state.get("test_failures", 0)
    if test_failures > 0:
        print(f"  -> 质量评估: 发现 {test_failures} 个测试失败 (F-Cases)。判定为质量不达标，停止流程。")
        return {
            "evaluation_result": "FAIL_F_CASE",
            "evaluation_feedback": f"测试套件执行失败，存在 {test_failures} 个断言错误 (F-Cases)。请人工检查业务逻辑与测试预期是否一致。"
        }

    # 2. 检查 E-Case (执行/语法错误)
    # 这类错误通常是 import 错误、语法错误等，LLM 很有机会自动修复。
    test_errors = state.get("test_errors", 0)
    if test_errors > 0:
        print(f"  -> 质量评估: 发现 {test_errors} 个执行错误 (E-Cases)。触发重试修复。")
        feedback_for_refiner = "Please fix the following execution errors (Syntax/Import/Runtime):\n" + state.get('execution_feedback', '')
        return {
            "evaluation_result": "RETRY_QUALITY",
            "evaluation_feedback": feedback_for_refiner
        }
    
    # 3. 检查覆盖率
    coverage = state.get("coverage", 0.0)
    if coverage < coverage_threshold:
        print(f"  -> 质量评估: 覆盖率 ({coverage:.2%}) 未达标 (目标 {coverage_threshold:.2%})。触发重试补充。")
        feedback_for_refiner = f"Coverage ({coverage:.2%}) is below threshold ({coverage_threshold:.2%}). Please add tests for:\n" + state.get('execution_feedback', '')
        return {
            "evaluation_result": "RETRY_QUALITY",
            "evaluation_feedback": feedback_for_refiner
        }
    
    # 4. 质量达标 (0 F-Case, 0 E-Case, Coverage OK)
    # 只有到了这一步，才有资格进入变异测试
    print(f"  -> 质量评估: 基础质量达标 (0 Errors, 0 Failures, Coverage {coverage:.2%})。准备进行变异测试。")
    return {
        "evaluation_result": "QUALITY_MET",
        "evaluation_feedback": "" # 无需反馈
    }

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
        
        survived_details = result.get("survived_details", [])
        
        # --- Token 熔断保护：Top-N 截断策略 ---
        MAX_REPORT_LIMIT = 5  
        
        truncated_details = survived_details[:MAX_REPORT_LIMIT]
        remaining_count = len(survived_details) - MAX_REPORT_LIMIT
        
        feedback_intro = (
            f"**关于测试强度的反馈 (来自变异测试):**\n"
            f"测试用例的健壮性不足。变异得分为 {score:.2%} (低于 {mutation_threshold:.2%})。\n"
            f"共有 {len(survived_details)} 个变异体存活，以下是 **Top {len(truncated_details)}** 个典型案例：\n\n"
        )
        
        feedback_details = ""
        for i, mutant in enumerate(truncated_details, 1):
            # 限制单个变异体描述的长度（防止某个变异体代码过长）
            orig_code = mutant['original_code']
            mut_code = mutant['mutated_code']
            if len(orig_code) > 200: orig_code = orig_code[:200] + "..."
            if len(mut_code) > 200: mut_code = mut_code[:200] + "..."

            feedback_details += (
                f"  **案例 {i}. (第 {mutant['original_line_no']} 行):**\n"
                f"     - 原代码: `{orig_code}`\n"
                f"     - 变异为: `{mut_code}`\n"
                f"     - **问题:** 您的测试未报错（未杀死此变异）。请添加用例区分二者。\n\n"
            )
            
        # 如果有截断，添加提示
        if remaining_count > 0:
            feedback_details += (
                f"**... (还有 {remaining_count} 个变异体存活。为防止 Token 溢出已省略。**\n"
                f"**策略:** 请先专注修复上述 {len(truncated_details)} 个问题。修复后，下一轮迭代将显示剩余的问题。)\n"
            )

        qa_feedback = feedback_intro + feedback_details
        
        return {
            "mutation_score": score,
            "mutation_test_has_error": False,
            "evaluation_feedback": qa_feedback,  
            "evaluation_result": "RETRY_MUTATION",   
            "mutation_details": result.get("details", "No details available.")
        }

    # 如果工具正常运行，则正常返回分数
    return {
        "mutation_score": score,
        "mutation_test_has_error": False,
        "mutation_details": result.get("details", "No details available.")
    }