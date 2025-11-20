# src/workflows/nodes.py

import json
import os
from langchain_community.callbacks import get_openai_callback

# 导入工具和 schemas
from .schemas import WorkflowState,EvaluationResult
from ..tools.code_inspector import CodeAnalyzer
from ..tools.requirement_analyzer import RequirementAnalyzer
from ..tools.requirement_analyzer import TestModelAssembler, FullTestModel
from ..tools.validator import StaticDifferentialValidator, BatchValidationReport
from ..tools.code_executer import execute_tests_and_get_report
from ..tools.mutation_tester import run_mutation_test
from ..llm.client import get_llm_client

def _extract_code_from_llm(response_content: str) -> str:
    """从LLM的Markdown响应中提取Python代码。"""
    try:
        # 优先查找 ```python
        code = response_content.split("```python")[1].split("```")[0].strip()
    except IndexError:
        try:
            # 其次查找 ```
            code = response_content.split("```")[1].split("```")[0].strip()
        except IndexError:
            # 如果找不到markdown，则假定整个响应都是代码 (有风险)
            print("  -> WARNING: Could not find markdown fences. Using full response.")
            code = response_content.strip()
    return code

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

    **任务与核心约束 (必须严格遵守):**
    
    1.  **编写测试:** 基于上述所有信息，编写一个完整的 pytest 测试套件。
    2.  **导入:** 必须从 `{module_name}` 模块导入待测试的函数。
    3.  **标记装饰器:** 每一个测试函数都必须使用 `@pytest.mark.requirement("ID")` 装饰器，从上面的“需求覆盖项列表”中选择一个ID进行标记。
    4.  **核心约束：数据必须匹配需求ID**
        当你为测试函数分配一个需求ID时，你必须：
        a.  **回查** "需求覆盖项列表" 中该ID的描述 。
        b.  **确保** 你在该函数中提供的**输入数据**  **严格满足**该ID描述中的**所有触发条件** 。
        c.  **确保** 你的 `assert` 语句验证的是在该特定条件下的预期结果。
    5   **核心约束：输出case匹配源代码**
        这是最重要的规则。你的所有 `assert` 语句和 `pytest.raises` 检查，**必须** 100% 匹配 **"待测试源代码"** 的 *实际* 行为。
        **禁止幻觉 (F-Case 关键):** **绝对不要** `pytest.raises(Error)` 任何源代码中没有 *明确* `raise` 的异常
    6.  **输出格式:** **只在 markdown 的代码块 (```python ... ```) 中输出测试套件的 Python 代码。**
    """
    return {"generation_prompt": prompt_template}

def test_creator_node(state: WorkflowState) -> dict:
    """
    节点4.1: 测试用例创造者 (LLM 1)
    从0到1，根据完整的Prompt“创造”测试代码的第一个版本。
    (此节点不再处理反馈)
    """
    print("--- 步骤4.1：生成测试用例草稿 (Creator) ---")
    # llm = get_llm_client(model_name= "openai:gpt-5",temperature=0.1)
    llm = get_llm_client(temperature=0.1)
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
        * **重要** `assert` 的预期值是否符合代码逻辑？是否能正确反映代码的行为？
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
    current_state = state["evaluation_result"]
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
    is_incremental = False  # 标记是否为增量模式
    if current_state == "RETRY_E_CASE":
        # 策略1: 完全重写 (修复 E-Case)
        print("  -> 策略: REFRESH (修复 E-Case 错误)")
        is_incremental = False
        prompt = f"""
        **任务: 修复测试套件的执行错误**
        你是一名专业的Python测试工程师。上一轮的测试代码在执行时遇到了技术错误 (E-Case)。
        请根据“执行反馈”，修复“上一轮的测试代码”。

        **上一轮的测试代码 (包含错误):**
        ```python
        {state['test_code']}
        ```
        **执行反馈 (需要修复的错误):**
        ---
        {feedback}
        ---
        **待测试源代码 (供参考):**
        ```python
        {state['code']}
        ```
        **修复指南:**
        - 仔细阅读错误信息，修复 Python 语法、导入或运行时错误。
        - **不要**修改已有的 `@pytest.mark.requirement(...)` 装饰器。
        
        **输出要求:**
        - **只在 markdown 的代码块 (```python ... ```) 中输出修复后的、完整的 Python 测试代码。**
        - 不要包含任何额外的解释或评论。
        """

    elif current_state == "RETRY_COVERAGE":
        # 策略2: 增量添加 (补充 Coverage)
        print("  -> 策略: INCREMENTAL (补充覆盖率)")
        is_incremental = True
        prompt = f"""
        **任务: 增量补充测试用例以提高覆盖率**
        你是一名专业的Python测试工程师。现有的测试套件 100% 通过，但覆盖率不足。
        
        **执行反馈 (需要覆盖的缺失行):**
        ---
        {feedback}
        ---
        **待测试源代码 (供参考):**
        ```python
        {state['code']}
        ```
        **需求覆盖项列表 (供新用例标记):**
        {req_id_list_str}

        **修复指南:**
        - 你的任务是**只编写新的测试用例**来覆盖上述“缺失行”。
        - **不要**重复生成已有的测试用例。
        - **必须**为所有新添加的测试用例函数添加 `@pytest.mark.requirement("ID")` 装饰器，从上面的“需求列表”中选择最相关的一个。

        **输出要求:**
        - **只在 markdown 的代码块 (```python ... ```) 中输出你新编写的 Python 测试函数。**
        - 不要包含任何解释或已有的代码。
        """

    elif current_state == "RETRY_MUTATION":
        # 策略3: 增量添加 (杀死变异体)
        print("  -> 策略: INCREMENTAL (补充变异测试强度)")
        is_incremental = True
        prompt = f"""
        **任务: 增量补充更强的测试用例以杀死变异体**
        你是一名专业的Python测试工程师。现有的测试套件 100% 通过，但健壮性不足 (变异测试得分低)。
        
        **执行反馈 (需要杀死的存活变异体):**
        ---
        {feedback}
        ---
        **待测试源代码 (供参考):**
        ```python
        {state['code']}
        ```
        **需求覆盖项列表 (供新用例标记):**
        {req_id_list_str}

        **修复指南:**
        - 你的任务是**只编写新的、更强的测试用例**来杀死上述“存活变异体”。
        - 重点关注边界值、特殊值或能区分细微逻辑差异（如 `>` vs `>=`）的断言。
        - **不要**重复生成已有的测试用例。
        - **必须**为所有新添加的测试用例函数添加 `@pytest.mark.requirement("ID")` 装饰器。

        **输出要求:**
        - **只在 markdown 的代码块 (```python ... ```) 中输出你新编写的 Python 测试函数。**
        - 不要包含任何解释或已有的代码。
        """

    else:
        # 兜底情况，不应发生
        print(f"  -> WARNING: Refiner 收到未知状态 '{current_state}'，不执行任何操作。")
        return {}
    
    # --- 执行 LLM 调用 ---
    with get_openai_callback() as cb:
        response = llm.invoke(prompt)
        print(f"  -> LLM Call Tokens (Refiner - {current_state}): {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")

    # 提取代码
    llm_output_code = _extract_code_from_llm(response.content)

    final_test_code = ""
    if is_incremental:
        # 增量模式： 附加新代码
        print("  -> 增量模式：正在附加新生成的用例...")
        old_test_code = state['test_code']
        final_test_code = old_test_code + f"\n\n# --- 迭代补充 ({current_state}) ---\n" + llm_output_code
    else:
        # 重写模式：替换旧代码
        print("  -> 重写模式：正在替换为已修复的代码...")
        final_test_code = llm_output_code
    
    # print("  -> Refined Test Code:\n", final_test_code) # (可选：调试时取消注释)
    
    return {
        "test_code": final_test_code, # 覆盖上一轮代码
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
            "evaluation_result": "RETRY_E_CASE",
            "evaluation_feedback": feedback_for_refiner
        }
    
    # 3. 检查覆盖率
    coverage = state.get("coverage", 0.0)
    if coverage < coverage_threshold:
        print(f"  -> 质量评估: 覆盖率 ({coverage:.2%}) 未达标 (目标 {coverage_threshold:.2%})。触发重试补充。")
        feedback_for_refiner = f"Coverage ({coverage:.2%}) is below threshold ({coverage_threshold:.2%}). Please add tests for:\n" + state.get('execution_feedback', '')
        return {
            "evaluation_result": "RETRY_COVERAGE",
            "evaluation_feedback": feedback_for_refiner
        }
    
    # 4. 质量达标 (0 F-Case, 0 E-Case, Coverage OK)
    # 只有到了这一步，才有资格进入变异测试
    print(f"  -> 质量评估: 基础质量达标 (0 Errors, 0 Failures, Coverage {coverage:.2%})。准备进行变异测试。")
    return {
        "evaluation_result": "QUALITY_MET",
        "evaluation_feedback": "" # 无需反馈
    }

def feedback_summarizer_node(state: WorkflowState) -> dict:
    """
    节点 6.5: 告警反馈总结器 
    
    在 Refiner 之前调用。
    使用 LLM 将原始的、冗长的命令行输出 (E-Case, Coverage, Mutation)
    转换为简洁、可操作的修复指令。
    """
    print("--- 步骤 6.5: 总结失败反馈 (Summarizer) ---")
    
    # evaluation_feedback 此刻包含的是来自 executor 或 mutator 的 *原始* 反馈
    current_state: EvaluationResult = state["evaluation_result"]
    raw_feedback: str = state["evaluation_feedback"]
    
    # 我们使用一个低温 (deterministic) LLM 来做总结
    llm = get_llm_client(temperature=0.0) 
    prompt = ""

    if current_state == "RETRY_E_CASE":
        print("  -> 策略: 总结 E-Case (执行错误)")
        prompt = f"""
        **任务:** 你是一个资深的QA专家。请将以下冗长的 `pytest` 错误日志总结为一条清晰、简洁的指令，告诉开发者需要修复什么。

        **原始错误日志:**
        ---
        {raw_feedback}
        ---

        **总结要求:**
        - 重点关注 `AssertionError` 之外的错误，如 `ImportError`, `SyntaxError`, `NameError`, `TypeError` 等。
        - 明确指出是哪个文件、哪一行、哪个函数出了什么问题。
        - 示例: "请修复 `test_script.py` 中的 `ImportError`：无法导入 `logic_module.py`。"
        - **只输出总结后的指令，不要包含任何其他内容。**
        """
    
    elif current_state == "RETRY_COVERAGE":
        print("  -> 策略: 总结 Coverage (覆盖率不足)")
        prompt = f"""
        **任务:** 你是一个资深的测试分析师。请将以下 `coverage` 报告总结为需要补充的测试场景。

        **原始覆盖率报告 (包含缺失行):**
        ---
        {raw_feedback}
        ---
        **待测试源代码 (供参考):**
        ```python
        {state['code']}
        ```

        **总结要求:**
        - 分析缺失的代码行（Missing lines）。
        - 结合源代码，将这些缺失的行翻译成 1-3 个需要补充的“测试场景”。
        - 示例: "请补充测试用例：\n1. 测试当 `operation` 不是 'add' 或 'subtract' 时的默认返回 `None` 的情况 (覆盖第 10 行)。\n2. 测试 `b=0` 时的除法错误 (覆盖第 12 行)。"
        - **只输出总结后的指令，不要包含任何其他内容。**
        """

    elif current_state == "RETRY_MUTATION":
        print("  -> 策略: 总结 Mutation (变异测试失败)")
        prompt = f"""
        **任务:** 你是一个资深的变异测试专家。请将以下“存活变异体”报告总结为清晰的“测试用例强化指令”。

        **原始变异测试报告 (包含存活变异体):**
        ---
        {raw_feedback}
        ---
        **待测试源代码 (供参考):**
        ```python
        {state['code']}
        ```

        **总结要求:**
        - 针对报告中的 1-3 个*最典型*的存活变异体。
        - 明确说明“为什么”测试用例不够强。
        - 示例: "请强化测试用例：\n1. (第 5 行) 当前测试未能区分 `a + b` 和 `a - b`。请添加一个 `a=5, b=2, operation='add'` 的断言。\n2. (第 12 行) 当前测试未能区分 `b == 0` 和 `b != 0`。请添加一个 `b=0` 时的除法测试。"
        - **只输出总结后的指令，不要包含任何其他内容。**
        """
    else:
        # 状态不匹配 (e.g., QUALITY_MET)，理论上不应路由到此
        print(f"  -> INFO: Summarizer 收到状态 '{current_state}'，无需总结。")
        return {}

    # --- LLM 调用以获取总结 ---
    try:
        with get_openai_callback() as cb:
            response = llm.invoke(prompt)
            summarized_feedback = response.content.strip()
            print(f"  -> LLM Call Tokens (Summarizer - {current_state}): {cb.total_tokens} (P: {cb.prompt_tokens}, C: {cb.completion_tokens})")

        print(f"  -> 总结的反馈: {summarized_feedback}")
        
        # *** 关键 ***: 用总结后的简洁反馈，覆盖掉 state 中的原始反馈
        return {
            "evaluation_feedback": summarized_feedback 
        }
    except Exception as e:
        print(f"  -> ERROR: Feedback summarizer LLM call failed: {e}")
        # 如果总结失败，保留原始反馈，让 Refiner 至少可以尝试
        return {}

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