# src/workflows/schemas.py

from typing import TypedDict, Literal, List, Any
from pydantic import BaseModel, Field

class IterationLog(TypedDict):
    """记录单次迭代的关键信息"""
    iteration: int
    test_code: str
    pass_rate: float
    coverage: float
    feedback: str

EvaluationResult = Literal[
    "NOT_STARTED",        # 初始状态
    "RETRY_E_CASE",       # 因执行错误(E-Case)重试
    "RETRY_COVERAGE",     # 因覆盖率不足重试
    "QUALITY_MET",        # 质量循环：通过（0 E-Case, 覆盖率达标）
    "FAIL_F_CASE",        # 最终裁决：因F-Case而失败（硬停止）
    "PASS_TO_MUTATION",   # 最终裁决：通过（0 F-Case），进入变异测试
    "RETRY_MUTATION",     # 变异测试：失败，触发修复
    "PASS_FINAL",         # 变异测试：通过，流程成功结束
    "FAIL_CRITICAL"       # 任何阶段的工具或严重错误
]

class WorkflowState(TypedDict):
    """定义工作流中传递的状态"""
    # 核心数据
    code: str                  # 待测试的源代码
    requirement: str           # 用户的测试需求
    analysis_report: str       # 代码静态分析报告 (JSON 字符串)
    structured_requirement: str    # 结构化需求报告 (JSON 字符串)
    generation_prompt: str     # 用于生成测试用例的最终 Prompt
    test_code: str             # 生成的测试用例代码

    analysis_model: Any        # (来自 code_analyzer) 代码分析的原始字典模型
    requirement_model: Any     # (来自 requirement_analyzer) 需求分析的Pydantic模型 (FullTestModel)
    validation_report: str     # (来自 validator_node) 静态验证报告 (JSON string)
    
    # 结果与迭代
    coverage: float            # 测试覆盖率
    pass_rate: float           # 测试通过率
    test_failures: int         # 测试失败 (F cases) 的数量
    test_errors: int           # 测试错误 (E cases) 的数量

    execution_feedback: str    # 来自测试执行器的原始反馈

    evaluation_result: EvaluationResult  # 当前的评估状态
    evaluation_feedback: str   # *专门*用于“修复者”的反馈
    
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