# src/workflows/schemas.py

from typing import TypedDict, Literal, List
from pydantic import BaseModel, Field

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