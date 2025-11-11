# src/workflows/test_gen_workflow.py

import time
from functools import partial
from typing import Literal

from langgraph.graph import StateGraph, END

# +++ 导入模块 +++
from .schemas import WorkflowState
from . import nodes as wf_nodes
from ..config import app_config
from ..report.reporter_console import WorkflowReporter

class TestGenerationWorkflow:
    """
    基于图的自动化测试生成工作流。
    负责构建和编排计算图。
    """
    def __init__(self, config: dict):
        self.config = config
        self.max_retries = config.get("max_retries", 3)

    def quality_router(self, state: WorkflowState) -> Literal["test_refiner_node", "final_triage", "end"]:
        """
        决策点1 (在 Quality Evaluator 之后):
        - 质量不达标 (E-Case / Coverage) -> 重试 (Refiner)
        - 质量达标 (0 E-Case, Cov OK) -> 进入最终裁决 (Triage)
        """
        print("--- 决策点1: 质量路由 (Quality Router) ---")
        
        result = state.get("evaluation_result")
        
        if result == "RETRY_QUALITY":
            if state.get("retry_count", 0) < self.max_retries:
                print(f"  -> 路由: 质量不达标 (E-Case/Coverage)。路由到 [Refiner] (重试 {state.get('retry_count', 0) + 1}/{self.max_retries})")
                return "test_refiner_node"
            else:
                print(f"  -> 路由: 达到最大重试次数 ({self.max_retries})。流程终止。")
                return "end"
        
        elif result == "QUALITY_MET":
            print("  -> 路由: 质量达标。路由到 [Final Triage]")
            return "final_triage"
            
        else:
            # 兜底，理论上不应发生
            print(f"  -> 路由: 评估结果未知 ({result})。流程终止。")
            return "end"
        
    def final_triage_router(self, state: WorkflowState) -> Literal["mutation_test_node", "end"]:
        """
        决策点2 (在 Quality Met 之后):
        - 检查 F-Cases。
        - 0 F-Cases -> 进入变异测试
        - >0 F-Cases -> 硬停止
        """
        print("--- 决策点2: 最终裁决 (Final Triage) ---")
        
        test_failures = state.get("test_failures", 0)
        
        if test_failures > 0:
            print(f"  -> 裁决: 发现 {test_failures} 个测试失败 (F-Cases)。流程终止。")
            # 我们需要一个方法来设置最终状态为 FAIL_F_CASE
            # 但 langgraph 的 END 节点不接受输入。
            # 我们依赖 quality_evaluator_node 已经在 F-Case 时设置了正确的状态。
            return "end"
        else:
            print("  -> 裁决: 0 F-Cases。路由到 [Mutation Test]")
            return "mutation_test_node"
        
    def mutation_test_router(self, state: WorkflowState) -> Literal["test_refiner_node", "end"]:
        """
        决策点3 (在 Mutation Test 之后):
        - 变异测试失败 -> 重试 (Refiner)
        - 变异测试通过/工具错误 -> 结束
        """
        print("--- 决策点3: 变异测试路由 (Mutation Router) ---")
        
        result = state.get("evaluation_result")
        
        if result == "RETRY_MUTATION":
             if state.get("retry_count", 0) < self.max_retries:
                print(f"  -> 路由: 变异测试得分低。路由到 [Refiner] (重试 {state.get('retry_count', 0) + 1}/{self.max_retries})")
                return "test_refiner_node"
             else:
                print(f"  -> 路由: 达到最大重试次数 ({self.max_retries})。流程终止。")
                return "end"
        
        elif result == "PASS_FINAL":
            print("  -> 路由: 变异测试通过。流程结束。")
            return "end"
            
        elif result == "FAIL_CRITICAL":
            print("  -> 路由: 变异测试工具出错。流程终止。")
            return "end"
            
        else:
            print(f"  -> 路由: 变异测试评估结果未知 ({result})。流程终止。")
            return "end"


    # --- 图构建方法 ---
    def build(self):
        """将所有节点和逻辑边组装成一个可执行的 LangGraph 计算图。"""
        graph = StateGraph(WorkflowState)

        # --- 使用 functools.partial 绑定节点所需的额外参数 ---
        # 这样可以保持节点函数的签名纯粹，只接收 state
        test_executor_with_config = partial(wf_nodes.test_executor_node, 
                                            logic_filename=self.config["logic_filename"], 
                                            test_filename=self.config["test_filename"])
        quality_evaluator_with_config = partial(wf_nodes.result_evaluator_node, 
                                                coverage_threshold=self.config["coverage_threshold"])
        mutation_test_with_config = partial(wf_nodes.mutation_test_node, 
                                            app_config=self.config)
        # 注册所有节点
        graph.add_node("code_analyzer", wf_nodes.code_analyzer_node)
        graph.add_node("requirement_analyzer", wf_nodes.requirement_analyzer_node)
        graph.add_node("validator", wf_nodes.validator_node)
        graph.add_node("prompt_organizer", partial(wf_nodes.prompt_organizer_node, logic_filename=self.config["logic_filename"]))

        graph.add_node("test_creator_node", wf_nodes.test_creator_node)
        graph.add_node("test_reviewer_node", wf_nodes.test_reviewer_node)

        graph.add_node("test_executor", test_executor_with_config)
        graph.add_node("quality_evaluator_node", quality_evaluator_with_config)
        graph.add_node("test_refiner_node", wf_nodes.test_refiner_node) 

        graph.add_node("mutation_test_node", mutation_test_with_config)
        # 定义工作流的执行路径
        # 阶段 1: 线性分析流程
        graph.set_entry_point("code_analyzer")
        graph.add_edge("code_analyzer", "requirement_analyzer")
        graph.add_edge("requirement_analyzer", "validator")
        graph.add_edge("validator", "prompt_organizer")
        
        # 阶段 2: 创造与审查
        graph.add_edge("prompt_organizer", "test_creator_node")
        graph.add_edge("test_creator_node", "test_reviewer_node")

        # 阶段 3: 执行与质量循环
        graph.add_edge("test_reviewer_node", "test_executor") # 首次执行
        graph.add_edge("test_executor", "quality_evaluator_node")
        graph.add_edge("test_refiner_node", "test_executor")
        # 添加条件分支
        graph.add_conditional_edges(
            "quality_evaluator_node",
            self.quality_router,
            {
                "test_refiner_node": "test_refiner_node",
                "final_triage": "final_triage_router", # <-- 转到“最终裁决”
                "end": END
            }
        )
        
        # 阶段 4: 添加“最终裁决”路由
        # (我们使用一个 "dummy" 节点（lambda）作为路由锚点，因为 LangGraph 3.0+ 不推荐无节点路由)
        graph.add_node("final_triage_router", lambda state: state) # Dummy 节点
        graph.add_conditional_edges(
            "final_triage_router",
            self.final_triage_router,
            {
                "mutation_test_node": "mutation_test_node",
                "end": END
            }
        )

        # 阶段 4: 添加“变异测试”路由
        graph.add_conditional_edges(
            "mutation_test_node",
            self.mutation_test_router,
            {
                "test_refiner_node": "test_refiner_node", # 变异测试失败也送去修复
                "end": END
            }
        )

        return graph.compile()

# ------------------------- 示例运行 -------------------------

if __name__ == "__main__":
    #  定义待测试的业务逻辑代码
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

    #  用户的测试需求
    user_requirement = "Write a comprehensive test suite for the 'calculate' function. Ensure all operations ('add', 'subtract', 'multiply', 'divide') are tested. Also, specifically test the edge case of division by zero and invalid input types."

    #  初始化并构建工作流
    workflow_builder = TestGenerationWorkflow(config=app_config)
    app = workflow_builder.build()
    

    #  执行工作流
    print("\n" + "="*50)
    print("           开始测试用例生成           ")
    print("="*50 + "\n")
    
    # --- 确保所有新字段都已初始化 ---
    initial_state = {
        "code": sample_logic_code, 
        "requirement": user_requirement, 
        "retry_count": 0,
        "start_time": time.time(), 
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0, 
        "total_tokens": 0,
        "iteration_history": [],
        "mutation_test_has_error": False, 
        "mutation_error_details": "",
        "test_failures": 0,
        "test_errors": 0,
        "evaluation_result": "NOT_STARTED",
        "evaluation_feedback": "",
        "analysis_report": "",
        "structured_requirement": "",
        "generation_prompt": "",
        "test_code": "",
        "analysis_model": None,
        "requirement_model": None,
        "validation_report": ""
    }
    final_state = app.invoke(initial_state)

    #  打印最终结果
    reporter = WorkflowReporter(final_state, app_config)
    reporter.generate_report()
