# src/workflows/test_gen_workflow.py

import time
from functools import partial
from typing import Literal

from langgraph.graph import StateGraph, END

# +++ 导入模块 +++
from .schemas import WorkflowState
from . import nodes as wf_nodes
from ..config import app_config
from ..reporting import WorkflowReporter

class TestGenerationWorkflow:
    """
    基于图的自动化测试生成工作流。
    负责构建和编排计算图。
    """
    def __init__(self, config: dict):
        self.config = config

    # --- 决策逻辑 ---
    def route_after_initial_evaluation(self, state: WorkflowState) -> Literal["mutation_test", "retry", "end"]:
        print("--- 决策点1 ---")
        if state["evaluation_result"] == "pass":
            print("测试通过，无需变异测试或重试。")
            return "mutation_test"
        if state["retry_count"] >= self.config["max_retries"]:
            print("达到最大重试次数，结束流程。")
            return "end"
        print("测试未通过，准备重试。")
        return "retry"

    def route_after_mutation_test(self, state: WorkflowState) -> Literal["end", "retry", "critical_error"]:
        print("--- 决策点2 ---")
        if state.get("mutation_test_has_error"):
            print("变异测试过程中出现严重错误，结束流程。")
            return "critical_error"
        if state.get("mutation_score", 0.0) >= self.config["mutation_threshold"]:
            print("变异测试通过，结束流程。")
            return "end"
        else:
            if state["retry_count"] >= self.config["max_retries"]:
                print("达到最大重试次数，结束流程。")
                return "end"
            print("变异测试未通过，准备重试。")
            return "retry"

    # --- 图构建方法 ---
    def build(self):
        """将所有节点和逻辑边组装成一个可执行的 LangGraph 计算图。"""
        graph = StateGraph(WorkflowState)

        # --- 使用 functools.partial 绑定节点所需的额外参数 ---
        # 这样可以保持节点函数的签名纯粹，只接收 state
        prompt_organizer_with_config = partial(wf_nodes.prompt_organizer_node, logic_filename=self.config["logic_filename"])
        test_executor_with_config = partial(wf_nodes.test_executor_node, logic_filename=self.config["logic_filename"], test_filename=self.config["test_filename"])
        result_evaluator_with_config = partial(wf_nodes.result_evaluator_node, coverage_threshold=self.config["coverage_threshold"])
        mutation_test_with_config = partial(wf_nodes.mutation_test_node, app_config=self.config)
        
        # 注册所有节点
        graph.add_node("code_analyzer", wf_nodes.code_analyzer_node)
        graph.add_node("requirement_analyzer", wf_nodes.requirement_analyzer_node)
        graph.add_node("prompt_organizer", prompt_organizer_with_config)
        graph.add_node("test_generator", wf_nodes.test_generator_node)
        graph.add_node("test_executor", test_executor_with_config)
        graph.add_node("result_evaluator", result_evaluator_with_config)
        graph.add_node("mutation_test_node", mutation_test_with_config)

        # 定义工作流的执行路径
        graph.set_entry_point("code_analyzer")
        graph.add_edge("code_analyzer", "requirement_analyzer")
        graph.add_edge("requirement_analyzer", "prompt_organizer")
        graph.add_edge("prompt_organizer", "test_generator")
        graph.add_edge("test_generator", "test_executor")
        graph.add_edge("test_executor", "result_evaluator")

        # 添加条件分支
        graph.add_conditional_edges("result_evaluator", self.route_after_initial_evaluation,
            {"mutation_test": "mutation_test_node", "retry": "test_generator", "end": END})
        
        graph.add_conditional_edges("mutation_test_node", self.route_after_mutation_test,
            {"retry": "test_generator", "end": END, "critical_error": END})

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
    
    initial_state = {
        "code": sample_logic_code, "requirement": user_requirement, "retry_count": 0,
        "start_time": time.time(), "total_prompt_tokens": 0,
        "total_completion_tokens": 0, "total_tokens": 0,
        "iteration_history": [],
        "mutation_test_has_error": False, 
        "mutation_error_details": ""      
    }
    final_state = app.invoke(initial_state)

    #  打印最终结果
    reporter = WorkflowReporter(final_state, app_config)
    reporter.generate_report()
