# src/reporting.py

import time
from typing import Dict, Any
from .workflows.schemas import WorkflowState

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