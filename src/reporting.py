# src/reporting.py

import time
from typing import Dict, Any
from .workflows.schemas import WorkflowState, EvaluationResult
import json

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
    
    def _format_final_result(self, result: EvaluationResult) -> str:
        """将内部状态名转换为用户友好的报告字符串"""
        if result == "PASS_FINAL":
            return "PASS (变异测试通过)"
        if result == "FAIL_F_CASE":
            return "FAIL (测试用例执行失败)"
        if result == "FAIL_CRITICAL":
            return "FAIL (工具严重错误)"
        if result == "RETRY_MUTATION" or result == "RETRY_QUALITY":
            return f"FAIL (达到最大重试次数 {self.config.get('max_retries', 3)})"
        
        # 兜底
        return str(result).upper()    
    
    def generate_report(self):
        """
        生成并打印完整的成本与效率报告。
        """
        self._calculate_metrics()
        
        final_result_str = self._format_final_result(self.state.get('evaluation_result'))

        print("\n" + "="*50 + "\n            WORKFLOW COMPLETED - FINAL REPORT\n" + "="*50)

        if self.state.get("mutation_test_has_error"):
            print("\n[!!!] WORKFLOW HALTED DUE TO A CRITICAL ERROR.")
            print("  -> The mutation testing tool failed to execute.")
            print(f"  -> Error Details: {self.state.get('mutation_error_details')}")
        
        max_retries = self.config.get("max_retries", 3)
        if self.state.get('retry_count', 0) >= max_retries and "FAIL" not in final_result_str:
            print(f"\n[WARNING] Workflow stopped due to reaching the maximum retry limit ({max_retries}).")

        print(f"\nFinal Evaluation Result: {final_result_str}")
        
        # --- 新增：如果失败，打印最后一次的执行反馈 ---
        if "FAIL" in final_result_str and self.state.get("execution_feedback"):
             print("\n--- Last Execution Feedback (Failure Details) ---")
             print(self.state.get("execution_feedback"))
        # --- 结束 ---

        print("\n--- Generated Test Code (Final Version) ---")
        print(self.state.get('test_code', 'No code generated.'))
        
        print("\n" + "-"*20 + " Static Validation Report " + "-"*18)
        validation_report_str = self.state.get('validation_report')
        if validation_report_str:
            try:
                report_json = json.loads(validation_report_str)
                print(json.dumps(report_json, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(validation_report_str)
        else:
            print("No validation report was generated.")
        
        print("\n" + "-"*20 + " Cost & Efficiency Metrics " + "-"*20)
        print(f"  - Total Execution Time: {self.state.get('total_execution_time', 0.0):.2f} seconds")
        print(f"  - Total Iterations (Refinement): {self.state.get('retry_count', 0)}")
        print(f"  - Token Consumption:")
        print(f"    - Total Prompt Tokens:    {self.state.get('total_prompt_tokens', 0)}")
        print(f"    - Total Completion Tokens:  {self.state.get('total_completion_tokens', 0)}")
        print(f"    - Grand Total Tokens:     {self.state.get('total_tokens', 0)}")
        
        print("\n" + "-"*20 + " Quality Metrics " + "-"*26)
        print(f"  - Final Coverage: {self.state.get('coverage', 0.0):.2%}")
        print(f"  - Final Pass Rate: {self.state.get('pass_rate', 0.0):.2%}")
        print(f"  - Final Mutation Score: {self.state.get('mutation_score', 0.0):.2%}")
        print("="*67)