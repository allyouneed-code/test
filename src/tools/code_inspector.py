# static_analyzer.py

import ast
import json
import subprocess
import tempfile
import os
from typing import Dict, Any, List
import textwrap

class CodeAnalyzer:
    """
    一个静态分析工具，用于分析待测试的 Python 代码文件，
    提取其结构、逻辑、复杂度和质量特征，为LLM生成单元测试提供上下文。
    """

    def __init__(self, source_code: str):
        """
        初始化分析器。
        :param source_code: 待分析的源代码字符串。
        """
        if not source_code.strip():
            raise ValueError("源代码不能为空")
        self.source_code = source_code
        # 尝试解析 AST，如果解析失败（如因编码问题），会在此时抛出错误
        # 实际运行中，如果代码包含中文，可能需要在顶层添加 # -*- coding: utf-8 -*-
        try:
            self.tree = ast.parse(self.source_code)
        except Exception as e:
            # 这里的 ast.parse 失败可能也是编码问题，但我们假设输入是可解析的
            raise SyntaxError(f"AST 解析失败，可能是代码语法错误或编码问题: {e}")

    def analyze(self) -> Dict[str, Any]:
        """
        执行所有分析步骤，并返回一个包含所有信息的字典。
        修正: 确保临时文件使用 'utf-8' 编码。
        :return: 包含代码静态结构信息的字典。
        """
        analysis_result = {}
        self.filepath = None
        
        # 修正：明确指定 encoding='utf-8'，以避免平台默认编码不一致导致的问题
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            self.filepath = tmp.name
            tmp.write(self.source_code)
        
        try:
            analysis_result = {
                "a_structural_features": self._analyze_structure(),
                "b_logical_features": self._analyze_logic(),
                "c_complexity_features": self._analyze_complexity(),
                "d_quality_features": self._analyze_quality()
            }
        except Exception as e:
            # 捕获其他未知错误
            analysis_result["fatal_error"] = f"分析过程中发生未知错误: {e}"
        finally:
            # 清理临时文件
            if self.filepath and os.path.exists(self.filepath):
                os.unlink(self.filepath)

        return analysis_result

    # ==================================================================
    # a. 结构特征 (Structural Features) - 无需修正
    # ==================================================================
    def _analyze_structure(self) -> Dict[str, Any]:
        """
        分析代码的模块、类、函数和变量的层次关系与属性。
        """
        structure = {
            "module_docstring": ast.get_docstring(self.tree),
            "classes": [],
            "functions": [],
            "global_variables": []
        }

        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "methods": [],
                    "attributes": [] # 类级别属性
                }
                for body_item in node.body:
                    if isinstance(body_item, ast.FunctionDef):
                        method_info = self._extract_function_info(body_item)
                        class_info["methods"].append(method_info)
                    elif isinstance(body_item, ast.Assign):
                        for target in body_item.targets:
                            if isinstance(target, ast.Name):
                                class_info["attributes"].append(target.id)
                structure["classes"].append(class_info)
            elif isinstance(node, ast.FunctionDef):
                structure["functions"].append(self._extract_function_info(node))
            elif isinstance(node, ast.Assign):
                  for target in node.targets:
                      if isinstance(target, ast.Name):
                          structure["global_variables"].append(target.id)

        return structure

    def _extract_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """辅助函数：从 AST 节点提取函数信息"""
        return {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": [arg.arg for arg in node.args.args],
            "returns": ast.unparse(node.returns) if node.returns else None
        }

    # ==================================================================
    # b. 逻辑特征 (Logical Features) - 无需修正
    # ==================================================================
    def _analyze_logic(self) -> Dict[str, Any]:
        """
        分析代码的约束条件和外部依赖。
        """
        imports = set()
        conditions = [] # 例如 if, while, assert 等语句
        
        for node in ast.walk(self.tree):
            # 提取外部依赖
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
            
            # 提取逻辑约束（简化版，仅记录类型和行号）
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Assert, ast.Try)):
                conditions.append({
                    "type": node.__class__.__name__,
                    "lineno": node.lineno
                })

        return {
            "external_dependencies": sorted(list(imports)),
            "constraint_conditions": conditions
        }

    # ==================================================================
    # c. 复杂度特征 (Complexity Features)
    # ==================================================================
    def _analyze_complexity(self) -> Dict[str, Any]:
        """
        使用 radon 库分析圈复杂度和代码行数。
        修正: 确保 text=True 且处理编码，并捕获 JSONDecodeError。
        """
        try:
            # 圈复杂度 (Cyclomatic Complexity)
            cmd_cc = f"radon cc \"{self.filepath}\" -j"
            # 修正：明确 encoding='utf-8'，以防 radon 输出的 JSON 中包含中文
            result_cc = subprocess.run(cmd_cc, capture_output=True, text=True, encoding='gbk', shell=True)
            
            # 检查是否有错误或非JSON输出
            if result_cc.stderr or not result_cc.stdout.strip():
                 # radon 在遇到编码错误时会把错误信息输出到 stdout/stderr，并可能不是 JSON
                 error_msg = result_cc.stderr or result_cc.stdout
                 return {"cyclomatic_complexity": {"error": f"Radon CC error: {error_msg.strip()}"}, "lines_of_code": "N/A"}

            cc_data = json.loads(result_cc.stdout).get(self.filepath, [])
            
            # 代码行数 (Lines of Code)
            cmd_raw = f"radon raw \"{self.filepath}\" -j"
            result_raw = subprocess.run(cmd_raw, capture_output=True, text=True, encoding='utf-8', shell=True)
            raw_data = json.loads(result_raw.stdout).get(self.filepath, {})

            return {
                "cyclomatic_complexity": cc_data,
                "lines_of_code": {
                    "total_lines": raw_data.get('loc', 'N/A'),
                    "source_lines_of_code": raw_data.get('sloc', 'N/A'),
                    "comment_lines": raw_data.get('comments', 'N/A'),
                }
            }
        except json.JSONDecodeError as e:
            return {"error": f"Radon JSON 解析失败 (stdout: {result_cc.stdout[:50]}...): {str(e)}"}
        except Exception as e:
            return {"error": f"Radon analysis failed: {str(e)}"}


    # ==================================================================
    # d. 质量特征 (Quality Features)
    # ==================================================================
    def _analyze_quality(self) -> Dict[str, Any]:
        """
        分析代码异味和安全漏洞。
        """
        return {
            "test_coverage": "N/A - Test coverage is a runtime metric and cannot be determined by static analysis. It should be provided externally.",
            "code_smells": self._run_pylint(),
            "security_vulnerabilities": self._run_bandit()
        }

    def _run_bandit(self) -> List[Dict[str, Any]]:
        """
        使用 bandit 工具检测安全漏洞
        修正: 确保 text=True 且处理编码，并捕获 JSONDecodeError。
        """
        try:
            cmd = f"bandit -r \"{self.filepath}\" -f json --quiet" # 添加 --quiet 减少非 JSON 输出
            # 修正：明确 encoding='utf-8'
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', shell=True)
            
            # 检查 stderr，bandit 错误信息通常在 stderr
            if result.stderr:
                 return [{"error": f"Bandit execution error (stderr): {result.stderr.strip()}"}]

            # 检查 stdout，防止非 JSON 干扰
            stdout_str = result.stdout.strip()
            if not stdout_str:
                # 可能是没有发现漏洞且没有输出
                return []
            
            bandit_results = json.loads(stdout_str)
            return bandit_results.get("results", [])
        except json.JSONDecodeError as e:
            # 捕获 JSON 解析失败，这正是原问题中 'Expect value' 错误的来源
            return [{"error": f"Bandit JSON 解析失败 (stdout: {result.stdout[:50]}...): {str(e)}"}]
        except Exception as e:
            return [{"error": f"Bandit analysis failed: {str(e)}"}]

    def _run_pylint(self) -> List[Dict[str, Any]]:
        """
        使用 pylint 工具检测代码异味
        修正: 确保 text=True 且处理编码。
        """
        try:
            # 使用 JSON 输出格式
            cmd = f"pylint \"{self.filepath}\" --output-format=json"
            # 修正：明确 encoding='utf-8'
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', shell=True)
            
            if result.stderr:
                 # Pylint 错误或警告信息通常在 stdout，stderr 可能是致命错误
                 return [{"error": f"Pylint execution error (stderr): {result.stderr.strip()}"}]

            # Pylint 在没有问题时输出 '[]' 或空字符串
            stdout_str = result.stdout.strip()
            if not stdout_str or stdout_str == '[]':
                return []
            
            # Pylint 在解析失败时，stdout 可能不是有效的 JSON
            pylint_results = json.loads(stdout_str)
            return pylint_results
        except json.JSONDecodeError as e:
            return [{"error": f"Pylint JSON 解析失败 (stdout: {result.stdout[:50]}...): {str(e)}"}]
        except Exception as e:
            return [{"error": f"Pylint analysis failed: {str(e)}"}]
            
# 保持 sample_code_to_test 不变
sample_code_to_test = textwrap.dedent("""
    import os

    class Calculator:
        \"\"\"一个简单的计算器类\"\"\"
        def __init__(self):
            self.result = 0

        def add(self, a, b):
            \"\"\"加法运算\"\"\"
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise TypeError("Inputs must be numeric")
            self.result = a + b
            return self.result

        def subtract(self, a, b):
            \"\"\"减法运算\"\"\"
            return a - b

        def execute_command(self, cmd):
            \"\"\"执行一个系统命令（这是一个危险的操作，用于安全检测）\"\"\"
            os.system(cmd) # bandit应该能检测到这个漏洞

    def standalone_multiply(x, y):
        \"\"\"一个独立的乘法函数\"\"\"
        return x * y

    # 全局变量
    PI = 3.14159
""")

if __name__ == "__main__":
    # 1. 初始化分析器
    analyzer = CodeAnalyzer(source_code=sample_code_to_test)

    # 2. 执行分析
    static_info = analyzer.analyze()

    # 3. 将结果格式化输出（这个 JSON 就可以喂给 LLM）
    # 修正: 增加 ensure_ascii=False 以便正确显示中文
    pretty_json_output = json.dumps(static_info, indent=4, ensure_ascii=False)
    
    print("代码静态分析结果：")
    print(pretty_json_output)