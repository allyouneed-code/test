import subprocess
import os
import glob
import json
import re
import shutil 
import textwrap

# --- 配置 ---覆盖率 JSON 文件名
COV_JSON_FILE = "coverage.json"

def _write_code_to_file(content, filename):
    """将代码字符串写入指定文件，确保使用 UTF-8 编码。"""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            # 添加编码声明
            if not content.strip().startswith('# -*- coding: utf-8 -*-'):
                file.write("# -*- coding: utf-8 -*-\n")
            file.write(content)
        return True
    except Exception as e:
        print(f"写入文件 {filename} 时出错: {e}")
        return False

def _run_pytest_with_coverage(test_filename: str):
    """使用 coverage run -m pytest 执行测试，并返回 subprocess.run 结果。"""
    
    # 构造 Pytest Coverage 运行命令
    # --source=. 用于告诉 coverage 只测量当前目录下的代码
    # -m pytest 用于以模块形式运行 pytest
    # --parallel-mode: 确保在子进程中也能正确收集覆盖率
    coverage_command = [
        "coverage", "run", "--parallel-mode", 
        "--source=.",  # <--- 确保使用 '.' 来追踪当前目录
        "-m", "pytest", test_filename
    ]
    
    print(f"\n[Executor] Running tests: {' '.join(coverage_command)}")
    
    # 运行子进程并捕获结果
    # check=False 允许 pytest 失败（返回码为 1）而不抛出异常，以便继续生成报告
    result = subprocess.run(
        coverage_command, 
        capture_output=True, 
        text=True, 
        check=False
    )

    # 打印 Pytest 原始输出，方便调试
    print("-" * 50)
    print(f"[Pytest STDOUT]:\n{result.stdout}")
    print("-" * 50)
    
    return result

def _parse_pytest_summary(output):
    """从 pytest 的输出中解析用例的通过/失败统计数据。"""
    summary = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'skipped': 0,
        'pass_rate': 0.0
    }
    
    # 查找包含 "passed", "failed"... 的行
    counts = re.findall(r'(\d+)\s+(passed|failed|skipped|error|xfailed|xpassed)', output)
    
    for count_str, status in counts:
        count = int(count_str)
        summary['total_tests'] += count
        
        if status == 'passed':
            summary['passed'] = count
        elif status == 'failed':
            summary['failed'] = count
        elif status in ('skipped', 'error', 'xfailed', 'xpassed'):
            summary[status] = summary.get(status, 0) + count # 累加其他状态
            
    if summary['total_tests'] > 0:
        summary['pass_rate'] = round(summary['passed'] / summary['total_tests'], 4)
        
    return summary


def _get_coverage_metrics(logic_filename):
    """
    运行 coverage report -m 命令，解析其文本输出，提取覆盖率关键指标（仅针对 logic_filename）。
    """
    
    # 定义用于匹配报告行的正则表达式
    # 匹配文件名, Stmts, Miss, Cover, Missing 行号
    # 例如： logic_module.py    7      1    86%   8
    REPORT_PATTERN = re.compile(
        r"^(?P<name>\S+)\s+"  # 文件名 (name)
        r"(?P<stmts>\d+)\s+"  # 语句数 (stmts)
        r"(?P<miss>\d+)\s+"   # 缺失数 (miss)
        r"(?P<cover>\d+)%\s+" # 覆盖率 (cover)
        r"(?P<missing_lines>.*)$"  # 缺失行号 (missing_lines)
    )

    try:
        # 1. 合并数据文件
        subprocess.run(["coverage", "combine"], check=True, capture_output=True)

        # 2. 生成文本报告并捕获输出
        # -m 选项会输出缺失的行号
        report_result = subprocess.run(
            ["coverage", "report", "-m", logic_filename], 
            capture_output=True, 
            text=True, 
            check=False # 即使 coverage report 失败也要处理
        )
        
        report_output = report_result.stdout

        print(f"\n[Coverage Report Output]:\n{report_output}\n")

        # 3. 查找目标文件 logic_filename 的报告行
        target_line = None
        for line in report_output.splitlines():
            # 需要对文件名进行处理，确保匹配是精确的，特别是当文件名中有路径时
            if line.strip().startswith(logic_filename):
                target_line = line.strip()
                break

        if not target_line:
            # 文件未被追踪或未执行
            print(f"[Coverage] WARNING: Target file {logic_filename} not found in coverage data. Assuming 0% coverage.")
            return {
                'covered_percentage': 0.0,
                'total_statements': 0, 
                'executed_lines': 0,
                'missing_lines_count': 0, 
                'missing_lines': [],
                'error_detail': f"Target logic file '{logic_filename}' was not executed or tracked.",
            }
        
        # 4. 使用正则表达式解析数据
        match = REPORT_PATTERN.search(target_line)
        
        if not match:
            # 报告格式不匹配，通常不会发生，除非 coverage.py 版本变化
            raise ValueError(f"Failed to parse report line: {target_line}")

        data = match.groupdict()
        
        missing_lines_str = data.get('missing_lines', '').strip()

        # 5. 构建覆盖率字典
        coverage_metrics = {
            'covered_percentage': float(data.get('cover', 0.0)),
            'total_statements': int(data.get('stmts', 0)),
            'executed_lines': int(data.get('stmts', 0)) - int(data.get('miss', 0)),
            'missing_lines_count': int(data.get('miss', 0)),
            'missing_lines': missing_lines_str, # 直接返回原始缺失行字符串
        }
        
        return coverage_metrics

    except Exception as e:
        print(f"[Coverage] ERROR: Failed to process coverage metrics. {e}")
        return {'error': str(e)}

def _cleanup(files_to_clean):
    """清理生成的文件和 coverage 数据文件。"""
    for filename in files_to_clean:
        if os.path.exists(filename):
            os.remove(filename)
    
    # 清理所有 .coverage* 文件
    for f in glob.glob(".coverage*"):
        os.remove(f)
    if os.path.exists('htmlcov'):
        shutil.rmtree('htmlcov') 
        
    
    print("\n[Executor] Cleanup complete.")

# ----------------------------------------------------------------------
#                         AGENT TOOL ENTRY POINT
# ----------------------------------------------------------------------

def execute_tests_and_get_report(logic_code: str, test_code: str, logic_filename: str, test_filename: str) -> dict:
    """
    Agent Tool 的主要入口点。
    将业务逻辑代码和 Pytest 测试代码写入文件，执行测试，并返回结构化的报告。

    Args:
        logic_code: 包含被测函数（如 calculate）的业务逻辑代码字符串。
        test_code: 包含 Pytest 风格测试函数（如 test_add）的测试代码字符串。
        logic_filename: 业务逻辑文件名（默认为 logic_module.py）。

    Returns:
        一个包含 'test_execution' 和 'coverage_metrics' 的结构化字典。
    """
    
    # 确保 logic_filename 和 TEST_FILENAME 存在于清理列表
    files_to_clean = [logic_filename, test_filename, COV_JSON_FILE]
    final_report = {'test_execution': {}, 'coverage_metrics': {}}
    
    try:
        # --- 1. 写入文件 ---
        if not _write_code_to_file(logic_code, logic_filename): return {'error': f"Failed to write logic code to {logic_filename}"}
        if not _write_code_to_file(test_code, test_filename): return {'error': f"Failed to write test code to {TEST_FILENAME}"}

        # --- 2. 执行测试 ---
        pytest_result = _run_pytest_with_coverage(test_filename)
        
        # --- 3. 收集指标 ---
        final_report['test_execution'] = _parse_pytest_summary(pytest_result.stdout)
        final_report['coverage_metrics'] = _get_coverage_metrics(logic_filename)
        
        # --- 4. 辅助报告生成 (可选: 生成HTML报告供人工调试) ---
        subprocess.run(["coverage", "html"], check=True, capture_output=True)
        print("\n[Executor] HTML report generated in 'htmlcov/' directory.")
        
        return final_report
        
    except FileNotFoundError as e:
        final_report['error'] = f"Required tool not found. Please ensure 'pytest', 'coverage', and 'python' are installed and in PATH. Error: {e}"
        return final_report
    except Exception as e:
        final_report['error'] = f"An unexpected error occurred during execution: {e}"
        return final_report
    finally:
        # --- 5. 清理 ---
        _cleanup(files_to_clean)


# ----------------------------------------------------------------------
#                            EXAMPLE USAGE
# ----------------------------------------------------------------------

if __name__ == "__main__":
    LOGIC_FILENAME = "logic_module.py"
    TEST_FILENAME = "test_script.py"
    # 1. 业务逻辑代码
    sample_logic_code = """
def calculate(a, b, operation):
    if operation == 'add':
        return a + b
    if operation == 'subtract':
        return a - b
    # 这一行是故意留下的未覆盖代码
    if operation == 'multiply':
        return a * b 
    return None
"""
    
    # 2. Pytest 风格测试用例代码
    sample_test_code = f"""
from {LOGIC_FILENAME.replace('.py', '')} import calculate 

def test_add_success():
    assert calculate(5, 3, 'add') == 8

def test_subtract_success():
    assert calculate(10, 4, 'subtract') == 6

def test_unsupported_operation():
    # 确保覆盖到最后的 return None
    assert calculate(10, 4, 'divide') is None

def test_add_failure_example():
    # 模拟一个失败的用例
    assert calculate(1, 1, 'add') == 3 # 实际是 2, 预期是 3
"""
    
    # 调用入口函数
    report = execute_tests_and_get_report(sample_logic_code, sample_test_code)
    
    print("\n" + "="*70)
    print("                 FINAL STRUCTURED REPORT (JSON FORMAT)                ")
    print("="*70)
    print(json.dumps(report, indent=4))
    print("="*70)