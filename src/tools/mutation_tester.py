# -*- coding: utf-8 -*-
import subprocess
import os
import re
import shutil
import sys
import textwrap
from typing import Dict, Any, List

def parse_survived_mutants(output: str) -> List[Dict[str, str]]:
    """
    从 mutpy 的详细输出中准确解析出所有存活的变异体信息。
    采用“先分割，再匹配”的健壮逻辑。
    """
    survived_mutants = []
    # 1. 定位到所有变异体报告开始的地方
    reports_section_match = re.search(r"\[\*\] Start mutants generation and execution:(.*?)\[\*\] Mutation score", output, re.DOTALL)
    if not reports_section_match:
        return []
    reports_section = reports_section_match.group(1)

    # 2. 将所有变异体报告分割成独立的块
    # 每个报告都以 "- [# " 开头
    mutant_blocks = re.split(r'\n\s*-\s\[#', reports_section)

    for block in mutant_blocks:
        if not block.strip():
            continue

        # 3. 处理那些明确标记为 "survived" 的块
        if re.search(r'\[.*?s\]\s+survived\s*$', block, re.DOTALL):
            id_match = re.search(r'^\s*(\d+)\].*?', block)
            original_match = re.search(r'^\s*-\s*(\d+):\s*(.*)', block, re.MULTILINE)
            mutated_match = re.search(r'^\s*\+\s*(\d+):\s*(.*)', block, re.MULTILINE)

            if id_match and original_match and mutated_match:
                survived_mutants.append({
                    "id": id_match.group(1).strip(),
                    "original_line_no": original_match.group(1).strip(),
                    "original_code": original_match.group(2).strip(),
                    "mutated_code": mutated_match.group(2).strip(),
                })
    return survived_mutants

def run_mutation_test(source_code: str, test_code: str, logic_filename: str, test_filename: str) -> Dict[str, Any]:
    print("  -> 使用mutpy运行变异测试...")
    base_dir = "temp_mutation_test_pytest"
    if os.path.exists(base_dir): shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, logic_filename), 'w', encoding='utf-8') as f: f.write(source_code)
    with open(os.path.join(base_dir, test_filename), 'w', encoding='utf-8') as f: f.write(test_code)

    output = ""
    try:
        python_executable = sys.executable
        scripts_dir = os.path.join(os.path.dirname(python_executable), 'Scripts')
        mutpy_script_path = os.path.join(scripts_dir, 'mut.py')
        if not os.path.exists(mutpy_script_path): raise FileNotFoundError(f"找不到 'mut.py'，请确认已正确安装 mutpy: {mutpy_script_path}")

        module_name = os.path.splitext(logic_filename)[0]
        cmd = (f'"{python_executable}" "{mutpy_script_path}" '
               f'--target {module_name} --runner pytest --unit-test {test_filename} --show-mutants')

        env = os.environ.copy(); env['PYTHONUTF8'] = '1'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True, errors="ignore", env=env, cwd=base_dir)
        output = result.stdout + result.stderr
        # print(output)

        if "0 tests passed" in output or "no tests ran" in output:
             raise ValueError("mutpy 成功运行但发现了0个测试。请检查测试代码是否符合 pytest 规范。")

        all_match = re.search(r'all: (\d+)', output)
        if not all_match: raise ValueError(f"mutpy 执行失败或解析输出失败。原始输出:\n{output}")
        
        killed_match = re.search(r'killed: (\d+)', output)
        incompetent_match = re.search(r'incompetent: (\d+)', output)
        total = int(all_match.group(1))
        killed = int(killed_match.group(1)) if killed_match else 0
        incompetent = int(incompetent_match.group(1)) if incompetent_match else 0
        effective_mutants = total - incompetent
        score = killed / effective_mutants if effective_mutants > 0 else 1.0
        
        survived_details = parse_survived_mutants(output)

        print(f"  -> Mutation Test Score: {score:.2%}")
        return {"mutation_score": score, "survived_count": len(survived_details), "survived_details": survived_details, "details": output}
    except Exception as e:
        return {"mutation_score": 0.0, "error": f"发生异常: {e}. 原始输出: \n{output}"}
    finally:
        if os.path.exists(base_dir): shutil.rmtree(base_dir)

# ==================================================================
#                       example usage
# ==================================================================
if __name__ == "__main__":
    
    # 1. 准备业务逻辑代码
    logic_code_to_test = textwrap.dedent("""
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
    """)

    # 2. 准备一个“不够强”的 pytest 测试用例
    # 这个用例故意遗漏了对浮点数除法和小数情况的测试
    weak_pytest_code = textwrap.dedent("""
    import pytest
    from logic_module import calculate

    def test_addition():
        assert calculate(2, 3, 'add') == 5

    def test_subtraction():
        assert calculate(10, 5, 'subtract') == 5
    
    def test_division_integer():
        # 这个测试用例无法区分 a/b 和 a//b
        assert calculate(10, 2, 'divide') == 5
        
    def test_invalid_type_single():
        # 这个测试用例无法区分 or 和 and 的逻辑错误
        with pytest.raises(TypeError):
            calculate('a', 5, 'add')
    """)

    print("="*60)
    print(">>> 场景一：使用“弱”测试用例进行变异测试 <<<")
    print("="*60)
    
    # 3. 运行变异测试
    result_weak = run_mutation_test(
        source_code=logic_code_to_test,
        test_code=weak_pytest_code,
        logic_filename="logic_module.py",
        test_filename="test_script.py"
    )

    #