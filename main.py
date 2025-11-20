# main.py
import os
import sys
import time
import argparse
from docx import Document

from src.config import app_config
from src.workflows.test_gen_workflow import TestGenerationWorkflow
from src.report import WorkflowReporter, DocxWorkflowReporter


# --- 导入项目模块 ---
# 确保 src 目录在 Python 路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)



def read_file_content(filepath: str) -> str:
    """读取普通文本文件 (.py)"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def read_requirements_from_docx(filepath: str) -> str:
    """
    读取 Word 文档 (.docx) 中的所有文本，包括段落和表格内容。
    针对表格数据，会将每一行的数据用 " | " 连接，保持键值对的上下文关系。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到文件: {filepath}")
    
    if not filepath.endswith('.docx'):
        raise ValueError("需求文件必须是 .docx 格式")

    try:
        doc = Document(filepath)
        full_text = []

        # 1. 读取文档正文段落 (如果有的话)
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())

        # 2. 读取文档中的表格 (这是您需求文档的核心部分)
        for table in doc.tables:
            for row in table.rows:
                row_cells = []
                for cell in row.cells:
                    # 获取单元格文本并去除多余空白
                    cell_text = cell.text.strip()
                    # 只有非空单元格才加入
                    if cell_text: 
                        # 处理单元格内可能有换行的情况，替换为空格以免打断结构
                        clean_text = cell_text.replace('\n', ' ')
                        row_cells.append(clean_text)
                
                # 将同一行的单元格用 " | " 连接
                # 例如： "关键函数 | GetAngle"
                # 这样 LLM 能够理解它们之间的键值对应关系
                if row_cells:
                    full_text.append(" | ".join(row_cells))

        return "\n".join(full_text)

    except Exception as e:
        raise RuntimeError(f"读取 Word 文件失败: {e}")

def run_workflow(code_text: str, requirement_text: str):
    """
    运行测试生成工作流并打印报告。
    (此函数封装了 main() 中的 try...except 逻辑)
    """
    try:
        # 3. 更新配置 (使用默认文件名，因为我们是动态传入内容)
        # app_config["logic_filename"] = "logic_module.py" # 已在 config.py 中设置
        # app_config["test_filename"] = "test_script.py"

        # 4. 初始化工作流
        print("正在构建工作流图...")
        workflow_builder = TestGenerationWorkflow(config=app_config)
        app = workflow_builder.build()

        # 5. 构建初始状态 (与 main.py 相同)
        initial_state = {
            "code": code_text, 
            "requirement": requirement_text, 
            "retry_count": 0,
            "start_time": time.time(), 
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0, 
            "total_tokens": 0,
            "iteration_history": [],
            "execution_feedback": "",
            "mutation_test_has_error": False, 
            "mutation_error_details": "",
            "test_failures": 0,
            "test_errors": 0,
            "evaluation_result": "NOT_STARTED",
            "analysis_report": "",
            "structured_requirement": "",
            "generation_prompt": "",
            "test_code": "",
            "analysis_model": None,
            "requirement_model": None,
            "validation_report": ""
        }

        # 6. 执行工作流
        print("\n--- 开始执行 ---")
        final_state = app.invoke(initial_state)

        # 7. 生成最终报告
        print("\n--- 生成报告 ---")
        reporter = WorkflowReporter(final_state, app_config)
        reporter.generate_report()

        # 8. 生成 Word 报告 (Docx Report)
        print("\n--- 生成 Word 报告 ---")
        output_docx = f"Test_Report_{int(time.time())}.docx"
        docx_reporter = DocxWorkflowReporter(final_state, output_filename=output_docx)
        docx_reporter.generate()
        
        print(f"\n✅ 工作流完成！报告已保存到 {output_docx}")

    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="基于 LLM 的自动化单元测试生成工具")
    parser.add_argument("req_file", help="需求文档路径 (.docx)")
    parser.add_argument("code_file", help="待测试源代码路径 (.py)")
    parser.add_argument("--logic_filename", default="logic_module.py", help="执行器使用的逻辑文件名 (默认: logic_module.py)")
    parser.add_argument("--test_filename", default="test_script.py", help="执行器使用的测试文件名 (默认: test_script.py)")
    
    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀  启动自动化测试生成工作流")
    print("="*60)

    try:
        # 2. 读取输入文件
        print(f"正在读取需求文件: {args.req_file} ...")
        requirement_text = read_requirements_from_docx(args.req_file)
        
        print(f"正在读取代码文件: {args.code_file} ...")
        code_text = read_file_content(args.code_file)
        
        # --- 调用新的核心函数 ---
        run_workflow(code_text, requirement_text)

    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()