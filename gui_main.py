# gui_main.py
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading
import sys
import os

# 导入您在 main.py 中重构的函数和原始帮助函数
from main import run_workflow, read_requirements_from_docx, read_file_content

# 确保 src 目录在 Python 路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

class WorkflowGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("自动化测试生成工具")
        self.root.geometry("800x600")

        self.req_file_path = tk.StringVar()
        self.code_file_path = tk.StringVar()

        # --- 1. 文件选择 ---
        frame_files = tk.Frame(root, pady=10)
        frame_files.pack(fill="x")

        tk.Label(frame_files, text="需求文件 (.docx):").pack(side=tk.LEFT, padx=5)
        tk.Entry(frame_files, textvariable=self.req_file_path, width=50).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(frame_files, text="浏览...", command=self.load_req_file).pack(side=tk.LEFT, padx=5)

        frame_code = tk.Frame(root, pady=5)
        frame_code.pack(fill="x")

        tk.Label(frame_code, text="源代码文件 (.py):").pack(side=tk.LEFT, padx=5)
        tk.Entry(frame_code, textvariable=self.code_file_path, width=50).pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(frame_code, text="浏览...", command=self.load_code_file).pack(side=tk.LEFT, padx=5)

        # --- 2. 运行按钮 ---
        self.run_button = tk.Button(root, text="开始分析", command=self.start_analysis_thread, font=("Arial", 12, "bold"), bg="green", fg="white")
        self.run_button.pack(pady=10, fill="x", padx=10)

        # --- 3. 输出控制台 ---
        tk.Label(root, text="工作流输出:").pack(anchor="w", padx=10)
        self.console_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", height=25)
        self.console_output.pack(pady=10, padx=10, fill="both", expand=True)

        # --- 4. 重定向 stdout ---
        self.stdout_redirector = TextRedirector(self.console_output)
        
    def load_req_file(self):
        path = filedialog.askopenfilename(filetypes=[("Word 文档", "*.docx")])
        if path:
            self.req_file_path.set(path)

    def load_code_file(self):
        path = filedialog.askopenfilename(filetypes=[("Python 文件", "*.py")])
        if path:
            self.code_file_path.set(path)

    def start_analysis_thread(self):
        req_path = self.req_file_path.get()
        code_path = self.code_file_path.get()

        if not req_path or not code_path:
            messagebox.showerror("错误", "请同时选择需求文件和源代码文件。")
            return

        # 禁用按钮，防止重复点击
        self.run_button.config(text="正在运行... 请稍候 ...", state="disabled")
        
        # 清空控制台
        self.console_output.config(state="normal")
        self.console_output.delete(1.0, tk.END)
        self.console_output.config(state="disabled")

        # 将 stdout 重定向到我们的文本框
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector

        # 在新线程中运行工作流，以防GUI冻结
        thread = threading.Thread(
            target=self.run_workflow_in_background, 
            args=(code_path, req_path),
            daemon=True
        )
        thread.start()

    def run_workflow_in_background(self, code_path, req_path):
        """这个函数在后台线程中运行"""
        try:
            # 1. 从文件读取内容
            print(f"读取需求: {req_path}")
            req_text = read_requirements_from_docx(req_path)
            
            print(f"读取代码: {code_path}")
            code_text = read_file_content(code_path)

            # 2. 调用重构后的核心函数
            run_workflow(code_text, req_text)

        except Exception as e:
            # 确保即使线程崩溃，错误也会被打印到GUI
            print(f"\n❌ GUI 线程中发生严重错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 恢复 stdout 并重新启用按钮
            # 我们必须使用 root.after 确保这些 GUI 操作在主线程中执行
            self.root.after(0, self.on_workflow_complete)

    def on_workflow_complete(self):
        """在主线程中恢复GUI状态"""
        sys.stdout = sys.__stdout__  # 恢复标准输出
        sys.stderr = sys.__stderr__
        self.run_button.config(text="开始分析", state="normal")
        messagebox.showinfo("完成", "工作流执行完毕！")


class TextRedirector:
    """一个辅助类，用于将 print 语句重定向到 Tkinter 文本框"""
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        # 确保 GUI 更新在主线程中
        def update_gui():
            self.widget.config(state="normal")
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END) # 自动滚动到底部
            self.widget.config(state="disabled")
        
        # 使用 root.after 将 GUI 更新调度到主线程
        self.widget.master.after(0, update_gui)

    def flush(self):
        pass # Pytest/Coverage 可能需要这个


if __name__ == "__main__":
    # 确保 src 目录在路径中
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, 'src'))

    root = tk.Tk()
    app = WorkflowGUI(root)
    root.mainloop()