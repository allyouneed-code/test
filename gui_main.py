# gui_main.py
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import threading
import sys
import os

# 导入 main.py 中的函数
from main import run_workflow, read_requirements_from_docx, read_file_content

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

class WorkflowGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("自动化测试生成工具")
        self.root.geometry("800x650")

        self.req_file_path = tk.StringVar()
        self.code_file_path = tk.StringVar()
        self.target_name = tk.StringVar()

        # --- 使用 Grid 布局的容器 Frame ---
        # Grid 是解决对齐问题最稳健的方法：
        # 第0列放标签，第1列放输入框，第2列放按钮
        input_frame = tk.Frame(root, pady=15, padx=15)
        input_frame.pack(fill="x")

        # 配置列权重，让中间的输入框(第1列)自动拉伸填满空间
        input_frame.grid_columnconfigure(1, weight=1)

        # --- Row 0: 被测件名称 ---
        # 移除加粗，使用默认字体或统一字体
        tk.Label(input_frame, text="被测件名称:", anchor="w").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(input_frame, textvariable=self.target_name).grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        # 第一行没有按钮，所以不需要 column=2

        # --- Row 1: 需求文件 ---
        tk.Label(input_frame, text="需求文件 (.docx):", anchor="w").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(input_frame, textvariable=self.req_file_path).grid(row=1, column=1, sticky="ew", pady=5, padx=5)
        tk.Button(input_frame, text="浏览...", command=self.load_req_file).grid(row=1, column=2, sticky="e", pady=5)

        # --- Row 2: 源代码文件 ---
        tk.Label(input_frame, text="源代码文件 (.py):", anchor="w").grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(input_frame, textvariable=self.code_file_path).grid(row=2, column=1, sticky="ew", pady=5, padx=5)
        tk.Button(input_frame, text="浏览...", command=self.load_code_file).grid(row=2, column=2, sticky="e", pady=5)

        # --- 4. 运行按钮 ---
        self.run_button = tk.Button(root, text="开始生成测试", command=self.start_analysis_thread, 
                                    font=("Arial", 12, "bold"), bg="#4CAF50", fg="white")
        self.run_button.pack(pady=10, fill="x", padx=20)

        # --- 5. 输出控制台 ---
        tk.Label(root, text="执行日志:").pack(anchor="w", padx=10)
        self.console_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, state="disabled", height=20)
        self.console_output.pack(pady=5, padx=10, fill="both", expand=True)

        self.stdout_redirector = TextRedirector(self.console_output)
        
    def load_req_file(self):
        path = filedialog.askopenfilename(filetypes=[("Word 文档", "*.docx")])
        if path: self.req_file_path.set(path)

    def load_code_file(self):
        path = filedialog.askopenfilename(filetypes=[("Python 文件", "*.py")])
        if path: self.code_file_path.set(path)

    def start_analysis_thread(self):
        req_path = self.req_file_path.get()
        code_path = self.code_file_path.get()
        target_name_val = self.target_name.get().strip()

        if not target_name_val:
            messagebox.showwarning("提示", "请输入被测件名称 (用于生成报告标题)。")
            return
        if not req_path or not code_path:
            messagebox.showerror("错误", "请选择需求文件和源代码文件。")
            return

        # UI 交互锁定
        self.run_button.config(text="正在运行...", state="disabled")
        self.console_output.config(state="normal")
        self.console_output.delete(1.0, tk.END)
        self.console_output.config(state="disabled")

        # 重定向输出
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector

        # 启动后台线程
        thread = threading.Thread(
            target=self.run_workflow_in_background, 
            args=(code_path, req_path, target_name_val),
            daemon=True
        )
        thread.start()

    def run_workflow_in_background(self, code_path, req_path, target_name):
        """后台执行逻辑"""
        try:
            print(f"正在读取文件...")
            req_text = read_requirements_from_docx(req_path)
            code_text = read_file_content(code_path)

            # 调用核心逻辑，传入文件名和被测件名称
            run_workflow(
                code_text=code_text, 
                requirement_text=req_text,
                req_filename=os.path.basename(req_path),
                code_filename=os.path.basename(code_path),
                target_name=target_name
            )

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.root.after(0, self.on_workflow_complete)

    def on_workflow_complete(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.run_button.config(text="开始生成测试", state="normal")
        messagebox.showinfo("完成", "测试报告生成完毕！")


class TextRedirector:
    def __init__(self, widget):
        self.widget = widget
    def write(self, text):
        def update():
            self.widget.config(state="normal")
            self.widget.insert(tk.END, text)
            self.widget.see(tk.END)
            self.widget.config(state="disabled")
        self.widget.master.after(0, update)
    def flush(self): pass

if __name__ == "__main__":
    root = tk.Tk()
    app = WorkflowGUI(root)
    root.mainloop()