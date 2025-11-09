import os
import json
import ast
from staticfg import CFGBuilder
import astunparse
from typing import List, Dict, Any, Optional

class CodeAnalyzer:
    """
    一个面向对象的封装器，用于解析 Python 源代码，
    并提取一个符合 M_code = (A, S, G, P) 规范的代码模型。

    当前实现：
    - G (图形表示): 完全实现 (使用 staticfg)
    - P (谓词约束): 完全实现 (嵌入 G.Edges)
    - A (被分析单元): 仅存根 (Stub)
    - S (静态接口): 仅存根 (Stub)
    """
# --- 内部类：用于AST遍历 ---
    class _CallVisitor(ast.NodeVisitor):
        """一个简单的访问者，用于收集函数体内的所有调用。"""
        def __init__(self):
            self.calls = []
        
        def visit_Call(self, node):
            # 尝试获取一个可读的函数名 (e.g., 'print', 'os.path.exists')
            try:
                func_name = ast.unparse(node.func)
            except: 
                func_name = "complex_call"
            
            # 简化：只收集函数名，不合并行号
            self.calls.append(func_name)
            self.generic_visit(node)

    def __init__(self, source_code_str: str, filename: str = "source_code_string"):
        """
        初始化提取器。
        
        参数:
            source_code_str (str): 要分析的原始 Python 源代码字符串。
        """
        # 移除了所有 try-except
        if not isinstance(source_code_str, str) or not source_code_str.strip():
            print("错误：source_code_str 必须是一个非空的字符串。")
            exit()
            
        self.source_code = source_code_str
        self.filename = filename
        # cfg_blocks 将存储来自 staticfg 的原始块
        self.cfg_blocks = []
        # model 将存储最终的 M_code 字典
        self.ast_tree = None
        self.primary_function_node = None # 存储找到的第一个 FunctionDef
        self.external_calls = []
        self.model = {
            "A": {},
            "S": {},
            "G": {}
        }
        self.ast_tree = ast.parse(self.source_code)
            # 立即分析 AST 以填充 A 和 S 的数据
        self._find_primary_function()
        self._find_external_calls()

    def process(self) -> dict:
        """
        处理源代码并返回完整的代码模型。
        这是一个“协调器”方法，按顺序调用所有私有构建器。

        返回:
            dict: 包含 A, S, G 模型的完整字典。
        """
        print("（1/3）正在运行 staticfg 解析器...")
        self._run_staticfg()
        
        self._filter_zombie_block()
        
        print("（2/3）正在构建 G (CFG) 和 P (谓词) 模型...")
        self._build_model_G_and_P()
        
        print("（3/3）正在构建 A 和 S 存根模型...")
        self._build_model_A() # 构建 A 的存根
        self._build_model_S() # 构建 S 的存根
        
        print("--- 模型提取完成 ---")
        return self.model

    # --- 私有方法 (Private Methods) ---
    def _find_primary_function(self):
        """遍历 AST，找到第一个函数定义并存储它。"""
        if not self.ast_tree:
            return
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef):
                self.primary_function_node = node
                return # 找到第一个函数，停止

    def _find_external_calls(self):
        """使用 _CallVisitor 遍历主函数体，收集外部调用。"""
        if not self.primary_function_node:
            return # 没有函数可供分析
        
        visitor = self._CallVisitor()
        # 只访问主函数体，而不是整个文件
        visitor.visit(self.primary_function_node) 
        
        # 去重并格式化
        unique_calls = set(visitor.calls)
        self.external_calls = [
            {"target_signature": call_name, "call_locations": []} # 位置信息被简化
            for call_name in unique_calls
        ]


    def _run_staticfg(self):
        """
        步骤 1: 运行 staticfg 库并填充 self.cfg_blocks。
        """
        builder = CFGBuilder()
        cfg = builder.build_from_src(
            name='source_cfg',
            src=self.source_code
        )
        self.cfg_blocks = list(cfg)

    def _filter_zombie_block(self):
        """
        步骤 2: 移除 staticfg 产生的、代表 'def' 函数定义的“僵尸块”。
        """
        if self.cfg_blocks:
            first_block_statements = self.cfg_blocks[0].statements
            if first_block_statements:
                first_statement = astunparse.unparse(first_block_statements[0]).strip()
                if first_statement.startswith('def '):
                    self.cfg_blocks.pop(0) # 移除第一个元素

    def _build_model_G_and_P(self):
        """
        步骤 3: 遍历已清理的 cfg_blocks，构建 G 和 P 模型。
        P (谓词) 被嵌入 G.Edges 中。
        """
        nodes = []
        edges = []
        entry = None
        exit_points = []

        # 填充 G.Nodes
        for block in self.cfg_blocks:
            statements_str = [astunparse.unparse(stmt).strip() for stmt in block.statements]
            nodes.append({
                "id": f"B{block.id}",
                "statements": statements_str
            })

        # 填充 G.Edges (和 P)
        for block in self.cfg_blocks:
            if not block.exits:
                continue
            for edge in block.exits:
                predicate_str = "None"
                if edge.exitcase is not None:
                    predicate_str = astunparse.unparse(edge.exitcase).strip()
                
                edges.append({
                    "from_node_id": f"B{edge.source.id}",
                    "to_node_id": f"B{edge.target.id}",
                    "predicate": predicate_str  # P (谓词)
                })

        # 填充 G.Entry
        if self.cfg_blocks:
            entry = f"B{self.cfg_blocks[0].id}"

        # 填充 G.Exit_Points
        for block in self.cfg_blocks:
            if not block.exits:
                exit_points.append({
                    "node_id": f"B{block.id}",
                    "exit_type": "Return" # 假设无出边即为返回
                })
        
        self.model["G"] = {
            "Nodes": nodes,
            "Edges": edges,
            "Entry": entry,
            "Exit_Points": exit_points
        }

    def _build_model_A(self):
        """
        步骤 4a: 构建 A (被分析单元) 模型。
        (存根实现)
        """
        if self.primary_function_node:
            self.model["A"] = {
                "Id": self.primary_function_node.name,
                "Type": "Function",
                "Location": self.filename
            }
        else:
            self.model["A"] = {"Id": "Not_Found", "Type": "Unknown", "Location": self.filename}

    def _build_model_S(self):
        """
        步骤 4b: 构建 S (静态接口) 模型。
        (存根实现 - 这需要完整的 AST 解析)
        """
        if not self.primary_function_node:
            self.model["S"] = {"Arg_in": [], "T_out": "Unknown", "C_ext": []}
            return

        func_node = self.primary_function_node
        
        # 1. 解析参数 (Arg_in)
        arg_list = []
        args = func_node.args.args
        defaults = func_node.args.defaults
        defaults_offset = len(args) - len(defaults)
        
        for i, arg in enumerate(args):
            arg_name = arg.arg
            static_type = ast.unparse(arg.annotation) if arg.annotation else "Any"
            
            default_val = None
            if i >= defaults_offset:
                default_ast = defaults[i - defaults_offset]
                try:
                    # 尝试获取常量值 (e.g., 5, "hello", None)
                    default_val = ast.literal_eval(default_ast)
                except (ValueError, TypeError, SyntaxError):
                    # 如果不是简单常量 (e.g., a function call), 则unparse
                    try:
                        default_val = ast.unparse(default_ast)
                    except:
                        default_val = "ComplexDefaultValue"
                        
            arg_list.append({
                "name": arg_name, 
                "static_type": static_type, 
                "default_value": default_val
            })
        
        # 2. 解析返回类型 (T_out)
        return_type = ast.unparse(func_node.returns) if func_node.returns else "Any"
        
        # 3. 填充模型
        self.model["S"] = {
            "Arg_in": arg_list,
            "T_out": return_type,
            "C_ext": self.external_calls # 使用 __init__ 中收集的结果
        }

class CodeUnitDecomposer:
    """
    一个代码单元分解器。
    
    它负责解析一个完整的 Python 源代码文件（字符串），
    并将其拆分为一个“被测单元”列表，其中每个单元都是一个
    独立的函数（FunctionDef）或类方法（ClassDef 内的 FunctionDef）。
    
    这个类的输出是 CodeAnalyzer 的理想输入。
    """

    class _FunctionVisitor(ast.NodeVisitor):
        """
        一个AST访问者，用于查找并提取所有函数和方法定义。
        """
        def __init__(self):
            # 存储提取的单元
            # "name": 函数名
            # "class_context": 如果是方法，则为类名；否则为 None
            # "code": 该函数的完整源代码字符串
            self.units: List[Dict[str, Any]] = []
            self.current_class_name: Optional[str] = None

        def visit_ClassDef(self, node: ast.ClassDef):
            """
            当我们访问一个类时，记录下当前类的名称，
            然后继续访问该类的子节点（即方法）。
            """
            self.current_class_name = node.name
            # 遍历类的主体以查找方法
            self.generic_visit(node)
            # 离开类后，重置类名
            self.current_class_name = None

        def visit_FunctionDef(self, node: ast.FunctionDef):
            """
            当我们访问一个函数（或方法）时，将其提取出来。
            """
            try:
                # 使用 ast.unparse 将AST节点转换回源代码字符串
                # 这是自 Python 3.9+ 以来的标准方法
                code_str = ast.unparse(node)
            except AttributeError:
                # 备用方案，以防 unparse 不可用 (尽管它应该是)
                code_str = "# [Error] 无法 unparse 此函数"
            
            unit = {
                "name": node.name,
                "class_context": self.current_class_name,
                "code": code_str,
                "ast_node": node  # 存储原始节点以供将来使用
            }
            self.units.append(unit)
            
            # **重要**:
            # 我们在这里故意 *不* 调用 self.generic_visit(node)。
            # 这可以防止它递归地查找并提取 *嵌套函数*。
            # 对于单元测试，我们通常只关心最外层的函数/方法
            # 作为“被测单元”。嵌套函数被视为其父函数的一部分。

    def __init__(self, source_code_str: str):
        """
        初始化分解器。
        
        参数:
            source_code_str (str): 要分析的完整 Python 源代码。
        """
        self.source_code = source_code_str
        self.ast_tree = None
        
        try:
            self.ast_tree = ast.parse(self.source_code)
        except SyntaxError as e:
            print(f"代码分解器错误：源代码存在语法错误，无法解析。 {e}")
            raise

    def decompose(self) -> List[Dict[str, Any]]:
        """
        执行分解，返回所有找到的单元。

        返回:
            一个字典列表，每个字典代表一个函数或方法。
        """
        if not self.ast_tree:
            return []
            
        visitor = self._FunctionVisitor()
        visitor.visit(self.ast_tree)
        return visitor.units

if __name__ == "__main__":
    
    # 6. 更新示例代码，使其自包含
    sample_code = """
import os # 外部导入

def calculate(a: int, b: int, operation: str = 'add'):
    '''一个简单的计算器'''
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Inputs must be numeric")
    
    print(f"Calculating {a} {operation} {b}") # 外部调用
    
    if operation == 'add':
        return a + b
    if operation == 'subtract':
        return a - b
    if operation == 'divide':
        if b == 0:
            return "Error: Division by zero"
        return a / b
    return None
"""

    # (核心) 初始化对象并处理
    # 传入 "calculate_example.py" 作为虚拟文件名
    extractor = CodeAnalyzer(sample_code, filename="calculate_example.py")
    final_model = extractor.process()

    # 4. 打印最终的、精简的 JSON
    print("\n--- 最终 M_code 模型 (JSON) ---")
    print(json.dumps(final_model, indent=2, ensure_ascii=False))