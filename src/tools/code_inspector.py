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

    更新支持：
    - 既支持独立函数 (Function-Level)
    - 也支持类定义 (Class-Level)
    """
    
    class _CallVisitor(ast.NodeVisitor):
        """一个简单的访问者，用于收集作用域内的所有调用。"""
        def __init__(self):
            self.calls = []
        
        def visit_Call(self, node):
            try:
                func_name = ast.unparse(node.func)
            except: 
                func_name = "complex_call"
            self.calls.append(func_name)
            self.generic_visit(node)

    def __init__(self, source_code_str: str, filename: str = "source_code_string", target_name: str = None):
        if not isinstance(source_code_str, str) or not source_code_str.strip():
            print("错误：source_code_str 必须是一个非空的字符串。")
            # 实际项目中建议 raise ValueError 而不是 exit()
            return 
            
        self.source_code = source_code_str
        self.filename = filename
        self.cfg_blocks = []
        self.ast_tree = None
        self.target_name = target_name
        
        # 修改：使用 generic target_node 而不是 primary_function_node
        self.target_node = None 
        self.target_type = "Unknown" # "Class" or "Function"
        
        self.external_calls = []
        self.model = {
            "A": {},
            "S": {},
            "G": {}
        }
        
        try:
            self.ast_tree = ast.parse(self.source_code)
            # 1. 识别目标（类优先，其次是函数）
            self._identify_target_unit()
            # 2. 分析外部调用
            self._find_external_calls()
        except Exception as e:
            print(f"AST 解析失败: {e}")

    def process(self) -> dict:
        print("（1/3）正在运行 staticfg 解析器...")
        self._run_staticfg()
        
        self._filter_zombie_block()
        
        print("（2/3）正在构建 G (CFG) 和 P (谓词) 模型...")
        self._build_model_G_and_P()
        
        print("（3/3）正在构建 A 和 S 存根模型...")
        self._build_model_A() 
        self._build_model_S() 
        
        print("--- 模型提取完成 ---")
        return self.model

    # --- 核心逻辑修改区域 ---

    def _identify_target_unit(self):
        """
        识别分析目标。
        策略：
        1. 如果指定了 target_name，精确查找。
        2. 如果没指定：
           - 优先找“非 dataclass”的类（通常是逻辑类）。
           - 其次找任意类。
           - 最后找函数。
        """
        if not self.ast_tree:
            return

        candidates_class = []
        candidates_func = []

        # 遍历所有节点收集候选者
        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否指定了名称
                if self.target_name and node.name == self.target_name:
                    self.target_node = node
                    self.target_type = "Class"
                    return
                candidates_class.append(node)
            elif isinstance(node, ast.FunctionDef):
                if self.target_name and node.name == self.target_name:
                    self.target_node = node
                    self.target_type = "Function"
                    return
                candidates_func.append(node)

        # 如果指定了名字但没找到
        if self.target_name:
            print(f"警告: 未在代码中找到名为 '{self.target_name}' 的类或函数。")
            return

        # --- 自动推断逻辑 ---
        
        # 1. 优先选择“有逻辑的类” (排除 dataclass)
        for cls_node in candidates_class:
            is_dataclass = False
            for decorator in cls_node.decorator_list:
                # 简单的装饰器检查，检查是否名为 'dataclass'
                if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
                    is_dataclass = True
                elif isinstance(decorator, ast.Call) and getattr(decorator.func, 'id', '') == 'dataclass': # handle @dataclass()
                    is_dataclass = True
            
            if not is_dataclass:
                self.target_node = cls_node
                self.target_type = "Class"
                return

        # 2. 如果都是 dataclass，或者没有普通类，选第一个类
        if candidates_class:
            self.target_node = candidates_class[0]
            self.target_type = "Class"
            return

        # 3. 最后选第一个函数
        if candidates_func:
            self.target_node = candidates_func[0]
            self.target_type = "Function"
            return
        
    def _find_external_calls(self):
        """遍历目标节点（类或函数），收集外部调用。"""
        if not self.target_node:
            return 
        
        visitor = self._CallVisitor()
        visitor.visit(self.target_node) # 如果是类，会自动遍历所有方法
        
        unique_calls = set(visitor.calls)
        self.external_calls = [
            {"target_signature": call_name, "call_locations": []}
            for call_name in unique_calls
        ]

    def _run_staticfg(self):
        try:
            builder = CFGBuilder()
            cfg = builder.build_from_src(name='source_cfg', src=self.source_code)
            self.cfg_blocks = list(cfg)
        except Exception as e:
            print(f"StaticFG 运行警告: {e}")
            self.cfg_blocks = []

    def _filter_zombie_block(self):
        """移除定义头部的僵尸块 (def ... 或 class ...)"""
        if self.cfg_blocks:
            first_block_statements = self.cfg_blocks[0].statements
            if first_block_statements:
                try:
                    first_statement = astunparse.unparse(first_block_statements[0]).strip()
                    # 同时过滤 def 和 class
                    if first_statement.startswith('def ') or first_statement.startswith('class '):
                        self.cfg_blocks.pop(0)
                except:
                    pass

    def _build_model_G_and_P(self):
        # (逻辑保持不变)
        nodes = []
        edges = []
        entry = None
        exit_points = []

        for block in self.cfg_blocks:
            statements_str = [astunparse.unparse(stmt).strip() for stmt in block.statements]
            nodes.append({"id": f"B{block.id}", "statements": statements_str})

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
                    "predicate": predicate_str 
                })

        if self.cfg_blocks:
            entry = f"B{self.cfg_blocks[0].id}"

        for block in self.cfg_blocks:
            if not block.exits:
                exit_points.append({"node_id": f"B{block.id}", "exit_type": "Return"})
        
        self.model["G"] = {
            "Nodes": nodes, "Edges": edges, "Entry": entry, "Exit_Points": exit_points
        }

    def _build_model_A(self):
        """构建单元信息 (Unit Info)"""
        if self.target_node:
            self.model["A"] = {
                "Id": self.target_node.name,
                "Type": self.target_type, # 动态类型：Class 或 Function
                "Location": self.filename
            }
        else:
            self.model["A"] = {"Id": "Not_Found", "Type": "Unknown", "Location": self.filename}

    def _build_model_S(self):
        """
        构建静态接口 (Static Interface)。
        - 如果是 Function: 解析该函数的参数。
        - 如果是 Class: 解析 __init__ 方法的参数。
        """
        if not self.target_node:
            self.model["S"] = {"Arg_in": [], "T_out": "Unknown", "C_ext": []}
            return

        # 1. 确定用于提取参数的节点
        args_node = None
        return_annotation = None

        if self.target_type == "Function":
            args_node = self.target_node.args
            return_annotation = self.target_node.returns
        
        elif self.target_type == "Class":
            # 在类中寻找 __init__
            init_method = None
            for item in self.target_node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    init_method = item
                    break
            
            if init_method:
                args_node = init_method.args
                return_annotation = init_method.returns
            else:
                # 没有 __init__，也就是默认构造函数，无参数
                self.model["S"] = {
                    "Arg_in": [], 
                    "T_out": self.target_node.name, # 返回的是实例类型
                    "C_ext": self.external_calls
                }
                return

        # 2. 解析参数 (通用逻辑)
        arg_list = []
        if args_node:
            args = args_node.args
            defaults = args_node.defaults
            defaults_offset = len(args) - len(defaults)
            
            for i, arg in enumerate(args):
                arg_name = arg.arg
                
                # 关键修改：过滤掉 self
                if arg_name == 'self':
                    continue

                static_type = ast.unparse(arg.annotation) if arg.annotation else "Any"
                
                default_val = None
                if i >= defaults_offset:
                    default_ast = defaults[i - defaults_offset]
                    try:
                        default_val = ast.literal_eval(default_ast)
                    except:
                        try:
                            default_val = ast.unparse(default_ast)
                        except:
                            default_val = "Complex"
                            
                arg_list.append({
                    "name": arg_name, 
                    "static_type": static_type, 
                    "default_value": default_val
                })
        
        # 3. 解析返回类型
        if self.target_type == "Class":
            return_type = self.target_node.name # 类的返回类型是它自己
        else:
            return_type = ast.unparse(return_annotation) if return_annotation else "Any"
        
        self.model["S"] = {
            "Arg_in": arg_list,
            "T_out": return_type,
            "C_ext": self.external_calls
        }

class CodeUnitDecomposer:
    """
    代码单元分解器 (优化版)。
    
    策略：
    1. 顶层函数 -> 作为一个独立单元。
    2. 类 -> 作为一个独立单元 (包含其所有方法)。
       注意：我们不再把类的方法拆出来单独作为单元，因为方法通常依赖类的上下文。
    """

    class _UnitVisitor(ast.NodeVisitor):
        def __init__(self):
            self.units: List[Dict[str, Any]] = []

        def visit_FunctionDef(self, node: ast.FunctionDef):
            """处理顶层函数"""
            # 只有当函数是顶层函数时才提取（不属于任何类）
            # 在 AST 遍历中，visit_ClassDef 会拦截类内部的方法
            self._add_unit(node, "Function", node.name)

        def visit_ClassDef(self, node: ast.ClassDef):
            """处理类"""
            # 将整个类作为一个单元提取
            self._add_unit(node, "Class", node.name)
            # 注意：这里我们 *不* 调用 generic_visit，
            # 这意味着我们不会进入类内部去把方法单独拆出来。
            # 这种策略保证了“类”的完整性。

        def _add_unit(self, node, unit_type, name):
            try:
                code_str = ast.unparse(node)
            except AttributeError:
                code_str = ""
            
            self.units.append({
                "name": name,
                "type": unit_type, # "Class" or "Function"
                "code": code_str,  # 完整的类代码 或 完整的函数代码
                "ast_node": node
            })

    def __init__(self, source_code_str: str):
        self.source_code = source_code_str
        try:
            self.ast_tree = ast.parse(self.source_code)
        except SyntaxError as e:
            print(f"语法错误: {e}")
            self.ast_tree = None

    def decompose(self) -> List[Dict[str, Any]]:
        if not self.ast_tree:
            return []
        visitor = self._UnitVisitor()
        visitor.visit(self.ast_tree)
        return visitor.units
    
if __name__ == "__main__":
    # 测试用例：C语言转过来的 Python 类
    sample_class_code = """
import math
from dataclasses import dataclass

@dataclass
class MpuData:
    accX: float
    accY: float
    accZ: float

class AHRSSolver:
    def __init__(self, kp: float = 0.5):
        self.q0 = 1.0
        self.Kp = kp

    def GetAngle(self, pMpu: MpuData, dt: float) -> tuple:
        norm = math.sqrt(pMpu.accX**2 + pMpu.accY**2)
        return (0.0, 0.0, 0.0)
"""
    sample_class_code2 = """
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

    print("--- 分析 Class ---")
    extractor = CodeAnalyzer(sample_class_code2, filename="ahrs.py")
    model = extractor.process()
    print(json.dumps(model, indent=2, ensure_ascii=False))