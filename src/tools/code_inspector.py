import os
import json
from staticfg import CFGBuilder
import astunparse

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

    def __init__(self, source_code_str: str):
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
        # cfg_blocks 将存储来自 staticfg 的原始块
        self.cfg_blocks = []
        # model 将存储最终的 M_code 字典
        self.model = {
            "A": {},
            "S": {},
            "G": {}
        }

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
        self.model["A"] = {
            "Id": "check_age",  # 假设值
            "Type": "Function", # 假设值
            "Location": "my_function.py" # 假设值
        }

    def _build_model_S(self):
        """
        步骤 4b: 构建 S (静态接口) 模型。
        (存根实现 - 这需要完整的 AST 解析)
        """
        self.model["S"] = {
            "Arg_in": [
                {"name": "age", "static_type": "Any", "default_value": None} # 假设值
            ],
            "T_out": "String", # 假设值
            "C_ext": [
                {"target_signature": "print", "call_locations": []} # 假设值
            ]
        }


if __name__ == "__main__":
    
    # 1. 定义源文件路径 (假设 my_function.py 在上一级目录)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TARGET_FILE = os.path.join(SCRIPT_DIR, '..', 'my_function.py')

    if not os.path.exists(TARGET_FILE):
        print(f"错误：在 {TARGET_FILE} 未找到目标文件 my_function.py")
        exit()

    # 2. 读取源代码
    print(f"正在从 {TARGET_FILE} 读取源文件...")
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        code_to_analyze = f.read()

    # 3. (核心) 初始化对象并处理
    extractor = CodeAnalyzer(code_to_analyze)
    final_model = extractor.process()

    # 4. 打印最终的、精简的 JSON
    print("\n--- 最终 M_code 模型 (JSON) ---")
    print(json.dumps(final_model, indent=2, ensure_ascii=False))