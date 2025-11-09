# 假设文件路径位于: src/tools/static_validator.py

from typing import Dict, List, Set, Any, Optional,Tuple
import re
from pydantic import BaseModel, Field
import json

# ------------------------------------------------------------------
# 1. 导入依赖的模型
# 假设 requirement_analyzer.py 和 code_inspector.py 位于同级
# （如果它们在不同的模块中，你需要调整这些导入路径）
# ------------------------------------------------------------------

from ..llm.client import BaseChatModel
from .requirement_analyzer import FullTestModel

class MatchPair(BaseModel):
    """一个语义等价的匹配对"""
    requirement_nl: str = Field(description="来自需求列表（自然语言）的谓词")
    code_predicate: str = Field(description="来自代码列表（Python 表达式）的谓词")

class BatchValidationReport(BaseModel):
    """LLM 必须返回的完整匹配报告"""
    matches: List[MatchPair] = Field(description="所有语义上等价的匹配对")
    unmatched_reqs: List[str] = Field(description="需求列表中未找到任何匹配的谓词")
    unmatched_code: List[str] = Field(description="代码列表中未找到任何匹配的谓词")

class StaticDifferentialValidator:
    """
    静态差分验证器。

    负责对比 M_req (需求模型) 和 M_code (代码模型)，
    以识别两者之间的一致性、差距和冗余。
    """

    def __init__(self, m_req: FullTestModel, m_code: Dict[str, Any], llm_client: BaseChatModel):
        """
        初始化验证器。

        参数:
            m_req: 来自 RequirementAnalyzer 的完整需求模型。
            m_code: 来自 CodeAnalyzer.process() 的完整代码模型字典。
            llm_client: 一个 LangChain 兼容的 LLM 客户端实例。
        """
        self.m_req = m_req
        self.m_code = m_code
        self.llm_client = llm_client

        self.llm_cache: Dict[tuple, bool] = {}
        
        # 报告结构
        self.report = {
            "matches": [],              # 一致的项
            "gaps_req_to_code": [],     # 需求已定义，但代码未实现
            "gaps_code_to_req": [],     # 代码已实现，但需求未定义
            "errors": []                # 验证过程中的错误
        }
        
        # 用于存储已规范化的谓词，以便于比较
        self.code_predicates: Set[str] = set()
        self.req_conditions: Set[str] = set()

    def validate(self) -> Dict[str, List[str]]:
        """
        执行完整的验证流程。
        """
        print("--- 启动静态差分验证 ---")
        
        # 步骤 1: 验证接口 (M_req.I vs M_code.S)
        print("  (1/2) 正在验证接口...")
        self._validate_interface()

        # 步骤 3: 验证行为 (M_req.B vs M_code.G/P)
        print("  (2/2) 正在验证行为逻辑...")
        self._validate_behavior()

        print("--- 验证完成 ---")
        return self.report

    # --- 检查点 1: 接口验证 ---
    def _validate_interface(self):
        """
        检查点 1: 验证 M_req.I (接口) 和 M_code.S (静态接口)。
        """
        try:
            # 1a: 验证单元ID
            req_id = self.m_req.unit_under_test.identifier
            code_id = self.m_code['A']['Id']
            if req_id == code_id:
                self.report['matches'].append(f"单元ID一致: {req_id}")
            else:
                self.report['gaps_req_to_code'].append(f"ID不匹配: 需求为 '{req_id}', 代码为 '{code_id}'")

            # 1b: 验证参数 (仅检查名称)
            req_params = {p.name for p in self.m_req.interface_specification.input_parameters}
            code_params = {p['name'] for p in self.m_code['S']['Arg_in']}
            
            if req_params == code_params:
                self.report['matches'].append(f"参数列表一致: {req_params}")
            else:
                missing_in_code = req_params - code_params
                missing_in_req = code_params - req_params
                if missing_in_code:
                    self.report['gaps_req_to_code'].append(f"代码中缺失参数: {missing_in_code}")
                if missing_in_req:
                    self.report['gaps_code_to_req'].append(f"需求中缺失参数: {missing_in_req}")

            # (未来扩展: 在此处添加对 M_req.I.external_dependencies 和 M_code.S.C_ext 的验证)

        except KeyError as e:
            self.report['errors'].append(f"接口验证失败：模型中缺少键 {e}")
        except Exception as e:
            self.report['errors'].append(f"接口验证时发生未知错误: {e}")

    # --- 检查点 2: 行为验证 (准备) ---
    def _extract_predicates(self) -> Tuple[Set[str], Set[str]]:
        """从两个模型中提取原始的谓词字符串集合。"""
        req_conditions = set()
        code_predicates = set()
        try:
            for scenario in self.m_req.behavioral_model.functional_scenarios:
                req_conditions.update(scenario.pre_conditions)
            for scenario in self.m_req.behavioral_model.error_scenarios:
                req_conditions.update(scenario.error_conditions)
            
            for edge in self.m_code['G']['Edges']:
                if edge['predicate'] != "None":
                    code_predicates.add(edge['predicate'])
        except KeyError as e:
            self.report['errors'].append(f"谓词提取失败：模型中缺少键 {e}")
        return req_conditions, code_predicates

    # --- 检查点 3: 行为验证 (执行) ---
    def _build_batch_validation_prompt(self, req_list: List[str], code_list: List[str]) -> str:
        """
        构建 O(1) 批处理 Prompt。
        """
        # 将列表转换为易于阅读的 JSON 字符串
        req_json = json.dumps(req_list, indent=2)
        code_json = json.dumps(code_list, indent=2)

        prompt = f"""
        你是一个精通 Python 语言和软件需求的逻辑匹配专家。
        你的任务是比较“需求列表（自然语言）”和“代码列表（Python 表达式）”，
        找出它们之间所有语义等价的匹配项。

        [需求列表 (List A - 自然语言)]
        {req_json}

        [代码列表 (List B - Python 表达式)]
        {code_json}

        [任务]
        请执行以下匹配逻辑：
        1.  对于 List A 中的**每一项**，在 List B 中寻找一个语义上**完全等价**的项。
        2.  一个项只能被匹配一次。
        3.  
        [示例匹配规则]
        - "operation is 'add'" (List A) 等价于 "operation == 'add'" (List B)。
        - "a is not a number" (List A) 等价于 "not isinstance(a, (int, float))" (List B)。
        - "operation is 'divide' and b is 0" (List A) 等价于 "(operation == 'divide') and (b == 0)" (List B)。
        - "b is not 0" (List A) *不等价*于 "b == 0" (List B)。

        [输出]
        请严格按照 "BatchValidationReport" Pydantic 模式返回一个 JSON 对象，包含：
        1.  `matches`: 一个 `MatchPair` 列表，包含所有找到的等价对。
        2.  `unmatched_reqs`: List A 中未找到任何匹配的项。
        3.  `unmatched_code`: List B 中未找到任何匹配的项。
        """
        return prompt.strip()
    
    def _validate_behavior_batch(self):
        """
        执行LLM 语义比较。
        """
        # 1. 提取所有原始谓词
        req_nl_set, code_pred_set = self._extract_predicates()
        
        if not req_nl_set and not code_pred_set:
            print("    -> 行为列表为空，跳过比较。")
            return

        # 2. 构建 O(1) Prompt
        prompt = self._build_batch_validation_prompt(list(req_nl_set), list(code_pred_set))
        
        try:
            # 3. 执行单次 LLM 调用
            print(f"    -> 正在执行LLM批量比较 (Reqs: {len(req_nl_set)}, Code: {len(code_pred_set)})...")
            llm_report: BatchValidationReport = self.structured_llm.invoke(prompt)
            print("    -> LLM 批量比较完成。")

            # 4. 解析 LLM 返回的报告并填充到主报告
            for pair in llm_report.matches:
                self.report['matches'].append(f"行为逻辑一致: '{pair.requirement_nl}' <=> '{pair.code_predicate}'")
            
            for gap in llm_report.unmatched_reqs:
                self.report['gaps_req_to_code'].append(f"行为缺失: D: 需求条件 '{gap}' 未在代码中找到语义等价的实现。")

            for gap in llm_report.unmatched_code:
                # 过滤掉由 'if' 产生的否定分支
                if '!=' not in gap and 'not (' not in gap:
                    self.report['gaps_code_to_req'].append(f"冗余/未定义代码: C: 代码分支 '{gap}' 未在需求中定义。")

        except Exception as e:
            self.report['errors'].append(f"LLM 批量验证失败: {e}. 可能 Prompt 过长或返回 JSON 格式错误。")


