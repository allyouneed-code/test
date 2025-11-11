# src/report/docx_reporter.py

import json
import os
import time
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

class DocxWorkflowReporter:
    """
    [Word版] 生成详尽的测试报告，完整可视化 M_req=(U,I,B,C) 和 M_code=(A,S,G)。
    """
    def __init__(self, state: dict, output_filename: str = "Test_Generation_Report.docx"):
        self.state = state
        self.output_filename = output_filename
        self.doc = Document()
        self._setup_styles()

    def _setup_styles(self):
        """配置中文字体和基础样式"""
        style = self.doc.styles['Normal']
        style.font.name = 'Times New Roman'
        style.element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
        style.font.size = Pt(10.5)
        
        # 设置标题样式颜色（可选）
        for i in range(1, 4):
            if f'Heading {i}' in self.doc.styles:
                self.doc.styles[f'Heading {i}'].font.color.rgb = RGBColor(46, 116, 181)

    def generate(self):
        print("  -> 正在生成 Word 报告 (包含完整模型定义)...")
        self._add_title()
        self._add_summary()
        
        # 加载数据
        req_data = self._load_json(self.state.get('structured_requirement', '{}'))
        code_data = self._load_json(self.state.get('analysis_report', '{}'))
        
        # 2. 需求模型
        self.doc.add_heading('2. 需求模型分析 (M_req)', level=1)
        if req_data:
            self._add_req_U_unit_under_test(req_data)
            self._add_req_I_interface(req_data)
            self._add_req_B_behavior(req_data)
            self._add_req_C_constraints(req_data)
        else:
            self.doc.add_paragraph("需求模型数据为空。")

        # 3. 代码模型
        self.doc.add_heading('3. 代码模型分析 (M_code)', level=1)
        self.doc.add_heading('3.1 源代码快照', level=2)
        self._add_code_block(self.state.get('code', ''))
        
        if code_data:
            self._add_code_A_unit_info(code_data)
            self._add_code_S_static_interface(code_data)
            self._add_code_G_control_flow(code_data)
        else:
            self.doc.add_paragraph("代码分析数据为空。")

        # 4. 验证与测试
        self._add_validation_section()
        self._add_test_code_section()
        
        try:
            self.doc.save(self.output_filename)
            print(f"[Report] Word document generated: {os.path.abspath(self.output_filename)}")
        except Exception as e:
            print(f"[Report] Error saving document: {e}")

    def _load_json(self, json_str):
        try:
            return json.loads(json_str)
        except:
            return None

    # ================= M_req 部分 =================

    def _add_req_U_unit_under_test(self, data):
        """2.1 被测单元 (U)"""
        self.doc.add_heading('2.1 被测单元 (U - Unit Under Test)', level=2)
        uut = data.get('unit_under_test', {})
        
        table = self.doc.add_table(rows=0, cols=2)
        table.style = 'Table Grid'
        self._add_table_row(table, "标识符 (ID)", uut.get('identifier', 'N/A'))
        self._add_table_row(table, "描述 (Desc)", uut.get('desc', 'N/A'))
        self._add_table_row(table, "需求溯源", str(uut.get('source', 'N/A')))

    def _add_req_I_interface(self, data):
        """2.2 接口规约 (I)"""
        self.doc.add_heading('2.2 接口规约 (I - Interface)', level=2)
        i_spec = data.get('interface_specification', {})
        
        # 输入参数
        self.doc.add_paragraph("输入参数集合:", style='Caption')
        inputs = i_spec.get('input_parameters', [])
        if inputs:
            table = self.doc.add_table(rows=1, cols=3)
            table.style = 'Table Grid'
            self._set_table_header(table, ['参数名', '描述', '约束'])
            for param in inputs:
                constraints = param.get('constraints', [])
                cons_str = "\n".join(constraints) if isinstance(constraints, list) else str(constraints)
                self._add_table_row(table, param.get('name', ''), param.get('desc', ''), cons_str)
        else:
            self.doc.add_paragraph("无输入参数定义。", style='List Bullet')

        # 外部依赖
        deps = i_spec.get('external_dependencies', [])
        if deps:
            self.doc.add_paragraph("外部依赖 (Mock对象):", style='Caption')
            table = self.doc.add_table(rows=1, cols=2)
            table.style = 'Table Grid'
            self._set_table_header(table, ['依赖名称', '交互契约'])
            for dep in deps:
                self._add_table_row(table, dep.get('name', ''), dep.get('contract', ''))

    def _add_req_B_behavior(self, data):
        """2.3 行为模型 (B)"""
        self.doc.add_heading('2.3 行为模型 (B - Behavior)', level=2)
        b_model = data.get('behavioral_model', {})
        
        # 功能场景
        scenarios = b_model.get('functional_scenarios', [])
        if scenarios:
            self.doc.add_paragraph("功能场景 (Functional Scenarios):", style='Heading 3')
            for sc in scenarios:
                self._add_scenario_card(sc, "FUNC")
        
        # 错误场景
        err_scenarios = b_model.get('error_scenarios', [])
        if err_scenarios:
            self.doc.add_paragraph("异常场景 (Error Scenarios):", style='Heading 3')
            for sc in err_scenarios:
                self._add_scenario_card(sc, "ERR")

    def _add_req_C_constraints(self, data):
        """2.4 约束向量 (C) - 之前缺失的部分"""
        self.doc.add_heading('2.4 约束向量 (C - Constraints)', level=2)
        
        c_model = data.get('constraints', {})
        # 兼容不同的 json 结构 (可能是 list 或 dict)
        param_constraints = c_model.get('param_constraints', []) if isinstance(c_model, dict) else []
        
        if not param_constraints:
            self.doc.add_paragraph("未提取到详细的等价类约束。")
            return

        for pc in param_constraints:
            param_name = pc.get('param_name', 'Unknown')
            self.doc.add_paragraph(f"参数: {param_name}", style='Heading 3')
            
            table = self.doc.add_table(rows=1, cols=3)
            table.style = 'Table Grid'
            self._set_table_header(table, ['类型', '描述', '边界值 (Derived)'])
            
            # 有效等价类
            for valid in pc.get('P_valid', []):
                vals = valid.get('derived_values', [])
                val_str = ", ".join(map(str, vals))
                self._add_table_row(table, "有效 (Valid)", valid.get('description', ''), val_str)
            
            # 无效等价类
            for invalid in pc.get('P_invalid', []):
                vals = invalid.get('derived_values', [])
                val_str = ", ".join(map(str, vals))
                self._add_table_row(table, "无效 (Invalid)", invalid.get('description', ''), val_str)
            
            self.doc.add_paragraph("") # 空行

    # ================= M_code 部分 =================

    def _add_code_A_unit_info(self, data):
        """3.2 单元信息 (A) - 之前缺失的部分"""
        self.doc.add_heading('3.2 单元信息 (A - Unit Info)', level=2)
        a_model = data.get('A', {})
        
        table = self.doc.add_table(rows=0, cols=2)
        table.style = 'Table Grid'
        self._add_table_row(table, "单元ID", a_model.get('Id', 'N/A'))
        self._add_table_row(table, "类型", a_model.get('Type', 'N/A'))
        self._add_table_row(table, "位置", a_model.get('Location', 'N/A'))

    def _add_code_S_static_interface(self, data):
        """3.3 静态接口 (S) - 之前缺失的部分"""
        self.doc.add_heading('3.3 静态接口 (S - Static Interface)', level=2)
        s_model = data.get('S', {})
        
        # 参数列表
        self.doc.add_paragraph("代码定义的参数:", style='Caption')
        args_in = s_model.get('Arg_in', [])
        if args_in:
            table = self.doc.add_table(rows=1, cols=3)
            table.style = 'Table Grid'
            self._set_table_header(table, ['参数名', '静态类型', '默认值'])
            for arg in args_in:
                self._add_table_row(table, arg.get('name', ''), arg.get('static_type', ''), str(arg.get('default_value', '')))
        else:
            self.doc.add_paragraph("无参数或无法解析参数。", style='List Bullet')

        # 外部调用
        c_ext = s_model.get('C_ext', [])
        if c_ext:
            self.doc.add_paragraph("检测到的外部调用:", style='Caption')
            for call in c_ext:
                self.doc.add_paragraph(f"调用: {call.get('target_signature', '')}", style='List Bullet')

    def _add_code_G_control_flow(self, data):
        """3.4 控制流图 (G)"""
        self.doc.add_heading('3.4 控制流与谓词 (G - CFG)', level=2)
        g_model = data.get('G', {})
        
        nodes = g_model.get('Nodes', [])
        edges = g_model.get('Edges', [])
        
        p = self.doc.add_paragraph()
        p.add_run(f"总计: {len(nodes)} 个基本块, {len(edges)} 条跳转边。")
        
        if edges:
            self.doc.add_paragraph("关键路径谓词 (Predicates):", style='Caption')
            # 使用表格展示谓词，更整齐
            table = self.doc.add_table(rows=1, cols=3)
            table.style = 'Table Grid'
            self._set_table_header(table, ['源节点', '目标节点', '跳转条件 (Predicate)'])
            
            for edge in edges:
                pred = edge.get('predicate', 'None')
                if pred != 'None':
                    self._add_table_row(table, edge.get('from_node_id'), edge.get('to_node_id'), pred)

    # ================= 通用辅助方法 =================

    def _add_validation_section(self):
        """4. 验证结果"""
        self.doc.add_heading('4. 一致性检查结果', level=1)
        val_data = self._load_json(self.state.get('validation_report', '{}'))
        if not val_data:
            self.doc.add_paragraph("无验证数据。")
            return

        # 4.1 缺失
        gaps_req = val_data.get('gaps_req_to_code', [])
        if gaps_req:
            self.doc.add_heading(f'4.1 代码未实现的需求 ({len(gaps_req)})', level=2)
            for gap in gaps_req:
                p = self.doc.add_paragraph()
                run = p.add_run("✖ " + str(gap))
                run.font.color.rgb = RGBColor(255, 0, 0) # Red
        
        # 4.2 匹配
        matches = val_data.get('matches', [])
        if matches:
            self.doc.add_heading(f'4.2 一致项 ({len(matches)})', level=2)
            for m in matches:
                p = self.doc.add_paragraph()
                run = p.add_run("✔ " + str(m))
                run.font.color.rgb = RGBColor(0, 128, 0) # Green

    def _add_test_code_section(self):
        self.doc.add_heading('5. 最终测试代码', level=1)
        self._add_code_block(self.state.get('test_code', ''))

    def _add_title(self):
        self.doc.add_heading('自动化单元测试生成报告', 0).alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}").font.color.rgb = RGBColor(128,128,128)
        self.doc.add_page_break()

    def _add_summary(self):
        self.doc.add_heading('1. 执行摘要', level=1)
        table = self.doc.add_table(rows=0, cols=2)
        table.style = 'Table Grid'
        self._add_table_row(table, "最终结果", self.state.get('evaluation_result', 'N/A'))
        self._add_table_row(table, "覆盖率", f"{self.state.get('coverage', 0.0):.2%}")
        self._add_table_row(table, "通过率", f"{self.state.get('pass_rate', 0.0):.2%}")
        self._add_table_row(table, "F-Cases", str(self.state.get('test_failures', 0)))

    def _add_scenario_card(self, sc, type_tag):
        """辅助：以卡片形式展示场景"""
        p = self.doc.add_paragraph()
        # 边框绘制比较复杂，这里用背景色或粗体模拟
        prefix = "[功能]" if type_tag == "FUNC" else "[异常]"
        color = RGBColor(0, 0, 255) if type_tag == "FUNC" else RGBColor(200, 0, 0)
        
        run = p.add_run(f"{prefix} {sc.get('id', '')}: {sc.get('description', '')}")
        run.bold = True
        run.font.color.rgb = color
        
        # 详情
        detail_text = []
        if 'pre_conditions' in sc: detail_text.append(f"前置: {sc['pre_conditions']}")
        if 'error_conditions' in sc: detail_text.append(f"触发: {sc['error_conditions']}")
        if 'expected_outcome' in sc: detail_text.append(f"预期: {sc['expected_outcome']}")
        
        if detail_text:
            p2 = self.doc.add_paragraph("\n".join(detail_text))
            p2.paragraph_format.left_indent = Inches(0.5)

    def _add_table_row(self, table, *args):
        row_cells = table.add_row().cells
        for i, text in enumerate(args):
            if i < len(row_cells):
                row_cells[i].text = str(text)

    def _set_table_header(self, table, headers):
        hdr_cells = table.rows[0].cells
        for i, text in enumerate(headers):
            if i < len(hdr_cells):
                hdr_cells[i].text = text
                # 加粗
                hdr_cells[i].paragraphs[0].runs[0].bold = True
                # 设置背景色需要操作 XML，这里略过以保持代码简洁

    def _add_code_block(self, text):
        table = self.doc.add_table(rows=1, cols=1)
        cell = table.cell(0, 0)
        # 设置浅灰色背景
        shading_elm = OxmlElement('w:shd')
        shading_elm.set(qn('w:val'), 'clear')
        shading_elm.set(qn('w:fill'), 'F2F2F2') # 浅灰
        cell._tc.get_or_add_tcPr().append(shading_elm)
        
        p = cell.paragraphs[0]
        p.text = text
        for run in p.runs:
            run.font.name = 'Courier New'
            run.font.size = Pt(9)