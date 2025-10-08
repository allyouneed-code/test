import json
import openai
import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import ast
from pydantic import BaseModel, Field
from typing_extensions import Literal
from IPython.display import Image, display
from langchain_core.tools import tool


os.environ["OPENAI_API_KEY"] = "sk-OjjN3nmNeSZxEE8c2QJz985fdY3b9XegsKi7lTcl8z6Sr2de"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"



# State schema
class State(TypedDict):
    code: str
    requirement: str
    prompt: str
    ast_info: str
    test_code: str
    coverage: float
    pass_rate: float
    feedback : str
    results: str

class Feedback(BaseModel):
    grade: Literal["pass", "not pass"] = Field(
        description="Decide if the testcases is passed or not.",
    )
    feedback: str = Field(
        description="If the cases is not passed, provide feedback on how to improve it.",
    )


class GraphBuilder:
    # 为每个节点预设 system-level 回复模板
    ORGANIZER_SYS = (
        "You are the Prompt Organizer. "
        "Take the user's requirement and source code (with AST info) "
        "and package them into a single structured prompt."
    )
    GENERATOR_SYS = (
        "You are the Test Generator. "
        "Given a well-structured prompt, generate pytest-compatible unit tests. "
        "Only output valid Python code in markdown fences."
    )
    EVALUATOR_SYS = (
        "You are the Result Evaluator. "
        "Given coverage and pass rate results, decide if the tests meet the quality thresholds. coverage>=80"
        "If not, explain what failed."
    )

    def __init__(self):
        self.llm = init_chat_model("openai:gpt-3.5-turbo")
        self.graph = StateGraph(State)
        # Augment the LLM with schema for structured output
        self.evaluator = self.llm.with_structured_output(Feedback)

    def ast_analyze(self, state: State) -> str:
        """
        Perform static AST analysis on the provided code.
        Returns a string representation of the AST.
        """
        tree = ast.parse(state["code"])
        ast_info = ast.dump(tree, annotate_fields=True, include_attributes=False)
        return {"ast_info": ast_info}

    # Node 1: Prompt Organizer
    def prompt_organizer(self, state: State) -> dict:
        code = state["code"]
        msg = self.llm.invoke(
                f"Write prompt about code:{state['code']} requirement:{state['requirement']} and ast_info"
            )
        # 构造供 LLM 使用的 prompt
        return {"prompt": msg.content}

    # Node 2: Test Generator
    def test_generator(self, state: State) -> dict:
        if state.get("feedback"):
            resp = self.llm.invoke(
                f"Write case about {state['prompt']} but take into account the feedback: {state['feedback']}"
            )
        else:
            resp = self.llm.invoke(state["prompt"])
        return {"test_code": resp.content}

    # Node 3: Test Executor
    def test_executor(self, state: State) -> dict:
        # 从 prompt 里提取原始代码
        test_code = state["test_code"]
        return {"coverage": 0.9, "pass_rate": 0.9}

    # Node 4: Result Evaluator
    def result_evaluator(self, state: State) -> dict:
        grade = self.evaluator.invoke(f"Grade the case {state['test_code']} with coverage {state['coverage']} and pass rate {state['pass_rate']}")
        return {"results": grade.grade, "feedback": grade.feedback}
    
    def route_case(self,state: State):
        if state["results"] == "pass":
            return "Accepted"
        elif state["results"] == "not pass":
            return "Rejected + Feedback"
        
    def check_ast(self, state: State) -> str:
        """
        Check if the AST info is present in the state.
        If not, perform AST analysis.
        """
        if not state["ast_info"]:
            return "ast_analyze"
        else:
            return "next"

    def build(self):
        # 注册节点
        self.graph.add_node("organizer", self.prompt_organizer)
        self.graph.add_node("designer", self.test_generator)
        self.graph.add_node("generator", self.test_executor)
        self.graph.add_node("evaluator", self.result_evaluator)

        # 构建有向图
        self.graph.add_edge(START, "organizer")
        self.graph.add_edge("organizer", "designer")
        self.graph.add_edge("designer", "generator")
        self.graph.add_edge("generator", "evaluator")
        # 失败则回到 organizer 重试，成功则结束
        self.graph.add_conditional_edges(
            "evaluator",
            self.route_case,
            {  # Name returned by route_joke : Name of next node to visit
                "Accepted": END,
                "Rejected + Feedback": "organizer",
            },
        )
        return self.graph.compile()

if __name__ == "__main__":
    builder = GraphBuilder()
    graph = builder.build()
    # example usage
    # state = graph.invoke({"requirement": "Cover edge cases for divide by zero", "code": "def div(a, b): return a/b"})
    # print(state["test_code"])
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("my_graph.png", "wb") as f:
        f.write(png_bytes)

