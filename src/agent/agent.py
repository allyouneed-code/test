# -*- coding: utf-8 -*-
"""
File: base_agent.py
Description: Defines the abstract base class for all agents in the IntelliTest-Agent system.
Author: Your Name
Date: 2025-10-08
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

# To represent any LLM client from LangChain or other frameworks.
# For example: langchain_openai.ChatOpenAI, langchain_anthropic.ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel


class BaseAgent(ABC):
    """
    An abstract base class that defines the common interface for all agents.

    This class ensures that any concrete agent implementation within the system
    will have a consistent structure, including initialization and a primary
    execution method.
    """

    def __init__(self, role: str, llm_client: BaseChatModel):
        """
        Initializes the BaseAgent.

        Args:
            role (str): A string describing the agent's specific role or purpose.
                        (e.g., "Test Generation Agent", "Prompt Engineering Meta-Agent").
                        This is useful for logging and debugging.
            llm_client (BaseChatModel): An instance of a LangChain-compatible LLM client
                                       (e.g., ChatOpenAI) that the agent will use
                                       to perform its tasks.
        """
        if not role:
            raise ValueError("Agent 'role' cannot be empty.")
        if not llm_client:
            raise ValueError("Agent 'llm_client' must be provided.")
            
        self.role = role
        self.llm_client = llm_client
        print(f"[{self.__class__.__name__}] initialized with role: '{self.role}'")

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """
        The main entry point for the agent to perform its primary task.

        This method must be implemented by all subclasses. It is designed to be
        flexible by accepting arbitrary keyword arguments, allowing each agent
        to define its own specific inputs.

        Args:
            **kwargs: A dictionary of inputs required by the specific agent's task.
                      For example, a TestGenAgent might expect 'structured_context'
                      and 'prompt', while a PromptEngAgent might expect 'goal' and
                      'performance_history'.

        Returns:
            Any: The output of the agent's task. The type will vary depending
                 on the agent's role.
        """
        pass

    def get_role(self) -> str:
        """
        Returns the role of the agent.
        
        Returns:
            str: The agent's role description.
        """
        return self.role

# --- Example of a Concrete Implementation (for demonstration purposes) ---
# This part would typically be in another file like `test_gen_agent.py`.
#
# class TestGenAgent(BaseAgent):
#     def __init__(self, llm_client: BaseChatModel):
#         super().__init__(
#             role="A specialized agent for generating unit test cases from code context.",
#             llm_client=llm_client
#         )
#
#     def run(self, structured_context: Dict[str, Any], prompt: str) -> str:
#         """
#         Generates test cases based on the provided context and prompt.
#
#         Args:
#             structured_context (Dict[str, Any]): The combined analysis of code and requirements.
#             prompt (str): The prompt (potentially from PromptEngAgent) to use for generation.
#
#         Returns:
#             str: The generated test case code as a string.
#         """
#         print(f"[{self.get_role()}] is running with a given prompt...")
#         # In a real scenario, this would involve creating a chain and invoking the LLM.
#         # e.g., chain = ...
#         # response = self.llm_client.invoke(prompt.format(context=structured_context))
#         # return response.content
#         return "import pytest\n\ndef test_example():\n    assert 1 + 1 == 2\n"