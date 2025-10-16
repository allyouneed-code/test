# -*- coding: utf-8 -*-

import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

# ------------------------- 环境配置 -------------------------
os.environ["OPENAI_API_KEY"] = "sk-OjjN3nmNeSZxEE8c2QJz985fdY3b9XegsKi7lTcl8z6Sr2de"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"

def get_llm_client(model_name: str = "openai:gpt-3.5-turbo-1106", **kwargs) -> BaseChatModel:
    """
    输入模型名称和参数，返回一个LangChain兼容的LLM客户端实例。
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running the application.")

    # langchain's init_chat_model handles the environment variables automatically,
    # but we check for the key's existence for a clearer error message.
    
    print(f"--- Initializing LLM Client: {model_name} with temperature={kwargs.get('temperature', 'default')} ---")
    
    return init_chat_model(model_name, **kwargs)