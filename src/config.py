# src/config.py

import os
import json
from dotenv import load_dotenv
from typing import Dict, Any

def load_app_config() -> Dict[str, Any]:
    """
    加载应用的完整配置。

    优先级顺序:
    1. 环境变量 (通过 .env 文件或系统直接设置)
    2. config.json 文件
    3. 代码中定义的默认值

    Returns:
        一个包含所有配置项的字典。
    """
    # 1. 加载 .env 文件中的环境变量
    # 这会寻找项目根目录下的 .env 文件并加载它
    load_dotenv() 

    # 2. 设置默认值
    config = {
        "max_retries": 3,
        "coverage_threshold": 0.8,
        "mutation_threshold": 0.9,
        "logic_filename": "logic_module.py",
        "test_filename": "test_script.py",
    }

    # 3. 尝试从 config.json 文件加载并覆盖默认值
    try:
        # 路径相对于当前文件，向上两级找到项目根目录下的 config.json
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
            # 更新工作流和代码执行器的设置
            config.update(file_config.get("workflow_settings", {}))
            config.update(file_config.get("code_executer_settings", {}))
    except (FileNotFoundError, json.JSONDecodeError):
        print("--- config.json 未找到或无效，将使用默认设置。 ---")
        pass

    return config

# --- 全局配置实例 ---
# 在应用启动时加载一次，之后所有模块都可以从这里导入配置
app_config = load_app_config()