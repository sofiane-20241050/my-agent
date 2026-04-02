import os
from dotenv import load_dotenv
from typing import *
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    provider: str = "zai"        # openai / anthropic / openai
    model: str = "GLM-4-Flash"

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class AgentConfig:
    max_steps: int = 10             # Agent loop 最大轮数
    enable_tools: bool = True
    verbose: bool = True            # 打印 debug 信息
        


# ===== Tool 基类 =====
class Tool:
    def __init__(self, name, func, description, schema):
        self.name = name
        self.func = func
        self.description = description
        self.schema = schema

    def run(self, **kwargs):
        return self.func(**kwargs)






def main():
    modelconfig = ModelConfig(
        provider=os.getenv("MODEL_PROVIDER", "zai"),
        model=os.getenv("MODEL", "GLM-4-Flash"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=float(os.getenv("TEMPERATURE", 0.7)),
        max_tokens=int(os.getenv("MAX_TOKENS", 1024))
    )