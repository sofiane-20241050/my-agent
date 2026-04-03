import os
from dotenv import load_dotenv
import json
import inspect
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
    name: str = None
    description: str = None
    parameters: Optional[dict] = None
    
    def get_schema(self) -> Dict:
        """转换为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
    def exec(self, **kwargs):
        raise NotImplementedError("子类需要实现 exec 方法")
    

class ToolRegistry:
    def __init__(self):
        self.tools = {}  # mapping: name -> callable tool

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str):
        return self.tools.get(name)

    def list(self):
        return self.tools.values()


class Add(Tool):
    name = "add"
    description = "执行加法运算"
    parameters = {
        "type": "object",
        "properties": {
            "a": {
                "type": "integer",
                "description": "加数"
            },
            "b": {
                "type": "integer",
                "description": "被加数"
            }
    },
        "required": ["a", "b"],
    }

    def exec(self, a: str, b: str):
        # 将字符串转换为数字
        a = float(a)  # 可以用 int(a) 或 float(a) 取决于你的需求
        b = float(b)
        result = a + b
        return str(result)




def main():
    modelconfig = ModelConfig(
        provider=os.getenv("MODEL_PROVIDER", "zai"),
        model=os.getenv("MODEL", "GLM-4-Flash"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=float(os.getenv("TEMPERATURE", 0.7)),
        max_tokens=int(os.getenv("MAX_TOKENS", 1024))
    )
    add = Add()
    schema = add.get_schema()
    print(json.dumps(schema, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
