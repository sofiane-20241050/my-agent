import os
import re   
from dotenv import load_dotenv
import json
import inspect
import asyncio
from typing import *
from dataclasses import dataclass, field
from openai import OpenAI
from prompt import SYSTEM_PROMPT
import logging


logger = logging.getLogger(__name__)


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


class ToolCallParseError(Exception):
    pass


def parse_tool_call(text: str):
    """
    从 LLM 输出中解析 tool call

    返回:
        (tool_name, args_dict)

    失败:
        raise ToolCallParseError
    """

    try:
        # 1️⃣ 去掉 ```json ``` 包裹
        text = text.strip()
        text = re.sub(r"^```json", "", text)
        text = re.sub(r"```$", "", text)

        # 2️⃣ 提取 JSON（防止模型前后说废话）
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ToolCallParseError("No JSON found")

        json_str = match.group(0)

        # 3️⃣ 解析 JSON
        data = json.loads(json_str)

        # 4️⃣ 校验结构
        if "tool" not in data or "args" not in data:
            raise ToolCallParseError("Invalid tool format")

        tool_name = data["tool"]
        args = data["args"]

        if not isinstance(args, dict):
            raise ToolCallParseError("Args must be dict")

        return tool_name, args

    except json.JSONDecodeError as e:
        raise ToolCallParseError(f"JSON decode error: {e}")

    except Exception as e:
        raise ToolCallParseError(str(e))


# ===== 大模型 Client =====
class LLMClient:
    def __init__(self, model_config):
        self.model_config = model_config
        self.client = OpenAI(
            api_key=model_config.api_key,
            base_url=model_config.base_url  
        )

    def completion(self, messages):
        """
        标准 chat completion（不走 tools）
        """
        response = self.client.chat.completions.create(
            model=self.model_config.model,
            messages=messages,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens
        )
        return response.choices[0].message.content


# ===== Agent ===== 

class Agent:
    def __init__(self, llm: LLMClient, registry: ToolRegistry, agent_config: AgentConfig):
        self.llm = llm
        self.registry = registry
        self.agent_config = agent_config

    async def run(self, query: str):
        messages = []

        # system
        messages.append({
            "role": "system",
            "content": self._build_system_prompt()
        })

        messages.append({"role": "user", "content": query})
        logger.info(f"[User] {query}")

        step = 0

        while step < self.agent_config.max_steps:
            step += 1

            response = self.llm.completion(messages)

            if self.agent_config.verbose:
                logger.info(f"[Agent] {response}")

            try:
                tool_name, args = parse_tool_call(response)

                tool = self.registry.get(tool_name)
                if tool is None:
                    raise Exception(f"Tool {tool_name} not found")
                logger.info(f"===STEP {step}===")
                if self.agent_config.verbose:
                    logger.info(f"[TOOL CALL] {tool_name} {args}")

                result = await tool.exec(**args)

                if self.agent_config.verbose:
                    logger.info(f"[TOOL RESULT] {result}")

                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "assistant", "content": f"[TOOL RESULT]\n{result}"})

            except Exception:
                messages.append({"role": "assistant", "content": response})
                return response

        return "Max steps exceeded"

    def _build_system_prompt(self):
        tools = self.registry.list()

        tools_str = "\n".join([
            # f"{t.name}: {t.description}\nparameters: {t.parameters}"
            t.get_schema()
            for t in tools
        ])

        return SYSTEM_PROMPT.format(tools=tools_str)

async def main():
    modelconfig = ModelConfig(
        provider=os.getenv("MODEL_PROVIDER", "zai"),
        model=os.getenv("MODEL", "GLM-4-Flash"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=float(os.getenv("TEMPERATURE", 0.7)),
        max_tokens=int(os.getenv("MAX_TOKENS", 1024))
    )
    agentConfig = AgentConfig(
        max_steps=int(os.getenv("MAX_STEPS", 10)),
        verbose=True
    )
    add = Add()
    schema = add.get_schema()
    logger.info("启用的工具有：")
    print(json.dumps(schema, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
