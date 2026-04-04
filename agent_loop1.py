import os
import re   
from dotenv import load_dotenv
import json
import inspect
import requests
from tavily import TavilyClient
import asyncio
from typing import *
from dataclasses import dataclass
from openai import OpenAI
import logging
from datetime import datetime
from prompt import SYSTEM_PROMPT


# 创建日志目录
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 配置 logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            f"{log_dir}/agent_loop_{datetime.now():%Y%m%d_%H%M%S}.log",
            encoding='utf-8'
        )
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    provider: str = "zai"        # openai / anthropic / openai
    model: str = "GLM-4.5-Air"

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
    async def exec(self, **kwargs):
        raise NotImplementedError("子类需要实现 exec 方法")
    

class ToolRegistry:
    def __init__(self):
        self.tools = {}  # mapping: name -> callable tool

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str):
        return self.tools.get(name)

    def _list(self):
        return list(self.tools.values())


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

    async def exec(self, a: str, b: str):
        # 将字符串转换为数字
        a = float(a)  # 可以用 int(a) 或 float(a) 取决于你的需求
        b = float(b)
        result = a + b
        return str(result)


class Weather(Tool):
    name = "get_weather"
    description = "获取指定城市的当前天气信息"
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称"
            }
        },
        "required": ["city"],
    }

    async def exec(self, city: str):
        url = f"https://wttr.in/{city}?format=j1"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()

            current = data["current_condition"][0]
            desc = current["weatherDesc"][0]["value"]
            temp = current["temp_C"]

            return f"{city}当前天气：{desc}，气温{temp}°C"

        except Exception as e:
            return f"天气查询失败：{str(e)}"


class TavilySearch(Tool):
    name = "tavily_search"
    description = "使用Tavily搜索引擎获取互联网信息摘要"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或问题"
            }
        },
        "required": ["query"],
    }

    async def exec(self, query: str):
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "未配置搜索API Key"

        tavily = TavilyClient(api_key=api_key)

        try:
            response = tavily.search(
                query=query,
                search_depth="basic",
                include_answer=True
            )

            if response.get("answer"):
                return response["answer"]

            results = response.get("results", [])
            if not results:
                return "未找到相关信息"

            formatted = [
                f"{item['title']}: {item['content']}"
                for item in results[:5]
            ]

            return "\n".join(formatted)

        except Exception as e:
            return f"搜索失败：{str(e)}"


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
        text = re.sub(r"^```", "", text)    

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
                # 部分模型没有tool这个role，且不支持连续为assistant的角色
                messages.append({"role": "user", "content": f"[TOOL RESULT]\n{result}"})
                continue

            except Exception:
                messages.append({"role": "assistant", "content": response})
                logger.info(f"[Final RESULT] {response}")
                return response

        return "Max steps exceeded"

    def _build_system_prompt(self):
        tools = self.registry._list()

        tools_str = "\n".join([
            # f"{t.name}: {t.description}\nparameters: {t.parameters}"
            json.dumps(t.get_schema(), indent=4, ensure_ascii=False)
            for t in tools
        ])
        system_prompt = SYSTEM_PROMPT.replace("{{tools}}", tools_str)
        logger.info(f"[SYSTEM] {system_prompt}")
        return system_prompt

async def main():
    load_dotenv()
    logger.info("已加载环境变量")

    # Config
    modelconfig = ModelConfig(
        provider=os.getenv("MODEL_PROVIDER", "zai"),
        model=os.getenv("MODEL", "GLM-4.5-Air"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=float(os.getenv("TEMPERATURE", 0.7)),
        max_tokens=int(os.getenv("MAX_TOKENS", 1024))
    )
    agentConfig = AgentConfig(
        max_steps=int(os.getenv("MAX_STEPS", 10)),
        verbose=True
    )
    logger.info(f"已加载模型配置")

    # Tools
    add = Add()
    weather = Weather()
    search = TavilySearch()
    tools = ToolRegistry()
    tools.register(add)
    tools.register(weather)
    tools.register(search)
    logger.info(f"启用的工具有：{[t.name for t in tools._list()]}")
    print([t.name for t in tools._list()])
    
    # Agent
    llm = LLMClient(modelconfig)
    agent = Agent(llm, tools, agentConfig)  
    logger.info("已创建Agent工作流")
    prompt = "我想知道现在北京的天气适合到哪些景点游玩啊"
    await agent.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
