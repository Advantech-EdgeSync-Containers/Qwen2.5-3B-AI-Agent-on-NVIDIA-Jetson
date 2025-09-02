# agent_setup.py
import codecs
import json
import os
import re
from typing import List, Tuple, Any, Optional
from pydantic import PrivateAttr
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import tools, BaseTool


class MyLLMDecisionAgent(BaseSingleActionAgent):
    _tools: List[BaseTool] = PrivateAttr()
    _llm: Any = PrivateAttr()
    _prompt: Any = PrivateAttr()
    _output_parser: Any = PrivateAttr()

    def __init__(self, tools: List[BaseTool], llm: Any):
        super().__init__()
        object.__setattr__(self, "_tools", tools)
        object.__setattr__(self, "_llm", llm)

        system_prompt = """
        You are a smart assistant operating in two distinct modes:

        ---
        ## 🔧 MODE 1: HARDWARE CONTROL TASKS

        Use this mode only when the user's request is related to **hardware control or monitoring**.

        ### ✅ Use tools ONLY if the prompt contains hardware-specific keywords such as:
        - "GPIO", "pin", "read pin", "set pin", "voltage", "temperature", "fan", "BIOS", "motherboard", "sensor", "hardware status", etc.

        You have access to the following hardware tools:

        - `device_info_tool` → Retrieves motherboard and BIOS details.
        - `device_voltage_tool` → Returns onboard voltage sensor readings.
        - `device_temperature_tool` → Returns onboard temperature sensor readings.
        - `device_fans_tool` → Returns real-time fan speeds.
        - `gpio_pins_overview` → Returns GPIO pin direction and logic level.
        - `gpio_set_tool` → Sets GPIO pin level. Format: pin=PIN_NAME, level=LEVEL.
        - `gpio_read_tool` → Reads a GPIO pin's level. Format: pin=PIN_NAME.

        ### ⚠️ Rules:
        - ❌ Do NOT call tools for **math**, **percentages**, **logical reasoning**, or **general natural language tasks**.
        - ❌ Do NOT invent tool names or modify existing ones.
        - ❌ Do NOT use tools if the request is conversational, mathematical, or linguistic.

        ### 🔌 Response Format:
        Respond strictly using the format below—**with explanations or extra text**.

        - General tools (info, voltage, temp, fan, overview):
        tool_name: <tool_name>

        - GPIO Set:
        tool_name: gpio_set_tool
        tool_pin: <PIN_NAME>
        tool_level: <LEVEL>

        - GPIO Read:
        tool_name: gpio_read_tool
        tool_pin: <PIN_NAME>

        ---

        ## 🧠 MODE 2: GENERAL NATURAL LANGUAGE TASKS

        Use this mode for anything not related to hardware control.

        ### ✅ Includes:
        - Arithmetic and percentage calculations (e.g., "What is 15% of 200?")
        - Logical questions or puzzles (e.g., "Two ropes burn at different rates...")
        - Definitions, facts, summaries
        - Creative tasks (e.g., stories, jokes, explanations, opinions)

        ### ⚠️ Rules:
        - ❌ Do NOT call tools in this mode.
        - ✅ Respond naturally in fluent language.
        - ✅ Perform math or logic directly using reasoning—do NOT offload to tools.

        ---

        ## 📌 EXAMPLES

        User: What's the board info?
        →
        tool_name: device_info_tool

        User: Show system voltages
        →
        tool_name: device_voltage_tool

        User: Set GPIO pin GPIO9 to HIGH
        →
        tool_name: gpio_set_tool
        tool_pin: GPIO9
        tool_level: HIGH

        User: Read GPIO pin GPIO7
        →
        tool_name: gpio_read_tool
        tool_pin: GPIO7

        User: Hi or Hello?
        →
        Hi, How can I assist you ?
        
        User: What is 25% of 640?
        →
        160

        User: Define photosynthesis
        →
        Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water...

        User: Tell me a joke
        →
        Why did the computer go to therapy? It had too many bytes from its past!

        ---

        ## ❗ ABSOLUTE RULES

        - ❌ Never explain your internal reasoning or tool selection process.
        - ❌ Never guess or hallucinate tool names.
        - ❌ Never mix tool output with natural responses.
        - ❌ Never mix tool output with natural responses.
        - ✅ Avoid using prefixes like "Answer:", "AI:", or "Thinking...".
        - ✅ Do not use role-based or formatting prompts.
        - ✅ Always use direct reasoning for math or general knowledge.

        Stay accurate. Be direct. Choose the correct mode.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt.strip()),
            ("human", "{question}")
        ])

        object.__setattr__(self, "_prompt", prompt)
        object.__setattr__(self, "_output_parser", StrOutputParser())
        object.__setattr__(self, "_chain", self._prompt | self._llm | self._output_parser)

    def extract_json_key(self, code_block: str, key: str) -> Optional[str]:
        """
        Extract the value of a specified key from a markdown-wrapped JSON code block.
        """
        try:
            # Remove triple backticks and optional language specifier like ```json
            cleaned = re.sub(r"```[a-zA-Z]*", "", code_block).replace("```", "").strip()
            parsed = json.loads(cleaned)
            return parsed.get(key)
        except (json.JSONDecodeError, TypeError):
            return None

    def remove_think_block(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    def replace_tool_with_output(self, text: str, output: str) -> str:
        return re.sub(r"(?i)^tool_name:\s*\w+\s*$", output, text, flags=re.MULTILINE)

    def extract_tool(self, text: str) -> dict:
        tool_names = [tool.name.lower() for tool in tools]
        text = codecs.decode(text, 'unicode_escape')

        # Match the tool name
        pattern = r"\b(" + "|".join(re.escape(tool) for tool in tool_names) + r")\b"
        tool_match = re.search(pattern, text)

        if not tool_match:
            return {}

        tool_name = tool_match.group(1).strip()

        result = {"tool_name": tool_name}

        if tool_name in ["gpio_set_tool", 'gpio_read_tool']:
            # Extract tool_pin
            pin_match = re.search(r"tool_pin:\s*([\w\d]+)", text, re.IGNORECASE)
            level_match = re.search(r"tool_level:\s*([\w\d]+)", text, re.IGNORECASE)

            if pin_match:
                result["tool_pin"] = pin_match.group(1).strip()
            if level_match:
                result["tool_level"] = level_match.group(1).strip().upper()

        return result

    async def ainvoke_tool_from_text(self, text: str, original_input: str, callback_handler=None) -> str:

        tool_name, tool_input = self.extract_tool_and_input(text)
        tool_lookup = {tool.name.lower(): tool for tool in self._tools}

        print(f"tool: {str(tool_name)}")
        print(f"tool_input: {str(tool_input)}")
        if not tool_name or tool_name == "none" or tool_name not in tool_lookup:
            if callback_handler:
                # Stream response from LLM with callback
                self._llm.callbacks = [callback_handler]
                await self._llm.ainvoke(original_input)
                return None  # signal: already streamed
            else:
                result = await self._llm.ainvoke(original_input)
                return self.remove_think_block(result)

        if tool_name in tool_lookup:
            tool_result = tool_lookup[tool_name].run({"input": tool_input})
            return tool_result

        return f"Unknown tool: {tool_name}"

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> AgentFinish:
        raise NotImplementedError("Use aplan() with streaming")

    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> AgentFinish:
        prompt = kwargs["input"]
        streamed_text = await self._chain.ainvoke({"question": prompt})
        final_result = await self.ainvoke_tool_from_text(streamed_text, prompt)
        return AgentFinish(return_values={"output": final_result}, log="Streaming + tool resolved")

    @property
    def input_keys(self) -> List[str]:
        return ["input"]
