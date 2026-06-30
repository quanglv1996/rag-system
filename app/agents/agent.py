"""AI Agent — autonomous agent with memory, tools, and reasoning."""

from __future__ import annotations

import json
from typing import Any

from app.core.logger import get_logger
from app.interfaces.ai_provider import AIProvider
from app.schemas.ai import ChatMessage, ChatRequest
from app.common.enums import MessageRole
from app.tools.registry import BaseTool, ToolRegistry, ToolResult

logger = get_logger(__name__)

# Maximum agent reasoning steps before halting
MAX_ITERATIONS = 10


class AgentMemory:
    """Short-term conversation memory for the agent.

    Maintains the message history for multi-turn reasoning.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        """Initialize agent memory.

        Args:
            system_prompt: Optional system instruction for the agent.
        """
        self._messages: list[ChatMessage] = []
        if system_prompt:
            self._messages.append(
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            )

    def add_user(self, content: str) -> None:
        """Add a user message."""
        self._messages.append(ChatMessage(role=MessageRole.USER, content=content))

    def add_assistant(self, content: str) -> None:
        """Add an assistant message."""
        self._messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=content))

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """Add a tool execution result as a user message."""
        content = f"[Tool: {tool_name}] Result: {json.dumps(result, default=str)}"
        self._messages.append(ChatMessage(role=MessageRole.USER, content=content))

    @property
    def messages(self) -> list[ChatMessage]:
        """Return full message history."""
        return self._messages.copy()


class AIAgent:
    """Autonomous AI agent with tool-calling capabilities.

    Implements a ReAct-style (Reason + Act) loop:
    1. Generate a response or tool call.
    2. If tool call: execute → add result → continue reasoning.
    3. If final answer: return.

    Attributes:
        _provider: AI provider for generation.
        _memory: Short-term conversation memory.
        _tools: Available tools by name.
        _max_iterations: Maximum reasoning steps.
    """

    def __init__(
        self,
        provider: AIProvider,
        system_prompt: str | None = None,
        tools: list[BaseTool] | None = None,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        """Initialize the agent.

        Args:
            provider: AI provider for completions.
            system_prompt: Optional agent persona/instruction.
            tools: List of tools available to this agent.
            max_iterations: Safety limit for reasoning loop.
        """
        default_system = (
            "You are a helpful AI agent. When you need information or need to take action, "
            "use the available tools. Think step by step. "
            "When you have a final answer, provide it directly."
        )

        self._provider = provider
        self._memory = AgentMemory(system_prompt=system_prompt or default_system)
        self._tools: dict[str, BaseTool] = {}
        self._max_iterations = max_iterations

        # Register provided tools (or fall back to global registry)
        if tools:
            self._tools = {t.name: t for t in tools}
        else:
            self._tools = {t.name: t for t in ToolRegistry.all_tools()}

    async def run(self, user_input: str) -> str:
        """Run the agent on a user input.

        Executes the ReAct reasoning loop until a final answer is produced
        or the maximum iteration limit is reached.

        Args:
            user_input: Natural language query or task.

        Returns:
            str: Agent's final answer.
        """
        self._memory.add_user(user_input)

        for iteration in range(1, self._max_iterations + 1):
            logger.debug("Agent iteration", iteration=iteration)

            # Build request with tool schemas
            tool_schemas = [t.definition.to_openai_schema() for t in self._tools.values()]

            request = ChatRequest(
                messages=self._memory.messages,
                temperature=0.1,  # Low temperature for consistent reasoning
                max_tokens=2048,
            )

            try:
                response = await self._provider.chat(request)
            except Exception as exc:
                logger.error("Agent LLM call failed", error=str(exc))
                return f"I encountered an error: {exc}"

            content = response.content
            self._memory.add_assistant(content)

            # Check if the LLM wants to call a tool
            tool_call = self._parse_tool_call(content)

            if tool_call is None:
                # No tool call — this is the final answer
                return content

            # Execute the tool
            tool_name = tool_call.get("name", "")
            tool_params = tool_call.get("parameters", {})

            tool = self._tools.get(tool_name)
            if tool is None:
                self._memory.add_tool_result(
                    tool_name, {"error": f"Tool '{tool_name}' not found"}
                )
                continue

            logger.info("Agent executing tool", tool=tool_name, params=tool_params)
            result: ToolResult = await tool.execute(**tool_params)
            self._memory.add_tool_result(tool_name, result.output if result.success else {"error": result.error})

        return "Maximum reasoning steps reached. Unable to complete the task."

    def _parse_tool_call(self, content: str) -> dict[str, Any] | None:
        """Parse a tool call from the LLM response.

        Looks for JSON blocks with 'tool_call' key. The LLM is instructed
        via the system prompt to use this format.

        Args:
            content: LLM response text.

        Returns:
            dict | None: Parsed tool call or None if not a tool call.
        """
        # Look for a JSON block in the response
        import re

        json_pattern = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
        match = json_pattern.search(content)

        if match:
            try:
                parsed = json.loads(match.group(1))
                if "tool_call" in parsed:
                    return parsed["tool_call"]  # type: ignore[return-value]
            except json.JSONDecodeError:
                pass

        # Also try inline JSON with tool_call key
        try:
            if "tool_call" in content:
                # Extract first JSON object from the content
                start = content.index("{")
                parsed = json.loads(content[start:])
                return parsed.get("tool_call")  # type: ignore[return-value]
        except (ValueError, json.JSONDecodeError):
            pass

        return None
