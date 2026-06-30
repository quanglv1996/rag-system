"""Tool registry and base class for the AI Agent Framework.

Tools are discrete capabilities an Agent can invoke (RAG search,
HTTP calls, database queries, social posting, etc.). Each tool
implements the BaseTool ABC and registers itself in ToolRegistry.

Agents only interact with tools through the interface — new tools
can be added without modifying the agent or registry code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolDefinition:
    """Describes a tool for the LLM's function-calling schema.

    Attributes:
        name: Unique tool identifier (snake_case).
        description: Human + LLM readable description of what the tool does.
        parameters: JSON Schema describing accepted parameters.
        required: List of required parameter names.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI function calling schema format.

        Returns:
            dict: OpenAI-compatible tool schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


@dataclass
class ToolResult:
    """Result returned by a tool execution.

    Attributes:
        success: Whether execution succeeded.
        output: The tool's output data.
        error: Error message if execution failed.
        tool_name: Name of the tool that produced this result.
    """

    success: bool
    output: Any
    error: str | None = None
    tool_name: str = ""


class BaseTool(ABC):
    """Abstract base class for all agent tools.

    Each tool must implement:
    - definition: Returns the ToolDefinition for LLM schema.
    - execute: Runs the tool with provided parameters.
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool definition for schema generation.

        Returns:
            ToolDefinition: Tool metadata and parameter schema.
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Tool-specific parameters matching the definition schema.

        Returns:
            ToolResult: Execution result with success/failure status.
        """
        ...

    @property
    def name(self) -> str:
        """Return the tool's name.

        Returns:
            str: Tool name from definition.
        """
        return self.definition.name


class ToolRegistry:
    """Central registry for all agent tools.

    Tools register themselves by calling ToolRegistry.register().
    Agents query the registry to discover available tools and their schemas.
    """

    _tools: dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool) -> None:
        """Register a tool instance.

        Args:
            tool: Tool instance to register.
        """
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            BaseTool | None: Tool instance or None.
        """
        return cls._tools.get(name)

    @classmethod
    def get_or_raise(cls, name: str) -> BaseTool:
        """Get a tool by name or raise.

        Args:
            name: Tool name.

        Returns:
            BaseTool: Tool instance.

        Raises:
            ValueError: If the tool is not registered.
        """
        tool = cls._tools.get(name)
        if tool is None:
            raise ValueError(
                f"Tool '{name}' is not registered. Available: {list(cls._tools.keys())}"
            )
        return tool

    @classmethod
    def all_tools(cls) -> list[BaseTool]:
        """Return all registered tools.

        Returns:
            list[BaseTool]: All registered tool instances.
        """
        return list(cls._tools.values())

    @classmethod
    def get_schemas(cls) -> list[dict[str, Any]]:
        """Get OpenAI function calling schemas for all tools.

        Returns:
            list[dict]: List of OpenAI-compatible tool schemas.
        """
        return [t.definition.to_openai_schema() for t in cls._tools.values()]

    @classmethod
    def list_names(cls) -> list[str]:
        """List names of all registered tools.

        Returns:
            list[str]: Tool name strings.
        """
        return list(cls._tools.keys())
