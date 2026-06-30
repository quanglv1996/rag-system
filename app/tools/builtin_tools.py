"""Built-in tools for the AI Agent Framework."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.tools.registry import BaseTool, ToolDefinition, ToolRegistry, ToolResult


class RAGTool(BaseTool):
    """Tool that queries the RAG pipeline for knowledge retrieval.

    Allows an agent to answer questions using indexed documents.
    """

    def __init__(self, rag_service: Any) -> None:
        """Initialize with a RAG service instance.

        Args:
            rag_service: RAGService instance.
        """
        self._rag_service = rag_service

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition."""
        return ToolDefinition(
            name="rag_search",
            description=(
                "Search the knowledge base for information. Use this when the user "
                "asks a question that might be answered by documents in the system."
            ),
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to search for in the knowledge base",
                },
                "collection": {
                    "type": "string",
                    "description": "Knowledge base collection to search (default: 'default')",
                },
            },
            required=["question"],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute a RAG query.

        Args:
            question: The question to answer.
            collection: Optional collection name.

        Returns:
            ToolResult: RAG answer and sources.
        """
        question = kwargs.get("question", "")
        collection = kwargs.get("collection", "default")

        try:
            result = await self._rag_service.query(
                question=question, collection=collection
            )
            return ToolResult(
                success=True,
                output={"answer": result.answer, "sources_count": result.retrieved_count},
                tool_name=self.name,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool_name=self.name, output=None)


class HTTPTool(BaseTool):
    """Tool for making arbitrary HTTP GET/POST requests."""

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition."""
        return ToolDefinition(
            name="http_request",
            description="Make an HTTP request to a given URL and return the response body.",
            parameters={
                "url": {"type": "string", "description": "Target URL"},
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE"],
                    "description": "HTTP method",
                },
                "body": {
                    "type": "object",
                    "description": "Optional JSON body for POST/PUT requests",
                },
            },
            required=["url"],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute an HTTP request.

        Args:
            url: Request URL.
            method: HTTP method (default: GET).
            body: Optional request body.

        Returns:
            ToolResult: Response body and status code.
        """
        url = kwargs.get("url", "")
        method = kwargs.get("method", "GET").upper()
        body = kwargs.get("body", {})

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if method in ("POST", "PUT", "PATCH"):
                    response = await client.request(method, url, json=body)
                else:
                    response = await client.request(method, url)

            return ToolResult(
                success=response.status_code < 400,
                output={
                    "status_code": response.status_code,
                    "body": response.text[:2000],
                },
                tool_name=self.name,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool_name=self.name, output=None)


class CalculatorTool(BaseTool):
    """Tool for safe mathematical expression evaluation."""

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition."""
        return ToolDefinition(
            name="calculator",
            description="Evaluate a mathematical expression and return the result.",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')",
                }
            },
            required=["expression"],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Evaluate a math expression.

        Args:
            expression: Math expression string.

        Returns:
            ToolResult: Numeric result.
        """
        expression = kwargs.get("expression", "")

        try:
            import ast

            # Only allow safe AST nodes (numbers and arithmetic operations)
            tree = ast.parse(expression, mode="eval")
            allowed_types = (
                ast.Expression, ast.Num, ast.BinOp, ast.UnaryOp,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
                ast.USub, ast.UAdd, ast.Constant,
            )
            for node in ast.walk(tree):
                if not isinstance(node, allowed_types):
                    raise ValueError(f"Unsafe expression node: {type(node).__name__}")

            result = eval(compile(tree, "<expr>", "eval"))  # noqa: S307
            return ToolResult(
                success=True,
                output={"result": result, "expression": expression},
                tool_name=self.name,
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool_name=self.name, output=None)


class TelegramTool(BaseTool):
    """Tool for sending Telegram messages from an agent."""

    def __init__(self, telegram_service: Any) -> None:
        """Initialize with a Telegram service.

        Args:
            telegram_service: TelegramService instance.
        """
        self._service = telegram_service

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition."""
        return ToolDefinition(
            name="send_telegram",
            description="Send a text message to a Telegram chat or user.",
            parameters={
                "chat_id": {"type": "string", "description": "Target chat ID"},
                "message": {"type": "string", "description": "Message text to send"},
            },
            required=["chat_id", "message"],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Send a Telegram message.

        Args:
            chat_id: Target chat ID.
            message: Message text.

        Returns:
            ToolResult: Send result.
        """
        chat_id = kwargs.get("chat_id", "")
        message = kwargs.get("message", "")

        try:
            result = await self._service.send_text(chat_id=chat_id, text=message)
            return ToolResult(success=True, output=result, tool_name=self.name)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc), tool_name=self.name, output=None)


def register_default_tools(dependencies: dict[str, Any] | None = None) -> None:
    """Register all default tools in the ToolRegistry.

    Args:
        dependencies: Service dependencies for tools that need them.
    """
    deps = dependencies or {}

    ToolRegistry.register(HTTPTool())
    ToolRegistry.register(CalculatorTool())

    if rag_service := deps.get("rag_service"):
        ToolRegistry.register(RAGTool(rag_service=rag_service))

    if telegram_service := deps.get("telegram_service"):
        ToolRegistry.register(TelegramTool(telegram_service=telegram_service))
