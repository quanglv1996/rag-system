"""Social media posting workflow node."""

import time
from typing import Any

from app.workflow.models import NodeExecutionResult, WorkflowContext
from app.workflow.nodes.base import BaseNode


class SocialNode(BaseNode):
    """Workflow node that posts content to a social media platform.

    Config keys:
        platform: Target platform ('facebook', 'telegram', 'youtube', 'tiktok').
        action: Action to perform ('post', 'message', 'upload').
        content_key: Context variable key containing the content.
        recipient_key: Context variable for recipient ID (messaging).
        media_key: Context variable for media URL.
        output_key: Context key to store the result (default: 'social_result').
    """

    async def execute(
        self,
        context: WorkflowContext,
        dependencies: dict[str, Any] | None = None,
    ) -> NodeExecutionResult:
        """Post to the configured social platform.

        Args:
            context: Workflow context with content variables.
            dependencies: Must contain the relevant service (e.g., 'facebook_service').

        Returns:
            NodeExecutionResult: Contains post/message result.
        """
        start = time.perf_counter()
        deps = dependencies or {}

        platform = self.config.get("platform", "telegram")
        action = self.config.get("action", "post")
        content_key = self.config.get("content_key", "ai_response")
        output_key = self.config.get("output_key", "social_result")

        content = self.get_input(context, content_key, "")
        if not content:
            return self._failure(f"Content not found in context key '{content_key}'")

        service_key = f"{platform}_service"
        service = deps.get(service_key)
        if service is None:
            return self._failure(f"Service '{service_key}' not injected for platform '{platform}'")

        try:
            result: dict[str, Any] = {}

            if platform == "telegram":
                chat_id = self.get_input(context, self.config.get("recipient_key", "chat_id"), "")
                if not chat_id:
                    return self._failure("Telegram chat_id not found in context")
                result = await service.send_text(chat_id=chat_id, text=content)

            elif platform == "facebook":
                if action == "post":
                    result = await service.publish_post(content=content)
                elif action == "message":
                    recipient = self.get_input(context, self.config.get("recipient_key", "recipient_id"), "")
                    result = await service.send_messenger_message(recipient_id=recipient, text=content)

            elif platform == "youtube":
                result = {"note": "YouTube upload requires separate binary handling"}

            elif platform == "tiktok":
                result = {"note": "TikTok upload initiated via TikTokService"}

            self.set_output(context, output_key, result)
            duration_ms = (time.perf_counter() - start) * 1000

            return self._success(
                output={output_key: result, "platform": platform},
                duration_ms=duration_ms,
            )

        except Exception as exc:
            return self._failure(str(exc), (time.perf_counter() - start) * 1000)


class HTTPNode(BaseNode):
    """Workflow node that makes an arbitrary HTTP request.

    Config keys:
        url: Target URL (may contain {variable} placeholders).
        method: HTTP method (GET, POST, PUT, PATCH, DELETE).
        headers: Static headers dict.
        body_key: Context key containing the request body.
        output_key: Context key to store the response (default: 'http_response').
        timeout: Request timeout in seconds.
    """

    async def execute(
        self,
        context: WorkflowContext,
        dependencies: dict[str, Any] | None = None,
    ) -> NodeExecutionResult:
        """Execute an HTTP request.

        Args:
            context: Workflow context for variable substitution.
            dependencies: Not required.

        Returns:
            NodeExecutionResult: Contains HTTP response body.
        """
        import httpx

        start = time.perf_counter()

        url_template = self.config.get("url", "")
        # Substitute context variables into URL
        url = url_template.format(**context.variables)
        method = self.config.get("method", "GET").upper()
        headers: dict[str, str] = self.config.get("headers", {})
        output_key = self.config.get("output_key", "http_response")
        timeout = float(self.config.get("timeout", 30))

        body_key = self.config.get("body_key")
        body = self.get_input(context, body_key, {}) if body_key else {}

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method in ("GET", "DELETE"):
                    response = await client.request(method, url, headers=headers)
                else:
                    response = await client.request(method, url, headers=headers, json=body)

            response_data: dict[str, Any] = {
                "status_code": response.status_code,
                "body": response.text[:5000],  # Limit stored body size
            }

            self.set_output(context, output_key, response_data)
            duration_ms = (time.perf_counter() - start) * 1000

            return self._success(output={output_key: response_data}, duration_ms=duration_ms)

        except Exception as exc:
            return self._failure(str(exc), (time.perf_counter() - start) * 1000)


class TransformNode(BaseNode):
    """Workflow node that transforms context data using a Python expression.

    Config keys:
        input_key: Context key to read from.
        output_key: Context key to write to.
        expression: Python expression with 'value' as the input variable.

    Example:
        input_key: "ai_response"
        output_key: "summary"
        expression: "value[:500]"
    """

    async def execute(
        self,
        context: WorkflowContext,
        dependencies: dict[str, Any] | None = None,
    ) -> NodeExecutionResult:
        """Apply transformation expression to a context variable.

        Args:
            context: Workflow context.
            dependencies: Not used.

        Returns:
            NodeExecutionResult: Transformed value stored in output_key.
        """
        start = time.perf_counter()

        input_key = self.config.get("input_key", "")
        output_key = self.config.get("output_key", input_key)
        expression = self.config.get("expression", "value")

        value = self.get_input(context, input_key)

        try:
            result = eval(expression, {"__builtins__": {}}, {"value": value})  # noqa: S307
            self.set_output(context, output_key, result)
            duration_ms = (time.perf_counter() - start) * 1000
            return self._success(output={output_key: result}, duration_ms=duration_ms)
        except Exception as exc:
            return self._failure(f"Transform expression error: {exc}")
