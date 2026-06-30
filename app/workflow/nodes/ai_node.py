"""AI node — calls the AI provider for chat completions."""

import time
from typing import Any

from app.workflow.models import NodeDefinition, NodeExecutionResult, WorkflowContext
from app.workflow.nodes.base import BaseNode


class AINode(BaseNode):
    """Workflow node that calls the AI provider for text generation.

    Config keys:
        prompt_template: Jinja2 template string or variable reference.
        prompt_key: Key in context to use as the prompt.
        output_key: Context key to store the response (default: "ai_response").
        system_prompt: Optional system instruction.
        temperature: Sampling temperature (default: 0.7).
        max_tokens: Maximum response tokens (default: 2048).
        model: Optional model override.
    """

    async def execute(
        self,
        context: WorkflowContext,
        dependencies: dict[str, Any] | None = None,
    ) -> NodeExecutionResult:
        """Execute an AI chat completion.

        Args:
            context: Shared workflow context.
            dependencies: Must contain 'ai_service'.

        Returns:
            NodeExecutionResult: Contains ai_response in output.
        """
        start = time.perf_counter()
        deps = dependencies or {}

        ai_service = deps.get("ai_service")
        if ai_service is None:
            return self._failure("AI service not injected", (time.perf_counter() - start) * 1000)

        # Resolve prompt from config or context variable
        prompt_template = self.config.get("prompt_template")
        prompt_key = self.config.get("prompt_key", "prompt")
        output_key = self.config.get("output_key", "ai_response")

        if prompt_template:
            # Render Jinja2 template against context variables
            try:
                from jinja2 import Template
                prompt = Template(prompt_template).render(**context.variables)
            except Exception as exc:
                return self._failure(f"Prompt template render failed: {exc}")
        else:
            prompt = self.get_input(context, prompt_key, "")

        if not prompt:
            return self._failure("Empty prompt — set 'prompt_template' or 'prompt_key'")

        try:
            from app.schemas.ai import ChatMessage
            from app.common.enums import MessageRole

            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            response = await ai_service.chat(
                messages=messages,
                model=self.config.get("model"),
                temperature=self.config.get("temperature", 0.7),
                max_tokens=self.config.get("max_tokens", 2048),
                system_prompt=self.config.get("system_prompt"),
            )

            self.set_output(context, output_key, response.content)
            duration_ms = (time.perf_counter() - start) * 1000

            return self._success(
                output={
                    output_key: response.content,
                    "model": response.model,
                    "provider": response.provider,
                    "tokens_used": response.usage.total_tokens,
                },
                duration_ms=duration_ms,
            )

        except Exception as exc:
            return self._failure(str(exc), (time.perf_counter() - start) * 1000)
