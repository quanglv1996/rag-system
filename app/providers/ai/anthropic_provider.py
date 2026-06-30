"""Anthropic Claude provider implementation.

Implements the AIProvider interface using the Anthropic SDK.
Supports Claude 3 Opus/Sonnet/Haiku for chat, streaming, and vision.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from app.core.config import get_settings
from app.core.exception import ProviderException, RateLimitException
from app.core.logger import get_logger
from app.interfaces.ai_provider import AIProvider
from app.schemas.ai import (
    ChatRequest,
    ChatResponse,
    ChatUsage,
    EmbeddingRequest,
    EmbeddingResponse,
    VisionRequest,
    VisionResponse,
)

logger = get_logger(__name__)


class AnthropicProvider(AIProvider):
    """Anthropic Claude AI provider.

    Supports chat, streaming, and vision capabilities.
    Does NOT support embeddings (Claude has no embedding API).
    """

    def __init__(self) -> None:
        """Initialize the Anthropic provider."""
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderException(
                "anthropic package not installed. Run: pip install anthropic",
                provider="anthropic",
                operation="init",
            ) from exc

        settings = get_settings()
        # Expect ANTHROPIC_API_KEY in settings via .env
        api_key = getattr(settings, "anthropic_api_key", "")
        if not api_key:
            raise ProviderException(
                "ANTHROPIC_API_KEY is not configured",
                provider="anthropic",
                operation="init",
            )

        import anthropic as anthropic_lib
        self._client = anthropic_lib.AsyncAnthropic(api_key=api_key)
        self._model = getattr(settings, "anthropic_model", "claude-3-5-sonnet-20241022")
        logger.info("Anthropic provider initialized", model=self._model)

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "anthropic"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request to Anthropic Claude.

        Args:
            request: Chat request with messages.

        Returns:
            ChatResponse: Claude's response.
        """
        model = request.model or self._model

        # Separate system message from conversation
        system_content = ""
        messages = []
        for msg in request.messages:
            if msg.role.value == "system":
                system_content = msg.content
            else:
                messages.append({"role": msg.role.value, "content": msg.content})

        try:
            response = await self._client.messages.create(
                model=model,
                max_tokens=request.max_tokens,
                system=system_content or "You are a helpful assistant.",
                messages=messages,  # type: ignore[arg-type]
                temperature=request.temperature,
            )

            content = response.content[0].text if response.content else ""
            usage = response.usage

            return ChatResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage=ChatUsage(
                    prompt_tokens=usage.input_tokens,
                    completion_tokens=usage.output_tokens,
                    total_tokens=usage.input_tokens + usage.output_tokens,
                ),
                finish_reason=response.stop_reason,
            )

        except Exception as exc:
            err = str(exc).lower()
            if "rate" in err or "overload" in err:
                raise RateLimitException("Anthropic rate limit exceeded") from exc
            raise ProviderException(str(exc), provider="anthropic", operation="chat") from exc

    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Stream Claude's response token by token.

        Args:
            request: Chat request.

        Yields:
            str: Response tokens.
        """
        model = request.model or self._model
        system_content = ""
        messages = []
        for msg in request.messages:
            if msg.role.value == "system":
                system_content = msg.content
            else:
                messages.append({"role": msg.role.value, "content": msg.content})

        try:
            async with self._client.messages.stream(
                model=model,
                max_tokens=request.max_tokens,
                system=system_content or "You are a helpful assistant.",
                messages=messages,  # type: ignore[arg-type]
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:
            raise ProviderException(str(exc), provider="anthropic", operation="stream") from exc

    async def embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Anthropic does not provide embedding API.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Anthropic does not support embeddings via this API")

    def get_capabilities(self) -> dict[str, bool]:
        """Return capability map for Anthropic.

        Returns:
            dict: Supported capabilities.
        """
        return {
            "chat": True,
            "stream": True,
            "embedding": False,
            "image_generation": False,
            "speech_to_text": False,
            "text_to_speech": False,
            "vision": True,
        }
