"""Ollama local LLM provider.

Implements the AIProvider interface for locally running Ollama models
(Llama 3, Mistral, Qwen, etc.). Enables fully offline AI without any
external API dependency.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import httpx

from app.core.exception import ConfigurationException, ProviderException
from app.core.logger import get_logger
from app.interfaces.ai_provider import AIProvider
from app.schemas.ai import (
    ChatRequest,
    ChatResponse,
    ChatUsage,
    EmbeddingRequest,
    EmbeddingResponse,
)

logger = get_logger(__name__)


class OllamaProvider(AIProvider):
    """Ollama local LLM provider.

    Connects to a running Ollama server (default: http://localhost:11434).
    Supports any model available in the local Ollama installation.

    Config env vars:
        OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
        OLLAMA_MODEL: Default model name (default: llama3.2)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL.
            model: Default model to use.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=120.0,
        )
        logger.info("Ollama provider initialized", base_url=base_url, model=model)

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return "ollama"

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat request to Ollama.

        Args:
            request: Chat request.

        Returns:
            ChatResponse: Model response.
        """
        model = request.model or self._model
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }

        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            content = data.get("message", {}).get("content", "")
            return ChatResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                usage=ChatUsage(
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                ),
                finish_reason=data.get("done_reason"),
            )

        except httpx.ConnectError as exc:
            raise ProviderException(
                f"Cannot connect to Ollama at {self._base_url}. Is Ollama running?",
                provider="ollama",
                operation="chat",
            ) from exc
        except Exception as exc:
            raise ProviderException(str(exc), provider="ollama", operation="chat") from exc

    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Stream a chat response from Ollama.

        Args:
            request: Chat request.

        Yields:
            str: Response tokens.
        """
        import json

        model = request.model or self._model
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
        except Exception as exc:
            raise ProviderException(str(exc), provider="ollama", operation="stream") from exc

    async def embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using Ollama.

        Args:
            request: Embedding request.

        Returns:
            EmbeddingResponse: Embedding vectors.
        """
        model = request.model or "nomic-embed-text"
        embeddings = []

        try:
            for text in request.texts:
                response = await self._client.post(
                    "/api/embeddings",
                    json={"model": model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])

            dimensions = len(embeddings[0]) if embeddings else 0
            return EmbeddingResponse(
                embeddings=embeddings,
                model=model,
                provider=self.provider_name,
                dimensions=dimensions,
            )

        except Exception as exc:
            raise ProviderException(str(exc), provider="ollama", operation="embedding") from exc

    def get_capabilities(self) -> dict[str, bool]:
        """Return Ollama capability map."""
        return {
            "chat": True,
            "stream": True,
            "embedding": True,
            "image_generation": False,
            "speech_to_text": False,
            "text_to_speech": False,
            "vision": False,
        }
