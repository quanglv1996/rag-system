"""Abstract interface for LLM wrappers used inside the RAG pipeline.

Provides a clean boundary between the RAG retriever/pipeline code
and the underlying language model, allowing the LLM to be swapped
without changing the pipeline logic.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


class LLMInterface(ABC):
    """Abstract base class for LLM wrapper implementations.

    Used by the RAG pipeline's prompt builder to call the underlying
    language model with a constructed prompt.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a text completion for a given prompt.

        Args:
            prompt: The user/query prompt to complete.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.

        Returns:
            str: Generated text response.

        Raises:
            ProviderException: If generation fails.
        """
        ...

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream a text completion token by token.

        Args:
            prompt: The user/query prompt to complete.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Yields:
            str: Individual tokens as generated.
        """
        ...
