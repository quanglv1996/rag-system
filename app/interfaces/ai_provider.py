"""Abstract interface for AI providers.

Defines the contract that all AI provider implementations (OpenAI, Google,
Anthropic, etc.) must satisfy. Services depend on this interface, never
on concrete implementations — enabling the Strategy Pattern.

Concrete providers are selected via AIProviderFactory based on config.
New providers can be added by implementing this ABC without touching
any existing service or router code.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from app.schemas.ai import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    SpeechToTextRequest,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
    VisionRequest,
    VisionResponse,
)


class AIProvider(ABC):
    """Abstract base class for all AI provider implementations.

    Each method corresponds to a distinct AI capability. Providers
    implement only the capabilities they support; unsupported methods
    raise NotImplementedError.

    Subclasses must implement:
        - chat: Single-turn or multi-turn conversation.
        - stream: Streaming chat (token by token).
        - embedding: Convert text to vector embeddings.

    Subclasses may implement:
        - image_generation: Generate images from text prompts.
        - speech_to_text: Transcribe audio to text.
        - text_to_speech: Convert text to audio.
        - vision: Analyze images with text instructions.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique identifier for this provider.

        Returns:
            str: Provider name (e.g., 'openai', 'google').
        """
        ...

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request.

        Args:
            request: Structured chat request with messages and parameters.

        Returns:
            ChatResponse: The AI response with content and usage stats.

        Raises:
            ProviderException: If the API call fails.
            RateLimitException: If rate limit is exceeded.
        """
        ...

    @abstractmethod
    async def stream(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion response token by token.

        Args:
            request: Structured chat request with messages and parameters.

        Yields:
            str: Individual text tokens as they are generated.

        Raises:
            ProviderException: If the API call fails.
        """
        ...

    @abstractmethod
    async def embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate vector embeddings for a list of texts.

        Args:
            request: Embedding request with input texts and model.

        Returns:
            EmbeddingResponse: Vectors and usage statistics.

        Raises:
            ProviderException: If the API call fails.
        """
        ...

    async def image_generation(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """Generate images from a text prompt.

        Args:
            request: Image generation request with prompt and parameters.

        Returns:
            ImageGenerationResponse: Generated image URLs or base64 data.

        Raises:
            NotImplementedError: If the provider does not support this feature.
        """
        raise NotImplementedError(
            f"Provider '{self.provider_name}' does not support image generation"
        )

    async def speech_to_text(
        self, request: SpeechToTextRequest
    ) -> SpeechToTextResponse:
        """Transcribe audio data to text.

        Args:
            request: Speech-to-text request with audio bytes and language.

        Returns:
            SpeechToTextResponse: Transcribed text and metadata.

        Raises:
            NotImplementedError: If the provider does not support this feature.
        """
        raise NotImplementedError(
            f"Provider '{self.provider_name}' does not support speech-to-text"
        )

    async def text_to_speech(
        self, request: TextToSpeechRequest
    ) -> TextToSpeechResponse:
        """Convert text to speech audio.

        Args:
            request: Text-to-speech request with text and voice settings.

        Returns:
            TextToSpeechResponse: Audio bytes and metadata.

        Raises:
            NotImplementedError: If the provider does not support this feature.
        """
        raise NotImplementedError(
            f"Provider '{self.provider_name}' does not support text-to-speech"
        )

    async def vision(self, request: VisionRequest) -> VisionResponse:
        """Analyze an image with a text instruction.

        Args:
            request: Vision request with image data and instruction prompt.

        Returns:
            VisionResponse: Text description or analysis result.

        Raises:
            NotImplementedError: If the provider does not support this feature.
        """
        raise NotImplementedError(
            f"Provider '{self.provider_name}' does not support vision"
        )

    def get_capabilities(self) -> dict[str, bool]:
        """Return the capability map for this provider.

        Returns:
            dict[str, bool]: Map of capability name to availability.
        """
        return {
            "chat": True,
            "stream": True,
            "embedding": True,
            "image_generation": False,
            "speech_to_text": False,
            "text_to_speech": False,
            "vision": False,
        }
