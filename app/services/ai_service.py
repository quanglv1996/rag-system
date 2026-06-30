"""AI Service — business logic for AI chat, embeddings, and multimodal tasks.

Orchestrates AI provider calls, handles conversation history,
implements retry logic, and enforces business rules.
No HTTP or database code should appear here.
"""

from collections.abc import AsyncGenerator
from typing import Any

from app.core.exception import ProviderException, ValidationException
from app.core.logger import TimingLogger, get_logger
from app.interfaces.ai_provider import AIProvider
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
from app.common.enums import MessageRole

logger = get_logger(__name__)


class AIService:
    """Service layer for all AI provider operations.

    Contains the business logic for AI interactions. Depends on the
    AIProvider interface — not on any concrete provider class.

    Attributes:
        _provider: AI provider implementing the AIProvider interface.
    """

    def __init__(self, provider: AIProvider) -> None:
        """Initialize the AI service with a provider.

        Args:
            provider: AIProvider implementation (OpenAI, Google, etc.).
        """
        self._provider = provider
        logger.info("AIService initialized", provider=provider.provider_name)

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
    ) -> ChatResponse:
        """Process a chat completion request.

        Optionally prepends a system message to the conversation.
        Validates input before forwarding to the provider.

        Args:
            messages: Conversation history including the latest user message.
            model: Optional model override.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum response tokens.
            system_prompt: Optional system instruction to prepend.

        Returns:
            ChatResponse: AI-generated response.

        Raises:
            ValidationException: If the message list is empty.
            ProviderException: If the AI API call fails.
        """
        if not messages:
            raise ValidationException("At least one message is required", field="messages")

        # Build full message list with optional system prompt
        full_messages = list(messages)
        if system_prompt:
            full_messages.insert(
                0, ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            )

        request = ChatRequest(
            messages=full_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        with TimingLogger("ai_chat", logger):
            response = await self._provider.chat(request)

        logger.info(
            "Chat completed",
            provider=response.provider,
            model=response.model,
            total_tokens=response.usage.total_tokens,
        )

        return response

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion response token by token.

        Args:
            messages: Conversation history.
            model: Optional model override.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            system_prompt: Optional system instruction.

        Yields:
            str: Individual response tokens.

        Raises:
            ValidationException: If messages are empty.
        """
        if not messages:
            raise ValidationException("At least one message is required", field="messages")

        full_messages = list(messages)
        if system_prompt:
            full_messages.insert(
                0, ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
            )

        request = ChatRequest(
            messages=full_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for token in self._provider.stream(request):
            yield token

    async def generate_embeddings(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> EmbeddingResponse:
        """Generate vector embeddings for a list of texts.

        Args:
            texts: Strings to embed. Must not be empty.
            model: Optional model override.

        Returns:
            EmbeddingResponse: Embedding vectors and metadata.

        Raises:
            ValidationException: If texts list is empty.
            ProviderException: If the embedding API call fails.
        """
        if not texts:
            raise ValidationException("At least one text is required", field="texts")

        # Remove empty strings
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            raise ValidationException("All texts are empty", field="texts")

        request = EmbeddingRequest(texts=valid_texts, model=model)

        with TimingLogger("ai_embedding", logger):
            response = await self._provider.embedding(request)

        return response

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> ImageGenerationResponse:
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the desired image.
            model: Optional model override.
            size: Image dimensions.
            quality: Image quality level.
            n: Number of images to generate.

        Returns:
            ImageGenerationResponse: Generated image URLs.

        Raises:
            ValidationException: If prompt is empty.
            ProviderException: If image generation fails or is unsupported.
        """
        if not prompt.strip():
            raise ValidationException("Image prompt must not be empty", field="prompt")

        request = ImageGenerationRequest(
            prompt=prompt, model=model, size=size, quality=quality, n=n
        )
        return await self._provider.image_generation(request)

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        model: str | None = None,
    ) -> SpeechToTextResponse:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio file bytes.
            language: Optional BCP-47 language code.
            model: Optional model override.

        Returns:
            SpeechToTextResponse: Transcription result.

        Raises:
            ValidationException: If audio_bytes is empty.
            ProviderException: If transcription fails.
        """
        if not audio_bytes:
            raise ValidationException("Audio bytes must not be empty", field="audio_bytes")

        request = SpeechToTextRequest(
            audio_bytes=audio_bytes,
            language=language,
            model=model,
        )
        return await self._provider.speech_to_text(request)

    async def synthesize_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str | None = None,
        speed: float = 1.0,
    ) -> TextToSpeechResponse:
        """Convert text to speech audio.

        Args:
            text: Text to synthesize.
            voice: Voice identifier.
            model: Optional model override.
            speed: Speech speed multiplier.

        Returns:
            TextToSpeechResponse: Synthesized audio bytes.

        Raises:
            ValidationException: If text is empty.
        """
        if not text.strip():
            raise ValidationException("Text must not be empty", field="text")

        request = TextToSpeechRequest(
            text=text,
            voice=voice,
            model=model,
            speed=speed,
        )
        return await self._provider.text_to_speech(request)

    async def analyze_image(
        self,
        prompt: str,
        image_url: str | None = None,
        image_base64: str | None = None,
        model: str | None = None,
    ) -> VisionResponse:
        """Analyze an image with a text prompt.

        Args:
            prompt: Instruction or question about the image.
            image_url: URL of the image (mutually exclusive with image_base64).
            image_base64: Base64-encoded image data.
            model: Optional model override.

        Returns:
            VisionResponse: Text analysis result.

        Raises:
            ValidationException: If neither image source is provided.
        """
        if not image_url and not image_base64:
            raise ValidationException(
                "Either image_url or image_base64 must be provided",
                field="image",
            )

        request = VisionRequest(
            image_url=image_url,
            image_base64=image_base64,
            prompt=prompt,
            model=model,
        )
        return await self._provider.vision(request)

    def get_provider_capabilities(self) -> dict[str, bool]:
        """Return the current provider's capability map.

        Returns:
            dict[str, bool]: Capability name → availability.
        """
        return self._provider.get_capabilities()

    @property
    def provider_name(self) -> str:
        """Return the active provider name.

        Returns:
            str: Provider identifier.
        """
        return self._provider.provider_name
