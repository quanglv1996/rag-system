"""OpenAI provider implementation.

Implements the AIProvider interface using the official OpenAI Python SDK.
Supports chat completions, streaming, embeddings, image generation,
speech-to-text, text-to-speech, and vision (GPT-4o).

All API errors are mapped to domain exceptions to keep callers
decoupled from the OpenAI SDK's exception hierarchy.
"""

from collections.abc import AsyncGenerator

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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
    ImageGenerationRequest,
    ImageGenerationResponse,
    SpeechToTextRequest,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
    VisionRequest,
    VisionResponse,
)

logger = get_logger(__name__)

# Retry configuration: 3 attempts with exponential backoff
_RETRY_CONFIG = dict(
    retry=retry_if_exception_type((openai.APIConnectionError, openai.APITimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)


class OpenAIProvider(AIProvider):
    """OpenAI API provider implementation.

    Wraps the OpenAI async client and maps all capabilities to
    the AIProvider interface. Handles retries, rate limits, and
    error mapping automatically.

    Attributes:
        _client: Async OpenAI client instance.
        _model: Default chat model to use.
        _embedding_model: Default embedding model to use.
    """

    def __init__(self) -> None:
        """Initialize the OpenAI provider from application settings."""
        settings = get_settings()

        if not settings.openai_api_key:
            raise ProviderException(
                "OpenAI API key is not configured",
                provider="openai",
                operation="init",
            )

        self._client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.http_timeout,
            max_retries=0,  # We handle retries via tenacity
        )
        self._model = settings.openai_model
        self._embedding_model = settings.openai_embedding_model

        logger.info(
            "OpenAI provider initialized",
            model=self._model,
            embedding_model=self._embedding_model,
        )

    @property
    def provider_name(self) -> str:
        """Return provider identifier.

        Returns:
            str: 'openai'
        """
        return "openai"

    @retry(**_RETRY_CONFIG)
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request to OpenAI.

        Args:
            request: Chat request with messages and parameters.

        Returns:
            ChatResponse: Completed response from the model.

        Raises:
            RateLimitException: If OpenAI rate limits are exceeded.
            ProviderException: For any other API error.
        """
        model = request.model or self._model
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        logger.debug(
            "Sending chat request to OpenAI",
            model=model,
            message_count=len(messages),
        )

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )

            choice = response.choices[0]
            usage = response.usage

            return ChatResponse(
                content=choice.message.content or "",
                model=response.model,
                provider=self.provider_name,
                usage=ChatUsage(
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                ),
                finish_reason=choice.finish_reason,
            )

        except openai.RateLimitError as exc:
            raise RateLimitException(
                message="OpenAI rate limit exceeded",
                retry_after=60,
            ) from exc
        except openai.AuthenticationError as exc:
            raise ProviderException(
                "OpenAI authentication failed — check OPENAI_API_KEY",
                provider="openai",
                operation="chat",
            ) from exc
        except openai.APIError as exc:
            raise ProviderException(
                f"OpenAI API error: {exc.message}",
                provider="openai",
                operation="chat",
            ) from exc

    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Stream a chat completion response token by token.

        Args:
            request: Chat request with messages and parameters.

        Yields:
            str: Individual text tokens as generated.

        Raises:
            RateLimitException: If OpenAI rate limits are exceeded.
            ProviderException: For any other API error.
        """
        model = request.model or self._model
        messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        try:
            async with await self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

        except openai.RateLimitError as exc:
            raise RateLimitException("OpenAI rate limit exceeded") from exc
        except openai.APIError as exc:
            raise ProviderException(
                f"OpenAI streaming error: {exc.message}",
                provider="openai",
                operation="stream",
            ) from exc

    @retry(**_RETRY_CONFIG)
    async def embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate vector embeddings using OpenAI.

        Args:
            request: Embedding request with input texts.

        Returns:
            EmbeddingResponse: Vectors for each input text.

        Raises:
            ProviderException: If the embedding API call fails.
        """
        model = request.model or self._embedding_model

        logger.debug(
            "Generating embeddings via OpenAI",
            model=model,
            text_count=len(request.texts),
        )

        try:
            response = await self._client.embeddings.create(
                model=model,
                input=request.texts,
            )

            embeddings = [item.embedding for item in response.data]
            dimensions = len(embeddings[0]) if embeddings else 0
            usage = response.usage

            return EmbeddingResponse(
                embeddings=embeddings,
                model=response.model,
                provider=self.provider_name,
                dimensions=dimensions,
                usage=ChatUsage(
                    prompt_tokens=usage.prompt_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )

        except openai.RateLimitError as exc:
            raise RateLimitException("OpenAI rate limit exceeded") from exc
        except openai.APIError as exc:
            raise ProviderException(
                f"OpenAI embedding error: {exc.message}",
                provider="openai",
                operation="embedding",
            ) from exc

    @retry(**_RETRY_CONFIG)
    async def image_generation(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """Generate images using DALL-E.

        Args:
            request: Image generation request.

        Returns:
            ImageGenerationResponse: List of image URLs.

        Raises:
            ProviderException: If image generation fails.
        """
        model = request.model or "dall-e-3"

        try:
            response = await self._client.images.generate(
                model=model,
                prompt=request.prompt,
                size=request.size,  # type: ignore[arg-type]
                quality=request.quality,  # type: ignore[arg-type]
                n=request.n,
                response_format=request.response_format,  # type: ignore[arg-type]
            )

            images: list[str] = []
            for img in response.data:
                if request.response_format == "url" and img.url:
                    images.append(img.url)
                elif img.b64_json:
                    images.append(img.b64_json)

            return ImageGenerationResponse(
                images=images,
                model=model,
                provider=self.provider_name,
            )

        except openai.APIError as exc:
            raise ProviderException(
                f"OpenAI image generation error: {exc.message}",
                provider="openai",
                operation="image_generation",
            ) from exc

    async def speech_to_text(
        self, request: SpeechToTextRequest
    ) -> SpeechToTextResponse:
        """Transcribe audio using OpenAI Whisper.

        Args:
            request: Speech-to-text request with audio data.

        Returns:
            SpeechToTextResponse: Transcribed text.

        Raises:
            ProviderException: If transcription fails.
        """
        try:
            # Whisper API expects a file-like object
            import io

            audio_file = io.BytesIO(request.audio_bytes)
            audio_file.name = "audio.mp3"

            response = await self._client.audio.transcriptions.create(
                model=request.model or "whisper-1",
                file=audio_file,
                language=request.language,
                prompt=request.prompt,
            )

            return SpeechToTextResponse(
                text=response.text,
                language=request.language,
                provider=self.provider_name,
            )

        except openai.APIError as exc:
            raise ProviderException(
                f"OpenAI speech-to-text error: {exc.message}",
                provider="openai",
                operation="speech_to_text",
            ) from exc

    async def text_to_speech(
        self, request: TextToSpeechRequest
    ) -> TextToSpeechResponse:
        """Synthesize speech using OpenAI TTS.

        Args:
            request: Text-to-speech request.

        Returns:
            TextToSpeechResponse: Audio bytes.

        Raises:
            ProviderException: If synthesis fails.
        """
        try:
            response = await self._client.audio.speech.create(
                model=request.model or "tts-1",
                voice=request.voice,  # type: ignore[arg-type]
                input=request.text,
                speed=request.speed,
                response_format=request.response_format,  # type: ignore[arg-type]
            )

            return TextToSpeechResponse(
                audio_bytes=response.content,
                format=request.response_format,
                provider=self.provider_name,
            )

        except openai.APIError as exc:
            raise ProviderException(
                f"OpenAI TTS error: {exc.message}",
                provider="openai",
                operation="text_to_speech",
            ) from exc

    async def vision(self, request: VisionRequest) -> VisionResponse:
        """Analyze an image using GPT-4o vision.

        Args:
            request: Vision request with image and prompt.

        Returns:
            VisionResponse: Analysis result text.

        Raises:
            ProviderException: If vision analysis fails.
            ValueError: If neither image_url nor image_base64 is provided.
        """
        if not request.image_url and not request.image_base64:
            raise ValueError("Either image_url or image_base64 must be provided")

        # Build the image content part
        if request.image_url:
            image_content = {
                "type": "image_url",
                "image_url": {"url": request.image_url},
            }
        else:
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{request.image_base64}"
                },
            }

        model = request.model or "gpt-4o"

        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": request.prompt},
                            image_content,
                        ],
                    }
                ],
                max_tokens=request.max_tokens,
            )

            choice = response.choices[0]
            usage = response.usage

            return VisionResponse(
                content=choice.message.content or "",
                model=response.model,
                provider=self.provider_name,
                usage=ChatUsage(
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                ),
            )

        except openai.APIError as exc:
            raise ProviderException(
                f"OpenAI vision error: {exc.message}",
                provider="openai",
                operation="vision",
            ) from exc

    def get_capabilities(self) -> dict[str, bool]:
        """Return OpenAI capability map.

        Returns:
            dict[str, bool]: All capabilities supported by OpenAI.
        """
        return {
            "chat": True,
            "stream": True,
            "embedding": True,
            "image_generation": True,
            "speech_to_text": True,
            "text_to_speech": True,
            "vision": True,
        }
