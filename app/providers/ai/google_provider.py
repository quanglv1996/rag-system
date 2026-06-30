"""Google Generative AI provider implementation.

Implements the AIProvider interface using the Google Generative AI SDK
(Gemini). Supports chat completions, streaming, embeddings, and vision.

Maps Google-specific exceptions to domain exceptions for clean separation.
"""

from collections.abc import AsyncGenerator

import google.generativeai as genai
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


class GoogleProvider(AIProvider):
    """Google Generative AI (Gemini) provider implementation.

    Wraps the google-generativeai SDK and maps capabilities to
    the AIProvider interface. Supports Gemini 1.5 Pro/Flash models.

    Attributes:
        _model_name: Default Gemini model identifier.
        _embedding_model_name: Default embedding model identifier.
    """

    def __init__(self) -> None:
        """Initialize the Google provider from application settings."""
        settings = get_settings()

        if not settings.google_api_key:
            raise ProviderException(
                "Google API key is not configured",
                provider="google",
                operation="init",
            )

        # Configure the global genai client
        genai.configure(api_key=settings.google_api_key)

        self._model_name = settings.google_model
        self._embedding_model_name = settings.google_embedding_model

        logger.info(
            "Google provider initialized",
            model=self._model_name,
            embedding_model=self._embedding_model_name,
        )

    @property
    def provider_name(self) -> str:
        """Return provider identifier.

        Returns:
            str: 'google'
        """
        return "google"

    def _build_generation_config(
        self, temperature: float, max_tokens: int
    ) -> genai.types.GenerationConfig:
        """Build Gemini generation configuration.

        Args:
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.

        Returns:
            genai.types.GenerationConfig: Configured generation parameters.
        """
        return genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send a chat completion request to Gemini.

        Args:
            request: Chat request with messages and parameters.

        Returns:
            ChatResponse: Completed response from the model.

        Raises:
            RateLimitException: If Google rate limits are exceeded.
            ProviderException: For any other API error.
        """
        model_name = request.model or self._model_name
        model = genai.GenerativeModel(model_name)

        # Build message history for Gemini's expected format
        history: list[dict[str, str]] = []
        last_user_message = ""

        for msg in request.messages:
            if msg.role.value == "system":
                # Gemini doesn't have a system role — prepend to first user message
                last_user_message = f"[System]: {msg.content}\n\n"
            elif msg.role.value == "user":
                history.append(
                    {"role": "user", "parts": [last_user_message + msg.content]}
                )
                last_user_message = ""
            elif msg.role.value == "assistant":
                history.append({"role": "model", "parts": [msg.content]})

        # The last user message drives the current turn
        current_message = history.pop()["parts"][0] if history else last_user_message

        generation_config = self._build_generation_config(
            request.temperature, request.max_tokens
        )

        logger.debug(
            "Sending chat request to Gemini",
            model=model_name,
            message_count=len(request.messages),
        )

        try:
            chat_session = model.start_chat(history=history[:-1] if len(history) > 1 else [])
            response = await chat_session.send_message_async(
                current_message,
                generation_config=generation_config,
            )

            usage_metadata = response.usage_metadata
            return ChatResponse(
                content=response.text,
                model=model_name,
                provider=self.provider_name,
                usage=ChatUsage(
                    prompt_tokens=usage_metadata.prompt_token_count or 0,
                    completion_tokens=usage_metadata.candidates_token_count or 0,
                    total_tokens=usage_metadata.total_token_count or 0,
                ),
                finish_reason=str(response.candidates[0].finish_reason)
                if response.candidates
                else None,
            )

        except Exception as exc:
            error_str = str(exc).lower()
            if "quota" in error_str or "rate" in error_str:
                raise RateLimitException("Google API rate limit exceeded") from exc
            raise ProviderException(
                f"Google chat error: {exc}",
                provider="google",
                operation="chat",
            ) from exc

    async def stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Stream a Gemini chat response token by token.

        Args:
            request: Chat request with messages and parameters.

        Yields:
            str: Text chunks as they are generated.

        Raises:
            ProviderException: If streaming fails.
        """
        model_name = request.model or self._model_name
        model = genai.GenerativeModel(model_name)

        # Combine all messages into a single prompt for streaming
        prompt_parts: list[str] = []
        for msg in request.messages:
            if msg.role.value == "system":
                prompt_parts.append(f"[System instruction]: {msg.content}")
            elif msg.role.value == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role.value == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        full_prompt = "\n".join(prompt_parts)
        generation_config = self._build_generation_config(
            request.temperature, request.max_tokens
        )

        try:
            response = await model.generate_content_async(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as exc:
            raise ProviderException(
                f"Google streaming error: {exc}",
                provider="google",
                operation="stream",
            ) from exc

    async def embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate vector embeddings using Google embedding model.

        Args:
            request: Embedding request with input texts.

        Returns:
            EmbeddingResponse: Vectors for each input text.

        Raises:
            ProviderException: If embedding generation fails.
        """
        model_name = request.model or self._embedding_model_name

        logger.debug(
            "Generating embeddings via Google",
            model=model_name,
            text_count=len(request.texts),
        )

        try:
            embeddings: list[list[float]] = []
            for text in request.texts:
                result = genai.embed_content(
                    model=model_name,
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(result["embedding"])

            dimensions = len(embeddings[0]) if embeddings else 0

            return EmbeddingResponse(
                embeddings=embeddings,
                model=model_name,
                provider=self.provider_name,
                dimensions=dimensions,
            )

        except Exception as exc:
            raise ProviderException(
                f"Google embedding error: {exc}",
                provider="google",
                operation="embedding",
            ) from exc

    async def vision(self, request: VisionRequest) -> VisionResponse:
        """Analyze an image using Gemini vision capabilities.

        Args:
            request: Vision request with image data and prompt.

        Returns:
            VisionResponse: Analysis result.

        Raises:
            ProviderException: If vision analysis fails.
            ValueError: If neither image_url nor image_base64 is provided.
        """
        if not request.image_url and not request.image_base64:
            raise ValueError("Either image_url or image_base64 must be provided")

        model_name = request.model or "gemini-1.5-pro-vision"
        model = genai.GenerativeModel(model_name)

        try:
            import base64

            if request.image_base64:
                image_data = base64.b64decode(request.image_base64)
                image_part = {"mime_type": "image/jpeg", "data": image_data}
            else:
                # For URL-based images, use httpx to fetch first
                import httpx

                async with httpx.AsyncClient() as client:
                    img_response = await client.get(request.image_url or "")
                    image_data = img_response.content
                    mime_type = img_response.headers.get(
                        "content-type", "image/jpeg"
                    ).split(";")[0]
                image_part = {"mime_type": mime_type, "data": image_data}

            response = await model.generate_content_async(
                [request.prompt, image_part]
            )

            return VisionResponse(
                content=response.text,
                model=model_name,
                provider=self.provider_name,
            )

        except Exception as exc:
            raise ProviderException(
                f"Google vision error: {exc}",
                provider="google",
                operation="vision",
            ) from exc

    def get_capabilities(self) -> dict[str, bool]:
        """Return Google provider capability map.

        Returns:
            dict[str, bool]: Capabilities supported by this provider.
        """
        return {
            "chat": True,
            "stream": True,
            "embedding": True,
            "image_generation": False,
            "speech_to_text": False,
            "text_to_speech": False,
            "vision": True,
        }
