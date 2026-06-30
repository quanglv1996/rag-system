"""AI API router — chat, embeddings, image, speech, vision endpoints."""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.common.schemas import BaseResponse
from app.core.dependency import AppSettings, CurrentUserId, get_ai_service
from app.schemas.ai import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    SpeechToTextResponse,
    TextToSpeechRequest,
    VisionRequest,
    VisionResponse,
)
from app.services.ai_service import AIService

router = APIRouter(prefix="/ai", tags=["AI"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat completion",
    description="Send messages to the configured AI provider and get a response.",
)
async def chat(
    request: ChatRequest,
    ai_service: Annotated[AIService, Depends(get_ai_service)],
) -> ChatResponse:
    """Process a chat completion request.

    Args:
        request: Chat request with message history and parameters.
        ai_service: Injected AI service.

    Returns:
        ChatResponse: AI-generated response.
    """
    return await ai_service.chat(
        messages=request.messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )


@router.post(
    "/chat/stream",
    summary="Streaming chat completion",
    description="Stream chat completion tokens as Server-Sent Events.",
)
async def chat_stream(
    request: ChatRequest,
    ai_service: Annotated[AIService, Depends(get_ai_service)],
) -> StreamingResponse:
    """Stream a chat completion response.

    Args:
        request: Chat request with stream=True.
        ai_service: Injected AI service.

    Returns:
        StreamingResponse: SSE stream of tokens.
    """

    async def generate() -> AsyncGenerator[str, None]:
        async for token in ai_service.stream_chat(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/embedding",
    response_model=EmbeddingResponse,
    summary="Generate text embeddings",
    description="Convert texts to vector embeddings using the configured embedding provider.",
)
async def embedding(
    request: EmbeddingRequest,
    ai_service: Annotated[AIService, Depends(get_ai_service)],
) -> EmbeddingResponse:
    """Generate vector embeddings for a list of texts.

    Args:
        request: Embedding request with input texts.
        ai_service: Injected AI service.

    Returns:
        EmbeddingResponse: Embedding vectors and metadata.
    """
    return await ai_service.generate_embeddings(
        texts=request.texts, model=request.model
    )


@router.post(
    "/image",
    response_model=ImageGenerationResponse,
    summary="Generate images",
    description="Generate images from a text prompt using DALL-E or compatible model.",
)
async def generate_image(
    request: ImageGenerationRequest,
    ai_service: Annotated[AIService, Depends(get_ai_service)],
) -> ImageGenerationResponse:
    """Generate images from a text prompt.

    Args:
        request: Image generation request.
        ai_service: Injected AI service.

    Returns:
        ImageGenerationResponse: Generated image URLs.
    """
    return await ai_service.generate_image(
        prompt=request.prompt,
        model=request.model,
        size=request.size,
        quality=request.quality,
        n=request.n,
    )


@router.post(
    "/vision",
    response_model=VisionResponse,
    summary="Analyze image",
    description="Analyze an image with a text prompt using vision-capable models.",
)
async def analyze_image(
    request: VisionRequest,
    ai_service: Annotated[AIService, Depends(get_ai_service)],
) -> VisionResponse:
    """Analyze an image with a text prompt.

    Args:
        request: Vision request with image and prompt.
        ai_service: Injected AI service.

    Returns:
        VisionResponse: Text analysis result.
    """
    return await ai_service.analyze_image(
        prompt=request.prompt,
        image_url=request.image_url,
        image_base64=request.image_base64,
        model=request.model,
    )


@router.get(
    "/capabilities",
    summary="Get provider capabilities",
    description="Returns which AI capabilities are available for the current provider.",
)
async def get_capabilities(
    ai_service: Annotated[AIService, Depends(get_ai_service)],
) -> dict[str, bool]:
    """Return the active AI provider's capability map.

    Returns:
        dict[str, bool]: Capability name → availability.
    """
    return ai_service.get_provider_capabilities()
