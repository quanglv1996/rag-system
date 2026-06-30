"""Pydantic schemas for AI provider requests and responses."""

from typing import Any

from pydantic import BaseModel, Field

from app.common.enums import MessageRole


class ChatMessage(BaseModel):
    """A single message in a chat conversation.

    Attributes:
        role: Message author role (system/user/assistant).
        content: Text content of the message.
        name: Optional author name for function/tool messages.
    """

    role: MessageRole
    content: str
    name: str | None = Field(default=None)


class ChatRequest(BaseModel):
    """Request schema for chat completion.

    Attributes:
        messages: Conversation history including the current user message.
        model: Optional model override (uses provider default if None).
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens in the response.
        stream: Whether to stream the response.
        metadata: Arbitrary metadata passed through to the provider.
    """

    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=128000)
    stream: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatUsage(BaseModel):
    """Token usage statistics for a chat completion.

    Attributes:
        prompt_tokens: Tokens consumed by the prompt.
        completion_tokens: Tokens generated in the response.
        total_tokens: Sum of prompt and completion tokens.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Response schema for chat completion.

    Attributes:
        content: Generated text response.
        model: Model actually used for generation.
        provider: Provider that generated the response.
        usage: Token usage statistics.
        finish_reason: Why generation stopped (stop/length/etc.).
        metadata: Provider-specific additional data.
    """

    content: str
    model: str
    provider: str
    usage: ChatUsage = Field(default_factory=ChatUsage)
    finish_reason: str | None = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingRequest(BaseModel):
    """Request schema for text embedding.

    Attributes:
        texts: List of strings to embed.
        model: Optional model override.
    """

    texts: list[str] = Field(min_length=1)
    model: str | None = Field(default=None)


class EmbeddingResponse(BaseModel):
    """Response schema for text embedding.

    Attributes:
        embeddings: List of embedding vectors, one per input text.
        model: Model used for embedding.
        provider: Provider that generated the embeddings.
        dimensions: Vector dimension count.
        usage: Token usage statistics.
    """

    embeddings: list[list[float]]
    model: str
    provider: str
    dimensions: int
    usage: ChatUsage = Field(default_factory=ChatUsage)


class ImageGenerationRequest(BaseModel):
    """Request schema for image generation.

    Attributes:
        prompt: Text description of the desired image.
        model: Optional model override.
        size: Desired image dimensions (e.g., '1024x1024').
        quality: Image quality level (standard/hd).
        n: Number of images to generate.
        response_format: Output format (url/b64_json).
    """

    prompt: str = Field(min_length=1, max_length=4000)
    model: str | None = Field(default=None)
    size: str = Field(default="1024x1024")
    quality: str = Field(default="standard")
    n: int = Field(default=1, ge=1, le=10)
    response_format: str = Field(default="url")


class ImageGenerationResponse(BaseModel):
    """Response schema for image generation.

    Attributes:
        images: List of generated image URLs or base64 data.
        model: Model used for generation.
        provider: Provider that generated the images.
    """

    images: list[str]
    model: str
    provider: str


class SpeechToTextRequest(BaseModel):
    """Request schema for speech-to-text transcription.

    Attributes:
        audio_bytes: Raw audio data as bytes.
        language: Optional BCP-47 language code (auto-detect if None).
        model: Optional model override.
        prompt: Optional context to guide transcription.
    """

    audio_bytes: bytes
    language: str | None = Field(default=None)
    model: str | None = Field(default=None)
    prompt: str | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}


class SpeechToTextResponse(BaseModel):
    """Response schema for speech-to-text.

    Attributes:
        text: Transcribed text.
        language: Detected or specified language.
        duration: Audio duration in seconds.
        provider: Provider that transcribed the audio.
    """

    text: str
    language: str | None = None
    duration: float | None = None
    provider: str


class TextToSpeechRequest(BaseModel):
    """Request schema for text-to-speech synthesis.

    Attributes:
        text: Text to convert to speech.
        voice: Voice identifier.
        model: Optional model override.
        speed: Speaking speed multiplier.
        response_format: Audio format (mp3/opus/aac/flac).
    """

    text: str = Field(min_length=1, max_length=4096)
    voice: str = Field(default="alloy")
    model: str | None = Field(default=None)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    response_format: str = Field(default="mp3")


class TextToSpeechResponse(BaseModel):
    """Response schema for text-to-speech.

    Attributes:
        audio_bytes: Generated audio data.
        format: Audio format.
        provider: Provider that synthesized the audio.
    """

    audio_bytes: bytes
    format: str
    provider: str

    model_config = {"arbitrary_types_allowed": True}


class VisionRequest(BaseModel):
    """Request schema for vision/image analysis.

    Attributes:
        image_url: URL of the image to analyze (use image_base64 as alternative).
        image_base64: Base64-encoded image data.
        prompt: Instruction or question about the image.
        model: Optional model override.
        max_tokens: Maximum tokens in the response.
    """

    image_url: str | None = Field(default=None)
    image_base64: str | None = Field(default=None)
    prompt: str = Field(min_length=1)
    model: str | None = Field(default=None)
    max_tokens: int = Field(default=1024, ge=1)


class VisionResponse(BaseModel):
    """Response schema for vision analysis.

    Attributes:
        content: Text response describing or answering about the image.
        model: Model used.
        provider: Provider that analyzed the image.
        usage: Token usage statistics.
    """

    content: str
    model: str
    provider: str
    usage: ChatUsage = Field(default_factory=ChatUsage)
