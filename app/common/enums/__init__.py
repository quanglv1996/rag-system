"""Application-wide enumerations.

All enums used across services, models, and API schemas are defined here
to avoid duplication and ensure consistency.
"""

from enum import Enum


class AIProvider(str, Enum):
    """Supported AI provider identifiers.

    Used in configuration and provider factory to select the correct
    implementation via Strategy Pattern.
    """

    OPENAI = "openai"
    GOOGLE = "google"


class VectorDB(str, Enum):
    """Supported vector database identifiers."""

    FAISS = "faiss"
    CHROMA = "chroma"


class MessageRole(str, Enum):
    """Chat message roles following the OpenAI convention."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class DocumentStatus(str, Enum):
    """Document processing pipeline status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Supported document types for the RAG pipeline."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "md"
    HTML = "html"


class ChunkStrategy(str, Enum):
    """Text chunking strategies for the RAG pipeline."""

    FIXED = "fixed"          # Fixed character count
    RECURSIVE = "recursive"  # LangChain RecursiveCharacterTextSplitter
    SENTENCE = "sentence"    # Split on sentence boundaries
    SEMANTIC = "semantic"    # Semantic similarity-based splitting


class SocialPlatform(str, Enum):
    """Supported social media platforms."""

    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    TELEGRAM = "telegram"


class PostStatus(str, Enum):
    """Social media post lifecycle status."""

    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    FAILED = "failed"
    DELETED = "deleted"


class Environment(str, Enum):
    """Application deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Standard logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TokenType(str, Enum):
    """JWT token types."""

    ACCESS = "access"
    REFRESH = "refresh"


class HTTPMethod(str, Enum):
    """HTTP request methods."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
