"""Application-wide constants.

Centralizes all magic numbers, string literals, and configuration
constants to avoid duplication and simplify maintenance.
"""

# =============================================================================
# HTTP
# =============================================================================

# Default request timeout in seconds
HTTP_DEFAULT_TIMEOUT: float = 30.0

# Default number of retry attempts for failed HTTP requests
HTTP_DEFAULT_RETRIES: int = 3

# Exponential backoff multiplier (seconds)
HTTP_BACKOFF_FACTOR: float = 2.0

# Minimum seconds between retries
HTTP_MIN_RETRY_WAIT: float = 1.0

# Maximum seconds between retries
HTTP_MAX_RETRY_WAIT: float = 60.0

# =============================================================================
# RAG
# =============================================================================

# Default text chunk size in characters
RAG_DEFAULT_CHUNK_SIZE: int = 1000

# Default overlap between consecutive chunks
RAG_DEFAULT_CHUNK_OVERLAP: int = 200

# Number of top-k chunks to retrieve per query
RAG_DEFAULT_TOP_K: int = 5

# Minimum similarity score threshold for retrieval (0.0 - 1.0)
RAG_MIN_SIMILARITY_SCORE: float = 0.0

# Maximum document size (bytes) allowed for upload
RAG_MAX_DOCUMENT_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

# Supported MIME types for document ingestion
RAG_SUPPORTED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
        "text/html",
    }
)

# =============================================================================
# AI
# =============================================================================

# Default temperature for chat completions
AI_DEFAULT_TEMPERATURE: float = 0.7

# Default max tokens for chat completions
AI_DEFAULT_MAX_TOKENS: int = 2048

# Embedding vector dimension for OpenAI text-embedding-3-small
OPENAI_EMBEDDING_DIMENSION: int = 1536

# Embedding vector dimension for Google embedding-001
GOOGLE_EMBEDDING_DIMENSION: int = 768

# =============================================================================
# Cache
# =============================================================================

# Cache TTL for embeddings (24 hours)
CACHE_EMBEDDING_TTL: int = 86400

# Cache TTL for chat responses (1 hour)
CACHE_CHAT_TTL: int = 3600

# Cache TTL for social platform tokens (1 hour)
CACHE_TOKEN_TTL: int = 3600

# Cache key prefixes
CACHE_PREFIX_EMBEDDING: str = "embedding:"
CACHE_PREFIX_CHAT: str = "chat:"
CACHE_PREFIX_TOKEN: str = "token:"
CACHE_PREFIX_RATE_LIMIT: str = "rate_limit:"

# =============================================================================
# Authentication
# =============================================================================

# Maximum allowed failed login attempts before lockout
AUTH_MAX_LOGIN_ATTEMPTS: int = 5

# Account lockout duration in seconds (15 minutes)
AUTH_LOCKOUT_DURATION: int = 900

# =============================================================================
# Facebook API
# =============================================================================

FACEBOOK_BASE_URL: str = "https://graph.facebook.com"
FACEBOOK_OAUTH_URL: str = "https://www.facebook.com/dialog/oauth"
FACEBOOK_TOKEN_URL: str = "https://graph.facebook.com/oauth/access_token"

# Rate limit: 200 calls per hour per page token
FACEBOOK_RATE_LIMIT_CALLS: int = 200
FACEBOOK_RATE_LIMIT_WINDOW: int = 3600  # seconds

# =============================================================================
# TikTok API
# =============================================================================

TIKTOK_BASE_URL: str = "https://open.tiktokapis.com/v2"
TIKTOK_AUTH_URL: str = "https://www.tiktok.com/v2/auth/authorize"
TIKTOK_TOKEN_URL: str = "https://open.tiktokapis.com/v2/oauth/token"

# =============================================================================
# YouTube / Google
# =============================================================================

YOUTUBE_BASE_URL: str = "https://www.googleapis.com/youtube/v3"
YOUTUBE_UPLOAD_URL: str = "https://www.googleapis.com/upload/youtube/v3/videos"
GOOGLE_AUTH_URL: str = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL: str = "https://oauth2.googleapis.com/token"

# =============================================================================
# Telegram API
# =============================================================================

TELEGRAM_BASE_URL: str = "https://api.telegram.org/bot"

# =============================================================================
# Pagination
# =============================================================================

# Default page size for list endpoints
PAGINATION_DEFAULT_PAGE_SIZE: int = 20

# Maximum page size for list endpoints
PAGINATION_MAX_PAGE_SIZE: int = 100

# =============================================================================
# Logging
# =============================================================================

# Log file rotation size (10 MB)
LOG_MAX_BYTES: int = 10 * 1024 * 1024

# Number of rotated log files to keep
LOG_BACKUP_COUNT: int = 5
