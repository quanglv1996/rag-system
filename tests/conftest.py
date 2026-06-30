"""Shared pytest fixtures and configuration."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.common.enums import MessageRole
from app.schemas.ai import (
    ChatMessage,
    ChatResponse,
    ChatUsage,
    EmbeddingResponse,
)


# =============================================================================
# Event Loop
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Provide a session-scoped asyncio event loop."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Application Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def app():
    """Provide the FastAPI app instance for testing."""
    import os

    # Use test environment variables
    os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test_db")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
    os.environ.setdefault("SECRET_KEY", "test-secret-key-at-least-32-characters-long")

    from app.main import create_app

    return create_app()


@pytest.fixture
def client(app) -> TestClient:
    """Provide a synchronous test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest_asyncio.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Provide an async HTTP client for testing async endpoints."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# =============================================================================
# Mock AI Provider Fixture
# =============================================================================


@pytest.fixture
def mock_ai_provider():
    """Provide a mock AIProvider with pre-configured responses."""
    provider = MagicMock()
    provider.provider_name = "mock"

    # Mock chat
    provider.chat = AsyncMock(
        return_value=ChatResponse(
            content="This is a test response.",
            model="mock-model",
            provider="mock",
            usage=ChatUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            finish_reason="stop",
        )
    )

    # Mock stream
    async def mock_stream(request):
        for token in ["Hello", " ", "World"]:
            yield token

    provider.stream = mock_stream

    # Mock embedding
    provider.embedding = AsyncMock(
        return_value=EmbeddingResponse(
            embeddings=[[0.1, 0.2, 0.3] * 512],  # 1536-dim
            model="mock-embedding",
            provider="mock",
            dimensions=1536,
        )
    )

    provider.get_capabilities = MagicMock(
        return_value={
            "chat": True,
            "stream": True,
            "embedding": True,
            "image_generation": False,
            "speech_to_text": False,
            "text_to_speech": False,
            "vision": False,
        }
    )

    return provider


# =============================================================================
# Mock Vector DB Fixture
# =============================================================================


@pytest.fixture
def mock_vector_db():
    """Provide a mock VectorDatabase with pre-configured responses."""
    from app.interfaces.vector_database import VectorDocument

    db = MagicMock()
    db.provider_name = "mock_vector"

    db.add = AsyncMock(return_value=["chunk-0", "chunk-1"])
    db.search = AsyncMock(
        return_value=[
            VectorDocument(
                id="chunk-0",
                content="Test document chunk content.",
                score=0.92,
                metadata={"source": "test.pdf", "document_id": "doc-001"},
            )
        ]
    )
    db.delete = AsyncMock(return_value=2)
    db.clear = AsyncMock(return_value=None)
    db.count = AsyncMock(return_value=10)

    return db


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_chat_messages() -> list[ChatMessage]:
    """Provide sample chat messages for testing."""
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="What is 2+2?"),
    ]


@pytest.fixture
def sample_document_content() -> str:
    """Provide sample document text for RAG tests."""
    return """
    Artificial Intelligence (AI) refers to the simulation of human intelligence
    in machines. AI systems are designed to perform tasks that typically require
    human intelligence, such as learning, reasoning, problem-solving, perception,
    and language understanding.

    Machine learning is a subset of AI that enables systems to learn and improve
    from experience without being explicitly programmed. Deep learning is a type
    of machine learning that uses neural networks with many layers.

    Natural Language Processing (NLP) is another key area of AI focused on
    the interaction between computers and human language.
    """
