"""Memory system for AI agents and conversations.

Provides pluggable memory backends:
- ConversationMemory: In-memory sliding window of recent messages.
- SessionMemory: Redis-backed per-session key-value store.
- SummaryMemory: Periodically summarizes history to manage token limits.
- VectorMemory: Long-term semantic memory via vector store.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from app.core.logger import get_logger

logger = get_logger(__name__)


class BaseMemory(ABC):
    """Abstract base for all memory implementations."""

    @abstractmethod
    async def add(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a message to memory.

        Args:
            role: Message role (user/assistant/system).
            content: Message content.
            metadata: Optional metadata.
        """
        ...

    @abstractmethod
    async def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Retrieve conversation history.

        Args:
            limit: Optional max number of messages to return (most recent).

        Returns:
            list[dict]: Message dicts with role and content.
        """
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored messages."""
        ...


class ConversationMemory(BaseMemory):
    """In-memory sliding window conversation history.

    Maintains the most recent N messages in memory.
    Suitable for single-session use where persistence is not required.

    Attributes:
        _messages: List of message dicts.
        _max_messages: Maximum messages to retain.
    """

    def __init__(self, max_messages: int = 50) -> None:
        """Initialize conversation memory.

        Args:
            max_messages: Maximum number of messages to retain.
        """
        self._messages: list[dict[str, Any]] = []
        self._max_messages = max_messages

    async def add(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a message, evicting oldest if at capacity.

        Args:
            role: Message role.
            content: Message text.
            metadata: Optional metadata attached to the message.
        """
        self._messages.append(
            {"role": role, "content": content, "metadata": metadata or {}}
        )
        # Evict oldest messages if over limit
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]

    async def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return conversation history.

        Args:
            limit: Optional max messages (most recent).

        Returns:
            list[dict]: Message history.
        """
        if limit:
            return self._messages[-limit:]
        return self._messages.copy()

    async def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()


class SessionMemory(BaseMemory):
    """Redis-backed per-session key-value memory.

    Persists conversation history in Redis with TTL.
    Suitable for web sessions where memory must survive request boundaries.

    Attributes:
        _redis: Async Redis client.
        _session_id: Unique session identifier.
        _ttl: Cache TTL in seconds.
    """

    def __init__(
        self,
        redis: Any,
        session_id: str,
        ttl: int = 86400,
    ) -> None:
        """Initialize session memory.

        Args:
            redis: Async Redis client.
            session_id: Unique session key.
            ttl: Storage TTL in seconds (default: 24 hours).
        """
        self._redis = redis
        self._session_id = session_id
        self._ttl = ttl
        self._key = f"memory:session:{session_id}"

    async def add(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Append a message to the session.

        Args:
            role: Message role.
            content: Message text.
            metadata: Optional metadata.
        """
        history = await self.get_history()
        history.append({"role": role, "content": content, "metadata": metadata or {}})

        await self._redis.setex(
            self._key,
            self._ttl,
            json.dumps(history),
        )

    async def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Load conversation history from Redis.

        Args:
            limit: Optional max messages.

        Returns:
            list[dict]: Message history.
        """
        raw = await self._redis.get(self._key)
        if not raw:
            return []

        history: list[dict[str, Any]] = json.loads(raw)
        if limit:
            return history[-limit:]
        return history

    async def clear(self) -> None:
        """Delete the session from Redis."""
        await self._redis.delete(self._key)


class SummaryMemory(BaseMemory):
    """Memory that summarizes old messages to manage token limits.

    Keeps the last N messages verbatim. Older messages are periodically
    summarized by the LLM and stored as a compact summary string.

    Attributes:
        _recent: ConversationMemory for recent messages.
        _summary: Current summary of older messages.
        _ai_provider: Provider for generating summaries.
        _summary_threshold: Summarize when this many messages are exceeded.
    """

    def __init__(
        self,
        ai_provider: Any,
        max_recent: int = 10,
        summary_threshold: int = 20,
    ) -> None:
        """Initialize summary memory.

        Args:
            ai_provider: AI provider for summarization.
            max_recent: Recent messages to keep verbatim.
            summary_threshold: Trigger summarization at this count.
        """
        self._recent = ConversationMemory(max_messages=max_recent)
        self._summary: str = ""
        self._ai_provider = ai_provider
        self._summary_threshold = summary_threshold
        self._total_added = 0

    async def add(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a message and summarize old history if needed.

        Args:
            role: Message role.
            content: Message content.
            metadata: Optional metadata.
        """
        await self._recent.add(role, content, metadata)
        self._total_added += 1

        if self._total_added % self._summary_threshold == 0:
            await self._summarize()

    async def get_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return summary + recent messages.

        Args:
            limit: Optional limit on recent messages.

        Returns:
            list[dict]: Combined message history.
        """
        history: list[dict[str, Any]] = []

        if self._summary:
            history.append({
                "role": "system",
                "content": f"[Conversation Summary]: {self._summary}",
            })

        recent = await self._recent.get_history(limit)
        history.extend(recent)
        return history

    async def clear(self) -> None:
        """Clear all memory and reset summary."""
        await self._recent.clear()
        self._summary = ""
        self._total_added = 0

    async def _summarize(self) -> None:
        """Generate a summary of the current conversation history."""
        history = await self._recent.get_history()
        if not history:
            return

        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        )

        prompt = (
            f"Summarize the following conversation concisely, preserving key facts:\n\n"
            f"{conversation_text}\n\nSummary:"
        )

        try:
            from app.schemas.ai import ChatMessage
            from app.common.enums import MessageRole

            request_messages = [ChatMessage(role=MessageRole.USER, content=prompt)]

            from app.schemas.ai import ChatRequest

            response = await self._ai_provider.chat(
                ChatRequest(messages=request_messages, max_tokens=512, temperature=0.3)
            )
            self._summary = response.content
            logger.debug("Memory summary updated", summary_length=len(self._summary))
        except Exception as exc:
            logger.warning("Memory summarization failed", error=str(exc))
