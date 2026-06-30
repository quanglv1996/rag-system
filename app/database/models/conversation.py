"""Conversation history ORM model."""

import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin


class Conversation(Base, TimestampMixin):
    """Database model for AI chat conversations.

    Attributes:
        id: UUID primary key.
        user_id: Optional user ID if conversation is user-scoped.
        title: Optional conversation title.
        provider: AI provider used (openai, google).
        model: Specific model used.
    """

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4, index=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)

    messages: Mapped[list["ConversationMessage"]] = relationship(
        "ConversationMessage", back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"Conversation(id={self.id!r}, provider={self.provider!r})"


class ConversationMessage(Base, TimestampMixin):
    """Database model for individual messages in a conversation.

    Attributes:
        id: UUID primary key.
        conversation_id: Foreign key to parent Conversation.
        role: Message role (system, user, assistant).
        content: Message text content.
        token_count: Number of tokens used.
    """

    __tablename__ = "conversation_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, default=uuid.uuid4, index=True
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user/assistant/system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(nullable=True)

    conversation: Mapped[Conversation] = relationship(
        "Conversation", back_populates="messages"
    )

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"ConversationMessage(id={self.id!r}, role={self.role!r})"
