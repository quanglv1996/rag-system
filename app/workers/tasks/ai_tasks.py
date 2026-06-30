"""AI background tasks — chat, embedding, image generation."""

from __future__ import annotations

from typing import Any

from celery import Task

from app.workers.celery_app import celery_app


class AITask(Task):
    """Base class for AI tasks with lazy provider initialization."""

    _provider: Any = None

    @property
    def provider(self) -> Any:
        """Lazily initialize the AI provider."""
        if self._provider is None:
            from app.core.config import get_settings
            from app.providers.ai.factory import AIProviderFactory

            settings = get_settings()
            self._provider = AIProviderFactory.create(settings.llm_provider)
        return self._provider


@celery_app.task(
    bind=True,
    base=AITask,
    name="app.workers.tasks.ai_tasks.generate_chat_response",
    max_retries=3,
    default_retry_delay=5,
    queue="ai",
)
def generate_chat_response(
    self: Task,
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Background task to generate a chat response.

    Args:
        messages: List of {role, content} message dicts.
        model: Optional model override.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        task_id: Optional caller-supplied task tracking ID.

    Returns:
        dict: Contains 'content', 'model', 'provider', 'tokens_used'.
    """
    import asyncio

    from app.core.logger import get_logger
    from app.schemas.ai import ChatMessage
    from app.common.enums import MessageRole

    logger = get_logger(__name__)
    logger.info("AI chat task started", task_id=task_id or self.request.id)

    async def _run() -> dict[str, Any]:
        from app.services.ai_service import AIService

        service = AIService(provider=self.provider)
        chat_messages = [
            ChatMessage(role=MessageRole(m["role"]), content=m["content"])
            for m in messages
        ]
        response = await service.chat(
            messages=chat_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "content": response.content,
            "model": response.model,
            "provider": response.provider,
            "tokens_used": response.usage.total_tokens,
        }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error("AI chat task failed", error=str(exc))
        raise self.retry(exc=exc, countdown=5 * (2 ** self.request.retries))


@celery_app.task(
    bind=True,
    base=AITask,
    name="app.workers.tasks.ai_tasks.generate_embeddings_batch",
    max_retries=3,
    default_retry_delay=5,
    queue="ai",
)
def generate_embeddings_batch(
    self: Task,
    texts: list[str],
    model: str | None = None,
) -> dict[str, Any]:
    """Generate embeddings for a batch of texts in the background.

    Args:
        texts: List of strings to embed.
        model: Optional embedding model override.

    Returns:
        dict: Contains 'embeddings' (list of vectors) and 'dimensions'.
    """
    import asyncio

    async def _run() -> dict[str, Any]:
        from app.services.ai_service import AIService

        service = AIService(provider=self.provider)
        response = await service.generate_embeddings(texts=texts, model=model)
        return {
            "embeddings": response.embeddings,
            "dimensions": response.dimensions,
            "model": response.model,
        }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5 * (2 ** self.request.retries))


@celery_app.task(
    bind=True,
    name="app.workers.tasks.ai_tasks.generate_image_task",
    max_retries=2,
    queue="ai",
)
def generate_image_task(
    self: Task,
    prompt: str,
    size: str = "1024x1024",
    n: int = 1,
) -> dict[str, Any]:
    """Generate images from a prompt in the background.

    Args:
        prompt: Image description.
        size: Output dimensions.
        n: Number of images.

    Returns:
        dict: Contains 'images' (list of URLs).
    """
    import asyncio

    async def _run() -> dict[str, Any]:
        from app.core.config import get_settings
        from app.providers.ai.factory import AIProviderFactory
        from app.services.ai_service import AIService

        settings = get_settings()
        provider = AIProviderFactory.create(settings.llm_provider)
        service = AIService(provider=provider)
        response = await service.generate_image(prompt=prompt, size=size, n=n)
        return {"images": response.images}

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc)
