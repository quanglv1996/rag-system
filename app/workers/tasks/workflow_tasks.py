"""Workflow background tasks."""

from __future__ import annotations

from typing import Any

from celery import Task

from app.workers.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="app.workers.tasks.workflow_tasks.execute_workflow",
    max_retries=1,
    soft_time_limit=1800,
    queue="workflow",
)
def execute_workflow(
    self: Task,
    workflow_definition: dict[str, Any],
    initial_context: dict[str, Any] | None = None,
    execution_id: str | None = None,
) -> dict[str, Any]:
    """Execute a complete workflow in the background.

    Args:
        workflow_definition: WorkflowDefinition as a dict.
        initial_context: Initial context variables.
        execution_id: Optional execution tracking ID.

    Returns:
        dict: Final workflow context as dict.
    """
    import asyncio

    from app.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Workflow task started", execution_id=execution_id or self.request.id)

    async def _run() -> dict[str, Any]:
        from app.core.config import get_settings
        from app.providers.ai.factory import AIProviderFactory
        from app.providers.vector.factory import VectorProviderFactory
        from app.services.ai_service import AIService
        from app.services.facebook_service import FacebookService
        from app.services.rag_service import RAGService
        from app.services.telegram_service import TelegramService
        from app.workflow.engine import WorkflowEngine
        from app.workflow.loader import WorkflowLoader

        settings = get_settings()
        ai_provider = AIProviderFactory.create(settings.llm_provider)
        vector_provider = VectorProviderFactory.create(settings.vector_db)

        dependencies = {
            "ai_service": AIService(provider=ai_provider),
            "rag_service": RAGService(
                settings=settings,
                vector_provider=vector_provider,
                ai_provider=ai_provider,
            ),
            "facebook_service": FacebookService(settings=settings),
            "telegram_service": TelegramService(settings=settings),
        }

        workflow = WorkflowLoader.from_dict(workflow_definition)
        engine = WorkflowEngine(dependencies=dependencies)
        context = await engine.execute(
            workflow=workflow,
            initial_context=initial_context,
            execution_id=execution_id,
        )

        return {
            "workflow_id": context.workflow_id,
            "execution_id": context.execution_id,
            "status": context.status.value,
            "variables": context.variables,
        }

    try:
        return asyncio.run(_run())
    except Exception as exc:
        logger.error("Workflow task failed", error=str(exc))
        raise self.retry(exc=exc)
