"""Celery application configuration for the AI Automation Platform.

Provides the central Celery app instance with Redis as broker and backend.
All background tasks are defined in sub-modules and auto-discovered.
"""

from celery import Celery
from celery.signals import worker_ready
from kombu import Exchange, Queue

from app.core.config import get_settings

settings = get_settings()

# =============================================================================
# Celery Application Instance
# =============================================================================

celery_app = Celery(
    "ai_platform",
    broker=str(settings.redis_url),
    backend=str(settings.redis_url),
    include=[
        "app.workers.tasks.ai_tasks",
        "app.workers.tasks.rag_tasks",
        "app.workers.tasks.social_tasks",
        "app.workers.tasks.workflow_tasks",
    ],
)

# =============================================================================
# Configuration
# =============================================================================

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task behavior
    task_acks_late=True,               # Acknowledge only after task completes
    task_reject_on_worker_lost=True,   # Re-queue on worker crash
    task_track_started=True,           # Report STARTED state
    task_soft_time_limit=600,          # Soft kill after 10 min (raises SoftTimeLimitExceeded)
    task_time_limit=660,               # Hard kill after 11 min
    worker_prefetch_multiplier=1,      # One task at a time per worker
    worker_max_tasks_per_child=100,    # Restart worker after 100 tasks (memory leak prevention)

    # Result backend
    result_expires=86400,              # Keep results for 24 hours

    # Dead Letter Queue — failed tasks after max_retries go here
    task_queues=(
        Queue("default",    Exchange("default"),    routing_key="default"),
        Queue("high",       Exchange("high"),       routing_key="high"),
        Queue("ai",         Exchange("ai"),         routing_key="ai"),
        Queue("rag",        Exchange("rag"),        routing_key="rag"),
        Queue("social",     Exchange("social"),     routing_key="social"),
        Queue("workflow",   Exchange("workflow"),   routing_key="workflow"),
        Queue("dead_letter",Exchange("dead_letter"),routing_key="dead_letter"),
    ),
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",

    # Task routing by task name prefix
    task_routes={
        "app.workers.tasks.ai_tasks.*":       {"queue": "ai"},
        "app.workers.tasks.rag_tasks.*":      {"queue": "rag"},
        "app.workers.tasks.social_tasks.*":   {"queue": "social"},
        "app.workers.tasks.workflow_tasks.*": {"queue": "workflow"},
    },

    # Flower monitoring
    flower_basic_auth=["admin:admin"],  # Override in production
)


@worker_ready.connect
def on_worker_ready(**kwargs: object) -> None:
    """Log when Celery worker is ready to accept tasks."""
    from app.core.logger import get_logger
    logger = get_logger("celery.worker")
    logger.info("Celery worker ready", queues=list(celery_app.conf.task_queues))
