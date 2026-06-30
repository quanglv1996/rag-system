"""Scheduler Manager using APScheduler with AsyncIO support.

Manages scheduled jobs for:
- Scheduled social media posts
- Periodic workflow execution
- Token refresh jobs
- Data synchronization
- Cron-based recurring tasks

Jobs are persisted in Redis (jobstore) for restart safety.
"""

from __future__ import annotations

from typing import Any, Callable

from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class SchedulerManager:
    """Centralized async job scheduler.

    Uses APScheduler with Redis job store for persistence across restarts.
    Supports cron expressions, interval triggers, and one-shot date triggers.

    Attributes:
        _scheduler: The underlying APScheduler instance.
    """

    def __init__(self) -> None:
        """Initialize the scheduler with Redis persistence."""
        settings = get_settings()

        # Parse Redis connection info from URL
        redis_url = str(settings.redis_url)
        host = "localhost"
        port = 6379
        db = 0

        if "@" in redis_url:
            # redis://user:pass@host:port/db
            parts = redis_url.split("@")[-1].split("/")
            host_port = parts[0].split(":")
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 6379
            db = int(parts[1]) if len(parts) > 1 else 0
        elif redis_url.startswith("redis://"):
            # redis://host:port/db
            stripped = redis_url[8:]
            parts = stripped.split("/")
            host_port = parts[0].split(":")
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 6379
            db = int(parts[1]) if len(parts) > 1 else 0

        jobstores = {
            "default": RedisJobStore(
                host=host,
                port=port,
                db=db,
                jobs_key="apscheduler.jobs",
                run_times_key="apscheduler.run_times",
            )
        }

        executors = {
            "default": AsyncIOExecutor(),
        }

        job_defaults = {
            "coalesce": True,           # Run once even if missed multiple triggers
            "max_instances": 1,         # Prevent overlapping execution
            "misfire_grace_time": 60,   # Allow 60s late execution window
        }

        self._scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone="UTC",
        )

    def start(self) -> None:
        """Start the scheduler."""
        self._scheduler.start()
        logger.info("Scheduler started")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the scheduler gracefully.

        Args:
            wait: If True, wait for running jobs to complete.
        """
        self._scheduler.shutdown(wait=wait)
        logger.info("Scheduler stopped")

    def add_cron_job(
        self,
        func: Callable[..., Any],
        cron_expression: str,
        job_id: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        replace_existing: bool = True,
    ) -> str:
        """Schedule a job using a cron expression.

        Args:
            func: Callable to execute.
            cron_expression: Standard cron string (e.g., '0 9 * * 1-5').
            job_id: Unique job identifier.
            args: Positional arguments for func.
            kwargs: Keyword arguments for func.
            replace_existing: Replace existing job with same ID.

        Returns:
            str: Job ID.
        """
        # Parse cron expression into APScheduler CronTrigger
        parts = cron_expression.split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression '{cron_expression}'. Must have 5 fields."
            )

        trigger = CronTrigger(
            minute=parts[0],
            hour=parts[1],
            day=parts[2],
            month=parts[3],
            day_of_week=parts[4],
            timezone="UTC",
        )

        job = self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            args=args or [],
            kwargs=kwargs or {},
            replace_existing=replace_existing,
        )

        logger.info("Cron job scheduled", job_id=job_id, cron=cron_expression)
        return job.id

    def add_interval_job(
        self,
        func: Callable[..., Any],
        job_id: str,
        seconds: int = 0,
        minutes: int = 0,
        hours: int = 0,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Schedule a recurring interval job.

        Args:
            func: Callable to execute.
            job_id: Unique job identifier.
            seconds: Interval in seconds.
            minutes: Interval in minutes.
            hours: Interval in hours.
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            str: Job ID.
        """
        trigger = IntervalTrigger(seconds=seconds, minutes=minutes, hours=hours)

        job = self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            args=args or [],
            kwargs=kwargs or {},
            replace_existing=True,
        )

        logger.info("Interval job scheduled", job_id=job_id)
        return job.id

    def add_one_time_job(
        self,
        func: Callable[..., Any],
        job_id: str,
        run_at: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Schedule a one-time job at a specific date/time.

        Args:
            func: Callable to execute.
            job_id: Unique job identifier.
            run_at: ISO 8601 datetime string (e.g., '2026-07-01T09:00:00').
            args: Positional arguments.
            kwargs: Keyword arguments.

        Returns:
            str: Job ID.
        """
        from datetime import datetime

        run_time = datetime.fromisoformat(run_at)
        trigger = DateTrigger(run_date=run_time, timezone="UTC")

        job = self._scheduler.add_job(
            func,
            trigger=trigger,
            id=job_id,
            args=args or [],
            kwargs=kwargs or {},
        )

        logger.info("One-time job scheduled", job_id=job_id, run_at=run_at)
        return job.id

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job by ID.

        Args:
            job_id: Job identifier to remove.

        Returns:
            bool: True if removed successfully.
        """
        try:
            self._scheduler.remove_job(job_id)
            logger.info("Job removed", job_id=job_id)
            return True
        except Exception:
            return False

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all scheduled jobs.

        Returns:
            list[dict]: Job metadata including id, next_run_time, trigger.
        """
        jobs = self._scheduler.get_jobs()
        return [
            {
                "id": job.id,
                "name": job.name or job.id,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger),
            }
            for job in jobs
        ]

    def pause_job(self, job_id: str) -> None:
        """Pause a scheduled job.

        Args:
            job_id: Job to pause.
        """
        self._scheduler.pause_job(job_id)

    def resume_job(self, job_id: str) -> None:
        """Resume a paused job.

        Args:
            job_id: Job to resume.
        """
        self._scheduler.resume_job(job_id)


# =============================================================================
# Built-in scheduled jobs
# =============================================================================


async def refresh_oauth_tokens_job() -> None:
    """Periodic job: refresh all expiring OAuth tokens.

    Runs every 30 minutes to refresh tokens before they expire.
    """
    from app.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Token refresh job running")

    # In a real implementation, query CredentialManager for expiring tokens
    # and call the respective OAuth provider's refresh endpoint.


async def sync_facebook_inbox_job() -> None:
    """Periodic job: sync Facebook Messenger inbox."""
    from app.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Facebook inbox sync job running")


# Singleton scheduler instance
_scheduler_manager: SchedulerManager | None = None


def get_scheduler() -> SchedulerManager:
    """Get the singleton SchedulerManager instance.

    Returns:
        SchedulerManager: Configured scheduler.
    """
    global _scheduler_manager
    if _scheduler_manager is None:
        _scheduler_manager = SchedulerManager()
    return _scheduler_manager
