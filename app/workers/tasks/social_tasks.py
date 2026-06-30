"""Social media background tasks — upload, post, message."""

from __future__ import annotations

from typing import Any

from celery import Task

from app.workers.celery_app import celery_app


@celery_app.task(
    bind=True,
    name="app.workers.tasks.social_tasks.publish_facebook_post",
    max_retries=3,
    default_retry_delay=30,
    queue="social",
)
def publish_facebook_post(
    self: Task,
    content: str,
    link: str | None = None,
    media_urls: list[str] | None = None,
) -> dict[str, Any]:
    """Publish a Facebook post in the background.

    Args:
        content: Post text content.
        link: Optional URL attachment.
        media_urls: Optional image URLs.

    Returns:
        dict: Facebook API response with post_id.
    """
    import asyncio

    async def _run() -> dict[str, Any]:
        from app.core.config import get_settings
        from app.services.facebook_service import FacebookService

        settings = get_settings()
        service = FacebookService(settings=settings)
        return await service.publish_post(content=content, link=link, media_urls=media_urls)

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))


@celery_app.task(
    bind=True,
    name="app.workers.tasks.social_tasks.send_telegram_message",
    max_retries=3,
    default_retry_delay=5,
    queue="social",
)
def send_telegram_message(
    self: Task,
    chat_id: str | int,
    text: str,
    parse_mode: str = "HTML",
) -> dict[str, Any]:
    """Send a Telegram message in the background.

    Args:
        chat_id: Target chat ID.
        text: Message text.
        parse_mode: 'HTML' or 'Markdown'.

    Returns:
        dict: Telegram API response.
    """
    import asyncio

    async def _run() -> dict[str, Any]:
        from app.core.config import get_settings
        from app.services.telegram_service import TelegramService

        settings = get_settings()
        service = TelegramService(settings=settings)
        return await service.send_text(chat_id=chat_id, text=text, parse_mode=parse_mode)

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)


@celery_app.task(
    bind=True,
    name="app.workers.tasks.social_tasks.upload_youtube_video",
    max_retries=2,
    soft_time_limit=3600,   # 1 hour soft limit for large uploads
    time_limit=3660,
    queue="social",
)
def upload_youtube_video(
    self: Task,
    access_token: str,
    title: str,
    description: str,
    video_url: str | None = None,
) -> dict[str, Any]:
    """Upload a video to YouTube in the background.

    Args:
        access_token: OAuth2 access token.
        title: Video title.
        description: Video description.
        video_url: Public video URL.

    Returns:
        dict: YouTube upload response.
    """
    import asyncio

    async def _run() -> dict[str, Any]:
        from app.core.config import get_settings
        from app.services.youtube_service import YouTubeService

        settings = get_settings()
        service = YouTubeService(settings=settings)
        return await service.upload_video(
            access_token=access_token,
            title=title,
            description=description,
            video_url=video_url,
        )

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(
    bind=True,
    name="app.workers.tasks.social_tasks.publish_tiktok_video",
    max_retries=2,
    queue="social",
)
def publish_tiktok_video(
    self: Task,
    access_token: str,
    video_url: str,
    title: str,
    privacy_level: str = "PUBLIC_TO_EVERYONE",
) -> dict[str, Any]:
    """Publish a video to TikTok in the background.

    Args:
        access_token: TikTok access token.
        video_url: Public video URL.
        title: Video title.
        privacy_level: TikTok privacy setting.

    Returns:
        dict: TikTok publish response.
    """
    import asyncio

    async def _run() -> dict[str, Any]:
        from app.core.config import get_settings
        from app.services.tiktok_service import TikTokService

        settings = get_settings()
        service = TikTokService(settings=settings)
        return await service.publish_video(
            access_token=access_token,
            video_url=video_url,
            title=title,
            privacy_level=privacy_level,
        )

    try:
        return asyncio.run(_run())
    except Exception as exc:
        raise self.retry(exc=exc)
