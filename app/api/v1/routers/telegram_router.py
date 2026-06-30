"""Telegram API router."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from app.core.dependency import get_telegram_service
from app.schemas.social import (
    TelegramInlineKeyboardRequest,
    TelegramSendMediaRequest,
    TelegramSendMessageRequest,
    TelegramWebhookSetRequest,
)
from app.services.telegram_service import TelegramService

router = APIRouter(prefix="/telegram", tags=["Telegram"])


@router.post("/send", summary="Send a Telegram message")
async def send_message(
    request: TelegramSendMessageRequest,
    telegram_service: Annotated[TelegramService, Depends(get_telegram_service)],
) -> dict[str, Any]:
    """Send a text message to a Telegram chat.

    Args:
        request: Chat ID and message text.
        telegram_service: Injected service.
    """
    return await telegram_service.send_text(
        chat_id=request.chat_id,
        text=request.text,
        parse_mode=request.parse_mode,
    )


@router.post("/send/media", summary="Send media to Telegram")
async def send_media(
    request: TelegramSendMediaRequest,
    telegram_service: Annotated[TelegramService, Depends(get_telegram_service)],
) -> dict[str, Any]:
    """Send media (photo, video, document, voice) to a Telegram chat.

    Args:
        request: Media URL and type.
        telegram_service: Injected service.
    """
    if request.media_type == "photo":
        return await telegram_service.send_photo(request.chat_id, request.url, request.caption)
    elif request.media_type == "video":
        return await telegram_service.send_video(request.chat_id, request.url, request.caption)
    elif request.media_type == "document":
        return await telegram_service.send_document(request.chat_id, request.url, request.caption)
    else:
        from app.providers.social.telegram_provider import TelegramProvider
        provider = TelegramProvider()
        return await provider.send_voice(request.chat_id, request.url)


@router.post("/send/keyboard", summary="Send message with inline keyboard")
async def send_keyboard(
    request: TelegramInlineKeyboardRequest,
    telegram_service: Annotated[TelegramService, Depends(get_telegram_service)],
) -> dict[str, Any]:
    """Send a Telegram message with inline keyboard buttons.

    Args:
        request: Message text and keyboard layout.
        telegram_service: Injected service.
    """
    return await telegram_service.send_inline_keyboard(
        chat_id=request.chat_id,
        text=request.text,
        buttons=request.buttons,
    )


@router.post("/webhook/set", summary="Set Telegram webhook")
async def set_webhook(
    request: TelegramWebhookSetRequest,
    telegram_service: Annotated[TelegramService, Depends(get_telegram_service)],
) -> dict[str, Any]:
    """Register the Telegram webhook URL.

    Args:
        request: Webhook URL and optional secret token.
        telegram_service: Injected service.
    """
    return await telegram_service.setup_webhook(
        webhook_url=request.url,
        secret_token=request.secret_token,
    )


@router.post("/webhook", summary="Receive Telegram updates", include_in_schema=False)
async def webhook_events(
    raw_request: Request,
    telegram_service: Annotated[TelegramService, Depends(get_telegram_service)],
) -> dict[str, str]:
    """Handle incoming Telegram webhook updates.

    In production, dispatch to a Celery task queue for processing.

    Args:
        raw_request: Raw request with Telegram update payload.
        telegram_service: Injected service.
    """
    # Parse the update
    update = await raw_request.json()

    # TODO: dispatch update to Celery task for async processing
    return {"status": "ok"}


@router.get("/bot/info", summary="Get bot information")
async def get_bot_info(
    telegram_service: Annotated[TelegramService, Depends(get_telegram_service)],
) -> dict[str, Any]:
    """Get information about the Telegram bot.

    Args:
        telegram_service: Injected service.
    """
    return await telegram_service.get_bot_info()
