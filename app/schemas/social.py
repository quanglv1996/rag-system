"""Pydantic schemas for social platform API endpoints."""

from typing import Any

from pydantic import BaseModel, Field, HttpUrl


# =============================================================================
# Facebook Schemas
# =============================================================================


class FacebookPostRequest(BaseModel):
    """Request to publish a post to a Facebook Page."""

    content: str = Field(min_length=1, max_length=63206)
    link: str | None = Field(default=None)
    media_urls: list[str] = Field(default_factory=list)


class FacebookPostResponse(BaseModel):
    """Response after publishing a Facebook post."""

    post_id: str
    success: bool = True


class FacebookMessageRequest(BaseModel):
    """Request to send a Messenger message."""

    recipient_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=2000)
    media_url: str | None = Field(default=None)


class FacebookCommentRequest(BaseModel):
    """Request to add a comment to a Facebook object."""

    object_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=8000)


class FacebookWebhookVerifyParams(BaseModel):
    """Query parameters for Facebook webhook verification."""

    hub_mode: str = Field(alias="hub.mode")
    hub_verify_token: str = Field(alias="hub.verify_token")
    hub_challenge: str = Field(alias="hub.challenge")

    model_config = {"populate_by_name": True}


# =============================================================================
# YouTube Schemas
# =============================================================================


class YouTubeUploadRequest(BaseModel):
    """Request to upload a video to YouTube."""

    title: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=5000)
    video_url: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    privacy_status: str = Field(default="public")
    category_id: str = Field(default="22")


class YouTubeUploadResponse(BaseModel):
    """Response after initiating a YouTube video upload."""

    video_id: str | None = None
    upload_uri: str | None = None
    title: str
    status: str


class YouTubeCommentReplyRequest(BaseModel):
    """Request to reply to a YouTube comment."""

    parent_comment_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=10000)


class YouTubePlaylistRequest(BaseModel):
    """Request to create a YouTube playlist."""

    title: str = Field(min_length=1, max_length=150)
    description: str = Field(default="")
    privacy_status: str = Field(default="public")


# =============================================================================
# Telegram Schemas
# =============================================================================


class TelegramSendMessageRequest(BaseModel):
    """Request to send a Telegram message."""

    chat_id: str | int
    text: str = Field(min_length=1, max_length=4096)
    parse_mode: str = Field(default="HTML")


class TelegramSendMediaRequest(BaseModel):
    """Request to send media to a Telegram chat."""

    chat_id: str | int
    url: str = Field(min_length=1)
    caption: str = Field(default="", max_length=1024)
    media_type: str = Field(default="photo")  # photo/video/document/voice


class TelegramInlineKeyboardRequest(BaseModel):
    """Request to send a message with an inline keyboard."""

    chat_id: str | int
    text: str = Field(min_length=1)
    buttons: list[list[dict[str, str]]] = Field(min_length=1)
    parse_mode: str = Field(default="HTML")


class TelegramWebhookSetRequest(BaseModel):
    """Request to set the Telegram webhook URL."""

    url: str = Field(min_length=1)
    secret_token: str | None = Field(default=None)


# =============================================================================
# TikTok Schemas
# =============================================================================


class TikTokPublishRequest(BaseModel):
    """Request to publish a video to TikTok."""

    video_url: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=2200)
    privacy_level: str = Field(default="PUBLIC_TO_EVERYONE")
    disable_comment: bool = Field(default=False)
    disable_duet: bool = Field(default=False)
    disable_stitch: bool = Field(default=False)


class TikTokPublishResponse(BaseModel):
    """Response after initiating a TikTok video publish."""

    publish_id: str | None = None
    status: str


class TikTokTokenCallbackRequest(BaseModel):
    """Request parameters from TikTok OAuth callback."""

    code: str = Field(min_length=1)
    state: str | None = Field(default=None)
