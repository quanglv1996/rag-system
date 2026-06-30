"""TikTok API router."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header

from app.core.dependency import get_tiktok_service
from app.schemas.social import (
    TikTokPublishRequest,
    TikTokPublishResponse,
    TikTokTokenCallbackRequest,
)
from app.services.tiktok_service import TikTokService

router = APIRouter(prefix="/tiktok", tags=["TikTok"])


@router.post(
    "/upload",
    response_model=TikTokPublishResponse,
    summary="Publish a video to TikTok",
)
async def publish_video(
    request: TikTokPublishRequest,
    authorization: str = Header(..., description="Bearer TikTok access token"),
    tiktok_service: TikTokService = Depends(get_tiktok_service),
) -> TikTokPublishResponse:
    """Publish a video to TikTok via Content Posting API.

    Args:
        request: Video URL, title, and privacy settings.
        authorization: Bearer access token.
        tiktok_service: Injected service.
    """
    access_token = authorization.replace("Bearer ", "")
    result = await tiktok_service.publish_video(
        access_token=access_token,
        video_url=request.video_url,
        title=request.title,
        privacy_level=request.privacy_level,
        disable_comment=request.disable_comment,
        disable_duet=request.disable_duet,
        disable_stitch=request.disable_stitch,
    )
    return TikTokPublishResponse(
        publish_id=result.get("publish_id"),
        status="pending",
    )


@router.get(
    "/callback",
    summary="TikTok OAuth callback",
    description="Handle the OAuth2 callback and exchange code for tokens.",
)
async def oauth_callback(
    code: str,
    state: str | None = None,
    tiktok_service: TikTokService = Depends(get_tiktok_service),
) -> dict[str, Any]:
    """Handle TikTok OAuth2 callback.

    Args:
        code: Authorization code from TikTok.
        state: Optional state parameter.
        tiktok_service: Injected service.
    """
    return await tiktok_service.exchange_auth_code(code)


@router.get(
    "/creator/info",
    summary="Get TikTok creator info",
)
async def get_creator_info(
    authorization: str = Header(...),
    tiktok_service: TikTokService = Depends(get_tiktok_service),
) -> dict[str, Any]:
    """Get information about the authenticated TikTok creator.

    Args:
        authorization: Bearer access token.
        tiktok_service: Injected service.
    """
    access_token = authorization.replace("Bearer ", "")
    return await tiktok_service.get_creator_info(access_token)


@router.get(
    "/videos",
    summary="List creator's TikTok videos",
)
async def list_videos(
    max_count: int = 20,
    authorization: str = Header(...),
    tiktok_service: TikTokService = Depends(get_tiktok_service),
) -> dict[str, Any]:
    """List videos for the authenticated TikTok creator.

    Args:
        max_count: Maximum number of videos to return.
        authorization: Bearer access token.
        tiktok_service: Injected service.
    """
    access_token = authorization.replace("Bearer ", "")
    return await tiktok_service.get_video_list(access_token, max_count)
