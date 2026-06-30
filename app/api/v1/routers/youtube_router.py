"""YouTube API router."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Header, UploadFile

from app.core.dependency import get_youtube_service
from app.schemas.social import (
    YouTubeCommentReplyRequest,
    YouTubePlaylistRequest,
    YouTubeUploadRequest,
    YouTubeUploadResponse,
)
from app.services.youtube_service import YouTubeService

router = APIRouter(prefix="/youtube", tags=["YouTube"])


@router.post(
    "/upload",
    response_model=YouTubeUploadResponse,
    summary="Upload a video to YouTube",
)
async def upload_video(
    request: YouTubeUploadRequest,
    authorization: str = Header(..., description="Bearer OAuth2 access token"),
    youtube_service: YouTubeService = Depends(get_youtube_service),
) -> YouTubeUploadResponse:
    """Initiate a YouTube video upload.

    Args:
        request: Video metadata and URL.
        authorization: OAuth2 Bearer token.
        youtube_service: Injected service.
    """
    access_token = authorization.replace("Bearer ", "")
    result = await youtube_service.upload_video(
        access_token=access_token,
        title=request.title,
        description=request.description,
        video_url=request.video_url,
        tags=request.tags,
        privacy_status=request.privacy_status,
        category_id=request.category_id,
    )
    return YouTubeUploadResponse(
        video_id=result.get("id"),
        upload_uri=result.get("upload_uri"),
        title=request.title,
        status="initiated",
    )


@router.post(
    "/playlists",
    summary="Create a YouTube playlist",
)
async def create_playlist(
    request: YouTubePlaylistRequest,
    authorization: str = Header(...),
    youtube_service: YouTubeService = Depends(get_youtube_service),
) -> dict[str, Any]:
    """Create a new YouTube playlist.

    Args:
        request: Playlist title and settings.
        authorization: OAuth2 Bearer token.
        youtube_service: Injected service.
    """
    access_token = authorization.replace("Bearer ", "")
    return await youtube_service.create_playlist(
        access_token=access_token,
        title=request.title,
        description=request.description,
        privacy_status=request.privacy_status,
    )


@router.post(
    "/comments/reply",
    summary="Reply to a YouTube comment",
)
async def reply_comment(
    request: YouTubeCommentReplyRequest,
    authorization: str = Header(...),
    youtube_service: YouTubeService = Depends(get_youtube_service),
) -> dict[str, Any]:
    """Reply to a YouTube comment.

    Args:
        request: Parent comment ID and reply text.
        authorization: OAuth2 Bearer token.
        youtube_service: Injected service.
    """
    access_token = authorization.replace("Bearer ", "")
    return await youtube_service.reply_to_comment(
        access_token=access_token,
        parent_comment_id=request.parent_comment_id,
        text=request.text,
    )


@router.get(
    "/videos/{video_id}/analytics",
    summary="Get video analytics",
)
async def get_analytics(
    video_id: str,
    authorization: str = Header(...),
    youtube_service: YouTubeService = Depends(get_youtube_service),
) -> dict[str, Any]:
    """Retrieve analytics for a YouTube video.

    Args:
        video_id: YouTube video ID.
        authorization: OAuth2 Bearer token.
        youtube_service: Injected service.
    """
    access_token = authorization.replace("Bearer ", "")
    return await youtube_service.get_analytics(access_token, video_id)
