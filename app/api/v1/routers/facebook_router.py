"""Facebook API router — posts, messages, comments, webhooks, insights."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from app.core.dependency import AppSettings, get_facebook_service
from app.schemas.social import (
    FacebookCommentRequest,
    FacebookMessageRequest,
    FacebookPostRequest,
    FacebookPostResponse,
)
from app.services.facebook_service import FacebookService

router = APIRouter(prefix="/facebook", tags=["Facebook"])


@router.post(
    "/post",
    response_model=FacebookPostResponse,
    summary="Publish a Facebook post",
    description="Publish a text or media post to the configured Facebook Page.",
)
async def create_post(
    request: FacebookPostRequest,
    facebook_service: Annotated[FacebookService, Depends(get_facebook_service)],
) -> FacebookPostResponse:
    """Publish a post to the Facebook Page.

    Args:
        request: Post content and optional media.
        facebook_service: Injected Facebook service.

    Returns:
        FacebookPostResponse: Created post ID.
    """
    result = await facebook_service.publish_post(
        content=request.content,
        link=request.link,
        media_urls=request.media_urls,
    )
    return FacebookPostResponse(post_id=result.get("id", ""))


@router.patch(
    "/post/{post_id}",
    summary="Edit a Facebook post",
    description="Update the text content of an existing Facebook post.",
)
async def update_post(
    post_id: str,
    content: str,
    facebook_service: Annotated[FacebookService, Depends(get_facebook_service)],
) -> dict[str, Any]:
    """Edit an existing Facebook post.

    Args:
        post_id: ID of the post to edit.
        content: New text content.
        facebook_service: Injected Facebook service.

    Returns:
        dict: API response.
    """
    return await facebook_service.update_post(post_id, content)


@router.delete(
    "/post/{post_id}",
    summary="Delete a Facebook post",
)
async def delete_post(
    post_id: str,
    facebook_service: Annotated[FacebookService, Depends(get_facebook_service)],
) -> dict[str, Any]:
    """Delete a Facebook post by ID.

    Args:
        post_id: Post to delete.
        facebook_service: Injected service.
    """
    return await facebook_service.remove_post(post_id)


@router.post(
    "/message",
    summary="Send a Messenger message",
    description="Send a direct message via Facebook Messenger.",
)
async def send_message(
    request: FacebookMessageRequest,
    facebook_service: Annotated[FacebookService, Depends(get_facebook_service)],
) -> dict[str, Any]:
    """Send a Messenger message.

    Args:
        request: Message recipient and content.
        facebook_service: Injected service.
    """
    return await facebook_service.send_messenger_message(
        recipient_id=request.recipient_id,
        text=request.text,
        media_url=request.media_url,
    )


@router.post(
    "/comment",
    summary="Add a comment",
    description="Add a comment to a Facebook post or photo.",
)
async def add_comment(
    request: FacebookCommentRequest,
    facebook_service: Annotated[FacebookService, Depends(get_facebook_service)],
) -> dict[str, Any]:
    """Add a comment to a Facebook object.

    Args:
        request: Object ID and comment text.
        facebook_service: Injected service.
    """
    return await facebook_service.add_comment(request.object_id, request.text)


@router.get(
    "/insights/{post_id}",
    summary="Get post insights",
    description="Retrieve analytics data for a Facebook post.",
)
async def get_insights(
    post_id: str,
    facebook_service: Annotated[FacebookService, Depends(get_facebook_service)],
) -> dict[str, Any]:
    """Get insights for a Facebook post.

    Args:
        post_id: Post ID.
        facebook_service: Injected service.
    """
    return await facebook_service.get_post_insights(post_id)


@router.get(
    "/webhook",
    summary="Facebook webhook verification",
    description="Handles the hub challenge for Facebook webhook subscription.",
)
async def webhook_verify(
    request: Request,
    facebook_service: Annotated[FacebookService, Depends(get_facebook_service)],
) -> Any:
    """Verify a Facebook webhook subscription.

    Args:
        request: Incoming request with hub params.
        facebook_service: Injected service.
    """
    params = dict(request.query_params)
    mode = params.get("hub.mode", "")
    verify_token = params.get("hub.verify_token", "")
    challenge = params.get("hub.challenge", "")

    result = facebook_service.handle_webhook_challenge(mode, verify_token, challenge)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid webhook verification token",
        )
    return int(result)


@router.post(
    "/webhook",
    summary="Facebook webhook events",
    description="Receives and processes Facebook webhook event notifications.",
)
async def webhook_events(
    request: Request,
    facebook_service: Annotated[FacebookService, Depends(get_facebook_service)],
    x_hub_signature_256: str = Header(default=""),
) -> dict[str, str]:
    """Process incoming Facebook webhook events.

    Args:
        request: Raw request with webhook payload.
        facebook_service: Injected service.
        x_hub_signature_256: HMAC signature header for verification.

    Returns:
        dict: Acknowledgement response.
    """
    payload = await request.body()

    if not facebook_service.verify_webhook_signature(payload, x_hub_signature_256):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )

    # Process events asynchronously (fire and forget)
    # In production, dispatch to Celery task queue here
    return {"status": "received"}


@router.get(
    "/page",
    summary="Get page info",
    description="Retrieve information about the configured Facebook Page.",
)
async def get_page_info(
    facebook_service: Annotated[FacebookService, Depends(get_facebook_service)],
) -> dict[str, Any]:
    """Get the Facebook page information.

    Args:
        facebook_service: Injected service.
    """
    return await facebook_service.get_page_info()
