"""API v1 router aggregator."""

from fastapi import APIRouter

from app.api.v1.routers.ai_router import router as ai_router
from app.api.v1.routers.facebook_router import router as facebook_router
from app.api.v1.routers.rag_router import router as rag_router
from app.api.v1.routers.telegram_router import router as telegram_router
from app.api.v1.routers.tiktok_router import router as tiktok_router
from app.api.v1.routers.youtube_router import router as youtube_router

# Aggregate all v1 routers under /api/v1
api_router = APIRouter()

api_router.include_router(ai_router)
api_router.include_router(rag_router)
api_router.include_router(facebook_router)
api_router.include_router(youtube_router)
api_router.include_router(telegram_router)
api_router.include_router(tiktok_router)

__all__ = ["api_router"]
