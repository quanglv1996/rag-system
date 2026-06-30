"""Webhook package."""

from app.webhook.gateway import WebhookEvent, WebhookGateway

__all__ = ["WebhookGateway", "WebhookEvent"]
