"""RBAC — Role-Based Access Control for the API.

Defines roles (Admin, Developer, Operator, Viewer) and maps them to
permission sets. FastAPI dependencies enforce role requirements.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from fastapi import Depends, HTTPException, status

from app.core.dependency import CurrentUserId
from app.core.exception import AuthorizationException


class Role(str, Enum):
    """User roles in the RBAC system."""

    ADMIN = "admin"         # Full access
    DEVELOPER = "developer" # AI, RAG, read social
    OPERATOR = "operator"   # Social posting, workflow execution
    VIEWER = "viewer"       # Read-only access


# Permission → allowed roles mapping
ROLE_PERMISSIONS: dict[str, set[Role]] = {
    # AI operations
    "ai:chat": {Role.ADMIN, Role.DEVELOPER, Role.OPERATOR},
    "ai:embedding": {Role.ADMIN, Role.DEVELOPER},
    "ai:image": {Role.ADMIN, Role.DEVELOPER, Role.OPERATOR},

    # RAG operations
    "rag:query": {Role.ADMIN, Role.DEVELOPER, Role.OPERATOR, Role.VIEWER},
    "rag:upload": {Role.ADMIN, Role.DEVELOPER},
    "rag:delete": {Role.ADMIN},

    # Social operations
    "social:post": {Role.ADMIN, Role.OPERATOR},
    "social:read": {Role.ADMIN, Role.DEVELOPER, Role.OPERATOR, Role.VIEWER},

    # Workflow operations
    "workflow:execute": {Role.ADMIN, Role.OPERATOR},
    "workflow:read": {Role.ADMIN, Role.DEVELOPER, Role.OPERATOR, Role.VIEWER},

    # Admin operations
    "admin:all": {Role.ADMIN},
    "credentials:write": {Role.ADMIN},
    "users:manage": {Role.ADMIN},
}


class RBACGuard:
    """FastAPI dependency for permission-based access control.

    Usage:
        >>> @router.post("/rag/upload")
        ... async def upload(
        ...     _: None = Depends(RBACGuard("rag:upload")),
        ... ):
        ...     ...
    """

    def __init__(self, required_permission: str) -> None:
        """Initialize the guard with a required permission.

        Args:
            required_permission: Permission key from ROLE_PERMISSIONS.
        """
        self.required_permission = required_permission

    async def __call__(
        self,
        user_id: CurrentUserId,
    ) -> None:
        """Check that the current user has the required permission.

        Args:
            user_id: Current authenticated user ID.

        Raises:
            HTTPException: 403 if the user lacks the required permission.
        """
        # In a real implementation, look up the user's role from database
        # For now, we demonstrate the pattern with a mock role lookup
        user_role = await self._get_user_role(user_id)

        allowed_roles = ROLE_PERMISSIONS.get(self.required_permission, set())

        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": {
                        "code": "PERMISSION_DENIED",
                        "message": f"Permission '{self.required_permission}' required",
                        "details": {
                            "required_permission": self.required_permission,
                            "user_role": user_role.value if user_role else None,
                        },
                    }
                },
            )

    async def _get_user_role(self, user_id: str) -> Role | None:
        """Look up a user's role.

        In production, this queries the database.
        This default implementation always returns ADMIN for any authenticated user.

        Args:
            user_id: User identifier.

        Returns:
            Role | None: User's assigned role.
        """
        # TODO: Implement DB lookup via UserRepository
        return Role.ADMIN


def require_permission(permission: str) -> Annotated[None, Depends]:
    """Convenience function for declaring permission requirements.

    Args:
        permission: Permission key string.

    Returns:
        FastAPI Depends annotation.

    Example:
        >>> CanUploadRAG = require_permission("rag:upload")
        >>> @router.post("/upload")
        ... async def upload(_: CanUploadRAG):
        ...     ...
    """
    return Depends(RBACGuard(permission))


# Pre-built permission guards
CanChat = Annotated[None, Depends(RBACGuard("ai:chat"))]
CanUploadRAG = Annotated[None, Depends(RBACGuard("rag:upload"))]
CanPost = Annotated[None, Depends(RBACGuard("social:post"))]
CanAdmin = Annotated[None, Depends(RBACGuard("admin:all"))]
CanExecuteWorkflow = Annotated[None, Depends(RBACGuard("workflow:execute"))]
