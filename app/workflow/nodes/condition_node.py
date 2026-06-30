"""Condition / branch node — evaluates a Python expression for routing."""

import time
from typing import Any

from app.workflow.models import NodeExecutionResult, NodeStatus, WorkflowContext
from app.workflow.nodes.base import BaseNode


class ConditionNode(BaseNode):
    """Workflow node that branches based on a boolean expression.

    Config keys:
        expression: Python expression evaluated against context.variables.
                    Must return a truthy value to take the 'true_branch'.
        true_branch: Node ID to execute when expression is truthy.
        false_branch: Node ID to execute when expression is falsy.

    Example config:
        expression: "len(rag_answer) > 100"
        true_branch: "publish_node"
        false_branch: "fallback_node"
    """

    async def execute(
        self,
        context: WorkflowContext,
        dependencies: dict[str, Any] | None = None,
    ) -> NodeExecutionResult:
        """Evaluate the condition and route to the appropriate branch.

        Args:
            context: Workflow context with variables available to the expression.
            dependencies: Not used by this node.

        Returns:
            NodeExecutionResult: Routes to true_branch or false_branch.
        """
        start = time.perf_counter()

        expression = self.config.get("expression", "True")
        true_branch = self.config.get("true_branch")
        false_branch = self.config.get("false_branch")

        try:
            # Evaluate in a safe-ish context with only context.variables available
            result = bool(eval(expression, {"__builtins__": {}}, dict(context.variables)))  # noqa: S307
        except Exception as exc:
            return self._failure(f"Condition expression error: {exc}")

        next_node = true_branch if result else false_branch
        next_nodes = [next_node] if next_node else []

        duration_ms = (time.perf_counter() - start) * 1000
        return NodeExecutionResult(
            node_id=self.node_id,
            status=NodeStatus.COMPLETED,
            output={"condition_result": result, "expression": expression},
            next_node_ids=next_nodes,
            duration_ms=duration_ms,
        )


class DelayNode(BaseNode):
    """Workflow node that introduces a configurable delay.

    Config keys:
        seconds: Number of seconds to delay (may be overridden by context).
        seconds_key: Context variable key containing delay duration.
    """

    async def execute(
        self,
        context: WorkflowContext,
        dependencies: dict[str, Any] | None = None,
    ) -> NodeExecutionResult:
        """Sleep for the configured duration.

        Args:
            context: Workflow context.
            dependencies: Not used.

        Returns:
            NodeExecutionResult: Success after delay completes.
        """
        import asyncio

        start = time.perf_counter()

        seconds_key = self.config.get("seconds_key")
        if seconds_key:
            seconds = float(context.variables.get(seconds_key, 0))
        else:
            seconds = float(self.config.get("seconds", 0))

        await asyncio.sleep(seconds)

        duration_ms = (time.perf_counter() - start) * 1000
        return self._success(
            output={"delayed_seconds": seconds},
            duration_ms=duration_ms,
        )


class ApprovalNode(BaseNode):
    """Workflow node that pauses for human approval.

    Publishes an approval request (via notification service or stores
    a pending approval in context) and halts execution.
    On resumption, the 'approved' context variable determines routing.

    Config keys:
        approvers: List of approver identifiers (email, user IDs).
        approval_message: Message describing what needs approval.
        approved_branch: Node ID if approved.
        rejected_branch: Node ID if rejected.
    """

    async def execute(
        self,
        context: WorkflowContext,
        dependencies: dict[str, Any] | None = None,
    ) -> NodeExecutionResult:
        """Request human approval and route based on the stored decision.

        Args:
            context: Workflow context.
            dependencies: Optional 'notification_manager' for approval request.

        Returns:
            NodeExecutionResult: Routes based on approval decision.
        """
        start = time.perf_counter()

        # Check if approval has already been provided (workflow resumption)
        approved = context.variables.get("approved")

        if approved is None:
            # First pass — emit approval request, pause the workflow
            message = self.config.get("approval_message", "Workflow requires your approval")
            approvers = self.config.get("approvers", [])

            deps = dependencies or {}
            notification_manager = deps.get("notification_manager")
            if notification_manager:
                try:
                    await notification_manager.notify(
                        title="Workflow Approval Required",
                        message=f"{message}\nWorkflow: {context.workflow_id}\nExecution: {context.execution_id}",
                        recipients=approvers,
                    )
                except Exception:
                    pass  # Notification failure should not block workflow

            # Store approval request state in context
            context.variables["_approval_pending"] = True
            context.variables["_approval_node_id"] = self.node_id

            from app.workflow.models import WorkflowStatus
            context.status = WorkflowStatus.PAUSED

            return NodeExecutionResult(
                node_id=self.node_id,
                status=NodeStatus.WAITING,
                output={"approval_requested": True, "approvers": approvers},
                next_node_ids=[],
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        # Workflow resumed — route based on approval decision
        approved_branch = self.config.get("approved_branch")
        rejected_branch = self.config.get("rejected_branch")
        next_node = approved_branch if approved else rejected_branch
        next_nodes = [next_node] if next_node else []

        return NodeExecutionResult(
            node_id=self.node_id,
            status=NodeStatus.COMPLETED,
            output={"approved": approved},
            next_node_ids=next_nodes,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
