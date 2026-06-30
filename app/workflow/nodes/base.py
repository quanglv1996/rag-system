"""Abstract base class for all workflow nodes.

Each node type (AI, RAG, Social, Condition, etc.) implements this ABC.
New node types can be added by implementing execute() without modifying
the workflow engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.workflow.models import NodeDefinition, NodeExecutionResult, NodeStatus, WorkflowContext


class BaseNode(ABC):
    """Abstract base for all workflow node executors.

    Subclasses implement execute() to define the node's logic.
    The engine calls execute() and handles retry, timeout, and error
    routing automatically.

    Attributes:
        definition: The node's configuration from the workflow YAML/JSON.
    """

    def __init__(self, definition: NodeDefinition) -> None:
        """Initialize the node with its definition.

        Args:
            definition: Node configuration from the workflow definition.
        """
        self.definition = definition
        self.node_id = definition.id
        self.config = definition.config

    @abstractmethod
    async def execute(
        self,
        context: WorkflowContext,
        dependencies: dict[str, Any] | None = None,
    ) -> NodeExecutionResult:
        """Execute the node's core logic.

        Args:
            context: Shared workflow execution context.
            dependencies: Injected services (AI provider, RAG service, etc.).

        Returns:
            NodeExecutionResult: Execution result with output and status.
        """
        ...

    def _success(
        self,
        output: dict[str, Any],
        next_nodes: list[str] | None = None,
        duration_ms: float = 0.0,
    ) -> NodeExecutionResult:
        """Build a successful NodeExecutionResult.

        Args:
            output: Data to pass downstream.
            next_nodes: IDs of next nodes to execute.
            duration_ms: Execution duration.

        Returns:
            NodeExecutionResult: Success result.
        """
        # Determine next nodes from definition if not explicitly provided
        if next_nodes is None:
            on_success = self.definition.on_success
            if isinstance(on_success, str):
                next_nodes = [on_success]
            elif isinstance(on_success, list):
                next_nodes = on_success
            else:
                next_nodes = []

        return NodeExecutionResult(
            node_id=self.node_id,
            status=NodeStatus.COMPLETED,
            output=output,
            next_node_ids=next_nodes,
            duration_ms=duration_ms,
        )

    def _failure(
        self,
        error: str,
        duration_ms: float = 0.0,
    ) -> NodeExecutionResult:
        """Build a failed NodeExecutionResult.

        Args:
            error: Error description.
            duration_ms: Execution duration.

        Returns:
            NodeExecutionResult: Failure result.
        """
        next_nodes: list[str] = []
        if self.definition.on_failure:
            next_nodes = [self.definition.on_failure]

        return NodeExecutionResult(
            node_id=self.node_id,
            status=NodeStatus.FAILED,
            error=error,
            next_node_ids=next_nodes,
            duration_ms=duration_ms,
        )

    def get_input(self, context: WorkflowContext, key: str, default: Any = None) -> Any:
        """Retrieve a value from the shared workflow context.

        Args:
            context: Workflow execution context.
            key: Variable name to retrieve.
            default: Default value if key is not found.

        Returns:
            Any: Retrieved value or default.
        """
        return context.variables.get(key, default)

    def set_output(self, context: WorkflowContext, key: str, value: Any) -> None:
        """Write a value to the shared workflow context.

        Args:
            context: Workflow execution context.
            key: Variable name to set.
            value: Value to store.
        """
        context.variables[key] = value
