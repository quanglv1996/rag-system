"""Pydantic models for Workflow Engine definitions.

Workflows are defined in JSON or YAML. Each workflow consists of
nodes connected by edges. Nodes are typed components that the engine
instantiates and executes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Supported node types in the workflow graph."""

    AI = "ai"
    RAG = "rag"
    SOCIAL = "social"
    CONDITION = "condition"
    BRANCH = "branch"
    LOOP = "loop"
    DELAY = "delay"
    HTTP = "http"
    APPROVAL = "approval"
    PARALLEL = "parallel"
    TRANSFORM = "transform"
    NOTIFY = "notify"
    STORAGE = "storage"
    SCHEDULE = "schedule"


class WorkflowStatus(str, Enum):
    """Workflow execution lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"         # Waiting for human approval
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SCHEDULED = "scheduled"


class NodeStatus(str, Enum):
    """Individual node execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"       # Waiting for approval / delay


class RetryPolicy(BaseModel):
    """Retry configuration for a workflow node.

    Attributes:
        max_attempts: Maximum number of retry attempts.
        delay_seconds: Base delay between retries (exponential backoff applied).
        backoff_multiplier: Exponential backoff multiplier.
        retry_on: Exception type names that trigger retry.
    """

    max_attempts: int = Field(default=3, ge=1, le=10)
    delay_seconds: float = Field(default=2.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    retry_on: list[str] = Field(default_factory=lambda: ["Exception"])


class NodeDefinition(BaseModel):
    """Definition of a single node in the workflow graph.

    Attributes:
        id: Unique node identifier within the workflow.
        type: Node type determining which executor to use.
        name: Human-readable node name.
        config: Node-specific configuration parameters.
        retry: Optional retry policy for transient failures.
        timeout_seconds: Maximum execution time before timeout.
        on_success: ID(s) of next node(s) on success.
        on_failure: ID of node to execute on failure.
        on_condition: Condition expression for branch nodes.
        delay_seconds: Seconds to delay before execution (delay nodes).
        parallel: Whether this node runs in parallel with siblings.
    """

    id: str = Field(min_length=1)
    type: NodeType
    name: str = Field(default="")
    config: dict[str, Any] = Field(default_factory=dict)
    retry: RetryPolicy | None = Field(default=None)
    timeout_seconds: float | None = Field(default=None, ge=1.0)
    on_success: list[str] | str | None = Field(default=None)
    on_failure: str | None = Field(default=None)
    on_condition: str | None = Field(default=None)
    delay_seconds: float | None = Field(default=None, ge=0.0)
    parallel: bool = Field(default=False)


class WorkflowDefinition(BaseModel):
    """Complete workflow definition loaded from JSON or YAML.

    Attributes:
        id: Unique workflow identifier.
        name: Human-readable name.
        description: Purpose description.
        version: Semantic version string.
        trigger: Optional trigger configuration.
        nodes: Ordered list of node definitions.
        context: Default context variables available to all nodes.
        schedule: Optional cron expression for scheduled execution.
        tags: Categorization tags.
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(default="")
    version: str = Field(default="1.0.0")
    trigger: dict[str, Any] = Field(default_factory=dict)
    nodes: list[NodeDefinition] = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    schedule: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)

    def get_node(self, node_id: str) -> NodeDefinition | None:
        """Find a node by its ID.

        Args:
            node_id: Node identifier to search for.

        Returns:
            NodeDefinition | None: Found node or None.
        """
        return next((n for n in self.nodes if n.id == node_id), None)

    @property
    def entry_node(self) -> NodeDefinition:
        """Return the first node (entry point) of the workflow.

        Returns:
            NodeDefinition: First node in the list.
        """
        return self.nodes[0]


class NodeExecutionResult(BaseModel):
    """Result produced by a single node execution.

    Attributes:
        node_id: ID of the executed node.
        status: Execution status.
        output: Data produced by the node, available to downstream nodes.
        error: Error message if execution failed.
        duration_ms: Execution duration in milliseconds.
        next_node_ids: IDs of nodes to execute next.
    """

    node_id: str
    status: NodeStatus
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0
    next_node_ids: list[str] = Field(default_factory=list)


class WorkflowContext(BaseModel):
    """Shared mutable execution context passed between nodes.

    Attributes:
        workflow_id: Workflow definition ID.
        execution_id: Unique ID for this execution run.
        variables: Shared key-value store for inter-node data passing.
        node_results: Map of node_id → NodeExecutionResult.
        status: Overall workflow execution status.
        current_node_id: ID of the currently executing node.
    """

    workflow_id: str
    execution_id: str
    variables: dict[str, Any] = Field(default_factory=dict)
    node_results: dict[str, NodeExecutionResult] = Field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_node_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}
