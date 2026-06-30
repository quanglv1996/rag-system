"""Async Workflow Engine — orchestrates node execution.

The engine executes workflow nodes in dependency order, handles
parallel execution, retry policies, timeouts, and branch routing.
All state is stored in WorkflowContext and is serializable to Redis
for long-running or resumable workflows.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from app.core.logger import get_logger
from app.workflow.models import (
    NodeDefinition,
    NodeExecutionResult,
    NodeStatus,
    WorkflowContext,
    WorkflowDefinition,
    WorkflowStatus,
)
from app.workflow.registry import NodeRegistry

logger = get_logger(__name__)


class WorkflowEngine:
    """Orchestrates the execution of a workflow definition.

    Responsibilities:
    - Instantiate nodes from the registry.
    - Execute nodes in order, following on_success / on_failure edges.
    - Handle parallel execution via asyncio.gather.
    - Apply retry policies and timeouts.
    - Persist and restore WorkflowContext.

    Attributes:
        _dependencies: Services injected into every node execution.
    """

    def __init__(self, dependencies: dict[str, Any] | None = None) -> None:
        """Initialize the workflow engine.

        Args:
            dependencies: Services available to all nodes
                          (ai_service, rag_service, facebook_service, etc.).
        """
        self._dependencies = dependencies or {}

    async def execute(
        self,
        workflow: WorkflowDefinition,
        initial_context: dict[str, Any] | None = None,
        execution_id: str | None = None,
    ) -> WorkflowContext:
        """Execute a complete workflow from start to finish.

        Args:
            workflow: Parsed workflow definition.
            initial_context: Initial variables to inject into the context.
            execution_id: Optional unique execution ID for tracing.

        Returns:
            WorkflowContext: Final execution context with all results.
        """
        exec_id = execution_id or str(uuid.uuid4())

        context = WorkflowContext(
            workflow_id=workflow.id,
            execution_id=exec_id,
            variables={**workflow.context, **(initial_context or {})},
            status=WorkflowStatus.RUNNING,
        )

        logger.info(
            "Workflow execution started",
            workflow_id=workflow.id,
            execution_id=exec_id,
        )

        start_time = time.perf_counter()

        try:
            # Begin from the entry node
            await self._execute_node(workflow, workflow.entry_node, context)

            if context.status not in (WorkflowStatus.FAILED, WorkflowStatus.PAUSED):
                context.status = WorkflowStatus.COMPLETED

        except Exception as exc:
            context.status = WorkflowStatus.FAILED
            logger.exception(
                "Workflow execution failed",
                workflow_id=workflow.id,
                execution_id=exec_id,
                error=str(exc),
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Workflow execution finished",
            workflow_id=workflow.id,
            execution_id=exec_id,
            status=context.status,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return context

    async def resume(
        self,
        workflow: WorkflowDefinition,
        context: WorkflowContext,
        resume_data: dict[str, Any] | None = None,
    ) -> WorkflowContext:
        """Resume a paused workflow (e.g., after human approval).

        Args:
            workflow: Original workflow definition.
            context: Previously stored context.
            resume_data: Additional variables to inject on resumption.

        Returns:
            WorkflowContext: Updated context after resumption.
        """
        if resume_data:
            context.variables.update(resume_data)

        context.status = WorkflowStatus.RUNNING

        # Find the paused node and re-execute from it
        paused_node_id = context.variables.get("_approval_node_id")
        if paused_node_id:
            node_def = workflow.get_node(paused_node_id)
            if node_def:
                context.variables["_approval_pending"] = False
                await self._execute_node(workflow, node_def, context)

        if context.status not in (WorkflowStatus.FAILED, WorkflowStatus.PAUSED):
            context.status = WorkflowStatus.COMPLETED

        return context

    async def _execute_node(
        self,
        workflow: WorkflowDefinition,
        node_def: NodeDefinition,
        context: WorkflowContext,
    ) -> None:
        """Instantiate and execute a single node, then route to successors.

        Args:
            workflow: Full workflow definition for successor lookup.
            node_def: Node definition to execute.
            context: Shared execution context.
        """
        context.current_node_id = node_def.id

        logger.debug(
            "Executing node",
            node_id=node_def.id,
            node_type=node_def.type,
        )

        # Instantiate the executor
        try:
            executor_class = NodeRegistry.get(node_def.type)
            executor = executor_class(node_def)
        except ValueError as exc:
            logger.error("Node type not registered", error=str(exc), node_id=node_def.id)
            context.status = WorkflowStatus.FAILED
            return

        # Execute with retry and timeout
        result = await self._run_with_retry(executor, context, node_def)
        context.node_results[node_def.id] = result

        # Stop if paused (human approval)
        if context.status == WorkflowStatus.PAUSED:
            return

        # Handle failure
        if result.status == NodeStatus.FAILED:
            if not result.next_node_ids:
                context.status = WorkflowStatus.FAILED
                return

        # Execute next nodes (may be parallel)
        next_ids = result.next_node_ids
        if not next_ids:
            return

        # Filter out already-executed nodes (prevent cycles)
        pending_ids = [nid for nid in next_ids if nid not in context.node_results]

        if not pending_ids:
            return

        # Parallel execution if multiple successors
        if len(pending_ids) > 1:
            next_defs = [workflow.get_node(nid) for nid in pending_ids]
            valid_defs = [n for n in next_defs if n is not None]
            await asyncio.gather(
                *[self._execute_node(workflow, nd, context) for nd in valid_defs]
            )
        else:
            next_def = workflow.get_node(pending_ids[0])
            if next_def:
                await self._execute_node(workflow, next_def, context)

    async def _run_with_retry(
        self,
        executor: Any,
        context: WorkflowContext,
        node_def: NodeDefinition,
    ) -> NodeExecutionResult:
        """Execute a node with retry and timeout policies.

        Args:
            executor: Instantiated node executor.
            context: Shared workflow context.
            node_def: Node configuration including retry policy.

        Returns:
            NodeExecutionResult: Final result after retries.
        """
        retry = node_def.retry
        max_attempts = retry.max_attempts if retry else 1
        delay = retry.delay_seconds if retry else 0.0
        backoff = retry.backoff_multiplier if retry else 1.0
        timeout = node_def.timeout_seconds

        last_result: NodeExecutionResult | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        executor.execute(context, self._dependencies),
                        timeout=timeout,
                    )
                else:
                    result = await executor.execute(context, self._dependencies)

                if result.status != NodeStatus.FAILED:
                    return result

                last_result = result

            except asyncio.TimeoutError:
                last_result = NodeExecutionResult(
                    node_id=node_def.id,
                    status=NodeStatus.FAILED,
                    error=f"Node timed out after {timeout}s",
                    next_node_ids=[node_def.on_failure] if node_def.on_failure else [],
                )

            except Exception as exc:
                last_result = NodeExecutionResult(
                    node_id=node_def.id,
                    status=NodeStatus.FAILED,
                    error=str(exc),
                    next_node_ids=[node_def.on_failure] if node_def.on_failure else [],
                )

            if attempt < max_attempts:
                wait = delay * (backoff ** (attempt - 1))
                logger.warning(
                    "Node failed, retrying",
                    node_id=node_def.id,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)

        return last_result or NodeExecutionResult(
            node_id=node_def.id,
            status=NodeStatus.FAILED,
            error="All retry attempts exhausted",
        )
