"""Node registry — maps node type strings to their executor classes."""

from app.workflow.models import NodeType
from app.workflow.nodes.base import BaseNode


class NodeRegistry:
    """Registry mapping NodeType enum values to executor classes.

    New node types can be registered without modifying existing code.
    This implements the Open/Closed Principle for the workflow engine.
    """

    _registry: dict[NodeType, type[BaseNode]] = {}

    @classmethod
    def register(cls, node_type: NodeType, node_class: type[BaseNode]) -> None:
        """Register a node executor class.

        Args:
            node_type: NodeType enum value.
            node_class: Executor class implementing BaseNode.
        """
        cls._registry[node_type] = node_class

    @classmethod
    def get(cls, node_type: NodeType) -> type[BaseNode]:
        """Get the executor class for a node type.

        Args:
            node_type: NodeType to look up.

        Returns:
            type[BaseNode]: Executor class.

        Raises:
            ValueError: If the node type is not registered.
        """
        klass = cls._registry.get(node_type)
        if klass is None:
            raise ValueError(
                f"Node type '{node_type}' is not registered. "
                f"Available: {list(cls._registry.keys())}"
            )
        return klass

    @classmethod
    def available(cls) -> list[str]:
        """Return a list of all registered node type names.

        Returns:
            list[str]: Registered node type strings.
        """
        return [t.value for t in cls._registry]


def _register_builtin_nodes() -> None:
    """Register all built-in node types."""
    from app.workflow.nodes.ai_node import AINode
    from app.workflow.nodes.condition_node import ApprovalNode, ConditionNode, DelayNode
    from app.workflow.nodes.rag_node import RAGNode
    from app.workflow.nodes.social_node import HTTPNode, SocialNode, TransformNode

    NodeRegistry.register(NodeType.AI, AINode)
    NodeRegistry.register(NodeType.RAG, RAGNode)
    NodeRegistry.register(NodeType.SOCIAL, SocialNode)
    NodeRegistry.register(NodeType.CONDITION, ConditionNode)
    NodeRegistry.register(NodeType.BRANCH, ConditionNode)  # alias
    NodeRegistry.register(NodeType.DELAY, DelayNode)
    NodeRegistry.register(NodeType.APPROVAL, ApprovalNode)
    NodeRegistry.register(NodeType.HTTP, HTTPNode)
    NodeRegistry.register(NodeType.TRANSFORM, TransformNode)


# Auto-register built-ins on module import
_register_builtin_nodes()
