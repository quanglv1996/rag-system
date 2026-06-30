"""RAG node — performs document retrieval and Q&A."""

import time
from typing import Any

from app.workflow.models import NodeExecutionResult, WorkflowContext
from app.workflow.nodes.base import BaseNode


class RAGNode(BaseNode):
    """Workflow node that queries the RAG pipeline.

    Config keys:
        question_key: Context key containing the question (default: 'question').
        output_key: Context key to store the answer (default: 'rag_answer').
        collection: Vector store collection to query (default: 'default').
        top_k: Number of chunks to retrieve.
        temperature: LLM temperature for answer generation.
        system_prompt: Optional system instruction.
    """

    async def execute(
        self,
        context: WorkflowContext,
        dependencies: dict[str, Any] | None = None,
    ) -> NodeExecutionResult:
        """Run a RAG query.

        Args:
            context: Workflow context with the question variable.
            dependencies: Must contain 'rag_service'.

        Returns:
            NodeExecutionResult: Contains rag_answer and sources.
        """
        start = time.perf_counter()
        deps = dependencies or {}

        rag_service = deps.get("rag_service")
        if rag_service is None:
            return self._failure("RAG service not injected")

        question_key = self.config.get("question_key", "question")
        output_key = self.config.get("output_key", "rag_answer")
        question = self.get_input(context, question_key, "")

        if not question:
            return self._failure(f"Question not found in context under key '{question_key}'")

        try:
            result = await rag_service.query(
                question=question,
                collection=self.config.get("collection", "default"),
                top_k=self.config.get("top_k"),
                system_prompt=self.config.get("system_prompt"),
                temperature=self.config.get("temperature", 0.3),
            )

            self.set_output(context, output_key, result.answer)
            self.set_output(context, "rag_sources", [s.content for s in result.sources])

            duration_ms = (time.perf_counter() - start) * 1000
            return self._success(
                output={
                    output_key: result.answer,
                    "sources_count": result.retrieved_count,
                    "rag_sources": [s.content[:200] for s in result.sources],
                },
                duration_ms=duration_ms,
            )

        except Exception as exc:
            return self._failure(str(exc), (time.perf_counter() - start) * 1000)
