"""Prompt management module.

Provides versioned, cached, Jinja2-rendered prompt templates.
Supports variable substitution, version tracking, and in-memory
caching with Redis-backed persistence.

Usage:
    >>> from app.prompts.manager import PromptManager
    >>> manager = PromptManager()
    >>> prompt = manager.render("rag_query", context={"question": "..."})
"""

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from app.core.exception import ValidationException
from app.core.logger import get_logger

logger = get_logger(__name__)

# Default templates directory relative to this file
TEMPLATES_DIR = Path(__file__).parent / "templates"


class PromptTemplate:
    """Represents a versioned prompt template.

    Attributes:
        name: Template identifier.
        version: Semantic version string.
        description: Human-readable description.
        variables: List of required variable names.
        raw_content: Raw Jinja2 template string.
    """

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        variables: list[str],
        raw_content: str,
    ) -> None:
        """Initialize a prompt template.

        Args:
            name: Template identifier.
            version: Semantic version.
            description: Purpose description.
            variables: Required variable names.
            raw_content: Jinja2 template string.
        """
        self.name = name
        self.version = version
        self.description = description
        self.variables = variables
        self.raw_content = raw_content

    def to_dict(self) -> dict[str, Any]:
        """Serialize the template metadata.

        Returns:
            dict[str, Any]: Template metadata dictionary.
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "variables": self.variables,
        }


class PromptManager:
    """Manages loading, rendering, versioning, and caching of prompts.

    Loads templates from the filesystem (Jinja2) and maintains an
    in-memory registry. Rendered prompts can be optionally cached in
    Redis to avoid re-rendering identical contexts.

    Attributes:
        _env: Jinja2 environment configured with the templates directory.
        _registry: In-memory map of template name → PromptTemplate.
        _redis: Optional Redis client for caching rendered prompts.
    """

    def __init__(
        self,
        templates_dir: Path | str | None = None,
        redis: Any | None = None,
    ) -> None:
        """Initialize the prompt manager.

        Args:
            templates_dir: Path to Jinja2 templates directory.
                           Defaults to app/prompts/templates/.
            redis: Optional async Redis client for prompt caching.
        """
        self._templates_dir = Path(templates_dir or TEMPLATES_DIR)
        self._templates_dir.mkdir(parents=True, exist_ok=True)
        self._redis = redis
        self._registry: dict[str, PromptTemplate] = {}

        # Configure Jinja2 with autoescape disabled for prompt templates
        # (prompts are internal, not user-facing HTML)
        self._env = Environment(
            loader=FileSystemLoader(str(self._templates_dir)),
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Load built-in templates
        self._register_builtin_templates()

        logger.info(
            "PromptManager initialized",
            templates_dir=str(self._templates_dir),
        )

    def _register_builtin_templates(self) -> None:
        """Register the built-in prompt templates."""
        built_ins: list[PromptTemplate] = [
            PromptTemplate(
                name="rag_query",
                version="1.0.0",
                description="Standard RAG Q&A prompt with context injection",
                variables=["question", "context"],
                raw_content=(
                    "You are a helpful AI assistant. Answer the question based "
                    "only on the provided context. Be concise and accurate.\n\n"
                    "Context:\n{{ context }}\n\n"
                    "Question: {{ question }}\n\n"
                    "Answer:"
                ),
            ),
            PromptTemplate(
                name="summarize",
                version="1.0.0",
                description="Summarize a piece of text",
                variables=["text", "max_words"],
                raw_content=(
                    "Please provide a concise summary of the following text "
                    "in at most {{ max_words | default(200) }} words:\n\n"
                    "{{ text }}\n\n"
                    "Summary:"
                ),
            ),
            PromptTemplate(
                name="social_caption",
                version="1.0.0",
                description="Generate an engaging social media caption",
                variables=["topic", "platform", "tone"],
                raw_content=(
                    "Write an engaging {{ platform }} caption about '{{ topic }}'. "
                    "Tone: {{ tone | default('professional') }}. "
                    "Include relevant hashtags. Keep it under 280 characters."
                ),
            ),
        ]

        for template in built_ins:
            self._registry[template.name] = template

    def register(self, template: PromptTemplate) -> None:
        """Register a custom prompt template.

        Args:
            template: PromptTemplate instance to register.
        """
        self._registry[template.name] = template
        logger.debug("Prompt template registered", name=template.name, version=template.version)

    def get(self, name: str) -> PromptTemplate | None:
        """Retrieve a template by name.

        Args:
            name: Template identifier.

        Returns:
            PromptTemplate | None: Template if found.
        """
        return self._registry.get(name)

    def render(
        self,
        name: str,
        variables: dict[str, Any] | None = None,
    ) -> str:
        """Render a prompt template with the given variables.

        Supports both registry-based templates and filesystem Jinja2 files.
        Registry templates are rendered directly from their raw_content.

        Args:
            name: Template identifier.
            variables: Variable substitutions for the template.

        Returns:
            str: Rendered prompt string.

        Raises:
            ValidationException: If the template is not found.
        """
        ctx = variables or {}

        # Try registry first
        template = self._registry.get(name)
        if template is not None:
            jinja_template = self._env.from_string(template.raw_content)
            return jinja_template.render(**ctx)

        # Try filesystem Jinja2 file
        template_file = f"{name}.j2"
        try:
            jinja_template = self._env.get_template(template_file)
            return jinja_template.render(**ctx)
        except TemplateNotFound:
            raise ValidationException(
                f"Prompt template '{name}' not found",
                field="name",
            )

    def list_templates(self) -> list[dict[str, Any]]:
        """List all registered prompt templates.

        Returns:
            list[dict[str, Any]]: Template metadata list.
        """
        return [t.to_dict() for t in self._registry.values()]

    def build_rag_prompt(self, question: str, context: str) -> str:
        """Convenience method to render the standard RAG prompt.

        Args:
            question: User's question.
            context: Retrieved context from vector store.

        Returns:
            str: Rendered RAG prompt.
        """
        return self.render("rag_query", {"question": question, "context": context})

    def build_summary_prompt(self, text: str, max_words: int = 200) -> str:
        """Convenience method to render the summarization prompt.

        Args:
            text: Text to summarize.
            max_words: Maximum word count for the summary.

        Returns:
            str: Rendered summarization prompt.
        """
        return self.render("summarize", {"text": text, "max_words": max_words})
