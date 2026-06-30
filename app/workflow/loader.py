"""JSON / YAML workflow definition loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.exception import ValidationException
from app.workflow.models import WorkflowDefinition


class WorkflowLoader:
    """Loads WorkflowDefinition objects from JSON or YAML sources."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> WorkflowDefinition:
        """Parse a workflow from a Python dictionary.

        Args:
            data: Raw dictionary matching the WorkflowDefinition schema.

        Returns:
            WorkflowDefinition: Validated workflow object.

        Raises:
            ValidationException: If the data is invalid.
        """
        try:
            return WorkflowDefinition.model_validate(data)
        except Exception as exc:
            raise ValidationException(
                f"Invalid workflow definition: {exc}",
                field="workflow",
            ) from exc

    @staticmethod
    def from_json(json_str: str) -> WorkflowDefinition:
        """Parse a workflow from a JSON string.

        Args:
            json_str: JSON string containing the workflow definition.

        Returns:
            WorkflowDefinition: Validated workflow object.
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValidationException(f"Invalid JSON: {exc}", field="workflow") from exc

        return WorkflowLoader.from_dict(data)

    @staticmethod
    def from_yaml(yaml_str: str) -> WorkflowDefinition:
        """Parse a workflow from a YAML string.

        Args:
            yaml_str: YAML string containing the workflow definition.

        Returns:
            WorkflowDefinition: Validated workflow object.
        """
        try:
            import yaml  # type: ignore[import-untyped]

            data = yaml.safe_load(yaml_str)
        except Exception as exc:
            raise ValidationException(f"Invalid YAML: {exc}", field="workflow") from exc

        return WorkflowLoader.from_dict(data)

    @staticmethod
    def from_file(path: str | Path) -> WorkflowDefinition:
        """Load a workflow definition from a JSON or YAML file.

        Args:
            path: Path to the workflow file (.json or .yaml/.yml).

        Returns:
            WorkflowDefinition: Validated workflow object.

        Raises:
            ValidationException: If the file format is unsupported or invalid.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise ValidationException(f"Workflow file not found: {path}", field="path")

        content = file_path.read_text(encoding="utf-8")
        ext = file_path.suffix.lower()

        if ext == ".json":
            return WorkflowLoader.from_json(content)
        elif ext in (".yaml", ".yml"):
            return WorkflowLoader.from_yaml(content)
        else:
            raise ValidationException(
                f"Unsupported workflow file format: '{ext}'. Use .json or .yaml",
                field="path",
            )
