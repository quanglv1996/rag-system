"""Plugin loader — dynamically discovers and loads provider plugins.

Plugins are Python packages placed under a configured plugins directory
or registered explicitly. Each plugin must implement the relevant
interface (AIProvider, SocialProvider, VectorDatabase) to be usable.

Discovery flow:
1. Scan the plugins directory for packages with a plugin.json manifest.
2. Validate the manifest.
3. Dynamically import the plugin class.
4. Register it in the appropriate factory.

This follows the Open/Closed Principle — adding a new provider
requires no changes to existing source code.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from app.core.logger import get_logger

logger = get_logger(__name__)

# Path to the user plugins directory (relative to project root)
DEFAULT_PLUGIN_DIR = Path("plugins")


class PluginManifest:
    """Parsed plugin.json manifest for a provider plugin.

    Attributes:
        name: Unique plugin identifier.
        version: Semantic version.
        plugin_type: 'ai', 'social', or 'vector'.
        entry_class: Fully-qualified class path.
        description: Human-readable description.
        requires: List of required pip packages.
    """

    def __init__(self, data: dict[str, Any], directory: Path) -> None:
        """Parse a plugin manifest dict.

        Args:
            data: Raw manifest dictionary.
            directory: Plugin package directory for error messages.

        Raises:
            ValueError: If required fields are missing.
        """
        required = {"name", "version", "plugin_type", "entry_class"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(
                f"Plugin manifest in '{directory}' is missing required fields: {missing}"
            )

        valid_types = {"ai", "social", "vector"}
        if data["plugin_type"] not in valid_types:
            raise ValueError(
                f"Invalid plugin_type '{data['plugin_type']}'. Must be one of {valid_types}"
            )

        self.name: str = data["name"]
        self.version: str = data["version"]
        self.plugin_type: str = data["plugin_type"]
        self.entry_class: str = data["entry_class"]
        self.description: str = data.get("description", "")
        self.requires: list[str] = data.get("requires", [])


class PluginLoader:
    """Discovers and loads provider plugins from the filesystem.

    Scanning supports plugins that ship with the application
    (in app/providers/) as well as user-installed plugins
    (in the configurable plugins/ directory).
    """

    _loaded: dict[str, PluginManifest] = {}

    @classmethod
    def discover(cls, plugin_dir: Path | str | None = None) -> list[PluginManifest]:
        """Scan the plugins directory and load all valid manifests.

        Args:
            plugin_dir: Directory to scan. Defaults to DEFAULT_PLUGIN_DIR.

        Returns:
            list[PluginManifest]: Successfully parsed manifests.
        """
        scan_dir = Path(plugin_dir or DEFAULT_PLUGIN_DIR)

        if not scan_dir.exists():
            logger.debug("Plugin directory does not exist", path=str(scan_dir))
            return []

        manifests: list[PluginManifest] = []

        for manifest_file in scan_dir.glob("*/plugin.json"):
            plugin_dir_path = manifest_file.parent
            try:
                raw = json.loads(manifest_file.read_text(encoding="utf-8"))
                manifest = PluginManifest(raw, plugin_dir_path)
                manifests.append(manifest)
                cls._loaded[manifest.name] = manifest
                logger.info(
                    "Plugin discovered",
                    name=manifest.name,
                    type=manifest.plugin_type,
                    version=manifest.version,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load plugin manifest",
                    path=str(manifest_file),
                    error=str(exc),
                )

        return manifests

    @classmethod
    def load_plugin(cls, manifest: PluginManifest) -> type:
        """Dynamically import the plugin class specified in the manifest.

        Args:
            manifest: Plugin manifest to load.

        Returns:
            type: The plugin class.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the class is not found in the module.
        """
        module_path, class_name = manifest.entry_class.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            logger.info("Plugin loaded", name=manifest.name, class_path=manifest.entry_class)
            return plugin_class
        except ImportError as exc:
            raise ImportError(
                f"Cannot import plugin '{manifest.name}': {exc}. "
                f"Ensure required packages are installed: {manifest.requires}"
            ) from exc

    @classmethod
    def register_all(cls, plugin_dir: Path | str | None = None) -> None:
        """Discover plugins and register them in the appropriate factory.

        Args:
            plugin_dir: Directory to scan for plugins.
        """
        manifests = cls.discover(plugin_dir)

        for manifest in manifests:
            try:
                plugin_class = cls.load_plugin(manifest)
                cls._register_in_factory(manifest, plugin_class)
            except Exception as exc:
                logger.error(
                    "Failed to register plugin",
                    name=manifest.name,
                    error=str(exc),
                )

    @classmethod
    def _register_in_factory(
        cls, manifest: PluginManifest, plugin_class: type
    ) -> None:
        """Register a loaded plugin class in the correct factory.

        Args:
            manifest: Plugin metadata.
            plugin_class: Imported class.
        """
        if manifest.plugin_type == "ai":
            from app.providers.ai.factory import AIProviderFactory

            AIProviderFactory.register(
                manifest.name,
                f"{plugin_class.__module__}.{plugin_class.__name__}",
            )

        elif manifest.plugin_type == "vector":
            from app.providers.vector.factory import VectorProviderFactory

            VectorProviderFactory.register(
                manifest.name,
                f"{plugin_class.__module__}.{plugin_class.__name__}",
            )

        logger.info(
            "Plugin registered in factory",
            name=manifest.name,
            type=manifest.plugin_type,
        )

    @classmethod
    def list_loaded(cls) -> list[dict[str, str]]:
        """List all currently loaded plugins.

        Returns:
            list[dict]: Plugin metadata dicts.
        """
        return [
            {
                "name": m.name,
                "type": m.plugin_type,
                "version": m.version,
                "description": m.description,
            }
            for m in cls._loaded.values()
        ]
