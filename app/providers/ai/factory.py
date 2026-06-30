"""Factory for creating AI provider instances.

Implements the Factory Pattern to instantiate the correct AIProvider
based on configuration. New providers can be registered here without
changing any service or router code.
"""

from app.core.exception import ConfigurationException
from app.interfaces.ai_provider import AIProvider


class AIProviderFactory:
    """Factory that creates AIProvider instances by name.

    Usage:
        >>> provider = AIProviderFactory.create("openai")
        >>> response = await provider.chat(request)
    """

    # Registry maps provider names to their classes (lazy imports)
    _registry: dict[str, str] = {
        "openai": "app.providers.ai.openai_provider.OpenAIProvider",
        "google": "app.providers.ai.google_provider.GoogleProvider",
    }

    @classmethod
    def create(cls, provider_name: str) -> AIProvider:
        """Create and return an AI provider instance.

        Args:
            provider_name: Registered provider identifier (e.g., 'openai').

        Returns:
            AIProvider: Configured provider instance.

        Raises:
            ConfigurationException: If provider_name is not registered.
        """
        if provider_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ConfigurationException(
                f"AI provider '{provider_name}' is not registered. "
                f"Available providers: {available}",
                config_key="LLM_PROVIDER",
            )

        # Lazy import to avoid loading all provider SDKs at startup
        module_path, class_name = cls._registry[provider_name].rsplit(".", 1)

        import importlib

        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)

        return provider_class()  # type: ignore[no-any-return]

    @classmethod
    def register(cls, name: str, class_path: str) -> None:
        """Register a new AI provider class.

        Args:
            name: Unique identifier for the provider.
            class_path: Fully-qualified class path (e.g., 'my_pkg.MyProvider').
        """
        cls._registry[name] = class_path

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return list of all registered provider names.

        Returns:
            list[str]: Registered provider identifiers.
        """
        return list(cls._registry.keys())
