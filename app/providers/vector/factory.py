"""Factory for creating vector database provider instances."""

from app.core.exception import ConfigurationException
from app.interfaces.vector_database import VectorDatabase


class VectorProviderFactory:
    """Factory that creates VectorDatabase instances by name.

    Usage:
        >>> provider = VectorProviderFactory.create("chroma")
        >>> await provider.add(documents)
    """

    _registry: dict[str, str] = {
        "chroma": "app.providers.vector.chroma_provider.ChromaProvider",
        "faiss": "app.providers.vector.faiss_provider.FAISSProvider",
    }

    @classmethod
    def create(cls, provider_name: str) -> VectorDatabase:
        """Create a VectorDatabase provider instance.

        Args:
            provider_name: Registered provider name ('chroma' or 'faiss').

        Returns:
            VectorDatabase: Configured provider instance.

        Raises:
            ConfigurationException: If provider_name is not registered.
        """
        if provider_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ConfigurationException(
                f"Vector provider '{provider_name}' is not registered. "
                f"Available: {available}",
                config_key="VECTOR_DB",
            )

        import importlib

        module_path, class_name = cls._registry[provider_name].rsplit(".", 1)
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)

        return provider_class()  # type: ignore[no-any-return]

    @classmethod
    def register(cls, name: str, class_path: str) -> None:
        """Register a new vector provider class.

        Args:
            name: Unique provider identifier.
            class_path: Fully-qualified class path.
        """
        cls._registry[name] = class_path
