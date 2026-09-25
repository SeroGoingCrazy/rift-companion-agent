"""Factory for selecting a document Loader by file extension.

Loaders register the extensions they handle; the ingestion pipeline uses
``ExtensionRoutingLoader`` so one ``load()`` call dispatches ``.pdf`` to
``PdfLoader`` and ``.md`` to ``MarkdownLoader`` without format checks of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader


class LoaderFactory:
    """Registry mapping file extensions to BaseLoader subclasses."""

    _PROVIDERS: dict[str, type[BaseLoader]] = {}

    @classmethod
    def register_provider(cls, extension: str, provider_class: type[BaseLoader]) -> None:
        """Register a loader for an extension such as ``".md"``.

        Raises:
            ValueError: If provider_class doesn't inherit from BaseLoader.
        """
        if not issubclass(provider_class, BaseLoader):
            raise ValueError(
                f"Provider class {provider_class.__name__} must inherit from BaseLoader"
            )
        cls._PROVIDERS[cls._normalize(extension)] = provider_class

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return sorted(cls._PROVIDERS)

    @classmethod
    def get_provider(cls, file_path: str | Path) -> type[BaseLoader]:
        """Return the loader class registered for the file's extension.

        Raises:
            ValueError: If no loader handles the extension.
        """
        ext = cls._normalize(Path(file_path).suffix)
        try:
            return cls._PROVIDERS[ext]
        except KeyError:
            raise ValueError(
                f"Unsupported file type: '{ext or Path(file_path).name}'. "
                f"Supported: {', '.join(cls.supported_extensions())}"
            ) from None

    @classmethod
    def create(cls, file_path: str | Path, **kwargs: Any) -> BaseLoader:
        """Instantiate the loader for ``file_path`` with ``kwargs``."""
        return cls.get_provider(file_path)(**kwargs)

    @staticmethod
    def _normalize(extension: str) -> str:
        ext = extension.lower()
        return ext if ext.startswith(".") or not ext else f".{ext}"


class ExtensionRoutingLoader(BaseLoader):
    """Loader that delegates to the registered loader for each file's extension.

    Loader instances are created lazily and reused per loader class.

    Args:
        options: Constructor kwargs per extension, e.g.
            ``{".pdf": {"extract_images": True}, ".md": {"collection": "rules"}}``.
    """

    def __init__(self, options: Optional[Dict[str, Dict[str, Any]]] = None):
        self._options = {
            LoaderFactory._normalize(ext): kwargs for ext, kwargs in (options or {}).items()
        }
        self._loaders: dict[str, BaseLoader] = {}

    def loader_for(self, file_path: str | Path) -> BaseLoader:
        ext = LoaderFactory._normalize(Path(file_path).suffix)
        if ext not in self._loaders:
            self._loaders[ext] = LoaderFactory.create(file_path, **self._options.get(ext, {}))
        return self._loaders[ext]

    def load(self, file_path: str | Path) -> Document:
        return self.loader_for(file_path).load(file_path)


def _register_builtin_providers() -> None:
    from src.libs.loader.markdown_loader import MarkdownLoader
    from src.libs.loader.pdf_loader import PdfLoader

    LoaderFactory.register_provider(".pdf", PdfLoader)
    LoaderFactory.register_provider(".md", MarkdownLoader)
    LoaderFactory.register_provider(".markdown", MarkdownLoader)


_register_builtin_providers()
