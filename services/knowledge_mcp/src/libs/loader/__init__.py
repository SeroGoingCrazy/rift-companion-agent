"""
Loader Module.

This package contains document loader components:
- Base loader class
- PDF loader
- Markdown loader
- Loader factory (extension-based selection)
- File integrity checker
"""

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.markdown_loader import MarkdownLoader
from src.libs.loader.loader_factory import ExtensionRoutingLoader, LoaderFactory
from src.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker

__all__ = [
    "BaseLoader",
    "PdfLoader",
    "MarkdownLoader",
    "LoaderFactory",
    "ExtensionRoutingLoader",
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
]
