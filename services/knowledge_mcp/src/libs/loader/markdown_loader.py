"""Markdown Loader implementation.

Parses a Markdown file into a Document and splits it into sections along the
ATX heading hierarchy (``#`` .. ``######``). Each section keeps the heading
path that leads to it, so downstream chunks never cross a section boundary
and always carry their context (e.g. ``退款规则 > 取消档位``).

Document metadata:
- source_path: absolute file path (BaseLoader contract)
- source: path relative to ``base_dir`` (posix), or the file name
- collection: explicit ``collection`` or the parent directory name
- doc_type: ``"markdown"``
- title: first H1, falling back to the file stem
- sections: list of ``{"title_path", "level", "text", "start_offset", "end_offset"}``
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

TITLE_PATH_SEPARATOR = " > "

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*(?:#+[ \t]*)?$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


class MarkdownLoader(BaseLoader):
    """Loader for ``.md`` files that splits content by heading level."""

    def __init__(
        self,
        collection: Optional[str] = None,
        base_dir: Optional[str | Path] = None,
        encoding: str = "utf-8",
    ):
        """Initialize Markdown Loader.

        Args:
            collection: Collection name written into metadata. Defaults to
                the name of the file's parent directory.
            base_dir: Directory that ``source`` is made relative to. Defaults
                to the file's parent directory (``source`` is the file name).
            encoding: File encoding.
        """
        self.collection = collection
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else None
        self.encoding = encoding

    def load(self, file_path: str | Path) -> Document:
        """Load a Markdown file and split it into heading sections.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a Markdown file.
        """
        path = self._validate_file(file_path)
        if path.suffix.lower() not in (".md", ".markdown"):
            raise ValueError(f"File is not a Markdown file: {path}")

        raw = path.read_bytes()
        text = raw.decode(self.encoding).lstrip("﻿").replace("\r\n", "\n")
        doc_hash = hashlib.sha256(raw).hexdigest()

        title = self._extract_title(text) or path.stem
        sections = split_sections(text, default_title=title)

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "source": self._relative_source(path),
            "collection": self.collection or path.parent.name,
            "doc_type": "markdown",
            "doc_hash": doc_hash,
            "title": title,
            "sections": sections,
        }
        return Document(id=f"doc_{doc_hash[:16]}", text=text, metadata=metadata)

    def _relative_source(self, path: Path) -> str:
        if self.base_dir is not None:
            try:
                return path.relative_to(self.base_dir).as_posix()
            except ValueError:
                pass
        return path.name

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        for level, heading, _start, _end in _iter_headings(text):
            if level == 1:
                return heading
        return None


def split_sections(text: str, default_title: str = "") -> List[Dict[str, Any]]:
    """Split Markdown text into sections along the heading hierarchy.

    A section runs from a heading to the next heading of any level. Its text
    starts with the heading lines of all its ancestors, then its own heading
    and body, so it reads correctly on its own. Headings without a body (pure
    containers such as an H1 directly followed by an H2) produce no section.
    Text before the first heading becomes a section titled ``default_title``.
    Headings inside fenced code blocks are ignored.
    """
    headings = list(_iter_headings(text))
    sections: List[Dict[str, Any]] = []

    preamble_end = headings[0][2] if headings else len(text)
    preamble = text[:preamble_end].strip()
    if preamble:
        sections.append({
            "title_path": default_title,
            "level": 0,
            "text": preamble,
            "start_offset": 0,
            "end_offset": preamble_end,
        })

    stack: List[tuple[int, str]] = []
    for i, (level, heading, start, body_start) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))

        end = headings[i + 1][2] if i + 1 < len(headings) else len(text)
        body = text[body_start:end].strip()
        if not body:
            continue

        heading_lines = "\n".join(f"{'#' * lvl} {name}" for lvl, name in stack)
        sections.append({
            "title_path": TITLE_PATH_SEPARATOR.join(name for _lvl, name in stack),
            "level": level,
            "text": f"{heading_lines}\n\n{body}",
            "start_offset": start,
            "end_offset": end,
        })
    return sections


def _iter_headings(text: str):
    """Yield ``(level, heading, line_start, body_start)`` for ATX headings."""
    offset = 0
    fence: Optional[str] = None
    for line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(line)
        stripped = line.rstrip("\n")

        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0] * 3
            elif marker.startswith(fence):
                fence = None
            continue
        if fence is not None:
            continue

        match = _HEADING_RE.match(stripped)
        if match:
            yield len(match.group(1)), match.group(2).strip(), line_start, offset
