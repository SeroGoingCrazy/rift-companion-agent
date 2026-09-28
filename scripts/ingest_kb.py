"""Ingest the generated knowledge base (``kb/``) into knowledge-mcp, one collection per folder.

- ``kb/platform_rules/*.md``     -> collection ``platform_rules``
- ``kb/modes_and_ranks/*.md``    -> collection ``modes_and_ranks``
- ``kb/companion_profiles/*.md`` -> collection ``companion_profiles``

Each file goes through the knowledge-mcp ingestion pipeline (MarkdownLoader -> section
chunks -> local embedding -> Chroma + BM25). Unchanged files are skipped by their SHA256;
when a file changed or disappeared, its old chunks are deleted first so stale numbers
never stay searchable.

Usage::

    uv run python scripts/gen_kb_docs.py      # regenerate kb/ from domain.yaml + seed
    uv run python scripts/ingest_kb.py        # incremental
    uv run python scripts/ingest_kb.py --force --collections platform_rules
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
COLLECTIONS = ("platform_rules", "modes_and_ranks", "companion_profiles")


class Pipeline(Protocol):
    def run(self, file_path: str) -> Any: ...


class Remover(Protocol):
    def delete_document(
        self, source_path: str, collection: str = ..., source_hash: str | None = ...
    ) -> Any: ...


class Integrity(Protocol):
    def compute_sha256(self, file_path: str) -> str: ...

    def list_processed(self, collection: str | None = None) -> list[dict[str, Any]]: ...


@dataclass
class CollectionReport:
    collection: str
    ingested: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks: int = 0

    def summary(self) -> str:
        return (
            f"{self.collection}: ingested {len(self.ingested)} ({self.chunks} chunks), "
            f"unchanged {len(self.skipped)}, removed {len(self.removed)}, "
            f"failed {len(self.failed)}"
        )


def sync_collection(
    collection: str,
    files: Iterable[Path],
    *,
    pipeline: Pipeline,
    integrity: Integrity,
    remover: Remover,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> CollectionReport:
    """Bring one collection in line with ``files``.

    Old chunks of a file are deleted before it is re-ingested (content changed or
    ``force``), and files that no longer exist are deleted from the collection.
    """
    report = CollectionReport(collection)
    paths = [str(Path(f).resolve()) for f in files]
    recorded = {rec["file_path"]: rec["file_hash"] for rec in integrity.list_processed(collection)}

    for path in sorted(set(recorded) - set(paths)):
        remover.delete_document(path, collection, source_hash=recorded[path])
        report.removed.append(path)
        log(f"  - removed {Path(path).name} (file no longer exists)")

    for path in paths:
        name = Path(path).name
        old_hash = recorded.get(path)
        if old_hash is not None and (force or integrity.compute_sha256(path) != old_hash):
            remover.delete_document(path, collection, source_hash=old_hash)

        result = pipeline.run(path)
        if not result.success:
            report.failed[path] = result.error or "unknown error"
            log(f"  ! failed {name}: {result.error}")
        elif result.stages.get("integrity", {}).get("skipped"):
            report.skipped.append(path)
        else:
            report.ingested.append(path)
            report.chunks += result.chunk_count
            log(f"  + {name}: {result.chunk_count} chunks")
    return report


def discover(kb_dir: Path, collection: str) -> list[Path]:
    folder = kb_dir / collection
    if not folder.is_dir():
        raise FileNotFoundError(
            f"{folder} not found; generate it first: uv run python scripts/gen_kb_docs.py"
        )
    return sorted(folder.glob("*.md"))


def _build_backends(settings: Any, collection: str, force: bool) -> tuple[Any, Any, Any]:
    """Create the real pipeline, integrity checker and document manager for a collection."""
    from src.core.settings import resolve_path
    from src.ingestion.document_manager import DocumentManager
    from src.ingestion.pipeline import IngestionPipeline
    from src.ingestion.storage.image_storage import ImageStorage
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    pipeline = IngestionPipeline(settings, collection=collection, force=force)
    manager = DocumentManager(
        VectorStoreFactory.create(settings, collection_name=collection),
        pipeline.bm25_indexer,
        ImageStorage(
            db_path=str(resolve_path("data/db/image_index.db")),
            images_root=str(resolve_path("data/images")),
        ),
        pipeline.integrity_checker,
    )
    return pipeline, pipeline.integrity_checker, manager


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kb-dir", type=Path, default=REPO_ROOT / "kb")
    parser.add_argument(
        "--collections",
        nargs="+",
        choices=COLLECTIONS,
        default=list(COLLECTIONS),
        help="collections to ingest (default: all three)",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=REPO_ROOT / "services" / "knowledge_mcp" / "config" / "settings.yaml",
    )
    parser.add_argument("--force", action="store_true", help="re-ingest files even when unchanged")
    parser.add_argument("--verbose", action="store_true", help="show pipeline stage logs")
    args = parser.parse_args(argv)
    if not args.verbose:
        logging.getLogger("src").setLevel(logging.WARNING)

    from src.core.settings import load_settings

    settings = load_settings(str(args.settings))
    print(f"embedding: {settings.embedding.provider} / {settings.embedding.model}")

    reports = []
    for collection in args.collections:
        files = discover(args.kb_dir, collection)
        print(f"[{collection}] {len(files)} files")
        pipeline, integrity, manager = _build_backends(settings, collection, args.force)
        reports.append(
            sync_collection(
                collection,
                files,
                pipeline=pipeline,
                integrity=integrity,
                remover=manager,
                force=args.force,
            )
        )

    print()
    for report in reports:
        print(report.summary())
    return 1 if any(r.failed for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
