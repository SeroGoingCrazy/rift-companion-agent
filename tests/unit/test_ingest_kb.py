"""E4: ingest_kb.py keeps each collection in line with kb/ (skip, re-ingest, remove)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ingest_kb", REPO_ROOT / "scripts" / "ingest_kb.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ingest_kb = _load()


@dataclass
class Result:
    success: bool = True
    chunk_count: int = 0
    error: str | None = None
    stages: dict[str, Any] = field(default_factory=dict)


class FakeStore:
    """Integrity records + pipeline + remover sharing one in-memory state."""

    def __init__(self) -> None:
        self.records: dict[str, str] = {}  # path -> hash
        self.runs: list[str] = []
        self.deleted: list[tuple[str, str]] = []  # (path, hash)
        self.force = False
        self.fail: set[str] = set()

    # Integrity
    def compute_sha256(self, file_path: str) -> str:
        return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()

    def list_processed(self, collection: str | None = None) -> list[dict[str, Any]]:
        return [{"file_path": p, "file_hash": h} for p, h in self.records.items()]

    # Pipeline (mirrors IngestionPipeline's SHA256 skip)
    def run(self, file_path: str) -> Result:
        self.runs.append(Path(file_path).name)
        if Path(file_path).name in self.fail:
            return Result(success=False, error="boom")
        file_hash = self.compute_sha256(file_path)
        if not self.force and file_hash in self.records.values():
            return Result(stages={"integrity": {"skipped": True}})
        self.records[file_path] = file_hash
        return Result(chunk_count=2, stages={"integrity": {"skipped": False}})

    # Remover
    def delete_document(
        self, source_path: str, collection: str = "default", source_hash: str | None = None
    ) -> None:
        self.deleted.append((Path(source_path).name, source_hash or ""))
        self.records.pop(source_path, None)


@pytest.fixture
def kb(tmp_path: Path) -> Path:
    folder = tmp_path / "kb" / "platform_rules"
    folder.mkdir(parents=True)
    (folder / "billing.md").write_text("# 计费\n\n系数 1.2", encoding="utf-8")
    (folder / "refund.md").write_text("# 退款\n\n2–24 小时退 50%", encoding="utf-8")
    return tmp_path / "kb"


def _sync(kb: Path, store: FakeStore, force: bool = False) -> Any:
    store.force = force
    files = ingest_kb.discover(kb, "platform_rules")
    return ingest_kb.sync_collection(
        "platform_rules",
        files,
        pipeline=store,
        integrity=store,
        remover=store,
        force=force,
        log=lambda _msg: None,
    )


def test_first_run_ingests_every_file(kb: Path) -> None:
    store = FakeStore()

    report = _sync(kb, store)

    assert [Path(p).name for p in report.ingested] == ["billing.md", "refund.md"]
    assert report.chunks == 4
    assert store.deleted == []


def test_unchanged_files_are_skipped(kb: Path) -> None:
    store = FakeStore()
    _sync(kb, store)

    report = _sync(kb, store)

    assert report.ingested == []
    assert len(report.skipped) == 2
    assert store.deleted == []


def test_changed_file_drops_old_chunks_before_reingest(kb: Path) -> None:
    store = FakeStore()
    _sync(kb, store)
    old_hash = store.records[str((kb / "platform_rules" / "refund.md").resolve())]

    (kb / "platform_rules" / "refund.md").write_text("# 退款\n\n2–24 小时退 60%", encoding="utf-8")
    report = _sync(kb, store)

    assert store.deleted == [("refund.md", old_hash)]
    assert [Path(p).name for p in report.ingested] == ["refund.md"]
    assert [Path(p).name for p in report.skipped] == ["billing.md"]


def test_deleted_file_is_removed_from_collection(kb: Path) -> None:
    store = FakeStore()
    _sync(kb, store)
    billing = kb / "platform_rules" / "billing.md"
    billing_hash = store.compute_sha256(str(billing))

    billing.unlink()
    report = _sync(kb, store)

    assert [Path(p).name for p in report.removed] == ["billing.md"]
    assert store.deleted == [("billing.md", billing_hash)]
    assert all("billing.md" not in p for p in store.records)


def test_force_reingests_and_replaces_everything(kb: Path) -> None:
    store = FakeStore()
    _sync(kb, store)

    report = _sync(kb, store, force=True)

    assert len(report.ingested) == 2
    assert sorted(name for name, _ in store.deleted) == ["billing.md", "refund.md"]


def test_failures_are_reported(kb: Path) -> None:
    store = FakeStore()
    store.fail = {"refund.md"}

    report = _sync(kb, store)

    assert [Path(p).name for p in report.failed] == ["refund.md"]
    assert "failed 1" in report.summary()


def test_missing_collection_folder_points_to_generator(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"gen_kb_docs\.py"):
        ingest_kb.discover(tmp_path, "platform_rules")


def test_rejects_unknown_collection() -> None:
    with pytest.raises(SystemExit):
        ingest_kb.main(["--collections", "faq"])
