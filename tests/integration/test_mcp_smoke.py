"""D7: the stdio smoke script runs the Claude Desktop flow end to end."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mcp_smoke", REPO_ROOT / "scripts" / "mcp_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_smoke_script_books_over_stdio(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    code = _load().main(["--db", str(tmp_path / "desktop.db"), "--now", "2026-10-01T14:00"])
    out = capfd.readouterr().out
    assert code == 0, out
    assert "booked #1: pending_payment" in out
    assert out.strip().endswith("OK")


@pytest.mark.slow
def test_smoke_script_reports_failure(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    # Unknown user: create_booking fails with NOT_FOUND and the script exits 1.
    code = _load().main(["--db", str(tmp_path / "d.db"), "--user-id", "999"])
    assert code == 1
    assert "NOT_FOUND" in capfd.readouterr().err
