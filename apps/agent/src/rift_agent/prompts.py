"""Prompt files live in ``config/prompts`` (shared with training); loaded once and cached."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PROMPTS_DIR = REPO_ROOT / "config" / "prompts"


@cache
def load_prompt(name: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    return (prompts_dir / name).read_text(encoding="utf-8").strip()


def prompt_sha256(name: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    """Hash recorded by training data cards; the agent compares it at start-up."""
    return hashlib.sha256((prompts_dir / name).read_bytes()).hexdigest()
