"""Teacher configs (``training/configs/inference/teacher_<name>.yaml``) and backends."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from rift_training.inference.base import Teacher
from rift_training.inference.mock import MockTeacher
from rift_training.inference.teacher_openai import OpenAITeacher, TeacherConfig

INFERENCE_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "inference"
#: Overrides the configured model without editing the YAML.
MODEL_ENV = "TEACHER_MODEL"


def load_teacher_config(name_or_path: str | Path, *, model: str | None = None) -> TeacherConfig:
    """``openai`` / ``mock`` or a YAML path; ``model`` > ``$TEACHER_MODEL`` > the file."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = INFERENCE_CONFIGS / f"teacher_{name_or_path}.yaml"
    data = yaml.safe_load(path.read_text("utf-8"))
    override = model or os.environ.get(MODEL_ENV, "").strip()
    if override:
        data["model"] = override
    return TeacherConfig.model_validate(data)


def make_teacher(config: TeacherConfig) -> Teacher:
    if config.provider == "mock":
        return MockTeacher(model=config.model)
    return OpenAITeacher(config)
