"""Live DeepSeek call (needs DEEPSEEK_API_KEY).

Run manually: ``uv run pytest -q -m llm tests/integration/test_deepseek_live.py``.
"""

import json
import os
from pathlib import Path

import pytest

from rift_common.llm import LLMFactory
from rift_common.settings import load_settings

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set"),
]


def test_deepseek_returns_valid_json() -> None:
    settings = load_settings(REPO_ROOT / "config" / "settings.yaml")
    llm = LLMFactory.create(settings.llm["default"])
    try:
        result = llm.chat(
            [
                {"role": "system", "content": '只输出 JSON 对象，格式为 {"mode": string}。'},
                {"role": "user", "content": "我想打大乱斗，mode 用英文小写。"},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=50,
        )
    finally:
        llm.close()
    data = json.loads(result.text)
    assert isinstance(data.get("mode"), str)
    assert result.usage.total_tokens > 0
