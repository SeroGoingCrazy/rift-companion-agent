from pathlib import Path
from typing import Any

import pytest
import yaml

from rift_common.settings import (
    Settings,
    SettingsError,
    expand_env,
    load_settings,
    validate_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_config() -> dict[str, Any]:
    return {
        "llm": {
            "default": {
                "provider": "openai_compatible",
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat",
                "api_key": "${DEEPSEEK_API_KEY}",
            },
            "local_slot": {
                "provider": "llama_server",
                "base_url": "http://localhost:8080/v1",
                "model": "rift-slot-0.6b-q4km",
                "timeout_s": 5,
            },
        },
        "embedding": {"provider": "mock"},
        "slot_extractor": {"primary": "llm", "fallback": None, "shadow": None},
        "mcp": {
            "booking": {"url": "http://localhost:8101/mcp"},
            "knowledge": {
                "url": "http://localhost:8102/mcp",
                "collection_default": "platform_rules",
            },
        },
        "trace": {
            "sqlite_path": "data/traces.db",
            "langfuse": {
                "enabled": True,
                "public_key": "${LANGFUSE_PUBLIC_KEY:-}",
                "secret_key": "${LANGFUSE_SECRET_KEY:-}",
            },
        },
        "session": {"checkpoint_path": "data/checkpoints.db"},
    }


ENV = {"DEEPSEEK_API_KEY": "test-key"}


def _write(tmp_path: Path, cfg: dict[str, Any]) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


# --- happy path -------------------------------------------------------------


def test_load_valid_settings(tmp_path: Path) -> None:
    s = load_settings(_write(tmp_path, _base_config()), env=ENV)
    assert isinstance(s, Settings)
    assert s.llm["default"].api_key == "test-key"
    assert s.llm["local_slot"].timeout_s == 5
    assert s.slot_extractor.max_retries == 1
    assert s.reply.polish is False
    assert s.session.history_turns == 6


def test_optional_env_with_empty_default_becomes_none(tmp_path: Path) -> None:
    s = load_settings(_write(tmp_path, _base_config()), env=ENV)
    assert s.trace.langfuse.public_key is None
    assert s.trace.langfuse.secret_key is None


def test_settings_are_frozen(tmp_path: Path) -> None:
    s = load_settings(_write(tmp_path, _base_config()), env=ENV)
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError
        s.reply.polish = True  # type: ignore[misc]


def test_repo_settings_yaml_loads() -> None:
    s = load_settings(REPO_ROOT / "config" / "settings.yaml", env=ENV)
    assert s.llm["default"].provider == "openai_compatible"
    assert s.mcp.knowledge.collection_default == "platform_rules"


# --- fail-fast on missing fields ---------------------------------------------


def test_missing_llm_default_provider_reports_path(tmp_path: Path) -> None:
    cfg = _base_config()
    del cfg["llm"]["default"]["provider"]
    with pytest.raises(SettingsError, match=r"llm\.default\.provider is required"):
        load_settings(_write(tmp_path, cfg), env=ENV)


def test_missing_section_reports_path(tmp_path: Path) -> None:
    cfg = _base_config()
    del cfg["mcp"]["booking"]
    with pytest.raises(SettingsError, match=r"mcp\.booking is required"):
        load_settings(_write(tmp_path, cfg), env=ENV)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["reply"] = {"polish": False, "polsh": True}
    with pytest.raises(SettingsError, match=r"reply\.polsh is not a recognized setting"):
        load_settings(_write(tmp_path, cfg), env=ENV)


def test_missing_llm_default_entry(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["llm"] = {"local_slot": cfg["llm"]["local_slot"]}
    with pytest.raises(SettingsError, match=r"llm\.default is required"):
        load_settings(_write(tmp_path, cfg), env=ENV)


def test_missing_file() -> None:
    with pytest.raises(SettingsError, match="settings file not found"):
        load_settings("does/not/exist.yaml")


# --- secrets only from env ---------------------------------------------------


def test_plaintext_sk_key_rejected(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["llm"]["default"]["api_key"] = "sk-abc123"
    with pytest.raises(SettingsError, match=r"llm\.default\.api_key: plaintext secret"):
        load_settings(_write(tmp_path, cfg), env=ENV)


def test_plaintext_sk_anywhere_rejected(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["mcp"]["booking"]["url"] = "sk-leaked"
    with pytest.raises(SettingsError, match="plaintext secret"):
        load_settings(_write(tmp_path, cfg), env=ENV)


def test_literal_value_in_secret_field_rejected(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["trace"]["langfuse"]["public_key"] = "pk-lf-123"
    with pytest.raises(SettingsError, match=r"trace\.langfuse\.public_key: secrets must be"):
        load_settings(_write(tmp_path, cfg), env=ENV)


# --- env expansion -----------------------------------------------------------


def test_missing_env_var_named(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="DEEPSEEK_API_KEY is not set"):
        load_settings(_write(tmp_path, _base_config()), env={})


@pytest.mark.parametrize(
    ("value", "env", "expected"),
    [
        ("${A}", {"A": "1"}, "1"),
        ("http://${HOST}:8080/v1", {"HOST": "llama"}, "http://llama:8080/v1"),
        ("${A:-fallback}", {}, "fallback"),
        ("${A:-fallback}", {"A": "set"}, "set"),
        ("${A:-}", {}, None),
        ("plain", {}, "plain"),
        (42, {}, 42),
        ([{"k": "${A}"}], {"A": "x"}, [{"k": "x"}]),
    ],
)
def test_expand_env(value: Any, env: dict[str, str], expected: Any) -> None:
    assert expand_env(value, env) == expected


def test_expand_env_reports_path() -> None:
    with pytest.raises(SettingsError, match=r"MISSING is not set \(referenced by a\.b\)"):
        expand_env({"a": {"b": "${MISSING}"}}, {})


# --- cross-field validation --------------------------------------------------


def test_fallback_same_as_primary_rejected(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["slot_extractor"]["fallback"] = "llm"
    with pytest.raises(SettingsError, match="fallback must differ"):
        load_settings(_write(tmp_path, cfg), env=ENV)


def test_local_extractor_requires_local_slot_llm(tmp_path: Path) -> None:
    cfg = _base_config()
    del cfg["llm"]["local_slot"]
    cfg["slot_extractor"]["shadow"] = "local"
    with pytest.raises(SettingsError, match=r"shadow=local requires llm\.local_slot"):
        load_settings(_write(tmp_path, cfg), env=ENV)


def test_invalid_extractor_kind(tmp_path: Path) -> None:
    cfg = _base_config()
    cfg["slot_extractor"]["primary"] = "gpt"
    with pytest.raises(SettingsError, match=r"slot_extractor\.primary"):
        load_settings(_write(tmp_path, cfg), env=ENV)


def test_validate_settings_returns_same_object(tmp_path: Path) -> None:
    s = load_settings(_write(tmp_path, _base_config()), env=ENV)
    assert validate_settings(s) is s
