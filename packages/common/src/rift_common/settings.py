"""Application settings: load ``config/settings.yaml``, expand ``${ENV}`` refs, validate.

Rules:
- Every section is a frozen Pydantic model with ``extra="forbid"``; a missing field fails fast
  with its dotted path (``llm.default.provider is required``).
- Secrets may only come from environment variables. A literal ``sk-...`` anywhere in the YAML,
  or a literal value in a secret field (``api_key`` / ``secret_key`` / ``public_key``), is rejected.
- ``${VAR}`` requires ``VAR`` to be set; ``${VAR:-default}`` falls back to ``default``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SECRET_FIELDS = frozenset({"api_key", "secret_key", "public_key"})
_PLAINTEXT_KEY = re.compile(r"^\s*sk-")


class SettingsError(ValueError):
    """Raised when settings cannot be loaded or are invalid."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LLMConfig(_Frozen):
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    timeout_s: float = Field(default=30.0, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    #: Mock provider only: YAML/JSON fixture with canned responses.
    fixture: str | None = None


class EmbeddingConfig(_Frozen):
    provider: str
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout_s: float = Field(default=30.0, gt=0)
    #: Mock provider only: vector size (default 64).
    dimension: int | None = Field(default=None, gt=0)


ExtractorKind = Literal["llm", "local"]


class SlotExtractorConfig(_Frozen):
    primary: ExtractorKind
    fallback: ExtractorKind | None = None
    shadow: ExtractorKind | None = None
    max_retries: int = Field(default=1, ge=0)


class ReplyConfig(_Frozen):
    polish: bool = False


class MCPServerConfig(_Frozen):
    url: str
    timeout_s: float = Field(default=15.0, gt=0)
    collection_default: str | None = None
    #: Knowledge server only: collections a consult question is searched in.
    collections: tuple[str, ...] = ()


class MCPConfig(_Frozen):
    booking: MCPServerConfig
    knowledge: MCPServerConfig


class LangfuseConfig(_Frozen):
    enabled: bool = False
    host: str = "https://cloud.langfuse.com"
    public_key: str | None = None
    secret_key: str | None = None


class TraceConfig(_Frozen):
    sqlite_path: str
    langfuse: LangfuseConfig = LangfuseConfig()


class SessionConfig(_Frozen):
    checkpoint_path: str
    history_turns: int = Field(default=6, ge=0)


class Settings(_Frozen):
    llm: dict[str, LLMConfig]
    embedding: EmbeddingConfig
    slot_extractor: SlotExtractorConfig
    reply: ReplyConfig = ReplyConfig()
    mcp: MCPConfig
    trace: TraceConfig
    session: SessionConfig


# Which ``llm.<name>`` entry each extractor kind needs.
_EXTRACTOR_LLM_KEY: dict[str, str] = {"llm": "default", "local": "local_slot"}


def _join(path: tuple[str | int, ...]) -> str:
    return ".".join(str(p) for p in path) or "<root>"


def _check_no_plaintext_secrets(obj: Any, path: tuple[str | int, ...] = ()) -> None:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            _check_no_plaintext_secrets(value, (*path, str(key)))
        return
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            _check_no_plaintext_secrets(value, (*path, i))
        return
    if not isinstance(obj, str):
        return
    if _PLAINTEXT_KEY.match(obj):
        raise SettingsError(
            f"{_join(path)}: plaintext secret found in settings; use an env var like ${{VAR}}"
        )
    is_secret_field = bool(path) and path[-1] in _SECRET_FIELDS
    if is_secret_field and obj.strip() and not _ENV_REF.fullmatch(obj.strip()):
        raise SettingsError(f"{_join(path)}: secrets must be referenced as ${{ENV_VAR}}")


def expand_env(
    obj: Any, env: Mapping[str, str] | None = None, _path: tuple[str | int, ...] = ()
) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in all strings of ``obj``.

    Raises ``SettingsError`` naming the variable (and its settings path) when a required
    variable is unset.
    """
    environ = os.environ if env is None else env
    if isinstance(obj, Mapping):
        return {k: expand_env(v, environ, (*_path, str(k))) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v, environ, (*_path, i)) for i, v in enumerate(obj)]
    if not isinstance(obj, str):
        return obj

    def _sub(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = environ.get(name)
        if value is not None:
            return value
        if default is not None:
            return default
        raise SettingsError(
            f"environment variable {name} is not set (referenced by {_join(_path)})"
        )

    expanded = _ENV_REF.sub(_sub, obj)
    # An env var that expands to empty for an optional field means "not configured".
    if expanded == "" and _ENV_REF.fullmatch(obj):
        return None
    return expanded


def _format_validation_error(err: ValidationError) -> str:
    messages = []
    for e in err.errors():
        path = _join(tuple(e["loc"]))
        if e["type"] == "missing":
            messages.append(f"{path} is required")
        elif e["type"] == "extra_forbidden":
            messages.append(f"{path} is not a recognized setting")
        else:
            messages.append(f"{path}: {e['msg']}")
    return "; ".join(messages)


def validate_settings(s: Settings) -> Settings:
    """Cross-field checks that a single-field schema cannot express."""
    if "default" not in s.llm:
        raise SettingsError("llm.default is required")

    sx = s.slot_extractor
    if sx.fallback is not None and sx.fallback == sx.primary:
        raise SettingsError("slot_extractor.fallback must differ from slot_extractor.primary")
    if sx.shadow is not None and sx.shadow == sx.primary:
        raise SettingsError("slot_extractor.shadow must differ from slot_extractor.primary")
    for role in ("primary", "fallback", "shadow"):
        kind = getattr(sx, role)
        if kind is None:
            continue
        needed = _EXTRACTOR_LLM_KEY[kind]
        if needed not in s.llm:
            raise SettingsError(f"slot_extractor.{role}={kind} requires llm.{needed}")
    return s


def load_settings(
    path: str | Path = DEFAULT_SETTINGS_PATH, env: Mapping[str, str] | None = None
) -> Settings:
    """Load, expand and validate settings from a YAML file."""
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SettingsError(f"settings file not found: {p}") from exc
    except yaml.YAMLError as exc:
        raise SettingsError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SettingsError(f"{p}: top level must be a mapping")

    _check_no_plaintext_secrets(raw)
    expanded = expand_env(raw, env)
    try:
        settings = Settings.model_validate(expanded)
    except ValidationError as exc:
        raise SettingsError(_format_validation_error(exc)) from exc
    return validate_settings(settings)
