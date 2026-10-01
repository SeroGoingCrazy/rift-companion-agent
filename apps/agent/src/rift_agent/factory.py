"""Build ``AgentDeps`` and trace sinks from ``config/settings.yaml``."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from rift_agent.deps import AgentDeps
from rift_agent.extractors.base import SlotExtractor
from rift_agent.extractors.llm import LLMSlotExtractor
from rift_agent.extractors.routed import RoutedSlotExtractor
from rift_agent.mcp_clients import BookingClient, KnowledgeClient, McpToolClient, http_connector
from rift_agent.replies.polisher import ReplyPolisher
from rift_agent.replies.templates import ReplyRenderer
from rift_common.llm import BaseLLM, LLMFactory
from rift_common.settings import ExtractorKind, Settings, SettingsError
from rift_common.trace.sinks import create_sinks
from rift_common.trace.sinks.base import TraceSink
from rift_domain.config import DomainConfig

#: Evaluation / deployment configurations (DEV_SPEC 3.4.2) -> (primary, fallback).
EXTRACTOR_CONFIGS: dict[str, tuple[ExtractorKind, ExtractorKind | None]] = {
    "llm": ("llm", None),
    "local": ("local", None),
    "routed": ("local", "llm"),
}


def with_extractor_config(settings: Settings, config: str) -> Settings:
    """``settings`` with the slot extractor routing of ``config`` (llm / local / routed)."""
    if config not in EXTRACTOR_CONFIGS:
        raise SettingsError(f"unknown extractor config {config!r}")
    primary, fallback = EXTRACTOR_CONFIGS[config]
    sx = settings.slot_extractor.model_copy(
        update={"primary": primary, "fallback": fallback, "shadow": None}
    )
    return settings.model_copy(update={"slot_extractor": sx})


def build_single_extractor(kind: str, settings: Settings, llm: BaseLLM) -> SlotExtractor:
    """One extractor without routing: ``llm`` (remote) or ``local`` (llama-server).

    ``local`` talks to llama-server's OpenAI-compatible API with the same prompt and
    strict validation as ``llm``; stage J replaces it with a dedicated LocalSlotExtractor.
    """
    if kind == "llm":
        return LLMSlotExtractor(llm)
    if kind == "local":
        cfg = settings.llm.get("local_slot")
        if cfg is None:
            raise SettingsError("slot_extractor kind 'local' needs llm.local_slot")
        return LLMSlotExtractor(
            LLMFactory.create(cfg), max_tokens=cfg.max_tokens or 300, name="local"
        )
    raise SettingsError(f"unknown slot_extractor kind {kind!r}")


def make_extractor(settings: Settings, llm: BaseLLM) -> SlotExtractor:
    sx = settings.slot_extractor

    def build(kind: str) -> SlotExtractor:
        return build_single_extractor(kind, settings, llm)

    local_cfg = settings.llm.get("local_slot")
    timeout = local_cfg.timeout_s if sx.primary == "local" and local_cfg else 30.0
    return RoutedSlotExtractor(
        build(sx.primary),
        fallback=build(sx.fallback) if sx.fallback else None,
        shadow=build(sx.shadow) if sx.shadow else None,
        max_retries=sx.max_retries,
        timeout_s=timeout,
    )


def build_deps(
    settings: Settings,
    domain: DomainConfig,
    *,
    llm: BaseLLM | None = None,
    booking: BookingClient | None = None,
    knowledge: KnowledgeClient | None = None,
    clock: Callable[[], datetime] = datetime.now,
) -> AgentDeps:
    """Real deps: remote LLM, routed extractor, HTTP MCP clients (overridable for demos)."""
    llm = llm or LLMFactory.create(settings.llm["default"])
    b, k = settings.mcp.booking, settings.mcp.knowledge
    return AgentDeps(
        domain=domain,
        llm=llm,
        extractor=make_extractor(settings, llm),
        booking=booking
        or BookingClient(McpToolClient("booking", http_connector(b.url, timeout_s=b.timeout_s))),
        knowledge=knowledge
        or KnowledgeClient(
            McpToolClient("knowledge", http_connector(k.url, timeout_s=k.timeout_s)),
            collections=k.collections or ((k.collection_default,) if k.collection_default else ()),
        ),
        replies=ReplyRenderer(domain),
        polisher=ReplyPolisher(llm) if settings.reply.polish else None,
        clock=clock,
        history_turns=settings.session.history_turns,
    )


def build_sinks(settings: Settings) -> list[TraceSink]:
    return create_sinks(settings.trace)
