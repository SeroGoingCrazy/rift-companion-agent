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
from rift_common.settings import Settings, SettingsError
from rift_common.trace.sinks import create_sinks
from rift_common.trace.sinks.base import TraceSink
from rift_domain.config import DomainConfig


def make_extractor(settings: Settings, llm: BaseLLM) -> SlotExtractor:
    sx = settings.slot_extractor

    def build(kind: str) -> SlotExtractor:
        if kind == "llm":
            return LLMSlotExtractor(llm)
        # The local llama-server extractor (stage J) plugs in here.
        raise SettingsError(f"slot_extractor kind {kind!r} is not available yet")

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
