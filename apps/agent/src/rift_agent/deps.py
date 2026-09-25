"""Runtime dependencies of the graph, passed as ``config["configurable"]["deps"]``.

They are never checkpointed: a restarted process builds new deps and resumes the same
``thread_id`` from SQLite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig

from rift_common.llm import BaseLLM
from rift_domain.config import DomainConfig

if TYPE_CHECKING:
    from rift_agent.extractors.base import SlotExtractor
    from rift_agent.mcp_clients import BookingClient, KnowledgeClient
    from rift_agent.replies.templates import ReplyRenderer


@dataclass
class AgentDeps:
    domain: DomainConfig
    #: Remote LLM for classification, consult answers and reply polishing.
    llm: BaseLLM
    extractor: SlotExtractor | None = None
    booking: BookingClient | None = None
    knowledge: KnowledgeClient | None = None
    replies: ReplyRenderer | None = None
    clock: Callable[[], datetime] = datetime.now
    history_turns: int = 6
    extras: dict[str, Any] = field(default_factory=dict)

    def now(self) -> datetime:
        return self.clock().replace(microsecond=0)

    def require_extractor(self) -> SlotExtractor:
        if self.extractor is None:
            raise RuntimeError("AgentDeps.extractor is not configured")
        return self.extractor

    def require_booking(self) -> BookingClient:
        if self.booking is None:
            raise RuntimeError("AgentDeps.booking is not configured")
        return self.booking

    def require_replies(self) -> ReplyRenderer:
        if self.replies is None:
            raise RuntimeError("AgentDeps.replies is not configured")
        return self.replies


def get_deps(config: RunnableConfig) -> AgentDeps:
    deps = (config.get("configurable") or {}).get("deps")
    if not isinstance(deps, AgentDeps):
        raise RuntimeError("graph config is missing configurable.deps (AgentDeps)")
    return deps
