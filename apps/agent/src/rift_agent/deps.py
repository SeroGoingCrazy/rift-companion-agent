"""Runtime dependencies of the graph, passed as ``config["configurable"]["deps"]``.

They are never checkpointed: a restarted process builds new deps and resumes the same
``thread_id`` from SQLite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_core.runnables import RunnableConfig

from rift_common.llm import BaseLLM
from rift_domain.config import DomainConfig


@dataclass
class AgentDeps:
    domain: DomainConfig
    #: Remote LLM for classification, consult answers and reply polishing.
    llm: BaseLLM
    clock: Callable[[], datetime] = datetime.now
    history_turns: int = 6
    extras: dict[str, Any] = field(default_factory=dict)

    def now(self) -> datetime:
        return self.clock().replace(microsecond=0)


def get_deps(config: RunnableConfig) -> AgentDeps:
    deps = (config.get("configurable") or {}).get("deps")
    if not isinstance(deps, AgentDeps):
        raise RuntimeError("graph config is missing configurable.deps (AgentDeps)")
    return deps
