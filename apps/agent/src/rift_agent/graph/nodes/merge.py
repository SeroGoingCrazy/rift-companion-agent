"""``merge_state``: delta -> state (tri-state merge + time resolution) -> domain rules.

Wraps ``rift_domain.merge.merge`` (which also resolves ``start_time_expr``) and
``rift_domain.rules.apply_rules``. When a change invalidates the selected companion
(e.g. "改成三小时"), their id is kept in ``diff.keep_companion_id`` so the next search
can re-select them if they are still free.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from rift_agent.deps import get_deps
from rift_agent.graph.nodes.extract import load_extraction
from rift_agent.graph.state import AgentState, Phase, booking_state, dump_booking
from rift_agent.graph.tracing import traced_node
from rift_domain.merge import merge
from rift_domain.rules import apply_rules


@traced_node("merge_state")
async def merge_state(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    extraction = load_extraction(state)
    assert extraction is not None  # routed here only with a valid extraction
    old = booking_state(state)
    new, diff = merge(old, extraction.delta, deps.now())
    new, rules = apply_rules(new, deps.domain)

    time = diff.time_result
    summary: dict[str, Any] = {
        "changed": list(diff.changed_fields),
        "invalidated": list(diff.invalidated),
        "cleared": [f.value for f in rules.cleared_fields],
        "invalid": {f.value: reason for f, reason in rules.invalid_fields.items()},
        "time_expr": new.start_time_expr,
        "time_reason": time.reason if time is not None and not time.ok else None,
    }
    if old.selected_companion_id is not None and new.selected_companion_id is None:
        name_changed = "companion_name" in diff.changed
        if not name_changed:
            summary["keep_companion_id"] = old.selected_companion_id

    return {
        "booking": dump_booking(new),
        "pending_action": new.pending_action.value if new.pending_action else None,
        "diff": summary,
        "phase": Phase.BOOKING.value,
    }
