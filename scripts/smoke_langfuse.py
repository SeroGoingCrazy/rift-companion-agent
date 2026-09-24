"""Send one fake conversation turn to Langfuse Cloud to verify the sink end to end.

Usage (needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY and DEEPSEEK_API_KEY set; no LLM call
is made, the key is only needed to load settings):

    uv run python scripts/smoke_langfuse.py
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

from rift_common.llm import LLMFactory
from rift_common.settings import LLMConfig, load_settings
from rift_common.trace import span, start_turn
from rift_common.trace.sinks import LangfuseSink

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    settings = load_settings(REPO_ROOT / "config" / "settings.yaml")
    sink = LangfuseSink(settings.trace.langfuse)
    if not sink.enabled:
        print("Langfuse sink disabled: check trace.langfuse.enabled and LANGFUSE_* env vars.")
        return 1

    llm = LLMFactory.create(LLMConfig(provider="mock", model="smoke-mock"), default="好的")
    session_id = f"smoke-{uuid.uuid4().hex[:8]}"
    with start_turn(
        session_id=session_id, user_id="smoke_user", turn_idx=0, sinks=[sink], input="约个双排"
    ) as root:
        with span("extract_slots"):
            llm.chat([{"role": "user", "content": "约个双排"}])
        with span("tool:find_companions", kind="tool", input={"top_k": 3}) as tool:
            tool.set_attr("output", ["阿狸酱"])
        root.set_attr("reply", "smoke test reply")
    sink.close()
    print(f"sent trace {root.trace_id} (session {session_id}); check it in Langfuse Cloud.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
