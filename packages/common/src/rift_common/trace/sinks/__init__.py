from rift_common.settings import TraceConfig
from rift_common.trace.sinks.base import TraceSink
from rift_common.trace.sinks.langfuse import LangfuseSink
from rift_common.trace.sinks.sqlite import SqliteSink


def create_sinks(config: TraceConfig) -> list[TraceSink]:
    """Build the configured dual-write sinks: SQLite always, Langfuse when enabled."""
    sinks: list[TraceSink] = [SqliteSink(config.sqlite_path)]
    if config.langfuse.enabled:
        sinks.append(LangfuseSink(config.langfuse))
    return sinks


__all__ = ["LangfuseSink", "SqliteSink", "TraceSink", "create_sinks"]
