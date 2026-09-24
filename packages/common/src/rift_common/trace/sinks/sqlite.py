"""SQLite sink: one ``turns`` row and N ``spans`` rows per turn, written in one transaction."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from rift_common.trace.context import TraceContext
from rift_common.trace.sinks.base import TraceSink
from rift_common.trace.span import Span

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    trace_id      TEXT PRIMARY KEY,
    session_id    TEXT,
    user_id       TEXT,
    turn_idx      INTEGER,
    input         TEXT,
    reply         TEXT,
    phase_before  TEXT,
    phase_after   TEXT,
    status        TEXT NOT NULL,
    error         TEXT,
    start_time    REAL NOT NULL,
    end_time      REAL,
    latency_ms    REAL,
    attrs_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, turn_idx);

CREATE TABLE IF NOT EXISTS spans (
    span_id     TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    parent_id   TEXT,
    session_id  TEXT,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    start_time  REAL NOT NULL,
    end_time    REAL,
    latency_ms  REAL,
    attrs_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_session_name ON spans(session_id, name);
"""

# Root-span attributes promoted to dedicated ``turns`` columns.
_TURN_COLUMNS = ("input", "reply", "phase_before", "phase_after")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return _dumps(value)


class SqliteSink(TraceSink):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def on_turn_end(self, trace: TraceContext) -> None:
        root = trace.root
        if root is None:
            return
        extra = {k: v for k, v in root.attrs.items() if k not in _TURN_COLUMNS}
        turn_row = (
            trace.trace_id,
            trace.session_id,
            trace.user_id,
            trace.turn_idx,
            *(_text(root.attrs.get(k)) for k in _TURN_COLUMNS),
            root.status,
            root.error,
            root.start_time,
            root.end_time,
            root.duration_ms,
            _dumps(extra),
        )
        span_rows = [self._span_row(s, trace) for s in trace.spans]
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", turn_row
            )
            conn.executemany(
                "INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", span_rows
            )

    @staticmethod
    def _span_row(s: Span, trace: TraceContext) -> tuple[Any, ...]:
        return (
            s.span_id,
            s.trace_id,
            s.parent_id,
            trace.session_id,
            s.name,
            s.kind,
            s.status,
            s.error,
            s.start_time,
            s.end_time,
            s.duration_ms,
            _dumps(s.attrs),
        )

    # --- queries (dashboard / tests) -------------------------------------------------

    def list_turns(self, session_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY turn_idx, start_time",
                (session_id,),
            ).fetchall()
        return [self._decode(r) for r in rows]

    def list_spans(
        self, *, trace_id: str | None = None, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        if (trace_id is None) == (session_id is None):
            raise ValueError("pass exactly one of trace_id or session_id")
        column, value = ("trace_id", trace_id) if trace_id else ("session_id", session_id)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM spans WHERE {column} = ? ORDER BY start_time", (value,)
            ).fetchall()
        return [self._decode(r) for r in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["attrs"] = json.loads(data.pop("attrs_json") or "{}")
        return data
