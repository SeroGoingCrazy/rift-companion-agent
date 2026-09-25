"""SQLite engine and sessions.

pysqlite's own transaction handling is turned off so SQLAlchemy's ``begin`` event decides
how a transaction starts: plain ``BEGIN`` for reads, ``BEGIN IMMEDIATE`` for
``write_session`` so the write lock is taken *before* the conflict check. Two concurrent
bookings for the same slot therefore serialize and the second one sees the first.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Connection, Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from booking_mcp.db.models import Base

#: Connection execution option read by the ``begin`` listener.
BEGIN_MODE_OPTION = "sqlite_begin_mode"

SessionFactory = sessionmaker[Session]


def database_url(target: str | Path) -> str:
    """``sqlite:///...`` URL for a file path; URLs and ``:memory:`` pass through."""
    text = str(target)
    if "://" in text:
        return text
    if text == ":memory:":
        return "sqlite://"
    return f"sqlite:///{Path(text).as_posix()}"


def create_db_engine(target: str | Path, *, busy_timeout_s: float = 30.0) -> Engine:
    url = database_url(target)
    in_memory = url in ("sqlite://", "sqlite:///:memory:")
    kwargs: dict[str, Any] = {
        "connect_args": {"check_same_thread": False, "timeout": busy_timeout_s},
    }
    if in_memory:
        # One shared connection, otherwise every checkout would see an empty database.
        kwargs["poolclass"] = StaticPool
    else:
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, **kwargs)

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn: Any, _record: Any) -> None:
        dbapi_conn.isolation_level = None  # let the "begin" listener emit BEGIN
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_s * 1000)}")
        if not in_memory:
            cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    @event.listens_for(engine, "begin")
    def _on_begin(conn: Connection) -> None:
        mode = conn.get_execution_options().get(BEGIN_MODE_OPTION, "DEFERRED")
        conn.exec_driver_sql(f"BEGIN {mode}")

    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def reset_db(engine: Engine) -> None:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> SessionFactory:
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def read_session(factory: SessionFactory) -> Iterator[Session]:
    with factory() as session:
        yield session


@contextmanager
def write_session(factory: SessionFactory) -> Iterator[Session]:
    """A session whose transaction starts with ``BEGIN IMMEDIATE``; commits on success."""
    with factory() as session:
        session.connection(execution_options={BEGIN_MODE_OPTION: "IMMEDIATE"})
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
