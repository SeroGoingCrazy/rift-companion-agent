"""MCP Server entry point using official MCP SDK.

This module implements the MCP server using the official Python MCP SDK
over stdio or Streamable HTTP. Logs always go to stderr, so stdout only
carries protocol messages in stdio mode.

Usage::

    knowledge-mcp --transport stdio
    knowledge-mcp --transport http --port 8102   # served at http://127.0.0.1:8102/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional, Sequence

from src.mcp_server.protocol_handler import create_mcp_server
from src.observability.logger import get_logger

if TYPE_CHECKING:
    pass


SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


def _redirect_all_loggers_to_stderr() -> None:
    """Redirect all root logger handlers to stderr.

    MCP stdio transport reserves stdout for JSON-RPC messages.
    Any logging to stdout corrupts the protocol stream.
    """
    import logging as _logging

    root = _logging.getLogger()
    stderr_handler = _logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    # Replace any existing stream handlers that might point to stdout
    for handler in root.handlers[:]:
        if isinstance(handler, _logging.StreamHandler) and not isinstance(
            handler, _logging.FileHandler
        ):
            root.removeHandler(handler)
    root.addHandler(stderr_handler)


def _preload_heavy_imports() -> None:
    """Eagerly import heavy third-party modules in the **main thread**.

    MCP SDK uses anyio + background threads for stdin/stdout I/O.
    When a tool handler runs ``asyncio.to_thread(fn)``, *fn* executes in
    a new worker thread.  If it tries to ``import chromadb`` (which
    transitively pulls in onnxruntime, numpy, sqlite3 C extensions …),
    that import can deadlock with the stdin-reader thread because both
    compete for Python's global *import lock*.

    Pre-importing here – before anyio spins up its I/O threads – avoids
    the deadlock entirely: subsequent ``import`` statements in worker
    threads simply hit ``sys.modules`` and return immediately.
    """
    # chromadb is the heaviest culprit (onnxruntime, numpy, …)
    try:
        import chromadb  # noqa: F401
        import chromadb.config  # noqa: F401
    except ImportError:
        pass  # optional at install time

    # Internal modules that tools lazy-import inside asyncio.to_thread
    try:
        import src.core.query_engine.query_processor  # noqa: F401
        import src.core.query_engine.hybrid_search  # noqa: F401
        import src.core.query_engine.dense_retriever  # noqa: F401
        import src.core.query_engine.sparse_retriever  # noqa: F401
        import src.core.query_engine.reranker  # noqa: F401
        import src.ingestion.storage.bm25_indexer  # noqa: F401
        import src.libs.embedding.embedding_factory  # noqa: F401
        import src.libs.vector_store.vector_store_factory  # noqa: F401
    except ImportError:
        pass


async def run_stdio_server_async() -> int:
    """Run MCP server over stdio asynchronously.

    Returns:
        Exit code.
    """
    # Import here to avoid import errors if mcp not installed
    import mcp.server.stdio

    # Ensure ALL logging goes to stderr (stdout is reserved for JSON-RPC)
    _redirect_all_loggers_to_stderr()

    # Pre-load heavy deps in main thread to prevent import-lock deadlocks
    # when tool handlers later call asyncio.to_thread().
    _preload_heavy_imports()

    logger = get_logger(log_level="INFO")
    logger.info("Starting MCP server (stdio transport) with official SDK.")

    # Create server with protocol handler
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)

    # Run with stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

    logger.info("MCP server shutting down.")
    return 0


def run_stdio_server() -> int:
    """Run MCP server over stdio (synchronous wrapper).

    Returns:
        Exit code.
    """
    return asyncio.run(run_stdio_server_async())


def create_http_app(server: Any) -> Any:
    """Stateless Streamable HTTP (JSON response) Starlette app serving MCP at ``/mcp``."""
    from mcp.server.fastmcp.server import StreamableHTTPASGIApp
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Route

    manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=True)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return Starlette(
        routes=[Route("/mcp", endpoint=StreamableHTTPASGIApp(manager))], lifespan=lifespan
    )


def run_http_server(host: str, port: int) -> int:
    """Run MCP server over Streamable HTTP at ``http://{host}:{port}/mcp``."""
    import uvicorn

    _redirect_all_loggers_to_stderr()
    _preload_heavy_imports()

    logger = get_logger(log_level="INFO")
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)
    logger.info("knowledge-mcp listening on http://%s:%s/mcp", host, port)
    uvicorn.run(create_http_app(server), host=host, port=port, log_level="warning")
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="knowledge-mcp", description="Knowledge MCP server")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point: stdio (default) or Streamable HTTP MCP server."""
    args = parse_args(argv)
    if args.transport == "http":
        return run_http_server(args.host, args.port)
    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())