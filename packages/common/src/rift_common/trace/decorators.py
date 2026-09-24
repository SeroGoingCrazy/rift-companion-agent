"""``@traced`` decorator for sync and async functions."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast

from rift_common.trace.context import span
from rift_common.trace.span import SpanKind


def traced[F: Callable[..., Any]](
    name: str | None = None, *, kind: SpanKind = "span", **static_attrs: Any
) -> Callable[[F], F]:
    """Wrap a function call in a span named ``name`` (defaults to the function's qualname)."""

    def decorate(fn: F) -> F:
        span_name = name or fn.__qualname__

        if inspect.iscoroutinefunction(fn):
            async_fn = cast(Callable[..., Awaitable[Any]], fn)

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(span_name, kind=kind, **static_attrs):
                    return await async_fn(*args, **kwargs)

            return cast(F, async_wrapper)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name, kind=kind, **static_attrs):
                return fn(*args, **kwargs)

        return cast(F, wrapper)

    return decorate
