"""Name -> class registry shared by the LLM and embedding provider factories."""

from __future__ import annotations


class UnknownProviderError(ValueError):
    """Raised when a config names a provider that was never registered."""


class ProviderRegistry[T]:
    """Registry of provider classes that must subclass ``base``.

    Registration validates the class eagerly so a typo fails at import time, not at the
    first request.
    """

    def __init__(self, kind: str, base: type[object]) -> None:
        # ``type[object]`` rather than ``type[T]`` so abstract bases are accepted by mypy.
        self._kind = kind
        self._base = base
        self._providers: dict[str, type[T]] = {}

    def register(self, name: str, cls: type[T]) -> None:
        if not name:
            raise ValueError(f"{self._kind} provider name must be non-empty")
        if not (isinstance(cls, type) and issubclass(cls, self._base)):
            raise TypeError(
                f"{self._kind} provider {name!r} must be a subclass of {self._base.__name__}, "
                f"got {cls!r}"
            )
        existing = self._providers.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"{self._kind} provider {name!r} is already registered to {existing.__name__}"
            )
        self._providers[name] = cls

    def get(self, name: str) -> type[T]:
        try:
            return self._providers[name]
        except KeyError:
            available = ", ".join(sorted(self._providers)) or "<none>"
            raise UnknownProviderError(
                f"unknown {self._kind} provider {name!r}; available: {available}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._providers)
