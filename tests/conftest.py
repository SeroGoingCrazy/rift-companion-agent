"""Shared pytest configuration.

Tests are auto-marked by their top-level directory (``tests/unit`` -> ``unit`` etc.),
so ``pytest -m "unit or integration"`` selects the offline CI suite.
"""

from pathlib import Path

import pytest

from rift_domain.config import DomainConfig, load_domain_config

_TESTS_ROOT = Path(__file__).parent
_DIR_MARKERS = {"unit", "integration", "e2e"}

REPO_ROOT = _TESTS_ROOT.parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        try:
            layer = Path(item.path).relative_to(_TESTS_ROOT).parts[0]
        except ValueError:
            continue
        if layer in _DIR_MARKERS:
            item.add_marker(layer)


@pytest.fixture(scope="session")
def domain() -> DomainConfig:
    """The repo's real ``config/domain.yaml``."""
    return load_domain_config(REPO_ROOT / "config" / "domain.yaml")
