import importlib

import pytest

PACKAGES = [
    "rift_common",
    "rift_domain",
    "booking_mcp",
    "rift_agent",
    "rift_web",
    "rift_training",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_workspace_package_importable(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__version__ == "0.1.0"


def test_directory_marker_applied(request: pytest.FixtureRequest) -> None:
    assert request.node.get_closest_marker("unit") is not None
