from __future__ import annotations

from pathlib import Path

from scripts.inspect_net_boundaries import inspect_boundaries


def test_inspect_boundaries_tiny_net():
    net_path = Path(__file__).resolve().parents[1] / "tests" / "assets" / "tiny.net.xml"
    entries, exits = inspect_boundaries(net_path)

    assert entries == ["e_in"]
    assert set(exits) == {"e_dead", "e_mid"}
