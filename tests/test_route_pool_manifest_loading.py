from __future__ import annotations

import pytest
from pathlib import Path

from scripts.route_pool_loader import load_route_pool_from_config


def test_manifest_loader_strips_comments_and_blanks(tmp_path: Path):
    routes_dir = tmp_path / "routes"
    routes_dir.mkdir()
    files = []
    for name in ["a.rou.xml", "b.rou.xml", "c.rou.xml"]:
        path = routes_dir / name
        path.write_text("", encoding="utf-8")
        files.append(path)

    manifest = tmp_path / "manifest.txt"
    rel_paths = [p.relative_to(manifest.parent) for p in files]
    manifest.write_text(
        "\n".join([
            "# comment",
            "",
            str(rel_paths[0]),
            "   " + str(rel_paths[1]) + "   ",
            "",
            str(rel_paths[2]),
        ]),
        encoding="utf-8",
    )

    config = {"train": {"route_pool_manifest": str(manifest)}}
    routes = load_route_pool_from_config(config, split="train", project_root=tmp_path)

    assert routes == [str(p.resolve()) for p in files]
    assert config["env"]["sumo"]["route_pool"] == routes


def test_missing_manifest_raises(tmp_path: Path):
    manifest = tmp_path / "missing.txt"
    config = {"train": {"route_pool_manifest": str(manifest)}}
    with pytest.raises(FileNotFoundError):
        load_route_pool_from_config(config, split="train", project_root=tmp_path)
