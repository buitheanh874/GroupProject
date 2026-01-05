from __future__ import annotations

import random
from pathlib import Path

from scripts.route_pool_loader import load_route_pool_from_config


def _select_route_from_pool(route_pool, episode_seed: int, episode_index: int) -> str:
    rng = random.Random(int(episode_seed) + int(episode_index))
    return rng.choice(route_pool)


def test_route_pool_manifest_selection_varied(tmp_path: Path):
    project_root = tmp_path
    train_dir = project_root / "train"
    train_dir.mkdir()

    route_files = []
    for idx in range(3):
        path = train_dir / f"route_{idx}.rou.xml"
        path.write_text("<routes></routes>", encoding="utf-8")
        route_files.append(path)

    manifest = train_dir / "manifest.txt"
    manifest.write_text("\n".join(p.name for p in route_files), encoding="utf-8")

    config = {"train": {"route_pool_manifest": str(manifest.relative_to(project_root))}}
    routes = load_route_pool_from_config(config, split="train", project_root=project_root)
    assert len(routes) == len(route_files)
    assert all(Path(p).exists() for p in routes)

    picks_a = [_select_route_from_pool(routes, episode_seed=123, episode_index=i) for i in range(1, 8)]
    picks_b = [_select_route_from_pool(routes, episode_seed=123, episode_index=i) for i in range(1, 8)]

    assert picks_a == picks_b  # deterministic for same seed
    assert len(set(picks_a)) > 1  # but not constant route
