from __future__ import annotations

from pathlib import Path

import yaml

from scripts.generate_hanoi_route_variants import generate_routes, write_manifest
from scripts.route_pool_loader import load_route_pool_from_config
from scripts.scenario_config_bridge import apply_calibration_overrides


def _calib_body() -> dict:
    return {
        "map_prefix": "hanoi",
        "net_file": "net.xml",
        "entry_edges": ["A_IN", "B_IN"],
        "exit_edges": ["X_OUT", "Y_OUT"],
        "turn_mapping": {
            "A_IN": {"L": ["X_OUT"], "S": ["Y_OUT"], "R": ["X_OUT"]},
            "B_IN": {"L": ["Y_OUT"], "S": ["X_OUT"], "R": ["Y_OUT"]},
        },
        "pcu_weights": {"motorcycle": 0.25, "passenger": 1.0},
        "demand": {"total_pcu_per_hour": {"low": 1000}, "entry_dirichlet_alpha": 2.0},
        "vehicle_mix": {"mean": {"motorcycle": 0.8, "passenger": 0.2}, "kappa": 20},
        "turning": {"mean_LSR": [0.2, 0.6, 0.2], "kappa": 30},
        "stages": {"enabled": False, "intervals": []},
        "min_total_vehicles": 1,
        "horizon_sec": 3600,
    }


def test_end_to_end_wiring_smoke(tmp_path: Path):
    project_root = tmp_path

    calib_body = _calib_body()
    calib_dir = project_root / "configs"
    calib_dir.mkdir()
    calib_path = calib_dir / "calib.yaml"
    calib_path.write_text(yaml.safe_dump({"scenario": calib_body}), encoding="utf-8")

    out_dir = project_root / "variants"
    routes = generate_routes(
        calib=calib_body,
        split="eval",
        profile="auto",
        seeds=[7],
        out_dir=out_dir,
        skip_router=True,
    )
    assert len(routes) == 1
    assert routes[0].exists()

    manifest_path = out_dir / "eval" / "manifest.txt"
    write_manifest(manifest_path, routes)
    assert manifest_path.exists()

    config = {
        "scenario_calibration": str(calib_path.relative_to(project_root)),
        "eval": {"route_pool_manifest": str(manifest_path.relative_to(project_root))},
    }

    updated = apply_calibration_overrides(config, project_root=project_root)
    veh_weights = updated["env"]["sumo"]["vehicle_weights"]
    assert veh_weights == calib_body["pcu_weights"]

    pool = load_route_pool_from_config(updated, split="eval", project_root=project_root)
    assert len(pool) == len(routes)
    assert all(Path(p).exists() for p in pool)
