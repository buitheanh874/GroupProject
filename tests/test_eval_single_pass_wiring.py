from __future__ import annotations

from pathlib import Path

import yaml

import scripts.eval as eval_mod


def test_eval_single_pass_wiring(tmp_path, monkeypatch):
    project_root = tmp_path
    monkeypatch.setattr(eval_mod, "repo_root", project_root)

    calib_body = {
        "scenario": {
            "entry_edges": ["A_IN"],
            "exit_edges": ["X_OUT"],
            "pcu_weights": {"motorcycle": 0.25, "passenger": 1.0},
            "turning": {"mean_LSR": [0.2, 0.6, 0.2], "kappa": 5},
            "turn_mapping": {"A_IN": {"L": ["X_OUT"], "S": ["X_OUT"], "R": ["X_OUT"]}},
        }
    }
    calib_dir = project_root / "configs"
    calib_dir.mkdir()
    calib_path = calib_dir / "calib.yaml"
    calib_path.write_text(yaml.safe_dump(calib_body), encoding="utf-8")

    route_dir = project_root / "eval"
    route_dir.mkdir()
    route_file = route_dir / "route.rou.xml"
    route_file.write_text("<routes><vehicle id='v0'/></routes>", encoding="utf-8")
    manifest = route_dir / "manifest.txt"
    manifest.write_text(route_file.name, encoding="utf-8")

    call_counts = {"load_yaml": 0, "route_pool": 0, "build_env": 0}

    original_load_yaml = eval_mod.load_yaml_config
    original_route_pool = eval_mod.load_route_pool_from_config

    def fake_load_yaml(path):
        call_counts["load_yaml"] += 1
        return original_load_yaml(path)

    def fake_route_pool(cfg, split, project_root):
        call_counts["route_pool"] += 1
        return original_route_pool(cfg, split=split, project_root=project_root)

    class DummyEnv:
        def __init__(self) -> None:
            self.route_pool = None

        def set_route_file_pool(self, pool):
            self.route_pool = list(pool)

        def set_seed(self, seed: int) -> None:
            pass

        def reset(self):
            return 0

        def step(self, action):
            return 0, 1.0, True, {"episode_kpi": {"arrived_vehicles": 1}}

        def close(self):
            pass

    def fake_build_env(cfg):
        call_counts["build_env"] += 1
        assert cfg["env"]["sumo"]["vehicle_weights"] == calib_body["scenario"]["pcu_weights"]
        assert all("rho_ns" in entry for entry in cfg.get("action_table", []))
        return DummyEnv()

    monkeypatch.setattr(eval_mod, "load_yaml_config", fake_load_yaml)
    monkeypatch.setattr(eval_mod, "load_route_pool_from_config", fake_route_pool)
    monkeypatch.setattr(eval_mod, "build_env", fake_build_env)

    config_body = {
        "scenario_calibration": "configs/calib.yaml",
        "action_table": [{"ns_ratio": 0.5}],
        "eval": {
            "route_pool_manifest": str(manifest.relative_to(project_root)),
            "model_path": "ignored.pt",
        },
        "logging": {"results_dir": str(project_root / "results")},
    }
    config_path = project_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_body), encoding="utf-8")

    eval_mod.main(
        argv=[
            "--config",
            str(config_path),
            "--controller",
            "fixed",
            "--runs",
            "1",
        ]
    )

    assert call_counts["load_yaml"] == 1
    assert call_counts["route_pool"] == 1
    assert call_counts["build_env"] == 1
