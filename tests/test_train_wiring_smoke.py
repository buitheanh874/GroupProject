from __future__ import annotations

from pathlib import Path

import yaml

import scripts.train as train_mod


def test_train_wiring_calibration_and_action_table(tmp_path, monkeypatch):
    project_root = tmp_path
    monkeypatch.setattr(train_mod, "project_root", project_root)

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

    build_env_calls = []

    class DummyEnv:
        def __init__(self) -> None:
            self.route_pool_set = None
            self._action_defs = []

        def set_route_file_pool(self, routes) -> None:
            self.route_pool_set = list(routes)

        def close(self) -> None:
            pass

    class DummyAgent:
        def to_train_mode(self) -> None:
            pass

        def save_model(self, *args, **kwargs) -> None:
            pass

    def fake_build_env(cfg):
        build_env_calls.append(cfg)
        return DummyEnv()

    def fake_build_agent(cfg, env):
        return DummyAgent(), None

    monkeypatch.setattr(train_mod, "build_env", fake_build_env)
    monkeypatch.setattr(train_mod, "build_agent", fake_build_agent)

    config = {
        "scenario_calibration": "configs/calib.yaml",
        "action_table": [{"ns_ratio": 0.4}],
        "train": {"episodes": 0},
        "logging": {
            "log_dir": str(project_root / "logs"),
            "model_dir": str(project_root / "models"),
            "results_dir": str(project_root / "results"),
        },
    }

    train_mod.run_training(config)

    assert len(build_env_calls) == 1
    cfg = build_env_calls[0]
    veh_weights = cfg["env"]["sumo"]["vehicle_weights"]
    assert veh_weights == calib_body["scenario"]["pcu_weights"]
    assert all("rho_ns" in entry for entry in cfg.get("action_table", []))
