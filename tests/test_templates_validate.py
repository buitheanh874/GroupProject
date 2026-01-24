from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.config_normalization import normalize_action_table_schema
from scripts.validation import validate_action_table, validate_scalar_params


def _load_template(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _prepare_config(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg.pop("scenario_calibration", None)  # avoid file dependency in template check
    cfg = normalize_action_table_schema(cfg)
    return cfg


@pytest.mark.parametrize(
    "template_path",
    [
        Path("configs/train_bignet_9tls.yaml"),
        Path("configs/eval_bignet_9tls.yaml"),
    ],
)
def test_templates_validate_action_table_and_scalars(template_path: Path):
    cfg = _prepare_config(_load_template(template_path))
    sumo_cfg = cfg.get("env", {}).get("sumo", {})

    action_table = cfg.get("action_table", [])
    allowed_cycles = [60, 90, 120]
    action_splits = [
        (0.30, 0.70),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.70, 0.30),
    ]
    rho_min = 0.1
    g_min_sec = min(entry.get("g_min_sec", 0) for entry in action_table) if action_table else 5

    processed = validate_action_table(
        action_table_raw=action_table,
        action_splits=action_splits,
        state_dim=4,
        allowed_cycles=allowed_cycles,
        rho_min=rho_min,
        g_min_sec=int(g_min_sec),
    )
    assert len(processed) == len(action_table)

    validate_scalar_params(
        yellow_sec=sumo_cfg.get("yellow_sec", 0),
        all_red_sec=sumo_cfg.get("all_red_sec", 0),
        rho_min=rho_min,
        g_min_sec=int(g_min_sec),
        queue_count_mode=sumo_cfg.get("queue_count_mode", "distinct_cycle"),
        halt_speed_threshold=sumo_cfg.get("halt_speed_threshold", 0.1),
        use_enhanced_reward=sumo_cfg.get("use_enhanced_reward", False),
        reward_exponent=sumo_cfg.get("reward_exponent", 1.0),
        enable_spillback_penalty=sumo_cfg.get("enable_spillback_penalty", False),
        alpha_spillback=sumo_cfg.get("alpha_spillback", 1.0),
        allowed_cycles=allowed_cycles,
    )
