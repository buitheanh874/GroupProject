from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch

from env.base_env import BaseEnv
from env.normalization import StateNormalizer
from env.sumo_env import SUMOEnv, SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram
from scripts.validation import validate_action_table
from env.toy_queue_env import ToyQueueEnv, ToyQueueEnvConfig
from rl.agent import AgentConfig, DQNAgent
from rl.utils import resolve_device


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config_with_inheritance(config_path: str) -> Dict[str, Any]:
    from rl.utils import load_yaml_config
    
    config = load_yaml_config(config_path)
    
    if "_base" in config:
        base_path = Path(config_path).parent / config["_base"]
        base_config = load_yaml_config(str(base_path))
        
        merged = deep_merge(base_config, config)
        merged.pop("_base", None)
        return merged
    
    return config


def _default_action_splits() -> List[Tuple[float, float]]:
    return [
        (0.30, 0.70),
        (0.40, 0.60),
        (0.50, 0.50),
        (0.60, 0.40),
        (0.70, 0.30),
    ]


def build_env(config: Dict[str, Any]) -> BaseEnv:
    env_config = config.get("env", {})
    env_type = str(env_config.get("type", "")).strip().lower()

    if env_type == "toy":
        toy_cfg = env_config.get("toy", {})
        toy_env_config = ToyQueueEnvConfig(
            max_steps=int(toy_cfg.get("max_steps", 200)),
            arrival_prob=float(toy_cfg.get("arrival_prob", 0.7)),
            serve_slow_rate=int(toy_cfg.get("serve_slow_rate", 1)),
            serve_fast_rate=int(toy_cfg.get("serve_fast_rate", 3)),
            seed=int(config.get("run", {}).get("seed", 0)),
        )
        return ToyQueueEnv(toy_env_config)

    if env_type == "sumo":
        sumo_cfg = env_config.get("sumo", {})
        net_path = Path(sumo_cfg.get("net_file", ""))
        route_path = Path(sumo_cfg.get("route_file", ""))
        
        if not net_path.exists():
            raise FileNotFoundError(
                f"Network file not found: {net_path}\n"
                f"Current working directory: {Path.cwd()}\n"
                f"Please check the path in config: env.sumo.net_file"
            )
        
        if not route_path.exists():
            raise FileNotFoundError(
                f"Route file not found: {route_path}\n"
                f"Current working directory: {Path.cwd()}\n"
                f"Please check the path in config: env.sumo.route_file"
            )
        lane_cfg = sumo_cfg.get("lane_groups", {})
        lane_cfg_by_tls = sumo_cfg.get("lane_groups_by_tls", {})
        phase_cfg = sumo_cfg.get("phase_program", {})

        lanes_by_tls: Dict[str, SumoLaneGroups] = {}
        if isinstance(lane_cfg_by_tls, dict) and len(lane_cfg_by_tls) > 0:
            for tls_key, cfg in lane_cfg_by_tls.items():
                lanes_by_tls[str(tls_key)] = SumoLaneGroups(
                    lanes_ns_ctrl=[str(x) for x in cfg.get("lanes_ns_ctrl", [])],
                    lanes_ew_ctrl=[str(x) for x in cfg.get("lanes_ew_ctrl", [])],
                    lanes_right_turn_slip_ns=[str(x) for x in cfg.get("lanes_right_turn_slip_ns", [])],
                    lanes_right_turn_slip_ew=[str(x) for x in cfg.get("lanes_right_turn_slip_ew", [])],
                    approach_lanes={str(k): [str(vv) for vv in v] for k, v in cfg.get("approach_lanes", {}).items()} if isinstance(cfg.get("approach_lanes", {}), dict) else {},
                )
        else:
            lanes_by_tls[str(sumo_cfg.get("tls_id", "tls0"))] = SumoLaneGroups(
                lanes_ns_ctrl=[str(x) for x in lane_cfg.get("lanes_ns_ctrl", [])],
                lanes_ew_ctrl=[str(x) for x in lane_cfg.get("lanes_ew_ctrl", [])],
                lanes_right_turn_slip_ns=[str(x) for x in lane_cfg.get("lanes_right_turn_slip_ns", [])],
                lanes_right_turn_slip_ew=[str(x) for x in lane_cfg.get("lanes_right_turn_slip_ew", [])],
                approach_lanes={str(k): [str(vv) for vv in v] for k, v in lane_cfg.get("approach_lanes", {}).items()} if isinstance(lane_cfg.get("approach_lanes", {}), dict) else {},
            )

        phases = SumoPhaseProgram(
            ns_green=int(phase_cfg.get("ns_green", 0)),
            ew_green=int(phase_cfg.get("ew_green", 1)),
            ns_yellow=phase_cfg.get("ns_yellow"),
            ew_yellow=phase_cfg.get("ew_yellow"),
            all_red=phase_cfg.get("all_red"),
        )

        tls_ids = [str(x) for x in sumo_cfg.get("tls_ids", [])]
        center_tls_id = sumo_cfg.get("center_tls_id")
        downstream_links = {str(k): v for k, v in sumo_cfg.get("downstream_links", {}).items()}
        vehicle_weights_raw = sumo_cfg.get("vehicle_weights", {})
        vehicle_weights = {str(k): float(v) for k, v in vehicle_weights_raw.items()}
        yellow_sec = int(sumo_cfg.get("yellow_sec", 0))
        all_red_sec = int(sumo_cfg.get("all_red_sec", 0))
        rho_min = float(sumo_cfg.get("rho_min", 0.1))
        g_min_sec = int(sumo_cfg.get("g_min_sec", 5))
        lambda_fairness = float(sumo_cfg.get("lambda_fairness", 0.12))
        fairness_metric = str(sumo_cfg.get("fairness_metric", "max")).lower()
        queue_count_mode = str(sumo_cfg.get("queue_count_mode", "distinct_cycle")).lower()
        halt_speed_threshold = float(sumo_cfg.get("halt_speed_threshold", 0.1))
        use_pcu_weighted_wait = sumo_cfg.get("use_pcu_weighted_wait")
        use_enhanced_reward = bool(sumo_cfg.get("use_enhanced_reward", False))
        reward_exponent = float(sumo_cfg.get("reward_exponent", 1.0))
        enable_anti_flicker = bool(sumo_cfg.get("enable_anti_flicker", False))
        kappa = float(sumo_cfg.get("kappa", 0.0))
        enable_spillback_penalty = bool(sumo_cfg.get("enable_spillback_penalty", False))
        beta = float(sumo_cfg.get("beta", 0.0))
        occ_threshold = float(sumo_cfg.get("occ_threshold", 0.0))
        allowed_cycles = [int(x) for x in sumo_cfg.get("allowed_cycles_sec", [30, 60, 90])]

        # Validate scalar params upfront.
        from scripts.validation import validate_scalar_params

        validate_scalar_params(
            yellow_sec=yellow_sec,
            all_red_sec=all_red_sec,
            rho_min=rho_min,
            g_min_sec=g_min_sec,
            lambda_fairness=lambda_fairness,
            fairness_metric=fairness_metric,
            queue_count_mode=queue_count_mode,
            halt_speed_threshold=halt_speed_threshold,
            use_enhanced_reward=use_enhanced_reward,
            reward_exponent=reward_exponent,
            enable_anti_flicker=enable_anti_flicker,
            kappa=kappa,
            enable_spillback_penalty=enable_spillback_penalty,
            beta=beta,
            occ_threshold=occ_threshold,
            allowed_cycles=allowed_cycles,
        )

        action_splits_raw = sumo_cfg.get("action_splits", [])
        action_splits = [(float(x[0]), float(x[1])) for x in action_splits_raw] if len(action_splits_raw) > 0 else _default_action_splits()
        action_table_raw = sumo_cfg.get("action_table", [])

        if yellow_sec < 0:
            raise ValueError("yellow_sec must be >=0")
        if all_red_sec < 0:
            raise ValueError("all_red_sec must be >=0")
        if rho_min <= 0.0 or rho_min > 0.5:
            raise ValueError("rho_min must be in (0, 0.5]")
        if g_min_sec < 0:
            raise ValueError("g_min_sec must be >=0")
        if lambda_fairness < 0.0:
            raise ValueError("lambda_fairness must be >=0")
        if fairness_metric not in {"max", "p95"}:
            raise ValueError("fairness_metric must be max or p95")
        if queue_count_mode not in {"distinct_cycle", "snapshot_last_step"}:
            raise ValueError("queue_count_mode must be distinct_cycle or snapshot_last_step")
        if halt_speed_threshold < 0.0:
            raise ValueError("halt_speed_threshold must be >=0")
        if use_enhanced_reward and reward_exponent < 1.0:
            raise ValueError("reward_exponent must be >=1 when use_enhanced_reward is True")
        if enable_anti_flicker and kappa < 0.0:
            raise ValueError("kappa must be >=0 when enable_anti_flicker is True")
        if enable_spillback_penalty:
            if beta < 0.0:
                raise ValueError("beta must be >=0 when enable_spillback_penalty is True")
            if occ_threshold < 0.0 or occ_threshold > 1.0:
                raise ValueError("occ_threshold must be in [0,1] when enable_spillback_penalty is True")
        if len(allowed_cycles) == 0 or any(cycle <= 0 for cycle in allowed_cycles):
            raise ValueError("allowed_cycles_sec must contain positive cycle lengths")

        state_dim = int(sumo_cfg.get("state_dim", 12 if len(tls_ids) > 0 or len(action_table_raw) > 0 else 4))
        normalize_state = bool(sumo_cfg.get("normalize_state", True))
        occupancy_enabled = bool(sumo_cfg.get("enable_downstream_occupancy", True))

        if len(tls_ids) > 1 and state_dim != 12:
            raise ValueError("When specifying multiple tls_ids, state_dim must be 12")

        if state_dim not in (4, 12):
            raise ValueError(f"state_dim must be 4 or 12, got {state_dim}")

        tls_ids_effective = tls_ids if len(tls_ids) > 0 else [str(sumo_cfg.get("tls_id", "tls0"))]
        center_tls_effective = str(center_tls_id) if center_tls_id is not None else str(tls_ids_effective[0])

        if state_dim == 12:
            if center_tls_effective not in tls_ids_effective:
                raise ValueError("center_tls_id must be in tls_ids (or tls_id) when state_dim=12")
            if occupancy_enabled:
                required_dirs = {"N", "E", "S", "W"}
                missing_dirs = [d for d in required_dirs if d not in {k.upper() for k in downstream_links.keys()}]
                if len(missing_dirs) > 0:
                    raise ValueError(f"downstream_links must include N/E/S/W when state_dim=12 (missing: {missing_dirs})")
            if len(tls_ids_effective) > 1:
                if len(lane_cfg_by_tls) == 0:
                    raise ValueError("lane_groups_by_tls must be provided for each tls_id when tls_ids has multiple entries")
                missing_lane_defs = [tid for tid in tls_ids_effective if str(tid) not in lane_cfg_by_tls]
                if len(missing_lane_defs) > 0:
                    raise ValueError(f"lane_groups_by_tls missing definitions for tls_ids: {missing_lane_defs}")

        if len(vehicle_weights) > 0:
            bad_weights = {k: v for k, v in vehicle_weights.items() if v <= 0}
            if len(bad_weights) > 0:
                raise ValueError(f"vehicle_weights must be >0 for all entries, got invalid: {bad_weights}")

        processed_action_table = validate_action_table(
            action_table_raw=action_table_raw,
            action_splits=action_splits,
            state_dim=state_dim,
            allowed_cycles=allowed_cycles,
            rho_min=rho_min,
            g_min_sec=g_min_sec,
        )

        sumo_env_config = SumoEnvConfig(
            sumo_binary=str(sumo_cfg.get("sumo_binary", "sumo")),
            net_file=str(sumo_cfg.get("net_file", "")),
            route_file=str(sumo_cfg.get("route_file", "")),
            additional_files=[str(x) for x in sumo_cfg.get("additional_files", [])],
            tls_id=str(sumo_cfg.get("tls_id", "tls0")),
            tls_ids=tls_ids_effective,
            center_tls_id=center_tls_effective,
            downstream_links=downstream_links,
            vehicle_weights=vehicle_weights,
            step_length_sec=float(sumo_cfg.get("step_length_sec", 1.0)),
            green_cycle_sec=int(sumo_cfg.get("green_cycle_sec", sumo_cfg.get("cycle_length_sec", 60))),
            yellow_sec=yellow_sec,
            all_red_sec=all_red_sec,
            max_cycles=int(sumo_cfg.get("max_cycles", 60)),
            max_sim_seconds=int(sumo_cfg["max_sim_seconds"]) if sumo_cfg.get("max_sim_seconds") is not None else None,
            seed=int(config.get("run", {}).get("seed", 0)),
            rho_min=rho_min,
            g_min_sec=g_min_sec,
            lambda_fairness=lambda_fairness,
            fairness_metric=fairness_metric,
            action_splits=action_splits,
            action_table=processed_action_table,
            queue_count_mode=queue_count_mode,
            include_transition_in_waiting=bool(sumo_cfg.get("include_transition_in_waiting", True)),
            use_pcu_weighted_wait=use_pcu_weighted_wait,
            use_enhanced_reward=use_enhanced_reward,
            reward_exponent=reward_exponent,
            enable_anti_flicker=enable_anti_flicker,
            kappa=kappa,
            enable_spillback_penalty=enable_spillback_penalty,
            beta=beta,
            occ_threshold=occ_threshold,
            halt_speed_threshold=halt_speed_threshold,
            terminate_on_empty=bool(sumo_cfg.get("terminate_on_empty", True)),
            sumo_extra_args=[str(x) for x in sumo_cfg.get("sumo_extra_args", [])],
            normalize_state=normalize_state,
            return_raw_state=bool(sumo_cfg.get("return_raw_state", False)),
            enable_kpi_tracker=bool(sumo_cfg.get("enable_kpi_tracker", False)),
            state_dim=state_dim,
            enable_downstream_occupancy=occupancy_enabled,
        )

        normalization_cfg = config.get("normalization", {})
        mean: Any = normalization_cfg.get("mean")
        std: Any = normalization_cfg.get("std")
        norm_file = normalization_cfg.get("file")

        if norm_file:
            import json

            with open(str(norm_file), "r", encoding="utf-8") as f:
                data = json.load(f)
            mean = data.get("mean", mean)
            std = data.get("std", std)

        if not normalize_state:
            mean = [0.0 for _ in range(state_dim)]
            std = [1.0 for _ in range(state_dim)]
        else:
            if mean is None or std is None:
                raise ValueError("Normalization stats are required when normalize_state is True. Provide mean/std or a file path.")
            mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1)
            std_arr = np.asarray(std, dtype=np.float32).reshape(-1)
            if mean_arr.size != state_dim or std_arr.size != state_dim:
                raise ValueError(f"Normalization mean/std must match state_dim={state_dim}, got mean={mean_arr.size}, std={std_arr.size}")
            mean = mean_arr.tolist()
            std = std_arr.tolist()

        normalizer = StateNormalizer(mean=mean, std=std, expected_dim=state_dim)

        return SUMOEnv(config=sumo_env_config, lanes=lanes_by_tls, phases=phases, normalizer=normalizer)

    raise ValueError(f"Unsupported env type: {env_type}")


def build_agent(config: Dict[str, Any], env: BaseEnv) -> Tuple[DQNAgent, torch.device]:
    run_cfg = config.get("run", {})
    device = resolve_device(str(run_cfg.get("device", "cpu")))

    agent_cfg = config.get("agent", {})
    hidden_dims = agent_cfg.get("hidden_dims", [128, 128])
    use_time_aware_gamma = bool(agent_cfg.get("use_time_aware_gamma", False))
    gamma_0 = float(agent_cfg.get("gamma_0", agent_cfg.get("gamma", 0.98)))
    t_ref = float(agent_cfg.get("T_ref", agent_cfg.get("t_ref", 60.0)))
    if use_time_aware_gamma and t_ref <= 0.0:
        raise ValueError("T_ref must be >0 when use_time_aware_gamma is True")

    agent_config = AgentConfig(
        state_dim=int(env.state_dim),
        action_dim=int(env.action_dim),
        hidden_dims=[int(x) for x in hidden_dims],
        gamma=float(agent_cfg.get("gamma", 0.98)),
        use_time_aware_gamma=use_time_aware_gamma,
        gamma_0=gamma_0,
        t_ref=t_ref,
        learning_rate=float(agent_cfg.get("learning_rate", 1e-3)),
        batch_size=int(agent_cfg.get("batch_size", 64)),
        replay_buffer_size=int(agent_cfg.get("replay_buffer_size", 100000)),
        target_update_freq=int(agent_cfg.get("target_update_freq", 1000)),
        seed=int(run_cfg.get("seed", 0)),
    )

    agent = DQNAgent(config=agent_config, device=device)
    return agent, device


def format_state(state: np.ndarray) -> str:
    values = [float(x) for x in np.asarray(state).reshape(-1).tolist()]
    return "[" + ", ".join([f"{v:.3f}" for v in values]) + "]"
