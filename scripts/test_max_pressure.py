from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_env
from controllers.max_pressure import MaxPressureSplitController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_low.yaml")
    parser.add_argument("--cycles", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config.setdefault("run", {})
    config["run"]["seed"] = int(args.seed)

    config.setdefault("env", {})
    config["env"].setdefault("sumo", {})
    config["env"]["sumo"]["max_cycles"] = int(args.cycles)
    config["env"]["sumo"]["enable_kpi_tracker"] = True
    config["env"]["sumo"]["terminate_on_empty"] = False

    set_global_seed(int(args.seed))

    env = build_env(config)
    
    sumo_cfg = config.get("env", {}).get("sumo", {})
    lane_cfg = sumo_cfg.get("lane_groups", {})
    
    lanes_ns = [str(x) for x in lane_cfg.get("lanes_ns_ctrl", [])]
    lanes_ew = [str(x) for x in lane_cfg.get("lanes_ew_ctrl", [])]
    
    splits_raw = sumo_cfg.get("action_splits", [])
    splits_ns = [float(x[0]) for x in splits_raw]
    
    controller = MaxPressureSplitController(
        lanes_ns=lanes_ns,
        lanes_ew=lanes_ew,
        splits_ns=splits_ns,
    )

    if hasattr(env, "set_seed"):
        env.set_seed(int(args.seed))

    print("=" * 80)
    print(f"MAX PRESSURE CONTROLLER TEST: {args.cycles} cycles")
    print("=" * 80)

    state = env.reset()
    done = False
    cycle_count = 0
    total_reward = 0.0

    try:
        while not done:
            if hasattr(env, "get_last_state_raw"):
                state_raw = env.get_last_state_raw()
                if state_raw is not None:
                    action_id = controller.select_action(state_raw)
                else:
                    action_id = 2
            else:
                action_id = 2

            next_state, reward, done, info = env.step(action_id)

            cycle_count += 1
            total_reward += float(reward)

            if isinstance(info, dict):
                sim_time = info.get("sim_time", 0.0)
                state_raw_info = info.get("state_raw", [0, 0, 0, 0])
                q_ns, q_ew = state_raw_info[0], state_raw_info[1]
                waiting_total = info.get("waiting_total", 0.0)
                split_rho_ns = info.get("split_rho_ns", 0.5)

                print(
                    f"Cycle {cycle_count:3d} | "
                    f"SimTime={sim_time:7.1f}s | "
                    f"Action={action_id} Split={split_rho_ns:.2f} | "
                    f"Q_NS={q_ns:5.1f} Q_EW={q_ew:5.1f} | "
                    f"Wait={waiting_total:8.1f} | "
                    f"Reward={reward:8.1f}"
                )

            state = next_state

        if isinstance(info, dict) and "episode_kpi" in info:
            kpi = info["episode_kpi"]
            print("\n" + "=" * 80)
            print("FINAL KPIs:")
            print(f"  Arrived vehicles: {kpi.get('arrived_vehicles', 0)}")
            print(f"  Avg wait time:    {kpi.get('avg_wait_time', 0.0):.2f}s")
            print(f"  Avg travel time:  {kpi.get('avg_travel_time', 0.0):.2f}s")
            print(f"  Avg stops:        {kpi.get('avg_stops', 0.0):.2f}")
            print(f"  Avg queue:        {kpi.get('avg_queue', 0.0):.2f}")
            print(f"  Total reward:     {total_reward:.2f}")
            print("=" * 80)

    finally:
        env.close()


if __name__ == "__main__":
    main()