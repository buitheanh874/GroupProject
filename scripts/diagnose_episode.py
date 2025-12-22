from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_sumo.yaml")
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
    fixed_action_id = int(config.get("baseline", {}).get("fixed_action_id", 2))

    if hasattr(env, "set_seed"):
        env.set_seed(int(args.seed))

    print("=" * 80)
    print(f"DIAGNOSTIC RUN: {args.cycles} cycles")
    print("=" * 80)

    state = env.reset()
    done = False
    cycle_count = 0
    total_reward = 0.0

    try:
        while not done:
            action_id = int(fixed_action_id)
            next_state, reward, done, info = env.step(action_id)

            cycle_count += 1
            total_reward += float(reward)

            if isinstance(info, dict):
                sim_time = info.get("sim_time", 0.0)
                state_raw = info.get("state_raw", [0, 0, 0, 0])
                q_ns, q_ew = state_raw[0], state_raw[1]
                waiting_total = info.get("waiting_total", 0.0)

                print(
                    f"Cycle {cycle_count:3d} | "
                    f"SimTime={sim_time:7.1f}s | "
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
            print("=" * 80)

            arrived = kpi.get("arrived_vehicles", 0)
            sim_time_final = info.get("sim_time", 0.0)

            if arrived == 0:
                print("\nPROBLEM DETECTED: No vehicles arrived!")
                print("\nPossible causes:")
                print(f"1. Episode too short: {sim_time_final:.0f}s may not be enough")
                print("   - Vehicles need ~100-200s to traverse network")
                print("   - Try increasing max_cycles to 100+")
                print("2. Route file issue: Check if vehicles are spawning")
                print("3. Network issue: Check if paths are connected")
                print("\nRecommended fixes:")
                print(f"   max_cycles: {max(100, args.cycles * 2)} (currently {args.cycles})")
                print("   OR use max_sim_seconds: 3600 instead of max_cycles")
            else:
                print(f"\nSUCCESS: {arrived} vehicles completed their routes")
                print(f"Episode length was adequate ({sim_time_final:.0f}s)")

    finally:
        env.close()


if __name__ == "__main__":
    main()
