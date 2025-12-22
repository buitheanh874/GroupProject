from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rl.utils import load_yaml_config, set_global_seed
from scripts.common import build_env


def test_scenario(
    config_path: str,
    max_cycles: int = 0,
    max_sim_seconds: int = 600,
    enable_kpi: bool = True,
    seed: int = 0,
):
    config = load_yaml_config(config_path)
    config.setdefault("run", {})
    config["run"]["seed"] = seed

    config.setdefault("env", {})
    config["env"].setdefault("sumo", {})
    config["env"]["sumo"]["max_cycles"] = max_cycles
    config["env"]["sumo"]["max_sim_seconds"] = max_sim_seconds if max_sim_seconds > 0 else None
    config["env"]["sumo"]["enable_kpi_tracker"] = enable_kpi
    config["env"]["sumo"]["terminate_on_empty"] = True

    set_global_seed(seed)

    env = build_env(config)
    fixed_action = 2

    print(f"\nTesting: max_cycles={max_cycles}, max_sim_seconds={max_sim_seconds}, kpi={enable_kpi}")
    print("-" * 80)

    try:
        if hasattr(env, "set_seed"):
            env.set_seed(seed)

        state = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done:
            _, reward, done, info = env.step(fixed_action)
            steps += 1
            total_reward += reward

            if steps % 10 == 0:
                sim_time = info.get("sim_time", 0) if isinstance(info, dict) else 0
                print(f"  Step {steps:3d} | SimTime={sim_time:6.1f}s | Reward={reward:8.1f}", end="\r")

        print()

        kpi = info.get("episode_kpi", {}) if isinstance(info, dict) else {}
        sim_time = info.get("sim_time", 0) if isinstance(info, dict) else 0

        print("\nResults:")
        print(f"  Steps:            {steps}")
        print(f"  Sim time:         {sim_time:.1f}s")
        print(f"  Total reward:     {total_reward:.1f}")
        print(f"  Arrived vehicles: {kpi.get('arrived_vehicles', 0)}")
        print(f"  Avg wait time:    {kpi.get('avg_wait_time', 0.0):.2f}s")
        print(f"  Avg queue:        {kpi.get('avg_queue', 0.0):.2f}")

        arrived = kpi.get("arrived_vehicles", 0)
        if arrived > 0:
            print(f"\nSUCCESS: {arrived} vehicles arrived")
            return True

        print(f"\nFAILED: No vehicles arrived in {sim_time:.0f}s")
        return False

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return False
    finally:
        env.close()


def main():
    config_path = "configs/train_sumo.yaml"

    print("=" * 80)
    print("EPISODE LENGTH TESTS")
    print("=" * 80)

    print("\n[TEST 1] Original: max_cycles=60, no time limit")
    test_scenario(config_path, max_cycles=60, max_sim_seconds=0, enable_kpi=True)

    print("\n[TEST 2] Time-based: 600s (10 min)")
    test_scenario(config_path, max_cycles=0, max_sim_seconds=600, enable_kpi=True)

    print("\n[TEST 3] Time-based: 1800s (30 min)")
    test_scenario(config_path, max_cycles=0, max_sim_seconds=1800, enable_kpi=True)

    print("\n[TEST 4] Time-based: 3600s (1 hour)")
    success = test_scenario(config_path, max_cycles=0, max_sim_seconds=3600, enable_kpi=True)

    print("\n" + "=" * 80)
    if success:
        print("RECOMMENDED CONFIG:")
        print("   max_cycles: 0")
        print("   max_sim_seconds: 3600")
        print("   terminate_on_empty: true")
        print("   enable_kpi_tracker: true")
    else:
        print("All tests failed. Check:")
        print("   1. Route file has vehicle flows")
        print("   2. Network connections are valid")
        print("   3. SUMO configuration is correct")
    print("=" * 80)


if __name__ == "__main__":
    main()
