from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from env.sumo_env import SUMOEnv, SumoEnvConfig, SumoLaneGroups, SumoPhaseProgram
from env.normalization import StateNormalizer


def main() -> None:
    print("=" * 80)
    print("VERIFYING ACTION TABLE STRUCTURE")
    print("=" * 80)

    config = SumoEnvConfig(
        sumo_binary="sumo",
        net_file="networks/BIGMAP.net.xml",
        route_file="networks/BIGMAP.rou.xml",
        tls_id="J0",
        tls_ids=["J0", "J1"],
        state_dim=12,
        action_table=[],
    )

    lanes = SumoLaneGroups(
        lanes_ns_ctrl=["dummy_ns"],
        lanes_ew_ctrl=["dummy_ew"],
    )

    phases = SumoPhaseProgram(ns_green=0, ew_green=4)

    normalizer = StateNormalizer(
        mean=[0.0] * 12,
        std=[1.0] * 12,
        expected_dim=12
    )

    print("Initializing SUMOEnv to generate action definitions...")
    
    try:
        env = SUMOEnv(config=config, lanes={config.tls_id: lanes}, phases=phases, normalizer=normalizer)
        
        print("\nAction Table Verification:")
        print(f"{'ID':<4} | {'Cycle':<6} | {'rho_NS':<6} | {'rho_EW':<6} | {'Group':<10}")
        print("-" * 50)
        
        actions = env._action_defs
        
        for i, action in enumerate(actions):
            group = "Low" if action.cycle_sec == 30 else "Med" if action.cycle_sec == 60 else "High"
            print(f"{i:<4d} | {action.cycle_sec:<5d}s | {action.rho_ns:<6.2f} | {action.rho_ew:<6.2f} | {group:<10}")

        print("-" * 50)
        print(f"Total actions generated: {len(actions)}")

        expected_count = 15
        if len(actions) != expected_count:
            print(f"\n[FAIL] Expected {expected_count} actions, got {len(actions)}")
            sys.exit(1)
        
        c30 = [a for a in actions if a.cycle_sec == 30]
        if len(c30) != 3:
            print(f"[FAIL] Expected 3 actions for cycle 30s, got {len(c30)}")
            sys.exit(1)
            
        c60 = [a for a in actions if a.cycle_sec == 60]
        if len(c60) != 7:
            print(f"[FAIL] Expected 7 actions for cycle 60s, got {len(c60)}")
            sys.exit(1)
            
        c90 = [a for a in actions if a.cycle_sec == 90]
        if len(c90) != 5:
            print(f"[FAIL] Expected 5 actions for cycle 90s, got {len(c90)}")
            sys.exit(1)

        action_7 = actions[7]
        print(f"\nVerifying Action 7 (New Baseline):")
        print(f"  Cycle: {action_7.cycle_sec}s")
        print(f"  Split: {action_7.rho_ns:.2f}/{action_7.rho_ew:.2f}")
        
        if action_7.cycle_sec == 60 and abs(action_7.rho_ns - 0.60) < 1e-5:
             print("  [OK] Action 7 is (60s, 0.6/0.4) as expected.")
        elif action_7.cycle_sec == 60 and abs(action_7.rho_ns - 0.50) < 1e-5:
             print("  [OK] Action 7 is (60s, 0.5/0.5).")
        else:
             print("  [WARN] Action 7 properties unexpected. Please check index alignment.")

        print("\n[SUCCESS] Action table structure verified.")

    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()