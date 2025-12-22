from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from rl.utils import ensure_dir, generate_run_id, load_yaml_config, set_global_seed
from scripts.common import build_agent, build_env


def run_episode(env: Any, controller: str, agent: Any, fixed_action_id: int, seed: int) -> Dict[str, Any]:
    if hasattr(env, "set_seed"):
        env.set_seed(int(seed))

    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    info: Dict[str, Any] = {}

    while not done:
        if controller == "fixed":
            action_id = int(fixed_action_id)
        else:
            action_id = int(agent.select_action(state=state, epsilon=0.0))

        state, reward, done, info = env.step(action_id)
        total_reward += float(reward)
        steps += 1

    kpi = {}
    if isinstance(info, dict):
        kpi = info.get("episode_kpi", {})

    return {
        "reward": float(total_reward),
        "steps": int(steps),
        "arrived_vehicles": int(kpi.get("arrived_vehicles", 0)),
        "avg_wait_time": float(kpi.get("avg_wait_time", 0.0)),
        "avg_travel_time": float(kpi.get("avg_travel_time", 0.0)),
        "avg_stops": float(kpi.get("avg_stops", 0.0)),
        "avg_queue": float(kpi.get("avg_queue", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_sumo.yaml")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller", choices=["fixed", "rl"], default="fixed")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config.setdefault("run", {})
    config["run"]["seed"] = int(args.seed)

    set_global_seed(int(args.seed))

    env = build_env(config)
    agent = None

    if args.controller == "rl":
        if not args.model_path:
            sys.exit("model-path is required when controller=rl")
        agent, _ = build_agent(config, env)
        agent.load_model(str(args.model_path))
        agent.to_eval_mode()

    fixed_action_id = int(config.get("baseline", {}).get("fixed_action_id", 2))

    run_id = generate_run_id(prefix="eval_sumo")
    out_path = Path(args.out) if args.out else Path(ensure_dir("logs")) / f"{run_id}.csv"
    ensure_dir(str(out_path.parent))

    fieldnames = [
        "episode",
        "controller",
        "seed",
        "reward",
        "steps",
        "arrived_vehicles",
        "avg_wait_time",
        "avg_travel_time",
        "avg_stops",
        "avg_queue",
    ]

    results: List[Dict[str, Any]] = []

    try:
        for ep in range(int(args.episodes)):
            episode_seed = int(args.seed + ep)
            result = run_episode(env, args.controller, agent, fixed_action_id, episode_seed)
            row = {
                "episode": int(ep + 1),
                "controller": args.controller,
                "seed": int(episode_seed),
                **result,
            }
            results.append(row)
    finally:
        try:
            env.close()
        except Exception:
            pass

    aggregate: Dict[str, Any] = {}
    if len(results) > 0:
        aggregate = {"episode": "mean", "controller": args.controller, "seed": ""}
        for key in ["reward", "steps", "arrived_vehicles", "avg_wait_time", "avg_travel_time", "avg_stops", "avg_queue"]:
            aggregate[key] = float(np.mean([float(r[key]) for r in results])) if hasattr(np, "mean") else sum(float(r[key]) for r in results) / len(results)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
        if aggregate:
            writer.writerow(aggregate)

    print(f"Evaluation complete. Results: {out_path}")


if __name__ == "__main__":
    main()
