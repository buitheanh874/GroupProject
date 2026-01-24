from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: str) -> Tuple[List[int], List[float]]:
    episodes: List[int] = []
    rewards: List[float] = []

    with open(str(path), "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            episode = int(row.get("episode", 0))
            reward = float(row.get("episode_reward", 0.0))
            episodes.append(episode)
            rewards.append(reward)

    return episodes, rewards


def moving_average(values: List[float], window: int) -> List[float]:
    if int(window) <= 1:
        return list(values)

    array = np.asarray(values, dtype=np.float32)
    kernel = np.ones(int(window), dtype=np.float32) / float(int(window))
    smoothed = np.convolve(array, kernel, mode="valid")
    padding = [float(smoothed[0])] * (int(window) - 1)
    output = padding + [float(x) for x in smoothed.tolist()]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    episodes, rewards = load_metrics(args.csv)
    smoothed = moving_average(rewards, window=int(args.window))

    plt.figure()
    plt.plot(episodes, rewards, label="Reward")
    plt.plot(episodes, smoothed, label="MovingAverage")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.grid(True)

    if str(args.out).strip() == "":
        csv_path = Path(args.csv)
        out_path = csv_path.with_suffix(".png")
    else:
        out_path = Path(args.out)

    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {str(out_path)}")


if __name__ == "__main__":
    main()
