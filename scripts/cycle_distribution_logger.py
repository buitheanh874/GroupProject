from __future__ import annotations

from collections import Counter
from typing import Dict, List


class CycleDistributionTracker:
    def __init__(self, allowed_cycles: List[int]):
        self.allowed_cycles = sorted(set(allowed_cycles))
        self.reset()

    def reset(self) -> None:
        self.cycle_counts = Counter({cycle: 0 for cycle in self.allowed_cycles})
        self.total_steps = 0

    def record(self, cycle_sec: int) -> None:
        if cycle_sec in self.cycle_counts:
            self.cycle_counts[cycle_sec] += 1
            self.total_steps += 1
        else:
            print(f"[WARN] Unexpected cycle_sec={cycle_sec}, allowed={self.allowed_cycles}")

    def get_distribution(self) -> Dict[int, float]:
        if self.total_steps == 0:
            return {cycle: 0.0 for cycle in self.allowed_cycles}

        return {
            cycle: float(count) / float(self.total_steps) * 100.0
            for cycle, count in self.cycle_counts.items()
        }

    def get_summary_str(self) -> str:
        if self.total_steps == 0:
            return "No cycles recorded"

        dist = self.get_distribution()
        parts = [f"{cycle}s: {pct:.1f}%" for cycle, pct in sorted(dist.items())]
        return f"Cycle distribution (n={self.total_steps}): " + ", ".join(parts)

    def get_entropy(self) -> float:
        import math

        if self.total_steps == 0:
            return 0.0

        entropy = 0.0
        for count in self.cycle_counts.values():
            if count > 0:
                p = float(count) / float(self.total_steps)
                entropy -= p * math.log2(p)

        return entropy


def example_usage():
    tracker = CycleDistributionTracker(allowed_cycles=[30, 60, 90])

    import random

    for _ in range(100):
        cycle = random.choices([30, 60, 90], weights=[0.1, 0.8, 0.1])[0]
        tracker.record(cycle)

    print(tracker.get_summary_str())
    print(f"Entropy: {tracker.get_entropy():.3f} (max={2.585:.3f} for uniform)")
    print()

    tracker.reset()
    for _ in range(100):
        cycle = random.choice([30, 60, 90])
        tracker.record(cycle)

    print(tracker.get_summary_str())
    print(f"Entropy: {tracker.get_entropy():.3f} (should be higher)")


if __name__ == "__main__":
    example_usage()

    print("\n" + "=" * 80)
    print("INTEGRATION GUIDE")
    print("=" * 80)
    print("1. Copy CycleDistributionTracker class to scripts/train.py")
    print("2. Follow PATCH INSTRUCTIONS in docstring above")
    print("3. Run training and check logs for cycle distribution")
    print("4. Analyze entropy: low = stuck on one cycle, high = exploring")
