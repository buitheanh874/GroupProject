from __future__ import annotations

import csv
from pathlib import Path

from scripts.eval import build_eval_row


def test_build_eval_row_and_csv(tmp_path: Path):
    kpi = {
        "arrived_vehicles": 20,
        "avg_wait_time": 5.0,
        "avg_travel_time": 30.0,
        "avg_stops": 1.2,
        "avg_queue": 3.0,
        "max_wait_time": 15.0,
        "p95_wait_time": 12.0,
    }
    row = build_eval_row(
        controller="rl",
        scenario="demo",
        run_id=1,
        total_reward=100.0,
        episode_steps=50,
        kpi=kpi,
    )
    assert row["throughput"] == 20 / 50
    assert set(row.keys()) == {
        "controller",
        "scenario",
        "run_id",
        "total_reward",
        "episode_steps",
        "arrived_vehicles",
        "avg_wait_time",
        "avg_travel_time",
        "avg_stops",
        "avg_queue",
        "max_wait_time",
        "p95_wait_time",
        "throughput",
        "teleport_started_total",
        "teleport_unique",
        "teleport_rate",
        "arrived_corr",
        "teleported_arrived",
        "completion_rate",
        "failed_corr",
        "avg_wait_time_corr",
        "avg_travel_time_corr",
        "p95_wait_time_corr",
        "max_wait_time_corr",
        "throughput_corr",
        "deadlock_triggered",
        "deadlock_reason",
        "deadlock_no_arrival_sec",
    }

    csv_path = tmp_path / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
