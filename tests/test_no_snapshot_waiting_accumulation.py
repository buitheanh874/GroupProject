from __future__ import annotations

from pathlib import Path


def test_no_snapshot_waiting_accumulation_patterns():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "env" / "sumo_env.py"
    data = path.read_text(encoding="utf-8")
    assert "w_ns += float(len(queued_ns))" not in data
    assert "w_ew += float(len(queued_ew))" not in data
    assert "w_dir[tls_id] += snapshot_counts" not in data


if __name__ == "__main__":
    test_no_snapshot_waiting_accumulation_patterns()
    print("test_no_snapshot_waiting_accumulation passed")
