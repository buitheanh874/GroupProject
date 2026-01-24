#!/usr/bin/env python
"""
Curriculum Report - Analyze curriculum training metrics

Reports:
1. Buffer phase histogram (distribution of phases in replay buffer)
2. Sampled batch phase histogram (distribution in sampled batches)
3. Route overlap between workers (Jaccard similarity)
4. Planned vs executed episodes per phase

Usage:
    python scripts/report_curriculum.py --log-dir logs/1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curriculum training report")
    parser.add_argument("--log-dir", type=str, required=True, help="Training log directory")
    parser.add_argument("--config", type=str, default="configs/train_1.yaml", help="Config file")
    parser.add_argument("--output", type=str, default=None, help="Output report file")
    return parser.parse_args(argv)


def load_training_logs(log_dir: Path) -> Dict[str, Any]:
    """Load training logs from log directory."""
    logs = {
        "buffer_histograms": [],
        "sampled_batch_histograms": [],
        "episode_route_log": [],
        "worker_routes": defaultdict(set),
    }
    
    # Look for curriculum log files
    curriculum_log = log_dir / "curriculum_stats.jsonl"
    if curriculum_log.exists():
        with open(curriculum_log, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if "buffer_phase_histogram" in entry:
                        logs["buffer_histograms"].append(entry)
                    if "sampled_batch_phase_histogram" in entry:
                        logs["sampled_batch_histograms"].append(entry)
                except json.JSONDecodeError:
                    continue
    
    # Look for episode logs from workers
    for worker_log in log_dir.glob("worker_*.log"):
        try:
            with open(worker_log, 'r') as f:
                content = f.read()
                # Parse route files used
                for line in content.split('\n'):
                    if "Route" in line and ".rou.xml" in line:
                        # Extract route filename
                        parts = line.split()
                        for part in parts:
                            if ".rou.xml" in part:
                                worker_id = worker_log.stem.split('_')[-1]
                                logs["worker_routes"][worker_id].add(part)
        except Exception:
            continue
    
    return logs


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(intersection) / float(union) if union > 0 else 0.0


def get_planned_episodes(config_path: str) -> Dict[str, int]:
    """Get planned episodes per phase from config."""
    try:
        from rl.utils import load_yaml_config
        config = load_yaml_config(config_path)
        curriculum = config.get("curriculum", {})
        phases = curriculum.get("phases", [])
        return {p.get("name", f"phase_{i}"): p.get("episodes", 0) for i, p in enumerate(phases)}
    except Exception as e:
        print(f"Failed to load config: {e}")
        return {}


def generate_report(
    logs: Dict[str, Any],
    planned_episodes: Dict[str, int],
    output_path: Optional[Path] = None,
) -> str:
    """Generate curriculum report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CURRICULUM TRAINING REPORT")
    lines.append("=" * 70)
    
    # 1. Planned vs executed episodes
    lines.append("\n## Planned Episodes per Phase")
    lines.append("-" * 40)
    if planned_episodes:
        for phase, eps in planned_episodes.items():
            lines.append(f"  {phase}: {eps} episodes")
    else:
        lines.append("  (No curriculum phases found)")
    
    # 2. Buffer phase histogram
    lines.append("\n## Buffer Phase Histogram (last recorded)")
    lines.append("-" * 40)
    if logs["buffer_histograms"]:
        last_hist = logs["buffer_histograms"][-1]
        hist = last_hist.get("buffer_phase_histogram", {})
        total = sum(hist.values())
        for phase_idx, count in sorted(hist.items(), key=lambda x: int(x[0]) if x[0] != '-1' else -1):
            pct = count / total * 100 if total > 0 else 0
            phase_name = f"Phase {phase_idx}" if phase_idx != '-1' else "Unknown"
            lines.append(f"  {phase_name}: {count} ({pct:.1f}%)")
    else:
        lines.append("  (No buffer histogram data)")
    
    # 3. Sampled batch phase histogram
    lines.append("\n## Sampled Batch Phase Histogram (last recorded)")
    lines.append("-" * 40)
    if logs["sampled_batch_histograms"]:
        last_hist = logs["sampled_batch_histograms"][-1]
        hist = last_hist.get("sampled_batch_phase_histogram", {})
        total = sum(hist.values())
        for phase_idx, count in sorted(hist.items(), key=lambda x: int(x[0]) if x[0] != '-1' else -1):
            pct = count / total * 100 if total > 0 else 0
            phase_name = f"Phase {phase_idx}" if phase_idx != '-1' else "Unknown"
            lines.append(f"  {phase_name}: {count} ({pct:.1f}%)")
        
        # Check for curriculum evidence
        lines.append("")
        if len(hist) > 1:
            lines.append("  ✅ Multiple phases found in sampled batches - curriculum evidence present")
        else:
            lines.append("  ⚠️ Single phase or unknown - check curriculum implementation")
    else:
        lines.append("  (No sampled batch histogram data)")
    
    # 4. Route overlap between workers
    lines.append("\n## Route Overlap Between Workers (Jaccard)")
    lines.append("-" * 40)
    worker_routes = logs["worker_routes"]
    if len(worker_routes) >= 2:
        worker_ids = sorted(worker_routes.keys())
        for i, w1 in enumerate(worker_ids):
            for w2 in worker_ids[i+1:]:
                jaccard = compute_jaccard_similarity(worker_routes[w1], worker_routes[w2])
                lines.append(f"  Worker {w1} ↔ Worker {w2}: {jaccard:.2f}")
        
        # Unique routes
        all_routes = set()
        for routes in worker_routes.values():
            all_routes.update(routes)
        lines.append(f"\n  Total unique routes used: {len(all_routes)}")
    else:
        lines.append("  (Insufficient worker data)")
    
    lines.append("\n" + "=" * 70)
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        print("\nNOTE: If training hasn't run yet, no data to report.")
        return 1
    
    logs = load_training_logs(log_dir)
    planned_episodes = get_planned_episodes(args.config)
    
    output_path = Path(args.output) if args.output else None
    report = generate_report(logs, planned_episodes, output_path)
    print(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
