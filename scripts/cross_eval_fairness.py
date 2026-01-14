from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _patch_config(base_cfg: Dict[str, Any], lambda_fairness: float, checkpoint: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    env_cfg = cfg.setdefault("env", {}).setdefault("sumo", {})
    env_cfg["lambda_fairness"] = float(lambda_fairness)
    eval_cfg = cfg.setdefault("eval", {})
    eval_cfg["model_path"] = checkpoint
    return cfg


def _run_eval(eval_script: Path, base_config_path: str, base_cfg: Dict[str, Any], checkpoint: str, lambda_fairness: float, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_cfg_path = output_dir / f"config_lambda{lambda_fairness:.3f}.yaml"
    patched = _patch_config(base_cfg, lambda_fairness=lambda_fairness, checkpoint=checkpoint)
    with tmp_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(patched, f, sort_keys=False)

    log_path = output_dir / f"eval_lambda{lambda_fairness:.3f}.log"
    cmd = [
        sys.executable,
        str(eval_script),
        "--config",
        str(tmp_cfg_path),
    ]
    print("Executing:", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"[WARN] Eval failed for lambda={lambda_fairness}. See {log_path}")
    return {"config_path": str(tmp_cfg_path), "log_path": str(log_path)}


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run cross-evaluation of fairness vs pure checkpoints.")
    parser.add_argument("--eval-script", default="scripts/eval.py", help="Path to eval entrypoint.")
    parser.add_argument("--config", required=True, help="Base eval config path.")
    parser.add_argument("--pure_ckpt", required=True, help="Checkpoint path for pure policy.")
    parser.add_argument("--fair_ckpt", required=True, help="Checkpoint path for fairness policy.")
    parser.add_argument("--lambda_values", nargs="+", type=float, default=[0.0, 0.12], help="Lambda fairness values to evaluate.")
    parser.add_argument("--output-dir", default="results/cross_eval", help="Directory to store logs and summary.")
    args = parser.parse_args(argv)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    eval_script = Path(args.eval_script)
    if not eval_script.exists():
        sys.exit(f"Eval script not found: {eval_script}")

    summary_rows = []
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        sys.exit(f"Failed to load base config: {exc}")

    for ckpt_label, ckpt_path in [("pure", args.pure_ckpt), ("fair", args.fair_ckpt)]:
        for lambda_val in args.lambda_values:
            out_dir = output_root / f"{ckpt_label}_lambda{lambda_val:.3f}"
            result_paths = _run_eval(
                eval_script,
                base_config_path=args.config,
                base_cfg=base_cfg,
                checkpoint=ckpt_path,
                lambda_fairness=lambda_val,
                output_dir=out_dir,
            )
            summary_rows.append(
                {
                    "policy": ckpt_label,
                    "lambda_fairness": lambda_val,
                    "config_path": result_paths["config_path"],
                    "log_path": result_paths["log_path"],
                }
            )

    csv_path = output_root / "cross_eval_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "lambda_fairness", "config_path", "log_path"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote summary to {csv_path}")


if __name__ == "__main__":
    main()
