from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def main() -> None:
    print("=" * 80)
    print("Collecting Normalization Stats for Ultimate Scenario")
    print("=" * 80)

    config_path = "configs/train_ultimate_pure.yaml"
    output_path = "configs/norm_stats_ultimate_clean.json"

    config_abs = (repo_root / config_path).resolve()
    output_abs = (repo_root / output_path).resolve()

    if not config_abs.exists():
        sys.exit(f"Config file not found: {config_abs}")

    output_abs.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = output_abs.with_suffix(output_abs.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "collect_norm_stats.py"),
        "--config", str(config_abs),
        "--episodes", "5",
        "--seed", "0",
        "--out", str(tmp_path),
    ]

    try:
        print("\nRunning normalization collection...")
        result = subprocess.run(cmd, cwd=str(repo_root), check=True)
        
        if not tmp_path.exists():
            sys.exit(f"Collector succeeded but temp file not found: {tmp_path}")
        
        tmp_path.replace(output_abs)
        
    except subprocess.CalledProcessError as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        sys.exit(f"Normalization collection failed with exit code {exc.returncode}")
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise exc

    print("\n" + "=" * 80)
    print(f"Normalization stats saved to: {output_abs}")
    print("=" * 80)
    
    if output_abs.exists():
        with open(output_abs, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()