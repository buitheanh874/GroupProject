from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

LEGACY_PATTERNS: List[str] = [
    "networks/" + "BI" + "*.net.xml",
    "networks/" + "BI" + "*.rou.xml",
    "networks/" + "BIG" + "MAP" + "*.net.xml",
    "networks/" + "BIG" + "MAP" + "*.rou.xml",
    "networks/variants",
    "networks/test_variants",
    "configs/train_*" + "bigmap" + "*.yaml",
    "configs/train_*" + "bi" + "*.yaml",
    "configs/eval_*" + "bigmap" + "*.yaml",
    "configs/eval_*" + "bi" + "*.yaml",
]

REFERENCE_REGEX = (
    r"BI" + r"\.net\.xml"
    + r"|BI" + r"_"
    + r"|BIG" + r"MAP"
    + r"|\btls0\b"
    + r"|single[-_ ]?intersection|one[-_ ]?intersection"
)


def _rg_available() -> bool:
    try:
        subprocess.run(["rg", "--version"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def print_references() -> None:
    if not _rg_available():
        print("[WARN] ripgrep not available; skipping reference scan")
        return
    result = subprocess.run(["rg", "-n", REFERENCE_REGEX, "."], text=True, capture_output=True)
    if result.returncode == 0 and result.stdout.strip():
        print("[WARN] Found references to legacy patterns:")
        print(result.stdout.strip())
    else:
        print("[OK] No references to legacy patterns found")


def find_candidates(patterns: Iterable[str]) -> list[Path]:
    seen = set()
    candidates: list[Path] = []
    for pattern in patterns:
        for path in Path(".").glob(pattern):
            if path.exists() and path not in seen:
                seen.add(path)
                candidates.append(path)
    return sorted(candidates)


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanup legacy single-node assets (dry-run by default).")
    parser.add_argument("--apply", action="store_true", help="Actually delete the files/directories.")
    args = parser.parse_args()

    print_references()

    candidates = find_candidates(LEGACY_PATTERNS)
    if not candidates:
        print("No legacy assets found.")
        return

    mode = "DELETE" if args.apply else "DRY-RUN"
    print(f"\n[{mode}] Legacy assets matching allowlist:")
    for path in candidates:
        print(f" - {path}")

    if args.apply:
        for path in candidates:
            delete_path(path)
        print("\n[OK] Legacy assets deleted.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(f"cleanup_legacy_assets failed: {exc}")
