from __future__ import annotations

import sys
from pathlib import Path
import re


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "configs"
    
    print("=" * 80)
    print("UPDATING BASELINE ACTION IDs (2 -> 7)")
    print("=" * 80)
    
    yaml_files = list(config_dir.glob("*.yaml"))
    
    if not yaml_files:
        print("No .yaml files found in configs/")
        return

    updated_count = 0
    pattern = re.compile(r"(fixed_action_id\s*:\s*)2\b")
    
    for yaml_file in yaml_files:
        try:
            content = yaml_file.read_text(encoding="utf-8")
            
            if pattern.search(content):
                new_content = pattern.sub(r"\g<1>7", content)
                yaml_file.write_text(new_content, encoding="utf-8")
                print(f"[UPDATED] {yaml_file.name}")
                updated_count += 1
            else:
                if "fixed_action_id" in content:
                    pass
                    
        except Exception as e:
            print(f"[ERROR] Could not process {yaml_file.name}: {e}")

    print("-" * 80)
    print(f"Total files updated: {updated_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()