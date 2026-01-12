from __future__ import annotations
import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


def fix_route_file(filepath: Path, min_veh_per_hour: float = 0.1) -> int:
    try:
        tree = ET.parse(str(filepath))
    except ET.ParseError as e:
        print(f"  [WARN] Parse error: {filepath.name}: {e}")
        return 0
    
    root = tree.getroot()
    removed = 0
    
    flows_to_remove = []
    for elem in root.iter("flow"):
        vph = elem.get("vehsPerHour")
        if vph is not None:
            try:
                vph_val = float(vph)
                if vph_val < min_veh_per_hour:
                    flows_to_remove.append(elem)
            except (ValueError, TypeError):
                pass
        
        begin = elem.get("begin")
        if begin is not None:
            try:
                if float(begin) < 0:
                    elem.set("begin", "0.0")
            except (ValueError, TypeError):
                pass
        
        end = elem.get("end")
        if end is not None:
            try:
                if float(end) < 0:
                    elem.set("end", "3600.0")
            except (ValueError, TypeError):
                pass
    
    for elem in flows_to_remove:
        root.remove(elem)
        removed += 1
    
    if removed > 0 or True:
        tree.write(str(filepath), encoding="utf-8", xml_declaration=True)
    
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--min-veh-per-hour", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        sys.exit(f"Directory not found: {input_dir}")
    
    folders = ["400", "600", "800", "1000", "1200"]
    total_fixed = 0
    total_removed = 0
    
    for folder in folders:
        folder_path = input_dir / folder
        if not folder_path.exists():
            print(f"[SKIP] {folder} not found")
            continue
        
        rou_files = sorted(folder_path.glob("*.rou.xml"))
        print(f"\n[{folder}] Processing {len(rou_files)} files...")
        
        folder_removed = 0
        for f in rou_files:
            if args.dry_run:
                continue
            removed = fix_route_file(f, args.min_veh_per_hour)
            if removed > 0:
                folder_removed += removed
                total_fixed += 1
        
        total_removed += folder_removed
        print(f"  Removed {folder_removed} low-rate flows from {len(rou_files)} files")
    
    print(f"\n[Done] Fixed {total_fixed} files, removed {total_removed} invalid flows")


if __name__ == "__main__":
    main()
