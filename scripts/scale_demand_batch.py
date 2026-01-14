"""
Batch-generate scaled route files for curriculum learning.

Based on demand scaling methodology from:
- MetaLight (AAAI 2020, Zang et al.) - Curriculum learning approach
- Advanced-MPLight (arXiv 2021, Zhang et al.) - Demand scaling from 0.5x to 2.0x

Usage:
    python scripts/scale_demand_batch.py --input-dir networks/variants/train --scales 0.5 0.75 1.0 --sample 30
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET


def _scale_depart_times(tree: ET.ElementTree, scale_factor: float) -> ET.ElementTree:
    """
    Scale demand by adjusting departure times.
    
    Lower scale_factor = less traffic (vehicles spread out more in time).
    scale_factor=0.5 means half the vehicles per unit time.
    
    For flow-based routes: scale probability/vehsPerHour attributes.
    For vehicle-based routes: spread depart times.
    """
    root = tree.getroot()
    
    flow_demand_keys = {"probability", "vehsPerHour", "number"}
    for elem in root.iter("flow"):
        for key in list(elem.attrib.keys()):
            if key not in flow_demand_keys:
                continue
            try:
                raw_value = float(elem.get(key, "0"))
                scaled = raw_value * scale_factor
                if key == "number":
                    scaled = max(0, int(round(scaled)))
                    elem.set(key, str(scaled))
                elif key == "probability":
                    scaled = min(1.0, max(0.0, scaled))
                    elem.set(key, f"{scaled:.6f}")
                else:
                    elem.set(key, f"{scaled:.6f}")
            except (ValueError, TypeError):
                pass
    
    for elem in root.iter("vehicle"):
        depart = elem.get("depart")
        if depart is None:
            continue
        try:
            depart_val = float(depart)
            new_depart = depart_val / max(0.01, scale_factor)
            elem.set("depart", f"{new_depart:.2f}")
        except (ValueError, TypeError):
            pass
    
    return tree


def _scale_route_file(input_path: Path, output_path: Path, scale_factor: float) -> None:
    """Scale a single route file and save to output_path."""
    try:
        tree = ET.parse(str(input_path))
    except ET.ParseError as e:
        print(f"  [WARN] Failed to parse {input_path.name}: {e}")
        return
    
    tree = _scale_depart_times(tree, scale_factor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)


def _collect_base_routes(input_dir: Path, sample: int) -> List[Path]:
    """Collect base route files (non-scaled originals)."""
    all_rou = sorted(input_dir.glob("*.rou.xml"))
    base_routes = [
        p for p in all_rou 
        if "_scale" not in p.name and "_scaled_" not in p.name
    ]
    
    if len(base_routes) == 0:
        base_routes = list(all_rou)
    
    if sample > 0 and len(base_routes) > sample:
        rng = random.Random(42)
        base_routes = rng.sample(base_routes, sample)
    
    return base_routes


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate scaled route files for curriculum learning."
    )
    parser.add_argument(
        "--input-dir", 
        type=str, 
        required=True,
        help="Directory containing base .rou.xml files"
    )
    parser.add_argument(
        "--scales", 
        type=float, 
        nargs="+", 
        default=[0.5, 0.75, 1.0],
        help="Demand scale factors (e.g., 0.5 0.75 1.0)"
    )
    parser.add_argument(
        "--sample", 
        type=int, 
        default=0,
        help="Number of base routes to sample (0 = all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    args = parser.parse_args(argv)
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        sys.exit(f"Input directory not found: {input_dir}")
    
    base_routes = _collect_base_routes(input_dir, args.sample)
    if len(base_routes) == 0:
        sys.exit(f"No .rou.xml files found in {input_dir}")
    
    print(f"Found {len(base_routes)} base route files")
    print(f"Scales to generate: {args.scales}")
    
    manifests: dict[float, List[str]] = {s: [] for s in args.scales}
    
    for scale in args.scales:
        scale_pct = int(scale * 100)
        output_subdir = input_dir / f"scaled_{scale_pct}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Scale {scale_pct}%] Generating to {output_subdir}")
        
        for route_path in base_routes:
            stem = route_path.stem
            out_name = f"{stem}_scale{scale_pct}.rou.xml"
            out_path = output_subdir / out_name
            
            _scale_route_file(route_path, out_path, scale)
            
            rel_path = out_path.relative_to(input_dir)
            manifests[scale].append(str(rel_path))
        
        print(f"  Generated {len(base_routes)} files")
    
    print("\n[Manifests]")
    for scale, file_list in manifests.items():
        scale_pct = int(scale * 100)
        manifest_path = input_dir / f"manifest_scale{scale_pct}.txt"
        manifest_path.write_text("\n".join(file_list), encoding="utf-8")
        print(f"  {manifest_path.name}: {len(file_list)} files")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
