from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET


def _scale_value(value: str, factor: float) -> str:
    try:
        numeric = float(value)
        scaled = max(0.0, numeric * factor)
        if value.isdigit():
            return str(int(round(scaled)))
        return str(scaled)
    except Exception:
        return value


def _apply_scaling(tree: ET.ElementTree, global_factor: float, noise_range: Tuple[float, float], rng: random.Random) -> ET.ElementTree:
    root = tree.getroot()
    demand_tags = {"flow"}
    demand_keys = {"probability", "vehsPerHour", "number"}
    for elem in root.iter():
        if elem.tag not in demand_tags:
            continue
        noise = rng.uniform(noise_range[0], noise_range[1])
        factor = max(0.0, global_factor * (1.0 + noise))
        for key in list(elem.attrib.keys()):
            if key not in demand_keys:
                continue
            raw = elem.get(key, "")
            scaled = _scale_value(raw, factor)
            if key == "number":
                try:
                    scaled_int = max(0, int(round(float(scaled))))
                    elem.set(key, str(scaled_int))
                except Exception:
                    elem.set(key, scaled)
            else:
                try:
                    scaled_float = float(scaled)
                    if key == "probability":
                        scaled_float = min(1.0, max(0.0, scaled_float))
                    elem.set(key, f"{scaled_float:.6f}")
                except Exception:
                    elem.set(key, scaled)
    return tree


def _write_variant(tree: ET.ElementTree, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate randomized SUMO route variants by scaling flow demands.")
    parser.add_argument("--input", required=True, help="Input .rou.xml file to randomize.")
    parser.add_argument("--output-dir", required=True, help="Directory to write randomized variants.")
    parser.add_argument("--variants", type=int, default=3, help="Number of randomized variants to produce.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic scaling factors.")
    parser.add_argument("--global-range", nargs=2, type=float, default=[0.7, 1.3], metavar=("MIN", "MAX"), help="Range for global scaling factor.")
    parser.add_argument("--per-flow-noise", type=float, default=0.1, help="Per-flow multiplicative noise magnitude (e.g., 0.1 for +/-10%).")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input route file not found: {input_path}")

    output_dir = Path(args.output_dir)
    try:
        tree_orig = ET.parse(str(input_path))
    except Exception as exc:
        sys.exit(f"Failed to parse route file: {exc}")

    min_scale, max_scale = float(args.global_range[0]), float(args.global_range[1])
    if min_scale <= 0 or max_scale <= 0 or min_scale > max_scale:
        sys.exit("global-range values must be positive and MIN <= MAX.")

    noise_mag = max(0.0, float(args.per_flow_noise))
    noise_range = (-noise_mag, noise_mag)
    rng_global = random.Random(int(args.seed))

    for idx in range(int(args.variants)):
        global_factor = rng_global.uniform(min_scale, max_scale)
        rng_variant = random.Random(int(args.seed) + idx)
        tree_variant = ET.ElementTree(copy.deepcopy(tree_orig.getroot()))
        tree_variant = _apply_scaling(tree_variant, global_factor=global_factor, noise_range=noise_range, rng=rng_variant)
        output_path = output_dir / f"{input_path.stem}_scaled_seed{args.seed}_idx{idx}.rou.xml"
        _write_variant(tree_variant, output_path)
        print(f"Wrote variant {idx}: {output_path} (global_factor={global_factor:.3f}, noise=+/-{noise_mag:.2f})")


if __name__ == "__main__":
    main()
