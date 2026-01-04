from __future__ import annotations

from pathlib import Path

from scripts.generate_hanoi_route_variants import (
    generate_routes,
    generate_variant,
    select_seeds,
    write_manifest,
)


def _base_calib() -> dict:
    return {
        "map_prefix": "hanoi",
        "net_file": "net.xml",
        "entry_edges": ["A", "B", "C"],
        "exit_edges": ["X", "Y"],
        "turn_mapping": {
            "A": {"L": ["X"], "S": ["Y"], "R": ["X"]},
            "B": {"L": ["Y"], "S": ["X"], "R": ["Y"]},
            "C": {"L": ["X"], "S": ["Y"], "R": ["X"]},
        },
        "pcu_weights": {"motorcycle": 0.25, "passenger": 1.0},
        "demand": {
            "total_pcu_per_hour": {"low": 1000, "med": 2000, "high": 3000},
            "entry_dirichlet_alpha": 2.0,
        },
        "vehicle_mix": {"mean": {"motorcycle": 0.8, "passenger": 0.2}, "kappa": 20},
        "turning": {"mean_LSR": [0.2, 0.6, 0.2], "kappa": 30, "turning_overrides": {}},
        "stages": {"enabled": False, "intervals": []},
        "min_total_vehicles": 1,
        "horizon_sec": 3600,
    }


def test_seed_split_train_eval_non_overlap():
    seeds_train = set(select_seeds("train", 5, base_seed=42))
    seeds_eval = set(select_seeds("eval", 5, base_seed=42))
    assert seeds_train.isdisjoint(seeds_eval)


def test_determinism_same_seed_same_meta():
    calib = _base_calib()
    _, meta_a = generate_variant(calib, profile="low", seed=123, turn_map=calib["turn_mapping"])
    _, meta_b = generate_variant(calib, profile="low", seed=123, turn_map=calib["turn_mapping"])
    assert meta_a == meta_b


def test_manifest_and_naming(tmp_path: Path):
    calib = _base_calib()
    seeds = [1, 2]
    routes = generate_routes(
        calib=calib,
        split="train",
        profile="auto",
        seeds=seeds,
        out_dir=tmp_path,
        skip_router=True,
    )

    assert len(routes) == len(seeds)
    assert all(route.exists() for route in routes)
    for seed, route in zip(seeds, routes):
        assert f"seed{seed:05d}" in route.name
        assert "_train_" in route.name

    manifest_path = tmp_path / "train" / "manifest.txt"
    write_manifest(manifest_path, routes)
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(seeds)
    assert all(line.endswith(".rou.xml") for line in lines)


def test_high_unbalanced_entry_weight():
    calib = _base_calib()
    _, meta = generate_variant(calib, profile="high_unbalanced", seed=7, turn_map=calib["turn_mapping"])
    pi = meta["entry_split"]
    assert max(pi.values()) >= 0.7


def test_ultimate_uses_multiple_intervals_when_present():
    calib = _base_calib()
    calib["stages"] = {
        "enabled": True,
        "intervals": [
            {"begin": 0, "end": 100, "level": "low"},
            {"begin": 100, "end": 200, "level": "high"},
        ],
    }
    _, meta = generate_variant(calib, profile="ultimate", seed=9, turn_map=calib["turn_mapping"])
    assert len(meta["level_plan"]) > 1
    levels = {interval["level"] for interval in meta["level_plan"]}
    assert "low" in levels and "high" in levels


def test_turns_xml_deterministic_and_counts(tmp_path: Path):
    calib = {
        "map_prefix": "hanoi",
        "net_file": "net.xml",
        "entry_edges": ["N_IN", "E_IN", "S_IN", "W_IN"],
        "exit_edges": ["N_OUT", "E_OUT", "S_OUT", "W_OUT"],
        "approach_order": ["N", "E", "S", "W"],
        "entry_by_dir": {"N": "N_IN", "E": "E_IN", "S": "S_IN", "W": "W_IN"},
        "exit_by_dir": {"N": "N_OUT", "E": "E_OUT", "S": "S_OUT", "W": "W_OUT"},
        "pcu_weights": {"motorcycle": 0.25, "passenger": 1.0},
        "demand": {"total_pcu_per_hour": {"low": 1000, "high": 2000}, "entry_dirichlet_alpha": 2.0},
        "vehicle_mix": {"mean": {"motorcycle": 0.8, "passenger": 0.2}, "kappa": 20},
        "turning": {"mean_LSR": [0.2, 0.6, 0.2], "kappa": 30},
        "stages": {"enabled": False, "intervals": []},
        "min_total_vehicles": 1,
        "horizon_sec": 3600,
    }
    routes = generate_routes(
        calib=calib,
        split="train",
        profile="auto",
        seeds=[42],
        out_dir=tmp_path,
        skip_router=True,
    )
    route_path = routes[0]
    base = route_path.stem.replace(".rou", "")
    turns_path = route_path.parent / f"turns_{base}.xml"
    flows_path = route_path.parent / f"flows_{base}.xml"
    assert turns_path.exists()
    assert flows_path.exists()
    import xml.etree.ElementTree as ET
    xml_root = ET.fromstring(turns_path.read_text(encoding="utf-8"))
    relations = xml_root.findall(".//edgeRelation")
    assert len(relations) == 12
    from_edges = {rel.get("from") for rel in relations}
    assert from_edges == {"N_IN", "E_IN", "S_IN", "W_IN"}

    flow_root = ET.fromstring(flows_path.read_text(encoding="utf-8"))
    vtypes = {vt.get("id") for vt in flow_root.findall("vType")}
    assert {"motorcycle", "passenger", "bus", "other"}.issubset(vtypes)
