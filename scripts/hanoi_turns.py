from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Mapping


def _dir_index_map(order: Iterable[str]) -> Dict[str, int]:
    return {d.upper(): i for i, d in enumerate([str(x).upper() for x in order])}


def _left_dir(idx: int, order: List[str]) -> str:
    return order[(idx - 1) % len(order)]


def _right_dir(idx: int, order: List[str]) -> str:
    return order[(idx + 1) % len(order)]


def resolve_turn_mapping(calib: Mapping[str, object]) -> Dict[str, Dict[str, List[str]]]:
    """
    Resolve entry->turn directions mapping to exit edges.

    Priority:
    1) turn_mapping explicit: {entry: {L: [...], S: [...], R: [...]}}
    2) Fallback cardinal mapping when 4-entry/4-exit with approach_order + entry_by_dir + exit_by_dir.
    """
    explicit = calib.get("turn_mapping")
    if isinstance(explicit, dict) and len(explicit) > 0:
        mapping: Dict[str, Dict[str, List[str]]] = {}
        for entry, dirs in explicit.items():
            mapping[str(entry)] = {k: [str(x) for x in v] for k, v in dirs.items()}
        return mapping

    entry_by_dir = calib.get("entry_by_dir", {})
    exit_by_dir = calib.get("exit_by_dir", {})
    approach_order = [str(x).upper() for x in calib.get("approach_order", ["N", "E", "S", "W"])]

    if len(entry_by_dir) == 4 and len(exit_by_dir) == 4:
        order_map = _dir_index_map(approach_order)
        mapping: Dict[str, Dict[str, List[str]]] = {}
        for dir_key, entry_edge in entry_by_dir.items():
            dir_upper = str(dir_key).upper()
            if dir_upper not in order_map:
                continue
            idx = order_map[dir_upper]
            left_dir = _left_dir(idx, approach_order)
            right_dir = _right_dir(idx, approach_order)
            straight_dir = dir_upper
            try:
                mapping[str(entry_edge)] = {
                    "L": [str(exit_by_dir[left_dir])],
                    "S": [str(exit_by_dir[straight_dir])],
                    "R": [str(exit_by_dir[right_dir])],
                }
            except KeyError:
                continue
        if len(mapping) == len(entry_by_dir):
            return mapping

    raise ValueError("turn mapping required to apply L/S/R probabilities; provide turn_mapping or cardinal mapping helpers")


def build_turn_ratios_xml(
    turn_map: Dict[str, Dict[str, List[str]]],
    turning_probs: Dict[str, Dict[str, float]],
    begin: float,
    end: float,
) -> str:
    root = ET.Element("turns")
    interval = ET.SubElement(root, "interval", begin=f"{float(begin):.1f}", end=f"{float(end):.1f}")

    for entry, dirs in turn_map.items():
        probs = turning_probs.get(entry, {})
        for dir_key, exits in dirs.items():
            prob_dir = float(probs.get(dir_key, 0.0))
            if prob_dir < 0.0 or len(exits) == 0:
                continue
            share = prob_dir / float(len(exits))
            for exit_edge in exits:
                ET.SubElement(
                    interval,
                    "edgeRelation",
                    attrib={
                        "from": str(entry),
                        "to": str(exit_edge),
                        "probability": f"{share:.6f}",
                    },
                )

    xml_str = ET.tostring(root, encoding="unicode")
    return xml_str
