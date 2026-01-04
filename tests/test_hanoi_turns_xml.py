from __future__ import annotations

import xml.etree.ElementTree as ET

from scripts.hanoi_turns import build_turn_ratios_xml


def test_turn_ratios_xml_probabilities_sum_to_one():
    turn_map = {
        "A_IN": {"L": ["X_OUT"], "S": ["Y_OUT"], "R": ["Z_OUT"]},
    }
    probs = {"A_IN": {"L": 0.2, "S": 0.6, "R": 0.2}}
    xml_text = build_turn_ratios_xml(turn_map, probs, begin=0.0, end=3600.0)
    root = ET.fromstring(xml_text)
    interval = root.find("interval")
    assert interval is not None
    relations = interval.findall("edgeRelation")
    assert len(relations) == 3
    total_prob = sum(float(el.get("probability")) for el in relations if el.get("from") == "A_IN")
    assert abs(total_prob - 1.0) < 1e-6
    assert {el.get("to") for el in relations} == {"X_OUT", "Y_OUT", "Z_OUT"}


def test_turn_ratios_xml_multiple_exits_share_probability():
    turn_map = {"A": {"L": ["X1", "X2"], "S": ["Y"], "R": []}}
    probs = {"A": {"L": 0.6, "S": 0.4}}
    xml_text = build_turn_ratios_xml(turn_map, probs, begin=0.0, end=60.0)
    root = ET.fromstring(xml_text)
    rels = root.findall(".//edgeRelation")
    left_probs = [float(el.get("probability")) for el in rels if el.get("to") in {"X1", "X2"}]
    assert len(left_probs) == 2
    assert abs(sum(left_probs) - 0.6) < 1e-6
