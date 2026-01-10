from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from controllers.max_pressure import select_action_from_defs
from env.sumo_env import SumoActionDefinition
from scripts.common import resolve_tls_ids_from_sumo_cfg
from scripts.sumo_network_tools import extract_tls_ids


def test_extract_tls_ids_bignet_file():
    ids = extract_tls_ids(Path("networks/BIGNET.net.xml"))
    assert len(ids) == 9
    for expected in ["J0", "J1", "J14", "J17", "J2", "J3", "J4", "J6", "J7"]:
        assert expected in ids


def test_auto_tls_ids_in_build_env(tmp_path: Path):
    net_file = tmp_path / "net.xml"
    net_file.write_text(
        """
        <net>
          <tlLogic id="TL1" type="static" programID="0" offset="0"></tlLogic>
          <tlLogic id="TL2" type="static" programID="0" offset="0"></tlLogic>
        </net>
        """,
        encoding="utf-8",
    )

    sumo_cfg = {
        "net_file": str(net_file),
        "tls_ids": "auto",
        "center_tls_id": None,
    }

    tls_ids, center = resolve_tls_ids_from_sumo_cfg(sumo_cfg, net_file)
    assert center == "TL1"
    assert sorted(tls_ids) == ["TL1", "TL2"]


def test_tls_id_validation_rejects_duplicates(tmp_path: Path):
    net_file = tmp_path / "net.xml"
    route_file = tmp_path / "route.rou.xml"
    net_file.write_text("<net><tlLogic id='A'/></net>", encoding="utf-8")
    route_file.write_text("<routes><vehicle id='v0'/></routes>", encoding="utf-8")

    sumo_cfg = {
        "net_file": str(net_file),
        "route_file": str(route_file),
        "tls_ids": ["A", "A"],
        "center_tls_id": "A",
    }

    with pytest.raises(ValueError):
        resolve_tls_ids_from_sumo_cfg(sumo_cfg, net_file)


def test_center_tls_must_be_in_tls_ids(tmp_path: Path):
    net_file = tmp_path / "net.xml"
    route_file = tmp_path / "route.rou.xml"
    net_file.write_text("<net><tlLogic id='A'/><tlLogic id='B'/></net>", encoding="utf-8")
    route_file.write_text("<routes><vehicle id='v0'/></routes>", encoding="utf-8")

    sumo_cfg = {
        "net_file": str(net_file),
        "route_file": str(route_file),
        "tls_ids": ["A", "B"],
        "center_tls_id": "C",
    }

    with pytest.raises(ValueError):
        resolve_tls_ids_from_sumo_cfg(sumo_cfg, net_file)


def test_controller_actions_cover_all_tls_ids():
    tls_ids = [f"J{i}" for i in range(9)]
    action_defs = [
        SumoActionDefinition(cycle_sec=30, rho_ns=0.30, rho_ew=0.70),
        SumoActionDefinition(cycle_sec=30, rho_ns=0.50, rho_ew=0.50),
        SumoActionDefinition(cycle_sec=30, rho_ns=0.70, rho_ew=0.30),
    ]
    allowed = [0, 1, 2]
    state_map = {
        tls: np.array([float(idx + 1), float(idx), float(idx + 0.5), float(idx * 0.1)], dtype=np.float32)
        for idx, tls in enumerate(tls_ids)
    }

    actions = {
        tls: select_action_from_defs(
            state_raw=state_map[tls],
            action_defs=action_defs,
            allowed_action_ids=allowed,
            default_action_id=allowed[0],
        )
        for tls in tls_ids
    }

    assert set(actions.keys()) == set(tls_ids)
    assert set(actions.values()).issubset(set(allowed))
