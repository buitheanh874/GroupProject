from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.sumo_env import validate_downstream_links_config


def test_missing_downstream_direction_fail_fast():
    with pytest.raises(ValueError, match="missing directions"):
        validate_downstream_links_config(
            downstream_links={"N": "EDGE_N", "E": "EDGE_E"},
            lane_id_set={"E2C_0", "N2C_0"},
            edge_id_set={"EDGE_E", "EDGE_N"},
            center_tls_id="CENTER",
        )


def test_invalid_downstream_ids_rejected():
    with pytest.raises(ValueError, match="invalid mappings"):
        validate_downstream_links_config(
            downstream_links={"N": "EDGE_N", "E": "EDGE_E", "S": "EDGE_S", "W": "EDGE_W"},
            lane_id_set={"E2C_0", "N2C_0"},
            edge_id_set={"EDGE_E", "EDGE_N"},
            center_tls_id="CENTER",
        )


def test_downstream_links_valid_passes():
    validate_downstream_links_config(
        downstream_links={"N": "EDGE_N", "E": "EDGE_E", "S": "EDGE_S", "W": "EDGE_W"},
        lane_id_set={"EDGE_S"},
        edge_id_set={"EDGE_E", "EDGE_N", "EDGE_W"},
        center_tls_id="CENTER",
    )
