from __future__ import annotations

from pathlib import Path
from typing import List, Union
import xml.etree.ElementTree as ET


def extract_tls_ids(net_file: Union[str, Path]) -> List[str]:
    """
    Extract traffic light IDs from a SUMO .net.xml file.

    The IDs are returned in the order they appear in the file (first occurrence wins)
    with duplicates removed. Raises a clear error when the file is missing or when
    no <tlLogic> elements are present.
    """
    net_path = Path(net_file)
    if not net_path.exists():
        raise FileNotFoundError(f"Network file not found: {net_path}")

    try:
        tree = ET.parse(net_path)
    except Exception as exc:  # pragma: no cover - XML parsing errors are surfaced
        raise ValueError(f"Failed to parse SUMO network XML: {net_path}") from exc

    root = tree.getroot()

    tls_ids: List[str] = []
    seen = set()
    for tl in root.findall(".//tlLogic"):
        tls_id = tl.get("id")
        if not tls_id:
            continue
        if tls_id in seen:
            continue
        seen.add(tls_id)
        tls_ids.append(tls_id)

    if len(tls_ids) == 0:
        raise ValueError(f"No <tlLogic> elements found in network: {net_path}")

    return tls_ids
