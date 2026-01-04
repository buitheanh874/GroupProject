from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List


def _contains_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in ["*", "?", "["])


def _resolve_path(path_str: str, base_dir: Path, project_root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path) if base_dir else (project_root / path)
    return path.resolve()


def _load_manifest(manifest_path: Path, project_root: Path) -> List[str]:
    manifest_path = _resolve_path(str(manifest_path), project_root, project_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Route pool manifest not found: {manifest_path}")

    routes: List[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        route_path = _resolve_path(stripped, manifest_path.parent, project_root)
        if not route_path.exists():
            raise FileNotFoundError(f"Route file from manifest missing: {route_path} (manifest: {manifest_path})")
        routes.append(str(route_path))
    if len(routes) == 0:
        raise ValueError(f"Route pool manifest is empty after filtering comments/blank lines: {manifest_path}")
    return routes


def _expand_route_pool(entries: List[Any], project_root: Path) -> List[str]:
    expanded: List[str] = []
    for entry in entries:
        path_str = str(entry)
        if _contains_glob(path_str):
            pattern = _resolve_path(path_str, project_root, project_root)
            matches = sorted(Path(p).resolve() for p in glob.glob(str(pattern)))
            if len(matches) == 0:
                raise FileNotFoundError(f"No route files matched pattern: {path_str}")
            expanded.extend(str(p) for p in matches)
        else:
            path = _resolve_path(path_str, project_root, project_root)
            if not path.exists():
                raise FileNotFoundError(f"Route file not found: {path}")
            expanded.append(str(path))
    return expanded


def load_route_pool_from_config(config: Dict[str, Any], split: str, project_root: Path) -> List[str]:
    """
    Load route pool from config for a given split ("train" or "eval").

    Priority:
        1) <split>.route_pool_manifest
        2) <split>.route_pool (with runtime glob expansion)
    Returns the resolved route list (absolute paths) and mutates config['env']['sumo']['route_pool'].
    """
    split_cfg = config.get(split, {})
    manifest = split_cfg.get("route_pool_manifest")
    route_pool_entries = split_cfg.get("route_pool", [])

    routes: List[str] = []
    if manifest:
        routes = _load_manifest(Path(manifest), project_root)
    elif route_pool_entries:
        routes = _expand_route_pool(list(route_pool_entries), project_root)

    if routes:
        config.setdefault("env", {}).setdefault("sumo", {})["route_pool"] = list(routes)

    return routes
