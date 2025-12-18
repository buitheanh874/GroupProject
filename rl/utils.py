from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def set_global_seed(seed: int) -> None:
    seed_value = int(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def resolve_device(device_str: str) -> torch.device:
    device_name = str(device_str).strip().lower()
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    return torch.device("cpu")


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(str(path), "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a dict")
    return data


def save_yaml_config(data: Dict[str, Any], path: str) -> None:
    with open(str(path), "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def ensure_dir(path: str) -> str:
    os.makedirs(str(path), exist_ok=True)
    return str(path)


def generate_run_id(prefix: Optional[str] = None) -> str:
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if prefix is None or str(prefix).strip() == "":
        return str(now)
    return f"{str(prefix)}_{now}"


def linear_epsilon(global_step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    step = int(max(0, global_step))
    total = int(max(1, decay_steps))

    if step >= total:
        return float(eps_end)

    fraction = float(step) / float(total)
    value = float(eps_start) + fraction * (float(eps_end) - float(eps_start))
    return float(value)
