from pathlib import Path
from typing import Optional


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    base_path = Path(start_path) if start_path is not None else Path(__file__)
    current = base_path.resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current] + list(current.parents):
        if (candidate / ".git").is_dir():
            return candidate
        if (candidate / "pyproject.toml").exists():
            return candidate

    return current
