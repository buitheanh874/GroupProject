from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _choose_column(df: pd.DataFrame, candidates: Iterable[str], name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Required column for {name} not found; tried {list(candidates)}")


def _summary_by_controller(df: pd.DataFrame, column: str) -> Tuple[List[str], List[float], List[float]]:
    grouped = df.groupby("controller")[column].agg(["mean", "std"]).reset_index()
    controllers = grouped["controller"].astype(str).tolist()
    means = grouped["mean"].astype(float).tolist()
    stds = grouped["std"].fillna(0.0).astype(float).tolist()
    return controllers, means, stds


def _plot_bar(out_path: str, title: str, y_label: str, controllers: List[str], means: List[float], stds: List[float]) -> None:
    x_pos = list(range(len(controllers)))
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(controllers)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for rect, mean in zip(bars, means):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to eval CSV")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for figures")
    args = parser.parse_args()

    outdir = _ensure_outdir(args.outdir)

    df = pd.read_csv(args.input)
    if "controller" not in df.columns:
        raise ValueError("CSV must contain a 'controller' column")

    avg_col = _choose_column(df, ["avg_wait_time", "avg_wait"], "avg_wait_time")
    max_col: Optional[str] = None
    try:
        max_col = _choose_column(df, ["max_wait_time", "p95_wait_time"], "max_wait_time")
    except ValueError:
        pass
    arrived_col = _choose_column(df, ["arrived_vehicles", "arrived"], "arrived_vehicles")

    controllers, avg_means, avg_stds = _summary_by_controller(df, avg_col)
    _plot_bar(
        out_path=os.path.join(outdir, "avg_wait.png"),
        title="Average Wait Time by Controller",
        y_label="Average Wait Time",
        controllers=controllers,
        means=avg_means,
        stds=avg_stds,
    )

    if max_col is not None:
        controllers_m, max_means, max_stds = _summary_by_controller(df, max_col)
        _plot_bar(
            out_path=os.path.join(outdir, "max_wait.png"),
            title="Tail Wait Time by Controller",
            y_label=max_col,
            controllers=controllers_m,
            means=max_means,
            stds=max_stds,
        )

    controllers_a, arr_means, arr_stds = _summary_by_controller(df, arrived_col)
    _plot_bar(
        out_path=os.path.join(outdir, "arrived.png"),
        title="Arrived Vehicles by Controller",
        y_label="Arrived Vehicles",
        controllers=controllers_a,
        means=arr_means,
        stds=arr_stds,
    )


if __name__ == "__main__":
    main()
