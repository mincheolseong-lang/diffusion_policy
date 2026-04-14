#!/usr/bin/env python3
"""
Plot one or more diffusion_policy JsonLogger outputs (logs.json.txt).

Layouts:
  --triple   three panels (horizontal): train loss | val loss | test success  (compare runs)
  --split    two stacked panels: train loss + success
  (default)  single axes + twin y: train loss + success

Example (object XYZ in obs vs not — short legend: no_pos / pos):
  python scripts/plot_training_log.py \\
    data/outputs/.../logs.json.txt data/outputs/.../logs.json.txt \\
    --labels no_pos pos --title "Lift" -o results/compare.png \\
    --triple --success-key test/mean_score --max-epoch 150 --no-show
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import pandas as pd


def _load_log(path: pathlib.Path, max_epoch: int | None) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    if max_epoch is not None and "epoch" in df.columns:
        df = df[df["epoch"] <= max_epoch]
    return df


def _series_per_epoch(df: pd.DataFrame, col: str) -> pd.DataFrame | None:
    if col not in df.columns:
        return None
    sub = df.dropna(subset=[col])
    if sub.empty:
        return None
    g = sub.groupby("epoch", as_index=False)[col].mean()
    return g


def main() -> int:
    p = argparse.ArgumentParser(description="Plot training logs.json.txt (JsonLogger format)")
    p.add_argument("logs", nargs="+", type=pathlib.Path, help="paths to logs.json.txt")
    p.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="one legend label per log (e.g. no_pos pos — vision-only vs +object XYZ)",
    )
    p.add_argument("--title", default="", help="figure suptitle")
    p.add_argument("-o", "--output", type=pathlib.Path, required=True, help="output .png path")
    layout = p.add_mutually_exclusive_group()
    layout.add_argument(
        "--triple",
        action="store_true",
        help="1×3 subplots: training loss | validation loss | test success (per-run colors)",
    )
    layout.add_argument(
        "--split",
        action="store_true",
        help="2×1 subplots: train loss + success metric",
    )
    p.add_argument(
        "--success-key",
        default="test/mean_score",
        help="column for rollout success (e.g. test/mean_score)",
    )
    p.add_argument(
        "--val-key",
        default="val_loss",
        help="column for validation loss (default val_loss)",
    )
    p.add_argument("--max-epoch", type=int, default=None, help="keep rows with epoch <= this")
    p.add_argument("--no-show", action="store_true", help="do not call plt.show()")
    args = p.parse_args()

    if len(args.logs) != len(args.labels):
        print(
            f"error: got {len(args.logs)} log file(s) but {len(args.labels)} label(s)",
            file=sys.stderr,
        )
        return 1

    runs: list[tuple[str, pd.DataFrame]] = []
    for path, label in zip(args.logs, args.labels):
        if not path.is_file():
            print(f"error: missing log file: {path}", file=sys.stderr)
            return 1
        runs.append((label, _load_log(path, args.max_epoch)))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.triple:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, constrained_layout=True)
        ax_tr, ax_va, ax_ts = axes[0], axes[1], axes[2]
    elif args.split:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        ax_tr, ax_va, ax_ts = axes[0], None, axes[1]
    else:
        fig, ax_tr = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax_va = None
        ax_ts = ax_tr.twinx()

    for i, (label, df) in enumerate(runs):
        if "epoch" not in df.columns:
            print(f"warning: {label}: missing epoch column", file=sys.stderr)
            continue
        color = f"C{i}"

        if "train_loss" in df.columns:
            g = df.groupby("epoch", as_index=False)["train_loss"].mean()
            ax_tr.plot(g["epoch"], g["train_loss"], label=label, color=color, linewidth=1.2)
        else:
            print(f"warning: {label}: no train_loss", file=sys.stderr)

        if args.triple or args.split:
            if ax_va is not None:
                vg = _series_per_epoch(df, args.val_key)
                if vg is not None:
                    ax_va.plot(vg["epoch"], vg[args.val_key], label=label, color=color, linewidth=1.2)
                else:
                    print(
                        f"warning: {label}: no {args.val_key!r} (skip validation curve)",
                        file=sys.stderr,
                    )

        sk = args.success_key
        sg = _series_per_epoch(df, sk)
        if sg is not None:
            if args.triple:
                ax_ts.plot(
                    sg["epoch"],
                    sg[sk],
                    label=label,
                    color=color,
                    marker="o",
                    markersize=4,
                    linestyle="-",
                    linewidth=1.0,
                )
            elif args.split:
                ax_ts.plot(
                    sg["epoch"],
                    sg[sk],
                    label=label,
                    color=color,
                    marker="o",
                    markersize=3,
                    linestyle="--",
                    linewidth=1.0,
                )
            else:
                ax_ts.plot(
                    sg["epoch"],
                    sg[sk],
                    label=label,
                    color=color,
                    marker="o",
                    markersize=3,
                    linestyle="--",
                    linewidth=1.0,
                )
        else:
            print(f"warning: {label}: no {sk!r} (skip success curve)", file=sys.stderr)

    ax_tr.set_ylabel("Loss (MSE)")
    ax_tr.set_title("Training Loss")
    ax_tr.grid(True, alpha=0.3)
    ax_tr.legend(loc="upper right")

    if args.triple:
        ax_va.set_ylabel("Loss (MSE)")
        ax_va.set_title("Validation Loss")
        ax_va.grid(True, alpha=0.3)
        ax_va.legend(loc="upper right")

        ax_ts.set_ylabel("Success Rate")
        ax_ts.set_title("Test Success Rate")
        ax_ts.set_ylim(-0.05, 1.05)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(loc="center right")
        fig.supxlabel("Epoch")
    elif args.split:
        ax_tr.set_ylabel("train_loss (mean / epoch)")
        ax_ts.set_ylabel(args.success_key)
        ax_ts.set_xlabel("epoch")
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(loc="upper right")
    else:
        ax_tr.set_ylabel("train_loss (mean / epoch)")
        ax_ts.set_ylabel(args.success_key)
        ax_ts.legend(loc="upper left")

    if args.title:
        fig.suptitle(args.title)

    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
