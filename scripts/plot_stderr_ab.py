#!/usr/bin/env python3
"""
Plot no-obs vs obs comparison from reconstructed stderr CSV.

Example:
python scripts/plot_stderr_ab.py \
  --csv results/lift_150ep_noobs_vs_obs_from_stderr.csv \
  --out results/lift_150ep_noobs_vs_obs_from_stderr.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot A/B training curves from stderr-reconstructed CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/lift_150ep_noobs_vs_obs_from_stderr.csv"),
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/lift_150ep_noobs_vs_obs_from_stderr.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Lift 150-epoch",
        help="Plot title",
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default="epoch",
        help="X axis label",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="mean train loss",
        help="Y axis label",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epochs, noobs, obs = [], [], []

    with args.csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = row["epoch"]
            a = row["noobs_mean_train_loss_from_stderr"]
            b = row["obs_mean_train_loss_from_stderr"]
            if e and a and b:
                epochs.append(int(e))
                noobs.append(float(a))
                obs.append(float(b))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, noobs, label="no_obs")
    plt.plot(epochs, obs, label="obs")
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
