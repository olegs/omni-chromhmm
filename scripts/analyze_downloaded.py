#!/usr/bin/env python3
"""
Analyze ChromHMM state lengths from BED files and plot distributions.

Usage:
    python analyze_downloaded.py [--dir <data_dir>]

Expects subdirectories 15state/ and 18state/ with .bed.gz files.
Produces:
  - plots/state_length_dist_15state.png   per-state violin plots for 15-state
  - plots/state_length_dist_18state.png   per-state violin plots for 18-state
  - plots/overall_dist_15state.png        overall length distribution per sample
  - plots/overall_dist_18state.png        overall length distribution per sample
  - stats_15state.tsv / stats_18state.tsv summary statistics
"""

import argparse
import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── helpers ────────────────────────────────────────────────────────────────────

def rgb_str_to_hex(rgb: str) -> str:
    """Convert BED itemRgb '255,128,0' to matplotlib hex color '#FF8000'."""
    try:
        r, g, b = (int(x) for x in rgb.split(","))
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "#888888"


def read_bed_gz(path: Path) -> pd.DataFrame:
    """Read a (possibly gzipped) ChromHMM BED file into a DataFrame.

    Parses column 9 (itemRgb, 0-based index 8) when present.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue
            chrom, start, end, state = parts[0], int(parts[1]), int(parts[2]), parts[3]
            rgb = parts[8] if len(parts) > 8 else "128,128,128"
            rows.append((chrom, start, end, state, end - start, rgb))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "state", "length", "rgb"])


def state_color_map(df: pd.DataFrame) -> dict:
    """Return {state: hex_color} using the first observed itemRgb per state."""
    return (
        df.groupby("state")["rgb"]
        .first()
        .apply(rgb_str_to_hex)
        .to_dict()
    )


def load_model(directory: Path, model_label: str) -> pd.DataFrame:
    """Load all BED files from *directory* and return a combined DataFrame."""
    files = sorted(directory.glob("*.bed.gz")) + sorted(directory.glob("*.bed"))
    if not files:
        sys.exit(f"No BED files found in {directory}")
    frames = []
    for f in files:
        sample = f.name.split("_")[0]          # e.g. E003
        print(f"  reading {f.name} …")
        df = read_bed_gz(f)
        df["sample"] = sample
        df["model"] = model_label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── plotting helpers ────────────────────────────────────────────────────────────

def _log_yticks(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


def plot_per_state_violin(df: pd.DataFrame, model_label: str, out_path: Path):
    """Violin plot of state lengths per chromatin state (all samples combined),
    coloured by the itemRgb field from the BED file."""
    states = sorted(df["state"].unique(),
                    key=lambda s: (s.split("_")[0].lstrip("Ee"), s))
    colors = state_color_map(df)
    n = len(states)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.7), 6))

    data_by_state = [np.log10(df.loc[df["state"] == s, "length"].values + 1) for s in states]
    parts = ax.violinplot(data_by_state, positions=range(n),
                          showmedians=True, showextrema=True)

    for i, (pc, state) in enumerate(zip(parts["bodies"], states)):
        pc.set_facecolor(colors.get(state, "#888888"))
        pc.set_alpha(0.85)

    # Style median and extrema lines to match body color
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

    ax.set_xticks(range(n))
    ax.set_xticklabels(states, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Chromatin state")
    ax.set_ylabel("log10(segment length + 1)  [bp]")
    ax.set_ylim(0, 8)
    ax.set_title(f"{model_label}: segment length distribution per state\n(all {df['sample'].nunique()} samples combined)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_coverage_per_state(df: pd.DataFrame, model_label: str, out_path: Path):
    """Violin plot of total genomic coverage (bp) per state across samples,
    coloured by itemRgb from the BED file."""
    states = sorted(df["state"].unique(),
                    key=lambda s: (s.split("_")[0].lstrip("Ee"), s))
    colors = state_color_map(df)

    # total bp per (sample, state)
    coverage = (df.groupby(["sample", "state"])["length"]
                  .sum()
                  .reset_index(name="total_bp"))

    n = len(states)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.7), 6))

    data_by_state = [
        np.log10(coverage.loc[coverage["state"] == s, "total_bp"].values + 1)
        for s in states
    ]
    parts = ax.violinplot(data_by_state, positions=range(n),
                          showmedians=True, showextrema=True)

    for pc, state in zip(parts["bodies"], states):
        pc.set_facecolor(colors.get(state, "#888888"))
        pc.set_alpha(0.85)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

    ax.set_xticks(range(n))
    ax.set_xticklabels(states, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Chromatin state")
    ax.set_ylabel("log10(total coverage + 1)  [bp]")
    ax.set_ylim(2, 10)
    ax.set_title(f"{model_label}: total genomic coverage per state\n"
                 f"(distribution across {df['sample'].nunique()} samples)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_overall_per_sample(df: pd.DataFrame, model_label: str, out_path: Path):
    """Box/violin plot of overall segment lengths per sample."""
    samples = sorted(df["sample"].unique())
    fig, ax = plt.subplots(figsize=(max(10, len(samples) * 0.9), 5))

    data_by_sample = [np.log10(df.loc[df["sample"] == s, "length"].values + 1) for s in samples]
    parts = ax.violinplot(data_by_sample, positions=range(len(samples)),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.75)

    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels(samples, rotation=45, ha="right")
    ax.set_xlabel("Sample")
    ax.set_ylabel("log10(segment length + 1)  [bp]")
    ax.set_title(f"{model_label}: overall segment length per sample")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_state_heatmap(df: pd.DataFrame, model_label: str, out_path: Path):
    """Heatmap: median segment length (log10) per sample × state."""
    pivot = (df.groupby(["sample", "state"])["length"]
               .median()
               .unstack("state"))
    # sort states naturally
    cols = sorted(pivot.columns,
                  key=lambda s: (s.split("_")[0].lstrip("Ee"), s))
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(max(12, len(cols) * 0.65), max(5, len(pivot) * 0.5)))
    sns.heatmap(np.log10(pivot + 1), ax=ax, cmap="YlOrRd",
                linewidths=0.3, annot=True, fmt=".1f",
                cbar_kws={"label": "log10(median length + 1)"})
    ax.set_title(f"{model_label}: median segment length [log10 bp]")
    ax.set_xlabel("Chromatin state")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def save_stats(df: pd.DataFrame, out_path: Path):
    """Save per-sample × per-state summary statistics."""
    stats = (df.groupby(["model", "sample", "state"])["length"]
               .agg(count="count", mean="mean", median="median",
                    std="std", p5=lambda x: np.percentile(x, 5),
                    p95=lambda x: np.percentile(x, 95),
                    total_bp="sum")
               .reset_index())
    stats.to_csv(out_path, sep="\t", index=False, float_format="%.1f")
    print(f"  saved {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=".",
                        help="Base directory containing 15state/ and 18state/ sub-dirs")
    args = parser.parse_args()

    base = Path(args.dir).resolve()
    plots_dir = base / "plots"
    plots_dir.mkdir(exist_ok=True)

    for model_label, subdir_name in [("15-state", "15state"), ("18-state", "18state")]:
        subdir = base / subdir_name
        if not subdir.is_dir():
            print(f"WARNING: {subdir} not found, skipping {model_label}")
            continue

        print(f"\n=== {model_label} ===")
        df = load_model(subdir, model_label)
        print(f"  total segments: {len(df):,}  |  states: {df['state'].nunique()}  |  samples: {df['sample'].nunique()}")

        tag = model_label.replace("-", "")
        plot_per_state_violin(df, model_label, plots_dir / f"state_length_violin_{tag}.png")
        plot_coverage_per_state(df, model_label, plots_dir / f"state_coverage_{tag}.png")
        plot_overall_per_sample(df, model_label, plots_dir / f"overall_per_sample_{tag}.png")
        plot_state_heatmap(df, model_label, plots_dir / f"median_heatmap_{tag}.png")
        save_stats(df, base / f"stats_{tag}.tsv")

    print("\nAll done.")


if __name__ == "__main__":
    main()
