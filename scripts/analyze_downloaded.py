#!/usr/bin/env python3
"""
Analyze ChromHMM state lengths from BED files and plot distributions.

Usage:
    python analyze_downloaded.py [--dir <data_dir>]

Expects subdirectories 15state/ and 18state/ with .bed.gz files.
Produces:
  - plots/state_length_violin_15state.png   per-state violin plots for 15-state
  - plots/state_length_violin_18state.png   per-state violin plots for 18-state
  - plots/overall_per_sample_15state.png    overall length distribution per sample
  - plots/overall_per_sample_18state.png    overall length distribution per sample
  - stats_15state.tsv / stats_18state.tsv   summary statistics
"""

import argparse
import sys
from pathlib import Path

from analyze import (
    load_bed_df,
    plot_per_state_violin,
    plot_coverage_per_state,
    plot_overall_per_sample,
    plot_state_heatmap,
    save_stats,
)


def load_model(directory, model_label):
    """Load all BED files from *directory* and return a combined DataFrame."""
    files = sorted(directory.glob("*.bed.gz")) + sorted(directory.glob("*.bed"))
    if not files:
        sys.exit(f"No BED files found in {directory}")
    frames = []
    for f in files:
        sample = f.name.split("_")[0]          # e.g. E003
        print(f"  reading {f.name} …")
        df = load_bed_df(f, sample=sample)
        df["model"] = model_label
        frames.append(df)
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default=".",
                        help="Base directory containing 15state/ and 18state/ sub-dirs")
    args = parser.parse_args()

    base = Path(args.dir).resolve()
    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for model_label, subdir_name in [("15-state", "15state"), ("18-state", "18state")]:
        subdir = base / subdir_name
        if not subdir.is_dir():
            print(f"WARNING: {subdir} not found, skipping {model_label}")
            continue

        print(f"\n=== {model_label} ===")
        df = load_model(subdir, model_label)
        print(f"  total segments: {len(df):,}  |  states: {df['state'].nunique()}  |  samples: {df['sample'].nunique()}")

        tag = model_label.replace("-", "")
        n_samples = df["sample"].nunique()

        plot_per_state_violin(
            df, plots_dir / f"state_length_violin_{tag}.png",
            title=f"{model_label}: segment length distribution per state\n(all {n_samples} samples combined)")
        plot_coverage_per_state(
            df, plots_dir / f"state_coverage_{tag}.png",
            title=f"{model_label}: total genomic coverage per state\n(distribution across {n_samples} samples)")
        plot_overall_per_sample(
            df, plots_dir / f"overall_per_sample_{tag}.png",
            title=f"{model_label}: overall segment length per sample")
        plot_state_heatmap(
            df, plots_dir / f"median_heatmap_{tag}.png",
            title=f"{model_label}: median segment length [log10 bp]")
        save_stats(df, base / f"stats_{tag}.tsv", extra_groupby=["model"])

    print("\nAll done.")


if __name__ == "__main__":
    main()
