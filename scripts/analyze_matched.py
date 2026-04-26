#!/usr/bin/env python3
"""
Analyze ChromHMM state lengths from all *_matched.bed files found
recursively under a root directory.

Samples are grouped by binarization type (default | omnipeak | homer | reference)
using the structured method key derived from each file path.

Usage:
    python analyze_matched.py [--dir ~/data/2026_omni_chromhmm]

Produces in <dir>/plots/ (one file per group + one for "all"):
  - matched_state_length_violin_{group}.png   per-state violin (RGB-coloured)
  - matched_state_coverage_{group}.png        total coverage per state
  - matched_overall_per_sample_{group}.png    overall length distribution per sample
  - matched_median_heatmap_{group}.png        median segment length heatmap sample × state

Produces in <dir>/:
  - matched_stats_{group}.tsv                 summary statistics per group
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from analyze import (
    load_bed_df,
    plot_per_state_violin,
    plot_coverage_per_state,
    plot_overall_per_sample,
    plot_state_heatmap,
    save_stats,
)
from utils import seg_label, parse_method


def _method_group(label):
    """Group label: binarization type derived from the structured method key."""
    if label.startswith("ENCFF"):
        return "reference"
    try:
        binarization, _, _ = parse_method(label)
        return binarization
    except Exception:
        return "other"


def load_all(root):
    """Recursively find all *_matched.bed files and load them.

    Assigns a 'sample' column (structured method key from seg_label) and a
    'group' column (binarization type: default | omnipeak | homer | reference).
    """
    files = sorted(root.rglob("*_matched.bed"))
    if not files:
        sys.exit(f"No *_matched.bed files found under {root}")

    frames = []
    for f in files:
        rel = f.relative_to(root)
        sample = seg_label(str(f))
        group  = _method_group(sample)
        print(f"  [{group}]  {rel}  →  {sample}")
        df = load_bed_df(f, sample=sample)
        df["group"] = group
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _group_label(df):
    groups = df["group"].unique()
    return groups[0] if len(groups) == 1 else "all groups"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="~/data/2026_omni_chromhmm",
                        help="Root directory to search for *_matched.bed files")
    args = parser.parse_args()

    root = Path(args.dir).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Directory not found: {root}")

    plots_dir = root / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Searching for *_matched.bed files under {root} ...")
    df = load_all(root)
    print(f"\nLoaded {len(df):,} segments | "
          f"{df['state'].nunique()} states | "
          f"{df['sample'].nunique()} samples | "
          f"{df['group'].nunique()} groups")

    all_groups = sorted(df["group"].unique())
    groups = [(g, df[df["group"] == g]) for g in all_groups] + [("all", df)]

    for tag, sub in groups:
        if sub.empty:
            print(f"\nSkipping group '{tag}' (no data)")
            continue
        slug = tag.replace(" ", "_")
        label = _group_label(sub)
        n_samples = sub["sample"].nunique()
        print(f"\n=== group: {tag}  ({n_samples} samples) ===")

        plot_per_state_violin(
            sub, plots_dir / f"matched_state_length_violin_{slug}.png",
            title=f"Segment length per state — {label}  ({n_samples} samples)")
        plot_coverage_per_state(
            sub, plots_dir / f"matched_state_coverage_{slug}.png",
            title=f"Total genomic coverage per state — {label}  ({n_samples} samples)")
        plot_overall_per_sample(
            sub, plots_dir / f"matched_overall_per_sample_{slug}.png",
            title=f"Overall segment length per sample — {label}",
            short_names=True)
        plot_state_heatmap(
            sub, plots_dir / f"matched_median_heatmap_{slug}.png",
            title=f"Median segment length [log10 bp] — {label}",
            short_names=True)
        save_stats(sub, root / f"matched_stats_{slug}.tsv")

    print("\nAll done.")


if __name__ == "__main__":
    main()
