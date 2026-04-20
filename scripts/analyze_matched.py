#!/usr/bin/env python3
"""
Analyze ChromHMM state lengths from all *_matched.bed files found
recursively under a root directory.

Usage:
    python analyze_matched.py [--dir ~/data/2026_omni_chromhmm]

Produces in <dir>/plots/:
  - matched_state_length_violin.png   per-state violin (RGB-coloured, all samples)
  - matched_overall_per_sample.png    overall length distribution per sample
  - matched_median_heatmap.png        median segment length heatmap sample × state
  - matched_stats.tsv                 summary statistics
"""

import argparse
import matplotlib
import numpy as np
import pandas as pd
import sys
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── helpers ────────────────────────────────────────────────────────────────────

def rgb_str_to_hex(rgb: str) -> str:
    """Convert BED itemRgb '255,128,0' to matplotlib hex '#FF8000'."""
    try:
        r, g, b = (int(x) for x in rgb.split(","))
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "#888888"


def read_bed(path: Path, sample: str) -> pd.DataFrame:
    """Read a ChromHMM BED file (plain, not gzipped).

    Expects at minimum 4 columns (chrom, start, end, state).
    Column 9 (index 8) is itemRgb when present.
    """
    rows = []
    with open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue
            chrom, start, end, state = parts[0], int(parts[1]), int(parts[2]), parts[3]
            rgb = parts[8] if len(parts) > 8 else "128,128,128"
            rows.append((chrom, start, end, state, end - start, rgb, sample))
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "state", "length", "rgb", "sample"])
    df["length"] = df["length"].astype(np.int32)
    return df


GROUP_DEFAULT = "chromhmm_default_result"
GROUP_OTHER   = "other"


def load_all(root: Path) -> pd.DataFrame:
    """Recursively find all *_matched.bed files and load them.

    Assigns a 'group' column:
      - GROUP_DEFAULT  if 'chromhmm_default_result' appears anywhere in the path
      - GROUP_OTHER    otherwise
    """
    files = sorted(root.rglob("*_matched.bed"))
    if not files:
        sys.exit(f"No *_matched.bed files found under {root}")

    frames = []
    for f in files:
        rel = f.relative_to(root)
        sample = str(rel.parent / rel.name.replace("_matched.bed", ""))
        group = GROUP_DEFAULT if GROUP_DEFAULT in str(rel) else GROUP_OTHER
        print(f"  [{group}]  {rel}")
        df = read_bed(f, sample)
        df["group"] = group
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def state_color_map(df: pd.DataFrame) -> dict:
    """Return {state: hex_color} using the first observed itemRgb per state."""
    return (
        df.groupby("state")["rgb"]
        .first()
        .apply(rgb_str_to_hex)
        .to_dict()
    )


def sorted_states(df: pd.DataFrame) -> list:
    """Sort states by leading numeric prefix, then alphabetically."""
    return sorted(df["state"].unique(),
                  key=lambda s: (s.split("_")[0].lstrip("Ee"), s))


# ── plots ──────────────────────────────────────────────────────────────────────

def _group_label(df: pd.DataFrame) -> str:
    groups = df["group"].unique()
    return groups[0] if len(groups) == 1 else "all groups"


def plot_per_state_violin(df: pd.DataFrame, out_path: Path):
    """Violin plot of log10 segment lengths per state, coloured by itemRgb."""
    states = sorted_states(df)
    colors = state_color_map(df)
    n = len(states)

    fig, ax = plt.subplots(figsize=(max(12, n * 0.8), 6))
    data = [np.log10(df.loc[df["state"] == s, "length"].values + 1) for s in states]
    parts = ax.violinplot(data, positions=range(n), showmedians=True, showextrema=True)

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
    ax.set_ylabel("log10(segment length + 1)  [bp]")
    ax.set_ylim(0, 8)
    ax.set_title(f"Segment length per state — {_group_label(df)}  ({df['sample'].nunique()} samples)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_coverage_per_state(df: pd.DataFrame, out_path: Path):
    """Violin plot of total genomic coverage (bp) per state across samples,
    coloured by itemRgb from the BED file."""
    states = sorted_states(df)
    colors = state_color_map(df)

    coverage = (df.groupby(["sample", "state"])["length"]
                  .sum()
                  .reset_index(name="total_bp"))

    n = len(states)
    fig, ax = plt.subplots(figsize=(max(12, n * 0.8), 6))
    data = [
        np.log10(coverage.loc[coverage["state"] == s, "total_bp"].values + 1)
        for s in states
    ]
    parts = ax.violinplot(data, positions=range(n), showmedians=True, showextrema=True)

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
    ax.set_title(f"Total genomic coverage per state — {_group_label(df)}  ({df['sample'].nunique()} samples)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_overall_per_sample(df: pd.DataFrame, out_path: Path):
    """Violin plot of overall segment lengths per sample."""
    samples = sorted(df["sample"].unique())
    n = len(samples)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))
    data = [np.log10(df.loc[df["sample"] == s, "length"].values + 1) for s in samples]
    parts = ax.violinplot(data, positions=range(n), showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.75)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

    short = ["/".join(s.split("/")[-2:]) if "/" in s else s for s in samples]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Sample")
    ax.set_ylabel("log10(segment length + 1)  [bp]")
    ax.set_title(f"Overall segment length per sample — {_group_label(df)}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_state_heatmap(df: pd.DataFrame, out_path: Path):
    """Heatmap of median segment length (log10) per sample × state."""
    pivot = (df.groupby(["sample", "state"])["length"]
               .median()
               .unstack("state"))
    cols = sorted_states(df)
    cols = [c for c in cols if c in pivot.columns]
    pivot = pivot[cols]

    short_index = ["/".join(s.split("/")[-2:]) if "/" in s else s for s in pivot.index]
    pivot.index = short_index

    fig, ax = plt.subplots(figsize=(max(12, len(cols) * 0.65), max(5, len(pivot) * 0.45)))
    sns.heatmap(np.log10(pivot + 1), ax=ax, cmap="YlOrRd",
                linewidths=0.3, annot=True, fmt=".1f",
                cbar_kws={"label": "log10(median length + 1)"})
    ax.set_title(f"Median segment length [log10 bp] — {_group_label(df)}")
    ax.set_xlabel("Chromatin state")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def save_stats(df: pd.DataFrame, out_path: Path):
    """Save per-sample × per-state summary statistics."""
    stats = (df.groupby(["sample", "state"])["length"]
               .agg(count="count", mean="mean", median="median",
                    std="std",
                    p5=lambda x: np.percentile(x, 5),
                    p95=lambda x: np.percentile(x, 95),
                    total_bp="sum")
               .reset_index())
    stats.to_csv(out_path, sep="\t", index=False, float_format="%.1f")
    print(f"  saved {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="~/data/2025_omni_chromhmm",
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

    groups = [
        (GROUP_DEFAULT, df[df["group"] == GROUP_DEFAULT]),
        (GROUP_OTHER,   df[df["group"] == GROUP_OTHER]),
        ("all",         df),
    ]

    for tag, sub in groups:
        if sub.empty:
            print(f"\nSkipping group '{tag}' (no data)")
            continue
        slug = tag.replace(" ", "_")
        print(f"\n=== group: {tag}  ({sub['sample'].nunique()} samples) ===")
        plot_per_state_violin(sub, plots_dir / f"matched_state_length_violin_{slug}.png")
        plot_coverage_per_state(sub, plots_dir / f"matched_state_coverage_{slug}.png")
        plot_overall_per_sample(sub, plots_dir / f"matched_overall_per_sample_{slug}.png")
        plot_state_heatmap(sub, plots_dir / f"matched_median_heatmap_{slug}.png")
        save_stats(sub, root / f"matched_stats_{slug}.tsv")

    print("\nAll done.")


if __name__ == "__main__":
    main()
