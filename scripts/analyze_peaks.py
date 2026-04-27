#!/usr/bin/env python3
"""
Peak-file analysis: number of peaks, mean/median length, and replicate Jaccard.

Inputs:
  --ds          dataset root directory (e.g. imr90/)
  --marks       space-separated histone mark names
  --omni-bin    OmniPeak binarization bin size (default: 100)
  --outdir      output directory (e.g. imr90/peaks)

Peak files expected:
  OmniPeak   : {ds}/[rep{1,2}/]omni/{mark}_{omni_bin}.peak  (cols 1-3)
  HOMER       : {ds}/[rep{1,2}/]homer/{mark}.bed              (cols 1-3)
  ChromHMM    : {ds}/[rep{1,2}/]chromhmm_default/{cell}_{chrom}_binary.txt
                  → consecutive 1-bins (200 bp each) per mark are merged

Outputs (all under --outdir):
  peak_stats.tsv          tab-sep table of all metrics
  n_peaks.png             grouped bar chart: number of peaks per mark
  mean_length.png         grouped bar chart: mean peak length per mark
  median_length.png       grouped bar chart: median peak length per mark
  jaccard_rep1_vs_rep2.png grouped bar chart: Jaccard between replicates per mark
"""

import argparse
import glob
import gzip
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_bed_regions(path):
    """Return list of (chrom, start, end) from a BED/peak file (cols 0-2)."""
    regions = []
    opener = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    with opener as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                regions.append((parts[0], int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    return regions


def load_chromhmm_binary_peaks(binary_dir, cell, bin_size=200):
    """
    Convert ChromHMM binary text files to per-mark peak regions by merging
    consecutive 1-bins.

    Returns dict: mark -> list of (chrom, start, end)
    """
    mark_regions = defaultdict(list)
    pattern = os.path.join(binary_dir, f"{cell}_chr*_binary.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        # try without cell prefix
        pattern = os.path.join(binary_dir, "*_binary.txt")
        files = sorted(glob.glob(pattern))

    for path in files:
        with open(path) as fh:
            # line 1: cell  chrom
            header = fh.readline().rstrip("\n").split("\t")
            chrom = header[1] if len(header) > 1 else "chrUnk"
            # line 2: mark names
            marks = fh.readline().rstrip("\n").split("\t")
            # remaining lines: binary rows
            bin_idx = 0
            for line in fh:
                vals = line.rstrip("\n").split("\t")
                for i, v in enumerate(vals):
                    if i >= len(marks):
                        break
                    if v == "1":
                        mark_regions[marks[i]].append(
                            (chrom, bin_idx * bin_size, (bin_idx + 1) * bin_size)
                        )
                bin_idx += 1

    # merge consecutive / overlapping bins per mark
    result = {}
    for mark, raw in mark_regions.items():
        result[mark] = _merge_regions(raw)
    return result


def _merge_regions(regions):
    """Merge overlapping/adjacent (chrom, start, end) regions."""
    if not regions:
        return []
    by_chrom = defaultdict(list)
    for chrom, s, e in regions:
        by_chrom[chrom].append((s, e))
    merged = []
    for chrom, ivs in by_chrom.items():
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((chrom, cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((chrom, cur_s, cur_e))
    return merged


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def peak_stats(regions):
    """Return dict with n_peaks, mean_length, median_length."""
    if not regions:
        return {"n_peaks": 0, "mean_length": 0.0, "median_length": 0.0}
    lengths = [e - s for _, s, e in regions]
    return {
        "n_peaks": len(lengths),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
    }


def jaccard(a, b):
    """Compute Jaccard similarity (intersection_bp / union_bp) between two region lists."""
    def to_dict(regions):
        d = defaultdict(list)
        for chrom, s, e in regions:
            d[chrom].append((s, e))
        for chrom in d:
            d[chrom].sort()
        return d

    def bp_overlap(da, db):
        total = 0
        for chrom in da:
            if chrom not in db:
                continue
            ia, ib = 0, 0
            la, lb = da[chrom], db[chrom]
            while ia < len(la) and ib < len(lb):
                s = max(la[ia][0], lb[ib][0])
                e = min(la[ia][1], lb[ib][1])
                if s < e:
                    total += e - s
                if la[ia][1] <= lb[ib][1]:
                    ia += 1
                else:
                    ib += 1
        return total

    da, db = to_dict(_merge_regions(a)), to_dict(_merge_regions(b))
    inter = bp_overlap(da, db)
    sum_a = sum(e - s for _, s, e in a)
    sum_b = sum(e - s for _, s, e in b)
    union = sum_a + sum_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = {
    "OmniPeak": "#2196F3",
    "HOMER":    "#FF9800",
    "MACS2":    "#4CAF50",
    "Default":  "#9E9E9E",
}
METHOD_ORDER = ["OmniPeak", "HOMER", "MACS2", "Default"]


def _bar_plot(df, value_col, ylabel, title, outpath):
    marks = sorted(df["mark"].unique())
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    x = np.arange(len(marks))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(marks) * 1.2), 4))
    for i, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("mark")
        vals = [sub.loc[m, value_col] if m in sub.index else 0 for m in marks]
        ax.bar(x + i * width, vals, width, label=method,
               color=PALETTE.get(method, "#555555"))
    ax.set_xticks(x + width)
    ax.set_xticklabels(marks, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="Dataset root directory")
    ap.add_argument("--cell", required=True, help="Cell name (for binary file glob)")
    ap.add_argument("--marks", nargs="+", required=True)
    ap.add_argument("--omni-bin", type=int, default=100)
    ap.add_argument("--chromhmm-bin", type=int, default=200)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    marks = args.marks
    has_replicates = os.path.isdir(os.path.join(args.ds, "rep1"))

    # -----------------------------------------------------------------------
    # Collect regions per method / mark / replicate
    # -----------------------------------------------------------------------
    # Structure: regions[method][mark][folder_key] = list of (chrom, s, e)
    regions = {m: {mk: {} for mk in marks} for m in ["OmniPeak", "HOMER", "MACS2", "Default"]}

    folders = {"pooled": args.ds}
    if has_replicates:
        folders["rep1"] = os.path.join(args.ds, "rep1")
        folders["rep2"] = os.path.join(args.ds, "rep2")

    for folder_key, folder_path in folders.items():
        # OmniPeak
        for mark in marks:
            path = os.path.join(folder_path, "omni", f"{mark}_{args.omni_bin}.peak")
            if os.path.exists(path):
                regions["OmniPeak"][mark][folder_key] = load_bed_regions(path)
            else:
                print(f"  missing: {path}", file=sys.stderr)

        # HOMER
        for mark in marks:
            path = os.path.join(folder_path, "homer", f"{mark}.bed")
            if os.path.exists(path):
                regions["HOMER"][mark][folder_key] = load_bed_regions(path)
            else:
                print(f"  missing: {path}", file=sys.stderr)

        # MACS2
        for mark in marks:
            path = os.path.join(folder_path, "macs2", f"{mark}.bed")
            if os.path.exists(path):
                regions["MACS2"][mark][folder_key] = load_bed_regions(path)
            else:
                print(f"  missing: {path}", file=sys.stderr)

        # ChromHMM default — parse binary files
        binary_dir = os.path.join(folder_path, "chromhmm_default")
        if os.path.isdir(binary_dir):
            chrom_peaks = load_chromhmm_binary_peaks(binary_dir, args.cell,
                                                     bin_size=args.chromhmm_bin)
            for mark in marks:
                if mark in chrom_peaks:
                    regions["Default"][mark][folder_key] = chrom_peaks[mark]
                else:
                    print(f"  missing chromhmm mark {mark} in {binary_dir}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Build stats table
    # -----------------------------------------------------------------------
    rows = []
    for method in ["OmniPeak", "HOMER", "MACS2", "Default"]:
        for mark in marks:
            row = {"method": method, "mark": mark}
            # pooled stats
            pooled = regions[method][mark].get("pooled", [])
            s = peak_stats(pooled)
            row["n_peaks"] = s["n_peaks"]
            row["mean_length"] = s["mean_length"]
            row["median_length"] = s["median_length"]
            # per-replicate stats
            for rep in ["rep1", "rep2"]:
                rep_r = regions[method][mark].get(rep, [])
                rs = peak_stats(rep_r)
                row[f"n_peaks_{rep}"] = rs["n_peaks"]
                row[f"mean_length_{rep}"] = rs["mean_length"]
            # Jaccard between replicates
            if has_replicates:
                r1 = regions[method][mark].get("rep1", [])
                r2 = regions[method][mark].get("rep2", [])
                row["jaccard_rep1_vs_rep2"] = jaccard(r1, r2) if (r1 and r2) else None
            rows.append(row)

    df = pd.DataFrame(rows)
    stats_path = os.path.join(args.outdir, "peak_stats.tsv")
    df.to_csv(stats_path, sep="\t", index=False, float_format="%.4f")
    print(f"Saved {stats_path}", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Plots (pooled stats)
    # -----------------------------------------------------------------------
    _bar_plot(df, "n_peaks", "Number of peaks", "Peaks per mark (pooled)",
              os.path.join(args.outdir, "n_peaks.png"))
    _bar_plot(df, "mean_length", "Mean length (bp)", "Mean peak length (pooled)",
              os.path.join(args.outdir, "mean_length.png"))
    _bar_plot(df, "median_length", "Median length (bp)", "Median peak length (pooled)",
              os.path.join(args.outdir, "median_length.png"))

    if has_replicates and "jaccard_rep1_vs_rep2" in df.columns:
        _bar_plot(df, "jaccard_rep1_vs_rep2",
                  "Jaccard (rep1 vs rep2)", "Peak Jaccard: rep1 vs rep2",
                  os.path.join(args.outdir, "jaccard_rep1_vs_rep2.png"))

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
