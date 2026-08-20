#!/usr/bin/env python3
"""Peak-file analysis: number of peaks, mean/median length, replicate Jaccard.

Peak files expected under {ds}/[rep{1,2}/]:
  OmniPeak : omni/{mark}_{omni_bin}.peak
  HOMER    : homer/{mark}.bed
  MACS2    : macs2/{mark}.bed
  ChromHMM : chromhmm_default/{cell}_{chrom}_binary.txt — consecutive 1-bins
             per mark are merged into peaks
"""

import glob
import gzip
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import BIN_COLORS, save_fig


def load_bed_regions(path):
    """Return list of (chrom, start, end) from a BED/peak file (cols 0-2)."""
    regions = []
    try:
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
    except (EOFError, gzip.BadGzipFile) as e:
        print(f"Warning: Corrupted gzip file {path}: {e}. Data might be incomplete.", file=sys.stderr)
    return regions


def load_chromhmm_binary_peaks(binary_dir, cell, bin_size=200):
    """Convert ChromHMM binary text files to {mark: [(chrom, start, end), ...]}
    by merging consecutive 1-bins.
    """
    mark_regions = defaultdict(list)
    pattern = os.path.join(binary_dir, f"{cell}_chr*_binary.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        # Fall back to files without the cell prefix.
        pattern = os.path.join(binary_dir, "*_binary.txt")
        files = sorted(glob.glob(pattern))

    for path in files:
        with open(path) as fh:
            header = fh.readline().rstrip("\n").split("\t")   # cell, chrom
            chrom = header[1] if len(header) > 1 else "chrUnk"
            marks = fh.readline().rstrip("\n").split("\t")
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

    result = {}
    for mark, raw in mark_regions.items():
        result[mark] = _merge_regions(raw)
    return result


def _merge_regions(regions):
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


def peak_stats(regions):
    if not regions:
        return {"n_peaks": 0, "mean_length": 0.0, "median_length": 0.0}
    lengths = [e - s for _, s, e in regions]
    return {
        "n_peaks": len(lengths),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
    }


def jaccard(a, b):
    """Jaccard similarity (intersection_bp / union_bp) of two region lists."""
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


PALETTE = {
    "OmniPeak": BIN_COLORS["omnipeak"],
    "HOMER":    BIN_COLORS["homer"],
    "MACS2":    BIN_COLORS["macs2"],
    "Default":  BIN_COLORS["default"],
}
METHOD_ORDER = ["Default", "HOMER", "MACS2", "OmniPeak"]


def _bar_plot(df, value_col, ylabel, title, outpath):
    marks = sorted(df["mark"].unique())
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    
    fig, ax = plt.subplots(figsize=(max(5, len(marks) * 0.8), 4.2))
    
    sns.barplot(data=df, x="mark", y=value_col, hue="method",
                order=marks, hue_order=methods, palette=PALETTE,
                ax=ax, edgecolor="lightgrey", linewidth=1)

    ax.set_xticks(range(len(marks)))
    ax.set_xticklabels(marks, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Method", fontsize=8, title_fontsize=9)

    save_fig(fig, outpath)


def run_analyze_peaks(ds, cell, marks, outdir, omni_bin=100, chromhmm_bin=200):
    """Per-mark peak statistics and replicate Jaccard for all callers.

    Writes peak_stats.tsv and bar plots under *outdir*; called from analysis.ipynb.
    """
    args = SimpleNamespace(ds=ds, cell=cell, marks=marks, outdir=outdir,
                           omni_bin=omni_bin, chromhmm_bin=chromhmm_bin)

    os.makedirs(args.outdir, exist_ok=True)

    marks = args.marks
    has_replicates = os.path.isdir(os.path.join(args.ds, "rep1")) or os.path.isdir(os.path.join(args.ds, "replicate1"))

    # regions[method][mark][folder_key] = [(chrom, s, e), ...]
    regions = {m: {mk: {} for mk in marks} for m in ["OmniPeak", "HOMER", "MACS2", "Default"]}

    folders = {"pooled": args.ds}
    if has_replicates:
        folders["rep1"] = os.path.join(args.ds, "rep1") if os.path.isdir(os.path.join(args.ds, "rep1")) else os.path.join(args.ds, "replicate1")
        folders["rep2"] = os.path.join(args.ds, "rep2") if os.path.isdir(os.path.join(args.ds, "rep2")) else os.path.join(args.ds, "replicate2")

    for folder_key, folder_path in folders.items():
        # OmniPeak
        for mark in marks:
            path = os.path.join(folder_path, "omni", f"{mark}_{args.omni_bin}.peak")
            if not os.path.exists(path):
                p_cand = os.path.join(folder_path, "omni", f"{args.cell}-{mark}.peak")
                if os.path.exists(p_cand):
                    path = p_cand
                else:
                    cand = [f for f in glob.glob(os.path.join(folder_path, "omni", f"*{mark}*.peak"))
                            if not f.endswith(".txt") and not f.endswith(".idx")]
                    if cand:
                        path = cand[0]
            if os.path.exists(path):
                regions["OmniPeak"][mark][folder_key] = load_bed_regions(path)
            else:
                print(f"  missing: {path}", file=sys.stderr)

        # HOMER
        for mark in marks:
            path = os.path.join(folder_path, "homer", f"{mark}.bed")
            if not os.path.exists(path):
                p_cand = os.path.join(folder_path, "homer", f"{args.cell}-{mark}_homer.bed")
                if os.path.exists(p_cand):
                    path = p_cand
                else:
                    cand = glob.glob(os.path.join(folder_path, "homer", f"*{mark}*_homer.bed"))
                    if cand:
                        path = cand[0]
            if os.path.exists(path):
                regions["HOMER"][mark][folder_key] = load_bed_regions(path)
            else:
                print(f"  missing: {path}", file=sys.stderr)

        # MACS2
        for mark in marks:
            path = os.path.join(folder_path, "macs2", f"{mark}.bed")
            if not os.path.exists(path):
                cand = glob.glob(os.path.join(folder_path, "macs2", f"*{mark}*Peak"))
                if cand:
                    path = cand[0]
            if os.path.exists(path):
                regions["MACS2"][mark][folder_key] = load_bed_regions(path)
            else:
                print(f"  missing: {path}", file=sys.stderr)

        # ChromHMM default — binary files, or the per-mark result BEDs when absent.
        binary_dir = os.path.join(folder_path, "chromhmm_default")
        result_dir = os.path.join(folder_path, "chromhmm_default_result")
        cell_chromhmm_dir = os.path.join(folder_path, f"{args.cell}_chromhmm")
        chrom_peaks = {}
        if os.path.isdir(binary_dir):
            chrom_peaks = load_chromhmm_binary_peaks(binary_dir, args.cell,
                                                     bin_size=args.chromhmm_bin)
        for mark in marks:
            if chrom_peaks.get(mark):
                regions["Default"][mark][folder_key] = chrom_peaks[mark]
                continue
            bed = os.path.join(result_dir, f"{mark}.bed")
            if not os.path.exists(bed):
                bed = os.path.join(cell_chromhmm_dir, f"{mark}.bed")
            if not os.path.exists(bed):
                cand = glob.glob(os.path.join(folder_path, "*_chromhmm", f"{mark}.bed"))
                if cand:
                    bed = cand[0]
            if os.path.exists(bed):
                regions["Default"][mark][folder_key] = load_bed_regions(bed)
            else:
                print(f"  missing chromhmm default peaks for {mark} in {folder_path}",
                      file=sys.stderr)

    rows = []
    for method in ["OmniPeak", "HOMER", "MACS2", "Default"]:
        for mark in marks:
            row = {"method": method, "mark": mark}

            rep_stats = {}
            for rep in ["rep1", "rep2"]:
                rep_r = regions[method][mark].get(rep, [])
                rs = peak_stats(rep_r)
                rep_stats[rep] = rs
                row[f"n_peaks_{rep}"] = rs["n_peaks"]
                row[f"mean_length_{rep}"] = rs["mean_length"]

            # Pooled stats fall back to the mean of the replicates.
            pooled = regions[method][mark].get("pooled", [])
            s = peak_stats(pooled)
            if s["n_peaks"] == 0 and has_replicates:
                s1, s2 = rep_stats["rep1"], rep_stats["rep2"]
                if s1["n_peaks"] > 0 or s2["n_peaks"] > 0:
                    active = [ss for ss in [s1, s2] if ss["n_peaks"] > 0]
                    row["n_peaks"] = sum(ss["n_peaks"] for ss in active) / len(active)
                    row["mean_length"] = sum(ss["mean_length"] for ss in active) / len(active)
                    row["median_length"] = sum(ss["median_length"] for ss in active) / len(active)
                else:
                    row["n_peaks"] = 0
                    row["mean_length"] = 0
                    row["median_length"] = 0
            else:
                row["n_peaks"] = s["n_peaks"]
                row["mean_length"] = s["mean_length"]
                row["median_length"] = s["median_length"]

            if has_replicates:
                r1 = regions[method][mark].get("rep1", [])
                r2 = regions[method][mark].get("rep2", [])
                row["jaccard_rep1_vs_rep2"] = jaccard(r1, r2) if (r1 and r2) else None
            rows.append(row)

    df = pd.DataFrame(rows)
    stats_path = os.path.join(args.outdir, "peak_stats.tsv")
    df.to_csv(stats_path, sep="\t", index=False, float_format="%.4f")
    print(f"Saved {stats_path}", file=sys.stderr)

    _bar_plot(df, "n_peaks", "Number of peaks", "Peak count per mark (pooled)",
              os.path.join(args.outdir, "n_peaks.png"))
    _bar_plot(df, "mean_length", "Mean peak length (bp)", "Mean peak length per mark (pooled)",
              os.path.join(args.outdir, "peak_length.png"))
    _bar_plot(df, "median_length", "Median length (bp)", "Median peak length (pooled)",
              os.path.join(args.outdir, "median_length.png"))

    if has_replicates and "jaccard_rep1_vs_rep2" in df.columns:
        _bar_plot(df, "jaccard_rep1_vs_rep2",
                  "Jaccard (rep1 vs rep2)", "Peak Jaccard: rep1 vs rep2",
                  os.path.join(args.outdir, "jaccard_rep1_vs_rep2.png"))

    print("Done.", file=sys.stderr)
