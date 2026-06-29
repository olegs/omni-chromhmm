#!/usr/bin/env python3
# Per-segmentation analysis: report, segment lengths, emissions, enrichment.
#
# Also provides shared IO and plotting helpers imported by other scripts.
#
# Importable module (no CLI). Drive it from analysis.ipynb:
#   from analyze import run_analyze
#   run_analyze(seg="SEG.bed", bin_size=200, outdir="OUT",
#               inputs=["chromhmm/*.txt"], annotations=["COORDS/*.bed.gz"],
#               rnaseq="rnaseq.tsv", gtf="annotation.gtf.gz", emissions_only=False)

import csv
import glob
import gzip
import os
import re
import sys
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
import seaborn as sns


# --- IO helpers ----------------------------------------------------------

def expand_globs(paths):
    out = []
    for p in paths:
        if any(c in p for c in "*?[]"):
            out.extend(sorted(glob.glob(p)))
        else:
            out.append(p)
    return out


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def _get_skiprows(path):
    """Count 'track' or 'browser' lines at the beginning of a BED file."""
    skip = 0
    try:
        with open_text(path) as f:
            for line in f:
                if line.startswith(("track", "browser")):
                    skip += 1
                else:
                    break
    except Exception:
        pass
    return skip


def load_bed(path):
    """Load a BED file as a list of (chrom, start, end, name) tuples."""
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        skiprows=_get_skiprows(path),
        engine="c",
        on_bad_lines="skip",
        low_memory=False
    )
    # Drop rows where chrom, start, or end are NaN
    subset = [c for c in [0, 1, 2] if c in df.columns]
    if subset:
        df = df.dropna(subset=subset)
    # Take first 4 columns at most
    df = df.iloc[:, :4]
    # Ensure we have all 4 columns for return
    for col in range(4):
        if col not in df.columns:
            df[col] = "." if col == 3 else 0
    # Ensure correct types
    df[0] = df[0].astype(str)
    df[1] = pd.to_numeric(df[1], errors="coerce").fillna(0).astype(int)
    df[2] = pd.to_numeric(df[2], errors="coerce").fillna(0).astype(int)
    df[3] = df[3].fillna(".").astype(str)
    # Convert to a list of tuples
    return df[[0, 1, 2, 3]].to_records(index=False).tolist()


def _load_seg_full(path):
    """Load a BED file (plain or gzipped) as a list of (chrom, start, end, name, color) 5-tuples."""
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        skiprows=_get_skiprows(path),
        engine="c",
        on_bad_lines="skip",
        low_memory=False
    )
    # Drop rows where chrom, start, or end are NaN
    subset = [c for c in [0, 1, 2] if c in df.columns]
    if subset:
        df = df.dropna(subset=subset)
    # Ensure we have enough columns. ChromHMM dense bed has at least 9.
    for col in [0, 1, 2, 3, 8]:
        if col not in df.columns:
            if col == 0: df[col] = "chrUnk"
            elif col in (1, 2): df[col] = 0
            elif col == 3: df[col] = "."
            elif col == 8: df[col] = "0,0,0"
    df[0] = df[0].astype(str)
    df[1] = pd.to_numeric(df[1], errors="coerce").fillna(0).astype(int)
    df[2] = pd.to_numeric(df[2], errors="coerce").fillna(0).astype(int)
    df[3] = df[3].fillna(".").astype(str)
    df[8] = df[8].fillna("0,0,0").astype(str)
    return df[[0, 1, 2, 3, 8]].to_records(index=False).tolist()


def load_bed_df(path, sample=None):
    """Load a (possibly gzipped) ChromHMM BED file into a DataFrame.

    Returns DataFrame with columns: chrom, start, end, state, length, rgb.
    Optionally adds a 'sample' column.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        skiprows=_get_skiprows(path),
        engine="c",
        on_bad_lines="skip",
        low_memory=False
    )
    # Drop rows where chrom, start, end or state are NaN
    for col in [0, 1, 2, 3]:
        if col not in df.columns:
            df[col] = "." if col in (0, 3) else 0
    df = df.dropna(subset=[0, 1, 2, 3])

    res = pd.DataFrame()
    res["chrom"] = df[0].astype(str)
    res["start"] = pd.to_numeric(df[1], errors="coerce").fillna(0).astype(np.int64)
    res["end"] = pd.to_numeric(df[2], errors="coerce").fillna(0).astype(np.int64)
    res["state"] = df[3].astype(str)
    res["length"] = (res["end"] - res["start"]).astype(np.int64)

    rgb_col = 8
    if rgb_col not in df.columns:
        res["rgb"] = "128,128,128"
    else:
        res["rgb"] = df[8].fillna("128,128,128").astype(str)

    if sample is not None:
        res["sample"] = sample
    return res


def load_binary(path):
    chrom = "chrUnknown"
    marks = []
    try:
        with open_text(path) as f:
            line1 = f.readline()
            if not line1:
                return chrom, marks, np.array([], dtype=np.int8)
            head = line1.rstrip("\n").split("\t")
            chrom = head[1] if len(head) > 1 else "chrUnknown"

            line2 = f.readline()
            if not line2:
                return chrom, marks, np.array([], dtype=np.int8)
            marks = line2.rstrip("\n").split("\t")

            try:
                df = pd.read_csv(f, sep="\t", header=None, dtype=np.int8)
                return chrom, marks, df.values
            except pd.errors.EmptyDataError:
                return chrom, marks, np.array([], dtype=np.int8)
    except (EOFError, gzip.BadGzipFile) as e:
        print(f"Warning: Corrupted gzip file {path}: {e}. Data might be incomplete.", file=sys.stderr)
    return chrom, marks, np.array([], dtype=np.int8)


# --- DataFrame helpers ---------------------------------------------------

def rgb_str_to_hex(rgb):
    """Convert BED itemRgb '255,128,0' to matplotlib hex '#FF8000'."""
    try:
        r, g, b = (int(x) for x in rgb.split(","))
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "#888888"


def state_color_map(df):
    """Return {state: hex_color} using the first observed itemRgb per state."""
    return (
        df.groupby("state")["rgb"]
        .first()
        .apply(rgb_str_to_hex)
        .to_dict()
    )


def _natural_sort_key(s):
    """Sort key for natural ordering: 'E1' < 'E2' < 'E10', 'Tss' < 'Tx'."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def sorted_states(df):
    """Sort states in natural order from a DataFrame."""
    return sorted(df["state"].unique(), key=_natural_sort_key)


# --- multi-sample plots (shared) ----------------------------------------

def plot_per_state_violin(df, out_path, title=None):
    """Violin plot of log10 segment lengths per state, coloured by itemRgb."""
    states = sorted_states(df)
    colors = state_color_map(df)
    n = len(states)

    fig, ax = plt.subplots(figsize=(max(3, n * 0.4), 4))
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
    ax.set_xticklabels(states, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Chromatin state")
    ax.set_ylabel("log10(segment length + 1)  [bp]")
    ax.set_ylim(0, 8)
    if title is None:
        title = f"Segment length per state ({df['sample'].nunique()} samples)"
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_coverage_per_state(df, out_path, title=None):
    """Violin plot of total genomic coverage (bp) per state across samples."""
    states = sorted_states(df)
    colors = state_color_map(df)

    coverage = (df.groupby(["sample", "state"])["length"]
                  .sum()
                  .reset_index(name="total_bp"))

    n = len(states)
    fig, ax = plt.subplots(figsize=(max(3, n * 0.4), 4))
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
    ax.set_xticklabels(states, rotation=60, ha="right", fontsize=7)
    ax.set_xlabel("Chromatin state")
    ax.set_ylabel("log10(total coverage + 1)  [bp]")
    ax.set_ylim(2, 10)
    if title is None:
        title = f"Total genomic coverage per state ({df['sample'].nunique()} samples)"
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_overall_per_sample(df, out_path, title=None, short_names=False):
    """Violin plot of overall segment lengths per sample."""
    samples = sorted(df["sample"].unique())
    n = len(samples)

    fig, ax = plt.subplots(figsize=(max(3, n * 0.35), 4))
    data = [np.log10(df.loc[df["sample"] == s, "length"].values + 1) for s in samples]
    parts = ax.violinplot(data, positions=range(n), showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.75)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(0.8)

    labels = samples
    if short_names:
        labels = ["/".join(s.split("/")[-2:]) if "/" in s else s for s in samples]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("Sample")
    ax.set_ylabel("log10(segment length + 1)  [bp]")
    if title is None:
        title = "Overall segment length per sample"
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_state_heatmap(df, out_path, title=None, short_names=False):
    """Heatmap of median segment length (log10) per sample x state."""
    pivot = (df.groupby(["sample", "state"])["length"]
               .median()
               .unstack("state"))
    cols = sorted_states(df)
    cols = [c for c in cols if c in pivot.columns]
    pivot = pivot[cols]

    if short_names:
        pivot.index = ["/".join(s.split("/")[-2:]) if "/" in s else s for s in pivot.index]

    fig, ax = plt.subplots(figsize=(max(5, len(cols) * 0.35),
                                    max(3, len(pivot) * 0.3)))
    sns.heatmap(np.log10(pivot + 1), ax=ax, cmap="YlOrRd",
                linewidths=0.3, annot=True, fmt=".1f",
                annot_kws={"fontsize": 5},
                cbar_kws={"label": "log10(median length + 1)"})
    if title is None:
        title = "Median segment length [log10 bp]"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Chromatin state")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path}")


def save_stats(df, out_path, extra_groupby=None):
    """Save per-sample x per-state summary statistics."""
    group_cols = list(extra_groupby or []) + ["sample", "state"]
    stats = (df.groupby(group_cols)["length"]
               .agg(count="count", mean="mean", median="median",
                    std="std",
                    p5=lambda x: np.percentile(x, 5),
                    p95=lambda x: np.percentile(x, 95),
                    total_bp="sum")
               .reset_index())
    stats.to_csv(out_path, sep="\t", index=False, float_format="%.1f")
    print(f"  saved {out_path}")


# --- RNA-seq expressed gene annotations ----------------------------------

def load_expressed_gene_ids(rnaseq_path, tpm_threshold=1.0):
    """Parse ENCODE RNA-seq quantification TSV; return set of expressed gene IDs."""
    expressed = set()
    with open(rnaseq_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                tpm = float(row["TPM"])
                gene_id = row["gene_id"].strip()
                if tpm >= tpm_threshold:
                    expressed.add(gene_id)
                    if "." in gene_id:
                        expressed.add(gene_id.split(".")[0])
            except (ValueError, KeyError):
                continue
    return expressed


def load_expressed_gene_coords(gtf_path, expressed_ids):
    """Parse GENCODE GTF for expressed gene bodies and TSS regions."""
    gene_bodies = []
    tss_regions = []
    try:
        with open_text(gtf_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 9 or cols[2] != "gene":
                    continue
                attrs = cols[8]
                gene_id = None
                gene_name = None
                for attr in attrs.split(";"):
                    attr = attr.strip()
                    if attr.startswith("gene_id"):
                        gene_id = attr.split('"')[1] if '"' in attr else attr.split()[-1]
                    elif attr.startswith("gene_name"):
                        gene_name = attr.split('"')[1] if '"' in attr else attr.split()[-1]
                matched = False
                if gene_id and (gene_id in expressed_ids or gene_id.split(".")[0] in expressed_ids):
                    matched = True
                if gene_name and gene_name in expressed_ids:
                    matched = True
                if not matched:
                    continue
                label = gene_name or gene_id
                chrom = cols[0]
                start = int(cols[3]) - 1
                end = int(cols[4])
                strand = cols[6]
                gene_bodies.append((chrom, start, end, label))
                tss = start if strand == "+" else end - 1
                tss_regions.append((chrom, tss, tss + 1, label))
    except (EOFError, gzip.BadGzipFile) as e:
        print(f"Warning: Corrupted gzip file {gtf_path}: {e}. Data might be incomplete.", file=sys.stderr)
    return gene_bodies, tss_regions


def make_expressed_annotations(rnaseq_path, gtf_path):
    """Build expressed gene body and TSS BED annotations from RNA-seq + GTF."""
    expressed_ids = load_expressed_gene_ids(rnaseq_path)
    print(f"  RNA-seq: {len(expressed_ids)} expressed gene IDs (TPM >= 1)", file=sys.stderr)

    gene_bodies, tss_regions = load_expressed_gene_coords(gtf_path, expressed_ids)
    print(f"  GTF: {len(gene_bodies)} expressed gene bodies, {len(tss_regions)} TSS regions",
          file=sys.stderr)

    result = []
    if gene_bodies:
        result.append(("ExpressedGeneBodies", gene_bodies))
    if tss_regions:
        result.append(("ExpressedTSS", tss_regions))
    return result


# --- Transition entropy (per-segmentation) --------------------------------

QUIESCENT_STATES = {"Quies", "Quiescent", "8_ZNF/Rpts", "9_Het", "Quies_low"}


def build_transition_matrix(segs, bin_size, exclude_states=None, mapping=None):
    """Build empirical transition count matrix at bin resolution."""
    exclude = set(exclude_states or [])
    by_chrom = defaultdict(dict)
    for row in segs:
        chrom, s, e, state = row[:4]
        if state in exclude:
            continue
        st = mapping.get(state, state) if mapping else state
        for b in range(s // bin_size, e // bin_size):
            by_chrom[chrom][b] = st

    all_states = sorted(
        {mapping.get(row[3], row[3]) if mapping else row[3]
         for row in segs if row[3] not in exclude},
        key=_natural_sort_key)
    state_idx = {s: i for i, s in enumerate(all_states)}
    n = len(all_states)
    counts = np.zeros((n, n), dtype=np.float64)
    state_bp = np.zeros(n, dtype=np.float64)

    for chrom, bins in by_chrom.items():
        sorted_idxs = sorted(bins.keys())
        for k, b in enumerate(sorted_idxs):
            st = bins[b]
            state_bp[state_idx[st]] += bin_size
            if k > 0 and sorted_idxs[k - 1] == b - 1:
                prev_st = bins[sorted_idxs[k - 1]]
                counts[state_idx[prev_st], state_idx[st]] += 1

    return all_states, counts, state_bp


def transition_entropy(states, counts, state_bp):
    """Compute total and per-state transition matrix entropy.

    Returns (total_entropy, per_state_entropy, transition_prob_matrix, stationary_dist).
    """
    n = len(states)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A = counts / row_sums
    H = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                H[i] -= A[i, j] * np.log2(A[i, j])
    total_bp = state_bp.sum()
    pi = state_bp / total_bp if total_bp > 0 else np.ones(n) / n
    return np.dot(pi, H), H, A, pi


def save_transition_entropy(segs, bin_size, outdir):
    """Compute and save transition entropy + matrix for a single segmentation."""
    edir = os.path.join(outdir, "entropy")
    os.makedirs(edir, exist_ok=True)

    for suffix, excl in [("", None), ("_noqh", QUIESCENT_STATES)]:
        states, counts, state_bp = build_transition_matrix(segs, bin_size, excl)
        if not states:
            continue
        total_H, H, A, pi = transition_entropy(states, counts, state_bp)

        detail = os.path.join(edir, f"transition_entropy{suffix}.tsv")
        with open(detail, "w") as f:
            f.write("state\tstationary_prob\tentropy\tself_transition_prob\n")
            for i, s in enumerate(states):
                f.write(f"{s}\t{pi[i]:.6f}\t{H[i]:.4f}\t{A[i, i]:.6f}\n")

        pd.DataFrame(A, index=states, columns=states).to_csv(
            os.path.join(edir, f"transition_matrix{suffix}.tsv"),
            sep="\t", float_format="%.6f")

        label = os.path.basename(outdir)
        excl_label = f" (excl. quiescent)" if excl else ""
        fig, ax = plt.subplots(figsize=(max(5, len(states) * 0.4),
                                        max(4, len(states) * 0.35)))
        sns.heatmap(A, xticklabels=states, yticklabels=states,
                    cmap="Blues", vmin=0, vmax=1, ax=ax,
                    linewidths=0.3, annot=True, fmt=".2f",
                    annot_kws={"fontsize": 6})
        ax.set_title(f"Transition matrix — {label}{excl_label}\n"
                     f"(total entropy = {total_H:.4f})")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        fig.tight_layout()
        fig.savefig(os.path.join(edir, f"transition_matrix{suffix}.png"), dpi=300)
        plt.close(fig)


# --- consistency analysis ------------------------------------------------

def merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for curr in intervals[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1]:
            prev[1] = max(prev[1], curr[1])
        else:
            merged.append(list(curr))
    return merged


def _compute_state_consistency_chrom(chrom_data, bin_size, epsilon):
    """Worker function for per-chromosome consistency computation."""
    chrom_state_depth_counts = {}
    for state, segs_by_sample in chrom_data.items():
        events = []
        for sample_segs in segs_by_sample:
            if not sample_segs:
                continue
            if epsilon > 0:
                expanded = [(max(0, s[0] - epsilon), s[1] + epsilon) for s in sample_segs]
                merged = merge_intervals(expanded)
            else:
                merged = sample_segs
            for start, end in merged:
                events.append((start, 1))
                events.append((end, -1))
        if not events:
            continue
        events.sort()
        depth_counts = defaultdict(int)
        current_depth = 0
        last_pos = events[0][0]
        for pos, delta in events:
            if pos > last_pos and current_depth > 0:
                depth_counts[current_depth] += (pos - last_pos) // bin_size
            current_depth += delta
            last_pos = pos
        chrom_state_depth_counts[state] = dict(depth_counts)
    return chrom_state_depth_counts


def compute_state_consistency(segs_list, bin_size=200, epsilon=0, show_progress=False):
    """Compute depth of state coverage (cumulative <= N) across multiple segmentations.

    For given epsilon, count that bin X is covered by state Y, if there is a state Y in <= epsilon distance to the bin.

    Returns: {state -> {depth -> cumulative_count_bins}} where depth is 1..len(segs_list)
    """
    # 1. Pre-group data by chrom and state to optimize performance
    # data[chrom][state][sample_idx] = [(start, end), ...]
    data = defaultdict(lambda: defaultdict(lambda: [[] for _ in range(len(segs_list))]))
    for i, segs in enumerate(segs_list):
        for s in segs:
            data[s[0]][s[3]][i].append((s[1], s[2]))

    all_chroms = sorted(data.keys())

    # 2. Parallelize over chromosomes
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_compute_state_consistency_chrom,
                                   {state: list(v) for state, v in data[chrom].items()},
                                   bin_size, epsilon)
                   for chrom in all_chroms]

        if show_progress:
            try:
                from tqdm.auto import tqdm
                for f in tqdm(as_completed(futures), total=len(futures),
                              desc=f"Consistency (eps={epsilon})", leave=False):
                    results.append(f.result())
            except ImportError:
                results = [f.result() for f in futures]
        else:
            results = [f.result() for f in futures]

    # 3. Merge results
    state_depth_counts = defaultdict(lambda: defaultdict(int))
    for res in results:
        for state, depth_counts in res.items():
            for d, count in depth_counts.items():
                state_depth_counts[state][d] += count

    # Compute cumulative counts: supported by <= N segmentations
    M = len(segs_list)
    for state in state_depth_counts:
        depths = state_depth_counts[state]
        cumulative = 0
        for d in range(1, M + 1):
            cumulative += depths[d]
            depths[d] = cumulative

    # Convert defaultdict to regular dict for return
    return {state: dict(depths) for state, depths in state_depth_counts.items()}


def plot_state_consistency(state_depth_counts, title, out_path, colors=None):
    """Build a cumulative plot of percentage of state coverage supported by <= N segmentations."""
    if not state_depth_counts:
        return

    M = max(max(d.keys()) for d in state_depth_counts.values())
    states = sorted(state_depth_counts.keys(), key=_natural_sort_key)

    plt.figure(figsize=(8, 5))
    for state in states:
        depths = state_depth_counts[state]
        total_bins = depths[M]
        if total_bins == 0:
            continue

        x = np.arange(1, M + 1)
        y = [depths[n] / total_bins * 100 for n in x]

        color = colors.get(state) if colors else None
        if isinstance(color, str) and "," in color:
            color = rgb_str_to_hex(color)

        # Replace white or quiescent with black for visibility
        if isinstance(color, str):
            c_upper = color.upper()
            if c_upper in ["#FFFFFF", "#FFF", "WHITE"] or "QUIES" in state.upper():
                color = "black"

        plt.plot(x, y, label=state, marker='.', markersize=2, color=color, alpha=0.8)

    plt.title(f"State consistency ({title})", fontsize=11, fontweight="bold")
    plt.xlabel("Supported by <= N segmentations", fontsize=9)
    plt.ylabel("% of state coverage", fontsize=9)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='x-small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# --- single-segmentation analysis ----------------------------------------

def save_report(segs, outdir):
    """Save report.tsv with state-level statistics."""
    lengths = defaultdict(list)
    for row in segs:
        _, s, e, name = row[:4]
        lengths[name].append(e - s)
    states = sorted(lengths, key=_natural_sort_key)

    path = os.path.join(outdir, "report.tsv")
    with open(path, "w") as f:
        f.write("state\tn_segments\ttotal_bp\tmean_length\tmedian_length\n")
        for st in states:
            ll = lengths[st]
            f.write(f"{st}\t{len(ll)}\t{sum(ll)}\t{np.mean(ll):.1f}\t{np.median(ll):.1f}\n")
    print(f"  Report: {len(states)} states -> {path}", file=sys.stderr)


def plot_segment_lengths(segs, outdir):
    lengths = defaultdict(list)
    for row in segs:
        _, s, e, name = row[:4]
        lengths[name].append(e - s)
    states = sorted(lengths, key=_natural_sort_key)
    means = [np.mean(lengths[s]) for s in states]

    fig, ax = plt.subplots(figsize=(max(4, 0.3 * len(states)), 3))
    ax.bar(states, means, edgecolor="lightgrey", linewidth=1)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=90)
    ax.set_title("Average segment length per state")
    ax.set_xlabel("state")
    ax.set_ylabel("bp")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "segment_length.png"), dpi=300)
    plt.close(fig)


# Preferred display order for histone marks
MARKS_ORDER = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3"]


def _reorder_marks(marks, mat):
    """Reorder columns of *mat* to match MARKS_ORDER. Unknown marks are appended."""
    known = [m for m in MARKS_ORDER if m in marks]
    unknown = [m for m in marks if m not in MARKS_ORDER]
    new_order = known + unknown
    idx = [marks.index(m) for m in new_order]
    return new_order, mat[:, idx]


def compute_emissions(segs, inputs, bin_size):
    """Compute state emission matrix (states x marks). Returns (states, marks, matrix)."""
    by_chrom, marks = {}, None
    for p in sorted(inputs):
        chrom, m, data = load_binary(p)
        if marks is None:
            marks = m
        by_chrom[chrom] = data

    sums = defaultdict(lambda: np.zeros(len(marks), dtype=np.float64))
    counts = defaultdict(int)
    for row in segs:
        chrom, s, e, name = row[:4]
        data = by_chrom.get(chrom)
        if data is None:
            continue
        b0 = s // bin_size
        b1 = min(e // bin_size, data.shape[0])
        if b1 > b0:
            sums[name] += data[b0:b1].sum(axis=0)
            counts[name] += (b1 - b0)

    states = sorted(sums, key=_natural_sort_key)
    mat = np.array([sums[s] / max(counts[s], 1) for s in states])
    marks, mat = _reorder_marks(marks, mat)
    return states, marks, mat


def save_emissions_table(states, marks, mat, outdir, subdir="bin_emissions"):
    """Save {subdir}/state_emissions.tsv: rows=states, cols=marks."""
    edir = os.path.join(outdir, subdir)
    os.makedirs(edir, exist_ok=True)
    path = os.path.join(edir, "state_emissions.tsv")
    with open(path, "w") as f:
        f.write("state\t" + "\t".join(marks) + "\n")
        for i, st in enumerate(states):
            vals = "\t".join(f"{v:.4f}" for v in mat[i])
            f.write(f"{st}\t{vals}\n")


def plot_emissions(states, marks, mat, outdir, subdir="bin_emissions"):
    """Plot {subdir}/state_emissions.png heatmap."""
    edir = os.path.join(outdir, subdir)
    os.makedirs(edir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(3, 0.4 * len(marks)),
                                    max(4, 0.35 * len(states))))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(marks))); ax.set_xticklabels(marks, rotation=90)
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states)
    ax.set_title("State emissions")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(edir, "state_emissions.png"), dpi=300)
    plt.close(fig)


# Preferred display order for enrichment annotations (TSV only)
ANNOTATIONS_ORDER = [
    "Genome %", "CpGIsland", "RefSeqExon", "RefSeqGene",
    "RefSeqTES", "RefSeqTSS", "RefSeqTSS2kb",
]

_PLOT_ANNOTATION_PREFIX = "atac_"


def _reorder_annotations(labels):
    def _base(lbl):
        return lbl.replace(".hg38", "")
    known = [l for ao in ANNOTATIONS_ORDER
             for l in labels if _base(l) == ao]
    unknown = [l for l in labels if _base(l) not in ANNOTATIONS_ORDER]
    return known + unknown


def _rename_plot_label(label):
    """Human-readable label for enrichment.png."""
    if label.startswith(_PLOT_ANNOTATION_PREFIX):
        return "OpenChromatin ATAC"
    return label


def _compute_overlap_bp(by_chrom, starts, ann_segs):
    state_hit = defaultdict(int)
    for row in ann_segs:
        chrom, s, e = row[:3]
        if chrom not in by_chrom:
            continue
        arr = by_chrom[chrom]
        i = max(0, bisect_left(starts[chrom], s) - 1)
        while i < len(arr) and arr[i][0] < e:
            ss, se, st = arr[i]
            ov = min(se, e) - max(ss, s)
            if ov > 0:
                state_hit[st] += ov
            i += 1
    return state_hit


def compute_enrichment(segs, annotation_items):
    """Fold enrichment of each state vs each annotation (ChromHMM-style)."""
    by_chrom = defaultdict(list)
    state_total = defaultdict(int)
    for row in segs:
        chrom, s, e, name = row[:4]
        by_chrom[chrom].append((s, e, name))
        state_total[name] += e - s
    for chrom in by_chrom:
        by_chrom[chrom].sort()
    starts = {c: [s for s, _, _ in v] for c, v in by_chrom.items()}

    total_bp = sum(state_total.values())
    states = sorted(state_total, key=_natural_sort_key)

    rows = []
    for label, bed_data in annotation_items:
        if isinstance(bed_data, str):
            try:
                ann_segs = load_bed(bed_data)
            except Exception as err:
                print(f"skipping annotation {bed_data}: {err}", file=sys.stderr)
                continue
        else:
            ann_segs = bed_data

        ann_bp = sum(row[2] - row[1] for row in ann_segs)
        ann_frac = ann_bp / total_bp if total_bp > 0 else 0

        state_hit = _compute_overlap_bp(by_chrom, starts, ann_segs)

        for st in states:
            overlap = state_hit.get(st, 0)
            state_frac = overlap / state_total[st] if state_total[st] > 0 else 0
            fold = state_frac / ann_frac if ann_frac > 0 else 0
            union = state_total[st] + ann_bp - overlap
            jaccard = overlap / union if union > 0 else 0
            rows.append({"state": st, "label": label,
                         "fold_enrichment": fold, "jaccard": jaccard, "coverage": state_frac})

    if not rows:
        return pd.DataFrame(columns=["state", "label", "fold_enrichment"])
    return pd.DataFrame(rows)


def save_enrichment_table(enrich_df, outdir):
    edir = os.path.join(outdir, "enrichment")
    os.makedirs(edir, exist_ok=True)
    enrich_df.to_csv(os.path.join(edir, "enrichment.tsv"),
                     sep="\t", index=False, float_format="%.4f")
    # Save Jaccard-only table for quick loading by compare_methods.py
    if "jaccard" in enrich_df.columns:
        (enrich_df[["state", "label", "jaccard"]]
         .to_csv(os.path.join(edir, "jaccard.tsv"),
                 sep="\t", index=False, float_format="%.6f"))
    if "coverage" in enrich_df.columns:
        (enrich_df[["state", "label", "coverage"]]
         .to_csv(os.path.join(edir, "coverage.tsv"),
                 sep="\t", index=False, float_format="%.6f"))


def _column_minmax_scale(mat):
    scaled = mat.copy()
    for c in scaled.columns:
        col_range = scaled[c].max() - scaled[c].min()
        if col_range < 1e-10:
            scaled[c] = 1.0 / len(scaled[c])
        else:
            scaled[c] = (scaled[c] - scaled[c].min()) / col_range
    return scaled


def plot_enrichment(enrich_df, segs, outdir):
    """Plot enrichment.png: odds ratio heatmap, per-column min-max scaled."""
    if enrich_df.empty:
        return
    edir = os.path.join(outdir, "enrichment")
    os.makedirs(edir, exist_ok=True)

    sorted_idx = sorted(enrich_df["state"].unique(), key=_natural_sort_key)
    sorted_cols = _reorder_annotations(sorted(enrich_df["label"].unique()))

    fold_mat = enrich_df.pivot(index="state", columns="label", values="fold_enrichment")
    fold_mat = fold_mat.loc[sorted_idx, sorted_cols]
    fold_mat.columns = [_rename_plot_label(c) for c in fold_mat.columns]

    state_bp = defaultdict(int)
    for row in segs:
        _, s, e, name = row[:4]
        state_bp[name] += e - s
    total_bp = sum(state_bp.values())
    genome_pct = pd.Series(
        {st: 100.0 * state_bp.get(st, 0) / total_bp for st in sorted_idx},
        name="Genome %")
    fold_mat.insert(0, "Genome %", genome_pct)

    scaled = _column_minmax_scale(fold_mat)

    fig, ax = plt.subplots(figsize=(max(3, 0.4 * len(scaled.columns)),
                                    max(4, 0.35 * len(scaled))))
    ax.imshow(scaled.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(scaled.columns)))
    ax.set_xticklabels(scaled.columns, rotation=90)
    ax.set_yticks(range(len(scaled.index)))
    ax.set_yticklabels(scaled.index)
    ax.set_title("Functional enrichment")
    fig.tight_layout()
    fig.savefig(os.path.join(edir, "enrichment.png"), dpi=300)
    plt.close(fig)


# --- direct-call entry point ---------------------------------------------

def run_analyze(seg, bin_size, outdir, inputs=None, annotations=None,
                rnaseq=None, gtf=None, bw_emissions=None, emissions_only=False):
    """Per-segmentation analysis: report, emissions, enrichment.

    Direct-call entry point (the former CLI). Writes report / emission /
    enrichment tables and plots under *outdir*; called from analysis.ipynb.
    """
    os.makedirs(outdir, exist_ok=True)
    segs = load_bed(seg)

    if not emissions_only:
        save_report(segs, outdir)
        plot_segment_lengths(segs, outdir)
        save_transition_entropy(segs, bin_size, outdir)

    inputs = expand_globs(inputs or [])
    if inputs:
        states, marks, emission_mat = compute_emissions(segs, inputs, bin_size)
        save_emissions_table(states, marks, emission_mat, outdir)
        plot_emissions(states, marks, emission_mat, outdir)
        # Save alongside BED for fast lookup by compare.py (same format as bw_emissions.npz).
        npz_path = os.path.splitext(seg)[0] + ".bin_emissions.npz"
        np.savez_compressed(npz_path,
                            states=np.array(states),
                            marks=np.array(marks),
                            mat=emission_mat)

    if bw_emissions and os.path.exists(bw_emissions):
        data = np.load(bw_emissions, allow_pickle=False)
        bw_states = list(data["states"])
        bw_marks  = list(data["marks"])
        bw_mat    = data["mat"]
        # Apply the same axis ordering as compute_emissions:
        # states → natural sort, marks → MARKS_ORDER
        state_order = sorted(range(len(bw_states)),
                             key=lambda i: _natural_sort_key(bw_states[i]))
        bw_states = [bw_states[i] for i in state_order]
        bw_mat    = bw_mat[state_order]
        bw_marks, bw_mat = _reorder_marks(bw_marks, bw_mat)
        save_emissions_table(bw_states, bw_marks, bw_mat, outdir,
                             subdir="bw_emissions")
        plot_emissions(bw_states, bw_marks, bw_mat, outdir,
                       subdir="bw_emissions")

    annotation_items = []
    for p in expand_globs(annotations or []):
        if os.path.exists(p):
            label = os.path.basename(p).replace(".bed.gz", "").replace(".bed", "")
            annotation_items.append((label, p))

    if rnaseq and gtf:
        if os.path.exists(rnaseq) and os.path.exists(gtf):
            annotation_items.extend(make_expressed_annotations(rnaseq, gtf))
        else:
            if not os.path.exists(rnaseq):
                print(f"Warning: RNA-seq file {rnaseq} not found, skipping expressed annotations.", file=sys.stderr)
            if not os.path.exists(gtf):
                print(f"Warning: GTF file {gtf} not found, skipping expressed annotations.", file=sys.stderr)

    if annotation_items:
        enrich_df = compute_enrichment(segs, annotation_items)
        save_enrichment_table(enrich_df, outdir)
        plot_enrichment(enrich_df, segs, outdir)
