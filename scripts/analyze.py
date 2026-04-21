#!/usr/bin/env python3
# Segmentation analysis and quality metrics.
#
# Single --seg: per-segmentation report, emissions, enrichment.
# Multiple --seg: cross-segmentation entropy, kappa, segment stats.
#
# Also provides shared helpers used by analyze_downloaded.py / analyze_matched.py.
#
# Usage:
#   # Per-segmentation analysis
#   analyze.py --seg SEG.bed --bin 200 --outdir OUT \
#       [--inputs chromhmm/*.txt] [--annotations COORDS/*.bed.gz]
#
#   # Cross-segmentation metrics (entropy, kappa, segment stats)
#   analyze.py --seg SEG1.bed SEG2.bed ... --bin 200 --outdir OUT

import argparse
import csv
import glob
import gzip
import os
import re
import sys
import types
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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


def load_bed(path):
    """Load a BED file as a list of (chrom, start, end, name) tuples."""
    segs = []
    with open_text(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                chrom, s, e = p[0], int(p[1]), int(p[2])
            except ValueError:
                continue
            name = p[3] if len(p) > 3 else "."
            segs.append((chrom, s, e, name))
    return segs


def load_bed_df(path, sample=None):
    """Load a (possibly gzipped) ChromHMM BED file into a DataFrame.

    Returns DataFrame with columns: chrom, start, end, state, length, rgb.
    Optionally adds a 'sample' column.
    """
    rows = []
    with open_text(path) as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue
            chrom, start, end, state = parts[0], int(parts[1]), int(parts[2]), parts[3]
            rgb = parts[8] if len(parts) > 8 else "128,128,128"
            rows.append((chrom, start, end, state, end - start, rgb))
    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "state", "length", "rgb"])
    if sample is not None:
        df["sample"] = sample
    return df


def load_binary(path):
    with open_text(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        chrom = head[1] if len(head) > 1 else "chrUnknown"
        marks = f.readline().rstrip("\n").split("\t")
        rows = [list(map(int, line.rstrip("\n").split("\t")))
                for line in f if line.strip()]
    return chrom, marks, np.asarray(rows, dtype=np.int8)


# --- DataFrame helpers (shared with analyze_downloaded / analyze_matched) -

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

    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), 4))
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
    fig.savefig(out_path, dpi=150)
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
    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), 4))
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_overall_per_sample(df, out_path, title=None, short_names=False):
    """Violin plot of overall segment lengths per sample."""
    samples = sorted(df["sample"].unique())
    n = len(samples)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.35), 4))
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
    fig.savefig(out_path, dpi=150)
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def save_stats(df, out_path, extra_groupby=None):
    """Save per-sample x per-state summary statistics.

    extra_groupby: additional column names to group by (e.g. ["model"]).
    """
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
    """Parse ENCODE RNA-seq quantification TSV; return set of expressed gene IDs.

    Gene IDs may be Ensembl (ENSG...) or Entrez (numeric). Version suffixes
    (e.g. ENSG00000141510.18) are stripped for matching.
    """
    expressed = set()
    with open(rnaseq_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                tpm = float(row["TPM"])
                gene_id = row["gene_id"].strip()
                if tpm >= tpm_threshold:
                    # Store both versioned and unversioned forms
                    expressed.add(gene_id)
                    if "." in gene_id:
                        expressed.add(gene_id.split(".")[0])
            except (ValueError, KeyError):
                continue
    return expressed


def load_expressed_gene_coords(gtf_path, expressed_ids):
    """Parse GENCODE GTF for gene-level entries; return gene bodies and TSS for expressed genes.

    Matches against both gene_id (Ensembl) and gene_name fields in the GTF.
    Returns (gene_bodies, tss_regions) where each is a list of (chrom, start, end, name).
    """
    gene_bodies = []
    tss_regions = []
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
            # Match by Ensembl gene_id (with or without version) or gene_name
            matched = False
            if gene_id and (gene_id in expressed_ids or gene_id.split(".")[0] in expressed_ids):
                matched = True
            if gene_name and gene_name in expressed_ids:
                matched = True
            if not matched:
                continue
            label = gene_name or gene_id
            chrom = cols[0]
            start = int(cols[3]) - 1  # GTF is 1-based
            end = int(cols[4])
            strand = cols[6]
            gene_bodies.append((chrom, start, end, label))
            tss = start if strand == "+" else end - 1
            tss_regions.append((chrom, tss, tss + 1, label))
    return gene_bodies, tss_regions


def make_expressed_annotations(rnaseq_path, gtf_path):
    """Build expressed gene body and TSS BED annotations from RNA-seq + GTF.

    Returns list of (label, bed_regions) pairs.
    """
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


# --- single-segmentation analysis ---------------------------------------

def save_report(segs, outdir):
    """Save report.tsv with state-level statistics."""
    lengths = defaultdict(list)
    for _, s, e, name in segs:
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
    for _, s, e, name in segs:
        lengths[name].append(e - s)
    states = sorted(lengths, key=_natural_sort_key)
    means = [np.mean(lengths[s]) for s in states]

    fig, ax = plt.subplots(figsize=(max(4, 0.3 * len(states)), 3))
    ax.bar(states, means)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=90)
    ax.set_title("Average segment length per state")
    ax.set_xlabel("state")
    ax.set_ylabel("bp")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "segment_length.png"), dpi=120)
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
    for chrom, s, e, name in segs:
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


def save_emissions_table(states, marks, mat, outdir):
    """Save emissions/state_emissions.tsv: rows=states, cols=marks."""
    edir = os.path.join(outdir, "emissions")
    os.makedirs(edir, exist_ok=True)
    path = os.path.join(edir, "state_emissions.tsv")
    with open(path, "w") as f:
        f.write("state\t" + "\t".join(marks) + "\n")
        for i, st in enumerate(states):
            vals = "\t".join(f"{v:.4f}" for v in mat[i])
            f.write(f"{st}\t{vals}\n")


def plot_emissions(states, marks, mat, outdir):
    """Plot emissions/state_emissions.png heatmap."""
    edir = os.path.join(outdir, "emissions")
    os.makedirs(edir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(3, 0.4 * len(marks)),
                                    max(4, 0.35 * len(states))))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(marks))); ax.set_xticklabels(marks, rotation=90)
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states)
    ax.set_title("State emissions")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(edir, "state_emissions.png"), dpi=120)
    plt.close(fig)


# Preferred display order for enrichment annotations
ANNOTATIONS_ORDER = [
    "Genome %", "CpGIsland", "RefSeqExon", "RefSeqGene",
    "RefSeqTES", "RefSeqTSS", "RefSeqTSS2kb",
]


def _reorder_annotations(labels):
    """Return sorted label list: known annotations first in ANNOTATIONS_ORDER, then others."""
    # Strip .hg38 suffix for matching
    def _base(lbl):
        return lbl.replace(".hg38", "")

    known = [l for ao in ANNOTATIONS_ORDER
             for l in labels if _base(l) == ao]
    unknown = [l for l in labels if _base(l) not in ANNOTATIONS_ORDER]
    return known + unknown


def _compute_overlap_bp(by_chrom, starts, ann_segs):
    """Compute bp overlap between segmentation states and an annotation.

    Returns dict: state → overlap_bp.
    """
    state_hit = defaultdict(int)
    for chrom, s, e, _ in ann_segs:
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
    """Fisher enrichment of each state vs each annotation using bp overlaps.

    annotation_items: list of (label, bed_regions_or_path).

    For each (state, annotation):
        a = bp(state ∩ ann) + 1
        b = bp(state) - a + 1
        c = bp(ann ∩ any_state) - a + 1
        d = total_bp - (a + b + c) + 1
    Fisher exact test (greater) on [[a, b], [c, d]].

    Returns a long-form DataFrame with columns:
        state, label, odds_ratio, p_value, q_value
    """
    from scipy.stats import fisher_exact
    from scipy.stats import false_discovery_control

    by_chrom = defaultdict(list)
    state_total = defaultdict(int)
    for chrom, s, e, name in segs:
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

        state_hit = _compute_overlap_bp(by_chrom, starts, ann_segs)
        ann_total = sum(state_hit.values())

        for st in states:
            a = state_hit.get(st, 0) + 1
            b = max(0, state_total[st] - a) + 1
            c = max(0, ann_total - a) + 1
            d = max(0, total_bp - (a + b + c)) + 1
            odds, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append({"state": st, "label": label,
                         "odds_ratio": odds, "p_value": pval})

    if not rows:
        return pd.DataFrame(columns=["state", "label", "odds_ratio", "p_value", "q_value"])

    df = pd.DataFrame(rows)
    df["q_value"] = false_discovery_control(df["p_value"].values)
    return df


def save_enrichment_table(enrich_df, outdir):
    """Save enrichment/enrichment.tsv (long-form) and pivoted odds_ratio table."""
    edir = os.path.join(outdir, "enrichment")
    os.makedirs(edir, exist_ok=True)
    enrich_df.to_csv(os.path.join(edir, "enrichment.tsv"),
                      sep="\t", index=False, float_format="%.4f")


def _column_minmax_scale(mat):
    """Per-column min-max scaling to [0, 1]."""
    scaled = mat.copy()
    for c in scaled.columns:
        col_range = scaled[c].max() - scaled[c].min()
        if col_range < 1e-10:
            scaled[c] = 1.0 / len(scaled[c])
        else:
            scaled[c] = (scaled[c] - scaled[c].min()) / col_range
    return scaled


def plot_enrichment(enrich_df, segs, outdir):
    """Plot enrichment.png: odds ratio heatmap with Genome %, per-column min-max scaled."""
    if enrich_df.empty:
        return
    edir = os.path.join(outdir, "enrichment")
    os.makedirs(edir, exist_ok=True)

    sorted_idx = sorted(enrich_df["state"].unique(), key=_natural_sort_key)
    sorted_cols = _reorder_annotations(sorted(enrich_df["label"].unique()))

    odds_mat = enrich_df.pivot(index="state", columns="label", values="odds_ratio")
    odds_mat = odds_mat.loc[sorted_idx, sorted_cols]

    # Add Genome % column
    state_bp = defaultdict(int)
    for _, s, e, name in segs:
        state_bp[name] += e - s
    total_bp = sum(state_bp.values())
    genome_pct = pd.Series(
        {st: 100.0 * state_bp.get(st, 0) / total_bp for st in sorted_idx},
        name="Genome %")
    odds_mat.insert(0, "Genome %", genome_pct)

    scaled = _column_minmax_scale(odds_mat)

    fig, ax = plt.subplots(figsize=(max(3, 0.4 * len(scaled.columns)),
                                    max(4, 0.35 * len(scaled))))
    ax.imshow(scaled.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(scaled.columns)))
    ax.set_xticklabels(scaled.columns, rotation=90)
    ax.set_yticks(range(len(scaled.index)))
    ax.set_yticklabels(scaled.index)
    ax.set_title("Functional enrichment")
    fig.tight_layout()
    fig.savefig(os.path.join(edir, "enrichment.png"), dpi=120)
    plt.close(fig)


# --- Transition matrix entropy --------------------------------------------

def build_transition_matrix(segs, bin_size, exclude_states=None):
    """Build empirical transition count matrix at bin resolution.

    Counts transitions between consecutive *bins* on the same chromosome,
    so self-transitions within long segments are properly represented.
    Segments whose state is in *exclude_states* are skipped.

    Returns (states, count_matrix, state_bp) where state_bp[i] = total bp
    in state i (for computing stationary distribution).
    """
    exclude = set(exclude_states or [])

    # Build per-chromosome bin arrays: {chrom: {bin_idx: state}}
    by_chrom = defaultdict(dict)
    for chrom, s, e, state in segs:
        if state in exclude:
            continue
        b0 = s // bin_size
        b1 = e // bin_size
        for b in range(b0, b1):
            by_chrom[chrom][b] = state

    all_states = sorted(set(state for _, _, _, state in segs if state not in exclude),
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
    """Compute total transition matrix entropy.

    For each state i, per-row entropy:
        H(i) = -sum_j A[i][j] * log2(A[i][j])

    Total entropy = sum_i pi[i] * H(i)
    where pi[i] = state_bp[i] / sum(state_bp)  (stationary distribution).

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

    total_H = np.dot(pi, H)
    return total_H, H, A, pi


def _build_seg_to_analysis_map(seg_paths, analysis_dir):
    """Map segmentation BED paths to their analysis subdirectories.

    Scans *analysis_dir* for subdirectories and matches each segmentation
    label to the best-matching subdir.  Returns {seg_path: subdir_path}.
    """
    if not analysis_dir or not os.path.isdir(analysis_dir):
        return {}
    subdirs = sorted([
        d for d in os.listdir(analysis_dir)
        if os.path.isdir(os.path.join(analysis_dir, d))
    ])
    mapping = {}
    for seg_path in seg_paths:
        label = os.path.basename(seg_path).replace(".bed", "")
        label_core = label.replace("_matched", "")
        best = None
        for sd in subdirs:
            if sd == "ref":
                if label.startswith("ENCFF"):
                    best = sd
                continue
            if sd in label_core or sd in label:
                best = sd
        if best is not None:
            mapping[seg_path] = os.path.join(analysis_dir, best)
    return mapping


def _compute_and_save_entropy(seg_paths, bin_size, outdir, exclude_states=None,
                              suffix="", seg_outdirs=None):
    """Compute transition entropy for each segmentation, save details + summary.

    *suffix* is appended to output filenames (e.g. "_no_quies").
    *seg_outdirs* maps seg_path → per-segmentation output directory.
    Returns list of {segmentation, total_entropy} dicts.
    """
    os.makedirs(outdir, exist_ok=True)
    seg_outdirs = seg_outdirs or {}
    results = []
    exclude_label = f" (excluding {', '.join(sorted(exclude_states))})" if exclude_states else ""

    for seg_path in seg_paths:
        segs = load_bed(seg_path)
        if not segs:
            print(f"  WARNING: empty segmentation {seg_path}", file=sys.stderr)
            continue

        states, counts, state_bp = build_transition_matrix(segs, bin_size, exclude_states)
        if not states:
            print(f"  WARNING: no states left after exclusion in {seg_path}", file=sys.stderr)
            continue
        total_H, H, A, pi = transition_entropy(states, counts, state_bp)

        label = os.path.basename(seg_path).replace(".bed", "")
        results.append({"segmentation": label, "total_entropy": total_H})
        print(f"  {label}{exclude_label}: total transition entropy = {total_H:.4f}")

        # Per-segmentation files go to analysis subdir when available
        seg_dir = seg_outdirs.get(seg_path)
        if seg_dir:
            seg_dir = os.path.join(seg_dir, "entropy")
            os.makedirs(seg_dir, exist_ok=True)
        else:
            seg_dir = outdir

        # Save per-state details
        detail_path = os.path.join(seg_dir, f"transition_entropy{suffix}.tsv")
        with open(detail_path, "w") as f:
            f.write("state\tstationary_prob\tentropy\tself_transition_prob\n")
            idx = {s: i for i, s in enumerate(states)}
            for s in states:
                i = idx[s]
                f.write(f"{s}\t{pi[i]:.6f}\t{H[i]:.4f}\t{A[i, i]:.6f}\n")
        print(f"  saved {detail_path}")

        # Save transition probability matrix
        trans_path = os.path.join(seg_dir, f"transition_matrix{suffix}.tsv")
        df_trans = pd.DataFrame(A, index=states, columns=states)
        df_trans.to_csv(trans_path, sep="\t", float_format="%.6f")
        print(f"  saved {trans_path}")

        # Plot transition matrix heatmap
        fig, ax = plt.subplots(figsize=(max(5, len(states) * 0.4),
                                        max(4, len(states) * 0.35)))
        sns.heatmap(A, xticklabels=states, yticklabels=states,
                    cmap="Blues", vmin=0, vmax=1, ax=ax,
                    linewidths=0.3, annot=True, fmt=".2f",
                    annot_kws={"fontsize": 6})
        ax.set_title(f"Transition matrix — {label}{exclude_label}\n"
                     f"(total entropy = {total_H:.4f})")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        fig.tight_layout()
        fig_path = os.path.join(seg_dir, f"transition_matrix{suffix}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  saved {fig_path}")

    return results


def _save_entropy_summary(results, outdir, suffix="", title_extra=""):
    """Save summary TSV and bar chart for a set of entropy results."""
    if not results:
        return
    summary_path = os.path.join(outdir, f"entropy_summary{suffix}.tsv")
    pd.DataFrame(results).to_csv(summary_path, sep="\t", index=False,
                                  float_format="%.4f")
    print(f"  saved {summary_path}")

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(max(5, len(df) * 0.5), 3.5))
    ax.bar(range(len(df)), df["total_entropy"])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["segmentation"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total transition matrix entropy (bits)")
    ax.set_title(f"Transition matrix entropy comparison{title_extra}")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(df["total_entropy"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig_path = os.path.join(outdir, f"entropy_summary{suffix}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  saved {fig_path}")


def _save_entropy_combined_plot(results_full, results_active, outdir):
    """Grouped bar chart comparing all-states vs excluding Quies/Het entropy."""
    if not results_full:
        return
    df_full = pd.DataFrame(results_full)
    df_active = pd.DataFrame(results_active) if results_active else pd.DataFrame()

    x = np.arange(len(df_full))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(5, len(df_full) * 0.5), 3.5))

    ax.bar(x - width / 2, df_full["total_entropy"], width,
           label="All states", color="#4878CF")
    if not df_active.empty:
        ax.bar(x + width / 2, df_active["total_entropy"], width,
               label="Excl. Quies/Het", color="#E8833A")

    ax.set_xticks(x)
    ax.set_xticklabels(df_full["segmentation"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total transition matrix entropy (bits)")
    ax.set_title("Transition matrix entropy comparison")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(outdir, "entropy_summary_combined.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  saved {fig_path}")


QUIESCENT_STATES = {"Quies", "Het"}


# --- Cohen's Kappa -------------------------------------------------------

def segmentation_to_bins(segs, bin_size):
    """Convert a segmentation to a dict: {chrom: {bin_index: state}}."""
    bins = defaultdict(dict)
    for chrom, s, e, state in segs:
        b0 = s // bin_size
        b1 = e // bin_size
        for b in range(b0, b1):
            bins[chrom][b] = state
    return bins


def compute_kappa(bins1, bins2):
    """Compute Cohen's Kappa between two bin-level segmentations.

    Returns (kappa, po, pe, n_bins, confusion_matrix_df).
    """
    common_chroms = set(bins1.keys()) & set(bins2.keys())
    labels1 = []
    labels2 = []
    for chrom in sorted(common_chroms):
        common_bins = set(bins1[chrom].keys()) & set(bins2[chrom].keys())
        for b in sorted(common_bins):
            labels1.append(bins1[chrom][b])
            labels2.append(bins2[chrom][b])

    n = len(labels1)
    if n == 0:
        return 0.0, 0.0, 0.0, 0, pd.DataFrame()

    labels1 = np.array(labels1)
    labels2 = np.array(labels2)

    all_states = sorted(set(labels1) | set(labels2), key=_natural_sort_key)
    state_idx = {s: i for i, s in enumerate(all_states)}
    k = len(all_states)

    conf = np.zeros((k, k), dtype=np.int64)
    for l1, l2 in zip(labels1, labels2):
        conf[state_idx[l1], state_idx[l2]] += 1

    po = np.diag(conf).sum() / n
    p1 = conf.sum(axis=1) / n
    p2 = conf.sum(axis=0) / n
    pe = np.dot(p1, p2)

    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0

    conf_df = pd.DataFrame(conf, index=all_states, columns=all_states)
    return kappa, po, pe, n, conf_df



def _seg_label(path):
    """Short human-readable label from a segmentation path."""
    return os.path.basename(path).replace(".bed", "")


def _load_emissions_tsv(path):
    """Load state_emissions.tsv → (states, marks, matrix) or None."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    states = df["state"].tolist()
    marks = [c for c in df.columns if c != "state"]
    mat = df[marks].values.astype(np.float64)
    return states, marks, mat


def _emission_cosine_similarity(states1, mat1, states2, mat2):
    """Compute optimal state correspondence by cosine similarity.

    Returns (best_avg_similarity, mapping_dict) where mapping maps
    states1 → states2 via Hungarian on cosine distance.
    """
    from scipy.optimize import linear_sum_assignment

    n1, n2 = len(states1), len(states2)
    cost = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            dot = np.dot(mat1[i], mat2[j])
            norm1 = np.linalg.norm(mat1[i])
            norm2 = np.linalg.norm(mat2[j])
            if norm1 > 0 and norm2 > 0:
                cost[i, j] = 1.0 - dot / (norm1 * norm2)
            else:
                cost[i, j] = 1.0

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    total_sim = 0.0
    for r, c in zip(row_ind, col_ind):
        mapping[states1[r]] = states2[c]
        total_sim += 1.0 - cost[r, c]
    avg_sim = total_sim / len(row_ind) if len(row_ind) > 0 else 0.0
    return avg_sim, mapping


def compare_all(args):
    """Pairwise comparison of all segmentations: kappa, jaccard, emission similarity."""
    from match import pair_overlap as match_pair_overlap
    from match import best_mapping as match_best_mapping
    from match import compare as match_compare

    os.makedirs(args.outdir, exist_ok=True)
    analysis_dir = getattr(args, "analysis_dir", None)
    seg_outdirs = _build_seg_to_analysis_map(args.seg, analysis_dir)

    paths = args.seg
    n = len(paths)
    labels = [_seg_label(p) for p in paths]

    # Load all segmentations
    print(f"  Loading {n} segmentations ...", file=sys.stderr)
    all_segs = []       # list of (chrom, start, end, name) for analyze.py functions
    all_segs_full = []  # list of (chrom, start, end, name, color) for match.py functions
    all_bins = []
    for p in paths:
        segs = load_bed(p)
        all_segs.append(segs)
        # Convert to match.py format (5-tuple with color)
        segs_full = []
        with open(p) as f:
            for line in f:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                parts = line.rstrip("\n").split("\t")
                chrom, s, e = parts[0], int(parts[1]), int(parts[2])
                name = parts[3] if len(parts) > 3 else "."
                color = parts[8] if len(parts) >= 9 else "0,0,0"
                segs_full.append((chrom, s, e, name, color))
        all_segs_full.append(segs_full)
        all_bins.append(segmentation_to_bins(segs, args.bin))

    # Load emissions where available
    emissions = {}
    for i, p in enumerate(paths):
        seg_dir = seg_outdirs.get(p)
        if not seg_dir:
            continue
        epath = os.path.join(seg_dir, "emissions", "state_emissions.tsv")
        result = _load_emissions_tsv(epath)
        if result is not None:
            emissions[i] = result

    # Pairwise comparisons
    kappa_mat = np.ones((n, n), dtype=np.float64)
    jaccard_mat = np.full((n, n), np.nan)
    em_sim_mat = np.full((n, n), np.nan)
    np.fill_diagonal(jaccard_mat, 1.0)
    np.fill_diagonal(em_sim_mat, 1.0)
    comparison_rows = []

    for i in range(n):
        for j in range(i + 1, n):
            row = {"seg1": labels[i], "seg2": labels[j]}

            # Kappa
            kappa, po, pe, n_bins, _ = compute_kappa(all_bins[i], all_bins[j])
            kappa_mat[i, j] = kappa
            kappa_mat[j, i] = kappa
            row.update(kappa=kappa, po=po, pe=pe, n_bins=n_bins)

            # Jaccard (via match.py) — produces per-pair heatmap + similarity
            overlap = match_pair_overlap(all_segs_full[i], all_segs_full[j])
            work_states = sorted({x[3] for x in all_segs_full[j]}, key=_natural_sort_key)
            ref_states = sorted({x[3] for x in all_segs_full[i]}, key=_natural_sort_key)
            mapping = match_best_mapping(overlap, work_states, ref_states)
            pair_dir = os.path.join(args.outdir, "pairs", f"{labels[i]}_vs_{labels[j]}")
            match_compare(all_segs_full[i], all_segs_full[j], overlap, mapping, pair_dir)

            # Read back similarity score
            sim_path = os.path.join(pair_dir, "similarity.txt")
            sim = 0.0
            if os.path.exists(sim_path):
                with open(sim_path) as f:
                    for line in f:
                        if "=" in line:
                            sim = float(line.split("=")[1].strip())
            jaccard_mat[i, j] = sim
            jaccard_mat[j, i] = sim
            row["jaccard_similarity"] = sim

            # Emission correspondence
            if i in emissions and j in emissions:
                st1, _, mat1 = emissions[i]
                st2, _, mat2 = emissions[j]
                avg_sim, em_mapping = _emission_cosine_similarity(st1, mat1, st2, mat2)
                em_sim_mat[i, j] = avg_sim
                em_sim_mat[j, i] = avg_sim
                row["emission_similarity"] = avg_sim
                row["emission_mapping"] = "; ".join(
                    f"{k}->{v}" for k, v in sorted(em_mapping.items()))

            comparison_rows.append(row)
            print(f"  {labels[i]} vs {labels[j]}: "
                  f"kappa={kappa:.4f}, jaccard={sim:.4f}"
                  + (f", emission={row.get('emission_similarity', 'N/A')}"
                     if 'emission_similarity' in row else ""))

    # Save all-pairs summary
    summary_path = os.path.join(args.outdir, "comparison_all_pairs.tsv")
    pd.DataFrame(comparison_rows).to_csv(summary_path, sep="\t", index=False,
                                          float_format="%.4f")
    print(f"  saved {summary_path}")

    # Save and plot kappa matrix
    kappa_df = pd.DataFrame(kappa_mat, index=labels, columns=labels)
    kappa_df.to_csv(os.path.join(args.outdir, "kappa_matrix.tsv"),
                     sep="\t", float_format="%.4f")
    _plot_sim_heatmap(kappa_df, "Pairwise Cohen's Kappa",
                      os.path.join(args.outdir, "kappa_heatmap.png"),
                      cmap="YlGnBu", cbar_label="Cohen's Kappa")

    # Save and plot jaccard similarity matrix
    jaccard_df = pd.DataFrame(jaccard_mat, index=labels, columns=labels)
    jaccard_df.to_csv(os.path.join(args.outdir, "jaccard_similarity_matrix.tsv"),
                       sep="\t", float_format="%.4f")
    _plot_sim_heatmap(jaccard_df, "Pairwise Jaccard similarity (overlap-matched)",
                      os.path.join(args.outdir, "jaccard_similarity_heatmap.png"),
                      cmap="YlOrRd", cbar_label="Similarity")

    # Save and plot emission similarity matrix (if available)
    if len(emissions) >= 2:
        em_df = pd.DataFrame(em_sim_mat, index=labels, columns=labels)
        em_df.to_csv(os.path.join(args.outdir, "emission_similarity_matrix.tsv"),
                      sep="\t", float_format="%.4f")
        _plot_sim_heatmap(em_df, "Pairwise emission correspondence",
                          os.path.join(args.outdir, "emission_similarity_heatmap.png"),
                          cmap="YlOrRd", cbar_label="Avg cosine similarity",
                          mask=np.isnan(em_sim_mat))

    # Per-segmentation rows saved to analysis subdirs
    for i, p in enumerate(paths):
        seg_dir = seg_outdirs.get(p)
        if not seg_dir:
            continue
        comp_dir = os.path.join(seg_dir, "comparison")
        os.makedirs(comp_dir, exist_ok=True)
        kappa_df.iloc[[i]].to_csv(
            os.path.join(comp_dir, "kappa_vs_all.tsv"), sep="\t", float_format="%.4f")
        jaccard_df.iloc[[i]].to_csv(
            os.path.join(comp_dir, "jaccard_vs_all.tsv"), sep="\t", float_format="%.4f")
        if i in emissions and len(emissions) >= 2:
            em_df.iloc[[i]].dropna(axis=1).to_csv(
                os.path.join(comp_dir, "emission_similarity_vs_all.tsv"),
                sep="\t", float_format="%.4f")


def _plot_sim_heatmap(df, title, path, cmap="YlGnBu", cbar_label="Value",
                      mask=None):
    """Plot an annotated similarity heatmap."""
    n = len(df)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.45), max(5, n * 0.4)))
    sns.heatmap(df, cmap=cmap, vmin=0, vmax=1, ax=ax,
                mask=mask, linewidths=0.5, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7}, cbar_kws={"label": cbar_label})
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# --- Segment length statistics --------------------------------------------

def compute_segment_stats(segs):
    """Compute segment length statistics from a segmentation.

    Returns a dict with: n_states, n_segments, min_length, max_length,
    mean_length, median_length (all in bp).
    """
    if not segs:
        return {}
    lengths = np.array([e - s for _, s, e, _ in segs])
    states = set(st for _, _, _, st in segs)
    return {
        "n_states": len(states),
        "n_segments": len(lengths),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
    }


def run_segment_stats(args):
    """Compute and save segment length statistics for each segmentation."""
    os.makedirs(args.outdir, exist_ok=True)
    analysis_dir = getattr(args, "analysis_dir", None)
    seg_outdirs = _build_seg_to_analysis_map(args.seg, analysis_dir)
    results = []

    for seg_path in args.seg:
        segs = load_bed(seg_path)
        if not segs:
            print(f"  WARNING: empty segmentation {seg_path}", file=sys.stderr)
            continue
        stats = compute_segment_stats(segs)
        label = os.path.basename(seg_path).replace(".bed", "")
        stats["segmentation"] = label
        results.append(stats)
        print(f"  {label}: {stats['n_states']} states, "
              f"{stats['n_segments']} segments, "
              f"lengths [{stats['min_length']}, {stats['max_length']}], "
              f"mean={stats['mean_length']:.0f}, median={stats['median_length']:.0f}")

        # Per-segmentation stats to analysis subdir
        seg_dir = seg_outdirs.get(seg_path)
        if seg_dir:
            stats_dir = os.path.join(seg_dir, "segment_stats")
            os.makedirs(stats_dir, exist_ok=True)
            row_df = pd.DataFrame([stats])
            row_df.to_csv(os.path.join(stats_dir, "segment_stats.tsv"),
                           sep="\t", index=False, float_format="%.1f")
            print(f"  saved {stats_dir}/segment_stats.tsv")

    if not results:
        return

    df = pd.DataFrame(results)
    col_order = ["segmentation", "n_states", "n_segments",
                 "min_length", "max_length", "mean_length", "median_length"]
    df = df[col_order]
    summary_path = os.path.join(args.outdir, "segment_stats.tsv")
    df.to_csv(summary_path, sep="\t", index=False, float_format="%.1f")
    print(f"  saved {summary_path}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    x = np.arange(len(df))
    xlabels = df["segmentation"]

    metrics = [
        ("n_states", "Total number of states", "Count"),
        ("n_segments", "Total number of segments", "Count"),
        ("min_length", "Min segment length", "bp"),
        ("max_length", "Max segment length", "bp"),
        ("mean_length", "Mean segment length", "bp"),
        ("median_length", "Median segment length", "bp"),
    ]

    for idx, (col, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        vals = df[col].values
        ax.bar(x, vals, color="#4878CF", edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=55, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            fmt = f"{v:.0f}" if v == int(v) else f"{v:.1f}"
            ax.text(i, v + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                    fmt, ha="center", va="bottom", fontsize=6)

    fig.suptitle("Segment length statistics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(args.outdir, "segment_stats.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  saved {fig_path}")



# --- CLI ------------------------------------------------------------------

def _run_single(args):
    """Per-segmentation analysis: report, segment lengths, emissions, enrichment."""
    os.makedirs(args.outdir, exist_ok=True)
    seg_path = args.seg[0]
    segs = load_bed(seg_path)

    if not getattr(args, "emissions_only", False):
        save_report(segs, args.outdir)
        plot_segment_lengths(segs, args.outdir)

    inputs = expand_globs(args.inputs or [])
    if inputs:
        states, marks, emission_mat = compute_emissions(segs, inputs, args.bin)
        save_emissions_table(states, marks, emission_mat, args.outdir)
        plot_emissions(states, marks, emission_mat, args.outdir)

    annotation_items = []
    for p in expand_globs(args.annotations or []):
        if os.path.exists(p):
            label = os.path.basename(p).replace(".bed.gz", "").replace(".bed", "")
            annotation_items.append((label, p))

    if args.rnaseq and args.gtf:
        rna_annotations = make_expressed_annotations(args.rnaseq, args.gtf)
        annotation_items.extend(rna_annotations)

    if annotation_items:
        enrich_df = compute_enrichment(segs, annotation_items)
        save_enrichment_table(enrich_df, args.outdir)
        plot_enrichment(enrich_df, segs, args.outdir)


def _run_multi(args):
    """Cross-segmentation metrics: entropy, kappa, segment stats."""
    analysis_dir = args.analysis_dir or args.outdir

    # Entropy
    entropy_dir = os.path.join(args.outdir, "comparison")
    seg_outdirs = _build_seg_to_analysis_map(args.seg, analysis_dir)
    results_full = _compute_and_save_entropy(args.seg, args.bin, entropy_dir,
                                             seg_outdirs=seg_outdirs)
    _save_entropy_summary(results_full, entropy_dir)
    excl = QUIESCENT_STATES
    results_active = _compute_and_save_entropy(
        args.seg, args.bin, entropy_dir, exclude_states=excl, suffix="_no_quies",
        seg_outdirs=seg_outdirs)
    _save_entropy_summary(
        results_active, entropy_dir, suffix="_no_quies",
        title_extra=f"\n(excluding {', '.join(sorted(excl))})")
    _save_entropy_combined_plot(results_full, results_active, entropy_dir)

    # Pairwise comparison (kappa, jaccard, emission correspondence)
    compare_dir = os.path.join(args.outdir, "comparison")
    compare_args = types.SimpleNamespace(
        seg=args.seg, bin=args.bin, outdir=compare_dir,
        analysis_dir=analysis_dir)
    compare_all(compare_args)

    # Segment stats
    stats_dir = os.path.join(args.outdir, "comparison")
    stats_args = types.SimpleNamespace(
        seg=args.seg, outdir=stats_dir,
        analysis_dir=analysis_dir)
    run_segment_stats(stats_args)


def main():
    ap = argparse.ArgumentParser(
        description="Segmentation analysis and quality metrics.\n\n"
                    "Single --seg: per-segmentation report, emissions, enrichment.\n"
                    "Multiple --seg: cross-segmentation entropy, kappa, segment stats.")
    ap.add_argument("--seg", nargs="+", required=True,
                    help="One or more segmentation BED files")
    ap.add_argument("--bin", type=int, required=True, help="Bin size in bp")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--inputs", nargs="*",
                    help="ChromHMM binary input files (for emissions)")
    ap.add_argument("--annotations", nargs="*",
                    help="Annotation BED(.gz) files (for enrichment)")
    ap.add_argument("--rnaseq", default=None,
                    help="ENCODE RNA-seq quantification TSV")
    ap.add_argument("--gtf", default=None,
                    help="GENCODE GTF(.gz) gene annotation")
    ap.add_argument("--emissions-only", action="store_true", dest="emissions_only",
                    help="Only compute emissions and enrichment (skip report/lengths)")
    ap.add_argument("--analysis-dir", default=None, dest="analysis_dir",
                    help="Analysis root dir for per-segmentation metric output")
    args = ap.parse_args()

    if len(args.seg) == 1:
        _run_single(args)
    else:
        _run_multi(args)


if __name__ == "__main__":
    main()
