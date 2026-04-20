#!/usr/bin/env python3
# Unified analysis for any ChromHMM-style segmentation:
#   1. Report: state count, segment counts, average/median lengths  (report.tsv)
#   2. Segment length bar chart                                     (segment_length.png)
#   3. State emissions heatmap + table                              (state_emissions.{png,tsv})
#   4. Functional enrichment heatmap + table                        (enrichment.{png,tsv})
#
# Also provides shared helpers used by analyze_downloaded.py and analyze_matched.py:
#   - BED reading (load_bed, load_bed_df)
#   - Multi-sample violin / coverage / heatmap / stats plots
#
# Usage:
#   analyze.py --seg SEG.bed --bin BIN --outdir OUT \
#       [--inputs chromhmm/*.txt(.gz)] \
#       [--annotations COORDS/*.bed.gz] \
#       [--rnaseq RNA.tsv --gene-info gene_info.gz --gtf annotation.gtf.gz]

import argparse
import csv
import glob
import gzip
import os
import sys
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


def sorted_states(df):
    """Sort states by leading numeric prefix, then alphabetically."""
    return sorted(df["state"].unique(),
                  key=lambda s: (s.split("_")[0].lstrip("Ee"), s))


# --- multi-sample plots (shared) ----------------------------------------

def plot_per_state_violin(df, out_path, title=None):
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
    if title is None:
        title = f"Segment length per state ({df['sample'].nunique()} samples)"
    ax.set_title(title)
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
    if title is None:
        title = f"Total genomic coverage per state ({df['sample'].nunique()} samples)"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_overall_per_sample(df, out_path, title=None, short_names=False):
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

    labels = samples
    if short_names:
        labels = ["/".join(s.split("/")[-2:]) if "/" in s else s for s in samples]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7 if short_names else 8)
    ax.set_xlabel("Sample")
    ax.set_ylabel("log10(segment length + 1)  [bp]")
    if title is None:
        title = "Overall segment length per sample"
    ax.set_title(title)
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

    fig, ax = plt.subplots(figsize=(max(12, len(cols) * 0.65),
                                    max(5, len(pivot) * 0.45)))
    sns.heatmap(np.log10(pivot + 1), ax=ax, cmap="YlOrRd",
                linewidths=0.3, annot=True, fmt=".1f",
                cbar_kws={"label": "log10(median length + 1)"})
    if title is None:
        title = "Median segment length [log10 bp]"
    ax.set_title(title)
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

def load_expressed_entrez_ids(rnaseq_path, tpm_threshold=1.0):
    """Parse ENCODE RNA-seq quantification TSV; return set of expressed Entrez gene IDs."""
    expressed = set()
    with open(rnaseq_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                tpm = float(row["TPM"])
                gene_id = row["gene_id"].strip()
                if tpm >= tpm_threshold:
                    expressed.add(gene_id)
            except (ValueError, KeyError):
                continue
    return expressed


def load_entrez_to_symbol(gene_info_path):
    """Parse NCBI gene_info file; return dict mapping Entrez gene ID (str) -> gene symbol."""
    mapping = {}
    with open_text(gene_info_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            # Columns: tax_id, GeneID, Symbol, ...
            mapping[cols[1]] = cols[2]
    return mapping


def load_expressed_gene_coords(gtf_path, expressed_symbols):
    """Parse GENCODE GTF for gene-level entries; return gene bodies and TSS for expressed genes.

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
            gene_name = None
            for attr in attrs.split(";"):
                attr = attr.strip()
                if attr.startswith("gene_name"):
                    gene_name = attr.split('"')[1] if '"' in attr else attr.split()[-1]
                    break
            if gene_name is None or gene_name not in expressed_symbols:
                continue
            chrom = cols[0]
            start = int(cols[3]) - 1  # GTF is 1-based
            end = int(cols[4])
            strand = cols[6]
            gene_bodies.append((chrom, start, end, gene_name))
            tss = start if strand == "+" else end - 1
            tss_regions.append((chrom, tss, tss + 1, gene_name))
    return gene_bodies, tss_regions


def make_expressed_annotations(rnaseq_path, gene_info_path, gtf_path):
    """Build expressed gene body and TSS BED annotations from RNA-seq + gene annotation.

    Returns list of (label, bed_regions) pairs.
    """
    expressed_ids = load_expressed_entrez_ids(rnaseq_path)
    print(f"  RNA-seq: {len(expressed_ids)} expressed genes (TPM >= 1)", file=sys.stderr)

    entrez_to_sym = load_entrez_to_symbol(gene_info_path)
    expressed_symbols = {entrez_to_sym[gid] for gid in expressed_ids if gid in entrez_to_sym}
    print(f"  Mapped to {len(expressed_symbols)} gene symbols", file=sys.stderr)

    gene_bodies, tss_regions = load_expressed_gene_coords(gtf_path, expressed_symbols)
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
    states = sorted(lengths)

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
    states = sorted(lengths)
    means = [np.mean(lengths[s]) for s in states]

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(states)), 4))
    ax.bar(states, means)
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=90)
    ax.set_title("Average segment length per state")
    ax.set_xlabel("state")
    ax.set_ylabel("bp")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "segment_length.png"), dpi=120)
    plt.close(fig)


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

    states = sorted(sums)
    mat = np.array([sums[s] / max(counts[s], 1) for s in states])
    return states, marks, mat


def save_emissions_table(states, marks, mat, outdir):
    """Save state_emissions.tsv: rows=states, cols=marks."""
    path = os.path.join(outdir, "state_emissions.tsv")
    with open(path, "w") as f:
        f.write("state\t" + "\t".join(marks) + "\n")
        for i, st in enumerate(states):
            vals = "\t".join(f"{v:.4f}" for v in mat[i])
            f.write(f"{st}\t{vals}\n")


def plot_emissions(states, marks, mat, outdir):
    """Plot state_emissions.png heatmap."""
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(marks)),
                                    max(4, 0.35 * len(states))))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(marks))); ax.set_xticklabels(marks, rotation=90)
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states)
    ax.set_title("State emissions")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "state_emissions.png"), dpi=120)
    plt.close(fig)


def compute_enrichment(segs, annotation_items):
    """Compute enrichment matrix.

    annotation_items: list of (label, bed_regions_or_path) where bed_regions_or_path
    is either a file path (str) or a list of (chrom, start, end, name) tuples.

    Returns (states, labels, matrix).
    """
    by_chrom = defaultdict(list)
    state_total = defaultdict(int)
    for chrom, s, e, name in segs:
        by_chrom[chrom].append((s, e, name))
        state_total[name] += e - s
    for chrom in by_chrom:
        by_chrom[chrom].sort()
    starts = {c: [s for s, _, _ in v] for c, v in by_chrom.items()}

    states = sorted(state_total)
    labels = []
    mat_cols = []

    for label, bed_data in annotation_items:
        if isinstance(bed_data, str):
            try:
                ann_segs = load_bed(bed_data)
            except Exception as err:
                print(f"skipping annotation {bed_data}: {err}", file=sys.stderr)
                continue
        else:
            ann_segs = bed_data

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

        col = np.array([state_hit[st] / max(state_total[st], 1) for st in states])
        labels.append(label)
        mat_cols.append(col)

    mat = np.column_stack(mat_cols) if mat_cols else np.zeros((len(states), 0))
    return states, labels, mat


def save_enrichment_table(states, labels, mat, outdir):
    """Save enrichment.tsv: rows=states, cols=annotation labels."""
    path = os.path.join(outdir, "enrichment.tsv")
    with open(path, "w") as f:
        f.write("state\t" + "\t".join(labels) + "\n")
        for i, st in enumerate(states):
            vals = "\t".join(f"{v:.4f}" for v in mat[i])
            f.write(f"{st}\t{vals}\n")


def plot_enrichment(states, labels, mat, outdir):
    """Plot enrichment.png heatmap."""
    if mat.shape[1] == 0:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(labels)),
                                    max(4, 0.35 * len(states))))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states)
    ax.set_title("Functional enrichment: overlap fraction")
    fig.colorbar(im, ax=ax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if mat[i, j] < 0.5 else "white")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "enrichment.png"), dpi=120)
    plt.close(fig)


# --- main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Unified analysis for any ChromHMM-style segmentation.")
    ap.add_argument("--seg", required=True, help="Segmentation BED file")
    ap.add_argument("--bin", type=int, required=True, help="Bin size in bp")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--inputs", nargs="*", default=[],
                    help="ChromHMM binary input files (for emission computation)")
    ap.add_argument("--annotations", nargs="*", default=[],
                    help="Annotation BED(.gz) files for enrichment (e.g. COORDS/*.bed.gz)")
    ap.add_argument("--rnaseq", default=None,
                    help="ENCODE RNA-seq quantification TSV")
    ap.add_argument("--gene-info", default=None,
                    help="NCBI gene_info(.gz) file for Entrez ID -> symbol mapping")
    ap.add_argument("--gtf", default=None,
                    help="GENCODE GTF(.gz) gene annotation for coordinates")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    segs = load_bed(args.seg)

    # 1. Report
    save_report(segs, args.outdir)

    # 2. Segment length plot
    plot_segment_lengths(segs, args.outdir)

    # 3. State emissions
    inputs = expand_globs(args.inputs)
    if inputs:
        states, marks, emission_mat = compute_emissions(segs, inputs, args.bin)
        save_emissions_table(states, marks, emission_mat, args.outdir)
        plot_emissions(states, marks, emission_mat, args.outdir)

    # 4. Enrichment
    annotation_items = []
    for p in expand_globs(args.annotations):
        if os.path.exists(p):
            label = os.path.basename(p).replace(".bed.gz", "").replace(".bed", "")
            annotation_items.append((label, p))

    if args.rnaseq and args.gene_info and args.gtf:
        rna_annotations = make_expressed_annotations(args.rnaseq, args.gene_info, args.gtf)
        annotation_items.extend(rna_annotations)

    if annotation_items:
        states, labels, enr_mat = compute_enrichment(segs, annotation_items)
        save_enrichment_table(states, labels, enr_mat, args.outdir)
        plot_enrichment(states, labels, enr_mat, args.outdir)


if __name__ == "__main__":
    main()
