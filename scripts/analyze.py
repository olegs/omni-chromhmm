#!/usr/bin/env python3
# Analysis plots for a ChromHMM-style segmentation, mirroring the three figures
# in the PDF's "States analysis" slides:
#   * average segment length per state
#   * state feature means across marks  (if --inputs is given: ChromHMM binary files)
#   * functional enrichment vs. annotation beds + RNA-seq  (if --annotations is given)
#
# Usage mirrors the PDF:
#   analyze.py --seg SEG.bed --bin BIN --outdir OUT [--inputs chromhmm/*.txt(.gz)]
#              [--annotations COORDS/*.bed.gz RNA_TSV ...]

import argparse
import glob
import gzip
import os
import sys
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_bed(path):
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


def load_binary(path):
    with open_text(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        chrom = head[1] if len(head) > 1 else "chrUnknown"
        marks = f.readline().rstrip("\n").split("\t")
        rows = [list(map(int, line.rstrip("\n").split("\t")))
                for line in f if line.strip()]
    return chrom, marks, np.asarray(rows, dtype=np.int8)


# --- plots ---------------------------------------------------------------

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


def plot_feature_means(segs, inputs, bin_size, outdir):
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

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(marks)),
                                    max(4, 0.35 * len(states))))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(marks))); ax.set_xticklabels(marks, rotation=90)
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states)
    ax.set_title("State feature means")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "state_feature_means.png"), dpi=120)
    plt.close(fig)


def plot_enrichment(segs, annotation_paths, outdir):
    by_chrom = defaultdict(list)
    state_total = defaultdict(int)
    for chrom, s, e, name in segs:
        by_chrom[chrom].append((s, e, name))
        state_total[name] += e - s
    for chrom in by_chrom:
        by_chrom[chrom].sort()
    starts = {c: [s for s, _, _ in v] for c, v in by_chrom.items()}

    states = sorted(state_total)
    anns = annotation_paths
    mat = np.zeros((len(states), len(anns)))

    for j, ann_path in enumerate(anns):
        try:
            ann_segs = load_bed(ann_path)
        except Exception as err:
            print(f"skipping annotation {ann_path}: {err}", file=sys.stderr)
            continue
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
        for i, st in enumerate(states):
            if state_total[st] > 0:
                mat[i, j] = state_hit[st] / state_total[st]

    ann_labels = [os.path.basename(p).replace(".bed.gz", "").replace(".bed", "")
                  for p in anns]
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(ann_labels)),
                                    max(4, 0.35 * len(states))))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(ann_labels))); ax.set_xticklabels(ann_labels, rotation=90)
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states)
    ax.set_title("Functional enrichment: overlap")
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--bin", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--annotations", nargs="*", default=[])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    segs = load_bed(args.seg)
    plot_segment_lengths(segs, args.outdir)

    inputs = expand_globs(args.inputs)
    if inputs:
        plot_feature_means(segs, inputs, args.bin, args.outdir)

    anns = [p for p in expand_globs(args.annotations) if os.path.exists(p)]
    if anns:
        plot_enrichment(segs, anns, args.outdir)


if __name__ == "__main__":
    main()
