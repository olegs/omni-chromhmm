#!/usr/bin/env python3
"""Compute per-state emissions and save to .npz."""

import argparse
import sys
from collections import defaultdict

import numpy as np

from analysis.analyze import load_binary, _natural_sort_key, _load_seg_full as load_bed


def compute_emissions_from_bigwigs(segs, bigwig_paths, mark_names, bin_size):
    """Compute mean signal per state per mark from bigwig files.

    Returns (sorted_states, marks, matrix) where matrix is (n_states, n_marks).
    """
    import pyBigWig

    n_marks = len(mark_names)
    bws = [pyBigWig.open(p) for p in bigwig_paths]

    sums   = defaultdict(lambda: np.zeros(n_marks, dtype=np.float64))
    counts = defaultdict(int)
    chrom_sizes = [bw.chroms() for bw in bws]

    for chrom, s, e, name, _ in segs:
        n_bins = (e - s) // bin_size
        if n_bins == 0:
            continue
        for mi, bw in enumerate(bws):
            if chrom not in chrom_sizes[mi]:
                continue
            e_c = min(e, chrom_sizes[mi][chrom])
            if s >= e_c:
                continue
            val = bw.stats(chrom, s, e_c, type='mean', nBins=1)[0]
            if val is None:
                continue
            sums[name][mi] += val * (e - s)
        counts[name] += n_bins * bin_size

    for bw in bws:
        bw.close()

    states = sorted(sums.keys(), key=_natural_sort_key)
    mat = np.zeros((len(states), n_marks), dtype=np.float64)
    for i, st in enumerate(states):
        if counts[st] > 0:
            mat[i] = sums[st] / counts[st]
    return states, mark_names, mat


def compute_emissions_from_binarized(segs, binary_paths, bin_size):
    """Compute state emission matrix (states x marks) from binarized files."""
    by_chrom, marks = {}, None
    for p in sorted(binary_paths):
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
    return states, marks, mat


def _save_emissions_npz(path, states, marks, mat):
    np.savez_compressed(path, states=np.array(states), marks=np.array(marks), mat=mat)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bed",     required=True, help="Segmentation BED file")
    ap.add_argument("--bigwigs", nargs="+", default=None,
                    help="Bigwig files (one per mark, same order as --marks)")
    ap.add_argument("--binaries", nargs="+", default=None,
                    help="Binarized files (ChromHMM BinarizeBam output)")
    ap.add_argument("--marks",   nargs="+", default=None,
                    help="Mark names corresponding to bigwig files")
    ap.add_argument("--bin",     type=int, default=100, help="Bin size (default: 100)")
    ap.add_argument("--output",  required=True, help="Output .npz file")

    args = ap.parse_args()

    segs = load_bed(args.bed)
    if args.bigwigs:
        if not args.marks:
            print("Error: --marks must be provided with --bigwigs", file=sys.stderr)
            sys.exit(1)
        states, marks, mat = compute_emissions_from_bigwigs(
            segs, args.bigwigs, args.marks, args.bin)
    elif args.binaries:
        states, marks, mat = compute_emissions_from_binarized(
            segs, args.binaries, args.bin)
    else:
        print("Error: either --bigwigs or --binaries must be provided", file=sys.stderr)
        sys.exit(1)

    _save_emissions_npz(args.output, states, marks, mat)


if __name__ == "__main__":
    main()
