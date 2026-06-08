#!/usr/bin/env python3
# Cluster binary ChromHMM inputs into N states using KMeans.
# Emits BED to stdout; logs go to stderr.

import argparse
import gzip
import numpy as np
import sys
from pathlib import Path


def load_binary(path):
    """Read one ChromHMM BinarizeBam / multiinter file -> (chrom, marks, 0/1 matrix)."""
    rows = []
    chrom = "chrUnknown"
    marks = []
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as f:
            line1 = f.readline()
            if not line1:
                return chrom, marks, np.asarray(rows, dtype=np.int8)
            cell_chr = line1.rstrip("\n").split("\t")
            chrom = cell_chr[1] if len(cell_chr) > 1 else "chrUnknown"
            
            line2 = f.readline()
            if not line2:
                return chrom, marks, np.asarray(rows, dtype=np.int8)
            marks = line2.rstrip("\n").split("\t")
            
            for line in f:
                if line.strip():
                    rows.append(list(map(int, line.rstrip("\n").split("\t"))))
    except (EOFError, gzip.BadGzipFile) as e:
        print(f"Warning: Corrupted gzip file {path}: {e}. Data might be incomplete.", file=sys.stderr)
    return chrom, marks, np.asarray(rows, dtype=np.int8)


def segments_from_labels(chrom, labels, bin_size):
    """Collapse consecutive identical labels into (chrom, start, end, state_id) runs."""
    if len(labels) == 0:
        return
    start = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            yield chrom, start * bin_size, i * bin_size, int(cur) + 1
            start = i
            cur = labels[i]
    yield chrom, start * bin_size, len(labels) * bin_size, int(cur) + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=int, required=True, help="Bin size in bp")
    ap.add_argument("--states", type=int, required=True, help="Number of clusters")
    ap.add_argument("--inputs", nargs="+", required=True, help="Binary ChromHMM .txt(.gz) files")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inputs = sorted(args.inputs)
    print(f"Loading data files {len(inputs)}", file=sys.stderr)

    per_chrom = []
    marks_ref = None
    for p in inputs:
        chrom, marks, data = load_binary(p)
        if marks_ref is None:
            marks_ref = marks
        elif marks != marks_ref:
            sys.exit(f"Mark order mismatch in {p}: {marks} vs {marks_ref}")
        per_chrom.append((chrom, data))

    X = np.vstack([d for _, d in per_chrom])
    print(f"Total length {X.shape[0]}", file=sys.stderr)

    from sklearn.cluster import KMeans
    print("Fitting KMeans", file=sys.stderr)
    model = KMeans(n_clusters=args.states, random_state=args.seed)
    labels = model.fit_predict(X)

    detected = len(set(labels.tolist()))
    print(f"Detected states {detected}!", file=sys.stderr)

    offset = 0
    for chrom, data in per_chrom:
        n = data.shape[0]
        for chrom_, s, e, sid in segments_from_labels(chrom, labels[offset:offset + n], args.bin):
            print(f"{chrom_}\t{s}\t{e}\tE{sid}\t0\t.\t{s}\t{e}\t0,0,0")
        offset += n


if __name__ == "__main__":
    main()
