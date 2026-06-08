#!/usr/bin/env python3
"""Extract per-mark BED files from ChromHMM binarized per-chromosome files.

For each histone modification, bins where the mark is present (value=1) are
merged into contiguous intervals and written as a separate BED file.

Usage:
    binarized_to_bed.py --bin 200 --outdir OUT  chr1_binary.txt chr2_binary.txt ...
"""

import argparse
import gzip
import os
import sys


def load_binary(path):
    """Read one binarized file. Returns (chrom, marks, rows) where rows is list of int lists."""
    rows = []
    chrom = "chrUnknown"
    marks = []
    try:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as f:
            line1 = f.readline()
            if not line1:
                return chrom, marks, rows
            cell_chr = line1.rstrip("\n").split("\t")
            chrom = cell_chr[1] if len(cell_chr) > 1 else "chrUnknown"
            
            line2 = f.readline()
            if not line2:
                return chrom, marks, rows
            marks = line2.rstrip("\n").split("\t")
            
            for line in f:
                line = line.rstrip("\n")
                if line:
                    rows.append(list(map(int, line.split("\t"))))
    except (EOFError, gzip.BadGzipFile) as e:
        print(f"Warning: Corrupted gzip file {path}: {e}. Data might be incomplete.", file=sys.stderr)
    return chrom, marks, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=int, required=True, help="Bin size in bp")
    ap.add_argument("--outdir", required=True, help="Output directory for per-mark BED files")
    ap.add_argument("inputs", nargs="+", help="Binarized .txt(.gz) files")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    bin_size = args.bin

    # Collect intervals per mark across all chromosomes
    # mark_intervals[mark] = [(chrom, start, end), ...]
    mark_intervals = {}
    marks_order = None

    for path in sorted(args.inputs):
        chrom, marks, rows = load_binary(path)
        if marks_order is None:
            marks_order = marks
        for col_idx, mark in enumerate(marks):
            if mark not in mark_intervals:
                mark_intervals[mark] = []
            # Scan bins and merge consecutive 1s into intervals
            in_region = False
            region_start = 0
            for i, row in enumerate(rows):
                val = row[col_idx]
                if val == 1 and not in_region:
                    region_start = i * bin_size
                    in_region = True
                elif val == 0 and in_region:
                    mark_intervals[mark].append((chrom, region_start, i * bin_size))
                    in_region = False
            # Close last region
            if in_region:
                mark_intervals[mark].append((chrom, region_start, len(rows) * bin_size))

    # Write per-mark BED files
    for mark in (marks_order or sorted(mark_intervals)):
        intervals = mark_intervals.get(mark, [])
        out_path = os.path.join(args.outdir, f"{mark}.bed")
        with open(out_path, "w") as f:
            for chrom, start, end in intervals:
                f.write(f"{chrom}\t{start}\t{end}\t{mark}\n")
        print(f"  {mark}: {len(intervals)} regions -> {out_path}")


if __name__ == "__main__":
    main()
