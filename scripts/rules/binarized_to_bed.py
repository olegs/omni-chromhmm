#!/usr/bin/env python3
"""Extract per-mark BED files from ChromHMM binarized per-chromosome files.

For each histone modification, bins where the mark is present (value=1) are
merged into contiguous intervals and written as a separate BED file.

Usage:
    binarized_to_bed.py --bin 200 --outdir OUT  chr1_binary.txt chr2_binary.txt ...
"""

import argparse
import os
import sys

# Add scripts/analysis to sys.path so analyze.py can be imported.
_analysis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
if _analysis_dir not in sys.path:
    sys.path.insert(0, _analysis_dir)

from analyze import load_binary

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
