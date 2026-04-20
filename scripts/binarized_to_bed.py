#!/usr/bin/env python3
# Convert ChromHMM binarized files to BED format.
# Each 200-bp bin becomes a BED line with binary mark values as extra columns.
# Reads per-chromosome binarized files and writes a single merged BED to stdout.

import argparse
import gzip
import sys


def convert(path, bin_size):
    """Read one binarized file and yield BED lines."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        cell_chr = f.readline().rstrip("\n").split("\t")
        chrom = cell_chr[1]
        marks = f.readline().rstrip("\n").split("\t")
        for i, line in enumerate(f):
            vals = line.rstrip("\n")
            if vals:
                start = i * bin_size
                end = start + bin_size
                yield f"{chrom}\t{start}\t{end}\t{vals}"
    return marks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=int, required=True, help="Bin size in bp")
    ap.add_argument("inputs", nargs="+", help="Binarized .txt(.gz) files")
    args = ap.parse_args()

    # Read marks header from first file for the comment line
    opener = gzip.open if args.inputs[0].endswith(".gz") else open
    with opener(args.inputs[0], "rt") as f:
        f.readline()
        marks = f.readline().rstrip("\n").split("\t")
    print(f"#chrom\tstart\tend\t" + "\t".join(marks))

    for path in sorted(args.inputs):
        for line in convert(path, args.bin):
            print(line)


if __name__ == "__main__":
    main()
