#!/usr/bin/env python3
import argparse
import gzip
import numpy as np
import sys
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_chrom_sizes(path):
    chroms = []
    sizes = {}
    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.split()
            if len(parts) < 2: continue
            name, size = parts[0], int(parts[1])
            if '_' in name: continue
            chroms.append(name)
            sizes[name] = size
    return chroms, sizes

def binarize_peaks(peak_files_groups, chroms, sizes, bin_size, marks):
    """
    peak_files_groups: list of lists of peak files (one list per mark)
    """
    num_marks = len(marks)
    data = {chrom: np.zeros(((sizes[chrom] + bin_size - 1) // bin_size, num_marks), dtype=np.int8) 
            for chrom in chroms}
    
    for m_idx, peak_files in enumerate(peak_files_groups):
        for peak_file in peak_files:
            if peak_file == "NONE" or not peak_file:
                continue
            # Skip common non-peak files
            if any(peak_file.endswith(ext) for ext in [".png", ".pdf", ".jpg", ".jpeg", ".log", ".bw", ".bigWig"]):
                print(f"Skipping non-peak file {peak_file}", file=sys.stderr)
                continue
            if not os.path.exists(peak_file):
                print(f"Warning: Peak file {peak_file} not found. Skipping.", file=sys.stderr)
                continue
            print(f"Binarizing {peak_file} for mark {marks[m_idx]}...", file=sys.stderr)
            
            opener = gzip.open if peak_file.endswith(".gz") else open
            mode = "rt" if peak_file.endswith(".gz") else "r"
            
            try:
                with opener(peak_file, mode) as f:
                    for line in f:
                        if not line.startswith('chr'): continue
                        parts = line.split()
                        if len(parts) < 3: continue
                        chrom, start, end = parts[0], int(parts[1]), int(parts[2])
                        if chrom not in data: continue
                        
                        b_start = start // bin_size
                        b_end = end // bin_size
                        
                        if b_end > b_start:
                            b_end = min(b_end, data[chrom].shape[0])
                            data[chrom][b_start:b_end, m_idx] = 1
            except Exception as e:
                print(f"Error reading {peak_file}: {e}", file=sys.stderr)
                    
    return [(chrom, data[chrom]) for chrom in chroms]

def write_binary_files(per_chrom, marks, cell, outdir):
    os.makedirs(outdir, exist_ok=True)
    for chrom, matrix in per_chrom:
        outfile = os.path.join(outdir, f"{cell}_{chrom}_binary.txt.gz")
        print(f"Writing {outfile}...", file=sys.stderr)
        with gzip.open(outfile, "wt") as f:
            f.write(f"{cell}\t{chrom}\n")
            f.write("\t".join(marks) + "\n")
            for row in matrix:
                f.write("\t".join(map(str, row)) + "\n")

def segments_from_labels(chrom, labels, bin_size):
    if len(labels) == 0: return
    start = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            yield chrom, start * bin_size, i * bin_size, int(cur) + 1
            start = i
            cur = labels[i]
    yield chrom, start * bin_size, len(labels) * bin_size, int(cur) + 1

def main():
    parser = argparse.ArgumentParser(description="Binarize peaks and run KMeans clustering.")
    parser.add_argument("--bin", type=int, required=True, help="Bin size")
    parser.add_argument("--chromsizes", required=True, help="Path to chrom.sizes file")
    parser.add_argument("--marks", required=True, help="Comma-separated list of marks")
    parser.add_argument("--peaks", nargs="+", required=True, help="Peak files (use commas to group multiple files per mark, matching --marks order)")
    parser.add_argument("--states", type=int, default=15, help="Number of KMeans states")
    parser.add_argument("--cell", default="cell", help="Cell name for binary files")
    parser.add_argument("--out", help="Output BED file (stdout if omitted)")
    parser.add_argument("--save-binary", help="Optional directory to save ChromHMM binary files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for KMeans")
    
    args = parser.parse_args()
    
    marks = args.marks.split(",")
    # peaks can be provided as a list where some items might contain commas if multiple files per mark
    # but the simplest way is to expect one argument per mark and if multiple files, they should be pre-concatenated or handled
    # The shell script did: cat $(ls $E/$PC/*${M}* | grep -v log)
    # So we should support multiple files per mark.
    
    peak_files_groups = [p.split(",") for p in args.peaks]
    if len(peak_files_groups) != len(marks):
         sys.exit(f"Error: Number of peak groups ({len(peak_files_groups)}) does not match number of marks ({len(marks)})")
        
    chroms, sizes = read_chrom_sizes(args.chromsizes)
    per_chrom = binarize_peaks(peak_files_groups, chroms, sizes, args.bin, marks)
    
    if args.save_binary:
        write_binary_files(per_chrom, marks, args.cell, args.save_binary)
        
    print("Preparing data for KMeans...", file=sys.stderr)
    X = np.vstack([data for _, data in per_chrom])
    print(f"Total length: {X.shape[0]} bins", file=sys.stderr)
    
    print(f"Fitting KMeans with {args.states} states...", file=sys.stderr)
    model = KMeans(n_clusters=args.states, init='k-means++', random_state=args.seed, n_init=10)
    labels = model.fit_predict(X)
    
    print("Generating BED output...", file=sys.stderr)
    out_f = open(args.out, "w") if args.out else sys.stdout
    
    # Prepare colors
    cmap = plt.get_cmap("tab20")
    colors = []
    for i in range(args.states):
        rgb = cmap(i % 20)[:3]
        colors.append(",".join([str(int(c * 255)) for c in rgb]))

    offset = 0
    for chrom, data in per_chrom:
        n = data.shape[0]
        chrom_labels = labels[offset : offset + n]
        for chrom_, s, e, sid in segments_from_labels(chrom, chrom_labels, args.bin):
            color = colors[sid - 1]
            out_f.write(f"{chrom_}\t{s}\t{e}\tE{sid}\t0\t.\t{s}\t{e}\t{color}\n")
        offset += n
        
    if args.out:
        out_f.close()
    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
