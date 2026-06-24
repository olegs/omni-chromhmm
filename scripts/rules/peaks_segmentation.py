#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import sys
import os
import gzip
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_chrom_sizes(path):
    df = pd.read_csv(path, sep='\t', header=None, names=['chrom', 'size'])
    df = df[~df['chrom'].str.contains('_')]
    return df['chrom'].tolist(), df.set_index('chrom')['size'].to_dict()

def binarize_peaks(peak_files_groups, chroms, sizes, bin_size, marks):
    """
    peak_files_groups: list of lists of peak files (one list per mark)
    """
    num_marks = len(marks)
    # Pre-allocate single matrix for better memory efficiency
    chrom_bins = {chrom: (sizes[chrom] + bin_size - 1) // bin_size for chrom in chroms}
    total_bins = sum(chrom_bins.values())
    
    # Use uint8 for binary data
    data_matrix = np.zeros((total_bins, num_marks), dtype=np.uint8)
    
    offsets = {}
    cur_off = 0
    for chrom in chroms:
        offsets[chrom] = cur_off
        cur_off += chrom_bins[chrom]
    
    for m_idx, peak_files in enumerate(peak_files_groups):
        for peak_file in peak_files:
            if not peak_file or peak_file == "NONE" or not os.path.exists(peak_file):
                continue
            if any(peak_file.endswith(ext) for ext in [".png", ".pdf", ".jpg", ".jpeg", ".log", ".bw", ".bigWig"]):
                continue
            
            print(f"Binarizing {peak_file} for mark {marks[m_idx]}...", file=sys.stderr)
            try:
                df = pd.read_csv(peak_file, sep='\t', header=None, comment='#',
                                 usecols=[0, 1, 2], names=['chrom', 'start', 'end'],
                                 dtype={'chrom': str, 'start': int, 'end': int})
                df = df[df['chrom'].isin(chroms)]
                
                for chrom, group in df.groupby('chrom'):
                    off = offsets[chrom]
                    n_bins = chrom_bins[chrom]
                    starts = (group['start'] // bin_size).clip(lower=0, upper=n_bins - 1).values
                    ends = (group['end'] // bin_size).clip(lower=0, upper=n_bins).values
                    for s, e in zip(starts, ends):
                        if e > s:
                            data_matrix[off + s : off + e, m_idx] = 1
            except Exception as e:
                print(f"Error reading {peak_file}: {e}", file=sys.stderr)
                    
    # Return as list of (chrom, matrix_slice) to keep compatibility
    result = []
    for chrom in chroms:
        off = offsets[chrom]
        n = chrom_bins[chrom]
        result.append((chrom, data_matrix[off : off + n]))
    return data_matrix, result

def write_binary_files(per_chrom, marks, cell, outdir):
    os.makedirs(outdir, exist_ok=True)
    for chrom, matrix in per_chrom:
        outfile = os.path.join(outdir, f"{chrom}_binary.txt.gz")
        print(f"Writing {outfile}...", file=sys.stderr)
        # Use pandas for efficient gzipped output
        df = pd.DataFrame(matrix, columns=marks)
        with gzip.open(outfile, "wt") as f:
            f.write(f"{cell}\t{chrom}\n")
            df.to_csv(f, sep='\t', index=False)

def segments_from_labels(chrom, labels, bin_size):
    if len(labels) == 0: return
    # Efficient RLE using numpy
    change_idx = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(labels)]
    for s, e in zip(starts, ends):
        yield chrom, s * bin_size, e * bin_size, int(labels[s]) + 1

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
    data_matrix, per_chrom = binarize_peaks(peak_files_groups, chroms, sizes, args.bin, marks)
    
    if args.save_binary:
        write_binary_files(per_chrom, marks, args.cell, args.save_binary)
        
    print(f"Fitting KMeans with {args.states} states...", file=sys.stderr)
    model = KMeans(n_clusters=args.states, init='k-means++', random_state=args.seed, n_init=10)
    
    # Subsampling for training to save memory
    subsample_size = min(data_matrix.shape[0], 1000000)
    print(f"  Subsampling {subsample_size} bins for training...", file=sys.stderr)
    np.random.seed(args.seed)
    indices = np.random.choice(data_matrix.shape[0], subsample_size, replace=False)
    model.fit(data_matrix[indices])
    
    print("Generating labels...", file=sys.stderr)
    labels = model.predict(data_matrix)
    
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
