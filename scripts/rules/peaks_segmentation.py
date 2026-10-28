#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import sys
import os
import gzip
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def to_pattern_codes(X):
    """Pattern index of every row of a binary matrix."""
    X = np.asarray(X)
    codes = np.zeros(len(X), dtype=np.int64)
    for j in range(X.shape[1]):
        codes |= (X[:, j] != 0).astype(np.int64) << j
    return codes

def pattern_rows(n_features, codes):
    """The binary rows of the given pattern codes."""
    codes = np.asarray(codes, dtype=np.uint64)
    shifts = np.arange(n_features, dtype=np.uint64)
    return ((codes[:, np.newaxis] >> shifts) & np.uint64(1)).astype(np.float64)

def read_chrom_sizes(path):
    df = pd.read_csv(path, sep=r'\s+', header=None, comment='#', names=['chrom', 'size'])
    df = df[~df['chrom'].str.contains('_')]
    df['size'] = pd.to_numeric(df['size'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
    df = df.dropna(subset=['chrom', 'size'])
    df['size'] = df['size'].astype(int)
    return df['chrom'].tolist(), df.set_index('chrom')['size'].to_dict()

def binarize_peaks(peak_files_groups, chroms, sizes, bin_size, marks):
    """
    peak_files_groups: list of lists of peak files (one list per mark)
    """
    num_marks = len(marks)
    # Pre-allocate single matrix for better memory efficiency
    chrom_bins = {chrom: (int(sizes[chrom]) + bin_size - 1) // bin_size for chrom in chroms}
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
                    ends = ((group['end'] + bin_size - 1) // bin_size).clip(lower=0, upper=n_bins).values
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

    if args.states <= 0:
        print("States <= 0, skipping clustering.", file=sys.stderr)
        return
        
    n_marks = len(marks)
    slices = []
    offset = 0
    for chrom, data in per_chrom:
        slices.append((chrom, offset, offset + data.shape[0]))
        offset += data.shape[0]

    codes = to_pattern_codes(data_matrix)

    print(f"Fitting KMeans with {args.states} states...", file=sys.stderr)
    # Binarized data has at most 2^n_marks unique rows. Collapsing them
    # into a weighted pattern table makes the fit independent of the
    # number of bins.
    patterns, counts = np.unique(codes, return_counts=True)
    print(f"  {len(patterns)} distinct patterns", file=sys.stderr)
    x_collapsed = pattern_rows(n_marks, patterns)
    model = KMeans(n_clusters=args.states, init='k-means++', random_state=args.seed, n_init=10)
    model.fit(x_collapsed, sample_weight=counts)

    print("Generating labels...", file=sys.stderr)
    pattern_labels = model.predict(x_collapsed)
    if n_marks <= 24:
        lut = np.zeros(1 << n_marks, dtype=np.int8)
        lut[patterns] = pattern_labels
        labels = lut[codes]
    else:
        labels = pattern_labels[np.searchsorted(patterns, codes)]
    
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
