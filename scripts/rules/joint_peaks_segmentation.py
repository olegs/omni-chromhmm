#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Add current directory to path to import peaks_segmentation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peaks_segmentation import (read_chrom_sizes, segments_from_labels,
                                to_pattern_codes, pattern_rows)

def main():
    parser = argparse.ArgumentParser(description="Joint KMeans clustering across multiple cells.")
    parser.add_argument("--bin", type=int, required=True, help="Bin size")
    parser.add_argument("--chromsizes", required=True, help="Path to chrom.sizes file")
    parser.add_argument("--marks", required=True, help="Comma-separated list of marks")
    parser.add_argument("--cells", required=True, help="Comma-separated list of cell names")
    parser.add_argument("--peaks", nargs="+", required=True, 
                        help="Peak files for all cells. Expected order: all marks for cell 1, then all marks for cell 2, etc. Use commas for multiple files per mark.")
    parser.add_argument("--states", type=int, default=15, help="Number of KMeans states")
    parser.add_argument("--outdir", default=None, help="Output directory for BED files")
    parser.add_argument("--outbeds", nargs="+", default=None, help="Explicit output BED file paths, matching order of cells")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for KMeans")
    args = parser.parse_args()
    
    marks = args.marks.split(",")
    cells = args.cells.split(",")
    num_marks = len(marks)
    num_cells = len(cells)
    
    if not args.outdir and not args.outbeds:
        sys.exit("Error: Either --outdir or --outbeds must be provided.")
    if args.outbeds and len(args.outbeds) != num_cells:
        sys.exit(f"Error: Expected {num_cells} output paths in --outbeds, but got {len(args.outbeds)}")

    chroms, sizes = read_chrom_sizes(args.chromsizes)
    
    # Use pandas to manage chromosome info efficiently
    chrom_info = pd.DataFrame({'chrom': chroms, 'size': [sizes[c] for c in chroms]})
    chrom_info['bins'] = (chrom_info['size'] + args.bin - 1) // args.bin
    chrom_info['offset'] = chrom_info['bins'].cumsum().shift(fill_value=0)
    chrom_info_dict = chrom_info.set_index('chrom').to_dict('index')
    total_bins_per_cell = chrom_info['bins'].sum()
    
    print(f"Pre-allocating joint data matrix: {total_bins_per_cell * num_cells} bins x {num_marks} marks", file=sys.stderr)
    # Using uint8 for memory efficiency
    X = np.zeros((total_bins_per_cell * num_cells, num_marks), dtype=np.uint8)
    
    for i, cell in enumerate(cells):
        print(f"Binarizing peaks for cell {cell}...", file=sys.stderr)
        cell_offset = i * total_bins_per_cell
        mark_offset = 0
        
        cell_peaks = args.peaks[i * num_marks : (i + 1) * num_marks]
        
        for m_idx, peak_group in enumerate(cell_peaks):
            for peak_file in peak_group.split(","):
                if not peak_file or peak_file == "NONE" or not os.path.exists(peak_file):
                    continue
                if any(peak_file.endswith(ext) for ext in [".png", ".pdf", ".jpg", ".jpeg", ".log", ".bw", ".bigWig"]):
                    continue
                
                print(f"  Reading {peak_file} for mark {marks[m_idx]}...", file=sys.stderr)
                try:
                    # Use pandas for efficient reading
                    df = pd.read_csv(peak_file, sep='\t', header=None, comment='#', 
                                     usecols=[0, 1, 2], names=['chrom', 'start', 'end'],
                                     dtype={'chrom': str, 'start': int, 'end': int})
                    
                    df = df[df['chrom'].isin(chroms)]
                    for chrom, group in df.groupby('chrom'):
                        c_info = chrom_info_dict[chrom]
                        off = cell_offset + c_info['offset']
                        n_bins = c_info['bins']
                        
                        starts = (group['start'] // args.bin).clip(lower=0, upper=n_bins - 1).values
                        ends = ((group['end'] + args.bin - 1) // args.bin).clip(lower=0, upper=n_bins).values
                        for s, e in zip(starts, ends):
                            if e > s:
                                X[off + s : off + e, mark_offset + m_idx] = 1
                except Exception as e:
                    print(f"  Warning: Error reading {peak_file}: {e}", file=sys.stderr)
        
    print(f"Total joint data size: {X.nbytes / 1024**2:.2f} MB", file=sys.stderr)
    
    codes = to_pattern_codes(X)
    print(f"Fitting joint KMeans with {args.states} states...", file=sys.stderr)
    # Binarized data has at most 2^num_marks unique rows. Collapsing them
    # into a weighted pattern table makes the fit independent of the
    # number of bins.
    patterns, counts = np.unique(codes, return_counts=True)
    print(f"  {len(patterns)} distinct patterns", file=sys.stderr)
    x_collapsed = pattern_rows(num_marks, patterns)
    model = KMeans(n_clusters=args.states, init='k-means++', random_state=args.seed, n_init=10)
    model.fit(x_collapsed, sample_weight=counts)
    pattern_labels = model.predict(x_collapsed)
    if num_marks <= 24:
        lut = np.zeros(1 << num_marks, dtype=np.int8)
        lut[patterns] = pattern_labels
    else:
        lut = None
    
    if args.outdir:
        print(f"Generating BED outputs in {args.outdir}...", file=sys.stderr)
        os.makedirs(args.outdir, exist_ok=True)
    else:
        print(f"Generating BED outputs...", file=sys.stderr)

    suffix = "kmeans"
    
    # Prepare colors (consistent with an individual script)
    cmap = plt.get_cmap("tab20")
    colors = []
    for i in range(args.states):
        rgb = cmap(i % 20)[:3]
        colors.append(",".join([str(int(c * 255)) for c in rgb]))

    offset = 0
    for i, cell in enumerate(cells):
        if args.outbeds:
            out_path = args.outbeds[i]
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        else:
            out_path = os.path.join(args.outdir, f"{cell}_{suffix}_joint_states.bed")
        print(f"Writing {out_path}...", file=sys.stderr)

        cell_codes = codes[offset : offset + total_bins_per_cell]
        if lut is not None:
            cell_labels = lut[cell_codes]
        else:
            cell_labels = pattern_labels[np.searchsorted(patterns, cell_codes)]
            
        with open(out_path, "w") as out_f:
            cell_offset = 0
            for chrom in chroms:
                n = chrom_info_dict[chrom]['bins']
                chrom_labels = cell_labels[cell_offset : cell_offset + n]
                for chrom_, s, e, sid in segments_from_labels(chrom, chrom_labels, args.bin):
                    color = colors[sid - 1]
                    out_f.write(f"{chrom_}\t{s}\t{e}\tE{sid}\t0\t.\t{s}\t{e}\t{color}\n")
                cell_offset += n
        offset += total_bins_per_cell
                
    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
