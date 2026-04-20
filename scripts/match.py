#!/usr/bin/env python3
# Match state IDs in --work to --ref by maximizing total overlap length
# (PDF: "Matching is better using overlap len vs jaccard"), then rewrite --work
# with reference names + colors to stdout.
#
# With --compare-only DIR: skip rewriting; instead dump jaccard heatmap and
# overall similarity into DIR.

import argparse
import os
import sys
from bisect import bisect_left
from collections import defaultdict


def load_bed(path):
    """Return list of (chrom, start, end, name, color)."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            chrom, s, e = p[0], int(p[1]), int(p[2])
            name = p[3] if len(p) > 3 else "."
            color = p[8] if len(p) >= 9 else "0,0,0"
            out.append((chrom, s, e, name, color))
    return out


def build_index(segs):
    """Group segs by chrom and sort; return (by_chr, starts_by_chr)."""
    by_chr = defaultdict(list)
    for chrom, s, e, name, _ in segs:
        by_chr[chrom].append((s, e, name))
    starts = {}
    for chrom in by_chr:
        by_chr[chrom].sort()
        starts[chrom] = [s for s, _, _ in by_chr[chrom]]
    return by_chr, starts


def pair_overlap(ref_segs, work_segs):
    """overlap[(work_name, ref_name)] = total overlapping bp."""
    ref_by_chr, ref_starts = build_index(ref_segs)
    overlap = defaultdict(int)
    for chrom, ws, we, wname, _ in work_segs:
        if chrom not in ref_by_chr:
            continue
        starts = ref_starts[chrom]
        i = bisect_left(starts, ws) - 1
        if i < 0:
            i = 0
        arr = ref_by_chr[chrom]
        while i < len(arr) and arr[i][0] < we:
            rs, re_, rname = arr[i]
            ov = min(re_, we) - max(rs, ws)
            if ov > 0:
                overlap[(wname, rname)] += ov
            i += 1
    return overlap


def state_lengths(segs):
    out = defaultdict(int)
    for _, s, e, name, _ in segs:
        out[name] += e - s
    return out


def best_mapping(overlap, work_states, ref_states):
    """One-to-one mapping from work states to ref states via the Hungarian
    algorithm, maximizing total overlap length while preserving the number
    of distinct states."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n_w, n_r = len(work_states), len(ref_states)
    # Build cost matrix (negative overlap for minimization)
    cost = np.zeros((n_w, n_r))
    for i, w in enumerate(work_states):
        for j, r in enumerate(ref_states):
            cost[i, j] = -overlap.get((w, r), 0)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        mapping[work_states[i]] = ref_states[j]
    # Any leftover work states (more work than ref) keep their name
    for w in work_states:
        if w not in mapping:
            mapping[w] = w
    return mapping


def state_colors(ref_segs):
    out = {}
    for _, _, _, name, color in ref_segs:
        out.setdefault(name, color)
    return out


def compare(ref_segs, work_segs, overlap, mapping, outdir):
    """Write jaccard heatmap + similarity.txt to outdir."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    ref_states = sorted({x[3] for x in ref_segs})
    work_states = sorted({x[3] for x in work_segs})
    ref_len = state_lengths(ref_segs)
    work_len = state_lengths(work_segs)

    mat = np.zeros((len(work_states), len(ref_states)))
    for i, w in enumerate(work_states):
        for j, r in enumerate(ref_states):
            ov = overlap.get((w, r), 0)
            union = ref_len[r] + work_len[w] - ov
            if union > 0:
                mat[i, j] = ov / union

    fig, ax = plt.subplots(figsize=(max(6, len(ref_states) * 0.5),
                                    max(6, len(work_states) * 0.4)))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ref_states))); ax.set_xticklabels(ref_states, rotation=90)
    ax.set_yticks(range(len(work_states))); ax.set_yticklabels(work_states)
    ax.set_title("Jaccard similarity")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if mat[i, j] < 0.5 else "white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "jaccard.png"), dpi=120)
    plt.close(fig)

    # Overall similarity: fraction of work length whose state, once relabelled,
    # falls into the same ref state on the overlap.
    total_hit = sum(overlap.get((w, mapping[w]), 0) for w in work_states)
    total = sum(work_len.values())
    sim = total_hit / total if total else 0.0
    with open(os.path.join(outdir, "similarity.txt"), "w") as f:
        f.write(f"similarity = {sim:.4f}\n")
    print(f"similarity = {sim:.4f}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--compare-only", default=None,
                    help="Write plots into this directory instead of rewriting bed")
    args = ap.parse_args()

    ref = load_bed(args.ref)
    work = load_bed(args.work)
    work_states = sorted({x[3] for x in work})
    ref_states = sorted({x[3] for x in ref})

    overlap = pair_overlap(ref, work)
    mapping = best_mapping(overlap, work_states, ref_states)

    if args.compare_only:
        compare(ref, work, overlap, mapping, args.compare_only)
        return

    colors = state_colors(ref)
    for chrom, s, e, name, color in work:
        new_name = mapping.get(name, name)
        new_color = colors.get(new_name, color)
        print(f"{chrom}\t{s}\t{e}\t{new_name}\t0\t.\t{s}\t{e}\t{new_color}")


if __name__ == "__main__":
    main()
