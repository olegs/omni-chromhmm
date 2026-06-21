#!/usr/bin/env python3
"""State matching: relabel segmentation states to a reference.

Sub-commands:

  compute  – Compute mean RPKM signal per state from bigwig files and save to .npz.

  match    – Match work segmentation to reference.

             When --ref-emissions and --work-emissions are provided, uses a combined
             score of Jaccard overlap and cosine emission similarity:
               combined = alpha * jaccard_overlap + (1 - alpha) * cosine_emission
             With alpha=1.0: overlap-only (Hungarian on raw total overlap bp).
             With alpha=0.0: emission-only (Hungarian on cosine matrix).
             With alpha=0.8 (default): overlap-weighted combined (comb).

             When no emission files are provided, falls back to overlap-only matching.

             Hungarian algorithm maximizes the combined similarity score.

Usage:
    match.py compute --bed SEG.bed \\
        --bigwigs H3K36me3.bw H3K9me3.bw ... \\
        --marks   H3K36me3    H3K9me3    ... \\
        [--bin 100] --output emissions.npz

    match.py match \\
        --ref REF.bed --work WORK.bed \\
        [--ref-emissions ref.npz --work-emissions work.npz] \\
        [--alpha 0.8] > MATCHED.bed
"""

import argparse
import os
import sys
import gzip
from bisect import bisect_left
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# BED I/O
# ---------------------------------------------------------------------------

def load_bed(path):
    """Return list of (chrom, start, end, name, color)."""
    out = []
    _open = gzip.open if path.endswith(".gz") else open
    with _open(path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            chrom, s, e = p[0], int(p[1]), int(p[2])
            name = p[3] if len(p) > 3 else "."
            color = p[8] if len(p) >= 9 else "0,0,0"
            out.append((chrom, s, e, name, color))
    return out


# ---------------------------------------------------------------------------
# Overlap helpers (also used by compare.py)
# ---------------------------------------------------------------------------

def build_index(segs):
    """Group segs by chrom and sort; return (by_chr, starts_by_chr)."""
    by_chr = defaultdict(list)
    for row in segs:
        chrom, s, e, name = row[:4]
        by_chr[chrom].append((s, e, name))
    starts = {}
    for chrom in by_chr:
        by_chr[chrom].sort()
        starts[chrom] = [s for s, _, _ in by_chr[chrom]]
    return by_chr, starts


def pair_overlap(ref_segs, work_segs, ref_index=None):
    """overlap[(work_name, ref_name)] = total overlapping bp."""
    ref_by_chr, ref_starts = ref_index if ref_index is not None else build_index(ref_segs)
    overlap = defaultdict(int)
    for row in work_segs:
        chrom, ws, we, wname = row[:4]
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
    for row in segs:
        name = row[3]
        s, e = row[1], row[2]
        out[name] += e - s
    return out


def state_colors(ref_segs):
    out = {}
    for row in ref_segs:
        name = row[3]
        color = row[4] if len(row) > 4 else "0,0,0"
        out.setdefault(name, color)
    return out


def best_mapping(overlap, work_states, ref_states):
    """One-to-one mapping work→ref via Hungarian algorithm on raw overlap length."""
    n_r = len(ref_states)
    cost = np.zeros((len(work_states), n_r))
    for i, w in enumerate(work_states):
        for j, r in enumerate(ref_states):
            cost[i, j] = -overlap.get((w, r), 0)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {work_states[i]: ref_states[j] for i, j in zip(row_ind, col_ind)}
    for w in work_states:
        if w not in mapping:
            mapping[w] = w
    return mapping


def remap_bins(bins, mapping):
    """Apply a state mapping to a bins dict {chrom: {pos: state}}.

    Any state not in mapping is left unchanged.
    """
    return {chrom: {pos: mapping.get(state, state) for pos, state in positions.items()}
            for chrom, positions in bins.items()}


def emission_cosine_mapping(states1, mat1, states2, mat2):
    """One-to-one mapping states1→states2 via cosine similarity (Hungarian).

    Returns (avg_similarity, mapping_dict).
    Both mat1 and mat2 are (n_states, n_marks) float arrays.
    """
    n1, n2 = len(states1), len(states2)
    cost = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            norm1 = np.linalg.norm(mat1[i])
            norm2 = np.linalg.norm(mat2[j])
            cost[i, j] = (1.0 - np.dot(mat1[i], mat2[j]) / (norm1 * norm2)
                          if norm1 > 0 and norm2 > 0 else 1.0)
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {states1[r]: states2[c] for r, c in zip(row_ind, col_ind)}
    avg_sim = sum(1.0 - cost[r, c] for r, c in zip(row_ind, col_ind)) / max(len(row_ind), 1)
    return avg_sim, mapping


def compare(ref_segs, work_segs, overlap, mapping, outdir):
    """Write Jaccard heatmap and similarity.txt to outdir."""
    os.makedirs(outdir, exist_ok=True)
    ref_states  = sorted({x[3] for x in ref_segs})
    work_states = sorted({x[3] for x in work_segs})
    ref_len  = state_lengths(ref_segs)
    work_len = state_lengths(work_segs)

    mat = np.zeros((len(work_states), len(ref_states)))
    for i, w in enumerate(work_states):
        for j, r in enumerate(ref_states):
            ov    = overlap.get((w, r), 0)
            union = ref_len[r] + work_len[w] - ov
            if union > 0:
                mat[i, j] = ov / union

    fig, ax = plt.subplots(figsize=(max(6, len(ref_states) * 0.5),
                                    max(6, len(work_states) * 0.4)))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ref_states)))
    ax.set_xticklabels(ref_states, rotation=90)
    ax.set_yticks(range(len(work_states)))
    ax.set_yticklabels(work_states)
    ax.set_title("Jaccard similarity")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if mat[i, j] < 0.5 else "white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "jaccard.png"), dpi=120)
    plt.close(fig)

    total_hit = sum(overlap.get((w, mapping[w]), 0) for w in work_states)
    total = sum(work_len.values())
    sim = total_hit / total if total else 0.0
    with open(os.path.join(outdir, "similarity.txt"), "w") as f:
        f.write(f"similarity = {sim:.4f}\n")
    print(f"similarity = {sim:.4f}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Emissions helpers
# ---------------------------------------------------------------------------

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

    states = sorted(sums.keys())
    mat = np.zeros((len(states), n_marks), dtype=np.float64)
    for i, st in enumerate(states):
        if counts[st] > 0:
            mat[i] = sums[st] / counts[st]
    return states, mark_names, mat


def _save_emissions_npz(path, states, marks, mat):
    np.savez_compressed(path, states=np.array(states), marks=np.array(marks), mat=mat)


def _load_emissions_npz(path):
    data = np.load(path, allow_pickle=False)
    return list(data["states"]), list(data["marks"]), data["mat"]


def _save_match_matrices(out_prefix, work_states, ref_states, mapping,
                         jaccard=None, cosine=None, combined=None):
    """Persist the per-state work→ref matching matrices used by the Hungarian
    assignment, plus the chosen mapping, so they can be inspected/plotted.

    Writes (rows = work states, cols = ref states):
      {out_prefix}.score.tsv     score matrix actually assigned on (combined,
                                 or jaccard/overlap when emissions are absent)
      {out_prefix}.jaccard.tsv   overlap Jaccard matrix         (when available)
      {out_prefix}.cosine.tsv    emission cosine matrix         (when available)
      {out_prefix}.mapping.tsv   work_state, ref_state, score[, jaccard, cosine]
      {out_prefix}.png           annotated score heatmap, matched cell outlined
    """
    out_dir = os.path.dirname(os.path.abspath(out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _save_tsv(mat, path):
        with open(path, "w") as f:
            f.write("\t" + "\t".join(map(str, ref_states)) + "\n")
            for i, w in enumerate(work_states):
                row = [str(w)] + [f"{mat[i, j]:.4f}" for j in range(len(ref_states))]
                f.write("\t".join(row) + "\n")

    if jaccard is not None:
        _save_tsv(jaccard, out_prefix + ".jaccard.tsv")
    if cosine is not None:
        _save_tsv(cosine, out_prefix + ".cosine.tsv")
    score = combined if combined is not None else jaccard
    if score is None:
        return
    _save_tsv(score, out_prefix + ".score.tsv")

    w_idx = {s: i for i, s in enumerate(work_states)}
    r_idx = {s: i for i, s in enumerate(ref_states)}
    with open(out_prefix + ".mapping.tsv", "w") as f:
        cols = ["work_state", "ref_state", "score"]
        if jaccard is not None:
            cols.append("jaccard")
        if cosine is not None:
            cols.append("cosine")
        f.write("\t".join(cols) + "\n")
        for w in work_states:
            r = mapping.get(w, w)
            if r not in r_idx:
                continue
            wi, ri = w_idx[w], r_idx[r]
            row = [w, r, f"{score[wi, ri]:.4f}"]
            if jaccard is not None:
                row.append(f"{jaccard[wi, ri]:.4f}")
            if cosine is not None:
                row.append(f"{cosine[wi, ri]:.4f}")
            f.write("\t".join(row) + "\n")

    fig, ax = plt.subplots(figsize=(max(6, len(ref_states) * 0.5),
                                    max(6, len(work_states) * 0.4)))
    im = ax.imshow(score, cmap="Blues", vmin=0, aspect="auto")
    ax.set_xticks(range(len(ref_states)))
    ax.set_xticklabels(ref_states, rotation=90, fontsize=7)
    ax.set_yticks(range(len(work_states)))
    ax.set_yticklabels(work_states, fontsize=7)
    ax.set_xlabel("Reference state")
    ax.set_ylabel("Work state")
    ax.set_title("Per-state matching score (work → reference)")
    for w in work_states:
        r = mapping.get(w, w)
        if r in r_idx:
            ax.add_patch(plt.Rectangle((r_idx[r] - 0.5, w_idx[w] - 0.5), 1, 1,
                                       fill=False, edgecolor="red", linewidth=1.5))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Unified matching
# ---------------------------------------------------------------------------

def match_states(ref_segs, work_segs,
                 states_ref=None, mat_ref=None,
                 states_work=None, mat_work=None,
                 alpha=0.8, matrix_out=None):
    """Match work states to ref states using overlap and/or cosine emission.

    alpha=1.0 (overlap-only): maximizes total overlapping bp directly (Hungarian
      on raw bp matrix — no normalization, consistent with best_mapping).

    0 < alpha < 1 (combined): Jaccard similarity is used as the normalized overlap
      signal so it can be linearly combined with cosine emission in [0,1]:
        combined[i,j] = alpha * jaccard(work_i, ref_j)
                      + (1 - alpha) * cosine(emission_work_i, emission_ref_j)

    alpha=0.0 (emission-only): cosine emission similarity only.

    Hungarian algorithm maximizes the total combined score.
    Returns dict mapping work_state -> ref_state.
    """
    work_states_sorted = sorted({x[3] for x in work_segs})
    ref_states_sorted  = sorted({x[3] for x in ref_segs})
    n_w = len(work_states_sorted)
    n_r = len(ref_states_sorted)

    overlap  = pair_overlap(ref_segs, work_segs)
    ref_len  = state_lengths(ref_segs)
    work_len = state_lengths(work_segs)

    use_emissions = (mat_ref is not None and mat_work is not None
                     and states_ref is not None and states_work is not None
                     and alpha < 1.0)

    if alpha >= 1.0:
        # Overlap-only: normalize by genome length so scores are in [0, 1].
        genome_len = sum(ref_len.values()) or 1
        raw_overlap = np.zeros((n_w, n_r))
        for i, w in enumerate(work_states_sorted):
            for j, r in enumerate(ref_states_sorted):
                raw_overlap[i, j] = overlap.get((w, r), 0) / genome_len
        combined = raw_overlap
        jaccard  = None
        cosine   = None
    else:
        # Combined or emission-only: normalize overlap to [0,1] via Jaccard.
        jaccard = np.zeros((n_w, n_r))
        for i, w in enumerate(work_states_sorted):
            for j, r in enumerate(ref_states_sorted):
                ov    = overlap.get((w, r), 0)
                union = ref_len.get(r, 0) + work_len.get(w, 0) - ov
                jaccard[i, j] = ov / union if union > 0 else 0.0

        if use_emissions:
            ref_idx  = {s: i for i, s in enumerate(states_ref)}
            work_idx = {s: i for i, s in enumerate(states_work)}

            cosine = np.zeros((n_w, n_r))
            for i, w in enumerate(work_states_sorted):
                for j, r in enumerate(ref_states_sorted):
                    wi = work_idx.get(w)
                    ri = ref_idx.get(r)
                    if wi is None or ri is None:
                        continue
                    norm_w = np.linalg.norm(mat_work[wi])
                    norm_r = np.linalg.norm(mat_ref[ri])
                    if norm_w > 0 and norm_r > 0:
                        cosine[i, j] = np.dot(mat_work[wi], mat_ref[ri]) / (norm_w * norm_r)

            combined = alpha * jaccard + (1.0 - alpha) * cosine
        else:
            combined = jaccard
            cosine   = None

    row_ind, col_ind = linear_sum_assignment(-combined)  # maximize
    mapping = {work_states_sorted[r]: ref_states_sorted[c]
               for r, c in zip(row_ind, col_ind)}

    # Logging
    avg  = combined[row_ind, col_ind].mean() if len(row_ind) > 0 else 0.0
    mode = ("overlap-only" if alpha >= 1.0
            else "emission-only" if alpha == 0.0
            else f"combined alpha={alpha}")
    print(f"avg_similarity = {avg:.4f}  ({mode})", file=sys.stderr)
    w_idx = {s: i for i, s in enumerate(work_states_sorted)}
    r_idx = {s: i for i, s in enumerate(ref_states_sorted)}
    for w, r in sorted(mapping.items()):
        wi, ri = w_idx[w], r_idx[r]
        if jaccard is not None:
            msg = f"  {w} -> {r}  (jaccard={jaccard[wi, ri]:.4f}"
            if cosine is not None:
                msg += f", cosine={cosine[wi, ri]:.4f}"
            msg += f", combined={combined[wi, ri]:.4f})"
        else:
            msg = f"  {w} -> {r}  (overlap={combined[wi, ri]:.4f})"
        print(msg, file=sys.stderr)

    for w in work_states_sorted:
        if w not in mapping:
            mapping[w] = w

    if matrix_out:
        _save_match_matrices(matrix_out, work_states_sorted, ref_states_sorted,
                             mapping, jaccard=jaccard, cosine=cosine, combined=combined)
    return mapping


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_compute(args):
    segs = load_bed(args.bed)
    states, marks, mat = compute_emissions_from_bigwigs(
        segs, args.bigwigs, args.marks, args.bin)
    _save_emissions_npz(args.output, states, marks, mat)


def cmd_match(args):
    ref_segs  = load_bed(args.ref)
    work_segs = load_bed(args.work)

    states_ref = mat_ref = states_work = mat_work = marks_work = None
    if args.ref_emissions and args.work_emissions:
        states_ref,  _,           mat_ref   = _load_emissions_npz(args.ref_emissions)
        states_work, marks_work,  mat_work  = _load_emissions_npz(args.work_emissions)
    elif args.work_emissions:
        # Load work emissions even when ref_emissions is absent (ovlp-only matching)
        # so the remapped emissions can still be written out.
        states_work, marks_work, mat_work = _load_emissions_npz(args.work_emissions)

    work_states = sorted({x[3] for x in work_segs})
    ref_states  = sorted({x[3] for x in ref_segs})

    if args.compare_only:
        overlap = pair_overlap(ref_segs, work_segs)
        mapping = best_mapping(overlap, work_states, ref_states)
        compare(ref_segs, work_segs, overlap, mapping, args.compare_only)
        return

    mapping = match_states(ref_segs, work_segs,
                           states_ref, mat_ref,
                           states_work, mat_work,
                           alpha=args.alpha, matrix_out=args.matrix_out)

    colors = state_colors(ref_segs)
    for row in work_segs:
        chrom, s, e, name = row[:4]
        color = row[4] if len(row) > 4 else "0,0,0"
        new_name  = mapping.get(name, name)
        new_color = colors.get(new_name, color)
        print(f"{chrom}\t{s}\t{e}\t{new_name}\t0\t.\t{s}\t{e}\t{new_color}")

    if args.remap_emissions and states_work is not None:
        remapped = [mapping.get(s, s) for s in states_work]
        _save_emissions_npz(args.remap_emissions, remapped, marks_work, mat_work)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    cp = sub.add_parser("compute", help="Compute per-state bigwig emissions and save to .npz")
    cp.add_argument("--bed",     required=True, help="Segmentation BED file")
    cp.add_argument("--bigwigs", nargs="+", required=True,
                    help="Bigwig files (one per mark, same order as --marks)")
    cp.add_argument("--marks",   nargs="+", required=True,
                    help="Mark names corresponding to bigwig files")
    cp.add_argument("--bin",     type=int, default=100, help="Bin size (default: 100)")
    cp.add_argument("--output",  required=True, help="Output .npz file")

    mp = sub.add_parser("match", help="Match work segmentation to reference")
    mp.add_argument("--ref",            required=True, help="Reference segmentation BED")
    mp.add_argument("--work",           required=True, help="Work segmentation BED to relabel")
    mp.add_argument("--ref-emissions",  default=None, dest="ref_emissions",
                    help="Pre-computed reference emissions .npz (enables combined/bwem matching)")
    mp.add_argument("--work-emissions", default=None, dest="work_emissions",
                    help="Pre-computed work emissions .npz (enables combined/bwem matching)")
    mp.add_argument("--alpha",          type=float, default=0.8,
                    help="Overlap weight: 1.0=overlap-only, 0.0=emission-only, 0.8=combined/comb (default)")
    mp.add_argument("--remap-emissions", default=None, dest="remap_emissions",
                    help="Save work emissions .npz with remapped state names to this path")
    mp.add_argument("--compare-only",   default=None, dest="compare_only",
                    help="Write Jaccard heatmap to this directory instead of rewriting BED")
    mp.add_argument("--matrix-out",     default=None, dest="matrix_out",
                    help="Path prefix to persist the per-state matching matrices "
                         "(.score.tsv/.jaccard.tsv/.cosine.tsv/.mapping.tsv/.png)")

    args = ap.parse_args()
    if args.cmd == "compute":
        cmd_compute(args)
    else:
        cmd_match(args)


if __name__ == "__main__":
    main()
