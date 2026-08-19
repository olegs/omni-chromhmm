#!/usr/bin/env python3
"""Match work segmentation to reference using overlap."""

import argparse
import os
import sys
import gzip
import re
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
# Overlap helpers
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


def agreement_metrics(overlap, lengths1, lengths2, exclude=()):
    """Agreement of two segmentations of the same genome, as a dict.

    *overlap* is a pair_overlap() result and *lengths1* / *lengths2* the
    state_lengths() of the two sides. Their states, minus *exclude*, span the bp
    confusion matrix that all three metrics are read from:

      Kappa   : Cohen's kappa - agreement on identically named states, corrected
                for the agreement expected from the state compositions alone
      Jaccard : per-state intersection over union, averaged over the states
      Cosine  : cosine similarity of the two state-composition vectors

    All three are 0.0 when the two sides share no state or never overlap.
    *exclude* is matched exactly, so a model that numbers its states needs the
    numbered names ("15_Quies", not "Quies") to drop the background.
    """
    states = (set(lengths1) | set(lengths2)) - set(exclude)
    total = sum(overlap.get((s1, s2), 0) for s1 in states for s2 in states)
    if not states or total == 0:
        return {"Jaccard": 0.0, "Kappa": 0.0, "Cosine": 0.0}

    # Marginals of the confusion matrix restricted to *states*: how much each
    # side calls a state, as measured against the other side.
    a1 = {s: sum(overlap.get((s, s2), 0) for s2 in states) for s in states}
    a2 = {s: sum(overlap.get((s1, s), 0) for s1 in states) for s in states}

    po = sum(overlap.get((s, s), 0) for s in states) / total
    pe = sum((a1[s] / total) * (a2[s] / total) for s in states)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    jaccards = []
    for s in states:
        intersection = overlap.get((s, s), 0)
        union = a1[s] + a2[s] - intersection
        if union > 0:
            jaccards.append(intersection / union)

    ordered = sorted(states)
    v1 = np.array([a1[s] for s in ordered])
    v2 = np.array([a2[s] for s in ordered])
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return {
        "Jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "Kappa": kappa,
        "Cosine": float(np.dot(v1, v2) / norms) if norms > 0 else 0.0,
    }


def agreement_by_mode(overlap, lengths1, lengths2, background=()):
    """agreement_metrics() with and without the background states.

    Returns {"full": ..., "noqh": ...}: "full" keeps every state, "noqh" drops
    *background* (the Quies/Het bulk of the genome, which otherwise dominates
    both kappa and Jaccard).
    """
    return {
        "full": agreement_metrics(overlap, lengths1, lengths2),
        "noqh": agreement_metrics(overlap, lengths1, lengths2, exclude=background),
    }


def state_colors(ref_segs):
    out = {}
    for row in ref_segs:
        name = row[3]
        color = row[4] if len(row) > 4 else "0,0,0"
        out.setdefault(name, color)
    return out


def normalize_state_name(name):
    """Normalize ENCODE state names to canonical 15-state labels."""
    # Remove leading number: "1_TssA" -> "TssA", "01_TssA" -> "TssA"
    return re.sub(r'^\d+_', '', name)


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


def _natural_sort_key(s):
    """Sort key for natural ordering: 'E1' < 'E2' < 'E10'."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


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
    ref_states  = sorted({x[3] for x in ref_segs}, key=_natural_sort_key)
    work_states = sorted({x[3] for x in work_segs}, key=_natural_sort_key)
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


def _save_emissions_npz(path, states, marks, mat):
    np.savez_compressed(path, states=np.array(states), marks=np.array(marks), mat=mat)


def _load_emissions_npz(path):
    data = np.load(path, allow_pickle=False)
    return list(data["states"]), list(data["marks"]), data["mat"]


def _save_match_matrices(out_prefix, work_states, ref_states, mapping,
                         scores=None, jaccard=None):
    """Persist the per-state work→ref matching matrices used by the Hungarian
    assignment, plus the chosen mapping, so they can be inspected/plotted.

    Writes (rows = work states, cols = ref states):
      {out_prefix}.score.tsv     score matrix actually assigned on (raw overlap normalized)
      {out_prefix}.jaccard.tsv   overlap Jaccard matrix
      {out_prefix}.mapping.tsv   work_state, ref_state, score, jaccard
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
    if scores is None:
        return
    _save_tsv(scores, out_prefix + ".score.tsv")

    w_idx = {s: i for i, s in enumerate(work_states)}
    r_idx = {s: i for i, s in enumerate(ref_states)}
    with open(out_prefix + ".mapping.tsv", "w") as f:
        cols = ["work_state", "ref_state", "score"]
        if jaccard is not None:
            cols.append("jaccard")
        f.write("\t".join(cols) + "\n")
        for w in work_states:
            r = mapping.get(w, w)
            if r not in r_idx:
                continue
            wi, ri = w_idx[w], r_idx[r]
            row = [w, r, f"{scores[wi, ri]:.4f}"]
            if jaccard is not None:
                row.append(f"{jaccard[wi, ri]:.4f}")
            f.write("\t".join(row) + "\n")

    fig, ax = plt.subplots(figsize=(max(6, len(ref_states) * 0.5),
                                    max(6, len(work_states) * 0.4)))
    im = ax.imshow(scores, cmap="Blues", vmin=0, aspect="auto")
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


def match_states(ref_segs_list, work_segs_list, matrix_out=None, method='overlap'):
    """Match work states to ref states using overlap or jaccard.

    Maximizes total overlapping bp or Jaccard index (Hungarian algorithm).
    Returns dict mapping work_state -> ref_state.
    """
    if not isinstance(ref_segs_list, list) or (ref_segs_list and not isinstance(ref_segs_list[0], list)):
        ref_segs_list = [ref_segs_list]
    if not isinstance(work_segs_list, list) or (work_segs_list and not isinstance(work_segs_list[0], list)):
        work_segs_list = [work_segs_list]

    all_work_states = set()
    all_ref_states = set()
    total_overlap = defaultdict(int)
    total_ref_len = defaultdict(int)
    total_work_len = defaultdict(int)

    for ref_segs, work_segs in zip(ref_segs_list, work_segs_list):
        for x in work_segs: all_work_states.add(x[3])
        for x in ref_segs: all_ref_states.add(x[3])
        overlap = pair_overlap(ref_segs, work_segs)
        rl = state_lengths(ref_segs)
        wl = state_lengths(work_segs)
        for k, v in overlap.items(): total_overlap[k] += v
        for k, v in rl.items(): total_ref_len[k] += v
        for k, v in wl.items(): total_work_len[k] += v

    work_states_sorted = sorted(all_work_states, key=_natural_sort_key)
    ref_states_sorted  = sorted(all_ref_states, key=_natural_sort_key)
    n_w = len(work_states_sorted)
    n_r = len(ref_states_sorted)

    # Overlap matrix: normalize by genome length so scores are in [0, 1] for logging.
    # Assignment is invariant to global scaling.
    genome_len = sum(total_ref_len.values()) or 1
    scores = np.zeros((n_w, n_r))
    jaccard = np.zeros((n_w, n_r))
    for i, w in enumerate(work_states_sorted):
        for j, r in enumerate(ref_states_sorted):
            ov = total_overlap.get((w, r), 0)
            scores[i, j] = ov / genome_len
            union = total_ref_len.get(r, 0) + total_work_len.get(w, 0) - ov
            jaccard[i, j] = ov / union if union > 0 else 0.0

    if method == 'jaccard':
        row_ind, col_ind = linear_sum_assignment(-jaccard)
    else:
        row_ind, col_ind = linear_sum_assignment(-scores)  # maximize
    mapping = {work_states_sorted[r]: ref_states_sorted[c]
               for r, c in zip(row_ind, col_ind)}

    # Logging
    avg = (jaccard[row_ind, col_ind].mean() if method == 'jaccard' else scores[row_ind, col_ind].mean()) if len(row_ind) > 0 else 0.0
    print(f"avg_similarity = {avg:.4f}  ({method}-only)", file=sys.stderr)
    w_idx = {s: i for i, s in enumerate(work_states_sorted)}
    r_idx = {s: i for i, s in enumerate(ref_states_sorted)}
    for w, r in sorted(mapping.items()):
        wi, ri = w_idx[w], r_idx[r]
        print(f"  {w} -> {r}  (overlap={scores[wi, ri]:.4f}, jaccard={jaccard[wi, ri]:.4f})",
              file=sys.stderr)

    for w in work_states_sorted:
        if w not in mapping:
            mapping[w] = w

    if matrix_out:
        _save_match_matrices(matrix_out, work_states_sorted, ref_states_sorted,
                             mapping, scores=scores, jaccard=jaccard)
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref",            nargs='+', required=True, help="Reference segmentation BED(s)")
    ap.add_argument("--work",           nargs='+', required=True, help="Work segmentation BED(s) to relabel")
    ap.add_argument("--out",            nargs='+', help="Output remapped BED(s)")
    ap.add_argument("--work-bw-emissions", nargs='+', default=None, dest="work_emissions",
                    help="Pre-computed work emissions .npz (for remapping only)")
    ap.add_argument("--remap-bw-emissions", nargs='+', default=None, dest="remap_emissions",
                    help="Save work emissions .npz with remapped state names to this path")
    ap.add_argument("--work-bin-emissions", nargs='+', default=None, dest="work_bin_emissions",
                    help="Pre-computed work binarized emissions .npz (for remapping only)")
    ap.add_argument("--remap-bin-emissions", nargs='+', default=None, dest="remap_bin_emissions",
                    help="Save work binarized emissions .npz with remapped state names to this path")
    ap.add_argument("--compare-only",   default=None, dest="compare_only",
                    help="Write Jaccard heatmap to this directory instead of rewriting BED")
    ap.add_argument("--matrix-out",     default=None, dest="matrix_out",
                    help="Path prefix to persist the per-state matching matrices "
                         "(.score.tsv/.jaccard.tsv/.mapping.tsv/.png)")
    ap.add_argument("--method",         default="jaccard", choices=["overlap", "jaccard"],
                    help="Matching method: overlap or jaccard (default: jaccard)")

    args = ap.parse_args()

    if len(args.ref) != len(args.work):
        if len(args.ref) == 1:
            args.ref = args.ref * len(args.work)
        elif len(args.work) == 1:
            args.work = args.work * len(args.ref)
        else:
            sys.exit("Error: --ref and --work must have same number of files, or one must be 1.")

    if args.out and len(args.out) != len(args.work):
        sys.exit("Error: --out must have same number of files as --work.")

    ref_segs_list = [load_bed(p) for p in args.ref]
    work_segs_list = [load_bed(p) for p in args.work]

    if args.compare_only:
        mapping = match_states(ref_segs_list, work_segs_list, matrix_out=args.matrix_out, method=args.method)
        # Aggregated stats for compare()
        total_overlap = defaultdict(int)
        total_ref_len = defaultdict(int)
        total_work_len = defaultdict(int)
        for r_segs, w_segs in zip(ref_segs_list, work_segs_list):
            ov = pair_overlap(r_segs, w_segs)
            rl = state_lengths(r_segs)
            wl = state_lengths(w_segs)
            for k, v in ov.items(): total_overlap[k] += v
            for k, v in rl.items(): total_ref_len[k] += v
            for k, v in wl.items(): total_work_len[k] += v

        ref_states = sorted(total_ref_len.keys(), key=_natural_sort_key)
        work_states = sorted(total_work_len.keys(), key=_natural_sort_key)

        dummy_ref = [("chr1", 0, total_ref_len[s], s, "0,0,0") for s in ref_states]
        dummy_work = [("chr1", 0, total_work_len[s], s, "0,0,0") for s in work_states]
        compare(dummy_ref, dummy_work, total_overlap, mapping, args.compare_only)
        return

    mapping = match_states(ref_segs_list, work_segs_list, matrix_out=args.matrix_out, method=args.method)

    # Apply mapping to all work segmentations
    for i, work_segs in enumerate(work_segs_list):
        colors = state_colors(ref_segs_list[i])
        out_f = open(args.out[i], "w") if args.out else sys.stdout
        for row in work_segs:
            chrom, s, e, name = row[:4]
            color = row[4] if len(row) > 4 else "0,0,0"
            raw_new_name = mapping.get(name, name)
            new_name  = normalize_state_name(raw_new_name)
            new_color = colors.get(raw_new_name, color)
            out_f.write(f"{chrom}\t{s}\t{e}\t{new_name}\t0\t.\t{s}\t{e}\t{new_color}\n")
        if args.out:
            out_f.close()

    # Remap emissions
    def _remap_em_list(in_paths, out_paths):
        if not in_paths or not out_paths: return
        if len(in_paths) != len(out_paths):
            print(f"Warning: number of emission files doesn't match remapped paths", file=sys.stderr)
            return
        for ip, op in zip(in_paths, out_paths):
            states, marks, mat = _load_emissions_npz(ip)
            remapped = [normalize_state_name(mapping.get(s, s)) for s in states]
            _save_emissions_npz(op, remapped, marks, mat)

    _remap_em_list(args.work_emissions, args.remap_emissions)
    _remap_em_list(args.work_bin_emissions, args.remap_bin_emissions)


if __name__ == "__main__":
    main()
