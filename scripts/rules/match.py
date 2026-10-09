#!/usr/bin/env python3
"""Match work segmentation to reference by Hungarian assignment."""

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

# utils lives next door, in scripts/analysis; put it on the path so the metric
# names below are the shared constants whether this runs as a CLI or an import.
_analysis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
if _analysis_dir not in sys.path:
    sys.path.insert(0, _analysis_dir)

import utils


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
    """overlap[(work_name, ref_name)] = total overlapping bp.

    The work state comes first, so agreement_metrics(pair_overlap(side2, side1),
    lengths1, lengths2) keeps its lengths on the sides they name.
    """
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


def per_state_agreement(overlap, lengths1, lengths2, exclude=()):
    """One-vs-rest agreement of every state, as {state: {jaccard, kappa}}."""
    states = (set(lengths1) | set(lengths2)) - set(exclude)
    total = sum(overlap.values())
    a1, a2 = defaultdict(int), defaultdict(int)
    for (s1, s2), bp in overlap.items():
        a1[s1] += bp
        a2[s2] += bp

    out = {}
    for s in sorted(states):
        intersection = overlap.get((s, s), 0)
        union = lengths1.get(s, 0) + lengths2.get(s, 0) - intersection
        if union <= 0:
            continue
        metrics = {utils.JACCARD: intersection / union, utils.KAPPA: 0.0}
        if total > 0:
            p1, p2 = a1[s] / total, a2[s] / total
            po = (intersection + (total - a1[s] - a2[s] + intersection)) / total
            pe = p1 * p2 + (1 - p1) * (1 - p2)
            metrics[utils.KAPPA] = (po - pe) / (1 - pe) if pe < 1 else 1.0
        out[s] = metrics
    return out


def per_state_diagonal(overlap, states, metric):
    """Agreement of every state with itself, as {state: value}."""
    total = sum(overlap.get((s1, s2), 0) for s1 in states for s2 in states)
    if total == 0:
        return {}
    a1 = {s: sum(overlap.get((s, s2), 0) for s2 in states) for s in states}
    a2 = {s: sum(overlap.get((s1, s), 0) for s1 in states) for s in states}

    out = {}
    for s in states:
        shared = overlap.get((s, s), 0)
        if metric == utils.JACCARD:
            denom = a1[s] + a2[s] - shared
            value = shared / denom if denom > 0 else None
        elif metric == utils.KAPPA:
            p1, p2, p12 = a1[s] / total, a2[s] / total, shared / total
            denom = p1 + p2 - 2 * p1 * p2
            value = 2 * (p12 - p1 * p2) / denom if denom > 0 else None
        elif metric == utils.COSINE:
            denom = np.sqrt(a1[s] * a2[s])
            value = shared / denom if denom > 0 else None
        else:
            raise ValueError(f"unknown metric {metric}")
        if value is not None:
            out[s] = value
    return out


def agreement_metrics(overlap, lengths1, lengths2, exclude=()):
    """Agreement metrics (kappa, jaccard, cosine) of two segmentations."""
    all_states = set(lengths1) | set(lengths2)
    def _is_excluded(s):
        if s in exclude: return True
        norm = utils.normalize_state_name(s)
        if norm in exclude: return True
        if exclude is utils.NOQH_STATES:
            return utils.is_noqh(s)
        return False

    states = {s for s in all_states if not _is_excluded(s)}
    total = sum(overlap.get((s1, s2), 0) for s1 in states for s2 in states)
    if not states or total == 0:
        return {
            utils.JACCARD: 0.0,
            utils.KAPPA: 0.0,
            utils.COSINE: 0.0
        }

    a1 = {s: sum(overlap.get((s, s2), 0) for s2 in states) for s in states}
    a2 = {s: sum(overlap.get((s1, s), 0) for s1 in states) for s in states}

    po = sum(overlap.get((s, s), 0) for s in states) / total
    pe = sum((a1[s] / total) * (a2[s] / total) for s in states)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0

    per_state = per_state_agreement(overlap, lengths1, lengths2, exclude=exclude)
    jaccards = [m[utils.JACCARD] for m in per_state.values()]

    ordered = sorted(states)
    v1 = np.array([lengths1.get(s, 0) for s in ordered])
    v2 = np.array([lengths2.get(s, 0) for s in ordered])
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return {
        utils.JACCARD: float(np.mean(jaccards)) if jaccards else 0.0,
        utils.KAPPA: kappa,
        utils.COSINE: float(np.dot(v1, v2) / norms) if norms > 0 else 0.0,
    }


def agreement_by_mode(overlap, lengths1, lengths2, background=()):
    """Return agreement metrics for 'full' and 'noqh' (background excluded) modes."""
    return {
        utils.FULL: agreement_metrics(overlap, lengths1, lengths2),
        utils.NOQH: agreement_metrics(overlap, lengths1, lengths2,
                                      exclude=background),
    }


def state_colors(ref_segs):
    out = {}
    for row in ref_segs:
        name = row[3]
        color = row[4] if len(row) > 4 else "0,0,0"
        out.setdefault(name, color)
    return out




# Match quality = Jaccard * alpha + EmissionSim * beta.
# Jaccard provides physical overlap concordance; emissions provide biochemical matching.
JACCARD_WEIGHT = 0.6
EMISSION_WEIGHT = 0.4


def jaccard_matrix(overlap, lengths_w, lengths_r, work_states, ref_states):
    """Pairwise Jaccard similarity of work and ref states in [0, 1]."""
    mat = np.zeros((len(work_states), len(ref_states)), dtype=float)
    for i, w in enumerate(work_states):
        lw = lengths_w.get(w, 0)
        for j, r in enumerate(ref_states):
            ov = overlap.get((w, r), 0)
            union = lw + lengths_r.get(r, 0) - ov
            if union > 0:
                mat[i, j] = ov / union
    return mat


def best_mapping(overlap, work_states, ref_states):
    """One-to-one mapping work→ref maximising shared bp (Hungarian)."""
    matrix = np.zeros((len(work_states), len(ref_states)))
    for i, w in enumerate(work_states):
        for j, r in enumerate(ref_states):
            matrix[i, j] = -overlap.get((w, r), 0)
    row_ind, col_ind = linear_sum_assignment(matrix)
    mapping = {work_states[i]: ref_states[j] for i, j in zip(row_ind, col_ind)}
    for w in work_states:
        mapping.setdefault(w, w)
    return mapping


def _natural_sort_key(s):
    """Sort key for natural ordering: 'E1' < 'E2' < 'E10'."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def remap_bins(bins, mapping):
    """Apply a state mapping to a bins dict {chrom: {pos: state}}."""
    return {chrom: {pos: mapping.get(state, state) for pos, state in positions.items()}
            for chrom, positions in bins.items()}


def _cosine_matrix(mat1, mat2):
    """Pairwise cosine of the rows of mat1 and mat2; 0 where a row is all zeros."""
    denom = np.outer(np.linalg.norm(mat1, axis=1), np.linalg.norm(mat2, axis=1))
    sim = np.zeros(denom.shape)
    nonzero = denom > 0
    sim[nonzero] = (mat1 @ mat2.T)[nonzero] / denom[nonzero]
    return sim


def background_augmented(mat):
    """Emission vectors with a 1 - max(mark) background component appended."""
    if mat.size == 0 or mat.shape[1] == 0 or mat.max() > 1.0 or mat.min() < 0.0:
        return mat
    return np.column_stack([mat, 1.0 - mat.max(axis=1)])


def _is_binarized(mat):
    """Whether a matrix holds binarization fractions rather than raw signal."""
    return mat.size > 0 and mat.max() <= 1.01 and mat.min() >= -0.01


# Bigwig signal -> expected binarized fraction, fitted on ChromHMM segmentations
# that have both: fraction = sigmoid(log2(signal / background) + intercept).
# The background is the least-marked state of each mark; half of the bins are
# called present at ~15x over it. Slope ~1 in every ENCODE dataset.
BIGWIG_PRESENCE_INTERCEPT = -3.9
# Floor of the background, relative to the mark's maximum, for marks with a zero state.
BIGWIG_BACKGROUND_FLOOR = 1e-3


def bigwig_presence(mat):
    """Bigwig emissions converted to the binarized scale, per mark."""
    mat = np.maximum(np.asarray(mat, dtype=float), 0.0)
    if mat.size == 0:
        return mat
    mark_max = mat.max(axis=0)
    background = np.maximum(mat.min(axis=0), BIGWIG_BACKGROUND_FLOOR * mark_max)
    background = np.where(background > 0, background, 1.0)
    fold = np.log2(np.maximum(mat, 1e-12) / background)
    presence = 1.0 / (1.0 + np.exp(-(fold + BIGWIG_PRESENCE_INTERCEPT)))
    return np.where(mark_max > 0, presence, 0.0)


def emission_similarity(mat_w, mat_r, binarized_w=None, binarized_r=None):
    """Pairwise similarity of two aligned emission matrices in [0, 1].

    Cosine of the background-augmented vectors: the 1 - max(mark) column keeps
    the signal level, so a near-empty state is not similar to an all-high one.
    Bigwig signal is first converted to the binarized scale (bigwig_presence),
    so bigwig and binarized emissions can be compared with each other.
    binarized_* says which a matrix holds; None guesses from its value range.
    """
    def _augment(mat, binarized):
        mat = np.asarray(mat, dtype=float)
        if binarized is None:
            binarized = _is_binarized(mat)
        if not binarized:
            mat = bigwig_presence(mat)
        return background_augmented(np.clip(mat, 0.0, 1.0))

    return _cosine_matrix(_augment(mat_w, binarized_w), _augment(mat_r, binarized_r))


def emission_cosine_mapping(states1, mat1, states2, mat2):
    """One-to-one mapping states1→states2 via cosine similarity (Hungarian).

    Returns (avg_similarity, mapping_dict).
    """
    mat1 = np.asarray(mat1, dtype=float)
    mat2 = np.asarray(mat2, dtype=float)
    cost = 1.0 - _cosine_matrix(background_augmented(mat1),
                                background_augmented(mat2))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {states1[r]: states2[c] for r, c in zip(row_ind, col_ind)}
    sim = _cosine_matrix(mat1, mat2)
    avg_sim = sum(sim[r, c] for r, c in zip(row_ind, col_ind)) / max(len(row_ind), 1)
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
    ax.set_xticklabels(ref_states, rotation=90, fontsize=utils.TICK_FONTSIZE)
    ax.set_yticks(range(len(work_states)))
    ax.set_yticklabels(work_states, fontsize=utils.TICK_FONTSIZE)
    ax.set_title("Jaccard similarity", **utils.TITLE_STYLE)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=utils.ANNOTATION_FONTSIZE, color="black" if mat[i, j] < 0.5 else "white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "jaccard.png"), dpi=300)
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
                         jaccard=None, emission=None, quality=None):
    """Persist work→ref matching matrices and chosen mapping.

    Writes:
      {out_prefix}.jaccard.tsv     Jaccard similarity matrix
      {out_prefix}.emission.tsv    emission similarity matrix
      {out_prefix}.quality.tsv     match quality matrix (composite utility)
      {out_prefix}.mapping.tsv     state mapping with individual scores
      {out_prefix}.png             annotated heatmap of match quality / Jaccard
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

    if emission is not None:
        _save_tsv(emission, out_prefix + ".emission.tsv")
    if quality is not None:
        _save_tsv(quality, out_prefix + ".quality.tsv")
    if jaccard is None:
        return
    _save_tsv(jaccard, out_prefix + ".jaccard.tsv")

    w_idx = {s: i for i, s in enumerate(work_states)}
    r_idx = {s: i for i, s in enumerate(ref_states)}
    with open(out_prefix + ".mapping.tsv", "w") as f:
        cols = ["work_state", "ref_state", "jaccard"]
        if emission is not None:
            cols.append("emission")
        if quality is not None:
            cols.append("quality")
        f.write("\t".join(cols) + "\n")
        for w in work_states:
            r = mapping.get(w, w)
            if r not in r_idx:
                # More work states than reference states: this one got no name.
                f.write("\t".join([str(w), "(unassigned)"] + [""] * (len(cols) - 2)) + "\n")
                continue
            wi, ri = w_idx[w], r_idx[r]
            row = [str(w), str(r), f"{jaccard[wi, ri]:.4f}"]
            if emission is not None:
                row.append(f"{emission[wi, ri]:.4f}")
            if quality is not None:
                row.append(f"{quality[wi, ri]:.4f}")
            f.write("\t".join(row) + "\n")

    fig, ax = plt.subplots(figsize=(max(6, len(ref_states) * 0.5),
                                    max(6, len(work_states) * 0.4)))
    # Row-normalize for the heatmap to make states with small coverage visible.
    plot_scores = (jaccard if quality is None else quality).copy()
    row_max = plot_scores.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    plot_scores /= row_max

    im = ax.imshow(plot_scores, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ref_states)))
    ax.set_xticklabels(ref_states, rotation=90, fontsize=utils.TICK_FONTSIZE)
    ax.set_yticks(range(len(work_states)))
    ax.set_yticklabels(work_states, fontsize=utils.TICK_FONTSIZE)
    ax.set_xlabel("Reference state", fontsize=utils.AXIS_FONTSIZE)
    ax.set_ylabel("Work state", fontsize=utils.AXIS_FONTSIZE)
    ax.set_title("Per-state Jaccard (work → reference)"
                 if quality is None else
                 "Per-state match quality: composite utility (work → reference)",
                 **utils.TITLE_STYLE)
    for w in work_states:
        r = mapping.get(w, w)
        if r in r_idx:
            ax.add_patch(plt.Rectangle((r_idx[r] - 0.5, w_idx[w] - 0.5), 1, 1,
                                       fill=False, edgecolor="red", linewidth=1.5))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _emission_rows(names, mat, states):
    """Rows of mat for states, looked up by name; None for a state with no row.

    Handles numeric equivalence such as "1.0" vs "1".
    """
    def _key(name):
        try:
            return str(int(float(name)))
        except (TypeError, ValueError):
            return str(name)
    by_name = {}
    for i, name in enumerate(names):
        by_name.setdefault(str(name), i)
        by_name.setdefault(_key(name), i)
    rows = []
    for s in states:
        i = by_name.get(str(s), by_name.get(_key(s)))
        rows.append(None if i is None else mat[i])
    return rows


def _align_emissions(w_names, w_marks, w_mat, r_names, r_marks, r_mat,
                     work_states, ref_states):
    """Emission rows of work_states and ref_states, by state name, on common marks.

    Returns (aligned_w_mat, aligned_r_mat, common_marks, w_found, r_found), where
    the *_found boolean masks mark the states the emission files have a row for.
    """
    # Fuzzy match marks: H3K4me3_100 -> H3K4me3
    def _base(m): return str(m).split('_')[0].upper()
    w_base = {_base(m): i for i, m in enumerate(w_marks)}
    r_base = {_base(m): i for i, m in enumerate(r_marks)}
    common_base = sorted(set(w_base) & set(r_base))
    if not common_base:
        return None, None, [], None, None

    w_cols = [w_base[b] for b in common_base]
    r_cols = [r_base[b] for b in common_base]

    def _aligned(names, mat, cols, states):
        mat = np.asarray(mat, dtype=float)
        rows = _emission_rows(names, mat, states)
        found = np.array([r is not None for r in rows], dtype=bool)
        out = np.zeros((len(states), len(cols)))
        for i, r in enumerate(rows):
            if r is not None:
                out[i] = r[cols]
        return out, found

    aligned_w, w_found = _aligned(w_names, w_mat, w_cols, work_states)
    aligned_r, r_found = _aligned(r_names, r_mat, r_cols, ref_states)
    return aligned_w, aligned_r, common_base, w_found, r_found


def _emissions_binarized(path):
    """Whether an emission .npz holds binarized fractions, from its name; None if unknown."""
    name = os.path.basename(path)
    if "bin_emissions" in name:
        return True
    if "bw_emissions" in name:
        return False
    return None


def _load_emission_similarity(work_path, ref_path, work_states, ref_states):
    """emission_similarity() of work_states x ref_states from emission .npz files.

    None when the files are missing, share no marks, or cover no state on a side.
    States absent from an emission file get the mean similarity, so they neither
    win nor lose a match on emissions.
    """
    if not (work_path and ref_path):
        return None
    try:
        w_names, w_marks, w_mat = _load_emissions_npz(work_path)
        r_names, r_marks, r_mat = _load_emissions_npz(ref_path)
    except Exception as e:
        print(f"Warning: could not load emissions for matching: {e}", file=sys.stderr)
        return None
    aligned_w, aligned_r, common, w_found, r_found = _align_emissions(
        w_names, w_marks, w_mat, r_names, r_marks, r_mat, work_states, ref_states)
    if not common:
        print("Warning: emission files share no marks, matching on Jaccard only",
              file=sys.stderr)
        return None
    if not w_found.any() or not r_found.any():
        print("Warning: emission state names match no segmentation state, "
              "matching on Jaccard only", file=sys.stderr)
        return None
    for side, states, found in (("work", work_states, w_found), ("reference", ref_states, r_found)):
        missing = [s for s, f in zip(states, found) if not f]
        if missing:
            print(f"Warning: no {side} emissions for states {missing}", file=sys.stderr)

    kinds = {True: "binarized", False: "bigwig", None: "unknown type"}
    print(f"Incorporating emission similarity (marks: {common}; work "
          f"{kinds[_emissions_binarized(work_path)]}, reference "
          f"{kinds[_emissions_binarized(ref_path)]})", file=sys.stderr)
    sim = emission_similarity(aligned_w, aligned_r,
                              _emissions_binarized(work_path), _emissions_binarized(ref_path))
    known = np.outer(w_found, r_found)
    sim[~known] = sim[known].mean()
    return sim


def match_states(ref_segs_list, work_segs_list, matrix_out=None,
                 work_emissions_path=None, ref_emissions_path=None):
    """Match work states to reference states by Hungarian assignment on match quality.

    Quality is JACCARD_WEIGHT * Jaccard + EMISSION_WEIGHT * emission similarity,
    or Jaccard alone when no usable emissions are given. Utility applies square root
    to Jaccard to lift low-overlap focal states.

    Returns bijection mapping {work_state: ref_state}.
    """
    if not isinstance(ref_segs_list, list) or (ref_segs_list and not isinstance(ref_segs_list[0], list)):
        ref_segs_list = [ref_segs_list]
    if not isinstance(work_segs_list, list) or (work_segs_list and not isinstance(work_segs_list[0], list)):
        work_segs_list = [work_segs_list]
    
    # Broadcast ref_segs_list if we have multiple work segmentations but only one reference
    if len(ref_segs_list) == 1 and len(work_segs_list) > 1:
        ref_segs_list = ref_segs_list * len(work_segs_list)

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

    genome_len = sum(total_ref_len.values()) or 1
    jaccard = jaccard_matrix(total_overlap, total_work_len, total_ref_len,
                             work_states_sorted, ref_states_sorted)

    emission_sim = _load_emission_similarity(work_emissions_path, ref_emissions_path,
                                             work_states_sorted, ref_states_sorted)
    if emission_sim is not None:
        quality = JACCARD_WEIGHT * jaccard + EMISSION_WEIGHT * emission_sim
        utility = JACCARD_WEIGHT * np.sqrt(np.maximum(jaccard, 0.0)) + EMISSION_WEIGHT * emission_sim
    else:
        quality = jaccard
        utility = np.sqrt(np.maximum(jaccard, 0.0))

    # Concave transformation: diminishing marginal returns (square root utility on Jaccard) prioritize
    # lifting worst-off states while preserving polynomial-time Hungarian assignment.
    row_ind, col_ind = linear_sum_assignment(-utility)

    w_idx = {s: i for i, s in enumerate(work_states_sorted)}
    r_idx = {s: i for i, s in enumerate(ref_states_sorted)}

    mapping = {work_states_sorted[i]: ref_states_sorted[j]
               for i, j in zip(row_ind, col_ind)}
    unassigned = [w for w in work_states_sorted if w not in mapping]
    for w in unassigned:
        mapping[w] = w
        print(f"  {w} -> (unassigned: no reference state left)", file=sys.stderr)

    matched = [w for w in work_states_sorted if w not in unassigned]
    if matched:
        rows = [w_idx[w] for w in matched]
        cols = [r_idx[mapping[w]] for w in matched]
        emission_avg = (f"{emission_sim[rows, cols].mean():.4f} (emission) "
                        if emission_sim is not None else "")
        bp_accuracy = sum(total_overlap.get((w, mapping[w]), 0) for w in matched) / genome_len
        print(f"avg_similarity = {jaccard[rows, cols].mean():.4f} (jaccard) "
              f"{emission_avg}"
              f"{quality[rows, cols].mean():.4f} (quality) "
              f"over {len(matched)}/{n_w} matched states, "
              f"bp accuracy {bp_accuracy:.4f}",
              file=sys.stderr)
    for w in matched:
        r = mapping[w]
        wi, ri = w_idx[w], r_idx[r]
        emission = (f"emission={emission_sim[wi, ri]:.4f}, "
                    if emission_sim is not None else "")
        print(f"  {w} -> {r}  (jaccard={jaccard[wi, ri]:.4f}, {emission}"
              f"quality={quality[wi, ri]:.4f})", file=sys.stderr)

    if matrix_out:
        _save_match_matrices(matrix_out, work_states_sorted, ref_states_sorted,
                             mapping, jaccard=jaccard,
                             emission=emission_sim, quality=quality)
    return mapping


def _pick_emissions(args):
    """The work and reference emission files to match on, as (work, ref).

    Binarized emissions are preferred on each side: emission_similarity() puts
    bigwig ones on the binarized scale, so the two sides need not be the same type.
    """
    work_bw = args.work_emissions[0] if args.work_emissions else None
    work_bin = args.work_bin_emissions[0] if args.work_bin_emissions else None
    work = {"bw": work_bw, "bin": work_bin}.get(args.work_em_type) or work_bin or work_bw
    ref = ({"bw": args.ref_bw_emissions, "bin": args.ref_bin_emissions}.get(args.ref_em_type)
           or args.ref_bin_emissions or args.ref_bw_emissions)
    return work, ref


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
    ap.add_argument("--ref-bw-emissions", default=None, dest="ref_bw_emissions",
                    help="Reference bigwig emissions .npz for matching guidance")
    ap.add_argument("--ref-bin-emissions", default=None, dest="ref_bin_emissions",
                    help="Reference binarized emissions .npz for matching guidance")
    ap.add_argument("--compare-only",   default=None, dest="compare_only",
                    help="Write Jaccard heatmap to this directory instead of rewriting BED")
    ap.add_argument("--matrix-out",     default=None, dest="matrix_out",
                    help="Path prefix to persist the per-state matching matrices "
                         "(.jaccard.tsv/.emission.tsv/.quality.tsv/.mapping.tsv/.png)")
    ap.add_argument("--work-em-type",   default=None, choices=["bw", "bin"],
                    help="Preferred emission type for work segmentation")
    ap.add_argument("--ref-em-type",    default=None, choices=["bw", "bin"],
                    help="Preferred emission type for reference segmentation")

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

    work_em, ref_em = _pick_emissions(args)
    mapping = match_states(ref_segs_list, work_segs_list, matrix_out=args.matrix_out,
                           work_emissions_path=work_em, ref_emissions_path=ref_em)

    if args.compare_only:
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

    # Apply mapping to all work segmentations
    for i, work_segs in enumerate(work_segs_list):
        colors = state_colors(ref_segs_list[i])
        out_f = open(args.out[i], "w") if args.out else sys.stdout
        for row in work_segs:
            chrom, s, e, name = row[:4]
            color = row[4] if len(row) > 4 else "0,0,0"
            raw_new_name = mapping.get(name, name)
            new_name  = utils.normalize_state_name(raw_new_name)
            new_color = colors.get(raw_new_name, color)
            out_f.write(f"{chrom}\t{s}\t{e}\t{new_name}\t0\t.\t{s}\t{e}\t{new_color}\n")
        if args.out:
            out_f.close()

    # Remap emissions (handles numeric equivalence such as "1.0" vs "1")
    def _mapped(state):
        if state in mapping:
            return mapping[state]
        try:
            alt = str(int(float(state)))
        except (TypeError, ValueError):
            alt = None
        if alt is not None and alt in mapping:
            return mapping[alt]
        print(f"Warning: emission state {state!r} is in no mapping, left as is",
              file=sys.stderr)
        return state

    def _remap_em_list(in_paths, out_paths):
        if not in_paths or not out_paths: return
        if len(in_paths) != len(out_paths):
            print("Warning: number of emission files doesn't match remapped paths", file=sys.stderr)
            return
        for ip, op in zip(in_paths, out_paths):
            states, marks, mat = _load_emissions_npz(ip)
            remapped = [utils.normalize_state_name(_mapped(s)) for s in states]
            _save_emissions_npz(op, remapped, marks, mat)

    _remap_em_list(args.work_emissions, args.remap_emissions)
    _remap_em_list(args.work_bin_emissions, args.remap_bin_emissions)


if __name__ == "__main__":
    main()
