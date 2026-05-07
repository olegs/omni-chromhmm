#!/usr/bin/env python3
# Cross-segmentation comparison: entropy, kappa, Jaccard, segment stats.
#
# Shared IO helpers are imported from analyze.py.
#
# Usage:
#   compare.py --seg SEG1.bed SEG2.bed ... --bins BIN [BIN ...] --outdir OUT \
#       [--analysis-dir OUT] [--threads N]

import argparse
import os
import sys
from collections import defaultdict

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
import seaborn as sns

from analyze import (load_bed, _natural_sort_key, _load_seg_full,
                     build_transition_matrix, transition_entropy)

_EXCLUDE_STATES = {"Quies", "Het"}
from utils import seg_label as _seg_label, is_replicate as _is_replicate, \
                   should_compare as _should_compare


def _build_seg_to_analysis_map(seg_paths, analysis_dir):
    """Map segmentation BED paths to their analysis subdirectories.

    Pooled methods are looked up directly in analysis_dir.
    Replicate methods (label ending in _rep1/_rep2) are looked up in their
    per-replicate analysis dir: {ds}/repN/analysis/{variant}/{method_base}/.
    The variant is inferred from the basename of analysis_dir (comb/bwem/ovlp).
    Returns {seg_path: subdir_path}.
    """
    if not analysis_dir or not os.path.isdir(analysis_dir):
        return {}
    variant = os.path.basename(analysis_dir)
    available = set(os.listdir(analysis_dir))
    mapping = {}
    for seg_path in seg_paths:
        label = _seg_label(seg_path)
        if label.startswith("ENCFF"):
            if "ref" in available:
                mapping[seg_path] = os.path.join(analysis_dir, "ref")
        elif label in available:
            mapping[seg_path] = os.path.join(analysis_dir, label)
        elif _is_replicate(label):
            # Replicate seg: analysis lives in {ds}/repN/analysis/{variant}/{method_base}/
            parts = seg_path.replace("\\", "/").split("/")
            ds = parts[0]
            rep = next((p for p in parts if p in ("rep1", "rep2")), None)
            method_base = label.rsplit("_", 1)[0]  # strip _rep1 / _rep2
            if rep:
                rep_dir = os.path.join(ds, rep, "analysis", variant, method_base)
                if os.path.isdir(rep_dir):
                    mapping[seg_path] = rep_dir
    return mapping


# ---------------------------------------------------------------------------
# Transition entropy
# ---------------------------------------------------------------------------

def _compute_entropy(seg_paths, bin_sizes, exclude_states=None):
    """Compute transition entropy totals for each segmentation.

    bin_sizes: list of bin sizes parallel to seg_paths (one per segmentation).
    Returns list of {segmentation, total_entropy} dicts.
    """
    results = []
    exclude_label = (f" (excluding {', '.join(sorted(exclude_states))})"
                     if exclude_states else "")
    for seg_path, bin_size in zip(seg_paths, bin_sizes):
        segs = load_bed(seg_path)
        if not segs:
            print(f"  WARNING: empty segmentation {seg_path}", file=sys.stderr)
            continue
        states, counts, state_bp = build_transition_matrix(segs, bin_size, exclude_states)
        if not states:
            print(f"  WARNING: no states left after exclusion in {seg_path}", file=sys.stderr)
            continue
        total_H, _, _, _ = transition_entropy(states, counts, state_bp)
        label = _seg_label(seg_path)
        results.append({"segmentation": label, "total_entropy": total_H})
        print(f"  {label}{exclude_label}: total transition entropy = {total_H:.4f}")
    return results


def _save_entropy_summary(results, outdir, suffix="", title_extra=""):
    """Save entropy summary TSV and bar chart."""
    if not results:
        return
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results)
    summary_path = os.path.join(outdir, f"entropy_summary{suffix}.tsv")
    df.to_csv(summary_path, sep="\t", index=False, float_format="%.4f")
    print(f"  saved {summary_path}")

    fig, ax = plt.subplots(figsize=(max(5, len(df) * 0.5), 3.5))
    ax.bar(range(len(df)), df["total_entropy"])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["segmentation"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total transition matrix entropy (bits)")
    ax.set_title(f"Transition matrix entropy comparison{title_extra}")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(df["total_entropy"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig_path = os.path.join(outdir, f"entropy_summary{suffix}.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  saved {fig_path}")


def _save_entropy_combined_plot(results_full, results_active, outdir):
    """Grouped bar chart: all-states vs excluding Quies/Het entropy."""
    if not results_full:
        return
    df_full = pd.DataFrame(results_full)
    df_active = pd.DataFrame(results_active) if results_active else pd.DataFrame()

    x = np.arange(len(df_full))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(5, len(df_full) * 0.5), 3.5))
    ax.bar(x - width / 2, df_full["total_entropy"], width,
           label="All states", color="#4878CF")
    if not df_active.empty:
        ax.bar(x + width / 2, df_active["total_entropy"], width,
               label="Excl. Quies/Het", color="#E8833A")
    ax.set_xticks(x)
    ax.set_xticklabels(df_full["segmentation"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total transition matrix entropy (bits)")
    ax.set_title("Transition matrix entropy comparison")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(outdir, "entropy_summary_combined.png")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  saved {fig_path}")


# ---------------------------------------------------------------------------
# Cohen's Kappa
# ---------------------------------------------------------------------------

def segmentation_to_bins(segs, bin_size):
    """Convert a segmentation to a dict: {chrom: {bin_index: state}}."""
    bins = defaultdict(dict)
    for chrom, s, e, state in segs:
        for b in range(s // bin_size, e // bin_size):
            bins[chrom][b] = state
    return bins


def _filter_bins(bins, exclude_states):
    """Return bins with entries whose state is in exclude_states removed."""
    return {chrom: {b: s for b, s in bmap.items() if s not in exclude_states}
            for chrom, bmap in bins.items()}


def _aligned_label_arrays(bins1, bins2):
    """Aligned label arrays from two bin-level segmentations (shared bins only)."""
    common_chroms = set(bins1.keys()) & set(bins2.keys())
    l1, l2 = [], []
    for chrom in sorted(common_chroms):
        common_bins = set(bins1[chrom].keys()) & set(bins2[chrom].keys())
        for b in sorted(common_bins):
            l1.append(bins1[chrom][b])
            l2.append(bins2[chrom][b])
    return np.array(l1), np.array(l2)



def compute_kappa(bins1, bins2):
    """Compute Cohen's Kappa. Returns (kappa, po, pe, n_bins, confusion_df)."""
    labels1, labels2 = _aligned_label_arrays(bins1, bins2)
    n = len(labels1)
    if n == 0:
        return 0.0, 0.0, 0.0, 0, pd.DataFrame()

    all_states = sorted(set(labels1) | set(labels2), key=_natural_sort_key)
    state_idx = {s: i for i, s in enumerate(all_states)}
    k = len(all_states)

    conf = np.zeros((k, k), dtype=np.int64)
    for l1, l2 in zip(labels1, labels2):
        conf[state_idx[l1], state_idx[l2]] += 1

    po = np.diag(conf).sum() / n
    p1 = conf.sum(axis=1) / n
    p2 = conf.sum(axis=0) / n
    pe = np.dot(p1, p2)
    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0

    return kappa, po, pe, n, pd.DataFrame(conf, index=all_states, columns=all_states)


def compute_jaccard(bins1, bins2):
    """Mean per-state Jaccard index between two bin-level segmentations."""
    labels1, labels2 = _aligned_label_arrays(bins1, bins2)
    if len(labels1) == 0:
        return 0.0
    states = sorted(set(labels1) | set(labels2), key=_natural_sort_key)
    jaccards = []
    for s in states:
        a = labels1 == s
        b = labels2 == s
        tp = int((a & b).sum())
        denom = int((a | b).sum())
        jaccards.append(tp / denom if denom > 0 else 1.0)
    return float(np.mean(jaccards))


def compute_per_state_kappa(bins1, bins2):
    """One-vs-rest Cohen's Kappa for each state present in either segmentation."""
    labels1, labels2 = _aligned_label_arrays(bins1, bins2)
    if len(labels1) == 0:
        return {}
    states = sorted(set(labels1) | set(labels2), key=_natural_sort_key)
    out = {}
    for s in states:
        a = (labels1 == s)
        b = (labels2 == s)
        po = float((a == b).mean())
        pa, pb = float(a.mean()), float(b.mean())
        pe = pa * pb + (1 - pa) * (1 - pb)
        out[s] = (po - pe) / (1 - pe) if pe < 1.0 else 1.0
    return out


# ---------------------------------------------------------------------------
# Emission similarity
# ---------------------------------------------------------------------------


def _load_emissions_npz(path):
    """Load a .npz emissions file → (states, matrix) or None."""
    if not path or not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    states = list(data["states"])
    mat = data["mat"].astype(np.float64)
    return states, mat




# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------

def _compare_pair(i, j, path_i, path_j, label_i, label_j,
                  bin_emission_path_i, bin_emission_path_j,
                  bw_emission_path_i, bw_emission_path_j,
                  bin_size_i, bin_size_j, outdir):
    """Compare one pair of segmentations (runs in a worker process).

    Uses the finer of the two bin sizes for kappa so that 200bp segments
    are compared at 100bp resolution when paired with a 100bp segmentation.
    """
    from match import (pair_overlap as match_pair_overlap,
                       best_mapping as match_best_mapping,
                       compare as match_compare,
                       state_lengths as match_state_lengths,
                       emission_cosine_mapping)

    bin_size = min(bin_size_i, bin_size_j)
    segs_i = load_bed(path_i)
    segs_j = load_bed(path_j)
    bins_i = segmentation_to_bins(segs_i, bin_size)
    bins_j = segmentation_to_bins(segs_j, bin_size)

    row = {"seg1": label_i, "seg2": label_j, "_i": i, "_j": j}

    kappa, po, pe, n_bins, _ = compute_kappa(bins_i, bins_j)
    row.update(kappa=kappa, po=po, pe=pe, n_bins=n_bins)
    row["_per_state_kappa"] = compute_per_state_kappa(bins_i, bins_j)

    segs_full_i = _load_seg_full(path_i)
    segs_full_j = _load_seg_full(path_j)
    overlap = match_pair_overlap(segs_full_i, segs_full_j)
    work_states = sorted({x[3] for x in segs_full_j}, key=_natural_sort_key)
    ref_states  = sorted({x[3] for x in segs_full_i}, key=_natural_sort_key)
    mapping = match_best_mapping(overlap, work_states, ref_states)
    pair_dir = os.path.join(outdir, "pairs", f"{label_i}_vs_{label_j}")
    match_compare(segs_full_i, segs_full_j, overlap, mapping, pair_dir)

    sim = 0.0
    sim_path = os.path.join(pair_dir, "similarity.txt")
    if os.path.exists(sim_path):
        with open(sim_path) as f:
            for line in f:
                if "=" in line:
                    sim = float(line.split("=")[1].strip())
    row["jaccard_similarity"] = sim

    # Overlap fraction: fraction of genome bp with identical state labels.
    all_states = {x[3] for x in segs_full_i} | {x[3] for x in segs_full_j}
    total_same = sum(overlap.get((s, s), 0) for s in all_states)
    genome_len = sum(match_state_lengths(segs_full_i).values()) or 1
    row["overlap_fraction"] = total_same / genome_len

    em_i = _load_emissions_npz(bin_emission_path_i)
    em_j = _load_emissions_npz(bin_emission_path_j)
    if em_i is not None and em_j is not None:
        avg_sim, em_mapping = emission_cosine_mapping(em_i[0], em_i[1], em_j[0], em_j[1])
        row["emission_similarity"] = avg_sim
        row["emission_mapping"] = "; ".join(
            f"{k}->{v}" for k, v in sorted(em_mapping.items()))

    bw_i = _load_emissions_npz(bw_emission_path_i)
    bw_j = _load_emissions_npz(bw_emission_path_j)
    if bw_i is not None and bw_j is not None:
        avg_bw_sim, _ = emission_cosine_mapping(bw_i[0], bw_i[1], bw_j[0], bw_j[1])
        row["bw_emission_similarity"] = avg_bw_sim

    # No-Quies/Het variants.
    bins_i_noqh = _filter_bins(bins_i, _EXCLUDE_STATES)
    bins_j_noqh = _filter_bins(bins_j, _EXCLUDE_STATES)
    kappa_noqh, po_noqh, _, _, _ = compute_kappa(bins_i_noqh, bins_j_noqh)
    row["kappa_noqh"]   = kappa_noqh
    row["overlap_noqh"] = po_noqh
    row["jaccard_noqh"] = compute_jaccard(bins_i_noqh, bins_j_noqh)

    print(f"  {label_i} vs {label_j}: kappa={kappa:.4f}, "
          f"jaccard={sim:.4f}, overlap={row['overlap_fraction']:.4f}"
          + (f", emission={row['emission_similarity']:.4f}"
             if "emission_similarity" in row else ""))
    return row



def compare_all(seg_paths, bin_sizes, outdir, analysis_dir=None, threads=None,
                label_override=None, all_pairs=False):
    """Selective segmentation comparison; saves metric matrices as TSV.

    bin_sizes: list of bin sizes parallel to seg_paths.
    For each pair the finer (smaller) of the two bin sizes is used for kappa,
    so methods with different native resolutions are compared fairly.

    Two classes of pairs are compared by default:
      1. Each pooled segmentation vs the ENCODE reference.
      2. Rep1 vs rep2 within the same method (replicate consistency).
    All other cross-method or replicate-vs-reference pairs are skipped.

    label_override: optional dict {seg_path: label} to override _seg_label().
    all_pairs: if True, compare every pair regardless of _should_compare().
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    os.makedirs(outdir, exist_ok=True)
    seg_outdirs = _build_seg_to_analysis_map(seg_paths, analysis_dir)
    labels = [(label_override or {}).get(p) or _seg_label(p) for p in seg_paths]
    n = len(seg_paths)

    bin_emission_paths = {
        i: p.replace(".bed", ".bin_emissions.npz")
        for i, p in enumerate(seg_paths)
        if os.path.exists(p.replace(".bed", ".bin_emissions.npz"))
    }

    bw_emission_paths = {
        i: p.replace(".bed", ".bw_emissions.npz")
        for i, p in enumerate(seg_paths)
        if os.path.exists(p.replace(".bed", ".bw_emissions.npz"))
    }

    kappa_mat        = np.zeros((n, n))
    jaccard_mat      = np.zeros((n, n))
    overlap_mat      = np.zeros((n, n))
    kappa_noqh_mat   = np.zeros((n, n))
    overlap_noqh_mat = np.zeros((n, n))
    jaccard_noqh_mat = np.zeros((n, n))
    em_sim_mat       = np.zeros((n, n))
    bw_sim_mat       = np.zeros((n, n))

    pair_order = [
        (i, j)
        for i in range(n) for j in range(i + 1, n)
        if all_pairs or _should_compare(labels[i], labels[j])
    ]
    pair_desc = "all pairs" if all_pairs else "vs ref + same-method rep1/rep2"
    n_workers = min(len(pair_order), threads or os.cpu_count() or 4)
    print(f"  Comparing {len(pair_order)} pairs ({pair_desc}) "
          f"with {n_workers} processes ...", file=sys.stderr)

    comparison_rows = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_compare_pair, i, j,
                            seg_paths[i], seg_paths[j], labels[i], labels[j],
                            bin_emission_paths.get(i), bin_emission_paths.get(j),
                            bw_emission_paths.get(i), bw_emission_paths.get(j),
                            bin_sizes[i], bin_sizes[j], outdir): (i, j)
            for i, j in pair_order
        }
        for fut in as_completed(futures):
            row = fut.result()
            i, j = row.pop("_i"), row.pop("_j")
            kappa_mat[i, j] = kappa_mat[j, i] = row["kappa"]
            jaccard_mat[i, j] = jaccard_mat[j, i] = row["jaccard_similarity"]
            overlap_mat[i, j] = overlap_mat[j, i] = row["overlap_fraction"]
            if "emission_similarity" in row:
                em_sim_mat[i, j] = em_sim_mat[j, i] = row["emission_similarity"]
            if "bw_emission_similarity" in row:
                bw_sim_mat[i, j] = bw_sim_mat[j, i] = row["bw_emission_similarity"]
            kappa_noqh_mat[i, j]   = kappa_noqh_mat[j, i]   = row["kappa_noqh"]
            overlap_noqh_mat[i, j] = overlap_noqh_mat[j, i] = row["overlap_noqh"]
            jaccard_noqh_mat[i, j] = jaccard_noqh_mat[j, i] = row["jaccard_noqh"]
            comparison_rows.append(row)

    # Per-state kappa vs reference
    ps_rows = []
    for row in comparison_rows:
        for state, k in (row.pop("_per_state_kappa", None) or {}).items():
            ps_rows.append({"seg1": row["seg1"], "seg2": row["seg2"],
                            "state": state, "kappa": k})
    if ps_rows:
        ps_df = pd.DataFrame(ps_rows)
        ps_df.to_csv(os.path.join(outdir, "per_state_kappa.tsv"),
                     sep="\t", index=False, float_format="%.4f")
        ref_label = labels[0]
        wide_a = ps_df[ps_df["seg1"] == ref_label].pivot_table(
            index="state", columns="seg2", values="kappa", aggfunc="mean")
        wide_b = ps_df[ps_df["seg2"] == ref_label].pivot_table(
            index="state", columns="seg1", values="kappa", aggfunc="mean")
        wide = wide_a.combine_first(wide_b) if not wide_b.empty else wide_a
        if not wide.empty:
            wide = wide.reindex(sorted(wide.index, key=_natural_sort_key))
            wide.to_csv(os.path.join(outdir, f"per_state_kappa_vs_{ref_label}.tsv"),
                        sep="\t", float_format="%.4f")
            vmax = max(float(np.nanmax(np.abs(wide.values))), 0.1)
            fig, ax = plt.subplots(figsize=(max(6, wide.shape[1] * 0.8),
                                            max(4, wide.shape[0] * 0.35)))
            sns.heatmap(wide, cmap="RdYlGn", vmin=-vmax, vmax=vmax, center=0,
                        linewidths=0.5, annot=True, fmt=".2f",
                        annot_kws={"fontsize": 7},
                        cbar_kws={"label": "Per-state Cohen's Kappa"},
                        ax=ax, mask=wide.isna().values)
            ax.set_title(f"Per-state Cohen's Kappa vs {ref_label}", fontsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"per_state_kappa_vs_{ref_label}.png"))
            plt.close(fig)

    pd.DataFrame(comparison_rows).to_csv(
        os.path.join(outdir, "comparison_all_pairs.tsv"),
        sep="\t", index=False, float_format="%.4f")

    # Save matrices (TSV only — no heatmaps)
    def _save_matrix(mat, name):
        df = pd.DataFrame(mat, index=labels, columns=labels)
        df.to_csv(os.path.join(outdir, f"{name}_matrix.tsv"),
                  sep="\t", float_format="%.4f")
        print(f"  saved {outdir}/{name}_matrix.tsv")
        return df

    kappa_df   = _save_matrix(kappa_mat, "kappa")
    jaccard_df = _save_matrix(jaccard_mat, "jaccard_similarity")
    _save_matrix(overlap_mat, "overlap")
    em_df = _save_matrix(em_sim_mat, "emission_similarity")
    _save_matrix(bw_sim_mat, "bw_emission_similarity")
    _save_matrix(kappa_noqh_mat,   "kappa_noqh")
    _save_matrix(overlap_noqh_mat, "overlap_noqh")
    _save_matrix(jaccard_noqh_mat, "jaccard_noqh")

    # Per-seg comparison rows in analysis dirs
    for i, p in enumerate(seg_paths):
        seg_dir = seg_outdirs.get(p)
        if not seg_dir:
            continue
        comp_dir = os.path.join(seg_dir, "comparison")
        os.makedirs(comp_dir, exist_ok=True)
        kappa_df.iloc[[i]].to_csv(os.path.join(comp_dir, "kappa_vs_all.tsv"),
                                   sep="\t", float_format="%.4f")
        jaccard_df.iloc[[i]].to_csv(os.path.join(comp_dir, "jaccard_vs_all.tsv"),
                                     sep="\t", float_format="%.4f")
        if em_df is not None and i in bin_emission_paths:
            em_df.iloc[[i]].dropna(axis=1).to_csv(
                os.path.join(comp_dir, "emission_similarity_vs_all.tsv"),
                sep="\t", float_format="%.4f")


# ---------------------------------------------------------------------------
# Segment length statistics
# ---------------------------------------------------------------------------

def compute_segment_stats(segs):
    """Compute segment length statistics. Returns a dict."""
    if not segs:
        return {}
    lengths = np.array([e - s for _, s, e, _ in segs])
    return {
        "n_states":      len({st for _, _, _, st in segs}),
        "n_segments":    len(lengths),
        "min_length":    int(np.min(lengths)),
        "max_length":    int(np.max(lengths)),
        "mean_length":   float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
    }


def _plot_segment_stats(df, outdir):
    """Bar charts for each segment stats metric."""
    x = np.arange(len(df))
    xlabels = df["segmentation"]
    for col, title, ylabel in [
        ("n_states",      "Total number of states",   "Count"),
        ("n_segments",    "Total number of segments", "Count"),
        ("min_length",    "Min segment length",       "bp"),
        ("max_length",    "Max segment length",       "bp"),
        ("mean_length",   "Mean segment length",      "bp"),
        ("median_length", "Median segment length",    "bp"),
    ]:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(max(4, len(df) * 0.5), 3.5))
        vals = df[col].values
        ax.bar(x, vals, color="#4878CF", edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=55, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            fmt = f"{v:.0f}" if v == int(v) else f"{v:.1f}"
            ax.text(i, v + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                    fmt, ha="center", va="bottom", fontsize=6)
        fig.tight_layout()
        path = os.path.join(outdir, f"{col}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  saved {path}")


def run_segment_stats(seg_paths, outdir, analysis_dir=None):
    """Compute and save segment length statistics for each segmentation."""
    os.makedirs(outdir, exist_ok=True)
    seg_outdirs = _build_seg_to_analysis_map(seg_paths, analysis_dir)
    results = []

    for seg_path in seg_paths:
        segs = load_bed(seg_path)
        if not segs:
            print(f"  WARNING: empty segmentation {seg_path}", file=sys.stderr)
            continue
        stats = compute_segment_stats(segs)
        label = _seg_label(seg_path)
        stats["segmentation"] = label
        results.append(stats)
        print(f"  {label}: {stats['n_states']} states, {stats['n_segments']} segments, "
              f"lengths [{stats['min_length']}, {stats['max_length']}], "
              f"mean={stats['mean_length']:.0f}, median={stats['median_length']:.0f}")

        seg_dir = seg_outdirs.get(seg_path)
        if seg_dir:
            stats_dir = os.path.join(seg_dir, "segment_stats")
            os.makedirs(stats_dir, exist_ok=True)
            pd.DataFrame([stats]).to_csv(os.path.join(stats_dir, "segment_stats.tsv"),
                                          sep="\t", index=False, float_format="%.1f")

    if not results:
        return

    col_order = ["segmentation", "n_states", "n_segments",
                 "min_length", "max_length", "mean_length", "median_length"]
    df = pd.DataFrame(results)[col_order]
    summary_path = os.path.join(outdir, "segment_stats.tsv")
    df.to_csv(summary_path, sep="\t", index=False, float_format="%.1f")
    print(f"  saved {summary_path}")
    _plot_segment_stats(df, outdir)


# ---------------------------------------------------------------------------
# Plot-only mode
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Cross-segmentation comparison: entropy, kappa, Jaccard, segment stats.")
    ap.add_argument("--seg", nargs="+", required=True,
                    help="Two or more segmentation BED files")
    ap.add_argument("--bins", nargs="+", type=int, required=True,
                    help="Bin size(s) in bp — one per --seg file, or a single value for all")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--analysis-dir", default=None, dest="analysis_dir",
                    help="Analysis root dir for per-segmentation metric output")
    ap.add_argument("--threads", type=int, default=None,
                    help="Max parallel workers for comparison jobs (default: cpu_count)")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Explicit labels for --seg files (one per file); "
                         "overrides path-derived labels from seg_label()")
    ap.add_argument("--all-pairs", action="store_true", dest="all_pairs",
                    help="Compare every pair of segmentations (bypasses should_compare filter)")
    args = ap.parse_args()

    # Broadcast a single bin size to all segmentations if needed.
    bin_sizes = (args.bins * len(args.seg) if len(args.bins) == 1
                 else args.bins)
    if len(bin_sizes) != len(args.seg):
        ap.error(f"--bins must have 1 value or one per --seg file "
                 f"({len(args.seg)} files, {len(args.bins)} bin sizes given)")
    if args.labels and len(args.labels) != len(args.seg):
        ap.error(f"--labels must have one value per --seg file "
                 f"({len(args.seg)} files, {len(args.labels)} labels given)")

    label_override = dict(zip(args.seg, args.labels)) if args.labels else None
    analysis_dir   = args.analysis_dir or args.outdir
    comparison_dir = args.outdir

    results_full = _compute_entropy(args.seg, bin_sizes)
    _save_entropy_summary(results_full, comparison_dir)

    results_active = _compute_entropy(args.seg, bin_sizes, exclude_states=_EXCLUDE_STATES)
    _save_entropy_summary(results_active, comparison_dir, suffix="_noqh",
                          title_extra=f"\n(excluding {', '.join(sorted(_EXCLUDE_STATES))})")
    _save_entropy_combined_plot(results_full, results_active, comparison_dir)

    run_segment_stats(args.seg, comparison_dir, analysis_dir)
    compare_all(args.seg, bin_sizes, comparison_dir, analysis_dir, args.threads,
                label_override=label_override, all_pairs=args.all_pairs)


if __name__ == "__main__":
    main()
