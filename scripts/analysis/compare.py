#!/usr/bin/env python3
# Cross-segmentation comparison: entropy, kappa, Jaccard, segment stats.

import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
import seaborn as sns

_analysis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "analysis"))
if _analysis_dir not in sys.path:
    sys.path.insert(0, _analysis_dir)

from analyze import (load_bed, _natural_sort_key, _load_seg_full,
                     build_transition_matrix, transition_entropy)

_rules_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rules"))
if _rules_dir not in sys.path:
    sys.path.insert(0, _rules_dir)

import match
import utils
from utils import seg_label as _seg_label, is_replicate as _is_replicate, \
                   should_compare as _should_compare, display_name, \
                   method_color, save_fig, NOQH_STATES as _EXCLUDE_STATES, \
                   JACCARD, KAPPA, NOQH_SUFFIX, \
                   JACCARD_DISPLAY, KAPPA_DISPLAY, FULL_DISPLAY, NOQH_DISPLAY


def _build_seg_to_analysis_map(seg_paths, analysis_dir):
    """Return {seg_path: analysis subdir}.

    Pooled methods live directly in analysis_dir; replicates live in
    {ds}/repN/analysis/{variant}/{method_base}/, where the variant is the
    basename of analysis_dir (comb/bwem/ovlp).
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
            parts = seg_path.replace("\\", "/").split("/")
            ds = parts[0]
            rep = next((p for p in parts if p in ("rep1", "rep2", "replicate1", "replicate2")), None)
            method_base = label.rsplit("_", 1)[0]
            if rep:
                rep_dir = os.path.join(ds, rep, "analysis", variant, method_base)
                if os.path.isdir(rep_dir):
                    mapping[seg_path] = rep_dir
    return mapping


def _compute_entropy(seg_paths, bin_sizes, exclude_states=None, mappings=None):
    """Transition entropy totals: [{segmentation, total_entropy}, ...].

    bin_sizes: one bin size per segmentation, parallel to seg_paths.
    """
    results = []
    exclude_label = (f" (excluding {', '.join(sorted(exclude_states))})"
                     if exclude_states else "")
    for i, (seg_path, bin_size) in enumerate(zip(seg_paths, bin_sizes)):
        segs = load_bed(seg_path)
        if not segs:
            print(f"  WARNING: empty segmentation {seg_path}", file=sys.stderr)
            continue
        mapping = mappings[i] if mappings else None
        states, counts, state_bp = build_transition_matrix(segs, bin_size, exclude_states, mapping)
        if not states:
            print(f"  WARNING: no states left after exclusion in {seg_path}", file=sys.stderr)
            continue
        total_H, _, _, _ = transition_entropy(states, counts, state_bp)
        label = _seg_label(seg_path)
        results.append({"segmentation": label, "total_entropy": total_H})
        print(f"  {label}{exclude_label}: total transition entropy = {total_H:.4f}")
    return results


def _get_method_style(label):
    """(display_name, color) for a segmentation label."""
    return display_name(label), method_color(label)


def _save_entropy_summary(results, outdir, suffix="", title_extra=""):
    """Entropy summary TSV and bar chart."""
    if not results:
        return
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results)
    summary_path = os.path.join(outdir, f"entropy_summary{suffix}.tsv")
    df.to_csv(summary_path, sep="\t", index=False, float_format="%.4f")
    print(f"  saved {summary_path}")

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.8), 4.2))

    styles = [_get_method_style(l) for l in df["segmentation"]]
    display_names = [s[0] for s in styles]
    colors = [s[1] for s in styles]
    
    sns.barplot(data=df, x="segmentation", y="total_entropy", ax=ax,
                palette={l: c for l, c in zip(df["segmentation"], colors)},
                hue="segmentation", dodge=False,
                edgecolor="lightgrey", linewidth=1)
    
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total transition matrix entropy (bits)", fontsize=9)
    ax.set_title(f"Transition matrix entropy comparison{title_extra}", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, v in enumerate(df["total_entropy"]):
        ax.text(i, v + yrange * 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    
    save_fig(fig, os.path.join(outdir, f"entropy_summary{suffix}.png"))


def _save_entropy_combined_plot(results_full, results_active, outdir):
    """Grouped bar chart: all-states vs excluding Quies/Het entropy."""
    if not results_full:
        return
    df_full = pd.DataFrame(results_full)
    df_active = pd.DataFrame(results_active) if results_active else pd.DataFrame()

    df_full["Type"] = FULL_DISPLAY
    if not df_active.empty:
        df_active["Type"] = NOQH_DISPLAY
    
    df_combined = pd.concat([df_full, df_active], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(max(6, len(df_full) * 0.8), 4.2))
    
    sns.barplot(data=df_combined, x="segmentation", y="total_entropy", hue="Type",
                ax=ax, palette={FULL_DISPLAY: "#4878CF", NOQH_DISPLAY: "#E8833A"},
                capsize=0.05, edgecolor="lightgrey", linewidth=1)
    
    display_names = [_get_method_style(l)[0] for l in df_full["segmentation"]]
    ax.set_xticks(range(len(df_full)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total transition matrix entropy (bits)", fontsize=9)
    ax.set_title("Transition matrix entropy comparison", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    save_fig(fig, os.path.join(outdir, "entropy_summary_combined.png"))


def segmentation_to_bins(segs, bin_size):
    """Convert a segmentation to {chrom: {bin_index: state}}."""
    bins = defaultdict(dict)
    for row in segs:
        chrom, s, e, state = row[:4]
        for b in range(s // bin_size, e // bin_size):
            bins[chrom][b] = state
    return bins


def _filter_bins(bins, exclude_states):
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
    """Cohen's Kappa; returns (kappa, po, pe, n_bins, confusion_df)."""
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
    return float(np.mean(list(compute_per_state_jaccard(bins1, bins2).values())))


def compute_per_state_jaccard(bins1, bins2):
    """Jaccard similarity for each state present in either segmentation."""
    labels1, labels2 = _aligned_label_arrays(bins1, bins2)
    if len(labels1) == 0:
        return {}
    states = sorted(set(labels1) | set(labels2), key=_natural_sort_key)
    out = {}
    for s in states:
        a = (labels1 == s)
        b = (labels2 == s)
        tp = int((a & b).sum())
        denom = int((a | b).sum())
        out[s] = tp / denom if denom > 0 else 1.0
    return out


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


def _load_emissions_npz(path):
    """Load a .npz emissions file → (states, matrix) or None."""
    if not path or not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    states = list(data["states"])
    mat = data["mat"].astype(np.float64)
    return states, mat


def compute_composition_similarity(segs1, segs2, exclude_states=None):
    """Cosine similarity of state distributions (by total bp)."""
    import match
    l1 = match.state_lengths(segs1)
    l2 = match.state_lengths(segs2)
    if exclude_states:
        l1 = {s: v for s, v in l1.items() if s not in exclude_states}
        l2 = {s: v for s, v in l2.items() if s not in exclude_states}
    states = sorted(set(l1.keys()) | set(l2.keys()))
    v1 = np.array([l1.get(s, 0) for s in states], dtype=np.float64)
    v2 = np.array([l2.get(s, 0) for s in states], dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return np.dot(v1, v2) / (n1 * n2)


def _compare_pair(i, j, path_i, path_j, label_i, label_j,
                  bin_emission_path_i, bin_emission_path_j,
                  bw_emission_path_i, bw_emission_path_j,
                  bin_size_i, bin_size_j, outdir, skip_noqh=False):
    """Compare one pair of segmentations (runs in a worker process)."""
    import match
    match_pair_overlap = match.pair_overlap
    match_best_mapping = match.best_mapping
    match_compare = match.compare
    match_state_lengths = match.state_lengths
    match_emission_cosine_mapping = match.emission_cosine_mapping

    bin_size = min(bin_size_i, bin_size_j)
    segs_i = load_bed(path_i)
    segs_j = load_bed(path_j)

    row = {"seg1": label_i, "seg2": label_j, "_i": i, "_j": j}

    segs_full_i = _load_seg_full(path_i)
    segs_full_j = _load_seg_full(path_j)

    row["composition_similarity"] = compute_composition_similarity(segs_full_i, segs_full_j)
    if not skip_noqh:
        row["composition_noqh_similarity"] = compute_composition_similarity(segs_full_i, segs_full_j, exclude_states=_EXCLUDE_STATES)

    overlap = match_pair_overlap(segs_full_i, segs_full_j)
    work_states = sorted({x[3] for x in segs_full_j}, key=_natural_sort_key)
    ref_states  = sorted({x[3] for x in segs_full_i}, key=_natural_sort_key)
    mapping = match_best_mapping(overlap, work_states, ref_states)

    bins_i = segmentation_to_bins(segs_i, bin_size)
    bins_j = segmentation_to_bins(segs_j, bin_size)

    eff_bins_j = bins_j   # base-wise metrics use bins_j as-is, no remapping

    kappa, po, pe, n_bins, _ = compute_kappa(bins_i, eff_bins_j)
    row.update(kappa=kappa, po=po, pe=pe, n_bins=n_bins)
    row["_per_state_kappa"] = compute_per_state_kappa(bins_i, eff_bins_j)

    pair_dir = os.path.join(outdir, "pairs", f"{label_i}_vs_{label_j}")
    match_compare(segs_full_i, segs_full_j, overlap, mapping, pair_dir)

    row["jaccard_similarity"] = compute_jaccard(bins_i, eff_bins_j)
    row["_per_state_jaccard"] = compute_per_state_jaccard(bins_i, eff_bins_j)

    # Overlap fraction: genome bp with identical state labels.
    all_ref_states = {x[3] for x in segs_full_i}
    total_same = sum(overlap.get((s, s), 0) for s in all_ref_states)
    
    genome_len = sum(match_state_lengths(segs_full_i).values()) or 1
    row["overlap_fraction"] = total_same / genome_len

    em_i = _load_emissions_npz(bin_emission_path_i)
    em_j = _load_emissions_npz(bin_emission_path_j)
    if em_i is not None and em_j is not None:
        avg_sim, em_mapping = match_emission_cosine_mapping(em_i[0], em_i[1], em_j[0], em_j[1])
        row["emission_similarity"] = avg_sim
        row["emission_mapping"] = "; ".join(
            f"{k}->{v}" for k, v in sorted(em_mapping.items()))

    bw_i = _load_emissions_npz(bw_emission_path_i)
    bw_j = _load_emissions_npz(bw_emission_path_j)
    if bw_i is not None and bw_j is not None:
        avg_bw_sim, _ = match_emission_cosine_mapping(bw_i[0], bw_i[1], bw_j[0], bw_j[1])
        row["bw_emission_similarity"] = avg_bw_sim

    if not skip_noqh:
        bins_i_noqh = _filter_bins(bins_i, _EXCLUDE_STATES)
        bins_j_noqh = _filter_bins(eff_bins_j, _EXCLUDE_STATES)
        kappa_noqh, po_noqh, _, _, _ = compute_kappa(bins_i_noqh, bins_j_noqh)
        row[f"{KAPPA}{NOQH_SUFFIX}"]   = kappa_noqh
        row[f"overlap{NOQH_SUFFIX}"] = po_noqh
        row[f"{JACCARD}{NOQH_SUFFIX}"] = compute_jaccard(bins_i_noqh, bins_j_noqh)

    print(f"  {label_i} vs {label_j}: "
          f"comp_sim={row['composition_similarity']:.4f}, "
          f"kappa={row['kappa']:.4f}, jaccard={row['jaccard_similarity']:.4f}"
          + (f", emission={row['emission_similarity']:.4f}"
             if "emission_similarity" in row else ""))
    return row


def compare_all(seg_paths, bin_sizes, outdir, analysis_dir=None, threads=None,
                label_override=None, all_pairs=False, skip_noqh=False):
    """Selective segmentation comparison; saves metric matrices as TSV.

    bin_sizes: one per segmentation; each pair uses the finer of the two for
    kappa, so methods with different native resolutions compare fairly.

    By default only pooled-vs-reference and same-method rep1-vs-rep2 pairs are
    compared; all_pairs=True compares every pair. label_override is an optional
    {seg_path: label} dict replacing _seg_label().
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

    kappa_mat           = np.eye(n)
    comp_sim_mat        = np.eye(n)
    comp_sim_noqh_mat   = np.eye(n)
    jaccard_mat         = np.eye(n)
    overlap_mat         = np.eye(n)
    kappa_noqh_mat      = np.eye(n)
    overlap_noqh_mat    = np.eye(n)
    jaccard_noqh_mat    = np.eye(n)
    em_sim_mat          = np.eye(n)
    bw_sim_mat          = np.eye(n)

    pair_order = [
        (i, j)
        for i in range(n) for j in range(i + 1, n)
        if all_pairs or _should_compare(labels[i], labels[j])
    ]
    pair_desc = "all pairs" if all_pairs else "vs ref + same-method rep1/rep2"
    n_workers = min(len(pair_order), threads or os.cpu_count() or 4)
    if len(pair_order) > 0:
        print(f"  Comparing {len(pair_order)} pairs ({pair_desc}) "
              f"with {n_workers} processes ...", file=sys.stderr)

    comparison_rows = []

    if len(pair_order) > 0:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_compare_pair, i, j,
                                seg_paths[i], seg_paths[j], labels[i], labels[j],
                                bin_emission_paths.get(i), bin_emission_paths.get(j),
                                bw_emission_paths.get(i), bw_emission_paths.get(j),
                                bin_sizes[i], bin_sizes[j], outdir, skip_noqh=skip_noqh): (i, j)
                for i, j in pair_order
            }
            for fut in as_completed(futures):
                row = fut.result()
                i, j = row.pop("_i"), row.pop("_j")
                comp_sim_mat[i, j] = comp_sim_mat[j, i] = row["composition_similarity"]
                kappa_mat[i, j] = kappa_mat[j, i] = row["kappa"]
                jaccard_mat[i, j] = jaccard_mat[j, i] = row["jaccard_similarity"]
                overlap_mat[i, j] = overlap_mat[j, i] = row["overlap_fraction"]
                if "emission_similarity" in row:
                    em_sim_mat[i, j] = em_sim_mat[j, i] = row["emission_similarity"]
                if "bw_emission_similarity" in row:
                    bw_sim_mat[i, j] = bw_sim_mat[j, i] = row["bw_emission_similarity"]
                
                if not skip_noqh:
                    comp_sim_noqh_mat[i, j] = comp_sim_noqh_mat[j, i] = row["composition_noqh_similarity"]
                    kappa_noqh_mat[i, j]   = kappa_noqh_mat[j, i]   = row[f"{KAPPA}{NOQH_SUFFIX}"]
                    overlap_noqh_mat[i, j] = overlap_noqh_mat[j, i] = row[f"overlap{NOQH_SUFFIX}"]
                    jaccard_noqh_mat[i, j] = jaccard_noqh_mat[j, i] = row[f"{JACCARD}{NOQH_SUFFIX}"]
                comparison_rows.append(row)
    else:
        print(f"  No pairs to compare ({pair_desc}).", file=sys.stderr)

    ps_rows = []
    for row in comparison_rows:
        pk = row.pop("_per_state_kappa", None) or {}
        pj = row.pop("_per_state_jaccard", None) or {}
        for state in sorted(set(pk.keys()) | set(pj.keys()), key=_natural_sort_key):
            ps_rows.append({"seg1": row["seg1"], "seg2": row["seg2"], "state": state,
                            KAPPA: pk.get(state),
                            JACCARD: pj.get(state)})

    if ps_rows:
        ps_df = pd.DataFrame(ps_rows)
        ps_df.to_csv(os.path.join(outdir, "per_state_metrics.tsv"),
                     sep="\t", index=False, float_format="%.4f")
        ref_label = labels[0]
        for metric, label in [(KAPPA, f"Cohen's {KAPPA_DISPLAY}"),
                              (JACCARD, JACCARD_DISPLAY)]:
            wide_a = ps_df[ps_df["seg1"] == ref_label].pivot_table(
                index="state", columns="seg2", values=metric, aggfunc="mean")
            wide_b = ps_df[ps_df["seg2"] == ref_label].pivot_table(
                index="state", columns="seg1", values=metric, aggfunc="mean")
            wide = wide_a.combine_first(wide_b) if not wide_b.empty else wide_a
            if not wide.empty:
                wide = wide.reindex(sorted(wide.index, key=_natural_sort_key))
                wide.to_csv(os.path.join(outdir, f"per_state_{metric}_vs_{ref_label}.tsv"),
                            sep="\t", float_format="%.4f")
                vmax = max(float(np.nanmax(np.abs(wide.values))), 0.1) if not wide.isna().all().all() else 0.1
                fig, ax = plt.subplots(figsize=(max(6, wide.shape[1] * 0.8),
                                                max(4, wide.shape[0] * 0.35)))
                is_kappa = metric == KAPPA
                sns.heatmap(wide, cmap="RdYlGn" if is_kappa else "YlGnBu",
                            vmin=-vmax if is_kappa else 0, vmax=vmax,
                            center=0 if is_kappa else None,
                            linewidths=0.5, annot=True, fmt=".2f",
                            annot_kws={"fontsize": 7},
                            cbar_kws={"label": f"Per-state {label}"},
                            ax=ax, mask=wide.isna().values)
                ax.set_title(f"Per-state {label} vs {ref_label}", fontsize=9)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
                save_fig(fig, os.path.join(outdir, f"per_state_{metric}_vs_{ref_label}.png"),
                         bbox_inches=None)

    pd.DataFrame(comparison_rows).to_csv(
        os.path.join(outdir, "comparison_all_pairs.tsv"),
        sep="\t", index=False, float_format="%.4f")

    def _save_matrix(mat, name):
        df = pd.DataFrame(mat, index=labels, columns=labels)
        df.to_csv(os.path.join(outdir, f"{name}_matrix.tsv"),
                  sep="\t", float_format="%.4f")
        print(f"  saved {outdir}/{name}_matrix.tsv")
        return df

    kappa_df   = _save_matrix(kappa_mat, KAPPA)
    _save_matrix(comp_sim_mat, "composition_similarity")
    jaccard_df = _save_matrix(jaccard_mat, f"{JACCARD}_similarity")
    _save_matrix(overlap_mat, "overlap")
    em_df = _save_matrix(em_sim_mat, "emission_similarity")
    _save_matrix(bw_sim_mat, "bw_emission_similarity")
    if not skip_noqh:
        _save_matrix(comp_sim_noqh_mat, f"composition{NOQH_SUFFIX}_similarity")
        _save_matrix(kappa_noqh_mat,   f"{KAPPA}{NOQH_SUFFIX}")
        _save_matrix(overlap_noqh_mat, f"overlap{NOQH_SUFFIX}")
        _save_matrix(jaccard_noqh_mat, f"{JACCARD}{NOQH_SUFFIX}")

    # Per-seg comparison rows, written into each analysis dir.
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


def compute_segment_stats(segs, exclude_states=None):
    """Segment length statistics; *exclude_states* drops the Quies/Het
    background first (NOQH mode).
    """
    if exclude_states:
        segs = [row for row in segs if row[3] not in exclude_states]
    if not segs:
        return {}
    lengths = np.array([row[2] - row[1] for row in segs])
    return {
        "n_states":      len({row[3] for row in segs}),
        "n_segments":    len(lengths),
        "min_length":    int(np.min(lengths)),
        "max_length":    int(np.max(lengths)),
        "mean_length":   float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
    }


def _plot_segment_stats(df, outdir, suffix=""):
    """Bar charts for each segment stats metric; *suffix* (e.g. "_noqh") is
    appended to filenames and titles so both variants coexist.
    """
    title_extra = " (excl. Quies/Het)" if suffix else ""
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
        fig, ax = plt.subplots(figsize=(max(5, len(df) * 0.8), 3.5))
        vals = df[col].values
        ax.bar(x, vals, color="#4878CF", edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=55, ha="right", fontsize=7)
        ax.set_title(title + title_extra, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            fmt = f"{v:.0f}" if v == int(v) else f"{v:.1f}"
            ax.text(i, v + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                    fmt, ha="center", va="bottom", fontsize=6)
        save_fig(fig, os.path.join(outdir, f"{col}{suffix}.png"), bbox_inches=None)


def run_segment_stats(seg_paths, outdir, analysis_dir=None, skip_noqh=False):
    """Segment length statistics per segmentation, in a full and a NOQH variant
    (segment_stats.tsv / segment_stats_noqh.tsv), each with its own plots.
    """
    os.makedirs(outdir, exist_ok=True)
    seg_outdirs = _build_seg_to_analysis_map(seg_paths, analysis_dir)
    col_order = ["segmentation", "n_states", "n_segments",
                 "min_length", "max_length", "mean_length", "median_length"]

    results = {"": [], NOQH_SUFFIX: []}   # suffix -> per-seg stats dicts

    for seg_path in seg_paths:
        segs = load_bed(seg_path)
        if not segs:
            print(f"  WARNING: empty segmentation {seg_path}", file=sys.stderr)
            continue
        label = _seg_label(seg_path)
        seg_dir = seg_outdirs.get(seg_path)

        variants = [("", None)]
        if not skip_noqh:
            variants.append((NOQH_SUFFIX, _EXCLUDE_STATES))

        for suffix, excl in variants:
            stats = compute_segment_stats(segs, exclude_states=excl)
            if not stats:
                print(f"  WARNING: no segments left for {label}{suffix}", file=sys.stderr)
                continue
            stats["segmentation"] = label
            results[suffix].append(stats)
            if not suffix:
                print(f"  {label}: {stats['n_states']} states, {stats['n_segments']} segments, "
                      f"lengths [{stats['min_length']}, {stats['max_length']}], "
                      f"mean={stats['mean_length']:.0f}, median={stats['median_length']:.0f}")

            if seg_dir:
                stats_dir = os.path.join(seg_dir, "segment_stats")
                os.makedirs(stats_dir, exist_ok=True)
                pd.DataFrame([stats]).to_csv(
                    os.path.join(stats_dir, f"segment_stats{suffix}.tsv"),
                    sep="\t", index=False, float_format="%.1f")

    for suffix in ("", NOQH_SUFFIX):
        if not results[suffix]:
            continue
        df = pd.DataFrame(results[suffix])[col_order]
        summary_path = os.path.join(outdir, f"segment_stats{suffix}.tsv")
        df.to_csv(summary_path, sep="\t", index=False, float_format="%.1f")
        print(f"  saved {summary_path}")
        _plot_segment_stats(df, outdir, suffix=suffix)


def run_compare(seg, bins, outdir, analysis_dir=None, threads=None,
                labels=None, all_pairs=False, skip_noqh=False):
    """Cross-segmentation comparison: entropy, kappa, Jaccard, segment stats.

    *bins* may be a single int (broadcast to all segs) or a list, one per seg.
    Called from analysis.ipynb.
    """
    if isinstance(bins, int):
        bins = [bins]
    args = SimpleNamespace(seg=seg, bins=bins, outdir=outdir,
                           analysis_dir=analysis_dir, threads=threads,
                           labels=labels, all_pairs=all_pairs)

    bin_sizes = (args.bins * len(args.seg) if len(args.bins) == 1
                 else args.bins)
    if len(bin_sizes) != len(args.seg):
        raise ValueError(f"bins must have 1 value or one per seg file "
                         f"({len(args.seg)} files, {len(args.bins)} bin sizes given)")
    if args.labels and len(args.labels) != len(args.seg):
        raise ValueError(f"labels must have one value per seg file "
                         f"({len(args.seg)} files, {len(args.labels)} labels given)")

    label_override = dict(zip(args.seg, args.labels)) if args.labels else None
    analysis_dir   = args.analysis_dir or args.outdir
    comparison_dir = args.outdir

    results_full = _compute_entropy(args.seg, bin_sizes)
    _save_entropy_summary(results_full, comparison_dir)

    if not skip_noqh:
        # The reference, if any, is what NOQH entropy maps states onto.
        ref_idx = next((i for i, p in enumerate(args.seg)
                        if _seg_label(p).startswith("ENCFF")), None)
        mappings = None
        if ref_idx is not None:
            import match
            ref_segs = _load_seg_full(args.seg[ref_idx])
            ref_states = sorted({x[3] for x in ref_segs}, key=_natural_sort_key)
            mappings = [None] * len(args.seg)
            for i, p in enumerate(args.seg):
                if i == ref_idx:
                    continue
                work_segs = _load_seg_full(p)
                work_states = sorted({x[3] for x in work_segs}, key=_natural_sort_key)
                overlap = match.pair_overlap(ref_segs, work_segs)
                mappings[i] = match.best_mapping(overlap, work_states, ref_states)

        results_active = _compute_entropy(args.seg, bin_sizes, exclude_states=_EXCLUDE_STATES,
                                          mappings=mappings)
        _save_entropy_summary(results_active, comparison_dir, suffix=NOQH_SUFFIX,
                              title_extra=f"\n(excluding {', '.join(sorted(_EXCLUDE_STATES))})")
        _save_entropy_combined_plot(results_full, results_active, comparison_dir)

    run_segment_stats(args.seg, comparison_dir, analysis_dir, skip_noqh=skip_noqh)
    compare_all(args.seg, bin_sizes, comparison_dir, analysis_dir, args.threads,
                label_override=label_override, all_pairs=args.all_pairs, skip_noqh=skip_noqh)
