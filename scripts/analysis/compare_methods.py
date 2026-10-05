#!/usr/bin/env python3
"""Aggregate per-method metrics into a unified comparison table and figures.

Reads the per-method report/enrichment tables under {analysis_dir} and the
metric matrices under {comparison_dir}; writes comparison_table.tsv and
per-metric PNG plots to {outdir}.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt

import utils
from utils import (METHOD_IDX, BIN_COLORS, METHOD_INFO, display_name,
                   bin_color, load_matrix, save_fig,
                   COMPOSITION, JACCARD, KAPPA, NOQH_SUFFIX,
                   COMPOSITION_DISPLAY, JACCARD_DISPLAY, KAPPA_DISPLAY)


def _build_analysis_to_seg_map(analysis_dirs, seg_names):
    """Map analysis subdir names → segmentation labels in metrics files.

    The labels match the dir names, except "ref" → the ENCFF... accession.
    """
    seg_set = set(seg_names)
    ref_accession = next((s for s in seg_names if s.startswith("ENCFF")), None)
    mapping = {}
    for adir in analysis_dirs:
        if adir == "ref":
            if ref_accession:
                mapping[adir] = ref_accession
        elif adir in seg_set:
            mapping[adir] = adir
    return mapping


def load_entropy(comparison_dir):
    """{seg_name: {entropy, entropy_noqh}}."""
    result = {}
    for suffix, col in [("", "entropy"), (NOQH_SUFFIX, f"entropy{NOQH_SUFFIX}")]:
        path = os.path.join(comparison_dir, f"entropy_summary{suffix}.tsv")
        if not os.path.exists(path):
            continue
        for _, row in pd.read_csv(path, sep="\t").iterrows():
            result.setdefault(row["segmentation"], {})[col] = row["total_entropy"]
    return result


def load_segment_stats(comparison_dir):
    """{seg_name: {n_states, n_segments, ..., max_length_noqh}}."""
    path = os.path.join(comparison_dir, "segment_stats.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    stats = {
        row["segmentation"]: {
            "n_states":         int(row["n_states"]),
            "n_segments":       int(row["n_segments"]),
            "min_length":       int(row["min_length"]),
            "max_length":       int(row["max_length"]),
            "mean_length":      float(row["mean_length"]),
            "median_length_all": float(row["median_length"]),
        }
        for _, row in df.iterrows()
    }

    noqh_path = os.path.join(comparison_dir, f"segment_stats{NOQH_SUFFIX}.tsv")
    if os.path.exists(noqh_path):
        for _, row in pd.read_csv(noqh_path, sep="\t").iterrows():
            if row["segmentation"] in stats:
                stats[row["segmentation"]][f"max_length{NOQH_SUFFIX}"] = int(row["max_length"])

    return stats


def load_report(analysis_dir, method):
    """{state: {n_segments, total_bp, median_length, ...}}."""
    path = os.path.join(analysis_dir, method, "report.tsv")
    if not os.path.exists(path):
        return {}
    return pd.read_csv(path, sep="\t").set_index("state").to_dict("index")


def load_enrichment(analysis_dir, method):
    """{state: {annotation: fold_enrichment}}."""
    path = os.path.join(analysis_dir, method, "enrichment", "enrichment.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    return df.pivot(index="state", columns="label", values="fold_enrichment").to_dict("index")


def load_coverage(analysis_dir, method):
    """{state: {annotation: coverage}}."""
    path = os.path.join(analysis_dir, method, "enrichment", "coverage.tsv")
    if not os.path.exists(path):
        path = os.path.join(analysis_dir, method, "enrichment", "enrichment.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    if "coverage" not in df.columns:
        return {}
    return df.pivot(index="state", columns="label", values="coverage").to_dict("index")


def load_jaccard(analysis_dir, method):
    """{state: {annotation: jaccard}}."""
    path = os.path.join(analysis_dir, method, "enrichment", "jaccard.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    return df.pivot(index="state", columns="label", values="jaccard").to_dict("index")


def load_sensitivity(analysis_dir, method):
    """{state: {annotation: sensitivity}}."""
    path = os.path.join(analysis_dir, method, "enrichment", "sensitivity.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    return df.pivot(index="state", columns="label", values="sensitivity").to_dict("index")


def build_table(analysis_dir, comparison_dir, ref_dir=None):
    """Unified comparison DataFrame, one row per pooled method.

    analysis_dir : variant-specific subdir (e.g. ds/analysis/comb/)
    ref_dir      : analysis dir containing ref/, defaults to analysis_dir
    """
    ref_dir = ref_dir or analysis_dir

    def _valid(d):
        full = os.path.join(analysis_dir, d)
        return (
            os.path.isdir(full)
            and d in METHOD_INFO
            and not d.endswith("_dense")
            and (os.path.exists(os.path.join(full, "report.tsv"))
                 or os.path.isdir(os.path.join(full, "bin_emissions")))
        )

    methods = sorted(
        [d for d in os.listdir(analysis_dir) if _valid(d)],
        key=lambda m: (METHOD_IDX.get(m, 999), m),
    )

    ref_full = os.path.join(ref_dir, "ref")
    if os.path.isdir(ref_full) and "ref" not in methods:
        methods = ["ref"] + methods

    def _adir(method):
        return ref_dir if method == "ref" else analysis_dir

    entropy_data = load_entropy(comparison_dir)
    kappa_mat        = load_matrix(os.path.join(comparison_dir, "kappa_matrix.tsv"))
    jaccard_mat      = load_matrix(os.path.join(comparison_dir, "jaccard_similarity_matrix.tsv"))
    comp_mat         = load_matrix(os.path.join(comparison_dir, "composition_similarity_matrix.tsv"))
    overlap_mat      = load_matrix(os.path.join(comparison_dir, "overlap_matrix.tsv"))
    emission_mat     = load_matrix(os.path.join(comparison_dir, "emission_similarity_matrix.tsv"))
    bw_emission_mat  = load_matrix(os.path.join(comparison_dir, "bw_emission_similarity_matrix.tsv"))
    kappa_noqh_mat   = load_matrix(os.path.join(comparison_dir, "kappa_noqh_matrix.tsv"))
    jaccard_noqh_mat = load_matrix(os.path.join(comparison_dir, "jaccard_noqh_matrix.tsv"))
    comp_noqh_mat    = load_matrix(os.path.join(comparison_dir, "composition_noqh_similarity_matrix.tsv"))
    overlap_noqh_mat = load_matrix(os.path.join(comparison_dir, "overlap_noqh_matrix.tsv"))
    seg_stats    = load_segment_stats(comparison_dir)

    # Union of all labels seen across every metric source.
    seg_names = set(entropy_data) | set(seg_stats)
    for mat in (kappa_mat, jaccard_mat, comp_mat, overlap_mat,
                emission_mat, bw_emission_mat,
                kappa_noqh_mat, jaccard_noqh_mat, comp_noqh_mat, overlap_noqh_mat):
        if mat is not None:
            seg_names |= set(mat.index)
    a2s = _build_analysis_to_seg_map(methods, seg_names)
    ref_seg = a2s.get("ref", "")

    rows = []
    for method in methods:
        binarization, state_model, rep = METHOD_INFO[method]
        seg_name = a2s.get(method, "")

        row = {
            "method":        method,
            "display_name":  display_name(method),
            "binarization":  binarization,
            "state_model":   state_model,
            "replicate":     rep or "",
        }

        if seg_name in entropy_data:
            row["entropy"] = entropy_data[seg_name].get("entropy", np.nan)
            row[f"entropy{NOQH_SUFFIX}"] = entropy_data[seg_name].get(
                f"entropy{NOQH_SUFFIX}", np.nan)

        if seg_name in seg_stats:
            row.update(seg_stats[seg_name])

        if seg_name and ref_seg:
            for mat, col in [
                (kappa_mat,        f"{KAPPA}_vs_ref"),
                (kappa_noqh_mat,   f"{KAPPA}{NOQH_SUFFIX}_vs_ref"),
                (jaccard_mat,      f"{JACCARD}_vs_ref"),
                (jaccard_noqh_mat, f"{JACCARD}{NOQH_SUFFIX}_vs_ref"),
                (comp_mat,         f"{COMPOSITION}_vs_ref"),
                (comp_noqh_mat,    f"{COMPOSITION}{NOQH_SUFFIX}_vs_ref"),
            ]:
                if mat is not None and seg_name in mat.index and ref_seg in mat.columns:
                    val = mat.loc[seg_name, ref_seg]
                    if not np.isnan(val):
                        row[col] = val

        # Replicate consistency: only for pooled (non-replicate) methods
        if rep is None and method != "ref" and seg_name:
            rep1_seg = f"{seg_name}_rep1"
            rep2_seg = f"{seg_name}_rep2"
            base_rep_cols = [
                (kappa_mat,        f"{KAPPA}_rep1_vs_rep2"),
                (jaccard_mat,      f"{JACCARD}_rep1_vs_rep2"),
                (comp_mat,         f"{COMPOSITION}_rep1_vs_rep2"),
                (overlap_mat,      "overlap_rep1_vs_rep2"),
                (kappa_noqh_mat,   f"{KAPPA}{NOQH_SUFFIX}_rep1_vs_rep2"),
                (jaccard_noqh_mat, f"{JACCARD}{NOQH_SUFFIX}_rep1_vs_rep2"),
                (comp_noqh_mat,    f"{COMPOSITION}{NOQH_SUFFIX}_rep1_vs_rep2"),
                (overlap_noqh_mat, f"overlap{NOQH_SUFFIX}_rep1_vs_rep2"),
            ]
            for mat, col_name in base_rep_cols:
                if mat is not None and rep1_seg in mat.index and rep2_seg in mat.columns:
                    row[col_name] = mat.loc[rep1_seg, rep2_seg]

        report = load_report(_adir(method), method)
        total_bp = sum(s["total_bp"] for s in report.values())
        for state, col_base in [("Tx", "Tx_length"),
                                ("Tss", "Tss_length"),
                                ("TxWk", "TxWk_length")]:
            if state in report:
                row[f"median_{col_base}"] = report[state].get("median_length", np.nan)
                row[f"mean_{col_base}"] = report[state].get("mean_length", np.nan)

        enrichment = load_enrichment(_adir(method), method)
        def _get_val(data, st, ann):
            """Look up data[state][annotation], tolerating name variants such as
            "Tss" -> "1_TssA" and "RefSeqTSS.hg38" -> "RefSeqTSS".
            """
            target_state = st
            if st not in data:
                matches = [s for s in data if st.lower() in s.lower()]
                if matches:
                    target_state = next((s for s in matches if s.lower() == st.lower()), matches[0])
                else:
                    return np.nan

            if ann in data[target_state]:
                return data[target_state][ann]
            base = ann.split(".")[0]
            for k in data[target_state]:
                if k.split(".")[0] == base:
                    return data[target_state][k]
            return np.nan

        row["enrich_Tx_RefSeqGene"] = _get_val(enrichment, "Tx", "RefSeqGene.hg38")
        row["enrich_Tx_ExpressedGeneBodies"] = _get_val(enrichment, "Tx", "ExpressedGeneBodies")
        for suffix in ("2kb",):
            row[f"enrich_Tss_RefSeqTSS{suffix}"] = _get_val(enrichment, "Tss", f"RefSeqTSS{suffix}.hg38")
        row["enrich_Enh1_ExpressedTSS"] = _get_val(enrichment, "Enh1", "ExpressedTSS")

        # ATAC-seq validation (pooled active states).
        coverage = load_coverage(_adir(method), method)
        sensitivity = load_sensitivity(_adir(method), method)

        atac_label = None
        for st_data in enrichment.values():
            atac_label = next((l for l in st_data if l.startswith("atac_")), None)
            if atac_label:
                break
            
        if atac_label:
            # ann_frac (ann_bp / total_bp) is recoverable from any state with fold > 0.
            ann_frac = np.nan
            for st in enrichment:
                if atac_label in enrichment[st] and atac_label in coverage.get(st, {}):
                    f = enrichment[st][atac_label]
                    c = coverage[st][atac_label]
                    if f > 0:
                        ann_frac = c / f
                        break
            
            for pool_name, check_fn in [("Tss", utils.is_promoter_core), 
                                       ("Enh", utils.is_distal_enhancer), 
                                       ("Active", utils.is_active_open),
                                       ("Quies", utils.is_noqh)]:
                pool_key = f"POOL:{pool_name}"
                if pool_key in sensitivity and atac_label in sensitivity[pool_key]:
                    row[f"sensitivity_{pool_name}_ATAC"] = sensitivity[pool_key][atac_label]
                if pool_key in coverage and atac_label in coverage[pool_key]:
                    row[f"coverage_{pool_name}_ATAC"] = coverage[pool_key][atac_label]
                
                pooled_states = [st for st in enrichment if check_fn(st) and not st.startswith("POOL:")]
                
                if pooled_states:
                    overlap_sum = 0
                    state_bp_sum = 0
                    for st in pooled_states:
                        st_bp = report.get(st, {}).get("total_bp", 0)
                        st_cov = coverage.get(st, {}).get(atac_label, 0)
                        overlap_sum += st_cov * st_bp
                        state_bp_sum += st_bp
                    
                    if state_bp_sum > 0:
                        pooled_cov = overlap_sum / state_bp_sum
                        if f"coverage_{pool_name}_ATAC" not in row:
                            row[f"coverage_{pool_name}_ATAC"] = pooled_cov
                        
                        if not np.isnan(ann_frac) and ann_frac > 0:
                            ann_bp = ann_frac * total_bp
                            row[f"enrich_{pool_name}_ATAC"] = pooled_cov / ann_frac
                            # Fraction of ATAC peaks covered by these states.
                            if f"sensitivity_{pool_name}_ATAC" not in row:
                                row[f"sensitivity_{pool_name}_ATAC"] = overlap_sum / ann_bp

        # Biological validation: fraction of each annotation covered by states.
        val_targets = [
            ("Tx", utils.is_tx, "ExpressedGeneBodies", "Tx_ExpressedGeneBodies"),
            ("Active", utils.is_active_open, "NonExpressedGeneBodies", "Active_NonExpGeneBodies"),
            ("Quies", utils.is_noqh, "NonExpressedGeneBodies", "Quies_NonExpGeneBodies"),
            ("Active", utils.is_active_open, "ExpressedTSS", "Active_ExpressedTSS"),
        ]
        for suffix in ("2kb",):
            val_targets.append(("Tss", utils.is_promoter_core, f"RefSeqTSS{suffix}.hg38", f"Tss_RefSeqTSS{suffix}"))
            val_targets.append(("Tss", utils.is_promoter_core, f"ExpressedTSS{suffix}", f"Tss_ExpressedTSS{suffix}"))

        for pool_name, check_fn, ann_name, col_prefix in val_targets:
            target_ann = None
            for st in enrichment:
                if ann_name in enrichment[st]:
                    target_ann = ann_name
                    break
            if not target_ann:
                base = ann_name.split(".")[0]
                for st in enrichment:
                    for k in enrichment[st]:
                        if k.split(".")[0] == base:
                            target_ann = k
                            break
                    if target_ann: break
            
            if target_ann and total_bp > 0:
                pool_key = f"POOL:{pool_name}"
                val_sens = _get_val(sensitivity, pool_key, target_ann)
                val_cov = _get_val(coverage, pool_key, target_ann)
                
                if not np.isnan(val_sens):
                    row[f"sensitivity_{col_prefix}"] = val_sens
                if not np.isnan(val_cov):
                    row[f"coverage_{col_prefix}"] = val_cov

                ann_frac = np.nan
                for st in enrichment:
                    if target_ann in enrichment[st] and target_ann in coverage.get(st, {}):
                        f = enrichment[st][target_ann]
                        c = coverage[st][target_ann]
                        if f > 0:
                            ann_frac = c / f
                            break
                
                if not np.isnan(ann_frac) and ann_frac > 0:
                    pooled_states = [st for st in enrichment if check_fn(st) and not st.startswith("POOL:")]
                    if pooled_states:
                        overlap_sum = 0
                        state_bp_sum = 0
                        for st in pooled_states:
                            st_bp = report.get(st, {}).get("total_bp", 0)
                            st_cov = coverage.get(st, {}).get(target_ann, 0)
                            overlap_sum += st_cov * st_bp
                            state_bp_sum += st_bp
                        
                        ann_bp = ann_frac * total_bp
                        if f"sensitivity_{col_prefix}" not in row:
                            row[f"sensitivity_{col_prefix}"] = overlap_sum / ann_bp
                        if f"coverage_{col_prefix}" not in row:
                            row[f"coverage_{col_prefix}"] = overlap_sum / state_bp_sum if state_bp_sum > 0 else np.nan
                        row[f"enrich_{col_prefix}"] = (overlap_sum / state_bp_sum) / ann_frac if state_bp_sum > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _order_methods(df):
    """Sort by METHOD_ORDER; exclude _dense entries."""
    df = df[~df["method"].str.endswith("_dense")].copy()
    df["_sort"] = df["method"].map(lambda m: (METHOD_IDX.get(m, 999), m))
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def _filter_valid(df, cols):
    """Keep rows where at least one of *cols* is non-NaN."""
    cols = [cols] if isinstance(cols, str) else cols
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return df
    return df.dropna(subset=existing, how="all").reset_index(drop=True)


def _bar_panel(ax, df, col, title, ylabel=None):
    df = _filter_valid(df, col)
    vals = df[col].values if col in df.columns else np.full(len(df), np.nan)
    x = np.arange(len(df))
    ax.bar(x, vals, color=[bin_color(b) for b in df["binarization"]], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["display_name"], rotation=55, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6)


def plot_comparison(df, outdir):
    """Per-metric bar chart PNGs."""
    from matplotlib.patches import Patch

    df_main = df[df["replicate"] == ""].reset_index(drop=True)
    w = max(5, len(df_main) * 0.6)

    legend_elements = [
        Patch(facecolor=BIN_COLORS["default"],   label="Default binarization"),
        Patch(facecolor=BIN_COLORS["omnipeak"],  label="OmniPeak binarization"),
        Patch(facecolor=BIN_COLORS["homer"],     label="Homer binarization"),
        Patch(facecolor=BIN_COLORS["macs2"],     label="MACS2 binarization"),
        Patch(facecolor=BIN_COLORS["reference"], label="ENCODE reference"),
    ]

    def _make_fig():
        fig, ax = plt.subplots(figsize=(w, 3.5))
        ax.legend(handles=legend_elements, fontsize=6,
                  bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        return fig, ax

    # (column, title, ylabel) per metric, first over all states then without the
    # Quies/Het background.
    _REP_METRICS = [(KAPPA, KAPPA_DISPLAY, KAPPA_DISPLAY),
                    (JACCARD, JACCARD_DISPLAY, "Similarity"),
                    (COMPOSITION, COMPOSITION_DISPLAY, "Cosine composition similarity"),
                    ("overlap", "Overlap", "Overlap fraction")]
    _REP_COLS = [
        (f"{metric}{suffix}_rep1_vs_rep2",
         f"Replicate reproducibility ({metric_display}{excl})", ylabel)
        for suffix, excl in (("", ""), (NOQH_SUFFIX, " excl. Quies/Het"))
        for metric, metric_display, ylabel in _REP_METRICS
    ]
    for col, title, ylabel in _REP_COLS:
        if col in df_main.columns and df_main[col].notna().any():
            fig, ax = _make_fig()
            _bar_panel(ax, df_main, col, title, ylabel)
            save_fig(fig, os.path.join(outdir, f"{col}.png"))

    plot_targets = [
        ("enrich_Tx_ExpressedGeneBodies",  "Tx enrichment vs expressed gene bodies"),
        ("sensitivity_Tx_ExpressedGeneBodies", "Tx bp overlap at expressed gene bodies"),
        ("coverage_Tx_ExpressedGeneBodies",    "Tx bp overlap by expressed genes"),
        ("enrich_Active_ATAC",             "Active chromatin enrichment at ATAC-seq peaks"),
        ("sensitivity_Active_ATAC",        "Active chromatin bp overlap at ATAC-seq peaks"),
        ("coverage_Active_ATAC",           "Active chromatin bp overlap by ATAC-seq peaks"),
        # Expressed TSS validation (±2kb window by default).
        ("enrich_Tss_ExpressedTSS2kb",     "Tss enrichment at Expressed TSS", "enrich_Tss_ExpressedTSS"),
        ("sensitivity_Tss_ExpressedTSS2kb", "Tss bp overlap at Expressed TSS", "sensitivity_Tss_ExpressedTSS"),
        ("coverage_Tss_ExpressedTSS2kb",    "Tss bp overlap by Expressed TSS", "coverage_Tss_ExpressedTSS"),
        ("enrich_Active_NonExpGeneBodies",  "Active states enrichment at non-expressed genes"),
        ("enrich_Quies_NonExpGeneBodies",   "Quiescent states enrichment at non-expressed genes"),
        ("median_Tx_length",               "Median Tx (transcription) segment length"),
        ("mean_Tx_length",                 "Mean Tx (transcription) segment length"),
    ]
    # RefSeq TSS validation (±2kb window by default).
    plot_targets.extend([
        ("enrich_Tss_RefSeqTSS2kb",      "Tss enrichment at RefSeq TSS", "enrich_Tss_RefSeqTSS"),
        ("sensitivity_Tss_RefSeqTSS2kb", "Tss bp overlap at RefSeq TSS", "sensitivity_Tss_RefSeqTSS"),
        ("coverage_Tss_RefSeqTSS2kb",    "Tss bp overlap by RefSeq TSS", "coverage_Tss_RefSeqTSS"),
    ])

    for col_info in plot_targets:
        if len(col_info) == 3:
            col, title, filename = col_info
        else:
            col, title = col_info
            filename = col
        ylabel = "Fold enrichment" if col.startswith("enrich") else \
                 "Fraction" if col.startswith("sensitivity") else \
                 JACCARD_DISPLAY if col.startswith(JACCARD) else \
                 "Fraction" if col.startswith("coverage") else "bp"
        if col in df_main.columns and df_main[col].notna().any():
            fig, ax = _make_fig()
            _bar_panel(ax, df_main, col, title, ylabel)
            save_fig(fig, os.path.join(outdir, f"{filename}.png"))


def run_compare_methods(analysis_dir, comparison_dir, outdir, ref_dir=None):
    """Aggregate metrics into a unified method comparison table and plots."""
    os.makedirs(outdir, exist_ok=True)

    df = build_table(analysis_dir, comparison_dir, ref_dir=ref_dir)
    table_path = os.path.join(outdir, "comparison_table.tsv")
    df.to_csv(table_path, sep="\t", index=False, float_format="%.4f")
    print(f"  saved {table_path}")
    print(df.to_string(index=False))

    # Backfill columns that may be absent in old TSVs.
    if "display_name" not in df.columns:
        df["display_name"] = df["method"].map(display_name)
    if "replicate" not in df.columns:
        df["replicate"] = df["method"].map(lambda m: METHOD_INFO.get(m, (None, None, None))[2] or "")

    df = _order_methods(df)
    plot_comparison(df, outdir)
