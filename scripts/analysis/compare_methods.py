#!/usr/bin/env python3
"""Aggregate per-method metrics into a unified comparison table and figure.

Method keys follow the structured naming convention from analyze.smk / compare.py:
  {state_model}_{binarization}[_{rep}]
  state_model   : chromhmm | kmeans
  binarization  : default | omni | homer
  rep           : rep1 | rep2  (optional)
  special       : ref  (ENCODE reference)

Reads from:
  - {analysis_dir}/{method}/report.tsv
  - {analysis_dir}/{method}/enrichment/enrichment.tsv
  - {analysis_dir}/{method}/enrichment/jaccard.tsv
  - {comparison_dir}/entropy_summary.tsv
  - {comparison_dir}/kappa_matrix.tsv
  - {comparison_dir}/jaccard_similarity_matrix.tsv
  - {comparison_dir}/segment_stats.tsv
  - {comparison_dir}/emission_similarity_matrix.tsv
  - {comparison_dir}/bw_emission_similarity_matrix.tsv

Produces in {outdir}/:
  - comparison_table.tsv
  - per-metric PNG plots
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt

from utils import (METHOD_ORDER, METHOD_IDX, DISPLAY_NAMES, BIN_COLORS,
                   METHOD_INFO, display_name, seg_label)



# ---------------------------------------------------------------------------
# Analysis-dir → segmentation-label mapping
# ---------------------------------------------------------------------------

def _build_analysis_to_seg_map(analysis_dirs, seg_names):
    """Map analysis subdir names → segmentation labels in metrics files.

    Labels in compare.py now match analysis dir names directly.
    The only exception is "ref" which maps to the ENCFF... accession.
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_matrix(directory, filename):
    """Load a seg × seg matrix TSV as a DataFrame, or None."""
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t", index_col=0)


def load_entropy(comparison_dir):
    """Returns dict: seg_name → {entropy, entropy_noqh}."""
    result = {}
    for suffix, col in [("", "entropy"), ("_noqh", "entropy_noqh")]:
        path = os.path.join(comparison_dir, f"entropy_summary{suffix}.tsv")
        if not os.path.exists(path):
            continue
        for _, row in pd.read_csv(path, sep="\t").iterrows():
            result.setdefault(row["segmentation"], {})[col] = row["total_entropy"]
    return result


def load_segment_stats(comparison_dir):
    """Returns dict: seg_name → {n_states, n_segments, ..., max_length_noqh}."""
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

    # NOQH variant (excl. Quies/Het): surface max_length_noqh alongside max_length.
    noqh_path = os.path.join(comparison_dir, "segment_stats_noqh.tsv")
    if os.path.exists(noqh_path):
        for _, row in pd.read_csv(noqh_path, sep="\t").iterrows():
            if row["segmentation"] in stats:
                stats[row["segmentation"]]["max_length_noqh"] = int(row["max_length"])

    return stats


def load_report(analysis_dir, method):
    """Returns dict: state → {n_segments, total_bp, median_length, ...}."""
    path = os.path.join(analysis_dir, method, "report.tsv")
    if not os.path.exists(path):
        return {}
    return pd.read_csv(path, sep="\t").set_index("state").to_dict("index")


def load_enrichment(analysis_dir, method):
    """Returns dict: state → {annotation → fold_enrichment}."""
    path = os.path.join(analysis_dir, method, "enrichment", "enrichment.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    return df.pivot(index="state", columns="label", values="fold_enrichment").to_dict("index")


def load_coverage(analysis_dir, method):
    """Returns dict: state → {annotation → coverage}."""
    path = os.path.join(analysis_dir, method, "enrichment", "coverage.tsv")
    if not os.path.exists(path):
        # Fallback to enrichment.tsv if coverage.tsv doesn't exist
        path = os.path.join(analysis_dir, method, "enrichment", "enrichment.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    if "coverage" not in df.columns:
        return {}
    return df.pivot(index="state", columns="label", values="coverage").to_dict("index")


def load_jaccard(analysis_dir, method):
    """Returns dict: state → {annotation → jaccard}."""
    path = os.path.join(analysis_dir, method, "enrichment", "jaccard.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    return df.pivot(index="state", columns="label", values="jaccard").to_dict("index")


# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------

def build_table(analysis_dir, comparison_dir, ref_dir=None):
    """Build the unified comparison DataFrame (one row per pooled method).

    analysis_dir : variant-specific subdir (e.g. ds/analysis/comb/)
    ref_dir      : top-level analysis dir containing ref/ (e.g. ds/analysis/).
                   Defaults to analysis_dir when not provided.
    """
    ref_dir = ref_dir or analysis_dir

    def _valid(d):
        full = os.path.join(analysis_dir, d)
        return (
            os.path.isdir(full)
            and d in METHOD_INFO          # must be a known, structured key
            and not d.endswith("_dense")
            and (os.path.exists(os.path.join(full, "report.tsv"))
                 or os.path.isdir(os.path.join(full, "bin_emissions")))
        )

    methods = sorted(
        [d for d in os.listdir(analysis_dir) if _valid(d)],
        key=lambda m: (METHOD_IDX.get(m, 999), m),
    )

    # Prepend ref if found in ref_dir but not already in analysis_dir.
    ref_full = os.path.join(ref_dir, "ref")
    if os.path.isdir(ref_full) and "ref" not in methods:
        methods = ["ref"] + methods

    def _adir(method):
        """Return the analysis base dir for a given method."""
        return ref_dir if method == "ref" else analysis_dir

    # Load cross-segmentation metrics
    entropy_data = load_entropy(comparison_dir)
    kappa_mat        = _load_matrix(comparison_dir, "kappa_matrix.tsv")
    jaccard_mat      = _load_matrix(comparison_dir, "jaccard_similarity_matrix.tsv")
    comp_mat         = _load_matrix(comparison_dir, "composition_similarity_matrix.tsv")
    overlap_mat      = _load_matrix(comparison_dir, "overlap_matrix.tsv")
    emission_mat     = _load_matrix(comparison_dir, "emission_similarity_matrix.tsv")
    bw_emission_mat  = _load_matrix(comparison_dir, "bw_emission_similarity_matrix.tsv")
    kappa_noqh_mat   = _load_matrix(comparison_dir, "kappa_noqh_matrix.tsv")
    jaccard_noqh_mat = _load_matrix(comparison_dir, "jaccard_noqh_matrix.tsv")
    comp_noqh_mat    = _load_matrix(comparison_dir, "composition_noqh_similarity_matrix.tsv")
    overlap_noqh_mat = _load_matrix(comparison_dir, "overlap_noqh_matrix.tsv")
    seg_stats    = load_segment_stats(comparison_dir)

    # seg_names = union of all labels seen across every metric source
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

        # Entropy
        if seg_name in entropy_data:
            row["entropy"]      = entropy_data[seg_name].get("entropy", np.nan)
            row["entropy_noqh"] = entropy_data[seg_name].get("entropy_noqh", np.nan)

        # Segment stats
        if seg_name in seg_stats:
            row.update(seg_stats[seg_name])

        # vs reference
        if seg_name and ref_seg:
            for mat, col in [
                (kappa_mat,        "kappa_vs_ref"),
                (kappa_noqh_mat,   "kappa_noqh_vs_ref"),
                (jaccard_mat,      "jaccard_vs_ref"),
                (jaccard_noqh_mat, "jaccard_noqh_vs_ref"),
                (comp_mat,         "composition_vs_ref"),
                (comp_noqh_mat,    "composition_noqh_vs_ref"),
            ]:
                if mat is not None and seg_name in mat.index and ref_seg in mat.columns:
                    val = mat.loc[seg_name, ref_seg]
                    if not np.isnan(val):
                        row[col] = val

        # Replicate consistency: only for pooled (non-replicate) methods
        if rep is None and method != "ref" and seg_name:
            rep1_seg = f"{seg_name}_rep1"
            rep2_seg = f"{seg_name}_rep2"
            # Base as-is replicate columns (always included).
            base_rep_cols = [
                (kappa_mat,      "kappa_rep1_vs_rep2"),
                (jaccard_mat,    "jaccard_rep1_vs_rep2"),
                (comp_mat,       "composition_rep1_vs_rep2"),
                (overlap_mat,    "overlap_rep1_vs_rep2"),
                (kappa_noqh_mat,   "kappa_noqh_rep1_vs_rep2"),
                (jaccard_noqh_mat, "jaccard_noqh_rep1_vs_rep2"),
                (comp_noqh_mat,    "composition_noqh_rep1_vs_rep2"),
                (overlap_noqh_mat, "overlap_noqh_rep1_vs_rep2"),
            ]
            for mat, col_name in base_rep_cols:
                if mat is not None and rep1_seg in mat.index and rep2_seg in mat.columns:
                    row[col_name] = mat.loc[rep1_seg, rep2_seg]

        # Report: mean lengths for key states
        report = load_report(_adir(method), method)
        total_bp = sum(s["total_bp"] for s in report.values())
        for state, col_base in [("Tx", "Tx_length"),
                                ("Tss", "Tss_length"),
                                ("TxWk", "TxWk_length")]:
            if state in report:
                row[f"median_{col_base}"] = report[state].get("median_length", np.nan)
                row[f"mean_{col_base}"] = report[state].get("mean_length", np.nan)

        # Enrichment
        enrichment = load_enrichment(_adir(method), method)
        def _get_val(data, st, ann):
            # 1. Flexible state lookup
            target_state = st
            if st not in data:
                # Try case-insensitive or substring match (e.g. "Tss" -> "1_TssA")
                matches = [s for s in data if st.lower() in s.lower()]
                if matches:
                    # Prefer exact match if possible, otherwise first match
                    target_state = next((s for s in matches if s.lower() == st.lower()), matches[0])
                else:
                    return np.nan
            
            # 2. Flexible annotation lookup
            if ann in data[target_state]:
                return data[target_state][ann]
            # Fallback to label without genome suffix
            base = ann.split(".")[0]
            for k in data[target_state]:
                if k.split(".")[0] == base:
                    return data[target_state][k]
            return np.nan

        for state, annotation, col in [
            ("Tx",   "RefSeqGene.hg38",        "enrich_Tx_RefSeqGene"),
            ("Tx",   "ExpressedGeneBodies",     "enrich_Tx_ExpressedGeneBodies"),
            ("Tss",  "RefSeqTSS.hg38",          "enrich_Tss_RefSeqTSS"),
            ("Tss",  "RefSeqTSS2kb.hg38",       "enrich_Tss_RefSeqTSS2kb"),
            ("Enh1", "ExpressedTSS",            "enrich_Enh1_ExpressedTSS"),
        ]:
            val = _get_val(enrichment, state, annotation)
            if not np.isnan(val):
                row[col] = val

        # Jaccard vs expressed annotations
        jaccard = load_jaccard(_adir(method), method)
        for state, annotation, col in [
            ("Tx",  "ExpressedGeneBodies", "jaccard_Tx_ExpressedGeneBodies"),
        ]:
            val = _get_val(jaccard, state, annotation)
            if not np.isnan(val):
                row[col] = val

        # ATAC-seq validation (pooled active states)
        coverage = load_coverage(_adir(method), method)
        
        # Find any atac_* label
        atac_label = None
        for st_data in enrichment.values():
            atac_label = next((l for l in st_data if l.startswith("atac_")), None)
            if atac_label:
                break
            
        if atac_label:
            # Find any state that has fold > 0 to calculate ann_frac (ann_bp / total_bp)
            ann_frac = np.nan
            for st in enrichment:
                if atac_label in enrichment[st] and atac_label in coverage.get(st, {}):
                    f = enrichment[st][atac_label]
                    c = coverage[st][atac_label]
                    if f > 0:
                        ann_frac = c / f
                        break
            
            for pool_name, substrs in [("Tss", ["Tss"]), 
                                       ("Enh", ["Enh"]), 
                                       ("Active", ["Tss", "Enh"]),
                                       ("Quies", ["Quies", "Het", "ZNF"])]:
                # Pool states: match any substr, but exclude "Biv"
                pooled_states = [st for st in enrichment 
                                if any(sub.lower() in st.lower() for sub in substrs)
                                and "biv" not in st.lower()]
                
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
                        row[f"coverage_{pool_name}_ATAC"] = pooled_cov
                        
                        if not np.isnan(ann_frac) and ann_frac > 0:
                            ann_bp = ann_frac * total_bp
                            row[f"enrich_{pool_name}_ATAC"] = pooled_cov / ann_frac
                            # Sensitivity: fraction of ATAC peaks covered by these states
                            row[f"sensitivity_{pool_name}_ATAC"] = overlap_sum / ann_bp
                            # Correct pooled Jaccard
                            row[f"jaccard_{pool_name}_ATAC"] = overlap_sum / (state_bp_sum + ann_bp - overlap_sum)

        # Additional biological validation sensitivities (fraction of annotation covered by states)
        for pool_name, substrs, ann_name, col_prefix in [
            ("Tss",    ["Tss"], "RefSeqTSS2kb.hg38",      "Tss_RefSeqTSS2kb"),
            ("Tx",     ["Tx"],  "ExpressedGeneBodies",    "Tx_ExpressedGeneBodies"),
            ("Tss",    ["Tss"], "ExpressedTSS",           "Tss_ExpressedTSS"),
            ("Tss",    ["Tss"], "ExpressedTSS2kb",         "Tss_ExpressedTSS2kb"),
            ("Active", ["Tss", "Enh"], "ExpressedTSS",    "Active_ExpressedTSS"),
            ("Active", ["Tss", "Enh"], "NonExpressedGeneBodies", "Active_NonExpGeneBodies"),
            ("Quies",  ["Quies", "Het", "ZNF"], "NonExpressedGeneBodies", "Quies_NonExpGeneBodies"),
        ]:
            # Find actual annotation label (flexible matching)
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
                ann_frac = np.nan
                for st in enrichment:
                    if target_ann in enrichment[st] and target_ann in coverage.get(st, {}):
                        f = enrichment[st][target_ann]
                        c = coverage[st][target_ann]
                        if f > 0:
                            ann_frac = c / f
                            break
                
                if not np.isnan(ann_frac) and ann_frac > 0:
                    pooled_states = [st for st in enrichment 
                                    if any(sub.lower() in st.lower() for sub in substrs)
                                    and "biv" not in st.lower()]
                    if pooled_states:
                        overlap_sum = 0
                        state_bp_sum = 0
                        for st in pooled_states:
                            st_bp = report.get(st, {}).get("total_bp", 0)
                            st_cov = coverage.get(st, {}).get(target_ann, 0)
                            overlap_sum += st_cov * st_bp
                            state_bp_sum += st_bp
                        
                        ann_bp = ann_frac * total_bp
                        row[f"sensitivity_{col_prefix}"] = overlap_sum / ann_bp
                        row[f"coverage_{col_prefix}"] = overlap_sum / state_bp_sum if state_bp_sum > 0 else np.nan
                        row[f"enrich_{col_prefix}"] = (overlap_sum / state_bp_sum) / ann_frac if state_bp_sum > 0 else np.nan
                        row[f"jaccard_{col_prefix}"] = overlap_sum / (state_bp_sum + ann_bp - overlap_sum) if (state_bp_sum + ann_bp - overlap_sum) > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _order_methods(df):
    """Sort by METHOD_ORDER; exclude _dense entries."""
    df = df[~df["method"].str.endswith("_dense")].copy()
    df["_sort"] = df["method"].map(lambda m: (METHOD_IDX.get(m, 999), m))
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def _method_colors(df):
    return [BIN_COLORS.get(b, "#888888") for b in df["binarization"]]


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
    ax.bar(x, vals, color=_method_colors(df), edgecolor="white", linewidth=0.5)
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


def _save_panel(fig, outdir, name):
    fig.tight_layout()
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_comparison(df, outdir):
    """Generate per-metric bar chart PNGs."""
    from matplotlib.patches import Patch

    # Pooled methods only (no replicates) for main plots
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

    # Replicate consistency plots
    _REP_COLS = [
        ("kappa_rep1_vs_rep2",          "Replicate reproducibility (Kappa)",                    "Kappa"),
        ("jaccard_rep1_vs_rep2",        "Replicate reproducibility (Jaccard)",                  "Similarity"),
        ("composition_rep1_vs_rep2",    "Replicate reproducibility (Composition)",              "Cosine similarity"),
        ("overlap_rep1_vs_rep2",        "Replicate reproducibility (Overlap)",                  "Overlap fraction"),
        ("kappa_noqh_rep1_vs_rep2",     "Replicate reproducibility (Kappa excl. Quies/Het)",    "Kappa"),
        ("jaccard_noqh_rep1_vs_rep2",   "Replicate reproducibility (Jaccard excl. Quies/Het)",  "Similarity"),
        ("composition_noqh_rep1_vs_rep2", "Replicate reproducibility (Composition excl. Quies/Het)", "Cosine similarity"),
        ("overlap_noqh_rep1_vs_rep2",   "Replicate reproducibility (Overlap excl. Quies/Het)",  "Overlap fraction"),
    ]
    for col, title, ylabel in _REP_COLS:
        if col in df_main.columns and df_main[col].notna().any():
            fig, ax = _make_fig()
            _bar_panel(ax, df_main, col, title, ylabel)
            _save_panel(fig, outdir, col)

    # Enrichment / Jaccard plots
    for col, title in [
        ("enrich_Tx_ExpressedGeneBodies",  "Tx enrichment vs expressed gene bodies"),
        ("jaccard_Tx_ExpressedGeneBodies", "Jaccard: Tx state vs expressed gene bodies"),
        ("sensitivity_Tx_ExpressedGeneBodies", "Fraction of expressed gene bodies covered by Tx states"),
        ("enrich_Active_ATAC",             "Active chromatin enrichment at ATAC-seq peaks"),
        ("sensitivity_Active_ATAC",        "Fraction of ATAC-seq peaks covered by Active states"),
        ("jaccard_Active_ATAC",            "Jaccard: Active states vs ATAC-seq"),
        ("coverage_Active_ATAC",           "Fraction of Active states covered by ATAC-seq peaks"),
        ("enrich_Tss_RefSeqTSS2kb",        "Tss enrichment at RefSeq TSS ±2 kb"),
        ("jaccard_Tss_RefSeqTSS2kb",       "Jaccard: Tss state vs RefSeq TSS ±2 kb"),
        ("sensitivity_Tss_RefSeqTSS2kb",   "Fraction of RefSeq TSS ±2 kb covered by Tss states"),
        ("coverage_Tss_RefSeqTSS2kb",      "Fraction of Tss states covered by RefSeq TSS ±2 kb"),
        ("enrich_Tss_ExpressedTSS",        "Tss enrichment at Expressed TSS"),
        ("jaccard_Tss_ExpressedTSS",       "Jaccard: Tss state vs Expressed TSS"),
        ("jaccard_Tss_ExpressedTSS2kb",    "Jaccard: Tss state vs Expressed TSS ±2 kb"),
        ("sensitivity_Tss_ExpressedTSS",   "Fraction of Expressed TSS covered by Tss states"),
        ("coverage_Tss_ExpressedTSS",      "Fraction of Tss states covered by Expressed TSS"),
        ("enrich_Active_NonExpGeneBodies",  "Active states enrichment at non-expressed genes"),
        ("enrich_Quies_NonExpGeneBodies",   "Quiescent states enrichment at non-expressed genes"),
        ("median_Tx_length",               "Median Tx (transcription) segment length"),
        ("mean_Tx_length",                 "Mean Tx (transcription) segment length"),
    ]:
        ylabel = "Fold enrichment" if col.startswith("enrich") else \
                 "Fraction" if col.startswith("sensitivity") else \
                 "Jaccard" if col.startswith("jaccard") else \
                 "Fraction" if col.startswith("coverage") else "bp"
        if col in df_main.columns and df_main[col].notna().any():
            fig, ax = _make_fig()
            _bar_panel(ax, df_main, col, title, ylabel)
            _save_panel(fig, outdir, col)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_compare_methods(analysis_dir, comparison_dir, outdir, ref_dir=None):
    """Aggregate metrics into a unified method comparison table & plots.

    Direct-call entry point (the former CLI); called from analysis.ipynb.
    """
    os.makedirs(outdir, exist_ok=True)

    df = build_table(analysis_dir, comparison_dir, ref_dir=ref_dir)
    table_path = os.path.join(outdir, "comparison_table.tsv")
    df.to_csv(table_path, sep="\t", index=False, float_format="%.4f")
    print(f"  saved {table_path}")
    print(df.to_string(index=False))

    # Backfill columns that may be absent in old TSVs
    if "display_name" not in df.columns:
        df["display_name"] = df["method"].map(display_name)
    if "replicate" not in df.columns:
        df["replicate"] = df["method"].map(lambda m: METHOD_INFO.get(m, (None, None, None))[2] or "")

    df = _order_methods(df)
    plot_comparison(df, outdir)
