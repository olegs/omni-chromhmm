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
        for state, col_base in [("Tx", "Tx_length"),
                                ("Tss", "Tss_length"),
                                ("TxWk", "TxWk_length")]:
            if state in report:
                row[f"median_{col_base}"] = report[state].get("median_length", np.nan)
                row[f"mean_{col_base}"] = report[state].get("mean_length", np.nan)

        # Enrichment
        enrichment = load_enrichment(_adir(method), method)
        def _get_val(data, st, ann):
            if st not in data: return np.nan
            if ann in data[st]: return data[st][ann]
            # Fallback to label without genome suffix
            base = ann.split(".")[0]
            for k in data[st]:
                if k.split(".")[0] == base:
                    return data[st][k]
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

        # Jaccard vs ATAC (any atac_* label)
        if "Tss" in jaccard:
            atac_label = next((l for l in jaccard["Tss"] if l.startswith("atac_")), None)
            if atac_label:
                row["jaccard_Tss_ATAC"] = jaccard["Tss"][atac_label]

        # Enrichment vs ATAC (any atac_* label)
        if "Tss" in enrichment:
            atac_label = next((l for l in enrichment["Tss"] if l.startswith("atac_")), None)
            if atac_label:
                row["enrich_Tss_ATAC"] = enrichment["Tss"][atac_label]

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
        ("jaccard_Tss_ATAC",               "Jaccard: Tss state vs ATAC-seq"),
        ("median_Tx_length",               "Median Tx (transcription) segment length"),
        ("mean_Tx_length",                 "Mean Tx (transcription) segment length"),
    ]:
        ylabel = "Fold enrichment" if col.startswith("enrich") else \
                 "Jaccard" if col.startswith("jaccard") else "bp"
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
