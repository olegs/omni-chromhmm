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
  - {comparison_dir}/ami_matrix.tsv
  - {comparison_dir}/jaccard_similarity_matrix.tsv
  - {comparison_dir}/segment_stats.tsv

Produces in {outdir}/:
  - comparison_table.tsv
  - per-metric PNG plots
"""

import argparse
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
    """Returns dict: seg_name → {entropy, entropy_no_quies}."""
    result = {}
    for suffix, col in [("", "entropy"), ("_no_quies", "entropy_no_quies")]:
        path = os.path.join(comparison_dir, f"entropy_summary{suffix}.tsv")
        if not os.path.exists(path):
            continue
        for _, row in pd.read_csv(path, sep="\t").iterrows():
            result.setdefault(row["segmentation"], {})[col] = row["total_entropy"]
    return result


def load_segment_stats(comparison_dir):
    """Returns dict: seg_name → {n_states, n_segments, ...}."""
    path = os.path.join(comparison_dir, "segment_stats.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    return {
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

def build_table(analysis_dir, comparison_dir, ref_dir=None, rematch=None):
    """Build the unified comparison DataFrame (one row per pooled method).

    analysis_dir : variant-specific subdir (e.g. ds/analysis/comb/)
    ref_dir      : top-level analysis dir containing ref/ (e.g. ds/analysis/).
                   Defaults to analysis_dir when not provided.
    rematch      : one of 'ovlp', 'binem', 'bwem' — when given, only replicate
                   re-match columns for that method are populated (used for
                   rematched_{rematch}/ output dirs).
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
    kappa_mat              = _load_matrix(comparison_dir, "kappa_matrix.tsv")
    ami_mat                = _load_matrix(comparison_dir, "ami_matrix.tsv")
    jaccard_mat            = _load_matrix(comparison_dir, "jaccard_similarity_matrix.tsv")
    overlap_mat            = _load_matrix(comparison_dir, "overlap_matrix.tsv")
    kappa_rematch_ovlp_mat   = _load_matrix(comparison_dir, "kappa_rematch_ovlp_matrix.tsv")
    jaccard_rematch_ovlp_mat = _load_matrix(comparison_dir, "jaccard_rematch_ovlp_matrix.tsv")
    overlap_rematch_ovlp_mat = _load_matrix(comparison_dir, "overlap_rematch_ovlp_matrix.tsv")
    kappa_rematch_binem_mat     = _load_matrix(comparison_dir, "kappa_rematch_binem_matrix.tsv")
    jaccard_rematch_binem_mat   = _load_matrix(comparison_dir, "jaccard_rematch_binem_matrix.tsv")
    overlap_rematch_binem_mat   = _load_matrix(comparison_dir, "overlap_rematch_binem_matrix.tsv")
    emission_mat             = _load_matrix(comparison_dir, "emission_similarity_matrix.tsv")
    kappa_rematch_bwem_mat     = _load_matrix(comparison_dir, "kappa_rematch_bwem_matrix.tsv")
    jaccard_rematch_bwem_mat   = _load_matrix(comparison_dir, "jaccard_rematch_bwem_matrix.tsv")
    overlap_rematch_bwem_mat   = _load_matrix(comparison_dir, "overlap_rematch_bwem_matrix.tsv")
    bw_emission_mat          = _load_matrix(comparison_dir, "bw_emission_similarity_matrix.tsv")
    seg_stats    = load_segment_stats(comparison_dir)

    # seg_names = union of all labels seen across every metric source
    seg_names = set(entropy_data) | set(seg_stats)
    for mat in (kappa_mat, ami_mat, jaccard_mat, overlap_mat,
                kappa_rematch_ovlp_mat, jaccard_rematch_ovlp_mat, overlap_rematch_ovlp_mat,
                kappa_rematch_binem_mat, jaccard_rematch_binem_mat, overlap_rematch_binem_mat,
                emission_mat,
                kappa_rematch_bwem_mat, jaccard_rematch_bwem_mat, overlap_rematch_bwem_mat,
                bw_emission_mat):
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

        if not rematch:
            # Entropy
            if seg_name in entropy_data:
                row["entropy"]          = entropy_data[seg_name].get("entropy", np.nan)
                row["entropy_no_quies"] = entropy_data[seg_name].get("entropy_no_quies", np.nan)

            # Segment stats
            if seg_name in seg_stats:
                row.update(seg_stats[seg_name])

            # vs reference (base metrics only; rematch vs-ref cols live in rematched_*/)
            if seg_name and ref_seg:
                for mat, col in [
                    (kappa_mat, "kappa_vs_ref"),
                    (ami_mat,   "ami_vs_ref"),
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
                (kappa_mat,   "kappa_rep1_vs_rep2"),
                (ami_mat,     "ami_rep1_vs_rep2"),
                (jaccard_mat, "jaccard_rep1_vs_rep2"),
                (overlap_mat, "overlap_rep1_vs_rep2"),
            ]
            # Rematch replicate columns, grouped by method.
            rematch_rep_cols = {
                "ovlp": [
                    (kappa_rematch_ovlp_mat,   "kappa_rematch_ovlp_rep1_vs_rep2"),
                    (jaccard_rematch_ovlp_mat, "jaccard_rematch_ovlp_rep1_vs_rep2"),
                    (overlap_rematch_ovlp_mat, "overlap_rematch_ovlp_rep1_vs_rep2"),
                ],
                "binem": [
                    (kappa_rematch_binem_mat,   "kappa_rematch_binem_rep1_vs_rep2"),
                    (jaccard_rematch_binem_mat, "jaccard_rematch_binem_rep1_vs_rep2"),
                    (overlap_rematch_binem_mat, "overlap_rematch_binem_rep1_vs_rep2"),
                    (emission_mat,           "emission_rep1_vs_rep2"),
                ],
                "bwem": [
                    (kappa_rematch_bwem_mat,   "kappa_rematch_bwem_rep1_vs_rep2"),
                    (jaccard_rematch_bwem_mat, "jaccard_rematch_bwem_rep1_vs_rep2"),
                    (overlap_rematch_bwem_mat, "overlap_rematch_bwem_rep1_vs_rep2"),
                    (bw_emission_mat,        "bw_emission_rep1_vs_rep2"),
                ],
            }
            # When rematch is given include only that method's cols;
            # otherwise only base as-is cols (rematch cols live in rematched_{rematch}/).
            if rematch:
                rep_pairs = base_rep_cols + rematch_rep_cols.get(rematch, [])
            else:
                rep_pairs = base_rep_cols
            for mat, col_name in rep_pairs:
                if mat is not None and rep1_seg in mat.index and rep2_seg in mat.columns:
                    row[col_name] = mat.loc[rep1_seg, rep2_seg]

        if not rematch:
            # Report: median lengths for key states
            report = load_report(_adir(method), method)
            for state, col in [("Tx", "median_Tx_length"),
                                ("Tss", "median_Tss_length"),
                                ("TxWk", "median_TxWk_length")]:
                if state in report:
                    row[col] = report[state].get("median_length", np.nan)

            # Enrichment
            enrichment = load_enrichment(_adir(method), method)
            for state, annotation, col in [
                ("Tx",   "RefSeqGene.hg38",        "enrich_Tx_RefSeqGene"),
                ("Tx",   "ExpressedGeneBodies",     "enrich_Tx_ExpressedGeneBodies"),
                ("Tss",  "RefSeqTSS.hg38",          "enrich_Tss_RefSeqTSS"),
                ("Tss",  "RefSeqTSS2kb.hg38",       "enrich_Tss_RefSeqTSS2kb"),
                ("Enh1", "ExpressedTSS",            "enrich_Enh1_ExpressedTSS"),
            ]:
                if state in enrichment and annotation in enrichment[state]:
                    row[col] = enrichment[state][annotation]

        if not rematch:
            # Jaccard vs expressed annotations
            jaccard = load_jaccard(_adir(method), method)
            for state, annotation, col in [
                ("Tx",  "ExpressedGeneBodies", "jaccard_Tx_ExpressedGeneBodies"),
                ("Tss", "ExpressedTSS",        "jaccard_Tss_ExpressedTSS"),
            ]:
                if state in jaccard and annotation in jaccard[state]:
                    row[col] = jaccard[state][annotation]

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


def plot_comparison(df, outdir, rematch=None):
    """Generate per-metric bar chart PNGs."""
    from matplotlib.patches import Patch

    # Pooled methods only (no replicates) for main plots
    df_main = df[df["replicate"] == ""].reset_index(drop=True)
    w = max(5, len(df_main) * 0.6)

    legend_elements = [
        Patch(facecolor=BIN_COLORS["default"],   label="Default binarization"),
        Patch(facecolor=BIN_COLORS["omnipeak"],  label="OmniPeak binarization"),
        Patch(facecolor=BIN_COLORS["homer"],     label="Homer binarization"),
        Patch(facecolor=BIN_COLORS["reference"], label="ENCODE reference"),
    ]

    def _make_fig():
        fig, ax = plt.subplots(figsize=(w, 3.5))
        ax.legend(handles=legend_elements, fontsize=6,
                  bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        return fig, ax

    # Replicate consistency plots — filtered by rematch when specified.
    _REMATCH_REP_COLS = {
        None: [
            ("kappa_rep1_vs_rep2",   "Replicate reproducibility (Kappa)",    "Kappa"),
            ("ami_rep1_vs_rep2",     "Replicate reproducibility (AMI)",      "AMI"),
            ("jaccard_rep1_vs_rep2", "Replicate reproducibility (Jaccard)",  "Similarity"),
            ("overlap_rep1_vs_rep2", "Replicate reproducibility (Overlap)",  "Overlap fraction"),
        ],
        "ovlp": [
            ("kappa_rep1_vs_rep2",                "Replicate reproducibility (Kappa)",                 "Kappa"),
            ("jaccard_rep1_vs_rep2",              "Replicate reproducibility (Jaccard)",               "Similarity"),
            ("overlap_rep1_vs_rep2",              "Replicate reproducibility (Overlap)",               "Overlap fraction"),
            ("kappa_rematch_ovlp_rep1_vs_rep2",   "Replicate reproducibility (Kappa re-match ovlp)",   "Kappa"),
            ("jaccard_rematch_ovlp_rep1_vs_rep2", "Replicate reproducibility (Jaccard re-match ovlp)", "Jaccard"),
            ("overlap_rematch_ovlp_rep1_vs_rep2", "Replicate reproducibility (Overlap re-match ovlp)", "Overlap fraction"),
        ],
        "binem": [
            ("kappa_rep1_vs_rep2",                "Replicate reproducibility (Kappa)",                  "Kappa"),
            ("jaccard_rep1_vs_rep2",              "Replicate reproducibility (Jaccard)",                "Similarity"),
            ("overlap_rep1_vs_rep2",              "Replicate reproducibility (Overlap)",                "Overlap fraction"),
            ("kappa_rematch_binem_rep1_vs_rep2",   "Replicate reproducibility (Kappa re-match binem)",   "Kappa"),
            ("jaccard_rematch_binem_rep1_vs_rep2", "Replicate reproducibility (Jaccard re-match binem)", "Jaccard"),
            ("overlap_rematch_binem_rep1_vs_rep2", "Replicate reproducibility (Overlap re-match binem)", "Overlap fraction"),
            ("emission_rep1_vs_rep2",             "Replicate reproducibility (Emission bin)",           "Cosine similarity"),
        ],
        "bwem": [
            ("kappa_rep1_vs_rep2",                "Replicate reproducibility (Kappa)",                  "Kappa"),
            ("jaccard_rep1_vs_rep2",              "Replicate reproducibility (Jaccard)",                "Similarity"),
            ("overlap_rep1_vs_rep2",              "Replicate reproducibility (Overlap)",                "Overlap fraction"),
            ("kappa_rematch_bwem_rep1_vs_rep2",   "Replicate reproducibility (Kappa re-match bwem)",    "Kappa"),
            ("jaccard_rematch_bwem_rep1_vs_rep2", "Replicate reproducibility (Jaccard re-match bwem)",  "Jaccard"),
            ("overlap_rematch_bwem_rep1_vs_rep2", "Replicate reproducibility (Overlap re-match bwem)",  "Overlap fraction"),
            ("bw_emission_rep1_vs_rep2",          "Replicate reproducibility (Emission bw)",            "Cosine similarity"),
        ],
    }
    for col, title, ylabel in _REMATCH_REP_COLS[rematch]:
        if col in df_main.columns and df_main[col].notna().any():
            fig, ax = _make_fig()
            _bar_panel(ax, df_main, col, title, ylabel)
            fname = col.replace(f"_rematch_{rematch}", "_rematch") if rematch else col
            _save_panel(fig, outdir, fname)

    if not rematch:
        # Enrichment / Jaccard plots
        for col, title in [
            ("enrich_Tx_ExpressedGeneBodies",  "Tx enrichment vs expressed gene bodies"),
            ("jaccard_Tx_ExpressedGeneBodies", "Jaccard: Tx state vs expressed gene bodies"),
            ("jaccard_Tss_ExpressedTSS",       "Jaccard: Tss state vs TSS of expressed genes"),
            ("median_Tx_length",               "Median Tx (transcription) segment length"),
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

def main():
    ap = argparse.ArgumentParser(description="Unified method comparison table & plots")
    ap.add_argument("--analysis-dir",   required=True,
                    help="Variant-specific analysis subdir (e.g. ds/analysis/comb/)")
    ap.add_argument("--ref-dir",        default=None, dest="ref_dir",
                    help="Top-level analysis dir containing ref/ subdir "
                         "(defaults to --analysis-dir)")
    ap.add_argument("--comparison-dir", required=True,
                    help="Directory with entropy, kappa, segment_stats TSVs")
    ap.add_argument("--outdir",         required=True, help="Output directory")
    ap.add_argument("--rematch",        default=None, choices=["ovlp", "binem", "bwem"],
                    help="When given, produce focused replicate re-match table "
                         "for this method (for rematched_{rematch}/ output dirs)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = build_table(args.analysis_dir, args.comparison_dir,
                     ref_dir=args.ref_dir, rematch=args.rematch)
    table_path = os.path.join(args.outdir, "comparison_table.tsv")
    df.to_csv(table_path, sep="\t", index=False, float_format="%.4f")
    print(f"  saved {table_path}")
    print(df.to_string(index=False))

    # Backfill columns that may be absent in old TSVs
    if "display_name" not in df.columns:
        df["display_name"] = df["method"].map(display_name)
    if "replicate" not in df.columns:
        df["replicate"] = df["method"].map(lambda m: METHOD_INFO.get(m, (None, None, None))[2] or "")

    df = _order_methods(df)
    plot_comparison(df, args.outdir, rematch=args.rematch)


if __name__ == "__main__":
    main()
