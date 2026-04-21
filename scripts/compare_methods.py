#!/usr/bin/env python3
"""Aggregate per-method metrics into a unified comparison table and figure.

Reads from:
  - {analysis_dir}/{method}/report.tsv                (segment lengths)
  - {analysis_dir}/{method}/enrichment/enrichment.tsv  (functional enrichment)
  - {comparison_dir}/entropy_summary.tsv               (transition entropy)
  - {comparison_dir}/entropy_summary_no_quies.tsv      (entropy excl. Quies/Het)
  - {comparison_dir}/kappa_matrix.tsv                  (pairwise Kappa)
  - {comparison_dir}/segment_stats.tsv                 (segment length statistics)

Produces in {outdir}/:
  - comparison_table.tsv   — one row per method, all metrics as columns
  - comparison_figure.png  — multi-panel bar chart comparison

Usage:
  python compare_methods.py \
      --analysis-dir imr90/analysis \
      --comparison-dir imr90/analysis/comparison \
      --outdir imr90/analysis/methods
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Method metadata — map analysis subdirectory name to classification
# ---------------------------------------------------------------------------

def classify_method(name):
    """Return (binarization, model) for an analysis subdirectory name."""
    if name == "ref":
        return "reference", "chromhmm"
    parts = name.split("_")
    # model: chromhmm_default, chromhmm, gmm, kmeans
    model = parts[0]  # chromhmm / gmm / kmeans
    if model == "chromhmm" and len(parts) > 1 and parts[1] == "default":
        binarization = "default"
    elif model == "chromhmm":
        binarization = "omnipeak"
    else:
        binarization = "omnipeak"
    return binarization, model


# Map from analysis subdir name → segmentation name in entropy/kappa files.
# Built dynamically by matching analysis dir names to available metric names.

def _build_analysis_to_seg_map(analysis_dirs, seg_names):
    """Heuristic mapping from analysis subdir → segmentation file basename.

    Analysis dirs:  chromhmm_default, chromhmm_omni, gmm_rep1, ref, ...
    Seg names:      IMR90_15_chromhmm_default_matched, gmm_omni_matched,
                    ENCFF714POQ_chromhmm, ...

    Strategy: for each analysis dir, find the seg name whose suffix best
    matches after stripping cell prefix and _matched.
    """
    mapping = {}
    set(seg_names)

    for adir in analysis_dirs:
        # Direct name match attempts
        candidates = []
        for seg in seg_names:
            # Strip trailing _matched for comparison
            seg_core = seg.replace("_matched", "")
            # "ref" maps to the reference (ENCODE accession)
            if adir == "ref":
                # Reference is the one without _matched and with ENCFF prefix
                if seg.startswith("ENCFF"):
                    candidates.append(seg)
                continue
            # Check if analysis dir name appears in seg name
            if adir in seg_core or adir in seg:
                candidates.append(seg)

        if len(candidates) == 1:
            mapping[adir] = candidates[0]
        elif len(candidates) > 1:
            # Prefer exact suffix match
            for c in candidates:
                core = c.replace("_matched", "")
                if core.endswith(adir) or adir in core:
                    mapping[adir] = c
                    break
            else:
                mapping[adir] = candidates[0]

    return mapping


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_entropy(entropy_dir):
    """Load entropy summaries. Returns dict: seg_name → {entropy, entropy_no_quies}."""
    result = {}
    for suffix, col in [("", "entropy"), ("_no_quies", "entropy_no_quies")]:
        path = os.path.join(entropy_dir, f"entropy_summary{suffix}.tsv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep="\t")
        for _, row in df.iterrows():
            name = row["segmentation"]
            if name not in result:
                result[name] = {}
            result[name][col] = row["total_entropy"]
    return result


def load_kappa_matrix(kappa_dir):
    """Load kappa matrix as a DataFrame (seg x seg)."""
    path = os.path.join(kappa_dir, "kappa_matrix.tsv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t", index_col=0)


def load_segment_stats(segment_stats_dir):
    """Load segment_stats.tsv. Returns dict: seg_name → {n_states, n_segments, ...}."""
    path = os.path.join(segment_stats_dir, "segment_stats.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    result = {}
    for _, row in df.iterrows():
        result[row["segmentation"]] = {
            "n_states": int(row["n_states"]),
            "n_segments": int(row["n_segments"]),
            "min_length": int(row["min_length"]),
            "max_length": int(row["max_length"]),
            "mean_length": float(row["mean_length"]),
            "median_length_all": float(row["median_length"]),
        }
    return result


def load_report(analysis_dir, method):
    """Load report.tsv for a method. Returns dict: state → {n_segments, total_bp, ...}."""
    path = os.path.join(analysis_dir, method, "report.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    return df.set_index("state").to_dict("index")


def load_enrichment(analysis_dir, method):
    """Load enrichment for a method. Returns dict: state → {annotation → odds_ratio}."""
    path = os.path.join(analysis_dir, method, "enrichment", "enrichment.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t")
    pivot = df.pivot(index="state", columns="label", values="odds_ratio")
    return pivot.to_dict("index")


# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------

def build_table(analysis_dir, comparison_dir):
    """Build the unified comparison DataFrame."""
    # Discover analysis methods
    methods = sorted([
        d for d in os.listdir(analysis_dir)
        if os.path.isdir(os.path.join(analysis_dir, d))
    ])

    # Load metrics (all from comparison_dir)
    entropy_data = load_entropy(comparison_dir)
    kappa_mat = load_kappa_matrix(comparison_dir)
    seg_stats = load_segment_stats(comparison_dir)

    # Build mapping from analysis dir → segmentation name in metrics
    seg_names = list(entropy_data.keys())
    if kappa_mat is not None:
        seg_names = list(set(seg_names) | set(kappa_mat.index))
    if seg_stats:
        seg_names = list(set(seg_names) | set(seg_stats.keys()))
    a2s = _build_analysis_to_seg_map(methods, seg_names)

    # Find reference segmentation name (for kappa_vs_ref)
    ref_seg = a2s.get("ref", "")

    rows = []
    for method in methods:
        binarization, model = classify_method(method)
        seg_name = a2s.get(method, "")

        row = {
            "method": method,
            "binarization": binarization,
            "model": model,
        }

        # Entropy
        if seg_name in entropy_data:
            row["entropy"] = entropy_data[seg_name].get("entropy", np.nan)
            row["entropy_no_quies"] = entropy_data[seg_name].get("entropy_no_quies", np.nan)

        # Segment length statistics
        if seg_name in seg_stats:
            for k in ("n_states", "n_segments", "min_length", "max_length",
                       "mean_length", "median_length_all"):
                row[k] = seg_stats[seg_name][k]

        # Kappa vs reference
        if kappa_mat is not None and seg_name in kappa_mat.index and ref_seg in kappa_mat.columns:
            row["kappa_vs_ref"] = kappa_mat.loc[seg_name, ref_seg]

        # Kappa rep1 vs rep2 (find matching rep pair)
        # Only for non-replicate, non-ref methods (pooled or mode-level)
        if kappa_mat is not None and method not in ("ref",) and "_rep" not in method:
            # Derive the model prefix (chromhmm, gmm, kmeans)
            # chromhmm_default → chromhmm_default_rep{1,2}
            # chromhmm_omni / chromhmm_replicated → chromhmm_rep{1,2}
            # gmm_omni / gmm_replicated → gmm_rep{1,2}
            parts = method.split("_")
            model_prefix = parts[0]  # chromhmm / gmm / kmeans
            if model_prefix == "chromhmm" and len(parts) > 1 and parts[1] == "default":
                rep1_method = "chromhmm_default_rep1"
                rep2_method = "chromhmm_default_rep2"
            else:
                rep1_method = f"{model_prefix}_rep1"
                rep2_method = f"{model_prefix}_rep2"
            rep1_seg = a2s.get(rep1_method, "")
            rep2_seg = a2s.get(rep2_method, "")
            if rep1_seg in kappa_mat.index and rep2_seg in kappa_mat.columns:
                row["kappa_rep1_vs_rep2"] = kappa_mat.loc[rep1_seg, rep2_seg]

        # Report: median lengths for key states
        report = load_report(analysis_dir, method)
        for state, col in [("Tx", "median_Tx_length"),
                           ("Tss", "median_Tss_length"),
                           ("TxWk", "median_TxWk_length")]:
            if state in report:
                row[col] = report[state].get("median_length", np.nan)

        # Enrichment: key state-annotation pairs
        enrichment = load_enrichment(analysis_dir, method)
        enrich_pairs = [
            ("Tx", "RefSeqGene.hg38", "enrich_Tx_RefSeqGene"),
            ("Tx", "ExpressedGeneBodies", "enrich_Tx_ExpressedGeneBodies"),
            ("Tss", "RefSeqTSS.hg38", "enrich_Tss_RefSeqTSS"),
            ("Tss", "RefSeqTSS2kb.hg38", "enrich_Tss_RefSeqTSS2kb"),
            ("Enh1", "ExpressedTSS", "enrich_Enh1_ExpressedTSS"),
        ]
        for state, annotation, col in enrich_pairs:
            if state in enrichment and annotation in enrichment[state]:
                row[col] = enrichment[state][annotation]

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Colors by binarization type
BIN_COLORS = {"default": "#4878CF", "omnipeak": "#E8833A", "reference": "#888888"}


def _method_colors(df):
    """Return list of colors based on binarization column."""
    return [BIN_COLORS.get(b, "#888888") for b in df["binarization"]]


def _bar_panel(ax, df, col, title, ylabel=None):
    """Draw a bar chart for one column of the comparison table."""
    vals = df[col].values if col in df.columns else np.full(len(df), np.nan)
    colors = _method_colors(df)
    x = np.arange(len(df))
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=55, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # Value labels
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6)


def _grouped_bar_panel(ax, df, cols, labels, title, ylabel=None):
    """Draw grouped bar chart for multiple columns."""
    x = np.arange(len(df))
    width = 0.8 / len(cols)
    colors_base = _method_colors(df)
    for j, (col, label) in enumerate(zip(cols, labels)):
        vals = df[col].values if col in df.columns else np.full(len(df), np.nan)
        offsets = x + (j - len(cols) / 2 + 0.5) * width
        # Darken/lighten colors for groups
        alpha = 0.6 + 0.4 * j / max(len(cols) - 1, 1)
        ax.bar(offsets, vals, width, color=colors_base, alpha=alpha, label=label,
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=55, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def plot_comparison(df, outdir):
    """Create multi-panel comparison figure."""
    # Filter to non-replicate methods for cleaner view
    # (keep ref, pooled default, pooled omni variants)
    mask = ~df["method"].str.contains("rep[12]", regex=True)
    df_main = df[mask].reset_index(drop=True)

    n_panels = 8
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(10, len(df_main) * 0.9), n_panels * 4))

    # Panel A: Kappa vs reference
    _bar_panel(axes[0], df_main, "kappa_vs_ref",
               "A. Cohen's Kappa vs ENCODE reference", "Kappa")

    # Panel B: Kappa rep1 vs rep2
    has_rep_kappa = "kappa_rep1_vs_rep2" in df_main.columns and df_main["kappa_rep1_vs_rep2"].notna().any()
    if has_rep_kappa:
        _bar_panel(axes[1], df_main, "kappa_rep1_vs_rep2",
                   "B. Replicate reproducibility (Kappa rep1 vs rep2)", "Kappa")
    else:
        axes[1].text(0.5, 0.5, "No replicate data available",
                     transform=axes[1].transAxes, ha="center", va="center",
                     fontsize=12, color="gray")
        axes[1].set_title("B. Replicate reproducibility", fontsize=10, fontweight="bold")

    # Panel C: Total number of states
    _bar_panel(axes[2], df_main, "n_states",
               "C. Total number of states", "Count")

    # Panel D: Segment length statistics (min, max, mean, median)
    len_cols = [c for c in ["min_length", "max_length", "mean_length", "median_length_all"]
                if c in df_main.columns and df_main[c].notna().any()]
    len_labels = [c.replace("_length", "").replace("_all", "").replace("_", " ").capitalize()
                  for c in len_cols]
    if len_cols:
        _grouped_bar_panel(axes[3], df_main, len_cols, len_labels,
                           "D. Segment length statistics", "bp")
    else:
        axes[3].text(0.5, 0.5, "No segment stats data", transform=axes[3].transAxes,
                     ha="center", va="center", fontsize=12, color="gray")
        axes[3].set_title("D. Segment length statistics", fontsize=10, fontweight="bold")

    # Panel E: Key enrichments
    enrich_cols = [c for c in ["enrich_Tx_RefSeqGene", "enrich_Tss_RefSeqTSS",
                               "enrich_Tx_ExpressedGeneBodies"]
                   if c in df_main.columns and df_main[c].notna().any()]
    enrich_labels = [c.replace("enrich_", "").replace("_", " → ") for c in enrich_cols]
    if enrich_cols:
        _grouped_bar_panel(axes[4], df_main, enrich_cols, enrich_labels,
                           "E. Functional enrichment (key state-annotation pairs)",
                           "Odds ratio")
    else:
        axes[4].text(0.5, 0.5, "No enrichment data", transform=axes[4].transAxes,
                     ha="center", va="center", fontsize=12, color="gray")
        axes[4].set_title("E. Functional enrichment", fontsize=10, fontweight="bold")

    # Panel F: Median Tx segment length
    _bar_panel(axes[5], df_main, "median_Tx_length",
               "F. Median Tx (transcription) segment length", "bp")

    # Panel G: Entropy comparison
    ent_cols = [c for c in ["entropy", "entropy_no_quies"]
                if c in df_main.columns and df_main[c].notna().any()]
    ent_labels = ["All states", "Excl. Quies/Het"][:len(ent_cols)]
    if ent_cols:
        _grouped_bar_panel(axes[6], df_main, ent_cols, ent_labels,
                           "G. Transition matrix entropy", "bits")
    else:
        axes[6].set_title("G. Transition matrix entropy", fontsize=10, fontweight="bold")

    # Panel H: Median segment length (all states)
    _bar_panel(axes[7], df_main, "median_length_all",
               "H. Median segment length (all states)", "bp")

    # Legend for binarization colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BIN_COLORS["default"], label="Default binarization"),
                       Patch(facecolor=BIN_COLORS["omnipeak"], label="Omnipeak binarization"),
                       Patch(facecolor=BIN_COLORS["reference"], label="ENCODE reference")]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = os.path.join(outdir, "comparison_figure.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Unified method comparison table & figure")
    ap.add_argument("--analysis-dir", required=True,
                    help="Directory containing per-method analysis subdirs")
    ap.add_argument("--comparison-dir", required=True,
                    help="Directory with entropy, kappa, segment_stats TSVs")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = build_table(args.analysis_dir, args.comparison_dir)

    # Save table
    table_path = os.path.join(args.outdir, "comparison_table.tsv")
    df.to_csv(table_path, sep="\t", index=False, float_format="%.4f")
    print(f"  saved {table_path}")
    print(df.to_string(index=False))

    # Plot
    plot_comparison(df, args.outdir)


if __name__ == "__main__":
    main()
