#!/usr/bin/env python3
"""Cross-dataset summary bar plots with mean ± std error bars.

Reads per-dataset comparison_table.tsv and per-method jaccard.tsv files,
then produces one grouped bar chart per metric showing performance across
all datasets (mean ± std).

Plots are generated:
  summary_entropy_noqh.png        — transition matrix entropy (NOQH)
  summary_jaccard_tx.png          — Jaccard: Tx state vs expressed gene bodies
  summary_enrich_tx.png           — Tx fold enrichment at expressed gene bodies
  summary_mean_tx_length.png      — mean Tx segment length
  summary_jaccard_tss.png         — Jaccard: Tss state vs RefSeqTSS ±1 kb
  summary_jaccard_tss_atac.png    — Jaccard: Tss state vs ATAC-seq peaks
  summary_n_segments.png          — total number of segments

Importable module (no CLI). Drive it from analysis.ipynb:
    from summary_plots import run_summary_plots
    run_summary_plots(datasets=[...], methods_dirs=[...], analysis_dirs=[...],
                      outdir="out/summary_plots")
Each *_outfile / *_outdir argument that is set selects one plot group.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from utils import METHOD_ORDER, DISPLAY_NAMES, BIN_COLORS, METHOD_INFO
from analyze import load_bed_df

METHODS_POOLED = [m for m in METHOD_ORDER
                  if not m.endswith("_rep1") and not m.endswith("_rep2")]


# ---------------------------------------------------------------------------
# Inter-dataset violin: state/method constants and helpers
# ---------------------------------------------------------------------------

# Canonical chromatin state order (ENCODE 15-state naming convention)
STATE_ORDER = [
    "Tss", "TssFlnk", "TssFlnkU", "TssFlnkD",
    "Tx", "TxWk",
    "EnhG", "EnhG1", "EnhG2",
    "Enh", "Enh1", "Enh2", "EnhLo",
    "ZNF/Rpts", "Het", "Biv",
    "ReprPC", "ReprPCWk",
    "Quies", "Unknown",
]
STATE_IDX = {s: i for i, s in enumerate(STATE_ORDER)}

# ENCODE 15-state canonical RGB colors (from BED column 9, consistent across all references)
_RGB = lambda r, g, b: (r / 255, g / 255, b / 255)
STATE_COLORS = {
    "Tss":      _RGB(255,   0,   0),
    "TssFlnk":  _RGB(255,  69,   0),
    "TssFlnkU": _RGB(255,  69,   0),
    "TssFlnkD": _RGB(255,  69,   0),
    "Tx":       _RGB(  0, 128,   0),
    "TxWk":     _RGB( 63, 154,  80),
    "EnhG":     _RGB(170, 223,   7),
    "EnhG1":    _RGB(170, 223,   7),
    "EnhG2":    _RGB(170, 223,   7),
    "Enh":      _RGB(255, 223,   0),
    "Enh1":     _RGB(255, 223,   0),
    "Enh2":     _RGB(255, 223,   0),
    "EnhLo":    _RGB(255, 223,   0),
    "ZNF/Rpts": _RGB(104, 205, 170),
    "Het":      _RGB( 75,   0, 130),
    "Biv":      _RGB(205,  92,  92),
    "ReprPC":   _RGB(137,  55, 223),
    "ReprPCWk": _RGB(137,  55, 223),
    "Quies":    _RGB(220, 220, 220),
}

# (key, display label, hex color) — key matches METHOD_ORDER keys
INTER_DS_METHODS = [
    ("reference",        "ENCODE Ref",      BIN_COLORS["reference"]),
    ("chromhmm_default", "Default ChromHMM", BIN_COLORS["default"]),
    ("chromhmm_homer",   "ChromHMM HOMER",     BIN_COLORS["homer"]),
    ("kmeans_homer",     "KMeans HOMER",       BIN_COLORS["homer"]),
    ("chromhmm_macs2",   "ChromHMM MACS2",     BIN_COLORS["macs2"]),
    ("kmeans_macs2",     "KMeans MACS2",       BIN_COLORS["macs2"]),
    ("chromhmm_omni",    "ChromHMM OmniPeak",  BIN_COLORS["omnipeak"]),
    ("kmeans_omni",      "KMeans OmniPeak",    BIN_COLORS["omnipeak"]),
]
METHOD_PALETTE = {label: color for _, label, color in INTER_DS_METHODS}

MAX_SEGS_PER_METHOD = 500_000


def sort_states(states):
    """Sort chromatin state names in canonical ENCODE order."""
    return sorted(states, key=lambda s: (STATE_IDX.get(s, 999), s))


def ds_method_bed(workdir, ds, cell, nstates, method_key, match_method="comb"):
    """Return Path for a single (dataset, method) {match_method}_matched BED."""
    root = Path(workdir) / ds
    sfx = f"{match_method}_matched"
    mapping = {
        "chromhmm_default": root / "chromhmm_default_result" / f"{cell}_{nstates}_dense_{sfx}.bed",
        "chromhmm_omni":    root / "omni"  / "chromhmm_result" / f"omni_{cell}_{nstates}_dense_{sfx}.bed",
        "kmeans_omni":      root / "omni"  / f"omni_kmeans_states_{sfx}.bed",
        "chromhmm_homer":   root / "homer" / "chromhmm_result" / f"homer_{cell}_{nstates}_dense_{sfx}.bed",
        "kmeans_homer":     root / "homer" / f"homer_kmeans_states_{sfx}.bed",
        "chromhmm_macs2":   root / "macs2" / "chromhmm_result" / f"macs2_{cell}_{nstates}_dense_{sfx}.bed",
        "kmeans_macs2":     root / "macs2" / f"macs2_kmeans_states_{sfx}.bed",
    }
    return mapping[method_key]


def ref_beds(markups_dir):
    """Return sorted list of ENCODE reference BED paths from markups/15state/."""
    markups_path = Path(markups_dir) / "15state"
    files = sorted(markups_path.glob("*.bed.gz")) + sorted(markups_path.glob("*.bed"))
    if not files:
        print(f"  WARNING: no reference BED files in {markups_path}", file=sys.stderr)
    return [str(f) for f in files]


def load_bed_segments(path, max_rows=MAX_SEGS_PER_METHOD):
    """Load a BED file; return DataFrame[state, log_length] (downsampled)."""
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: missing {p}", file=sys.stderr)
        return pd.DataFrame(columns=["state", "log_length"])
    df = load_bed_df(str(p))[["state", "length"]]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    df["log_length"] = np.log10(df["length"].values + 1)
    return df[["state", "log_length"]]


def load_reference_segments(markups_dir, max_rows=MAX_SEGS_PER_METHOD):
    """Pool all ENCODE reference BED files; return DataFrame[state, log_length, method]."""
    frames = [load_bed_df(p)[["state", "length"]] for p in ref_beds(markups_dir)]
    if not frames:
        return pd.DataFrame(columns=["state", "log_length", "method"])
    df = pd.concat(frames, ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42)
    df["log_length"] = np.log10(df["length"].values + 1)
    df["method"] = "ENCODE Ref"
    return df[["state", "log_length", "method"]]


# ---------------------------------------------------------------------------
# Data loading (summary bar plots)
# ---------------------------------------------------------------------------

def _load_comparison_table(methods_dir):
    """Return pooled-only rows from {methods_dir}/comparison_table.tsv, or None."""
    path = os.path.join(methods_dir, "comparison_table.tsv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    return df[df["replicate"].fillna("") == ""].copy()


def _load_jaccard_value(analysis_dir, method, state, label):
    """Return the Jaccard value for (state, label) from the per-method jaccard.tsv.

    For the special 'ref' method the jaccard.tsv lives one level above the
    variant dir (e.g. analysis/ref/ rather than analysis/ovlp/ref/).
    """
    if method == "ref":
        path = os.path.join(os.path.dirname(analysis_dir), "ref",
                            "enrichment", "jaccard.tsv")
    else:
        path = os.path.join(analysis_dir, method, "enrichment", "jaccard.tsv")
    if not os.path.exists(path):
        return np.nan
    df = pd.read_csv(path, sep="\t")
    mask = (df["state"] == state) & (df["label"] == label)
    if not mask.any():
        return np.nan
    return float(df.loc[mask, "jaccard"].iloc[0])


def _collect_table_col(datasets, methods_dirs, col, include_ref=False):
    """Collect one comparison_table column across datasets.

    include_ref: also keep the ENCODE reference row (method key "ref"), which is
    otherwise excluded because "ref" is not among the de-novo METHODS_POOLED.
    Returns DataFrame with index=method, columns=dataset.
    """
    allowed = set(METHODS_POOLED) | ({"ref"} if include_ref else set())
    records = {}
    for ds, mdir in zip(datasets, methods_dirs):
        table = _load_comparison_table(mdir)
        if table is None:
            continue
        for _, row in table.iterrows():
            method = row["method"]
            if method not in allowed:
                continue
            val = row.get(col, np.nan)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = np.nan
            records.setdefault(method, {})[ds] = val
    return pd.DataFrame(records).T  # index=method, columns=dataset


def _collect_jaccard(datasets, analysis_dirs, state, label):
    """Collect Jaccard values for (state, label) across datasets.

    Returns DataFrame with index=method, columns=dataset.
    """
    records = {}
    for ds, adir in zip(datasets, analysis_dirs):
        for method in METHODS_POOLED:
            val = _load_jaccard_value(adir, method, state, label)
            records.setdefault(method, {})[ds] = val
    return pd.DataFrame(records).T


def _load_jaccard_atac_value(analysis_dir, method, state):
    """Return Jaccard of (state, atac_*) — matches any atac_ label."""
    if method == "ref":
        path = os.path.join(os.path.dirname(analysis_dir), "ref",
                            "enrichment", "jaccard.tsv")
    else:
        path = os.path.join(analysis_dir, method, "enrichment", "jaccard.tsv")
    if not os.path.exists(path):
        return np.nan
    df = pd.read_csv(path, sep="\t")
    mask = (df["state"] == state) & df["label"].str.startswith("atac_")
    if not mask.any():
        return np.nan
    return float(df.loc[mask, "jaccard"].iloc[0])


def _collect_jaccard_atac(datasets, analysis_dirs, state):
    """Collect ATAC-seq Jaccard values for *state* across datasets.

    Matches any label starting with 'atac_' (dataset-specific accession).
    Datasets without ATAC data contribute NaN.
    Returns DataFrame with index=method, columns=dataset.
    """
    records = {}
    for ds, adir in zip(datasets, analysis_dirs):
        for method in METHODS_POOLED:
            val = _load_jaccard_atac_value(adir, method, state)
            records.setdefault(method, {})[ds] = val
    return pd.DataFrame(records).T


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_summary(data, title, ylabel, outpath, partial_note=False, order=None):
    """Grouped bar chart: mean ± std across datasets.

    order: explicit method-key order (e.g. ["ref", ...] to include the reference);
    defaults to METHODS_POOLED.
    """
    method_order = order if order is not None else METHODS_POOLED
    methods = [m for m in method_order if m in data.index]
    if not methods:
        print(f"  skipping {outpath}: no data")
        return

    # Drop methods that have no data at all (e.g. ENCODE Ref in rep-consistency plots)
    methods = [m for m in methods
               if not data.loc[m].isna().all()]

    display   = [DISPLAY_NAMES.get(m, m) for m in methods]
    colors    = [BIN_COLORS.get(METHOD_INFO[m][0], "#888888") for m in methods]
    means     = data.loc[methods].mean(axis=1, skipna=True).values
    stds      = data.loc[methods].std(axis=1, skipna=True).values
    counts    = data.loc[methods].count(axis=1).values

    # Write a "no data" placeholder rather than skipping, so Snakemake output files always exist
    if np.all(np.isnan(means)):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, fontweight="bold")
        os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {outpath} (no data)")
        return

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 0.65), 4.2))

    ax.bar(x, np.nan_to_num(means), color=colors,
           edgecolor="white", linewidth=0.5)

    for i, (m, s, c) in enumerate(zip(means, stds, counts)):
        if np.isnan(m) or c == 0:
            continue
        if not np.isnan(s) and c > 1:
            ax.errorbar(x[i], m, yerr=s, fmt="none", color="black",
                        capsize=3, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    n_ds = data.shape[1]
    for i, (m, s, c) in enumerate(zip(means, stds, counts)):
        if np.isnan(m) or c == 0:
            continue
        top = m + (s if not np.isnan(s) and c > 1 else 0)
        lbl = f"{m:.2f}"
        if partial_note and c < n_ds:
            lbl += f"\n(n={c})"
        ax.text(x[i], top + yrange * 0.01, lbl,
                ha="center", va="bottom", fontsize=6)

    legend_elements = []
    bin_labels = {
        "reference": "ENCODE reference",
        "default":   "Default binarization",
        "omnipeak":  "OmniPeak binarization",
        "homer":     "Homer binarization",
        "macs2":     "MACS2 binarization",
    }
    # Only show binarizations that are actually present in the plotted methods
    plotted_bins = {METHOD_INFO.get(m, (None,))[0] for m in methods}
    active_binarizations = [b for b in ["reference", "default", "omnipeak", "homer", "macs2"]
                            if b in plotted_bins]

    for b in active_binarizations:
        legend_elements.append(Patch(facecolor=BIN_COLORS[b], label=bin_labels[b]))

    if legend_elements:
        ax.legend(handles=legend_elements, fontsize=6,
                  bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    n_note = f"mean ± std across {n_ds} datasets"
    if partial_note:
        n_note += " (n = datasets with data)"
    ax.set_xlabel(n_note, fontsize=7, color="grey")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outpath}")


def _collect_per_state_kappa(datasets, analysis_dirs, match_method="comb"):
    """Collect per-state kappa vs reference for all datasets."""
    all_stacked = []
    for ds, adir in zip(datasets, analysis_dirs):
        # We look into potential locations for per-state kappa TSVs.
        # 1. {ds}/comparison/{match_method} (standard notebook location)
        # 2. {adir}/comparison/{match_method}
        # 3. {adir}/comparison
        comp_dirs = [
            os.path.join(ds, "comparison", match_method),
            os.path.join(adir, "comparison", match_method),
            os.path.join(adir, "comparison")
        ]
        
        comp_dir = None
        for d in comp_dirs:
            if os.path.isdir(d):
                comp_dir = d
                break
        
        if not comp_dir:
            continue

        # Look for per_state_kappa_vs_*.tsv
        try:
            tsv_files = [f for f in os.listdir(comp_dir)
                         if f.startswith("per_state_kappa_vs_") and f.endswith(".tsv")]
        except OSError:
            continue

        if not tsv_files:
            continue

        tsv_path = os.path.join(comp_dir, tsv_files[0])
        try:
            df = pd.read_csv(tsv_path, sep="\t", index_col=0)
            if df.empty:
                continue

            # Stack to get (state, method, kappa)
            stacked = df.stack().reset_index()
            stacked.columns = ["state", "method", "kappa"]
            # Exclude NaNs/Nones as per instructions
            stacked = stacked.dropna(subset=["kappa"])
            stacked["dataset"] = ds
            all_stacked.append(stacked)
        except Exception as e:
            print(f"  WARNING: could not load {tsv_path}: {e}", file=sys.stderr)

    if not all_stacked:
        return pd.DataFrame()
    return pd.concat(all_stacked, ignore_index=True)


def _save_kappa_heatmap(df, title, outfile):
    """Save a per-state kappa heatmap, matching compare.py style."""
    if df.empty:
        return
    # Exclude all-NaN rows/cols to keep plot clean
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    if df.empty:
        return

    vmax = max(float(np.nanmax(np.abs(df.values))), 0.1) if not df.isna().all().all() else 0.1
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1] * 0.8),
                                    max(4, df.shape[0] * 0.35)))
    sns.heatmap(df, cmap="RdYlGn", vmin=-vmax, vmax=vmax, center=0,
                linewidths=0.5, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7},
                cbar_kws={"label": "Per-state Cohen's Kappa"},
                ax=ax, mask=df.isna().values)
    ax.set_title(title, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    fig.savefig(outfile)
    plt.close(fig)
    print(f"  saved {outfile}")


def _plot_per_state_kappa(datasets, analysis_dirs, outdir, match_method="comb"):
    """Generate per-dataset and summary per-state kappa plots."""
    df = _collect_per_state_kappa(datasets, analysis_dirs, match_method)
    if df.empty:
        return

    # 1. Per-dataset plots
    for ds in datasets:
        subset = df[df["dataset"] == ds]
        if subset.empty:
            continue
        ds_df = subset.pivot(index="state", columns="method", values="kappa")
        if ds_df.empty:
            continue
        ds_df = ds_df.reindex(sort_states(ds_df.index))
        # Use METHODS_POOLED for canonical order (if present in df)
        methods_in_plot = [m for m in METHODS_POOLED if m in ds_df.columns and m != "ref"]
        # Also include any other methods that might be there but not in METHODS_POOLED
        other_methods = [m for m in ds_df.columns if m not in METHODS_POOLED and m != "ref"]
        ds_df = ds_df[methods_in_plot + other_methods]

        out_path = os.path.join(outdir, f"per_state_kappa_{ds}.png")
        _save_kappa_heatmap(ds_df, f"Per-state Cohen's Kappa: {ds} vs Reference", out_path)

    # 2. Summary plot
    summary_df = df.groupby(["state", "method"])["kappa"].mean().unstack()
    summary_df = summary_df.reindex(sort_states(summary_df.index))
    methods_in_plot = [m for m in METHODS_POOLED if m in summary_df.columns and m != "ref"]
    other_methods = [m for m in summary_df.columns if m not in METHODS_POOLED and m != "ref"]
    summary_df = summary_df[methods_in_plot + other_methods]

    out_path = os.path.join(outdir, "per_state_kappa_summary.png")
    _save_kappa_heatmap(summary_df, "Mean Per-state Cohen's Kappa vs Reference (Summary)", out_path)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Replicate consistency summary plots
# ---------------------------------------------------------------------------

_REP_CONSISTENCY_PLOTS = [
    # (col_from_methods_table, title, ylabel, outfile_stem)
    ("kappa_noqh_rep1_vs_rep2",
     "Rep. consistency: Kappa (NOQH, raw)",   "Kappa",
     "rep_consistency_kappa_noqh_rep1_vs_rep2"),
    ("kappa_rep1_vs_rep2",
     "Rep. consistency: Kappa (full, raw)",   "Kappa",
     "rep_consistency_kappa_rep1_vs_rep2"),
    ("jaccard_noqh_rep1_vs_rep2",
     "Rep. consistency: Jaccard (NOQH, raw)", "Jaccard",
     "rep_consistency_jaccard_noqh_rep1_vs_rep2"),
    ("jaccard_rep1_vs_rep2",
     "Rep. consistency: Jaccard (full, raw)", "Jaccard",
     "rep_consistency_jaccard_rep1_vs_rep2"),
]


def _plot_rep_consistency(datasets, methods_dirs, outdir):
    """Generate replicate consistency bar plots (mean ± std across datasets with replicates)."""
    os.makedirs(outdir, exist_ok=True)
    for col, title, ylabel, stem in _REP_CONSISTENCY_PLOTS:
        outpath = os.path.join(outdir, f"{stem}.png")
        data = _collect_table_col(datasets, methods_dirs, col)
        _plot_summary(data, title, ylabel, outpath, partial_note=True)


# Segment length violin (from matched_stats_all.tsv)
# ---------------------------------------------------------------------------

# Map matched_stats sample key → (display label, binarization key)
_SAMPLE_TO_INFO = {
    "chromhmm_default": ("Default ChromHMM",  "default"),
    "chromhmm_omni":    ("OmniPeak ChromHMM", "omnipeak"),
    "kmeans_omni":      ("KMeans OmniPeak",   "omnipeak"),
    "chromhmm_homer":   ("HOMER ChromHMM",    "homer"),
    "kmeans_homer":     ("KMeans HOMER",      "homer"),
    "chromhmm_macs2":   ("MACS2 ChromHMM",    "macs2"),
    "kmeans_macs2":     ("KMeans MACS2",      "macs2"),
}


# ---------------------------------------------------------------------------
# Peak statistics (from {ds}/peaks/peak_stats.tsv)
# ---------------------------------------------------------------------------

_PEAK_METHOD_COLORS = {
    "OmniPeak": BIN_COLORS["omnipeak"],
    "Default":  BIN_COLORS["default"],
    "HOMER":    BIN_COLORS["homer"],
    "MACS2":    BIN_COLORS["macs2"],
}
_PEAK_METHOD_ORDER = ["OmniPeak", "Default", "HOMER", "MACS2"]
_MARK_ORDER = ["H3K4me3", "H3K27ac", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3"]


def _load_peak_frames(datasets, workdir):
    frames = []
    for ds in datasets:
        path = os.path.join(workdir, ds, "peaks", "peak_stats.tsv")
        if not os.path.exists(path):
            print(f"  skipping {path}: not found")
            continue
        df = pd.read_csv(path, sep="\t")[["method", "mark", "n_peaks", "mean_length"]]
        df["dataset"] = ds
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def _peak_bar(data, col, ylabel, title, outpath):
    marks   = [m for m in _MARK_ORDER if m in data["mark"].unique()]
    methods = [m for m in _PEAK_METHOD_ORDER if m in data["method"].unique()]
    x = np.arange(len(marks))
    width = 0.8 / len(methods)
    offsets = np.linspace(-(len(methods)-1)/2, (len(methods)-1)/2, len(methods)) * width
    fig, ax = plt.subplots(figsize=(max(6, len(marks) * len(methods) * 0.22 + 2), 4.5))
    for offset, method in zip(offsets, methods):
        means, stds = [], []
        for mark in marks:
            vals = data.loc[(data["method"] == method) & (data["mark"] == mark), col].values
            means.append(np.mean(vals) if len(vals) else np.nan)
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        means, stds = np.array(means), np.array(stds)
        ax.bar(x + offset, np.nan_to_num(means), width=width * 0.9,
               color=_PEAK_METHOD_COLORS[method], label=method,
               edgecolor="white", linewidth=0.4)
        valid = ~np.isnan(means)
        ax.errorbar(x[valid] + offset, means[valid], yerr=stds[valid],
                    fmt="none", color="black", capsize=2, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(marks, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outpath}")


def _load_gap_frames(datasets, workdir):
    frames = []
    for ds in datasets:
        path = os.path.join(workdir, ds, "peaks", "gap_lengths.tsv.gz")
        if not os.path.exists(path):
            print(f"  skipping {path}: not found")
            continue
        df = pd.read_csv(path, sep="\t", compression="gzip")[["method", "mark", "gap_length"]]
        df["dataset"] = ds
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def _plot_peak_gap_violin(datasets, workdir, outpath):
    """Violin plot: distribution of gap lengths between adjacent binarized elements.

    x-axis: method; hue: histone modification (mark); y-axis: log10(gap_length+1).
    Data pooled across all datasets.
    """
    data = _load_gap_frames(datasets, workdir)
    if data is None:
        print(f"  skipping {outpath}: no data")
        return

    marks   = [m for m in _MARK_ORDER if m in data["mark"].unique()]
    methods = [m for m in _PEAK_METHOD_ORDER if m in data["method"].unique()]

    data = data[data["method"].isin(methods) & data["mark"].isin(marks)].copy()
    data["log_gap"] = np.log10(data["gap_length"].values + 1)
    data["method"] = pd.Categorical(data["method"], categories=methods, ordered=True)
    data["mark"]   = pd.Categorical(data["mark"],   categories=marks,   ordered=True)

    mark_palette = sns.color_palette("tab10", n_colors=len(marks))
    mark_colors  = {m: mark_palette[i] for i, m in enumerate(marks)}

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * len(marks) * 0.35 + 2), 5))
    sns.violinplot(
        data=data, x="method", y="log_gap", hue="mark",
        order=methods, hue_order=marks,
        palette=mark_colors,
        linewidth=0.4, density_norm="width", inner="quartile", cut=0,
        ax=ax,
    )
    ax.set_xlabel("Binarization method", fontsize=9)
    ax.set_ylabel("log₁₀(gap length + 1)  [bp]", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(title="Mark", fontsize=7, title_fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    n = len(datasets)
    ax.set_title(
        f"Gap lengths between adjacent binarized elements\n"
        f"(pooled across {n} datasets)",
        fontsize=10, fontweight="bold",
    )
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outpath}")


def _plot_peak_count(datasets, workdir, outpath):
    """Grouped bar chart: n_peaks per mark per method, mean ± std across datasets."""
    data = _load_peak_frames(datasets, workdir)
    if data is None:
        print(f"  skipping {outpath}: no data")
        return
    n = len(datasets)
    _peak_bar(data, "n_peaks", "Number of peaks",
              f"Peak count per mark and method  (mean ± std, n={n} datasets)", outpath)


def _plot_peak_length(datasets, workdir, outpath):
    """Grouped bar chart: mean peak length per mark per method, mean ± std across datasets."""
    data = _load_peak_frames(datasets, workdir)
    if data is None:
        print(f"  skipping {outpath}: no data")
        return
    n = len(datasets)
    _peak_bar(data, "mean_length", "Mean peak length (bp)",
              f"Mean peak length per mark and method  (mean ± std, n={n} datasets)", outpath)


# ---------------------------------------------------------------------------
# State coverage (total bp per state)
# ---------------------------------------------------------------------------

def _plot_state_coverage(datasets, cells, workdir, markups_dir, nstates, outfile, match_method="comb"):
    """Grouped bar chart: fraction of genome per chromatin state, method as hue, all datasets pooled."""
    from analyze import load_bed_df

    def _coverage(paths):
        """Return {state: total_bp} pooled across paths."""
        totals = {}
        for p in paths:
            p = Path(p)
            if not p.exists():
                print(f"  WARNING: missing {p}", file=sys.stderr)
                continue
            df = load_bed_df(str(p))[["state", "length"]]
            for state, bp in df.groupby("state")["length"].sum().items():
                totals[state] = totals.get(state, 0) + bp
        return totals

    # Collect per-dataset coverage fractions per method
    all_states = set()
    method_fracs = {}  # label -> list of {state: fraction} (one per dataset)

    for key, label, _ in INTER_DS_METHODS:
        per_ds = []
        if key == "reference":
            cov = _coverage(ref_beds(markups_dir))
            if cov:
                total = sum(cov.values())
                per_ds.append({s: bp / total for s, bp in cov.items()})
        else:
            for ds, cell in zip(datasets, cells):
                p = ds_method_bed(workdir, ds, cell, nstates, key, match_method)
                cov = _coverage([p])
                if cov:
                    total = sum(cov.values())
                    per_ds.append({s: bp / total for s, bp in cov.items()})
        if not per_ds:
            continue
        for d in per_ds:
            all_states.update(d.keys())
        method_fracs[label] = per_ds

    if not method_fracs:
        print(f"  skipping {outfile}: no data", file=sys.stderr)
        return

    states = sort_states(all_states)
    labels = [lbl for _, lbl, _ in INTER_DS_METHODS if lbl in method_fracs]
    colors = {lbl: col for _, lbl, col in INTER_DS_METHODS if lbl in method_fracs}

    # Build rows for seaborn: one row per (state, dataset, method)
    rows = []
    for label in labels:
        for d in method_fracs[label]:
            for s in states:
                rows.append({"State": s, "Method": label, "fraction": d.get(s, 0.0)})
    plot_df = pd.DataFrame(rows)

    n_methods = len(labels)
    figw = max(16, len(states) * n_methods * 0.22)

    # Broken y-axis: upper panel shows 0.40–1.0 (Quies/Het), lower shows 0–0.20
    BREAK_LOW  = 0.20   # top of lower panel
    BREAK_HIGH = 0.40   # bottom of upper panel

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        figsize=(figw, 6),
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.06},
    )

    for ax in (ax_top, ax_bot):
        sns.barplot(
            data=plot_df, x="State", y="fraction", hue="Method",
            hue_order=labels, palette=colors,
            estimator="mean", errorbar="sd",
            ax=ax, capsize=0.15, err_kws={"linewidth": 1.0},
            legend=(ax is ax_top),
        )

    ax_top.set_ylim(BREAK_HIGH, 1.02)
    ax_bot.set_ylim(0, BREAK_LOW)

    # Hide the inner spines to create the visual break
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False)

    # Draw diagonal break marks on both panels
    d = 0.012
    kwargs = dict(transform=fig.transFigure, color="k", clip_on=False, linewidth=0.8)
    # Positions in figure coordinates: left and right edges of the plot area
    for ax, sign in [(ax_top, -1), (ax_bot, 1)]:
        x0, x1 = ax.get_position().x0, ax.get_position().x1
        y  = ax.get_position().y0 if sign == 1 else ax.get_position().y1
        for x in (x0, x1):
            fig.add_artist(plt.Line2D([x - d, x + d], [y + sign * d * 1.5, y - sign * d * 1.5], **kwargs))

    for ax in (ax_top, ax_bot):
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.tick_params(axis="y", labelsize=8)

    ax_bot.tick_params(axis="x", labelsize=8, rotation=45)
    ax_bot.set_xlabel("Chromatin state", fontsize=9)
    ax_top.set_xlabel("")

    ax_top.legend(title="Method", fontsize=8, title_fontsize=9,
                  bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    if ax_bot.get_legend():
        ax_bot.get_legend().remove()

    ax_top.set_title(
        "Genomic coverage per chromatin state — ENCODE reference vs de-novo methods\n"
        f"(datasets: {', '.join(datasets)})",
        fontsize=10, fontweight="bold",
    )
    ax_top.set_ylabel("")
    ax_bot.set_ylabel("Fraction of genome", fontsize=9)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outfile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _plot_violin(datasets, cells, workdir, markups_dir, nstates, outfile, match_method="comb"):
    """Single violin panel: per-state segment length, method as hue, all datasets pooled."""
    print("Loading reference ...")
    frames = [load_reference_segments(markups_dir)]

    for key, label, _ in INTER_DS_METHODS:
        if key == "reference":
            continue
        paths = [str(ds_method_bed(workdir, ds, cell, nstates, key, match_method))
                 for ds, cell in zip(datasets, cells)]
        print(f"Loading {key} ...")
        sub = [f for f in (load_bed_segments(p) for p in paths) if not f.empty]
        if sub:
            df = pd.concat(sub, ignore_index=True)
            if len(df) > MAX_SEGS_PER_METHOD:
                df = df.sample(MAX_SEGS_PER_METHOD, random_state=42)
            df["method"] = label
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    present = [s for s in sort_states(combined["state"].unique())
               if (combined["state"] == s).sum() >= 5]
    combined = combined[combined["state"].isin(present)].copy()
    combined["state"] = pd.Categorical(combined["state"], categories=present, ordered=True)
    hue_order = [lbl for _, lbl, _ in INTER_DS_METHODS if lbl in combined["method"].unique()]

    fig_w = max(16, len(present) * len(INTER_DS_METHODS) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    sns.violinplot(
        data=combined, x="state", y="log_length",
        hue="method", hue_order=hue_order,
        palette=METHOD_PALETTE, order=present,
        linewidth=0.4, density_norm="width", inner="quartile", cut=0,
        ax=ax,
    )
    ax.set_ylim(1.8, 6)
    ax.set_ylabel("log₁₀(segment length + 1)  [bp]", fontsize=9)
    ax.set_xlabel("Chromatin state", fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=55)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(title="Method", fontsize=7, title_fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.set_title(
        "Segment length per chromatin state — ENCODE reference vs de-novo methods\n"
        f"(datasets: {', '.join(datasets)})",
        fontsize=10, fontweight="bold",
    )
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outfile}")


def _stacked_composition_chart(coverages, labels, title, outfile, label_fontsize=7):
    """Shared helper: stacked 100% bar chart from {label: {state: fraction}} dict."""
    all_states = set()
    for fracs in coverages.values():
        all_states.update(fracs.keys())
    all_states.discard("Unknown")
    states = sort_states(all_states)
    cmap = plt.get_cmap("tab20")
    state_colors = {s: STATE_COLORS.get(s, cmap(i % 20)) for i, s in enumerate(states)}

    figw = max(8, len(labels) * 0.8)
    fig, ax = plt.subplots(figsize=(figw, 5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for s in states:
        vals = np.array([coverages[lbl].get(s, 0.0) for lbl in labels])
        ax.bar(x, vals, bottom=bottom, label=s, color=state_colors[s],
               edgecolor="white", linewidth=0.3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_fontsize, rotation=45, ha="right")
    ax.set_ylabel("Fraction of genome", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(title="State", fontsize=7, title_fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0, ncol=1)
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outfile}")


def _plot_reference_composition(markups_dir, outfile):
    """Stacked bar chart: state fraction per ENCODE reference segmentation."""
    from analyze import load_bed_df
    beds = ref_beds(markups_dir)
    if not beds:
        print(f"  skipping {outfile}: no reference beds", file=sys.stderr)
        return

    coverages = {}
    labels = []
    for path in beds:
        name = Path(path).name.replace(".bed.gz", "").replace(".bed", "")
        label = "_".join(name.split("_")[1:]).replace("_", " ")
        df = load_bed_df(path)[["state", "length"]]
        totals = df.groupby("state")["length"].sum()
        total_bp = totals.sum()
        coverages[label] = (totals / total_bp).to_dict() if total_bp > 0 else {}
        labels.append(label)

    _stacked_composition_chart(
        coverages, labels,
        f"State composition across ENCODE reference segmentations ({len(labels)} cell types)",
        outfile, label_fontsize=7,
    )


def _plot_method_composition(datasets, cells, workdir, markups_dir, nstates, outfile, match_method="comb"):
    """Stacked bar chart: mean state fraction per method, averaged across datasets."""
    from analyze import load_bed_df

    def _fracs_from_path(path):
        p = Path(path)
        if not p.exists():
            return {}
        df = load_bed_df(str(p))[["state", "length"]]
        totals = df.groupby("state")["length"].sum()
        total_bp = totals.sum()
        return (totals / total_bp).to_dict() if total_bp > 0 else {}

    coverages = {}
    labels_out = []
    for key, label, _ in INTER_DS_METHODS:
        if key == "reference":
            beds = ref_beds(markups_dir)
            per_ds = [_fracs_from_path(b) for b in beds]
        else:
            per_ds = [
                _fracs_from_path(ds_method_bed(workdir, ds, cell, nstates, key, match_method))
                for ds, cell in zip(datasets, cells)
            ]
        per_ds = [d for d in per_ds if d]
        if not per_ds:
            continue
        all_states = set().union(*[d.keys() for d in per_ds])
        mean_fracs = {
            s: float(np.mean([d.get(s, 0.0) for d in per_ds]))
            for s in all_states
        }
        coverages[label] = mean_fracs
        labels_out.append(label)

    _stacked_composition_chart(
        coverages, labels_out,
        "State composition per method — mean across datasets",
        outfile, label_fontsize=9,
    )


def _plot_per_dataset_method_composition(datasets, cells, workdir, nstates,
                                         method_key, method_label, outfile, match_method="comb"):
    """Stacked bar chart: state fraction per dataset for a single method."""
    from analyze import load_bed_df

    def _fracs_from_path(path):
        p = Path(path)
        if not p.exists():
            return {}
        df = load_bed_df(str(p))[["state", "length"]]
        totals = df.groupby("state")["length"].sum()
        total_bp = totals.sum()
        return (totals / total_bp).to_dict() if total_bp > 0 else {}

    coverages = {}
    labels_out = []
    for ds, cell in zip(datasets, cells):
        bed = ds_method_bed(workdir, ds, cell, nstates, method_key, match_method)
        fracs = _fracs_from_path(bed)
        if fracs:
            coverages[ds] = fracs
            labels_out.append(ds)

    if not coverages:
        print(f"  skipping {outfile}: no data for method {method_key}", file=sys.stderr)
        return

    _stacked_composition_chart(
        coverages, labels_out,
        f"State composition — {method_label} (per dataset)",
        outfile, label_fontsize=9,
    )


def _plot_method_similarity_distribution(inter_ds_dir, methods, outfile, noqh=False,
                                         group_a=None, group_b=None):
    """Violin plot: pairwise similarity distributions per de-novo method and metric.

    For each method the upper triangle of the kappa/Jaccard matrix is extracted
    (one value per dataset pair) and drawn as a violin, grouped by metric.

    If group_a and group_b are provided (sets of dataset name prefixes), only pairs
    where one dataset is in group_a and the other is in group_b are included.
    """
    suffix = "_noqh" if noqh else ""
    metric_configs = [
        ("Composition", f"composition{suffix}_similarity_matrix.tsv",                 "#E8833A"),
        ("Kappa",   f"kappa{suffix}_matrix.tsv",                                        "#4878CF"),
        ("Jaccard", f"jaccard_noqh_matrix.tsv" if noqh else "jaccard_similarity_matrix.tsv",
                    "#2CA02C"),
    ]

    def _ds_name(label):
        """Extract dataset name prefix from a matrix index label like 'imr90:method'."""
        return label.split(":")[0]

    rows = []
    for method in methods:
        label = _SAMPLE_TO_INFO.get(method, (method,))[0]
        for metric, filename, _ in metric_configs:
            path = os.path.join(inter_ds_dir, method, filename)
            if not os.path.exists(path):
                print(f"  WARNING: missing {path}", file=sys.stderr)
                continue
            mat = pd.read_csv(path, sep="\t", index_col=0)
            n = len(mat)
            for i in range(n):
                for j in range(i + 1, n):
                    if group_a is not None and group_b is not None:
                        ds_i = _ds_name(mat.index[i])
                        ds_j = _ds_name(mat.index[j])
                        if not ((ds_i in group_a and ds_j in group_b) or
                                (ds_i in group_b and ds_j in group_a)):
                            continue
                    rows.append({"Method": label, "Metric": metric, "value": float(mat.iloc[i, j])})

    if not rows:
        print(f"  skipping {outfile}: no data", file=sys.stderr)
        return

    plot_df = pd.DataFrame(rows)
    method_labels = [_SAMPLE_TO_INFO.get(m, (m,))[0] for m in methods
                     if _SAMPLE_TO_INFO.get(m, (m,))[0] in plot_df["Method"].unique()]
    palette = {"Composition": "#E8833A", "Kappa": "#4878CF", "Jaccard": "#2CA02C"}

    n_methods = len(method_labels)
    fig, ax = plt.subplots(figsize=(max(10, n_methods * 1.4 + 3), 5))
    sns.violinplot(
        data=plot_df, x="Method", y="value", hue="Metric",
        order=method_labels,
        hue_order=["Composition", "Kappa", "Jaccard"],
        palette=palette,
        inner="box", cut=0, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Pairwise similarity", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    mode = "NOQH (excl. Quies/Het)" if noqh else "Full"
    n_pairs = plot_df[plot_df["Metric"] == "Composition"]["Method"].count() // max(n_methods, 1)
    pair_desc = " — ChIP↔Mint-ChIP pairs only" if (group_a and group_b) else ""
    ax.set_title(
        f"Inter-dataset similarity by method — {mode}{pair_desc}\n"
        f"({n_methods} methods, {n_pairs} dataset pairs each)",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(title="Metric", fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outfile}")


def _plot_reference_distribution(comp_path, kappa_path, jaccard_path, outfile,
                                  title_suffix=""):
    """Violin plot of pairwise similarity among ENCODE reference segmentations."""
    metrics = [
        ("Composition", comp_path),
        ("Kappa",   kappa_path),
        ("Jaccard", jaccard_path),
    ]
    rows = []
    for metric, path in metrics:
        mat = pd.read_csv(path, sep="\t", index_col=0)
        n = len(mat)
        for i in range(n):
            for j in range(i + 1, n):
                rows.append({"Metric": metric, "value": float(mat.iloc[i, j])})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(data=plot_df, x="Metric", y="value", hue="Metric",
                   order=["Composition", "Kappa", "Jaccard"],
                   inner="box", cut=0, ax=ax, legend=False,
                   palette={"Composition": "#E8833A", "Kappa": "#4878CF", "Jaccard": "#2CA02C"})
    ax.set_xlabel("")
    ax.set_ylabel("Pairwise similarity", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    n_refs_pairs = plot_df[plot_df["Metric"] == "Composition"]["value"].count()
    title = (
        f"Inter-reference similarity distribution{title_suffix}\n"
        f"({int((-1 + (1 + 8 * n_refs_pairs) ** 0.5) / 2 + 1)} ENCODE references, {n_refs_pairs} pairs each metric)"
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outfile}")


def plot_reference_n_segments(datasets, methods_dirs, labels, outfile, title):
    """Bar chart: number of segments of each dataset's own ENCODE reference.

    One grey bar per dataset (the "ref" row of its comparison_table). Used to
    show the per-dataset ENCODE references split by assay (ChIP-seq vs Mint-ChIP).
    """
    vals, labs = [], []
    for ds, mdir, lab in zip(datasets, methods_dirs, labels):
        table = _load_comparison_table(mdir)
        if table is None:
            continue
        row = table[table["method"] == "ref"]
        if row.empty or "n_segments" not in row.columns:
            continue
        try:
            vals.append(float(row["n_segments"].iloc[0]) / 1000.0)
            labs.append(lab)
        except (TypeError, ValueError):
            continue
    if not vals:
        print(f"  skipping {outfile}: no reference data")
        return
    fig, ax = plt.subplots(figsize=(max(4, len(labs) * 1.2), 4.2))
    x = np.arange(len(labs))
    ax.bar(x, vals, color=BIN_COLORS["reference"], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Segments (×10³)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, v in enumerate(vals):
        ax.text(i, v + yr * 0.01, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outfile}")


def run_summary_plots(datasets=None, methods_dirs=None, analysis_dirs=None,
                      outdir=None, workdir=None, markups_dir=None, cells=None,
                      methods=None, nstates=15, match_method="comb",
                      violin_outfile=None, state_coverage_outfile=None,
                      peak_count_outfile=None, peak_length_outfile=None,
                      peak_gap_violin_outfile=None,
                      ref_composition_outfile=None,
                      ref_kappa_matrix=None, ref_jaccard_matrix=None,
                      ref_dist_outfile=None,
                      ref_kappa_noqh_matrix=None, ref_jaccard_noqh_matrix=None,
                      ref_dist_noqh_outfile=None,
                      method_composition_outfile=None,
                      method_ds_composition_outdir=None,
                      rep_consistency_outdir=None,
                      method_sim_dist_indir=None, method_sim_dist_methods=None,
                      method_sim_dist_outfile=None,
                      method_sim_dist_noqh_outfile=None,
                      method_sim_dist_group_a=None, method_sim_dist_group_b=None,
                      method_sim_dist_filtered_outfile=None,
                      method_sim_dist_filtered_noqh_outfile=None):
    """Cross-dataset summary bar plots, violins and similarity distributions.

    Direct-call entry point (the former CLI). Each *_outfile / *_outdir argument
    that is set selects one plot group to produce, mirroring the old out
    Snakemake rules. Called from analysis.ipynb.
    """
    args = SimpleNamespace(
        datasets=datasets or [], methods_dirs=methods_dirs or [],
        analysis_dirs=analysis_dirs or [], outdir=outdir, workdir=workdir,
        markups_dir=markups_dir, cells=cells or [], methods=methods or [],
        nstates=nstates, match_method=match_method,
        violin_outfile=violin_outfile, state_coverage_outfile=state_coverage_outfile,
        peak_count_outfile=peak_count_outfile, peak_length_outfile=peak_length_outfile,
        peak_gap_violin_outfile=peak_gap_violin_outfile,
        ref_composition_outfile=ref_composition_outfile,
        ref_kappa_matrix=ref_kappa_matrix, ref_jaccard_matrix=ref_jaccard_matrix,
        ref_dist_outfile=ref_dist_outfile,
        ref_kappa_noqh_matrix=ref_kappa_noqh_matrix,
        ref_jaccard_noqh_matrix=ref_jaccard_noqh_matrix,
        ref_dist_noqh_outfile=ref_dist_noqh_outfile,
        method_composition_outfile=method_composition_outfile,
        method_ds_composition_outdir=method_ds_composition_outdir,
        rep_consistency_outdir=rep_consistency_outdir,
        method_sim_dist_indir=method_sim_dist_indir,
        method_sim_dist_methods=method_sim_dist_methods or [],
        method_sim_dist_outfile=method_sim_dist_outfile,
        method_sim_dist_noqh_outfile=method_sim_dist_noqh_outfile,
        method_sim_dist_group_a=method_sim_dist_group_a,
        method_sim_dist_group_b=method_sim_dist_group_b,
        method_sim_dist_filtered_outfile=method_sim_dist_filtered_outfile,
        method_sim_dist_filtered_noqh_outfile=method_sim_dist_filtered_noqh_outfile)

    if args.methods:
        # Update global method lists/palettes to the requested subset.
        global INTER_DS_METHODS, METHOD_PALETTE, METHODS_POOLED
        methods_map = {m[0]: m for m in INTER_DS_METHODS}
        new_methods = []
        # Always keep the ENCODE reference on state-level plots (coverage,
        # composition, violin) — their titles read "ENCODE reference vs de-novo
        # methods", and the bar/table plots key the reference off "ref" instead,
        # so a stray "reference" entry here is harmless to them.
        if "reference" not in args.methods and "reference" in methods_map:
            new_methods.append(methods_map["reference"])
        for k in args.methods:
            if k in methods_map:
                new_methods.append(methods_map[k])
            else:
                new_methods.append((k, k.replace("_", " ").title(), "#808080"))
        INTER_DS_METHODS = new_methods
        METHOD_PALETTE = {label: color for _, label, color in INTER_DS_METHODS}
        METHODS_POOLED = [m[0] for m in INTER_DS_METHODS]

    # --- summary bar plots -----------------------------------------------
    if args.outdir:
        if not (len(args.datasets) == len(args.methods_dirs) == len(args.analysis_dirs)):
            raise ValueError("--datasets, --methods-dirs and --analysis-dirs must have equal lengths")
        os.makedirs(args.outdir, exist_ok=True)
        ds, mdirs, adirs = args.datasets, args.methods_dirs, args.analysis_dirs

        data = _collect_table_col(ds, mdirs, "entropy", include_ref=True)
        _plot_summary(data, "Transition matrix entropy (full)",
                      "Entropy (bits)",
                      os.path.join(args.outdir, "summary_entropy.png"),
                      order=["ref"] + METHODS_POOLED)

        data = _collect_table_col(ds, mdirs, "entropy_noqh", include_ref=True)
        _plot_summary(data, "Transition matrix entropy (NOQH, excl. Quies/Het)",
                      "Entropy (bits)",
                      os.path.join(args.outdir, "summary_entropy_noqh.png"),
                      order=["ref"] + METHODS_POOLED)

        data = _collect_table_col(ds, mdirs, "jaccard_Tx_ExpressedGeneBodies")
        _plot_summary(data, "Jaccard: Tx state vs expressed gene bodies", "Jaccard",
                      os.path.join(args.outdir, "summary_jaccard_tx.png"),
                      partial_note=True)

        data = _collect_table_col(ds, mdirs, "enrich_Tx_ExpressedGeneBodies")
        _plot_summary(data, "Tx enrichment at expressed gene bodies", "Fold enrichment",
                      os.path.join(args.outdir, "summary_enrich_tx.png"),
                      partial_note=True)

        data = _collect_table_col(ds, mdirs, "median_Tx_length", include_ref=True)
        _plot_summary(data, "Median Tx (transcription) segment length", "bp",
                      os.path.join(args.outdir, "summary_median_tx_length.png"),
                      order=["ref"] + METHODS_POOLED)

        data = _collect_table_col(ds, mdirs, "mean_Tx_length", include_ref=True)
        _plot_summary(data, "Mean Tx (transcription) segment length", "bp",
                      os.path.join(args.outdir, "summary_mean_tx_length.png"),
                      order=["ref"] + METHODS_POOLED)

        data = _collect_jaccard(ds, adirs, "Tss", "RefSeqTSS2kb.hg38")
        _plot_summary(data, "Jaccard: Tss state vs RefSeq TSS ±1 kb", "Jaccard",
                      os.path.join(args.outdir, "summary_jaccard_tss.png"))

        data = _collect_jaccard_atac(ds, adirs, "Tss")
        _plot_summary(data, "Jaccard: Tss state vs ATAC-seq", "Jaccard",
                      os.path.join(args.outdir, "summary_jaccard_tss_atac.png"),
                      partial_note=True)

        # Total number of segments, including the ENCODE reference.
        data = _collect_table_col(ds, mdirs, "n_segments", include_ref=True) / 1000.0
        _plot_summary(data, "Total number of segments", "Segments (×10³)",
                      os.path.join(args.outdir, "summary_n_segments.png"),
                      order=["ref"] + METHODS_POOLED)

        # Similarity vs ENCODE reference (3 variants)
        _plot_summary(_collect_table_col(ds, mdirs, "kappa_vs_ref"),
                      "Agreement vs ENCODE reference (Kappa)", "Cohen's Kappa",
                      os.path.join(args.outdir, "summary_kappa_vs_ref.png"),
                      order=METHODS_POOLED)

        _plot_summary(_collect_table_col(ds, mdirs, "jaccard_vs_ref"),
                      "Agreement vs ENCODE reference (Jaccard)", "Mean per-state Jaccard",
                      os.path.join(args.outdir, "summary_jaccard_vs_ref.png"),
                      order=METHODS_POOLED)

        _plot_summary(_collect_table_col(ds, mdirs, "composition_vs_ref"),
                      "State composition similarity vs ENCODE reference", "Cosine similarity",
                      os.path.join(args.outdir, "summary_composition_similarity.png"),
                      order=METHODS_POOLED)

        _plot_per_state_kappa(ds, adirs, args.outdir, match_method=args.match_method)

    if args.violin_outfile or args.state_coverage_outfile:
        if not (args.workdir and args.markups_dir and args.cells):
            raise ValueError("--workdir, --markups-dir and --cells are required for violin/coverage plots")
        if len(args.datasets) != len(args.cells):
            raise ValueError("--datasets and --cells must have equal lengths")

        if args.violin_outfile:
            os.makedirs(os.path.dirname(os.path.abspath(args.violin_outfile)), exist_ok=True)
            _plot_violin(args.datasets, args.cells, args.workdir, args.markups_dir,
                         args.nstates, args.violin_outfile, args.match_method)
        if args.state_coverage_outfile:
            os.makedirs(os.path.dirname(os.path.abspath(args.state_coverage_outfile)), exist_ok=True)
            _plot_state_coverage(args.datasets, args.cells, args.workdir, args.markups_dir,
                                 args.nstates, args.state_coverage_outfile, args.match_method)

    # --- peak count bar chart ----------------------------------------------
    if args.peak_count_outfile:
        if not args.workdir:
            raise ValueError("--workdir is required for --peak-count-outfile")
        os.makedirs(os.path.dirname(os.path.abspath(args.peak_count_outfile)), exist_ok=True)
        _plot_peak_count(args.datasets, args.workdir, args.peak_count_outfile)

    # --- mean peak length bar chart ----------------------------------------
    if args.peak_length_outfile:
        if not args.workdir:
            raise ValueError("--workdir is required for --peak-length-outfile")
        os.makedirs(os.path.dirname(os.path.abspath(args.peak_length_outfile)), exist_ok=True)
        _plot_peak_length(args.datasets, args.workdir, args.peak_length_outfile)

    # --- gap-length violin -------------------------------------------------
    if args.peak_gap_violin_outfile:
        if not args.workdir:
            raise ValueError("--workdir is required for --peak-gap-violin-outfile")
        _plot_peak_gap_violin(args.datasets, args.workdir, args.peak_gap_violin_outfile)

    # --- inter-reference state composition ----------------------------------
    if args.ref_composition_outfile:
        if not args.markups_dir:
            raise ValueError("--markups-dir is required for --ref-composition-outfile")
        os.makedirs(os.path.dirname(os.path.abspath(args.ref_composition_outfile)), exist_ok=True)
        _plot_reference_composition(args.markups_dir, args.ref_composition_outfile)

    # --- inter-reference similarity distribution ----------------------------
    if args.ref_dist_outfile:
        if not (args.ref_comp_matrix and args.ref_kappa_matrix and args.ref_jaccard_matrix):
            raise ValueError("--ref-comp-matrix, --ref-kappa-matrix and --ref-jaccard-matrix are required "
                     "for --ref-dist-outfile")
        os.makedirs(os.path.dirname(os.path.abspath(args.ref_dist_outfile)), exist_ok=True)
        _plot_reference_distribution(args.ref_comp_matrix,
                                     args.ref_kappa_matrix,
                                     args.ref_jaccard_matrix, args.ref_dist_outfile,
                                     title_suffix=" — Full")

    if args.ref_dist_noqh_outfile:
        if not (args.ref_comp_noqh_matrix and args.ref_kappa_noqh_matrix and args.ref_jaccard_noqh_matrix):
            raise ValueError("--ref-comp-noqh-matrix, --ref-kappa-noqh-matrix and "
                     "--ref-jaccard-noqh-matrix are required for --ref-dist-noqh-outfile")
        os.makedirs(os.path.dirname(os.path.abspath(args.ref_dist_noqh_outfile)), exist_ok=True)
        _plot_reference_distribution(args.ref_comp_noqh_matrix,
                                     args.ref_kappa_noqh_matrix,
                                     args.ref_jaccard_noqh_matrix, args.ref_dist_noqh_outfile,
                                     title_suffix=" — NOQH (excl. Quies/Het)")

    # --- inter-method similarity distribution --------------------------------
    if args.method_sim_dist_outfile or args.method_sim_dist_noqh_outfile:
        if not (args.method_sim_dist_indir and args.method_sim_dist_methods):
            raise ValueError("--method-sim-dist-indir and --method-sim-dist-methods are required "
                     "for --method-sim-dist-outfile / --method-sim-dist-noqh-outfile")
        if args.method_sim_dist_outfile:
            os.makedirs(os.path.dirname(os.path.abspath(args.method_sim_dist_outfile)), exist_ok=True)
            _plot_method_similarity_distribution(
                args.method_sim_dist_indir, args.method_sim_dist_methods,
                args.method_sim_dist_outfile, noqh=False,
            )
        if args.method_sim_dist_noqh_outfile:
            os.makedirs(os.path.dirname(os.path.abspath(args.method_sim_dist_noqh_outfile)), exist_ok=True)
            _plot_method_similarity_distribution(
                args.method_sim_dist_indir, args.method_sim_dist_methods,
                args.method_sim_dist_noqh_outfile, noqh=True,
            )

    # --- cross-group filtered similarity distribution -------------------------
    if args.method_sim_dist_filtered_outfile or args.method_sim_dist_filtered_noqh_outfile:
        if not (args.method_sim_dist_indir and args.method_sim_dist_methods):
            raise ValueError("--method-sim-dist-indir and --method-sim-dist-methods are required "
                     "for --method-sim-dist-filtered-outfile / --method-sim-dist-filtered-noqh-outfile")
        if not (args.method_sim_dist_group_a and args.method_sim_dist_group_b):
            print("  WARNING: skipping filtered similarity distribution: "
                  "both --method-sim-dist-group-a and --method-sim-dist-group-b "
                  "must be non-empty", file=sys.stderr)
        else:
            ga = set(args.method_sim_dist_group_a)
            gb = set(args.method_sim_dist_group_b)
            if args.method_sim_dist_filtered_outfile:
                os.makedirs(os.path.dirname(os.path.abspath(args.method_sim_dist_filtered_outfile)), exist_ok=True)
                _plot_method_similarity_distribution(
                    args.method_sim_dist_indir, args.method_sim_dist_methods,
                    args.method_sim_dist_filtered_outfile, noqh=False,
                    group_a=ga, group_b=gb,
                )
            if args.method_sim_dist_filtered_noqh_outfile:
                os.makedirs(os.path.dirname(os.path.abspath(args.method_sim_dist_filtered_noqh_outfile)), exist_ok=True)
                _plot_method_similarity_distribution(
                    args.method_sim_dist_indir, args.method_sim_dist_methods,
                    args.method_sim_dist_filtered_noqh_outfile, noqh=True,
                    group_a=ga, group_b=gb,
                )

    # --- replicate consistency plots ----------------------------------------
    if args.rep_consistency_outdir:
        if len(args.datasets) != len(args.methods_dirs):
            raise ValueError("--datasets and --methods-dirs must have equal lengths")
        _plot_rep_consistency(args.datasets, args.methods_dirs, args.rep_consistency_outdir)

    # --- method state composition -------------------------------------------
    if args.method_composition_outfile:
        if not args.markups_dir:
            raise ValueError("--markups-dir is required for --method-composition-outfile")
        os.makedirs(os.path.dirname(os.path.abspath(args.method_composition_outfile)), exist_ok=True)
        _plot_method_composition(args.datasets, args.cells, args.workdir, args.markups_dir,
                                 args.nstates, args.method_composition_outfile, args.match_method)

    # --- per-dataset method state composition (4 supplementary plots) -------
    if args.method_ds_composition_outdir:
        if not (args.workdir and args.cells and args.datasets):
            raise ValueError("--workdir, --datasets and --cells are required for "
                     "--method-ds-composition-outdir")
        if len(args.datasets) != len(args.cells):
            raise ValueError("--datasets and --cells must have equal lengths")
        os.makedirs(args.method_ds_composition_outdir, exist_ok=True)
        _supp_methods = [
            ("chromhmm_default", "Default ChromHMM"),
            ("chromhmm_omni",    "ChromHMM OmniPeak"),
            ("kmeans_omni",      "KMeans OmniPeak"),
            ("chromhmm_homer",   "ChromHMM HOMER"),
            ("kmeans_homer",     "KMeans HOMER"),
            ("chromhmm_macs2",   "ChromHMM MACS2"),
            ("kmeans_macs2",     "KMeans MACS2"),
        ]
        for method_key, method_label in _supp_methods:
            outfile = os.path.join(args.method_ds_composition_outdir,
                                   f"method_ds_composition_{method_key}.png")
            _plot_per_dataset_method_composition(
                args.datasets, args.cells, args.workdir, args.nstates,
                method_key, method_label, outfile, args.match_method,
            )
