#!/usr/bin/env python3
"""Cross-dataset summary bar plots with mean ± std error bars and individual points.

Reads the per-dataset comparison_table.tsv and per-method jaccard.tsv files and
produces one grouped bar chart per metric across all datasets.

Drive it from analysis.ipynb via run_summary_plots(); each *_outfile / *_outdir
argument that is set selects one plot group.
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
from utils import (METHOD_ORDER, DISPLAY_NAMES, BIN_COLORS, METHOD_INFO,
                   strip_points, method_color, save_fig)
from analyze import load_bed_df

METHODS_POOLED = [m for m in METHOD_ORDER
                  if not m.endswith("_rep1") and not m.endswith("_rep2")]


# Canonical ENCODE 15-state order.
STATE_ORDER = [
    "TssA", "Tss", "TssAFlnk", "TssFlnk", "TssFlnkU", "TssFlnkD", "TxFlnk",
    "Tx", "TxWk",
    "EnhG", "EnhG1", "EnhG2",
    "Enh", "Enh1", "Enh2", "EnhLo",
    "ZNF/Rpts", "Het",
    "TssBiv", "BivFlnk", "EnhBiv", "Biv",
    "ReprPC", "ReprPCWk",
    "Quies",
]
STATE_IDX = {s: i for i, s in enumerate(STATE_ORDER)}

# Canonical state colors, from BED column 9 of the ENCODE references.
_RGB = lambda r, g, b: (r / 255, g / 255, b / 255)
STATE_COLORS = {
    "TssA":     _RGB(255,   0,   0),
    "Tss":      _RGB(255,   0,   0),
    "TssAFlnk": _RGB(255,  69,   0),
    "TssFlnk":  _RGB(255,  69,   0),
    "TssFlnkU": _RGB(255,  69,   0),
    "TssFlnkD": _RGB(255,  69,   0),
    "TxFlnk":   _RGB( 50, 205,  50),
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
    "TssBiv":   _RGB(205,  92,  92),
    "BivFlnk":  _RGB(233, 150, 122),
    "EnhBiv":   _RGB(189, 183, 107),
    "Biv":      _RGB(205,  92,  92),
    "ReprPC":   _RGB(137,  55, 223),
    "ReprPCWk": _RGB(137,  55, 223),
    "Quies":    _RGB(220, 220, 220),
}

# (METHOD_ORDER key, display label, hex color)
INTER_DS_METHODS = [
    ("ref",              "ENCODE Ref",      BIN_COLORS["reference"]),
    ("chromhmm_default", "Default ChromHMM", BIN_COLORS["default"]),
    ("chromhmm_homer",   "ChromHMM HOMER",     BIN_COLORS["homer"]),
    ("kmeans_homer",     "KMeans HOMER",       BIN_COLORS["homer"]),
    ("chromhmm_macs2",   "ChromHMM MACS2",     BIN_COLORS["macs2"]),
    ("kmeans_macs2",     "KMeans MACS2",       BIN_COLORS["macs2"]),
    ("chromhmm_omni",    "ChromHMM OmniPeak",  BIN_COLORS["omnipeak"]),
    ("kmeans_omni",      "KMeans OmniPeak",    BIN_COLORS["omnipeak"]),
]
METHOD_PALETTE = {label: color for _, label, color in INTER_DS_METHODS}


def sort_states(states):
    """Sort chromatin state names in canonical ENCODE order.

    Numbered names ("1_TssA", "10_TssBiv") are not in STATE_IDX and fall to the
    end, ordered by state number rather than alphabetically.
    """
    def key(s):
        prefix = s.split("_")[0]
        return (STATE_IDX.get(s, 999),
                int(prefix) if prefix.isdigit() else 999,
                s)
    return sorted(states, key=key)


def ds_method_bed(workdir, ds, cell, nstates, method_key, match_method):
    """Return Path for a single (dataset, method) {match_method}_matched BED."""
    root = Path(workdir) / ds
    sfx = match_method if match_method.endswith("matched") else f"{match_method}_matched"
    mapping = {
        "chromhmm_default": root / "chromhmm_default_result" / f"{cell}_{nstates}_dense_{sfx}.bed",
        "kmeans_omni":      root / "omni"  / f"omni_kmeans_states_{sfx}.bed",
        "kmeans_homer":     root / "homer" / f"homer_kmeans_states_{sfx}.bed",
        "kmeans_macs2":     root / "macs2" / f"macs2_kmeans_states_{sfx}.bed",
        "chromhmm_omni":    root / "omni"  / f"omni_chromhmm_states_{sfx}.bed",
        "chromhmm_homer":   root / "homer" / f"homer_chromhmm_states_{sfx}.bed",
        "chromhmm_macs2":   root / "macs2" / f"macs2_chromhmm_states_{sfx}.bed",
    }
    return mapping[method_key]


def ref_beds(markups_dir):
    """Sorted ENCODE reference BED paths from markups/15state/."""
    markups_path = Path(markups_dir) / "15state"
    files = sorted(markups_path.glob("*.bed.gz")) + sorted(markups_path.glob("*.bed"))
    if not files:
        print(f"  WARNING: no reference BED files in {markups_path}", file=sys.stderr)
    return [str(f) for f in files]


def _load_comparison_table(methods_dir):
    """Pooled-only rows from {methods_dir}/comparison_table.tsv, or None."""
    path = os.path.join(methods_dir, "comparison_table.tsv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    return df[df["replicate"].fillna("") == ""].copy()


def _collect_table_col(datasets, methods_dirs, col, include_ref=False):
    """One comparison_table column across datasets, as index=method,
    columns=dataset.

    include_ref keeps the ENCODE reference row, which is otherwise excluded
    because "ref" is not among the de-novo METHODS_POOLED.
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
    return pd.DataFrame(records).T


def _plot_summary(data, title, ylabel, outpath, partial_note=False, order=None):
    """Grouped bar chart: mean ± std across datasets.

    order: explicit method-key order, defaults to METHODS_POOLED.
    """
    method_order = order if order is not None else METHODS_POOLED
    methods = [m for m in method_order if m in data.index]
    if not methods:
        print(f"  skipping {outpath}: no data")
        return

    methods = [m for m in methods
               if not data.loc[m].isna().all()]

    display   = [DISPLAY_NAMES.get(m, m) for m in methods]
    means     = data.loc[methods].mean(axis=1, skipna=True).values
    stds      = data.loc[methods].std(axis=1, skipna=True).values
    counts    = data.loc[methods].count(axis=1).values
    ses       = stds / np.sqrt(counts)

    # A placeholder rather than a skip, so the output file always exists.
    if np.all(np.isnan(means)):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="grey")
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, fontweight="bold")
        save_fig(fig, outpath, tight=False, note="(no data)")
        return

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 0.8), 4.2))

    df_melted = data.loc[methods].reset_index().melt(id_vars='index', var_name='dataset', value_name='value')
    df_melted = df_melted.rename(columns={'index': 'method'})
    df_melted['display_name'] = df_melted['method'].map(lambda m: DISPLAY_NAMES.get(m, m))
    palette = {DISPLAY_NAMES.get(m, m): method_color(m) for m in methods}
    display_order = [DISPLAY_NAMES.get(m, m) for m in methods]

    sns.barplot(data=df_melted, x="display_name", y="value", order=display_order,
                palette=palette, hue="display_name", dodge=False,
                capsize=0.05, errorbar="se", err_kws={"linewidth": 2.0},
                ax=ax, edgecolor="lightgrey", linewidth=1)

    strip_points(ax, data=df_melted, x="display_name", y="value",
                 order=display_order, dodge=False, size=2)

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    # Only datasets with data for the plotted methods count, so RNA-seq/ATAC-seq
    # plots report the datasets where that information was available.
    n_ds = data.loc[methods].notna().any(axis=0).sum()
    for i, (m, se, c) in enumerate(zip(means, ses, counts)):
        if np.isnan(m) or c == 0:
            continue
        top = m + (se if not np.isnan(se) and c > 1 else 0)
        # Keep the label clear of the individual points as well.
        vmax = data.loc[methods[i]].max(skipna=True)
        if not pd.isna(vmax):
            top = max(top, float(vmax))
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
    plotted_bins = {METHOD_INFO.get(m, (None,))[0] for m in methods}
    active_binarizations = [b for b in ["reference", "default", "omnipeak", "homer", "macs2"]
                            if b in plotted_bins]

    for b in active_binarizations:
        legend_elements.append(Patch(facecolor=BIN_COLORS[b], label=bin_labels[b]))

    if legend_elements:
        ax.legend(handles=legend_elements, fontsize=6,
                  bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    n_note = f"mean ± SE across {n_ds} datasets (points: individual datasets)"
    if partial_note:
        n_note += " (n = datasets with data)"
    ax.set_xlabel(n_note, fontsize=7, color="grey")

    save_fig(fig, outpath)


def _plot_2way_scatter(x_data, y_data, title, xlabel, ylabel, outpath, order=None):
    """Scatter plot: X vs Y (mean ± SE) across datasets, from DataFrames with
    index=method, columns=dataset.
    """
    method_order = order if order is not None else METHODS_POOLED
    methods = [m for m in method_order if m in x_data.index and m in y_data.index]
    if not methods:
        print(f"  skipping {outpath}: no data")
        return

    methods = [m for m in methods
               if not x_data.loc[m].isna().all() and not y_data.loc[m].isna().all()]

    if not methods:
        print(f"  skipping {outpath}: no data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for m in methods:
        color = method_color(m)
        label = DISPLAY_NAMES.get(m, m)

        ds_common = x_data.columns.intersection(y_data.columns)
        x_vals = x_data.loc[m, ds_common].astype(float)
        y_vals = y_data.loc[m, ds_common].astype(float)

        mask = x_vals.notna() & y_vals.notna()
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

        if len(x_vals) == 0:
            continue

        x_mean = x_vals.mean()
        y_mean = y_vals.mean()
        count = len(x_vals)
        x_se = x_vals.std() / np.sqrt(count) if count > 1 else 0
        y_se = y_vals.std() / np.sqrt(count) if count > 1 else 0

        ax.scatter(x_vals, y_vals, color=color, alpha=0.2, s=20, edgecolors='none')
        ax.errorbar(x_mean, y_mean, xerr=x_se, yerr=y_se, fmt='o',
                    color=color, label=label, markersize=7, markeredgecolor='white', markeredgewidth=1)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    save_fig(fig, outpath, tight=False)


def _collect_per_state_metrics(datasets, analysis_dirs, match_method):
    """Collect per-state kappa and jaccard vs reference for all datasets."""
    all_stacked = []
    for ds, adir in zip(datasets, analysis_dirs):
        comp_dirs = [
            os.path.join(ds, "comparison", match_method),
            os.path.join(adir, "comparison", match_method),
            os.path.join(adir, "comparison"),
            adir
        ]
        
        comp_dir = None
        for d in comp_dirs:
            if os.path.isdir(d):
                comp_dir = d
                break
        
        if not comp_dir:
            continue

        metrics_path = os.path.join(comp_dir, "per_state_metrics.tsv")
        if os.path.exists(metrics_path):
            try:
                df = pd.read_csv(metrics_path, sep="\t")
                if not df.empty:
                    df["dataset"] = ds
                    all_stacked.append(df)
                    continue
            except Exception as e:
                print(f"  WARNING: could not load {metrics_path}: {e}", file=sys.stderr)

        # Fall back to the per-state kappa/jaccard TSVs.
        for metric in ["kappa", "jaccard"]:
            try:
                tsv_files = [f for f in os.listdir(comp_dir)
                             if f.startswith(f"per_state_{metric}_vs_") and f.endswith(".tsv")]
                if tsv_files:
                    tsv_path = os.path.join(comp_dir, tsv_files[0])
                    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
                    if not df.empty:
                        stacked = df.stack().reset_index()
                        stacked.columns = ["state", "method", metric]
                        stacked = stacked.dropna(subset=[metric])
                        stacked["dataset"] = ds
                        all_stacked.append(stacked)
            except Exception as e:
                print(f"  WARNING: could not load legacy {metric} TSV in {comp_dir}: {e}", file=sys.stderr)

    if not all_stacked:
        return pd.DataFrame()
    
    return pd.concat(all_stacked, ignore_index=True)


def _save_metric_heatmap(df, title, outfile, metric_label):
    if df.empty:
        return
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    if df.empty:
        return

    is_kappa = "kappa" in metric_label.lower()
    vmax = max(float(np.nanmax(np.abs(df.values))), 0.1) if not df.isna().all().all() else 0.1
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1] * 0.8),
                                    max(4, df.shape[0] * 0.35)))
    sns.heatmap(df, cmap="RdYlGn" if is_kappa else "YlGnBu",
                vmin=-vmax if is_kappa else 0, vmax=vmax, center=0 if is_kappa else None,
                linewidths=0.5, annot=True, fmt=".2f",
                annot_kws={"fontsize": 7},
                cbar_kws={"label": metric_label},
                ax=ax, mask=df.isna().values)
    ax.set_title(title, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    save_fig(fig, outfile, bbox_inches=None)


def _plot_per_state_metrics(datasets, analysis_dirs, outdir, match_method):
    """Per-dataset and summary per-state metrics plots."""
    df = _collect_per_state_metrics(datasets, analysis_dirs, match_method)
    if df.empty:
        return

    df["Method"] = df["method"].apply(lambda m: DISPLAY_NAMES.get(m, m))

    for metric, label in [("kappa", "Cohen's Kappa"), ("jaccard", "Jaccard")]:
        if metric not in df.columns or df[metric].dropna().empty:
            continue

        summary_df = df.groupby(["state", "Method"])[metric].mean().unstack()
        summary_df = summary_df[~summary_df.index.isin(["FULL", "NOQH"])]
        summary_df = summary_df.reindex(sort_states(summary_df.index))

        pooled_labels = [DISPLAY_NAMES.get(m, m) for m in METHODS_POOLED if m != "ref"]
        methods_in_plot = [m for m in pooled_labels if m in summary_df.columns]
        other_methods = [m for m in summary_df.columns if m not in pooled_labels and m != DISPLAY_NAMES.get("ref", "ref")]
        summary_df = summary_df[methods_in_plot + other_methods]

        out_path = os.path.join(outdir, f"per_state_{metric}_summary.png")
        _save_metric_heatmap(summary_df, f"Mean Per-state {label} vs Reference (Summary)", out_path, f"Per-state {label}")

        # Average metric per state, combined across methods; the FULL/NOQH
        # pseudo-states are not real state types.
        plot_df = df.dropna(subset=[metric])
        plot_df = plot_df[~plot_df["state"].isin(["FULL", "NOQH"])]
        
        states_order = sort_states(plot_df["state"].unique())
        fig, ax = plt.subplots(figsize=(max(12, len(states_order) * 0.8), 5))
        
        sns.barplot(data=plot_df, x="state", y=metric, 
                    order=states_order, color="#4878CF",
                    capsize=0.1, errorbar="se", ax=ax, edgecolor="lightgrey", linewidth=1)
        
        strip_points(ax, data=plot_df, x="state", y=metric,
                     order=states_order,
                     size=1.5, alpha=0.4, jitter=0.2)
        
        ax.set_title(f"Average Per-state {label} vs Reference", fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlabel("State", fontsize=9)
        ax.set_ylim(0 if metric == "jaccard" else None, 1.05)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        save_fig(fig, os.path.join(outdir, f"per_state_{metric}_bar.png"))


# (comparison_table column, title, ylabel, outfile stem)
_REP_CONSISTENCY_PLOTS = [
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
    """Replicate consistency bar plots, mean ± std across datasets."""
    os.makedirs(outdir, exist_ok=True)
    for col, title, ylabel, stem in _REP_CONSISTENCY_PLOTS:
        outpath = os.path.join(outdir, f"{stem}.png")
        data = _collect_table_col(datasets, methods_dirs, col)
        _plot_summary(data, title, ylabel, outpath, partial_note=True)


def _plot_rep_similarity_distribution(datasets, methods_dirs, outfile, noqh=False):
    """Bar plot: replicate consistency distribution per de-novo method and metric."""
    suffix = "_noqh" if noqh else ""
    metric_configs = [
        ("Composition", f"composition{suffix}_rep1_vs_rep2", "#E8833A"),
        ("Kappa",       f"kappa{suffix}_rep1_vs_rep2",       "#4878CF"),
        ("Jaccard",     f"jaccard{suffix}_rep1_vs_rep2",     "#2CA02C"),
    ]

    rows = []
    for metric, col, _ in metric_configs:
        data = _collect_table_col(datasets, methods_dirs, col)
        for method in data.index:
            label = _SAMPLE_TO_INFO.get(method, (method,))[0]
            for ds_name, val in data.loc[method].dropna().items():
                rows.append({"Method": label, "Metric": metric, "value": float(val), "dataset": ds_name})

    if not rows:
        print(f"  skipping {outfile}: no data", file=sys.stderr)
        return

    plot_df = pd.DataFrame(rows)
    method_labels = [_SAMPLE_TO_INFO.get(m, (m,))[0] for m in METHODS_POOLED
                     if _SAMPLE_TO_INFO.get(m, (m,))[0] in plot_df["Method"].unique()]

    palette = {metric: color for metric, _, color in metric_configs}

    n_methods = len(method_labels)
    fig, ax = plt.subplots(figsize=(max(8, n_methods * 1.2 + 2), 5))
    sns.barplot(
        data=plot_df, x="Method", y="value", hue="Metric",
        order=method_labels,
        hue_order=[m[0] for m in metric_configs],
        palette=palette,
        estimator="mean", errorbar="se",
        ax=ax, capsize=0.1, err_kws={"linewidth": 1.0},
        edgecolor="lightgrey", linewidth=1,
    )
    strip_points(ax, data=plot_df, x="Method", y="value", hue="Metric",
                 order=method_labels, hue_order=[m[0] for m in metric_configs])
    ax.set_xlabel("")
    ax.set_ylabel("Replicate similarity", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    mode = "NOQH (excl. Quies/Het)" if noqh else "Full"
    n_ds = plot_df["dataset"].nunique()
    ax.set_title(
        f"Replicate consistency by method — {mode}\n"
        f"({n_methods} methods, {n_ds} datasets with replicates)",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(title="Metric", fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    save_fig(fig, outfile)


def _plot_rep_consistency_per_state(datasets, methods_dirs, outdir):
    """Bar plots of per-state replicate consistency (Jaccard and Kappa)."""
    df = _collect_rep_per_state_metrics(datasets, methods_dirs)
    if df.empty:
        print("  no per-state replicate consistency data found", file=sys.stderr)
        return

    df["Method"] = df["method"].apply(lambda m: _SAMPLE_TO_INFO.get(m, (m,))[0])
    states = sort_states(df["state"].unique())
    method_labels = [_SAMPLE_TO_INFO.get(m, (m,))[0] for m in METHODS_POOLED
                     if _SAMPLE_TO_INFO.get(m, (m,))[0] in df["Method"].unique()]

    for metric in ["jaccard", "kappa"]:
        title = "Jaccard" if metric == "jaccard" else "Cohen's Kappa"
        outpath = os.path.join(outdir, f"rep_consistency_per_state_{metric}.png")

        fig, ax = plt.subplots(figsize=(max(10, len(states) * 0.8), 5))

        sns.barplot(
            data=df, x="state", y=metric, hue="Method",
            order=states, hue_order=method_labels,
            palette=METHOD_PALETTE,
            estimator="mean", errorbar="se",
            ax=ax, capsize=0.05, err_kws={"linewidth": 1.0},
            edgecolor="lightgrey", linewidth=0.5
        )

        strip_points(ax, data=df, x="state", y=metric, hue="Method",
                     order=states, hue_order=method_labels)

        ax.set_xlabel("Chromatin State")
        ax.set_ylabel(f"Replicate {title}")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_title(f"Replicate consistency by state: {title}", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Method", fontsize=8, title_fontsize=9,
                  bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)

        save_fig(fig, outpath)


def _collect_rep_per_state_metrics(datasets, methods_dirs):
    """Collect per-state metrics from per_state_metrics.tsv across all datasets."""
    rows = []
    for ds, mdir in zip(datasets, methods_dirs):
        path = os.path.join(mdir, "per_state_metrics.tsv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found. Re-run comparison to generate it.", file=sys.stderr)
            continue

        try:
            df = pd.read_csv(path, sep="\t")
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}", file=sys.stderr)
            continue

        def get_method(l):
            l = str(l).split(":")[-1]   # "ds:method_rep1" or "method_rep1"
            if l.endswith("_rep1"): return l[:-5]
            if l.endswith("_rep2"): return l[:-5]
            return None

        for _, row in df.iterrows():
            s1, s2 = row["seg1"], row["seg2"]
            m1 = get_method(s1)
            m2 = get_method(s2)

            if m1 and m1 == m2 and ((str(s1).endswith("_rep1") and str(s2).endswith("_rep2")) or
                                    (str(s1).endswith("_rep2") and str(s2).endswith("_rep1"))):
                rows.append({
                    "dataset": ds,
                    "method": m1,
                    "state": row["state"],
                    "kappa": row["kappa"],
                    "jaccard": row["jaccard"]
                })
    return pd.DataFrame(rows)


# sample key → (display label, binarization key)
_SAMPLE_TO_INFO = {
    "chromhmm_default": ("Default ChromHMM",  "default"),
    "chromhmm_omni":    ("OmniPeak ChromHMM", "omnipeak"),
    "kmeans_omni":      ("KMeans OmniPeak",   "omnipeak"),
    "chromhmm_homer":   ("HOMER ChromHMM",    "homer"),
    "kmeans_homer":     ("KMeans HOMER",      "homer"),
    "chromhmm_macs2":   ("MACS2 ChromHMM",    "macs2"),
    "kmeans_macs2":     ("KMeans MACS2",      "macs2"),
}


# Peak statistics, from {ds}/peaks/peak_stats.tsv.
_PEAK_METHOD_COLORS = {
    "OmniPeak": BIN_COLORS["omnipeak"],
    "Default":  BIN_COLORS["default"],
    "ChromHMM": BIN_COLORS["default"],
    "HOMER":    BIN_COLORS["homer"],
    "MACS2":    BIN_COLORS["macs2"],
}
_PEAK_METHOD_ORDER = ["Default", "ChromHMM", "HOMER", "MACS2", "OmniPeak"]
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


def _peak_bar(data, col, ylabel, title, outpath, p_low=None, p_high=None, marks=None):
    if marks is None:
        present = data["mark"].unique()
        marks = [m for m in _MARK_ORDER if m in present]
        marks += sorted([m for m in present if m not in _MARK_ORDER])
    else:
        marks = [m for m in marks if m in data["mark"].unique()]
    methods = [m for m in _PEAK_METHOD_ORDER if m in data["method"].unique()]
    x = np.arange(len(marks))
    fig, ax = plt.subplots(figsize=(max(6, len(marks) * len(methods) * 0.22 + 2), 4.5))

    sns.barplot(data=data, x="mark", y=col, hue="method",
                order=marks, hue_order=methods, palette=_PEAK_METHOD_COLORS,
                capsize=0.05, errorbar="se", err_kws={"linewidth": 2.0},
                ax=ax, edgecolor="lightgrey", linewidth=1)

    strip_points(ax, data=data, x="mark", y=col, hue="method",
                 order=marks, hue_order=methods, dodge=True, size=2)

    ax.set_xticks(x)
    ax.set_xticklabels(marks, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(title="Method", fontsize=7, title_fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    save_fig(fig, outpath)


def log_peak_outliers(df, p_low=5, p_high=95, marks=None):
    """Log peak count / mean length outliers (< p_low% or > p_high%) per method and mark."""
    methods = [m for m in _PEAK_METHOD_ORDER if m in df["method"].unique()]
    if marks is None:
        present = df["mark"].unique()
        marks = [m for m in _MARK_ORDER if m in present]
        marks += sorted([m for m in present if m not in _MARK_ORDER])
    else:
        marks = [m for m in marks if m in df["mark"].unique()]

    for col, col_name, unit in [("n_peaks", "Peak counts (n_peaks)", ""),
                                ("mean_length", "Mean peak length (bp)", " bp")]:
        print(f"=== Outliers for {col_name} (< {p_low}% or > {p_high}%) ===")
        total_count = 0
        for method in methods:
            for mark in marks:
                sub = df[(df["method"] == method) & (df["mark"] == mark)]
                if sub.empty:
                    continue
                vals = sub[col].dropna().values
                if len(vals) == 0:
                    continue
                q_low, q_high = np.percentile(vals, [p_low, p_high])
                lows = sub[sub[col] < q_low]
                highs = sub[sub[col] > q_high]
                outliers_count = len(lows) + len(highs)
                if outliers_count > 0:
                    total_count += outliers_count
                    print(f"  [{method}] {mark} (5th%: {q_low:.1f}, 95th%: {q_high:.1f}, total={len(vals)}):")
                    if not lows.empty:
                        low_strs = [f"{r['dataset']} ({r[col]:.1f}{unit})" if isinstance(r[col], float)
                                    else f"{r['dataset']} ({r[col]}{unit})"
                                    for _, r in lows.iterrows()]
                        print(f"    - Low outliers (< {p_low}%): {', '.join(low_strs)}")
                    if not highs.empty:
                        high_strs = [f"{r['dataset']} ({r[col]:.1f}{unit})" if isinstance(r[col], float)
                                     else f"{r['dataset']} ({r[col]}{unit})"
                                     for _, r in highs.iterrows()]
                        print(f"    - High outliers (> {p_high}%): {', '.join(high_strs)}")
        print(f"Total {col_name} outliers excluded: {total_count}\n")


def _plot_peak_count(datasets, workdir, outpath, p_low=None, p_high=None, marks=None):
    """Grouped bar chart: n_peaks per mark and method, mean ± std across datasets."""
    if isinstance(datasets, pd.DataFrame):
        data = datasets
    else:
        data = _load_peak_frames(datasets, workdir)
    if data is None or data.empty:
        print(f"  skipping {outpath}: no data")
        return
    n = data["dataset"].nunique()
    _peak_bar(data, "n_peaks", "Number of peaks",
              f"Peak count per mark and method  (mean ± std, n={n} datasets)", outpath,
              p_low=p_low, p_high=p_high, marks=marks)


def _plot_peak_length(datasets, workdir, outpath, p_low=None, p_high=None, marks=None):
    """Grouped bar chart: peak length per mark and method, mean ± std across datasets."""
    if isinstance(datasets, pd.DataFrame):
        data = datasets
    else:
        data = _load_peak_frames(datasets, workdir)
    if data is None or data.empty:
        print(f"  skipping {outpath}: no data")
        return
    n = data["dataset"].nunique()
    _peak_bar(data, "mean_length", "Mean peak length (bp)",
              f"Mean peak length per mark and method  (mean ± std, n={n} datasets)", outpath,
              p_low=p_low, p_high=p_high, marks=marks)


def _plot_state_coverage(datasets, cells, workdir, markups_dir, nstates, outfile, match_method):
    """Grouped bar chart: fraction of genome per state, method as hue, all
    datasets pooled.
    """

    def _coverage(paths):
        """{state: total_bp} pooled across paths."""
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

    all_states = set()
    method_fracs = {}  # label -> [{state: fraction}, ...], one per dataset

    for key, label, _ in INTER_DS_METHODS:
        per_ds = []
        if key == "ref":
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

    rows = []   # one row per (state, dataset, method)
    for label in labels:
        for d in method_fracs[label]:
            for s in states:
                rows.append({"State": s, "Method": label, "fraction": d.get(s, 0.0)})
    plot_df = pd.DataFrame(rows)

    n_methods = len(labels)
    figw = max(16, len(states) * n_methods * 0.22)

    # Broken y-axis: the upper panel holds Quies/Het, the lower one the rest.
    BREAK_LOW  = 0.20
    BREAK_HIGH = 0.40

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        figsize=(figw, 6),
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.06},
    )

    for ax in (ax_top, ax_bot):
        sns.barplot(
            data=plot_df, x="State", y="fraction", hue="Method",
            hue_order=labels, palette=colors,
            estimator="mean", errorbar="se",
            ax=ax, capsize=0.15, err_kws={"linewidth": 1.0},
            legend=(ax is ax_top),
            edgecolor="lightgrey", linewidth=1,
        )
        strip_points(ax, data=plot_df, x="State", y="fraction", hue="Method",
                     order=states, hue_order=labels,
                     size=1.5, alpha=0.4, jitter=0.2)

    ax_top.set_ylim(BREAK_HIGH, 1.02)
    ax_bot.set_ylim(0, BREAK_LOW)

    # Hiding the inner spines creates the visual break.
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False)

    # Diagonal break marks at the left and right edges of both panels.
    d = 0.012
    kwargs = dict(transform=fig.transFigure, color="k", clip_on=False, linewidth=0.8)
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
    save_fig(fig, outfile, tight=False)


def _stacked_composition_chart(coverages, labels, title, outfile, label_fontsize=7):
    """Stacked 100% bar chart from a {label: {state: fraction}} dict."""
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

    # Renormalize so the fractions still sum to 1.0 after discarding "Unknown".
    norm_coverages = {}
    for lbl in labels:
        fracs = coverages[lbl]
        total = sum(fracs.get(st, 0.0) for st in states)
        if total > 0:
            norm_coverages[lbl] = {st: fracs.get(st, 0.0) / total for st in states}
        else:
            norm_coverages[lbl] = {st: 0.0 for st in states}

    for s in states:
        vals = np.array([norm_coverages[lbl].get(s, 0.0) for lbl in labels])
        ax.bar(x, vals, bottom=bottom, label=s, color=state_colors[s],
               edgecolor='none', width=0.8)
        bottom += vals

    # A single border around the whole stacked bar.
    ax.bar(x, bottom, color='none', edgecolor='lightgrey', linewidth=1,
           width=0.8, label='_nolegend_')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_fontsize, rotation=45, ha="right")
    ax.set_ylabel("Fraction of genome", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(title="State", fontsize=7, title_fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0, ncol=1)
    ax.set_title(title, fontsize=10, fontweight="bold")
    save_fig(fig, outfile)


def _plot_reference_composition(markups_dir, outfile):
    """Stacked bar chart: state fraction per ENCODE reference segmentation."""
    beds = ref_beds(markups_dir)
    # Sort by cell type label, the part after the ENCFF..._ prefix.
    beds = sorted(beds, key=lambda p: "_".join(Path(p).name.replace(".bed.gz", "").replace(".bed", "").split("_")[1:]))
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


def _plot_method_composition(datasets, cells, workdir, markups_dir, nstates, outfile, match_method):
    """Stacked bar chart: mean state fraction per method, averaged across datasets."""

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
        if key == "ref":
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
                                         method_key, method_label, outfile, match_method):
    """Stacked bar chart: state fraction per dataset for a single method."""

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
    """Bar plot: pairwise similarity distributions per de-novo method and metric.

    Each bar is the upper triangle of a method's kappa/Jaccard matrix, one value
    per dataset pair. group_a / group_b, when given, keep only pairs with one
    dataset from each group.
    """
    suffix = "_noqh" if noqh else ""
    metric_configs = [
        ("Composition", f"composition{suffix}_similarity_matrix.tsv",                 "#E8833A"),
        ("Kappa",   f"kappa{suffix}_matrix.tsv",                                        "#4878CF"),
        ("Jaccard", f"jaccard_noqh_matrix.tsv" if noqh else "jaccard_similarity_matrix.tsv",
                    "#2CA02C"),
    ]

    def _ds_name(label):
        """Dataset prefix of a matrix index label like 'imr90:method'."""
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
                    rows.append({"Method": label, "Metric": metric, "value": float(mat.iloc[i, j]),
                                 "ds_i": ds_i, "ds_j": ds_j})

    if not rows:
        print(f"  skipping {outfile}: no data", file=sys.stderr)
        return

    plot_df = pd.DataFrame(rows)
    method_labels = [_SAMPLE_TO_INFO.get(m, (m,))[0] for m in methods
                     if _SAMPLE_TO_INFO.get(m, (m,))[0] in plot_df["Method"].unique()]
    palette = {"Composition": "#E8833A", "Kappa": "#4878CF", "Jaccard": "#2CA02C"}

    n_methods = len(method_labels)
    fig, ax = plt.subplots(figsize=(max(10, n_methods * 1.4 + 3), 5))
    sns.barplot(
        data=plot_df, x="Method", y="value", hue="Metric",
        order=method_labels,
        hue_order=["Composition", "Kappa", "Jaccard"],
        palette=palette,
        estimator="mean", errorbar="se",
        ax=ax, capsize=0.1, err_kws={"linewidth": 1.0},
        edgecolor="lightgrey", linewidth=1,
    )
    strip_points(ax, data=plot_df, x="Method", y="value", hue="Metric",
                 order=method_labels,
                 hue_order=["Composition", "Kappa", "Jaccard"], size=2)
    ax.set_xlabel("")
    ax.set_ylabel("Pairwise similarity", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    mode = "NOQH (excl. Quies/Het)" if noqh else "Full"
    n_ds = pd.concat([plot_df["ds_i"], plot_df["ds_j"]]).nunique()
    n_pairs = plot_df[plot_df["Metric"] == "Composition"]["Method"].count() // max(n_methods, 1)
    pair_desc = " — ChIP↔Mint-ChIP pairs only" if (group_a and group_b) else ""
    ax.set_title(
        f"Inter-dataset similarity by method — {mode}{pair_desc}\n"
        f"({n_methods} methods, n={n_ds} datasets, {n_pairs} pairs each)",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(title="Metric", fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    save_fig(fig, outfile)


def _plot_reference_distribution(comp_path, kappa_path, jaccard_path, outfile,
                                  title_suffix=""):
    """Bar plot of pairwise similarity among ENCODE reference segmentations."""
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
    sns.barplot(data=plot_df, x="Metric", y="value", hue="Metric",
                order=["Composition", "Kappa", "Jaccard"],
                palette={"Composition": "#E8833A", "Kappa": "#4878CF", "Jaccard": "#2CA02C"},
                estimator="mean", errorbar="se",
                ax=ax, capsize=0.15, err_kws={"linewidth": 1.0},
                legend=False, edgecolor="lightgrey", linewidth=1)
    strip_points(ax, data=plot_df, x="Metric", y="value",
                 order=["Composition", "Kappa", "Jaccard"], dodge=False)
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
    save_fig(fig, outfile)


def plot_reference_n_segments(datasets, methods_dirs, labels, outfile, title):
    """Bar chart: number of segments of each dataset's own ENCODE reference,
    one grey bar per dataset (the "ref" row of its comparison_table).
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
    fig, ax = plt.subplots(figsize=(max(5, len(labs) * 1.2), 4.2))
    x = np.arange(len(labs))
    ax.bar(x, vals, color=BIN_COLORS["reference"], edgecolor="lightgrey", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Segments (×10³)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, v in enumerate(vals):
        ax.text(i, v + yr * 0.01, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    save_fig(fig, outfile)


def _plot_per_dataset_all_methods_composition(datasets, cells, workdir, nstates,
                                              methods, outdir, match_method):
    """Stacked bar chart: state fraction per method for each dataset."""

    def _fracs_from_path(path):
        p = Path(path)
        if not p.exists():
            return {}
        df = load_bed_df(str(p))[["state", "length"]]
        totals = df.groupby("state")["length"].sum()
        total_bp = totals.sum()
        return (totals / total_bp).to_dict() if total_bp > 0 else {}

    for ds, cell in zip(datasets, cells):
        coverages = {}
        labels_out = []
        for key, label, _ in methods:
            if key == "ref":
                continue
            try:
                bed = ds_method_bed(workdir, ds, cell, nstates, key, match_method)
                fracs = _fracs_from_path(bed)
                if fracs:
                    coverages[label] = fracs
                    labels_out.append(label)
            except KeyError:
                continue

        if not coverages:
            continue

        outfile = os.path.join(outdir, f"ds_composition_{ds}.png")
        _stacked_composition_chart(
            coverages, labels_out,
            f"State composition — {ds} (all de-novo methods)",
            outfile, label_fontsize=9,
        )


def run_summary_plots(datasets=None, methods_dirs=None, analysis_dirs=None,
                      outdir=None, workdir=None, markups_dir=None, cells=None,
                      methods=None, marks=None, nstates=15, match_method='jaccard',
                      state_coverage_outfile=None,
                      peak_count_outfile=None, peak_length_outfile=None,
                      peak_stats_outfile=None,
                      ref_composition_outfile=None,
                      ref_comp_matrix=None,
                      ref_kappa_matrix=None, ref_jaccard_matrix=None,
                      ref_dist_outfile=None,
                      ref_comp_noqh_matrix=None,
                      ref_kappa_noqh_matrix=None, ref_jaccard_noqh_matrix=None,
                      ref_dist_noqh_outfile=None,
                      method_composition_outfile=None,
                      method_ds_composition_outdir=None,
                      all_methods_composition_outdir=None,
                      rep_consistency_outdir=None,
                      method_sim_dist_indir=None, method_sim_dist_methods=None,
                      method_sim_dist_outfile=None,
                      method_sim_dist_noqh_outfile=None,
                      method_sim_dist_group_a=None, method_sim_dist_group_b=None,
                      method_sim_dist_filtered_outfile=None,
                      method_sim_dist_filtered_noqh_outfile=None):
    """Cross-dataset summary bar plots and similarity distributions.

    Each *_outfile / *_outdir argument that is set selects one plot group.
    """
    args = SimpleNamespace(
        datasets=datasets or [], methods_dirs=methods_dirs or [],
        analysis_dirs=analysis_dirs or [], outdir=outdir, workdir=workdir,
        markups_dir=markups_dir, cells=cells or [], methods=methods or [],
        nstates=nstates, marks=marks, match_method=match_method,
        state_coverage_outfile=state_coverage_outfile,
        peak_count_outfile=peak_count_outfile, peak_length_outfile=peak_length_outfile,
        peak_stats_outfile=peak_stats_outfile,
        ref_composition_outfile=ref_composition_outfile,
        ref_comp_matrix=ref_comp_matrix,
        ref_kappa_matrix=ref_kappa_matrix, ref_jaccard_matrix=ref_jaccard_matrix,
        ref_dist_outfile=ref_dist_outfile,
        ref_comp_noqh_matrix=ref_comp_noqh_matrix,
        ref_kappa_noqh_matrix=ref_kappa_noqh_matrix,
        ref_jaccard_noqh_matrix=ref_jaccard_noqh_matrix,
        ref_dist_noqh_outfile=ref_dist_noqh_outfile,
        method_composition_outfile=method_composition_outfile,
        method_ds_composition_outdir=method_ds_composition_outdir,
        all_methods_composition_outdir=all_methods_composition_outdir,
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
        # Narrow the global method lists/palettes to the requested subset.
        global INTER_DS_METHODS, METHOD_PALETTE, METHODS_POOLED
        methods_map = {m[0] if isinstance(m, (list, tuple)) else m: m
                       for m in INTER_DS_METHODS}
        new_methods = []
        # The state-level plots (coverage, composition) always show the ENCODE
        # reference; the bar/table plots key it off "ref" instead, so a stray
        # entry here is harmless to them.
        if "ref" not in args.methods and "ref" in methods_map:
            new_methods.append(methods_map["ref"])
        for k in args.methods:
            if k in methods_map:
                m = methods_map[k]
                if isinstance(m, (list, tuple)):
                    new_methods.append(m)
                else:
                    new_methods.append((m, m.replace("_", " ").title(), "#808080"))
            else:
                new_methods.append((k, k.replace("_", " ").title(), "#808080"))
        INTER_DS_METHODS = new_methods
        METHOD_PALETTE = {label: color for _, label, color in INTER_DS_METHODS}
        METHODS_POOLED = [m[0] for m in INTER_DS_METHODS]

    if args.outdir:
        if not (len(args.datasets) == len(args.methods_dirs) == len(args.analysis_dirs)):
            raise ValueError("--datasets, --methods-dirs and --analysis-dirs must have equal lengths")
        os.makedirs(args.outdir, exist_ok=True)
        ds, mdirs, adirs = args.datasets, args.methods_dirs, args.analysis_dirs

        data = _collect_table_col(ds, mdirs, "entropy", include_ref=True)
        _plot_summary(data, "Transition matrix entropy (full)",
                      "Entropy (bits)",
                      os.path.join(args.outdir, "summary_entropy.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "entropy_noqh", include_ref=True)
        _plot_summary(data, "Transition matrix entropy (NOQH, excl. Quies/Het)",
                      "Entropy (bits)",
                      os.path.join(args.outdir, "summary_entropy_noqh.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        # RNA-seq validation (expressed genes).
        data = _collect_table_col(ds, mdirs, "jaccard_Tx_ExpressedGeneBodies", include_ref=True)
        _plot_summary(data, "Jaccard: Tx state vs expressed gene bodies", "Jaccard",
                      os.path.join(args.outdir, "summary_jaccard_tx.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "enrich_Tx_ExpressedGeneBodies", include_ref=True)
        _plot_summary(data, "Tx enrichment at expressed gene bodies", "Fold enrichment",
                      os.path.join(args.outdir, "summary_enrich_tx.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "sensitivity_Tx_ExpressedGeneBodies", include_ref=True) * 100.0
        _plot_summary(data, "Fraction of expressed gene bodies covered by Tx states", "% overlap",
                      os.path.join(args.outdir, "summary_sensitivity_tx.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "coverage_Tx_ExpressedGeneBodies", include_ref=True) * 100.0
        _plot_summary(data, "Fraction of Tx states covered by expressed genes", "% overlap",
                      os.path.join(args.outdir, "summary_coverage_tx.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        tx_sens = _collect_table_col(ds, mdirs, "sensitivity_Tx_ExpressedGeneBodies", include_ref=True) * 100.0
        tx_cov = _collect_table_col(ds, mdirs, "coverage_Tx_ExpressedGeneBodies", include_ref=True) * 100.0
        _plot_2way_scatter(tx_sens, tx_cov,
                           "Tx state validation (Expressed Gene Bodies)",
                           "Fraction of expressed gene bodies covered by Tx states (%)",
                           "Fraction of Tx states covered by expressed genes (%)",
                           os.path.join(args.outdir, "summary_2way_tx.png"),
                           order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "jaccard_Tss_RefSeqTSS2kb", include_ref=True)
        _plot_summary(data, "Jaccard: Tss state vs RefSeq TSS ±2 kb", "Jaccard",
                      os.path.join(args.outdir, "summary_jaccard_tss.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "enrich_Tss_RefSeqTSS2kb", include_ref=True)
        _plot_summary(data, "Tss enrichment at RefSeq TSS ±2 kb", "Fold enrichment",
                      os.path.join(args.outdir, "summary_enrich_tss.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "sensitivity_Tss_RefSeqTSS2kb", include_ref=True) * 100.0
        _plot_summary(data, "Fraction of RefSeq TSS ±2 kb covered by Tss states", "% overlap",
                      os.path.join(args.outdir, "summary_sensitivity_tss.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "coverage_Tss_RefSeqTSS2kb", include_ref=True) * 100.0
        _plot_summary(data, "Fraction of Tss states covered by RefSeq TSS ±2 kb", "% overlap",
                      os.path.join(args.outdir, "summary_coverage_tss.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        tss_sens = _collect_table_col(ds, mdirs, "sensitivity_Tss_RefSeqTSS2kb", include_ref=True) * 100.0
        tss_cov = _collect_table_col(ds, mdirs, "coverage_Tss_RefSeqTSS2kb", include_ref=True) * 100.0
        _plot_2way_scatter(tss_sens, tss_cov,
                           "Tss state validation (RefSeq TSS ±2 kb)",
                           "Fraction of RefSeq TSS ±2 kb covered by Tss states (%)",
                           "Fraction of Tss states covered by RefSeq TSS ±2 kb (%)",
                           os.path.join(args.outdir, "summary_2way_tss.png"),
                           order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "jaccard_Tss_ExpressedTSS", include_ref=True)
        _plot_summary(data, "Jaccard: Tss state vs Expressed TSS", "Jaccard",
                      os.path.join(args.outdir, "summary_jaccard_tss_exptss.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "jaccard_Tss_ExpressedTSS2kb", include_ref=True)
        _plot_summary(data, "Jaccard: Tss state vs Expressed TSS ±2 kb", "Jaccard",
                      os.path.join(args.outdir, "summary_jaccard_tss_exptss2kb.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "enrich_Tss_ExpressedTSS2kb", include_ref=True)
        _plot_summary(data, "Tss enrichment at Expressed TSS ±2 kb", "Fold enrichment",
                      os.path.join(args.outdir, "summary_enrich_tss_exptss2kb.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "sensitivity_Tss_ExpressedTSS", include_ref=True) * 100.0
        _plot_summary(data, "Fraction of Expressed TSS covered by Tss states", "% overlap",
                      os.path.join(args.outdir, "summary_sensitivity_tss_exptss.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "coverage_Tss_ExpressedTSS", include_ref=True) * 100.0
        _plot_summary(data, "Fraction of Tss states covered by Expressed TSS", "% overlap",
                      os.path.join(args.outdir, "summary_coverage_tss_exptss.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        exptss_sens = _collect_table_col(ds, mdirs, "sensitivity_Tss_ExpressedTSS", include_ref=True) * 100.0
        exptss_cov = _collect_table_col(ds, mdirs, "coverage_Tss_ExpressedTSS", include_ref=True) * 100.0
        _plot_2way_scatter(exptss_sens, exptss_cov,
                           "Tss state validation (Expressed TSS)",
                           "Fraction of Expressed TSS covered by Tss states (%)",
                           "Fraction of Tss states covered by Expressed TSS (%)",
                           os.path.join(args.outdir, "summary_2way_tss_exptss.png"),
                           order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        exptss2kb_sens = _collect_table_col(ds, mdirs, "sensitivity_Tss_ExpressedTSS2kb", include_ref=True) * 100.0
        exptss2kb_cov = _collect_table_col(ds, mdirs, "coverage_Tss_ExpressedTSS2kb", include_ref=True) * 100.0
        _plot_2way_scatter(exptss2kb_sens, exptss2kb_cov,
                           "Tss state validation (Expressed TSS ±2 kb)",
                           "Fraction of Expressed TSS ±2 kb covered by Tss states (%)",
                           "Fraction of Tss states covered by Expressed TSS ±2 kb (%)",
                           os.path.join(args.outdir, "summary_2way_tss_exptss2kb.png"),
                           order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "enrich_Active_NonExpGeneBodies", include_ref=True)
        _plot_summary(data, "Active states enrichment at Non-expressed Gene Bodies", "Fold enrichment",
                      os.path.join(args.outdir, "summary_enrich_active_nonexp.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "enrich_Quies_NonExpGeneBodies", include_ref=True)
        _plot_summary(data, "Quiescent states enrichment at Non-expressed Gene Bodies", "Fold enrichment",
                      os.path.join(args.outdir, "summary_enrich_quies_nonexp.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        # ATAC-seq validation.
        data = _collect_table_col(ds, mdirs, "jaccard_Active_ATAC", include_ref=True)
        _plot_summary(data, "Jaccard: Active states vs ATAC-seq", "Jaccard",
                      os.path.join(args.outdir, "summary_jaccard_active_atac.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "enrich_Active_ATAC", include_ref=True)
        _plot_summary(data, "Active chromatin enrichment at ATAC-seq peaks", "Fold enrichment",
                      os.path.join(args.outdir, "summary_enrich_active_atac.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "sensitivity_Active_ATAC", include_ref=True) * 100.0
        _plot_summary(data, "Fraction of ATAC-seq peaks covered by Active states", "% overlap",
                      os.path.join(args.outdir, "summary_sensitivity_active_atac.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "coverage_Active_ATAC", include_ref=True) * 100.0
        _plot_summary(data, "Fraction of Active states covered by ATAC-seq peaks", "% overlap",
                      os.path.join(args.outdir, "summary_coverage_active_atac.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        atac_sens = _collect_table_col(ds, mdirs, "sensitivity_Active_ATAC", include_ref=True) * 100.0
        atac_cov = _collect_table_col(ds, mdirs, "coverage_Active_ATAC", include_ref=True) * 100.0
        _plot_2way_scatter(atac_sens, atac_cov,
                           "Active chromatin validation (ATAC-seq)",
                           "Fraction of ATAC-seq peaks covered by Active states (%)",
                           "Fraction of Active states covered by ATAC-seq peaks (%)",
                           os.path.join(args.outdir, "summary_2way_active_atac.png"),
                           order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "sensitivity_Tss_ATAC", include_ref=True) * 100.0
        _plot_summary(data, "Fraction of ATAC-seq peaks covered by Tss states", "% overlap",
                      os.path.join(args.outdir, "summary_sensitivity_tss_atac.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "sensitivity_Enh_ATAC", include_ref=True) * 100.0
        _plot_summary(data, "Fraction of ATAC-seq peaks covered by Enh states", "% overlap",
                      os.path.join(args.outdir, "summary_sensitivity_enh_atac.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "enrich_Quies_ATAC", include_ref=True)
        _plot_summary(data, "Quiescent states enrichment at ATAC-seq peaks", "Fold enrichment",
                      os.path.join(args.outdir, "summary_enrich_quies_atac.png"),
                      partial_note=True, order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        # Tx segment lengths.
        data = _collect_table_col(ds, mdirs, "median_Tx_length", include_ref=True)
        _plot_summary(data, "Median Tx (transcription) segment length", "bp",
                      os.path.join(args.outdir, "summary_median_tx_length.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "mean_Tx_length", include_ref=True)
        _plot_summary(data, "Mean Tx (transcription) segment length", "bp",
                      os.path.join(args.outdir, "summary_mean_tx_length.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        data = _collect_table_col(ds, mdirs, "n_segments", include_ref=True) / 1000.0
        _plot_summary(data, "Total number of segments", "Segments (×10³)",
                      os.path.join(args.outdir, "summary_n_segments.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        # Similarity vs ENCODE reference.
        _plot_summary(_collect_table_col(ds, mdirs, "kappa_vs_ref", include_ref=True),
                      "Agreement vs ENCODE reference (Kappa)", "Cohen's Kappa",
                      os.path.join(args.outdir, "summary_kappa_vs_ref.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))

        _plot_summary(_collect_table_col(ds, mdirs, "jaccard_vs_ref", include_ref=True),
                      "Agreement vs ENCODE reference (Jaccard)", "Mean per-state Jaccard",
                      os.path.join(args.outdir, "summary_jaccard_vs_ref.png"),
                      order=list(dict.fromkeys(["ref"] + METHODS_POOLED)))


        _plot_per_state_metrics(ds, adirs, args.outdir, match_method=args.match_method)

    if args.state_coverage_outfile:
        if not (args.workdir and args.markups_dir and args.cells):
            raise ValueError("--workdir, --markups-dir and --cells are required for coverage plots")
        if len(args.datasets) != len(args.cells):
            raise ValueError("--datasets and --cells must have equal lengths")

        os.makedirs(os.path.dirname(os.path.abspath(args.state_coverage_outfile)), exist_ok=True)
        _plot_state_coverage(args.datasets, args.cells, args.workdir, args.markups_dir,
                             args.nstates, args.state_coverage_outfile, args.match_method)

    if args.peak_count_outfile:
        if not args.workdir:
            raise ValueError("--workdir is required for --peak-count-outfile")
        os.makedirs(os.path.dirname(os.path.abspath(args.peak_count_outfile)), exist_ok=True)
        _plot_peak_count(args.datasets, args.workdir, args.peak_count_outfile, marks=args.marks)

    if args.peak_length_outfile:
        if not args.workdir:
            raise ValueError("--workdir is required for --peak-length-outfile")
        os.makedirs(os.path.dirname(os.path.abspath(args.peak_length_outfile)), exist_ok=True)
        _plot_peak_length(args.datasets, args.workdir, args.peak_length_outfile, marks=args.marks)

    # Deprecated: split into the two plots above.
    if args.peak_stats_outfile:
        print("  WARNING: --peak-stats-outfile is deprecated, splitting into n_peaks and peak_length", 
              file=sys.stderr)
        if not args.workdir:
            raise ValueError("--workdir is required for --peak-stats-outfile")
        
        outdir = os.path.dirname(os.path.abspath(args.peak_stats_outfile))
        os.makedirs(outdir, exist_ok=True)
        _plot_peak_count(args.datasets, args.workdir, os.path.join(outdir, "n_peaks.png"), marks=args.marks)
        _plot_peak_length(args.datasets, args.workdir, os.path.join(outdir, "peak_length.png"), marks=args.marks)

    if args.ref_composition_outfile:
        if not args.markups_dir:
            raise ValueError("--markups-dir is required for --ref-composition-outfile")
        os.makedirs(os.path.dirname(os.path.abspath(args.ref_composition_outfile)), exist_ok=True)
        _plot_reference_composition(args.markups_dir, args.ref_composition_outfile)

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

    if args.rep_consistency_outdir:
        if len(args.datasets) != len(args.methods_dirs):
            raise ValueError("--datasets and --methods-dirs must have equal lengths")
        _plot_rep_consistency(args.datasets, args.methods_dirs, args.rep_consistency_outdir)
        _plot_rep_similarity_distribution(
            args.datasets, args.methods_dirs,
            os.path.join(args.rep_consistency_outdir, "rep_consistency_distribution.png"),
            noqh=False,
        )
        _plot_rep_similarity_distribution(
            args.datasets, args.methods_dirs,
            os.path.join(args.rep_consistency_outdir, "rep_consistency_distribution_noqh.png"),
            noqh=True,
        )
        _plot_rep_consistency_per_state(args.datasets, args.methods_dirs, args.rep_consistency_outdir)

    if args.method_composition_outfile:
        if not args.markups_dir:
            raise ValueError("--markups-dir is required for --method-composition-outfile")
        os.makedirs(os.path.dirname(os.path.abspath(args.method_composition_outfile)), exist_ok=True)
        _plot_method_composition(args.datasets, args.cells, args.workdir, args.markups_dir,
                                 args.nstates, args.method_composition_outfile, args.match_method)

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

    if args.all_methods_composition_outdir:
        if not (args.workdir and args.cells and args.datasets):
            raise ValueError("--workdir, --datasets and --cells are required for "
                             "--all-methods-composition-outdir")
        os.makedirs(args.all_methods_composition_outdir, exist_ok=True)
        _plot_per_dataset_all_methods_composition(
            args.datasets, args.cells, args.workdir, args.nstates,
            INTER_DS_METHODS, args.all_methods_composition_outdir, args.match_method
        )
