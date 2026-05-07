#!/usr/bin/env python3
"""State emission discriminability: pairwise cosine similarity and Gini index.

For each (dataset, method) pair, loads:
  {analysis_dir}/{method}/bin_emissions/state_emissions.tsv
  {analysis_dir}/{method}/bw_emissions/state_emissions.tsv

Two complementary metrics are computed per method × emission type:

  1. Mean pairwise cosine similarity (between states)
       Higher = states look more alike = less discriminative label space.

  2. Mean Gini index (per state, averaged across states)
       Gini measures how concentrated each state's signal is across marks.
       Higher Gini = signal concentrated in a few marks = sharp, specific state
       signature = more discriminative.
       Lower Gini = signal spread uniformly across marks = state less distinctive.

Expected result: bigwig (continuous) emissions show lower Gini than binarized
emissions, confirming that signal averaging produces less specific state profiles
and motivating overlap-based (rather than emission-based) label transfer.

Outputs:
  {outdir}/emission_cosine_sim_{dataset}.png   — per-dataset cosine similarity
  {outdir}/emission_cosine_sim_summary.png     — cosine summary across datasets
  {outdir}/emission_gini_{dataset}.png         — per-dataset Gini index
  {outdir}/emission_gini_summary.png           — Gini summary across datasets

Usage:
    emission_similarity.py \\
        --datasets      imr90 monocytes monocytes_mint gm12878_mint \\
        --analysis-dirs imr90/analysis/ovlp monocytes/analysis/ovlp ... \\
        --outdir        inter_dataset/summary_plots
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from utils import METHOD_ORDER, DISPLAY_NAMES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODS_POOLED = [m for m in METHOD_ORDER
                  if not m.endswith("_rep1") and not m.endswith("_rep2")]

EMISSION_COLORS = {
    "bin": "#5B8DB8",   # blue   — binarized
    "bw":  "#E07B54",   # orange — bigwig (continuous)
}
EMISSION_LABELS = {
    "bin": "Binarized",
    "bw":  "Bigwig (continuous)",
}
EMISSION_TYPES = ["bin", "bw"]
EMISSION_SUBDIRS = {
    "bin": "bin_emissions",
    "bw":  "bw_emissions",
}


# ---------------------------------------------------------------------------
# Emission loading
# ---------------------------------------------------------------------------

def _load_emissions_tsv(path):
    """Load state_emissions.tsv; return (states, mat) or None if missing."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", index_col=0)
    return list(df.index), df.values.astype(float)


def _analysis_base(analysis_dir, method):
    """Return the analysis subdirectory for *method* inside *analysis_dir*.

    'ref' lives one level above the variant dir (e.g. analysis/ref/ rather
    than analysis/ovlp/ref/).
    """
    if method == "ref":
        return os.path.join(os.path.dirname(analysis_dir), "ref")
    return os.path.join(analysis_dir, method)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _pairwise_cosine_sim(mat):
    """Mean and std of all N*(N-1)/2 off-diagonal pairwise cosine similarities.

    mat : (n_states, n_marks) — each row is a state emission vector.
    """
    n = len(mat)
    if n < 2:
        return np.nan, np.nan
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    normed = mat / np.where(norms > 0, norms, 1.0)
    sims = (normed @ normed.T)[np.triu_indices(n, k=1)]
    return float(np.mean(sims)), float(np.std(sims))


def _gini_index(vec):
    """Gini coefficient of a non-negative 1-D emission vector for one state.

    G = 0  — signal equal across all marks (maximally uniform, least specific).
    G = 1  — all signal in one mark (maximally concentrated, most specific).

    Formula (sorted ascending):
      G = (2 * sum((i+1) * x_i)) / (n * sum(x)) - (n+1)/n
    """
    vec = np.abs(np.asarray(vec, dtype=float))
    total = vec.sum()
    if total == 0:
        return np.nan
    vec = np.sort(vec)
    n = len(vec)
    return float((2.0 * np.dot(np.arange(1, n + 1), vec) / (n * total)) - (n + 1.0) / n)


def _mean_gini(mat):
    """Mean and std of per-state Gini indices across all states.

    mat : (n_states, n_marks)
    """
    ginis = np.array([_gini_index(row) for row in mat])
    valid = ginis[~np.isnan(ginis)]
    if len(valid) == 0:
        return np.nan, np.nan
    return float(np.mean(valid)), float(np.std(valid) if len(valid) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_records(datasets, analysis_dirs, methods):
    """Return DataFrame with columns:
      dataset, method, emission_type,
      mean_sim, std_sim, mean_gini, std_gini.

    Rows are produced only when the TSV file exists. Missing bigwig emission
    files are silently skipped so the plot degrades gracefully.
    """
    records = []
    for ds, adir in zip(datasets, analysis_dirs):
        for method in methods:
            base = _analysis_base(adir, method)
            for etype in EMISSION_TYPES:
                path = os.path.join(base, EMISSION_SUBDIRS[etype],
                                    "state_emissions.tsv")
                result = _load_emissions_tsv(path)
                if result is None:
                    continue
                _, mat = result
                mean_sim, std_sim = _pairwise_cosine_sim(mat)
                mean_g, std_g = _mean_gini(mat)
                records.append({
                    "dataset":       ds,
                    "method":        method,
                    "emission_type": etype,
                    "mean_sim":      mean_sim,
                    "std_sim":       std_sim,
                    "mean_gini":     mean_g,
                    "std_gini":      std_g,
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Generic grouped-bar plotting
# ---------------------------------------------------------------------------

def _grouped_bars(ax, methods, df_slice, y_col, err_col):
    """Grouped bars (bin/bw hue) for the methods list on *ax*."""
    n = len(methods)
    width = 0.35
    offsets = {"bin": -width / 2, "bw": width / 2}
    x = np.arange(n)

    for etype in EMISSION_TYPES:
        means, errs = [], []
        for method in methods:
            row = df_slice[
                (df_slice["method"] == method) &
                (df_slice["emission_type"] == etype)
            ]
            means.append(float(row[y_col].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row[err_col].iloc[0]) if not row.empty else np.nan)

        means, errs = np.array(means), np.array(errs)
        ax.bar(x + offsets[etype], np.nan_to_num(means),
               width=width * 0.9, color=EMISSION_COLORS[etype],
               label=EMISSION_LABELS[etype], edgecolor="white", linewidth=0.5)
        valid = ~np.isnan(means) & ~np.isnan(errs)
        if valid.any():
            ax.errorbar(x[valid] + offsets[etype], means[valid],
                        yerr=errs[valid], fmt="none", color="black",
                        capsize=3, linewidth=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in methods],
                       rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0)


def _save_fig(fig, ax, ylabel, title, xlabel, outpath):
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=7, color="grey")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outpath}")


# ---------------------------------------------------------------------------
# Per-dataset plots
# ---------------------------------------------------------------------------

def _plot_per_dataset(df_ds, dataset, y_col, err_col, ylabel,
                      title_metric, title_note, xlabel_note, outpath):
    methods = [m for m in METHODS_POOLED if m in df_ds["method"].values]
    if not methods:
        print(f"  skipping {outpath}: no data")
        return
    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 0.9), 4.5))
    _grouped_bars(ax, methods, df_ds, y_col, err_col)
    _save_fig(fig, ax,
              ylabel=ylabel,
              title=f"{title_metric} — {dataset}\n({title_note})",
              xlabel=xlabel_note,
              outpath=outpath)


def plot_cosine_per_dataset(df_ds, dataset, outpath):
    _plot_per_dataset(
        df_ds, dataset,
        y_col="mean_sim", err_col="std_sim",
        ylabel="Mean pairwise cosine similarity",
        title_metric="State emission pairwise cosine similarity",
        title_note="higher = states more similar = less discriminative",
        xlabel_note="mean ± std of all N×(N−1)/2 state-pair similarities",
        outpath=outpath,
    )


def plot_gini_per_dataset(df_ds, dataset, outpath):
    _plot_per_dataset(
        df_ds, dataset,
        y_col="mean_gini", err_col="std_gini",
        ylabel="Mean Gini index (across states)",
        title_metric="State emission Gini index",
        title_note="higher = signal concentrated in fewer marks = more specific state",
        xlabel_note="mean ± std of per-state Gini coefficients",
        outpath=outpath,
    )


# ---------------------------------------------------------------------------
# Summary plots (across datasets)
# ---------------------------------------------------------------------------

def _aggregate(df, methods, y_col, std_col):
    rows = []
    for method in methods:
        for etype in EMISSION_TYPES:
            vals = df[(df["method"] == method) &
                      (df["emission_type"] == etype)][y_col].dropna().values
            if len(vals) == 0:
                continue
            rows.append({
                "method":        method,
                "emission_type": etype,
                y_col:           float(np.mean(vals)),
                std_col:         float(np.std(vals)) if len(vals) > 1 else 0.0,
            })
    return pd.DataFrame(rows)


def _plot_summary(df, datasets, y_col, std_col, ylabel,
                  title_metric, title_note, outpath):
    methods = [m for m in METHODS_POOLED if m in df["method"].values]
    if not methods:
        print(f"  skipping {outpath}: no data")
        return
    df_agg = _aggregate(df, methods, y_col, std_col)
    if df_agg.empty:
        return
    n_ds = len(datasets)
    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 0.9), 4.5))
    _grouped_bars(ax, methods, df_agg, y_col, std_col)
    _save_fig(fig, ax,
              ylabel=ylabel,
              title=f"{title_metric} — all datasets\n({title_note}; n={n_ds})",
              xlabel=f"mean ± std across {n_ds} datasets",
              outpath=outpath)


def plot_cosine_summary(df, datasets, outpath):
    _plot_summary(df, datasets,
                  y_col="mean_sim", std_col="std_sim",
                  ylabel="Mean pairwise cosine similarity",
                  title_metric="State emission pairwise cosine similarity",
                  title_note="higher = states more similar = less discriminative",
                  outpath=outpath)


def plot_gini_summary(df, datasets, outpath):
    _plot_summary(df, datasets,
                  y_col="mean_gini", std_col="std_gini",
                  ylabel="Mean Gini index (across states)",
                  title_metric="State emission Gini index",
                  title_note="higher = signal concentrated in fewer marks = more specific state",
                  outpath=outpath)


# ---------------------------------------------------------------------------
# Inter-dataset binarized emission cosine similarity
# ---------------------------------------------------------------------------

def _cosine_sim(vec_a, vec_b):
    """Cosine similarity between two emission vectors."""
    na, nb = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(vec_a, vec_b) / (na * nb))


def collect_inter_dataset_binem_records(datasets, analysis_dirs, methods,
                                        group_a=None, group_b=None):
    """For each method and dataset pair, compute mean cosine similarity between
    same-named state binarized emission vectors (name-based matching on
    comb-matched state labels — no additional Hungarian rematch).

    group_a / group_b: when both are given, only pairs with one dataset from
    each group are included (used for ChIP↔Mint-ChIP cross-assay filtering).

    Returns DataFrame with columns: method, ds_a, ds_b, mean_sim.
    """
    emissions = {}
    for ds, adir in zip(datasets, analysis_dirs):
        for method in methods:
            base = _analysis_base(adir, method)
            path = os.path.join(base, EMISSION_SUBDIRS["bin"], "state_emissions.tsv")
            result = _load_emissions_tsv(path)
            if result is not None:
                states, mat = result
                emissions[(ds, method)] = {s: mat[i] for i, s in enumerate(states)}

    records = []
    n = len(datasets)
    for i in range(n):
        for j in range(i + 1, n):
            ds_a, ds_b = datasets[i], datasets[j]
            if group_a is not None and group_b is not None:
                if not ((ds_a in group_a and ds_b in group_b) or
                        (ds_a in group_b and ds_b in group_a)):
                    continue
            for method in methods:
                em_a = emissions.get((ds_a, method))
                em_b = emissions.get((ds_b, method))
                if em_a is None or em_b is None:
                    continue
                common = set(em_a) & set(em_b)
                if not common:
                    continue
                sims = [_cosine_sim(em_a[s], em_b[s]) for s in common]
                sims = [s for s in sims if not np.isnan(s)]
                if sims:
                    records.append({
                        "method":   method,
                        "ds_a":     ds_a,
                        "ds_b":     ds_b,
                        "mean_sim": float(np.mean(sims)),
                    })
    return pd.DataFrame(records)


def plot_inter_dataset_binem(df, methods, outfile, cross_assay=False):
    """Violin plot: per-method distribution of inter-dataset binarized emission cosine similarity."""
    if df.empty:
        print(f"  skipping {outfile}: no data", file=sys.stderr)
        return

    plot_df = df.copy()
    plot_df["Method"] = plot_df["method"].map(lambda m: DISPLAY_NAMES.get(m, m))
    method_labels = [DISPLAY_NAMES.get(m, m) for m in methods
                     if DISPLAY_NAMES.get(m, m) in plot_df["Method"].values]

    n_methods = len(method_labels)
    n_pairs = int(plot_df.groupby("method").size().max()) if not plot_df.empty else 0

    fig, ax = plt.subplots(figsize=(max(8, n_methods * 1.4 + 2), 5))
    sns.violinplot(
        data=plot_df, x="Method", y="mean_sim",
        order=method_labels,
        color="#5B8DB8",
        inner="box", cut=0, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean cosine similarity (matched states)", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    pair_note = " — ChIP↔Mint-ChIP pairs" if cross_assay else ""
    ax.set_title(
        f"Inter-dataset binarized emission similarity{pair_note}\n"
        f"({n_methods} methods, {n_pairs} dataset pairs per method)",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(outfile)), exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outfile}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="State emission discriminability: cosine similarity and Gini index.")
    ap.add_argument("--datasets",      nargs="+", required=True)
    ap.add_argument("--analysis-dirs", nargs="+", required=True,
                    dest="analysis_dirs")
    ap.add_argument("--methods",       nargs="*", default=None)
    ap.add_argument("--outdir",        required=True)
    ap.add_argument("--inter-dataset-binem-outfile", default=None,
                    dest="inter_dataset_binem_outfile",
                    help="Output PNG for inter-dataset binarized emission cosine similarity")
    ap.add_argument("--cross-assay-binem-outfile", default=None,
                    dest="cross_assay_binem_outfile",
                    help="Output PNG for ChIP↔Mint-ChIP binarized emission cosine similarity")
    ap.add_argument("--group-a", nargs="*", default=None, dest="group_a",
                    help="Dataset names for group A (ChIP) for cross-assay filtering")
    ap.add_argument("--group-b", nargs="*", default=None, dest="group_b",
                    help="Dataset names for group B (Mint-ChIP) for cross-assay filtering")
    args = ap.parse_args()

    if len(args.datasets) != len(args.analysis_dirs):
        ap.error("--datasets and --analysis-dirs must have equal lengths")

    methods = args.methods or [m for m in METHODS_POOLED if m != "ref"]
    os.makedirs(args.outdir, exist_ok=True)

    print("Collecting emission data ...")
    df = collect_records(args.datasets, args.analysis_dirs, methods)
    if df.empty:
        print("WARNING: no emission TSV files found.", file=sys.stderr)
        sys.exit(0)

    for ds in args.datasets:
        df_ds = df[df["dataset"] == ds]
        plot_cosine_per_dataset(df_ds, ds,
            os.path.join(args.outdir, f"emission_cosine_sim_{ds}.png"))
        plot_gini_per_dataset(df_ds, ds,
            os.path.join(args.outdir, f"emission_gini_{ds}.png"))

    plot_cosine_summary(df, args.datasets,
        os.path.join(args.outdir, "emission_cosine_sim_summary.png"))
    plot_gini_summary(df, args.datasets,
        os.path.join(args.outdir, "emission_gini_summary.png"))

    if args.inter_dataset_binem_outfile or args.cross_assay_binem_outfile:
        print("Collecting inter-dataset binarized emission data ...")
        df_inter = collect_inter_dataset_binem_records(
            args.datasets, args.analysis_dirs, methods)
        if args.inter_dataset_binem_outfile:
            plot_inter_dataset_binem(df_inter, methods,
                                     args.inter_dataset_binem_outfile,
                                     cross_assay=False)
        if args.cross_assay_binem_outfile:
            df_cross = collect_inter_dataset_binem_records(
                args.datasets, args.analysis_dirs, methods,
                group_a=args.group_a, group_b=args.group_b)
            plot_inter_dataset_binem(df_cross, methods,
                                     args.cross_assay_binem_outfile,
                                     cross_assay=True)


if __name__ == "__main__":
    main()
