#!/usr/bin/env python3
"""Aggregate inter-dataset kappa matrices into a cross-dataset comparison table.

For each method, reads the per-method kappa matrices produced by compare.py
(with --all-pairs --labels ds:method) and extracts the upper-triangle pairs.
Outputs one row per (method, ds_a, ds_b) with raw kappa and NOQH variants.

Usage:
    compare_inter_dataset.py --methods M1 M2 ... --indir DIR --outfile OUT.tsv
"""

import os
from types import SimpleNamespace

import pandas as pd


def _read_matrix(path):
    """Read a symmetric kappa matrix TSV; return as DataFrame or None."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", index_col=0)
    # Normalize index/column names: strip whitespace
    df.index   = df.index.str.strip()
    df.columns = df.columns.str.strip()
    return df


def _upper_pairs(df):
    """Yield (label_i, label_j, value) for the upper triangle of a square matrix."""
    labels = list(df.index)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if j > i:
                yield li, lj, df.loc[li, lj]


def run_compare_inter_dataset(methods, indir, outfile):
    """Aggregate inter-dataset kappa matrices into a comparison table.

    Direct-call entry point (the former CLI); called from analysis.ipynb.
    """
    args = SimpleNamespace(methods=methods, indir=indir, outfile=outfile)

    rows = []
    for method in args.methods:
        method_dir = os.path.join(args.indir, method)

        mats = {
            "kappa":        _read_matrix(os.path.join(method_dir, "kappa_matrix.tsv")),
            "kappa_noqh":   _read_matrix(os.path.join(method_dir, "kappa_noqh_matrix.tsv")),
            "jaccard":      _read_matrix(os.path.join(method_dir, "jaccard_similarity_matrix.tsv")),
            "jaccard_noqh": _read_matrix(os.path.join(method_dir, "jaccard_noqh_matrix.tsv")),
        }

        base_mat = mats["kappa"]
        if base_mat is None:
            print(f"  WARNING: missing kappa_matrix.tsv for {method}, skipping")
            continue

        for label_i, label_j, _ in _upper_pairs(base_mat):
            # Labels are "{ds}:{method}" — extract dataset names
            ds_a = label_i.split(":")[0]
            ds_b = label_j.split(":")[0]
            row = {"method": method, "ds_a": ds_a, "ds_b": ds_b}
            for metric, mat in mats.items():
                if mat is not None and label_i in mat.index and label_j in mat.columns:
                    row[metric] = mat.loc[label_i, label_j]
                else:
                    row[metric] = float("nan")
            rows.append(row)

    if not rows:
        print("WARNING: no data collected — output will be empty")
        pd.DataFrame().to_csv(args.outfile, sep="\t", index=False)
        return

    df = pd.DataFrame(rows)
    # Column order
    metric_cols = ["kappa", "kappa_noqh", "jaccard", "jaccard_noqh"]
    cols = ["method", "ds_a", "ds_b"] + [c for c in metric_cols if c in df.columns]
    df = df[cols].sort_values(["method", "ds_a", "ds_b"])
    df.to_csv(args.outfile, sep="\t", index=False, float_format="%.4f")
    print(f"Saved {len(df)} rows to {args.outfile}")
