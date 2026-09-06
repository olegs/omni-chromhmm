#!/usr/bin/env python3
"""Aggregate inter-dataset kappa matrices into a cross-dataset comparison table.

Reads the per-method matrices produced by compare.py with all_pairs and
ds:method labels, and emits one row per (method, ds_a, ds_b).
"""

import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_matrix, JACCARD, KAPPA, NOQH_SUFFIX


def _upper_pairs(df):
    """Yield (label_i, label_j, value) for the upper triangle."""
    labels = list(df.index)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if j > i:
                yield li, lj, df.loc[li, lj]


def run_compare_out(methods, indir, outfile):
    """Aggregate inter-dataset kappa matrices into a comparison table."""
    args = SimpleNamespace(methods=methods, indir=indir, outfile=outfile)

    rows = []
    for method in args.methods:
        method_dir = os.path.join(args.indir, method)

        mats = {
            KAPPA:                    load_matrix(os.path.join(method_dir, f"{KAPPA}_matrix.tsv")),
            f"{KAPPA}{NOQH_SUFFIX}":   load_matrix(os.path.join(method_dir, f"{KAPPA}{NOQH_SUFFIX}_matrix.tsv")),
            JACCARD:                  load_matrix(os.path.join(method_dir, f"{JACCARD}_similarity_matrix.tsv")),
            f"{JACCARD}{NOQH_SUFFIX}": load_matrix(os.path.join(method_dir, f"{JACCARD}{NOQH_SUFFIX}_matrix.tsv")),
        }

        base_mat = mats[KAPPA]
        if base_mat is None:
            print(f"  WARNING: missing {KAPPA}_matrix.tsv for {method}, skipping")
            continue

        for label_i, label_j, _ in _upper_pairs(base_mat):
            # Labels are "{ds}:{method}".
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
    metric_cols = [KAPPA, f"{KAPPA}{NOQH_SUFFIX}",
                   JACCARD, f"{JACCARD}{NOQH_SUFFIX}"]
    cols = ["method", "ds_a", "ds_b"] + [c for c in metric_cols if c in df.columns]
    df = df[cols].sort_values(["method", "ds_a", "ds_b"])
    df.to_csv(args.outfile, sep="\t", index=False, float_format="%.4f")
    print(f"Saved {len(df)} rows to {args.outfile}")
