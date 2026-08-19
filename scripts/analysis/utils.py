#!/usr/bin/env python3
"""Shared segmentation method naming for the omni-chromhmm pipeline.

Method keys follow the structured naming convention used throughout the pipeline
(analyze.smk analysis dirs, compare.py labels, compare_methods.py registry):

  {state_model}_{binarization}[_{rep}]
  state_model   : chromhmm | kmeans
  binarization  : default | omni | homer
  rep           : rep1 | rep2  (optional)
  special       : ref  (ENCODE reference segmentation)

Public API
----------
METHOD_ORDER     : canonical ordered list of all known method keys
METHOD_IDX       : {method: rank} for sorting
DISPLAY_NAMES    : human-readable plot labels
BIN_COLORS       : {binarization: hex color}
METHOD_INFO      : {method: (binarization, state_model, rep)} — pre-built
parse_method(name)            → (binarization, state_model, rep)
display_name(method)          → str
seg_label(path)               → method key derived from a BED file path
is_replicate(label)           → bool
should_compare(label_i, label_j) → bool
scatter_points(ax, x, values) → overlay individual points on a matplotlib bar
strip_points(ax, ...)         → overlay individual points on a sns.barplot
bar_label_y(ax, *values)      → y for a value label, kept inside the axes
"""

import os

import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Canonical method registry
# ---------------------------------------------------------------------------

# Ordered list of all known method keys.
METHOD_ORDER = [
    "ref",
    "chromhmm_default",
    "kmeans_homer",
    "kmeans_macs2",
    "kmeans_omni",
    "joint_chromhmm",
    "joint_kmeans_homer",
    "joint_kmeans_macs2",
    "joint_kmeans_omni",
    "chromhmm_default_rep1",
    "kmeans_homer_rep1",
    "kmeans_macs2_rep1",
    "kmeans_omni_rep1",
    "chromhmm_default_rep2",
    "kmeans_homer_rep2",
    "kmeans_macs2_rep2",
    "kmeans_omni_rep2",
]

# {method: rank} — for deterministic sorting.
METHOD_IDX = {m: i for i, m in enumerate(METHOD_ORDER)}

# Human-readable display labels (used on plot axes).
DISPLAY_NAMES = {
    "ref":                   "ENCODE Ref",
    "chromhmm_default":      "Default ChromHMM",
    "kmeans_omni":           "OmniPeak KMeans",
    "kmeans_homer":          "Homer KMeans",
    "kmeans_macs2":          "MACS2 KMeans",
    "joint_chromhmm":        "Joint ChromHMM",
    "joint_kmeans_omni":     "Joint OmniPeak KMeans",
    "joint_kmeans_homer":    "Joint Homer KMeans",
    "joint_kmeans_macs2":    "Joint MACS2 KMeans",
    "chromhmm_default_rep1": "Default ChromHMM (rep1)",
    "kmeans_omni_rep1":      "OmniPeak KMeans (rep1)",
    "kmeans_homer_rep1":     "Homer KMeans (rep1)",
    "kmeans_macs2_rep1":     "MACS2 KMeans (rep1)",
    "chromhmm_default_rep2": "Default ChromHMM (rep2)",
    "kmeans_omni_rep2":      "OmniPeak KMeans (rep2)",
    "kmeans_homer_rep2":     "Homer KMeans (rep2)",
    "kmeans_macs2_rep2":     "MACS2 KMeans (rep2)",
}

# Colors keyed by binarization type.
BIN_COLORS = {
    "default":   "#4878CF",
    "omnipeak":  "#E8833A",
    "homer":     "#2CA02C",
    "macs2":     "#9467BD",
    "reference": "#888888",
}


# ---------------------------------------------------------------------------
# Method parsing
# ---------------------------------------------------------------------------

def parse_method(name):
    """Parse a structured method key into (binarization, state_model, rep).

    Method key format: {state_model}_{binarization}[_{rep}]
      state_model  : chromhmm | kmeans
      binarization : default | omni → omnipeak | homer
      rep          : rep1 | rep2 | replicate1 | replicate2 | None
    Special case: "ref" → ("reference", "chromhmm", None)
    """
    if name == "ref":
        return "reference", "chromhmm", None
    parts = name.split("_")
    rep = parts[-1] if parts[-1] in ("rep1", "rep2", "replicate1", "replicate2") else None
    if rep and rep.startswith("replicate"):
        rep = "rep" + rep[len("replicate"):]
    core = parts[:-1] if rep else parts

    if name.startswith("joint_chromhmm"):
        return "default", "joint_chromhmm", rep
    if name.startswith("joint_kmeans"):
        binarization = core[2] if len(core) > 2 else "default"
        if binarization == "omni": binarization = "omnipeak"
        return binarization, "joint_kmeans", rep

    state_model = core[0]
    binarization_key = core[1] if len(core) > 1 else ""
    if binarization_key == "default":
        binarization = "default"
    elif binarization_key == "omni":
        binarization = "omnipeak"
    elif binarization_key == "homer":
        binarization = "homer"
    elif binarization_key == "macs2":
        binarization = "macs2"
    else:
        binarization = "default"
    return binarization, state_model, rep


# Pre-built info for all known methods: method → (binarization, state_model, rep).
METHOD_INFO = {m: parse_method(m) for m in METHOD_ORDER}


def display_name(method):
    """Return the human-readable display label for a method key."""
    return DISPLAY_NAMES.get(method, method)


# ---------------------------------------------------------------------------
# BED path → method key
# ---------------------------------------------------------------------------

def seg_label(path):
    """Derive structured method key from a segmentation BED file path.

    Maps known path patterns to analysis-dir-compatible labels:
      chromhmm_default[_rep]  — default ChromHMM binarization
      chromhmm_omni[_rep]     — OmniPeak ChromHMM
      chromhmm_homer[_rep]    — Homer ChromHMM
      kmeans_omni[_rep]       — OmniPeak KMeans
      kmeans_homer[_rep]      — Homer KMeans
      ENCFF...                — reference (kept as-is)
    """
    parts = path.replace("\\", "/").split("/")
    basename = os.path.basename(path)

    if basename.startswith("ENCFF"):
        return basename.replace(".bed", "")

    caller = next((p for p in parts if p in ("omni", "homer", "macs2")), None)
    rep    = next((p for p in parts if p in ("rep1", "rep2", "replicate1", "replicate2")), None)
    if rep and rep.startswith("replicate"):
        rep = "rep" + rep[len("replicate"):]

    if "kmeans_states" in basename:
        model = f"kmeans_{caller}" if caller else "kmeans"
    elif "joint_kmeans" in parts or "joint_kmeans" in basename:
        model = f"joint_kmeans_{caller}" if caller else "joint_kmeans"
    elif "joint_chromhmm" in parts or "joint_chromhmm" in basename:
        model = "joint_chromhmm"
    elif "chromhmm_default_result" in parts:
        model = "chromhmm_default"
    else:
        model = basename.replace(".bed", "").replace("_matched", "")

    if rep:
        model = f"{model}_{rep}"
    return model


# ---------------------------------------------------------------------------
# Pair filtering helpers
# ---------------------------------------------------------------------------

def is_replicate(label):
    """True if the label belongs to a replicate segmentation (_rep1 or _rep2)."""
    return label.endswith("_rep1") or label.endswith("_rep2")


def should_compare(label_i, label_j):
    """True if this pair of segmentations should be compared.

    Two classes of valid comparisons:
      1. Pooled segmentation vs ENCODE reference (replicates excluded from ref comparison).
      2. Rep1 vs rep2 of the *same* method.
    """
    ref_i = label_i.startswith("ENCFF")
    ref_j = label_j.startswith("ENCFF")
    if ref_i or ref_j:
        other = label_j if ref_i else label_i
        return not is_replicate(other)
    if not (is_replicate(label_i) and is_replicate(label_j)):
        return False
    return label_i[:-5] == label_j[:-5]  # strip "_rep1" / "_rep2" (5 chars)


# ---------------------------------------------------------------------------
# Individual data points on top of bars with error bars
# ---------------------------------------------------------------------------

# Every bar chart that shows an error bar also shows the underlying observations,
# so the spread behind the mean stays visible.
POINT_STYLE = dict(color="#333333", alpha=0.75, linewidth=0.3, edgecolor="white",
                   zorder=5)
POINT_SIZE = 12   # matplotlib scatter marker area
STRIP_SIZE = 2    # seaborn stripplot marker diameter
_STYLE_KEYS = ("color", "alpha", "linewidth", "edgecolor", "zorder")


def _point_style(size, small, kwargs):
    """POINT_STYLE with per-call overrides pulled out of *kwargs*."""
    style = dict(POINT_STYLE)
    if size < small:
        style["linewidth"] = 0   # a white outline would swallow tiny markers
    for key in _STYLE_KEYS:
        if key in kwargs:
            style[key] = kwargs.pop(key)
    return style


def scatter_points(ax, xpos, values, jitter=0.08, size=POINT_SIZE, **kwargs):
    """Scatter individual observations on top of a matplotlib bar centred at *xpos*.

    Jitter comes from an RNG seeded by *xpos*, so each bar gets its own pattern
    and re-running reproduces the same figure. Style keys (color, alpha, ...)
    may be overridden per call.
    """
    vals = np.asarray(np.ravel(values), dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return
    rng = np.random.default_rng(int(round(abs(xpos) * 1000)))
    offs = rng.uniform(-jitter, jitter, len(vals)) if len(vals) > 1 else np.zeros(1)
    ax.scatter(xpos + offs, vals, s=size, **_point_style(size, 8, kwargs), **kwargs)


def bar_label_y(ax, *values):
    """Y position for a value label: clear of *values* but inside the axes.

    Keeps the number readable on crowded bars, where the label would otherwise
    end up on top of the point cloud or off the figure.
    """
    lo, hi = ax.get_ylim()
    top = max([v for v in values if v is not None and not pd.isna(v)], default=lo)
    return min(top + 0.01 * (hi - lo), hi - 0.05 * (hi - lo))


def strip_points(ax, jitter=0.15, size=STRIP_SIZE, dodge=True, **kwargs):
    """Overlay individual observations matching a sns.barplot on *ax*.

    Pass the same data/x/y/hue/order/hue_order as the barplot; use dodge=False
    when the barplot itself is not dodged (hue == x). Style keys (color, alpha,
    ...) may be overridden per call — crowded bars (hundreds of observations)
    stay readable with a smaller size and lower alpha.
    """
    style = _point_style(size, 2.5, kwargs)
    hue = kwargs.get("hue")
    if hue is not None:
        # With a hue, color= would build a gradient palette (one shade per
        # level); a flat palette keeps every point the same neutral colour.
        levels = kwargs.get("hue_order")
        if levels is None:
            data = kwargs.get("data")
            levels = pd.unique(data[hue] if data is not None else hue)
        color = style.pop("color")
        kwargs["palette"] = {lvl: color for lvl in levels}
    sns.stripplot(ax=ax, dodge=dodge, jitter=jitter, size=size, legend=False,
                  **style, **kwargs)
