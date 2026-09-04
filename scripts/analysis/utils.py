#!/usr/bin/env python3
"""Shared naming, plotting and caching helpers for the omni-chromhmm pipeline.

Method keys are {state_model}_{binarization}[_{rep}], plus "ref" for the
ENCODE reference segmentation.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

METHOD_IDX = {m: i for i, m in enumerate(METHOD_ORDER)}

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

# The Quies/Het bulk of the genome, dropped by the NOQH variant of every metric,
# where it would otherwise dominate both kappa and Jaccard.
NOQH_STATES = {"Quies", "Het"}

BIN_COLORS = {
    "default":   "#4878CF",
    "omnipeak":  "#E8833A",
    "homer":     "#2CA02C",
    "macs2":     "#9467BD",
    "reference": "#888888",
}


def parse_method(name):
    """Parse a method key into (binarization, state_model, rep)."""
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


METHOD_INFO = {m: parse_method(m) for m in METHOD_ORDER}


def display_name(method):
    return DISPLAY_NAMES.get(method, method)


def bin_color(binarization):
    """Plot color for a binarization type; the neutral grey when unknown."""
    return BIN_COLORS.get(binarization, BIN_COLORS["reference"])


def method_color(method):
    return bin_color(parse_method(method)[0])


def seg_label(path):
    """Derive a method key from a segmentation BED file path."""
    parts = path.replace("\\", "/").split("/")
    basename = os.path.basename(path)

    if basename.startswith("ENCFF"):
        return basename.replace(".bed", "")

    reps = ("rep1", "rep2", "replicate1", "replicate2")
    caller = next((p for p in parts if p in ("omni", "homer", "macs2")), None)
    rep    = next((p for p in parts if p in reps), None)
    if rep is None:
        # A joint model writes one segmentation per replicate into a folder shared
        # by them, so its replicate is in the file name: rep1_15_dense.bed.
        rep = next((r for r in reps if basename.startswith(f"{r}_")), None)
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


def is_replicate(label):
    return label.endswith("_rep1") or label.endswith("_rep2")


def should_compare(label_i, label_j):
    """True for pooled-vs-reference pairs and rep1-vs-rep2 of the same method."""
    ref_i = label_i.startswith("ENCFF")
    ref_j = label_j.startswith("ENCFF")
    if ref_i or ref_j:
        other = label_j if ref_i else label_i
        return not is_replicate(other)
    if not (is_replicate(label_i) and is_replicate(label_j)):
        return False
    return label_i[:-5] == label_j[:-5]


# Every bar chart with an error bar also shows the underlying observations.
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
    """Scatter observations on top of a matplotlib bar centred at *xpos*.

    Jitter is seeded by *xpos*, so re-running reproduces the same figure.
    """
    vals = np.asarray(np.ravel(values), dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return
    rng = np.random.default_rng(int(round(abs(xpos) * 1000)))
    offs = rng.uniform(-jitter, jitter, len(vals)) if len(vals) > 1 else np.zeros(1)
    ax.scatter(xpos + offs, vals, s=size, **_point_style(size, 8, kwargs), **kwargs)


def bar_label_y(ax, *values):
    """Y position for a value label: clear of *values* but inside the axes."""
    lo, hi = ax.get_ylim()
    top = max([v for v in values if v is not None and not pd.isna(v)], default=lo)
    return min(top + 0.01 * (hi - lo), hi - 0.05 * (hi - lo))


def strip_points(ax, jitter=0.15, size=STRIP_SIZE, dodge=True, **kwargs):
    """Overlay observations matching a sns.barplot on *ax*.

    Pass the same data/x/y/hue/order/hue_order as the barplot; use dodge=False
    when the barplot itself is not dodged (hue == x).
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


JOINT_HATCH = "//"


def _is_joint(method):
    """True for a joint model, by key (joint_omni) or display name (Joint ...)."""
    return str(method).lower().startswith("joint")


def hatch_joint(ax, order, joint=_is_joint):
    """Hatch the bars of the joint models, which share their caller's colour.

    sns.barplot draws one bar container per hue level, in hue_order order, so
    pass the same hue_order as the barplot (with x == hue, that is its order).
    Without a hue there is a single container holding one bar per x level —
    pass the barplot order instead.
    """
    groups = ax.containers
    if len(groups) == 1 and len(groups[0]) == len(order):
        groups = [[bar] for bar in groups[0]]
    for group, method in zip(groups, order):
        if joint(method):
            for bar in group:
                bar.set_hatch(JOINT_HATCH)


def hatch_all(ax):
    """Hatch every bar of *ax*, for a plot of a single joint model.

    The published 18-state and 15-state reference segmentations come from one
    model trained over every epigenome; a plot showing only such a model has no
    individual counterpart to pick out, so all of its bars carry the hatch.
    """
    for container in ax.containers:
        for bar in container:
            bar.set_hatch(JOINT_HATCH)


def save_fig(fig, path, tight=True, note=None, **kwargs):
    """Write *fig* to *path*, then close it and report the file.

    *tight* runs tight_layout() first — pass False for a figure that manages
    its own layout. Extra keywords go to fig.savefig().
    """
    if tight:
        fig.tight_layout()
    kwargs.setdefault("bbox_inches", "tight")
    _save(path, lambda p: fig.savefig(p, **kwargs))
    plt.close(fig)
    print(f"  saved {path}{' ' + note if note else ''}")


def load_matrix(path):
    """Read a seg × seg matrix TSV into a DataFrame; None when it is missing."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def cached_pickle(path, compute, label=None, valid=None):
    """Result of compute(), cached as a pickle in *path*.

    The cache is reused when it exists and *valid* — an optional predicate on
    the loaded value — accepts it; otherwise compute() runs and is written back.
    """
    what = label or os.path.basename(path)
    if os.path.exists(path):
        with open(path, "rb") as f:
            value = pickle.load(f)
        if valid is None or valid(value):
            print(f"Loaded cached {what} from {path}")
            return value
        print(f"Cached {what} in {path} no longer matches the inputs, recomputing...")
    else:
        print(f"Computing {what}...")
    value = compute()
    _save(path, lambda p: _dump_pickle(p, value))
    print(f"Saved {what} to {path}")
    return value


def cached_csv(path, compute, label=None, index=False, valid=None, **read_kwargs):
    """DataFrame returned by compute(), cached as CSV in *path*.

    Same contract as cached_pickle(); *read_kwargs* go to pd.read_csv, and its
    *sep* is written back too, so a tab-separated cache reads as it was written.
    """
    what = label or os.path.basename(path)
    if os.path.exists(path):
        df = pd.read_csv(path, **read_kwargs)
        if valid is None or valid(df):
            print(f"Loaded cached {what} from {path}")
            return df
        print(f"Cached {what} in {path} no longer matches the inputs, recomputing...")
    print(f"Computing {what}...")
    df = compute()
    sep = read_kwargs.get("sep") or ","
    _save(path, lambda p: df.to_csv(p, index=index, sep=sep))
    print(f"Saved {what} to {path}")
    return df


def file_stamp(path):
    """(mtime, size) identifying *path*, None when missing.

    A list, not a tuple, so a signature built from it survives a JSON round trip.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return [st.st_mtime_ns, st.st_size]


def stamp_current(stamp_path, signature, outputs=()):
    """True when *stamp_path* records exactly *signature* and *outputs* exist.

    For steps whose result is a set of files; any missing output invalidates the
    stamp, so deleting a result forces a rerun.
    """
    if any(not os.path.exists(o) for o in outputs):
        return False
    try:
        with open(stamp_path) as f:
            return json.load(f) == signature
    except (OSError, ValueError):
        return False


def save_stamp(stamp_path, signature):
    _save(stamp_path, lambda p: _dump_json(p, signature))


def _save(path, write):
    """Run write(path) with the parent directory of *path* in place."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write(path)


def _dump_pickle(path, value):
    with open(path, "wb") as f:
        pickle.dump(value, f)


def _dump_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f, indent=1)
