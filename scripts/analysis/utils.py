#!/usr/bin/env python3
"""Shared naming, plotting and caching helpers for the omni-chromhmm pipeline.

Method keys are {state_model}_{binarization}[_{rep}], plus "ref" for the
ENCODE reference segmentation.
"""

import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics, plus the two comparison domains every agreement is measured in: full
# over all states, noqh with the Quies/Het background dropped. These are the
# on-disk spelling — the keys of agreement_by_mode() and the "_noqh" file
# suffix — so they are kept lowercase and the *_DISPLAY forms below carry the
# human-readable spelling.
JACCARD = "jaccard"
KAPPA = "kappa"
COSINE = "cosine"
FULL = "full"
NOQH = "noqh"

# The cosine is the cosine of the two state-composition vectors (bp per state),
# and nothing else. It is blind to *where* the states are: it asks whether the
# two segmentations spend the genome on states in the same proportions, so it
# saturates near 1 once a background state dominates both sides, and two
# segmentations that agree nowhere still score 1.0 when their state budgets
# match. Read it next to kappa, which does look at placement.
#
# compare.py reaches the same quantity from the per-state bp vectors of two
# segment lists and writes it as "composition", which is why that spelling is
# an alias below.
COMPOSITION = "composition"

# Agreement of the state *signatures* of two segmentations rather than of their
# placement: the mean cosine of the emission vectors of the states paired up by
# match.emission_cosine_mapping(). It says whether two segmentations describe
# the genome with the same state definitions, so it stays meaningful where the
# placement metrics do not — across samples that differ biologically.
#
# Not one of the metrics normalize_metric() ranks: it is not cached next to
# jaccard/kappa/cosine, compare.py writes it per pair under this spelling.
EMISSION = "emission_similarity"

# Metric and domain display names, for plot titles, labels and legends, and for
# the columns of the notebook caches.
JACCARD_DISPLAY = "Jaccard"
KAPPA_DISPLAY = "Kappa"
COSINE_DISPLAY = "Cosine"
COMPOSITION_DISPLAY = "Composition"
EMISSION_DISPLAY = "Emission similarity"

# Overlap of a state family with an independent annotation of the same sample:
# the harmonic mean of the annotated share of the family (precision) and of the
# share of the annotation the family covers (recall). Computed from the
# per-state enrichment tables by annotation_f1(), so - like the emission
# similarity - it is not one of the metrics normalize_metric() ranks, and it has
# no lowercase on-disk spelling because nothing caches it next to
# jaccard/kappa/cosine.
#
# A monotone transform of the Jaccard of the same two sets (F1 = 2J / (1 + J)),
# so it orders the callers of one annotation exactly as the Jaccard does; it is
# read instead of the Jaccard because precision and recall are what a validation
# against an annotation is asking about.
F1_DISPLAY = "F1"

FULL_DISPLAY = "FULL"
NOQH_DISPLAY = "NOQH"

METRIC_DISPLAY = {JACCARD: JACCARD_DISPLAY, KAPPA: KAPPA_DISPLAY,
                  COSINE: COSINE_DISPLAY, COMPOSITION: COMPOSITION_DISPLAY}

# Every on-disk spelling a metric can appear under, most canonical first: the
# cosine is written as "cosine" by the notebook agreement caches and as
# "composition" by the comparison tables compare.py feeds.
METRIC_ALIASES = {COSINE: (COSINE, COMPOSITION)}
DOMAIN_DISPLAY = {FULL: FULL_DISPLAY, NOQH: NOQH_DISPLAY}

# "" for FULL, "_noqh" for NOQH: the suffix every domain-specific column, TSV
# and matrix name carries.
NOQH_SUFFIX = f"_{NOQH}"


def normalize_metric(name):
    """Normalize a metric name (jaccard, kappa, cosine) or None when unknown.

    The display spellings lowercase to the keys, so a cache column name
    round-trips back to its metric.
    """
    name = str(name).strip().lower()
    return name if name in (JACCARD, KAPPA, COSINE) else None


def metric_aliases(name):
    """The on-disk spellings of a metric, canonical first — what a cache column
    may be called. Unknown names are returned unchanged, as a single spelling."""
    name = str(name).strip().lower()
    return METRIC_ALIASES.get(name, (name,))


def normalize_domain(name):
    """Normalize a comparison domain (full, noqh) or None when unknown."""
    name = str(name).strip().lower()
    return name if name in (FULL, NOQH) else None


def metric_display(name):
    """Display spelling of a metric name, or None when unknown."""
    return METRIC_DISPLAY.get(str(name).strip().lower())


def domain_display(name):
    """Display spelling of a comparison domain, or None when unknown."""
    return DOMAIN_DISPLAY.get(str(name).strip().lower())

# Methods
CHROMHMM = "chromhmm"
HOMER = "homer"
MACS2 = "macs2"
OMNI = "omni"

# Method family display names
CHROMHMM_DISPLAY = "ChromHMM"
HOMER_DISPLAY = "HOMER"
MACS2_DISPLAY = "MACS2"
OMNI_DISPLAY = "OmniPeak"

# Interpreted state types
QUIESCENT = "Quiescent"
FACULTATIVE_HET = "FacultativeHet"
CONSTITUTIVE_HET = "ConstitutiveHet"

# Canonical keys
CHROMHMM_DEFAULT = "chromhmm_default"
CHROMHMM_HOMER = "chromhmm_homer"
CHROMHMM_MACS2 = "chromhmm_macs2"
CHROMHMM_OMNI = "chromhmm_omni"
KMEANS_HOMER = "kmeans_homer"
KMEANS_MACS2 = "kmeans_macs2"
KMEANS_OMNI = "kmeans_omni"
JOINT_CHROMHMM = "joint_chromhmm"
JOINT_KMEANS_HOMER = "joint_kmeans_homer"
JOINT_KMEANS_MACS2 = "joint_kmeans_macs2"
JOINT_KMEANS_OMNI = "joint_kmeans_omni"

CALLER_KEYS = {
    CHROMHMM: (CHROMHMM_DEFAULT, JOINT_CHROMHMM),
    HOMER:    (KMEANS_HOMER,    JOINT_KMEANS_HOMER),
    MACS2:    (KMEANS_MACS2,    JOINT_KMEANS_MACS2),
    OMNI:     (KMEANS_OMNI,     JOINT_KMEANS_OMNI),
}


def method_key(caller, joint=False):
    """Canonical key of a caller's model: its individual one, or its joint one."""
    return CALLER_KEYS[caller][1 if joint else 0]


METHOD_ORDER = [
    "ref",
    CHROMHMM_DEFAULT,
    KMEANS_HOMER,
    KMEANS_MACS2,
    KMEANS_OMNI,
    CHROMHMM_HOMER,
    CHROMHMM_MACS2,
    CHROMHMM_OMNI,
    JOINT_CHROMHMM,
    JOINT_KMEANS_HOMER,
    JOINT_KMEANS_MACS2,
    JOINT_KMEANS_OMNI,
    f"{CHROMHMM_DEFAULT}_rep1",
    f"{KMEANS_HOMER}_rep1",
    f"{KMEANS_MACS2}_rep1",
    f"{KMEANS_OMNI}_rep1",
    f"{CHROMHMM_DEFAULT}_rep2",
    f"{KMEANS_HOMER}_rep2",
    f"{KMEANS_MACS2}_rep2",
    f"{KMEANS_OMNI}_rep2",
]

METHOD_IDX = {m: i for i, m in enumerate(METHOD_ORDER)}

DISPLAY_NAMES = {
    "ref":                   "ENCODE Ref",
    CHROMHMM_DEFAULT:      "Default ChromHMM",
    CHROMHMM_OMNI:         "OmniPeak ChromHMM",
    CHROMHMM_HOMER:        "Homer ChromHMM",
    CHROMHMM_MACS2:        "MACS2 ChromHMM",
    KMEANS_OMNI:           "OmniPeak KMeans",
    KMEANS_HOMER:          "Homer KMeans",
    KMEANS_MACS2:          "MACS2 KMeans",
    JOINT_CHROMHMM:        "Joint ChromHMM",
    JOINT_KMEANS_OMNI:     "Joint OmniPeak KMeans",
    JOINT_KMEANS_HOMER:    "Joint Homer KMeans",
    JOINT_KMEANS_MACS2:    "Joint MACS2 KMeans",
    f"{CHROMHMM_DEFAULT}_rep1": "Default ChromHMM (rep1)",
    f"{KMEANS_OMNI}_rep1":      "OmniPeak KMeans (rep1)",
    f"{KMEANS_HOMER}_rep1":     "Homer KMeans (rep1)",
    f"{KMEANS_MACS2}_rep1":     "MACS2 KMeans (rep1)",
    f"{CHROMHMM_DEFAULT}_rep2": "Default ChromHMM (rep2)",
    f"{KMEANS_OMNI}_rep2":      "OmniPeak KMeans (rep2)",
    f"{KMEANS_HOMER}_rep2":     "Homer KMeans (rep2)",
    f"{KMEANS_MACS2}_rep2":     "MACS2 KMeans (rep2)",
}

def normalize_method(name):
    """Normalize a method name to its canonical key, or None when unknown."""
    name = str(name).strip().lower().replace(" ", "_").replace("omnipeak", "omni")
    if name in (CHROMHMM, "default_chromhmm", CHROMHMM_DEFAULT, "ref_15", "individual"):
        return CHROMHMM_DEFAULT
    if name in (JOINT_CHROMHMM, "joint_ref_15", "joint"):
        return JOINT_CHROMHMM
    is_joint = name.startswith("joint_")
    if is_joint:
        name = name[len("joint_"):]
    if name.startswith("kmeans_"):
        name = name[len("kmeans_"):]
    elif name.endswith("_kmeans"):
        name = name[:-len("_kmeans")]
    if name in (HOMER, MACS2, OMNI):
        if name == HOMER: canonical = KMEANS_HOMER
        elif name == MACS2: canonical = KMEANS_MACS2
        else: canonical = KMEANS_OMNI
        if is_joint:
            return "joint_" + canonical
        return canonical
    return None

# The Quies/Het bulk of the genome, dropped by the NOQH variant of every metric,
# where it would otherwise dominate both kappa and Jaccard.
NOQH_STATES = {
    "Quies", "Quiescent", "Quies_low",
    "Het", "9_Het", "13_Het",
    "15_Quies", "18_Quies",
    "8_ZNF/Rpts", "ZNF/Rpts"
}

# The same background, named in the vocabulary of interpretation.py: what the
# NOQH domain drops once the states have been interpreted from their emissions
# rather than matched to a reference by name. A comparison of two segmentations
# whose references name their states differently has to go through the types,
# because the name-based metrics count every one-sided name as a disagreement.
NOQH_TYPES = (QUIESCENT, FACULTATIVE_HET, CONSTITUTIVE_HET)

# The state families the functional validations score, over the names the
# 15-state markups use: the promoter family is Tss with its flanking states,
# the active family adds the enhancers. Biv is in neither - a bivalent promoter
# is as much repressed as active, so counting it as active would charge a
# caller for finding one.
PROMOTER_STATES = ("Tss", "TssFlnk", "TssFlnkU", "TssFlnkD")
TX_STATES = ("Tx", "TxWk")
ENHANCER_STATES = ("Enh", "Enh1", "Enh2", "EnhG", "EnhG1", "EnhG2", "EnhLo")
ACTIVE_STATES = PROMOTER_STATES + ENHANCER_STATES

# (key, label, state family, annotation label prefix) of the functional
# validations: a state family against an annotation of the same sample that
# says which loci are actually active in it, scored by annotation_f1().
#
# The RefSeq annotations are deliberately not among them - a TSS the sample
# does not transcribe is no evidence that a state placed there is wrong or
# right - and the ATAC-seq label carries the accession of the experiment
# ("atac_ENCFF243NTP"), so annotations are matched by prefix rather than by
# name.
FUNCTIONAL_TARGETS = (
    ("functional_atac", "Active chromatin vs ATAC-seq",
     ACTIVE_STATES, "atac_"),
    ("functional_tss", "Tss states vs expressed TSS \u00b12 kb",
     PROMOTER_STATES, "ExpressedTSS2kb"),
    ("functional_tx", "Tx states vs expressed gene bodies",
     TX_STATES, "ExpressedGeneBodies"),
)

# (key, label, state, annotation label prefix) of the one validation scored on
# a single state, by annotation_jaccard(): the Tx state itself against the
# expressed gene bodies, the quantity analysis_encode.ipynb plots as
# summary_jaccard_tx.png. Apart from FUNCTIONAL_TARGETS because it is neither a
# family nor an F1 - swapping the metric on a family would reorder nothing, so
# what this adds over ("functional_tx", ...) above is the narrower state set:
# Tx alone, without the TxWk the callers disagree most about.
FUNCTIONAL_TX_JACCARD = ("functional_tx_jaccard",
                         "Tx state vs expressed gene bodies",
                         "Tx", "ExpressedGeneBodies")


def annotation_f1(dirpath, states, prefix):
    """F1 of the union of `states` against every annotation `prefix` names.

    Read off the enrichment and report tables of one segmentation in
    `dirpath` ({analysis_dir}/{method}), as [{"Label", F1_DISPLAY}, ...] - one
    entry per matching annotation, empty when either table is missing or no
    annotation matches.

    The enrichment table holds, per state and annotation, the annotated share
    of the state (coverage) and that share over the share of the genome the
    annotation takes (fold_enrichment), so the annotation size comes back as
    coverage / fold_enrichment x genome - as a median over the states that
    overlap it at all, since a state with no overlap has both at 0.

    The family is scored as one set: its bp are summed and so are its overlaps,
    rather than averaging the per-state numbers, which would let a tiny state
    that happens to sit inside the annotation outweigh the family's bulk.
    """
    enrichment = os.path.join(dirpath, "enrichment", "enrichment.tsv")
    report = os.path.join(dirpath, "report.tsv")
    if not (os.path.exists(enrichment) and os.path.exists(report)):
        return []
    enrich = pd.read_csv(enrichment, sep="\t")
    sizes = pd.read_csv(report, sep="\t").set_index("state")["total_bp"]
    genome = sizes.sum()

    rows = []
    for label, group in enrich.groupby("label"):
        if not str(label).startswith(prefix):
            continue
        positive = group[group["fold_enrichment"] > 0]
        if positive.empty:
            continue
        annotation = float(np.median(positive["coverage"]
                                     / positive["fold_enrichment"])) * genome
        family = group[group["state"].isin(states)]
        family_bp = float(sizes.reindex(family["state"]).sum())
        if annotation <= 0 or family_bp <= 0:
            continue
        overlap = float((family["coverage"].values
                         * sizes.reindex(family["state"]).values).sum())
        rows.append({"Label": str(label),
                     F1_DISPLAY: 2 * overlap / (family_bp + annotation)})
    return rows


def annotation_jaccard(dirpath, state, prefix):
    """Jaccard of a single `state` against every annotation `prefix` names.

    Read off enrichment/jaccard.tsv of one segmentation in `dirpath`
    ({analysis_dir}/{method}), as [{"Label", JACCARD_DISPLAY}, ...] - one entry
    per matching annotation, empty when the table is missing or no annotation
    matches.

    One state rather than a family, and the exact bp Jaccard
    analyze.compute_enrichment() wrote rather than a number recovered from the
    enrichment table: a family Jaccard would need the union of its states,
    which the per-state table cannot give, and it would in any case order the
    callers exactly as annotation_f1() already does (F1 = 2J / (1 + J)).
    """
    path = os.path.join(dirpath, "enrichment", "jaccard.tsv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, sep="\t")
    hits = df[(df["state"].astype(str) == state)
              & df["label"].astype(str).str.startswith(prefix)]
    return [{"Label": str(label), JACCARD_DISPLAY: float(value)}
            for label, value in zip(hits["label"], hits["jaccard"])]

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


def caller_key(method):
    """The caller family (chromhmm, omni, homer, macs2) for a method key."""
    binarization = parse_method(method)[0]
    return {"default": CHROMHMM, "omnipeak": OMNI}.get(binarization, binarization)


def display_name(method):
    return DISPLAY_NAMES.get(method, method)


def slug(name):
    """Filename-safe form of a display name: "Joint ChromHMM" -> joint_chromhmm."""
    return str(name).lower().replace(" ", "_")


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


# Every bar chart of the notebooks is the same figure: mean +- SE bars in the
# style above, the observations on top of them, the joint models hatched, and
# the mean of each bar written inside the axes. bar_plot() draws it, and the
# two variants below cover the composition plots, whose Quiescent state needs
# either a broken axis or a stack.
BAR_STYLE = dict(capsize=0.05, errorbar="se", err_kws={"linewidth": 2.0},
                 edgecolor="lightgrey", linewidth=1)
TITLE_STYLE = dict(fontsize=11, fontweight="bold")
AXIS_FONTSIZE = 9
TICK_FONTSIZE = 8


def _bars(ax, data, x, y, order, hue, hue_order, palette, color, hatch, points,
          bar_kwargs):
    """One sns.barplot in the shared style, with its points and hatching.

    Bars are dodged when *hue* names a second variable; without it the colour
    follows *x*, which seaborn wants spelled as an undodged hue.
    """
    dodge = hue is not None and hue != x
    plot_hue = hue if hue is not None else (x if palette is not None else None)
    plot_order = hue_order if dodge else (order if plot_hue is not None else None)
    style = dict(BAR_STYLE, **bar_kwargs)
    sns.barplot(data=data, x=x, y=y, hue=plot_hue, order=order,
                hue_order=plot_order, palette=palette if plot_hue else None,
                color=color, dodge=dodge, legend=dodge, ax=ax, **style)
    strip_points(ax, data=data, x=x, y=y,
                 **({"hue": hue, "hue_order": hue_order} if dodge else {}),
                 order=order, dodge=dodge, **(points or {"size": 2}))
    if hatch == "joint":
        hatch_joint(ax, plot_order or order)
    elif hatch == "all":
        hatch_all(ax)


def _bar_legend(ax, levels, legend, title, kwargs):
    """Restyle the barplot legend outside the axes, or drop it."""
    if not legend:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        return
    opts = dict(title=title, fontsize=8, title_fontsize=9,
                bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    opts.update(kwargs or {})
    labels = [display_name(level) for level in levels]
    # One bar of every group as the handle, so the swatch is the patch itself -
    # its colour and its hatch. Labelled handles alone give a line per entry.
    bars = [group[0] for group in ax.containers if len(group)]
    if len(bars) == len(labels):
        ax.legend(bars, labels, **opts)
    else:
        ax.legend(labels=labels, **opts)


def _xticklabels(order, xticklabels):
    """Tick labels for *order*: as given, display names, or the levels."""
    if xticklabels == "display":
        return [display_name(level) for level in order]
    return order if xticklabels is None else xticklabels


def bar_labels(ax, data, x, y, order, fmt="{:.2f}", fontsize=6):
    """Write the mean of every bar of *order* above it, inside the axes."""
    for i, level in enumerate(order):
        vals = pd.to_numeric(data.loc[data[x] == level, y], errors="coerce").dropna()
        if vals.empty:
            continue
        mean, err = vals.mean(), vals.sem()
        ax.text(i, bar_label_y(ax, mean + (0 if pd.isna(err) else err), vals.max()),
                fmt.format(mean), ha="center", va="bottom", fontsize=fontsize)


def bar_plot(data, x, y, order=None, hue=None, hue_order=None, palette=None,
             color=None, ax=None, figsize=(6, 4.2), title=None, xlabel="",
             ylabel=None, xticklabels=None, rotation=45, tick_fontsize=TICK_FONTSIZE,
             ylim=None, log=False, labels=None, label_fontsize=6, hatch="joint",
             legend=False, legend_title="Method", legend_kwargs=None, points=None,
             path=None, **bar_kwargs):
    """Bar chart of *y* per *x* level, mean +- SE with the observations on top.

    *order* fixes the x levels (defaults to their order of appearance) and
    *palette* colours them; pass *hue* / *hue_order* instead for grouped bars,
    or *color* for a single-colour chart. *xticklabels* replaces the tick
    labels - "display" for display_name() of *order*. *labels* is a format
    string for the per-bar mean, which only makes sense without a *hue*, where
    one bar is one group of values. *hatch* is "joint" for hatch_joint(), "all"
    for hatch_all(), None for neither, and *points* overrides the
    strip_points() keywords. Extra keywords go to sns.barplot().

    Writes the figure to *path* and closes it, or returns the axes when *path*
    is None; pass *ax* to draw into an existing figure instead.
    """
    if order is None:
        order = list(pd.unique(data[x]))
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    _bars(ax, data, x, y, order, hue, hue_order, palette, color, hatch, points,
          bar_kwargs)

    if title:
        ax.set_title(title, **TITLE_STYLE)
    ax.set_xlabel(xlabel or "", fontsize=AXIS_FONTSIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AXIS_FONTSIZE)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(_xticklabels(order, xticklabels), rotation=rotation,
                       ha="right" if rotation else "center", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    if log and (pd.to_numeric(data[y], errors="coerce") > 0).any():
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    _bar_legend(ax, hue_order or order, legend, legend_title, legend_kwargs)
    if labels:
        if ylim is None and not log:
            # Headroom for the labels, which bar_label_y() keeps inside the axes
            # and would otherwise write over the tallest bar.
            low, high = ax.get_ylim()
            ax.set_ylim(low, high + 0.07 * (high - low))
        # After the y limits: a label is placed relative to them.
        bar_labels(ax, data, x, y, order, fmt=labels, fontsize=label_fontsize)
    if path is not None and fig is not None:
        save_fig(fig, path)
        return None
    return ax


def broken_bar_plot(data, x, y, order, break_low=0.20, break_high=0.40, top=1.02,
                    height_ratios=(1, 4), figsize=(12, 6), title=None, xlabel=None,
                    ylabel=None, xticklabels=None, rotation=45,
                    tick_fontsize=TICK_FONTSIZE, path=None, **kwargs):
    """bar_plot() with the y axis broken between *break_low* and *break_high*.

    What the state composition plots need: the Quiescent state covers more than
    half of the genome, and on a shared 0..1 axis it flattens every other state
    to nothing. The two axes hold the same bars over the two y ranges, and the
    break is marked by the hidden inner spines plus a pair of diagonals.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": list(height_ratios), "hspace": 0.06})
    for ax in (ax_top, ax_bot):
        bar_plot(data, x, y, order=order, ax=ax, rotation=rotation,
                 tick_fontsize=tick_fontsize, xticklabels=xticklabels,
                 legend=kwargs.get("legend", False) and ax is ax_top, **{
                     k: v for k, v in kwargs.items() if k != "legend"})
    # The break is what carries the scale, so the ranges come after the bars.
    ax_top.set_ylim(break_high, top)
    ax_bot.set_ylim(0, break_low)
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False)

    d = 0.012   # half-length of a break mark, in figure coordinates
    for ax, sign in ((ax_top, -1), (ax_bot, 1)):
        pos = ax.get_position()
        y_fig = pos.y0 if sign == 1 else pos.y1
        for x_fig in (pos.x0, pos.x1):
            fig.add_artist(plt.Line2D([x_fig - d, x_fig + d],
                                      [y_fig + sign * d * 1.5, y_fig - sign * d * 1.5],
                                      transform=fig.transFigure, color="k",
                                      clip_on=False, linewidth=0.8))

    if title:
        ax_top.set_title(title, **TITLE_STYLE)
    ax_top.set_xlabel("")
    ax_bot.set_xlabel(xlabel or "", fontsize=AXIS_FONTSIZE)
    ax_top.set_ylabel("")
    ax_bot.set_ylabel(ylabel or "", fontsize=AXIS_FONTSIZE)
    if path is not None:
        # tight_layout() would move the axes the break marks are placed against.
        save_fig(fig, path, tight=False)
        return None
    return ax_top, ax_bot


def stacked_bar_plot(pivot, colors=None, figsize=(15, 6), width=0.8, title=None,
                     xlabel=None, ylabel=None, xticklabels=None, rotation=90,
                     tick_fontsize=6, legend_title="State",
                     legend_fontsize="x-small", path=None):
    """Stacked bars of a fraction table, with the row totals outlined.

    *pivot* is indexed by the bars (dataset or method) and its columns are the
    components, stacked in column order and coloured by *colors*. The outline
    of the row sums is what shows how much of the genome a segmentation covers
    at all, which the stack alone hides.
    """
    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="bar", stacked=True, ax=ax, width=width, color=colors, linewidth=0)
    # Drawn after the stack so ax.legend() below picks up the components only.
    pivot.sum(axis=1).plot(kind="bar", ax=ax, width=width, facecolor="none",
                           edgecolor="lightgrey", linewidth=1, legend=False)
    if title:
        ax.set_title(title, **TITLE_STYLE)
    ax.legend(title=legend_title, fontsize=legend_fontsize,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xlabel(xlabel or "", fontsize=AXIS_FONTSIZE)
    ax.set_ylabel(ylabel or "", fontsize=AXIS_FONTSIZE)
    if xticklabels is not None:
        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels(xticklabels, rotation=rotation,
                           ha="right" if rotation not in (0, 90) else "center",
                           fontsize=tick_fontsize)
    else:
        ax.tick_params(axis="x", rotation=rotation, labelsize=tick_fontsize)
    ax.grid(axis="y", alpha=0.3)
    if path is not None:
        save_fig(fig, path)
        return None
    return ax


def sample_pairs(pairs, limit=1000, seed=42):
    """At most *limit* of *pairs*, sampled reproducibly and sorted.

    The pairwise consistency of a hundred segmentations is tens of thousands of
    pairs, which the notebooks read off a sample of. Sorting the sample keeps
    it lined up with a cached list of overlaps computed from the same call.
    """
    pairs = list(pairs)
    if len(pairs) <= limit:
        return pairs
    return sorted(random.Random(seed).sample(pairs, limit))


def keyed_cache(path, keys, compute, label=None, valid=None, progress=None):
    """{key: compute(key)} over *keys*, backed by a {key: value} pickle.

    Only the keys the cache does not hold - or holds a value *valid* rejects -
    are computed, and the file is rewritten only when some were, which is what
    makes the per-sample and per-pair tables of the notebooks resumable across
    runs. *progress* wraps the missing keys: pass tqdm to show a bar.
    """
    what = label or os.path.basename(path)
    cache = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)
        if valid is not None:
            stale = [key for key, value in cache.items() if not valid(value)]
            if stale:
                print(f"Dropping {len(stale)} cached {what} entries that no "
                      f"longer match the inputs")
                for key in stale:
                    del cache[key]
        print(f"Loaded {len(cache)} cached {what} entries from {path}")
    missing = [key for key in keys if key not in cache]
    if missing:
        print(f"Computing {what} for {len(missing)} of {len(keys)} keys...")
        for key in (progress(missing) if progress else missing):
            cache[key] = compute(key)
        _save(path, lambda p: _dump_pickle(p, cache))
        print(f"Saved {what} to {path}")
    return {key: cache[key] for key in keys}


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


def cached_json(path, compute, label=None, valid=None):
    """Result of compute(), cached as JSON in *path*.

    Same contract as cached_pickle(), for a value a plain JSON object holds -
    the state colour maps of the notebooks.
    """
    what = label or os.path.basename(path)
    if os.path.exists(path):
        with open(path) as f:
            value = json.load(f)
        if valid is None or valid(value):
            print(f"Loaded cached {what} from {path}")
            return value
        print(f"Cached {what} in {path} no longer matches the inputs, recomputing...")
    else:
        print(f"Computing {what}...")
    value = compute()
    _save(path, lambda p: _dump_json(p, value))
    print(f"Saved {what} to {path}")
    return value


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
