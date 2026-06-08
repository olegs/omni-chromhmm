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
"""

import os

# ---------------------------------------------------------------------------
# Canonical method registry
# ---------------------------------------------------------------------------

# Ordered list of all known method keys.
METHOD_ORDER = [
    "ref",
    "chromhmm_default",
    "chromhmm_omni",   "kmeans_omni",
    "chromhmm_homer",  "kmeans_homer",
    "chromhmm_macs2",  "kmeans_macs2",
    "chromhmm_default_rep1",
    "chromhmm_omni_rep1",  "kmeans_omni_rep1",
    "chromhmm_homer_rep1", "kmeans_homer_rep1",
    "chromhmm_macs2_rep1", "kmeans_macs2_rep1",
    "chromhmm_default_rep2",
    "chromhmm_omni_rep2",  "kmeans_omni_rep2",
    "chromhmm_homer_rep2", "kmeans_homer_rep2",
    "chromhmm_macs2_rep2", "kmeans_macs2_rep2",
]

# {method: rank} — for deterministic sorting.
METHOD_IDX = {m: i for i, m in enumerate(METHOD_ORDER)}

# Human-readable display labels (used on plot axes).
DISPLAY_NAMES = {
    "ref":                   "ENCODE Ref",
    "chromhmm_default":      "Default ChromHMM",
    "chromhmm_omni":         "OmniPeak ChromHMM",
    "chromhmm_homer":        "Homer ChromHMM",
    "kmeans_omni":           "OmniPeak KMeans",
    "kmeans_homer":          "Homer KMeans",
    "chromhmm_macs2":        "MACS2 ChromHMM",
    "kmeans_macs2":          "MACS2 KMeans",
    "chromhmm_default_rep1": "Default ChromHMM (rep1)",
    "chromhmm_omni_rep1":    "OmniPeak ChromHMM (rep1)",
    "chromhmm_homer_rep1":   "Homer ChromHMM (rep1)",
    "kmeans_omni_rep1":      "OmniPeak KMeans (rep1)",
    "kmeans_homer_rep1":     "Homer KMeans (rep1)",
    "chromhmm_macs2_rep1":   "MACS2 ChromHMM (rep1)",
    "kmeans_macs2_rep1":     "MACS2 KMeans (rep1)",
    "chromhmm_default_rep2": "Default ChromHMM (rep2)",
    "chromhmm_omni_rep2":    "OmniPeak ChromHMM (rep2)",
    "chromhmm_homer_rep2":   "Homer ChromHMM (rep2)",
    "kmeans_omni_rep2":      "OmniPeak KMeans (rep2)",
    "kmeans_homer_rep2":     "Homer KMeans (rep2)",
    "chromhmm_macs2_rep2":   "MACS2 ChromHMM (rep2)",
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
      rep          : rep1 | rep2 | None
    Special case: "ref" → ("reference", "chromhmm", None)
    """
    if name == "ref":
        return "reference", "chromhmm", None
    parts = name.split("_")
    rep = parts[-1] if parts[-1] in ("rep1", "rep2") else None
    core = parts[:-1] if rep else parts
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
    rep    = next((p for p in parts if p in ("rep1", "rep2")), None)

    if "kmeans_states" in basename:
        model = f"kmeans_{caller}" if caller else "kmeans"
    elif "chromhmm_default_result" in parts:
        model = "chromhmm_default"
    elif "chromhmm_result" in parts and caller:
        model = f"chromhmm_{caller}"
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
