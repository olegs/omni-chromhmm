#!/usr/bin/env python3
"""Interpretation of anonymous segmentation states as chromatin state types.
Described in https://genome.cshlp.org/content/genome/suppl/2024/04/16/gr.278343.123.DC1/Supplemental_Document.pdf

Emissions are rescaled per mark by the strongest state of the same segmentation,
which makes ChromHMM emissions and near-binary k-means centroids comparable

Required files:
  bin_emissions/state_emissions.tsv  presence of each mark in the state
  enrichment/enrichment.tsv          fold enrichment in genomic annotations
  report.tsv                         genome fraction, segment lengths
  entropy/transition_matrix.tsv      self-transition probability

"""

import os
import re

import numpy as np
import pandas as pd

# The types table ships next to this module; pass a path to use another copy.
DEFAULT_INTERPRETATION_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "interpretation.csv")


def load_interpretation_table(path=DEFAULT_INTERPRETATION_CSV):
    """Chromatin state types: index = type, columns Short Name / Description."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", comment="#")
    df.columns = [c.strip() for c in df.columns]
    df["Description"] = df["Description"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df.set_index("Chromatin State Type")


# Roadmap/ENCODE-like color per interpreted type.
TYPE_COLORS = {
    "Promoter":         "#FF0000",
    "PromoterFlanking": "#FF4500",
    "Enhancer":         "#FFDF00",
    "EnhancerLow":      "#FFF4B0",
    "Bivalent":         "#CD5C5C",
    "CTCF":             "#00B0F0",
    "Transcribed":      "#008000",
    "K9K36":            "#68CDAA",
    "FacultativeHet":   "#8937DF",
    "ConstitutiveHet":  "#4B0082",
    "Quiescent":        "#DCDCDC",
}

# Marks defining each type. The marks named in the types table carry weight 1,
# marks that merely co-occur with them support the call with a smaller weight.
MARK_GROUPS = {
    "prom": {"H3K4me3": 1.0, "H3K9ac": 0.8, "H3K4me2": 0.4, "H2AFZ": 0.3},
    "enh":  {"H3K27ac": 1.0, "H3K4me1": 1.0},
    "tx":   {"H3K36me3": 1.0, "H3K79me2": 0.4, "H4K20me1": 0.3},
    "facu": {"H3K27me3": 1.0},
    "cons": {"H3K9me3": 1.0, "H3K9me2": 0.4},
    "ctcf": {"CTCF": 1.0},
}
# Annotations supporting promoters (TSS proximity) and transcription (gene body).
TSS_ANNOTATIONS = ["RefSeqTSS", "RefSeqTSS2kb", "CpGIsland"]
GENE_ANNOTATIONS = ["RefSeqGene", "RefSeqExon", "RefSeqTES"]

# Thresholds, all on 0..1 scores (see state_evidence for the normalization).
MIN_SCALE = 0.20     # a mark never reaching this in any state is weak everywhere
QUIES_MAX = 0.15     # "lack of any marks": no mark above this, relative and raw
STRONG = 0.50        # mark group counts as present
BIVA_FACU = 0.35     # H3K27me3 level required next to an activating mark
CTCF_OTHER = 0.30    # CTCF states carry no other mark above this
TIER_RATIO = 0.60    # below 60% of the strongest state of the family -> low tier
TIER_FLOOR = 0.50    # ... and never call a state Promoter/Enhancer below this
FE_REF_TSS = 20.0    # fold enrichment mapped to 1.0 (log scale)
FE_REF_GENE = 2.5    # fold enrichment mapped to 1.0 (linear scale)
UNCLASSIFIED = 0.05  # no family reaches this -> Quiescent


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

def load_segmentation_evidence(segdir):
    """Load every per-state evidence table of one segmentation directory.

    Returns None when emissions are missing, i.e. the segmentation was not
    analyzed; the other tables are optional and default to neutral values.
    """
    em_path = os.path.join(segdir, "bin_emissions", "state_emissions.tsv")
    if not os.path.exists(em_path):
        return None
    emissions = pd.read_csv(em_path, sep="\t", index_col=0)
    emissions.index = emissions.index.astype(str)

    enrichment = None
    en_path = os.path.join(segdir, "enrichment", "enrichment.tsv")
    if os.path.exists(en_path):
        df = pd.read_csv(en_path, sep="\t")
        df["state"] = df["state"].astype(str)
        df["label"] = df["label"].str.replace(r"\.[^.]+$", "", regex=True)  # drop .hg38
        enrichment = df.pivot_table(index="state", columns="label", values="fold_enrichment")

    report = None
    rp_path = os.path.join(segdir, "report.tsv")
    if os.path.exists(rp_path):
        report = pd.read_csv(rp_path, sep="\t", index_col=0)
        report.index = report.index.astype(str)

    self_transition = None
    tr_path = os.path.join(segdir, "entropy", "transition_matrix.tsv")
    if os.path.exists(tr_path):
        tm = pd.read_csv(tr_path, sep="\t", index_col=0)
        tm.index, tm.columns = tm.index.astype(str), tm.columns.astype(str)
        self_transition = pd.Series({s: tm.at[s, s] for s in tm.index if s in tm.columns})

    return {"emissions": emissions, "enrichment": enrichment,
            "report": report, "self_transition": self_transition}


def _nanmax(values):
    """Max ignoring NaN; 0.0 when nothing is available (mark not measured)."""
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return max(vals) if vals else 0.0


def _fe_norm_log(fe, ref):
    """Fold enrichment -> 0..1, log scale (1x -> 0, ref -> 1)."""
    return np.clip(np.log10(np.clip(fe, 1.0, None)) / np.log10(ref), 0, 1)


def _fe_norm_lin(fe, ref):
    """Fold enrichment -> 0..1, linear scale (1x -> 0, ref -> 1)."""
    return np.clip((fe - 1.0) / (ref - 1.0), 0, 1)


def state_evidence(ev):
    """Per-state evidence table: mark group scores, annotation scores, statistics.

    Emissions are scaled per mark by the strongest state of the same
    segmentation, which makes the scores comparable across methods with very
    different binarizations (ChromHMM emissions vs near-binary k-means
    centroids). A mark whose maximum stays below MIN_SCALE is weak in every
    state and is not rescaled up.
    """
    em = ev["emissions"]
    raw = em.to_numpy(float)
    scale = np.maximum(np.nanmax(raw, axis=0), MIN_SCALE)
    rel = pd.DataFrame(np.clip(raw / scale, 0, 1), index=em.index, columns=em.columns)

    out = pd.DataFrame(index=em.index)
    for group, weights in MARK_GROUPS.items():
        avail = {m: w for m, w in weights.items() if m in rel.columns}
        if avail:
            w = pd.Series(avail)
            out[group] = (rel[list(avail)] * w).sum(axis=1) / w.sum()
        else:
            out[group] = np.nan            # mark not measured in this dataset
    # Single marks the type definitions name explicitly.
    out["k27ac"] = rel["H3K27ac"] if "H3K27ac" in rel.columns else np.nan
    out["k4me3"] = rel["H3K4me3"] if "H3K4me3" in rel.columns else np.nan
    out["max_mark"] = rel.max(axis=1)
    out["max_raw"] = em.max(axis=1)

    enr = ev["enrichment"]
    for name, labels, norm, ref in (("tss_ann", TSS_ANNOTATIONS, _fe_norm_log, FE_REF_TSS),
                                    ("gene_ann", GENE_ANNOTATIONS, _fe_norm_lin, FE_REF_GENE)):
        cols = [c for c in labels if enr is not None and c in enr.columns]
        fe = enr[cols].reindex(out.index).max(axis=1) if cols else pd.Series(1.0, index=out.index)
        out[name] = norm(fe.fillna(1.0), ref)
    for col in TSS_ANNOTATIONS + GENE_ANNOTATIONS:
        out[f"fe_{col}"] = enr[col].reindex(out.index) if enr is not None and col in enr.columns else np.nan

    rep = ev["report"]
    out["Fraction"] = rep["total_bp"].reindex(out.index) / rep["total_bp"].sum() if rep is not None else np.nan
    out["MedianLength"] = rep["median_length"].reindex(out.index) if rep is not None else np.nan
    st = ev["self_transition"]
    out["SelfTransition"] = st.reindex(out.index) if st is not None else np.nan

    # Family scores: marks, reinforced by the annotation that defines the type.
    out["s_prom"] = 0.6 * out["prom"] + 0.4 * out["tss_ann"]
    out["s_enh"] = out["enh"]
    out["s_tran"] = 0.7 * out["tx"] + 0.3 * out["gene_ann"]
    out["s_facu"] = out["facu"]
    out["s_cons"] = out["cons"]
    out["s_ctcf"] = out["ctcf"]
    return out


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

FAMILY_SCORES = {"Promoter": "s_prom", "Enhancer": "s_enh", "Transcribed": "s_tran",
                 "FacultativeHet": "s_facu", "ConstitutiveHet": "s_cons", "CTCF": "s_ctcf"}


def _family(row):
    """Type family, the rule that fired and the evidence strength for one state."""
    # "Quiescent: lack of any marks" — nothing stands out, relative or raw.
    if row["max_mark"] < QUIES_MAX and row["max_raw"] < QUIES_MAX:
        return "Quiescent", f"no mark above {QUIES_MAX:.2f}", 1.0 - row["max_mark"]
    # "K9K36: presence of the marks H3K9me3 and H3K36me3".
    if row["cons"] >= STRONG and row["tx"] >= STRONG:
        return "K9K36", "H3K9me3+H3K36me3", min(row["cons"], row["tx"])
    # "Bivalent: both activating (H3K27ac) and repressive (H3K27me3) marks".
    activating = _nanmax([row["k27ac"], row["k4me3"]])
    if row["facu"] >= BIVA_FACU and activating >= STRONG:
        return "Bivalent", "H3K27me3+activating", min(row["facu"], activating)
    # "CTCF: presence of the transcription factor CTCF" — only if measured.
    others = _nanmax([row["prom"], row["enh"], row["tx"], row["facu"], row["cons"]])
    if not np.isnan(row["ctcf"]) and row["ctcf"] >= STRONG and others < CTCF_OTHER:
        return "CTCF", "CTCF only", row["ctcf"] - others
    # Otherwise the dominant family wins.
    scores = {fam: row[col] for fam, col in FAMILY_SCORES.items() if not np.isnan(row[col])}
    fam = max(scores, key=scores.get)
    if scores[fam] < UNCLASSIFIED:
        return "Quiescent", f"no family above {UNCLASSIFIED:.2f}", 1.0 - row["max_mark"]
    return fam, f"dominant {fam}={scores[fam]:.2f}", scores[fam]


def _tier(states, strength, strong_type, low_type):
    """Split a family into a strong and a low tier by within-segmentation signal.

    The types table defines PromoterFlanking and EnhancerLow relative to their
    strong counterpart ("at lower levels than Promoters", "same as Enhancer, but
    with lower signal values"), so the split is relative to the best state of the
    same family in the same segmentation, with an absolute floor: a family whose
    strongest state stays below TIER_FLOOR is low-signal throughout.
    """
    if not states:
        return {}
    cut = max(TIER_RATIO * max(strength[s] for s in states), TIER_FLOOR)
    return {s: (strong_type if strength[s] >= cut else low_type) for s in states}


def interpret_states(ev_table, types_table=None):
    """Assign a chromatin state type to every state of one segmentation.

    Adds to the evidence table:
      Family    type before the strong/low tier split
      Type      chromatin state type from the types table, Short its short name
      Reason    which rule fired
      Strength  evidence for the assigned type (0..1)
      Margin    winning family score minus runner-up — low means mixed marks
    """
    if types_table is None:
        types_table = load_interpretation_table()
    short_names = types_table["Short Name"].to_dict()

    df = ev_table.copy()
    decided = df.apply(_family, axis=1)
    df["Family"] = [f for f, _, _ in decided]
    df["Reason"] = [r for _, r, _ in decided]
    df["Strength"] = [s for _, _, s in decided]

    types = dict(zip(df.index, df["Family"]))
    types.update(_tier([s for s in df.index if types[s] == "Promoter"],
                       df["s_prom"].to_dict(), "Promoter", "PromoterFlanking"))
    types.update(_tier([s for s in df.index if types[s] == "Enhancer"],
                       df["s_enh"].to_dict(), "Enhancer", "EnhancerLow"))
    df["Type"] = [types[s] for s in df.index]

    unknown = set(df["Type"]) - set(short_names)
    if unknown:
        raise ValueError(f"Types missing from the types table: {sorted(unknown)}")
    df["Short"] = df["Type"].map(short_names)

    top2 = np.sort(np.nan_to_num(df[list(FAMILY_SCORES.values())].to_numpy(float)), axis=1)[:, -2:]
    df["Margin"] = top2[:, 1] - top2[:, 0]
    return df


def interpret_segmentation(segdir, types_table=None):
    """Interpret one analysis directory; None when it has not been analyzed."""
    ev = load_segmentation_evidence(segdir)
    return None if ev is None else interpret_states(state_evidence(ev), types_table)


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def natural_state_order(states):
    """States of one segmentation in their own order: 1..15, or E1..E15."""
    return sorted(set(states), key=lambda s: (int(re.sub(r"\D", "", s) or 0), s))


def type_matrix(df, dataset, rep="rep1", column="Short", methods=None):
    """Interpreted types as rows = method, columns = state index within method.

    *df* holds the interpretation of several segmentations, with Dataset / Rep /
    Method / State columns; *methods* fixes the row order.
    """
    sub = df[(df["Dataset"] == dataset) & (df["Rep"] == rep)]
    order = methods if methods is not None else list(dict.fromkeys(sub["Method"]))
    order = [m for m in order if m in set(sub["Method"])]
    rows = {}
    for method in order:
        states = sub[sub["Method"] == method].set_index("State")
        rows[method] = pd.Series([states.at[s, column] for s in natural_state_order(states.index)],
                                 index=range(1, len(states) + 1))
    return pd.DataFrame(rows).T.reindex(order)


def type_text_color(state_type):
    """Text color that stays readable on top of TYPE_COLORS[state_type]."""
    r, g, b = (int(TYPE_COLORS[state_type].lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"
