#!/usr/bin/env python3
"""Per-state decomposition of the pairwise Cohen's kappa consistency score.

The bar charts in the notebook report a single kappa per method, which says
*how much* two segmentations of different epigenomes agree but not *where* the
agreement comes from or which states lose it. This module splits kappa into
exactly additive per-state pieces so methods can be compared state by state.

Notation — for one pair of segmentations, C is the bp confusion matrix
(C[i, j] = bp labelled i by segmentation A and j by segmentation B),
p = C / C.sum(), r = p.sum(1) (A marginal), c = p.sum(0) (B marginal):

    po    = sum_i p_ii                        observed agreement
    pe    = sum_i r_i c_i                     chance agreement
    kappa = (po - pe) / (1 - pe)

Diagonal contribution (what each state *earns*):

    kappa_i = (p_ii - r_i c_i) / (1 - pe)     and   sum_i kappa_i == kappa

Off-diagonal confusion (where kappa *leaks*), chance-corrected excess:

    e_ij = (p_ij - r_i c_j) / (1 - pe)        and   sum_{i != j} e_ij == -kappa

Because sum_ij (p_ij - r_i c_j) == 0, the off-diagonal excess matrix is a
complete accounting of the lost agreement. Most e_ij are negative (the pair
co-occurs *less* than chance, which is good); a *positive* e_ij is a systematic
label swap between states i and j — the actual confusion to investigate.

Loss budget — the shortfall splits into two independent causes, exactly:

    ceiling_i   = min(r_i, c_i)               max possible p_ii given marginals
    kappa_max   = (sum_i ceiling_i - pe) / (1 - pe)
    loss_i      = (ceiling_i - p_ii) / (1 - pe)          placement loss
    1 - kappa   = TV(r, c) / (1 - pe)  +  sum_i loss_i

where TV(r, c) = 1 - sum_i min(r_i, c_i) is the total-variation distance between
the two state compositions. The first term is *composition drift* (the two
segmentations call different *amounts* of each state — sensitivity/threshold
differences), the second is *misplacement* (right amounts, wrong places —
boundary jitter, fragmentation, state confusion, matching swaps).

Two per-state kappas — do not confuse them
------------------------------------------
The notebook's pairwise heatmaps put a *different* per-state kappa on their
diagonal: the one-vs-rest 2x2 kappa, which collapses every other state into
"not i" and scores state i on its own 0..1 scale,

    kappa_1vr_i = 2 (p_ii - r_i c_i) / (r_i + c_i - 2 r_i c_i)

Same numerator as kappa_i above, but divided by that state's *own* chance
headroom instead of the global one, so

    kappa_1vr_i = kappa_i * 2 (1 - pe) / (r_i + c_i - 2 r_i c_i)

The two therefore differ per state by a factor that itself varies per state
(2x for Quies up to >100x for rare states in the Omnipeak cache), and only
kappa_i is additive: sum_i kappa_i == kappa, while sum_i kappa_1vr_i is an
arbitrary number (2.57 for Omnipeak FULL). Use kappa_i to ask "where does this
method's kappa come from", kappa_1vr_i to ask "how well is state i recovered".
Both are computed here — kappa_diag and kappa_1vr in the per_state frame — so
the heatmap diagonals can be reproduced and cross-checked from one place.

Importable module (no CLI). Drive it from analysis_epi_1000.ipynb:
    from kappa_decomposition import run_kappa_decomposition
    run_kappa_decomposition(KAPPA_SOURCES, outdir="out/kappa_decomp")

It reads the pairwise caches the notebook already writes (out/pw_cache_*.pkl,
out/pw_15_cache.pkl, out/pw_18_core_cache.pkl), so nothing is recomputed.
"""

import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["savefig.dpi"] = 300
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from summary_plots import sort_states
from utils import BIN_COLORS

# Cached pairwise overlaps written by the notebook: label -> (path, type filter).
# The filter selects Individual/Joint inside out/pw_15_cache.pkl, which keys on
# (Type, Dataset1, Dataset2); None means "the cache holds a single method".
KAPPA_SOURCES = {
    "ChromHMM":        ("out/pw_cache_chromhmm.pkl", None),
    "HOMER":           ("out/pw_cache_homer.pkl", None),
    "MACS2":           ("out/pw_cache_macs2.pkl", None),
    "Omnipeak":        ("out/pw_cache_omnipeak.pkl", None),
    "Joint HOMER":     ("out/pw_cache_joint homer.pkl", None),
    "Joint MACS2":     ("out/pw_cache_joint macs2.pkl", None),
    "Joint Omnipeak":  ("out/pw_cache_joint omnipeak.pkl", None),
    "15-state Individual": ("out/pw_15_cache.pkl", "Individual"),
    "15-state Joint":      ("out/pw_15_cache.pkl", "Joint"),
    "18-core":         ("out/pw_18_core_cache.pkl", None),
}

# Method colours come from the project-wide binarization palette (utils.BIN_COLORS,
# the same source as summary_plots.METHOD_PALETTE and the notebook's
# `method_palette`), so a method keeps its colour in every figure. Individual and
# joint runs of one caller share a colour by design — the hatch separates them.
METHOD_STYLE = {
    "ChromHMM":            (BIN_COLORS["default"],   ""),
    "HOMER":               (BIN_COLORS["homer"],     ""),
    "MACS2":               (BIN_COLORS["macs2"],     ""),
    "Omnipeak":            (BIN_COLORS["omnipeak"],  ""),
    "Joint HOMER":         (BIN_COLORS["homer"],     "//"),
    "Joint MACS2":         (BIN_COLORS["macs2"],     "//"),
    "Joint Omnipeak":      (BIN_COLORS["omnipeak"],  "//"),
    "15-state Individual": (BIN_COLORS["default"],   ".."),
    "15-state Joint":      (BIN_COLORS["default"],   "xx"),
    "18-core":             (BIN_COLORS["reference"], ""),
}

NOQH_STATES = {"Quies", "Het"}


def method_style(label):
    """(colour, hatch) for a method label, falling back to keyword inference."""
    if label in METHOD_STYLE:
        return METHOD_STYLE[label]
    low = label.lower()
    for key, colour in (("omni", "omnipeak"), ("homer", "homer"),
                        ("macs2", "macs2"), ("chromhmm", "default"),
                        ("15-state", "default")):
        if key in low:
            return BIN_COLORS[colour], "//" if "joint" in low else ""
    return BIN_COLORS["reference"], "//" if "joint" in low else ""


def _norm(state):
    """Drop the ChromHMM state-number prefix: '15_Quies' -> 'Quies'."""
    return re.sub(r"^\d+_", "", state)


def load_cached_overlaps(path, type_filter=None):
    """Return the list of bp confusion dicts stored in a notebook pairwise cache."""
    with open(path, "rb") as f:
        cache = pickle.load(f)
    if "overlaps" in cache:            # out/pw_cache_<method>.pkl
        return cache["overlaps"]
    return [v["overlap"] for k, v in cache.items()   # out/pw_{15,18_core}_cache.pkl
            if type_filter is None or k[0] == type_filter]


def confusion_matrix(overlap, exclude_states=()):
    """Collapse an overlap dict into (states, C) with normalised state names."""
    excl = {_norm(s) for s in exclude_states}
    merged = defaultdict(float)
    for (s1, s2), bp in overlap.items():
        n1, n2 = _norm(s1), _norm(s2)
        if n1 in excl or n2 in excl:
            continue
        merged[(n1, n2)] += bp
    states = sort_states({s for key in merged for s in key})
    idx = {s: i for i, s in enumerate(states)}
    C = np.zeros((len(states), len(states)))
    for (s1, s2), bp in merged.items():
        C[idx[s1], idx[s2]] = bp
    return states, C


def decompose_pair(overlap, exclude_states=()):
    """Decompose one pair's kappa. Returns None if the pair has no shared bp.

    Keys: states, kappa, po, pe, kappa_max, tv, kappa_diag, kappa_1vr, ceiling,
    loss, excess. kappa_diag sums to kappa; excess off-diagonal sums to -kappa;
    kappa_1vr is the non-additive one-vs-rest per-state kappa that the notebook
    heatmaps put on their diagonal (see the module docstring), NaN for states
    absent from both segmentations of the pair.
    """
    states, C = confusion_matrix(overlap, exclude_states)
    total = C.sum()
    if total == 0:
        return None
    p = C / total
    r, c = p.sum(1), p.sum(0)
    po, pe = float(np.trace(p)), float(r @ c)
    if pe >= 1:
        return None
    den = 1.0 - pe
    ceiling = np.minimum(r, c)
    # One-vs-rest 2x2 kappa: same numerator as kappa_diag, but each state divided
    # by its own chance headroom. Undefined (not zero) where a state is missing
    # from both segmentations — averaging such pairs in as 0 biases the mean down.
    den_1vr = r + c - 2 * r * c
    kappa_1vr = np.where(den_1vr > 0,
                         2 * (np.diag(p) - r * c) / np.where(den_1vr > 0, den_1vr, 1),
                         np.nan)
    return {
        "states": states,
        "kappa": (po - pe) / den,
        "po": po,
        "pe": pe,
        "kappa_max": (ceiling.sum() - pe) / den,   # achievable given compositions
        "tv": 1.0 - ceiling.sum(),                 # composition drift
        "kappa_diag": (np.diag(p) - r * c) / den,  # per-state earnings (additive)
        "kappa_1vr": kappa_1vr,                    # per-state 2x2 (not additive)
        "ceiling": (ceiling - r * c) / den,        # per-state earnings ceiling
        "loss": (ceiling - np.diag(p)) / den,      # per-state placement loss
        "excess": (p - np.outer(r, c)) / den,      # chance-corrected confusion
    }


def decompose_method(overlaps, exclude_states=(), min_states=None):
    """Average the decomposition over all pairs of one method.

    Pairs whose state set differs from the modal one are skipped for the matrix
    and additive per-state pieces (a segmentation that lost a state to matching
    cannot be averaged cell-by-cell); the count of skipped pairs is returned so
    silent truncation stays visible.

    kappa_1vr needs no cell-by-cell alignment, so it is averaged over *every*
    pair in which the state is defined, per state — the same quantity and the
    same averaging as the fixed pairwise heatmap diagonals. Its per-state pair
    counts come back in the "one_vs_rest" frame.
    """
    decomposed = [d for d in (decompose_pair(o, exclude_states) for o in overlaps)
                  if d is not None]
    if not decomposed:
        return None
    modal = pd.Series([tuple(d["states"]) for d in decomposed]).mode()[0]
    kept = [d for d in decomposed if tuple(d["states"]) == modal]
    if min_states and len(modal) < min_states:
        return None
    n = len(kept)
    per_state = pd.DataFrame({
        "kappa_diag": np.mean([d["kappa_diag"] for d in kept], axis=0),
        "ceiling": np.mean([d["ceiling"] for d in kept], axis=0),
        "loss": np.mean([d["loss"] for d in kept], axis=0),
    }, index=list(modal))
    excess = pd.DataFrame(np.mean([d["excess"] for d in kept], axis=0),
                          index=list(modal), columns=list(modal))

    sums, counts = defaultdict(float), defaultdict(int)
    for d in decomposed:
        for state, value in zip(d["states"], d["kappa_1vr"]):
            if not np.isnan(value):
                sums[state] += value
                counts[state] += 1
    order = sort_states(counts)
    one_vs_rest = pd.DataFrame({
        "kappa_1vr": [sums[s] / counts[s] for s in order],
        "n_pairs": [counts[s] for s in order],
    }, index=order)

    return {
        "n_pairs": n,
        "n_skipped": len(decomposed) - n,
        "global": pd.Series({
            "kappa": np.mean([d["kappa"] for d in kept]),
            "po": np.mean([d["po"] for d in kept]),
            "pe": np.mean([d["pe"] for d in kept]),
            "kappa_max": np.mean([d["kappa_max"] for d in kept]),
            "kappa_over_max": np.mean([d["kappa"] / d["kappa_max"] for d in kept
                                       if d["kappa_max"] > 0]),
            "composition_TV": np.mean([d["tv"] for d in kept]),
        }),
        "per_state": per_state,
        "one_vs_rest": one_vs_rest,
        "excess": excess,
    }


def confusion_pairs(excess, top=15):
    """Rank unordered state pairs by symmetrised off-diagonal excess.

    Positive excess = the two segmentations swap these labels more often than
    chance, i.e. a systematic confusion rather than independent noise.
    """
    sym = excess.values + excess.values.T
    states = list(excess.index)
    rows = [{"State A": states[i], "State B": states[j], "Excess": sym[i, j]}
            for i in range(len(states)) for j in range(i + 1, len(states))]
    return (pd.DataFrame(rows).sort_values("Excess", ascending=False)
            .head(top).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_state_contributions_all(results, outfile, mode_label="FULL", top=12,
                                 verbose=True):
    """Grouped bars: per-state additive kappa contribution for *every* method.

    Coloured with the project method palette; each method's bars sum to its
    total kappa (shown in the legend).

    States are ordered by their summed contribution across methods; everything
    past `top` is pooled into one explicit "Other (n states)" group rather than
    dropped, so the bars still add up to kappa. Pass top=None for every state
    (the figure then gets very wide); exact numbers are in
    kappa_per_state_<mode>.csv.
    """
    methods = [m for m in results if results[m]]
    if not methods:
        return
    df = pd.DataFrame({m: results[m]["per_state"]["kappa_diag"] for m in methods})
    ranked = list(df.fillna(0).sum(axis=1).sort_values(ascending=False).index)
    shown = ranked[:top] if top else ranked
    pooled = ranked[len(shown):]
    plot_df = df.loc[shown]
    if pooled:
        plot_df.loc[f"Other ({len(pooled)} states)"] = df.loc[pooled].sum(min_count=1)
        if verbose:
            print(f"  {mode_label}: pooled {len(pooled)} low-contribution states "
                  f"into 'Other': {', '.join(pooled)}")

    groups = list(plot_df.index)
    x = np.arange(len(groups))
    w = 0.84 / len(methods)

    fig, ax = plt.subplots(figsize=(max(9.0, 1.0 * len(groups)), 5.2))
    for j, method in enumerate(methods):
        colour, hatch = method_style(method)
        offset = (j - (len(methods) - 1) / 2) * w
        ax.bar(x + offset, plot_df[method].values, w, color=colour, hatch=hatch,
               edgecolor="white", linewidth=0.5,
               label=f"{method}  (kappa={results[method]['global']['kappa']:.3f})")

    ax.axhline(0, color="0.4", linewidth=0.8)
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color="0.85", linewidth=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=55, ha="right", fontsize=8)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("Additive kappa contribution  (sums to total kappa)", fontsize=9)
    ax.set_xlabel("Chromatin state", fontsize=9)
    ax.set_title(f"Where pairwise kappa comes from ({mode_label}) — all methods\n"
                 "additive share (p_ii - r_i c_i)/(1 - pe): bars sum to the method's "
                 "kappa — NOT the one-vs-rest kappa on the heatmap diagonals\n"
                 "Colour = binarization, hatch separates methods sharing one",
                 fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=7.5, frameon=False, title="Method", title_fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


def plot_loss_budget(results, outfile, mode_label="FULL"):
    """Stacked bars: kappa + composition drift + placement loss == 1 per method."""
    methods = [m for m in results if results[m]]
    kappa = np.array([results[m]["global"]["kappa"] for m in methods])
    ceiling = np.array([results[m]["global"]["kappa_max"] for m in methods])
    placement = ceiling - kappa
    composition = 1.0 - ceiling
    order = np.argsort(-kappa)
    methods = [methods[i] for i in order]
    kappa, placement, composition = kappa[order], placement[order], composition[order]

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(methods)), 4.4))
    x = np.arange(len(methods))
    parts = (("Kappa achieved", kappa, "#4878CF"),
             ("Lost to misplacement", placement, "#E8833A"),
             ("Lost to composition drift", composition, "#BDBDBD"))
    bottom = np.zeros(len(methods))
    for label, vals, color in parts:
        ax.bar(x, vals, 0.68, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=1.2)
        bottom += vals
    for xi, k in zip(x, kappa):
        ax.text(xi, k / 2, f"{k:.3f}", ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of the agreement budget", fontsize=9)
    ax.set_title(f"Kappa loss budget ({mode_label}): 1 - kappa splits into\n"
                 "composition drift (different state amounts) + misplacement "
                 "(same amounts, wrong places)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left",
              borderaxespad=0)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_kappa_decomposition(sources=None, outdir="out/kappa_decomp",
                            modes=("full", "noqh"), verbose=True):
    """Decompose every cached method and write tables + plots to outdir.

    Returns {mode: {"global": DataFrame, "per_state": DataFrame, "results": dict}}.
    """
    sources = sources or KAPPA_SOURCES
    os.makedirs(outdir, exist_ok=True)
    out = {}

    for mode in modes:
        exclude = NOQH_STATES if mode == "noqh" else ()
        results = {}
        for method, (path, type_filter) in sources.items():
            if not os.path.exists(path):
                if verbose:
                    print(f"  WARNING: missing {path}", file=sys.stderr)
                continue
            res = decompose_method(load_cached_overlaps(path, type_filter), exclude)
            if res is None:
                continue
            results[method] = res
            if verbose:
                print(f"  {mode.upper():5s} {method:22s} kappa="
                      f"{res['global']['kappa']:.4f}  pairs={res['n_pairs']}"
                      + (f"  (skipped {res['n_skipped']} with odd state sets)"
                         if res["n_skipped"] else ""))

        df_global = pd.DataFrame({m: r["global"] for m, r in results.items()}).T
        # Joining methods with different state sets yields an alphabetical union
        # index; put it back into canonical ENCODE order.
        def _by_method(field):
            df = pd.DataFrame({m: r["per_state"][field] for m, r in results.items()})
            return df.reindex(sort_states(df.index))

        df_state = _by_method("kappa_diag")
        df_loss = _by_method("loss")
        # The heatmaps' per-state kappa: same numerator, own denominator, not
        # additive — kept next to df_state so the two are never mixed up.
        df_1vr = pd.DataFrame({m: r["one_vs_rest"]["kappa_1vr"]
                               for m, r in results.items()})
        df_1vr = df_1vr.reindex(sort_states(df_1vr.index))
        df_global.to_csv(f"{outdir}/kappa_global_{mode}.csv")
        df_state.to_csv(f"{outdir}/kappa_per_state_{mode}.csv")
        df_loss.to_csv(f"{outdir}/kappa_placement_loss_{mode}.csv")
        df_1vr.to_csv(f"{outdir}/kappa_one_vs_rest_{mode}.csv")

        for method, res in results.items():
            slug = method.lower().replace(" ", "_").replace("-", "_")
            confusion_pairs(res["excess"]).to_csv(
                f"{outdir}/confusion_pairs_{slug}_{mode}.csv", index=False)

        plot_loss_budget(results, f"{outdir}/loss_budget_{mode}.png", mode.upper())
        plot_state_contributions_all(
            results, f"{outdir}/state_contributions_all_{mode}.png",
            mode.upper(), verbose=verbose)
        out[mode] = {"global": df_global, "per_state": df_state,
                     "placement_loss": df_loss, "one_vs_rest": df_1vr,
                     "results": results}
    return out


if __name__ == "__main__":
    workdir = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/data/2026_epi_1000")
    os.chdir(workdir)
    run_kappa_decomposition()
