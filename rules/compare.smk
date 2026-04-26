# Cross-segmentation comparison and method summary (driven by compare.py / compare_methods.py):
#   - Transition matrix entropy
#   - Pairwise Cohen's Kappa, AMI, Jaccard similarity
#   - Segment length statistics
#   - Unified method comparison table
#
# Three parallel comparison runs, one per label-matching strategy (variant):
#   comparison/comb/   — combined (overlap + bw-emission, default)
#   comparison/bwem/   — bigwig-emission-only
#   comparison/ovlp/   — overlap-only
#
# compare_methods aggregates each variant into methods/{variant}/comparison_table.tsv
# (no rematch columns; base replicate consistency only).
#
# Three replicate re-match runs (always from comb matrices):
#   methods/rematched_ovlp/  — re-align rep labels by bp overlap
#   methods/rematched_binem/ — re-align rep labels by cosine sim of binarized emissions
#   methods/rematched_bwem/  — re-align rep labels by cosine sim of bigwig emissions


_VARIANT_SUFFIX = {
    "comb":   "comb_matched",
    "bwem":   "bwem_matched",
    "ovlp":   "ovlp_matched",
}


def _seg_bin(path):
    """Native bin size for a segmentation BED path."""
    if "/omni/" in path:
        return OMNI_BIN
    if "/homer/" in path:
        return HOMER_BIN
    return CHROMHMM_BIN  # chromhmm_default and reference


def _compare_beds_for_folder(folder, variant):
    """Matched BED files for one folder for a given variant."""
    suffix = _VARIANT_SUFFIX[variant]
    cell = DATASETS[ds_of(folder)]["cell"]
    beds = [f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense_{suffix}.bed"]
    for caller in ["omni", "homer"]:
        beds.append(f"{folder}/{caller}/chromhmm_result/{cell}_{NSTATES}_dense_{suffix}.bed")
        beds.append(f"{folder}/{caller}/kmeans_states_{suffix}.bed")
    return beds


def _ds_compare_segs(ds, variant):
    """All BED files for dataset-level comparison: ref + every folder."""
    segs = [_ref_bed(ds)]
    for folder in _folders(ds):
        segs += _compare_beds_for_folder(folder, variant)
    return segs


def _ds_compare_inputs(w):
    """Snakemake inputs for compute_metrics: ref + matched beds for the given variant.

    When DO_ANALYZE is enabled the analysis sentinels are included so that
    compute_metrics runs after analyze_segmentations, ensuring .bin_emissions.npz
    files are present for binem re-matching.
    """
    segs = list(_ds_compare_segs(w.ds, w.variant))
    if DO_ANALYZE:
        for folder in _folders(w.ds):
            segs.append(f"{folder}/analysis/ref/report.tsv")
    return segs


rule compute_metrics:
    """Dataset-level comparison for one matching strategy: entropy, kappa, AMI, Jaccard."""
    input: _ds_compare_inputs
    output:
        entropy              = "{ds}/comparison/{variant}/entropy_summary.tsv",
        kappa                = "{ds}/comparison/{variant}/kappa_matrix.tsv",
        ami                  = "{ds}/comparison/{variant}/ami_matrix.tsv",
        jaccard              = "{ds}/comparison/{variant}/jaccard_similarity_matrix.tsv",
        overlap              = "{ds}/comparison/{variant}/overlap_matrix.tsv",
        stats                = "{ds}/comparison/{variant}/segment_stats.tsv",
        kappa_rematch_ovlp   = "{ds}/comparison/{variant}/kappa_rematch_ovlp_matrix.tsv",
        jaccard_rematch_ovlp = "{ds}/comparison/{variant}/jaccard_rematch_ovlp_matrix.tsv",
        overlap_rematch_ovlp = "{ds}/comparison/{variant}/overlap_rematch_ovlp_matrix.tsv",
        kappa_rematch_binem     = "{ds}/comparison/{variant}/kappa_rematch_binem_matrix.tsv",
        jaccard_rematch_binem   = "{ds}/comparison/{variant}/jaccard_rematch_binem_matrix.tsv",
        overlap_rematch_binem   = "{ds}/comparison/{variant}/overlap_rematch_binem_matrix.tsv",
        emission_sim         = "{ds}/comparison/{variant}/emission_similarity_matrix.tsv",
        kappa_rematch_bwem     = "{ds}/comparison/{variant}/kappa_rematch_bwem_matrix.tsv",
        jaccard_rematch_bwem   = "{ds}/comparison/{variant}/jaccard_rematch_bwem_matrix.tsv",
        overlap_rematch_bwem   = "{ds}/comparison/{variant}/overlap_rematch_bwem_matrix.tsv",
        bw_emission_sim      = "{ds}/comparison/{variant}/bw_emission_similarity_matrix.tsv",
    wildcard_constraints:
        ds      = r"[A-Za-z0-9_]+",
        variant = r"comb|bwem|ovlp",
    threads: workflow.cores
    conda: "../envs/python.yaml"
    params:
        scripts_dir = SCRIPTS_DIR,
        # Per-segmentation bin sizes: each method is evaluated at its native resolution.
        # For pair comparison (kappa/AMI), compare.py uses min(bin_i, bin_j) so that
        # 200bp segments are compared at 100bp when paired with OmniPeak segmentations.
        segs        = lambda w: " ".join(_ds_compare_segs(w.ds, w.variant)),
        bins        = lambda w: " ".join(str(_seg_bin(p)) for p in _ds_compare_segs(w.ds, w.variant)),
    shell:
        r"""
        python {params.scripts_dir}/compare.py \
            --seg {params.segs} \
            --bins {params.bins} \
            --outdir {wildcards.ds}/comparison/{wildcards.variant} \
            --analysis-dir {wildcards.ds}/analysis/{wildcards.variant} \
            --threads {threads}
        """


rule compare_methods:
    """Aggregate metrics for {ds}/{variant} into a unified comparison table."""
    input:
        entropy  = "{ds}/comparison/{variant}/entropy_summary.tsv",
        kappa    = "{ds}/comparison/{variant}/kappa_matrix.tsv",
        ami      = "{ds}/comparison/{variant}/ami_matrix.tsv",
        jaccard  = "{ds}/comparison/{variant}/jaccard_similarity_matrix.tsv",
        overlap  = "{ds}/comparison/{variant}/overlap_matrix.tsv",
        stats    = "{ds}/comparison/{variant}/segment_stats.tsv",
        analysis = "{ds}/analysis/ref/report.tsv",
    output: "{ds}/methods/{variant}/comparison_table.tsv"
    wildcard_constraints:
        ds      = r"[A-Za-z0-9_]+",
        variant = r"comb|bwem|ovlp",
    conda: "../envs/python.yaml"
    params:
        scripts_dir = SCRIPTS_DIR,
    shell:
        r"""
        python {params.scripts_dir}/compare_methods.py \
            --analysis-dir {wildcards.ds}/analysis/{wildcards.variant} \
            --ref-dir {wildcards.ds}/analysis \
            --comparison-dir {wildcards.ds}/comparison/{wildcards.variant} \
            --outdir {wildcards.ds}/methods/{wildcards.variant}
        """


def _rematched_inputs(w):
    """Declared compute_metrics outputs relevant to this rematch method."""
    base = [
        f"{w.ds}/comparison/comb/kappa_matrix.tsv",
        f"{w.ds}/comparison/comb/jaccard_similarity_matrix.tsv",
        f"{w.ds}/comparison/comb/overlap_matrix.tsv",
        f"{w.ds}/analysis/ref/report.tsv",
    ]
    extras = {
        "ovlp": [
            f"{w.ds}/comparison/comb/kappa_rematch_ovlp_matrix.tsv",
            f"{w.ds}/comparison/comb/jaccard_rematch_ovlp_matrix.tsv",
            f"{w.ds}/comparison/comb/overlap_rematch_ovlp_matrix.tsv",
        ],
        "binem": [
            f"{w.ds}/comparison/comb/kappa_rematch_binem_matrix.tsv",
            f"{w.ds}/comparison/comb/jaccard_rematch_binem_matrix.tsv",
            f"{w.ds}/comparison/comb/overlap_rematch_binem_matrix.tsv",
            f"{w.ds}/comparison/comb/emission_similarity_matrix.tsv",
        ],
        "bwem": [
            f"{w.ds}/comparison/comb/kappa_rematch_bwem_matrix.tsv",
            f"{w.ds}/comparison/comb/jaccard_rematch_bwem_matrix.tsv",
            f"{w.ds}/comparison/comb/overlap_rematch_bwem_matrix.tsv",
            f"{w.ds}/comparison/comb/bw_emission_similarity_matrix.tsv",
        ],
    }
    return base + extras[w.rematch]


rule compare_rematched:
    """Replicate reproducibility after {rematch} re-matching (using comb comparison)."""
    input: _rematched_inputs
    output: "{ds}/methods/rematched_{rematch}/comparison_table.tsv"
    wildcard_constraints:
        ds      = r"[A-Za-z0-9_]+",
        rematch = r"ovlp|binem|bwem",
    conda: "../envs/python.yaml"
    params:
        scripts_dir = SCRIPTS_DIR,
    shell:
        r"""
        python {params.scripts_dir}/compare_methods.py \
            --analysis-dir {wildcards.ds}/analysis/comb \
            --ref-dir {wildcards.ds}/analysis \
            --comparison-dir {wildcards.ds}/comparison/comb \
            --rematch {wildcards.rematch} \
            --outdir {wildcards.ds}/methods/rematched_{wildcards.rematch}
        """
