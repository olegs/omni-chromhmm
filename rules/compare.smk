# Cross-segmentation comparison and method summary (driven by compare.py / compare_methods.py):
#   - Transition matrix entropy
#   - Pairwise Cohen's Kappa, AMI, Jaccard similarity
#   - Segment length statistics
#   - Unified method comparison table
#
# Default: comparison/comb/ only — combined overlap+bwem matching (alpha=0.8).
# compare_methods aggregates results into methods/comb/comparison_table.tsv.


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
    for caller in ["omni", "homer", "macs2"]:
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
    files are present for emission similarity computation.
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
        emission_sim    = "{ds}/comparison/{variant}/emission_similarity_matrix.tsv",
        bw_emission_sim = "{ds}/comparison/{variant}/bw_emission_similarity_matrix.tsv",
        kappa_noqh      = "{ds}/comparison/{variant}/kappa_noqh_matrix.tsv",
        ami_noqh        = "{ds}/comparison/{variant}/ami_noqh_matrix.tsv",
        overlap_noqh    = "{ds}/comparison/{variant}/overlap_noqh_matrix.tsv",
        jaccard_noqh    = "{ds}/comparison/{variant}/jaccard_noqh_matrix.tsv",
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


