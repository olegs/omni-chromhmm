import glob as _glob
from pathlib import Path

# Inter-dataset reproducibility: compare the same method across all datasets.
#
# For each method the pooled ovlp_matched segmentation from every dataset is fed
# into compare.py with --all-pairs, so every cross-dataset pair is evaluated.
# Labels are prefixed with the dataset name (e.g. "imr90:chromhmm_omni") to
# avoid collisions when the same method key appears in multiple datasets.
#
# State re-matching (ovlp and binem) aligns any residual label differences that
# survive the per-dataset reference matching step.
#
# Outputs (under inter_dataset/):
#   {method}/kappa_matrix.tsv              — raw kappa, all dataset pairs
#   {method}/kappa_rematch_ovlp_matrix.tsv — after overlap re-matching
#   {method}/kappa_noqh_matrix.tsv         — NOQH variants
#   {method}/kappa_rematch_ovlp_noqh_matrix.tsv
#   comparison_table.tsv                   — one row per method × dataset-pair

INTER_DS_METHODS = [
    "chromhmm_default",
    "chromhmm_omni",  "kmeans_omni",
    "chromhmm_homer", "kmeans_homer",
    "chromhmm_macs2", "kmeans_macs2",
]


def _inter_ds_bed(ds, method):
    """Pooled ovlp_matched BED for *method* in *ds*."""
    cell = DATASETS[ds]["cell"]
    if method == "chromhmm_default":
        return f"{ds}/chromhmm_default_result/{cell}_{NSTATES}_dense_ovlp_matched.bed"
    parts  = method.split("_")           # ["chromhmm","omni"] or ["kmeans","homer"]
    model  = parts[0]                    # chromhmm | kmeans
    caller = parts[1]                    # omni | homer | macs2
    if model == "chromhmm":
        return f"{ds}/{caller}/chromhmm_result/{cell}_{NSTATES}_dense_ovlp_matched.bed"
    return f"{ds}/{caller}/kmeans_states_ovlp_matched.bed"


def _inter_ds_bin(method):
    """Native bin size for a method."""
    if "omni" in method or "macs2" in method:
        return OMNI_BIN    # 100 bp
    return CHROMHMM_BIN    # 200 bp


def _inter_ds_inputs(method):
    return [_inter_ds_bed(ds, method) for ds in DATASETS]


rule inter_dataset_compare_method:
    """Compare one method across all datasets with ovlp + binem re-matching."""
    input:
        lambda w: _inter_ds_inputs(w.method),
    output:
        entropy                    = "inter_dataset/{method}/entropy_summary.tsv",
        kappa                      = "inter_dataset/{method}/kappa_matrix.tsv",
        ami                        = "inter_dataset/{method}/ami_matrix.tsv",
        jaccard                    = "inter_dataset/{method}/jaccard_similarity_matrix.tsv",
        overlap                    = "inter_dataset/{method}/overlap_matrix.tsv",
        stats                      = "inter_dataset/{method}/segment_stats.tsv",
        kappa_rematch_ovlp         = "inter_dataset/{method}/kappa_rematch_ovlp_matrix.tsv",
        jaccard_rematch_ovlp       = "inter_dataset/{method}/jaccard_rematch_ovlp_matrix.tsv",
        overlap_rematch_ovlp       = "inter_dataset/{method}/overlap_rematch_ovlp_matrix.tsv",
        kappa_rematch_binem        = "inter_dataset/{method}/kappa_rematch_binem_matrix.tsv",
        jaccard_rematch_binem      = "inter_dataset/{method}/jaccard_rematch_binem_matrix.tsv",
        overlap_rematch_binem      = "inter_dataset/{method}/overlap_rematch_binem_matrix.tsv",
        emission_sim               = "inter_dataset/{method}/emission_similarity_matrix.tsv",
        kappa_rematch_bwem         = "inter_dataset/{method}/kappa_rematch_bwem_matrix.tsv",
        jaccard_rematch_bwem       = "inter_dataset/{method}/jaccard_rematch_bwem_matrix.tsv",
        overlap_rematch_bwem       = "inter_dataset/{method}/overlap_rematch_bwem_matrix.tsv",
        bw_emission_sim            = "inter_dataset/{method}/bw_emission_similarity_matrix.tsv",
        kappa_noqh                 = "inter_dataset/{method}/kappa_noqh_matrix.tsv",
        ami_noqh                   = "inter_dataset/{method}/ami_noqh_matrix.tsv",
        jaccard_noqh               = "inter_dataset/{method}/jaccard_noqh_matrix.tsv",
        overlap_noqh               = "inter_dataset/{method}/overlap_noqh_matrix.tsv",
        kappa_rematch_ovlp_noqh    = "inter_dataset/{method}/kappa_rematch_ovlp_noqh_matrix.tsv",
        jaccard_rematch_ovlp_noqh  = "inter_dataset/{method}/jaccard_rematch_ovlp_noqh_matrix.tsv",
        overlap_rematch_ovlp_noqh  = "inter_dataset/{method}/overlap_rematch_ovlp_noqh_matrix.tsv",
        kappa_rematch_binem_noqh   = "inter_dataset/{method}/kappa_rematch_binem_noqh_matrix.tsv",
        jaccard_rematch_binem_noqh = "inter_dataset/{method}/jaccard_rematch_binem_noqh_matrix.tsv",
        overlap_rematch_binem_noqh = "inter_dataset/{method}/overlap_rematch_binem_noqh_matrix.tsv",
        kappa_rematch_bwem_noqh    = "inter_dataset/{method}/kappa_rematch_bwem_noqh_matrix.tsv",
        jaccard_rematch_bwem_noqh  = "inter_dataset/{method}/jaccard_rematch_bwem_noqh_matrix.tsv",
        overlap_rematch_bwem_noqh  = "inter_dataset/{method}/overlap_rematch_bwem_noqh_matrix.tsv",
    wildcard_constraints:
        method = "|".join(INTER_DS_METHODS),
    threads: workflow.cores
    conda: "../envs/python.yaml"
    params:
        scripts_dir = SCRIPTS_DIR,
        segs   = lambda w: " ".join(_inter_ds_inputs(w.method)),
        bins   = lambda w: " ".join(
                     str(_inter_ds_bin(w.method)) for _ in DATASETS),
        labels = lambda w: " ".join(
                     f"{ds}:{w.method}" for ds in DATASETS),
    shell:
        r"""
        python {params.scripts_dir}/compare.py \
            --seg    {params.segs} \
            --bins   {params.bins} \
            --labels {params.labels} \
            --all-pairs \
            --outdir inter_dataset/{wildcards.method} \
            --threads {threads}
        """


rule inter_dataset_compare_summary:
    """Aggregate per-method kappa matrices into one cross-dataset comparison table."""
    input:
        expand("inter_dataset/{method}/kappa_rematch_ovlp_noqh_matrix.tsv",
               method=INTER_DS_METHODS),
    output:
        "inter_dataset/comparison_table.tsv",
    conda: "../envs/python.yaml"
    params:
        scripts_dir = SCRIPTS_DIR,
        methods     = " ".join(INTER_DS_METHODS),
        indir       = "inter_dataset",
    shell:
        r"""
        python {params.scripts_dir}/compare_inter_dataset.py \
            --methods {params.methods} \
            --indir   {params.indir} \
            --outfile {output}
        """


_SUMMARY_PLOTS = [
    "inter_dataset/summary_plots/summary_entropy_noqh.png",
    "inter_dataset/summary_plots/summary_jaccard_tx.png",
    "inter_dataset/summary_plots/summary_enrich_tx.png",
    "inter_dataset/summary_plots/summary_median_tx_length.png",
    "inter_dataset/summary_plots/summary_jaccard_tss.png",
    "inter_dataset/summary_plots/summary_n_segments.png",
]


rule inter_dataset_summary_plots:
    """Cross-dataset summary bar plots with mean ± std across all datasets."""
    input:
        expand("{ds}/methods/ovlp/comparison_table.tsv", ds=list(DATASETS)),
    output:
        _SUMMARY_PLOTS,
    conda: "../envs/python.yaml"
    params:
        scripts_dir   = SCRIPTS_DIR,
        datasets      = " ".join(list(DATASETS)),
        methods_dirs  = " ".join(f"{ds}/methods/ovlp"   for ds in DATASETS),
        analysis_dirs = " ".join(f"{ds}/analysis/ovlp"  for ds in DATASETS),
        outdir        = "inter_dataset/summary_plots",
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets      {params.datasets} \
            --methods-dirs  {params.methods_dirs} \
            --analysis-dirs {params.analysis_dirs} \
            --outdir        {params.outdir}
        """


_STATE_LENGTH_PLOT    = "inter_dataset/summary_plots/state_length_comparison.png"
_STATE_COVERAGE_PLOT  = "inter_dataset/summary_plots/state_coverage.png"
_PEAK_COUNT_PLOT      = "inter_dataset/summary_plots/peak_count.png"
_PEAK_LENGTH_PLOT     = "inter_dataset/summary_plots/peak_length.png"


rule inter_dataset_segment_lengths_comparison:
    """Per-state segment length violin: ENCODE reference vs de-novo methods, all datasets."""
    input:
        _MARKUPS_DIR,
        expand("{ds}/omni/kmeans_states_ovlp_matched.bed", ds=list(DATASETS)),
    output:
        _STATE_LENGTH_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir  = SCRIPTS_DIR,
        workdir      = lambda w: config.get("workdir", "."),
        markups_dir  = _MARKUPS_DIR,
        datasets     = " ".join(list(DATASETS)),
        cells        = " ".join(DATASETS[ds]["cell"] for ds in DATASETS),
        nstates      = NSTATES,
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets        {params.datasets} \
            --cells           {params.cells} \
            --workdir         {params.workdir} \
            --markups-dir     {params.markups_dir} \
            --nstates         {params.nstates} \
            --violin-outfile  {output}
        """


rule inter_dataset_state_coverage:
    """Per-state genomic coverage fraction: ENCODE reference vs de-novo methods, all datasets."""
    input:
        _MARKUPS_DIR,
        expand("{ds}/omni/kmeans_states_ovlp_matched.bed", ds=list(DATASETS)),
    output:
        _STATE_COVERAGE_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir  = SCRIPTS_DIR,
        workdir      = lambda w: config.get("workdir", "."),
        markups_dir  = _MARKUPS_DIR,
        datasets     = " ".join(list(DATASETS)),
        cells        = " ".join(DATASETS[ds]["cell"] for ds in DATASETS),
        nstates      = NSTATES,
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets               {params.datasets} \
            --cells                  {params.cells} \
            --workdir                {params.workdir} \
            --markups-dir            {params.markups_dir} \
            --nstates                {params.nstates} \
            --state-coverage-outfile {output}
        """


rule inter_dataset_peak_count:
    """Peak count per mark per method, mean ± std across datasets."""
    input:
        expand("{ds}/peaks/peak_stats.tsv", ds=list(DATASETS)),
    output:
        _PEAK_COUNT_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir = SCRIPTS_DIR,
        workdir     = lambda w: config.get("workdir", "."),
        datasets    = " ".join(list(DATASETS)),
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets             {params.datasets} \
            --workdir              {params.workdir} \
            --peak-count-outfile   {output}
        """


rule inter_dataset_peak_length:
    """Mean peak length per mark per method, mean ± std across datasets."""
    input:
        expand("{ds}/peaks/peak_stats.tsv", ds=list(DATASETS)),
    output:
        _PEAK_LENGTH_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir = SCRIPTS_DIR,
        workdir     = lambda w: config.get("workdir", "."),
        datasets    = " ".join(list(DATASETS)),
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets              {params.datasets} \
            --workdir               {params.workdir} \
            --peak-length-outfile   {output}
        """


_REF_KAPPA_MATRIX   = "inter_dataset/reference/kappa_matrix.tsv"
_REF_AMI_MATRIX     = "inter_dataset/reference/ami_matrix.tsv"
_REF_JACCARD_MATRIX = "inter_dataset/reference/jaccard_similarity_matrix.tsv"
_REF_DIST_PLOT      = "inter_dataset/reference/similarity_distribution.png"
_REF_COMPOSITION_PLOT = "inter_dataset/reference/state_composition.png"


rule inter_reference_compare:
    """Pairwise kappa/AMI/Jaccard among all downloaded ENCODE reference segmentations (no rematch)."""
    input:
        _MARKUPS_DIR,
    output:
        kappa       = _REF_KAPPA_MATRIX,
        ami         = _REF_AMI_MATRIX,
        jaccard     = _REF_JACCARD_MATRIX,
        dist        = _REF_DIST_PLOT,
        composition = _REF_COMPOSITION_PLOT,
    threads: workflow.cores
    conda: "../envs/python.yaml"
    params:
        scripts_dir = SCRIPTS_DIR,
        markups_dir = _MARKUPS_DIR,
        bin_size    = CHROMHMM_BIN,
        segs        = lambda w: " ".join(
                          str(p) for p in sorted(
                              Path(_MARKUPS_DIR, "15state").glob("*.bed.gz")
                          )
                      ),
        labels      = lambda w: " ".join(
                          "_".join(p.name.replace(".bed.gz", "").split("_")[1:])
                          for p in sorted(
                              Path(_MARKUPS_DIR, "15state").glob("*.bed.gz")
                          )
                      ),
    shell:
        r"""
        python {params.scripts_dir}/compare.py \
            --seg    {params.segs} \
            --bins   {params.bin_size} \
            --labels {params.labels} \
            --all-pairs \
            --outdir inter_dataset/reference \
            --threads {threads}
        python {params.scripts_dir}/summary_plots.py \
            --markups-dir        {params.markups_dir} \
            --ref-composition-outfile {output.composition} \
            --ref-kappa-matrix   {output.kappa} \
            --ref-ami-matrix     {output.ami} \
            --ref-jaccard-matrix {output.jaccard} \
            --ref-dist-outfile   {output.dist}
        """


_REP_DATASETS = [ds for ds in DATASETS if DATASETS[ds].get("replicates")]

_REP_CONSISTENCY_PLOTS = [
    "inter_dataset/summary_plots/rep_consistency_kappa_noqh_rep1_vs_rep2.png",
    "inter_dataset/summary_plots/rep_consistency_kappa_rematch_ovlp_noqh_rep1_vs_rep2.png",
    "inter_dataset/summary_plots/rep_consistency_kappa_rematch_ovlp_rep1_vs_rep2.png",
    "inter_dataset/summary_plots/rep_consistency_jaccard_noqh_rep1_vs_rep2.png",
    "inter_dataset/summary_plots/rep_consistency_jaccard_rematch_ovlp_noqh_rep1_vs_rep2.png",
    "inter_dataset/summary_plots/rep_consistency_jaccard_rematch_ovlp_rep1_vs_rep2.png",
    "inter_dataset/summary_plots/rep_consistency_ami_noqh_rep1_vs_rep2.png",
    "inter_dataset/summary_plots/rep_consistency_ami_rep1_vs_rep2.png",
]


rule inter_dataset_rep_consistency_plots:
    """Replicate consistency bar plots: Kappa/Jaccard/AMI (raw + ovlp-rematched) across datasets."""
    input:
        expand("{ds}/methods/ovlp/comparison_table.tsv",          ds=_REP_DATASETS),
        expand("{ds}/methods/rematched_ovlp/comparison_table.tsv", ds=_REP_DATASETS),
    output:
        _REP_CONSISTENCY_PLOTS,
    conda: "../envs/python.yaml"
    params:
        scripts_dir    = SCRIPTS_DIR,
        repo_plots_dir = os.path.join(workflow.basedir, "plots", "summary"),
        datasets       = " ".join(_REP_DATASETS),
        methods_dirs   = " ".join(f"{ds}/methods/ovlp"           for ds in _REP_DATASETS),
        rematched_dirs = " ".join(f"{ds}/methods/rematched_ovlp" for ds in _REP_DATASETS),
        outdir         = "inter_dataset/summary_plots",
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets             {params.datasets} \
            --methods-dirs         {params.methods_dirs} \
            --rematched-ovlp-dirs  {params.rematched_dirs} \
            --rep-consistency-outdir {params.outdir}
        cp {params.outdir}/rep_consistency_*.png {params.repo_plots_dir}/
        """


rule inter_dataset_all:
    """Run all inter-dataset method comparisons + summary table + summary plots."""
    input:
        "inter_dataset/comparison_table.tsv",
        expand("inter_dataset/{method}/kappa_rematch_ovlp_noqh_matrix.tsv",
               method=INTER_DS_METHODS),
        _SUMMARY_PLOTS,
        _STATE_LENGTH_PLOT,
        _STATE_COVERAGE_PLOT,
        _PEAK_COUNT_PLOT,
        _PEAK_LENGTH_PLOT,
        _REF_DIST_PLOT,
        _REF_COMPOSITION_PLOT,
        _REP_CONSISTENCY_PLOTS,
