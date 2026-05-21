from pathlib import Path

# Inter-dataset reproducibility: compare the same method across all datasets.
#
# For each method the pooled {MATCH_METHOD}_matched segmentation from every dataset is fed
# into compare.py with --all-pairs, so every cross-dataset pair is evaluated.
# Labels are prefixed with the dataset name (e.g. "imr90:chromhmm_omni") to
# avoid collisions when the same method key appears in multiple datasets.
#
# Outputs (under inter_dataset/):
#   {method}/kappa_matrix.tsv       — raw kappa, all dataset pairs
#   {method}/kappa_noqh_matrix.tsv  — NOQH variant
#   comparison_table.tsv            — one row per method × dataset-pair

INTER_DS_METHODS = (
        ["chromhmm_default"]
        + (["chromhmm_omni"] if DO_CHROMHMM_PEAKS and DO_OMNIPEAK else [])
        + (["chromhmm_homer"] if DO_CHROMHMM_PEAKS and DO_HOMER else [])
        + (["chromhmm_macs2"] if DO_CHROMHMM_PEAKS and DO_MACS2 else [])
        + (["kmeans_omni"] if DO_OMNIPEAK else [])
        + (["kmeans_homer"] if DO_HOMER else [])
        + (["kmeans_macs2"] if DO_MACS2 else [])
)


def _inter_ds_bed(ds, method):
    """Pooled {MATCH_METHOD}_matched BED for *method* in *ds*."""
    cell = DATASETS[ds]["cell"]
    sfx = f"{MATCH_METHOD}_matched"
    if method == "chromhmm_default":
        return f"{ds}/chromhmm_default_result/{cell}_{NSTATES}_dense_{sfx}.bed"
    parts = method.split("_")  # ["chromhmm","omni"] or ["kmeans","homer"]
    model = parts[0]  # chromhmm | kmeans
    caller = parts[1]  # omni | homer | macs2
    if model == "chromhmm":
        return f"{ds}/{caller}/chromhmm_result/{cell}_{NSTATES}_dense_{sfx}.bed"
    return f"{ds}/{caller}/{caller}_kmeans_states_{sfx}.bed"



def _inter_ds_inputs(method):
    return [_inter_ds_bed(ds,method) for ds in DATASETS]


rule inter_dataset_compare_method:
    """Compare one method across all datasets (no re-matching)."""
    input:
        lambda w: _inter_ds_inputs(w.method),
    output:
        entropy="inter_dataset/{method}/entropy_summary.tsv",
        kappa="inter_dataset/{method}/kappa_matrix.tsv",
        jaccard="inter_dataset/{method}/jaccard_similarity_matrix.tsv",
        overlap="inter_dataset/{method}/overlap_matrix.tsv",
        stats="inter_dataset/{method}/segment_stats.tsv",
        kappa_noqh="inter_dataset/{method}/kappa_noqh_matrix.tsv",
        jaccard_noqh="inter_dataset/{method}/jaccard_noqh_matrix.tsv",
        overlap_noqh="inter_dataset/{method}/overlap_noqh_matrix.tsv",
    wildcard_constraints:
        method="|".join(INTER_DS_METHODS),
    threads: workflow.cores
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        segs=lambda w: " ".join(_inter_ds_inputs(w.method)),
        bins=lambda w: " ".join(
            str(_seg_bin(_inter_ds_bed(ds, w.method))) for ds in DATASETS),
        labels=lambda w: " ".join(
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
        expand("inter_dataset/{method}/kappa_noqh_matrix.tsv",
            method=INTER_DS_METHODS),
    output:
        "inter_dataset/comparison_table.tsv",
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        methods=" ".join(INTER_DS_METHODS),
        indir="inter_dataset",
    shell:
        r"""
        python {params.scripts_dir}/compare_inter_dataset.py \
            --methods {params.methods} \
            --indir   {params.indir} \
            --outfile {output}
        """


_SUMMARY_PLOTS = [
    "inter_dataset/summary_plots/summary_entropy.png",
    "inter_dataset/summary_plots/summary_entropy_noqh.png",
    "inter_dataset/summary_plots/summary_jaccard_tx.png",
    "inter_dataset/summary_plots/summary_enrich_tx.png",
    "inter_dataset/summary_plots/summary_median_tx_length.png",
    "inter_dataset/summary_plots/summary_jaccard_tss.png",
    "inter_dataset/summary_plots/summary_jaccard_tss_atac.png",
    "inter_dataset/summary_plots/summary_n_segments.png",
]


rule inter_dataset_summary_plots:
    """Cross-dataset summary bar plots with mean ± std across all datasets."""
    input:
        expand("{ds}/methods/" + MATCH_METHOD + "/comparison_table.tsv", ds=list(DATASETS)),
    output:
        _SUMMARY_PLOTS,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        datasets=" ".join(list(DATASETS)),
        methods_dirs=" ".join(f"{ds}/methods/{MATCH_METHOD}" for ds in DATASETS),
        analysis_dirs=" ".join(f"{ds}/analysis/{MATCH_METHOD}" for ds in DATASETS),
        outdir="inter_dataset/summary_plots",
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets      {params.datasets} \
            --methods-dirs  {params.methods_dirs} \
            --analysis-dirs {params.analysis_dirs} \
            --outdir        {params.outdir}
        """


_STATE_LENGTH_PLOT = "inter_dataset/summary_plots/state_length_comparison.png"
_STATE_COVERAGE_PLOT = "inter_dataset/summary_plots/state_coverage.png"
_PEAK_COUNT_PLOT = "inter_dataset/summary_plots/peak_count.png"
_PEAK_LENGTH_PLOT = "inter_dataset/summary_plots/peak_length.png"
_PEAK_GAP_VIOLIN_PLOT = "inter_dataset/summary_plots/peak_gap_violin.png"


rule inter_dataset_segment_lengths_comparison:
    """Per-state segment length violin: ENCODE reference vs de-novo methods, all datasets."""
    input:
        _MARKUPS_DIR,
        *(expand("{ds}/omni/omni_kmeans_states_" + MATCH_METHOD + "_matched.bed", ds=list(DATASETS)) if DO_OMNIPEAK else []),
        *(expand("{ds}/homer/homer_kmeans_states_" + MATCH_METHOD + "_matched.bed", ds=list(DATASETS)) if DO_HOMER else []),
        *(expand("{ds}/macs2/macs2_kmeans_states_" + MATCH_METHOD + "_matched.bed", ds=list(DATASETS)) if DO_MACS2 else []),
    output:
        _STATE_LENGTH_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        workdir=lambda w: config.get("workdir","."),
        markups_dir=_MARKUPS_DIR,
        datasets=" ".join(list(DATASETS)),
        cells=" ".join(DATASETS[ds]["cell"] for ds in DATASETS),
        nstates=NSTATES,
        match_method=MATCH_METHOD,
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets        {params.datasets} \
            --cells           {params.cells} \
            --workdir         {params.workdir} \
            --markups-dir     {params.markups_dir} \
            --nstates         {params.nstates} \
            --match-method    {params.match_method} \
            --violin-outfile  {output}
        """


rule inter_dataset_state_coverage:
    """Per-state genomic coverage fraction: ENCODE reference vs de-novo methods, all datasets."""
    input:
        _MARKUPS_DIR,
        *(expand("{ds}/omni/omni_kmeans_states_" + MATCH_METHOD + "_matched.bed", ds=list(DATASETS)) if DO_OMNIPEAK else []),
        *(expand("{ds}/homer/homer_kmeans_states_" + MATCH_METHOD + "_matched.bed", ds=list(DATASETS)) if DO_HOMER else []),
        *(expand("{ds}/macs2/macs2_kmeans_states_" + MATCH_METHOD + "_matched.bed", ds=list(DATASETS)) if DO_MACS2 else []),
    output:
        _STATE_COVERAGE_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        workdir=lambda w: config.get("workdir","."),
        markups_dir=_MARKUPS_DIR,
        datasets=" ".join(list(DATASETS)),
        cells=" ".join(DATASETS[ds]["cell"] for ds in DATASETS),
        nstates=NSTATES,
        match_method=MATCH_METHOD,
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets               {params.datasets} \
            --cells                  {params.cells} \
            --workdir                {params.workdir} \
            --markups-dir            {params.markups_dir} \
            --nstates                {params.nstates} \
            --match-method           {params.match_method} \
            --state-coverage-outfile {output}
        """


rule inter_dataset_peak_count:
    """Peak count per mark per method, mean ± std across datasets."""
    input:
        expand("{ds}/peaks/peak_stats.tsv",ds=list(DATASETS)),
    output:
        _PEAK_COUNT_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        workdir=lambda w: config.get("workdir","."),
        datasets=" ".join(list(DATASETS)),
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
        expand("{ds}/peaks/peak_stats.tsv",ds=list(DATASETS)),
    output:
        _PEAK_LENGTH_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        workdir=lambda w: config.get("workdir","."),
        datasets=" ".join(list(DATASETS)),
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets              {params.datasets} \
            --workdir               {params.workdir} \
            --peak-length-outfile   {output}
        """


rule inter_dataset_peak_gap_violin:
    """Violin plot: gap lengths between adjacent binarized elements, per mark (hue) × method."""
    input:
        expand("{ds}/peaks/gap_lengths.tsv.gz",ds=list(DATASETS)),
    output:
        _PEAK_GAP_VIOLIN_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        workdir=lambda w: config.get("workdir","."),
        datasets=" ".join(list(DATASETS)),
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets                   {params.datasets} \
            --workdir                    {params.workdir} \
            --peak-gap-violin-outfile    {output}
        """


_REF_KAPPA_MATRIX = "inter_dataset/reference/kappa_matrix.tsv"
_REF_JACCARD_MATRIX = "inter_dataset/reference/jaccard_similarity_matrix.tsv"
_REF_KAPPA_NOQH_MATRIX = "inter_dataset/reference/kappa_noqh_matrix.tsv"
_REF_JACCARD_NOQH_MATRIX = "inter_dataset/reference/jaccard_noqh_matrix.tsv"
_REF_DIST_PLOT = "inter_dataset/reference/similarity_distribution.png"
_REF_DIST_NOQH_PLOT = "inter_dataset/reference/similarity_distribution_noqh.png"
_REF_COMPOSITION_PLOT = "inter_dataset/reference/state_composition.png"


rule inter_reference_compare:
    """Pairwise kappa/Jaccard among all downloaded ENCODE reference segmentations (no rematch)."""
    input:
        _MARKUPS_DIR,
    output:
        kappa=_REF_KAPPA_MATRIX,
        jaccard=_REF_JACCARD_MATRIX,
        kappa_noqh=_REF_KAPPA_NOQH_MATRIX,
        jaccard_noqh=_REF_JACCARD_NOQH_MATRIX,
        dist=_REF_DIST_PLOT,
        dist_noqh=_REF_DIST_NOQH_PLOT,
        composition=_REF_COMPOSITION_PLOT,
    threads: workflow.cores
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        markups_dir=_MARKUPS_DIR,
        bin_size=CHROMHMM_BIN,
        segs=lambda w: " ".join(
            str(p) for p in sorted(
                Path(_MARKUPS_DIR,"15state").glob("*.bed.gz")
            )
        ),
        labels=lambda w: " ".join(
            "_".join(p.name.replace(".bed.gz","").split("_")[1:])
            for p in sorted(
                Path(_MARKUPS_DIR,"15state").glob("*.bed.gz")
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
            --markups-dir              {params.markups_dir} \
            --ref-composition-outfile  {output.composition} \
            --ref-kappa-matrix         {output.kappa} \
            --ref-jaccard-matrix       {output.jaccard} \
            --ref-dist-outfile         {output.dist} \
            --ref-kappa-noqh-matrix    {output.kappa_noqh} \
            --ref-jaccard-noqh-matrix  {output.jaccard_noqh} \
            --ref-dist-noqh-outfile    {output.dist_noqh}
        """


_METHOD_SIM_DIST_PLOT = "inter_dataset/summary_plots/method_similarity_distribution.png"
_METHOD_SIM_DIST_NOQH_PLOT = "inter_dataset/summary_plots/method_similarity_distribution_noqh.png"
_METHOD_SIM_DIST_CHIP_MINT_PLOT = "inter_dataset/summary_plots/method_similarity_distribution_chip_vs_mint.png"
_METHOD_SIM_DIST_CHIP_MINT_NOQH_PLOT = "inter_dataset/summary_plots/method_similarity_distribution_chip_vs_mint_noqh.png"

_CHIP_DATASETS = [ds for ds in DATASETS if not ds.endswith("_mint")]
_MINT_DATASETS = [ds for ds in DATASETS if ds.endswith("_mint")]


rule inter_dataset_method_similarity_distribution:
    """Violin plots of inter-dataset pairwise similarity (kappa/Jaccard) per de-novo method."""
    input:
        expand("inter_dataset/{method}/kappa_noqh_matrix.tsv",method=INTER_DS_METHODS),
    output:
        dist=_METHOD_SIM_DIST_PLOT,
        dist_noqh=_METHOD_SIM_DIST_NOQH_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        repo_plots_dir=os.path.join(workflow.basedir,"plots","summary"),
        indir="inter_dataset",
        methods=" ".join(INTER_DS_METHODS),
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --method-sim-dist-indir          {params.indir} \
            --method-sim-dist-methods        {params.methods} \
            --method-sim-dist-outfile        {output.dist} \
            --method-sim-dist-noqh-outfile   {output.dist_noqh}
        mkdir -p {params.repo_plots_dir}
        cp {output.dist} {output.dist_noqh} {params.repo_plots_dir}/
        """


rule inter_dataset_method_similarity_distribution_chip_vs_mint:
    """Violin plots of inter-dataset similarity filtered to ChIP↔Mint-ChIP pairs only."""
    input:
        expand("inter_dataset/{method}/kappa_noqh_matrix.tsv",method=INTER_DS_METHODS),
    output:
        dist=_METHOD_SIM_DIST_CHIP_MINT_PLOT,
        dist_noqh=_METHOD_SIM_DIST_CHIP_MINT_NOQH_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        repo_plots_dir=os.path.join(workflow.basedir,"plots","summary"),
        indir="inter_dataset",
        methods=" ".join(INTER_DS_METHODS),
        chip_datasets=" ".join(_CHIP_DATASETS),
        mint_datasets=" ".join(_MINT_DATASETS),
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --method-sim-dist-indir                   {params.indir} \
            --method-sim-dist-methods                 {params.methods} \
            --method-sim-dist-group-a                 {params.chip_datasets} \
            --method-sim-dist-group-b                 {params.mint_datasets} \
            --method-sim-dist-filtered-outfile        {output.dist} \
            --method-sim-dist-filtered-noqh-outfile   {output.dist_noqh}
        mkdir -p {params.repo_plots_dir}
        cp {output.dist} {output.dist_noqh} {params.repo_plots_dir}/
        """


_REP_DATASETS = [ds for ds in DATASETS if DO_REPLICATES and DATASETS[ds].get("replicates")]

_REP_CONSISTENCY_PLOTS = [
    "inter_dataset/summary_plots/rep_consistency_kappa_noqh_rep1_vs_rep2.png",
    "inter_dataset/summary_plots/rep_consistency_kappa_rep1_vs_rep2.png",
    "inter_dataset/summary_plots/rep_consistency_jaccard_noqh_rep1_vs_rep2.png",
    "inter_dataset/summary_plots/rep_consistency_jaccard_rep1_vs_rep2.png",
]


rule inter_dataset_rep_consistency_plots:
    """Replicate consistency bar plots: Kappa/Jaccard (raw) across datasets."""
    input:
        expand("{ds}/methods/comb/comparison_table.tsv",ds=_REP_DATASETS),
    output:
        _REP_CONSISTENCY_PLOTS,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        repo_plots_dir=os.path.join(workflow.basedir,"plots","summary"),
        datasets=" ".join(_REP_DATASETS),
        methods_dirs=" ".join(f"{ds}/methods/comb" for ds in _REP_DATASETS),
        outdir="inter_dataset/summary_plots",
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets              {params.datasets} \
            --methods-dirs          {params.methods_dirs} \
            --rep-consistency-outdir {params.outdir}
        mkdir -p {params.repo_plots_dir}
        cp {params.outdir}/rep_consistency_*.png {params.repo_plots_dir}/
        """


# all 7 de-novo methods
_METHOD_DS_COMPOSITION_PLOTS = [
    f"inter_dataset/summary_plots/method_ds_composition_{m}.png"
    for m in (list(INTER_DS_METHODS))
]


rule inter_dataset_method_composition:
    """Per-dataset state composition for each de-novo method (4 supplementary plots)."""
    input:
        [_inter_ds_bed(ds,m) for ds in DATASETS for m in (list(INTER_DS_METHODS))],
    output:
        _METHOD_DS_COMPOSITION_PLOTS,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        repo_plots_dir=os.path.join(workflow.basedir,"plots","summary"),
        workdir=lambda w: config.get("workdir","."),
        datasets=" ".join(list(DATASETS)),
        cells=" ".join(DATASETS[ds]["cell"] for ds in DATASETS),
        nstates=NSTATES,
        outdir="inter_dataset/summary_plots",
        match_method=MATCH_METHOD,
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets                     {params.datasets} \
            --cells                        {params.cells} \
            --workdir                      {params.workdir} \
            --nstates                      {params.nstates} \
            --match-method                 {params.match_method} \
            --method-ds-composition-outdir {params.outdir}
        mkdir -p {params.repo_plots_dir}
        cp {params.outdir}/method_ds_composition_*.png {params.repo_plots_dir}/
        """


rule inter_dataset_method_state_composition:
    """Mean state composition across datasets per method (single summary plot)."""
    input:
        [_inter_ds_bed(ds,m) for ds in DATASETS for m in (list(INTER_DS_METHODS))],
        _MARKUPS_DIR,
    output:
        "inter_dataset/summary_plots/method_state_composition.png",
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        repo_plots_dir=os.path.join(workflow.basedir,"plots","summary"),
        workdir=lambda w: config.get("workdir","."),
        datasets=" ".join(list(DATASETS)),
        cells=" ".join(DATASETS[ds]["cell"] for ds in DATASETS),
        nstates=NSTATES,
        markups_dir=_MARKUPS_DIR,
        match_method=MATCH_METHOD,
    shell:
        r"""
        python {params.scripts_dir}/summary_plots.py \
            --datasets                   {params.datasets} \
            --cells                      {params.cells} \
            --workdir                    {params.workdir} \
            --markups-dir                {params.markups_dir} \
            --nstates                    {params.nstates} \
            --match-method               {params.match_method} \
            --method-composition-outfile {output}
        mkdir -p {params.repo_plots_dir}
        cp {output} {params.repo_plots_dir}/
        """


_EMISSION_SIM_PLOTS = (
        expand("inter_dataset/summary_plots/emission_cosine_sim_{ds}.png",
            ds=list(DATASETS))
        + expand("inter_dataset/summary_plots/emission_gini_{ds}.png",
    ds=list(DATASETS))
        + [
            "inter_dataset/summary_plots/emission_cosine_sim_summary.png",
            "inter_dataset/summary_plots/emission_gini_summary.png",
        ]
)


rule inter_dataset_emission_similarity:
    """Pairwise cosine similarity of state emissions: binarized vs bigwig, per dataset + summary.

    Demonstrates that bigwig (continuous) emissions produce more similar (less
    discriminative) state profiles than binarized emissions, supporting the choice
    of overlap-based label transfer as the primary matching strategy.
    """
    input:
        expand("{ds}/methods/comb/comparison_table.tsv",ds=list(DATASETS)),
    output:
        _EMISSION_SIM_PLOTS,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        datasets=" ".join(list(DATASETS)),
        analysis_dirs=" ".join(f"{ds}/analysis/comb" for ds in DATASETS),
        methods=" ".join(INTER_DS_METHODS),
        outdir="inter_dataset/summary_plots",
        repo_plots_dir=os.path.join(workflow.basedir,"plots","summary"),
    shell:
        r"""
        python {params.scripts_dir}/emission_similarity.py \
            --datasets      {params.datasets} \
            --analysis-dirs {params.analysis_dirs} \
            --methods       {params.methods} \
            --outdir        {params.outdir}
        mkdir -p {params.repo_plots_dir}
        cp {params.outdir}/emission_cosine_sim_*.png {params.repo_plots_dir}/
        """


_INTER_DS_BINEM_PLOT = "inter_dataset/summary_plots/inter_dataset_binem_similarity.png"
_CROSS_ASSAY_BINEM_PLOT = "inter_dataset/summary_plots/cross_assay_binem_similarity.png"


rule inter_dataset_binem_similarity:
    """Inter-dataset and cross-assay binarized emission cosine similarity per method."""
    input:
        expand("{ds}/methods/comb/comparison_table.tsv",ds=list(DATASETS)),
    output:
        inter_ds=_INTER_DS_BINEM_PLOT,
        cross_assay=_CROSS_ASSAY_BINEM_PLOT,
    conda: "../envs/python.yaml"
    params:
        scripts_dir=SCRIPTS_DIR,
        repo_plots_dir=os.path.join(workflow.basedir,"plots","summary"),
        datasets=" ".join(list(DATASETS)),
        analysis_dirs=" ".join(f"{ds}/analysis/comb" for ds in DATASETS),
        methods=" ".join(INTER_DS_METHODS),
        outdir="inter_dataset/summary_plots",
        chip_datasets=" ".join(_CHIP_DATASETS),
        mint_datasets=" ".join(_MINT_DATASETS),
    shell:
        r"""
        python {params.scripts_dir}/emission_similarity.py \
            --datasets                      {params.datasets} \
            --analysis-dirs                 {params.analysis_dirs} \
            --methods                       {params.methods} \
            --outdir                        {params.outdir} \
            --inter-dataset-binem-outfile   {output.inter_ds} \
            --cross-assay-binem-outfile     {output.cross_assay} \
            --group-a                       {params.chip_datasets} \
            --group-b                       {params.mint_datasets}
        mkdir -p {params.repo_plots_dir}
        cp {output.inter_ds} {output.cross_assay} {params.repo_plots_dir}/
        """


rule inter_dataset_all:
    """Run all inter-dataset method comparisons + summary table + summary plots."""
    input:
        "inter_dataset/comparison_table.tsv",
        expand("inter_dataset/{method}/kappa_noqh_matrix.tsv",
            method=INTER_DS_METHODS),
        _SUMMARY_PLOTS,
        _STATE_LENGTH_PLOT,
        _STATE_COVERAGE_PLOT,
        _PEAK_COUNT_PLOT,
        _PEAK_LENGTH_PLOT,
        _PEAK_GAP_VIOLIN_PLOT,
        _REF_DIST_PLOT,
        _REF_DIST_NOQH_PLOT,
        _REF_COMPOSITION_PLOT,
        _REP_CONSISTENCY_PLOTS,
        _EMISSION_SIM_PLOTS,
        _METHOD_DS_COMPOSITION_PLOTS,
        "inter_dataset/summary_plots/method_state_composition.png",
        _METHOD_SIM_DIST_PLOT,
        _METHOD_SIM_DIST_NOQH_PLOT,
        _METHOD_SIM_DIST_CHIP_MINT_PLOT,
        _METHOD_SIM_DIST_CHIP_MINT_NOQH_PLOT,
        _INTER_DS_BINEM_PLOT,
        _CROSS_ASSAY_BINEM_PLOT,
