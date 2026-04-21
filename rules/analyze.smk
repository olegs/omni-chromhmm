# Downstream analysis and quality metrics (all driven by analyze.py):
#   - Per-segmentation: report, segment lengths, state emissions, enrichment
#   - Transition matrix entropy
#   - Pairwise Cohen's Kappa
#   - Segment length statistics
#   - Unified method comparison (compare_methods.py)


# --- per-segmentation analysis plots --------------------------------------

def _analysis_inputs(w):
    """Every matched segmentation for a dataset plus optional RNA-seq inputs."""
    ds = w.ds
    cfg = DATASETS[ds]
    cell = cfg["cell"]
    files = [
        _ref_bed(ds),
        f"{ds}/chromhmm_default_result/{cell}_{NSTATES}_chromhmm_default_matched.bed",
    ]
    for mode in _modes_for(ds):
        files.append(f"{ds}/{mode}/chromhmm_result/{cell}_{NSTATES}_chromhmm_{mode}_matched.bed")
        files.append(f"{ds}/{mode}/gmm_{mode}_matched.bed")
        files.append(f"{ds}/{mode}/kmeans_{mode}_matched.bed")
    # Per-replicate segmentations
    if cfg.get("replicates"):
        for rep in ["rep1", "rep2"]:
            files.append(f"{ds}/{rep}/chromhmm_default_result/{cell}_{NSTATES}_chromhmm_default_{rep}_matched.bed")
            files.append(f"{ds}/{rep}/chromhmm_result/{cell}_{NSTATES}_chromhmm_{rep}_matched.bed")
            files.append(f"{ds}/{rep}/gmm_{rep}_matched.bed")
            files.append(f"{ds}/{rep}/kmeans_{rep}_matched.bed")
    # RNA-seq + gene annotations (when available)
    if cfg.get("rnaseq"):
        files.append(f"{ds}/rnaseq_{cfg['rnaseq']}.tsv")
        files.append(TOOLS["gencode_gtf"])
    return files


rule analyze_segmentations:
    """Run analyze.py on every matched segmentation for a dataset. Also triggers
    enrichment against ChromHMM COORDS and, when available, RNA-seq expressed
    gene bodies / TSS."""
    input: _analysis_inputs
    output: "{ds}/analysis/ref/report.tsv"
    conda: "../envs/python.yaml"
    params:
        coords = TOOLS["coords_dir"],
        cell   = lambda w: DATASETS[w.ds]["cell"],
        modes  = lambda w: _modes_for(w.ds),
        reps   = lambda w: "rep1 rep2" if DATASETS[w.ds].get("replicates") else "",
        ref    = lambda w: _ref_bed(w.ds),
        bin    = BIN,
        n      = NSTATES,
        scripts_dir = SCRIPTS_DIR,
        rnaseq    = lambda w: f"{w.ds}/rnaseq_{DATASETS[w.ds]['rnaseq']}.tsv"
                               if DATASETS[w.ds].get("rnaseq") else "",
        gtf       = lambda w: TOOLS["gencode_gtf"] if DATASETS[w.ds].get("rnaseq") else "",
    shell:
        r"""
        mkdir -p {wildcards.ds}/analysis

        # Build RNA-seq arguments if available
        RNA_ARGS=""
        if [ -n "{params.rnaseq}" ]; then
            RNA_ARGS="--rnaseq {params.rnaseq} --gtf {params.gtf}"
        fi

        # Helper: run analyze.py with common args
        run_analyze() {{
            local seg="$1"; local outdir="$2"; shift 2
            python {params.scripts_dir}/analyze.py --seg "$seg" --bin {params.bin} \
                --outdir "$outdir" \
                --annotations {params.coords}/*.bed.gz \
                $RNA_ARGS "$@"
        }}

        # Reference
        run_analyze {params.ref} {wildcards.ds}/analysis/ref

        # Default ChromHMM (pre-matching: emissions + enrichment only)
        run_analyze \
            {wildcards.ds}/chromhmm_default_result/{params.cell}_{params.n}_dense.bed \
            {wildcards.ds}/analysis/chromhmm_default_dense \
            --inputs {wildcards.ds}/chromhmm_default/*.txt --emissions-only

        # Default ChromHMM (post-matching)
        run_analyze \
            {wildcards.ds}/chromhmm_default_result/{params.cell}_{params.n}_chromhmm_default_matched.bed \
            {wildcards.ds}/analysis/chromhmm_default \
            --inputs {wildcards.ds}/chromhmm_default/*.txt

        # Omnipeak modes
        for mode in {params.modes}; do
            run_analyze \
                {wildcards.ds}/$mode/chromhmm_result/{params.cell}_{params.n}_chromhmm_${{mode}}_matched.bed \
                {wildcards.ds}/analysis/chromhmm_$mode \
                --inputs {wildcards.ds}/$mode/chromhmm_peaks/*.txt.gz

            run_analyze \
                {wildcards.ds}/$mode/gmm_${{mode}}_matched.bed \
                {wildcards.ds}/analysis/gmm_$mode \
                --inputs {wildcards.ds}/$mode/chromhmm_peaks/*.txt.gz

            run_analyze \
                {wildcards.ds}/$mode/kmeans_${{mode}}_matched.bed \
                {wildcards.ds}/analysis/kmeans_$mode \
                --inputs {wildcards.ds}/$mode/chromhmm_peaks/*.txt.gz
        done

        # Per-replicate analysis
        for rep in {params.reps}; do
            run_analyze \
                {wildcards.ds}/$rep/chromhmm_default_result/{params.cell}_{params.n}_chromhmm_default_${{rep}}_matched.bed \
                {wildcards.ds}/analysis/chromhmm_default_$rep \
                --inputs {wildcards.ds}/$rep/chromhmm_default/*.txt

            run_analyze \
                {wildcards.ds}/$rep/chromhmm_result/{params.cell}_{params.n}_chromhmm_${{rep}}_matched.bed \
                {wildcards.ds}/analysis/chromhmm_$rep \
                --inputs {wildcards.ds}/$rep/chromhmm_peaks/*.txt.gz

            run_analyze \
                {wildcards.ds}/$rep/gmm_${{rep}}_matched.bed \
                {wildcards.ds}/analysis/gmm_$rep \
                --inputs {wildcards.ds}/$rep/chromhmm_peaks/*.txt.gz

            run_analyze \
                {wildcards.ds}/$rep/kmeans_${{rep}}_matched.bed \
                {wildcards.ds}/analysis/kmeans_$rep \
                --inputs {wildcards.ds}/$rep/chromhmm_peaks/*.txt.gz
        done

        """


# --- Segmentation quality metrics ----------------------------------------

def _seg_files(w):
    """All matched segmentation BEDs for a dataset (for metrics computation).

    Order: reference, default chromhmm,
           pooled omni (chromhmm, kmeans, gmm),
           replicated (chromhmm, kmeans, gmm),
           per-replicate (default chromhmm, chromhmm on omnipeak, kmeans, gmm).
    """
    ds = w.ds
    cfg = DATASETS[ds]
    cell = cfg["cell"]
    files = [
        _ref_bed(ds),
        f"{ds}/chromhmm_default_result/{cell}_{NSTATES}_chromhmm_default_matched.bed",
    ]
    for mode in _modes_for(ds):
        files.append(f"{ds}/{mode}/chromhmm_result/{cell}_{NSTATES}_chromhmm_{mode}_matched.bed")
        files.append(f"{ds}/{mode}/kmeans_{mode}_matched.bed")
        files.append(f"{ds}/{mode}/gmm_{mode}_matched.bed")
    if cfg.get("replicates"):
        for rep in ["rep1", "rep2"]:
            files.append(f"{ds}/{rep}/chromhmm_default_result/{cell}_{NSTATES}_chromhmm_default_{rep}_matched.bed")
            files.append(f"{ds}/{rep}/chromhmm_result/{cell}_{NSTATES}_chromhmm_{rep}_matched.bed")
            files.append(f"{ds}/{rep}/kmeans_{rep}_matched.bed")
            files.append(f"{ds}/{rep}/gmm_{rep}_matched.bed")
    return files


rule compute_metrics:
    """Compute cross-segmentation metrics: entropy, comparison (kappa/jaccard/emission), segment stats."""
    input: _seg_files
    output:
        entropy = "{ds}/analysis/comparison/entropy_summary.tsv",
        kappa   = "{ds}/analysis/comparison/kappa_matrix.tsv",
        stats   = "{ds}/analysis/comparison/segment_stats.tsv",
    conda: "../envs/python.yaml"
    params:
        scripts_dir = SCRIPTS_DIR,
        bin = BIN,
    shell:
        r"""
        python {params.scripts_dir}/analyze.py \
            --seg {input} \
            --bin {params.bin} \
            --outdir {wildcards.ds}/analysis \
            --analysis-dir {wildcards.ds}/analysis
        """


rule compare_methods:
    """Aggregate all metrics into a unified comparison table and figure."""
    input:
        entropy   = "{ds}/analysis/comparison/entropy_summary.tsv",
        kappa     = "{ds}/analysis/comparison/kappa_matrix.tsv",
        stats     = "{ds}/analysis/comparison/segment_stats.tsv",
        analysis  = "{ds}/analysis/ref/report.tsv",
    output: "{ds}/analysis/methods/comparison_table.tsv"
    conda: "../envs/python.yaml"
    params:
        scripts_dir = SCRIPTS_DIR,
    shell:
        r"""
        python {params.scripts_dir}/compare_methods.py \
            --analysis-dir {wildcards.ds}/analysis \
            --comparison-dir {wildcards.ds}/analysis/comparison \
            --outdir {wildcards.ds}/analysis/methods
        """
