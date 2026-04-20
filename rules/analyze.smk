# Downstream analysis: analyze.py produces, per segmentation, the PDF's
# standard plots — segment-length distribution, state feature means,
# functional enrichment vs ChromHMM COORDS and (when available) RNA-seq
# expression quantiles.


# --- per-segmentation analysis plots --------------------------------------

def _analysis_inputs(w):
    """Every matched segmentation for a dataset: one sentinel covers all of
    them so analyze.py invocations stay in a single rule."""
    ds = w.ds
    cfg = DATASETS[ds]
    cell = cfg["cell"]
    files = [
        _ref_bed(ds),
        f"{ds}/chromhmm_default_input.bed",
        f"{ds}/chromhmm_default_result/{cell}_{NSTATES}_chromhmm_default_matched.bed",
    ]
    for mode in _modes_for(ds):
        files.append(f"{ds}/{mode}/chromhmm_result/{cell}_{NSTATES}_chromhmm_{mode}_matched.bed")
        files.append(f"{ds}/{mode}/gmm_{mode}_matched.bed")
        files.append(f"{ds}/{mode}/kmeans_{mode}_matched.bed")
    # Per-replicate segmentations
    if cfg.get("replicates"):
        for rep in ["rep1", "rep2"]:
            files.append(f"{ds}/{rep}/chromhmm_default_result/{cell}_{NSTATES}_chromhmm_default_matched.bed")
            files.append(f"{ds}/{rep}/chromhmm_result/{cell}_{NSTATES}_chromhmm_{rep}_matched.bed")
            files.append(f"{ds}/{rep}/gmm_{rep}_matched.bed")
            files.append(f"{ds}/{rep}/kmeans_{rep}_matched.bed")
    return files


rule analyze_segmentations:
    """Run analyze.py on every matched segmentation for a dataset. Also triggers
    enrichment against ChromHMM COORDS and, when available, RNA-seq quantiles."""
    input:
        segs = _analysis_inputs,
        rna  = lambda w: (f"{w.ds}/rnaseq_{DATASETS[w.ds]['rnaseq']}.tsv"
                          if DATASETS[w.ds].get("rnaseq") else []),
    output: touch("{ds}/analysis/.done")
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
    shell:
        r"""
        mkdir -p {wildcards.ds}/analysis
        RNA_ANN=""
        if [ -n "{input.rna}" ]; then RNA_ANN="{input.rna}"; fi

        # Reference
        python {params.scripts_dir}/analyze.py --seg {params.ref} --bin {params.bin} \
            --outdir {wildcards.ds}/analysis/ref --annotations {params.coords}/*.bed.gz $RNA_ANN

        # Default ChromHMM
        python {params.scripts_dir}/analyze.py --seg {wildcards.ds}/chromhmm_default_result/{params.cell}_{params.n}_chromhmm_default_matched.bed \
            --bin {params.bin} --outdir {wildcards.ds}/chromhmm_default_result \
            --inputs {wildcards.ds}/chromhmm_default/*.txt \
            --annotations {params.coords}/*.bed.gz $RNA_ANN

        # Omnipeak modes
        for mode in {params.modes}; do
            python {params.scripts_dir}/analyze.py --seg {wildcards.ds}/$mode/chromhmm_result/{params.cell}_{params.n}_chromhmm_${{mode}}_matched.bed \
                --bin {params.bin} --outdir {wildcards.ds}/$mode/chromhmm_result \
                --inputs {wildcards.ds}/$mode/chromhmm_peaks/*.txt.gz \
                --annotations {params.coords}/*.bed.gz $RNA_ANN

            python {params.scripts_dir}/analyze.py --seg {wildcards.ds}/$mode/gmm_${{mode}}_matched.bed \
                --bin {params.bin} --outdir {wildcards.ds}/$mode/gmm \
                --inputs {wildcards.ds}/$mode/chromhmm_peaks/*.txt.gz \
                --annotations {params.coords}/*.bed.gz $RNA_ANN

            python {params.scripts_dir}/analyze.py --seg {wildcards.ds}/$mode/kmeans_${{mode}}_matched.bed \
                --bin {params.bin} --outdir {wildcards.ds}/$mode/kmeans \
                --inputs {wildcards.ds}/$mode/chromhmm_peaks/*.txt.gz \
                --annotations {params.coords}/*.bed.gz $RNA_ANN
        done

        # Per-replicate analysis
        for rep in {params.reps}; do
            # Default ChromHMM per replicate
            python {params.scripts_dir}/analyze.py --seg {wildcards.ds}/$rep/chromhmm_default_result/{params.cell}_{params.n}_chromhmm_default_matched.bed \
                --bin {params.bin} --outdir {wildcards.ds}/$rep/chromhmm_default_result \
                --inputs {wildcards.ds}/$rep/chromhmm_default/*.txt \
                --annotations {params.coords}/*.bed.gz $RNA_ANN

            # ChromHMM over Omnipeak per replicate
            python {params.scripts_dir}/analyze.py --seg {wildcards.ds}/$rep/chromhmm_result/{params.cell}_{params.n}_chromhmm_${{rep}}_matched.bed \
                --bin {params.bin} --outdir {wildcards.ds}/$rep/chromhmm_result \
                --inputs {wildcards.ds}/$rep/chromhmm_peaks/*.txt.gz \
                --annotations {params.coords}/*.bed.gz $RNA_ANN

            # GMM per replicate
            python {params.scripts_dir}/analyze.py --seg {wildcards.ds}/$rep/gmm_${{rep}}_matched.bed \
                --bin {params.bin} --outdir {wildcards.ds}/$rep/gmm \
                --inputs {wildcards.ds}/$rep/chromhmm_peaks/*.txt.gz \
                --annotations {params.coords}/*.bed.gz $RNA_ANN

            # KMeans per replicate
            python {params.scripts_dir}/analyze.py --seg {wildcards.ds}/$rep/kmeans_${{rep}}_matched.bed \
                --bin {params.bin} --outdir {wildcards.ds}/$rep/kmeans \
                --inputs {wildcards.ds}/$rep/chromhmm_peaks/*.txt.gz \
                --annotations {params.coords}/*.bed.gz $RNA_ANN
        done

        """
