# Omnipeak peak calling and Omnipeak-driven segmentations.
#
# Omnipeak modes (via {mode} wildcard):
#   omni       - one Omnipeak call on the pooled BAM per mark.
#   replicated - Omnipeak's native multi-replicate mode (comma-separated -t).
#   rep1/rep2  - per-replicate Omnipeak on individual replicate BAMs.
#
# Downstream each mode feeds three segmentations:
#   ChromHMM LearnModel on Omnipeak-binarized inputs (chromhmm_result/).
#   GMM / MiniBatchKMeans states via states.py (states.bed).


# --- peak calling --------------------------------------------------------

rule omnipeak_pooled:
    input:
        bam = "{ds}/bams_pooled/{mark}.bam",
    output:
        peak  = f"{{ds}}/pooled_omni/{{mark}}_pooled_{BIN}.peak",
        model = f"{{ds}}/pooled_omni/{{mark}}_pooled_{BIN}.omni",
    threads: 4
    log: f"{{ds}}/pooled_omni/{{mark}}_pooled_{BIN}.log"
    params:
        bin = BIN,
        cs = TOOLS["chromsizes"],
        extra = lambda w: DATASETS[w.ds].get("omnipeak_extra", ""),
        wdir = lambda w: f"{w.ds}/pooled_omni",
    shell:
        "mkdir -p {params.wdir} && "
        "{OMNIPEAK} analyze -t {input.bam} -cs {params.cs} --bin {params.bin} "
        "--threads 4 -w {params.wdir} -m {output.model} -p {output.peak} --clip 0 {params.extra} "
        "&> {log}"


rule omnipeak_replicated:
    """Omnipeak's native multi-replicate mode (comma-separated BAMs)."""
    input:  lambda w: bams_for_mark(w.ds, w.mark)
    output:
        peak  = f"{{ds}}/replicated_omni/{{mark}}_replicated_{BIN}.peak",
        model = f"{{ds}}/replicated_omni/{{mark}}_replicated_{BIN}.omni",
    threads: 4
    log: f"{{ds}}/replicated_omni/{{mark}}_replicated_{BIN}.log"
    params:
        bin = BIN,
        cs  = TOOLS["chromsizes"],
        extra = lambda w: DATASETS[w.ds].get("omnipeak_extra", ""),
        wdir = lambda w: f"{w.ds}/replicated_omni",
    shell:
        r"""
        mkdir -p {params.wdir}
        T=$(echo {input} | tr ' ' ',')
        {OMNIPEAK} analyze -t "$T" -cs {params.cs} -b {params.bin} \
            --threads 4 -w {params.wdir} -m {output.model} -p {output.peak} --clip 0 {params.extra} \
            &> {log}
        """


rule rep_omnipeak_per_mark:
    """Per-replicate Omnipeak on the replicate's BAM."""
    input:  lambda w: bams_for_mark(w.ds, w.mark, rep=w.mode)
    output:
        peak  = f"{{ds}}/{{mode}}_omni/{{mark}}_{{mode}}_{BIN}.peak",
        model = f"{{ds}}/{{mode}}_omni/{{mark}}_{{mode}}_{BIN}.omni",
    wildcard_constraints: mode = "rep[12]"
    log: f"{{ds}}/{{mode}}_omni/{{mark}}_{{mode}}_{BIN}.log"
    params:
        bin = BIN, cs = TOOLS["chromsizes"],
        extra = lambda w: DATASETS[w.ds].get("omnipeak_extra", ""),
        wdir = lambda w: f"{w.ds}/{w.mode}_omni",
    shell:
        "mkdir -p {params.wdir} && "
        "{OMNIPEAK} analyze -t {input} -b {params.bin} -cs {params.cs} "
        "-w {params.wdir} -m {output.model} -p {output.peak} --clip 0 {params.extra} "
        "&> {log}"


# --- Omnipeak -> ChromHMM binary matrix ----------------------------------

rule mode_cat_peaks_per_mark:
    """Sort peaks for a given mode and mark."""
    input:  lambda w: _mode_peak(w.ds, w.mode, w.mark)
    output: "{ds}/{mode}/chromhmm_peaks/{mark}"
    shell:  "sort -k1,1 -k2,2n {input} > {output}"


rule mode_multiinter:
    input:
        bins = f"bins{BIN}.bed",
        peaks = lambda w: [f"{w.ds}/{w.mode}/chromhmm_peaks/{m}" for m in MARKS],
    output: "{ds}/{mode}/chromhmm_peaks/multiinter.tsv"
    conda: "../envs/bio.yaml"
    shell:
        "bash {SCRIPTS_DIR}/multiinter.sh {output} {input.bins} {input.peaks}"


rule mode_binarize_per_chr:
    """Extract per-chromosome binary matrix (6 mark columns) and gzip."""
    input:  "{ds}/{mode}/chromhmm_peaks/multiinter.tsv"
    output: "{ds}/{mode}/chromhmm_peaks/{cell}_{chr}_binary.txt.gz"
    shell:
        "bash {SCRIPTS_DIR}/binarize_per_chr.sh {input} {wildcards.cell} {wildcards.chr} {output}"


# --- Segmentations over Omnipeak binarization ----------------------------

rule chromhmm_learn_over_omnipeak:
    input:  lambda w: _mode_binary_files(w)
    output:
        dense = "{ds}/{mode}/chromhmm_result/" + "{cell}_" + str(NSTATES) + "_dense.bed",
    log: "{ds}/{mode}/chromhmm_result/{cell}_learn.log"
    params:
        indir  = lambda w: f"{w.ds}/{w.mode}/chromhmm_peaks",
        outdir = lambda w: f"{w.ds}/{w.mode}/chromhmm_result",
        bin = BIN,
        n = NSTATES,
        genome = GENOME,
        chromsizes = TOOLS["chromsizes"],
    shell:
        "mkdir -p {params.outdir} && "
        "{CHROMHMM} LearnModel -b {params.bin} -l {params.chromsizes} "
        "{params.indir} {params.outdir} "
        "{params.n} {params.genome} "
        "&> {log}"


rule gmm_states:
    input:  lambda w: _mode_binary_files(w)
    output: "{ds}/{mode}/gmm_states.bed"
    log: "{ds}/{mode}/gmm_states.log"
    conda: "../envs/python.yaml"
    params:
        bin = BIN,
        n = NSTATES,
        indir = lambda w: f"{w.ds}/{w.mode}/chromhmm_peaks",
    shell:
        "python {SCRIPTS_DIR}/states.py --method gmm --bin {params.bin} --states {params.n} "
        "--inputs {params.indir}/*.gz > {output} "
        "2> {log}"


rule kmeans_states:
    input:  lambda w: _mode_binary_files(w)
    output: "{ds}/{mode}/kmeans_states.bed"
    log: "{ds}/{mode}/kmeans_states.log"
    conda: "../envs/python.yaml"
    params:
        bin = BIN,
        n = NSTATES,
        indir = lambda w: f"{w.ds}/{w.mode}/chromhmm_peaks",
    shell:
        "python {SCRIPTS_DIR}/states.py --method kmeans --bin {params.bin} --states {params.n} "
        "--inputs {params.indir}/*.gz > {output} "
        "2> {log}"
