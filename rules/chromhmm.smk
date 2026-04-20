# Standard ChromHMM pipeline rules.
#
# Top-level default pipeline uses pooled BAMs at {ds}/bams_pooled/.
# Per-replicate pipelines (rep1, rep2) use {ds}/{mode}/bams_pooled/.


# --- Top-level default ChromHMM (pooled BAMs) ----------------------------

rule make_cellmark_table:
    input:  lambda w: [f"{w.ds}/bams_pooled/{m}.bam" for m in MARKS]
    output: "{ds}/chromhmm_default/cellmarkfiletable.tsv"
    params: cell = lambda w: DATASETS[w.ds]["cell"]
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], "w") as f:
            for m in MARKS:
                f.write(f"{params.cell}\t{m}\t{m}.bam\n")


rule chromhmm_binarize_bam:
    input:
        table = "{ds}/chromhmm_default/cellmarkfiletable.tsv",
        bams  = lambda w: [f"{w.ds}/bams_pooled/{m}.bam" for m in MARKS],
    output:
        flag = "{ds}/chromhmm_default/.binarized"
    params:
        bin = BIN,
        cs = TOOLS["chromsizes"],
        bamdir = "{ds}/bams_pooled",
        outdir = "{ds}/chromhmm_default",
    shell:
        "mkdir -p {params.outdir} && "
        "{CHROMHMM} BinarizeBam -b {params.bin} {params.cs} "
        "{params.bamdir} {input.table} {params.outdir} && "
        "touch {output.flag}"


rule chromhmm_default_input_bed:
    """Convert per-chromosome binarized files to a merged BED."""
    input:  "{ds}/chromhmm_default/.binarized"
    output: "{ds}/chromhmm_default_input.bed"
    params:
        bin = BIN,
        indir = "{ds}/chromhmm_default",
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/binarized_to_bed.py --bin {params.bin} "
        "{params.indir}/*_binary.txt > {output}"


rule chromhmm_learn_default:
    input:  "{ds}/chromhmm_default/.binarized"
    output:
        dense = "{ds}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_dense.bed",
    params:
        indir  = "{ds}/chromhmm_default",
        outdir = "{ds}/chromhmm_default_result",
        bin = BIN,
        n = NSTATES,
        genome = GENOME,
    shell:
        "mkdir -p {params.outdir} && "
        "{CHROMHMM} LearnModel -b {params.bin} {params.indir} {params.outdir} "
        "{params.n} {params.genome}"


# --- Per-replicate ChromHMM (rep1, rep2) ----------------------------------

rule rep_make_cellmark_table:
    input:  lambda w: [f"{w.ds}/{w.mode}/bams_pooled/{m}.bam" for m in MARKS]
    output: "{ds}/{mode}/bams_pooled/cellmarkfiletable.tsv"
    wildcard_constraints: mode = "rep[12]"
    params: cell = lambda w: DATASETS[w.ds]["cell"]
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], "w") as f:
            for m in MARKS:
                f.write(f"{params.cell}\t{m}\t{m}.bam\n")


rule rep_chromhmm_binarize_bam:
    input:
        table = "{ds}/{mode}/bams_pooled/cellmarkfiletable.tsv",
        bams  = lambda w: [f"{w.ds}/{w.mode}/bams_pooled/{m}.bam" for m in MARKS],
    output:
        flag = "{ds}/{mode}/chromhmm_default/.binarized"
    wildcard_constraints: mode = "rep[12]"
    params:
        bin = BIN,
        cs = TOOLS["chromsizes"],
        bamdir = "{ds}/{mode}/bams_pooled",
        outdir = "{ds}/{mode}/chromhmm_default",
    shell:
        "mkdir -p {params.outdir} && "
        "{CHROMHMM} BinarizeBam -b {params.bin} {params.cs} "
        "{params.bamdir} {input.table} {params.outdir} && "
        "touch {output.flag}"


rule rep_chromhmm_default_input_bed:
    """Convert per-chromosome binarized files to a merged BED."""
    input:  "{ds}/{mode}/chromhmm_default/.binarized"
    output: "{ds}/{mode}/chromhmm_default_input.bed"
    wildcard_constraints: mode = "rep[12]"
    params:
        bin = BIN,
        indir = "{ds}/{mode}/chromhmm_default",
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/binarized_to_bed.py --bin {params.bin} "
        "{params.indir}/*_binary.txt > {output}"


rule rep_chromhmm_learn_default:
    input:  "{ds}/{mode}/chromhmm_default/.binarized"
    output:
        dense = "{ds}/{mode}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_dense.bed",
    wildcard_constraints: mode = "rep[12]"
    params:
        indir  = "{ds}/{mode}/chromhmm_default",
        outdir = "{ds}/{mode}/chromhmm_default_result",
        bin = BIN,
        n = NSTATES,
        genome = GENOME,
    shell:
        "mkdir -p {params.outdir} && "
        "{CHROMHMM} LearnModel -b {params.bin} {params.indir} {params.outdir} "
        "{params.n} {params.genome}"
