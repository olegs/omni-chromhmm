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


def _default_binary_files(w):
    cell = DATASETS[w.ds]["cell"]
    return [f"{w.ds}/chromhmm_default/{cell}_{c}_binary.txt" for c in CHROMS]


rule chromhmm_binarize_bam:
    input:
        table = "{ds}/chromhmm_default/cellmarkfiletable.tsv",
        bams  = lambda w: [f"{w.ds}/bams_pooled/{m}.bam" for m in MARKS],
    output:
        bins = expand("{{ds}}/chromhmm_default/{{cell}}_{chr}_binary.txt", chr=CHROMS)
    params:
        bin = BIN,
        cs = TOOLS["chromsizes"],
        bamdir = "{ds}/bams_pooled",
        outdir = "{ds}/chromhmm_default",
    shell:
        "mkdir -p {params.outdir} && "
        "{CHROMHMM} BinarizeBam -b {params.bin} {params.cs} "
        "{params.bamdir} {input.table} {params.outdir}"



rule chromhmm_learn_default:
    input:  _default_binary_files
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


rule chromhmm_default_mark_beds:
    """Extract per-mark BED files from default binarized per-chromosome files."""
    input:  _default_binary_files
    output: expand("{{ds}}/chromhmm_default_result/{mark}.bed", mark=MARKS)
    params:
        bin = BIN,
        indir = "{ds}/chromhmm_default",
        outdir = "{ds}/chromhmm_default_result",
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/binarized_to_bed.py --bin {params.bin} "
        "--outdir {params.outdir} {params.indir}/*_binary.txt"


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


def _rep_default_binary_files(w):
    cell = DATASETS[w.ds]["cell"]
    return [f"{w.ds}/{w.mode}/chromhmm_default/{cell}_{c}_binary.txt" for c in CHROMS]


rule rep_chromhmm_binarize_bam:
    input:
        table = "{ds}/{mode}/bams_pooled/cellmarkfiletable.tsv",
        bams  = lambda w: [f"{w.ds}/{w.mode}/bams_pooled/{m}.bam" for m in MARKS],
    output:
        bins = expand("{{ds}}/{{mode}}/chromhmm_default/{{cell}}_{chr}_binary.txt", chr=CHROMS)
    wildcard_constraints: mode = "rep[12]"
    params:
        bin = BIN,
        cs = TOOLS["chromsizes"],
        bamdir = "{ds}/{mode}/bams_pooled",
        outdir = "{ds}/{mode}/chromhmm_default",
    shell:
        "mkdir -p {params.outdir} && "
        "{CHROMHMM} BinarizeBam -b {params.bin} {params.cs} "
        "{params.bamdir} {input.table} {params.outdir}"


rule rep_chromhmm_default_mark_beds:
    """Extract per-mark BED files from replicate default binarized files."""
    input:  _rep_default_binary_files
    output: expand("{{ds}}/{{mode}}/chromhmm_default_result/{mark}.bed", mark=MARKS)
    wildcard_constraints: mode = "rep[12]"
    params:
        bin = BIN,
        indir = "{ds}/{mode}/chromhmm_default",
        outdir = "{ds}/{mode}/chromhmm_default_result",
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/binarized_to_bed.py --bin {params.bin} "
        "--outdir {params.outdir} {params.indir}/*_binary.txt"


rule rep_chromhmm_learn_default:
    input:  _rep_default_binary_files
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
