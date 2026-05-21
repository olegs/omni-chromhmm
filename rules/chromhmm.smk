# Standard ChromHMM pipeline rules.
#
# All rules are parameterised by a {folder} wildcard that can be either the
# dataset root ({ds}) for pooled BAMs or a replicate subdirectory ({ds}/rep1,
# {ds}/rep2) for per-replicate runs.  The BAM source is always {folder}/bams/.

# All rules are parameterised by a {folder} wildcard (dataset root or replicate
# subdir) and a {caller} wildcard (omni | homer).
# This file owns:
#   - Peak -> binary matrix:  {folder}/{caller}/chromhmm_peaks/
#   - ChromHMM LearnModel over peaks:  {folder}/{caller}/chromhmm_result/
#   - KMeans segmentation:    {folder}/{caller}/kmeans_states.bed

# Reference ChromHMM markup download.

_MARKUPS_DIR = os.path.join(workflow.basedir,"markups")

rule download_markups:
    """Download ENCODE reference ChromHMM BED files into markups/."""
    output: directory(_MARKUPS_DIR)
    shell:
        "bash {SCRIPTS_DIR}/download_chromhmm.sh {_MARKUPS_DIR}"

# --- Default ChromHMM binarization ---------------------------------------

rule make_cellmark_table:
    input:
        bams=lambda w: [f"{w.folder}/bams/{m}.bam" for m in MARKS],
        controls=lambda w: ([f"{w.folder}/controls/{m}.bam" for m in MARKS]
                            if folder_has_controls(w.folder) else []),
    output: "{folder}/chromhmm_default/cellmarkfiletable.tsv"
    params: cell=lambda w: DATASETS[ds_of(w.folder)]["cell"]
    run:
        os.makedirs(os.path.dirname(output[0]),exist_ok=True)
        _has_ctrl = folder_has_controls(wildcards.folder)
        with open(output[0],"w") as f:
            for m in MARKS:
                if _has_ctrl:
                    f.write(f"{params.cell}\t{m}\t{m}.bam\t{m}.bam\n")
                else:
                    f.write(f"{params.cell}\t{m}\t{m}.bam\n")


rule chromhmm_binarize_bam:
    input:
        table="{folder}/chromhmm_default/cellmarkfiletable.tsv",
        bams=lambda w: [f"{w.folder}/bams/{m}.bam" for m in MARKS],
        controls=lambda w: ([f"{w.folder}/controls/{m}.bam" for m in MARKS]
                            if folder_has_controls(w.folder) else []),
    output:
        bins=temp(expand("{{folder}}/chromhmm_default/{{cell}}_{chr}_binary.txt",chr=CHROMS))
    params:
        bin=CHROMHMM_BIN,
        cs=TOOLS["chromsizes"],
        bamdir="{folder}/bams",
        controldir=lambda w: f"-c {w.folder}/controls" if folder_has_controls(w.folder) else "",
        outdir="{folder}/chromhmm_default",
    shell:
        "mkdir -p {params.outdir} && "
        "{CHROMHMM} BinarizeBam -b {params.bin} {params.controldir} {params.cs} "
        "{params.bamdir} {input.table} {params.outdir}"


rule chromhmm_learn_default:
    input: _default_binary_files
    output:
        dense="{folder}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_dense.bed",
    params:
        indir="{folder}/chromhmm_default",
        outdir="{folder}/chromhmm_default_result",
        bin=CHROMHMM_BIN,
        n=NSTATES,
        genome=GENOME,
    shell:
        "mkdir -p {params.outdir} && "
        "{CHROMHMM} LearnModel -b {params.bin} {params.indir} {params.outdir} "
        "{params.n} {params.genome}"


rule chromhmm_default_mark_beds:
    """Extract per-mark BED files from default binarized per-chromosome files."""
    input: _default_binary_files
    output: expand("{{folder}}/chromhmm_default_result/{mark}.bed",mark=MARKS)
    params:
        bin=CHROMHMM_BIN,
        indir="{folder}/chromhmm_default",
        outdir="{folder}/chromhmm_default_result",
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/binarized_to_bed.py --bin {params.bin} "
        "--outdir {params.outdir} {params.indir}/*_binary.txt"


# --- Peak -> ChromHMM binary matrix --------------------------------------

rule cat_peaks_per_mark:
    """Sort peaks for a given folder+caller and mark into the binary matrix input."""
    input: lambda w: peak_file(w.folder,w.caller,w.mark)
    output: temp("{folder}/{caller}/chromhmm_peaks/{mark}")
    shell: "sort -k1,1 -k2,2n {input} > {output}"


rule multiinter:
    input:
        bins=lambda w: f"bins{CALLER_BIN[w.caller]}.bed",
        peaks=lambda w: [f"{w.folder}/{w.caller}/chromhmm_peaks/{m}" for m in MARKS],
    output: temp("{folder}/{caller}/chromhmm_peaks/multiinter.tsv")
    conda: "../envs/bio.yaml"
    shell:
        "bash {SCRIPTS_DIR}/multiinter.sh {output} {input.bins} {input.peaks}"


rule binarize_per_chr:
    """Extract per-chromosome binary matrix (mark columns) and gzip."""
    input: "{folder}/{caller}/chromhmm_peaks/multiinter.tsv"
    output: temp("{folder}/{caller}/chromhmm_peaks/{cell}_{chr}_binary.txt.gz")
    shell:
        "bash {SCRIPTS_DIR}/binarize_per_chr.sh {input} {wildcards.cell} {wildcards.chr} {output}"

# --- Segmentations over peak binarization --------------------------------

rule chromhmm_learn_over_peaks:
    input: _peaks_binary_files
    output:
        dense="{folder}/{caller}/chromhmm_result/{cell}_" + str(NSTATES) + "_dense.bed",
    log: "{folder}/{caller}/chromhmm_result/{cell}_learn.log"
    params:
        indir="{folder}/{caller}/chromhmm_peaks",
        outdir="{folder}/{caller}/chromhmm_result",
        bin=lambda w: CALLER_BIN[w.caller],
        n=NSTATES,
        genome=GENOME,
        chromsizes=TOOLS["chromsizes"],
    shell:
        "mkdir -p {params.outdir} && "
        "{CHROMHMM} LearnModel -b {params.bin} -l {params.chromsizes} "
        "{params.indir} {params.outdir} "
        "{params.n} {params.genome} "
        "&> {log}"


rule kmeans_states:
    input: _peaks_binary_files
    output: "{folder}/{caller}/kmeans_states.bed"
    log: "{folder}/{caller}/kmeans_states.log"
    conda: "../envs/python.yaml"
    params:
        bin=lambda w: CALLER_BIN[w.caller],
        n=NSTATES,
        indir="{folder}/{caller}/chromhmm_peaks",
    shell:
        "python {SCRIPTS_DIR}/states.py --bin {params.bin} --states {params.n} "
        "--inputs {params.indir}/*.gz > {output} "
        "2> {log}"
