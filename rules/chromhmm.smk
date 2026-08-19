# Standard ChromHMM pipeline rules.
#
# All rules are parameterised by a {folder} wildcard that can be either the
# dataset root ({ds}) for pooled BAMs or a replicate subdirectory ({ds}/rep1,
# {ds}/rep2) for per-replicate runs.  The BAM source is always {folder}/bams/.

# All rules are parameterised by a {folder} wildcard (dataset root or replicate
# subdir) and a {caller} wildcard (omni | homer).
# This file owns:
#   - Peak -> binary matrix:  {folder}/{caller}/chromhmm_peaks/

# Reference ChromHMM markup download.

_MARKUPS_DIR = os.path.join(workflow.basedir, "markups")

rule download_markups:
    """Download ENCODE reference ChromHMM BED files into markups/."""
    output: directory(_MARKUPS_DIR)
    shell:
        "bash download_chromhmm.sh {_MARKUPS_DIR}"

# --- Default ChromHMM binarization ---------------------------------------

rule make_cellmark_table:
    input:
        bams=lambda w: [ancient(f"{w.folder}/bams/{m}.bam") for m in get_marks_for_folder(w.folder)],
        controls=lambda w: ([ancient(f"{w.folder}/controls/{m}.bam") for m in get_marks_for_folder(w.folder)]
                            if folder_has_controls(w.folder) else []),
    output: temp("{folder}/chromhmm_default/cellmarkfiletable.tsv")
    params:
        cell=lambda w: DATASETS[ds_of(w.folder)]["cell"],
        marks=lambda w: get_marks_for_folder(w.folder)
    run:
        os.makedirs(os.path.dirname(output[0]),exist_ok=True)
        _has_ctrl = folder_has_controls(wildcards.folder)
        with open(output[0],"w") as f:
            for m in params.marks:
                if _has_ctrl:
                    f.write(f"{params.cell}\t{m}\t{m}.bam\t{m}.bam\n")
                else:
                    f.write(f"{params.cell}\t{m}\t{m}.bam\n")


rule chromhmm_binarize_bam:
    input:
        table=ancient("{folder}/chromhmm_default/cellmarkfiletable.tsv"),
        bams=lambda w: [ancient(f"{w.folder}/bams/{m}.bam") for m in get_marks_for_folder(w.folder)],
        controls=lambda w: ([ancient(f"{w.folder}/controls/{m}.bam") for m in get_marks_for_folder(w.folder)]
                            if folder_has_controls(w.folder) else []),
    output:
        # Kept (not temp) so binarized-emission information remains available for
        # per-state emission analysis after the pipeline finishes.
        bins=expand("{{folder}}/chromhmm_default/{{cell}}_{chr}_binary.txt",chr=CHROMS)
    params:
        bin=CHROMHMM_BIN,
        cs=TOOLS["chromsizes"],
        bamdir="{folder}/bams",
        controldir=lambda w: f"-c {w.folder}/controls" if folder_has_controls(w.folder) else "",
        outdir="{folder}/chromhmm_default",
    shell:
        r"""
        trap "rm -f {output.bins} {params.outdir}/chromsizes.txt" ERR
        mkdir -p {params.outdir}
        # Filter chromsizes to only include canonical chromosomes
        grep -v "_" {params.cs} > {params.outdir}/chromsizes.txt
        {CHROMHMM} BinarizeBam -b {params.bin} {params.controldir} {params.outdir}/chromsizes.txt \
            {params.bamdir} {input.table} {params.outdir}
        rm {params.outdir}/chromsizes.txt
        """


rule chromhmm_learn_default:
    input: ancient(_default_binary_files)
    output:
        dense="{folder}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_dense.bed",
        segments="{folder}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_segments.bed",
        emissions="{folder}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_emissions.txt",
        transitions="{folder}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_transitions.txt",
    params:
        indir="{folder}/chromhmm_default",
        outdir="{folder}/chromhmm_default_result",
        bin=CHROMHMM_BIN,
        n=NSTATES,
        genome=GENOME,
    shell:
        r"""
        mkdir -p {params.outdir}
        {CHROMHMM} LearnModel -b {params.bin} {params.indir} {params.outdir} {params.n} {params.genome}
        # ChromHMM LearnModel might not prefix emissions/transitions correctly.
        # Ensure outputs match Snakemake's expectations.
        [ -f {params.outdir}/emissions_{params.n}.txt ] && mv {params.outdir}/emissions_{params.n}.txt {output.emissions}
        [ -f {params.outdir}/transitions_{params.n}.txt ] && mv {params.outdir}/transitions_{params.n}.txt {output.transitions}
        # dense and segments are usually already prefixed if the input files had the cell prefix,
        # but let's be safe if they aren't.
        [ -f {params.outdir}/{params.n}_dense.bed ] && mv {params.outdir}/{params.n}_dense.bed {output.dense}
        [ -f {params.outdir}/{params.n}_segments.bed ] && mv {params.outdir}/{params.n}_segments.bed {output.segments}
        true
        """


rule chromhmm_default_mark_beds:
    """Extract per-mark BED files from default binarized per-chromosome files."""
    input: ancient(_default_binary_files)
    output:
        directory("{folder}/chromhmm_default_result_files")
    params:
        bin=CHROMHMM_BIN,
        indir="{folder}/chromhmm_default",
        outdir="{folder}/chromhmm_default_result_files",
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/binarized_to_bed.py --bin {params.bin} "
        "--outdir {params.outdir} {input}"


rule chromhmm_default_mark_bed:
    """Individual mark BED file, linked from the bulk extraction directory."""
    input: "{folder}/chromhmm_default_result_files"
    output: "{folder}/chromhmm_default_result/{mark}.bed"
    shell:
        "mkdir -p $(dirname {output}) && ln -f {input}/{wildcards.mark}.bed {output}"




