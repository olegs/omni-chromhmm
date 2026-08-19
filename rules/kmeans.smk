# Standard Kmeans segmentation pipeline rules.
#
# All rules are parameterised by a {folder} wildcard that can be either the
# dataset root ({ds}) for pooled BAMs or a replicate subdirectory ({ds}/rep1,
# {ds}/rep2) for per-replicate runs.  The BAM source is always {folder}/bams/.

# All rules are parameterised by a {folder} wildcard (dataset root or replicate
# subdir) and a {caller} wildcard (omni | homer).
# This file owns:
#   - KMeans segmentation:    {folder}/{caller}/{caller}_kmeans_states.bed

rule kmeans_states:
    """Binarize peaks and run KMeans clustering in one step using peaks_segmentation.py."""
    input:
        peaks=lambda w: [ancient(peak_file(w.folder, w.caller, m)) for m in get_marks_for_folder(w.folder)],
        cs=ancient(TOOLS["chromsizes"])
    output:
        kmeans="{folder}/{caller}/{caller}_kmeans_states.bed",
        bins=expand("{{folder}}/{{caller}}/chromhmm_peaks/{chr}_binary.txt.gz", chr=CHROMS)
    log: "{folder}/{caller}/{caller}_kmeans_states.log"
    conda: "../envs/python.yaml"
    params:
        bin=lambda w: CALLER_BIN[w.caller],
        n=NSTATES,
        marks=lambda w: ",".join(get_marks_for_folder(w.folder)),
        cell=lambda w: DATASETS[ds_of(w.folder)]["cell"],
        outdir="{folder}/{caller}/chromhmm_peaks"
    shell:
        "python {SCRIPTS_DIR}/peaks_segmentation.py "
        "--bin {params.bin} --chromsizes {input.cs} --marks {params.marks} "
        "--peaks {input.peaks} --states {params.n} --out {output.kmeans} "
        "--save-binary {params.outdir} --cell {params.cell} "
        "&> {log}"
