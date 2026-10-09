# Standard Kmeans segmentation pipeline rules.
#
# All rules are parameterised by a {folder} wildcard that can be either the
# dataset root ({ds}) for pooled BAMs or a replicate subdirectory ({ds}/rep1,
# {ds}/rep2) for per-replicate runs.  The BAM source is always {folder}/bams/.

# All rules are parameterised by a {folder} wildcard (dataset root or replicate
# subdir) and a {caller} wildcard (omni | homer).
# This file owns:
#   - KMeans segmentation:    {folder}/{caller}/{caller}_kmeans_states.bed
#   - BMM3 segmentation:      {folder}/{caller}/{caller}_bmm3_states.bed

rule binarize_peaks:
    """Binarize peaks for a given caller and folder."""
    input:
        peaks=lambda w: [ancient(peak_file(w.folder, w.caller, m)) for m in get_marks_for_folder(w.folder)],
        cs=ancient(TOOLS["chromsizes"])
    output:
        bins=expand("{{folder}}/{{caller}}/chromhmm_peaks/{chr}_binary.txt.gz", chr=CHROMS)
    log: "{folder}/{caller}/binarize_peaks.log"
    conda: "../envs/python.yaml"
    resources: mem_mb=2000
    params:
        bin=lambda w: CALLER_BIN[w.caller],
        marks=lambda w: ",".join(get_marks_for_folder(w.folder)),
        cell=lambda w: DATASETS[ds_of(w.folder)]["cell"],
        outdir="{folder}/{caller}/chromhmm_peaks"
    shell:
        "python {SCRIPTS_DIR}/peaks_segmentation.py "
        "--bin {params.bin} --chromsizes {input.cs} --marks {params.marks} "
        "--peaks {input.peaks} --states 0 --save-binary {params.outdir} --cell {params.cell} "
        "> {log} 2>&1"


rule kmeans_states:
    """Run KMeans clustering on binarized peaks."""
    input:
        peaks=lambda w: [ancient(peak_file(w.folder, w.caller, m)) for m in get_marks_for_folder(w.folder)],
        cs=ancient(TOOLS["chromsizes"]),
        bins=expand("{{folder}}/{{caller}}/chromhmm_peaks/{chr}_binary.txt.gz", chr=CHROMS)
    output:
        kmeans="{folder}/{caller}/{caller}_kmeans_states.bed",
    log: "{folder}/{caller}/{caller}_kmeans_states.log"
    conda: "../envs/python.yaml"
    resources: mem_mb=4000
    params:
        bin=lambda w: CALLER_BIN[w.caller],
        n=NSTATES,
        marks=lambda w: ",".join(get_marks_for_folder(w.folder)),
        cell=lambda w: DATASETS[ds_of(w.folder)]["cell"],
    shell:
        "python {SCRIPTS_DIR}/peaks_segmentation.py "
        "--bin {params.bin} --chromsizes {input.cs} --marks {params.marks} "
        "--peaks {input.peaks} --states {params.n} --out {output.kmeans} "
        "--cell {params.cell} "
        "> {log} 2>&1"


rule bmm3_states:
    """Run the 3-bin spatial BMM clustering on binarized peaks."""
    input:
        peaks=lambda w: [ancient(peak_file(w.folder, w.caller, m)) for m in get_marks_for_folder(w.folder)],
        cs=ancient(TOOLS["chromsizes"]),
        bins=expand("{{folder}}/{{caller}}/chromhmm_peaks/{chr}_binary.txt.gz", chr=CHROMS)
    output:
        bmm="{folder}/{caller}/{caller}_bmm3_states.bed",
    log: "{folder}/{caller}/{caller}_bmm3_states.log"
    conda: "../envs/python.yaml"
    # The spatial fit holds the binarization, its pattern codes and the table
    # of the patterns the track shows - at the 13 marks and 100 bp bins of a
    # SAGAconf replicate that is well past what the plain mixture needed.
    resources: mem_mb=8000
    params:
        bin=lambda w: CALLER_BIN[w.caller],
        n=NSTATES,
        marks=lambda w: ",".join(get_marks_for_folder(w.folder)),
        cell=lambda w: DATASETS[ds_of(w.folder)]["cell"],
    shell:
        "python {SCRIPTS_DIR}/peaks_segmentation.py "
        "--bin {params.bin} --chromsizes {input.cs} --marks {params.marks} "
        "--peaks {input.peaks} --states {params.n} --out {output.bmm} "
        "--cell {params.cell} --mixture --spatial-bins 3 "
        "> {log} 2>&1"
