# Reference ChromHMM markup analysis and matched-segmentation analysis.

_MARKUPS_DIR = os.path.join(workflow.basedir,"markups")

rule download_markups:
    """Download ENCODE reference ChromHMM BED files into markups/."""
    output: os.path.join(_MARKUPS_DIR, "15state", "ENCFF393FJX_Heart_right_ventricle.bed.gz")
    shell:
        "bash {SCRIPTS_DIR}/download_chromhmm.sh {_MARKUPS_DIR}"


rule analyze_markups_chromhmm:
    """Violin plots and stats for ENCODE 15-state / 18-state reference annotations."""
    input: os.path.join(_MARKUPS_DIR, "15state", "ENCFF393FJX_Heart_right_ventricle.bed.gz")
    output: os.path.join(_MARKUPS_DIR, "stats_15state.tsv")
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/analyze_downloaded.py --dir {_MARKUPS_DIR}"


rule analyze_markups_matched:
    """Per-dataset analysis of all *_matched.bed segmentations."""
    input: _analysis_inputs
    output: "{ds}/matched_stats_all.tsv"
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/analyze_matched.py --dir {wildcards.ds}"
