# Reference ChromHMM markup analysis and matched-segmentation analysis.

_MARKUPS_DIR = os.path.join(workflow.basedir,"markups")

rule download_markups:
    """Download ENCODE reference ChromHMM BED files into markups/."""
    output: touch(os.path.join(_MARKUPS_DIR,".downloaded"))
    shell:
        "bash {SCRIPTS_DIR}/download_chromhmm.sh"


rule analyze_markups_chromhmm:
    """Violin plots and stats for ENCODE 15-state / 18-state reference annotations."""
    input: os.path.join(_MARKUPS_DIR,".downloaded")
    output: touch(os.path.join(_MARKUPS_DIR,"plots",".done"))
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/analyze_downloaded.py --dir {_MARKUPS_DIR}"


rule analyze_markups_matched:
    """Per-dataset analysis of all *_matched.bed segmentations."""
    input: _analysis_inputs
    output: touch("{ds}/plots_matched/.done")
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/analyze_matched.py --dir {wildcards.ds}"
