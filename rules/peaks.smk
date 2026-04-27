# Peak-file analysis: number of peaks, mean/median length, replicate Jaccard.
# Output: {ds}/peaks/


def _peak_analysis_inputs(w):
    """All peak files needed for the analysis of dataset {ds}."""
    ds  = w.ds
    cfg = DATASETS[ds]
    cell = cfg["cell"]
    files = []
    for folder in _folders(ds):
        for mark in MARKS:
            files.append(peak_file(folder, "omni",  mark))
            files.append(peak_file(folder, "homer", mark))
            files.append(peak_file(folder, "macs2", mark))
        # ChromHMM binary files (one representative chrom is enough as input
        # sentinel; analyze_peaks.py globs all of them at runtime)
        files += [f"{folder}/chromhmm_default/{cell}_{c}_binary.txt" for c in CHROMS]
    return files


rule analyze_peaks:
    """Compute per-mark peak stats and replicate Jaccard for all callers."""
    input: _peak_analysis_inputs
    output:
        stats   = "{ds}/peaks/peak_stats.tsv",
        n_peaks = "{ds}/peaks/n_peaks.png",
        mean    = "{ds}/peaks/mean_length.png",
        median  = "{ds}/peaks/median_length.png",
    conda: "../envs/python.yaml"
    params:
        cell         = lambda w: DATASETS[w.ds]["cell"],
        omni_bin     = OMNI_BIN,
        chromhmm_bin = CHROMHMM_BIN,
        marks        = " ".join(MARKS),
        scripts_dir  = SCRIPTS_DIR,
    shell:
        r"""
        mkdir -p {wildcards.ds}/peaks
        python {params.scripts_dir}/analyze_peaks.py \
            --ds        {wildcards.ds} \
            --cell      {params.cell} \
            --marks     {params.marks} \
            --omni-bin  {params.omni_bin} \
            --chromhmm-bin {params.chromhmm_bin} \
            --outdir    {wildcards.ds}/peaks
        """
