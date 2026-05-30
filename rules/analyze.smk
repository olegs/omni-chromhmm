# Per-segmentation analysis (driven by analyze.py):
#   - Report, segment lengths, state emissions, enrichment
#
# The rule is generic in {folder}: it operates on whichever segmentations
# live directly inside that folder (chromhmm_default*/, omni/, homer/).


def _folder_seg_files(w):
    """All segmentation BEDs analyzed by analyze.py within {folder}.

    Returns BEDs + their bw_emissions.npz (so that both binarized-track
    and bigwig emission plots are produced when analyze=True).
    """
    folder = w.folder
    cell = DATASETS[ds_of(folder)]["cell"]
    # Unmatched default ChromHMM (emissions-only pass)
    beds = [
        f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense.bed",
    ]
    # All matched BEDs (combined, bigwig-emission-only, and overlap-only)
    beds += [
        f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense_comb_matched.bed",
        f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense_bwem_matched.bed",
        f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense_ovlp_matched.bed",
    ]
    for caller in CALLERS:
        if DO_CHROMHMM_PEAKS:
            beds.append(f"{folder}/{caller}/chromhmm_result/{caller}_{cell}_{NSTATES}_dense_comb_matched.bed")
            beds.append(f"{folder}/{caller}/chromhmm_result/{caller}_{cell}_{NSTATES}_dense_bwem_matched.bed")
            beds.append(f"{folder}/{caller}/chromhmm_result/{caller}_{cell}_{NSTATES}_dense_ovlp_matched.bed")
        beds.append(f"{folder}/{caller}/{caller}_kmeans_states_comb_matched.bed")
        beds.append(f"{folder}/{caller}/{caller}_kmeans_states_bwem_matched.bed")
        beds.append(f"{folder}/{caller}/{caller}_kmeans_states_ovlp_matched.bed")
    # Require bw_emissions.npz for every BED.  For unmatched beds this triggers
    # compute_emissions (bigwig re-read); for _matched beds the match rules produce
    # a remapped copy without re-reading bigwigs (via --remap-emissions).
    npzs = [b.replace(".bed", ".bw_emissions.npz") for b in beds]
    return beds + npzs


def _folder_analysis_inputs(w):
    """Segmentations + reference BED + bw_emissions + binary inputs + optional RNA-seq."""
    ds     = ds_of(w.folder)
    folder = w.folder
    cfg    = DATASETS[ds]
    cell   = cfg["cell"]
    ref    = _ref_bed(ds)

    # Reference files are ancient
    files = [ancient(ref), ancient(ref.replace(".bed", ".bw_emissions.npz"))]
    # Segmentations and emissions are NOT ancient (trigger re-run if they change)
    files += _folder_seg_files(w)

    # Explicit binary-input files so Snakemake guarantees they exist before
    # analyze.py runs its binarized-track emission computation.
    files += [f"{folder}/chromhmm_default/{cell}_{c}_binary.txt" for c in CHROMS]
    for caller in CALLERS:
        files += [f"{folder}/{caller}/chromhmm_peaks/{cell}_{c}_binary.txt.gz"
                  for c in CHROMS]

    if cfg.get("rnaseq"):
        files.append(ancient(f"{ds}/rnaseq_{cfg['rnaseq']}.tsv"))
        files.append(ancient(TOOLS["gencode_gtf"]))
    if cfg.get("atac"):
        files.append(ancient(f"{ds}/atac_{cfg['atac']}.bed.gz"))
    return files


# _analysis_inputs is kept as a dataset-level aggregator (used by markups.smk).
def _analysis_inputs(w):
    """All matched segmentations across every folder for a dataset."""
    ds  = w.ds
    cfg = DATASETS[ds]
    cell = cfg["cell"]
    files = [_ref_bed(ds)]
    for folder in _folders(ds):
        files.append(f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense_comb_matched.bed")
        files.append(f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense_bwem_matched.bed")
        files.append(f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense_ovlp_matched.bed")
        for caller in CALLERS:
            if DO_CHROMHMM_PEAKS:
                files.append(f"{folder}/{caller}/chromhmm_result/{caller}_{cell}_{NSTATES}_dense_comb_matched.bed")
                files.append(f"{folder}/{caller}/chromhmm_result/{caller}_{cell}_{NSTATES}_dense_bwem_matched.bed")
                files.append(f"{folder}/{caller}/chromhmm_result/{caller}_{cell}_{NSTATES}_dense_ovlp_matched.bed")
            files.append(f"{folder}/{caller}/{caller}_kmeans_states_comb_matched.bed")
            files.append(f"{folder}/{caller}/{caller}_kmeans_states_bwem_matched.bed")
            files.append(f"{folder}/{caller}/{caller}_kmeans_states_ovlp_matched.bed")
    if cfg.get("rnaseq"):
        files.append(f"{ds}/rnaseq_{cfg['rnaseq']}.tsv")
        files.append(TOOLS["gencode_gtf"])
    if cfg.get("atac"):
        files.append(f"{ds}/atac_{cfg['atac']}.bed.gz")
    return files


# Explicitly track Functional Enrichment and Emissions plots as outputs
_ANALYZE_OUTPUTS = ["{folder}/analysis/ref/report.tsv",
                    "{folder}/analysis/ref/enrichment/enrichment.png",
                    "{folder}/analysis/ref/bw_emissions/state_emissions.png",
                    "{folder}/analysis/chromhmm_default_dense/bin_emissions/state_emissions.png"]
for _v in ["comb", "bwem", "ovlp"]:
    _ANALYZE_OUTPUTS.append(f"{{folder}}/analysis/{_v}/chromhmm_default/enrichment/enrichment.png")
    _ANALYZE_OUTPUTS.append(f"{{folder}}/analysis/{_v}/chromhmm_default/bw_emissions/state_emissions.png")
    _ANALYZE_OUTPUTS.append(f"{{folder}}/analysis/{_v}/chromhmm_default/bin_emissions/state_emissions.png")
    for _c in CALLERS:
        _ANALYZE_OUTPUTS.append(f"{{folder}}/analysis/{_v}/kmeans_{_c}/enrichment/enrichment.png")
        _ANALYZE_OUTPUTS.append(f"{{folder}}/analysis/{_v}/kmeans_{_c}/bw_emissions/state_emissions.png")
        _ANALYZE_OUTPUTS.append(f"{{folder}}/analysis/{_v}/kmeans_{_c}/bin_emissions/state_emissions.png")
        if DO_CHROMHMM_PEAKS:
            _ANALYZE_OUTPUTS.append(f"{{folder}}/analysis/{_v}/chromhmm_{_c}/enrichment/enrichment.png")
            _ANALYZE_OUTPUTS.append(f"{{folder}}/analysis/{_v}/chromhmm_{_c}/bw_emissions/state_emissions.png")
            _ANALYZE_OUTPUTS.append(f"{{folder}}/analysis/{_v}/chromhmm_{_c}/bin_emissions/state_emissions.png")


rule analyze_segmentations:
    """Run analyze.py on every matched segmentation within {folder}."""
    input: lambda w: [ancient(f) for f in _folder_analysis_inputs(w)]
    output: _ANALYZE_OUTPUTS
    conda: "../envs/python.yaml"
    params:
        coords      = TOOLS["coords_dir"],
        cell        = lambda w: DATASETS[ds_of(w.folder)]["cell"],
        ref         = lambda w: _ref_bed(ds_of(w.folder)),
        bin         = CHROMHMM_BIN,
        caller_bins = " ".join(f"{c}:{CALLER_BIN[c]}" for c in CALLERS),
        n           = NSTATES,
        chromhmm_peaks = DO_CHROMHMM_PEAKS,
        scripts_dir = SCRIPTS_DIR,
        rnaseq      = lambda w: (
                          f"{ds_of(w.folder)}/rnaseq_{DATASETS[ds_of(w.folder)]['rnaseq']}.tsv"
                          if DATASETS[ds_of(w.folder)].get("rnaseq") else ""),
        gtf         = lambda w: TOOLS["gencode_gtf"] if DATASETS[ds_of(w.folder)].get("rnaseq") else "",
        atac        = lambda w: (
                          f"{ds_of(w.folder)}/atac_{DATASETS[ds_of(w.folder)]['atac']}.bed.gz"
                          if DATASETS[ds_of(w.folder)].get("atac") else ""),
    shell:
        r"""
        mkdir -p {wildcards.folder}/analysis

        RNA_ARGS=""
        if [ -n "{params.rnaseq}" ]; then
            RNA_ARGS="--rnaseq {params.rnaseq} --gtf {params.gtf}"
        fi

        ATAC_EXTRA=""
        if [ -n "{params.atac}" ]; then
            ATAC_EXTRA="{params.atac}"
        fi

        run_analyze() {{
            local bin="$1"; local seg="$2"; local outdir="$3"; shift 3
            local npz="${{seg%.bed}}.bw_emissions.npz"
            local bw_arg=""
            [ -f "$npz" ] && bw_arg="--bw-emissions $npz"
            python {params.scripts_dir}/analyze.py --seg "$seg" --bin "$bin" \
                --outdir "$outdir" \
                --annotations {params.coords}/*.bed.gz $ATAC_EXTRA \
                $bw_arg $RNA_ARGS "$@"
        }}

        # Reference (no binary inputs; bin only affects transition entropy)
        run_analyze {params.bin} {params.ref} {wildcards.folder}/analysis/ref

        # Default ChromHMM
        run_analyze {params.bin} \
            {wildcards.folder}/chromhmm_default_result/{params.cell}_{params.n}_dense.bed \
            {wildcards.folder}/analysis/chromhmm_default_dense \
            --inputs {wildcards.folder}/chromhmm_default/*.txt --emissions-only

        for variant in comb bwem ovlp; do
            run_analyze {params.bin} \
                {wildcards.folder}/chromhmm_default_result/{params.cell}_{params.n}_dense_${{variant}}_matched.bed \
                {wildcards.folder}/analysis/${{variant}}/chromhmm_default \
                --inputs {wildcards.folder}/chromhmm_default/*.txt
        done

        # Peak-caller methods — loop over enabled callers only
        for caller_bin_pair in {params.caller_bins}; do
            caller="${{caller_bin_pair%%:*}}"
            cbin="${{caller_bin_pair##*:}}"
            peaks_dir="{wildcards.folder}/${{caller}}/chromhmm_peaks"
            for variant in comb bwem ovlp; do
                run_analyze "$cbin" \
                    {wildcards.folder}/${{caller}}/${{caller}}_kmeans_states_${{variant}}_matched.bed \
                    {wildcards.folder}/analysis/${{variant}}/kmeans_${{caller}} \
                    --inputs "$peaks_dir"/*.txt.gz
                if [ "{params.chromhmm_peaks}" = "True" ] && [ -f "{wildcards.folder}/${{caller}}/chromhmm_result/${{caller}}_{params.cell}_{params.n}_dense_${{variant}}_matched.bed" ]; then
                    run_analyze "$cbin" \
                        {wildcards.folder}/${{caller}}/chromhmm_result/${{caller}}_{params.cell}_{params.n}_dense_${{variant}}_matched.bed \
                        {wildcards.folder}/analysis/${{variant}}/chromhmm_${{caller}} \
                        --inputs "$peaks_dir"/*.txt.gz
                fi
            done
        done
        """


# Peak-file analysis: number of peaks, mean/median length, replicate Jaccard.
# Output: {ds}/peaks/


def _peak_analysis_inputs(w):
    """All peak files needed for the analysis of dataset {ds}."""
    ds = w.ds
    cfg = DATASETS[ds]
    cell = cfg["cell"]
    files = []
    for folder in _folders(ds):
        for mark in MARKS:
            for caller in CALLERS:
                files.append(peak_file(folder,caller,mark))
        # ChromHMM binary files (one representative chrom is enough as input
        # sentinel; analyze_peaks.py globs all of them at runtime)
        files += [f"{folder}/chromhmm_default/{cell}_{c}_binary.txt" for c in CHROMS]
    return files


rule analyze_peaks:
    """Compute per-mark peak stats and replicate Jaccard for all callers."""
    input: lambda w: [ancient(f) for f in _peak_analysis_inputs(w)]
    output:
        stats="{ds}/peaks/peak_stats.tsv",
        gaps="{ds}/peaks/gap_lengths.tsv.gz",
        n_peaks="{ds}/peaks/n_peaks.png",
        mean="{ds}/peaks/mean_length.png",
        median="{ds}/peaks/median_length.png",
    conda: "../envs/python.yaml"
    params:
        cell=lambda w: DATASETS[w.ds]["cell"],
        omni_bin=OMNI_BIN,
        chromhmm_bin=CHROMHMM_BIN,
        marks=" ".join(MARKS),
        scripts_dir=SCRIPTS_DIR,
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
