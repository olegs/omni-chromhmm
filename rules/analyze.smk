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
    for caller in ["omni", "homer"]:
        beds.append(f"{folder}/{caller}/chromhmm_result/{cell}_{NSTATES}_dense_comb_matched.bed")
        beds.append(f"{folder}/{caller}/chromhmm_result/{cell}_{NSTATES}_dense_bwem_matched.bed")
        beds.append(f"{folder}/{caller}/chromhmm_result/{cell}_{NSTATES}_dense_ovlp_matched.bed")
        beds.append(f"{folder}/{caller}/kmeans_states_comb_matched.bed")
        beds.append(f"{folder}/{caller}/kmeans_states_bwem_matched.bed")
        beds.append(f"{folder}/{caller}/kmeans_states_ovlp_matched.bed")
    # Require the bw_emissions.npz for every BED so Snakemake builds them first.
    npzs = [b.replace(".bed", ".bw_emissions.npz") for b in beds]
    return beds + npzs


def _folder_analysis_inputs(w):
    """Segmentations + reference BED + bw_emissions + binary inputs + optional RNA-seq."""
    ds     = ds_of(w.folder)
    folder = w.folder
    cfg    = DATASETS[ds]
    cell   = cfg["cell"]
    ref    = _ref_bed(ds)

    files = [ref, ref.replace(".bed", ".bw_emissions.npz")] + _folder_seg_files(w)

    # Explicit binary-input files so Snakemake guarantees they exist before
    # analyze.py runs its binarized-track emission computation.
    files += [f"{folder}/chromhmm_default/{cell}_{c}_binary.txt" for c in CHROMS]
    for caller in ["omni", "homer"]:
        files += [f"{folder}/{caller}/chromhmm_peaks/{cell}_{c}_binary.txt.gz"
                  for c in CHROMS]

    if cfg.get("rnaseq"):
        files.append(f"{ds}/rnaseq_{cfg['rnaseq']}.tsv")
        files.append(TOOLS["gencode_gtf"])
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
        for caller in ["omni", "homer"]:
            files.append(f"{folder}/{caller}/chromhmm_result/{cell}_{NSTATES}_dense_comb_matched.bed")
            files.append(f"{folder}/{caller}/chromhmm_result/{cell}_{NSTATES}_dense_bwem_matched.bed")
            files.append(f"{folder}/{caller}/chromhmm_result/{cell}_{NSTATES}_dense_ovlp_matched.bed")
            files.append(f"{folder}/{caller}/kmeans_states_comb_matched.bed")
            files.append(f"{folder}/{caller}/kmeans_states_bwem_matched.bed")
            files.append(f"{folder}/{caller}/kmeans_states_ovlp_matched.bed")
    if cfg.get("rnaseq"):
        files.append(f"{ds}/rnaseq_{cfg['rnaseq']}.tsv")
        files.append(TOOLS["gencode_gtf"])
    return files


rule analyze_segmentations:
    """Run analyze.py on every matched segmentation within {folder}."""
    input: _folder_analysis_inputs
    output: "{folder}/analysis/ref/report.tsv"
    conda: "../envs/python.yaml"
    params:
        coords      = TOOLS["coords_dir"],
        cell        = lambda w: DATASETS[ds_of(w.folder)]["cell"],
        ref         = lambda w: _ref_bed(ds_of(w.folder)),
        bin         = CHROMHMM_BIN,
        omni_bin    = OMNI_BIN,
        homer_bin   = HOMER_BIN,
        n           = NSTATES,
        scripts_dir = SCRIPTS_DIR,
        rnaseq      = lambda w: (
                          f"{ds_of(w.folder)}/rnaseq_{DATASETS[ds_of(w.folder)]['rnaseq']}.tsv"
                          if DATASETS[ds_of(w.folder)].get("rnaseq") else ""),
        gtf         = lambda w: TOOLS["gencode_gtf"] if DATASETS[ds_of(w.folder)].get("rnaseq") else "",
    shell:
        r"""
        mkdir -p {wildcards.folder}/analysis

        RNA_ARGS=""
        if [ -n "{params.rnaseq}" ]; then
            RNA_ARGS="--rnaseq {params.rnaseq} --gtf {params.gtf}"
        fi

        run_analyze() {{
            local bin="$1"; local seg="$2"; local outdir="$3"; shift 3
            local npz="${{seg%.bed}}.bw_emissions.npz"
            local bw_arg=""
            [ -f "$npz" ] && bw_arg="--bw-emissions $npz"
            python {params.scripts_dir}/analyze.py --seg "$seg" --bin "$bin" \
                --outdir "$outdir" \
                --annotations {params.coords}/*.bed.gz \
                $bw_arg $RNA_ARGS "$@"
        }}

        # Bin-size policy:
        #   ChromHMM default binarization uses {params.bin}-bp bins (binary files are at
        #   that resolution), so all default-ChromHMM analyses run with {params.bin}.
        #   OmniPeak binarization uses {params.omni_bin}-bp bins; Homer uses
        #   {params.homer_bin}-bp bins.

        # Reference (no binary inputs; bin only affects transition entropy)
        run_analyze {params.bin} {params.ref} {wildcards.folder}/analysis/ref

        # Default ChromHMM — bin={params.bin} matches chromhmm_default binary resolution
        run_analyze {params.bin} \
            {wildcards.folder}/chromhmm_default_result/{params.cell}_{params.n}_dense.bed \
            {wildcards.folder}/analysis/chromhmm_default_dense \
            --inputs {wildcards.folder}/chromhmm_default/*.txt --emissions-only

        # comb/ — combined (overlap + bw-emission) matched segmentations
        run_analyze {params.bin} \
            {wildcards.folder}/chromhmm_default_result/{params.cell}_{params.n}_dense_comb_matched.bed \
            {wildcards.folder}/analysis/comb/chromhmm_default \
            --inputs {wildcards.folder}/chromhmm_default/*.txt

        run_analyze {params.omni_bin} \
            {wildcards.folder}/omni/chromhmm_result/{params.cell}_{params.n}_dense_comb_matched.bed \
            {wildcards.folder}/analysis/comb/chromhmm_omni \
            --inputs {wildcards.folder}/omni/chromhmm_peaks/*.txt.gz

        run_analyze {params.omni_bin} \
            {wildcards.folder}/omni/kmeans_states_comb_matched.bed \
            {wildcards.folder}/analysis/comb/kmeans_omni \
            --inputs {wildcards.folder}/omni/chromhmm_peaks/*.txt.gz

        run_analyze {params.homer_bin} \
            {wildcards.folder}/homer/chromhmm_result/{params.cell}_{params.n}_dense_comb_matched.bed \
            {wildcards.folder}/analysis/comb/chromhmm_homer \
            --inputs {wildcards.folder}/homer/chromhmm_peaks/*.txt.gz

        run_analyze {params.homer_bin} \
            {wildcards.folder}/homer/kmeans_states_comb_matched.bed \
            {wildcards.folder}/analysis/comb/kmeans_homer \
            --inputs {wildcards.folder}/homer/chromhmm_peaks/*.txt.gz

        # bwem/ — bigwig-emission-only matched segmentations
        run_analyze {params.bin} \
            {wildcards.folder}/chromhmm_default_result/{params.cell}_{params.n}_dense_bwem_matched.bed \
            {wildcards.folder}/analysis/bwem/chromhmm_default \
            --inputs {wildcards.folder}/chromhmm_default/*.txt

        run_analyze {params.omni_bin} \
            {wildcards.folder}/omni/chromhmm_result/{params.cell}_{params.n}_dense_bwem_matched.bed \
            {wildcards.folder}/analysis/bwem/chromhmm_omni \
            --inputs {wildcards.folder}/omni/chromhmm_peaks/*.txt.gz

        run_analyze {params.omni_bin} \
            {wildcards.folder}/omni/kmeans_states_bwem_matched.bed \
            {wildcards.folder}/analysis/bwem/kmeans_omni \
            --inputs {wildcards.folder}/omni/chromhmm_peaks/*.txt.gz

        run_analyze {params.homer_bin} \
            {wildcards.folder}/homer/chromhmm_result/{params.cell}_{params.n}_dense_bwem_matched.bed \
            {wildcards.folder}/analysis/bwem/chromhmm_homer \
            --inputs {wildcards.folder}/homer/chromhmm_peaks/*.txt.gz

        run_analyze {params.homer_bin} \
            {wildcards.folder}/homer/kmeans_states_bwem_matched.bed \
            {wildcards.folder}/analysis/bwem/kmeans_homer \
            --inputs {wildcards.folder}/homer/chromhmm_peaks/*.txt.gz

        # ovlp/ — overlap-only matched segmentations
        run_analyze {params.bin} \
            {wildcards.folder}/chromhmm_default_result/{params.cell}_{params.n}_dense_ovlp_matched.bed \
            {wildcards.folder}/analysis/ovlp/chromhmm_default \
            --inputs {wildcards.folder}/chromhmm_default/*.txt

        run_analyze {params.omni_bin} \
            {wildcards.folder}/omni/chromhmm_result/{params.cell}_{params.n}_dense_ovlp_matched.bed \
            {wildcards.folder}/analysis/ovlp/chromhmm_omni \
            --inputs {wildcards.folder}/omni/chromhmm_peaks/*.txt.gz

        run_analyze {params.omni_bin} \
            {wildcards.folder}/omni/kmeans_states_ovlp_matched.bed \
            {wildcards.folder}/analysis/ovlp/kmeans_omni \
            --inputs {wildcards.folder}/omni/chromhmm_peaks/*.txt.gz

        run_analyze {params.homer_bin} \
            {wildcards.folder}/homer/chromhmm_result/{params.cell}_{params.n}_dense_ovlp_matched.bed \
            {wildcards.folder}/analysis/ovlp/chromhmm_homer \
            --inputs {wildcards.folder}/homer/chromhmm_peaks/*.txt.gz

        run_analyze {params.homer_bin} \
            {wildcards.folder}/homer/kmeans_states_ovlp_matched.bed \
            {wildcards.folder}/analysis/ovlp/kmeans_homer \
            --inputs {wildcards.folder}/homer/chromhmm_peaks/*.txt.gz
        """
