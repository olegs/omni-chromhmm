# State matching: relabel segmentations to the ENCODE reference.
#
# Strategy: overlap-only (Jaccard + Hungarian), driven by match.py.


def _binarized_files(bedpath):
    """All per-chromosome binarized data paths for a segmentation BED."""
    folder = _emissions_folder(bedpath)
    ds = ds_of(folder)
    cell = DATASETS[ds]["cell"]
    if "chromhmm_default_result" in bedpath:
        # Default ChromHMM (BinarizeBam output)
        return [f"{folder}/chromhmm_default/{cell}_{c}_binary.txt" for c in CHROMS]
    else:
        # KMeans (peaks_segmentation.py output)
        parts = bedpath.split("/")
        # Path: {folder}/{caller}/{caller}_kmeans_states...
        caller = parts[-2]
        return [f"{folder}/{caller}/chromhmm_peaks/{c}_binary.txt.gz"
                for c in CHROMS]


def _folder_bigwigs(folder):
    """All per-mark bigwig paths for a folder."""
    return [f"{folder}/bams/{mark}.bw" for mark in MARKS]


def _emissions_folder(bedpath):
    """Top-level folder (for bigwig lookup) from a bed file path stem.

    Handles both pooled ('ds/...') and per-replicate ('ds/repN/...') paths.
    """
    parts = bedpath.split("/")
    if len(parts) >= 2 and parts[1] in ("rep1", "rep2"):
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


# Rule priority: match rule takes priority over the generic compute_emissions rule
# so that _matched.bw_emissions.npz and _matched.bin_emissions.npz files are
# produced by state-name remapping rather than recomputing from scratch.
ruleorder: match_segmentation > compute_bw_emissions
ruleorder: match_segmentation > compute_bin_emissions


# --- Overlap matching -----------------------------------------------------

rule match_segmentation:
    """Overlap matching: {name}.bed → {name}_matched.bed + remapped emissions."""
    input:
        ref=lambda w: ancient(_ref_bed(ds_of(_emissions_folder(w.bedpath)))),
        work="{bedpath}.bed",
        work_bw_em="{bedpath}.bw_emissions.npz",
        work_bin_em="{bedpath}.bin_emissions.npz",
    output:
        bed="{bedpath}_matched.bed",
        bw_em="{bedpath}_matched.bw_emissions.npz",
        bin_em="{bedpath}_matched.bin_emissions.npz",
        matrix_png="{bedpath}_matched.match.png",
        matrix_map="{bedpath}_matched.match.mapping.tsv",
    params:
        mprefix="{bedpath}_matched.match",
        method=MATCH_METHOD,
    wildcard_constraints:
        bedpath=r"[A-Za-z0-9_./-]+",
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/match.py "
        "--ref {input.ref} --work {input.work} "
        "--work-bw-emissions {input.work_bw_em} --remap-bw-emissions {output.bw_em} "
        "--work-bin-emissions {input.work_bin_em} --remap-bin-emissions {output.bin_em} "
        "--matrix-out {params.mprefix} --method {params.method} > {output.bed}"


# --- Emissions pre-computation --------------------------------------------

rule compute_bw_emissions:
    """Compute per-state bigwig emissions for any segmentation BED.

    Generic rule: {bedpath}.bed → {bedpath}.bw_emissions.npz.
    The bams/ folder for bigwig lookup is derived from the leading path components.
    """
    input:
        bed="{bedpath}.bed",
        bigwigs=lambda w: [ancient(f) for f in _folder_bigwigs(_emissions_folder(w.bedpath))],
    output: "{bedpath}.bw_emissions.npz"
    wildcard_constraints:
        bedpath=r"[A-Za-z0-9_./-]+",
    conda: "../envs/python.yaml"
    params:
        marks=" ".join(MARKS),
        bigwigs=lambda w: " ".join(_folder_bigwigs(_emissions_folder(w.bedpath))),
        bin=CHROMHMM_BIN,
    shell:
        "python {SCRIPTS_DIR}/emissions.py "
        "--bed {input.bed} --bigwigs {params.bigwigs} --marks {params.marks} "
        "--bin {params.bin} --output {output}"


rule compute_bin_emissions:
    """Compute per-state binarized emissions for any segmentation BED.

    Generic rule: {bedpath}.bed → {bedpath}.bin_emissions.npz.
    """
    input:
        bed="{bedpath}.bed",
        binaries=lambda w: [ancient(f) for f in _binarized_files(w.bedpath)],
    output: "{bedpath}.bin_emissions.npz"
    wildcard_constraints:
        bedpath=r"[A-Za-z0-9_./-]+",
    conda: "../envs/python.yaml"
    params:
        bin=_seg_bin,
    shell:
        "python {SCRIPTS_DIR}/emissions.py "
        "--bed {input.bed} --binaries {input.binaries} "
        "--bin {params.bin} --output {output}"
