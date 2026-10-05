# State matching: relabel segmentations to the ENCODE reference.
#
# Strategy: Hungarian assignment over Jaccard plus emission-profile
# similarity, taken to a concave power so the largest states do not decide
# the whole mapping; driven by match.py. The result is a bijection; pairs
# that score poorly are named anyway and flagged in the .mapping.tsv.


def _binarized_files(bedpath):
    """All per-chromosome binarized data paths for a segmentation BED."""
    folder = _emissions_folder(bedpath)
    ds = ds_of(folder)
    cell = DATASETS[ds]["cell"]
    if "chromhmm_default_result" in bedpath:
        # Default ChromHMM (BinarizeBam output)
        return [f"{folder}/chromhmm_default/{cell}_{c}_binary.txt" for c in CHROMS]
    
    parts = bedpath.split("/")
    # KMeans paths look like {ds}/{caller}/{caller}_kmeans_states...
    # or {ds}/repN/{caller}/{caller}_kmeans_states...
    # In both cases, the caller is the directory containing the bed file.
    if len(parts) >= 3 and parts[-2] in CALLER_BIN:
        caller = parts[-2]
        return [f"{folder}/{caller}/chromhmm_peaks/{c}_binary.txt.gz"
                for c in CHROMS]
    
    # Reference BEDs or other beds in the root folder don't have binarized data
    # that compute_bin_emissions can automatically find.
    return []


def _folder_bigwigs(folder):
    """All per-mark bigwig paths for a folder."""
    return [f"{folder}/bams/{mark}.bw" for mark in get_marks_for_folder(folder)]


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


# --- Matching ---------------------------------------------------------

rule match_segmentation:
    """State matching: {name}.bed → {name}_matched.bed + remapped emissions."""
    input:
        ref=lambda w: ancient(_ref_bed(ds_of(_emissions_folder(w.bedpath)))),
        work="{bedpath}.bed",
        work_bw_em=lambda w: ["{bedpath}.bw_emissions.npz".format(**w)] if DO_BW_EMISSIONS else [],
        work_bin_em=lambda w: ["{bedpath}.bin_emissions.npz".format(**w)] if DO_BIN_EMISSIONS else [],
        ref_bw_em=lambda w: ([ancient(_ref_bed(ds_of(_emissions_folder(w.bedpath))).replace(".bed", ".bw_emissions.npz"))]
                             if DO_BW_EMISSIONS and P.get("ref_em_type", "") in ("", "bw") else []),
        ref_bin_em=lambda w: ([ancient(_ref_bed(ds_of(_emissions_folder(w.bedpath))).replace(".bed", ".bin_emissions.npz"))]
                              if DO_BIN_EMISSIONS and P.get("ref_em_type", "") in ("", "bin") and _binarized_files(_ref_bed(ds_of(_emissions_folder(w.bedpath))).replace(".bed", "")) else []),
    output:
        bed="{bedpath}_matched.bed",
        bw_em="{bedpath}_matched.bw_emissions.npz" if DO_BW_EMISSIONS else [],
        bin_em="{bedpath}_matched.bin_emissions.npz" if DO_BIN_EMISSIONS else [],
        matrix_png="{bedpath}_matched.match.png",
        matrix_map="{bedpath}_matched.match.mapping.tsv",
    params:
        mprefix="{bedpath}_matched.match",
        work_em_type=P.get("work_em_type", ""),
        ref_em_type=P.get("ref_em_type", ""),
        bw_flags=lambda w, input, output: f"--work-bw-emissions {input.work_bw_em} --remap-bw-emissions {output.bw_em}" if DO_BW_EMISSIONS else "",
        bin_flags=lambda w, input, output: f"--work-bin-emissions {input.work_bin_em} --remap-bin-emissions {output.bin_em}" if DO_BIN_EMISSIONS else "",
        ref_bw_flags=lambda w, input: f"--ref-bw-emissions {input.ref_bw_em}" if "ref_bw_em" in input.keys() and input.ref_bw_em else "",
        ref_bin_flags=lambda w, input: f"--ref-bin-emissions {input.ref_bin_em}" if "ref_bin_em" in input.keys() and input.ref_bin_em else "",
        type_flags=(f"--work-em-type {P.get('work_em_type', '')} " if P.get('work_em_type', '') else "") + (f"--ref-em-type {P.get('ref_em_type', '')}" if P.get('ref_em_type', '') else "")
    wildcard_constraints:
        bedpath=r"[A-Za-z0-9_./-]+",
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/match.py "
        "--ref {input.ref} --work {input.work} "
        "{params.bw_flags} {params.bin_flags} "
        "{params.ref_bw_flags} {params.ref_bin_flags} "
        "{params.type_flags} "
        "--matrix-out {params.mprefix} > {output.bed}"


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
        marks=lambda w: " ".join(get_marks_for_folder(_emissions_folder(w.bedpath))),
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
