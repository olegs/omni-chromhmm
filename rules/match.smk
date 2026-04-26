# State matching: relabel segmentations to the ENCODE reference.
#
# Three strategies, all driven by match.py:
#
#   _ovlp_matched.bed        — overlap-only (Jaccard + Hungarian)
#   _comb_matched.bed        — combined overlap+bw-emission (alpha=0.8, default)
#   _bwem_matched.bed        — bigwig-emission-only (cosine + Hungarian)


def _ref_emissions(ds):
    """Path of the pre-computed reference emissions .npz for a dataset."""
    return f"{ds}/{DATASETS[ds]['ref_chromhmm']}_chromhmm.bw_emissions.npz"


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


# Rule priority: more specific *_bwem_matched / *_ovlp_matched patterns take
# precedence over the generic *_matched pattern.
ruleorder: match_segmentation_em > match_segmentation_ovlp > match_segmentation


# --- Overlap-only matching ------------------------------------------------

rule match_segmentation_ovlp:
    """Overlap-only matching: {name}.bed → {name}_ovlp_matched.bed."""
    input:
        ref  = lambda w: _ref_bed(ds_of(_emissions_folder(w.bedpath))),
        work = "{bedpath}.bed",
    output: "{bedpath}_ovlp_matched.bed"
    wildcard_constraints:
        bedpath = r"[A-Za-z0-9_./-]+",
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/match.py match --alpha 1.0 --ref {input.ref} --work {input.work} > {output}"


# --- Emissions pre-computation --------------------------------------------

rule compute_emissions:
    """Compute per-state bigwig emissions for any segmentation BED.

    Generic rule: {bedpath}.bed → {bedpath}.bw_emissions.npz.
    The bams/ folder for bigwig lookup is derived from the leading path components.
    """
    input:
        bed     = "{bedpath}.bed",
        bigwigs = lambda w: _folder_bigwigs(_emissions_folder(w.bedpath)),
    output: "{bedpath}.bw_emissions.npz"
    wildcard_constraints:
        bedpath = r"[A-Za-z0-9_./-]+",
    conda: "../envs/bio.yaml"
    params:
        marks   = " ".join(MARKS),
        bigwigs = lambda w: " ".join(_folder_bigwigs(_emissions_folder(w.bedpath))),
        bin     = CHROMHMM_BIN,
    shell:
        "python {SCRIPTS_DIR}/match.py compute "
        "--bed {input.bed} --bigwigs {params.bigwigs} --marks {params.marks} "
        "--bin {params.bin} --output {output}"


# --- Combined matching (default) ------------------------------------------

rule match_segmentation:
    """Combined matching (overlap + bw-emission, alpha=0.8): {name}.bed → {name}_comb_matched.bed."""
    input:
        ref     = lambda w: _ref_bed(ds_of(_emissions_folder(w.bedpath))),
        ref_em  = lambda w: _ref_emissions(ds_of(_emissions_folder(w.bedpath))),
        work    = "{bedpath}.bed",
        work_em = "{bedpath}.bw_emissions.npz",
    output: "{bedpath}_comb_matched.bed"
    wildcard_constraints:
        bedpath = r"[A-Za-z0-9_./-]+",
    conda: "../envs/bio.yaml"
    shell:
        "python {SCRIPTS_DIR}/match.py match "
        "--ref {input.ref} --ref-emissions {input.ref_em} "
        "--work {input.work} --work-emissions {input.work_em} > {output}"


# --- Emissions-only matching ----------------------------------------------

rule match_segmentation_em:
    """Bigwig-emission-only matching (alpha=0): {name}.bed → {name}_bwem_matched.bed."""
    input:
        ref     = lambda w: _ref_bed(ds_of(_emissions_folder(w.bedpath))),
        ref_em  = lambda w: _ref_emissions(ds_of(_emissions_folder(w.bedpath))),
        work    = "{bedpath}.bed",
        work_em = "{bedpath}.bw_emissions.npz",
    output: "{bedpath}_bwem_matched.bed"
    wildcard_constraints:
        bedpath = r"[A-Za-z0-9_./-]+",
    conda: "../envs/bio.yaml"
    shell:
        "python {SCRIPTS_DIR}/match.py match --alpha 0 "
        "--ref {input.ref} --ref-emissions {input.ref_em} "
        "--work {input.work} --work-emissions {input.work_em} > {output}"
