# Omni ChromHMM Snakemake pipeline

import os

configfile: "config.yaml"

workdir: os.path.expanduser(config.get("workdir","."))

TOOLS = config["tools"]
P = config["params"]
MARKS = P["marks"]
CHROMHMM_BIN = P["chromhmm_bin"]
OMNI_BIN = P["omni_bin"]
HOMER_BIN = P["homer_bin"]
MACS2_BIN = P["macs2_bin"]
NSTATES = P["n_states"]
GENOME = P["genome"]
MATCH_METHOD = P["match_method"]

# Per-caller binarization bin sizes.
CALLER_BIN = {"omni": OMNI_BIN, "homer": HOMER_BIN, "macs2": MACS2_BIN}


def _seg_bin(path):
    """Native bin size for a segmentation BED path."""
    for caller, size in CALLER_BIN.items():
        if f"/{caller}/" in path:
            return size
    return CHROMHMM_BIN


def _flag(key, default=True):
    """Read a boolean flag from --config key=... (top-level) or params.key (yaml).

    CLI values arrive as strings ('True'/'False'), so handle both.
    """
    val = config.get(key,P.get(key,default))
    if isinstance(val,str):
        return val.lower() not in ("false", "0", "no", "")
    return bool(val)


DO_CHROMHMM_PEAKS = _flag("chromhmm_peaks")
DO_REPLICATES = _flag("replicates")
DO_OMNIPEAK = _flag("omnipeak")
DO_HOMER = _flag("homer")
DO_MACS2 = _flag("macs2")

# Callers enabled by the config flags above.
CALLERS = (
        (["omni"] if DO_OMNIPEAK else [])
        + (["homer"] if DO_HOMER else [])
        + (["macs2"] if DO_MACS2 else [])
)

CHROMHMM = f"java {TOOLS['java_opts']} -jar {TOOLS['chromhmm_jar']}"
OMNIPEAK = f"java -Xmx8G --add-modules=jdk.incubator.vector -jar {TOOLS['omnipeak_jar']}"

DATASETS = config["datasets"]

SCRIPTS_DIR = os.path.join(workflow.basedir,"scripts", "rules")

# Global wildcard constraints.
# {folder} covers both the pooled root ({ds}) and per-replicate subdirs ({ds}/rep1,
# {ds}/rep2), so it may contain a single path separator.
# {caller} is the peak-calling algorithm: omni or homer.
wildcard_constraints:
    ds=r"[A-Za-z0-9_]+",
    acc=r"ENCFF[A-Z0-9]+",
    mark=r"H3K[0-9]+(me[0-9]|ac)",
    caller=r"omni|homer|macs2",
    rep=r"rep[12]",
    folder=r"[A-Za-z0-9_]+(/rep[12])?",
    chr=r"chr[0-9XYM]+",
    cell=r"[A-Za-z0-9]+",


# --- helpers shared by every included rules/*.smk ------------------------

def ds_of(folder):
    """Extract dataset name from a folder path ('imr90' or 'imr90/rep1')."""
    return folder.split("/")[0]


def accs_of(ds, mark=None, rep=None):
    """Return accessions for a dataset filtered by mark/rep."""
    out = []
    for acc, meta in DATASETS[ds]["bams"].items():
        if mark is not None and meta["mark"] != mark:
            continue
        if rep is not None and meta.get("rep") != rep:
            continue
        out.append(acc)
    return out


def bam_path(ds, acc):
    mark = DATASETS[ds]["bams"][acc]["mark"]
    return f"{ds}/downloaded/{acc}_{mark}.bam"


def bams_for_mark(ds, mark, rep=None):
    accs = accs_of(ds,mark=mark,rep=rep)
    if not accs and rep is not None:
        # Fall back to untagged BAMs for marks that have no replicate-specific entry
        accs = [a for a, meta in DATASETS[ds]["bams"].items()
                if meta["mark"] == mark and "rep" not in meta]
    return [bam_path(ds,a) for a in accs]


# --- Control BAM helpers ---------------------------------------------------

def has_controls(ds):
    """True if any BAM in the dataset has a non-empty control accession."""
    return any(meta.get("control") for meta in DATASETS[ds]["bams"].values())


def control_accs_for_mark(ds, mark, rep=None):
    """Return deduplicated control accessions for a dataset's mark (optionally per rep).

    When rep is given but no rep-specific entry exists for the mark, falls back to
    untagged BAMs (marks with no 'rep' field) — mirroring bams_for_mark behaviour.
    """
    out = set()
    for acc, meta in DATASETS[ds]["bams"].items():
        if meta["mark"] != mark:
            continue
        if rep is not None and meta.get("rep") != rep:
            continue
        ctrl = meta.get("control")
        if ctrl:
            out.add(ctrl)
    if not out and rep is not None:
        # Fall back to untagged BAMs for marks that have no replicate-specific entry
        for acc, meta in DATASETS[ds]["bams"].items():
            if meta["mark"] == mark and "rep" not in meta:
                ctrl = meta.get("control")
                if ctrl:
                    out.add(ctrl)
    return sorted(out)


def controls_for_mark(ds, mark, rep=None):
    """Return downloaded control BAM paths for a mark."""
    return [f"{ds}/downloaded/{a}_control.bam" for a in control_accs_for_mark(ds,mark,rep)]


def folder_has_controls(folder):
    """Check whether the dataset behind a folder path has controls."""
    return has_controls(ds_of(folder))


def read_chromsizes(path):
    """Chromosomes with '_' filtered out (matches the `grep -v _` in the PDF)."""
    chroms = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                name = line.split()[0]
                if "_" not in name:
                    chroms.append(name)
    return chroms


# Prefer the live chromsizes file so we stay in sync with the ChromHMM install;
# fall back to the canonical hg38 primary assembly list so the DAG is complete
# before the file exists on disk.
CHROMS = read_chromsizes(TOOLS["chromsizes"]) or (
        [f"chr{i}" for i in range(1,23)] + ["chrX", "chrY"]
)


def _ref_bed(ds):
    """Path of the ENCODE ChromHMM reference bed for a dataset."""
    return f"{ds}/{DATASETS[ds]['ref_chromhmm']}_chromhmm.bed"


def _folders(ds):
    """All processing folders for a dataset: pooled root + per-replicate subdirs."""
    folders = [ds]
    if DO_REPLICATES and DATASETS[ds].get("replicates"):
        folders += [f"{ds}/rep1", f"{ds}/rep2"]
    return folders


def peak_file(folder, caller, mark):
    """Peak file path produced by a given caller for a mark inside folder."""
    if caller == "omni":
        return f"{folder}/omni/{mark}_{OMNI_BIN}.peak"
    elif caller == "homer":
        return f"{folder}/homer/{mark}.bed"
    else:  # macs2
        return f"{folder}/macs2/{mark}.bed"


def _peaks_binary_files(w):
    """Per-chromosome gz binary inputs for ChromHMM/KMeans over peaks."""
    cell = DATASETS[ds_of(w.folder)]["cell"]
    return [f"{w.folder}/{w.caller}/chromhmm_peaks/{cell}_{c}_binary.txt.gz"
            for c in CHROMS]


def _default_binary_files(w):
    """Per-chromosome binary inputs for default ChromHMM binarization."""
    cell = DATASETS[ds_of(w.folder)]["cell"]
    return [f"{w.folder}/chromhmm_default/{cell}_{c}_binary.txt" for c in CHROMS]


def all_results(ds):
    cfg = DATASETS[ds]
    cell = cfg["cell"]
    t = [f"{ds}/{cfg['ref_chromhmm']}_chromhmm.bed"]

    for folder in _folders(ds):
        t.append(f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense_ovlp_matched.bed")
        t.append(f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense_bwem_matched.bed")
        t.append(f"{folder}/chromhmm_default_result/{cell}_{NSTATES}_dense_comb_matched.bed")
        for mark in MARKS:
            t.append(f"{folder}/chromhmm_default_result/{mark}.bed")
        for caller in CALLERS:
            if DO_CHROMHMM_PEAKS:
                t.append(f"{folder}/{caller}/chromhmm_result/{caller}_{cell}_{NSTATES}_dense_ovlp_matched.bed")
                t.append(f"{folder}/{caller}/chromhmm_result/{caller}_{cell}_{NSTATES}_dense_bwem_matched.bed")
                t.append(f"{folder}/{caller}/chromhmm_result/{caller}_{cell}_{NSTATES}_dense_comb_matched.bed")
            t.append(f"{folder}/{caller}/{caller}_kmeans_states_ovlp_matched.bed")
            t.append(f"{folder}/{caller}/{caller}_kmeans_states_bwem_matched.bed")
            t.append(f"{folder}/{caller}/{caller}_kmeans_states_comb_matched.bed")

    # Per-state matching matrix (heatmap) produced alongside every matched BED.
    t += [f.replace("_matched.bed", "_matched.match.png")
          for f in list(t) if f.endswith("_matched.bed")]

    return t


rule all:
    input:
        lambda wildcards: [f"{ds}/.done" for ds in DATASETS],


rule dataset_done:
    """Per-dataset sentinel: all segmentations matched against the reference.

    Analysis, comparison and inter-dataset plots are produced separately in
    analysis.ipynb, not by this pipeline.
    """
    input:
        lambda w: [ancient(f) for f in all_results(w.ds)],
    output: touch("{ds}/.done")

# Download and prepare
include: "rules/data.smk"
# ChromHMM
include: "rules/chromhmm.smk"
# Peak callers
include: "rules/omni.smk"
include: "rules/homer.smk"
include: "rules/macs2.smk"
# Matching vs reference
include: "rules/match.smk"
# Analysis, comparison and inter-dataset summary plots live in analysis.ipynb.
