# Omni ChromHMM Snakemake pipeline

import os

configfile: "config.yaml"

workdir: os.path.expanduser(config.get("workdir", "."))

TOOLS =     config["tools"]
P =         config["params"]
MARKS  =    P["marks"]
BIN =       P["bin"]
NSTATES =   P["n_states"]
GENOME  =   P["genome"]

CHROMHMM = f"java {TOOLS['java_opts']} -jar {TOOLS['chromhmm_jar']}"
OMNIPEAK = f"java {TOOLS['java_opts']} --add-modules=jdk.incubator.vector -jar {TOOLS['omnipeak_jar']}"

DATASETS = config["datasets"]

SCRIPTS_DIR = os.path.join(workflow.basedir, "scripts")

# Global wildcard constraints — keep wildcards from swallowing path separators
# or colliding with similarly named directories (e.g. omni vs pooled_omni).
wildcard_constraints:
    ds    = r"[A-Za-z0-9_]+",
    acc   = r"ENCFF[A-Z0-9]+",
    mark  = r"H3K[0-9]+(me[0-9]|ac)",
    mode  = r"omni|replicated|rep[12]",
    chr   = r"chr[0-9XYM]+",
    cell  = r"[A-Za-z0-9]+",
    name  = r"[A-Za-z0-9_.]+",


# --- helpers shared by every included rules/*.smk ------------------------

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
    return f"{ds}/bams/{acc}_{mark}.bam"


def bams_for_mark(ds, mark, rep=None):
    return [bam_path(ds, a) for a in accs_of(ds, mark=mark, rep=rep)]


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
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
)


def _modes_for(ds):
    """Derive Omnipeak modes from dataset config: always omni, plus replicated if replicates."""
    return ["omni", "replicated"] if DATASETS[ds].get("replicates") else ["omni"]


def _ref_bed(ds):
    """Path of the ENCODE ChromHMM reference bed for a dataset."""
    return f"{ds}/{DATASETS[ds]['ref_chromhmm']}_chromhmm.bed"


def _mode_peak(ds, mode, mark):
    """Path of the Omnipeak .peak file for a given (dataset, mode, mark)."""
    prefix = "pooled" if mode == "omni" else mode
    return f"{ds}/{prefix}_omni/{mark}_{prefix}_{BIN}.peak"


def _mode_binary_files(w):
    """Per-chromosome ChromHMM binary inputs for a dataset+mode."""
    cell = DATASETS[w.ds]["cell"]
    return [f"{w.ds}/{w.mode}/chromhmm_peaks/{cell}_{c}_binary.txt.gz" for c in CHROMS]


def all_results(ds):
    cfg = DATASETS[ds]
    cell = cfg["cell"]
    t = []

    # Reference ChromHMM annotation
    t.append(f"{ds}/{cfg['ref_chromhmm']}_chromhmm.bed")

    # Default ChromHMM pipeline result on pooled BAMs (matched to reference)
    t.append(f"{ds}/chromhmm_default_result/{cell}_{NSTATES}_chromhmm_default_matched.bed")

    # Each Omnipeak mode: ChromHMM-over-Omnipeak + GMM/KMeans states, both matched
    for mode in _modes_for(ds):
        t.append(f"{ds}/{mode}/chromhmm_result/{cell}_{NSTATES}_chromhmm_{mode}_matched.bed")
        t.append(f"{ds}/{mode}/gmm_{mode}_matched.bed")
        t.append(f"{ds}/{mode}/kmeans_{mode}_matched.bed")

    # Per-replicate: repeat steps 1-4 for each replicate under {ds}/{rep}/
    if cfg.get("replicates"):
        for rep in ["rep1", "rep2"]:
            # Default ChromHMM on replicate BAMs
            t.append(f"{ds}/{rep}/chromhmm_default_result/{cell}_{NSTATES}_chromhmm_default_{rep}_matched.bed")
            # ChromHMM over Omnipeak on replicate BAMs
            t.append(f"{ds}/{rep}/chromhmm_result/{cell}_{NSTATES}_chromhmm_{rep}_matched.bed")
            # GMM and KMeans states on replicate Omnipeak binarization
            t.append(f"{ds}/{rep}/gmm_{rep}_matched.bed")
            t.append(f"{ds}/{rep}/kmeans_{rep}_matched.bed")

    return t


rule dataset_done:
    """Per-dataset sentinel: all segmentations + analysis + matched plots."""
    input:
        lambda w: all_results(w.ds)
                  + [f"{w.ds}/analysis/.done", f"{w.ds}/plots_matched/.done"],
    output: touch("{ds}/.done")


rule all:
    input:
        [f"{ds}/.done" for ds in DATASETS],


include: "rules/data.smk"
include: "rules/chromhmm.smk"
include: "rules/omni.smk"
include: "rules/match.smk"
include: "rules/analyze.smk"
include: "rules/markups.smk"
