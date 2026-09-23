#!/usr/bin/env python3
"""Init, updated and refitted models on newly arrived samples.

K-means and the Bernoulli mixture over the previous-current-next bin patterns
(BMM3) on the same Omnipeak pattern tracks, and ChromHMM over the same
binarization, and on HOMER and MACS2 peak calls, on the ENCODE split:
twenty-one arms, scored on the incoming samples only.
"""

import gzip
import itertools
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import patterns as pat                                           # noqa: E402
import bernoulli as bern                                          # noqa: E402
import utils                                                     # noqa: E402

INIT_ARM = "init"
UPDATED = "updated"
REFIT = "refit"
ARMS = (INIT_ARM, UPDATED, REFIT)

# The scenario a non-refit arm is trained under.
SCENARIOS = ("updated", "init")

THREADS = pat.THREADS
JAVA_MEM = "-mx8000M"
# One seed for every method. ChromHMM's `-init information` is deterministic
# and has no seed to vary, so a pattern model allowed several - a best-of pick,
# or a spread to average over - would not be answering the same question.
SEED = 42


# --- ChromHMM-format input from the cached pattern tracks ------------------

def _row_templates(marks):
    """The 64 possible binary rows, pre-rendered, indexed by pattern code."""
    rows = []
    for code in range(pat.N_PATTERNS):
        bits = [(code >> i) & 1 for i in range(len(marks))]
        rows.append("\t".join(str(b) for b in bits))
    return np.array(rows, dtype=object)


def write_binary(cache, mask, cell, sample, outdir, bin_size=None):
    """Write {cell}_{chrom}_binary.txt.gz for one sample, one file per chrom."""
    bin_size = bin_size or mask.bin
    if bin_size % mask.bin != 0:
        raise ValueError(f"bin_size {bin_size} must be a multiple of mask.bin {mask.bin}")
    ratio = bin_size // mask.bin

    os.makedirs(outdir, exist_ok=True)
    codes = cache.codes(sample)
    templates = _row_templates(pat.MARKS)
    header = f"{cell}\t{{chrom}}\n" + "\t".join(pat.MARKS) + "\n"
    written = []
    for chrom, lo, hi in mask.slices():
        path = os.path.join(outdir, f"{cell}_{chrom}_binary.txt.gz")
        c = codes[lo:hi]
        if ratio > 1:
            n = len(c) // ratio
            c = np.bitwise_or.reduce(c[:n * ratio].reshape(n, ratio), axis=1)

        with gzip.open(path, "wt") as out:
            out.write(header.format(chrom=chrom))
            out.write("\n".join(templates[c]))
            out.write("\n")
        written.append(path)
    return written


def link_cells(source_dirs, outdir, chroms=None):
    """One ChromHMM input directory holding several cells' binary files."""
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    for directory in source_dirs:
        for name in sorted(os.listdir(directory)):
            if not name.endswith("_binary.txt.gz"):
                continue
            if chroms is not None and name.rsplit("_", 2)[-2] not in chroms:
                continue
            os.symlink(os.path.abspath(os.path.join(directory, name)),
                       os.path.join(outdir, name))
    return outdir


# --- measured ChromHMM calls ----------------------------------------------

def run_java(jar, args, log):
    """Run one ChromHMM command; return (wall, cpu, peak_mb) of that process."""
    command = ["java", JAVA_MEM, "-jar", jar] + [str(a) for a in args]
    os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
    started = time.perf_counter()
    with open(log, "wb") as handle:
        pid = os.posix_spawn("/usr/bin/env", ["env"] + command, os.environ,
                             file_actions=[(os.POSIX_SPAWN_DUP2, handle.fileno(), 1),
                                           (os.POSIX_SPAWN_DUP2, handle.fileno(), 2)])
        _, status, usage = os.wait4(pid, 0)
    wall = time.perf_counter() - started
    if status != 0:
        raise RuntimeError(f"{' '.join(command)} failed ({status}); see {log}")
    return {"wall": wall, "cpu": usage.ru_utime + usage.ru_stime,
            "peak_mb": usage.ru_maxrss * pat._RSS_UNIT / 1024 ** 2,
            "command": " ".join(command)}


def learn_model(jar, input_dir, out_dir, states, assembly, bin_size, seed,
                threads=THREADS, log=None, model=None):
    """LearnModel over every cell in input_dir, optionally from a given model.

    The fresh fit is `-init information`, which is deterministic; *model* is
    the `-init load -m` warm start, where EM continues from an existing model
    instead, the ChromHMM equivalent of the pattern models' updated arm.
    """
    arguments = ["LearnModel", "-b", bin_size, "-p", threads,
                 "-init", "load" if model is not None else "information",
                 "-s", seed, "-nobrowser", "-noenrich"]
    if model is not None:
        arguments += ["-m", model]
    arguments += [input_dir, out_dir, states, assembly]
    return run_java(jar, arguments,
                    log or os.path.join(out_dir, "learnmodel.log"))


def make_segmentation(jar, model, input_dir, out_dir, bin_size, log=None):
    """MakeSegmentation: apply an existing model, no retraining."""
    return run_java(jar, ["MakeSegmentation", "-b", bin_size, "-gzip",
                          model, input_dir, out_dir],
                    log or os.path.join(out_dir, "makesegmentation.log"))


# --- reading a ChromHMM segmentation back ----------------------------------

def load_labels(path, mask):
    """Per-bin state ids (0-based, -1 where unlabelled) of a dense BED."""
    frame = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                        names=["chrom", "start", "end", "state"],
                        dtype={"chrom": str}, comment="t")
    labels = np.full(mask.total_bins, -1, dtype=np.int8)
    states = frame["state"].str.extract(r"E?(\d+)")[0].astype(int).to_numpy() - 1
    for chrom, group in frame.assign(state_id=states).groupby("chrom", sort=False):
        if chrom not in mask.nbins:
            continue
        offset, n = mask.offset[chrom], mask.nbins[chrom]
        starts = np.clip(group["start"].to_numpy() // mask.bin, 0, n)
        ends = np.clip(group["end"].to_numpy() // mask.bin, 0, n)
        for start, end, state in zip(starts, ends, group["state_id"].to_numpy()):
            labels[offset + start:offset + end] = state
    return labels


def load_emissions(path):
    """ChromHMM emissions_{k}.txt as a states x marks matrix in MARKS order."""
    frame = pd.read_csv(path, sep="\t", index_col=0)
    frame.columns = [c.replace(".sorted", "") for c in frame.columns]
    frame = frame.reindex(columns=list(pat.MARKS))
    return frame.sort_index().to_numpy()


# --- the experiment --------------------------------------------------------

# Samples of the same organ move together, so a unit rather than a sample is
# what a split draws.

UNITS = (("monocytes",), ("spleen",), ("neurosphere",), ("imr90",),
         ("sigmoid_colon",), ("adrenal_gland",), ("tibial_nerve",),
         ("heart_left_ventricle", "heart_right_ventricle"),
         ("substantia_nigra", "temporal_lobe"))
ALL_SAMPLES = tuple(s for unit in UNITS for s in unit)

N_INCOMING = 4          # 4 incoming, the remaining 7 initial
N_SPLITS = 7            # how many splits a run evaluates by default


def split_candidates(datasets, n_incoming=N_INCOMING):
    """Every split of the units into initial and incoming that is measurable."""
    both = {sample for sample in ALL_SAMPLES
            if datasets[sample].get("rnaseq") and datasets[sample].get("atac")}
    eligible = [unit for unit in UNITS if all(s in both for s in unit)]
    out = []
    for size in range(1, len(eligible) + 1):
        for combination in itertools.combinations(eligible, size):
            incoming = tuple(s for unit in combination for s in unit)
            if len(incoming) == n_incoming:
                out.append(tuple(sorted(incoming)))
    return sorted(set(out))


def choose_splits(datasets, n_splits=N_SPLITS, n_incoming=N_INCOMING, seed=42):
    """`n_splits` splits that between them test every eligible sample evenly."""
    candidates = split_candidates(datasets, n_incoming)
    primary = tuple(sorted(INCOMING))
    chosen = [primary] if primary in candidates else []
    usage = {sample: 0 for unit in UNITS for sample in unit}
    for incoming in chosen:
        for sample in incoming:
            usage[sample] += 1
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(candidates)))
    while len(chosen) < min(n_splits, len(candidates)):
        remaining = [c for c in order if candidates[c] not in chosen]
        best = max(remaining, key=lambda c: (
            len({s for s in candidates[c] if usage[s] == 0}),
            -sum(usage[s] for s in candidates[c]), -c))
        chosen.append(candidates[best])
        for sample in candidates[best]:
            usage[sample] += 1
    return [{"name": f"split{i + 1}", "incoming": list(incoming),
             "initial": [s for s in ALL_SAMPLES if s not in set(incoming)]}
            for i, incoming in enumerate(chosen)]


INITIAL = ("monocytes", "spleen", "heart_left_ventricle", "heart_right_ventricle",
           "neurosphere", "substantia_nigra", "temporal_lobe")
INCOMING = ("imr90", "sigmoid_colon", "adrenal_gland", "tibial_nerve")

# How the mixture arms are fitted, and how much endpoint loss still counts as
# the init model annotating the incoming samples well enough.
INIT = "kmeans"
# 200 restarts of the spatial mixture cost ~13 s against ChromHMM's ~85 s,
# and take the seed-to-seed per-state Jaccard of the fit from 0.92 to 0.98.
N_INIT = 200
TOLERANCE = 0.02

DATA_DIR = "~/data/2026_segmentations/encode"
CONFIG = "config_encode.yaml"


def encode_peak_path(omni_bin):
    """How the cache finds a mark's Omnipeak peaks (rules/omni.smk layout)."""
    def peak_path(sample_dir, mark):
        return os.path.join(sample_dir, "omni", f"{mark}_{omni_bin}.peak")
    return peak_path


def load_config():
    """The repository config every arm takes its parameters from."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    with open(os.path.join(repo, CONFIG)) as handle:
        return yaml.safe_load(handle)


def setup(data_dir=DATA_DIR):
    """The mask and the peak cache every arm shares."""
    config = load_config()
    data_dir = os.path.expanduser(data_dir)
    out_root = os.path.join(data_dir, "out", "transfer_cache")
    os.makedirs(out_root, exist_ok=True)
    excluded = set(config.get("outliers") or ()).intersection(INITIAL + INCOMING)
    if excluded:
        raise ValueError(f"split contains datasets the config marks as outliers: "
                         f"{sorted(excluded)}")
    mask = pat.Mask(os.path.join(data_dir, "hg38.chrom.sizes"),
                    bin_size=config["params"]["omni_bin"])
    cache = pat.Cache(out_root, data_dir, mask, marks=pat.MARKS,
                      peak_path=encode_peak_path(config["params"]["omni_bin"]))
    datasets = config["datasets"]
    return {"config": config, "datasets": datasets, "mask": mask, "cache": cache,
            "data_dir": data_dir}


LEARN_CHROMS = ("chr1",)
N_STATES = 15

# The mixture appears over the peak calls: BMM3 reads the
# previous, own and next bin together (2^18 patterns).
BMM3_INIT = "bmm_spatial_init"
BMM3_UPDATED = "bmm_spatial_updated"
BMM3_REFIT = "bmm_spatial_refit"
KM_INIT = "kmeans_init"
KM_UPDATED = "kmeans_updated"
KM_REFIT = "kmeans_refit"

HOMER_KM_INIT = "homer_kmeans_init"
HOMER_KM_UPDATED = "homer_kmeans_updated"
HOMER_KM_REFIT = "homer_kmeans_refit"
HOMER_BMM3_INIT = "homer_bmm3_init"
HOMER_BMM3_UPDATED = "homer_bmm3_updated"
HOMER_BMM3_REFIT = "homer_bmm3_refit"

MACS2_KM_INIT = "macs2_kmeans_init"
MACS2_KM_UPDATED = "macs2_kmeans_updated"
MACS2_KM_REFIT = "macs2_kmeans_refit"
MACS2_BMM3_INIT = "macs2_bmm3_init"
MACS2_BMM3_UPDATED = "macs2_bmm3_updated"
MACS2_BMM3_REFIT = "macs2_bmm3_refit"

# The refit is the incoming-only update's fresh counterpart.
PATTERN_ARMS = (
    (KM_INIT, "init", "omni_kmeans"), (KM_UPDATED, "updated", "omni_kmeans"),
    (KM_REFIT, "refit", "omni_kmeans"),
    (BMM3_INIT, "init", "omni_bmm3"), (BMM3_UPDATED, "updated", "omni_bmm3"),
    (BMM3_REFIT, "refit", "omni_bmm3"),
    (HOMER_KM_INIT, "init", "homer_kmeans"), (HOMER_KM_UPDATED, "updated", "homer_kmeans"),
    (HOMER_KM_REFIT, "refit", "homer_kmeans"),
    (HOMER_BMM3_INIT, "init", "homer_bmm3"), (HOMER_BMM3_UPDATED, "updated", "homer_bmm3"),
    (HOMER_BMM3_REFIT, "refit", "homer_bmm3"),
    (MACS2_KM_INIT, "init", "macs2_kmeans"), (MACS2_KM_UPDATED, "updated", "macs2_kmeans"),
    (MACS2_KM_REFIT, "refit", "macs2_kmeans"),
    (MACS2_BMM3_INIT, "init", "macs2_bmm3"), (MACS2_BMM3_UPDATED, "updated", "macs2_bmm3"),
    (MACS2_BMM3_REFIT, "refit", "macs2_bmm3")
)

METHODS = ("chromhmm", "omni_kmeans", "omni_bmm3",
           "homer_kmeans", "homer_bmm3", "macs2_kmeans", "macs2_bmm3")
MIXTURES = ("omni_bmm3", "homer_bmm3", "macs2_bmm3")          # the methods fitted by Bernoulli EM
SPATIAL_BINS_OF = {m: 3 for m in MIXTURES}    # ... of which these read spatial bins

METHOD_DISPLAY = {INIT_ARM: "ChromHMM Init", UPDATED: "ChromHMM updated",
                  REFIT: "ChromHMM refit",
                  BMM3_INIT: "BMM3 (Omni) Init", BMM3_UPDATED: "BMM3 (Omni) updated",
                  BMM3_REFIT: "BMM3 (Omni) refit",
                  KM_INIT: "KMeans (Omni) Init", KM_UPDATED: "KMeans (Omni) updated",
                  KM_REFIT: "KMeans (Omni) refit",
                  HOMER_KM_INIT: "KMeans (HOMER) Init",
                  HOMER_KM_UPDATED: "KMeans (HOMER) updated",
                  HOMER_KM_REFIT: "KMeans (HOMER) refit",
                  HOMER_BMM3_INIT: "BMM3 (HOMER) Init",
                  HOMER_BMM3_UPDATED: "BMM3 (HOMER) updated",
                  HOMER_BMM3_REFIT: "BMM3 (HOMER) refit",
                  MACS2_KM_INIT: "KMeans (MACS2) Init",
                  MACS2_KM_UPDATED: "KMeans (MACS2) updated",
                  MACS2_KM_REFIT: "KMeans (MACS2) refit",
                  MACS2_BMM3_INIT: "BMM3 (MACS2) Init",
                  MACS2_BMM3_UPDATED: "BMM3 (MACS2) updated",
                  MACS2_BMM3_REFIT: "BMM3 (MACS2) refit"}


def objective_gap(score, reference, mixture):
    """How much worse a warm-started fit is than a fresh one, in percent.

    K-means reports inertia and the mixture a log-likelihood, so the sign of
    "better" differs; the gap is positive either way when the warm start is
    the worse of the two, and scaled by the fresh fit's own objective.
    """
    if reference is None or score is None or not np.isfinite(reference) \
            or reference == 0:
        return float("nan")
    worse = reference - score if mixture else score - reference
    return 100.0 * worse / abs(reference)


def prepare_binary(cache, mask, datasets, samples, root, bin_size=None,
                   recompute=False, caller="binary"):
    """Per-sample ChromHMM binary files, written once from the pattern tracks."""
    dirs = {}
    for sample in samples:
        cell = datasets[sample]["cell"]
        suffix = "" if bin_size is None or bin_size == mask.bin else f"_{bin_size}"
        directory = os.path.join(root, caller + suffix, sample)
        marker = os.path.join(directory, f"{cell}_{mask.chroms[-1]}_binary.txt.gz")
        if recompute or not cache.has(sample):
            with pat.step(f"preprocess {sample}"):
                cache.process(sample)
        if recompute or not os.path.exists(marker):
            with pat.step(f"binarize {sample} at {bin_size or mask.bin} bp"):
                write_binary(cache, mask, cell, sample, directory,
                             bin_size=bin_size)
        dirs[sample] = directory
    return dirs


def chrom_histogram(cache, samples, mask, chroms, bins=1, ratio=1):
    """Pattern counts of `samples` restricted to `chroms` - the mixture input.

    With `bins > 1`, a bin is counted as its spatial context. For `bins=3`,
    this includes the previous, own and next bin together. The plain counts
    (bins=1) are what K-means reads.
    """
    if bins == 1:
        size = pat.N_PATTERNS
    elif bins == 3:
        size = pat.N_PATTERNS_SPATIAL
    else:
        raise ValueError(f"Unsupported bins={bins}")

    total = np.zeros(size, dtype=np.int64)
    for sample in samples:
        codes = cache.codes(sample)
        for chrom, lo, hi in mask.slices():
            if chrom in chroms:
                c = codes[lo:hi]
                if ratio > 1:
                    n = len(c) // ratio
                    c = np.bitwise_or.reduce(
                        c[:n * ratio].reshape(n, ratio), axis=1)
                if bins > 1:
                    total += bern.spatial_histogram(c, [(None, 0, len(c))],
                                                    len(pat.MARKS), bins=bins)
                else:
                    total += np.bincount(c, minlength=size)
    return total


def cross_sample_rows(labels_by_method, names_by_method, samples, states):
    """How alike two different incoming samples look under each arm."""
    rows = []
    for method, labels in labels_by_method.items():
        names = names_by_method[method]
        exclude = pat.noqh_states(names)
        for i, first in enumerate(samples):
            for second in samples[i + 1:]:
                table = pat.confusion_from_labels(labels[first], labels[second],
                                                  states)
                for domain, dropped in ((utils.FULL, ()), (utils.NOQH, exclude)):
                    rows.append({"method": method, "sample_a": first,
                                 "sample_b": second, "domain": domain,
                                 **pat.cross_sample_metrics(table, names, dropped)})
    return rows


def stability_rows(labels_by_method, names_by_method, samples, states, method):
    """How alike an init/updated arm looks compared to its own refit."""
    rows = []
    arm_refit = REFIT_ARMS[method]
    for scenario, arms in ARMS_OF_SCENARIO.items():
        arm = arms[method]
        names = names_by_method[arm]
        exclude = pat.noqh_states(names)
        for sample in samples:
            table = pat.confusion_from_labels(labels_by_method[arm][sample],
                                              labels_by_method[arm_refit][sample],
                                              states)
            # Refit is trained from scratch and its state order is arbitrary,
            # so even an updated fit that stays perfectly in its init
            # parameters needs matching before they can be compared.
            mapping = pat.state_mapping(table)
            table = table[:, mapping]
            for domain, dropped in ((utils.FULL, ()), (utils.NOQH, exclude)):
                rows.append({"method": method, "scenario": scenario,
                             "sample": sample, "domain": domain,
                             "comparison": comparison_key(method, scenario),
                             **pat.cross_sample_metrics(table, names, dropped)})
    return rows


def _cost_cache(out_dir):
    """Where a measured ChromHMM step's cost is kept."""
    return os.path.join(out_dir, "step_cost.json")


def _measured(out_dir, outputs_exist, run_step):
    """Run and measure a step, or read back the cost of an earlier run."""
    cache_path = _cost_cache(out_dir)
    if outputs_exist and os.path.exists(cache_path):
        with open(cache_path) as handle:
            pat.log(f"reading cost from {out_dir}", indent=2)
            return {**json.load(handle), "cached": True}
    cost = run_step()
    os.makedirs(out_dir, exist_ok=True)
    with open(cache_path, "w") as handle:
        json.dump(cost, handle, indent=2, default=float)
    return {**cost, "cached": False}


def _measured_op(out_dir, outputs_exist, run_step):
    """Run a picklable operation and cache both its result and cost."""
    cache_path = _cost_cache(out_dir)
    result_path = os.path.join(out_dir, "result.pkl")
    if outputs_exist and os.path.exists(cache_path) and os.path.exists(result_path):
        with open(cache_path) as handle:
            cost = json.load(handle)
        with open(result_path, "rb") as handle:
            result = pickle.load(handle)
        pat.log(f"reading result from {out_dir}", indent=2)
        return result, {**cost, "cached": True}
    result, cost = run_step()
    os.makedirs(out_dir, exist_ok=True)
    with open(cache_path, "w") as handle:
        json.dump(cost, handle, indent=2, default=float)
    with open(result_path, "wb") as handle:
        pickle.dump(result, handle)
    return result, {**cost, "cached": False}


# Every number read off the fits is cached separately and survives, but
# every number read off them does not.


def _rows_signature():
    """What a cached set of rows was produced by."""
    return {"reference_match": pat.REFERENCE_MATCH}


def _cached_rows(out_dir, recompute, compute_rows):
    """Cache the table rows of one method, keyed by what produced them."""
    rows_path = os.path.join(out_dir, "rows.json")
    if not recompute and os.path.exists(rows_path):
        with open(rows_path) as handle:
            cached = json.load(handle)
        if cached.get("signature") == _rows_signature():
            pat.log(f"reading rows from {out_dir}", indent=2)
            return cached["rows"]
        pat.log(f"rows in {out_dir} were produced by {cached.get('signature')}, "
                f"recomputing", indent=2)
    rows = compute_rows()
    os.makedirs(out_dir, exist_ok=True)
    with open(rows_path, "w") as handle:
        json.dump({"signature": _rows_signature(), "rows": rows}, handle,
                  indent=2, default=float)
    return rows


def _model_path(out_dir, states):
    """The model `-init information` writes: one name, no seed in it."""
    return os.path.join(out_dir, f"model_{states}.txt")


def _emissions_path(out_dir, states):
    return os.path.join(out_dir, f"emissions_{states}.txt")


def _method_dir(root, split_name, suffix):
    """Where a method's aggregated table rows are kept."""
    return os.path.join(root, f"method_{split_name}{suffix}")


def _tag(rows, split):
    return [{"split": split, **row} for row in rows]


def run(out_root=None, seed=SEED, states=N_STATES, learn_chroms=LEARN_CHROMS,
        n_splits=N_SPLITS, bin_size=None, peak_bin=None, recompute=False):
    """Every arm on `n_splits` initial/incoming splits of the same 11 samples.

    Every arm is fitted at `seed` and scored there, ChromHMM included.
    """
    env = setup()
    cache, mask, datasets = env["cache"], env["mask"], env["datasets"]
    params = env["config"]["params"]
    chmm_bin = bin_size or params["chromhmm_bin"]
    omni_bin = peak_bin or params.get("omni_bin", 100)
    homer_bin = peak_bin or params.get("homer_bin", 100)
    macs2_bin = peak_bin or params.get("macs2_bin", 100)
    shared = os.path.join(env["data_dir"], "out", "transfer")
    root = out_root or shared
    os.makedirs(root, exist_ok=True)
    jar = os.path.join(env["data_dir"], "ChromHMM", "ChromHMM.jar")
    assembly = params["genome"]
    splits = choose_splits(datasets, n_splits)

    # The binary files are the same for every output root, so they live beside
    # the experiment rather than inside one run's directory.
    def _peak_path(caller_key):
        return lambda sd, m: os.path.join(sd, caller_key, f"{m}.bed")

    caches = {
        "omni": cache,
        "homer": pat.Cache(os.path.join(env["data_dir"], "cache_homer"), env["data_dir"], mask, pat.MARKS, _peak_path("homer")),
        "macs2": pat.Cache(os.path.join(env["data_dir"], "cache_macs2"), env["data_dir"], mask, pat.MARKS, _peak_path("macs2")),
    }
    caller_bins = {"omni": omni_bin, "homer": homer_bin, "macs2": macs2_bin}

    binary_configs = [("omni", omni_bin), ("homer", homer_bin),
                      ("macs2", macs2_bin), ("omni", chmm_bin)]
    binaries = {}
    for c, b in binary_configs:
        key = f"{c}_{b}"
        if key not in binaries:
            binaries[key] = prepare_binary(
                caches[c], mask, datasets, list(ALL_SAMPLES), shared,
                bin_size=b, recompute=recompute, caller=f"binary_{c}_{b}")

    tables = {key: [] for key in TABLES}
    for split in tqdm(splits, desc="Splits"):
        name = split["name"]
        initial, incoming = split["initial"], split["incoming"]
        samples = list(initial) + list(incoming)
        pat.log(f"{name}: {len(initial)} initial, {len(incoming)} incoming",
                indent=0)
        # `updated` continues the init fit on the incoming samples alone -
        # what a lab holding only the new data can do.
        training_of = {"init": initial, "updated": incoming, "refit": samples}

        with pat.step("count patterns on " + ",".join(learn_chroms)):
            pools = {c: {b: {} for b in (1, 3)} for c in caches}
            for c in caches:
                ratio = caller_bins[c] // mask.bin
                for b in (1, 3):
                    for scenario in ("init", "updated"):
                        pools[c][b][scenario] = chrom_histogram(
                            caches[c], training_of[scenario], mask,
                            set(learn_chroms), bins=b, ratio=ratio)
                    pools[c][b]["refit"] = (pools[c][b]["init"]
                                            + pools[c][b]["updated"])

        reference = [None]
        def get_reference():
            if reference[0] is None:
                reference[0] = {s: pat.reference_labels(s, mask, datasets,
                                                       env["data_dir"])
                                for s in incoming}
            return reference[0]

        for method in METHODS:
            m_root = os.path.join(root, method)
            caller = method.split("_")[0] if "_" in method else "omni"
            if method == "chromhmm":
                m_bin_size = chmm_bin
                b_key = f"omni_{chmm_bin}"
            else:
                m_bin_size = caller_bins[caller]
                b_key = f"{caller}_{m_bin_size}"
            m_suffix = "" if m_bin_size == mask.bin else f"_{m_bin_size}"
            method_dir = _method_dir(m_root, name, m_suffix)

            def compute_method_rows(method=method, m_bin_size=m_bin_size, m_suffix=m_suffix, b_key=b_key):
                m_tables = {k: [] for k in TABLES}
                m_labels, m_models = {}, {}
                caller = method.split("_")[0] if "_" in method else "omni"
                m_binary = binaries[b_key]

                if method == "chromhmm":
                    for arm in ARMS:
                        training = training_of[SCENARIO_OF[arm]]
                        learn_dir = link_cells([m_binary[s] for s in training],
                                               os.path.join(m_root, f"input_{name}_{arm}"),
                                               set(learn_chroms))
                        out_dir = os.path.join(m_root, f"model_{name}_{arm}{m_suffix}")
                        warm = (m_models[INIT_ARM]["model"]
                                if arm == UPDATED else None)
                        cost = _measured(out_dir, not recompute and os.path.exists(
                            _model_path(out_dir, states)),
                                         lambda: learn_model(jar, learn_dir, out_dir, states,
                                                             assembly, m_bin_size,
                                                             seed, threads=THREADS,
                                                             model=warm))
                        m_models[arm] = {"model": _model_path(out_dir, states),
                                         "out_dir": out_dir,
                                         "emissions": load_emissions(
                                             _emissions_path(out_dir, states)),
                                         "cost": cost, "n_training": len(training)}

                    segment_dir = link_cells([m_binary[s] for s in incoming],
                                             os.path.join(m_root, f"input_{name}_incoming"))
                    for arm in ARMS:
                        out_dir = os.path.join(m_root, f"segmentation_{name}_{arm}{m_suffix}")
                        expected = [os.path.join(out_dir, f"{datasets[s]['cell']}_{states}"
                                                          f"_segments.bed.gz") for s in incoming]
                        cost = _measured(out_dir, not recompute and all(
                            os.path.exists(p) for p in expected),
                                         lambda: make_segmentation(jar, m_models[arm]["model"],
                                                                   segment_dir, out_dir,
                                                                   m_bin_size))
                        m_labels[arm] = {s: load_labels(p, mask)
                                         for s, p in zip(incoming, expected)}
                        m_models[arm].update({
                            "names": pat.interpret_centroids(m_models[arm]["emissions"]),
                            "annotate_cost": cost})

                        m_tables["cost"] += [
                            {"method": arm, "step": "learn_model",
                             "n_samples": m_models[arm]["n_training"],
                             **m_models[arm]["cost"]},
                            {"method": arm, "step": "make_segmentation",
                             "n_samples": len(incoming), **m_models[arm]["annotate_cost"]}]
                        m_tables["states"] += [{"method": arm, "state": state,
                                                **{mark: float(m_models[arm]["emissions"][j, i])
                                                   for i, mark in enumerate(pat.MARKS)}}
                                               for j, state in enumerate(m_models[arm]["names"])]
                        m_tables["cross_sample"] += cross_sample_rows(
                            {arm: m_labels[arm]}, {arm: m_models[arm]["names"]},
                            list(incoming), states)
                else:
                    init_fit, fresh_score = {}, {}
                    ratio = m_bin_size // mask.bin
                    for arm, scenario, m_type in PATTERN_ARMS:
                        if m_type != method:
                            continue
                        mixture = m_type in MIXTURES
                        spatial_bins = SPATIAL_BINS_OF.get(m_type, 1)
                        counts = pools[caller][spatial_bins][scenario]
                        training = training_of[scenario]
                        fresh = INIT if mixture else None
                        warm = (init_fit.get(m_type)
                                if scenario == UPDATED else None)
                        operation = pat.fit_op if mixture else pat.fit_kmeans_op
                        restarts = N_INIT if mixture else pat.KMEANS_N_INIT
                        out_dir = os.path.join(m_root, f"model_{name}_{arm}_{seed}{m_suffix}")

                        fit, cost = _measured_op(out_dir, not recompute,
                                                lambda: pat.measure(operation, counts,
                                                                    (warm[0], warm[1]) if warm is not None
                                                                    else fresh, seed, restarts))
                        params, weights, score = fit
                        if scenario == "init":
                            init_fit[m_type] = fit

                        # A warm start runs one pass from the init
                        # parameters while a fresh fit gets the full restart
                        # budget, so the two are not comparable until the
                        # fresh fit on the same counts is there to be read
                        # beside it.  For the incoming-only update nothing
                        # else fits those counts, so it is measured here and
                        # charged to its own step.
                        reference_score = None
                        if scenario == "refit":
                            fresh_score[m_type] = score
                        elif scenario == UPDATED:
                            ref_dir = os.path.join(
                                m_root, f"model_{name}_{arm}_fresh_{seed}{m_suffix}")
                            ref_fit, ref_cost = _measured_op(
                                ref_dir, not recompute,
                                lambda: pat.measure(operation, counts, fresh,
                                                    seed, restarts))
                            reference_score = ref_fit[2]
                            m_tables["cost"].append(
                                {"method": arm, "step": "reference_fit",
                                 "n_samples": len(training),
                                 "wall": ref_cost["inner_wall"],
                                 "cpu": ref_cost["inner_cpu"],
                                 "peak_mb": ref_cost["peak_mb"],
                                 "score": reference_score,
                                 "cached": ref_cost.get("cached", False),
                                 "command": ("fit_bernoulli" if mixture else "fit_kmeans")
                                            + " (fresh restart)"})

                        lut = (pat.lookup_table(params, weights) if mixture
                               else pat.kmeans_lookup_table(params))
                        names = pat.interpret_centroids(params)

                        started = time.perf_counter()
                        m_labels[arm] = {}
                        for s in incoming:
                            codes = caches[caller].codes(s)
                            all_labels = np.zeros(mask.total_bins, dtype=lut.dtype)
                            for chrom, lo, hi in mask.slices():
                                c = codes[lo:hi]
                                if ratio > 1:
                                    n = len(c) // ratio
                                    c = np.bitwise_or.reduce(
                                        c[:n * ratio].reshape(n, ratio), axis=1)
                                l = (bern.spatial_labels(lut, c, [(None, 0, len(c))],
                                                         len(pat.MARKS), bins=spatial_bins)
                                     if spatial_bins > 1 else lut[c])
                                if ratio > 1:
                                    l = np.repeat(l, ratio)
                                    # Pad to match original hi-lo
                                    if len(l) < hi - lo:
                                        l = np.pad(l, (0, hi - lo - len(l)), mode="edge")
                                all_labels[lo:hi] = l
                            m_labels[arm][s] = all_labels

                        annotate_wall = time.perf_counter() - started
                        m_models[arm] = {"emissions": params, "weights": weights,
                                         "names": names}

                        m_tables["cost"] += [
                            {"method": arm, "step": "learn_model",
                             "n_samples": len(training), "wall": cost["inner_wall"],
                             "cpu": cost["inner_cpu"], "peak_mb": cost["peak_mb"],
                             "score": score,
                             "score_reference": reference_score,
                             "score_gap_pct": objective_gap(score,
                                                            reference_score,
                                                            mixture),
                             "cached": cost.get("cached", False),
                             "command": ("fit_bernoulli" if mixture else "fit_kmeans")
                                        + ("(warm start)" if warm is not None else "")},
                            {"method": arm, "step": "make_segmentation",
                             "n_samples": len(incoming), "wall": annotate_wall,
                             "cpu": float("nan"), "peak_mb": float("nan"),
                             "cached": False, "command": "lut[codes]"}]
                        own = pat.own_marks(params)
                        m_tables["states"] += [{"method": arm, "state": state,
                                                **{mark: float(own[j, i])
                                                   for i, mark in enumerate(pat.MARKS)}}
                                               for j, state in enumerate(m_models[arm]["names"])]
                        m_tables["cross_sample"] += cross_sample_rows(
                            {arm: m_labels[arm]}, {arm: m_models[arm]["names"]}, list(incoming), states)

                # Multi-arm rows for this method
                ref = get_reference()
                maps = {arm: pat.reference_mapping(m_labels[arm], ref, incoming,
                                                   N_STATES)
                        for arm in m_labels}
                for sample in incoming:
                    arm_refit = REFIT_ARMS[method]
                    for scenario, arms in ARMS_OF_SCENARIO.items():
                        arm = arms[method]
                        m_tables["state_jaccard"] += pat.reference_state_rows(
                            m_labels[arm][sample], m_labels[arm_refit][sample],
                            maps[arm], maps[arm_refit], sample,
                            comparison_key(method, scenario))
                m_tables["stability"] = stability_rows(
                    m_labels, {arm: m_models[arm]["names"] for arm in m_labels}, list(incoming),
                    states, method)

                return m_tables

            with pat.step(f"{name}: {method}"):
                m_tables = _cached_rows(method_dir, recompute, compute_method_rows)
            for k, rows in m_tables.items():
                tables[k] += _tag(rows, name)

    frames = {key: pd.DataFrame(rows) for key, rows in tables.items() if rows}
    cached = frames["cost"].get("cached")
    report = {"settings": {"states": states, "seed": seed,
                           "n_splits": len(splits),
                           "pattern_bin": mask.bin,
                           "chromhmm_bin": bin_size,
                           "peak_bin": peak_bin,
                           "arms": list(METHOD_PLOT_ORDER),
                           "learn_chroms": list(learn_chroms), "threads": THREADS,
                           "tolerance": TOLERANCE, "assembly": assembly,
                           "reference_match": pat.REFERENCE_MATCH,
                           "splits": [{"name": s["name"], "initial": s["initial"],
                                       "incoming": s["incoming"]} for s in splits],
                           "mask": mask.signature()},
              "reused_cached_cost": (int(cached.fillna(False).astype(bool).sum())
                                     if cached is not None else 0),
              "summary": summarise(frames)}
    os.makedirs(root, exist_ok=True)
    pat.log(f"saving results to {root}")
    with open(os.path.join(root, "report.json"), "w") as handle:
        json.dump(report, handle, indent=2, default=float)
    for key, frame in frames.items():
        frame.to_csv(os.path.join(root, f"{key}.csv"), index=False)
    return report, frames


def summarise(frames):
    """Agreement and stability metrics across the cohort."""
    out = {}
    cross = frames.get("cross_sample")
    if cross is not None:
        out["cross_sample"] = json.loads(
            cross.groupby(["domain", "method"])[[utils.KAPPA]].mean()
            .reset_index().to_json(orient="records"))
    reference = frames.get("state_jaccard")
    if reference is not None and not reference.empty:
        out["state_kappa"] = json.loads(
            reference.pivot_table(index="state", columns="comparison",
                                  values="kappa")
            .reset_index().to_json(orient="records"))
        background = utils.NOQH_STATES
        for comparison, group in reference.groupby("comparison"):
            key = comparison.replace("_vs_refit", "")
            active = group[~group["state"].isin(background)]
            out[f"state_kappa_{key}"] = float(group["kappa"].mean())
            out[f"state_kappa_median_{key}"] = float(group["kappa"].median())
            # The quiescent bulk agrees with everything and is two thirds of
            # the genome, so the headline number is also read without it.
            out[f"state_kappa_noqh_{key}"] = float(active["kappa"].mean())
    stability = frames.get("stability")
    if stability is not None and not stability.empty:
        out["stability"] = json.loads(
            stability.groupby(["domain", "comparison"])[[utils.KAPPA]].mean()
            .reset_index().to_json(orient="records"))
    return out


TABLES = ("cross_sample", "state_jaccard", "stability", "cost",
          "states")


def default_root(out_root=None):
    """Where a run keeps its tables."""
    return (os.path.join(os.path.expanduser(DATA_DIR), "out", "transfer")
            if out_root is None else out_root)


def load(root):
    """Read back what a previous run wrote under `root`: (report, frames)."""
    with open(os.path.join(root, "report.json")) as handle:
        report = json.load(handle)
    frames = {}
    for name in TABLES:
        path = os.path.join(root, f"{name}.csv")
        if os.path.exists(path):
            pat.log(f"reading {name} from {root}")
            frames[name] = pd.read_csv(path)
    return report, frames


REQUIRED_TABLES = ("cost", "state_jaccard", "cross_sample",
                   "stability")


def stale(report, frames, **requested):
    """Why the tables on disk do not answer the question being asked.

    Existence of `report.json` is not enough: it can carry a different split
    count, a different matcher or arms this module no longer produces, and the
    notebook would plot it without noticing.
    """
    settings = report.get("settings", {})
    reasons = []
    for key, wanted in requested.items():
        found = settings.get(key)
        if found != wanted:
            reasons.append(f"{key}: {found!r} on disk, {wanted!r} requested")
    matcher = settings.get("reference_match")
    if matcher != pat.REFERENCE_MATCH:
        reasons.append(f"reference match: {matcher!r} on disk, "
                       f"{pat.REFERENCE_MATCH!r} now")
    arms = list(settings.get("arms") or ())
    if arms != list(METHOD_PLOT_ORDER):
        extra = [a for a in arms if a not in METHOD_PLOT_ORDER]
        missing = [a for a in METHOD_PLOT_ORDER if a not in arms]
        reasons.append(f"arms: on disk has {extra or 'none'} extra and "
                       f"{missing or 'none'} missing")
    for table in REQUIRED_TABLES:
        if table not in frames:
            reasons.append(f"{table}.csv is missing")
    return reasons


def load_or_run(out_root=None, recompute=False, **kwargs):
    """The tables of a previous run when they match `kwargs`, else a fresh run.

    A request for more splits than `choose_splits()` can build is clamped by
    the runner, so asking for more recomputes every time.
    """
    kwargs.setdefault("seed", SEED)
    kwargs.setdefault("states", N_STATES)
    kwargs.setdefault("n_splits", N_SPLITS)
    params = load_config()["params"]
    kwargs["bin_size"] = (kwargs.get("bin_size") or params["chromhmm_bin"])
    kwargs["peak_bin"] = (kwargs.get("peak_bin") or params.get("macs2_bin", 100))

    pat.log(f"Transfer experiment: {kwargs['states']} states, "
            f"seed {kwargs['seed']}, {kwargs['n_splits']} splits", indent=0)

    root = default_root(out_root)
    reasons = ["no report.json under that root"]
    if not recompute and os.path.exists(os.path.join(root, "report.json")):
        report, frames = load(root)
        reasons = stale(report, frames, seed=kwargs.get("seed"),
                        states=kwargs.get("states"),
                        chromhmm_bin=kwargs.get("bin_size"),
                        peak_bin=kwargs.get("peak_bin"),
                        n_splits=kwargs.get("n_splits"))
        if not reasons:
            pat.log(f"reading cache from {root}")
            _display_report(report)
            return report, frames
        print("  recomputing, the tables on disk do not match:")
        for reason in reasons:
            print(f"    - {reason}")
    report, frames = run(out_root=out_root, recompute=recompute, **kwargs)
    _display_report(report)
    return report, frames


def _display_report(report):
    """Print the provenance table if in a notebook."""
    try:
        from IPython.display import display
        display(provenance(report))
    except (ImportError, NameError):
        pass


def provenance(report):
    """What the tables on disk were produced from, as a one-column frame."""
    settings = report.get("settings", {})
    splits = settings.get("splits", [])
    rows = {
        "states": settings.get("states"),
        "learn_chroms": ", ".join(settings.get("learn_chroms", [])),
        "ChromHMM bin (bp)": settings.get("chromhmm_bin"),
        "pattern bin (bp)": settings.get("pattern_bin"),
        "peak bin (bp)": settings.get("peak_bin"),
        "seed": settings.get("seed"),
        "splits": len(splits),
        "incoming per split": ", ".join(
            "+".join(split["incoming"]) for split in splits),
        "arms": len(settings.get("arms", [])),
        "tolerance": settings.get("tolerance"),
        "reference match": settings.get("reference_match"),
        "cost rows read from cache": report.get("reused_cached_cost"),
    }
    return pd.DataFrame({"value": pd.Series(rows, dtype=object)})


def summary_table(report, frames):
    """Stability metrics, one row per arm.

    A row is a method under one of the three scenarios that have a refit to
    lose against; everything is read off the frames rather than the summary
    keys, so an arm the tables do not carry simply does not appear.
    """
    if "stability" not in frames:
        return pd.DataFrame()

    index = ["method", "scenario"]
    stability = frames["stability"].assign(
        method=lambda frame: frame["comparison"].map(comparison_method),
        scenario=lambda frame: frame["comparison"].map(comparison_scenario))

    full = stability[stability["domain"] == utils.FULL]
    noqh = stability[stability["domain"] == utils.NOQH]

    table = pd.DataFrame({
        "stability": full.groupby(index)["kappa"].mean(),
        "stability_noqh": noqh.groupby(index)["kappa"].mean(),
    })
    order = [(method, scenario) for method in PLOT_METHODS
             for scenario in ARMS_OF_SCENARIO if (method, scenario) in table.index]
    table = table.reindex(order + [k for k in table.index if k not in order])
    return table.rename(index=SCENARIO_LABEL, level="scenario")


def objective_table(frames):
    """What a warm start gives up against a fresh fit on the same counts.

    Only the pattern models have a restart budget to be short of: ChromHMM's
    `-init information` is a single deterministic pass, so its rows carry no
    score and do not appear.
    """
    cost = frames.get("cost")
    if cost is None or "score_gap_pct" not in cost:
        return pd.DataFrame()
    rows = cost[(cost["step"] == "learn_model")
                & cost["score_gap_pct"].notna()]
    if rows.empty:
        return pd.DataFrame()
    table = (rows.groupby("method")[["score", "score_reference",
                                     "score_gap_pct"]].mean()
             .rename(columns={"score": "warm objective",
                              "score_reference": "fresh objective",
                              "score_gap_pct": "warm is worse by (%)"}))
    learn = cost[cost["step"] == "learn_model"].groupby("method")["wall"].mean()
    fresh = cost[cost["step"] == "reference_fit"].groupby("method")["wall"].mean()
    table["warm fit (s)"] = learn
    table["fresh fit (s)"] = fresh
    return (table.reindex([a for a in METHOD_PLOT_ORDER if a in table.index])
            .rename(index=METHOD_DISPLAY))


# Twenty-one arms need more width than the five-method panels do.
ARM_FIG = (9.6, 4.4)


def plots(frames, summary, outdir):
    """Every panel as its own figure, for a notebook that shows them singly."""
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    written = []

    def draw_fig(name, size, build, tight=True):
        fig = plt.figure(figsize=size)
        build(fig)
        path = os.path.join(outdir, f"{name}.png")
        utils.save_fig(fig, path, tight=tight)
        written.append(path)

    def draw(name, size, panel):
        draw_fig(name, size, lambda fig: panel(fig.subplots()))

    if "cross_sample" in frames:
        for metric in (utils.KAPPA,):
            for domain in (utils.FULL, utils.NOQH):
                draw(f"cross_sample_{metric}_{domain}", ARM_FIG,
                     lambda ax, m=metric, d=domain: _cross_sample_panel(
                         ax, frames["cross_sample"], m, d))
    if "cost" in frames:
        draw("cost", ARM_FIG, lambda ax: _cost_panel(ax, frames["cost"]))
    if "state_jaccard" in frames:
        draw_fig("state_kappa", (13, 4.2 * len(ARMS_OF_SCENARIO)),
                 lambda fig: _state_figure(fig, frames["state_jaccard"]),
                 tight=False)
        for scenario in ARMS_OF_SCENARIO:
            draw_fig(f"state_kappa_{scenario}", (13, 4.8),
                     lambda fig, s=scenario: _state_figure(
                         fig, frames["state_jaccard"], scenarios=[s]),
                     tight=False)
            draw(f"state_kappa_summary_{scenario}", (7.0, 4.4),
                 lambda ax, s=scenario: _state_agreement_summary_panel(
                     ax, frames["state_jaccard"], s))
    if "stability" in frames:
        for metric in (utils.KAPPA,):
            for domain in (utils.FULL, utils.NOQH):
                for scenario in ARMS_OF_SCENARIO:
                    draw(f"stability_{metric}_{domain}_{scenario}", (7.0, 4.4),
                         lambda ax, m=metric, d=domain, s=scenario: _stability_panel(
                             ax, frames["stability"], m, d, s))
    return written


# --- figures ---------------------------------------------------------------
#
# A method is a colour and a scenario is a hatch, so an arm is readable from
# either alone.

METHOD_OF = {
    INIT_ARM: "chromhmm", UPDATED: "chromhmm", REFIT: "chromhmm",
    BMM3_INIT: "omni_bmm3", BMM3_UPDATED: "omni_bmm3",
    BMM3_REFIT: "omni_bmm3",
    KM_INIT: "omni_kmeans", KM_UPDATED: "omni_kmeans",
    KM_REFIT: "omni_kmeans",
    HOMER_KM_INIT: "homer_kmeans", HOMER_KM_UPDATED: "homer_kmeans",
    HOMER_KM_REFIT: "homer_kmeans",
    HOMER_BMM3_INIT: "homer_bmm3", HOMER_BMM3_UPDATED: "homer_bmm3",
    HOMER_BMM3_REFIT: "homer_bmm3",
    MACS2_KM_INIT: "macs2_kmeans", MACS2_KM_UPDATED: "macs2_kmeans",
    MACS2_KM_REFIT: "macs2_kmeans",
    MACS2_BMM3_INIT: "macs2_bmm3", MACS2_BMM3_UPDATED: "macs2_bmm3",
    MACS2_BMM3_REFIT: "macs2_bmm3"}
SCENARIO_OF = {
    INIT_ARM: "init", UPDATED: "updated", REFIT: "refit",
    BMM3_INIT: "init", BMM3_UPDATED: "updated", BMM3_REFIT: "refit",
    KM_INIT: "init", KM_UPDATED: "updated", KM_REFIT: "refit",
    HOMER_KM_INIT: "init", HOMER_KM_UPDATED: "updated",
    HOMER_KM_REFIT: "refit",
    HOMER_BMM3_INIT: "init", HOMER_BMM3_UPDATED: "updated",
    HOMER_BMM3_REFIT: "refit",
    MACS2_KM_INIT: "init", MACS2_KM_UPDATED: "updated",
    MACS2_KM_REFIT: "refit",
    MACS2_BMM3_INIT: "init", MACS2_BMM3_UPDATED: "updated",
    MACS2_BMM3_REFIT: "refit"}


INIT_ARMS = {METHOD_OF[arm]: arm
             for arm in (KM_INIT, BMM3_INIT, HOMER_KM_INIT,
                         HOMER_BMM3_INIT, MACS2_KM_INIT,
                         MACS2_BMM3_INIT, INIT_ARM)}
UPDATED_ARMS = {METHOD_OF[arm]: arm
                for arm in (KM_UPDATED, BMM3_UPDATED, HOMER_KM_UPDATED,
                            HOMER_BMM3_UPDATED, MACS2_KM_UPDATED,
                            MACS2_BMM3_UPDATED, UPDATED)}
REFIT_ARMS = {METHOD_OF[arm]: arm
              for arm in (KM_REFIT, BMM3_REFIT, HOMER_KM_REFIT,
                          HOMER_BMM3_REFIT, MACS2_KM_REFIT,
                          MACS2_BMM3_REFIT, REFIT)}
# Every scenario that has a refit to be compared against, in plot order.
ARMS_OF_SCENARIO = {"init": INIT_ARMS, "updated": UPDATED_ARMS}


def comparison_key(method, scenario):
    """How a state_jaccard row names one arm's comparison with its own refit."""
    return f"{method}_{scenario}_vs_refit"


def comparison_method(comparison):
    """The method a comparison key belongs to."""
    for scenario in SCENARIOS:
        suffix = f"_{scenario}_vs_refit"
        if comparison.endswith(suffix):
            return comparison[:-len(suffix)]
    return comparison


def comparison_scenario(comparison):
    """The scenario a comparison key belongs to."""
    for scenario in SCENARIOS:
        if comparison.endswith(f"_{scenario}_vs_refit"):
            return scenario
    return None


METHOD_COLORS = {
    "omni_kmeans":   utils.method_color(utils.KMEANS_OMNI),
    "omni_bmm3":     utils.method_color(utils.BMM3_OMNI),
    "chromhmm":      utils.method_color(utils.CHROMHMM_DEFAULT),
    "homer_kmeans":  utils.method_color(utils.KMEANS_HOMER),
    "homer_bmm3":    utils.method_color(utils.BMM3_HOMER),
    "macs2_kmeans":  utils.method_color(utils.KMEANS_MACS2),
    "macs2_bmm3":    utils.method_color(utils.BMM3_MACS2),
}
METHOD_LABEL = {"omni_kmeans": "KMeans (Omni)",
                "omni_bmm3": "BMM3 (Omni)",
                "chromhmm": "ChromHMM",
                "homer_kmeans": "KMeans (HOMER)",
                "homer_bmm3": "BMM3 (HOMER)",
                "macs2_kmeans": "KMeans (MACS2)",
                "macs2_bmm3": "BMM3 (MACS2)"}
SCENARIO_HATCH = {"init": "xx", "updated": "..", "refit": None}
SCENARIO_LABEL = {"init": "Init", "updated": "updated (new only)",
                  "refit": "refit"}
METHOD_PLOT_ORDER = (KM_INIT, KM_UPDATED, KM_REFIT,
                     BMM3_INIT, BMM3_UPDATED, BMM3_REFIT,
                     INIT_ARM, UPDATED, REFIT,
                     HOMER_KM_INIT, HOMER_KM_UPDATED, HOMER_KM_REFIT,
                     HOMER_BMM3_INIT, HOMER_BMM3_UPDATED, HOMER_BMM3_REFIT,
                     MACS2_KM_INIT, MACS2_KM_UPDATED, MACS2_KM_REFIT,
                     MACS2_BMM3_INIT, MACS2_BMM3_UPDATED, MACS2_BMM3_REFIT)
SHORT_LABEL = {KM_INIT: "KM\ninit", KM_UPDATED: "KM\nupd",
               KM_REFIT: "KM\nrefit",
               BMM3_INIT: "BMM3\ninit", BMM3_UPDATED: "BMM3\nupd",
               BMM3_REFIT: "BMM3\nrefit",
               INIT_ARM: "CHMM\ninit", UPDATED: "CHMM\nupd",
               REFIT: "CHMM\nrefit",
               HOMER_KM_INIT: "HKM\ninit", HOMER_KM_UPDATED: "HKM\nupd",
               HOMER_KM_REFIT: "HKM\nrefit",
               HOMER_BMM3_INIT: "HBMM\ninit", HOMER_BMM3_UPDATED: "HBMM\nupd",
               HOMER_BMM3_REFIT: "HBMM\nrefit",
               MACS2_KM_INIT: "MKM\ninit", MACS2_KM_UPDATED: "MKM\nupd",
               MACS2_KM_REFIT: "MKM\nrefit",
               MACS2_BMM3_INIT: "MBMM\ninit", MACS2_BMM3_UPDATED: "MBMM\nupd",
               MACS2_BMM3_REFIT: "MBMM\nrefit"}
PLOT_METHODS = ("omni_kmeans", "omni_bmm3", "chromhmm",
                "homer_kmeans", "homer_bmm3", "macs2_kmeans", "macs2_bmm3")
def comparison_series(scenario):
    """One entry per method, in plot order.

    The comparison rows it carries, the arm whose colour and hatch it takes,
    and its legend label.
    """
    arms = ARMS_OF_SCENARIO[scenario]
    return tuple((comparison_key(method, scenario), arms[method],
                  METHOD_LABEL[method]) for method in PLOT_METHODS)


def _series_offsets(n, width):
    """Bar offsets that centre `n` series of `width` on their tick."""
    return (np.arange(n) - (n - 1) / 2) * width


def _style(ax, title, ylabel, pad=None):
    ax.set_title(title, pad=pad, **utils.TITLE_STYLE)
    ax.set_ylabel(ylabel, fontsize=utils.AXIS_FONTSIZE)
    ax.tick_params(labelsize=utils.TICK_FONTSIZE)
    ax.grid(axis="y", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _bar_style(arm):
    return dict(color=METHOD_COLORS[METHOD_OF[arm]],
                hatch=SCENARIO_HATCH[SCENARIO_OF[arm]],
                edgecolor="white", linewidth=0.9)


def _arm_axis(ax):
    arms = METHOD_PLOT_ORDER
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([SHORT_LABEL[a] for a in arms], fontsize=5.5)
    # Twenty-one ticks on one axis: each is tinted its method's colour, so the
    # three scenarios of a method read as a group without reading the text.
    for tick, arm in zip(ax.get_xticklabels(), arms):
        tick.set_color(METHOD_COLORS[METHOD_OF[arm]])


def _legend(ax, scenarios=("init", "updated", "refit")):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=METHOD_COLORS[m], edgecolor="white",
                              label=METHOD_LABEL[m]) for m in METHOD_COLORS]
                       + [Patch(facecolor="#DDDDDD", edgecolor="white",
                                hatch=SCENARIO_HATCH[s], label=SCENARIO_LABEL[s])
                          for s in scenarios],
               loc="upper left", ncol=5, frameon=False, fontsize=6.5,
               handlelength=1.4, columnspacing=0.8)




def _cross_sample_panel(ax, cross_sample, metric, domain):
    """Cross-sample agreement per arm: do two different samples look alike?"""
    subset = cross_sample[cross_sample["domain"] == domain]
    for x, arm in enumerate(METHOD_PLOT_ORDER):
        values = subset[subset["method"] == arm][metric]
        if values.empty:
            continue
        ax.bar(x, float(values.mean()), 0.66, **_bar_style(arm))
        utils.scatter_points(ax, x, values, jitter=0.08, size=5, alpha=0.6)
        ax.annotate(f"{values.mean():.2f}", (x, float(values.mean())),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=6)
    _arm_axis(ax)
    low, high = ax.get_ylim()
    ax.set_ylim(0, high * 1.35)
    domain_label = "FULL" if domain == "full" else "NOQH"
    _style(ax, f"Cross-sample {metric}, {domain_label} (6 incoming pairs)",
           metric.capitalize())
    _legend(ax)


def _cost_panel(ax, cost):
    """Fitting and annotation cost per arm, log axis: they differ by 100x."""
    steps = (("learn_model", "Fit"), ("make_segmentation", "Annotate 4"))
    width = 0.38
    for offset, (step, label) in zip((-width / 2, width / 2), steps):
        for x, arm in enumerate(METHOD_PLOT_ORDER):
            values = cost[(cost["method"] == arm)
                          & (cost["step"] == step)]["wall"].dropna()
            if values.empty:
                continue
            value = float(values.mean())
            ax.bar(x + offset, value, width, **_bar_style(arm),
                   alpha=1.0 if step == "learn_model" else 0.55)
            utils.scatter_points(ax, x + offset, values, jitter=0.07, size=4,
                                 alpha=0.75)
            ax.annotate(f"{value:,.3g}", (x + offset, value),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=5.5, rotation=90)
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * 8)
    _arm_axis(ax)
    _style(ax, "Cost: fit (solid) and annotate (pale), mean of the splits",
           "Seconds (wall)")
    _legend(ax)


# The state order of REFERENCE_STATES is already grouped by function; these are
# the boundaries between those groups, drawn so a reader can tell a promoter
# state from the background bulk without reading the tick labels.
STATE_GROUPS = (("promoter", pat.PROMOTER_FAMILY),
                ("enhancer", ("Enh1", "Enh2", "EnhG1", "EnhG2")),
                ("bivalent / repressed", ("Biv", "ReprPC", "Het", "ZNF/Rpts")),
                ("transcribed", pat.TX_FAMILY),
                ("quiescent", ("Quies",)))


def _state_agreement_summary_panel(ax, state_jaccard, scenario):
    """Mean per-state agreement (Kappa) against the refit, averaged over all states."""
    series = comparison_series(scenario)
    for x, (comparison, arm, label) in enumerate(series):
        subset = state_jaccard[state_jaccard["comparison"] == comparison]
        if subset.empty:
            continue
        # Each row is one state in one sample in one split.
        # Average per split and sample first, to get points for scatter.
        values = subset.groupby(["split", "sample"])["kappa"].mean()
        ax.bar(x, values.mean(), 0.66, **_bar_style(arm))
        utils.scatter_points(ax, x, values, jitter=0.08, size=9)
        ax.annotate(f"{values.mean():.3f}", (x, float(values.mean())),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=6)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([label for _, _, label in series], rotation=45, ha="right", fontsize=7)
    _style(ax, f"{SCENARIO_LABEL[scenario].capitalize()} vs its own refit",
           "Mean per-state Kappa")
    ax.set_ylim(0, 1.2)
    _legend(ax, scenarios=(scenario,))


def _stability_panel(ax, stability, metric, domain, scenario):
    """Mean per-state agreement (Kappa) against the refit, averaged over all states."""
    series = comparison_series(scenario)
    subset = stability[(stability["domain"] == domain) &
                       (stability["scenario"] == scenario)]
    for x, (comparison, arm, label) in enumerate(series):
        values = subset[subset["comparison"] == comparison][metric]
        if values.empty:
            continue
        ax.bar(x, float(values.mean()), 0.66, **_bar_style(arm))
        utils.scatter_points(ax, x, values, jitter=0.08, size=9)
        ax.annotate(f"{values.mean():.3f}", (x, float(values.mean())),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=6)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([label for _, _, label in series], rotation=45, ha="right", fontsize=7)
    domain_label = "FULL" if domain == utils.FULL else "NOQH"
    _style(ax, f"{SCENARIO_LABEL[scenario].capitalize()} vs its own refit ({domain_label})",
           f"{metric.capitalize()}, mean over splits")
    ax.set_ylim(0, 1.2)
    _legend(ax, scenarios=(scenario,))


def _state_order(state_jaccard):
    """The reference states the tables carry, in the reference's own order."""
    present = set(state_jaccard["state"])
    return [state for state in pat.REFERENCE_STATES if state in present]


def _state_groups(ax, order, labels=True, label_y=-0.30):
    """Separators and, on the labelled axis, names for the functional groups."""
    edge = 0
    for label, members in STATE_GROUPS:
        width = sum(1 for state in members if state in order)
        if not width:
            continue
        edge += width
        if edge < len(order):
            ax.axvline(edge - 0.5, color="#CCCCCC", linewidth=0.7, zorder=0)
        if labels:
            ax.annotate(label, (edge - width / 2 - 0.5, label_y),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="top", fontsize=7, color="#777777")


def _state_panel(ax, state_jaccard, scenario="init", xlabels=True):
    """Per-state agreement of one scenario with its own refit, in one vocabulary."""
    if state_jaccard.empty:
        return
    series = comparison_series(scenario)
    order = _state_order(state_jaccard)
    width = 0.8 / len(series)
    offsets = _series_offsets(len(series), width)
    subsets = {comparison: state_jaccard[state_jaccard["comparison"] == comparison]
               for comparison, _, _ in series}
    if not any(len(subset) for subset in subsets.values()):
        return
    full = int(state_jaccard.groupby(["comparison", "state"]).size().max())
    for offset, (comparison, arm, label) in zip(offsets, series):
        subset = subsets[comparison]
        values = [subset[subset["state"] == state]["kappa"] for state in order]
        means = [float(v.mean()) if v.notna().any() else np.nan for v in values]
        ax.bar(np.arange(len(order)) + offset, means, width, label=label,
               **_bar_style(arm))
        for x, sample in enumerate(values):
            utils.scatter_points(ax, x + offset, sample, jitter=0.05, size=3,
                                 alpha=0.5)
            if len(sample) == 0:
                ax.plot([x + offset - width / 2.6, x + offset + width / 2.6],
                        [0.035, 0.035], color="#888888", linewidth=1.2,
                        solid_capstyle="butt")
            elif len(sample) < full:
                ax.annotate(f"{len(sample)}/{full}", (x + offset, 0.06),
                            ha="center", fontsize=4.5, color="#555555",
                            rotation=90)
    drawn = pd.concat(subsets.values()) if subsets else state_jaccard
    missing = drawn.groupby("state")["recovered"].any()
    missing = missing[~missing].index
    top = 1.2 if len(missing) == 0 else 1.55
    for index, state in enumerate(missing):
        if state not in order:
            continue
        x = order.index(state)
        ax.axvspan(x - 0.45, x + 0.45, color="#B00020", alpha=0.06, zorder=0)
        ax.text(x, 1.22 + 0.16 * (index % 2), "no arm puts\nany bin here",
                ha="center", va="top", fontsize=6, color="#B00020")
    ax.set_xticks(range(len(order)))
    if xlabels:
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6.5)
    else:
        ax.set_xticklabels([])
    _state_groups(ax, order, labels=xlabels)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, top)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _style(ax, f"{SCENARIO_LABEL[scenario].capitalize()} vs its own refit",
           "Kappa")


def _state_figure(fig, state_jaccard, scenarios=None):
    """One or both scenarios against the refit, one row each, on a shared axis.

    The Kappa is shown for every state matched onto the ENCODE reference
    vocabulary.
    """
    if scenarios is None:
        scenarios = [s for s in ARMS_OF_SCENARIO
                     if state_jaccard["comparison"].isin(
                         [comparison_key(m, s) for m in PLOT_METHODS]).any()]
    axes = np.atleast_1d(fig.subplots(len(scenarios), 1, sharex=True))
    for index, (ax, scenario) in enumerate(zip(axes, scenarios)):
        _state_panel(ax, state_jaccard, scenario,
                     xlabels=index == len(scenarios) - 1)
    fig.subplots_adjust(top=0.88 if len(scenarios) > 1 else 0.82,
                        bottom=0.16 if len(scenarios) > 1 else 0.22,
                        left=0.05, right=0.99, hspace=0.22)
    fig.suptitle("Per-state agreement (Kappa) against each arm's own refit, in the "
                 "ENCODE reference vocabulary (every arm matched onto it)",
                 y=0.985, **utils.TITLE_STYLE)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.955 if len(scenarios) > 1 else 0.91),
               ncol=len(labels), handlelength=1.6, columnspacing=1.2)






