#!/usr/bin/env python3
"""Weighted Bernoulli mixture over the binary patterns of a peak-caller binarization."""

import os
import pickle
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

import bernoulli as bern

THREADS = int(os.environ.get("PATTERNS_THREADS", "4"))
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(THREADS))

# The state matching of the whole project lives in scripts/rules/match.py; put
# it on the path, but leave the import to the functions that match (see the
# label alignment section).
_rules_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rules"))
if _rules_dir not in sys.path:
    sys.path.insert(0, _rules_dir)

# --- Init analysis settings ------------------------------------------------

MARKS = ("H3K4me3", "H3K4me1", "H3K36me3", "H3K9me3", "H3K27me3", "H3K27ac")
N_STATES = 15
SEED = 42
# Random restarts of a fresh fit.  A restart of the spatial mixture costs
# ~0.07 s, and at 10 of them the best-known optimum of a 15-state chr1 fit is
# essentially never reached: the restarts span 4.7% of the log-likelihood.
N_INIT = 50
KMEANS_N_INIT = 3000  # restarts of the K-means seeding, where it is stable
N_PATTERNS = 1 << len(MARKS)
SPATIAL_BINS = bern.SPATIAL_BINS       # previous, current and next bin
N_SPATIAL_MARKS = len(MARKS) * SPATIAL_BINS
N_PATTERNS_SPATIAL = 1 << N_SPATIAL_MARKS

# Chromosomes no mark is called on; keeping them would only add all-zero bins.
EXCLUDED_CHROMS = ("chrY", "chrM", "chrEBV")


# --- progress on stdout ----------------------------------------------------
#
# The scripts log what they are doing instead of drawing progress bars: the
# lines survive in a notebook and in a terminal log, they carry the wall time
# of every step, and only the outermost loop over splits still draws a bar.

def log(message, indent=1):
    """One line of progress on stdout, flushed so it appears as it happens."""
    print("  " * indent + message, flush=True)


@contextmanager
def step(message, indent=1):
    """Log one step of work as `message ... 1.2s`.

    The body may append words to the yielded list to have them shown after the
    time - which is how a step that was read back from disk says so.
    """
    print("  " * indent + message + " ... ", end="", flush=True)
    started = time.perf_counter()
    note = []
    yield note
    print(f"{time.perf_counter() - started:.1f}s"
          + (f" ({', '.join(note)})" if note else ""), flush=True)


# --- genome mask -----------------------------------------------------------

class Mask:
    """Genomic mask: chromosomes binned at `bin_size` bp."""

    def __init__(self, chromsizes, bin_size, excluded=EXCLUDED_CHROMS):
        sizes = pd.read_csv(chromsizes, sep="\t", header=None,
                            names=["chrom", "size"])
        sizes = sizes[~sizes["chrom"].str.contains("_")]
        sizes = sizes[~sizes["chrom"].isin(excluded)]
        self.bin = bin_size
        self.chroms = sizes["chrom"].tolist()
        self.sizes = dict(zip(sizes["chrom"], sizes["size"]))
        self.nbins = {c: (self.sizes[c] + bin_size - 1) // bin_size
                      for c in self.chroms}
        self.offset = {}
        off = 0
        for c in self.chroms:
            self.offset[c] = off
            off += self.nbins[c]
        self.total_bins = off

    def signature(self):
        return {"bin": self.bin, "chroms": len(self.chroms),
                "total_bins": self.total_bins}

    def slices(self):
        """(chrom, start_bin, end_bin) of every chromosome, in index order."""
        return [(c, self.offset[c], self.offset[c] + self.nbins[c])
                for c in self.chroms]


# --- preprocessing: peaks -> pattern codes ---------------------------------

def _peak_intervals(path, mask):
    """(chrom, start_bin, end_bin) arrays of a peak file, clipped to the mask."""
    df = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[0, 1, 2],
                     names=["chrom", "start", "end"],
                     dtype={"chrom": str, "start": np.int64, "end": np.int64})
    df = df[df["chrom"].isin(mask.nbins)]
    for chrom, group in df.groupby("chrom", sort=False):
        n = mask.nbins[chrom]
        starts = np.clip(group["start"].to_numpy() // mask.bin, 0, n)
        ends = np.clip(-(-group["end"].to_numpy() // mask.bin), 0, n)
        keep = ends > starts
        yield chrom, starts[keep], ends[keep]


def pattern_codes(sample_dir, mask, marks, peak_path):
    """uint8 pattern code per masked bin: bit i is set when marks[i] has a peak."""
    codes = np.zeros(mask.total_bins, dtype=np.uint8)
    covered = np.empty(mask.total_bins, dtype=bool)
    delta = np.zeros(mask.total_bins + 1, dtype=np.int32)
    for i, mark in enumerate(marks):
        path = peak_path(sample_dir, mark)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        delta[:] = 0
        for chrom, starts, ends in _peak_intervals(path, mask):
            off = mask.offset[chrom]
            np.add.at(delta, starts + off, 1)
            np.add.at(delta, ends + off, -1)
        np.cumsum(delta[:-1], out=delta[:-1])
        np.greater(delta[:-1], 0, out=covered)
        codes |= (covered.view(np.uint8) << np.uint8(i))
    return codes


# --- per-sample cache ------------------------------------------------------

class Cache:
    """Per-sample preprocessing cache, shared by every arm."""

    def __init__(self, root, data_dir, mask, marks, peak_path):
        self.root = root
        self.data_dir = data_dir
        self.mask = mask
        self.marks = tuple(marks)
        self.peak_path = peak_path
        self.codes_dir = os.path.join(root, "codes")
        os.makedirs(self.codes_dir, exist_ok=True)

    def codes_path(self, sample):
        return os.path.join(self.codes_dir, f"{sample}_codes.npz")

    def has(self, sample):
        return os.path.exists(self.codes_path(sample))

    def process(self, sample):
        """Peak calling output -> cached pattern track. Returns seconds."""
        started = time.perf_counter()
        codes = pattern_codes(os.path.join(self.data_dir, sample), self.mask,
                              marks=self.marks, peak_path=self.peak_path)
        np.savez_compressed(self.codes_path(sample), codes=codes)
        return time.perf_counter() - started

    def codes(self, sample):
        with np.load(self.codes_path(sample)) as data:
            return data["codes"]


# --- weighted Bernoulli mixture on the pattern counts ----------------------

def _slices(mask, chroms=None):
    """The mask's chromosome blocks, restricted to `chroms` when given."""
    return [s for s in mask.slices() if chroms is None or s[0] in chroms]


def spatial_histogram(codes, mask, chroms=None, bins=SPATIAL_BINS):
    """Spatial pattern counts over `chroms` (the whole mask by default).

    A spatial pattern is a bin widened to the marks of its previous, own and
    next bin (bins=3) or previous and own bin (bins=2).
    """
    return bern.spatial_histogram(codes, _slices(mask, chroms), len(MARKS),
                                  bins=bins)


def spatial_labels(lut, codes, mask, chroms=None, bins=SPATIAL_BINS):
    """Annotate a track through a spatial lookup table: `lut[spatial code]`."""
    return bern.spatial_labels(lut, codes, _slices(mask, chroms), len(MARKS),
                               bins=bins)


def fit_bernoulli(hist, init=None, k=N_STATES, seed=SEED, n_init=N_INIT):
    """Weighted Bernoulli-mixture EM over the pattern table.

    `hist` decides the vocabulary: 2^6 counts is the plain mark pattern of a
    bin, 2^18 the spatial pattern of spatial_histogram().  Everything the EM
    itself is tuned by - iterations, polish, tolerance, floor - stays at the
    bernoulli defaults; only the vocabulary and the restarts differ here.
    """
    return bern.fit_bernoulli(hist, init=init, k=k, seed=seed, n_init=n_init)


def fit_kmeans(hist, init=None, k=N_STATES, seed=SEED, n_init=KMEANS_N_INIT):
    """Weighted K-means over the pattern table, shaped like fit_bernoulli()."""
    from sklearn.cluster import KMeans

    counts = np.asarray(hist, dtype=np.float64)
    present = np.flatnonzero(counts)
    x = bern.pattern_rows(bern.n_pattern_features(counts.size), present)
    w = counts[present]
    warm = init is not None and not isinstance(init, str)
    model = KMeans(n_clusters=k, random_state=seed,
                   init=np.asarray(init[0], dtype=np.float64) if warm else "k-means++",
                   n_init=1 if warm else n_init)
    labels = model.fit_predict(x, sample_weight=w)
    mass = np.array([w[labels == j].sum() for j in range(k)]) + 1e-12
    return (np.clip(model.cluster_centers_, 1e-6, 1 - 1e-6), mass / mass.sum(),
            float(model.inertia_))


def kmeans_lookup_table(centroids):
    """Nearest-centroid state of every pattern."""
    centroids = np.asarray(centroids)
    table = bern.pattern_rows(centroids.shape[1])
    distance = ((table ** 2).sum(axis=1)[:, None] - 2 * table @ centroids.T
                + (centroids ** 2).sum(axis=1)[None, :])
    return distance.argmin(axis=1).astype(np.int8)


def lookup_table(probs, weights):
    """State of every pattern: the whole annotation model."""
    return bern.lookup_table(probs, weights)


# --- label alignment -------------------------------------------------------
#
# Every match here is match.best_mapping(), the matcher the rest of the project
# relabels segmentations with, and the agreement is match.agreement_metrics(),
# read under the metric names of utils. Both are imported where they are used
# rather than at the top of the module: measure() re-imports patterns in a
# fresh interpreter for every fit and reports that process's peak RSS as the
# cost of the fit, and match pulls matplotlib and seaborn in behind it - 50 MB
# on top of a pattern fit that peaks at 160. Fitting needs no matching.

def confusion_overlap(table):
    """A confusion table as the overlap dict match.py works on.

    The label arrays here are already binned, so a cell of the table is the
    overlap of two states in bins where match.py counts it in bp; every
    function over there is scale-free, so the two are interchangeable.  States
    go in under their row and column indices rather than their names: a fit
    can name two states alike - interpret_centroids() reads the names off the
    emissions - and a name-keyed dict would silently merge them.
    """
    table = np.asarray(table, dtype=np.float64)
    return {(i, j): table[i, j] for i in range(table.shape[0])
            for j in range(table.shape[1])}


def state_mapping(table):
    """Match the rows of a confusion table onto its columns, one to one.

    match.best_mapping(), the matcher the rest of the project runs on, read
    off a confusion table: the assignment maximising the total overlap of the
    matched pairs, so every column is spoken for and a row that matches
    nothing still takes one.  Returned as the partner column of every row,
    for indexing.
    """
    import match

    table = np.asarray(table, dtype=np.float64)
    rows, columns = list(range(table.shape[0])), list(range(table.shape[1]))
    mapping = match.best_mapping(confusion_overlap(table), rows, columns)
    return np.array([mapping[row] for row in rows])




# --- biological interpretation, fixed on the initial training fit ----------

def own_marks(values):
    """The bin's own mark columns of a fit, in MARKS order.

    A spatial centroid carries several bins side by side; the state is defined
    by the "current" one, the same one interpret_centroids() names it after.
    Works on a single row and on a states x marks matrix alike.
    """
    values = np.asarray(values)
    width = values.shape[-1]
    if width == N_SPATIAL_MARKS:        # 3 bins: prev, curr, next
        return values[..., len(MARKS):2 * len(MARKS)]
    return values


_RULES = (
    (("H3K4me3", "H3K27me3"), (), "TssBiv"),
    (("H3K4me3", "H3K27ac"), (), "Tss"),
    (("H3K4me3",), (), "TssFlnk"),
    (("H3K27ac", "H3K4me1"), (), "Enh"),
    (("H3K27ac",), (), "EnhA"),
    (("H3K4me1", "H3K36me3"), (), "EnhG"),
    (("H3K4me1", "H3K27me3"), (), "EnhBiv"),
    (("H3K4me1",), (), "EnhLo"),
    (("H3K36me3", "H3K9me3"), (), "ZNF"),
    (("H3K36me3",), (), "Tx"),
    (("H3K27me3", "H3K9me3"), (), "HetPC"),
    (("H3K27me3",), (), "ReprPC"),
    (("H3K9me3",), (), "Het"),
)

_VARIANTS = {
    "Tss": ("Tss", "TssFlnkU", "TssFlnkD"),
    "TssFlnk": ("TssFlnk", "TssFlnkU", "TssFlnkD"),
    "Enh": ("Enh", "Enh1", "Enh2"),
    "EnhA": ("EnhA1", "EnhA2"),
    "EnhG": ("EnhG", "EnhG1", "EnhG2"),
    "EnhLo": ("EnhLo", "EnhWk"),
    "Tx": ("Tx", "TxWk"),
    "ReprPC": ("ReprPC", "ReprPCWk"),
    "Het": ("Het", "HetWk"),
    "Quies": ("Quies", "Quies2", "Quies3"),
}

# The functional families, read on the reference states a fit was matched onto
# rather than on the names interpret_centroids() gives it: utils.PROMOTER_STATES,
# utils.TX_STATES and utils.ACTIVE_STATES, spelled out here so that importing
# this module costs a measured fit nothing (see the label alignment section).
PROMOTER_FAMILY = ("Tss", "TssFlnk", "TssFlnkU", "TssFlnkD")
TX_FAMILY = ("Tx", "TxWk")
ENHANCER_FAMILY = ("Enh", "Enh1", "Enh2", "EnhG", "EnhG1", "EnhG2", "EnhLo")

FAMILIES = {"promoter": PROMOTER_FAMILY, "transcribed": TX_FAMILY,
            "enhancer": ENHANCER_FAMILY,
            "active": PROMOTER_FAMILY + ENHANCER_FAMILY}


def _base_name(presence):
    """Base state name of a centroid's mark-presence pattern."""
    present = {m for m, flag in zip(MARKS, own_marks(presence)) if flag}
    for required, forbidden, name in _RULES:
        if present.issuperset(required) and not present.intersection(forbidden):
            return name
    return "Quies"


def interpret_centroids(centroids, threshold=0.5):
    """State names of a fit's clusters, from their emission profiles alone."""
    centroids = np.asarray(centroids, dtype=np.float64)
    presence = centroids >= threshold
    strength = centroids.sum(axis=1)
    bases = [_base_name(row) for row in presence]
    names = [None] * len(centroids)
    for base in sorted(set(bases)):
        members = [j for j in range(len(centroids)) if bases[j] == base]
        members.sort(key=lambda j: -strength[j])
        variants = _VARIANTS.get(base, (base,))
        for rank, j in enumerate(members):
            names[j] = variants[rank] if rank < len(variants) else f"{base}{rank + 1}"
    return names


# --- ENCODE 15-state reference vocabulary -----------------------------------

REFERENCE_STATES = ("Tss", "TssFlnk", "TssFlnkU", "TssFlnkD", "Enh1", "Enh2",
                    "EnhG1", "EnhG2", "Biv", "ReprPC", "Het", "ZNF/Rpts",
                    "Tx", "TxWk", "Quies")


def reference_labels(sample, mask, datasets, data_dir):
    """The sample's ENCODE markup as per-bin indices into REFERENCE_STATES."""
    import pandas as pd
    accession = datasets[sample].get("ref_chromhmm")
    path = os.path.join(data_dir, sample, f"{accession}_chromhmm.bed")
    if not accession or not os.path.exists(path):
        return None
    frame = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3],
                        names=["chrom", "start", "end", "state"],
                        dtype={"chrom": str}, comment="t")
    index = {name: i for i, name in enumerate(REFERENCE_STATES)}
    labels = np.full(mask.total_bins, -1, dtype=np.int8)
    frame = frame[frame["state"].isin(index)]
    for chrom, group in frame.groupby("chrom", sort=False):
        if chrom not in mask.nbins:
            continue
        offset, n = mask.offset[chrom], mask.nbins[chrom]
        starts = np.clip(group["start"].to_numpy() // mask.bin, 0, n)
        ends = np.clip(group["end"].to_numpy() // mask.bin, 0, n)
        states = group["state"].map(index).to_numpy()
        for start, end, state in zip(starts, ends, states):
            labels[offset + start:offset + end] = state
    return labels


# How a fit's states are read in the reference vocabulary: state_mapping(),
# i.e. match.best_mapping(), the same bijection every other comparison in the
# project is matched by.  Recorded in a run's settings so that tables written
# by an earlier matcher are not read as if they were these.
REFERENCE_MATCH = "match.best_mapping"


def reference_mapping(labels_by_sample, reference, samples, states=N_STATES):
    """Match of an arm's clusters onto the reference states.

    Pooled over `samples`, so one arm reads the same in all of them.  A
    bijection: a fit holding no second promoter still has a state assigned to
    the second promoter slot, and how well it fills it is what the endpoints
    then measure.
    """
    pooled = np.zeros((states, len(REFERENCE_STATES)), dtype=np.int64)
    for sample in samples:
        if reference.get(sample) is None:
            continue
        pooled += confusion_from_labels(labels_by_sample[sample], reference[sample],
                                        max(states, len(REFERENCE_STATES))
                                        )[:states, :len(REFERENCE_STATES)]
    return state_mapping(pooled)


def reference_families(mapping):
    """The states of each functional family, by what the reference matched.

    `mapping` is a reference_mapping() result: the reference state every state
    of a fit was assigned to.  Membership follows that assignment rather than
    the fit's own emissions, so a family means the same thing in every arm -
    the same convention the per-state agreement is already read in.  The
    matching is a bijection, so each family holds a fixed number of states and
    an arm short of, say, a second promoter still fills all four promoter
    slots; how well it fills them is what the endpoint then measures.
    """
    names = [REFERENCE_STATES[i] for i in mapping]
    return {key: [j for j, name in enumerate(names) if name in members]
            for key, members in FAMILIES.items()}


def reference_state_rows(labels_a, labels_b, map_a, map_b, sample, comparison):
    """Per-state Jaccard and Kappa of two annotations, both relabelled to the reference."""
    left, right = map_a[labels_a], map_b[labels_b]
    n = float(len(left))
    rows = []
    for index, state in enumerate(REFERENCE_STATES):
        in_left, in_right = left == index, right == index
        tp = float(np.logical_and(in_left, in_right).sum())
        l_sum, r_sum = float(in_left.sum()), float(in_right.sum())
        union = l_sum + r_sum - tp
        tn = n - union
        po = (tp + tn) / n
        pe = (l_sum * r_sum + (n - l_sum) * (n - r_sum)) / (n * n)
        # The two families the summary pivots on; the enhancers are read
        # through the endpoints rather than here, so they stay "other".
        family = ("promoter" if state in PROMOTER_FAMILY else
                  "transcribed" if state in TX_FAMILY else "other")
        recovered = bool(l_sum or r_sum)
        rows.append({"sample": sample, "comparison": comparison,
                     "state": state, "family": family,
                     "jaccard": tp / union if union > 0 else float("nan"),
                     "kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0 if recovered else float("nan"),
                     "recovered": recovered})
    return rows


# --- cross-sample agreement ------------------------------------------------

def confusion_from_labels(labels_a, labels_b, k=N_STATES):
    """State confusion of two label arrays, ignoring unlabelled bins (-1)."""
    keep = (labels_a >= 0) & (labels_b >= 0)
    index = labels_a[keep].astype(np.int64) * k + labels_b[keep].astype(np.int64)
    return np.bincount(index, minlength=k * k).reshape(k, k)


def noqh_states(names):
    """The quiescent / heterochromatin bulk, which the NOQH variant drops.

    utils.NOQH_STATES named by prefix: interpret_centroids() splits a base
    name into variants ("Quies2", "HetWk", "ReprPCWk"), and they all belong to
    the same background the exact set holds.
    """
    return {name for name in names
            if name.startswith(("Quies", "Het", "ReprPC")) or name == "ZNF"}


def cross_sample_metrics(table, names, exclude=()):
    """Mean per-state Jaccard and Cohen's kappa of one confusion table.

    match.agreement_metrics() over the table, which is how the ENCODE,
    SAGAconf and epi1000 notebooks read the same two numbers: kappa on the
    confusion restricted to the kept states, and the mean of the per-state
    Jaccards, whose union is the whole extent of a state - a bin an excluded
    state takes on the other side counts against it, rather than being
    dropped from the comparison altogether.
    """
    import match
    import utils

    table = np.asarray(table, dtype=np.float64)
    keep = [i for i, name in enumerate(names) if name not in exclude]
    dropped = [i for i, name in enumerate(names) if name in exclude]
    metrics = match.agreement_metrics(
        confusion_overlap(table),
        dict(enumerate(table.sum(axis=1))), dict(enumerate(table.sum(axis=0))),
        exclude=dropped)
    return {utils.JACCARD: metrics[utils.JACCARD],
            utils.KAPPA: metrics[utils.KAPPA],
            "bins": float(table[np.ix_(keep, keep)].sum()),
            "states": len(keep)}


# --- spatial structure -----------------------------------------------------

def segment_stats(labels, mask):
    """Segments and mean segment length of a labelling - how fragmented it is."""
    changes = 0
    covered = 0
    for _, lo, hi in mask.slices():
        piece = labels[lo:hi]
        if piece.size == 0:
            continue
        changes += int((piece[1:] != piece[:-1]).sum()) + 1
        covered += piece.size
    return {"segments": changes,
            "mean_segment_bp": covered * mask.bin / changes if changes else float("nan")}


# --- cost accounting -------------------------------------------------------

_RSS_UNIT = 1 if sys.platform == "darwin" else 1024   # ru_maxrss: bytes vs KiB
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def _child_main(path):
    """Run one measured operation in a fresh interpreter (see measure())."""
    with open(path, "rb") as handle:
        fn, args, kwargs = pickle.load(handle)
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        value = fn(*args, **kwargs)
        payload = (True, value,
                   {"inner_wall": time.perf_counter() - started,
                    "inner_cpu": time.process_time() - cpu_started})
    except BaseException:                            # noqa: BLE001 - sent to the parent
        import traceback
        payload = (False, traceback.format_exc(), {})
    with open(path + ".out", "wb") as handle:
        pickle.dump(payload, handle, protocol=4)


def measure(fn, *args, **kwargs):
    """Run `fn` in a fresh interpreter; return (result, cost)."""
    import tempfile
    handle, path = tempfile.mkstemp(suffix=".measure.pkl")
    with os.fdopen(handle, "wb") as channel:
        pickle.dump((fn, args, kwargs), channel, protocol=4)
    code = (f"import sys; sys.path.insert(0, {_MODULE_DIR!r}); "
            f"import patterns; patterns._child_main({path!r})")
    started = time.perf_counter()
    pid = os.posix_spawn(sys.executable, [sys.executable, "-c", code], os.environ)
    _, status, usage = os.wait4(pid, 0)
    wall = time.perf_counter() - started
    try:
        with open(path + ".out", "rb") as channel:
            ok, value, inner = pickle.load(channel)
    except FileNotFoundError:
        raise RuntimeError(f"measured child died, status {status}") from None
    finally:
        for leftover in (path, path + ".out"):
            if os.path.exists(leftover):
                os.remove(leftover)
    if not ok:
        raise RuntimeError(value)
    cpu = usage.ru_utime + usage.ru_stime
    peak = usage.ru_maxrss * _RSS_UNIT / 1024 ** 2
    return value, {"wall": wall, "cpu": cpu, "peak_mb": peak, **inner}


def fit_op(counts, init=None, seed=SEED, n_init=N_INIT):
    """fit_bernoulli() under a picklable name, for measure()."""
    return fit_bernoulli(counts, init=init, seed=seed, n_init=n_init)


def fit_kmeans_op(counts, init=None, seed=SEED, n_init=KMEANS_N_INIT):
    """fit_kmeans() under a picklable name, for measure()."""
    return fit_kmeans(counts, init=init, seed=seed, n_init=n_init)


# --- genomic annotations ---------------------------------------------------

def add_intervals(target, mask, chrom, starts, ends):
    """Mark every bin an interval touches, in place."""
    if chrom not in mask.nbins:
        return target
    n, off = mask.nbins[chrom], mask.offset[chrom]
    starts = np.clip(np.asarray(starts) // mask.bin, 0, n)
    ends = np.clip(-(-np.asarray(ends) // mask.bin), 0, n)
    for s, e in zip(starts, ends):
        if e > s:
            target[off + s:off + e] = True
    return target


def static_annotation(mask, path):
    """Bin mask of a ChromHMM COORDS bed.gz (sample-independent)."""
    frame = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2],
                        names=["chrom", "start", "end"], dtype={"chrom": str})
    out = np.zeros(mask.total_bins, dtype=bool)
    for chrom, group in frame.groupby("chrom", sort=False):
        add_intervals(out, mask, chrom, group["start"].to_numpy(),
                      group["end"].to_numpy())
    return out
