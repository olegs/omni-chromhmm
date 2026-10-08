# Chromatin State Extraction

## 1. Objective

Every method learns a fixed number of chromatin states from the same histone marks of a sample and writes a segmentation: each genomic bin is labelled with one state ($E_1, \dots, E_K$). The methods differ in how a bin is described (read-depth binarization or peak calls) and in the model that groups the bins. The labels are then mapped onto a reference model by `match.md`, so the methods can be compared state by state.

## 2. Methods

### 2.1. Binarization

A bin of mark $m$ is $1$ when the mark is present there, else $0$; a bin is the vector $x \in \{0, 1\}^M$ over the $M$ marks.

- **ChromHMM Default.** `ChromHMM BinarizeBam` on the read alignments, against controls when the dataset has them (`rules/chromhmm.smk`).
- **Peak calls.** A bin is $1$ when it overlaps a peak of OmniPeak, HOMER or MACS2 for that mark (`binarize_peaks()` in `scripts/rules/peaks_segmentation.py`). Each caller has its own bin size, set in the config. An empty peak file is an all-zero mark, not a missing one.

### 2.2. Models

- **ChromHMM** (`LearnModel`): a multivariate Bernoulli HMM. States have per-mark emission probabilities, and transitions between neighbouring bins make the labels spatially coherent.
- **K-means** (`*_kmeans_states.bed`): clusters the binary vectors of the bins; a state is a centroid, and every bin takes the nearest one.
- **BMM** (`*_bmm_states.bed`, `--mixture`): a Bernoulli mixture model, i.e. ChromHMM's emission model without transitions,
  $$p(x) = \sum_k w_k \prod_m p_{km}^{x_m} (1 - p_{km})^{1 - x_m}$$
  fitted by EM (`BernoulliMixture` in `scripts/analysis/bernoulli.py`). Restarts are seeded from K-means centroids shrunk towards the pooled mark frequencies, with mixing proportions taken from the cluster sizes, and the restart with the best likelihood wins. Every bin takes the state of highest posterior.

K-means and BMM run on every peak caller (`rules/kmeans.smk`), ChromHMM on the read binarization.

### 2.3. Pattern Table

$M$ binary marks allow at most $2^M$ distinct bin vectors, while a genome holds millions of bins. Both models see the bins only through sums over them, so a fit on the table of distinct patterns weighted by their bin counts (`pattern_histogram()`) is the same fit at a cost independent of genome size. Labelling is a lookup: each pattern gets one state, and every bin reads its state off its pattern (`track_labels()`). Both run block by block, so the codes of the whole genome are never held at once.

A model cannot realize more states than there are distinct patterns. When a degenerate binarization shows fewer patterns than requested states (several empty marks), the mixture is fitted with as many states as patterns (`mixture_states()`), and the log carries a warning.

### 2.4. Joint Models

A joint model is learned once over several samples and segments each of them: the replicates of an ENCODE or SAGAconf dataset, or the epigenomes of epi1000 (`scripts/rules/joint_peaks_segmentation.py`; joint ChromHMM concatenates the replicate binarizations). Pattern counts are summed over the samples, so all of them share one state space, and one shared mapping names it (`match.md` §2.6).

## 3. Rationale

### 3.1. Peaks as Binarization

The peak caller sets which bins carry a mark; the clustering only groups the resulting patterns. Using the same model on OmniPeak, HOMER and MACS2 therefore isolates the effect of the peak caller, and ChromHMM Default is the reference point for the read-depth route.

### 3.2. K-means and BMM

K-means is the simplest grouping of binary patterns and the baseline. BMM is the likelihood model for binary data: its parameters are per-state mark probabilities, the same kind of emissions ChromHMM reports, so the two are directly comparable and matched on the same emission scale. Leaving out transitions isolates what ChromHMM's HMM adds over the emissions alone.

### 3.3. A Bin's Own Marks Only

No neighbouring-bin context is added to the observation. Peak calls are already contiguous intervals, so a bin's neighbours nearly always repeat its own marks and carry little new information. What they do add sits at peak boundaries, where a window can only move a label at the edge of a peak. That fragments segments rather than lengthening them, while the pattern space and the fit cost grow with the window width. Spatial coherence is left to models that have transitions (ChromHMM).

## 4. Alternatives Not Used

- **Window features** (previous, own and next bin as one observation; formerly BMM3 and KMeans3). Same endpoints as a bin's own marks at several times the cost, and shorter segments.
- **Fitting on the bins directly.** The same fit as on the pattern table, at a cost that grows with the genome and without a cheaper result.
- **Random EM initialisation.** Seeding from shrunk K-means centroids lets EM start from a sensible partition without inheriting it: raw centroids are hard $0/1$, which turns the first E-step into the K-means assignment.
