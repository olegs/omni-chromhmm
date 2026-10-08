#!/usr/bin/env python3
"""Weighted Bernoulli mixture over the binary patterns of a binarization.

A binarization of m marks has at most 2^m distinct rows, so a fit here is
carried on the table of distinct patterns weighted by their bin counts: both EM
steps see the rows only through weight-weighted sums, so the fit on the
collapsed table is the same fit at a cost independent of the number of bins.

The pipeline's segmentations count a genome-wide track into that table with
pattern_histogram() and annotate it with track_labels(), block by block, so
neither ever holds the pattern codes of the whole track at once.
"""

import numpy as np
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

# A binarization of m marks has at most 2^m distinct rows, so duplicate rows
# can be folded into weighted patterns; above this many marks the table of
# patterns stops being the cheaper representation.
MAX_PATTERN_MARKS = 24

# Rows per block when the whole pattern table is walked; blocking keeps the
# temporaries of a wide table out of the way.
PATTERN_BLOCK = 1 << 15

# Bins per block when a genome-wide track is counted or annotated.
TRACK_BLOCK = 1 << 22

# Widest pattern space an array may span: 2^24 counters are 134 MB of int64,
# and one bit more doubles that.
MAX_DENSE_PATTERN_BITS = 24

# How much of the pooled mean is mixed into a K-means centroid before EM sees
# it.  A raw centroid is hard 0/1 on most features, and clipped to `floor` the
# component it defines is so nearly deterministic that the first E-step is a
# hard assignment: EM inherits the partition K-means guessed and cannot leave
# it.  Shrinking the centroids keeps the seeding soft enough for EM to move.
INIT_SHRINK = 0.1


# --- the pattern representation -------------------------------------------

def n_pattern_features(n_patterns):
    """Features behind a pattern histogram of this length: 2^m -> m."""
    n_features = int(n_patterns).bit_length() - 1
    if 1 << n_features != n_patterns:
        raise ValueError(f"{n_patterns} patterns is not a power of two")
    return n_features


def pattern_rows(n_features, codes=None):
    """The binary rows of the given pattern codes (the whole table by default)."""
    if codes is None:
        _check_dense(n_features, "enumerating every pattern")
        codes = np.arange(1 << n_features)
    # numpy refuses to shift a uint64 by a signed amount, so both sides are
    # made unsigned here.
    codes = np.asarray(codes, dtype=np.uint64)
    shifts = np.arange(n_features, dtype=np.uint64)
    return ((codes[:, np.newaxis] >> shifts) & np.uint64(1)).astype(np.float64)


def _check_dense(n_features, what):
    """Refuse to span a pattern space that does not fit in an array.

    The dense forms below are indexed by pattern code, so their size is set by
    the vocabulary and not by the track.
    """
    if n_features > MAX_DENSE_PATTERN_BITS:
        raise ValueError(
            f"{what} needs 2^{n_features} entries, over the "
            f"2^{MAX_DENSE_PATTERN_BITS} an array may span")


def _worth_collapsing(n_samples, n_features):
    """Whether folding rows into a pattern table pays off: the table of counters
    must not outweigh the matrix of rows it replaces."""
    return (0 < n_features <= MAX_PATTERN_MARKS
            and (1 << n_features) <= n_samples * n_features)


def to_pattern_codes(X):
    """Pattern index of every row of a binary matrix."""
    X = np.asarray(X)
    codes = np.zeros(len(X), dtype=np.int64)
    for j in range(X.shape[1]):
        codes |= (X[:, j] != 0).astype(np.int64) << j
    return codes


def _pattern_codes(X):
    """Pattern index of every row, or None when the pattern table would not pay
    off - too many features, too few rows, or non-binary rows."""
    n_samples, n_features = X.shape
    if not _worth_collapsing(n_samples, n_features):
        return None
    if not np.array_equal(X != 0, X):
        return None
    return to_pattern_codes(X)


def _collapse(X, sample_weight):
    """Fold duplicate binary rows into unique patterns with summed weights.

    Both EM steps see the rows only through weight-weighted sums, so the fit on
    the collapsed table is the same fit - and a genome-wide binary matrix
    collapses to at most 2^n_features rows.  Returns None when the rows are not
    binary or when collapsing them would not pay off.
    """
    codes = _pattern_codes(X)
    if codes is None:
        return None
    mass = np.bincount(codes, weights=sample_weight, minlength=1 << X.shape[1])
    present = np.flatnonzero(mass)
    return pattern_rows(X.shape[1], present), mass[present]


# --- genome-wide tracks, block by block -----------------------------------

def pattern_histogram(X, block=TRACK_BLOCK):
    """Bins per pattern of a binary matrix - the mixture's input.

    Counted block by block, so the pattern codes of the whole track are never
    held at once.
    """
    X = np.asarray(X)
    n_features = X.shape[1]
    _check_dense(n_features, f"a histogram of {n_features} marks")
    total = np.zeros(1 << n_features, dtype=np.int64)
    for start in range(0, len(X), block):
        total += np.bincount(to_pattern_codes(X[start:start + block]),
                             minlength=total.size)
    return total


def track_labels(probs, weights, X, block=TRACK_BLOCK, dtype=np.int8):
    """State of every bin of a binary matrix, through the lookup table of the
    fitted parameters, block by block."""
    X = np.asarray(X)
    lut = lookup_table(probs, weights, dtype=dtype)
    out = np.empty(len(X), dtype=dtype)
    for start in range(0, len(X), block):
        stop = min(start + block, len(X))
        out[start:stop] = lut[to_pattern_codes(X[start:stop])]
    return out


# --- the mixture -----------------------------------------------------------

def _logit_offset(probs, weights):
    """The (logit, offset) form of the mixture's log density.

    x log p + (1-x) log(1-p) + log w  =  x . (log p - log(1-p))
                                         + [sum log(1-p) + log w]

    which is one matrix product per E-step instead of two, and never
    materialises 1 - X.
    """
    probs = np.asarray(probs, dtype=np.float64)
    log_zero = np.log1p(-probs)
    return np.log(probs) - log_zero, log_zero.sum(axis=1) + np.log(weights)


def _log_weighted_prob(X, probs, weights):
    """Log of the joint p(x, component) for every row and component."""
    logit, offset = _logit_offset(probs, weights)
    return X @ logit.T + offset


def _logsumexp(a):
    """logsumexp over the last axis, shift-stabilized."""
    top = a.max(axis=1, keepdims=True)
    return top[:, 0] + np.log(np.exp(a - top).sum(axis=1))


def _responsibilities(X, probs, weights):
    """Normalised responsibilities and the per-row log-likelihood.

    The EM step wants them in linear space, so they are normalised in place -
    which also skips the second exponential the log-space form needs.
    """
    resp = _log_weighted_prob(X, probs, weights)
    top = resp.max(axis=1, keepdims=True)
    resp -= top
    np.exp(resp, out=resp)
    total = resp.sum(axis=1, keepdims=True)
    resp /= total
    return resp, top[:, 0] + np.log(total[:, 0])


class BernoulliMixture(BaseEstimator, DensityMixin):
    """Bernoulli Mixture Model.

    Binary input is fitted on its table of distinct patterns, which makes the
    cost of an EM iteration independent of the number of samples.

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.
    tol : float, default=1e-10
        The convergence threshold.
    max_iter : int, default=1000
        The number of EM iterations to perform.
    n_init : int, default=10
        The number of initializations to perform.
    random_state : int, RandomState instance or None, default=None
    floor : float, default=1e-6
        Floor for probabilities to avoid log(0).
    shrink : float, default=INIT_SHRINK
        How much of the pooled mean is mixed into the K-means centroids the
        restarts start from.
    """
    def __init__(self, n_components=1, tol=1e-10, max_iter=1000,
                 n_init=10, random_state=None, floor=1e-6,
                 shrink=INIT_SHRINK):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.floor = floor
        self.shrink = shrink

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        sample_weight : array-like of shape (n_samples,), default=None
        """
        X = np.asarray(X)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        else:
            sample_weight = np.asarray(sample_weight)
            
        random_state = check_random_state(self.random_state)
        total_weight = np.sum(sample_weight)

        # Weighted K-means over the patterns is the same fit as K-means over
        # the rows they stand for, so the seeding runs on the collapsed table
        # too rather than on every row.
        collapsed = _collapse(X, sample_weight)
        em_x, em_w = collapsed if collapsed is not None else (X, sample_weight)

        best_loglik = -np.inf

        for _ in range(self.n_init):
            # Initialization
            means, weights = self._initialize_means(em_x, em_w, random_state)

            loglik = -np.inf
            for i in range(self.max_iter):
                prev_loglik = loglik
                means, weights, loglik = self._em_step(em_x, em_w, means, weights)
                
                if i > 0 and abs(loglik - prev_loglik) < self.tol:
                    break
            
            if loglik > best_loglik:
                best_loglik = loglik
                self.means_ = means
                self.weights_ = weights
                self.lower_bound_ = loglik / total_weight
                
        self.labels_ = self.predict(X)
        return self

    def fit_histogram(self, hist, n_features):
        """Fit the model on a pattern histogram.

        Parameters
        ----------
        hist : array-like of shape (2^n_features,)
            Counts of every binary pattern.
        n_features : int
            Number of binary features.
        """
        counts = np.asarray(hist, dtype=np.float64)
        present = np.flatnonzero(counts)
        return self.fit_patterns(present, counts[present], n_features)

    def fit_patterns(self, codes, counts, n_features):
        """Fit the model on a sparse pattern histogram.

        Parameters
        ----------
        codes : array-like of shape (n_patterns,)
            Pattern codes with a non-zero count, as the non-zero entries of a
            histogram.
        counts : array-like of shape (n_patterns,)
            Bins carrying each of those patterns.
        n_features : int
            Number of binary features.
        """
        x = pattern_rows(n_features, codes)
        w = np.asarray(counts, dtype=np.float64)

        random_state = check_random_state(self.random_state)
        total_weight = np.sum(w)
        
        best_loglik = -np.inf
        for _ in range(self.n_init):
            means, weights = self._initialize_means(x, w, random_state)

            loglik = -np.inf
            for i in range(self.max_iter):
                prev_loglik = loglik
                means, weights, loglik = self._em_step(x, w, means, weights)
                
                if i > 0 and abs(loglik - prev_loglik) < self.tol:
                    break
                    
            if loglik > best_loglik:
                best_loglik = loglik
                self.means_ = means
                self.weights_ = weights
                self.lower_bound_ = loglik / total_weight
                
        return self.means_, self.weights_, self.lower_bound_

    def _initialize_means(self, X, sample_weight, random_state):
        """One restart's starting point: K-means centroids shrunk toward the
        pooled mean, and the mixing proportions of the clusters they came from.

        Both departures from the plain centroids matter.  Unshrunk they are
        hard 0/1 on most features, which at `floor` makes every component
        deterministic and the first E-step a hard assignment - EM then only
        ever polishes the partition K-means guessed.  And the clusters are
        wildly unequal in mass, so uniform mixing proportions misstate the
        largest of them by an order of magnitude before EM has seen a row.
        """
        model = KMeans(n_clusters=self.n_components, n_init=1,
                       random_state=random_state)
        model.fit(X, sample_weight=sample_weight)
        pooled = sample_weight @ X / sample_weight.sum()
        centers = ((1 - self.shrink) * model.cluster_centers_
                   + self.shrink * pooled)
        mass = np.bincount(model.labels_, weights=sample_weight,
                           minlength=self.n_components) + 1e-12
        return (np.clip(centers, self.floor, 1 - self.floor), mass / mass.sum())

    def _e_step(self, X, means, weights):
        log_resp = _log_weighted_prob(X, means, weights)
        loglik_norm = _logsumexp(log_resp)
        log_resp -= loglik_norm[:, np.newaxis]
        return log_resp, loglik_norm

    def _em_step(self, X, sample_weight, means, weights):
        """One EM iteration, returning the log-likelihood of the parameters
        that went in (as the E-step computed it, before the update)."""
        weighted_resp, loglik_norm = _responsibilities(X, means, weights)
        weighted_resp *= sample_weight[:, np.newaxis]
        mass = weighted_resp.sum(axis=0) + 1e-12
        means = np.clip(np.dot(weighted_resp.T, X) / mass[:, np.newaxis],
                        self.floor, 1 - self.floor)
        return means, mass / mass.sum(), float(sample_weight @ loglik_norm)

    def predict_proba(self, X):
        X = np.asarray(X)
        log_resp, _ = self._e_step(X, self.means_, self.weights_)
        return np.exp(log_resp, out=log_resp)

    def predict(self, X):
        X = np.asarray(X)
        codes = _pattern_codes(X)
        if codes is not None:
            # Label the patterns once and look the rows up.
            return lookup_table(self.means_, self.weights_,
                                dtype=np.int64)[codes]
        return np.argmax(_log_weighted_prob(X, self.means_, self.weights_),
                         axis=1)

    def score(self, X, y=None, sample_weight=None):
        X = np.asarray(X)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        collapsed = _collapse(X, sample_weight)
        if collapsed is not None:
            X, sample_weight = collapsed
        loglik_norm = _logsumexp(_log_weighted_prob(X, self.means_,
                                                    self.weights_))
        return float(sample_weight @ loglik_norm)

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        bic : float
            The lower the better.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        # Number of parameters:
        # means: n_components * n_features
        # weights: n_components - 1
        n_params = self.n_components * (n_features + 1) - 1
        return -2 * self.score(X) + n_params * np.log(n_samples)

    def entropy(self, X, sample_weight=None):
        """Total entropy of the cluster assignments for X."""
        X = np.asarray(X)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        collapsed = _collapse(X, sample_weight)
        if collapsed is not None:
            X, sample_weight = collapsed

        log_resp, _ = self._e_step(X, self.means_, self.weights_)
        resp = np.exp(log_resp)
        return -np.sum(sample_weight[:, np.newaxis] * resp * log_resp)

    def icl(self, X):
        """Integrated completed likelihood for the current model on the input X."""
        return self.bic(X) + 2 * self.entropy(X)


def fit_bernoulli(hist, init=None, k=15, seed=42, n_init=10, iterations=300,
                  polish=200_000, chunk=5000, tol=1e-10, floor=1e-6):
    """Fit the mixture on a pattern histogram: 2^n_features bin counts in
    pattern order.  Returns (probs, weights, loglik)."""
    counts = np.asarray(hist, dtype=np.float64)
    present = np.flatnonzero(counts)
    x, w = pattern_rows(n_pattern_features(counts.size), present), counts[present]

    model = BernoulliMixture(
        n_components=k,
        tol=tol,
        max_iter=iterations,
        n_init=n_init,
        random_state=seed,
        floor=floor
    )
    
    if init is not None and not isinstance(init, str):
        model.means_ = np.clip(np.asarray(init[0], dtype=float), floor, 1 - floor)
        model.weights_ = np.asarray(init[1], dtype=float)
    else:
        model.fit(x, sample_weight=w)
    
    # Polish step
    if polish > 0:
        means = model.means_
        weights = model.weights_
        loglik = -np.inf
        for _ in range(max(1, polish // chunk)):
            prev_labels = lookup_table(means, weights)
            
            for _ in range(chunk):
                prev_loglik = loglik
                means, weights, loglik = model._em_step(x, w, means, weights)
                if abs(loglik - prev_loglik) < tol:
                    break
            
            model.means_ = means
            model.weights_ = weights
            if np.array_equal(lookup_table(means, weights), prev_labels):
                break
                
    return model.means_, model.weights_, float(model.score(x, sample_weight=w))


def log_likelihood(counts, probs, weights):
    """sum_p C(p) log p(p) - the full-bin log-likelihood, from the counts."""
    counts = np.asarray(counts, dtype=np.float64)
    present = np.flatnonzero(counts)
    rows = pattern_rows(n_pattern_features(counts.size), present)
    return float(counts[present]
                 @ _logsumexp(_log_weighted_prob(rows, probs, weights)))


def lookup_table(probs, weights, dtype=np.int8):
    """State of every pattern: the whole annotation model.

    The table is walked in blocks, which keeps a wide model from materialising
    its rows and their scores all at once.
    """
    probs = np.asarray(probs)
    _check_dense(probs.shape[1], f"a lookup table over {probs.shape[1]} features")
    logit, offset = _logit_offset(probs, weights)
    out = np.empty(1 << probs.shape[1], dtype=dtype)
    for start in range(0, out.size, PATTERN_BLOCK):
        stop = min(start + PATTERN_BLOCK, out.size)
        rows = pattern_rows(probs.shape[1], np.arange(start, stop))
        out[start:stop] = np.argmax(rows @ logit.T + offset, axis=1)
    return out
