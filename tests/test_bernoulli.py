#!/usr/bin/env python3
"""Tests for the weighted Bernoulli mixture and its spatial pattern forms.

Every claim bernoulli.py makes about its own representation is a claim that two
computations agree: the collapsed fit is the fit on the rows, the sparse
histogram is the dense one, the blocked lookup is the unblocked one, the
logit/offset algebra is the Bernoulli log density.  So the tests here are
mostly equivalences, checked against a naive reference written out in full -
per-bin Python loops for the spatial codes, the explicit
`x log p + (1-x) log(1-p)` for the density, scipy for logsumexp.  The
references are slow and obviously right; the module is fast and has to match.

Runs under pytest, or standalone: `python tests/test_bernoulli.py`.
"""

import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "scripts", "analysis"))

import bernoulli as bern  # noqa: E402

SEED = 0


# --- naive references ------------------------------------------------------

def ref_spatial_codes(block, n_marks, bins=3):
    """block_spatial_codes() as a per-bin Python loop, in exact integers.

    Python ints do not wrap, so this also pins the layout independently of the
    uint32/uint64 the module picks.
    """
    codes = []
    for i in range(len(block)):
        if bins == 1:
            codes.append(int(block[i]))
            continue
        previous = int(block[i - 1]) if i > 0 else 0
        following = int(block[i + 1]) if i + 1 < len(block) else 0
        code = previous | (int(block[i]) << n_marks)
        if bins == 3:
            code |= following << 2 * n_marks
        codes.append(code)
    return codes


def ref_log_joint(X, probs, weights):
    """log p(x, component) written out, without the logit/offset rearrangement."""
    return (X @ np.log(probs).T + (1 - X) @ np.log1p(-probs).T
            + np.log(weights))


def ref_labels(codes, slices, n_marks, probs, weights, bins=3):
    """The spatial annotation, one bin at a time through the naive reference."""
    labels = np.zeros(len(codes), dtype=np.int64)
    for _, lo, hi in slices:
        for i, code in enumerate(ref_spatial_codes(codes[lo:hi], n_marks,
                                                   bins=bins)):
            row = np.array([(code >> b) & 1 for b in range(bins * n_marks)],
                           dtype=float)
            labels[lo + i] = np.argmax(ref_log_joint(row[np.newaxis],
                                                     probs, weights))
    return labels


def as_ints(codes):
    """A code array as exact Python ints, so a uint64 wrap cannot hide."""
    return [int(c) for c in codes]


# --- fixtures --------------------------------------------------------------

def random_binary(n_samples, n_features, density=0.3, seed=SEED):
    return (np.random.default_rng(seed).random((n_samples, n_features))
            < density).astype(float)


def random_params(n_components, n_features, seed=SEED):
    """Probabilities well inside (0, 1) and mixing proportions that sum to 1."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.05, 0.95, (n_components, n_features)),
            rng.dirichlet(np.ones(n_components)))


def peaky_codes(n_bins, n_marks, seed=SEED):
    """A track whose marks sit in contiguous runs, as real peak calls do.

    Chromatin marks are correlated and clustered, which is what keeps the
    spatial vocabulary small enough to fit sparsely; independent coin flips per
    bin would not exercise the same regime.
    """
    rng = np.random.default_rng(seed)
    codes = np.zeros(n_bins, dtype=np.int64)
    for mark in range(n_marks):
        coverage = 0.01 + 0.005 * mark
        column = np.zeros(n_bins, dtype=bool)
        n_peaks = max(1, int(n_bins * coverage / 15))
        for start, length in zip(rng.integers(0, n_bins - 30, n_peaks),
                                 rng.integers(5, 26, n_peaks)):
            column[start:start + length] = True
        codes |= column.astype(np.int64) << mark
    return codes


def even_slices(n_bins, n_slices):
    """`n_slices` contiguous blocks covering [0, n_bins), as chromosomes are."""
    edges = np.linspace(0, n_bins, n_slices + 1).astype(int)
    return [(f"chr{i + 1}", int(lo), int(hi))
            for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]


# --- the pattern representation --------------------------------------------

def test_n_pattern_features_inverts_a_power_of_two():
    for n_features in range(1, 21):
        assert bern.n_pattern_features(1 << n_features) == n_features


def test_n_pattern_features_rejects_a_non_power_of_two():
    for n_patterns in (0, 3, 100, (1 << 10) + 1):
        try:
            bern.n_pattern_features(n_patterns)
        except ValueError:
            continue
        raise AssertionError(f"{n_patterns} patterns was accepted")


def test_to_pattern_codes_is_little_endian_over_the_marks():
    X = random_binary(500, 7)
    expected = (X.astype(np.int64) * (1 << np.arange(7))).sum(axis=1)
    assert np.array_equal(bern.to_pattern_codes(X), expected)


def test_to_pattern_codes_reads_any_non_zero_as_a_one():
    X = np.array([[0.0, 2.0, -1.0, 0.0]])
    assert bern.to_pattern_codes(X)[0] == 0b0110


def test_pattern_rows_inverts_to_pattern_codes():
    for n_features in (1, 4, 10):
        table = bern.pattern_rows(n_features)
        assert table.shape == (1 << n_features, n_features)
        assert np.array_equal(bern.to_pattern_codes(table),
                              np.arange(1 << n_features))


def test_pattern_rows_decodes_codes_past_32_bits():
    """Spatial codes arrive unsigned and wider than an int32; the bits must
    still come back in order."""
    n_features = 40
    codes = np.array([0, 1, 1 << 39, (1 << 40) - 1, 0b1010 << 30],
                     dtype=np.uint64)
    rows = bern.pattern_rows(n_features, codes)
    for row, code in zip(rows, as_ints(codes)):
        assert np.array_equal(row, [(code >> b) & 1
                                    for b in range(n_features)])


def test_collapse_conserves_mass_and_groups_rows():
    rng = np.random.default_rng(SEED)
    X = random_binary(4000, 6)
    weight = rng.uniform(0.5, 2.0, len(X))
    rows, mass = bern._collapse(X, weight)

    assert np.isclose(mass.sum(), weight.sum())
    assert len(np.unique(bern.to_pattern_codes(rows))) == len(rows)
    for row, row_mass in zip(rows, mass):
        assert np.isclose(weight[np.all(X == row, axis=1)].sum(), row_mass)


def test_collapse_declines_when_it_would_not_pay():
    weight = np.ones(10)
    # Non-binary rows have no pattern code at all.
    assert bern._collapse(np.random.default_rng(SEED).random((10, 4)),
                          weight) is None
    # 2^30 counters for 10 rows is the wrong way round, and 30 marks is over
    # MAX_PATTERN_MARKS besides.
    assert bern._collapse(np.zeros((10, 30)), weight) is None


# --- spatial patterns ------------------------------------------------------

def test_block_spatial_codes_matches_the_naive_layout():
    rng = np.random.default_rng(SEED)
    for n_marks in (1, 2, 3, 5, 8, 11):
        block = rng.integers(0, 1 << n_marks, size=37)
        assert as_ints(bern.block_spatial_codes(block, n_marks)) \
            == ref_spatial_codes(block, n_marks)


def test_block_spatial_codes_holds_every_bit_it_promises():
    """All-ones bins are the widest code a vocabulary allows: 3 * n_marks set
    bits.  This is what catches a dtype too narrow for the layout."""
    for n_marks in (1, 10, 11, 21):
        block = np.full(5, (1 << n_marks) - 1)
        codes = bern.block_spatial_codes(block, n_marks)
        assert as_ints(codes) == ref_spatial_codes(block, n_marks)
        # The interior bins carry the previous, own and next marks all set.
        assert int(codes[2]) == (1 << 3 * n_marks) - 1


def test_block_spatial_codes_reads_off_the_ends_as_zero():
    n_marks = 4
    block = np.array([0b1111, 0b0001, 0b1000])
    codes = as_ints(bern.block_spatial_codes(block, n_marks))
    # First bin: no predecessor.
    assert codes[0] >> 0 & 0b1111 == 0
    assert codes[0] >> n_marks & 0b1111 == 0b1111
    # Last bin: no successor.
    assert codes[-1] >> 2 * n_marks & 0b1111 == 0
    assert codes[-1] >> n_marks & 0b1111 == 0b1000


def test_slices_keep_neighbours_from_crossing_a_chromosome_boundary():
    """The spatial window is what makes boundaries matter: without per-slice
    blocking the last bin of a chromosome would become the predecessor of the
    first bin of the next one."""
    n_marks = 4
    left = np.array([0b0001, 0b0010])
    right = np.array([0b0100, 0b1000])
    codes = np.concatenate((left, right))
    slices = [("chr1", 0, 2), ("chr2", 2, 4)]

    separate = (ref_spatial_codes(left, n_marks)
                + ref_spatial_codes(right, n_marks))
    labelled = np.concatenate([
        bern.block_spatial_codes(codes[lo:hi], n_marks)
        for _, lo, hi in slices])
    assert as_ints(labelled) == separate

    # And the joined-up track really would have leaked, so the test has teeth.
    assert as_ints(bern.block_spatial_codes(codes, n_marks)) != separate


def test_spatial_histogram_matches_the_naive_reference():
    n_marks = 3
    codes = np.random.default_rng(SEED).integers(0, 1 << n_marks, size=400)
    slices = even_slices(400, 4)

    expected = np.zeros(1 << 3 * n_marks, dtype=np.int64)
    for _, lo, hi in slices:
        for code in ref_spatial_codes(codes[lo:hi], n_marks):
            expected[code] += 1
    assert np.array_equal(bern.spatial_histogram(codes, slices, n_marks),
                          expected)


def test_histograms_accept_a_binary_matrix_as_well_as_codes():
    n_marks = 4
    codes = peaky_codes(1200, n_marks)
    X = bern.pattern_rows(n_marks, codes)
    slices = even_slices(1200, 2)

    assert np.array_equal(bern.spatial_histogram(X, slices, n_marks),
                          bern.spatial_histogram(codes, slices, n_marks))


def test_histograms_count_only_the_slices_given():
    """Restricting the slices to the training chromosomes has to restrict the
    counts, not just the work."""
    n_marks = 4
    codes = peaky_codes(3000, n_marks)
    every = even_slices(3000, 3)

    for slices in ([every[0]], every[:2], every):
        counted = sum(hi - lo for _, lo, hi in slices)
        assert bern.spatial_histogram(codes, slices, n_marks).sum() == counted


def test_histograms_tolerate_empty_and_degenerate_slices():
    n_marks = 3
    codes = peaky_codes(100, n_marks)
    for slices in ([], [("chr1", 5, 5)], [("chr1", 0, 1)]):
        counted = sum(hi - lo for _, lo, hi in slices)
        assert bern.spatial_histogram(codes, slices, n_marks).sum() == counted


def test_the_two_bin_window_holds_the_previous_bin_and_the_bin_itself():
    """What the pipeline's mixture reads: no successor, and half the bits of
    the 3-bin window."""
    n_marks = 4
    block = np.array([0b1111, 0b0001, 0b1000])
    codes = as_ints(bern.block_spatial_codes(block, n_marks, bins=2))
    assert codes == ref_spatial_codes(block, n_marks, bins=2)
    # First bin: no predecessor; every bin: its own marks above it.
    assert codes[0] == 0b1111 << n_marks
    assert codes[1] == 0b1111 | (0b0001 << n_marks)
    # And nothing above the two bins the window spans.
    assert max(codes) < 1 << 2 * n_marks


def test_every_window_width_matches_the_naive_layout():
    rng = np.random.default_rng(SEED)
    for bins in (1, 2, 3):
        for n_marks in (1, 3, 11, 13):
            block = rng.integers(0, 1 << n_marks, size=37)
            assert as_ints(bern.block_spatial_codes(block, n_marks, bins=bins)) \
                == ref_spatial_codes(block, n_marks, bins=bins)


def test_a_window_of_no_width_is_refused():
    for bins in (0, 4, -1):
        try:
            bern.block_spatial_codes(np.zeros(4, dtype=np.int64), 3, bins=bins)
        except ValueError as error:
            assert "1, 2 or 3 bins" in str(error)
            continue
        raise AssertionError(f"bins={bins} was accepted")


def test_the_sparse_histogram_is_the_dense_one_without_the_zeros():
    for bins in (1, 2, 3):
        n_marks = 4
        codes = peaky_codes(2000, n_marks)
        slices = even_slices(2000, 3)

        dense = bern.spatial_histogram(codes, slices, n_marks, bins=bins)
        keys, counts = bern.sparse_spatial_histogram(codes, slices, n_marks,
                                                     bins=bins)
        assert np.array_equal(keys, np.flatnonzero(dense))
        assert np.array_equal(counts, dense[np.flatnonzero(dense)])


def test_the_sparse_histogram_spans_a_vocabulary_no_array_could():
    """13 marks of 2-bin pattern is 2^26 counters and 11 marks of 3-bin one is
    2^33: the dense form refuses both, and the sparse form is what makes the
    SAGAconf datasets fittable."""
    for n_marks, bins in ((13, 2), (11, 3)):
        codes = peaky_codes(3000, n_marks)
        slices = even_slices(3000, 2)
        keys, counts = bern.sparse_spatial_histogram(codes, slices, n_marks,
                                                     bins=bins)
        assert counts.sum() == 3000
        assert len(keys) <= 3000
        assert int(keys.max()) < 1 << bins * n_marks


def test_the_sparse_histogram_counts_only_the_slices_given():
    n_marks = 5
    codes = peaky_codes(3000, n_marks)
    every = even_slices(3000, 3)
    for slices in ([], [("chr1", 5, 5)], [every[0]], every[:2], every):
        _, counts = bern.sparse_spatial_histogram(codes, slices, n_marks, bins=2)
        assert counts.sum() == sum(hi - lo for _, lo, hi in slices)


def test_the_sparse_annotation_agrees_with_the_reference():
    for bins in (1, 2, 3):
        n_marks = 4
        probs, weights = random_params(5, bins * n_marks)
        codes = peaky_codes(600, n_marks)
        slices = even_slices(600, 3)

        assert np.array_equal(
            bern.spatial_pattern_labels(probs, weights, codes, slices, n_marks,
                                        bins=bins, dtype=np.int64),
            ref_labels(codes, slices, n_marks, probs, weights, bins=bins))


def test_the_sparse_annotation_is_the_dense_lookup_without_the_table():
    """Same labels as spatial_labels() through a whole-vocabulary table, for a
    vocabulary small enough that both forms can be run."""
    n_marks = 4
    probs, weights = random_params(6, 2 * n_marks)
    codes = peaky_codes(1500, n_marks)
    slices = even_slices(1500, 3)

    table = bern.lookup_table(probs, weights, dtype=np.int64)
    assert np.array_equal(
        bern.spatial_labels(table, codes, slices, n_marks, bins=2),
        bern.spatial_pattern_labels(probs, weights, codes, slices, n_marks,
                                    bins=2, dtype=np.int64))


def test_the_sparse_annotation_does_not_depend_on_the_block_size():
    n_marks = 4
    probs, weights = random_params(5, 2 * n_marks)
    codes = peaky_codes(900, n_marks)
    slices = even_slices(900, 2)

    original = bern.PATTERN_BLOCK
    try:
        labels = bern.spatial_pattern_labels(probs, weights, codes, slices,
                                             n_marks, bins=2, dtype=np.int64)
        for block in (1, 7, 128):
            bern.PATTERN_BLOCK = block
            assert np.array_equal(
                bern.spatial_pattern_labels(probs, weights, codes, slices,
                                            n_marks, bins=2, dtype=np.int64),
                labels)
    finally:
        bern.PATTERN_BLOCK = original


def test_a_one_bin_window_fits_what_the_plain_mixture_fits():
    """bins=1 is the unwidened bin, so the sparse route has to reproduce the
    plain pattern histogram and the plain labels."""
    n_marks = 5
    codes = peaky_codes(2000, n_marks)
    slices = even_slices(2000, 3)

    keys, counts = bern.sparse_spatial_histogram(codes, slices, n_marks, bins=1)
    plain = np.bincount(codes, minlength=1 << n_marks)
    assert np.array_equal(keys, np.flatnonzero(plain))
    assert np.array_equal(counts, plain[np.flatnonzero(plain)])

    probs, weights = random_params(4, n_marks)
    assert np.array_equal(
        bern.spatial_pattern_labels(probs, weights, codes, slices, n_marks,
                                    bins=1, dtype=np.int64),
        bern.lookup_table(probs, weights, dtype=np.int64)[codes])


# --- the dense guard -------------------------------------------------------

def test_dense_forms_refuse_a_vocabulary_they_cannot_span():
    """Past MAX_DENSE_PATTERN_BITS an array indexed by pattern code stops
    fitting in memory - 11 marks of spatial pattern is 2^33 counters - so the
    dense entry points have to refuse rather than ask for it."""
    over = bern.MAX_DENSE_PATTERN_BITS + 1
    probs, weights = random_params(3, over)
    cases = [
        lambda: bern.pattern_rows(over),
        lambda: bern.lookup_table(probs, weights),
        lambda: bern.spatial_histogram(np.zeros(10, dtype=np.int64),
                                       [("chr1", 0, 10)], 11),
    ]
    for call in cases:
        try:
            call()
        except ValueError as error:
            assert "an array may span" in str(error)
            continue
        raise AssertionError("a dense form spanned an unspannable vocabulary")


# --- annotation ------------------------------------------------------------

def test_blocking_does_not_change_the_annotation():
    """PATTERN_BLOCK is a memory knob; the labels must not depend on it."""
    n_features = 12
    probs, weights = random_params(6, n_features)
    codes = np.random.default_rng(SEED).integers(0, 1 << n_features, size=300)

    original = bern.PATTERN_BLOCK
    try:
        bern.PATTERN_BLOCK = 1 << 15
        table = bern.lookup_table(probs, weights, dtype=np.int64)
        for block in (1, 7, 13, 512):
            bern.PATTERN_BLOCK = block
            assert np.array_equal(
                bern.lookup_table(probs, weights, dtype=np.int64), table)
    finally:
        bern.PATTERN_BLOCK = original


def test_spatial_annotation_agrees_with_the_reference():
    n_marks = 4
    probs, weights = random_params(5, 3 * n_marks)
    codes = peaky_codes(600, n_marks)
    slices = even_slices(600, 3)

    table = bern.lookup_table(probs, weights, dtype=np.int64)
    assert np.array_equal(bern.spatial_labels(table, codes, slices, n_marks),
                          ref_labels(codes, slices, n_marks, probs, weights))


def test_spatial_labels_keeps_the_lookup_tables_dtype():
    n_marks = 3
    probs, weights = random_params(4, 3 * n_marks)
    codes = peaky_codes(200, n_marks)
    slices = even_slices(200, 2)
    for dtype in (np.int8, np.int32, np.int64):
        table = bern.lookup_table(probs, weights, dtype=dtype)
        assert bern.spatial_labels(table, codes, slices,
                                   n_marks).dtype == dtype


# --- the log density -------------------------------------------------------

def test_logit_offset_reproduces_the_bernoulli_log_density():
    """The rearrangement to x . logit + offset is one matrix product instead of
    two and never materialises 1 - X; it has to be the same number."""
    n_features = 10
    probs, weights = random_params(7, n_features)
    X = random_binary(200, n_features, density=0.4)
    assert np.allclose(bern._log_weighted_prob(X, probs, weights),
                       ref_log_joint(X, probs, weights))


def test_logit_offset_survives_probabilities_at_the_floor():
    floor = 1e-6
    probs = np.array([[floor, 1 - floor, 0.5], [0.5, floor, 1 - floor]])
    weights = np.array([0.4, 0.6])
    X = bern.pattern_rows(3)
    values = bern._log_weighted_prob(X, probs, weights)
    assert np.all(np.isfinite(values))
    assert np.allclose(values, ref_log_joint(X, probs, weights))


def test_logsumexp_matches_scipy():
    values = np.random.default_rng(SEED).normal(0, 300, (50, 8))
    assert np.allclose(bern._logsumexp(values), logsumexp(values, axis=1))


def test_logsumexp_does_not_overflow_on_extreme_scores():
    values = np.array([[-1e5, -1e5 - 1.0], [1e5, 1e5 - 1.0]])
    assert np.allclose(bern._logsumexp(values), logsumexp(values, axis=1))


def test_responsibilities_are_normalised_and_carry_the_loglik():
    n_features = 8
    probs, weights = random_params(5, n_features)
    X = random_binary(300, n_features)

    resp, loglik = bern._responsibilities(X, probs, weights)
    joint = ref_log_joint(X, probs, weights)
    assert np.allclose(resp.sum(axis=1), 1.0)
    assert np.allclose(resp, np.exp(joint - logsumexp(joint, axis=1)[:, None]))
    assert np.allclose(loglik, logsumexp(joint, axis=1))


# --- EM --------------------------------------------------------------------

def test_em_step_is_the_textbook_weighted_update():
    n_features = 6
    probs, weights = random_params(5, n_features)
    X = random_binary(2000, n_features)
    rows, mass = bern._collapse(X, np.ones(len(X)))

    model = bern.BernoulliMixture(n_components=5, floor=1e-12)
    new_probs, new_weights, loglik = model._em_step(rows, mass, probs, weights)

    joint = ref_log_joint(rows, probs, weights)
    norm = logsumexp(joint, axis=1)
    resp = np.exp(joint - norm[:, np.newaxis]) * mass[:, np.newaxis]
    expected_mass = resp.sum(axis=0)

    assert np.allclose(new_probs,
                       (resp.T @ rows) / expected_mass[:, np.newaxis],
                       atol=1e-9)
    assert np.allclose(new_weights, expected_mass / expected_mass.sum())
    # The documented contract: the loglik returned is the one of the
    # parameters that went in, as the E-step saw them before the update.
    assert np.isclose(loglik, mass @ norm)


def test_em_never_decreases_the_loglik():
    n_features = 7
    probs, weights = random_params(6, n_features)
    rows, mass = bern._collapse(random_binary(3000, n_features), np.ones(3000))

    model = bern.BernoulliMixture(n_components=6, floor=1e-9)
    logliks = []
    for _ in range(80):
        probs, weights, loglik = model._em_step(rows, mass, probs, weights)
        logliks.append(loglik)
    assert np.all(np.diff(logliks) > -1e-7), np.diff(logliks).min()


def test_initialisation_shrinks_the_centroids_off_the_corners():
    """Unshrunk K-means centroids are hard 0/1 on most features, which at the
    floor makes the first E-step a hard assignment and pins EM to the partition
    K-means guessed.  The seeding must stay soft, and must carry the clusters'
    real masses rather than uniform proportions."""
    n_features = 6
    rows, mass = bern._collapse(random_binary(3000, n_features, density=0.2),
                               np.ones(3000))
    model = bern.BernoulliMixture(n_components=5, shrink=0.1, floor=1e-6)
    probs, weights = model._initialize_means(rows, mass,
                                             np.random.RandomState(SEED))

    assert probs.shape == (5, n_features)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(probs > model.floor) and np.all(probs < 1 - model.floor)
    # Shrinkage keeps every probability a shrink-fraction away from a corner.
    pooled = mass @ rows / mass.sum()
    assert probs.min() >= model.shrink * pooled.min() - 1e-12
    # A shrink of zero is what the hard corners look like, so the guard is
    # doing something.
    unshrunk = bern.BernoulliMixture(n_components=5, shrink=0.0, floor=1e-6)
    corners, _ = unshrunk._initialize_means(rows, mass,
                                            np.random.RandomState(SEED))
    assert corners.min() < probs.min()


def test_restarts_explore_different_seedings():
    """n_init only buys anything if the restarts differ; the shared
    RandomState has to advance between them."""
    n_features = 6
    rows, mass = bern._collapse(random_binary(2000, n_features), np.ones(2000))
    model = bern.BernoulliMixture(n_components=5)
    state = np.random.RandomState(SEED)
    first, _ = model._initialize_means(rows, mass, state)
    second, _ = model._initialize_means(rows, mass, state)
    assert not np.allclose(first, second)


def test_fit_is_reproducible_from_the_seed():
    X = random_binary(2000, 6)
    fits = [bern.BernoulliMixture(n_components=5, n_init=2, random_state=7,
                                  max_iter=50).fit(X) for _ in range(2)]
    assert np.array_equal(fits[0].means_, fits[1].means_)
    assert np.array_equal(fits[0].weights_, fits[1].weights_)
    assert np.array_equal(fits[0].labels_, fits[1].labels_)


def test_fit_keeps_its_parameters_in_range():
    X = random_binary(2000, 6)
    model = bern.BernoulliMixture(n_components=5, n_init=2, random_state=SEED,
                                  max_iter=50).fit(X)
    assert model.means_.shape == (5, 6)
    assert np.all(model.means_ >= model.floor)
    assert np.all(model.means_ <= 1 - model.floor)
    assert np.isclose(model.weights_.sum(), 1.0)
    assert np.all(model.weights_ >= 0)


def test_lower_bound_is_a_lower_bound_on_the_fitted_loglik():
    """lower_bound_ is reported per unit weight, and comes from the E-step that
    preceded the final update - so it can only understate the parameters it is
    stored beside, never overstate them."""
    X = random_binary(4000, 6)
    for max_iter in (2, 5, 300):
        model = bern.BernoulliMixture(n_components=5, n_init=1,
                                      random_state=SEED,
                                      max_iter=max_iter).fit(X)
        actual = model.score(X) / len(X)
        assert model.lower_bound_ <= actual + 1e-12
        assert np.isclose(model.lower_bound_, actual, atol=1e-2)


def test_more_restarts_do_not_find_a_worse_optimum():
    hist = np.bincount(bern.to_pattern_codes(random_binary(20_000, 8)),
                       minlength=1 << 8).astype(float)
    scores = []
    for n_init in (1, 5, 20):
        probs, weights, _ = bern.BernoulliMixture(
            n_components=6, n_init=n_init, random_state=SEED,
            max_iter=200).fit_histogram(hist, 8)
        scores.append(bern.log_likelihood(hist, probs, weights))
    assert scores[1] >= scores[0] - 1e-6
    assert scores[2] >= scores[1] - 1e-6


# --- the fit is the same whatever representation it is handed --------------

def test_fitting_rows_equals_fitting_their_pattern_table():
    """The whole point of the collapse: an EM iteration over the weighted
    patterns is an EM iteration over the rows they stand for."""
    n_features = 6
    X = random_binary(20_000, n_features)
    hist = np.bincount(bern.to_pattern_codes(X),
                       minlength=1 << n_features).astype(float)

    on_rows = bern.BernoulliMixture(n_components=5, n_init=2, random_state=7,
                                    max_iter=200)
    on_rows.fit(X)
    probs, weights, _ = bern.BernoulliMixture(
        n_components=5, n_init=2, random_state=7,
        max_iter=200).fit_histogram(hist, n_features)

    order = np.lexsort(on_rows.means_.T)
    assert np.allclose(on_rows.means_[order], probs[np.lexsort(probs.T)],
                       atol=1e-6)


def test_sample_weights_are_the_same_as_repeated_rows():
    n_features = 5
    unique = bern.pattern_rows(n_features)
    weight = np.arange(1, len(unique) + 1, dtype=float)
    repeated = np.repeat(unique, weight.astype(int), axis=0)

    weighted = bern.BernoulliMixture(n_components=4, n_init=2, random_state=3,
                                     max_iter=100)
    weighted.fit(unique, sample_weight=weight)
    plain = bern.BernoulliMixture(n_components=4, n_init=2, random_state=3,
                                  max_iter=100)
    plain.fit(repeated)
    assert np.allclose(np.sort(weighted.means_, axis=0),
                       np.sort(plain.means_, axis=0), atol=1e-6)


def test_fit_histogram_equals_fit_patterns():
    n_features = 8
    hist = np.bincount(bern.to_pattern_codes(random_binary(20_000,
                                                           n_features)),
                       minlength=1 << n_features).astype(float)
    present = np.flatnonzero(hist)

    dense = bern.BernoulliMixture(n_components=6, n_init=2,
                                  random_state=7).fit_histogram(hist,
                                                                n_features)
    sparse = bern.BernoulliMixture(n_components=6, n_init=2,
                                   random_state=7).fit_patterns(
        present, hist[present], n_features)
    for from_dense, from_sparse in zip(dense, sparse):
        assert np.allclose(from_dense, from_sparse)


def test_fit_patterns_returns_parameters_and_a_per_bin_loglik():
    n_features = 8
    hist = np.bincount(bern.to_pattern_codes(random_binary(10_000,
                                                           n_features)),
                       minlength=1 << n_features).astype(float)
    present = np.flatnonzero(hist)
    probs, weights, lower_bound = bern.BernoulliMixture(
        n_components=5, n_init=2, random_state=SEED,
        max_iter=300).fit_patterns(present, hist[present], n_features)

    assert probs.shape == (5, n_features)
    assert np.isclose(weights.sum(), 1.0)
    assert lower_bound <= bern.log_likelihood(hist, probs,
                                              weights) / hist.sum() + 1e-12


# --- scoring ---------------------------------------------------------------

def test_score_is_the_weighted_total_loglik():
    n_features = 6
    X = random_binary(3000, n_features)
    weight = np.random.default_rng(SEED).uniform(0.5, 2.0, len(X))
    model = bern.BernoulliMixture(n_components=5, n_init=2,
                                  random_state=SEED).fit(X)

    expected = weight @ logsumexp(ref_log_joint(X, model.means_,
                                                model.weights_), axis=1)
    assert np.isclose(model.score(X, sample_weight=weight), expected)
    assert np.isclose(model.score(X),
                      logsumexp(ref_log_joint(X, model.means_,
                                              model.weights_),
                                axis=1).sum())


def test_log_likelihood_of_a_histogram_equals_score_of_its_rows():
    n_features = 7
    X = random_binary(8000, n_features)
    hist = np.bincount(bern.to_pattern_codes(X),
                       minlength=1 << n_features).astype(float)
    present = np.flatnonzero(hist)
    model = bern.BernoulliMixture(n_components=5, n_init=2,
                                  random_state=SEED).fit(X)

    assert np.isclose(bern.log_likelihood(hist, model.means_, model.weights_),
                      model.score(bern.pattern_rows(n_features, present),
                                  sample_weight=hist[present]))
    assert np.isclose(bern.log_likelihood(hist, model.means_, model.weights_),
                      model.score(X))


def test_bic_counts_the_free_parameters():
    n_features, n_components = 6, 5
    X = random_binary(3000, n_features)
    model = bern.BernoulliMixture(n_components=n_components, n_init=2,
                                  random_state=SEED).fit(X)
    # n_components * n_features probabilities, plus n_components - 1 free
    # mixing proportions.
    n_params = n_components * n_features + n_components - 1
    assert np.isclose(model.bic(X),
                      -2 * model.score(X) + n_params * np.log(len(X)))


def test_entropy_is_the_responsibility_entropy_and_icl_adds_it_to_bic():
    n_features = 6
    X = random_binary(3000, n_features)
    model = bern.BernoulliMixture(n_components=5, n_init=2,
                                  random_state=SEED).fit(X)

    joint = ref_log_joint(X, model.means_, model.weights_)
    log_resp = joint - logsumexp(joint, axis=1)[:, np.newaxis]
    assert np.isclose(model.entropy(X), -np.sum(np.exp(log_resp) * log_resp))
    assert model.entropy(X) >= 0
    assert np.isclose(model.icl(X), model.bic(X) + 2 * model.entropy(X))


def test_a_confident_model_has_almost_no_assignment_entropy():
    X = bern.pattern_rows(3)
    probs = np.clip(X[[0, 7]], 1e-9, 1 - 1e-9)     # all-off and all-on
    model = bern.BernoulliMixture(n_components=2)
    model.means_, model.weights_ = probs, np.array([0.5, 0.5])
    assert model.entropy(np.array([[0.0, 0, 0], [1.0, 1, 1]])) < 1e-6


# --- prediction ------------------------------------------------------------

def test_predict_equals_the_brute_force_argmax():
    n_features = 6
    X = random_binary(3000, n_features)
    model = bern.BernoulliMixture(n_components=5, n_init=2,
                                  random_state=SEED).fit(X)
    expected = np.argmax(ref_log_joint(X, model.means_, model.weights_),
                         axis=1)
    assert np.array_equal(model.predict(X), expected)
    assert np.array_equal(model.labels_, expected)


def test_predict_agrees_whether_or_not_the_pattern_table_is_used():
    """predict() labels the pattern table and indexes into it when that pays
    off, and falls back to a row-wise argmax when it does not.  The two paths
    are the same answer."""
    n_features = 6
    model = bern.BernoulliMixture(n_components=5, n_init=2, random_state=SEED)
    model.fit(random_binary(3000, n_features))

    many = random_binary(3000, n_features)          # collapses
    few = many[:5]                                  # does not
    assert bern._pattern_codes(many) is not None
    assert bern._pattern_codes(few) is None
    assert np.array_equal(model.predict(many)[:5], model.predict(few))


def test_predict_proba_is_normalised_and_consistent_with_predict():
    n_features = 6
    X = random_binary(1000, n_features)
    model = bern.BernoulliMixture(n_components=5, n_init=2,
                                  random_state=SEED).fit(X)
    proba = model.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0)
    assert np.array_equal(np.argmax(proba, axis=1), model.predict(X))


# --- fit_bernoulli ---------------------------------------------------------

def test_fit_bernoulli_reports_the_loglik_of_what_it_returns():
    n_features = 8
    hist = np.bincount(bern.to_pattern_codes(random_binary(20_000,
                                                           n_features)),
                       minlength=1 << n_features).astype(float)
    probs, weights, loglik = bern.fit_bernoulli(hist, k=5, seed=5, n_init=2,
                                                iterations=50, polish=0)
    assert probs.shape == (5, n_features)
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(loglik, bern.log_likelihood(hist, probs, weights))


def test_polishing_only_improves_the_fit():
    n_features = 8
    hist = np.bincount(bern.to_pattern_codes(random_binary(20_000,
                                                           n_features)),
                       minlength=1 << n_features).astype(float)
    unpolished = bern.fit_bernoulli(hist, k=5, seed=5, n_init=2,
                                    iterations=30, polish=0)
    polished = bern.fit_bernoulli(hist, k=5, seed=5, n_init=2, iterations=30,
                                  polish=20_000, chunk=500)
    assert polished[2] >= unpolished[2] - 1e-6


def test_fit_bernoulli_starts_from_a_given_model():
    n_features = 8
    hist = np.bincount(bern.to_pattern_codes(random_binary(20_000,
                                                           n_features)),
                       minlength=1 << n_features).astype(float)
    start = bern.fit_bernoulli(hist, k=5, seed=5, n_init=2, iterations=30,
                               polish=0)

    # Handed a model and told not to polish, it hands the same one back.
    frozen = bern.fit_bernoulli(hist, init=start[:2], k=5, polish=0)
    assert np.allclose(frozen[0], start[0])
    assert np.allclose(frozen[1], start[1])
    assert np.isclose(frozen[2], bern.log_likelihood(hist, *start[:2]))

    # Polishing from it cannot make it worse.
    warmed = bern.fit_bernoulli(hist, init=start[:2], k=5, polish=5000,
                                chunk=500)
    assert warmed[2] >= frozen[2] - 1e-6


def test_fit_bernoulli_clips_a_given_model_into_range():
    n_features = 4
    hist = np.bincount(bern.to_pattern_codes(random_binary(2000, n_features)),
                       minlength=1 << n_features).astype(float)
    degenerate = (np.zeros((3, n_features)), np.ones(3) / 3)
    probs, _, loglik = bern.fit_bernoulli(hist, init=degenerate, k=3,
                                          polish=0, floor=1e-6)
    assert np.all(probs >= 1e-6)
    assert np.isfinite(loglik)


# --- does it actually find the mixture that generated the data? ------------

def test_a_planted_mixture_is_recovered():
    """The equivalences above would all hold for a broken optimiser, so one
    test asks the only question that matters: given data from a known
    Bernoulli mixture, does the fit come back with it?"""
    rng = np.random.default_rng(SEED)
    n_components, n_features, n_samples = 4, 8, 200_000
    true_probs = rng.uniform(0.05, 0.95, (n_components, n_features))
    true_weights = rng.dirichlet(np.ones(n_components) * 5)

    drawn = rng.choice(n_components, n_samples, p=true_weights)
    X = (rng.random((n_samples, n_features)) < true_probs[drawn]).astype(float)
    hist = np.bincount(bern.to_pattern_codes(X),
                       minlength=1 << n_features).astype(float)

    probs, weights, _ = bern.BernoulliMixture(
        n_components=n_components, n_init=10, random_state=SEED,
        max_iter=500).fit_histogram(hist, n_features)

    # The components come back in an arbitrary order.
    rows, cols = linear_sum_assignment(
        np.abs(true_probs[:, np.newaxis, :] - probs[np.newaxis]).sum(axis=2))
    assert np.abs(true_probs[rows] - probs[cols]).max() < 0.05
    assert np.abs(true_weights[rows] - weights[cols]).max() < 0.02
    # And the fit explains the sample at least as well as the truth does.
    assert bern.log_likelihood(hist, probs, weights) >= \
        bern.log_likelihood(hist, true_probs, true_weights)


def test_a_planted_spatial_mixture_beats_a_non_spatial_one():
    """A spatial fit sees each bin with its neighbours, so on a track whose
    structure is in the runs it has to explain the data better per bin than the
    same number of states fitted on single bins."""
    n_marks, n_bins = 5, 40_000
    codes = peaky_codes(n_bins, n_marks)
    slices = even_slices(n_bins, 4)

    plain = np.bincount(codes, minlength=1 << n_marks).astype(float)
    flat = bern.BernoulliMixture(n_components=6, n_init=3,
                                 random_state=SEED).fit_histogram(plain,
                                                                  n_marks)
    hist = bern.spatial_histogram(codes, slices, n_marks)
    spatial = bern.BernoulliMixture(n_components=6, n_init=3,
                                    random_state=SEED).fit_histogram(
        hist, 3 * n_marks)
    assert spatial[2] < flat[2]        # per-bin loglik over a wider vocabulary


def test_the_spatial_window_separates_bins_their_own_marks_cannot():
    """The defining property of the spatial variant, and the only thing it can
    do that a single-bin model cannot: two bins with identical marks of their
    own get different states when their neighbours differ.  A flat model reads
    a bin through its own marks alone, so it assigns one state per mark pattern
    by construction - an empty bin beside a peak and an empty bin in the middle
    of nowhere are the same call.  Here they need not be.
    """
    n_marks, n_bins = 5, 40_000
    codes = peaky_codes(n_bins, n_marks)
    slices = even_slices(n_bins, 4)

    hist = bern.spatial_histogram(codes, slices, n_marks)
    probs, weights, _ = bern.BernoulliMixture(
        n_components=6, n_init=3, random_state=SEED).fit_histogram(
        hist, 3 * n_marks)
    labels = bern.spatial_labels(
        bern.lookup_table(probs, weights, dtype=np.int64), codes, slices,
        n_marks)

    observed = np.unique(codes)
    split = [code for code in observed
             if len(np.unique(labels[codes == code])) > 1]
    assert len(split) > 0.2 * len(observed)
    # The empty bin is the clearest case: context alone has to move it.
    assert len(np.unique(labels[codes == 0])) > 1

    # A flat model over the same track cannot split any mark pattern at all.
    plain = np.bincount(codes, minlength=1 << n_marks).astype(float)
    flat_probs, flat_weights, _ = bern.BernoulliMixture(
        n_components=6, n_init=3, random_state=SEED).fit_histogram(plain,
                                                                   n_marks)
    flat = bern.lookup_table(flat_probs, flat_weights, dtype=np.int64)[codes]
    assert all(len(np.unique(flat[codes == code])) == 1 for code in observed)


# --- standalone runner -----------------------------------------------------

def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as error:                       # noqa: BLE001
            failed.append(name)
            print(f"FAIL {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
