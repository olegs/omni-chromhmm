#!/usr/bin/env python3
"""Tests for the weighted Bernoulli mixture and its pattern forms.

Every claim bernoulli.py makes about its own representation is a claim that two
computations agree: the collapsed fit is the fit on the rows, the blocked
histogram and lookup are the unblocked ones, the logit/offset algebra is the
Bernoulli log density.  So the tests here are mostly equivalences, checked
against a naive reference written out in full - the explicit
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

def ref_log_joint(X, probs, weights):
    """log p(x, component) written out, without the logit/offset rearrangement."""
    return (X @ np.log(probs).T + (1 - X) @ np.log1p(-probs).T
            + np.log(weights))


# --- fixtures --------------------------------------------------------------

def random_binary(n_samples, n_features, density=0.3, seed=SEED):
    return (np.random.default_rng(seed).random((n_samples, n_features))
            < density).astype(float)


def random_params(n_components, n_features, seed=SEED):
    """Probabilities well inside (0, 1) and mixing proportions that sum to 1."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.05, 0.95, (n_components, n_features)),
            rng.dirichlet(np.ones(n_components)))


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
    """Codes can arrive unsigned and wider than an int32; the bits must still
    come back in order."""
    n_features = 40
    codes = np.array([0, 1, 1 << 39, (1 << 40) - 1, 0b1010 << 30],
                     dtype=np.uint64)
    rows = bern.pattern_rows(n_features, codes)
    for row, code in zip(rows, (int(c) for c in codes)):
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


# --- the dense guard -------------------------------------------------------

def test_dense_forms_refuse_a_vocabulary_they_cannot_span():
    """Past MAX_DENSE_PATTERN_BITS an array indexed by pattern code stops
    fitting in memory, so the dense entry points have to refuse rather than
    ask for it."""
    over = bern.MAX_DENSE_PATTERN_BITS + 1
    probs, weights = random_params(3, over)
    cases = [
        lambda: bern.pattern_rows(over),
        lambda: bern.lookup_table(probs, weights),
        lambda: bern.pattern_histogram(np.zeros((10, over), dtype=np.uint8)),
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


# --- genome-wide tracks ----------------------------------------------------

def test_pattern_histogram_counts_every_bin_once():
    X = random_binary(10_001, 6)
    assert np.array_equal(bern.pattern_histogram(X),
                          np.bincount(bern.to_pattern_codes(X),
                                      minlength=1 << 6))


def test_pattern_histogram_does_not_depend_on_the_block_size():
    X = random_binary(5000, 5)
    whole = bern.pattern_histogram(X)
    for block in (1, 7, 999, 5000, 10_000):
        assert np.array_equal(bern.pattern_histogram(X, block=block), whole)


def test_pattern_histogram_of_no_bins_is_empty():
    assert bern.pattern_histogram(np.zeros((0, 4), dtype=np.uint8)).sum() == 0


def test_track_labels_are_the_brute_force_argmax():
    X = random_binary(3000, 5)
    probs, weights = random_params(4, 5)
    assert np.array_equal(
        bern.track_labels(probs, weights, X, block=101, dtype=np.int64),
        np.argmax(ref_log_joint(X, probs, weights), axis=1))


def test_track_labels_do_not_depend_on_the_block_size():
    X = random_binary(4000, 6)
    probs, weights = random_params(5, 6)
    whole = bern.track_labels(probs, weights, X)
    for block in (1, 13, 4000, 8000):
        assert np.array_equal(bern.track_labels(probs, weights, X, block=block),
                              whole)


def test_track_labels_keep_the_requested_dtype():
    X = random_binary(200, 4)
    probs, weights = random_params(3, 4)
    for dtype in (np.int8, np.int32, np.int64):
        assert bern.track_labels(probs, weights, X, dtype=dtype).dtype == dtype


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
