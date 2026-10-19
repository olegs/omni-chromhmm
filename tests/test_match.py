import os
import tempfile
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "rules")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis")))

import match
import utils


def test_state_lengths_and_pair_overlap():
    ref_segs = [
        ("chr1", 0, 1000, "Tss", "255,0,0"),
        ("chr1", 1000, 5000, "Tx", "0,255,0"),
        ("chr1", 5000, 10000, "Quies", "255,255,255"),
    ]
    work_segs = [
        ("chr1", 0, 1000, "E1", "0,0,0"),
        ("chr1", 1000, 3000, "E2", "0,0,0"),
        ("chr1", 3000, 10000, "E3", "0,0,0"),
    ]
    rl = match.state_lengths(ref_segs)
    wl = match.state_lengths(work_segs)
    assert rl == {"Tss": 1000, "Tx": 4000, "Quies": 5000}
    assert wl == {"E1": 1000, "E2": 2000, "E3": 7000}

    overlap = match.pair_overlap(ref_segs, work_segs)
    assert overlap[("E1", "Tss")] == 1000
    assert overlap[("E2", "Tx")] == 2000
    assert overlap[("E3", "Tx")] == 2000
    assert overlap[("E3", "Quies")] == 5000


def test_match_states_bijection(capsys):
    ref_segs = [
        ("chr1", 0, 1000, "Tss", "255,0,0"),
        ("chr1", 1000, 5000, "Tx", "0,255,0"),
        ("chr1", 5000, 10000, "Quies", "255,255,255"),
    ]
    work_segs = [
        ("chr1", 0, 1000, "E1", "0,0,0"),
        ("chr1", 1000, 5000, "E2", "0,0,0"),
        ("chr1", 5000, 10000, "E3", "0,0,0"),
    ]
    mapping = match.match_states([ref_segs], [work_segs])
    assert mapping == {"E1": "Tss", "E2": "Tx", "E3": "Quies"}
    captured = capsys.readouterr()
    assert "jaccard=" in captured.err
    assert "overlap=" not in captured.err
    assert "npmi=" not in captured.err
    assert "composition=" not in captured.err
    assert "emission=" not in captured.err
    assert "composition_similarity" not in captured.err
    assert "quality=" in captured.err


def test_agreement_metrics():
    ref_segs = [
        ("chr1", 0, 1000, "Tss", "255,0,0"),
        ("chr1", 1000, 5000, "Tx", "0,255,0"),
        ("chr1", 5000, 10000, "Quies", "255,255,255"),
    ]
    overlap = match.pair_overlap(ref_segs, ref_segs)
    lengths = match.state_lengths(ref_segs)
    metrics = match.agreement_metrics(overlap, lengths, lengths)
    assert np.isclose(metrics[utils.JACCARD], 1.0)
    assert np.isclose(metrics[utils.KAPPA], 1.0)
    assert np.isclose(metrics[utils.COSINE], 1.0)
    assert np.isclose(metrics[utils.AMI], 1.0)


def test_ami_ignores_state_names():
    segs1 = [
        ("chr1", 0, 1000, "Tss", "0,0,0"),
        ("chr1", 1000, 5000, "Tx", "0,0,0"),
        ("chr1", 5000, 10000, "Quies", "0,0,0"),
    ]
    # The same partition under other names: no state matching, a perfect AMI,
    # where kappa on the names sees no agreement at all.
    segs2 = [(c, s, e, {"Tss": "E3", "Tx": "E1", "Quies": "E2"}[n], col)
             for c, s, e, n, col in segs1]
    overlap = match.pair_overlap(segs2, segs1)
    metrics = match.agreement_metrics(overlap, match.state_lengths(segs1),
                                      match.state_lengths(segs2))
    assert np.isclose(metrics[utils.AMI], 1.0)
    assert metrics[utils.KAPPA] < 0.1


def test_ami_matches_sklearn():
    # The exact AMI of a bin-level contingency table: the bp version drops the
    # expected mutual information, which vanishes on a large table.
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 200_000)
    y = np.where(rng.random(len(x)) < 0.6, x, rng.integers(0, 4, len(x)))
    overlap = {}
    for i, j in zip(x, y):
        overlap[(i, j)] = overlap.get((i, j), 0) + 200
    exact = sklearn_metrics.adjusted_mutual_info_score(x, y)
    assert np.isclose(match.adjusted_mutual_information(overlap), exact, atol=1e-4)


def test_ami_independent_partitions():
    overlap = {("A", "X"): 100, ("A", "Y"): 100, ("B", "X"): 100, ("B", "Y"): 100}
    assert np.isclose(match.adjusted_mutual_information(overlap), 0.0)


def test_ami_full_domain_only():
    overlap = {("Tss", "Tss"): 100, ("Quies", "Quies"): 500}
    lengths = {"Tss": 100, "Quies": 500}
    modes = match.agreement_by_mode(overlap, lengths, lengths,
                                    background={"Quies"})
    assert np.isclose(modes[utils.FULL][utils.AMI], 1.0)
    assert utils.AMI not in modes[utils.NOQH]


def test_emission_similarity_keeps_signal_level():
    mat1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mat2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 0.0]])
    sim = match.emission_similarity(mat1, mat2)
    assert np.isclose(sim[0, 0], 1.0)
    assert np.isclose(sim[2, 2], 1.0)
    assert sim[1, 1] < 1.0
    assert sim[0, 2] == 0.0

    # Same profile shape, different level: a near-empty Quies is not an all-high state.
    quies = np.array([[0.02, 0.02, 0.02]])
    high = np.array([[0.9, 0.9, 0.9], [0.0, 0.0, 0.0]])
    sim = match.emission_similarity(quies, high)
    assert sim[0, 1] > 0.99
    assert sim[0, 0] < 0.2


def test_emission_similarity_raw_signal_is_scaled_per_mark():
    raw = np.array([[10.0, 0.0], [0.0, 4.0], [0.0, 0.0]])
    sim = match.emission_similarity(raw, raw)
    assert np.allclose(np.diag(sim), 1.0)
    assert sim[0, 1] < 0.1


def test_bigwig_scaled_is_on_the_binarized_scale():
    # Fold-enrichment signal: state 1 is the background (median) of both marks.
    bw = np.array([[25.0, 0.3], [0.25, 0.25], [1.0, 10.0], [0.2, 0.2]])
    scaled = match.bigwig_scaled(bw)
    assert scaled.min() >= 0.0 and scaled.max() <= 1.0
    assert np.allclose(scaled.max(axis=0), 1.0)      # top state of each mark
    assert np.allclose(scaled[[1, 3]], 0.0)          # background and below
    assert 0.0 < scaled[2, 0] < 0.25                 # weak signal stays low on the log scale
    assert np.allclose(match.bigwig_scaled(np.ones((3, 2))), 0.0)   # flat mark: no signal


def test_emission_similarity_bigwig_vs_binarized():
    marks = ["H3K4me3", "H3K27me3"]
    binarized = np.array([[0.95, 0.01], [0.02, 0.90], [0.01, 0.02]])     # Tss, ReprPC, Quies
    bigwig = np.array([[0.4, 0.3], [30.0, 0.35], [0.45, 12.0]])           # Quies, Tss, ReprPC
    sim = match.emission_similarity(binarized, bigwig, binarized_w=True, binarized_r=False)
    assert list(sim.argmax(axis=1)) == [1, 2, 0]
    assert sim[2, 0] > 0.95


def test_match_states_matrix_out():
    ref_segs = [
        ("chr1", 0, 1000, "Tss", "255,0,0"),
        ("chr1", 1000, 5000, "Tx", "0,255,0"),
        ("chr1", 5000, 10000, "Quies", "255,255,255"),
    ]
    work_segs = [
        ("chr1", 0, 1000, "E1", "0,0,0"),
        ("chr1", 1000, 5000, "E2", "0,0,0"),
        ("chr1", 5000, 10000, "E3", "0,0,0"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "test_match")
        mapping = match.match_states([ref_segs], [work_segs], matrix_out=prefix)
        assert mapping == {"E1": "Tss", "E2": "Tx", "E3": "Quies"}
        assert os.path.exists(f"{prefix}.jaccard.tsv")
        assert not os.path.exists(f"{prefix}.score.tsv")
        assert not os.path.exists(f"{prefix}.emission.tsv")
        assert not os.path.exists(f"{prefix}.composition.tsv")
        assert not os.path.exists(f"{prefix}.npmi.tsv")
        assert os.path.exists(f"{prefix}.quality.tsv")
        assert os.path.exists(f"{prefix}.mapping.tsv")
        assert os.path.exists(f"{prefix}.png")

        with open(f"{prefix}.mapping.tsv") as f:
            header = f.readline().strip().split("\t")
            assert "jaccard" in header
            assert "composition" not in header
            assert "npmi" not in header
            assert "quality" in header
            for line in f:
                row = line.strip().split("\t")
                assert len(row) == len(header)


def test_match_states_with_emissions(capsys):
    ref_segs = [
        ("chr1", 0, 1000, "Tss", "255,0,0"),
        ("chr1", 1000, 5000, "Tx", "0,255,0"),
        ("chr1", 5000, 10000, "Quies", "255,255,255"),
    ]
    work_segs = [
        ("chr1", 0, 1000, "E1", "0,0,0"),
        ("chr1", 1000, 5000, "E2", "0,0,0"),
        ("chr1", 5000, 10000, "E3", "0,0,0"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        work_em = os.path.join(tmpdir, "work.emissions.npz")
        ref_em = os.path.join(tmpdir, "ref.emissions.npz")
        prefix = os.path.join(tmpdir, "test_match_em")

        match._save_emissions_npz(work_em, ["E1", "E2", "E3"], ["H3K4me3", "H3K36me3"],
                                  np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
        match._save_emissions_npz(ref_em, ["Tss", "Tx", "Quies"], ["H3K4me3", "H3K36me3"],
                                  np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

        mapping = match.match_states([ref_segs], [work_segs], matrix_out=prefix,
                                     work_emissions_path=work_em, ref_emissions_path=ref_em)
        assert mapping == {"E1": "Tss", "E2": "Tx", "E3": "Quies"}
        assert os.path.exists(f"{prefix}.jaccard.tsv")
        assert not os.path.exists(f"{prefix}.score.tsv")
        assert os.path.exists(f"{prefix}.emission.tsv")
        assert not os.path.exists(f"{prefix}.composition.tsv")
        assert not os.path.exists(f"{prefix}.npmi.tsv")
        assert os.path.exists(f"{prefix}.quality.tsv")
        assert os.path.exists(f"{prefix}.mapping.tsv")
        assert os.path.exists(f"{prefix}.png")

        captured = capsys.readouterr()
        assert "jaccard=" in captured.err
        assert "overlap=" not in captured.err
        assert "npmi=" not in captured.err
        assert "emission=" in captured.err
        assert "composition=" not in captured.err
        assert "composition_similarity" not in captured.err
        assert "quality=" in captured.err

        with open(f"{prefix}.mapping.tsv") as f:
            header = f.readline().strip().split("\t")
            assert "jaccard" in header
            assert "emission" in header
            assert "composition" not in header
            assert "npmi" not in header
            assert "quality" in header


def _equal_overlap_segs():
    """Work E1/E2 overlap reference A/B equally, so only emissions decide."""
    ref_segs = [
        ("chr1", 0, 1000, "A", "0,0,0"),
        ("chr1", 1000, 2000, "B", "0,0,0"),
    ]
    work_segs = [
        ("chr1", 0, 500, "E1", "0,0,0"),
        ("chr1", 500, 1000, "E2", "0,0,0"),
        ("chr1", 1000, 1500, "E1", "0,0,0"),
        ("chr1", 1500, 2000, "E2", "0,0,0"),
    ]
    return ref_segs, work_segs


def test_match_states_emissions_rows_by_name():
    ref_segs, work_segs = _equal_overlap_segs()
    with tempfile.TemporaryDirectory() as tmpdir:
        work_em = os.path.join(tmpdir, "work.npz")
        ref_em = os.path.join(tmpdir, "ref.npz")
        # Rows stored out of sorted order: E2 first.
        match._save_emissions_npz(work_em, ["E2", "E1"], ["H3K4me3", "H3K36me3"],
                                  np.array([[0.0, 1.0], [1.0, 0.0]]))
        match._save_emissions_npz(ref_em, ["B", "A"], ["H3K36me3", "H3K4me3"],
                                  np.array([[0.0, 1.0], [1.0, 0.0]]))
        mapping = match.match_states([ref_segs], [work_segs],
                                     work_emissions_path=work_em, ref_emissions_path=ref_em)
        # A and B have H3K4me3 and H3K36me3 respectively; so do E2 and E1.
        assert mapping == {"E1": "B", "E2": "A"}


def test_match_states_emissions_missing_state(capsys):
    ref_segs = [
        ("chr1", 0, 1000, "Tss", "0,0,0"),
        ("chr1", 1000, 5000, "Tx", "0,0,0"),
        ("chr1", 5000, 10000, "Quies", "0,0,0"),
    ]
    work_segs = [
        ("chr1", 0, 1000, "E1", "0,0,0"),
        ("chr1", 1000, 5000, "E2", "0,0,0"),
        ("chr1", 5000, 10000, "E3", "0,0,0"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        work_em = os.path.join(tmpdir, "work.npz")
        ref_em = os.path.join(tmpdir, "ref.npz")
        match._save_emissions_npz(work_em, ["E1", "E2"], ["H3K4me3", "H3K36me3"],
                                  np.array([[1.0, 0.0], [0.0, 1.0]]))
        match._save_emissions_npz(ref_em, ["Tss", "Tx", "Quies"], ["H3K4me3", "H3K36me3"],
                                  np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
        mapping = match.match_states([ref_segs], [work_segs],
                                     work_emissions_path=work_em, ref_emissions_path=ref_em)
        assert mapping == {"E1": "Tss", "E2": "Tx", "E3": "Quies"}
        captured = capsys.readouterr()
        assert "no work emissions for states ['E3']" in captured.err


def test_match_states_emissions_with_other_names_are_ignored(capsys):
    ref_segs, work_segs = _equal_overlap_segs()
    with tempfile.TemporaryDirectory() as tmpdir:
        work_em = os.path.join(tmpdir, "work.npz")
        ref_em = os.path.join(tmpdir, "ref.npz")
        # Same size as the segmentation, but none of its state names.
        match._save_emissions_npz(work_em, ["X1", "X2"], ["H3K4me3"],
                                  np.array([[0.0], [1.0]]))
        match._save_emissions_npz(ref_em, ["A", "B"], ["H3K4me3"],
                                  np.array([[1.0], [0.0]]))
        match.match_states([ref_segs], [work_segs],
                           work_emissions_path=work_em, ref_emissions_path=ref_em)
        captured = capsys.readouterr()
        assert "match no segmentation state" in captured.err
        assert "emission=" not in captured.err


def test_emissions_binarized_from_file_name():
    assert match._emissions_binarized("a/x.bin_emissions.npz") is True
    assert match._emissions_binarized("a/ENCFF1_chromhmm.bw_emissions.npz") is False
    assert match._emissions_binarized("a/x.npz") is None


def test_pick_emissions_prefers_binarized():
    import argparse
    args = argparse.Namespace(work_emissions=["w.bw_emissions.npz"],
                              work_bin_emissions=["w.bin_emissions.npz"],
                              ref_bw_emissions="r.bw_emissions.npz", ref_bin_emissions=None,
                              work_em_type=None, ref_em_type=None)
    assert match._pick_emissions(args) == ("w.bin_emissions.npz", "r.bw_emissions.npz")
    args.ref_em_type = "bin"     # asked-for type missing: fall back to what exists
    assert match._pick_emissions(args) == ("w.bin_emissions.npz", "r.bw_emissions.npz")
    args.work_em_type = "bw"
    assert match._pick_emissions(args)[0] == "w.bw_emissions.npz"


def test_quality_is_jaccard_without_emissions():
    ref_segs = [("chr1", 0, 9000, "A", "0,0,0"), ("chr1", 9000, 10000, "B", "0,0,0")]
    work_segs = [("chr1", 0, 5000, "E1", "0,0,0"), ("chr1", 5000, 10000, "E2", "0,0,0")]
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "m")
        match.match_states([ref_segs], [work_segs], matrix_out=prefix)
        load = lambda p: np.loadtxt(p, skiprows=1, usecols=(1, 2))
        quality, jaccard = load(f"{prefix}.quality.tsv"), load(f"{prefix}.jaccard.tsv")
        assert np.allclose(quality, jaccard)
        assert np.allclose(jaccard, [[5000 / 9000, 0.0], [0.4, 0.2]], atol=1e-4)


def test_jaccard_matrix():
    overlap = {("E1", "A"): 5000, ("E2", "A"): 4000, ("E2", "B"): 1000}
    mat = match.jaccard_matrix(overlap, {"E1": 5000, "E2": 5000}, {"A": 9000, "B": 1000},
                               ["E1", "E2"], ["A", "B"])
    assert np.allclose(mat, [[5000 / 9000, 0.0], [0.4, 0.2]])


def test_match_states_concave_scoring_lifts_worst_off_state():
    # Demonstrates that concave utility prioritizes lifting the worst-off state (E2)
    # rather than letting E1 greedily capture A for a minor gain while exiling E2 to ~0 overlap.
    ref_segs = [
        ("chr1", 0, 1000, "A", "0,0,0"),
        ("chr1", 1000, 2000, "B", "0,0,0"),
    ]
    work_segs = [
        ("chr1", 0, 800, "E1", "0,0,0"),
        ("chr1", 1000, 1600, "E1", "0,0,0"),
        ("chr1", 800, 1000, "E2", "0,0,0"),
        ("chr1", 1600, 1610, "E2", "0,0,0"),
        ("chr1", 2000, 2790, "E2", "0,0,0"),
    ]
    mapping = match.match_states([ref_segs], [work_segs])
    assert mapping == {"E1": "B", "E2": "A"}


def test_match_states_separate_sqrt_jaccard_with_emissions():
    # E1 is a focal state with low raw Jaccard (0.04 vs 0.0) with A, and high emission with A (1.0 vs 0.0).
    # E2 is a broad state with moderate Jaccard with B.
    # Checks that utility combines 0.8 * sqrt(J) + 0.2 * E appropriately.
    ref_segs = [
        ("chr1", 0, 100, "Focal_Ref", "0,0,0"),
        ("chr1", 100, 10000, "Broad_Ref", "0,0,0"),
    ]
    work_segs = [
        ("chr1", 0, 100, "E_focal", "0,0,0"),
        ("chr1", 100, 10000, "E_broad", "0,0,0"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        work_em = os.path.join(tmpdir, "work.npz")
        ref_em = os.path.join(tmpdir, "ref.npz")
        match._save_emissions_npz(work_em, ["E_focal", "E_broad"], ["H3K4me3"],
                                  np.array([[1.0], [0.0]]))
        match._save_emissions_npz(ref_em, ["Focal_Ref", "Broad_Ref"], ["H3K4me3"],
                                  np.array([[1.0], [0.0]]))
        mapping = match.match_states([ref_segs], [work_segs],
                                     work_emissions_path=work_em, ref_emissions_path=ref_em)
        assert mapping == {"E_focal": "Focal_Ref", "E_broad": "Broad_Ref"}
