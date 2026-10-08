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


def test_bigwig_presence_is_on_the_binarized_scale():
    # Fold-enrichment signal over a 0.25 background (state 3).
    bw = np.array([[25.0, 0.3], [4.0, 0.25], [0.25, 10.0]])
    presence = match.bigwig_presence(bw)
    assert presence[0, 0] > 0.9          # 100x: called present almost everywhere
    assert 0.7 < presence[2, 1] < 0.9    # 40x
    assert 0.4 < presence[1, 0] < 0.6    # 16x: around half
    assert presence[2, 0] < 0.05         # background
    assert presence[1, 1] < 0.05


def test_emission_similarity_bigwig_vs_binarized():
    marks = ["H3K4me3", "H3K27me3"]
    binarized = np.array([[0.95, 0.01], [0.02, 0.90], [0.01, 0.02]])     # Tss, ReprPC, Quies
    bigwig = np.array([[0.4, 0.3], [30.0, 0.35], [0.45, 12.0]])           # Quies, Tss, ReprPC
    sim = match.emission_similarity(binarized, bigwig, binarized_w=True, binarized_r=False)
    assert list(sim.argmax(axis=1)) == [1, 2, 0]
    assert sim[2, 0] > 0.95
    # Max scaling alone would leave a 4x-over-background state looking half marked.
    assert match.emission_similarity(np.array([[0.0, 0.0]]), np.array([[1.0, 0.25], [4.0, 0.25]]),
                                     binarized_w=True, binarized_r=False)[0, 1] > 0.95


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
