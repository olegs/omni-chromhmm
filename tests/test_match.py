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


def test_jaccard_matrix_properties():
    ref_segs = [
        ("chr1", 0, 1000, "Tss", "255,0,0"),
        ("chr1", 1000, 5000, "Tx", "0,255,0"),
        ("chr1", 5000, 10000, "Quies", "255,255,255"),
    ]
    work_segs = [
        ("chr1", 0, 1000, "Tss", "255,0,0"),
        ("chr1", 1000, 5000, "Tx", "0,255,0"),
        ("chr1", 5000, 10000, "Quies", "255,255,255"),
    ]
    rl = match.state_lengths(ref_segs)
    wl = match.state_lengths(work_segs)
    overlap = match.pair_overlap(ref_segs, work_segs)
    states = ["Tss", "Tx", "Quies"]

    jaccard = match.jaccard_matrix(overlap, wl, rl, states, states)
    for i in range(len(states)):
        assert np.isclose(jaccard[i, i], 1.0)
    assert jaccard[0, 1] == 0.0
    assert jaccard[0, 2] == 0.0
    assert jaccard[1, 2] == 0.0


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
    assert "npmi=" not in captured.err
    assert "emission=" in captured.err
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


def test_profile_overlap_and_cosine():
    mat1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mat2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 0.0]])
    sim = match.emission_similarity(mat1, mat2)
    assert np.isclose(sim[0, 0], 1.0)
    assert np.isclose(sim[1, 1], 0.5)
    # Empty row falls back to augmented cosine
    assert sim[2, 2] > 0.9


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
        assert os.path.exists(f"{prefix}.score.tsv")
        assert os.path.exists(f"{prefix}.jaccard.tsv")
        assert not os.path.exists(f"{prefix}.npmi.tsv")
        assert os.path.exists(f"{prefix}.quality.tsv")
        assert os.path.exists(f"{prefix}.mapping.tsv")
        assert os.path.exists(f"{prefix}.png")

        with open(f"{prefix}.mapping.tsv") as f:
            header = f.readline().strip().split("\t")
            assert "jaccard" in header
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
        assert os.path.exists(f"{prefix}.score.tsv")
        assert os.path.exists(f"{prefix}.jaccard.tsv")
        assert not os.path.exists(f"{prefix}.npmi.tsv")
        assert os.path.exists(f"{prefix}.quality.tsv")
        assert os.path.exists(f"{prefix}.mapping.tsv")
        assert os.path.exists(f"{prefix}.png")

        captured = capsys.readouterr()
        assert "jaccard=" in captured.err
        assert "npmi=" not in captured.err
        assert "emission=" in captured.err
        assert "quality=" in captured.err

        with open(f"{prefix}.mapping.tsv") as f:
            header = f.readline().strip().split("\t")
            assert "jaccard" in header
            assert "npmi" not in header
            assert "quality" in header
