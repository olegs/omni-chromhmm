import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis")))
import summary_plots


def test_state_palette_numbered_states_tab20():
    states = ["1", "2", "10", "11", "12", "E1", "E2", "E10", "E11", "E12"]
    palette = summary_plots.state_palette(states)
    cmap = plt.get_cmap("tab20")

    def expected_hex(num):
        r, g, b = cmap((num - 1) % 20)[:3]
        return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

    for num in [1, 2, 10, 11, 12]:
        assert palette[str(num)] == expected_hex(num)
        assert palette[f"E{num}"] == expected_hex(num)

    # State 10, 11, 12 should not have the same color as State 1
    assert palette["11"] != palette["1"]
    assert palette["12"] != palette["1"]
    assert palette["E11"] != palette["E1"]
    assert palette["E12"] != palette["E1"]


def test_unknown_state_handling(tmp_path):
    states = ["Tss", "Tx", "Quies", "Unknown"]
    sorted_states = summary_plots.sort_states(states)
    assert sorted_states[-1] == "Unknown"
    assert sorted_states[0] == "Tss"

    palette = summary_plots.state_palette(states)
    assert "Unknown" in palette
    assert palette["Unknown"] == "#000000"

    coverages = {
        "dataset_a": {"Tss": 0.2, "Tx": 0.3, "Quies": 0.5},
        "dataset_b": {"Tss": 0.1, "Tx": 0.2, "Quies": 0.6, "Unknown": 0.1},
    }
    outfile = str(tmp_path / "composition_test.png")
    summary_plots._stacked_composition_chart(
        coverages, ["dataset_a", "dataset_b"], "Test Composition", outfile
    )
    assert os.path.exists(outfile)


def test_plot_method_composition_with_custom_title(tmp_path):
    # Mock reference bed file
    bed_path = tmp_path / "ref_chromhmm.bed"
    with open(bed_path, "w") as f:
        f.write("chr1\t0\t1000\tTss\t0\t.\t0\t1000\t255,0,0\n")
        f.write("chr1\t1000\t5000\tQuies\t0\t.\t1000\t5000\t220,220,220\n")

    outfile = str(tmp_path / "method_composition.png")
    summary_plots.run_summary_plots(
        datasets=["test_ds"],
        cells=["test_cell"],
        workdir=str(tmp_path),
        ref_paths=[str(bed_path)],
        methods=["ref"],
        method_composition_outfile=outfile,
        method_composition_title="State composition per method — mean across datasets (ChIP-seq)",
    )
    assert os.path.exists(outfile)


def test_plot_per_dataset_method_composition_reference(tmp_path):
    ds_dir = tmp_path / "ds1"
    ds_dir.mkdir()
    bed_path = ds_dir / "ENCFF123_chromhmm.bed"
    with open(bed_path, "w") as f:
        f.write("chr1\t0\t1000\tTss\t0\t.\t0\t1000\t255,0,0\n")
        f.write("chr1\t1000\t5000\tQuies\t0\t.\t1000\t5000\t220,220,220\n")

    outdir = str(tmp_path / "out")
    summary_plots.run_summary_plots(
        datasets=["ds1"],
        cells=["Cell1"],
        workdir=str(tmp_path),
        ref_paths=[str(bed_path)],
        method_ds_composition_outdir=outdir,
    )
    ref_file = os.path.join(outdir, "method_ds_composition_reference.png")
    assert os.path.exists(ref_file)


def test_plot_reference_distribution_combined(tmp_path):
    # Mock similarity matrix TSVs
    matrix_content = "cell\tCellA\tCellB\nCellA\t1.0\t0.8\nCellB\t0.8\t1.0\n"
    comp_file = tmp_path / "comp.tsv"
    kappa_file = tmp_path / "kappa.tsv"
    jaccard_file = tmp_path / "jaccard.tsv"
    comp_noqh_file = tmp_path / "comp_noqh.tsv"
    kappa_noqh_file = tmp_path / "kappa_noqh.tsv"
    jaccard_noqh_file = tmp_path / "jaccard_noqh.tsv"

    for f in [comp_file, kappa_file, jaccard_file, comp_noqh_file, kappa_noqh_file, jaccard_noqh_file]:
        with open(f, "w") as fp:
            fp.write(matrix_content)

    outfile = str(tmp_path / "similarity_distribution_combined.png")
    summary_plots.run_summary_plots(
        ref_comp_matrix=str(comp_file),
        ref_kappa_matrix=str(kappa_file),
        ref_jaccard_matrix=str(jaccard_file),
        ref_comp_noqh_matrix=str(comp_noqh_file),
        ref_kappa_noqh_matrix=str(kappa_noqh_file),
        ref_jaccard_noqh_matrix=str(jaccard_noqh_file),
        ref_dist_combined_outfile=outfile,
    )
    assert os.path.exists(outfile)
