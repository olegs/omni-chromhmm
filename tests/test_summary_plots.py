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
    outfile = str(tmp_path / "composition_test.pdf")
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

    outfile = str(tmp_path / "method_composition.pdf")
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
    ref_file = os.path.join(outdir, "method_ds_composition_reference.pdf")
    assert os.path.exists(ref_file)


def test_plot_reference_distribution(tmp_path):
    # Mock similarity matrix TSVs
    matrix_content = "cell\tCellA\tCellB\nCellA\t1.0\t0.8\nCellB\t0.8\t1.0\n"
    comp_file = tmp_path / "comp.tsv"
    kappa_file = tmp_path / "kappa.tsv"
    jaccard_file = tmp_path / "jaccard.tsv"

    for f in [comp_file, kappa_file, jaccard_file]:
        with open(f, "w") as fp:
            fp.write(matrix_content)

    outfile = str(tmp_path / "similarity_distribution.pdf")
    summary_plots.run_summary_plots(
        ref_comp_matrix=str(comp_file),
        ref_kappa_matrix=str(kappa_file),
        ref_jaccard_matrix=str(jaccard_file),
        ref_dist_outfile=outfile,
    )
    assert os.path.exists(outfile)


def test_plot_method_similarity_distribution(tmp_path):
    method_dir = tmp_path / "chromhmm"
    method_dir.mkdir()
    matrix_content = "\tds1:chromhmm\tds2:chromhmm\nds1:chromhmm\t1.0\t0.75\nds2:chromhmm\t0.75\t1.0\n"
    for filename in ["composition_similarity_matrix.tsv", "kappa_matrix.tsv", "jaccard_similarity_matrix.tsv"]:
        with open(method_dir / filename, "w") as f:
            f.write(matrix_content)

    outfile = str(tmp_path / "method_sim.pdf")
    summary_plots._plot_method_similarity_distribution(
        methods=["chromhmm"],
        inter_ds_dir=str(tmp_path),
        outfile=outfile,
    )
    assert os.path.exists(outfile)


def test_plot_replicate_consistency_bars(tmp_path):
    ds1_dir = tmp_path / "ds1"
    ds1_dir.mkdir()
    comp_table = ds1_dir / "comparison_table.tsv"
    with open(comp_table, "w") as f:
        f.write("method\tdisplay_name\treplicate\tkappa_rep1_vs_rep2\tjaccard_rep1_vs_rep2\n")
        f.write("chromhmm_default\tChromHMM\t\t0.85\t0.80\n")

    outdir = str(tmp_path / "out")
    summary_plots._plot_rep_consistency(
        datasets=["ds1"],
        methods_dirs=[str(ds1_dir)],
        outdir=outdir,
    )
    assert os.path.exists(os.path.join(outdir, "rep_consistency_kappa_rep1_vs_rep2.pdf"))
    assert os.path.exists(os.path.join(outdir, "rep_consistency_jaccard_rep1_vs_rep2.pdf"))


def test_collect_per_state_metrics_keeps_pooled_vs_ref(tmp_path):
    comp_dir = tmp_path / "ds1" / "comparison"
    comp_dir.mkdir(parents=True)
    with open(comp_dir / "per_state_metrics.tsv", "w") as f:
        f.write("seg1\tseg2\tstate\tkappa\tjaccard\n")
        f.write("ENCFF000AAA\tchromhmm_default\tTssA\t0.8\t0.7\n")
        f.write("ENCFF000AAA\tchromhmm_default_rep1\tTssA\t0.6\t0.5\n")
        f.write("kmeans_omni\tENCFF000AAA\tTssA\t0.4\t0.3\n")
        f.write("chromhmm_default\tkmeans_omni\tTssA\t0.2\t0.1\n")

    df = summary_plots._collect_per_state_metrics(
        ["ds1"], [str(tmp_path / "ds1")], "jaccard")
    assert sorted(df["method"]) == ["chromhmm_default", "kmeans_omni"]
    assert df.set_index("method").loc["kmeans_omni", "kappa"] == 0.4


def test_2way_scatter_labels(tmp_path, monkeypatch):
    import pandas as pd
    x_data = pd.DataFrame({"ds1": [50.0, 60.0]}, index=["ref", "chromhmm_default"])
    y_data = pd.DataFrame({"ds1": [40.0, 70.0]}, index=["ref", "chromhmm_default"])
    outfile = str(tmp_path / "test_2way.pdf")

    captured_labels = {}
    orig_save = summary_plots.save_fig

    def mock_save_fig(fig, outpath, **kwargs):
        ax = fig.axes[0]
        captured_labels["xlabel"] = ax.get_xlabel()
        captured_labels["ylabel"] = ax.get_ylabel()
        orig_save(fig, outpath, **kwargs)

    monkeypatch.setattr(summary_plots, "save_fig", mock_save_fig)

    summary_plots._plot_2way_scatter(
        x_data, y_data,
        "Test Title",
        "Expressed gene bodies bp overlap (Sensitivity, %)",
        "Tx state bp overlap (Precision, %)",
        outfile
    )
    assert os.path.exists(outfile)
    assert captured_labels["xlabel"] == "Expressed gene bodies bp overlap (Sensitivity, %)"
    assert captured_labels["ylabel"] == "Tx state bp overlap (Precision, %)"
