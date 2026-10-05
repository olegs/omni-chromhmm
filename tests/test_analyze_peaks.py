import gzip
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis")))

import analyze_peaks


def test_fast_count_marks():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create uncompressed binary file
        txt_path = os.path.join(tmpdir, "test_binary.txt")
        lines = [
            "cell1\tchr1\n",
            "H3K4me3\tH3K27ac\tH3K4me1\n",
            "0\t0\t0\n",  # 0 marks
            "1\t0\t0\n",  # 1 mark
            "1\t1\t0\n",  # 2 marks
            "1\t1\t1\n",  # 3 marks
            "0\t1\t0\n",  # 1 mark
        ]
        with open(txt_path, "w") as f:
            f.writelines(lines)

        counts = analyze_peaks.fast_count_marks(txt_path)
        # 0 marks: 1, 1 mark: 2, 2 marks: 1, 3 marks: 1
        assert len(counts) == 4
        assert counts[0] == 1
        assert counts[1] == 2
        assert counts[2] == 1
        assert counts[3] == 1

        # Create gzipped binary file
        gz_path = os.path.join(tmpdir, "test_binary.txt.gz")
        with gzip.open(gz_path, "wt") as f:
            f.writelines(lines)

        counts_gz = analyze_peaks.fast_count_marks(gz_path)
        assert np.array_equal(counts, counts_gz)


def test_binarization_mark_coverage_and_plot():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = "sample1"
        cell = "SampleCell"
        ds_dir = os.path.join(tmpdir, ds)
        # ChromHMM default dir
        ch_dir = os.path.join(ds_dir, "chromhmm_default")
        os.makedirs(ch_dir, exist_ok=True)
        ch_file = os.path.join(ch_dir, f"{cell}_chr1_binary.txt")
        lines_ch = [
            f"{cell}\tchr1\n",
            "H3K4me3\tH3K27ac\tH3K4me1\n",
            "0\t0\t0\n",
            "1\t0\t0\n",
            "1\t1\t0\n",
            "1\t1\t1\n",
        ]
        with open(ch_file, "w") as f:
            f.writelines(lines_ch)

        # OmniPeak dir
        omni_dir = os.path.join(ds_dir, "omni", "chromhmm_peaks")
        os.makedirs(omni_dir, exist_ok=True)
        omni_file = os.path.join(omni_dir, "chr1_binary.txt.gz")
        lines_omni = [
            f"{cell}\tchr1\n",
            "H3K4me3\tH3K27ac\tH3K4me1\n",
            "1\t0\t0\n",
            "1\t1\t0\n",
            "1\t1\t1\n",
            "1\t1\t1\n",
        ]
        with gzip.open(omni_file, "wt") as f:
            f.writelines(lines_omni)

        df = analyze_peaks.binarization_mark_coverage(
            ds, cell, methods=["ChromHMM", "OmniPeak"], workdir=tmpdir
        )
        assert not df.empty
        assert set(df["method"]) == {"ChromHMM", "OmniPeak"}
        assert set(df["N"]) == {1, 2, 3}

        # Check normalization at N=1
        for m in ["ChromHMM", "OmniPeak"]:
            sub = df[(df["method"] == m) & (df["N"] == 1)]
            assert sub["fraction"].iloc[0] == 1.0

        # Plot absolute and relative separately
        plot_abs_path = os.path.join(tmpdir, "plot_abs.png")
        fig_abs, ax_abs = analyze_peaks.plot_mark_coverage_absolute(df, outfile=plot_abs_path)
        assert fig_abs is not None
        assert ax_abs is not None
        assert os.path.exists(plot_abs_path)
        assert ax_abs.get_yscale() == "log"

        plot_rel_path = os.path.join(tmpdir, "plot_rel.png")
        fig_rel, ax_rel = analyze_peaks.plot_mark_coverage_relative(df, outfile=plot_rel_path)
        assert fig_rel is not None
        assert ax_rel is not None
        assert os.path.exists(plot_rel_path)
        assert ax_rel.get_yscale() == "linear"

        # Test summary_plots integration with mark_coverage_outfile
        import summary_plots
        summary_out = os.path.join(tmpdir, "summary_cov.png")
        summary_plots.run_summary_plots(
            datasets=[ds],
            cells=[cell],
            workdir=tmpdir,
            mark_coverage_outfile=summary_out
        )
        assert os.path.exists(summary_out)

        # Test summary_plots integration with outdir (same folder as summary_2way_tss_exptss.png)
        summary_dir = os.path.join(tmpdir, "summary_plots")
        analysis_dir = os.path.join(tmpdir, "analysis_dir")
        os.makedirs(analysis_dir, exist_ok=True)
        # Create dummy comparison table
        comp_df = pd.DataFrame([{"method": "chromhmm_default", "display_name": "Default ChromHMM",
                                 "binarization": "default", "state_model": "chromhmm", "replicate": None,
                                 "sensitivity_Tss_ExpressedTSS2kb": 0.5, "coverage_Tss_ExpressedTSS2kb": 0.5}])
        comp_df.to_csv(os.path.join(analysis_dir, "comparison_table.tsv"), sep="\t", index=False)

        summary_plots.run_summary_plots(
            datasets=[ds],
            methods_dirs=[analysis_dir],
            analysis_dirs=[analysis_dir],
            outdir=summary_dir,
            workdir=tmpdir,
            cells=[cell]
        )
        assert os.path.exists(os.path.join(summary_dir, "summary_2way_tss_exptss.png"))
        assert os.path.exists(os.path.join(summary_dir, "binarization_mark_coverage.png"))
        assert os.path.exists(os.path.join(summary_dir, "binarization_mark_coverage_absolute.png"))
        assert os.path.exists(os.path.join(summary_dir, "binarization_mark_coverage_relative.png"))


def test_binarization_mark_coverage_replicates():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = "rep_sample"
        cell = "MCF7"
        ds_dir = os.path.join(tmpdir, ds)
        # Replicate 1 ChromHMM dir
        ch_dir = os.path.join(ds_dir, "rep1", "chromhmm_default")
        os.makedirs(ch_dir, exist_ok=True)
        ch_file = os.path.join(ch_dir, f"{cell}_chr1_binary.txt")
        lines_ch = [
            f"{cell}\tchr1\n",
            "H3K4me3\tH3K27ac\tH3K4me1\n",
            "0\t0\t0\n",
            "1\t0\t0\n",
            "1\t1\t0\n",
            "1\t1\t1\n",
        ]
        with open(ch_file, "w") as f:
            f.writelines(lines_ch)

        # Replicate 2 OmniPeak dir
        omni_dir = os.path.join(ds_dir, "rep2", "omni", "chromhmm_peaks")
        os.makedirs(omni_dir, exist_ok=True)
        omni_file = os.path.join(omni_dir, "chr1_binary.txt.gz")
        lines_omni = [
            f"{cell}\tchr1\n",
            "H3K4me3\tH3K27ac\tH3K4me1\n",
            "1\t0\t0\n",
            "1\t1\t0\n",
            "1\t1\t1\n",
        ]
        with gzip.open(omni_file, "wt") as f:
            f.writelines(lines_omni)

        df = analyze_peaks.binarization_mark_coverage(
            ds, cell, methods=["ChromHMM", "OmniPeak"], workdir=tmpdir
        )
        assert not df.empty
        assert set(df["method"]) == {"ChromHMM", "OmniPeak"}
        assert set(df["N"]) == {1, 2, 3}

        # Test summary_plots with replicate structure
        import summary_plots
        summary_dir = os.path.join(tmpdir, "out", "summary_plots")
        os.makedirs(summary_dir, exist_ok=True)
        summary_plots._plot_mark_coverage(
            [ds], tmpdir, os.path.join(summary_dir, "binarization_mark_coverage_absolute.png"),
            cells={ds: cell}, relative=False
        )
        assert os.path.exists(os.path.join(summary_dir, "binarization_mark_coverage_absolute.png"))

        # Test summary_plots._plot_mark_coverage with DataFrame directly
        df_out = os.path.join(summary_dir, "df_cov_rel.png")
        summary_plots._plot_mark_coverage(df, tmpdir, df_out, relative=True)
        assert os.path.exists(df_out)


def test_binarization_mark_coverage_chromhmm_binary_and_peak_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = "E001"
        cell = "E001"
        ds_dir = os.path.join(tmpdir, ds)
        # ChromHMM in chromhmm_binary folder
        ch_dir = os.path.join(ds_dir, "chromhmm_binary")
        os.makedirs(ch_dir, exist_ok=True)
        ch_file = os.path.join(ch_dir, f"{cell}_chr1_binary.txt.gz")
        lines_ch = [
            f"{cell}\tchr1\n",
            "H3K4me3\tH3K27ac\tH3K4me1\n",
            "0\t0\t0\n",
            "1\t0\t0\n",
            "1\t1\t0\n",
            "1\t1\t1\n",
        ]
        with gzip.open(ch_file, "wt") as f:
            f.writelines(lines_ch)

        # HOMER raw peak files
        homer_dir = os.path.join(ds_dir, "homer")
        os.makedirs(homer_dir, exist_ok=True)
        with open(os.path.join(homer_dir, f"{cell}-H3K4me3_homer.bed"), "w") as f:
            f.write("chr1\t100\t200\nchr1\t300\t400\n")
        with open(os.path.join(homer_dir, f"{cell}-H3K27ac_homer.bed"), "w") as f:
            f.write("chr1\t150\t250\n")

        # Mock chrom.sizes
        chromsizes_path = os.path.join(tmpdir, "hg19.chrom.sizes")
        with open(chromsizes_path, "w") as f:
            f.write("chr1\t1000\n")

        df = analyze_peaks.binarization_mark_coverage(
            ds, cell, methods=["ChromHMM", "HOMER"], workdir=tmpdir
        )
        assert not df.empty
        assert set(df["method"]) == {"ChromHMM", "HOMER"}
        assert 1 in df[df["method"] == "ChromHMM"]["N"].values
        assert 1 in df[df["method"] == "HOMER"]["N"].values
