import csv
import gzip
import os
import tempfile
import pytest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "rules")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "analysis")))

import analyze
import utils


def write_rnaseq_tsv(path, rows):
    """Helper to write RNA-seq TSV fixture."""
    fieldnames = ["gene_id", "transcript_id(s)", "length", "effective_length",
                  "expected_count", "TPM", "FPKM", "IsoPct"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_gtf(path, entries, gzipped=False):
    """Helper to write GTF fixture."""
    lines = ["# Synthetic GTF header\n"]
    for e in entries:
        attrs = f'gene_id "{e["gene_id"]}"; gene_name "{e["gene_name"]}"; gene_type "protein_coding";'
        line = f'{e["chrom"]}\tHAVANA\tgene\t{e["start"]}\t{e["end"]}\t.\t{e["strand"]}\t.\t{attrs}\n'
        lines.append(line)
    
    if gzipped:
        with gzip.open(path, "wt") as f:
            f.writelines(lines)
    else:
        with open(path, "w") as f:
            f.writelines(lines)


def test_load_gene_tpms_ranking_and_top_n():
    with tempfile.TemporaryDirectory() as tmpdir:
        rnaseq_file = os.path.join(tmpdir, "rnaseq.tsv")
        rows = [
            {"gene_id": "ENSG00000001.1", "TPM": "50.5"},
            {"gene_id": "ENSG00000002.2", "TPM": "10.0"},
            {"gene_id": "ENSG00000003.1", "TPM": "100.2"},
            {"gene_id": "ENSG00000004.5", "TPM": "0.0"},
            {"gene_id": "ENSG00000005.1", "TPM": "0.05"},
            {"gene_id": "INVALID", "TPM": "N/A"},
        ]
        write_rnaseq_tsv(rnaseq_file, rows)

        # Select top 2 genes
        tpms, top_exp = analyze.load_gene_tpms(rnaseq_file, top_n=2)
        # Top 2 should be ENSG00000003.1 (100.2) and ENSG00000001.1 (50.5)
        assert "ENSG00000003.1" in top_exp
        assert "ENSG00000003" in top_exp
        assert "ENSG00000001.1" in top_exp
        assert "ENSG00000001" in top_exp
        # ENSG00000002.2 ranked 3rd so not in top 2
        assert "ENSG00000002.2" not in top_exp
        assert "ENSG00000002" not in top_exp
        # Zero TPM gene must never be in top expressed
        assert "ENSG00000004.5" not in top_exp

        # Check TPM lookup map
        assert tpms["ENSG00000003.1"] == pytest.approx(100.2)
        assert tpms["ENSG00000003"] == pytest.approx(100.2)
        assert tpms["ENSG00000004.5"] == pytest.approx(0.0)


def test_load_gene_tpms_zero_tpm_cutoff():
    with tempfile.TemporaryDirectory() as tmpdir:
        rnaseq_file = os.path.join(tmpdir, "rnaseq.tsv")
        rows = [
            {"gene_id": "ENSG00000001.1", "TPM": "1.5"},
            {"gene_id": "ENSG00000002.1", "TPM": "0.0"},
        ]
        write_rnaseq_tsv(rnaseq_file, rows)

        # Requesting top 12k when only 1 gene has TPM > 0
        tpms, top_exp = analyze.load_gene_tpms(rnaseq_file, top_n=12000)
        assert "ENSG00000001.1" in top_exp
        assert "ENSG00000001" in top_exp
        assert "ENSG00000002.1" not in top_exp
        assert "ENSG00000002" not in top_exp


def test_load_gene_coords_top_expressed_and_nonexpressed():
    with tempfile.TemporaryDirectory() as tmpdir:
        gtf_file = os.path.join(tmpdir, "genes.gtf")
        entries = [
            # Expressed + strand (1-based GTF: 1000..2000 -> 0-based: 999..2000, TSS=999)
            {"gene_id": "ENSG00000001.1", "gene_name": "GENE1", "chrom": "chr1",
             "start": 1000, "end": 2000, "strand": "+"},
            # Expressed - strand (1-based GTF: 3000..4000 -> 0-based: 2999..4000, TSS=3999)
            {"gene_id": "ENSG00000002.5", "gene_name": "GENE2", "chrom": "chr1",
             "start": 3000, "end": 4000, "strand": "-"},
            # Non-expressed (TPM <= 0.1)
            {"gene_id": "ENSG00000003.1", "gene_name": "GENE3", "chrom": "chr2",
             "start": 5000, "end": 6000, "strand": "+"},
            # Intermediate TPM (not in top expressed, but TPM > 0.1: should be excluded from both)
            {"gene_id": "ENSG00000004.1", "gene_name": "GENE4", "chrom": "chr2",
             "start": 7000, "end": 8000, "strand": "+"},
        ]
        write_gtf(gtf_file, entries)

        gene_tpms = {
            "ENSG00000001.1": 50.0, "ENSG00000001": 50.0,
            "ENSG00000002.5": 20.0, "ENSG00000002": 20.0,
            "ENSG00000003.1": 0.05, "ENSG00000003": 0.05,
            "ENSG00000004.1": 5.0, "ENSG00000004": 5.0,
        }
        top_exp_ids = {"ENSG00000001.1", "ENSG00000001", "ENSG00000002.5", "ENSG00000002"}

        exp_b, exp_t, nonexp_b, nonexp_t = analyze.load_gene_coords(
            gtf_file, gene_tpms, top_expressed_ids=top_exp_ids, nonexp_thresh=0.1
        )

        assert len(exp_b) == 2
        assert len(exp_tss := exp_t) == 2
        assert len(nonexp_b) == 1
        assert len(nonexp_t) == 1

        # Check coordinate conversion
        assert exp_b[0] == ("chr1", 999, 2000, "GENE1")
        assert exp_tss[0] == ("chr1", 999, 1000, "GENE1")

        assert exp_b[1] == ("chr1", 2999, 4000, "GENE2")
        assert exp_tss[1] == ("chr1", 3999, 4000, "GENE2")

        assert nonexp_b[0] == ("chr2", 4999, 6000, "GENE3")
        assert nonexp_t[0] == ("chr2", 4999, 5000, "GENE3")


def test_load_gene_coords_version_independent_matching():
    with tempfile.TemporaryDirectory() as tmpdir:
        gtf_file = os.path.join(tmpdir, "genes.gtf")
        # GTF has unversioned ID, RNA-seq had versioned ID
        entries = [
            {"gene_id": "ENSG00000099", "gene_name": "GENE99", "chrom": "chr1",
             "start": 100, "end": 500, "strand": "+"},
        ]
        write_gtf(gtf_file, entries)

        # Top expressed has base and versioned IDs
        top_exp_ids = {"ENSG00000099.3", "ENSG00000099"}
        gene_tpms = {"ENSG00000099.3": 40.0, "ENSG00000099": 40.0}

        exp_b, exp_t, nonexp_b, nonexp_t = analyze.load_gene_coords(
            gtf_file, gene_tpms, top_expressed_ids=top_exp_ids
        )
        assert len(exp_b) == 1
        assert exp_b[0] == ("chr1", 99, 500, "GENE99")


def test_load_gene_coords_fallback_threshold():
    with tempfile.TemporaryDirectory() as tmpdir:
        gtf_file = os.path.join(tmpdir, "genes.gtf")
        entries = [
            {"gene_id": "ENSG00000001", "gene_name": "G1", "chrom": "chr1",
             "start": 100, "end": 500, "strand": "+"},
            {"gene_id": "ENSG00000002", "gene_name": "G2", "chrom": "chr1",
             "start": 600, "end": 900, "strand": "+"},
        ]
        write_gtf(gtf_file, entries)

        gene_tpms = {"ENSG00000001": 2.5, "ENSG00000002": 0.05}
        exp_b, exp_t, nonexp_b, nonexp_t = analyze.load_gene_coords(
            gtf_file, gene_tpms, top_expressed_ids=None, exp_thresh=1.0, nonexp_thresh=0.1
        )
        assert len(exp_b) == 1
        assert exp_b[0][3] == "G1"
        assert len(nonexp_b) == 1
        assert nonexp_b[0][3] == "G2"


def test_gene_name_matching():
    with tempfile.TemporaryDirectory() as tmpdir:
        gtf_file = os.path.join(tmpdir, "genes.gtf")
        entries = [
            {"gene_id": "NOVEL_ID", "gene_name": "KNOWN_NAME", "chrom": "chr1",
             "start": 100, "end": 500, "strand": "+"},
        ]
        write_gtf(gtf_file, entries)

        # Matched via gene_name
        top_exp_ids = {"KNOWN_NAME"}
        gene_tpms = {"KNOWN_NAME": 15.0}

        exp_b, exp_t, nonexp_b, nonexp_t = analyze.load_gene_coords(
            gtf_file, gene_tpms, top_expressed_ids=top_exp_ids
        )
        assert len(exp_b) == 1
        assert exp_b[0][3] == "KNOWN_NAME"


def test_make_expressed_annotations_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        rnaseq_file = os.path.join(tmpdir, "rnaseq.tsv")
        gtf_file = os.path.join(tmpdir, "genes.gtf.gz")

        rows = [
            {"gene_id": "ENSG00000001.1", "TPM": "80.0"},
            {"gene_id": "ENSG00000002.1", "TPM": "0.0"},
        ]
        write_rnaseq_tsv(rnaseq_file, rows)

        entries = [
            {"gene_id": "ENSG00000001.1", "gene_name": "ACTB", "chrom": "chr1",
             "start": 10000, "end": 20000, "strand": "+"},
            {"gene_id": "ENSG00000002.1", "gene_name": "SILENT", "chrom": "chr1",
             "start": 30000, "end": 40000, "strand": "-"},
        ]
        write_gtf(gtf_file, entries, gzipped=True)

        annotations = analyze.make_expressed_annotations(
            rnaseq_file, gtf_file, top_expressed=12000, nonexp_thresh=0.1
        )

        labels = [name for name, _ in annotations]
        assert "ExpressedGeneBodies" in labels
        assert "ExpressedTSS" in labels
        assert "ExpressedTSS2kb" in labels
        assert "NonExpressedGeneBodies" in labels
        assert "NonExpressedTSS" in labels
        assert "NonExpressedTSS2kb" in labels

        annot_dict = dict(annotations)
        # Check ExpressedTSS2kb window: TSS interval [9999, 10000) extended by 2000 bp -> [7999, 12000)
        assert annot_dict["ExpressedTSS2kb"][0] == ("chr1", 7999, 12000, "ACTB")
        # Check NonExpressedTSS2kb window: TSS interval [39999, 40000) extended by 2000 bp -> [37999, 42000)
        assert annot_dict["NonExpressedTSS2kb"][0] == ("chr1", 37999, 42000, "SILENT")


def test_state_predicates():
    assert utils.is_promoter_core("Tss")
    assert utils.is_promoter_core("1_TssA")
    assert utils.is_promoter_core("TssFlnkU")
    assert not utils.is_promoter_core("TssFlnk")
    assert not utils.is_promoter_core("TssFlnkD")

    assert utils.is_promoter_flank("TssFlnk")
    assert utils.is_promoter_flank("TssFlnkD")
    assert not utils.is_promoter_flank("Tss")

    assert utils.is_promoter("Tss")
    assert utils.is_promoter("1_TssA")
    assert utils.is_promoter("TssFlnk")
    assert not utils.is_promoter("Tx")
    assert not utils.is_promoter("Enh")

    assert utils.is_tx("Tx")
    assert utils.is_tx("2_TxA")
    assert utils.is_tx("TxWk")
    assert not utils.is_tx("Tss")

    assert utils.is_tx_core("Tx")
    assert utils.is_tx_core("TxA")
    assert not utils.is_tx_core("TxWk")
    assert not utils.is_tx_core("EnhG1")

    assert utils.is_enhancer("Enh")
    assert utils.is_enhancer("5_EnhA")
    assert utils.is_enhancer("EnhG1")
    assert not utils.is_enhancer("EnhBiv")
    assert not utils.is_enhancer("Tss")

    assert utils.is_distal_enhancer("Enh")
    assert utils.is_distal_enhancer("5_EnhA")
    assert utils.is_distal_enhancer("Enh1")
    assert not utils.is_distal_enhancer("EnhG1")

    assert utils.is_genic_enhancer("EnhG")
    assert utils.is_genic_enhancer("EnhG1")
    assert not utils.is_genic_enhancer("Enh1")

    assert utils.is_active("1_TssA")
    assert utils.is_active("EnhG")
    assert not utils.is_active("Tx")
    assert not utils.is_active("Quies")

    assert utils.is_active_open("1_TssA")
    assert utils.is_active_open("5_EnhA")
    assert utils.is_active_open("Enh1")
    assert not utils.is_active_open("EnhG1")
    assert not utils.is_active_open("TssFlnkD")

    assert utils.is_biv("TssBiv")
    assert utils.is_biv("10_Biv")
    assert utils.is_biv("EnhBiv")
    assert not utils.is_biv("Tss")


def test_compute_enrichment_50_percent_overlap():
    # Segments
    # Seg1: chr1 1000..1600 (1_TssA, 600bp)
    # Seg2: chr1 3000..3400 (1_TssA, 400bp)
    # Seg3: chr1 3400..3800 (2_TssAFlnk, 400bp)
    # Seg4: chr1 5000..6000 (1_TssA, 1000bp)
    # Seg5: chr1 7000..8000 (15_Quies, 1000bp)
    segs = [
        ("chr1", 1000, 1600, "1_TssA"),
        ("chr1", 3000, 3400, "1_TssA"),
        ("chr1", 3400, 3800, "2_TssAFlnk"),
        ("chr1", 5000, 6000, "1_TssA"),
        ("chr1", 7000, 8000, "15_Quies"),
    ]

    # Two annotation loci
    # Ann1: chr1 1000..2000 (1000bp)
    # Ann2: chr1 3000..4000 (1000bp)
    ann_segs = [
        ("chr1", 1000, 2000),
        ("chr1", 3000, 4000),
    ]

    df = analyze.compute_enrichment(segs, [("TestAnnotation", ann_segs)])
    assert "sensitivity_50" in df.columns
    assert "coverage_50" in df.columns

    by_state = df.set_index("state")

    # 1_TssA:
    # Locus 1: 600bp/1000bp = 60% >= 50% (hit)
    # Locus 2: 400bp/1000bp = 40% < 50% (miss)
    # sensitivity_50 = 1 / 2 = 0.5
    assert by_state.loc["1_TssA", "sensitivity_50"] == pytest.approx(0.5)

    # 1_TssA segments: Seg1 (100%), Seg2 (100%), Seg4 (0%) -> 2 hits out of 3 segs
    assert by_state.loc["1_TssA", "coverage_50"] == pytest.approx(2.0 / 3.0)

    # 2_TssAFlnk:
    # Locus 1: 0bp (miss), Locus 2: 400bp (40% < 50% -> miss) -> sensitivity_50 = 0.0
    # Seg3 (100%) -> coverage_50 = 1.0
    assert by_state.loc["2_TssAFlnk", "sensitivity_50"] == pytest.approx(0.0)
    assert by_state.loc["2_TssAFlnk", "coverage_50"] == pytest.approx(1.0)

    # POOL:Tss (1_TssA + 2_TssAFlnk):
    # Locus 1: 600bp = 60% >= 50% (hit)
    # Locus 2: 400bp + 400bp = 800bp = 80% >= 50% (hit)
    # sensitivity_50 = 2 / 2 = 1.0
    assert by_state.loc["POOL:Tss", "sensitivity_50"] == pytest.approx(1.0)
    # Family segments: Seg1, Seg2, Seg3, Seg4 (3 hits out of 4 segs) -> coverage_50 = 3/4 = 0.75
    assert by_state.loc["POOL:Tss", "coverage_50"] == pytest.approx(0.75)


def test_stacked_bar_plot_xticklabels():
    import pandas as pd
    df = pd.DataFrame({
        "State1": [0.3, 0.4],
        "State2": [0.7, 0.6],
    }, index=["kmeans_homer", "bmm3_macs2"])

    ax = utils.stacked_bar_plot(df, xticklabels="group")
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["Homer KMeans", "MACS2 BMM3"]


def test_plot_summary_no_legend():
    import pandas as pd
    import summary_plots

    df = pd.DataFrame({
        "ds1": [1.0, 2.0],
        "ds2": [1.5, 2.5]
    }, index=["kmeans_homer", "bmm3_macs2"])

    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "test_summary.png")
        summary_plots._plot_summary(df, "Test Title", "Y Label", outpath)
        assert os.path.exists(outpath)
