# Functional Validation and Biological Benchmark Framework

## 1. Overview and Objective

De novo chromatin state learning produces discrete, genome-wide segmentations from combinatorial histone modification patterns. However, statistical goodness-of-fit or intra-model emission probabilities alone do not guarantee biological utility or regulatory fidelity.

To rigorously benchmark segmentation quality and measure practical utility across different peak calling, binarization, and segmentation strategies (e.g., OmniPeak, HOMER, MACS2, ChromHMM default, and reference annotations), predicted chromatin states are benchmarked against independent, orthogonal functional assays:
1. **ATAC-seq**: Transposase-accessible open chromatin marking active cis-regulatory elements (core promoters and distal enhancers).
2. **RNA-seq & GENCODE GTF**: Transcriptional activity quantified by gene-level expression (TPM), providing sample-matched expressed transcription start sites (TSS) and active gene bodies.
3. **RefSeq Gene Models & Genomic Features**: Curated genomic landmarks (RefSeq TSS, gene bodies, exons, TES, and CpG islands) defining structural genomic coordinates.

The functional validation pipeline (`scripts/analysis/analyze.py`, `scripts/analysis/compare_methods.py`, `scripts/analysis/summary_plots.py`) evaluates these alignments using continuous base-pair overlaps across genomic landmarks and functional targets.

---

## 2. Orthogonal Biological Data Sources and Preprocessing

```
====================================================================================================
Validation Target        Input Files & Sources           Extracted Biological Features
====================================================================================================
Open Chromatin           ATAC-seq NarrowPeak BED         Accessible regulatory elements (promoters & enhancers)
                         (e.g., `atac_{accession}.bed.gz`)

Expressed Transcription  ENCODE RNA-seq TSV              - ExpressedGeneBodies (top 12,000 expressed genes)
                         (gene TPM quantification) +     - ExpressedTSS (1 bp strand-aware TSS)
                         GENCODE Basic GTF               - ExpressedTSS2kb (TSS ± 2,000 bp window)
                         (`gencode.v46.basic.gtf.gz`)    - NonExpressedGeneBodies & NonExpressedTSS (TPM ≤ 0.1)

RefSeq Genomic Features  ChromHMM Coords hg38            - RefSeqTSS (1 bp) & RefSeqTSS2kb (TSS ± 2,000 bp)
                         (`ChromHMM/COORDS/hg38/`)       - RefSeqGene (gene bodies), RefSeqExon, RefSeqTES
                                                         - CpGIsland (CpG island coordinates)
====================================================================================================
```

### 2.1. Open Chromatin via ATAC-seq
- **Source**: High-confidence ATAC-seq narrowPeak BED files downloaded from ENCODE (e.g., `ENCFF243NTP` for IMR90, `ENCFF105FRE` for Spleen).
- **Processing**: Peak intervals are sorted and merged per chromosome (`merge_intervals`) to eliminate artificial boundary inflation from overlapping peak calls.
- **Role**: Serves as direct physical evidence of accessible, nucleosome-depleted open chromatin at active promoters and distal enhancers.

### 2.2. Transcriptional Activity via RNA-seq and GENCODE
- **Source**: ENCODE gene quantification TSV tables (providing gene-level TPMs) combined with GENCODE basic comprehensive annotation GTF (`gencode.v46.basic.annotation.gtf.gz`).
- **Processing** (`make_expressed_annotations` in `scripts/analysis/analyze.py`):
  1. **Gene Expression Stratification**: Genes are ranked descending by TPM. The top $N = 12,000$ active genes ($\text{TPM} > 0$) are categorized as *expressed*, while genes with $\text{TPM} \le 0.1$ are categorized as *non-expressed*.
  2. **Feature Extraction**:
     - **Expressed Gene Bodies (`ExpressedGeneBodies`)**: Full transcribed interval $[\text{start}, \text{end}]$ for expressed genes.
     - **Expressed TSS (`ExpressedTSS`)**: Strand-specific 1 bp initiation coordinate ($\text{start}$ for $+$ strand, $\text{end} - 1$ for $-$ strand).
     - **Expressed TSS 2kb Window (`ExpressedTSS2kb`)**: Extended window $[\text{TSS} - 2000, \text{TSS} + 2000]$ capturing the core promoter and proximal flanking regulatory region.
     - **Negative Controls (`NonExpressedGeneBodies`, `NonExpressedTSS2kb`)**: Inactive genomic loci used to evaluate non-specific background or false-positive active assignments.

### 2.3. Curated Genomic Landmarks (RefSeq)
- **Source**: Standardized ChromHMM coordinate beds (`RefSeqTSS.hg38`, `RefSeqTSS2kb.hg38`, `RefSeqGene.hg38`, `RefSeqExon.hg38`, `RefSeqTES.hg38`, `CpGIsland.hg38`).
- **Processing**: All intervals are merged per chromosome before scoring to establish precise ground-truth footprints.

---

## 3. Mathematical Evaluation Metrics

For a given chromatin state (or pooled state family) $S$ and an orthogonal biological annotation $A$ over a genome of total length $L_{\text{genome}}$:

### 3.1. Base-Pair Level Overlap Metrics

Let $\text{TotalBp}(S)$ be the total genomic footprint of state $S$, $\text{TotalBp}(A)$ be the total footprint of annotation $A$, and $\text{Overlap}(S, A)$ be the total number of shared base pairs:

1. **State Coverage (Precision / Purity)**:
   $$\text{Coverage}(S, A) = \frac{\text{Overlap}(S, A)}{\text{TotalBp}(S)} \quad \in [0, 1]$$
   Measures the fraction of state $S$'s genomic footprint that falls within annotation $A$. High coverage signifies high regulatory purity and low false-positive background.

2. **Annotation Sensitivity (Recall)**:
   $$\text{Sensitivity}(S, A) = \frac{\text{Overlap}(S, A)}{\text{TotalBp}(A)} \quad \in [0, 1]$$
   Measures the fraction of the ground-truth annotation $A$ recovered by state $S$.

3. **Fold Enrichment (ChromHMM Odds Ratio)**:
   $$\text{FoldEnrichment}(S, A) = \frac{\text{Coverage}(S, A)}{\text{AnnotationFraction}(A)} = \frac{\text{Overlap}(S, A) / \text{TotalBp}(S)}{\text{TotalBp}(A) / L_{\text{genome}}}$$
   Measures how much more frequently state $S$ overlaps annotation $A$ relative to a uniform genomic background.

4. **Jaccard Similarity Index**:
   $$\text{Jaccard}(S, A) = \frac{\text{Overlap}(S, A)}{\text{TotalBp}(S) + \text{TotalBp}(A) - \text{Overlap}(S, A)} \quad \in [0, 1]$$

---

## 4. Chromatin State Pooling and Functional Families

To enable balanced comparisons across models that split or lump related chromatin sub-states, states are grouped into target-specific functional pools (`scripts/analysis/utils.py`):

```
====================================================================================================
Functional Pool       Constituent States                     Validation Target & Rationale
====================================================================================================
POOL:Tss              Tss, TssA, TssAFlnk, TssFlnkU          ExpressedTSS2kb & RefSeqTSS2kb
(Core Promoter)                                              Evaluates transcription initiation without
                                                             flanking transition noise.

POOL:Active           Tss, TssA, TssAFlnk, TssFlnkU,         ATAC-seq NarrowPeaks
(Active Open)         Enh, EnhA, EnhAFlnk, Enh1, Enh2        Focuses on accessible promoters and distal
                                                             enhancers (excludes non-open genic enhancers).

POOL:Tx               Tx, TxA, TxWk, TxFlnk                  ExpressedGeneBodies
(Transcribed)                                                Captures transcription elongation and weak
                                                             transcription across active gene bodies.

POOL:Quies            Quies, Het                             NonExpressedGeneBodies
(Repressed/Silent)                                           Negative control confirming the absence of active
                                                             epigenetic marks in inactive genes.
====================================================================================================
```

### 4.1. Pooling Rationale by Biological Target
- **Promoter Evaluation (`POOL:Tss` vs. TSS 2kb)**:
  `Tss` and upstream accessible flank `TssFlnkU` mark active promoter summits. Downstream flank states (`TssFlnk`, `TssFlnkD`) transition into gene bodies and heterochromatin, and are evaluated separately to avoid diluting promoter precision.
- **Open Chromatin Evaluation (`POOL:Active` vs. ATAC-seq)**:
  ATAC-seq measures nucleosome displacement at active promoters and distal enhancers. Genic enhancers (`EnhG1`, `EnhG2`), marked by H3K36me3 and intragenic H3K4me1/H3K27ac, generally lack focal open chromatin summits. Excluding genic enhancers from `POOL:Active` aligns the active state footprint with physical accessibility.
- **Transcribed State Evaluation (`POOL:Tx` vs. Expressed Gene Bodies)**:
  Evaluates transcribed states (`Tx`, `TxA`, `TxWk`, `TxFlnk`) against expressed gene bodies, capturing both strong elongation and weak/transitional transcription.

