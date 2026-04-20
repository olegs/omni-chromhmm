# Omni ChromHMM

Snakemake pipeline that compares the default ChromHMM segmentation,
ChromHMM over Omnipeak-summits-binarized data, and GMM/KMeans states over Omnipeak
across four ENCODE datasets (IMR90, Monocytes, Monocytes Mint ChIP-seq, GM12878 Mint ChIP-seq).

## Pipeline overview

The pipeline runs the following steps for each dataset:

1. **Data acquisition** (`rules/data.smk`) — downloads ENCODE BAMs, reference
   ChromHMM annotations, and optional RNA-seq quantifications; sorts/indexes
   BAMs; builds bigWig tracks; pools BAMs per histone mark; generates
   non-overlapping bin layouts for ChromHMM binarization.

2. **Default ChromHMM** (`rules/chromhmm.smk`) — runs ChromHMM BinarizeBam on
   the pooled BAMs followed by LearnModel to produce a 15-state dense
   segmentation. When replicates are available, the same procedure is repeated
   independently for each replicate.

3. **Omnipeak peak calling** (`rules/omni.smk`) — calls peaks with Omnipeak in
   three modes: *pooled* (single call on merged BAMs), *replicated* (Omnipeak's
   native multi-replicate mode), and *per-replicate* (one call per replicate).
   Peaks are intersected with genomic bins and converted to per-chromosome
   ChromHMM binary matrices. Three segmentations are produced per mode:
   - ChromHMM LearnModel on Omnipeak-binarized inputs
   - GMM clustering (via `scripts/states.py --method gmm`)
   - KMeans clustering (via `scripts/states.py --method kmeans`)

4. **State matching** (`rules/match.smk`) — relabels every segmentation to the
   ENCODE reference annotation using maximum-overlap mapping (`scripts/match.py`),
   so all results share a single label space and are directly comparable.

5. **Analysis** (`rules/analyze.smk`) — runs `scripts/analyze.py` on each
   matched segmentation to produce per-segmentation reports and plots:
   - `report.tsv` — state count, number of segments, total bp, mean/median length
   - `segment_length.png` — bar chart of average segment length per state
   - `state_emissions.{tsv,png}` — emission matrix (fraction of binarized signal per state × mark)
   - `enrichment.{tsv,png}` — overlap-fraction enrichment vs ChromHMM COORDS annotations
     (CpGIsland, RefSeqExon, RefSeqGene, RefSeqTES, RefSeqTSS, RefSeqTSS2kb)
     plus ExpressedGeneBodies and ExpressedTSS when RNA-seq data is available

6. **Markup analysis** (`rules/markups.smk`) — downloads ENCODE reference
   ChromHMM annotations across cell types for violin plots and cross-dataset
   statistics; also produces per-dataset comparison plots for all matched
   segmentations.

## One-time setup

```bash
mkdir -p ~/data/2026_omni_chromhmm
cd ~/data/2026_omni_chromhmm
wget https://compbio.mit.edu/ChromHMM/ChromHMM.zip && unzip -q ChromHMM.zip
wget https://download.jetbrains.com/biolabs/omnipeak/omnipeak-1.3.6762.jar
```

## Run

From the directory containing `Snakefile`, `config.yaml`, and `rules/`:

```bash
# preview everything with -n
snakemake -p all --use-conda --cores all --directory $(pwd) \
  --snakefile ~/work/claude/omnichromhmm/Snakefile --configfile ~/work/claude/omnichromhmm/config.yaml -n
```

## Narrower targets

```bash
# one dataset end-to-end (builds everything it depends on)
snakemake -p imr90/.done --use-conda --cores all --directory $(pwd) \
  --snakefile ~/work/claude/omnichromhmm/Snakefile --configfile ~/work/claude/omnichromhmm/config.yaml

# stop after default ChromHMM for monocytes
snakemake -p monocytes/chromhmm_default_result/Monocyte_15_dense_matched.bed \
  --use-conda --cores all --directory $(pwd) \
  --snakefile ~/work/claude/omnichromhmm/Snakefile --configfile ~/work/claude/omnichromhmm/config.yaml
```

Useful flags: `-p` to echo shell commands, `-r` for reasons,
`--dag | dot -Tpng > dag.png` to visualize.

## Comparing segmentations

Use `scripts/match.py` to compare any two segmentation BED files.

```bash
# Relabel work segmentation to match reference state names (writes BED to stdout)
python scripts/match.py --ref ref.bed --work work.bed > work_matched.bed

# Compare two segmentations and produce a Jaccard heatmap + similarity score
python scripts/match.py --ref seg_a.bed --work seg_b.bed --compare-only outdir/
```

`--compare-only` skips BED rewriting and instead writes to the output directory:
- `jaccard_heatmap.png` — pairwise Jaccard similarity between all state pairs
- `similarity.txt` — overall similarity score (fraction of overlap length matching the mapped reference state)

Examples: compare replicates, compare segmentation methods, or compare across datasets.

```bash
# IMR90 rep1-vs-rep2 default ChromHMM
python scripts/match.py \
  --ref imr90/rep1/chromhmm_default_result/IMR90_15_dense_matched.bed \
  --work imr90/rep2/chromhmm_default_result/IMR90_15_dense_matched.bed \
  --compare-only imr90/plots_reps_chromhmm/

# Cross-dataset ENCODE reference markup analysis
snakemake -p markups/plots/.done \
  --use-conda --cores all --directory $(pwd) \
  --snakefile ~/work/claude/omnichromhmm/Snakefile \
  --configfile ~/work/claude/omnichromhmm/config.yaml
```

To enable replicate processing for a dataset, set `replicates: true` in
`config.yaml` and ensure each BAM entry has a `rep: rep1` or `rep: rep2` tag.
Currently only `imr90` has replicate annotations.

## Analyzing results

### Per-segmentation analysis (`scripts/analyze.py`)

Each matched segmentation is analyzed independently. The script produces six
output files in the specified `--outdir`:

| File | Description |
|------|-------------|
| `report.tsv` | State count, segment counts, total bp, mean/median length per state |
| `segment_length.png` | Bar chart of average segment length per state |
| `state_emissions.tsv` | Emission matrix: fraction of binarized signal per state x mark |
| `state_emissions.png` | Heatmap of the emission matrix |
| `enrichment.tsv` | Overlap-fraction enrichment vs annotation tracks |
| `enrichment.png` | Heatmap of the enrichment matrix with values |

Standalone usage:

```bash
# Minimal: report + segment length plot
python scripts/analyze.py --seg seg.bed --bin 200 --outdir out/

# With emissions (requires ChromHMM binary input files)
python scripts/analyze.py --seg seg.bed --bin 200 --outdir out/ \
  --inputs chromhmm_default/*.txt

# With enrichment vs COORDS annotations
python scripts/analyze.py --seg seg.bed --bin 200 --outdir out/ \
  --inputs chromhmm_default/*.txt \
  --annotations ChromHMM/COORDS/hg38/*.bed.gz

# With RNA-seq expressed gene enrichment (adds ExpressedGeneBodies + ExpressedTSS tracks)
python scripts/analyze.py --seg seg.bed --bin 200 --outdir out/ \
  --inputs chromhmm_default/*.txt \
  --annotations ChromHMM/COORDS/hg38/*.bed.gz \
  --rnaseq rnaseq.tsv --gene-info Homo_sapiens.gene_info.gz \
  --gtf gencode.v46.basic.annotation.gtf.gz
```

### Cross-dataset reference analysis (`scripts/analyze_downloaded.py`)

Analyzes downloaded ENCODE reference ChromHMM annotations (15-state and 18-state
models) across multiple cell types. Produces violin plots of segment length
distributions, coverage per state, per-sample distributions, median length
heatmaps, and summary statistics.

```bash
python scripts/analyze_downloaded.py --dir markups/
```

### Matched segmentation comparison (`scripts/analyze_matched.py`)

Finds all `*_matched.bed` files recursively under a directory, groups them
into `chromhmm_default_result` vs other methods, and produces comparative
violin plots, coverage plots, heatmaps, and statistics for each group.

```bash
python scripts/analyze_matched.py --dir imr90/
```