# Omni ChromHMM

Snakemake pipeline comparing default ChromHMM, ChromHMM over Omnipeak-binarized data, 
and GMM/KMeans clustering across ENCODE datasets.

## Pipeline steps

1. **Data** (`rules/data.smk`) -- download ENCODE BAMs, reference ChromHMM annotations, 
optional RNA-seq; pool BAMs per mark.

2. **Default ChromHMM** (`rules/chromhmm.smk`) -- BinarizeBam + LearnModel on pooled BAMs 
to produce 15-state segmentation. Per-mark BED files are extracted from the binarized profiles. 
Repeated per replicate when available.

3. **Omnipeak** (`rules/omni.smk`) -- peak calling in pooled/replicated/per-replicate modes; 
peaks converted to ChromHMM binary matrices. Three segmentations per mode: ChromHMM LearnModel, GMM, KMeans.

4. **State matching** (`rules/match.smk`) -- relabel all segmentations to the ENCODE reference label space 
via maximum-overlap mapping.

5. **Analysis & metrics** (`rules/analyze.smk`) -- per-segmentation reports (emissions, enrichment, segment lengths), 
transition entropy, pairwise Cohen's Kappa, segment statistics, and unified method comparison.

6. **Markup analysis** (`rules/markups.smk`) -- cross-cell-type ENCODE reference analysis and per-dataset matched 
segmentation comparisons.

## Setup

```bash
mkdir -p ~/data/2026_omni_chromhmm && cd ~/data/2026_omni_chromhmm
wget https://compbio.mit.edu/ChromHMM/ChromHMM.zip && unzip -q ChromHMM.zip
wget https://download.jetbrains.com/biolabs/omnipeak/omnipeak-1.3.6762.jar
```

## Run

```bash
# Dry run
snakemake -p all --use-conda --cores all --directory $(pwd) \
  --snakefile ~/work/claude/omnichromhmm/Snakefile \
  --configfile ~/work/claude/omnichromhmm/config.yaml -n

# Single dataset
snakemake -p imr90/.done --use-conda --cores all --directory $(pwd) \
  --snakefile ~/work/claude/omnichromhmm/Snakefile \
  --configfile ~/work/claude/omnichromhmm/config.yaml
```

Useful flags: `-p` (echo commands), `-r` (reasons), `--dag | dot -Tpng > dag.png` (DAG visualization).

## Scripts

### `analyze.py` -- unified analysis and metrics

Single `--seg`: per-segmentation report, emissions, enrichment.
Multiple `--seg`: cross-segmentation entropy, kappa, jaccard, emission similarity, segment stats.

```bash
# Per-segmentation analysis
python scripts/analyze.py --seg seg.bed --bin 200 --outdir out/ \
  --inputs chromhmm_default/*.txt \
  --annotations ChromHMM/COORDS/hg38/*.bed.gz

# Cross-segmentation metrics
python scripts/analyze.py --seg seg1.bed seg2.bed ... --bin 200 --outdir out/
```

### `match.py` -- segmentation comparison

```bash
# Relabel to reference state names
python scripts/match.py --ref ref.bed --work work.bed > work_matched.bed

# Jaccard heatmap + similarity score (no relabelling)
python scripts/match.py --ref seg_a.bed --work seg_b.bed --compare-only outdir/
```

### `compare_methods.py` -- unified method comparison

Aggregates entropy, kappa, segment stats, and per-method enrichment into a comparison table and individual bar chart figures.

### `analyze_downloaded.py` / `analyze_matched.py`

Cross-cell-type and per-dataset violin plots, coverage, heatmaps, and summary statistics.

## Rebuilding plots only

To regenerate plots without recomputing metrics (e.g. after tweaking plot styles):

```bash
# Cross-segmentation comparison plots (entropy, kappa, jaccard, segment stats heatmaps)
python scripts/analyze.py --seg imr90/analysis/comparison/*.bed --bin 200 \
  --outdir imr90/analysis --plot-only

# Method comparison bar charts
python scripts/compare_methods.py --analysis-dir imr90/analysis \
  --comparison-dir imr90/analysis/comparison \
  --outdir imr90/analysis/methods --plot-only
```

Both `--plot-only` flags read existing TSVs and regenerate only the PNG files.

## Replicates

Set `replicates: true` in `config.yaml` and tag each BAM with `rep: rep1` or `rep: rep2`.