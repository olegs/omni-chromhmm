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

Sub-commands:

| Command | Description |
|---------|-------------|
| `analyze` | Per-segmentation report, emissions, enrichment |
| `entropy` | Transition matrix entropy |
| `kappa` | Cohen's Kappa between two segmentations |
| `kappa-all` | Pairwise Kappa between all segmentations |
| `segment-stats` | Segment length statistics (n_states, min/max/mean/median) |

```bash
# Per-segmentation analysis
python scripts/analyze.py analyze --seg seg.bed --bin 200 --outdir out/ \
  --inputs chromhmm_default/*.txt \
  --annotations ChromHMM/COORDS/hg38/*.bed.gz

# Transition entropy
python scripts/analyze.py entropy --seg seg1.bed seg2.bed --bin 200 --outdir out/

# Pairwise Kappa
python scripts/analyze.py kappa-all --seg *.bed --bin 200 --outdir out/

# Segment length statistics
python scripts/analyze.py segment-stats --seg *.bed --outdir out/
```

### `match.py` -- segmentation comparison

```bash
# Relabel to reference state names
python scripts/match.py --ref ref.bed --work work.bed > work_matched.bed

# Jaccard heatmap + similarity score (no relabelling)
python scripts/match.py --ref seg_a.bed --work seg_b.bed --compare-only outdir/
```

### `compare_methods.py` -- unified method comparison

Aggregates entropy, kappa, segment stats, and per-method enrichment into a single comparison table and multi-panel figure.

### `analyze_downloaded.py` / `analyze_matched.py`

Cross-cell-type and per-dataset violin plots, coverage, heatmaps, and summary statistics.

## Replicates

Set `replicates: true` in `config.yaml` and tag each BAM with `rep: rep1` or `rep: rep2`.