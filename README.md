# Omni ChromHMM

Snakemake pipeline comparing default ChromHMM against peak-caller-based
binarization (Omnipeak, Homer `findPeaks -style histone`), with both ChromHMM
LearnModel and KMeans clustering applied on top of each peak-caller binarization.
All peak callers are run in control-free mode so the pipeline works on ENCODE
datasets that lack matched input BAMs.

## Pipeline steps

1. **Data** (`rules/data.smk`) -- download ENCODE BAMs, reference ChromHMM annotations,
optional RNA-seq; pool BAMs per mark.

2. **Default ChromHMM** (`rules/chromhmm.smk`) -- BinarizeBam + LearnModel on pooled BAMs
to produce 15-state segmentation. Per-mark BED files are extracted from the binarized profiles.
Repeated per replicate when available.

3. **Omnipeak** (`rules/omni.smk`) -- control-free peak calling in pooled/per-replicate modes;
peaks converted to ChromHMM binary matrices. Two segmentations per mode: ChromHMM LearnModel and KMeans.

4. **Homer** (`rules/homer.smk`) -- parallel control-free peak caller via `makeTagDirectory` +
`findPeaks -style histone`. Feeds the same downstream binarization / segmentation machinery
as Omnipeak, providing a caller-independent cross-check.

5. **State matching** (`rules/match.smk`) -- relabel all segmentations to the ENCODE reference label
space using three strategies: combined overlap+bw-emission (default, `_comb_matched`),
bigwig-emission-only (`_bwem_matched`), and overlap-only (`_ovlp_matched`). Bigwig emissions per
state are pre-computed as `.bw_emissions.npz` and used for combined/bwem matching.

6. **Analysis** (`rules/analyze.smk`) -- per-segmentation reports: binarized emissions
(`bin_emissions/`), bigwig emissions (`bw_emissions/`), enrichment, segment lengths,
transition entropy. Run for all three match variants under `analysis/{variant}/`.

7. **Comparison** (`rules/compare.smk`) -- per-pair metrics for each matching strategy
(`comparison/comb/`, `comparison/bwem/`, `comparison/ovlp/`):
   - Transition entropy (full and excluding Quies/Het)
   - Pairwise Cohen's Kappa, AMI, Jaccard, overlap fraction (as-is labels)
   - Overlap-rematch: Hungarian on bp overlap, then Kappa/Jaccard/Overlap
   - Bin-emission rematch: Hungarian on cosine similarity of binarized state emissions
   - Bigwig-emission rematch: Hungarian on cosine similarity of bigwig-based state emissions
   - Unified method comparison table without rematch (`methods/{variant}/comparison_table.tsv`)
   - Replicate reproducibility with focused rematch analysis, one per method:
     - `methods/rematched_ovlp/`  -- re-align rep1/rep2 state labels by bp overlap
     - `methods/rematched_binem/` -- re-align rep1/rep2 state labels by cosine similarity of binarized emissions
     - `methods/rematched_bwem/`  -- re-align rep1/rep2 state labels by cosine similarity of bigwig emissions

8. **Markup analysis** (`rules/markups.smk`) -- cross-cell-type ENCODE reference analysis and
per-dataset matched segmentation comparisons.

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

### `match.py` -- state matching and bw emissions

```bash
# Compute per-state bigwig emissions (required for combined/bwem matching)
python scripts/match.py compute --bed seg.bed --bigwigs *.bw --marks H3K27ac H3K4me3 \
  --bin 200 --output seg.bw_emissions.npz

# Relabel to reference state names (combined/comb, default alpha=0.8)
python scripts/match.py match --ref ref.bed --ref-emissions ref.bw_emissions.npz \
  --work work.bed --work-emissions work.bw_emissions.npz > work_comb_matched.bed

# Overlap-only (alpha=1.0)
python scripts/match.py match --alpha 1.0 --ref ref.bed --work work.bed > work_ovlp_matched.bed

# Bigwig-emission-only (alpha=0)
python scripts/match.py match --alpha 0 --ref ref.bed --ref-emissions ref.bw_emissions.npz \
  --work work.bed --work-emissions work.bw_emissions.npz > work_bwem_matched.bed
```

### `analyze.py` -- per-segmentation analysis

Per-segmentation report: binarized state emissions (`bin_emissions/`), bigwig emissions
(`bw_emissions/`), enrichment, segment lengths, transition entropy.

```bash
python scripts/analyze.py --seg seg.bed --bin 200 --outdir out/ \
  --inputs chromhmm_default/*.txt \
  --bw-emissions seg.bw_emissions.npz \
  --annotations ChromHMM/COORDS/hg38/*.bed.gz
```

### `compare.py` -- cross-segmentation metrics

Entropy, Kappa, AMI, Jaccard, overlap fraction (as-is), plus three rematch variants
(`ovlp`, `binem`, `bwem`) for all pairs. Finer bin size used for cross-method pairs.

```bash
python scripts/compare.py --seg seg1.bed seg2.bed ... --bins 200 100 200 \
  --outdir out/comparison/ --threads 8
```

### `compare_methods.py` -- unified method comparison

Aggregates entropy, kappa, segment stats, enrichment, and base replicate metrics into
a comparison table and per-metric bar chart PNGs (`methods/{variant}/`).

With `--rematch ovlp|binem|bwem`, produces a focused replicate re-match table containing
only the columns relevant to that re-match method (`methods/rematched_{rematch}/`).
PNG filenames omit the redundant method suffix (e.g. `kappa_rematch_rep1_vs_rep2.png`).

## Replicates

Set `replicates: true` in `config.yaml` and tag each BAM with `rep: rep1` or `rep: rep2`.
When enabled, the pipeline runs all segmentation steps per-replicate and produces three
focused replicate reproducibility reports under `{ds}/methods/rematched_{ovlp,binem,bwem}/`,
each re-aligning rep1/rep2 state labels by a different criterion before computing Kappa/Jaccard:
- `rematched_ovlp/`  — Hungarian assignment on bp overlap
- `rematched_binem/` — Hungarian assignment on cosine similarity of binarized state emissions
- `rematched_bwem/`  — Hungarian assignment on cosine similarity of bigwig-based state emissions
