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

6. Launch `analysis.ipynb` for analysis, cross-segmentation comparison and inter-dataset summary plots.

## Setup

```bash
mkdir -p ~/data/2026_omni_chromhmm && cd ~/data/2026_omni_chromhmm
wget https://compbio.mit.edu/ChromHMM/ChromHMM.zip && unzip -q ChromHMM.zip
wget https://download.jetbrains.com/biolabs/omnipeak/omnipeak-1.4.6808.jar
```

## Run

```bash
# Dry run
for ds in imr90 monocytes monocytes_mint gm12878_mint spleen; do
  snakemake -p imr90/.done --use-conda --cores all --directory $(pwd) \
  --snakefile ~/work/omni-chromhmm/Snakefile \
  --configfile ~/work/omni-chromhmm/config.yaml \
  --resources homer_tagdir=1 merge_bam=1 disk_mb=10000 -n;
done
```

Relaunch Omnipeak:
```bash
for ds in imr90 monocytes monocytes_mint gm12878_mint spleen; do
  snakemake -p $ds/omni/.done --use-conda --cores all --directory $(pwd) \
  --snakefile ~/work/omni-chromhmm/Snakefile \
  --configfile ~/work/omni-chromhmm/config.yaml \
  --resources homer_tagdir=1 merge_bam=1 disk_mb=10000 --rerun-incomplete --config omnipeak=True -n;
done
```

Resource limits (pass via `--resources`):
- `homer_tagdir=1` — limits Homer to 2 concurrent tag directories.
- `merge_bam=1` — limits concurrent BAM merges in `pool_bams`. Each merge writes a
  multi-GB temporary file; running too many in parallel can fill the disk. Default is 1
  (sequential); increase if disk space allows.


Useful flags: `-p` (echo commands), `-r` (reasons), `--dag | dot -Tpng > dag.png` (DAG visualization).


