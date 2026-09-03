# Omni ChromHMM

Snakemake pipeline comparing default ChromHMM against peak-caller-based
binarization (HOMER, MACS2, Omnipeak), with both ChromHMM LearnModel and KMeans 
clustering applied on top of each peak-caller binarization.
All peak callers support both with control and without control settings, 
so the pipeline works on datasets that lack matched input BAMs.

## Pipeline steps

1. **Config** (`config_encode.yaml` or `config_sagaconf.yaml`) -- configure dataset-specific parameters, input files. 

2. **Data** (`rules/data.smk`) -- download ENCODE BAMs, reference ChromHMM annotations,
optional RNA-seq; pool BAMs per mark.

3. **Default ChromHMM** (`rules/chromhmm.smk`) -- BinarizeBam + LearnModel on pooled BAMs
to produce 15-state segmentation. Per-mark BED files are extracted from the binarized profiles.
Repeated per replicate when available.

4. **Omnipeak** (`rules/omni.smk`) -- control-free peak calling in pooled/per-replicate modes;
peaks converted to ChromHMM binary matrices. Two segmentations per mode: ChromHMM LearnModel and KMeans.

5. **Homer** (`rules/homer.smk`) -- parallel control-free peak caller via `makeTagDirectory` +
`findPeaks -style histone`. Feeds the same downstream binarization / segmentation machinery
as Omnipeak, providing a caller-independent cross-check.

6. **State matching** (`rules/match.smk`) -- relabel all segmentations to the ENCODE reference label space using overlap or jaccard strategy (`_matched`). 
Binarize and Bigwig emissions per state are pre-computed and remapped to the matched states.


## Processing 

```bash
DIR=~/data/2026_segmentations
mkdir -p $DIR && cd $DIR 
wget https://compbio.mit.edu/ChromHMM/ChromHMM.zip && unzip -q ChromHMM.zip
wget https://download.jetbrains.com/biolabs/omnipeak/omnipeak-1.5.6815.jar
```

## Run

### ENCODE analysis


```bash
mkdir -p ~/data/2026_segmentations/encode
cd ~/data/2026_segmentations/encode

# Add -n to dry run
for ds in imr90 monocytes monocytes_mint gm12878_mint spleen; do
  snakemake -p $ds/.done --use-conda --cores all --directory $(pwd) \
  --snakefile ~/work/omni-chromhmm/Snakefile \
  --configfile ~/work/omni-chromhmm/config_encode.yaml \
  --config homer=True macs2=True omnipeak=True \
  --resources homer_tagdir=1 merge_bam=1 disk_mb=10000 \
  --rerun-incomplete --rerun-trigger mtime;
done
```

Resource limits (pass via `--resources`):
- `homer_tagdir=1` — limits Homer to 2 concurrent tag directories.
- `merge_bam=1` — limits concurrent BAM merges in `pool_bams`. Each merge writes a
  multi-GB temporary file; running too many in parallel can fill the disk. Default is 1
  (sequential); increase if disk space allows.


Useful flags: `-p` (echo commands), `-r` (reasons), `--dag | dot -Tpng > dag.png` (DAG visualization).

Joint segmentations across the replicates of a dataset — a single model over `rep1` and `rep2`,
relabelled to the ENCODE reference with one shared mapping, so both replicates keep the
joint state space:

```bash
bash process_encode.sh
```

### 1000 epigenomes analysis
 
[Universal annotation of the human genome through integration of over a thousand epigenomic datasets](https://link.springer.com/article/10.1186/s13059-021-02572-z)

```bash
mkdir -p ~/data/2026_segmentations/epi1000
cd ~/data/2026_segmentations/epi1000

bash process_epi_1000.sh
```

`epi1000_replicates.yaml` groups the epigenomes that are replicates of the same biological
condition (differing only in donor, cell line, sex or consortium); the notebook measures
segmentation reproducibility within those groups.

### SAGAconf dataset analysis

[Robust chromatin state annotation](https://genome.cshlp.org/content/34/3/469)

```bash
mkdir -p ~/data/2026_segmentations/sagaconf
cd ~/data/2026_segmentations/sagaconf

# Add -n to dry run
for ds in mcf7 gm12878 k562 cd14_monocyte hela_s3; do
  snakemake -p $ds/.done --use-conda --cores all --directory $(pwd) \
  --snakefile ~/work/omni-chromhmm/Snakefile \
  --configfile ~/work/omni-chromhmm/config_sagaconf.yaml \
  --config homer=True macs2=True omnipeak=True \
  --resources homer_tagdir=1 merge_bam=1 disk_mb=10000 \
  --rerun-incomplete --rerun-trigger mtime;
done
```

```bash
bash process_sagaconf.sh
```

## Analysis

1. Launch `analysis_encode.ipynb` for ENCODE analysis, cross-segmentation comparison and inter-dataset summary plots.
2. Launch `analysis_epi1000.ipynb` for analysis of the 1000 epigenomes dataset.
3. Launch `analysis_sagaconf.ipynb` for analysis of the SAGAconf dataset.

## Questions?
Contact Oleg Shpynov (oleg.shpynov@jetbrains.com).
