#!/bin/bash
# Project root directory
ROOT=$(cd "$(dirname "$0")" && pwd)

# Compatibility for Zsh
if [ -n "$ZSH_VERSION" ]; then
  emulate bash
  setopt shwordsplit
fi

# Joint segmentations across replicates of the ENCODE datasets.
# Please ensure that snakemake part was already processed, see README.md
DIR=~/data/2026_segmentations/encode
mkdir -p "$DIR"

CHROMSIZES="$DIR/hg38.chrom.sizes"
CHROMHMM_JAR="$DIR/ChromHMM/ChromHMM.jar"
GENOME=hg38
STATES=15
CHROMHMM_BIN=200
# Marks list for the ENCODE datasets
MARKS="H3K4me3,H3K27ac,H3K4me1,H3K36me3,H3K9me3,H3K27me3"
REPS="rep1,rep2"
# Datasets with two replicates, see config_encode.yaml

# ENCODE reference segmentation of a dataset
ref_bed() {
 case "$1" in
  imr90) echo "ENCFF714POQ_chromhmm.bed" ;;
  monocytes) echo "ENCFF227EMB_chromhmm.bed" ;;
  spleen) echo "ENCFF183AWD_chromhmm.bed" ;;
 esac
}

# Relabel both replicates of a joint model to the ENCODE reference in a single
# match.py call: one shared mapping is applied to them, so the joint state space
# survives the matching, rep1 / rep2 stay comparable, and both end up in the label
# space of the individual _matched segmentations.
match_joint() {
 if [[ -f "$1" ]] && [[ -f "$2" ]] && [[ -f "$3" ]]; then
  python "$ROOT/scripts/rules/match.py" --ref "$1" --work "$2" "$3" --out "${2%.bed}_matched.bed" "${3%.bed}_matched.bed"
 else
  echo "Skipping matching, missing $1, $2 or $3"
 fi
}

# 1. Joint KMeans replicates states processing
for ds in imr90 monocytes spleen
do
 echo "===================="; echo "$ds"
 cd "$DIR/$ds" || exit
 for PC in homer macs2 omni
 do
  echo "~~~~~~~~~~~~~~~~~~~~"; echo "$PC"
  case "$PC" in
   omni)  BIN=100 ;;
   homer) BIN=200 ;;
   macs2) BIN=100 ;;
  esac
  echo "Collecting peaks"
  ALL_PEAKS=""
  for R in rep1 rep2
  do
   for M in H3K4me3 H3K27ac H3K4me1 H3K36me3 H3K9me3 H3K27me3
   do
    case "$PC" in
     omni)  P="$R/omni/${M}_${BIN}.peak" ;;
     homer) P="$R/homer/${M}.bed" ;;
     macs2) P="$R/macs2/${M}.bed" ;;
    esac
    if [[ ! -f "$P" ]]; then echo "Missing $ds/$P"; P=NONE; fi
    ALL_PEAKS="$ALL_PEAKS $P"
   done
  done
  echo "Joint learning"
  if [[ -z $(ls -A "joint_kmeans/$PC") ]]; then
    mkdir -p "joint_kmeans/$PC"
    python "$ROOT/scripts/rules/joint_peaks_segmentation.py" \
   --bin "$BIN" --chromsizes "$CHROMSIZES" --marks "$MARKS" --cells "$REPS" \
   --peaks $ALL_PEAKS --states "$STATES" --outdir "joint_kmeans/$PC";
  fi
  if [[ -z $(ls -A "joint_bmm3/$PC") ]]; then
   mkdir -p "joint_bmm3/$PC"
   python "$ROOT/scripts/rules/joint_peaks_segmentation.py" \
   --bin "$BIN" --chromsizes "$CHROMSIZES" --marks "$MARKS" --cells "$REPS" \
   --peaks $ALL_PEAKS --states "$STATES" --outdir "joint_bmm3/$PC" \
   --mixture --spatial-bins 3;
  fi
 done
done

# 2. Match joint KMeans to the ENCODE reference
for ds in imr90 monocytes spleen
do
 echo "===================="; echo "$ds"
 cd "$DIR/$ds" || exit
 for PC in homer macs2 omni
 do
  echo "Matching $ds $PC joint KMeans to the ENCODE reference"
  match_joint "$(ref_bed "$ds")" "joint_kmeans/$PC/rep1_kmeans_joint_states.bed" "joint_kmeans/$PC/rep2_kmeans_joint_states.bed"
  match_joint "$(ref_bed "$ds")" "joint_bmm3/$PC/rep1_bmm3_joint_states.bed" "joint_bmm3/$PC/rep2_bmm3_joint_states.bed"
 done
done

# 3. Joint ChromHMM replicates states processing
for ds in imr90 monocytes spleen
do
 echo "===================="; echo "$ds"
 cd "$DIR/$ds" || exit
 echo "~~~~~~~~~~~~~~~~~~~~"; echo "Joint ChromHMM"
 mkdir -p joint_chromhmm
 JOINT_BINARIZED=$(mktemp -d)
 python "$ROOT/scripts/joint_chromhmm.py" concat --rep1 rep1/chromhmm_default --rep2 rep2/chromhmm_default --outdir "$JOINT_BINARIZED"
 java -mx4000M -jar "$CHROMHMM_JAR" LearnModel -p 8 -b "$CHROMHMM_BIN" "$JOINT_BINARIZED" joint_chromhmm "$STATES" "$GENOME"
 rm -rf "$JOINT_BINARIZED"
done

# 4. Match joint ChromHMM to the ENCODE reference
for ds in imr90 monocytes spleen
do
 echo "===================="; echo "$ds"
 cd "$DIR/$ds" || exit
 echo "Matching $ds joint ChromHMM to the ENCODE reference"
 match_joint "$(ref_bed "$ds")" "joint_chromhmm/rep1_${STATES}_dense.bed" "joint_chromhmm/rep2_${STATES}_dense.bed"
done
