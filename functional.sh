#!/bin/bash
# Joint modeling across sweep parameters and collective matching for functional ROC validation on chr1.
# Please ensure that snakemake peak calling and binarization steps were already processed.

# Compatibility for Zsh
if [ -n "$ZSH_VERSION" ]; then
  emulate -L bash 2>/dev/null || emulate bash
  setopt shwordsplit
  setopt nonomatch
  setopt interactivecomments
  SCRIPT_PATH="${(%):-%x}"
elif [ -n "${BASH_SOURCE[0]}" ]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
else
  SCRIPT_PATH="$0"
fi

ROOT=$(cd "$(dirname -- "$SCRIPT_PATH")" && pwd)
source "$ROOT/match.sh"

DIR=~/data/2026_segmentations/encode
mkdir -p "$DIR"

FSWEEP=functional_chr1_sweep
FCHROM=chr1
GENOME=hg38
STATES=15
CHROMHMM_BIN=200
MARKS_LIST="H3K4me3,H3K27ac,H3K4me1,H3K36me3,H3K9me3,H3K27me3"
MARKS_ARR=(H3K4me3 H3K27ac H3K4me1 H3K36me3 H3K9me3 H3K27me3)

THRESHOLDS=(1e-10 1e-08 1e-06 0.0001 0.001 0.01 0.05 0.1 0.2 0.5)
PVALUES=(1e-10 1e-08 1e-06 1e-05 0.0001 0.001 0.01 0.05)

JOINED_THRESHOLDS=$(IFS=,; echo "${THRESHOLDS[*]}")
JOINED_PVALUES=$(IFS=,; echo "${PVALUES[*]}")

CHROMHMM_JAR="$DIR/ChromHMM/ChromHMM.jar"

# ENCODE reference segmentation of a dataset
ref_bed() {
  case "$1" in
    imr90) echo "$DIR/$1/$FSWEEP/ENCFF714POQ_chromhmm_${FCHROM}.bed" ;;
    monocytes) echo "$DIR/$1/$FSWEEP/ENCFF227EMB_chromhmm_${FCHROM}.bed" ;;
    monocytes_mint) echo "$DIR/$1/$FSWEEP/ENCFF166IHY_chromhmm_${FCHROM}.bed" ;;
    gm12878_mint) echo "$DIR/$1/$FSWEEP/ENCFF870JAH_chromhmm_${FCHROM}.bed" ;;
    spleen) echo "$DIR/$1/$FSWEEP/ENCFF183AWD_chromhmm_${FCHROM}.bed" ;;
    heart_right_ventricle) echo "$DIR/$1/$FSWEEP/ENCFF393FJX_chromhmm_${FCHROM}.bed" ;;
    sigmoid_colon) echo "$DIR/$1/$FSWEEP/ENCFF130HAO_chromhmm_${FCHROM}.bed" ;;
    adrenal_gland) echo "$DIR/$1/$FSWEEP/ENCFF950TVI_chromhmm_${FCHROM}.bed" ;;
    tibial_nerve) echo "$DIR/$1/$FSWEEP/ENCFF771QDN_chromhmm_${FCHROM}.bed" ;;
    heart_left_ventricle) echo "$DIR/$1/$FSWEEP/ENCFF490FOH_chromhmm_${FCHROM}.bed" ;;
    substantia_nigra) echo "$DIR/$1/$FSWEEP/ENCFF675GXD_chromhmm_${FCHROM}.bed" ;;
    temporal_lobe) echo "$DIR/$1/$FSWEEP/ENCFF538HOC_chromhmm_${FCHROM}.bed" ;;
    *) ls "$DIR/$1/$FSWEEP/"*_chromhmm_${FCHROM}.bed 2>/dev/null | head -n 1 ;;
  esac
}

cell_name() {
  case "$1" in
    imr90) echo "IMR90" ;;
    monocytes) echo "Monocyte" ;;
    monocytes_mint) echo "MonocyteMint" ;;
    gm12878_mint) echo "GM12878Mint" ;;
    spleen) echo "Spleen" ;;
    heart_right_ventricle) echo "HeartRightVentricle" ;;
    sigmoid_colon) echo "SigmoidColon" ;;
    adrenal_gland) echo "AdrenalGland" ;;
    tibial_nerve) echo "TibialNerve" ;;
    heart_left_ventricle) echo "HeartLeftVentricle" ;;
    substantia_nigra) echo "SubstantiaNigra" ;;
    temporal_lobe) echo "TemporalLobe" ;;
  esac
}

DATASETS=(
  gm12878_mint
  monocytes_mint
  imr90
  monocytes
  spleen
  heart_right_ventricle
  sigmoid_colon
  adrenal_gland
  tibial_nerve
  heart_left_ventricle
  substantia_nigra
  temporal_lobe
)

# 1. Joint KMeans across sweep thresholds
for ds in "${DATASETS[@]}"
do
  echo "========================================"
  echo "Joint peak segmentation for dataset: $ds"
  echo "========================================"
  if [ ! -d "$DIR/$ds" ]; then
    echo "Warning: Dataset directory $DIR/$ds does not exist, skipping..."
    continue
  fi
  cd "$DIR/$ds" || continue
  CHROMSIZES="$DIR/$ds/$FSWEEP/$FCHROM.chrom.sizes"
  REF=$(ref_bed "$ds")

  for PC in omni macs2 homer
  do
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "Caller: $PC"
    BIN=$(get_bin_size "$PC")

    ALL_PEAKS=()
    KMEANS_BEDS=()
    KMEANS_MATCHED=()

    for thr in "${THRESHOLDS[@]}"
    do
      for M in "${MARKS_ARR[@]}"
      do
        case "$PC" in
          omni)  P="$FSWEEP/$PC/$thr/${M}.peak" ;;
          homer) P="$FSWEEP/$PC/$thr/${M}.bed" ;;
          macs2) P="$FSWEEP/$PC/$thr/${M}.bed" ;;
        esac
        if [[ ! -f "$P" ]]; then
          echo "Warning: missing peak file $ds/$P"
          P="NONE"
        fi
        ALL_PEAKS+=("$P")
      done
      KMEANS_BEDS+=("$FSWEEP/$PC/$thr/${PC}_kmeans_states.bed")
      KMEANS_MATCHED+=("$FSWEEP/$PC/$thr/${PC}_kmeans_states_matched.bed")
    done

    echo "Running joint KMeans for $PC across sweep thresholds..."
    python "$ROOT/scripts/rules/joint_peaks_segmentation.py" \
      --bin "$BIN" --chromsizes "$CHROMSIZES" --marks "$MARKS_LIST" \
      --cells "$JOINED_THRESHOLDS" --peaks "${ALL_PEAKS[@]}" --states "$STATES" \
      --outbeds "${KMEANS_BEDS[@]}"

    echo "Collective matching for $PC KMeans across sweep thresholds..."
    python "$ROOT/scripts/rules/match.py" --ref "$REF" \
      --work "${KMEANS_BEDS[@]}" --out "${KMEANS_MATCHED[@]}" \
      --matrix-out "$FSWEEP/$PC/${PC}_kmeans_joint_matched.match"
  done

  # 2. Joint ChromHMM across sweep p-values
  echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
  echo "Joint ChromHMM default across p-values for $ds"
  CELL=$(cell_name "$ds")
  P_BINS=()
  VALID_PVALUES=()
  CHROMHMM_BEDS=()
  CHROMHMM_MATCHED=()

  for p in "${PVALUES[@]}"
  do
    BIN_FILE="$FSWEEP/chromhmm_default/$p/binary/${CELL}_${FCHROM}_binary.txt"
    if [ -f "$BIN_FILE" ]; then
      P_BINS+=("$BIN_FILE")
      VALID_PVALUES+=("$p")
      CHROMHMM_BEDS+=("$FSWEEP/chromhmm_default/$p/model/${CELL}_${STATES}_dense.bed")
      CHROMHMM_MATCHED+=("$FSWEEP/chromhmm_default/$p/model/${CELL}_${STATES}_dense_matched.bed")
    else
      echo "Warning: missing binary file $BIN_FILE"
    fi
  done

  if [ ${#P_BINS[@]} -gt 0 ]; then
    JOINT_BIN_DIR="$FSWEEP/chromhmm_default/joint_binary"
    JOINT_MODEL_DIR="$FSWEEP/chromhmm_default/joint_model"
    mkdir -p "$JOINT_BIN_DIR" "$JOINT_MODEL_DIR"

    python "$ROOT/scripts/joint_chromhmm.py" sweep \
      --inputs "${P_BINS[@]}" --labels "${VALID_PVALUES[@]}" --outdir "$JOINT_BIN_DIR"

    java -mx4000M -jar "$CHROMHMM_JAR" LearnModel -noautoopen -p 8 -b "$CHROMHMM_BIN" \
      "$JOINT_BIN_DIR" "$JOINT_MODEL_DIR" "$STATES" "$GENOME"

    for p in "${VALID_PVALUES[@]}"
    do
      mkdir -p "$FSWEEP/chromhmm_default/$p/model"
      if [ -f "$JOINT_MODEL_DIR/${p}_${STATES}_dense.bed" ]; then
        cp "$JOINT_MODEL_DIR/${p}_${STATES}_dense.bed" "$FSWEEP/chromhmm_default/$p/model/${CELL}_${STATES}_dense.bed"
      fi
    done

    echo "Collective matching for ChromHMM across sweep p-values..."
    python "$ROOT/scripts/rules/match.py" --ref "$REF" \
      --work "${CHROMHMM_BEDS[@]}" --out "${CHROMHMM_MATCHED[@]}" \
      --matrix-out "$FSWEEP/chromhmm_default/chromhmm_default_joint_matched.match"
  fi
done
