#!/bin/bash
# Project root directory
ROOT=$(cd "$(dirname "$0")" && pwd)
source "$ROOT/match.sh"


# Please ensure that snakemake part was already processed
DIR=~/data/2026_segmentations/sagaconf
mkdir -p $DIR 

CHROMSIZES=$DIR/hg38.chrom.sizes;
BIN=100;

# 1. Joint KMeans replicates states processing
for ds in mcf7 gm12878 k562 cd14_monocyte hela_s3; do
 echo "===================="; echo $ds;
 cd $DIR/$ds;

 for PC in homer macs2 omni; do
  echo "~~~~~~~~~~~~~~~~~~~~"; echo $PC;
  BIN=$(get_bin_size "$PC")
  mkdir -p joint_kmeans/$PC;
  # Discover marks present in at least one replicate for this peak caller
  MARKS_LIST="H3K27me3 H3K9me2 H3K4me2 H3K4me3 H3F3A H3K79me2 H3K4me1 H3K9ac H4K20me1 H3K9me3 H3K27ac H2AFZ H3K36me3"
  PRESENT_MARKS=""
  MARKS=""
  for M in $MARKS_LIST; do
    if ls rep*/$PC/*${M}* >/dev/null 2>&1; then
      PRESENT_MARKS="$PRESENT_MARKS $M"
      MARKS="${MARKS:+$MARKS,}$M"
    fi
  done
  REPS="rep1,rep2"
  ALL_PEAKS=""
  for R in rep1 rep2; do
   for M in $PRESENT_MARKS; do
    case $PC in
     omni)  P=$(ls $R/$PC/*${M}*.peak 2>/dev/null | tr '\n' ',' | sed 's/,$//') ;;
     homer) P=$(ls $R/$PC/*${M}*.bed 2>/dev/null | tr '\n' ',' | sed 's/,$//') ;;
     macs2) P=$(ls $R/$PC/*${M}*Peak 2>/dev/null | tr '\n' ',' | sed 's/,$//') ;;
    esac
    ALL_PEAKS="$ALL_PEAKS ${P:-NONE}"
   done
  done
  # Concatenated model, not stacked: replicates are rows sharing one mark space,
  # so a single KMeans/BMM yields a shared state space but an own segmentation per replicate.
  python "$ROOT/scripts/rules/joint_peaks_segmentation.py" \
   --bin $BIN --chromsizes $CHROMSIZES --marks $MARKS --cells "$REPS" \
   --peaks $ALL_PEAKS --states 15 --outdir joint_kmeans/$PC;
  python "$ROOT/scripts/rules/joint_peaks_segmentation.py" \
   --bin $BIN --chromsizes $CHROMSIZES --marks $MARKS --cells "$REPS" \
   --peaks $ALL_PEAKS --states 15 --outdir joint_bmm3/$PC --mixture --spatial-bins 3;
 done
done

# 2. Match individual KMeans to joint KMeans
for ds in mcf7 gm12878 k562 cd14_monocyte hela_s3; do
 echo "===================="; echo $ds;
 cd $DIR/$ds;

 for PC in homer macs2 omni; do
  echo "Matching $ds $PC individual to joint model"
  BIN=$(get_bin_size "$PC")
  for R in rep1 rep2; do
   match_states "joint_kmeans/$PC/${R}_kmeans_joint_states.bed" "$R/$PC/${PC}_kmeans_states.bed" "$PC" "$BIN" "$R"
   match_states "joint_bmm3/$PC/${R}_bmm3_joint_states.bed" "$R/$PC/${PC}_bmm3_states.bed" "$PC" "$BIN" "$R"
  done
 done
done

# 3. Joint ChromHMM replicates states processing
for ds in mcf7 gm12878 k562 cd14_monocyte hela_s3; do
 echo "===================="; echo $ds;
 cd $DIR/$ds;

 echo "~~~~~~~~~~~~~~~~~~~~"; echo "Joint ChromHMM";
 mkdir -p joint_chromhmm;
 JOINT_BINARIZED=$(mktemp -d);
 # Concatenate the binarized signal of both replicates, same marks, replicate as cell
 python "$ROOT/scripts/joint_chromhmm.py" concat \
  --rep1 rep1/chromhmm_default --rep2 rep2/chromhmm_default --outdir $JOINT_BINARIZED;
 # A single model over both replicates, LearnModel segments each replicate as a cell
 # in the shared state space and writes rep1/rep2 _15_segments.bed and _dense.bed.
 java -mx4000M -jar $DIR/ChromHMM/ChromHMM.jar LearnModel -p 8 -b 200 $JOINT_BINARIZED joint_chromhmm 15 hg38;
 rm -rf $JOINT_BINARIZED;
done

# 4. Match individual ChromHMM to joint ChromHMM
for ds in mcf7 gm12878 k562 cd14_monocyte hela_s3; do
 echo "===================="; echo $ds;
 cd $DIR/$ds;

 case $ds in
  mcf7) cell="MCF7" ;;
  gm12878) cell="GM12878" ;;
  k562) cell="K562" ;;
  cd14_monocyte) cell="CD14Monocyte" ;;
  hela_s3) cell="HeLaS3" ;;
 esac
 echo "Matching $ds ChromHMM individual to joint"
 for R in rep1 rep2; do
  match_states "joint_chromhmm/${R}_15_dense.bed" "$R/chromhmm_default_result/${cell}_15_dense.bed" "chromhmm" 200 "$R"
 done
done
