# Joint segmentations across replicates of the ENCODE datasets.
# Please ensure that snakemake part was already processed, see README.md
DIR=~/data/2026_segmentations/encode
mkdir -p $DIR

CHROMSIZES=$DIR/hg38.chrom.sizes;
CHROMHMM_JAR=$DIR/ChromHMM/ChromHMM.jar;
GENOME=hg38;
STATES=15;
CHROMHMM_BIN=200;
MARKS_LIST=(H3K4me3 H3K27ac H3K4me1 H3K36me3 H3K9me3 H3K27me3);
MARKS=$(IFS=,; echo "${MARKS_LIST[*]}");
REPS="rep1,rep2";
# Datasets with two replicates, see config_encode.yaml
DATASETS="imr90 monocytes spleen";

# ENCODE reference segmentation of a dataset, see config_encode.yaml
ref_bed() {
 case $1 in
  imr90) echo "ENCFF714POQ_chromhmm.bed" ;;
  monocytes) echo "ENCFF227EMB_chromhmm.bed" ;;
  spleen) echo "ENCFF183AWD_chromhmm.bed" ;;
 esac
}

# Relabel both replicates of a joint model to the ENCODE reference in a single
# match.py call: one shared mapping is applied to them, so the joint state space
# survives the matching, rep1 / rep2 stay comparable, and both end up in the label
# space of the individual _matched segmentations.
match_joint() { # reference bed, rep1 bed, rep2 bed
 if [[ -f $1 ]] && [[ -f $2 ]] && [[ -f $3 ]]; then
  python ~/work/omni-chromhmm/scripts/rules/match.py \
   --ref $1 --work $2 $3 --out ${2/.bed/_matched.bed} ${3/.bed/_matched.bed};
 else
  echo "Skipping matching, missing $1, $2 or $3";
 fi;
}

# 1. Joint KMeans replicates states processing
for ds in $DATASETS; do
 echo "===================="; echo $ds;
 cd $DIR/$ds;

 for PC in homer macs2 omni; do
  echo "~~~~~~~~~~~~~~~~~~~~"; echo $PC;
  # Peak caller bin sizes, mirroring params of config_encode.yaml
  case $PC in
   omni)  BIN=100 ;;
   homer) BIN=200 ;;
   macs2) BIN=100 ;;
  esac
  mkdir -p joint_kmeans/$PC;
  # Peaks of every replicate, marks in the same order for each of them.
  # Paths are the ones the pipeline produces, see peak_file() in the Snakefile.
  ALL_PEAKS=();
  for R in rep1 rep2; do
   for M in "${MARKS_LIST[@]}"; do
    case $PC in
     omni)  P=$R/omni/${M}_${BIN}.peak ;;
     homer) P=$R/homer/${M}.bed ;;
     macs2) P=$R/macs2/${M}.bed ;;
    esac
    if [[ ! -f $P ]]; then echo "Missing $ds/$P"; P=NONE; fi
    ALL_PEAKS+=($P);
   done;
  done;
  # Concatenated (not stacked) model: replicates are rows sharing one mark space,
  # so a single KMeans yields a shared state space but an own segmentation per replicate.
  python ~/work/omni-chromhmm/scripts/rules/joint_peaks_segmentation.py \
   --bin $BIN --chromsizes $CHROMSIZES --marks $MARKS --cells "$REPS" \
   --peaks "${ALL_PEAKS[@]}" --states $STATES --outdir joint_kmeans/$PC;
 done;
done;

# 2. Match joint KMeans to the ENCODE reference
for ds in $DATASETS; do
 echo "===================="; echo $ds;
 cd $DIR/$ds;

 for PC in homer macs2 omni; do
  echo "Matching $ds $PC joint KMeans to the ENCODE reference";
  match_joint $(ref_bed $ds) \
   joint_kmeans/$PC/rep1_kmeans_joint_states.bed \
   joint_kmeans/$PC/rep2_kmeans_joint_states.bed;
 done;
done;

# 3. Joint ChromHMM replicates states processing
for ds in $DATASETS; do
 echo "===================="; echo $ds;
 cd $DIR/$ds;

 echo "~~~~~~~~~~~~~~~~~~~~"; echo "Joint ChromHMM";
 mkdir -p joint_chromhmm;
 JOINT_BINARIZED=$(mktemp -d);
 # Concatenate the binarized signal of both replicates, same marks, replicate as cell
 python ~/work/omni-chromhmm/scripts/joint_chromhmm.py concat \
  --rep1 rep1/chromhmm_default --rep2 rep2/chromhmm_default --outdir $JOINT_BINARIZED;
 # A single model over both replicates, LearnModel segments each replicate (cell)
 # in the shared state space and writes rep{1,2}_15_segments.bed / _dense.bed.
 java -mx4000M -jar $CHROMHMM_JAR LearnModel -b $CHROMHMM_BIN \
  $JOINT_BINARIZED joint_chromhmm $STATES $GENOME;
 rm -rf $JOINT_BINARIZED;
done;

# 4. Match joint ChromHMM to the ENCODE reference
for ds in $DATASETS; do
 echo "===================="; echo $ds;
 cd $DIR/$ds;

 echo "Matching $ds joint ChromHMM to the ENCODE reference";
 match_joint $(ref_bed $ds) \
  joint_chromhmm/rep1_${STATES}_dense.bed \
  joint_chromhmm/rep2_${STATES}_dense.bed;
done;
