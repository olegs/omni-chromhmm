cd ~/2026_epi_1000

# Download 1000 epigenomes datasets
rm marks.txt;
# Filter samples with all 6 marks
for E in $(cat names.txt | sed -E 's/-.*//g' | sort --unique); do
 X=0;
 for m in H3K4me1 H3K4me3 H3K9me3 H3K27ac H3K27me3 H3K36me3 Input; do
  if [[ ! -z $(cat names.txt | grep "$E-$m.tagAlign.gz") ]]; then X=$((X+1)); fi
 done;
 if [ "$X" -eq 7 ]; then echo "$E" >> names.txt; fi
done


############### Precomputed segmentations ###################

# Download joint model segmentations
for E in $(cat names.txt); do
 echo $E;
 wget https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/jointModel/n15/reordered/${E}_15_coreMarks_dense.bed.gz -O ${E}_15_coreMarks_dense_joint_reodered.bed.gz
done;

# Download individual segmentations
for E in $(cat names.txt); do
 echo $E;
 wget https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/coreMarks/indivModels/default_init/$E/n15/${E}_15_coreMarks_dense.bed;
gzip ${E}_15_coreMarks_dense.bed;
done;

# Rematch to the joint model
for E in $(cat marks.txt); do echo $E;
 REF=${E}_15_coreMarks_dense_joint_reodered.bed.gz;
 WORK=${E}_15_coreMarks_dense.bed.gz;
 MATCHED=${WORK/.bed.gz/_matched.bed};
 if [[ -f $REF ]] & [[ -f $WORK ]]; then
	python ~/work/omni-chromhmm/scripts/rules/match.py --ref $REF --work $WORK > $MATCHED;
 fi;
done;

# Download joint 18 state extended models (core marks with H3K27ac)
for E in $(cat names.txt); do
	echo $E;
	wget https://egg2.wustl.edu/roadmap/data/byFileType/chromhmmSegmentations/ChmmModels/core_K27ac/jointModel/final/${E}_18_core_K27ac_dense.bed.gz;
done


############### Denovo ###################

for E in $(cat names.txt); do
 echo $E;
 for m in H3K4me1 H3K4me3 H3K9me3 H3K27ac H3K27me3 H3K36me3 Input; do
  echo $m; wget "https://egg2.wustl.edu/roadmap/data/byFileType/alignments/consolidated/$E-$m.tagAlign.gz";
 done
done


# Omnipeak processing
wget https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.chrom.sizes
wget https://download.jetbrains.com/biolabs/omnipeak/omnipeak-1.4.6808.jar

for E in $(cat names.txt); do echo $E; mkdir -p $E/omni;
 for m in H3K4me1 H3K4me3 H3K9me3 H3K27ac H3K27me3 H3K36me3; do
  echo $m;
   java --add-modules=jdk.incubator.vector  -Xmx8G  -jar /home/oshpynov/omnipeak-1.4.6808.jar \
    analyze -t $E-$m.tagAlign.gz -c $E-Input.tagAlign.gz --cs hg19.chrom.sizes --peak $E/omni/$E-$m.peak –-bigwig;
 done;
done

# MACS2 processing
for E in $(cat names.txt); do echo $E; mkdir -p $E/macs2;
 for m in H3K4me3 H3K27ac; do echo $m;
  t=$(mktemp -d);
  gunzip -c $E-$m.tagAlign.gz > $t/treatment.bed; gunzip -c $E-Input.tagAlign.gz > $t/control.bed;
  macs2 callpeak -f BED -t $t/treatment.bed -c $t/control.bed -n $E/macs2/$E-$m -g hs -q 0.05;
 done;

 for m in H3K4me1 H3K9me3 H3K27me3 H3K36me3; do echo $m;
  t=$(mktemp -d);
  gunzip -c $E-$m.tagAlign.gz > $t/treatment.bed; gunzip -c $E-Input.tagAlign.gz > $t/control.bed;
  macs2 callpeak -f BED -t $t/treatment.bed -c $t/control.bed -n $E/macs2/$E-$m -g hs --broad --broad-cutoff 0.1;
 done;
done;

# Homer processing
for E in $(cat names.txt); do echo $E; mkdir -p $E/homer;
 for m in H3K4me3 H3K27ac H3K4me1 H3K9me3 H3K27me3 H3K36me3; do echo $m; 
  t=$(mktemp -d);
  makeTagDirectory $t/treatment_tags $E-$m.tagAlign.gz -format bed -single;
  makeTagDirectory $t/control_tags $E-Input.tagAlign.gz -format bed -single;
  findPeaks $t/treatment_tags -style histone -i $t/control_tags -o $E/homer/${E}-${m}_homer.txt;
  pos2bed.pl $E/homer/${E}-${m}_homer.txt > $E/homer/${E}-${m}_homer.bed; 
 done;
done;

#KMeans states preprocessing
CHROMSIZES=hg19.chrom.sizes;
BIN=100;

#Means states processing
for E in $(cat names.txt); do echo "===================="; echo $E;
for PC in homer macs2 omni; do echo "~~~~~~~~~~~~~~~~~~~~"; echo $PC;
 STATES=$E/$PC/${E}_${PC}_kmeans_states.bed;
 if [[ -f $STATES ]]; then continue; fi;
 echo "States (KMeans peaks)";
 MARKS="H3K4me3,H3K4me1,H3K36me3,H3K9me3,H3K27me3,H3K27ac";
 PEAKS=();
 for M in H3K4me3 H3K4me1 H3K36me3 H3K9me3 H3K27me3 H3K27ac; do
  if [[ $PC == "omni" ]]; then
    P=$(ls $E/$PC/*${M}*.peak 2>/dev/null | tr '\n' ',' | sed 's/,$//');
  elif [[ $PC == "homer" ]]; then
    P=$(ls $E/$PC/*${M}*_homer.bed 2>/dev/null | tr '\n' ',' | sed 's/,$//');
  elif [[ $PC == "macs2" ]]; then
    P=$(ls $E/$PC/*${M}*Peak 2>/dev/null | tr '\n' ',' | sed 's/,$//');
  else
    echo "Unknown peak caller"; exit 1;
  fi
  PEAKS+=("${P:-NONE}");
 done;
 python ~/work/omni-chromhmm/scripts/rules/peaks_segmentation.py --bin $BIN --chromsizes $CHROMSIZES --marks $MARKS --peaks "${PEAKS[@]}" --states 15 --out $STATES --cell $E;
 echo "Done: $STATES";
done;
done;


# ChromHMM processing
TAB=$'\t';
for E in $(cat names.txt); do echo $E; 
 t=$(mktemp -d); rm $t/cell_mark_table.txt || true;
 gunzip -c $E/$E-Input.tagAlign.gz > $t/Input.bed;
 for m in H3K4me3 H3K27ac H3K4me1 H3K9me3 H3K27me3 H3K36me3; do echo $m; 
  gunzip -c $E/$E-$m.tagAlign.gz > $t/$m.bed;
  echo "$E$TAB$m$TAB$m.bed${TAB}Input.bed" >> $t/cell_mark_table.txt; 
 done; 
 mkdir -p $E/chromhmm_binary;
 java -mx4000M -jar ChromHMM/ChromHMM.jar BinarizeBed hg19.chrom.sizes $t  $t/cell_mark_table.txt $E/chromhmm_binary;
 java -mx4000M -jar ChromHMM/ChromHMM.jar LearnModel $E/chromhmm_binary $E/${E}_chromhmm 15 hg19; 
done;


########## Matching ###################

# Rematch peak caller states
for E in $(cat names.txt); do echo $E;
for PC in homer macs2 omni; do echo $PC;
# REF=~/data/2026_omni_chromhmm/imr90/ENCFF714POQ_chromhmm.bed;
 REF=${E}_15_coreMarks_dense_joint_reodered.bed.gz;
 WORK=$E/$PC/${E}_${PC}_kmeans_states.bed;
 MATCHED=${WORK/.bed/_matched.bed};
 if [[ -f $REF ]] & [[ -f $WORK ]]; then
	python ~/work/omni-chromhmm/scripts/rules/match.py --ref $REF --work $WORK > $MATCHED;
 fi;
done;
done;

# Rematch de-novo chromhmm
for E in $(cat names.txt); do echo $E;
# REF=~/data/2026_omni_chromhmm/imr90/ENCFF714POQ_chromhmm.bed;
 REF=${E}_15_coreMarks_dense_joint_reodered.bed.gz;
 WORK=${E}/${E}_chromhmm/${E}_15_dense.bed;
 MATCHED=${WORK/.bed/_matched.bed};
 if [[ -f $REF ]] && [[ -f $WORK ]]; then
	python ~/work/omni-chromhmm/scripts/rules/match.py --ref $REF --work $WORK > $MATCHED;
 fi;
done;

########## Joint models ###################

# Joint KMeans states processing
for PC in homer macs2 omni; do echo "~~~~~~~~~~~~~~~~~~~~"; echo $PC;
 mkdir -p joint_kmeans/$PC;
 MARKS="H3K4me3,H3K4me1,H3K36me3,H3K9me3,H3K27me3,H3K27ac";
 CELLS=$(cat names.txt | tr '\n' ',' | sed 's/,$//');
 ALL_PEAKS=();
 for E in $(cat names.txt); do
  for M in H3K4me3 H3K4me1 H3K36me3 H3K9me3 H3K27me3 H3K27ac; do
   if [[ $PC == "omni" ]]; then
     P=$(ls $E/$PC/*${M}*.peak 2>/dev/null | tr '\n' ',' | sed 's/,$//');
   elif [[ $PC == "homer" ]]; then
     P=$(ls $E/$PC/*${M}*_homer.bed 2>/dev/null | tr '\n' ',' | sed 's/,$//');
   elif [[ $PC == "macs2" ]]; then
     P=$(ls $E/$PC/*${M}*Peak 2>/dev/null | tr '\n' ',' | sed 's/,$//');
   fi
   ALL_PEAKS+=("${P:-NONE}");
  done;
 done;
 python ~/work/omni-chromhmm/scripts/rules/joint_peaks_segmentation.py --bin $BIN --chromsizes $CHROMSIZES --marks $MARKS --cells "$CELLS" --peaks "${ALL_PEAKS[@]}" --states 15 --outdir joint_kmeans/$PC;
done;


# Rematch joint peak caller states
for PC in homer macs2 omni; do echo $PC;
 REFS=(); WORKS=(); MATCHEDS=();
 for E in $(cat names.txt); do
#  REF=~/data/2026_omni_chromhmm/imr90/ENCFF714POQ_chromhmm.bed;
  REF=${E}_15_coreMarks_dense_joint_reodered.bed.gz;
  WORK=joint_kmeans/$PC/${E}_kmeans_joint_states.bed;
  MATCHED=${WORK/.bed/_matched.bed};
  if [[ -f $REF ]] && [[ -f $WORK ]]; then
    REFS+=($REF); WORKS+=($WORK); MATCHEDS+=($MATCHED);
  fi;
 done;
 if [ ${#WORKS[@]} -gt 0 ]; then
  python ~/work/omni-chromhmm/scripts/rules/match.py --ref "${REFS[@]}" --work "${WORKS[@]}" --out "${MATCHEDS[@]}";
 fi;
done;

