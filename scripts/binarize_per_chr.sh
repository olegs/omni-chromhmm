#!/usr/bin/env bash
# Extract per-chromosome binary matrix from the intersect-based multiinter TSV
# and write a gzipped ChromHMM binary file.
# Usage: binarize_per_chr.sh <multiinter.tsv> <cell> <chr> <output.gz>
set -euo pipefail

INPUT=$1
CELL=$2
CHR=$3
OUTPUT=$4

T=$(printf '\t')
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# Line 1: cell  chr
echo "${CELL}${T}${CHR}" > "$TMP"
# Line 2: mark names (columns 4+)
head -n 1 "$INPUT" | awk -v OFS="$T" '{for(i=4;i<=NF;i++){printf "%s%s",$i,(i<NF?OFS:"\n")}}' >> "$TMP"
# Data rows: filter by chromosome, extract mark columns (4+)
awk -v chr="$CHR" -v OFS="$T" \
    'NR>1 && $1==chr {for(i=4;i<=NF;i++){printf "%s%s",$i,(i<NF?OFS:"\n")}}' "$INPUT" >> "$TMP"
gzip -c "$TMP" > "$OUTPUT"
