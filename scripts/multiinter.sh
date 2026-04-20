#!/usr/bin/env bash
# Build a binary matrix (one row per bin) from a bins file and sorted peak files.
#
# Uses bedtools intersect per mark to produce exactly one 0/1 value per bin,
# avoiding the interval fragmentation that bedtools multiinter causes when
# peak boundaries don't align with bin edges.
#
# Usage: multiinter.sh <output> <bins.bed> <peak1> [peak2 ...]
set -euo pipefail

OUTPUT=$1; shift
BINS=$1; shift
# Remaining args are sorted peak files

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

# Base: chr, start, end from the bins file (already sorted)
cut -f1-3 "$BINS" | sort -k1,1 -k2,2n > "$TMP_DIR/base.bed"

# Header row: chr  start  end  mark1  mark2  ...
HEADER="chr	start	end"
i=0
for peak in "$@"; do
    mark_name=$(basename "$peak")
    HEADER="${HEADER}	${mark_name}"
    # -c counts overlaps; awk converts count > 0 to 1
    bedtools intersect -a "$TMP_DIR/base.bed" -b "$peak" -c -sorted \
        | awk '{print ($4>0)?1:0}' > "$TMP_DIR/mark_$i.txt"
    i=$((i+1))
done

# Combine base coordinates with all mark columns
{
    echo "$HEADER"
    paste "$TMP_DIR/base.bed" "$TMP_DIR"/mark_*.txt
} > "$OUTPUT"
