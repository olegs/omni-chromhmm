#!/usr/bin/env bash
# Source search pages:
#   15-state: https://www.encodeproject.org/search/?type=File&searchTerm=Chromhmm+15+state&file_format=bed

set -euo pipefail

BASE="https://www.encodeproject.org/files"
OUT_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"

# Format: "ACCESSION:Label"  — all GRCh38
FILES_15=(
    # Compatible naming scheme: Tss/Biv/Enh1/Enh2/EnhG1/EnhG2/TssFlnkD
    # Excluded (TssBiv naming): ENCFF052RFH, ENCFF122ZKA, ENCFF184WVN
    # Excluded (Unknown state + old Enh/EnhG/EnhLo naming): ENCFF034AGC, ENCFF388ELU, ENCFF422PAH, ENCFF710DKH, ENCFF773GKK
    "ENCFF393FJX:Heart_right_ventricle"
    "ENCFF996IOZ:Neurosphere"
    "ENCFF227EMB:CD14-positive_monocyte"
    "ENCFF130HAO:Sigmoid_colon"
    "ENCFF950TVI:Adrenal_gland"
    "ENCFF370FZP:Thyroid_gland"
    "ENCFF183AWD:Spleen"
    "ENCFF781KNB:Uterus"
    "ENCFF771QDN:Tibial_nerve"
    "ENCFF490FOH:Heart_left_ventricle"
    "ENCFF675GXD:Substantia_nigra"
    "ENCFF538HOC:Temporal_lobe"
)

download_one() {
    local accession="$1" label="$2" dest_dir="$3"
    local dest="${dest_dir}/${accession}_${label}.bed.gz"
    if [[ -f "$dest" ]]; then
        echo "  already exists: $(basename "$dest")"
        return
    fi
    local url="${BASE}/${accession}/@@download/${accession}.bed.gz"
    echo "  downloading ${accession} (${label}) ..."
    curl -fsSL --retry 3 --retry-delay 2 -o "$dest" "$url"
}

echo "=== Downloading 15-state ChromHMM markups (${#FILES_15[@]} files) ==="
mkdir -p "$OUT_DIR/15state"
for entry in "${FILES_15[@]}"; do
    acc="${entry%%:*}"
    label="${entry##*:}"
    download_one "$acc" "$label" "$OUT_DIR/15state"
done

echo ""
echo "Done. Files are in:"
echo "  $OUT_DIR/15state/"
