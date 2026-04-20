#!/usr/bin/env bash
# Download 20 × 15-state and 20 × 18-state ChromHMM BED files from ENCODE.
#
# Source search pages:
#   15-state: https://www.encodeproject.org/search/?type=File&searchTerm=Chromhmm+15+state&file_format=bed
#   18-state: https://www.encodeproject.org/search/?type=File&searchTerm=Chromhmm+18+state&file_format=bed&assembly=GRCh38

set -euo pipefail

BASE="https://www.encodeproject.org/files"
OUT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Format: "ACCESSION:Label"  — all GRCh38
FILES_15=(
    "ENCFF393FJX:Heart_right_ventricle"
    "ENCFF996IOZ:Neurosphere"
    "ENCFF388ELU:Neural_progenitor_cell"
    "ENCFF122ZKA:Hematopoietic_multipotent_progenitor"
    "ENCFF227EMB:CD14-positive_monocyte"
    "ENCFF130HAO:Sigmoid_colon"
    "ENCFF184WVN:iPS-15b"
    "ENCFF950TVI:Adrenal_gland"
    "ENCFF370FZP:Thyroid_gland"
    "ENCFF183AWD:Spleen"
    "ENCFF781KNB:Uterus"
    "ENCFF710DKH:T-helper_1_cell"
    "ENCFF422PAH:T-cell"
    "ENCFF034AGC:CD4-positive_alpha-beta_memory_T_cell"
    "ENCFF771QDN:Tibial_nerve"
    "ENCFF490FOH:Heart_left_ventricle"
    "ENCFF052RFH:Common_myeloid_progenitor_CD34-positive"
    "ENCFF773GKK:Stimulated_activated_memory_B_cell"
    "ENCFF675GXD:Substantia_nigra"
    "ENCFF538HOC:Temporal_lobe"
)

FILES_18=(
    "ENCFF519CTX:iPS-18a"
    "ENCFF215VUI:iPS-18c"
    "ENCFF787AMD:Trophoblast_cell"
    "ENCFF863RHD:Renal_cortex_interstitium"
    "ENCFF404EVG:iPS-18a_WengLab"
    "ENCFF610TSR:Mammary_epithelial_cell"
    "ENCFF610AXB:Hematopoietic_multipotent_progenitor"
    "ENCFF932LXK:Heart_left_ventricle"
    "ENCFF276KJM:Common_myeloid_progenitor_CD34-positive"
    "ENCFF332FBB:Ovary"
    "ENCFF071QSH:Fibroblast_of_gingiva"
    "ENCFF956KCJ:Muscle_of_back"
    "ENCFF677IYN:Heart"
    "ENCFF223MLA:HUES64"
    "ENCFF867JQX:Renal_cortical_epithelial_cell"
    "ENCFF019LPH:Sigmoid_colon"
    "ENCFF212BTK:Midbrain"
    "ENCFF767HOF:Fibroblast_of_aortic_adventitia"
    "ENCFF080GQN:Thoracic_aorta"
    "ENCFF067UGQ:Ascending_aorta"
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
echo "=== Downloading 18-state ChromHMM markups (${#FILES_18[@]} files) ==="
mkdir -p "$OUT_DIR/18state"
for entry in "${FILES_18[@]}"; do
    acc="${entry%%:*}"
    label="${entry##*:}"
    download_one "$acc" "$label" "$OUT_DIR/18state"
done

echo ""
echo "Done. Files are in:"
echo "  $OUT_DIR/15state/"
echo "  $OUT_DIR/18state/"
