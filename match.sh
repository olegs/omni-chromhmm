#!/bin/bash

# Compatibility for Zsh
if [ -n "$ZSH_VERSION" ]; then
  emulate bash
  setopt shwordsplit
  # Peaks are discovered by globbing, let unmatched patterns through as in bash
  setopt nonomatch
fi

# Determine bin size for a given peak caller
get_bin_size() {
  case "$1" in
    omni)  echo 100 ;;
    homer) echo 200 ;;
    macs2) echo 100 ;;
    chromhmm) echo 200 ;;
    *) echo 200 ;;
  esac
}

# Relabel a model to the reference:
# handle emissions computation if binary tracks are available.
match_states() {
  REF=$1
  WORK=$2
  PC=$3
  BIN=$4
  BINS_DIR=$5
  if [[ -f "$REF" ]] && [[ -f "$WORK" ]]; then
    # Try to find emissions
    W_BIN_EMS=""
    R_BIN_EMS=""
    EM="${WORK%.bed}.bin_emissions.npz"
    if [[ ! -f "$EM" ]] && [[ -n "$PC" ]] && [[ -n "$BIN" ]] && [[ -n "$BINS_DIR" ]]; then
      # Discover binary tracks
      BINS=""
      if [[ "$PC" == "chromhmm" ]]; then
        # Check both common layouts
        BINS=$(ls "$BINS_DIR/chromhmm_default/"*_binary.txt 2>/dev/null)
        if [[ -z "$BINS" ]]; then
          BINS=$(ls "$BINS_DIR/chromhmm_binary/"*_binary.txt 2>/dev/null)
        fi
      else
        BINS=$(ls "$BINS_DIR/$PC/chromhmm_peaks/"*_binary.txt.gz 2>/dev/null)
      fi
      if [[ -n "$BINS" ]]; then
        echo "  Computing emissions for $WORK"
        python "$ROOT/scripts/rules/emissions.py" --bed "$WORK" --binaries $BINS --bin "$BIN" --output "$EM"
      fi
    fi
    if [[ -f "$EM" ]]; then
      W_BIN_EMS="$EM"
      R_BIN_EMS="${WORK%.bed}_matched.bin_emissions.npz"
    fi

    EM_FLAGS=""
    if [[ -n "$W_BIN_EMS" ]]; then
      EM_FLAGS="--work-bin-emissions $W_BIN_EMS --remap-bin-emissions $R_BIN_EMS --work-em-type bin"
    fi

    # Reference emissions BW
    REF_EM="${REF%.bed}.bw_emissions.npz"
    if [[ -f "$REF_EM" ]]; then
      EM_FLAGS="$EM_FLAGS --ref-bw-emissions $REF_EM --ref-em-type bw"
    fi

    # Reference emissions BIN
    REF_BIN_EM="${REF%.bed}.bin_emissions.npz"
    if [[ -f "$REF_BIN_EM" ]]; then
      EM_FLAGS="$EM_FLAGS --ref-bin-emissions $REF_BIN_EM --ref-em-type bin"
    fi

    python "$ROOT/scripts/rules/match.py" --ref "$REF" --work "$WORK" \
      --out "${WORK%.bed}_matched.bed" \
      $EM_FLAGS --matrix-out "${WORK%.bed}_matched.match"
  else
    echo "Skipping matching, missing $REF or $WORK"
  fi
}

# Relabel both replicates of a joint model to the reference in a single
# match.py call: one shared mapping is applied to them.
match_joint() {
  REF=$1
  REP1=$2
  REP2=$3
  PC=$4
  BIN=$5
  if [[ -f "$REF" ]] && [[ -f "$REP1" ]] && [[ -f "$REP2" ]]; then
    # Try to find emissions
    W_BIN_EMS=""
    R_BIN_EMS=""
    # Replicate names are assumed to be rep1 and rep2 relative to current dir
    for R in rep1 rep2; do
      if [[ "$R" == "rep1" ]]; then BED="$REP1"; else BED="$REP2"; fi
      EM="${BED%.bed}.bin_emissions.npz"
      if [[ ! -f "$EM" ]] && [[ -n "$PC" ]] && [[ -n "$BIN" ]]; then
        # Discover binary tracks
        BINS=""
        if [[ "$PC" == "chromhmm" ]]; then
          BINS=$(ls "$R/chromhmm_default/"*_binary.txt 2>/dev/null)
        else
          BINS=$(ls "$R/$PC/chromhmm_peaks/"*_binary.txt.gz 2>/dev/null)
        fi
        if [[ -n "$BINS" ]]; then
          echo "  Computing emissions for $BED"
          python "$ROOT/scripts/rules/emissions.py" --bed "$BED" --binaries $BINS --bin "$BIN" --output "$EM"
        fi
      fi
      if [[ -f "$EM" ]]; then
        W_BIN_EMS="$W_BIN_EMS $EM"
        R_BIN_EMS="$R_BIN_EMS ${BED%.bed}_matched.bin_emissions.npz"
      fi
    done

    EM_FLAGS=""
    if [[ -n "$W_BIN_EMS" ]]; then
      EM_FLAGS="--work-bin-emissions $W_BIN_EMS --remap-bin-emissions $R_BIN_EMS --work-em-type bin"
    fi

    # Reference emissions BW
    REF_EM="${REF%.bed}.bw_emissions.npz"
    if [[ -f "$REF_EM" ]]; then
      EM_FLAGS="$EM_FLAGS --ref-bw-emissions $REF_EM --ref-em-type bw"
    fi

    # Reference emissions BIN
    REF_BIN_EM="${REF%.bed}.bin_emissions.npz"
    if [[ -f "$REF_BIN_EM" ]]; then
      EM_FLAGS="$EM_FLAGS --ref-bin-emissions $REF_BIN_EM --ref-em-type bin"
    fi

    python "$ROOT/scripts/rules/match.py" --ref "$REF" --work "$REP1" "$REP2" \
      --out "${REP1%.bed}_matched.bed" "${REP2%.bed}_matched.bed" \
      $EM_FLAGS --matrix-out "${REP1%.bed}_matched.match"
  else
    echo "Skipping matching, missing $REF, $REP1 or $REP2"
  fi
}
