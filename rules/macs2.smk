# MACS2 peak calling in narrow or broad mode depending on mark type.
#
# Narrow marks (sharp peaks): H3K4me3, H3K27ac
# Broad marks (diffuse domains): H3K27me3, H3K36me3, H3K9me3, H3K4me1
#
# No control BAM is required; --nolambda disables local lambda estimation.
# Output: {folder}/macs2/{mark}.bed  (3-column BED, sorted)

MACS2_NARROW_MARKS = {"H3K4me3", "H3K27ac"}
MACS2_BIN = P.get("macs2_bin", 100)
CALLER_BIN["macs2"] = MACS2_BIN


def _macs2_mode(mark):
    return "" if mark in MACS2_NARROW_MARKS else "--broad"


rule macs2_callpeak:
    """Call peaks with MACS2 (narrow for H3K4me3/H3K27ac, broad for others)."""
    input:
        bam = "{folder}/bams/{mark}.bam",
    output:
        bed = "{folder}/macs2/{mark}.bed",
    conda: "../envs/macs2.yaml"
    log:   "{folder}/macs2/{mark}.log"
    params:
        genome = {"hg38": "hs", "hg19": "hs", "mm10": "mm", "mm9": "mm"}.get(GENOME, GENOME),
        mode   = lambda w: _macs2_mode(w.mark),
        outdir = "{folder}/macs2",
        name   = "{mark}",
    shell:
        r"""
        mkdir -p {params.outdir}
        macs2 callpeak \
            -t {input.bam} \
            -f BAM \
            -g {params.genome} \
            {params.mode} \
            --outdir {params.outdir} \
            -n {params.name} \
            &> {log}
        # Narrow mode produces .narrowPeak; broad mode produces .broadPeak.
        # Both share the same first 3 BED columns. Merge into sorted 3-col BED.
        peak_file=""
        for f in {params.outdir}/{params.name}_peaks.narrowPeak \
                 {params.outdir}/{params.name}_peaks.broadPeak; do
            [ -f "$f" ] && peak_file="$f" && break
        done
        awk 'BEGIN{{OFS="\t"}} {{print $1,$2,$3}}' "$peak_file" \
            | sort -k1,1 -k2,2n > {output.bed}
        """
