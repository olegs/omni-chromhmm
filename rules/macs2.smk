# MACS2 peak calling in narrow or broad mode depending on mark type.
#
# Narrow marks (sharp peaks): H3K4me3, H3K27ac, H3K4me2, H3K9ac, H2AFZ
# Broad marks (diffuse domains): H3K27me3, H3K36me3, H3K9me3, H3K4me1, H3K9me2, H3K79me2, H4K20me1, H3F3A
#
# When a control BAM is available it is passed via -c; otherwise MACS2 runs
# without a control.
# Output: {folder}/macs2/{mark}.bed  (3-column BED, sorted)

MACS2_NARROW_MARKS = {"H3K4me3", "H3K27ac", "H3K4me2", "H3K9ac", "H2AFZ"}
MACS2_BIN = P.get("macs2_bin", 100)
CALLER_BIN["macs2"] = MACS2_BIN


def _macs2_opts(mark):
    return "" if mark in MACS2_NARROW_MARKS else "--broad --broad-cutoff 0.1"


rule macs2_callpeak:
    """Call peaks with MACS2 (narrow for H3K4me3/H3K27ac, broad for others)."""
    input:
        bam     = ancient("{folder}/bams/{mark}.bam"),
        control = lambda w: [ancient(f"{w.folder}/controls/{w.mark}.bam")] if folder_has_controls(w.folder) else [],
    output:
        bed = "{folder}/macs2/{mark}.bed",
    conda: "../envs/macs2.yaml"
    log:   "{folder}/macs2/{mark}.log"
    params:
        genome  = {"hg38": "hs", "hg19": "hs", "mm10": "mm", "mm9": "mm"}.get(GENOME, GENOME),
        opts    = lambda w: _macs2_opts(w.mark),
        control = lambda w: f"-c {w.folder}/controls/{w.mark}.bam" if folder_has_controls(w.folder) else "",
        outdir  = "{folder}/macs2",
        name    = "{mark}",
    shell:
        r"""
        mkdir -p {params.outdir}
        macs2 callpeak \
            -t {input.bam} \
            {params.control} \
            -f BAM \
            -g {params.genome} \
            {params.opts} \
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
