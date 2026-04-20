# Data acquisition and BAM housekeeping.
#
# Downloads ENCODE BAMs, reference ChromHMM annotations, and optional RNA-seq
# quantifications; sorts/indexes BAMs and builds bigWigs tracks; pools
# BAMs per mark; materialises the two-file non-overlapping bin layout that the
# Omnipeak -> ChromHMM path uses to avoid bedtools multiinter joining peaks.

rule download_bam:
    output: "{ds}/bams/{acc}_{mark}.bam"
    params:
        url = lambda w: f"https://www.encodeproject.org/files/{w.acc}/@@download/{w.acc}.bam"
    shell:
        "wget -q {params.url} -O {output}"


rule download_chromhmm_ref:
    output: "{ds}/{acc}_chromhmm.bed"
    params:
        url = lambda w: f"https://www.encodeproject.org/files/{w.acc}/@@download/{w.acc}.bed.gz"
    shell:
        "wget -q {params.url} -O {output}.gz && gunzip -f {output}.gz"


rule download_rnaseq:
    output: "{ds}/rnaseq_{acc}.tsv"
    params:
        url = lambda w: f"https://www.encodeproject.org/files/{w.acc}/@@download/{w.acc}.tsv"
    shell:
        "wget -q {params.url} -O {output}"


rule sort_bam:
    input:  "{ds}/bams/{name}.bam"
    output: "{ds}/bams/{name}.sorted.bam"
    threads: 4
    conda: "../envs/bio.yaml"
    shell:  "samtools sort -@ {threads} -o {output} {input}"


rule index_bam:
    input:  "{ds}/bams/{name}.sorted.bam"
    output: "{ds}/bams/{name}.sorted.bam.bai"
    conda: "../envs/bio.yaml"
    shell:  "samtools index {input}"


rule bam_coverage_bw:
    input:
        bam = "{ds}/bams/{name}.sorted.bam",
        bai = "{ds}/bams/{name}.sorted.bam.bai",
    output: "{ds}/bams/{name}.sorted.bw"
    threads: 6
    conda: "../envs/bio.yaml"
    shell:  "bamCoverage -b {input.bam} -p {threads} -o {output}"


rule pool_bams:
    input:  lambda w: bams_for_mark(w.ds, w.mark)
    output: "{ds}/bams_pooled/{mark}.bam"
    conda: "../envs/bio.yaml"
    run:
        if len(input) == 1:
            os.makedirs(os.path.dirname(output[0]), exist_ok=True)
            os.symlink(os.path.abspath(input[0]), output[0])
        else:
            shell("samtools merge -f {output} {input}")


rule rep_link_bam:
    """Symlink per-replicate BAM into the expected bams_pooled layout."""
    input:  lambda w: bams_for_mark(w.ds, w.mark, rep=w.mode)
    output: "{ds}/{mode}/bams_pooled/{mark}.bam"
    wildcard_constraints: mode = "rep[12]"
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        os.symlink(os.path.abspath(input[0]), output[0])


rule make_bins:
    output:
        all  = f"bins{BIN}.bed",
        even = f"bins{BIN}-0.bed",
        odd  = f"bins{BIN}-1.bed",
    params:
        chromsizes = TOOLS["chromsizes"],
        bin = BIN,
    conda: "../envs/bio.yaml"
    shell:
        r"""
        bedtools makewindows -g {params.chromsizes} -w {params.bin} > {output.all}
        sort -k1,1 -k2,2n {output.all} | awk '(NR%2)'  > {output.even}
        sort -k1,1 -k2,2n {output.all} | awk '!(NR%2)' > {output.odd}
        """
