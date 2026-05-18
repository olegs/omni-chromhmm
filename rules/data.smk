# Data acquisition and BAM housekeeping.
#
# Downloads ENCODE BAMs to {ds}/downloaded/ (ENCODE BAMs are already
# coordinate-sorted, so no re-sort step is needed).
# Pooled BAMs (per mark) are assembled at {ds}/bams/ — either a symlink when
# there is a single source BAM or a merged file when multiple replicates exist.
# Per-replicate BAMs are symlinked into {ds}/{rep}/bams/ for downstream rules
# that accept a generic {folder}/bams/ layout.


rule download_bam:
    output: "{ds}/downloaded/{acc}_{mark}.bam"
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


rule download_atac:
    output: "{ds}/atac_{acc}.bed.gz"
    params:
        url = lambda w: f"https://www.encodeproject.org/files/{w.acc}/@@download/{w.acc}.bed.gz"
    shell:
        "wget -q {params.url} -O {output}"


rule download_gencode_gtf:
    output: TOOLS["gencode_gtf"]
    params:
        url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.basic.annotation.gtf.gz"
    shell:
        "wget -q {params.url} -O {output}"


rule download_control_bam:
    """Download a control/input BAM from ENCODE."""
    output: "{ds}/downloaded/{acc}_control.bam"
    params:
        url = lambda w: f"https://www.encodeproject.org/files/{w.acc}/@@download/{w.acc}.bam"
    shell:
        "wget -q {params.url} -O {output}"


rule pool_controls:
    """Pool per-mark control BAMs into {ds}/controls/{mark}.bam.

    Symlinks when there is only one source BAM; merges with samtools when
    multiple control BAMs exist.
    """
    input:  lambda w: controls_for_mark(w.ds, w.mark)
    output: "{ds}/controls/{mark}.bam"
    conda: "../envs/bio.yaml"
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        if len(input) == 1:
            os.symlink(os.path.abspath(input[0]), output[0])
        else:
            shell("samtools merge -f {output} {input}")


rule rep_link_control:
    """Symlink a per-replicate control BAM into {ds}/{rep}/controls/{mark}.bam."""
    input:  lambda w: controls_for_mark(w.ds, w.mark, rep=w.rep)
    output: "{ds}/{rep}/controls/{mark}.bam"
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        if len(input) == 1:
            os.symlink(os.path.abspath(input[0]), output[0])
        else:
            shell("samtools merge -f {output} {input}")


rule pool_bams:
    """Pool per-mark downloaded BAMs into {ds}/bams/{mark}.bam.

    Symlinks when there is only one source BAM (single replicate or no
    replicates); merges with samtools when multiple BAMs exist.
    """
    input:  lambda w: bams_for_mark(w.ds, w.mark)
    output: "{ds}/bams/{mark}.bam"
    conda: "../envs/bio.yaml"
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        if len(input) == 1:
            os.symlink(os.path.abspath(input[0]), output[0])
        else:
            shell("samtools merge -f {output} {input}")


rule rep_link_bam:
    """Symlink a per-replicate downloaded BAM into {ds}/{rep}/bams/{mark}.bam."""
    input:  lambda w: bams_for_mark(w.ds, w.mark, rep=w.rep)
    output: "{ds}/{rep}/bams/{mark}.bam"
    run:
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        os.symlink(os.path.abspath(input[0]), output[0])


rule index_folder_bam:
    """Index a BAM in any {folder}/bams/ directory."""
    input:  "{folder}/bams/{mark}.bam"
    output: "{folder}/bams/{mark}.bam.bai"
    conda: "../envs/bio.yaml"
    shell:  "samtools index {input}"


rule bam_coverage_bw:
    """BigWig track for each mark BAM, produced next to the bams/ folder."""
    input:
        bam = "{folder}/bams/{mark}.bam",
        bai = "{folder}/bams/{mark}.bam.bai",
    output: "{folder}/bams/{mark}.bw"
    threads: 6
    conda: "../envs/bio.yaml"
    shell:  "bamCoverage -b {input.bam} -p {threads} -o {output} --normalizeUsing RPKM"


rule make_bins:
    output: "bins{binsize}.bed"
    wildcard_constraints:
        binsize = r"\d+",
    params:
        chromsizes = TOOLS["chromsizes"],
    conda: "../envs/bio.yaml"
    shell:
        "bedtools makewindows -g {params.chromsizes} -w {wildcards.binsize} > {output}"
