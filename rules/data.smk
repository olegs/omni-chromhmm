# Data acquisition and BAM housekeeping.
#
# Downloads ENCODE BAMs to {ds}/downloaded/ (ENCODE BAMs are already
# coordinate-sorted, so no re-sort step is needed).
# Pooled BAMs (per mark) are assembled at {ds}/bams/ — either a symlink when
# there is a single source BAM or a merged file when multiple replicates exist.
# Per-replicate BAMs are symlinked into {ds}/{rep}/bams/ for downstream rules
# that accept a generic {folder}/bams/ layout.


rule download_bam:
    output: temp("{ds}/downloaded/{acc}_{mark}.bam")
    params:
        url=lambda w: f"https://www.encodeproject.org/files/{w.acc}/@@download/{w.acc}.bam"
    shell:
        "wget -q {params.url} -O {output}"


rule download_chromhmm_ref:
    output: protected("{ds}/{acc}_chromhmm.bed")
    params:
        url=lambda w: f"https://www.encodeproject.org/files/{w.acc}/@@download/{w.acc}.bed.gz"
    shell:
        "wget -q {params.url} -O {output}.gz && gunzip -f {output}.gz"


rule download_rnaseq:
    output: protected("{ds}/rnaseq_{acc}.tsv")
    params:
        url=lambda w: f"https://www.encodeproject.org/files/{w.acc}/@@download/{w.acc}.tsv"
    shell:
        "wget -q {params.url} -O {output}"


rule download_atac:
    output: protected("{ds}/atac_{acc}.bed.gz")
    params:
        url=lambda w: f"https://www.encodeproject.org/files/{w.acc}/@@download/{w.acc}.bed.gz"
    shell:
        "wget -q {params.url} -O {output}"


rule download_gencode_gtf:
    output: temp(TOOLS["gencode_gtf"])
    params:
        url="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.basic.annotation.gtf.gz"
    shell:
        "wget -q {params.url} -O {output}"


rule download_control_bam:
    """Download a control/input BAM from ENCODE."""
    output: temp("{ds}/downloaded/{acc}_control.bam")
    wildcard_constraints:
        acc=r"ENCFF[A-Z0-9]+"
    params:
        url=lambda w: f"https://www.encodeproject.org/files/{w.acc}/@@download/{w.acc}.bam"
    shell:
        "wget -q {params.url} -O {output}"


rule merge_control_bams:
    """Merge multiple control BAMs into one canonical file keyed by sorted accessions.

    Output path encodes the sorted accession list joined by '+', so the same
    set of controls is merged only once regardless of how many marks reference it.
    """
    input: lambda w: [ancient(f"{w.ds}/downloaded/{a}_control.bam") for a in w.accs.split("+")]
    output: temp("{ds}/downloaded/{accs}_merged_control.bam")
    wildcard_constraints:
        accs=r"ENCFF[A-Z0-9]+(\+ENCFF[A-Z0-9]+)+"
    conda: "../envs/bio.yaml"
    shell: "samtools merge -f {output} {input}"


def _canonical_control(ds, mark, rep=None):
    """Return the canonical control BAM path for a (ds, mark[, rep]) combination.

    Single-control marks resolve to the downloaded file directly; multi-control
    marks resolve to the canonical merged file (sorted accessions joined by '+').
    """
    accs = control_accs_for_mark(ds,mark,rep)
    if len(accs) == 1:
        return f"{ds}/downloaded/{accs[0]}_control.bam"
    return f"{ds}/downloaded/{'+'.join(sorted(accs))}_merged_control.bam"


rule pool_controls:
    """Symlink {ds}/controls/{mark}.bam to the canonical control BAM.

    The canonical BAM is either the single downloaded file or the deduplicated
    merged BAM produced by merge_control_bams — so the same merge is never
    repeated for marks that share the same control set.
    """
    input: lambda w: ancient(_canonical_control(w.ds,w.mark))
    output: temp("{ds}/controls/{mark}.bam")
    run:
        os.makedirs(os.path.dirname(output[0]),exist_ok=True)
        if os.path.exists(output[0]):
            os.remove(output[0])
        os.link(os.path.abspath(input[0]),output[0])


rule rep_link_control:
    """Symlink {ds}/{rep}/controls/{mark}.bam to the canonical control BAM."""
    input: lambda w: ancient(_canonical_control(w.ds,w.mark,rep=w.rep))
    output: temp("{ds}/{rep}/controls/{mark}.bam")
    run:
        os.makedirs(os.path.dirname(output[0]),exist_ok=True)
        if os.path.exists(output[0]):
            os.remove(output[0])
        os.link(os.path.abspath(input[0]),output[0])


rule pool_bams:
    """Pool per-mark downloaded BAMs into {ds}/bams/{mark}.bam.

    Hard-links when there is only one source BAM (single replicate or no
    replicates); merges with samtools when multiple BAMs exist.

    Pass --resources disk_mb=N to cap total concurrent disk use.
    """
    input: lambda w: [ancient(f) for f in bams_for_mark(w.ds,w.mark)]
    output: temp("{ds}/bams/{mark}.bam")
    resources: merge_bam=1, disk_mb=20000
    conda: "../envs/bio.yaml"
    shell:
        r"""
        mkdir -p $(dirname {output})
        rm -f {output}
        # Count inputs
        count=0
        for i in {input}; do
            if [ $count -eq 0 ]; then first="$i"; fi
            count=$((count+1))
        done
        if [ $count -eq 1 ]; then
            ln "$first" {output}
        else
            samtools merge -f {output} {input}
        fi
        """


rule rep_link_bam:
    """Symlink a per-replicate downloaded BAM into {ds}/{rep}/bams/{mark}.bam."""
    input: lambda w: [ancient(f) for f in bams_for_mark(w.ds,w.mark,rep=w.rep)]
    output: temp("{ds}/{rep}/bams/{mark}.bam")
    run:
        os.makedirs(os.path.dirname(output[0]),exist_ok=True)
        if os.path.exists(output[0]):
            os.remove(output[0])
        os.link(os.path.abspath(input[0]),output[0])


rule index_folder_bam:
    """Index a BAM in any {folder}/bams/ directory."""
    input: ancient("{folder}/bams/{mark}.bam")
    output: temp("{folder}/bams/{mark}.bam.bai")
    conda: "../envs/bio.yaml"
    shell: "samtools index {input}"


rule bam_coverage_bw:
    """BigWig track for each mark BAM, produced next to the bams/ folder."""
    input:
        bam=ancient("{folder}/bams/{mark}.bam"),
        bai=ancient("{folder}/bams/{mark}.bam.bai"),
    output: "{folder}/bams/{mark}.bw"
    threads: 6
    resources: disk_mb=10000
    conda: "../envs/bio.yaml"
    shell: "bamCoverage -b {input.bam} -p {threads} -o {output} --normalizeUsing RPKM"


rule make_bins:
    output: temp("bins{binsize}.bed")
    wildcard_constraints:
        binsize=r"\d+",
    params:
        chromsizes=TOOLS["chromsizes"],
    conda: "../envs/bio.yaml"
    shell:
        "bedtools makewindows -g {params.chromsizes} -w {wildcards.binsize} > {output}"
