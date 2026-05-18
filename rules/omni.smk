# Omnipeak peak calling

rule omnipeak_call:
    """Call peaks with Omnipeak on {folder}/bams/{mark}.bam."""
    input:
        bam     = "{folder}/bams/{mark}.bam",
        control = lambda w: [f"{w.folder}/controls/{w.mark}.bam"] if folder_has_controls(w.folder) else [],
    output:
        peak = f"{{folder}}/omni/{{mark}}_{OMNI_BIN}.peak",
    threads: 8
    resources: mem_mb=8192
    log: f"{{folder}}/omni/{{mark}}_{OMNI_BIN}.log"
    params:
        bin     = OMNI_BIN,
        cs      = TOOLS["chromsizes"],
        extra   = lambda w: DATASETS[ds_of(w.folder)].get("omnipeak_extra", ""),
        control = lambda w: f"-c {w.folder}/controls/{w.mark}.bam" if folder_has_controls(w.folder) else "",
        wdir    = "{folder}/omni",
    shell:
        "mkdir -p {params.wdir} && "
        "{OMNIPEAK} analyze -t {input.bam} {params.control} -cs {params.cs} --bin {params.bin} "
        "--threads {threads} -w {params.wdir} "
        "-p {output.peak} --clip 0 {params.extra} "
        "&> {log}"


