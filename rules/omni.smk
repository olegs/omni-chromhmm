# Omnipeak peak calling

rule omnipeak_call:
    """Call peaks with Omnipeak on {folder}/bams/{mark}.bam."""
    input:
        bam = "{folder}/bams/{mark}.bam",
    output:
        peak = f"{{folder}}/omni/{{mark}}_{OMNI_BIN}.peak",
    threads: 8
    resources: mem_mb=8192
    log: f"{{folder}}/omni/{{mark}}_{OMNI_BIN}.log"
    params:
        bin   = OMNI_BIN,
        cs    = TOOLS["chromsizes"],
        extra = lambda w: DATASETS[ds_of(w.folder)].get("omnipeak_extra", ""),
        wdir  = "{folder}/omni",
    shell:
        "mkdir -p {params.wdir} && "
        "{OMNIPEAK} analyze -t {input.bam} -cs {params.cs} --bin {params.bin} "
        "--threads {threads} -w {params.wdir} "
        "-p {output.peak} --clip 0 {params.extra} "
        "&> {log}"


