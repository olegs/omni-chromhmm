# Homer peak calling (findPeaks -style histone).
#
# All rules are parameterised by a {folder} wildcard (dataset root or replicate
# subdir).  BAMs are read from {folder}/bams/; peaks land in {folder}/homer/.
# The downstream binary-matrix and segmentation rules in omni.smk consume
# {folder}/homer/{mark}.bed via the shared cat_peaks_per_mark rule.
#
# Homer is installed locally via configureHomer.pl
# (not bioconda — no osx-arm64 build exists there).

HOMER_DIR = TOOLS["homer_dir"]
HOMER_EXEDIR = f"{HOMER_DIR}/bin"


# --- One-time install ----------------------------------------------------

rule install_homer:
    """Fetch configureHomer.pl and install the core Homer package locally."""
    output:
        make = f"{HOMER_EXEDIR}/makeTagDirectory",
        find = f"{HOMER_EXEDIR}/findPeaks",
    params:
        dir = HOMER_DIR,
        url = "http://homer.ucsd.edu/homer/configureHomer.pl",
    log: f"{HOMER_DIR}/install.log"
    shell:
        r"""
        mkdir -p {params.dir}
        cd {params.dir}
        test -f configureHomer.pl || curl -sSL -o configureHomer.pl {params.url}
        perl configureHomer.pl -install &> $(basename {log})
        """


# --- Peak calling --------------------------------------------------------

rule homer_tagdir:
    input:
        bam  = "{folder}/bams/{mark}.bam",
        tool = f"{HOMER_EXEDIR}/makeTagDirectory",
    output: temp(directory("{folder}/homer/{mark}_tagdir"))
    conda: "../envs/bio.yaml"  # provides samtools
    log:    "{folder}/homer/{mark}_tagdir.log"
    shell:
        "{input.tool} {output} {input.bam} &> {log}"


rule homer_findpeaks:
    input:
        tagdir = "{folder}/homer/{mark}_tagdir",
        tool   = f"{HOMER_EXEDIR}/findPeaks",
    output: temp("{folder}/homer/{mark}.peaks.txt")
    log:    "{folder}/homer/{mark}_findPeaks.log"
    shell:
        "{input.tool} {input.tagdir} -style histone -o {output} &> {log}"


rule homer_peak_to_bed:
    input:  "{folder}/homer/{mark}.peaks.txt"
    output: "{folder}/homer/{mark}.bed"
    shell:
        r"""
        grep -v '^#' {input} \
          | awk 'BEGIN{{OFS="\t"}} NF>=4 {{print $2,$3,$4}}' \
          | sort -k1,1 -k2,2n > {output}
        """
