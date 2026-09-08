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
        make=f"{HOMER_EXEDIR}/makeTagDirectory",
        find=f"{HOMER_EXEDIR}/findPeaks",
    params:
        dir=HOMER_DIR,
        url="http://homer.ucsd.edu/homer/configureHomer.pl",
    log: f"{HOMER_DIR}/install.log"
    shell:
        r"""
        mkdir -p {params.dir}
        # Use absolute path for installation to ensure HOMER internal paths are correct.
        ABS_DIR=$(cd {params.dir} && pwd)
        LOG=$ABS_DIR/$(basename {log})
        cd $ABS_DIR
        test -f configureHomer.pl || curl -sSL -o configureHomer.pl {params.url}
        perl configureHomer.pl -install > $LOG 2>&1
        # `-install` as the trailing argument leaves configureHomer.pl's %install
        # empty, so it never calls compileSoftware() -- the package is unzipped but
        # cpp/SeqTag.cpp keeps the upstream hard-coded homeDirectory
        # (/gpfs/data01/cbenner/...) and every binary dies with "HOMER not
        # configured properly".  `-make` calls compileSoftware() unconditionally:
        # it rewrites SeqTag.cpp with $ABS_DIR, then recompiles.
        perl configureHomer.pl -make >> $LOG 2>&1
        # compileSoftware() runs make via backticks, so build failures are silent;
        # rebuild here to surface any error in the log, then assert the binaries
        # really carry the local install path.
        make -C cpp >> $LOG 2>&1
        grep -aq "$ABS_DIR" {output.make}
        grep -aq "$ABS_DIR" {output.find}
        """


# --- Peak calling --------------------------------------------------------

rule homer_tagdir:
    input:
        bam=ancient("{folder}/bams/{mark}.bam"),
        tool=f"{HOMER_EXEDIR}/makeTagDirectory",
    output: temp(directory("{folder}/homer/{mark}_tagdir"))
    conda: "../envs/bio.yaml"  # provides samtools
    log: "{folder}/homer/{mark}_tagdir.log"
    resources: homer_tagdir=1, disk_mb=20000
    shell:
        # Avoid creating large intermediate SAM files on disk by piping BAM directly.
        # trap ERR removes the incomplete output directory on failure.
        """
        trap "rm -rf {output}" ERR
        samtools view -h {input.bam} | {input.tool} {output} /dev/stdin -format sam -single > {log} 2>&1
        test -f {output}/tagInfo.txt
        """


rule homer_control_tagdir:
    """Create a Homer tag directory from the control BAM for a mark."""
    input:
        bam=ancient("{folder}/controls/{mark}.bam"),
        tool=f"{HOMER_EXEDIR}/makeTagDirectory",
    output: temp(directory("{folder}/homer/{mark}_control_tagdir"))
    conda: "../envs/bio.yaml"
    log: "{folder}/homer/{mark}_control_tagdir.log"
    resources: homer_tagdir=1, disk_mb=20000
    shell:
        # Avoid creating large intermediate SAM files on disk by piping BAM directly.
        # trap ERR removes the incomplete output directory on failure.
        """
        trap "rm -rf {output}" ERR
        samtools view -h {input.bam} | {input.tool} {output} /dev/stdin -format sam -single > {log} 2>&1
        test -f {output}/tagInfo.txt
        """


rule homer_findpeaks:
    input:
        tagdir=ancient("{folder}/homer/{mark}_tagdir"),
        control_tag=lambda w: [ancient(f"{w.folder}/homer/{w.mark}_control_tagdir")]
        if folder_has_controls(w.folder) else [],
        tool=f"{HOMER_EXEDIR}/findPeaks",
    output: temp("{folder}/homer/{mark}.peaks.txt")
    log: "{folder}/homer/{mark}_findPeaks.log"
    params:
        control=lambda w: f"-i {w.folder}/homer/{w.mark}_control_tagdir"
        if folder_has_controls(w.folder) else "",
    shell:
        r"""
        {input.tool} {input.tagdir} -style histone {params.control} -o {output} > {log} 2>&1
        test -f {output}
        """


rule homer_peak_to_bed:
    input: ancient("{folder}/homer/{mark}.peaks.txt")
    output: "{folder}/homer/{mark}.bed"
    shell:
        r"""
        grep -v '^#' {input} \
          | awk 'BEGIN{{OFS="\t"}} NF>=4 {{print $2,$3,$4}}' \
          | sort -k1,1 -k2,2n > {output}
        """
