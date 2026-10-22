# Functional ROC sweep on chr1 (functional.ipynb).
#
# The significance threshold of every binarization is swept on the chr1 BAMs of
# a dataset ({ds}/bams_chr1, {ds}/controls_chr1, see data.smk), with the
# options of the pipeline rules and only the threshold changed. Pooled
# replicates only: the rules take {ds}, so no {ds}/repN path can match them.
#   - OmniPeak --fdr, one model per mark fitted once and reused for every FDR
#   - MACS2 -q (broad marks: -q q --broad-cutoff 2q, the pipeline's 0.05 / 0.1)
#   - HOMER findPeaks -style histone -fdr, one tag directory per BAM
#   - ChromHMM BinarizeBam -p, then joint 15-state LearnModel across p-values
# Every peak caller is trained jointly across thresholds with KMeans and BMM3.
# All segmentations of each method are collectively matched to the chr1 part
# of the ENCODE reference by match.py.
#
# Layout under {ds}/functional_chr1_sweep/:
#   {caller}/{thr}/{mark}.bed|.peak         peaks of one threshold
#   {caller}/{thr}/{caller}_{model}_states_matched.bed
#   chromhmm_default/{thr}/model/{cell}_15_dense_matched.bed
#
# Run (e.g. imr90): snakemake imr90/functional_chr1_sweep/.done --use-conda ...

FSWEEP = "functional_chr1_sweep"
FCHROM = "chr1"
_FS = P.get("functional_sweep", {})
FUNCTIONAL_CALLERS = ["omni", "macs2", "homer"]
# Thresholds as directory names: 1e-10, 0.001, 0.05. float() first, since YAML
# reads 1e-10 (no dot) as a string.
FUNCTIONAL_THRESHOLDS = [f"{float(t):g}" for t in _FS.get("thresholds", [0.001, 0.05])]
FUNCTIONAL_PVALUES = [f"{float(t):g}" for t in _FS.get("chromhmm_pvalues", [0.0001])]
# MACS2 sets its genome-wide lambda off the effective size it is given: the
# chr1 share of the hs effective size, as chr1 is all it gets here.
FUNCTIONAL_MACS2_GSIZE = int(2.7e9 * 248956422 / 3.1e9)

wildcard_constraints:
    thr=r"[0-9.e+-]+",
    kind=r"treatment|control",


def _fsweep_control(ds, mark):
    return [f"{ds}/controls_chr1/{mark}.bam"] if has_controls(ds) else []


def functional_peak_file(ds, caller, thr, mark):
    """Peaks of one sweep threshold: OmniPeak broadPeak as it writes it, BED3 otherwise."""
    ext = "peak" if caller == "omni" else "bed"
    return f"{ds}/{FSWEEP}/{caller}/{thr}/{mark}.{ext}"


def functional_ref_bed(ds):
    """The chr1 part of the dataset's ENCODE reference, the sweep is matched to."""
    acc = DATASETS[ds]["ref_chromhmm"]
    return f"{ds}/{FSWEEP}/{acc}_chromhmm_{FCHROM}.bed"


def functional_results(ds):
    """Every matched segmentation of the sweep."""
    cell = DATASETS[ds]["cell"]
    t = [f"{ds}/{FSWEEP}/chromhmm_default/{p}/model/{cell}_{NSTATES}_dense_matched.bed"
         for p in FUNCTIONAL_PVALUES]
    t += [f"{ds}/{FSWEEP}/{caller}/{thr}/{caller}_{model}_states_matched.bed"
          for caller in FUNCTIONAL_CALLERS for thr in FUNCTIONAL_THRESHOLDS
          for model in ("kmeans", "bmm3")]
    return t


rule functional_sweep:
    """The whole chr1 threshold sweep of a dataset (e.g. snakemake imr90/functional_chr1_sweep/.done)."""
    input: lambda w: functional_results(w.ds)
    output: touch(f"{{ds}}/{FSWEEP}/.done")


rule functional_chrom_sizes:
    input: ancient(TOOLS["chromsizes"])
    output: f"{{ds}}/{FSWEEP}/{FCHROM}.chrom.sizes"
    shell: "awk '$1 == \"{FCHROM}\"' {input} > {output}"


rule functional_ref_chr1:
    input: lambda w: ancient(_ref_bed(w.ds))
    output: f"{{ds}}/{FSWEEP}/{{acc}}_chromhmm_{FCHROM}.bed"
    shell: "awk '$1 == \"{FCHROM}\"' {input} > {output}"


# --- Peak calling --------------------------------------------------------

rule functional_omni_model:
    """Fit the OmniPeak model of a mark once; every FDR reuses it."""
    input:
        bam=ancient("{ds}/bams_chr1/{mark}.bam"),
        control=lambda w: [ancient(p) for p in _fsweep_control(w.ds, w.mark)],
        cs=f"{{ds}}/{FSWEEP}/{FCHROM}.chrom.sizes",
    output: f"{{ds}}/{FSWEEP}/omni/models/{{mark}}.omni"
    threads: 8
    resources: mem_mb=8192
    log: f"{{ds}}/{FSWEEP}/omni/models/{{mark}}.log"
    params:
        bin=OMNI_BIN,
        control=lambda w, input: f"-c {input.control[0]}" if input.control else "",
        wdir=f"{{ds}}/{FSWEEP}/omni/work/{{mark}}",
    shell:
        "mkdir -p {params.wdir} && "
        "{OMNIPEAK} analyze -t {input.bam} {params.control} -cs {input.cs} --bin {params.bin} "
        "--threads {threads} -w {params.wdir} -m {output} --clip 0 "
        "> {log} 2>&1"


rule functional_omni_peaks:
    input:
        bam=ancient("{ds}/bams_chr1/{mark}.bam"),
        control=lambda w: [ancient(p) for p in _fsweep_control(w.ds, w.mark)],
        cs=f"{{ds}}/{FSWEEP}/{FCHROM}.chrom.sizes",
        model=f"{{ds}}/{FSWEEP}/omni/models/{{mark}}.omni",
    output: f"{{ds}}/{FSWEEP}/omni/{{thr}}/{{mark}}.peak"
    threads: 4
    resources: mem_mb=8192
    log: f"{{ds}}/{FSWEEP}/omni/{{thr}}/{{mark}}.log"
    params:
        bin=OMNI_BIN,
        control=lambda w, input: f"-c {input.control[0]}" if input.control else "",
        wdir=f"{{ds}}/{FSWEEP}/omni/work/{{mark}}_{{thr}}",
    shell:
        "mkdir -p {params.wdir} && "
        "{OMNIPEAK} analyze -t {input.bam} {params.control} -cs {input.cs} --bin {params.bin} "
        "--threads {threads} -w {params.wdir} -m {input.model} --fdr {wildcards.thr} "
        "-p {output} --clip 0 > {log} 2>&1 && rm -rf {params.wdir}"


def _functional_macs2_opts(w):
    q = float(w.thr)
    if w.mark in MACS2_NARROW_MARKS:
        return f"-q {w.thr}"
    return f"-q {w.thr} --broad --broad-cutoff {min(2 * q, 1.0):g}"


rule functional_macs2:
    input:
        bam=ancient("{ds}/bams_chr1/{mark}.bam"),
        control=lambda w: [ancient(p) for p in _fsweep_control(w.ds, w.mark)],
    output: f"{{ds}}/{FSWEEP}/macs2/{{thr}}/{{mark}}.bed"
    conda: "../envs/macs2.yaml"
    log: f"{{ds}}/{FSWEEP}/macs2/{{thr}}/{{mark}}.log"
    params:
        opts=_functional_macs2_opts,
        gsize=FUNCTIONAL_MACS2_GSIZE,
        control=lambda w, input: f"-c {input.control[0]}" if input.control else "",
        outdir=f"{{ds}}/{FSWEEP}/macs2/{{thr}}",
    shell:
        # chr1 alone can hold too few paired peaks for MACS2 to build its shift
        # model; it then stops, and it is rerun with the fixed fragment size its
        # own message suggests (--nomodel --extsize 147).
        r"""
        macs2 callpeak -t {input.bam} {params.control} -f BAM -g {params.gsize} \
            {params.opts} --outdir {params.outdir} -n {wildcards.mark} > {log} 2>&1 || true
        if ls {params.outdir}/{wildcards.mark}_peaks.*Peak > /dev/null 2>&1; then :;
        elif grep -q "Too few paired peaks" {log}; then
            macs2 callpeak -t {input.bam} {params.control} -f BAM -g {params.gsize} \
                {params.opts} --nomodel --extsize 147 \
                --outdir {params.outdir} -n {wildcards.mark} >> {log} 2>&1
        else
            exit 1
        fi
        peak_file=""
        for f in {params.outdir}/{wildcards.mark}_peaks.narrowPeak \
                 {params.outdir}/{wildcards.mark}_peaks.broadPeak; do
            [ -f "$f" ] && peak_file="$f" && break
        done
        awk 'BEGIN{{OFS="\t"}} {{print $1,$2,$3}}' "$peak_file" | sort -k1,1 -k2,2n > {output}
        rm -f {params.outdir}/{wildcards.mark}_peaks.* {params.outdir}/{wildcards.mark}_summits.bed {params.outdir}/{wildcards.mark}_model.r
        """


rule functional_homer_tagdir:
    """Tag directory of a chr1 treatment or control BAM, shared by every FDR."""
    input:
        bam=lambda w: ancient(f"{w.ds}/{'bams' if w.kind == 'treatment' else 'controls'}_chr1/{w.mark}.bam"),
        tool=f"{HOMER_EXEDIR}/makeTagDirectory",
    output: directory(f"{{ds}}/{FSWEEP}/homer/tagdirs/{{mark}}_{{kind}}")
    conda: "../envs/bio.yaml"
    log: f"{{ds}}/{FSWEEP}/homer/tagdirs/{{mark}}_{{kind}}.log"
    resources: homer_tagdir=1
    shell:
        """
        trap "rm -rf {output}" ERR
        samtools view -h {input.bam} | {input.tool} {output} /dev/stdin -format sam -single > {log} 2>&1
        test -f {output}/tagInfo.txt
        """


rule functional_homer:
    input:
        tagdir=f"{{ds}}/{FSWEEP}/homer/tagdirs/{{mark}}_treatment",
        control=lambda w: [f"{w.ds}/{FSWEEP}/homer/tagdirs/{w.mark}_control"]
        if has_controls(w.ds) else [],
        tool=f"{HOMER_EXEDIR}/findPeaks",
    output: f"{{ds}}/{FSWEEP}/homer/{{thr}}/{{mark}}.bed"
    log: f"{{ds}}/{FSWEEP}/homer/{{thr}}/{{mark}}.log"
    params:
        control=lambda w, input: f"-i {input.control[0]}" if input.control else "",
        txt=f"{{ds}}/{FSWEEP}/homer/{{thr}}/{{mark}}.peaks.txt",
    shell:
        r"""
        {input.tool} {input.tagdir} -style histone {params.control} -fdr {wildcards.thr} \
            -o {params.txt} > {log} 2>&1
        grep -v '^#' {params.txt} \
          | awk 'BEGIN{{OFS="\t"}} NF>=4 {{print $2,$3,$4}}' \
          | sort -k1,1 -k2,2n > {output}
        rm -f {params.txt}
        """


# --- ChromHMM binarization -----------------------------------------------

rule functional_chromhmm_binarize:
    input:
        bams=lambda w: [ancient(f"{w.ds}/bams_chr1/{m}.bam") for m in get_marks(w.ds)],
        controls=lambda w: [ancient(p) for m in get_marks(w.ds)
                            for p in _fsweep_control(w.ds, m)],
        cs=f"{{ds}}/{FSWEEP}/{FCHROM}.chrom.sizes",
    output: f"{{ds}}/{FSWEEP}/chromhmm_default/{{thr}}/binary/{{cell}}_{FCHROM}_binary.txt"
    resources: mem_mb=5000
    log: f"{{ds}}/{FSWEEP}/chromhmm_default/{{thr}}/{{cell}}_binarize.log"
    params:
        bin=CHROMHMM_BIN,
        rows=lambda w: "".join(
            f"{w.cell}\\t{m}\\t{m}.bam" + (f"\\t{m}.bam" if has_controls(w.ds) else "") + "\\n"
            for m in get_marks(w.ds)),
        table=f"{{ds}}/{FSWEEP}/chromhmm_default/{{thr}}/cellmarkfiletable.tsv",
        bamdir="{ds}/bams_chr1",
        controldir=lambda w: f"-c {w.ds}/controls_chr1" if has_controls(w.ds) else "",
        outdir=f"{{ds}}/{FSWEEP}/chromhmm_default/{{thr}}/binary",
    shell:
        r"""
        mkdir -p {params.outdir}
        printf "{params.rows}" > {params.table}
        {CHROMHMM} BinarizeBam -b {params.bin} -p {wildcards.thr} {params.controldir} {input.cs} \
            {params.bamdir} {params.table} {params.outdir} > {log} 2>&1
        """


ruleorder: functional_match_chromhmm_default > match_segmentation
ruleorder: functional_match_caller > match_segmentation

rule functional_chromhmm_learn_joint:
    """Train a single joint ChromHMM model across all swept p-values on chr1."""
    input:
        bins=lambda w: [f"{w.ds}/{FSWEEP}/chromhmm_default/{p}/binary/{w.cell}_{FCHROM}_binary.txt"
                        for p in FUNCTIONAL_PVALUES],
    output:
        beds=[f"{{ds}}/{FSWEEP}/chromhmm_default/{p}/model/{{cell}}_{NSTATES}_dense.bed"
              for p in FUNCTIONAL_PVALUES],
    threads: 8
    resources: mem_mb=8000
    log: f"{{ds}}/{FSWEEP}/chromhmm_default/{{cell}}_joint_learn.log"
    params:
        joint_bin_dir=f"{{ds}}/{FSWEEP}/chromhmm_default/joint_binary",
        joint_model_dir=f"{{ds}}/{FSWEEP}/chromhmm_default/joint_model",
        bin=CHROMHMM_BIN,
        n=NSTATES,
        genome=GENOME,
        labels=FUNCTIONAL_PVALUES,
    shell:
        r"""
        mkdir -p {params.joint_bin_dir} {params.joint_model_dir}
        python scripts/joint_chromhmm.py sweep --inputs {input.bins} --labels {params.labels} --outdir {params.joint_bin_dir} > {log} 2>&1
        {CHROMHMM} LearnModel -noautoopen -p {threads} -b {params.bin} {params.joint_bin_dir} {params.joint_model_dir} \
            {params.n} {params.genome} >> {log} 2>&1
        for p in {params.labels}; do
            mkdir -p {wildcards.ds}/{FSWEEP}/chromhmm_default/$p/model
            if [ -f {params.joint_model_dir}/${{p}}_{params.n}_dense.bed ]; then
                cp {params.joint_model_dir}/${{p}}_{params.n}_dense.bed {wildcards.ds}/{FSWEEP}/chromhmm_default/$p/model/{wildcards.cell}_{params.n}_dense.bed
            fi
        done
        """


rule functional_match_chromhmm_default:
    """Collective matching of all swept ChromHMM segmentations against the ENCODE reference."""
    input:
        ref=lambda w: functional_ref_bed(w.ds),
        work=lambda w: [f"{w.ds}/{FSWEEP}/chromhmm_default/{p}/model/{w.cell}_{NSTATES}_dense.bed"
                        for p in FUNCTIONAL_PVALUES],
    output:
        beds=[f"{{ds}}/{FSWEEP}/chromhmm_default/{p}/model/{{cell}}_{NSTATES}_dense_matched.bed"
              for p in FUNCTIONAL_PVALUES],
    log: f"{{ds}}/{FSWEEP}/chromhmm_default/{{cell}}_joint_match.log"
    conda: "../envs/python.yaml"
    params:
        matrix_prefix=f"{{ds}}/{FSWEEP}/chromhmm_default/chromhmm_default_joint_matched.match",
    shell:
        "python {SCRIPTS_DIR}/match.py --ref {input.ref} --work {input.work} --out {output.beds} "
        "--matrix-out {params.matrix_prefix} > {log} 2>&1"


# --- Peak segmentations --------------------------------------------------

def _functional_peaks(w):
    return [functional_peak_file(w.ds, w.caller, w.thr, m) for m in get_marks(w.ds)]


rule functional_binarize_peaks:
    input:
        peaks=_functional_peaks,
        cs=f"{{ds}}/{FSWEEP}/{FCHROM}.chrom.sizes",
    output: f"{{ds}}/{FSWEEP}/{{caller}}/{{thr}}/chromhmm_peaks/{FCHROM}_binary.txt.gz"
    log: f"{{ds}}/{FSWEEP}/{{caller}}/{{thr}}/binarize_peaks.log"
    conda: "../envs/python.yaml"
    params:
        bin=lambda w: CALLER_BIN[w.caller],
        marks=lambda w: ",".join(get_marks(w.ds)),
        cell=lambda w: DATASETS[w.ds]["cell"],
        outdir=f"{{ds}}/{FSWEEP}/{{caller}}/{{thr}}/chromhmm_peaks",
    shell:
        "python {SCRIPTS_DIR}/peaks_segmentation.py "
        "--bin {params.bin} --chromsizes {input.cs} --marks {params.marks} "
        "--peaks {input.peaks} --states 0 --save-binary {params.outdir} --cell {params.cell} "
        "> {log} 2>&1"


rule functional_joint_segmentation:
    """Joint KMeans or BMM3 across all swept thresholds for one peak caller."""
    input:
        peaks=lambda w: [functional_peak_file(w.ds, w.caller, thr, m)
                         for thr in FUNCTIONAL_THRESHOLDS for m in get_marks(w.ds)],
        cs=f"{{ds}}/{FSWEEP}/{FCHROM}.chrom.sizes",
    output:
        beds=[f"{{ds}}/{FSWEEP}/{{caller}}/{thr}/{{caller}}_{{model}}_states.bed"
              for thr in FUNCTIONAL_THRESHOLDS],
    wildcard_constraints:
        caller=r"omni|macs2|homer",
        model=r"kmeans|bmm3",
    log: f"{{ds}}/{FSWEEP}/{{caller}}/{{caller}}_{{model}}_joint_states.log"
    conda: "../envs/python.yaml"
    resources: mem_mb=8000
    params:
        bin=lambda w: CALLER_BIN[w.caller],
        marks=lambda w: ",".join(get_marks(w.ds)),
        cells=",".join(FUNCTIONAL_THRESHOLDS),
        n=NSTATES,
        mixture=lambda w: "--mixture --spatial-bins 3" if w.model == "bmm3" else "",
    shell:
        "python {SCRIPTS_DIR}/joint_peaks_segmentation.py "
        "--bin {params.bin} --chromsizes {input.cs} --marks {params.marks} "
        "--cells {params.cells} --peaks {input.peaks} --states {params.n} "
        "--outbeds {output.beds} {params.mixture} "
        "> {log} 2>&1"


rule functional_match_caller:
    """Collective matching of all swept segmentations for one caller and model."""
    input:
        ref=lambda w: functional_ref_bed(w.ds),
        work=lambda w: [f"{w.ds}/{FSWEEP}/{w.caller}/{thr}/{w.caller}_{w.model}_states.bed"
                        for thr in FUNCTIONAL_THRESHOLDS],
    output:
        beds=[f"{{ds}}/{FSWEEP}/{{caller}}/{thr}/{{caller}}_{{model}}_states_matched.bed"
              for thr in FUNCTIONAL_THRESHOLDS],
    wildcard_constraints:
        caller=r"omni|macs2|homer",
        model=r"kmeans|bmm3",
    log: f"{{ds}}/{FSWEEP}/{{caller}}/{{caller}}_{{model}}_joint_match.log"
    conda: "../envs/python.yaml"
    params:
        matrix_prefix=lambda w: f"{w.ds}/{FSWEEP}/{w.caller}/{w.caller}_{w.model}_joint_matched.match",
    shell:
        "python {SCRIPTS_DIR}/match.py --ref {input.ref} --work {input.work} --out {output.beds} "
        "--matrix-out {params.matrix_prefix} > {log} 2>&1"
