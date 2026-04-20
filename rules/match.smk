# State matching: relabel segmentations to the ENCODE reference via
# max-overlap mapping (match.py), so every segmentation shares one
# label space.
#
# Output filenames encode the input type:
#   chromhmm_default           — standard ChromHMM binarization
#   chromhmm_{mode}            — ChromHMM over Omnipeak peaks
#   gmm_{mode}                 — GMM over Omnipeak peaks
#   kmeans_{mode}              — K-means over Omnipeak peaks

rule match_chromhmm_default:
    input:
        ref  = lambda w: _ref_bed(w.ds),
        work = "{ds}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_dense.bed",
    output:    "{ds}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_chromhmm_default_matched.bed"
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/match.py --ref {input.ref} --work {input.work} > {output}"


rule match_rep_chromhmm_default:
    """Match per-replicate default ChromHMM to the ENCODE reference."""
    input:
        ref  = lambda w: _ref_bed(w.ds),
        work = "{ds}/{mode}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_dense.bed",
    output:    "{ds}/{mode}/chromhmm_default_result/{cell}_" + str(NSTATES) + "_chromhmm_default_{mode}_matched.bed"
    wildcard_constraints: mode = "rep[12]"
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/match.py --ref {input.ref} --work {input.work} > {output}"


rule match_chromhmm_mode:
    input:
        ref  = lambda w: _ref_bed(w.ds),
        work = "{ds}/{mode}/chromhmm_result/{cell}_" + str(NSTATES) + "_dense.bed",
    output:    "{ds}/{mode}/chromhmm_result/{cell}_" + str(NSTATES) + "_chromhmm_{mode}_matched.bed"
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/match.py --ref {input.ref} --work {input.work} > {output}"


rule match_gmm_mode:
    input:
        ref  = lambda w: _ref_bed(w.ds),
        work = "{ds}/{mode}/gmm_states.bed",
    output:    "{ds}/{mode}/gmm_{mode}_matched.bed"
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/match.py --ref {input.ref} --work {input.work} > {output}"


rule match_kmeans_mode:
    input:
        ref  = lambda w: _ref_bed(w.ds),
        work = "{ds}/{mode}/kmeans_states.bed",
    output:    "{ds}/{mode}/kmeans_{mode}_matched.bed"
    conda: "../envs/python.yaml"
    shell:
        "python {SCRIPTS_DIR}/match.py --ref {input.ref} --work {input.work} > {output}"
