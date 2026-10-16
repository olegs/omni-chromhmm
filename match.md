# Chromatin State Matching

## 1. Objective

De novo chromatin state models (ChromHMM, K-means, Bernoulli mixtures) learn states with arbitrary identifiers ($E_1, \dots, E_K$). To compare methods and interpret states, each learned state is mapped to a reference label (e.g. the ENCODE 15-state model: `Tss`, `TssFlnk`, `Tx`, `TxWk`, `EnhG`, `Enh`, `ZNF/Rpts`, `Het`, `Biv`, `ReprPC`, `Quies`, ...).

`scripts/rules/match.py` (invoked via `rules/match.smk`, `match.sh` and the `process_*.sh` scripts) solves this as a one-to-one assignment between work states and reference states.

## 2. Matching Scheme

### 2.1. Steps

`match_states()` relabels a work segmentation as follows:

1. **Jaccard.** For every work × reference state pair, the shared base pairs over the union of the two states (§2.3). When several segmentations are passed together, overlaps and state lengths are summed over all of them first (§2.6). Saved as `PREFIX.jaccard.tsv`.
2. **Emission similarity**, when emission files are given (§2.4). Saved as `PREFIX.emission.tsv`.
3. **Match quality**, the weighted sum of the two terms (§2.2). Saved as `PREFIX.quality.tsv`.
4. **Assignment.** A single Hungarian assignment maximises total concave utility $U(w, r) = \alpha \sqrt{\text{Jaccard}(w, r)} + \beta \cdot \text{EmissionSim}(w, r)$ (§2.2, §3.3). The result is a strict bijection: every work state gets a distinct reference label; if there are more work states than reference states, the surplus keeps its own name.
5. **Output.** Each work BED is rewritten with the reference names (normalised) and colours, and the work emission files are renamed to match. `PREFIX.mapping.tsv` lists each chosen pair with its scores, and `PREFIX.png` shows the quality heatmap. The log reports the same per pair, plus the bp accuracy of the assignment.

### 2.2. Match Quality and Concave Utility

$$\text{Quality}(w, r) = \alpha \cdot \text{Jaccard}(w, r) + \beta \cdot \text{EmissionSim}(w, r)$$

The weights $\alpha$, $\beta$ are set in `match.py`. Without usable emissions (missing files, no shared marks, or no state name in common), quality is Jaccard alone.

To prioritize fair coverage across all states and lift poorly matched states with diminishing marginal returns, the Hungarian algorithm optimizes concave square-root utility on spatial overlap combined with linear emission similarity:

$$U(w, r) = \alpha \sqrt{\text{Jaccard}(w, r)} + \beta \cdot \text{EmissionSim}(w, r)$$

(or $U(w, r) = \sqrt{\text{Jaccard}(w, r)}$ without emissions).

### 2.3. Jaccard

$$\text{Jaccard}(w, r) = \frac{|w \cap r|}{|w \cup r|} \quad \in [0, 1]$$

Jaccard is normalised by the size of both states, so a promoter pair and a `Quies` pair are scored on the same scale.

### 2.4. Emission Similarity

Each state has an emission vector $E(m)$ over the marks $m$: the share of its bins called present for binarized emissions (`.bin_emissions.npz`), or its mean signal for bigwig ones (`.bw_emissions.npz`).

1. **Inputs.** Binarized emissions are preferred on each side when both exist; `--work-em-type` / `--ref-em-type` override the choice. ENCODE references only have bigwig emissions, so they are usually compared with binarized work emissions. The type is taken from the file name, otherwise guessed from the value range.
2. **Alignment by name.** Emission rows are looked up by state name (`1.0` = `1`) and marks by base name (`H3K4me3_100` = `H3K4me3`); only marks present on both sides are used. A state missing from its emission file gets the mean similarity of all known pairs, so emissions neither favour nor penalise its candidates; a warning lists such states. If no mark or no state name matches, the emission term is dropped.
3. **Bigwig scaling.** Bigwig signal is brought to the $[0, 1]$ range of binarized emissions by min-max scaling its log, $L = \log(1 + E)$, per mark, with the median state as the minimum:
   $$P = \operatorname{clip}\left(\frac{L - \operatorname{median} L}{\max L - \operatorname{median} L},\ 0,\ 1\right)$$
   where median and max are taken over states. A mark with no spread across states gets $P = 0$. Binarized emissions are used as they are, $P = E$.
4. **Level-aware cosine.** Each vector gets a background component $b = 1 - \max_m P(m)$:
   $$\text{EmissionSim}(w, r) = \cos\big([P_w, b_w], [P_r, b_r]\big) \quad \in [0, 1]$$

### 2.5. Joint Matching

`--ref` and `--work` accept several files, paired in order; a single reference is reused for every work file. Overlaps and state lengths are summed over all pairs before the assignment, so one shared mapping is solved and applied to every work segmentation. This keeps state names consistent across biological replicates of a joint model and across the epigenomes of a joint epi1000 model. Emissions for matching come from the first work emission file.

### 2.6. Usage

```bash
python scripts/rules/match.py --ref REF.bed --work WORK.bed --out WORK_matched.bed \
    --work-bin-emissions WORK.bin_emissions.npz --ref-bw-emissions REF.bw_emissions.npz \
    --remap-bin-emissions WORK_matched.bin_emissions.npz --matrix-out WORK_matched.match
```

Without `--out` the relabelled BED goes to stdout. `--compare-only DIR` writes the Jaccard heatmap of the mapping to `DIR` instead of rewriting any BED.

## 3. Rationale

### 3.1. Jaccard

Jaccard measures positional agreement relative to the size of both states. Every pair is scored on the same $[0, 1]$ range, so a small promoter state that lines up well with `Tss` counts as much as a `Quies` pair, and broad domains cannot decide the whole mapping on base pairs alone. The assignment therefore no longer maximises bp accuracy exactly; the log reports bp accuracy separately.

### 3.2. Emissions

A state is defined by its combinatorial mark signature (H3K4me3 promoters, H3K27ac/H3K4me1 enhancers, H3K36me3 transcription, H3K27me3 Polycomb, H3K9me3 heterochromatin). Emissions resolve spatially adjacent states with similar overlap (`Tss` vs. `TssFlnk`, `Enh` vs. `EnhG`), penalise biochemically incompatible pairs, and stay informative across cell types and assays where spatial concordance is weaker.

The bigwig scaling has no fitted parameters:

- **Median as background.** Most states lack any given mark, so the median state sits at its noise level; states at or below it get 0, as they would after binarization.
- **Log scale.** Some references have an artifact state high on every mark; on a linear scale it would push all other states towards 0.
- **Per-mark maximum.** Marks differ in signal strength, so each is scaled by its own most marked state, as binarization thresholds each mark separately.

The background component keeps the signal level in the cosine, so an empty `Quies`-like state is not similar to a state with every mark high.

### 3.3. Concave Utility (Diminishing Marginal Returns)

Instead of a standard linear sum $\max_{\pi} \sum_i Q(i, \pi(i))$, the assignment optimizes total utility $\max_{\pi} \sum_i (\alpha \sqrt{J(i, \pi(i))} + \beta E(i, \pi(i)))$ with square-root transformation applied to Jaccard overlap.

Because square-root utility exhibits diminishing marginal returns on spatial overlap, improving a poorly matched state from agreement $0.01$ to $0.16$ yields a $+0.30$ gain in $\sqrt{J}$ ($\sqrt{0.16} - \sqrt{0.01} = 0.40 - 0.10 = 0.30$), whereas the same $+0.15$ gain for an already well-matched state from $0.81$ to $0.96$ yields only $+0.08$ utility. Separating $\sqrt{J}$ from emission similarity prevents high baseline emission scores from dampening the concave expansion of low-Jaccard focal states, ensuring that rare functional elements (TSS, enhancers) compete effectively while emissions maintain biochemical disambiguation, all while remaining solvable in a single polynomial-time Hungarian assignment pass.

### 3.4. Alternatives Not Used

- **Pure linear sum maximization.** Maximises the net aggregate sum, but allows strong states to greedily capture labels for minor incremental gains while stranding weaker states with near-zero concordance.
- **Strict bottleneck matching (Maximin LBAP).** Maximises the worst-case single state agreement $\max \min_i Q(i, \pi(i))$, but can sacrifice net genomic fidelity and disrupt multiple high-confidence states just to marginally improve an unmatchable or noisy surplus state.
- **Overlap share** (shared bp as a share of the genome). Maximises bp accuracy, but large domains dominate it and small-state labels are left almost entirely to emissions.
- **State size similarity term.** Steers large states towards large labels; without emissions it outweighs the overlap of every small state and labels them by length.
- **Two-stage assignment** (longest states matched first among the longest reference states). A hard size cut: it removes the worst exiles, but mid-sized states on either side of the cut get wrong labels.
- **Relaxing the bijection** (many-to-one). Not allowed: the mapping must be strictly one-to-one.

## 4. Validation: Cross-Sample Consistency

Matching transfers the reference labels onto each sample, so two samples labelled through the same reference can be compared state by state. `match.ipynb` (§4) checks that this transfer keeps the agreement the two samples actually have. For every pair of datasets (A, B) that share one reference vocabulary it computes:

- **Via reference.** The bp accuracy of A and B after both are relabelled by `match_states()`, i.e. the genome share where they carry the same reference label, and its **Kappa**, the same accuracy corrected for the agreement the two label shares reach by chance.
- **Ceiling.** The same accuracy under the best direct bijection between A's and B's own states (Hungarian on their overlap). Two bijections through the reference compose into one bijection A → B, so the reference labels can at best equal it.
- **Random.** The mean accuracy and Kappa over random bijections A → B, the level of agreement labels reach by chance.
- **ARI** of A and B, which compares the two partitions without any state matching.

Two scatterplots, `<workdir>/match/cross_sample_ari_vs_accuracy.pdf` and `<workdir>/match/cross_sample_ari_vs_kappa.pdf`, put bp agreement and Kappa via reference against ARI, each with its Spearman ρ and p-value, the random-bijection level in grey and the line $y = x$; the ceiling is in `cross_sample_summary.tsv`. Transfer works if agreement via reference sits close to the ceiling, well above the random level, and rises with ARI. The check is annotation-free and complements the functional benchmarks: it shows that the labels are consistent across samples, not that they are biologically correct. Pairs across reference vocabularies (ChIP × Mint-ChIP) are left out, because their references name some states differently and would disagree on those labels whatever the mapping.
