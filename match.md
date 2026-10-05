# Chromatin State Matching Scheme and Rationale

## 1. Overview and Objective

De novo chromatin state learning methods (such as ChromHMM, KMeans clustering, or Bernoulli Mixture Models) discover recurrent combinations of histone modifications without prior knowledge of state semantics. Consequently, the learned states are assigned arbitrary identifiers ($E_1, E_2, \dots, E_K$).

To maximize the practical utility of these segmentations, enable rigorous cross-method benchmarking, and allow standardized biological interpretation, each learned state must be mapped to a standardized reference label space (e.g., the ENCODE 15-state reference model: `TssA`, `TssAFlnk`, `TxFlnk`, `Tx`, `TxWk`, `EnhG`, `Enh`, `ZNF/Rpts`, `Het`, `TssBiv`, `BivFlnk`, `EnhBiv`, `ReprPC`, `ReprPCWk`, `Quies`).

The matching pipeline (`scripts/rules/match.py`, invoked via `rules/match.smk` and `match.sh`) resolves this state assignment problem by computing a global, one-to-one optimal alignment between the work states and the reference states.

---

## 2. Current Matching Scheme

### 2.1. Optimization Framework: Maximum Weight Bipartite Matching

State matching is formulated as a maximum-weight linear sum assignment problem (the Hungarian algorithm) between the set of work states $W = \{w_1, \dots, w_{N_w}\}$ and reference states $R = \{r_1, \dots, r_{N_r}\}$:

$$\max_{\pi} \sum_{i} \text{Quality}(w_i, r_{\pi(i)})$$

where $\pi$ defines a bijective mapping from work states to reference states.

### 2.2. Composite Match Quality Metric

The optimization objective combines direct base-pair overlap (Jaccard similarity index, weight 0.80) with biochemical histone mark emission profile similarity (weight 0.20):

$$\text{Quality}(w, r) = \beta \cdot \text{Jaccard}(w, r) + \gamma \cdot \text{EmissionSim}(w, r)$$ 

where $\beta = 0.80$ and $\gamma = 0.20$ when emission profiles are present (and $\beta = 1.0, \gamma = 0.0$ when emissions are unavailable).

### 2.3. Spatial Component: Jaccard Similarity

**Jaccard Similarity Index**:
$$\text{Jaccard}(w, r) = \frac{\text{Overlap}(w, r)}{\text{Length}(w) + \text{Length}(r) - \text{Overlap}(w, r)} \quad \in [0, 1]$$
Provides volume-aware physical overlap concordance, ensuring large genomic domains (`Quies`, `Het`, `Tx`) and focal regulatory elements are anchored accurately according to intersection-over-union.

### 2.4. Epigenetic Emission Similarity

When emission profiles (either bigwig-derived continuous signal or binarized mark presence) are available, mark vectors across shared histone marks are compared:

1. **Profile Overlap (Histogram Intersection)** for active/marked states:
   $$\text{EmissionSim}(w, r) = \sum_{m \in \text{Marks}} \min\left( \frac{E_w(m)}{\sum_{k} E_w(k)}, \frac{E_r(m)}{\sum_{k} E_r(k)} \right)$$

2. **Augmented Cosine Similarity** for profile-less states:
   If a state has zero mark signal ($\max_m E(m) \le 0$), it is augmented with an explicit background component $1.0 - \max_m E(m)$ before calculating cosine similarity, providing a well-conditioned similarity metric for unannotated/quiescent genomic regions.

### 2.5. Joint Matching Across Biological Replicates

For joint models learned simultaneously across multiple replicates (e.g., `rep1` and `rep2`), the total overlap and marginal state lengths are accumulated across all replicates before assignment. A single shared bijective mapping is solved and applied identically to both replicate segmentations, preserving cross-replicate alignment.

---

## 3. Rationale Behind the Hybrid Composite Utility Design

The matching scheme combines biochemical emission profile similarity (20%) and direct base-pair overlap Jaccard index (80%) to maximize annotation utility, biological fidelity, and operational robustness across the genome:

### 3.1. Role of Jaccard Similarity

1. **Volume Awareness and Domain Anchoring**:
   - Chromatin states span widely different genome proportions (from >60% for `Quies` to <0.2% for `TssA`).
   - The Jaccard index ($J = \frac{|A \cap B|}{|A \cup B|}$) naturally normalizes across different state sizes and penalizes large volume disparities. This anchors dominant baseline domains (`Quies`, `Het`, `Tx`) so they cannot be displaced or hijacked by rare, unrepresented states during global Hungarian optimization.
   - For short promoter states (e.g. `TssA`), intersection-over-union provides scores on the same scale ($[0, 1]$) as broad domains.

### 3.2. Why Epigenetic Emission Matching is Required

While spatial co-occurrence provides strong positional evidence, incorporating histone mark emission profile similarity ($\gamma = 0.20$ when available) is essential for robust state matching:

1. **Biochemical Ground Truth Invariant to Spatial Drift**:
   - A chromatin state is fundamentally defined by its characteristic combinatorial histone mark signature (e.g., H3K4me3 for active promoters, H3K27ac/H3K4me1 for enhancers, H3K36me3 for actively transcribed gene bodies, H3K27me3 for Polycomb repression, and H3K9me3 for heterochromatin).
   - While spatial boundaries can drift across peak callers, binarization methods, or sample preparations, the biochemical emission vector remains the definitive physical signature of chromatin identity.

2. **Disambiguating Spatially Adjacent and Correlated States**:
   - Regulatory regions often feature adjacent, spatially concentric chromatin states with overlapping boundaries (e.g., `TssA` vs. `TssAFlnk`, `Enh` vs. `EnhG`, `ReprPC` vs. `ReprPCWk`).
   - Boundary jitter or difference in bin resolution can cause a work promoter state to exhibit comparable spatial overlap with both `TssA` and `TssAFlnk`. The histone mark emission profile (such as the distinct ratio of H3K4me3 to flanking histone marks) provides the crucial biochemical signal needed to resolve such ambiguities accurately.

3. **Preventing Biologically Nonsensical Matches**:
   - Purely spatial assignment can be misled by local spatial displacement or high-entropy boundary bins, occasionally pairing an active regulatory state with a sprawling transcribed or repressive state if spatial overlap happens to dominate.
   - Emission similarity acts as a strong regularizer that penalizes pairings between incompatible histone mark profiles (e.g., pairing a state enriched in repressive mark H3K27me3 with an active promoter state), ensuring biological validity.

4. **Cross-Sample, Cross-Cell-Type, and Cross-Assay Alignment**:
   - When aligning segmentations across different cell types, tissues, or experimental assays (e.g., ChIP-seq vs. Mint-ChIP), genomic loci undergo genuine biological chromatin reorganization and cell-type-specific state switching.
   - Under such biological divergence, spatial overlap naturally degrades. Emission profiles, however, remain invariant across cell types for any given state category, enabling reliable state identification and alignment even when spatial concordance is low.

