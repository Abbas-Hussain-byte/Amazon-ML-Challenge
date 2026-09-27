# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Abbas Hussain  
**Team Members:** Abbas Hussain  
**Submission Date:** September 27, 2026  

---

## 1. Executive Summary

This solution presents a scalable, fully offline, competition-compliant pipeline for multi-source business entity resolution across noisy enterprise datasets. To handle millions of records across diverse countries, varying scripts (English, Devanagari, Telugu), and severe class imbalance (~94.4% match entities vs. ~5.6% singletons), we developed a country-partitioned candidate generation system combining multilingual semantic embeddings (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0, ~118M parameters) with an inverted rare-token index and an exact/high-Levenshtein name-matching backstop. Candidate pairs are scored using character n-gram TF-IDF, token Jaccards, edit distances, and numeric/postal tokens via a LightGBM gradient boosted classifier tuned specifically for the competition's per-entity macro-$F_{0.5}$ metric, achieving a **90.07% overall blocking recall ceiling** (US 95.14%, India 82.59%) and a **0.9537 singleton-adjusted validation macro-$F_{0.5}$**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory analysis of the 2.2M Source-1 training entities and their corresponding Source-2/3 targets revealed several critical characteristics:
1. **Severe Imbalance & Singleton Sparsity:** Analysis of `train_ground_truth.tsv` revealed that **94.415%** of Source-1 entities possess true matches in Source-2 and/or Source-3, while only **5.585%** are true singletons. Evaluating on naive 50/50 samples creates a strong artificial bias toward overly conservative thresholds; calibrating to the true ~94.4%/5.6% ratio was vital for reliable validation.
2. **Multi-Script & Cross-Lingual Variations:** In the Indian partition, numerous entities appear in Latin script in Source 1 but in Devanagari or Telugu scripts (or transliterated variations) in Sources 2 and 3 (e.g., `'Real Care Pvt Ltd'` vs. `'रियल केयर प्रा. लि.'`). Monolingual English embeddings created a severe recall gap (~77% recall ceiling on India vs. ~95% on US).
3. **Empty / Abbreviated Address Fields:** A significant fraction of matching pairs share identical business names after stripping legal suffixes (e.g., `'Real Care Pvt Ltd'` vs. `'REAL PVT CARE LTD'`) but contain missing or truncated address strings, causing standard rare-token frequency caps to drop them.
4. **DBA / Brand vs. Legal Name Disambiguation:** Certain pairs represent holding entities vs. operating brand names (e.g., `'Team Air'` vs. `'Mirasol'`) sharing identical addresses but 0 common name tokens. Given that the competition metric ($F_{0.5}$) places double the weight on precision relative to recall ($\beta=0.5$), aggressively pursuing pure DBA address-only matches without lexical confirmation risks excessive false merges. We treated these as known residual errors to safeguard leaderboard precision.

### 2.2 Solution Strategy
We adopted a **Country-Partitioned Blocking + Gradient Boosted Matching** architecture designed for sub-linear scaling, zero API dependencies, and strict compliance with the $\le 8\text{B}$ parameter offline rule.

```
Source 1, 2, 3 Records
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 1. Normalization & Feature Extraction                  │
│    - Legal suffix stripping, address abbreviations     │
│    - Postal code & numeric token extraction            │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 2. Dynamic Country Partitioning (US, India, France)   │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 3. Scalable Dual-Stream Candidate Blocking             │
│    - Stream A: Multilingual MiniLM Nearest Neighbors   │
│    - Stream B: Rare-Token Backstop (DocFreq ≤ 2%)      │
│    - Stream C: Exact & High-Levenshtein Name Backstop  │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 4. Pairwise Feature Engineering                        │
│    - Embedding sim, Char-WB TF-IDF (Name/Address)      │
│    - Levenshtein & Jaccard, Numeric & Postal match     │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ 5. LightGBM Classifier & Threshold Calibration         │
│    - Per-entity macro-F0.5 optimization (thr = 0.95)   │
└────────────────────────────────────────────────────────┘
          │
          ▼
candidate_pairs.tsv & matching_results.tsv
```

**Approach Type:** Hybrid Multi-Stream Blocking + Gradient Boosted Tree Matching  
**Core Innovations:**
- **Dynamic Country Partitioning:** Strictly restricts comparisons to entities within the same country partition, eliminating cross-border false positives and cutting the quadratic candidate search space into independent, parallelizable sub-spaces.
- **Cross-Lingual Multilingual Sentence Embeddings:** Replaced monolingual embeddings with `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M params) to natively capture semantic similarities across Latin, Devanagari, and Indic scripts.
- **Frequency-Exempt Exact/Near-Match Name Backstop:** Added an inverted hash lookup for exact and token-sorted normalized names alongside a length-bucketed Levenshtein ($> 0.90$) index that bypasses document-frequency caps, recovering hundreds of matches where addresses were empty or short.

---

## 3. Candidate Generation (Blocking)

Candidate generation is the decisive recall bottleneck: any true pair omitted during blocking can never be recovered by the downstream classifier.

- **Blocking Keys & Algorithms Used:**
  1. **Country Partition:** Candidate matches are strictly constrained to records sharing the same country.
  2. **Approximate Nearest Neighbors via Multilingual Embeddings:** Every record is normalized and embedded once into a 384-dimensional unit-normalized vector using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. We query a cosine `NearestNeighbors` index with $K_{\text{broad}} = 50$.
  3. **Rare-Token Inverted Index Backstop:** Tokens appearing in $\le 2\%$ of the partition's records are indexed, guaranteeing that entities sharing distinctive, low-frequency tokens enter the candidate pool even if their full embedding distance is marginal.
  4. **Exact & High-Levenshtein Name Backstop:** Exact normalized name strings, token-sorted names (e.g. `'care real'` == `'real care'`), and name pairs with $\text{Levenshtein} > 0.90$ are retrieved regardless of individual word frequencies.
- **Candidate Set Size:** Bounded to a maximum of $K_{\text{final}} = 20$ candidates per Source-1 entity (average 20.0 candidates / entity), producing clean, compact candidate files that satisfy the candidate-set size evaluation criteria.
- **How True Matches Were Preserved (Empirical Progression):**
  - *Baseline (`all-MiniLM-L6-v2`, 50/50 sample, top-8):* Overall Recall Ceiling = **87.99%** (US: 95.32%, India: 77.10%).
  - *Multilingual Swap (`paraphrase-multilingual-MiniLM-L12-v2`):* Elevated cross-script similarity from 0.596 to 0.672, but diffused dense English embedding ranks when capped at top-8.
  - *Sample Stratification Fix:* Corrected sample ratio from 50/50 to the true 94.415% match / 5.585% singleton distribution.
  - *Name Backstop Rule + Parameter Calibration ($K_{\text{final}} = 20$):* Overall Recall Ceiling increased to **90.07%** (US: **95.14%**, India: **82.59%**), surpassing the 0.90 ceiling target.

---

## 4. Matching Model

### Features Used:
For every candidate pair $(A, B)$ emerging from blocking, we compute 13 dense lexical, semantic, and structural features:
1. `embed_cos`: Precomputed cosine similarity from multilingual sentence embeddings (reused directly from blocking).
2. `name_tfidf_cos`: Cosine similarity of character n-gram TF-IDF vectors (word boundary, $n \in [2, 4]$) on normalized business names.
3. `addr_tfidf_cos`: Cosine similarity of character n-gram TF-IDF vectors (word boundary, $n \in [2, 4]$) on normalized addresses.
4. `name_jaccard`: Word token Jaccard similarity over significant name tokens (stopwords and legal suffixes removed).
5. `addr_jaccard`: Word token Jaccard similarity over significant address tokens.
6. `name_lev`: Normalized Levenshtein similarity ($\in [0, 1]$) between normalized business names.
7. `addr_lev`: Normalized Levenshtein similarity between normalized addresses.
8. `country_match`: Binary indicator confirming matching countries.
9. `name_len_diff`: Absolute character length difference between normalized business names.
10. `common_name_tokens`: Exact count of shared significant name tokens.
11. `postal_match`: Trinary comparison of extracted postal/PIN codes ($+1.0$ for exact match, $-1.0$ for mismatch, $0.0$ if missing in either record).
12. `num_token_jaccard`: Jaccard similarity over numeric tokens extracted from address lines (street numbers, suite/flat numbers).
13. `num_token_overlap`: Raw count of overlapping numeric tokens.

### Model Architecture:
- **Model Type:** LightGBM Classifier (`LGBMClassifier`, 300 estimators, learning rate 0.05, 31 leaves, `class_weight='balanced'`). Fully offline, MIT-licensed, training in seconds on the generated feature matrices.
- **Validation Strategy:** `GroupShuffleSplit` (80/20 split) grouped strictly by `source1_entity_id`, ensuring no Source-1 entity appears in both training and validation folds.
- **Threshold Selection:** Grid search over thresholds $\tau \in [0.05, 0.95]$ evaluating the exact competition metric: macro-averaged per-entity $F_{0.5}$ (where $\text{Precision}$ is weighted $2\times$ as heavily as $\text{Recall}$). Optimal threshold selected: **$\tau = 0.95$**.

---

## 5. Results & Error Analysis

### Performance Metrics:
- **Blocking Recall Ceiling:** **90.07%** overall (US: 95.14%, India: 82.59%).
- **Validation Macro-$F_{0.5}$ (Raw):** **0.9535**
- **Singleton-Adjusted Validation Macro-$F_{0.5}$:** **0.9537**
  - Matched entities macro-$F_{0.5}$: **0.9515**
  - Singleton entities macro-$F_{0.5}$: **0.9903**
- **Submission Validation:** Validated via `utils/validate_submission.py --check-ids` yielding **`PASS — no blocking issues found. Safe to submit.`**

### Error Analysis:
- **Common False Positives (Wrong Merges):**
  - Co-located businesses in multi-tenant commercial centers (e.g. distinct retail stores located in the same mall or complex in Mumbai or New York sharing identical street addresses and generic suffixes like `'Enterprises'`).
  - Mitigated by selecting a high decision threshold ($\tau = 0.95$) that heavily penalizes uncertain merges under $F_{0.5}$.
- **Common False Negatives (Missed Matches):**
  - Holding company vs. operating brand name mismatches (e.g. `'Trusted Telecom Holdings'` vs. `'Haloumbrax'`) where names share zero lexical overlap and addresses are partially incomplete.
  - Extreme phonetic or spelling variations in localized Indic address components combined with missing PIN codes.
- **Disclosed Limitation (France Partition):**
  - The provided training dataset (`dataset/train/`) only contains ground-truth records for **US** and **India**; no training records exist for **France**.
  - While our dynamic country partitioning and character n-gram TF-IDF / multilingual embeddings generalize robustly to French entity syntax, France performance is an out-of-domain evaluation without country-specific ground truth supervision.

---

## 6. Conclusion

By systematically diagnosing blocking bottlenecks—transitioning from monolingual to multilingual embeddings, correcting sample stratification to match the 94.4%/5.6% real-world distribution, and augmenting the token backstop with exact and near-match normalized name rules—we achieved a **90.07% blocking recall ceiling** and a **0.9537 validation macro-$F_{0.5}$**. The resulting pipeline is lightweight, sub-linear in scaling, operates 100% offline without hosted LLMs, and strictly complies with all competition constraints.

---

## Appendix

### A. Code Artefacts & Structure
The complete runnable solution is packaged under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # End-to-end blocking, feature engineering, train, predict
│   ├── create_sample.py       # Representative stratification sampler
│   └── finetune_embedder.py   # Contrastive fine-tuning utility
├── utils/
│   └── validate_submission.py # Official competition submission validator
├── README.md                  # Detailed reproduction instructions
└── requirements.txt           # Pinned dependencies (pandas, scikit-learn, rapidfuzz, lightgbm, sentence-transformers, torch)
```

**Reproduction Commands:**
```bash
# 1. Train the pipeline (offline blocking + LightGBM fitting)
python src/pipeline.py train --data-dir dataset/train --out-dir output

# 2. Predict on test set (generates candidate_pairs.tsv and matching_results.tsv)
python src/pipeline.py predict --train-dir dataset/train --test-dir dataset/test --out-dir output

# 3. Validate output files
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

### B. Summary of Hyperparameters
| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| `EMBED_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0, 118M params, native Devanagari/Telugu subword tokenization |
| `TOP_K_BROAD` | 50 | Initial approximate nearest neighbor candidate net |
| `TOP_K_FINAL` | 20 | Bounded candidate submissions per Source-1 entity |
| `MIN_SIM` | 0.30 | Pre-filter threshold for cosine embedding similarity |
| `RARE_TOKEN_MAX_FRAC` | 0.02 | Frequency ceiling for inverted index token backstop |
| `NAME_LEV_THRESHOLD` | 0.90 | Levenshtein similarity cutoff for frequency-exempt name backstop |
| `CLASSIFIER_THRESHOLD`| 0.95 | Decision threshold calibrated for per-entity macro-$F_{0.5}$ precision weighting |
