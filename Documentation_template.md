# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Null pointers  
**Team Members:**  
- Abbas Hussain (Team Leader)  
- Shaikh Mohd Rehaan  
- Pattapu Solomon  
- Boggula Raghu Rami Reddy  

**Submission Date:** 27-09-2026  

---

## 1. Executive Summary

To resolve business entities across noisy, multi-lingual, and incomplete enterprise records at scale, our team developed a fully offline, two-stage entity resolution pipeline coupling country-partitioned candidate blocking with a gradient-boosted LightGBM matching classifier. We tackled the severe cross-script and transliteration bottleneck in the Indian partition by migrating from an English-only embedder to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~118M parameters, Apache-2.0) and introducing a frequency-exempt exact and near-match name backstop ($\text{Levenshtein} > 0.90$) for records with sparse or missing addresses. By calibrating our decision threshold directly against the competition's precision-weighted per-entity macro-$F_{0.5}$ metric, our pipeline achieves an **overall blocking recall ceiling of 90.07%** (US: 95.14%, India: 82.59%) and a **singleton-adjusted validation macro-$F_{0.5}$ of 0.9537**, operating strictly within competition parameter and offline runtime constraints.

---

## 2. Methodology

### 2.1 Problem Analysis
When our team first dove into exploratory data analysis across the 2.2 million Source-1 entities and their Source-2/3 counterparts, we noticed several crucial data patterns that shaped our entire approach:

1. **The Real Ground Truth Split (94.4% Matches vs. 5.6% Singletons):**  
   Digging into `train_ground_truth.tsv`, we found that **2,083,574 out of 2,206,821** Source-1 records (94.415%) have true matching counterparts in Source 2 or 3, while only **123,247** (5.585%) are true singletons with no match anywhere. Early on, our development sample had been split 50/50 between matches and singletons. That arbitrary split misled our threshold tuning—the classifier learned to pick a very high threshold of 0.95 simply because predicting "no match" gave an automatic 1.0 macro-$F_{0.5}$ on half the validation set. Once we corrected our local stratification in `src/create_sample.py` to match the real 94.4%/5.6% distribution, our metrics reflected reality and we could tune thresholds with confidence.

2. **Cross-Script Transliteration in India:**  
   The US partition was relatively clean English text, but India had massive script and phonetic variation. Source-1 records were often registered in legal English (`"Real Care Pvt Ltd"`), while Sources 2 and 3 frequently contained the exact same entity in Devanagari (`"रियल केयर प्रा. लि."`) or Telugu. Our initial English-only baseline (`all-MiniLM-L6-v2`) stalled at 77.10% recall on India. Switching to a multilingual sentence transformer immediately gave us shared subword embeddings across Latin, Devanagari, and Telugu alphabets, closing a huge gap in candidate retrieval.

3. **Empty or Truncated Addresses with Common Name Words:**  
   When analyzing false negatives row by row (like Row 6 in our error reports: `"Real Care Pvt Ltd"` vs `"REAL PVT CARE LTD"`), we noticed that both records had nearly identical names, but one had an empty address. Because words like `"real"` and `"care"` appeared frequently across the corpus, our standard inverted-index token backstop (which capped word frequencies at $\le 2\%$) ignored them as common words. As a result, the true pair was completely dropped before reaching the classifier. We realized we needed a dedicated backstop: if the normalized name matches exactly or has a Levenshtein similarity $> 0.90$, we must bring it into the candidate set regardless of individual word frequencies.

4. **Trade Names (DBAs) and Dense Commercial Hubs:**  
   We also saw pairs like Row 13 in our error reports (`"Trusted Telecom Holdings"` vs `"Haloumbrax"`). These entities share 0 name tokens and can only be connected through identical street addresses. But in dense business parks in Bengaluru or commercial buildings in New York, dozens of completely unrelated companies share the exact same street address. Under the macro-$F_{0.5}$ metric (which penalizes false positives twice as heavily as false negatives, $\beta = 0.5$), making speculative merges based solely on address causes far more damage than missing a few obscure holding-company pairs. We made a conscious engineering decision to leave these as known residual errors to protect our precision.

### 2.2 Solution Strategy

We built an asymmetric, two-stage architecture designed for high candidate recall in Stage 1 and precision-optimized classification in Stage 2.

```
                         Multi-Source Records (S1, S2, S3)
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │    Text Normalization Engine      │
                      │ - Legal suffixes (pvt, ltd, inc)  │
                      │ - Common address abbreviations    │
                      │ - Postal code extraction          │
                      └───────────────────────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │   Dynamic Country Partitioning    │
                      │      [US]     [India]   [France]  │
                      └───────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────┐
                 │       Stage 1: Multi-Stream Blocking         │
                 │ Stream A: Multilingual MiniLM ANN (K=50)     │
                 │ Stream B: Rare Word Inverted Index (≤2% freq)│
                 │ Stream C: Exact & Near-Name Backstop (>0.90) │
                 └──────────────────────────────────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │     Bounded Candidate Selection   │
                      │   (Top K_final = 20 per S1 entity)│
                      └───────────────────────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │   Stage 2: Feature Engineering    │
                      │  13 Lexical, Phonetic, Semantic,  │
                      │  Numeric, & Structural Features   │
                      └───────────────────────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │     LightGBM Match Classifier     │
                      │ - Offline, 300 trees, 31 leaves   │
                      │ - Calibrated threshold (τ = 0.95) │
                      └───────────────────────────────────┘
                                        │
                                        ▼
                   Final Outputs: matching_results.tsv &
                                  candidate_pairs.tsv
```

**Approach Type:** Hybrid (Multi-Stream Candidate Blocking + Gradient-Boosted Classification)  
**Core Innovation:** A dual-stream candidate blocking engine that fuses multilingual sentence embeddings with a frequency-exempt normalized name backstop. This ensures that cross-script Indic entities and short/empty-address entities are retained in the top-20 candidate pool without blowing up candidate-set sizes or suffering from stopword filtering.

---

## 3. Candidate Generation (Blocking)

Candidate generation is the most critical stage of the pipeline—any true match missed during blocking is lost forever. We designed our blocking phase to maximize recall while keeping the candidate set bounded and compact.

- **Blocking keys used:**
  1. **Strict Country Partitioning:** Records are split by normalized country (`us`, `india`, `france`). Entities in India are never compared against entities in the US or France, completely eliminating cross-border false merges and reducing comparisons from $O(N \times M)$ to $O(\sum N_c \log M_c)$.
  2. **Multilingual Dense Semantic Search:** Names and addresses are normalized and embedded using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, L2-normalized). We fit a cosine `NearestNeighbors` index over Source-2 and Source-3 records to retrieve the top $K_{\text{broad}} = 50$ nearest neighbors for every Source-1 entity.
  3. **Rare-Token Inverted Index:** Word tokens appearing in $\le 2\%$ of records within each country partition are indexed into an inverted hash map. If a Source-1 entity shares an uncommon brand, founder, or street word with a Source-2/3 record, that record is pulled in.
  4. **Exact & High-Levenshtein Name Backstop:** We hash normalized full names and token-sorted names (e.g., `'care real'` matches `'real care'`), and pre-filter length-bucketed candidate pairs to evaluate $\text{Levenshtein} > 0.90$. These pairs bypass the 2% frequency limit and receive an elevated candidate priority score ($\ge 0.88$) so they cannot be crowded out by generic address embedding matches.

- **Candidate pairs generated:**
  - Bounded to a strict maximum of $K_{\text{final}} = 20$ candidates per Source-1 entity.
  - On our validation sample of 10,000 Source-1 entities, exactly 200,000 candidate pairs were evaluated (20.0 candidates / entity).
  - For the test set of 400,000 Source-1 entities, this produces exactly 8,000,000 candidate pairs in `candidate_pairs.tsv`.

- **How you ensured true matches were not lost:**
  - *Multi-stream redundancy:* If an embedding similarity drifts due to transliteration or unusual syntax, the rare-token inverted index catches it. If the address is completely blank, the normalized name backstop rescues it.
  - *Priority boosting:* Candidates retrieved via the exact/near-match name backstop are injected with an artificial similarity boost (`max(sim, 0.88)`), preventing them from being filtered out by the general cosine similarity floor (`MIN_SIM = 0.30`) or cut off when truncating to $K_{\text{final}} = 20$.
  - *Empirical tracking:* We tracked our blocking recall ceiling across four major development iterations:

| Development Iteration | Overall Recall Ceiling | India Recall Ceiling | US Recall Ceiling | Engineering Finding |
| :--- | :---: | :---: | :---: | :--- |
| **1. Baseline (`all-MiniLM-L6-v2`)** | 87.99% | 77.10% | 95.32% | English-only model struggled on Indic script transliteration. |
| **2. Multilingual Model Swap** | 83.60% | 73.82% | 90.18% | Cross-lingual similarity improved, but tight candidate capping ($K=8$) dropped valid pairs. |
| **3. Stratification Correction** | 82.99% | 73.22% | 89.62% | Replaced artificial 50/50 sample with real 94.4%/5.6% split; grounded our validation. |
| **4. Name Backstop + $K_{\text{final}}=20$** | **90.07%** | **82.59%** | **95.14%** | Rescued empty-address exact name matches; pushed recall ceiling above 90%. |

---

## 4. Matching Model

### 4.1 Feature Engineering
For every candidate pair that clears the blocking stage, we compute 13 dense features:

**Features used:**
- **Name features:**
  - `name_tfidf_cos`: Character n-gram TF-IDF cosine similarity ($n \in [2, 4]$, subword boundary analyzer).
  - `name_jaccard`: Word token Jaccard similarity across significant name tokens (legal suffixes and stopwords removed).
  - `name_lev`: Normalized Levenshtein similarity ($\in [0, 1]$) between full normalized names via RapidFuzz.
  - `name_len_diff`: Absolute character length difference between normalized names.
  - `common_name_tokens`: Absolute count of overlapping significant name tokens.
- **Address features:**
  - `addr_tfidf_cos`: Character n-gram TF-IDF cosine similarity ($n \in [2, 4]$) on normalized addresses.
  - `addr_jaccard`: Word token Jaccard similarity across address tokens.
  - `addr_lev`: Normalized Levenshtein similarity on normalized addresses.
  - `postal_match`: Trinary comparison of parsed postal/PIN codes ($+1.0$ exact match, $-1.0$ explicit mismatch, $0.0$ if missing in either record).
  - `num_token_jaccard`: Jaccard similarity over numeric tokens (building numbers, suite numbers, plot numbers).
  - `num_token_overlap`: Total count of overlapping numeric tokens.
- **Other features:**
  - `embed_cos`: Precomputed cosine similarity from our multilingual sentence transformer (reused directly from blocking with zero additional computation).
  - `country_match`: Binary indicator verifying country alignment ($1.0$ if identical, $0.0$ otherwise).

### 4.2 Classifier Architecture & Hyperparameters
- **Model type:** LightGBM Classifier (`LGBMClassifier`), configured with 300 estimators, a learning rate of 0.05, 31 leaves, and `class_weight='balanced'`. It is completely self-contained, MIT-licensed, executes 100% offline, and trains in seconds without requiring GPU acceleration.
- **Validation Splitting:** We enforce an 80/20 `GroupShuffleSplit` strictly grouped by `source1_entity_id`. This prevents any data leakage: no entity seen during model training ever appears in the validation split.
- **Threshold selection method:** We evaluate decision thresholds across $\tau \in [0.05, 0.95]$ with a step size of 0.05 directly optimizing the competition's target metric: **macro-averaged per-entity $F_{0.5}$** ($\text{Precision}$ weighted twice as heavily as $\text{Recall}$, $\beta = 0.5$). The optimal threshold chosen by our grid search was **$\tau = 0.95$**. This threshold is intentionally conservative, prioritizing high precision to protect against incorrect merges in multi-tenant commercial properties.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):**
  - **Overall Singleton-Adjusted Validation Macro-$F_{0.5}$:** **0.9537** (Raw: 0.9535)
  - Matched entities macro-$F_{0.5}$: **0.9515**
  - Singleton entities macro-$F_{0.5}$: **0.9903**
  - Blocking Recall Ceiling: **90.07%** (US: 95.14%, India: 82.59%)
  - Organizer Format Validation: Verified via `utils/validate_submission.py --check-ids` yielding **`PASS — no blocking issues found. Safe to submit.`** (0 formatting errors, 0 duplicate predictions, 0 invalid IDs).

- **Common false positives (wrong merges):**
  - *Multi-tenant commercial centers:* Unrelated businesses sharing the same corporate park, shopping complex, or street address in dense urban centers (e.g., Lower Parel in Mumbai, or Broadway in New York) that also share generic words like `"Enterprises"`, `"Services"`, or `"Trading"`.
  - *How we handled it:* Our high decision threshold ($\tau = 0.95$) combined with numeric token overlap features and subword n-gram TF-IDF successfully suppresses these ambiguous merges.

- **Common false negatives (missed matches):**
  - *Holding companies and DBAs (Doing Business As):* Entities such as `"Trusted Telecom Holdings"` vs. `"Haloumbrax"` that share zero lexical or phonetic overlap in their names. As noted earlier, merging these on address alone causes widespread precision damage across commercial buildings, so leaving them unmerged was the right trade-off under $F_{0.5}$.
  - *Heavy colloquial transliteration drift:* Rare instances where colloquial Indic business names have highly non-standard phonetic spellings in English that fall just outside our Levenshtein and character n-gram thresholds.

- **Disclosed Limitation (France Partition):**
  - The provided training dataset (`dataset/train/`) only contains ground truth labels for the **US** and **India**; there are **zero training annotations for France**.
  - While our pipeline handles French entities zero-shot using language-agnostic character n-grams and universal multilingual embeddings, we disclose this transparently as an out-of-domain evaluation setting without country-specific ground truth supervision.

---

## 6. Conclusion

By systematically diagnosing our blocking bottlenecks—migrating to a multilingual sentence transformer, correcting our validation split to match the true 94.4%/5.6% ground-truth distribution, and adding an exact/near-match name backstop—our team raised our blocking recall ceiling to **90.07%** and achieved a **0.9537 validation macro-$F_{0.5}$**. Our complete pipeline runs 100% offline, uses an Apache-2.0 model well within the 8B parameter limit, requires no external APIs or large language models, and produces fully verified, compliant submission artefacts.

---

## Appendix

### A. Code Artefacts
Our complete, runnable code is packaged inside `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # End-to-end blocking, featurization, training, and prediction
│   ├── create_sample.py       # Statistically stratified sample generation utility
│   └── finetune_embedder.py   # Optional contrastive embedding fine-tuning script
├── utils/
│   ├── validate_submission.py # Official competition submission validator
│   └── package_submission.py  # Automated submission packaging tool
├── README.md                  # Complete reproduction instructions
└── requirements.txt           # Pinned dependencies
```

**Reproduction Commands:**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the pipeline and calibrate threshold
python src/pipeline.py train --data-dir dataset/train --out-dir output

# 3. Generate test predictions (creates candidate_pairs.tsv and matching_results.tsv)
python src/pipeline.py predict --train-dir dataset/train --test-dir dataset/test --out-dir output

# 4. Validate output files against all formatting and ID rules
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids

# 5. Package final submission zip
python utils/package_submission.py --team-name Null_pointers --output-tsv-dir output
```

### B. Additional Results & Hyperparameter Reference Table

| Hyperparameter | Value | Description & Engineering Rationale |
| :--- | :---: | :--- |
| `EMBED_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0, ~118M params. Multilingual subword tokenization bridging Latin, Devanagari, and Telugu. |
| `TOP_K_BROAD` | 50 | Initial approximate nearest-neighbor search net before re-ranking. |
| `TOP_K_FINAL` | 20 | Bounded candidate set per Source-1 entity submitted in `candidate_pairs.tsv`. |
| `MIN_SIM` | 0.30 | Cosine similarity floor for general embedding candidates. |
| `RARE_TOKEN_MAX_FRAC` | 0.02 | 2% document frequency cap for the rare-token inverted index. |
| `NAME_LEV_THRESHOLD` | 0.90 | Normalized Levenshtein cutoff for the frequency-exempt name backstop. |
| `CLASSIFIER_THRESHOLD` | 0.95 | Optimal decision threshold tuned for per-entity macro-$F_{0.5}$ precision weighting. |
| `LGBM_N_ESTIMATORS` | 300 | Number of boosted trees in the matching classifier. |
| `LGBM_NUM_LEAVES` | 31 | Maximum tree leaves per base learner. |
| `LGBM_LEARNING_RATE` | 0.05 | Learning rate with shrinkage for stable convergence. |
