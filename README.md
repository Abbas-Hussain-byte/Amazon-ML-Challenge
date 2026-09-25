# Business Entity Resolution — baseline (v2, offline-compliant)

## Compliance summary
- **Matcher** = GBM classifier (LightGBM/sklearn). MIT/BSD-licensed, not an
  LLM, trained only on your data, runs offline. No 8B-param question even
  applies to it.
- **Blocking** = local sentence-transformer (`all-MiniLM-L6-v2`, Apache-2.0,
  22M params). Downloaded once from HuggingFace, then every run is fully
  offline — no live API calls, satisfying the organizer's rule.
- **No hosted LLM APIs anywhere** in this pipeline (no Claude/Gemini/ChatGPT/
  Groq calls). Your coding assistant (Antigravity, etc.) is a separate,
  ungraded concern — it's the tool you use to write/run this code, not part
  of what gets evaluated.
- Candidate set size is now bounded per Source-1 entity (`TOP_K_FINAL` in
  `pipeline.py`, default 8) — directly addresses "smaller candidate set
  ranks higher," and scales to large datasets since blocking is O(n log n)
  approximate nearest-neighbor search, not O(n×m) all-pairs comparison.

## Setup (Google Colab recommended — free GPU, nothing to install manually)
```bash
pip install -r requirements.txt
```
First run downloads the embedding model once (~90MB); after that it's
offline. Place the competition's `dataset/` folder next to `src/`.

## Run
```bash
# 1. Train — prints blocking recall ceiling, avg candidates/entity, and
#    validation macro-F0.5 (the exact competition metric)
python src/pipeline.py train --data-dir dataset/train --out-dir output

# 2. (Optional but recommended) Fine-tune the embedder on your ground truth —
#    sharpens blocking for this dataset's specific noise patterns
python src/finetune_embedder.py --data-dir dataset/train --out-dir models/finetuned_embedder
# then edit EMBED_MODEL_NAME in pipeline.py to "models/finetuned_embedder" and re-run train

# 3. Predict — writes output/candidate_pairs.tsv and output/matching_results.tsv
python src/pipeline.py predict --train-dir dataset/train --test-dir dataset/test --out-dir output

# 4. Validate before uploading (organizer-provided)
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Beginner's guide: what to look at, in priority order

1. **Recall ceiling first, always.** Printed by `train`. If a true match
   never makes it into your candidate set, no amount of classifier tuning
   can recover it. Below ~0.90 → raise `TOP_K_BROAD`/`TOP_K_FINAL` or lower
   `MIN_SIM` in `pipeline.py`, re-run, re-check.
2. **Then avg candidates/entity.** This is your new scored dimension. Once
   recall ceiling looks solid, try lowering `TOP_K_FINAL` (8 → 6 → 5) and
   watch whether validation macro-F0.5 holds steady — if it does, you've
   found a smaller, equally-good candidate set, which now directly helps
   your ranking beyond the leaderboard score.
3. **Then validation macro-F0.5 from `train`.** This is your literal
   competition metric, computed exactly as the organizers compute it
   (per-entity, precision-weighted, singletons included). Trust this number
   over "does the code look sophisticated" — a good macro-F0.5 with a
   simple model beats a fancy model with a mediocre one.
4. **Feature ideas that tend to help most on this kind of noisy business
   data**, roughly in order of effort-to-payoff:
   - Numeric-token exact match (street numbers, PIN/ZIP/postal codes) as a
     precision feature — numbers rarely coincide by chance, unlike words.
   - DBA/trade-name splitting if you spot `"X (dba Y)"` patterns in names.
   - Country-specific address parsing (US ZIP formats vs. Indian PIN codes
     vs. French postal codes) as separate features instead of one blob.
5. **Fine-tuning the embedder** (`finetune_embedder.py`) is worth doing once
   your pipeline runs end-to-end and you have a baseline score to beat —
   don't start here, it's a multiplier on an already-working pipeline, not
   a fix for a broken one.
6. **Methodology document**: your candidate generation section should
   explicitly state the blocking approach (embedding top-K + rare-token
   backstop), why it scales (approximate nearest-neighbor, not all-pairs),
   and report the recall ceiling + avg candidate-set size numbers `train`
   prints — that's exactly what reviewers are asking to see.
