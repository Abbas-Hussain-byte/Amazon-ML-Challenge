"""
Business Entity Resolution — baseline pipeline v2 (Amazon ML Challenge 2026)

COMPLIANCE NOTE (read this first):
  - The only model used for actual entity matching here is a GBM classifier
    (LightGBM / sklearn HistGradientBoosting) — MIT/BSD licensed, trained only
    on the provided data, runs 100% offline. No LLM involved in matching.
  - Blocking uses a local sentence-embedding model (default:
    sentence-transformers/all-MiniLM-L6-v2, Apache-2.0, 22M params — far under
    the 8B cap). Weights are downloaded once from HuggingFace, then every
    encode() call runs fully offline on your machine/Colab GPU. No API calls
    at inference time, satisfying "run offline / no live API calls."
  - Nothing here calls Claude/Gemini/ChatGPT/Groq/any hosted LLM. Your coding
    assistant (Antigravity, Cursor, whatever) is a separate concern — it
    writes/runs this code but isn't part of the pipeline being graded.

Architecture:
  1. Normalize names/addresses
  2. Embed every record once with a local, offline sentence-transformer
  3. Block: for each Source-1 entity, retrieve only its top-K nearest
     candidates by cosine similarity (capped set size -> scales to millions
     of records, and keeps candidate_pairs.tsv small, which is now scored)
  4. Feature-engineer each candidate pair (string similarity + embedding sim)
  5. Train a GBM classifier on those features
  6. Tune a decision threshold to maximize macro-averaged per-entity F0.5
     (the exact competition metric — not global/micro F0.5)
  7. Write candidate_pairs.tsv and matching_results.tsv

Run from `code/business_entity_resolution/`:
    python src/pipeline.py train    --data-dir dataset/train --out-dir output
    python src/pipeline.py predict  --train-dir dataset/train --test-dir dataset/test --out-dir output

requirements.txt:
    pandas
    numpy
    scikit-learn
    rapidfuzz
    lightgbm            # optional, falls back to sklearn
    sentence-transformers
    torch
"""

import argparse
import re
import pickle
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupShuffleSplit

try:
    from rapidfuzz.distance import Levenshtein
    def lev_ratio(a, b):
        return Levenshtein.normalized_similarity(a, b)
except ImportError:
    import difflib
    def lev_ratio(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# Apache-2.0, 22M params, well under the 8B cap. Runs fully offline after the
# first download. Swap for 'paraphrase-multilingual-MiniLM-L12-v2' (also
# Apache-2.0, ~118M params) if you see many non-Latin-script names.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K_FINAL = 8          # final candidates per S1 entity (controls submitted set size)
TOP_K_BROAD = 30         # broader recall net before re-ranking/trimming to TOP_K_FINAL
MIN_SIM = 0.35            # drop candidates below this cosine similarity outright
RARE_TOKEN_MAX_FRAC = 0.02   # skip tokens appearing in > 2% of records for token-blocking backstop


# --------------------------------------------------------------------------
# 1. Normalization (unchanged logic, same as v1)
# --------------------------------------------------------------------------

LEGAL_SUFFIXES = {
    "corp", "corporation", "co", "company", "inc", "incorporated", "ltd",
    "limited", "llc", "llp", "pvt", "private", "plc", "gmbh", "sarl", "sa",
    "pty", "bv", "ag", "kg", "srl",
}
ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "apt": "apartment", "fl": "floor",
    "bldg": "building", "hwy": "highway", "ste": "suite", "sq": "square",
    "ct": "court", "pl": "place", "pkwy": "parkway",
}
STOPWORDS = {"the", "and", "of", "for", "near", "at", "in", "a", "an"}
_punct_re = re.compile(r"[^\w\s]")
_ws_re = re.compile(r"\s+")


def normalize_name(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = text.lower().replace("&", " and ")
    t = _punct_re.sub(" ", t)
    tokens = [tok for tok in t.split() if tok not in LEGAL_SUFFIXES]
    return _ws_re.sub(" ", " ".join(tokens)).strip()


def normalize_address(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = _punct_re.sub(" ", text.lower())
    tokens = [ADDR_ABBR.get(tok, tok) for tok in t.split()]
    return _ws_re.sub(" ", " ".join(tokens)).strip()


def significant_tokens(text: str, min_len: int = 3) -> set:
    return {tok for tok in text.split() if len(tok) >= min_len and tok not in STOPWORDS}


def load_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["norm_name"] = df["business_name"].apply(normalize_name)
    df["norm_addr"] = df["business_address"].apply(normalize_address)
    df["name_tokens"] = df["norm_name"].apply(significant_tokens)
    df["addr_tokens"] = df["norm_addr"].apply(significant_tokens)
    df["country_norm"] = df["country"].astype(str).str.strip().str.lower()
    df["embed_text"] = (df["norm_name"] + " " + df["norm_addr"]).str.strip()
    return df


# --------------------------------------------------------------------------
# 2. Local embedding model (offline)
# --------------------------------------------------------------------------

def load_embedder(model_name: str = EMBED_MODEL_NAME):
    if not HAS_ST:
        raise ImportError(
            "sentence-transformers not installed. `pip install sentence-transformers torch`. "
            "This downloads weights once, then all encode() calls run fully offline."
        )
    return SentenceTransformer(model_name)


def encode(model, texts, batch_size=256):
    vecs = model.encode(
        list(texts), batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,  # unit-norm -> dot product == cosine similarity
    )
    return np.asarray(vecs, dtype=np.float32)


# --------------------------------------------------------------------------
# 3. Blocking — capped top-K, scalable
# --------------------------------------------------------------------------

def rare_token_backstop(s1: pd.DataFrame, others: pd.DataFrame, max_frac=RARE_TOKEN_MAX_FRAC) -> dict:
    """
    Token-overlap candidates, but ONLY using tokens rare enough to be useful
    for blocking (skips generic words like 'store', 'restaurant' that would
    otherwise create huge, low-value blocks). Pure recall backstop —
    final size is still controlled downstream by the embedding re-ranking.
    """
    doc_freq = Counter()
    for tokens in others["name_tokens"]:
        for tok in tokens:
            doc_freq[tok] += 1
    n = max(len(others), 1)
    max_count = max(int(max_frac * n), 3)
    useful_tokens = {tok for tok, c in doc_freq.items() if c <= max_count}

    index = defaultdict(set)
    for eid, tokens in zip(others["entity_id"], others["name_tokens"]):
        for tok in tokens & useful_tokens:
            index[tok].add(eid)

    result = defaultdict(set)
    for eid, tokens in zip(s1["entity_id"], s1["name_tokens"]):
        for tok in tokens:
            result[eid] |= index.get(tok, set())
    return result


def generate_candidates(s1: pd.DataFrame, others: pd.DataFrame, embedder) -> pd.DataFrame:
    s1_vecs = encode(embedder, s1["embed_text"])
    other_vecs = encode(embedder, others["embed_text"])
    other_ids = others["entity_id"].values

    k_broad = min(TOP_K_BROAD, len(other_ids))
    nn = NearestNeighbors(n_neighbors=k_broad, metric="cosine").fit(other_vecs)
    dist, idx = nn.kneighbors(s1_vecs)
    embed_sim = 1 - dist  # cosine distance -> similarity

    token_backstop = rare_token_backstop(s1, others)
    other_pos = {eid: i for i, eid in enumerate(other_ids)}

    rows = []
    for row_i, s1_id in enumerate(s1["entity_id"].values):
        cand_scores = {}
        for j, sim in zip(idx[row_i], embed_sim[row_i]):
            cand_scores[other_ids[j]] = float(sim)

        # backstop: add rare-token matches even if outside the embedding top-K_BROAD,
        # scoring them via the already-computed vectors (cheap: one dot product each)
        for cid in token_backstop.get(s1_id, set()):
            if cid not in cand_scores and cid in other_pos:
                sim = float(np.dot(s1_vecs[row_i], other_vecs[other_pos[cid]]))
                cand_scores[cid] = sim

        ranked = sorted(cand_scores.items(), key=lambda x: -x[1])
        kept = [(cid, s) for cid, s in ranked if s >= MIN_SIM][:TOP_K_FINAL]
        for cid, sim in kept:
            rows.append((s1_id, cid, sim))

    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "embed_cos"])


# --------------------------------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------------------------------

def fit_tfidf(df_all: pd.DataFrame, col: str):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    vec.fit(df_all[col])
    return vec


def build_features(pairs: pd.DataFrame, s1: pd.DataFrame, others: pd.DataFrame,
                    name_vec: TfidfVectorizer, addr_vec: TfidfVectorizer) -> pd.DataFrame:
    s1_idx = s1.set_index("entity_id")
    other_idx = others.set_index("entity_id")
    a = s1_idx.loc[pairs["source1_entity_id"]].reset_index(drop=True)
    b = other_idx.loc[pairs["candidate_entity_id"]].reset_index(drop=True)

    feats = pd.DataFrame()
    feats["embed_cos"] = pairs["embed_cos"].values  # from blocking, strong signal — reuse it

    name_a = name_vec.transform(a["norm_name"]); name_b = name_vec.transform(b["norm_name"])
    feats["name_tfidf_cos"] = np.asarray((name_a.multiply(name_b)).sum(axis=1)).ravel()
    addr_a = addr_vec.transform(a["norm_addr"]); addr_b = addr_vec.transform(b["norm_addr"])
    feats["addr_tfidf_cos"] = np.asarray((addr_a.multiply(addr_b)).sum(axis=1)).ravel()

    def jaccard(sa, sb):
        if not sa and not sb: return 1.0
        if not sa or not sb: return 0.0
        u = len(sa | sb)
        return len(sa & sb) / u if u else 0.0

    feats["name_jaccard"] = [jaccard(x, y) for x, y in zip(a["name_tokens"], b["name_tokens"])]
    feats["addr_jaccard"] = [jaccard(x, y) for x, y in zip(a["addr_tokens"], b["addr_tokens"])]
    feats["name_lev"] = [lev_ratio(x, y) for x, y in zip(a["norm_name"], b["norm_name"])]
    feats["addr_lev"] = [lev_ratio(x, y) for x, y in zip(a["norm_addr"], b["norm_addr"])]
    feats["country_match"] = (a["country_norm"].values == b["country_norm"].values).astype(int)
    feats["name_len_diff"] = (a["norm_name"].str.len() - b["norm_name"].str.len()).abs()
    feats["common_name_tokens"] = [len(x & y) for x, y in zip(a["name_tokens"], b["name_tokens"])]

    feats["source1_entity_id"] = pairs["source1_entity_id"].values
    feats["candidate_entity_id"] = pairs["candidate_entity_id"].values
    return feats


FEATURE_COLS = [
    "embed_cos", "name_tfidf_cos", "addr_tfidf_cos", "name_jaccard", "addr_jaccard",
    "name_lev", "addr_lev", "country_match", "name_len_diff", "common_name_tokens",
]


# --------------------------------------------------------------------------
# 5. Ground truth + macro F0.5 (unchanged from v1)
# --------------------------------------------------------------------------

def load_ground_truth(path: Path) -> dict:
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    result = {}
    for _, row in gt.iterrows():
        ids = row["matched_entity_ids"]
        result[row["source1_entity_id"]] = set(ids.split(",")) if ids else set()
    return result


def label_pairs(feat_df: pd.DataFrame, gt: dict) -> pd.Series:
    return feat_df.apply(
        lambda r: int(r["candidate_entity_id"] in gt.get(r["source1_entity_id"], set())), axis=1
    )


def macro_f05(predictions: dict, gt: dict, all_s1_ids) -> float:
    scores = []
    for eid in all_s1_ids:
        pred, true = predictions.get(eid, set()), gt.get(eid, set())
        if not pred and not true:
            scores.append(1.0); continue
        tp = len(pred & true)
        precision = tp / len(pred) if pred else 0.0
        recall = tp / len(true) if true else 0.0
        denom = 0.25 * precision + recall
        scores.append((1.25 * precision * recall) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def tune_threshold(feat_df: pd.DataFrame, probs: np.ndarray, gt: dict, all_s1_ids, grid=None):
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    best_thr, best_score = 0.5, -1
    for thr in grid:
        preds = defaultdict(set)
        for s1, cand, p in zip(feat_df["source1_entity_id"], feat_df["candidate_entity_id"], probs):
            if p >= thr:
                preds[s1].add(cand)
        score = macro_f05(preds, gt, all_s1_ids)
        if score > best_score:
            best_score, best_thr = score, thr
    return best_thr, best_score


# --------------------------------------------------------------------------
# 6. Train
# --------------------------------------------------------------------------

def train(data_dir: Path, out_dir: Path):
    s1 = load_and_normalize(data_dir / "train_source1.tsv")
    s2 = load_and_normalize(data_dir / "train_source2.tsv")
    s3 = load_and_normalize(data_dir / "train_source3.tsv")
    others = pd.concat([s2, s3], ignore_index=True)
    gt = load_ground_truth(data_dir / "train_ground_truth.tsv")
    print(f"S1={len(s1)} S2={len(s2)} S3={len(s3)}")

    embedder = load_embedder()
    candidates = generate_candidates(s1, others, embedder)
    avg_cands = candidates.groupby("source1_entity_id").size().mean() if len(candidates) else 0
    print(f"Blocked candidate pairs: {len(candidates)}  (avg {avg_cands:.1f} candidates / S1 entity)")

    all_true_pairs = {(s1_id, m) for s1_id, ms in gt.items() for m in ms}
    blocked_pairs = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    missed = all_true_pairs - blocked_pairs
    recall_ceiling = 1 - len(missed) / max(len(all_true_pairs), 1)
    print(f"Blocking recall ceiling: {recall_ceiling:.4f} ({len(missed)} true pairs missed)")
    if recall_ceiling < 0.9:
        print("WARNING: raise TOP_K_FINAL/TOP_K_BROAD or lower MIN_SIM — you're losing true "
              "matches before the classifier ever sees them.")

    name_vec = fit_tfidf(pd.concat([s1[["norm_name"]], others[["norm_name"]]]), "norm_name")
    addr_vec = fit_tfidf(pd.concat([s1[["norm_addr"]], others[["norm_addr"]]]), "norm_addr")

    feats = build_features(candidates, s1, others, name_vec, addr_vec)
    feats["label"] = label_pairs(feats, gt)
    print(f"Positive pairs: {feats['label'].sum()} / {len(feats)}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(feats, groups=feats["source1_entity_id"]))
    train_feats, val_feats = feats.iloc[train_idx], feats.iloc[val_idx]
    X_train, y_train = train_feats[FEATURE_COLS], train_feats["label"]
    X_val = val_feats[FEATURE_COLS]

    if HAS_LGB:
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                    class_weight="balanced", random_state=42)
    else:
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_s1_ids = list(val_feats["source1_entity_id"].unique())
    best_thr, best_score = tune_threshold(val_feats, val_probs, gt, val_s1_ids)
    print(f"Best threshold: {best_thr:.2f}  ->  validation macro F0.5: {best_score:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "name_vec": name_vec, "addr_vec": addr_vec,
                     "threshold": best_thr, "embed_model_name": EMBED_MODEL_NAME}, f)
    print(f"Saved model to {out_dir / 'model.pkl'}")


# --------------------------------------------------------------------------
# 7. Predict
# --------------------------------------------------------------------------

def predict(train_dir: Path, test_dir: Path, out_dir: Path, model_path: Path):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model, name_vec, addr_vec, threshold = (
        bundle["model"], bundle["name_vec"], bundle["addr_vec"], bundle["threshold"]
    )
    embedder = load_embedder(bundle.get("embed_model_name", EMBED_MODEL_NAME))

    s1 = load_and_normalize(test_dir / "test_source1.tsv")
    s2 = load_and_normalize(test_dir / "test_source2.tsv")
    s3 = load_and_normalize(test_dir / "test_source3.tsv")
    others = pd.concat([s2, s3], ignore_index=True)

    candidates = generate_candidates(s1, others, embedder)
    feats = build_features(candidates, s1, others, name_vec, addr_vec)

    out_dir.mkdir(parents=True, exist_ok=True)

    cand_map = defaultdict(list)
    for s1_id, cid in zip(feats["source1_entity_id"], feats["candidate_entity_id"]):
        cand_map[s1_id].append(cid)
    with open(out_dir / "candidate_pairs.tsv", "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1["entity_id"]:
            f.write(f"{eid}\t{','.join(dict.fromkeys(cand_map.get(eid, [])))}\n")

    probs = model.predict_proba(feats[FEATURE_COLS])[:, 1] if len(feats) else np.array([])
    match_map = defaultdict(list)
    for s1_id, cid, p in zip(feats["source1_entity_id"], feats["candidate_entity_id"], probs):
        if p >= threshold:
            match_map[s1_id].append(cid)
    with open(out_dir / "matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in s1["entity_id"]:
            f.write(f"{eid}\t{','.join(dict.fromkeys(match_map.get(eid, [])))}\n")

    avg_cands = candidates.groupby("source1_entity_id").size().mean() if len(candidates) else 0
    print(f"Wrote candidate_pairs.tsv (avg {avg_cands:.1f} candidates/entity) and matching_results.tsv")
    print(f"Used threshold={threshold:.2f}. Run utils/validate_submission.py before uploading.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tp = sub.add_parser("train")
    tp.add_argument("--data-dir", type=Path, required=True)
    tp.add_argument("--out-dir", type=Path, required=True)
    pp = sub.add_parser("predict")
    pp.add_argument("--train-dir", type=Path, required=True)
    pp.add_argument("--test-dir", type=Path, required=True)
    pp.add_argument("--out-dir", type=Path, required=True)
    pp.add_argument("--model-path", type=Path, default=None)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.data_dir, args.out_dir)
    elif args.cmd == "predict":
        model_path = args.model_path or (args.out_dir / "model.pkl")
        predict(args.train_dir, args.test_dir, args.out_dir, model_path)


if __name__ == "__main__":
    main()
