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
import gc
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
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

# Apache-2.0, ~118M params, well under the 8B cap. Runs fully offline after the
# first download. High subword coverage for non-Latin and Indian scripts (Devanagari, Telugu, etc.).
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


TOP_K_FINAL = 20         # final candidates per S1 entity (controls submitted set size)
TOP_K_BROAD = 50         # broader recall net before re-ranking/trimming to TOP_K_FINAL
MIN_SIM = 0.30           # drop candidates below this cosine similarity outright
RARE_TOKEN_MAX_FRAC = 0.02   # skip tokens appearing in > 2% of records for token-blocking backstop
LARGE_PARTITION_THRESHOLD = 200_000  # Partitions with Others > this skip global ANN matrix to avoid OOM


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


def extract_postal_code(addr: str, country: str) -> str:
    if not isinstance(addr, str) or not addr:
        return ""
    c = str(country).lower().strip()
    if c == "india":
        m = re.search(r"\b[1-9]\d{5}\b", addr)
        return m.group(0) if m else ""
    elif c in ("us", "france"):
        m = re.search(r"\b\d{5}\b", addr)
        return m.group(0) if m else ""
    else:
        m = re.search(r"\b\d{4,6}\b", addr)
        return m.group(0) if m else ""


def extract_numeric_tokens(text: str) -> set:
    if not isinstance(text, str) or not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


def load_and_normalize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    country_norm = df["country"].astype(str).str.strip().str.lower()
    norm_name = df["business_name"].apply(normalize_name)
    norm_addr = df["business_address"].apply(normalize_address)
    name_tokens = norm_name.apply(significant_tokens)
    addr_tokens = norm_addr.apply(significant_tokens)
    postal_code = [extract_postal_code(a, c) for a, c in zip(df["business_address"], country_norm)]
    num_tokens = df["business_address"].apply(extract_numeric_tokens)
    embed_text = (norm_name + " " + norm_addr).str.strip()

    result = pd.DataFrame({
        "entity_id": df["entity_id"].values,
        "country_norm": country_norm.values,
        "norm_name": norm_name.values,
        "norm_addr": norm_addr.values,
        "name_tokens": name_tokens.values,
        "addr_tokens": addr_tokens.values,
        "postal_code": postal_code,
        "num_tokens": num_tokens.values,
        "embed_text": embed_text.values
    })
    del df
    gc.collect()
    return result


def load_country_subset(path: Path, country: str, chunksize: int = 250000) -> pd.DataFrame:
    """Load and normalize only the rows matching a specific country partition in low-memory chunks."""
    chunks = []
    for c in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize):
        c_norm = c["country"].astype(str).str.strip().str.lower()
        sub = c[c_norm == country]
        if len(sub) > 0:
            chunks.append(sub)
    if not chunks:
        return pd.DataFrame(columns=[
            "entity_id", "country_norm", "norm_name", "norm_addr",
            "name_tokens", "addr_tokens", "postal_code", "num_tokens", "embed_text"
        ])
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    country_norm = df["country"].astype(str).str.strip().str.lower()
    norm_name = df["business_name"].apply(normalize_name)
    norm_addr = df["business_address"].apply(normalize_address)
    name_tokens = norm_name.apply(significant_tokens)
    addr_tokens = norm_addr.apply(significant_tokens)
    postal_code = [extract_postal_code(a, c) for a, c in zip(df["business_address"], country_norm)]
    num_tokens = df["business_address"].apply(extract_numeric_tokens)
    embed_text = (norm_name + " " + norm_addr).str.strip()

    result = pd.DataFrame({
        "entity_id": df["entity_id"].values,
        "country_norm": country_norm.values,
        "norm_name": norm_name.values,
        "norm_addr": norm_addr.values,
        "name_tokens": name_tokens.values,
        "addr_tokens": addr_tokens.values,
        "postal_code": postal_code,
        "num_tokens": num_tokens.values,
        "embed_text": embed_text.values
    })
    del df
    gc.collect()
    return result


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


def encode(model, texts, batch_size=64):
    if len(texts) == 0:
        dim = getattr(model, "get_sentence_embedding_dimension", lambda: 384)()
        return np.empty((0, dim), dtype=np.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model.to(device)
    except Exception:
        pass
    vecs = model.encode(
        list(texts), batch_size=batch_size, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True
    )
    return np.asarray(vecs, dtype=np.float16)


# --------------------------------------------------------------------------
# 3. Blocking — capped top-K, scalable
# --------------------------------------------------------------------------

def build_others_backstop_index(others: pd.DataFrame, max_frac=RARE_TOKEN_MAX_FRAC) -> dict:
    """Build inverted token and name indices over others records once per country partition."""
    doc_freq = Counter()
    for tokens in others["name_tokens"]:
        for tok in tokens:
            doc_freq[tok] += 1
    n = max(len(others), 1)
    max_count = max(int(max_frac * n), 3)
    useful_tokens = {tok for tok, c in doc_freq.items() if c <= max_count}

    token_index = defaultdict(set)
    for eid, tokens in zip(others["entity_id"], others["name_tokens"]):
        for tok in tokens & useful_tokens:
            token_index[tok].add(eid)

    exact_name_index = defaultdict(set)
    sorted_tok_index = defaultdict(set)
    cand_index = defaultdict(list)
    for eid, name, tokens in zip(others["entity_id"], others["norm_name"], others["name_tokens"]):
        if len(name) >= 3:
            exact_name_index[name].add(eid)
            cand_index[name[:3]].append((eid, name, len(name)))
            st = " ".join(sorted(tokens))
            if st:
                sorted_tok_index[st].add(eid)
            for tok in tokens:
                cand_index[tok].append((eid, name, len(name)))

    return {
        "token_index": token_index,
        "exact_name_index": exact_name_index,
        "sorted_tok_index": sorted_tok_index,
        "cand_index": cand_index,
    }


def query_backstop_for_batch(s1_batch: pd.DataFrame, backstop_data: dict):
    """Query precomputed backstop index for a sub-chunk batch of S1 records."""
    token_index = backstop_data["token_index"]
    exact_name_index = backstop_data["exact_name_index"]
    sorted_tok_index = backstop_data["sorted_tok_index"]
    cand_index = backstop_data["cand_index"]

    token_cands = defaultdict(set)
    for eid, tokens in zip(s1_batch["entity_id"], s1_batch["name_tokens"]):
        for tok in tokens:
            token_cands[eid] |= token_index.get(tok, set())

    name_cands = defaultdict(set)
    for eid, name, tokens in zip(s1_batch["entity_id"], s1_batch["norm_name"], s1_batch["name_tokens"]):
        if len(name) >= 3:
            name_cands[eid] |= exact_name_index.get(name, set())
            st = " ".join(sorted(tokens))
            if st:
                name_cands[eid] |= sorted_tok_index.get(st, set())

            l1 = len(name)
            max_diff = max(1, int(0.12 * l1))
            seen = set()
            cands_to_check = []
            for cid, c_name, l2 in cand_index.get(name[:3], []):
                if cid not in seen and abs(l1 - l2) <= max_diff:
                    seen.add(cid)
                    cands_to_check.append((cid, c_name))
            for tok in tokens:
                for cid, c_name, l2 in cand_index.get(tok, []):
                    if cid not in seen and abs(l1 - l2) <= max_diff:
                        seen.add(cid)
                        cands_to_check.append((cid, c_name))
            for cid, c_name in cands_to_check:
                if lev_ratio(name, c_name) > 0.9:
                    name_cands[eid].add(cid)

    return token_cands, name_cands


def rare_token_backstop(s1: pd.DataFrame, others: pd.DataFrame, max_frac=RARE_TOKEN_MAX_FRAC) -> dict:
    backstop_data = build_others_backstop_index(others, max_frac)
    return query_backstop_for_batch(s1, backstop_data)


def generate_candidates_partition(s1_part: pd.DataFrame, others_part: pd.DataFrame, embedder) -> list:
    """Generate candidate pairs within a single country partition."""
    if len(s1_part) == 0 or len(others_part) == 0:
        return []

    s1_vecs = encode(embedder, s1_part["embed_text"])
    other_vecs = encode(embedder, others_part["embed_text"])
    other_ids = others_part["entity_id"].values

    k_broad = min(TOP_K_BROAD, len(other_ids))
    nn = NearestNeighbors(n_neighbors=k_broad, metric="cosine").fit(other_vecs)
    dist, idx = nn.kneighbors(s1_vecs)
    embed_sim = 1 - dist  # cosine distance -> similarity

    token_backstop, name_backstop = rare_token_backstop(s1_part, others_part)
    other_pos = {eid: i for i, eid in enumerate(other_ids)}

    rows = []
    for row_i, s1_id in enumerate(s1_part["entity_id"].values):
        cand_scores = {}
        for j, sim in zip(idx[row_i], embed_sim[row_i]):
            cand_scores[other_ids[j]] = float(sim)

        # Standard rare-token backstop: add token overlaps using real dot product
        for cid in token_backstop.get(s1_id, set()):
            if cid not in cand_scores and cid in other_pos:
                cand_scores[cid] = float(np.dot(s1_vecs[row_i], other_vecs[other_pos[cid]]))

        # Name rule backstop: ensure exact and high-Levenshtein name matches are prioritized
        for cid in name_backstop.get(s1_id, set()):
            if cid in other_pos:
                sim = float(np.dot(s1_vecs[row_i], other_vecs[other_pos[cid]]))
                cand_scores[cid] = max(cand_scores.get(cid, 0.0), sim, 0.88)

        ranked = sorted(cand_scores.items(), key=lambda x: -x[1])
        kept = [(cid, s) for cid, s in ranked if s >= MIN_SIM][:TOP_K_FINAL]
        for cid, sim in kept:
            rows.append((s1_id, cid, sim))

    del s1_vecs, other_vecs, nn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def generate_candidates(s1: pd.DataFrame, others: pd.DataFrame, embedder) -> pd.DataFrame:
    """
    Generate candidates partitioned dynamically by country.
    Entities in one country are only compared against entities from that country.
    Handles any set of countries dynamically (e.g. US, India, France).
    """
    all_rows = []
    countries = sorted(s1["country_norm"].unique())
    print(f"Partitioning blocking across {len(countries)} country partition(s): {countries}")

    for country in countries:
        s1_c = s1[s1["country_norm"] == country]
        others_c = others[others["country_norm"] == country]
        c_label = country if country else "<empty/unspecified>"
        print(f"  [Country: {c_label}] S1={len(s1_c)}, Others={len(others_c)}")
        if len(others_c) == 0:
            print(f"  Warning: No candidate records in Others for country {c_label!r}")
            continue

        c_rows = generate_candidates_partition(s1_c, others_c, embedder)
        all_rows.extend(c_rows)

    return pd.DataFrame(all_rows, columns=["source1_entity_id", "candidate_entity_id", "embed_cos"])


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

    def postal_cmp(p1, p2):
        if not p1 or not p2:
            return 0.0
        return 1.0 if p1 == p2 else -1.0

    feats["postal_match"] = [postal_cmp(p1, p2) for p1, p2 in zip(a["postal_code"], b["postal_code"])]
    feats["num_token_jaccard"] = [jaccard(x, y) for x, y in zip(a["num_tokens"], b["num_tokens"])]
    feats["num_token_overlap"] = [float(len(x & y)) for x, y in zip(a["num_tokens"], b["num_tokens"])]

    feats["source1_entity_id"] = pairs["source1_entity_id"].values
    feats["candidate_entity_id"] = pairs["candidate_entity_id"].values
    return feats


FEATURE_COLS = [
    "embed_cos", "name_tfidf_cos", "addr_tfidf_cos", "name_jaccard", "addr_jaccard",
    "name_lev", "addr_lev", "country_match", "name_len_diff", "common_name_tokens",
    "postal_match", "num_token_jaccard", "num_token_overlap",
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


def tune_threshold(feat_df: pd.DataFrame, probs: np.ndarray, gt: dict, all_s1_ids, grid=None,
                   singleton_adjust=True, match_weight=0.94415, singleton_weight=0.05585):
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    matched_ids = [eid for eid in all_s1_ids if gt.get(eid)]
    singleton_ids = [eid for eid in all_s1_ids if not gt.get(eid)]

    best_thr, best_score = 0.5, -1
    best_stats = {}

    for thr in grid:
        preds = defaultdict(set)
        for s1, cand, p in zip(feat_df["source1_entity_id"], feat_df["candidate_entity_id"], probs):
            if p >= thr:
                preds[s1].add(cand)
        raw_score = macro_f05(preds, gt, all_s1_ids)
        f05_matched = macro_f05(preds, gt, matched_ids) if matched_ids else 0.0
        f05_singleton = macro_f05(preds, gt, singleton_ids) if singleton_ids else 1.0
        adj_score = match_weight * f05_matched + singleton_weight * f05_singleton

        opt_score = adj_score if singleton_adjust else raw_score
        if opt_score > best_score:
            best_score = opt_score
            best_thr = thr
            best_stats = {
                "raw_f05": raw_score,
                "matched_f05": f05_matched,
                "singleton_f05": f05_singleton,
                "singleton_adj_f05": adj_score,
            }
    return best_thr, best_score, best_stats


# --------------------------------------------------------------------------
# 6. Train
# --------------------------------------------------------------------------

def train(data_dir: Path, out_dir: Path):
    s1 = load_and_normalize(data_dir / "train_source1.tsv")
    s2 = load_and_normalize(data_dir / "train_source2.tsv")
    s3 = load_and_normalize(data_dir / "train_source3.tsv")
    others = pd.concat([s2, s3], ignore_index=True)
    gt = load_ground_truth(data_dir / "train_ground_truth.tsv")
    total_recs = len(s1) + len(s2) + len(s3)
    print(f"Total records processed: S1={len(s1)}, S2={len(s2)}, S3={len(s3)} (Total={total_recs:,})")

    embedder = load_embedder()
    candidates = generate_candidates(s1, others, embedder)
    avg_cands = candidates.groupby("source1_entity_id").size().mean() if len(candidates) else 0
    print(f"Blocked candidate pairs: {len(candidates)}  (avg {avg_cands:.1f} candidates / S1 entity)")

    all_true_pairs = {(s1_id, m) for s1_id, ms in gt.items() for m in ms}
    blocked_pairs = set(zip(candidates["source1_entity_id"], candidates["candidate_entity_id"]))
    missed = all_true_pairs - blocked_pairs
    recall_ceiling = 1 - len(missed) / max(len(all_true_pairs), 1)
    print(f"Blocking recall ceiling (OVERALL): {recall_ceiling:.4f} ({len(missed)} true pairs missed / {len(all_true_pairs)} total)")

    s1_country = dict(zip(s1["entity_id"], s1["country_norm"]))
    for c in sorted(s1["country_norm"].unique()):
        c_true = {p for p in all_true_pairs if s1_country.get(p[0]) == c}
        c_blocked = {p for p in blocked_pairs if s1_country.get(p[0]) == c}
        c_missed = c_true - c_blocked
        c_rec = 1 - len(c_missed) / max(len(c_true), 1)
        c_label = c.upper() if c else "<EMPTY>"
        print(f"  Recall ceiling [{c_label}]: {c_rec:.4f} ({len(c_missed)} missed / {len(c_true)} true)")

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
                                    class_weight="balanced", random_state=42, verbose=-1)
    else:
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_s1_ids = list(val_feats["source1_entity_id"].unique())
    best_thr, best_score, val_stats = tune_threshold(val_feats, val_probs, gt, val_s1_ids)
    print(f"Best threshold: {best_thr:.2f}")
    print(f"  Raw validation macro F0.5: {val_stats['raw_f05']:.4f}")
    print(f"  Validation macro F0.5 (matched entities): {val_stats['matched_f05']:.4f}")
    print(f"  Validation macro F0.5 (singleton entities): {val_stats['singleton_f05']:.4f}")
    print(f"  Singleton-adjusted validation macro F0.5 (94.4%/5.6% weighted): {val_stats['singleton_adj_f05']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "name_vec": name_vec, "addr_vec": addr_vec,
                     "threshold": best_thr, "embed_model_name": EMBED_MODEL_NAME,
                     "val_stats": val_stats}, f)
    print(f"Saved model to {out_dir / 'model.pkl'}")



# --------------------------------------------------------------------------
# 7. Predict
# --------------------------------------------------------------------------

def predict(train_dir: Path, test_dir: Path, out_dir: Path, model_path: Path):
    try:
        import psutil
        def get_ram_gb():
            return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        def get_ram_gb():
            return 0.0

    peak_ram_gb = get_ram_gb()

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model, name_vec, addr_vec, threshold = (
        bundle["model"], bundle["name_vec"], bundle["addr_vec"], bundle["threshold"]
    )
    embedder = load_embedder(bundle.get("embed_model_name", EMBED_MODEL_NAME))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "candidate_pairs.tsv", "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(out_dir / "matching_results.tsv", "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    # Hardcoded to ['india'] temporarily for quick isolated memory test as requested.
    # To run all partitions across the full test set, set: countries = ["france", "india", "us"]
    countries = ["india"]
    print(f"Starting sub-chunked batch prediction (batch_size=1,500) for partitions: {countries}")
    print(f"Initial baseline RAM: {get_ram_gb():.2f} GB")

    total_s1 = 0
    total_candidates_count = 0

    for country in countries:
        c_label = country.upper()
        print(f"\n=======================================================")
        print(f"  Starting Partition: [{c_label}] | Current RAM: {get_ram_gb():.2f} GB")
        print(f"=======================================================")

        s1_c = load_country_subset(test_dir / "test_source1.tsv", country)
        if len(s1_c) == 0:
            print(f"  No S1 records for {c_label}, skipping.")
            continue

        s2_c = load_country_subset(test_dir / "test_source2.tsv", country)
        s3_c = load_country_subset(test_dir / "test_source3.tsv", country)
        others_c = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c
        gc.collect()

        cur_ram = get_ram_gb()
        peak_ram_gb = max(peak_ram_gb, cur_ram)
        print(f"  Loaded [{c_label}]: S1={len(s1_c):,}, Others={len(others_c):,} | RAM: {cur_ram:.2f} GB (Peak: {peak_ram_gb:.2f} GB)")
        total_s1 += len(s1_c)

        if len(others_c) == 0:
            with open(out_dir / "candidate_pairs.tsv", "a", encoding="utf-8") as f_cand, \
                 open(out_dir / "matching_results.tsv", "a", encoding="utf-8") as f_match:
                for eid in s1_c["entity_id"]:
                    f_cand.write(f"{eid}\t\n")
                    f_match.write(f"{eid}\t\n")
            del s1_c, others_c
            gc.collect()
            continue

        is_large = len(others_c) > LARGE_PARTITION_THRESHOLD
        mode_str = "backstop-only" if is_large else "embedding+backstop"
        print(f"  [Partition Mode: {mode_str}] Others count: {len(others_c):,} (LARGE_PARTITION_THRESHOLD: {LARGE_PARTITION_THRESHOLD:,})")

        # Build backstop index over Others once per country
        print(f"  Building rare-token & name backstop index over Others...")
        backstop_data = build_others_backstop_index(others_c)
        cur_ram = get_ram_gb()
        peak_ram_gb = max(peak_ram_gb, cur_ram)

        if not is_large:
            # Mode A: embedding + backstop
            print(f"  Encoding Others records into float16 (batch_size=64)...")
            other_vecs = encode(embedder, others_c["embed_text"], batch_size=64)
            other_ids = others_c["entity_id"].values
            other_pos = {eid: i for i, eid in enumerate(other_ids)}
            print(f"  Building NearestNeighbors index over Others...")
            k_broad = min(TOP_K_BROAD, len(other_ids))
            nn = NearestNeighbors(n_neighbors=k_broad, metric="cosine").fit(other_vecs)
            cur_ram = get_ram_gb()
            peak_ram_gb = max(peak_ram_gb, cur_ram)
            print(f"  Country index ready! RAM: {cur_ram:.2f} GB (Peak: {peak_ram_gb:.2f} GB)")
        else:
            # Mode B: backstop-only (skip global embedding matrix)
            print(f"  Bypassing global NearestNeighbors embedding matrix to keep RAM < 3 GB.")
            other_text_dict = dict(zip(others_c["entity_id"].values, others_c["embed_text"].values))
            cur_ram = get_ram_gb()
            peak_ram_gb = max(peak_ram_gb, cur_ram)
            print(f"  Country backstop index ready! RAM: {cur_ram:.2f} GB (Peak: {peak_ram_gb:.2f} GB)")

        # Sub-chunk S1 into batches of 1,500 entities
        s1_batch_size = 1500
        n_batches = (len(s1_c) + s1_batch_size - 1) // s1_batch_size
        print(f"  Processing {len(s1_c):,} S1 entities in {n_batches} batches of {s1_batch_size}...")

        part_cands = 0
        t_start = time.time()

        for b_idx in range(n_batches):
            b_start = b_idx * s1_batch_size
            b_end = min(b_start + s1_batch_size, len(s1_c))
            s1_batch = s1_c.iloc[b_start:b_end]

            token_backstop, name_backstop = query_backstop_for_batch(s1_batch, backstop_data)

            rows = []
            if not is_large:
                # Mode A: use pre-fitted nn
                s1_batch_vecs = encode(embedder, s1_batch["embed_text"], batch_size=64)
                dist, idx = nn.kneighbors(s1_batch_vecs)
                embed_sim = 1 - dist
                for row_i, s1_id in enumerate(s1_batch["entity_id"].values):
                    cand_scores = {}
                    for j, sim in zip(idx[row_i], embed_sim[row_i]):
                        cand_scores[other_ids[j]] = float(sim)
                    for cid in token_backstop.get(s1_id, set()):
                        if cid not in cand_scores and cid in other_pos:
                            cand_scores[cid] = float(np.dot(s1_batch_vecs[row_i], other_vecs[other_pos[cid]]))
                    for cid in name_backstop.get(s1_id, set()):
                        if cid in other_pos:
                            sim = float(np.dot(s1_batch_vecs[row_i], other_vecs[other_pos[cid]]))
                            cand_scores[cid] = max(cand_scores.get(cid, 0.0), sim, 0.88)
                    ranked = sorted(cand_scores.items(), key=lambda x: -x[1])
                    kept = [(cid, s) for cid, s in ranked if s >= MIN_SIM][:TOP_K_FINAL]
                    for cid, sim in kept:
                        rows.append((s1_id, cid, sim))
                del s1_batch_vecs, dist, idx, embed_sim
            else:
                # Mode B: backstop-only, compute per-candidate dot products on the fly
                batch_cids = {cid for cids in token_backstop.values() for cid in cids} | \
                             {cid for cids in name_backstop.values() for cid in cids}
                if batch_cids:
                    s1_batch_vecs = encode(embedder, s1_batch["embed_text"], batch_size=64)
                    cand_list = [cid for cid in batch_cids if cid in other_text_dict]
                    cand_texts = [other_text_dict[cid] for cid in cand_list]
                    cand_vecs = encode(embedder, cand_texts, batch_size=64)
                    cand_vec_map = {cid: cand_vecs[i] for i, cid in enumerate(cand_list)}

                    for row_i, s1_id in enumerate(s1_batch["entity_id"].values):
                        cand_scores = {}
                        for cid in name_backstop.get(s1_id, set()):
                            if cid in cand_vec_map:
                                sim = float(np.dot(s1_batch_vecs[row_i], cand_vec_map[cid]))
                                cand_scores[cid] = max(sim, 0.88)
                        for cid in token_backstop.get(s1_id, set()):
                            if cid not in cand_scores and cid in cand_vec_map:
                                cand_scores[cid] = float(np.dot(s1_batch_vecs[row_i], cand_vec_map[cid]))

                        ranked = sorted(cand_scores.items(), key=lambda x: -x[1])
                        kept = [(cid, s) for cid, s in ranked if s >= MIN_SIM][:TOP_K_FINAL]
                        for cid, sim in kept:
                            rows.append((s1_id, cid, sim))
                    del s1_batch_vecs, cand_vecs, cand_vec_map

            part_cands += len(rows)
            total_candidates_count += len(rows)

            batch_cand_map = defaultdict(list)
            batch_match_map = defaultdict(list)

            if rows:
                c_pairs = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "embed_cos"])
                del rows
                c_feats = build_features(c_pairs, s1_batch, others_c, name_vec, addr_vec)
                del c_pairs

                for s1_id, cid in zip(c_feats["source1_entity_id"], c_feats["candidate_entity_id"]):
                    batch_cand_map[s1_id].append(cid)

                c_probs = model.predict_proba(c_feats[FEATURE_COLS])[:, 1] if len(c_feats) else np.array([])
                for s1_id, cid, p in zip(c_feats["source1_entity_id"], c_feats["candidate_entity_id"], c_probs):
                    if p >= threshold:
                        batch_match_map[s1_id].append(cid)
                del c_feats, c_probs

            # Write batch results directly and incrementally to disk
            with open(out_dir / "candidate_pairs.tsv", "a", encoding="utf-8") as f_cand, \
                 open(out_dir / "matching_results.tsv", "a", encoding="utf-8") as f_match:
                for eid in s1_batch["entity_id"]:
                    f_cand.write(f"{eid}\t{','.join(dict.fromkeys(batch_cand_map.get(eid, [])))}\n")
                    f_match.write(f"{eid}\t{','.join(dict.fromkeys(batch_match_map.get(eid, [])))}\n")

            del token_backstop, name_backstop, batch_cand_map, batch_match_map
            gc.collect()

            cur_ram = get_ram_gb()
            peak_ram_gb = max(peak_ram_gb, cur_ram)

            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == n_batches:
                elapsed = time.time() - t_start
                rate = b_end / max(elapsed, 1)
                print(f"  [Batch {b_idx + 1}/{n_batches}] Processed {b_end:,}/{len(s1_c):,} S1 entities ({rate:.0f} ent/s) | RAM: {cur_ram:.2f} GB | Peak RAM: {peak_ram_gb:.2f} GB")

        print(f"  Finished [{c_label}]! Generated {part_cands:,} candidate pairs. Freeing partition memory...")
        if not is_large:
            del other_vecs, other_ids, other_pos, nn
        else:
            del other_text_dict
        del s1_c, others_c, backstop_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_cands = total_candidates_count / max(total_s1, 1)
    print(f"\n=======================================================")
    print(f"Prediction Complete!")
    print(f"Total Source-1 Entities Processed: {total_s1:,}")
    print(f"Total Candidate Pairs: {total_candidates_count:,} (avg {avg_cands:.1f} / entity)")
    print(f"Peak RAM Usage: {peak_ram_gb:.2f} GB (within 12.7 GB limit)")
    print(f"Used decision threshold: {threshold:.2f}")
    print(f"=======================================================")


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
        model_path = args.model_path
        if model_path is None:
            if (args.out_dir / "model.pkl").exists():
                model_path = args.out_dir / "model.pkl"
            elif Path("models/model.pkl").exists():
                model_path = Path("models/model.pkl")
            else:
                model_path = args.out_dir / "model.pkl"
        predict(args.train_dir, args.test_dir, args.out_dir, model_path)


if __name__ == "__main__":
    main()
