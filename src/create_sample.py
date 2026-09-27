"""
Utility to create a valid, representative sample of the dataset for fast local development.
Maintains ground truth linkage so blocking recall and macro-F0.5 can be accurately tested.
"""

import argparse
from pathlib import Path
import random
import pandas as pd


def create_sample(full_train_dir: Path, full_test_dir: Path, 
                  out_train_dir: Path, out_test_dir: Path,
                  n_s1_train: int = 10000, n_s1_test: int = 3000,
                  seed: int = 42):
    random.seed(seed)
    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_test_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Sampling Training Data ({n_s1_train} S1 entities) ---")
    gt_df = pd.read_csv(full_train_dir / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    # Stratified-style sample: match the real distribution (~94.415% matches, ~5.585% singletons)
    has_matches = gt_df["matched_entity_ids"] != ""
    multi_df = gt_df[has_matches]
    single_df = gt_df[~has_matches]
    
    # Real ~94.4% entities with matches, ~5.6% singletons
    match_ratio = 0.94415
    n_multi = min(len(multi_df), int(round(n_s1_train * match_ratio)))
    n_single = n_s1_train - n_multi
    
    sample_gt = pd.concat([
        multi_df.sample(n=n_multi, random_state=seed),
        single_df.sample(n=n_single, random_state=seed)
    ], ignore_index=True)
    
    s1_ids = set(sample_gt["source1_entity_id"])
    target_s2_ids = set()
    target_s3_ids = set()
    
    for ids_str in sample_gt["matched_entity_ids"]:
        if ids_str:
            for mid in ids_str.split(","):
                mid = mid.strip()
                if mid.startswith("S2-"):
                    target_s2_ids.add(mid)
                elif mid.startswith("S3-"):
                    target_s3_ids.add(mid)
                    
    print(f"Selected {len(s1_ids)} S1 entities.")
    print(f"Ground-truth required matches: {len(target_s2_ids)} S2 IDs, {len(target_s3_ids)} S3 IDs.")

    # Save sampled ground truth
    sample_gt.to_csv(out_train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    # Filter S1
    print("Filtering train_source1...")
    s1_df = pd.read_csv(full_train_dir / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s1_sample = s1_df[s1_df["entity_id"].isin(s1_ids)].copy()
    s1_sample.to_csv(out_train_dir / "train_source1.tsv", sep="\t", index=False)
    print(f"Saved {len(s1_sample)} records to train_source1.tsv")

    # Filter S2: Keep all true matches + add distractors
    print("Filtering train_source2...")
    s2_df = pd.read_csv(full_train_dir / "train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2_matches = s2_df[s2_df["entity_id"].isin(target_s2_ids)]
    s2_distractors = s2_df[~s2_df["entity_id"].isin(target_s2_ids)].sample(n=min(len(s2_df) - len(s2_matches), n_s1_train * 2), random_state=seed)
    s2_sample = pd.concat([s2_matches, s2_distractors], ignore_index=True)
    s2_sample.to_csv(out_train_dir / "train_source2.tsv", sep="\t", index=False)
    print(f"Saved {len(s2_sample)} records to train_source2.tsv ({len(s2_matches)} true matches, {len(s2_distractors)} distractors)")

    # Filter S3: Keep all true matches + add distractors
    print("Filtering train_source3...")
    s3_df = pd.read_csv(full_train_dir / "train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3_matches = s3_df[s3_df["entity_id"].isin(target_s3_ids)]
    s3_distractors = s3_df[~s3_df["entity_id"].isin(target_s3_ids)].sample(n=min(len(s3_df) - len(s3_matches), n_s1_train * 2), random_state=seed)
    s3_sample = pd.concat([s3_matches, s3_distractors], ignore_index=True)
    s3_sample.to_csv(out_train_dir / "train_source3.tsv", sep="\t", index=False)
    print(f"Saved {len(s3_sample)} records to train_source3.tsv ({len(s3_matches)} true matches, {len(s3_distractors)} distractors)")

    print(f"\n--- Sampling Test Data ({n_s1_test} S1 entities) ---")
    test_s1_df = pd.read_csv(full_test_dir / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    # Ensure France, US, India are all included
    subsets = []
    for c_val, group in test_s1_df.groupby("country"):
        n_pick = min(len(group), max(1, n_s1_test // test_s1_df["country"].nunique()))
        subsets.append(group.sample(n=n_pick, random_state=seed))
    test_s1_sample = pd.concat(subsets, ignore_index=True)
    test_s1_sample.to_csv(out_test_dir / "test_source1.tsv", sep="\t", index=False)
    print(f"Saved {len(test_s1_sample)} test S1 records (Countries: {test_s1_sample['country'].value_counts().to_dict()})")

    test_s2_df = pd.read_csv(full_test_dir / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    test_s2_sample = test_s2_df.sample(n=min(len(test_s2_df), n_s1_test * 3), random_state=seed)
    test_s2_sample.to_csv(out_test_dir / "test_source2.tsv", sep="\t", index=False)
    print(f"Saved {len(test_s2_sample)} test S2 records")

    test_s3_df = pd.read_csv(full_test_dir / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    test_s3_sample = test_s3_df.sample(n=min(len(test_s3_df), n_s1_test * 3), random_state=seed)
    test_s3_sample.to_csv(out_test_dir / "test_source3.tsv", sep="\t", index=False)
    print(f"Saved {len(test_s3_sample)} test S3 records")
    print("\nSample dataset successfully created!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-train-dir", type=Path, default=Path("dataset/train"))
    ap.add_argument("--full-test-dir", type=Path, default=Path("dataset/test"))
    ap.add_argument("--out-train-dir", type=Path, default=Path("dataset/sample_train"))
    ap.add_argument("--out-test-dir", type=Path, default=Path("dataset/sample_test"))
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-test", type=int, default=3000)
    args = ap.parse_args()

    create_sample(args.full_train_dir, args.full_test_dir, 
                  args.out_train_dir, args.out_test_dir, 
                  n_s1_train=args.n_train, n_s1_test=args.n_test)
