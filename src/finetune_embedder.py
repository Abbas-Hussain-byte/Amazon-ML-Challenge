"""
OPTIONAL step. Fine-tunes the local sentence embedder (Apache-2.0, <8B params)
on the competition's own ground-truth match pairs, so blocking gets sharper
at telling apart *this* dataset's noise patterns (abbreviations, transliteration,
typos). Trained only on the provided data -> compliant with the fine-tuning rule.

Everything here runs locally (Colab GPU is fine, CPU works too, just slower).
No external data, no API calls.

Usage:
    python src/finetune_embedder.py \
        --data-dir dataset/train --out-dir models/finetuned_embedder --epochs 3

Then point pipeline.py's EMBED_MODEL_NAME at the saved folder, e.g.:
    EMBED_MODEL_NAME = "models/finetuned_embedder"
"""

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

try:
    from pipeline import load_and_normalize, load_ground_truth, EMBED_MODEL_NAME
except ImportError:
    from src.pipeline import load_and_normalize, load_ground_truth, EMBED_MODEL_NAME


def build_training_examples(s1, others, gt):
    other_idx = others.set_index("entity_id")
    s1_idx = s1.set_index("entity_id")
    examples = []
    for s1_id, matched_ids in gt.items():
        if s1_id not in s1_idx.index:
            continue
        text_a = s1_idx.loc[s1_id, "embed_text"]
        for mid in matched_ids:
            if mid in other_idx.index:
                text_b = other_idx.loc[mid, "embed_text"]
                # MultipleNegativesRankingLoss treats every other item in the
                # batch as an implicit negative for this pair — no need to
                # hand-mine negative pairs.
                examples.append(InputExample(texts=[text_a, text_b]))
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    s1 = load_and_normalize(args.data_dir / "train_source1.tsv")
    s2 = load_and_normalize(args.data_dir / "train_source2.tsv")
    s3 = load_and_normalize(args.data_dir / "train_source3.tsv")
    others = __import__("pandas").concat([s2, s3], ignore_index=True)
    gt = load_ground_truth(args.data_dir / "train_ground_truth.tsv")

    examples = build_training_examples(s1, others, gt)
    print(f"Training on {len(examples)} positive pairs from ground truth")
    if len(examples) < 50:
        print("WARNING: very few positive pairs — fine-tuning may overfit or barely move the "
              "embedder. Consider skipping this step and relying on the pretrained model instead.")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=args.epochs,
        warmup_steps=int(0.1 * len(loader) * args.epochs),
        show_progress_bar=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out_dir))
    print(f"Saved fine-tuned embedder to {args.out_dir}")
    print(f"Update pipeline.py: EMBED_MODEL_NAME = {str(args.out_dir)!r}")


if __name__ == "__main__":
    main()
