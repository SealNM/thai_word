#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune EmbeddingGemma on Thai Words V3 relation triplets."
    )
    parser.add_argument(
        "--train",
        default="artifacts/v3/training_triplets.jsonl",
    )
    parser.add_argument(
        "--base-model",
        default="google/embeddinggemma-300m",
    )
    parser.add_argument(
        "--output",
        default="models/thai-words-embeddinggemma-v3",
    )
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for lower activation memory.",
    )
    parser.add_argument(
        "--no-matryoshka",
        action="store_true",
        help="Train only the full 768d representation (not recommended for V3).",
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        raise SystemExit(f"Training data not found: {train_path}")

    try:
        import torch
        from datasets import load_dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
        )
        from sentence_transformers.losses import (
            CachedMultipleNegativesRankingLoss,
            MatryoshkaLoss,
        )
        from sentence_transformers.training_args import BatchSamplers
    except ImportError as exc:
        raise SystemExit(
            "V3 training dependencies are missing. Install with: "
            "pip install -r requirements.txt"
        ) from exc

    dataset = load_dataset(
        "json",
        data_files=str(train_path),
        split="train",
    )
    required = {"anchor", "positive", "negative"}
    missing = required - set(dataset.column_names)
    if missing:
        raise SystemExit(f"Training dataset is missing columns: {sorted(missing)}")
    dataset = dataset.select_columns(["anchor", "positive", "negative"])

    if len(dataset) < 2:
        raise SystemExit("Need at least 2 training triplets.")

    eval_dataset = None
    train_dataset = dataset
    if args.eval_fraction > 0 and len(dataset) >= 20:
        split = dataset.train_test_split(
            test_size=args.eval_fraction,
            seed=args.seed,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]

    model = SentenceTransformer(args.base_model)

    base_loss = CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=args.mini_batch_size,
    )
    if args.no_matryoshka:
        loss = base_loss
        matryoshka_dims = None
    else:
        matryoshka_dims = [768, 512, 256, 128]
        loss = MatryoshkaLoss(
            model,
            base_loss,
            matryoshka_dims=matryoshka_dims,
        )

    use_bf16 = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        bf16=use_bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={
            "anchor": model.prompts["query"],
            "positive": model.prompts["document"],
            "negative": model.prompts["document"],
        },
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=100 if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=20,
        seed=args.seed,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    summary = {
        "base_model": args.base_model,
        "train_triplets": len(train_dataset),
        "eval_triplets": len(eval_dataset) if eval_dataset is not None else 0,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "learning_rate": args.learning_rate,
        "precision": "bf16" if use_bf16 else "float32",
        "matryoshka_dims": matryoshka_dims,
        "gradient_checkpointing": args.gradient_checkpointing,
        "output": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    trainer.train()

    final_dir = output / "final"
    model.save_pretrained(str(final_dir))
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved final model to {final_dir}")


if __name__ == "__main__":
    main()
