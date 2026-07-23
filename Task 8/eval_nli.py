"""
loads the best checkpoint produced by train_qwen_nli.py following the Nautilus Job
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def find_best_checkpoint(output_dir):
    """Return the trainer's recorded best checkpoint, else the latest one."""
    checkpoints = sorted(
        (d for d in os.listdir(output_dir) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[1]),
    )
    if not checkpoints:
        raise SystemExit(f"No checkpoints found under {output_dir}")
    latest = os.path.join(output_dir, checkpoints[-1])
    state_path = os.path.join(latest, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            best = json.load(f).get("best_model_checkpoint")
        if best and os.path.isdir(best):
            return best
    return latest


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an SNLI checkpoint.")
    parser.add_argument(
        "--output-dir",
        default="Task 8/outputs/qwen3-0.6b-nli-full",
        help="Run directory containing checkpoint-* folders; metrics.json is written here.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Specific checkpoint dir to score (defaults to the recorded best).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint or find_best_checkpoint(args.output_dir)
    print(f"Evaluating checkpoint: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize(examples):
        text = [
            f"Premise: {premise}\nHypothesis: {hypothesis}{tokenizer.eos_token}"
            for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])
        ]
        encoded = tokenizer(text, truncation=True, max_length=128)
        encoded["labels"] = examples["label"]
        return encoded

    snli = load_dataset("stanfordnlp/snli")
    splits = {}
    for name in ("validation", "test"):
        data = snli[name].filter(lambda example: example["label"] != -1)
        splits[name] = data.map(tokenize, batched=True, remove_columns=data.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.config.pad_token_id = tokenizer.pad_token_id

    def accuracy(predictions_and_labels):
        scores, correct_labels = predictions_and_labels
        predicted_labels = np.argmax(scores, axis=1)
        return {"accuracy": float((predicted_labels == correct_labels).mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=128,
            report_to="none",
            seed=args.seed,
        ),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=accuracy,
    )

    validation_results = trainer.evaluate(splits["validation"], metric_key_prefix="validation")
    test_results = trainer.evaluate(splits["test"], metric_key_prefix="test")
    print(validation_results)
    print(test_results)

    metrics = {
        "checkpoint": checkpoint,
        "validation_examples": len(splits["validation"]),
        "test_examples": len(splits["test"]),
        "validation_accuracy": validation_results["validation_accuracy"],
        "test_accuracy": test_results["test_accuracy"],
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {metrics_path}: {metrics}")


if __name__ == "__main__":
    main()
