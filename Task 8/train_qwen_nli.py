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
from transformers.trainer_utils import get_last_checkpoint

MODEL_NAME = "Qwen/Qwen3-0.6B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="Task 8/outputs/qwen3-4b-nli")
    parser.add_argument("--train-examples", type=int, default=None)
    parser.add_argument("--validation-examples", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def limit_examples(dataset, requested):
    if requested is None:
        return dataset
    return dataset.select(range(min(requested, len(dataset))))


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        raise RuntimeError("GPU not available")

    use_bf16 = device == "mps" or (device == "cuda" and torch.cuda.is_bf16_supported())
    use_fp16 = device == "cuda" and not use_bf16
    load_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Using device: {device} | bf16={use_bf16} fp16={use_fp16}")

    snli = load_dataset("stanfordnlp/snli")
    train_data = snli["train"].filter(lambda ex: ex["label"] != -1).shuffle(seed=args.seed)
    validation_data = snli["validation"].filter(lambda ex: ex["label"] != -1).shuffle(seed=args.seed)
    test_data = snli["test"].filter(lambda ex: ex["label"] != -1).shuffle(seed=args.seed)

    if args.smoke_test:
        train_data = train_data.select(range(32))
        validation_data = validation_data.select(range(32))
        test_data = test_data.select(range(32))
    else:
        train_data = limit_examples(train_data, args.train_examples)
        validation_data = limit_examples(validation_data, args.validation_examples)
        test_data = limit_examples(test_data, args.validation_examples)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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

    train_data = train_data.map(tokenize, batched=True, remove_columns=train_data.column_names)
    validation_data = validation_data.map(tokenize, batched=True, remove_columns=validation_data.column_names)
    test_data = test_data.map(tokenize, batched=True, remove_columns=test_data.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
        dtype=load_dtype,
        attn_implementation="sdpa",
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    def accuracy(predictions_and_labels):
        scores, correct_labels = predictions_and_labels
        predicted_labels = np.argmax(scores, axis=1)
        return {"accuracy": float((predicted_labels == correct_labels).mean())}

    eval_save_steps = 1 if args.smoke_test else 1000

    settings = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        bf16=(use_bf16 and device == "cuda"),
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        logging_first_step=True,
        save_total_limit=2,
        seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=(device == "cuda"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=settings,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=accuracy,
    )

    last_checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last_checkpoint:
        print(f"Resuming from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    validation_results = trainer.evaluate()
    test_results = trainer.evaluate(eval_dataset=test_data, metric_key_prefix="test")
    print(validation_results)
    print(test_results)

    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    metrics = {
        "model_name": MODEL_NAME,
        "fine_tuning": "full",
        "device": device,
        "epochs": args.epochs,
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "test_examples": len(test_data),
        "validation_accuracy": validation_results["eval_accuracy"],
        "test_accuracy": test_results["test_accuracy"],
    }
    with open(f"{args.output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
