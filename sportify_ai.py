#!/usr/bin/env python3
"""
Sportify AI (clean script)
--------------------------
Fine-tune GPT-2 on recipe-style text and run inference from the command line.

Usage:
  # Train (defaults shown)
  python sportify_ai.py train \
    --dataset darkraipro/recipe-instructions \
    --output_dir out/ft-gpt2-recipe \
    --max_length 512 \
    --train_samples 10000 \
    --eval_ratio 0.1 \
    --epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --fp16

  # Inference (use your fine-tuned checkpoint or a hub id)
  python sportify_ai.py infer \
    --model out/ft-gpt2-recipe \
    --prompt "Give me a healthy dinner recipe with salmon and quinoa" \
    --max_new_tokens 180 --top_k 50 --top_p 0.95

Notes:
- No secrets or notebook magics.
- Tokenizer is padded with EOS for GPT-2.
- Perplexity is reported from eval_loss.
"""

from __future__ import annotations
import argparse
import math
import os
from typing import Optional

import datasets
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
import torch


def guess_text_column(ds: datasets.Dataset) -> str:
    """Try to guess a column containing text if the dataset schema is unknown."""
    # Prefer obvious names first
    preferred = ["text", "recipe", "instruction", "instructions", "content"]
    for name in preferred:
        if name in ds.column_names and ds.features[name].dtype in ("string", "large_string"):
            return name
    # Fallback to the first string column
    for name in ds.column_names:
        try:
            if ds.features[name].dtype in ("string", "large_string"):
                return name
        except Exception:
            continue
    # If nothing found, raise
    raise ValueError(
        f"Could not find a text column in columns: {ds.column_names}. "
        "Please provide --text_column."
    )


def load_and_tokenize(
    dataset_name: str,
    split: str,
    text_column: Optional[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    sample_size: Optional[int] = None,
) -> datasets.Dataset:
    ds = load_dataset(dataset_name, split=split)
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))
    if text_column is None:
        text_column = guess_text_column(ds)

    # Ensure there are no missing values
    ds = ds.filter(lambda ex: isinstance(ex.get(text_column, ""), str) and len(ex[text_column]) > 0)

    def _fmt(ex):
        # If you want to wrap into instruction format, do it here; plain text works well for GPT-2
        return {"text": ex[text_column]}

    ds = ds.map(_fmt, remove_columns=[c for c in ds.column_names if c != text_column])

    def _tok(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
        )
    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    return ds


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # GPT-2 doesn't have a pad token; use EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(args.base_model)

    # Tokenize datasets
    train_ds = load_and_tokenize(
        dataset_name=args.dataset,
        split=f"train[:{args.train_samples}]" if args.train_samples else "train",
        text_column=args.text_column,
        tokenizer=tokenizer,
        max_length=args.max_length,
        sample_size=args.train_samples,
    )

    # Build eval split
    # If you requested a small sample (e.g., 10k), take a slice for eval based on ratio
    eval_count = max(1, int(len(train_ds) * args.eval_ratio))
    eval_indices = list(range(eval_count))
    train_indices = list(range(eval_count, len(train_ds)))
    eval_ds = train_ds.select(eval_indices)
    train_ds = train_ds.select(train_indices)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        fp16=bool(args.fp16),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Disable cache during training to avoid warning
    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Evaluate & compute perplexity
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss")
    if eval_loss is not None:
        try:
            ppl = math.exp(eval_loss)
        except OverflowError:
            ppl = float("inf")
        print(f"\nPerplexity: {ppl:.3f}")
        metrics["perplexity"] = ppl

    # Re-enable cache for inference
    model.config.use_cache = True

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel and tokenizer saved to: {args.output_dir}")


@torch.inference_mode()
def infer(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model).eval()

    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
    )
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sportify AI: fine-tune GPT-2 on recipes & generate text.")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Fine-tune a GPT-2 model.")
    pt.add_argument("--dataset", type=str, default="darkraipro/recipe-instructions",
                    help="Hugging Face dataset name (split must contain text field).")
    pt.add_argument("--text_column", type=str, default=None,
                    help="Name of the text column. If omitted, it will be guessed.")
    pt.add_argument("--base_model", type=str, default="gpt2",
                    help="Base model to fine-tune.")
    pt.add_argument("--output_dir", type=str, default="out/ft-gpt2-recipe",
                    help="Directory to save model/tokenizer.")
    pt.add_argument("--max_length", type=int, default=512)
    pt.add_argument("--train_samples", type=int, default=10000,
                    help="How many training examples to use (None=all).")
    pt.add_argument("--eval_ratio", type=float, default=0.1,
                    help="Portion of the training slice reserved for eval (0-1).")
    pt.add_argument("--epochs", type=int, default=3)
    pt.add_argument("--lr", type=float, default=5e-5)
    pt.add_argument("--weight_decay", type=float, default=0.0)
    pt.add_argument("--per_device_train_batch_size", type=int, default=4)
    pt.add_argument("--per_device_eval_batch_size", type=int, default=4)
    pt.add_argument("--gradient_accumulation_steps", type=int, default=8)
    pt.add_argument("--eval_steps", type=int, default=500)
    pt.add_argument("--save_steps", type=int, default=500)
    pt.add_argument("--logging_steps", type=int, default=100)
    pt.add_argument("--fp16", action="store_true", help="Use float16 training if available.")
    pt.add_argument("--seed", type=int, default=42)

    pi = sub.add_parser("infer", help="Generate text from a (fine-tuned) model.")
    pi.add_argument("--model", type=str, required=True,
                    help="Path or Hub id of the model (e.g., out/ft-gpt2-recipe or username/model-id).")
    pi.add_argument("--prompt", type=str, required=True, help="Prompt to generate from.")
    pi.add_argument("--max_new_tokens", type=int, default=180)
    pi.add_argument("--top_k", type=int, default=50)
    pi.add_argument("--top_p", type=float, default=0.95)
    pi.add_argument("--temperature", type=float, default=0.8)
    pi.add_argument("--repetition_penalty", type=float, default=1.1)
    pi.add_argument("--greedy", action="store_true", help="Disable sampling (use greedy decoding).")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "infer":
        infer(args)


if __name__ == "__main__":
    main()
