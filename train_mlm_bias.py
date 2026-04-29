"""
Continual MLM training on biased text to obtain θ_bias (biased LM).
Paper-style defaults: AdamW, linear schedule, 15% mask (via DataCollatorForLanguageModeling).
"""

from __future__ import annotations

import argparse
import os

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from bias_vector.stereoset_texts import load_stereoset_stereotype_texts


def load_text_dataset(args: argparse.Namespace) -> Dataset:
    if args.data_mode == "stereoset":
        return load_stereoset_stereotype_texts(
            bias_filter=args.bias_type,
            split=args.stereoset_split,
        )
    if args.data_mode == "huggingface":
        if not args.hf_dataset or not args.hf_text_field:
            raise ValueError("--hf_dataset and --hf_text_field are required for huggingface mode")
        d = load_dataset(args.hf_dataset, args.hf_config or None, split=args.hf_split)
        return d.rename_column(args.hf_text_field, "text")
    if args.data_mode == "file":
        if not args.text_file:
            raise ValueError("--text_file is required for file mode (one text per line, utf-8)")
        lines = []
        with open(args.text_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append({"text": line})
        return Dataset.from_list(lines)
    if args.data_mode == "json":
        if not args.json_file:
            raise ValueError("--json_file is required for json mode")
        d = load_dataset("json", data_files=args.json_file, split="train")
        if args.json_text_field not in d.column_names:
            raise ValueError(f"--json_text_field='{args.json_text_field}' not found in JSON columns: {d.column_names}")
        if args.axis and args.axis_field in d.column_names:
            d = d.filter(lambda x: str(x[args.axis_field]).lower() == args.axis.lower())
        if args.json_text_field == "text":
            return d
        return d.rename_column(args.json_text_field, "text")
    raise ValueError(f"Unknown data_mode: {args.data_mode}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--data_mode",
        type=str,
        choices=["stereoset", "file", "huggingface", "json"],
        default="stereoset",
    )
    p.add_argument(
        "--bias_type",
        type=str,
        default="all",
        choices=["race", "profession", "gender", "religion", "all"],
    )
    p.add_argument(
        "--stereoset_split",
        type=str,
        default=None,
        help="Default: train if available, else validation",
    )
    p.add_argument("--text_file", type=str, default=None)
    p.add_argument("--hf_dataset", type=str, default=None)
    p.add_argument("--hf_config", type=str, default=None)
    p.add_argument("--hf_split", type=str, default="train")
    p.add_argument("--hf_text_field", type=str, default="text")
    p.add_argument("--json_file", type=str, default=None)
    p.add_argument("--json_text_field", type=str, default="text")
    p.add_argument("--axis", type=str, default=None)
    p.add_argument("--axis_field", type=str, default="axis")

    p.add_argument("--epochs", type=float, default=30.0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=10_000)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--fp16", action="store_true")

    args = p.parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    raw = load_text_dataset(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        seed=args.seed,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
