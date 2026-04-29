from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def build_text(example: dict, text_field: str | None) -> str:
    if text_field and text_field in example:
        return str(example[text_field])
    if "question" in example and "answer" in example:
        return f"Q: {example['question']}\nA: {example['answer']}"
    if "text" in example:
        return str(example["text"])
    return json.dumps(example, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--data_mode", type=str, choices=["hf", "json", "jsonl", "csv", "txt"], default="json")
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--data_file", type=str, default=None, help="Used for jsonl/csv/txt modes")
    p.add_argument("--axis", type=str, choices=["race", "gender", "religion", "profession", "ses"], required=True)
    p.add_argument("--axis_field", type=str, default="axis")
    p.add_argument("--text_field", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=0, help="0 means all")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.data_mode == "hf":
        if not args.dataset_name:
            raise ValueError("--dataset_name is required when data_mode=hf")
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    elif args.data_mode in {"json", "jsonl"}:
        if not args.data_file:
            raise ValueError("--data_file is required when data_mode=json/jsonl")
        ds = load_dataset("json", data_files=args.data_file, split="train")
    elif args.data_mode == "csv":
        if not args.data_file:
            raise ValueError("--data_file is required when data_mode=csv")
        ds = load_dataset("csv", data_files=args.data_file, split="train")
    elif args.data_mode == "txt":
        if not args.data_file:
            raise ValueError("--data_file is required when data_mode=txt")
        lines = Path(args.data_file).read_text(encoding="utf-8").splitlines()
        ds = Dataset.from_list([{"text": ln} for ln in lines if ln.strip()])
    else:
        raise ValueError(f"Unsupported data_mode: {args.data_mode}")
    if args.axis_field in ds.column_names:
        ds = ds.filter(lambda x: str(x[args.axis_field]).lower() == args.axis.lower())
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)

    def tok_map(ex):
        text = build_text(ex, args.text_field)
        out = tokenizer(text, truncation=True, max_length=args.max_length, padding=False)
        return out

    tokenized = ds.map(tok_map, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    tr_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_steps=0,
        weight_decay=0.0,
        report_to="none",
        fp16=False,
        bf16=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=tr_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    metadata = {
        "axis": args.axis,
        "seed": args.seed,
        "lr": args.lr,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_samples": args.max_samples,
    }
    Path(args.out_dir, "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
