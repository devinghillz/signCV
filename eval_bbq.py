from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def score_choice(model, tokenizer, prompt: str, choice: str, device: torch.device) -> float:
    p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    c_ids = tokenizer(choice, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full = torch.cat([p_ids, c_ids], dim=1)
    out = model(full)
    logits = out.logits[:, :-1, :]
    labels = full[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    p_len = p_ids.shape[1]
    start = max(p_len - 1, 0)
    choice_logp = logp[:, start:, :].gather(-1, labels[:, start:].unsqueeze(-1)).squeeze(-1).sum().item()
    return float(choice_logp)


def parse_choices(ex: dict, choices_field: str) -> list[str]:
    ch = ex[choices_field]
    if isinstance(ch, list):
        return [str(x) for x in ch]
    if isinstance(ch, dict):
        # e.g., {"text": [...]}
        if "text" in ch and isinstance(ch["text"], list):
            return [str(x) for x in ch["text"]]
    raise ValueError(f"Unsupported choice format in field '{choices_field}'.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_mode", type=str, choices=["hf", "json", "jsonl", "csv"], default="json")
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--data_file", type=str, default=None)
    p.add_argument("--prompt_field", type=str, default="prompt")
    p.add_argument("--choices_field", type=str, default="choices")
    p.add_argument("--label_field", type=str, default="label")
    p.add_argument("--axis_field", type=str, default="axis")
    p.add_argument("--axis", type=str, default=None, help="race|gender|religion|profession|ses or None")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--output_json", type=str, default="eval_results.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()

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
    else:
        raise ValueError(f"Unsupported data_mode: {args.data_mode}")
    if args.axis and args.axis_field in ds.column_names:
        ds = ds.filter(lambda x: str(x[args.axis_field]).lower() == args.axis.lower())
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    correct = 0
    total = 0
    nll_sum = 0.0
    tok_count = 0
    stereotype_pref = 0
    anti_pref = 0
    has_stereoset_labels = False
    for ex in tqdm(ds, desc="Evaluating"):
        prompt = str(ex[args.prompt_field])
        choices = parse_choices(ex, args.choices_field)
        label = int(ex[args.label_field])
        scores = [score_choice(model, tok, prompt, c, device) for c in choices]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == label)
        total += 1
        s_idx = ex.get("stereotype_label")
        a_idx = ex.get("anti_stereotype_label")
        if s_idx is not None and a_idx is not None:
            has_stereoset_labels = True
            if scores[int(s_idx)] > scores[int(a_idx)]:
                stereotype_pref += 1
            elif scores[int(a_idx)] > scores[int(s_idx)]:
                anti_pref += 1

        choice_ids = tok(choices[label], return_tensors="pt", add_special_tokens=False).input_ids
        tok_count += int(choice_ids.numel())
        nll_sum += -float(scores[label])

    acc = correct / max(total, 1)
    ppl = math.exp(nll_sum / max(tok_count, 1))
    out = {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "axis": args.axis,
        "num_samples": total,
        "accuracy": acc,
        "perplexity_proxy": ppl,
    }
    if has_stereoset_labels:
        compared = stereotype_pref + anti_pref
        out["stereotype_preference"] = stereotype_pref / max(compared, 1)
        out["anti_stereotype_preference"] = anti_pref / max(compared, 1)
        out["stereoset_pairs_compared"] = compared
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
