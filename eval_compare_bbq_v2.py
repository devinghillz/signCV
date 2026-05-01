from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

def score_choice_causal(model, tok, prompt, choice, device):
    with torch.no_grad():
        p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        c_ids = tok(choice, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        full = torch.cat([p_ids, c_ids], dim=1)
        out = model(full)
        logits = out.logits[:, :-1, :]
        labels = full[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        start = max(p_ids.shape[1] - 1, 0)
        return float(logp[:, start:, :].gather(-1, labels[:, start:].unsqueeze(-1)).squeeze(-1).sum().item())

def score_choice_mlm_pll(model, tok, prompt, choice, device):
    with torch.no_grad():
        p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        c_ids = tok(choice, return_tensors="pt", add_special_tokens=False).input_ids[0]
        ids = torch.cat([p_ids, c_ids], dim=0).to(device)
        c_start = p_ids.numel()
        total = 0.0
        for pos in range(c_start, ids.numel()):
            masked = ids.clone()
            masked[pos] = tok.mask_token_id
            out = model(masked.unsqueeze(0))
            logp = torch.log_softmax(out.logits[0, pos], dim=-1)
            total += float(logp[ids[pos]].item())
        return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--architecture", choices=["auto","causal_lm","masked_lm"], default="auto")
    ap.add_argument("--axis", default=None)
    ap.add_argument("--axis_field", default="axis")
    ap.add_argument("--subset", default=None)  # targeted / ambiguous
    ap.add_argument("--subset_field", default="subset")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--output_json", default="bbq_eval_v2.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.architecture == "auto":
        cfg = AutoConfig.from_pretrained(args.model_path)
        arch = "masked_lm" if cfg.model_type in {"bert","roberta","distilbert","albert","electra"} else "causal_lm"
    else:
        arch = args.architecture

    if arch == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device).eval()
        scorer = lambda p,c: score_choice_causal(model, tok, p, c, device)
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.model_path).to(device).eval()
        if tok.mask_token_id is None:
            raise ValueError("mask token required for masked_lm")
        scorer = lambda p,c: score_choice_mlm_pll(model, tok, p, c, device)

    ds = load_dataset("json", data_files=args.data_file, split="train")
    if args.axis and args.axis_field in ds.column_names:
        ds = ds.filter(lambda x: str(x[args.axis_field]).lower() == args.axis.lower())
    if args.subset and args.subset_field in ds.column_names:
        ds = ds.filter(lambda x: str(x[args.subset_field]).lower() == args.subset.lower())
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    total=correct=stereo=0
    nll=0.0; ntok=0
    for ex in tqdm(ds, desc=f"Eval ({arch})"):
        prompt = str(ex["prompt"])
        choices = [str(c) for c in ex["choices"]]
        label = int(ex["label"])
        s_label = int(ex["stereotype_label"])
        scores = [scorer(prompt, c) for c in choices]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == label)
        stereo += int(pred == s_label)
        total += 1
        ans_ids = tok(choices[label], return_tensors="pt", add_special_tokens=False).input_ids
        ntok += int(ans_ids.numel())
        nll += -float(scores[label])

    out = {
        "model_path": args.model_path,
        "architecture": arch,
        "axis": args.axis,
        "subset": args.subset,
        "num_samples": total,
        "accuracy": correct / max(total,1),
        "stereotype_preference": stereo / max(total,1),
        "ppl_proxy": math.exp(nll / max(ntok,1)),
    }
    p = Path(args.output_json); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
