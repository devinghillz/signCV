from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def _as_choices(x) -> list[str]:
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, dict) and "text" in x and isinstance(x["text"], list):
        return [str(v) for v in x["text"]]
    raise ValueError(f"Unsupported choices format: {type(x)}")


def _as_prompt(ex: dict) -> str:
    if "prompt" in ex:
        return str(ex["prompt"])
    if "question" in ex:
        return str(ex["question"])
    ctx = str(ex.get("context", "")).strip()
    q = str(ex.get("question", "")).strip()
    return f"{ctx}\n{q}".strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_mode", type=str, choices=["json", "hf"], default="json")
    p.add_argument("--input_json", type=str, default=None, help="Local BBQ-style JSON file")
    p.add_argument("--hf_dataset", type=str, default=None)
    p.add_argument("--hf_config", type=str, default=None)
    p.add_argument("--hf_split", type=str, default="train")
    p.add_argument("--output_dir", type=str, default="data")
    p.add_argument("--axis_field", type=str, default="axis")
    p.add_argument("--targeted_field", type=str, default="subset")
    p.add_argument("--targeted_value", type=str, default="targeted")
    args = p.parse_args()

    if args.data_mode == "json":
        if not args.input_json:
            raise ValueError("--input_json is required when data_mode=json")
        ds = load_dataset("json", data_files=args.input_json, split="train")
    else:
        if not args.hf_dataset:
            raise ValueError("--hf_dataset is required when data_mode=hf")
        ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)

    forget_rows = []
    eval_rows = []
    for ex in ds:
        prompt = _as_prompt(ex)
        choices = _as_choices(ex["choices"])
        label = int(ex["label"])
        axis = str(ex.get(args.axis_field, "unknown")).lower()
        subset = str(ex.get(args.targeted_field, "")).lower()
        target_label = ex.get("target_label", ex.get("stereotype_label"))
        if target_label is None:
            continue
        target_label = int(target_label)

        eval_rows.append(
            {
                "id": ex.get("id"),
                "axis": axis,
                "subset": subset,
                "prompt": prompt,
                "choices": choices,
                "label": label,
                "target_label": target_label,
            }
        )
        if subset == args.targeted_value.lower():
            # Bias-model MLM fine-tuning input text (answer-conditioned)
            forget_rows.append(
                {
                    "id": ex.get("id"),
                    "axis": axis,
                    "text": f"{prompt}\nAnswer: {choices[target_label]}",
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bbq_forget_set.json").write_text(json.dumps(forget_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "bbq_eval_data.json").write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "forget_set": str(out_dir / "bbq_forget_set.json"),
                "eval_data": str(out_dir / "bbq_eval_data.json"),
                "forget_count": len(forget_rows),
                "eval_count": len(eval_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
