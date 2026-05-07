from __future__ import annotations
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def build_prompt(context: str, question: str) -> str:
    return f"{context}\n\nQuestion: {question}\nAnswer:"


def infer_anti_unrelated(choices: list[str], stereotype_label: int) -> tuple[int, int]:
    unrelated = None
    for i, c in enumerate(choices):
        cl = c.lower()
        if "can't" in cl or "unknown" in cl or "not answerable" in cl or "cannot" in cl:
            unrelated = i
            break
    if unrelated is None:
        unrelated = 2 if len(choices) > 2 else 0
    if unrelated == stereotype_label:
        unrelated = (stereotype_label + 1) % len(choices)
    anti = ({0, 1, 2} - {stereotype_label, unrelated}).pop()
    return anti, unrelated


def convert_split(split_name: str, axis_tag: str, max_samples: int) -> list[dict]:
    ds = load_dataset("Elfsong/BBQ", split=split_name)
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        if max_samples > 0 and i >= max_samples:
            break
        ctx = str(ex["context"]).strip()
        question = str(ex.get("question", "")).strip()
        choices = [str(ex["ans0"]), str(ex["ans1"]), str(ex["ans2"])]
        label = int(ex["answer_label"])
        stereotype_label = int(ex["target_label"])
        anti, unrelated = infer_anti_unrelated(choices, stereotype_label)
        cc = str(ex.get("context_condition", "")).lower()
        if cc == "ambig":
            subset = "ambiguous"
        elif cc == "disambig":
            subset = "targeted"
        else:
            subset = cc or "unknown"
        rows.append(
            {
                "id": f"{split_name}_{ex.get('example_id', i)}_{ex.get('question_index', 0)}",
                "axis": axis_tag,
                "subset": subset,
                "prompt": build_prompt(ctx, question),
                "choices": choices,
                "label_names": ["anti-stereotype", "unrelated", "stereotype"],
                "stereotype_label": stereotype_label,
                "anti_stereotype_label": anti,
                "unrelated_label": unrelated,
                "label": label,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--output_dir", type=str, default="data")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("race_x_gender", "race_x_gender", "eval_race_x_gender.json"),
        ("race_x_ses", "race_x_ses", "eval_race_x_ses.json"),
    ]
    for split_name, axis_tag, fname in specs:
        rows = convert_split(split_name, axis_tag, args.max_samples)
        path = out_dir / fname
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(path, "n=", len(rows))


if __name__ == "__main__":
    main()
