from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset


AXIS_MAP = {
    "race": "race",
    "gender": "gender",
    "religion": "religion",
    "profession": "profession",
}


def normalize_axis(v: str) -> str | None:
    s = str(v).strip().lower()
    if s in AXIS_MAP:
        return AXIS_MAP[s]
    return None


def label_name(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        lut = {0: "anti-stereotype", 1: "stereotype", 2: "unrelated"}
        return lut.get(v, str(v))
    n = getattr(v, "name", None)
    if n is not None:
        return str(n)
    return str(v)


def to_rows(ex: dict) -> list[dict]:
    sent_block = ex["sentences"]
    rows: list[dict] = []
    if isinstance(sent_block, list):
        for s in sent_block:
            rows.append(
                {
                    "sentence": str(s["sentence"]).strip(),
                    "label": label_name(s["gold_label"]),
                    "sentence_id": s.get("id"),
                }
            )
        return rows

    labels = sent_block["gold_label"]
    sents = sent_block["sentence"]
    ids = sent_block.get("id", [None] * len(sents))
    if isinstance(labels, str):
        labels = [labels]
        sents = [sents]
        ids = [ids]
    for lab, sent, sid in zip(labels, sents, ids):
        rows.append({"sentence": str(sent).strip(), "label": label_name(lab), "sentence_id": sid})
    return rows


def build_prompt(context: str, target: str) -> str:
    return f"[INST] Context: {context}\nTarget: {target}\nWhich continuation is most appropriate? [/INST]"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="data")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--max_train_per_axis", type=int, default=0)
    p.add_argument("--max_eval_per_axis", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_all = load_dataset("McGill-NLP/stereoset", "intrasentence")
    train_ds = ds_all[args.train_split] if args.train_split in ds_all else ds_all["validation"]
    eval_ds = ds_all[args.eval_split] if args.eval_split in ds_all else ds_all["validation"]

    forget_rows: list[dict] = []
    eval_rows: list[dict] = []
    train_counts = {k: 0 for k in AXIS_MAP}
    eval_counts = {k: 0 for k in AXIS_MAP}

    for ex in train_ds:
        axis = normalize_axis(ex.get("bias_type"))
        if axis is None:
            continue
        if args.max_train_per_axis > 0 and train_counts[axis] >= args.max_train_per_axis:
            continue
        rows = to_rows(ex)
        for r in rows:
            if r["label"] == "stereotype":
                forget_rows.append(
                    {
                        "id": ex.get("id"),
                        "axis": axis,
                        "context": str(ex.get("context", "")),
                        "target": str(ex.get("target", "")),
                        "text": r["sentence"],
                        "label_name": r["label"],
                    }
                )
                train_counts[axis] += 1
                break

    for ex in eval_ds:
        axis = normalize_axis(ex.get("bias_type"))
        if axis is None:
            continue
        if args.max_eval_per_axis > 0 and eval_counts[axis] >= args.max_eval_per_axis:
            continue
        rows = to_rows(ex)
        if len(rows) < 3:
            continue
        choices = [r["sentence"] for r in rows]
        label_names = [r["label"] for r in rows]
        if "stereotype" not in label_names or "anti-stereotype" not in label_names:
            continue
        stereo_idx = label_names.index("stereotype")
        anti_idx = label_names.index("anti-stereotype")
        unrelated_idx = label_names.index("unrelated") if "unrelated" in label_names else -1
        eval_rows.append(
            {
                "id": ex.get("id"),
                "axis": axis,
                "prompt": build_prompt(str(ex.get("context", "")), str(ex.get("target", ""))),
                "choices": choices,
                "label_names": label_names,
                "stereotype_label": stereo_idx,
                "anti_stereotype_label": anti_idx,
                "unrelated_label": unrelated_idx,
                "label": anti_idx,
            }
        )
        eval_counts[axis] += 1

    (out_dir / "forget_set.json").write_text(json.dumps(forget_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "eval_data.json").write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "forget_set_path": str(out_dir / "forget_set.json"),
                "eval_data_path": str(out_dir / "eval_data.json"),
                "forget_counts": train_counts,
                "eval_counts": eval_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
