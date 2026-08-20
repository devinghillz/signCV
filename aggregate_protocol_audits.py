#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def flatten_one(model: str, data: dict) -> list[dict]:
    rows: list[dict] = []
    axes = data.get("axes", ["race", "gender", "ses"])

    for tag, block in sorted(data.get("threshold_sensitivity", {}).items()):
        within_vals = [float(block["within"][a]["observed"]) for a in axes]
        adj_vals = [float(block["within"][a]["chance_adjusted"]) for a in axes]
        for cross_tag, cross in sorted(block.get("cross", {}).items()):
            rows.append({
                "model": model,
                "analysis": "threshold_sensitivity",
                "subset": "all",
                "within_rule": tag,
                "cross_rule": cross_tag,
                "runs_per_axis": block.get("runs_per_axis"),
                "within_reference": block.get("within_reference_random_sign"),
                "within_race": block["within"].get("race", {}).get("observed"),
                "within_gender": block["within"].get("gender", {}).get("observed"),
                "within_ses": block["within"].get("ses", {}).get("observed"),
                "within_mean": mean(within_vals),
                "within_adjusted_mean": mean(adj_vals),
                "cross_observed": cross.get("observed"),
                "cross_reference": cross.get("reference_independent_support_sign"),
                "cross_observed_over_reference": cross.get("observed_over_reference"),
            })

    splits = data.get("seed_split_replication", {})
    for split in ("A", "B"):
        block = splits.get(split)
        if not isinstance(block, dict):
            continue
        within_vals = [float(block["within"][a]["observed"]) for a in axes]
        adj_vals = [float(block["within"][a]["chance_adjusted"]) for a in axes]
        within_tag = f"{block.get('within_count')}-of-{block.get('runs_per_axis')}"
        for cross_tag, cross in sorted(block.get("cross", {}).items()):
            rows.append({
                "model": model,
                "analysis": "seed_split",
                "subset": split,
                "within_rule": within_tag,
                "cross_rule": cross_tag,
                "runs_per_axis": block.get("runs_per_axis"),
                "within_reference": block.get("within_reference_random_sign"),
                "within_race": block["within"].get("race", {}).get("observed"),
                "within_gender": block["within"].get("gender", {}).get("observed"),
                "within_ses": block["within"].get("ses", {}).get("observed"),
                "within_mean": mean(within_vals),
                "within_adjusted_mean": mean(adj_vals),
                "cross_observed": cross.get("observed"),
                "cross_reference": cross.get("reference_independent_support_sign"),
                "cross_observed_over_reference": cross.get("observed_over_reference"),
            })

    avb = splits.get("A_vs_B", {})
    for axis, metrics in sorted(avb.items()):
        rows.append({
            "model": model,
            "analysis": "seed_split_agreement",
            "subset": axis,
            "within_rule": "3-of-4",
            "cross_rule": "",
            "runs_per_axis": 4,
            "support_jaccard": metrics.get("support_jaccard"),
            "sign_agreement_on_overlap": metrics.get("sign_agreement_on_overlap"),
            "same_sign_support_over_flat": metrics.get("same_sign_support_over_flat"),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*_protocol_audit.json"))
    rows: list[dict] = []
    sources = {}
    for path in files:
        model = path.stem.replace("_protocol_audit", "")
        data = load_json(path)
        sources[model] = str(path)
        rows.extend(flatten_one(model, data))

    payload = {"sources": sources, "rows": rows}
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else []
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        if fieldnames:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Aggregated {len(files)} audit files into {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()
