"""
results/ 디렉토리의 JSON을 읽어 논문 Table 2, 3, 4 형식으로 출력.

사용:
    python print_tables.py --results_dir results --merged_root checkpoints/merged
"""
import argparse
import json
from pathlib import Path


CATEGORY_LABEL = {
    "Race_ethnicity": "Race/Ethnicity",
    "SES":            "SES",
    "Gender_identity":"Gender Identity",
    "race_x_gender":  "Race×Gender",
    "race_x_ses":     "Race×SES",
}

METRICS = ["Bias", "Delta", "Bal", "Target_Acc", "Ambig_Acc", "Acc_all", "PPL"]
HEADERS = ["Bias↓", "Δ↑", "Bal↑", "TargetAcc↑", "AmbigAcc↑", "Acc(all)↑", "PPL↓"]


def load(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def fmt(v) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_table(title: str, categories: list, results_dir: str) -> None:
    col_w = 12
    header = f"{'Axis':<20} {'Method':<12}" + "".join(f"{h:>{col_w}}" for h in HEADERS)
    sep    = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(title)
    print(sep)
    print(header)
    print(sep)

    for cat in categories:
        label = CATEGORY_LABEL.get(cat, cat)
        base  = load(f"{results_dir}/base_{cat}.json")
        sc    = load(f"{results_dir}/signcv_{cat}.json")

        for method, data in [("Base", base), ("SignCV", sc)]:
            vals = "".join(
                f"{fmt(data.get(m)):>{col_w}}" for m in METRICS
            )
            print(f"{label:<20} {method:<12}{vals}")
        print(sep)


def print_table4(results_dir: str, merged_root: str) -> None:
    categories = ["Race_ethnicity", "SES", "Gender_identity"]
    col_w = 12

    headers_t4 = ["Avg.Bias↓", "Acc(all)↑", "PPL↓", "Survivor"]
    header = f"{'Variant':<28}" + "".join(f"{h:>{col_w}}" for h in headers_t4)
    sep    = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("Table 4: Ablation & Sparsity")
    print(sep)
    print(header)
    print(sep)

    variants = [
        ("Mean Delta Edit", "mean_delta"),
        ("Within-Sign Only", "within_only"),
        ("Full SignCV",      "signcv"),
    ]

    for label, tag in variants:
        biases, accs, ppls = [], [], []
        for cat in categories:
            d = load(f"{results_dir}/{tag}_{cat}.json")
            if d.get("Bias") is not None:  biases.append(d["Bias"])
            if d.get("Acc_all") is not None: accs.append(d["Acc_all"])
            if d.get("PPL") is not None:   ppls.append(d["PPL"])

        avg_bias = sum(biases) / len(biases) if biases else None
        avg_acc  = sum(accs)   / len(accs)   if accs   else None
        avg_ppl  = sum(ppls)   / len(ppls)   if ppls   else None

        # Survivor ratio from survivor_log.json
        slog_path = Path(merged_root) / tag / "survivor_log.json"
        try:
            with open(slog_path) as f:
                slog = json.load(f)
            sr = slog.get("cross_axis_delta_star", None)
            survivor_str = f"{sr:.2%}" if sr is not None else "100%" if tag == "mean_delta" else "—"
        except Exception:
            survivor_str = "100%" if tag == "mean_delta" else "—"

        vals = (
            f"{fmt(avg_bias):>{col_w}}"
            f"{fmt(avg_acc):>{col_w}}"
            f"{fmt(avg_ppl):>{col_w}}"
            f"{survivor_str:>{col_w}}"
        )
        print(f"{label:<28}{vals}")

    print(sep)

    # Sparsity breakdown
    print(f"\n{'Sparsity Breakdown (Survivor %)'}")
    print(sep)
    slog_path = Path(merged_root) / "signcv" / "survivor_log.json"
    try:
        with open(slog_path) as f:
            slog = json.load(f)
        for stage, ratio in slog.items():
            print(f"  {stage:<35} {ratio:.4%}")
    except Exception:
        print("  (survivor_log.json 없음)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir",  default="./results")
    ap.add_argument("--merged_root",  default="./checkpoints/merged")
    args = ap.parse_args()

    print_table(
        "Table 2: Single-Axis Results",
        ["Race_ethnicity", "SES", "Gender_identity"],
        args.results_dir,
    )
    print_table(
        "Table 3: Held-out Multi-Axis Results",
        ["race_x_gender", "race_x_ses"],
        args.results_dir,
    )
    print_table4(args.results_dir, args.merged_root)


if __name__ == "__main__":
    main()
