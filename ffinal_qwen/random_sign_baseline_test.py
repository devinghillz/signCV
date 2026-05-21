#!/usr/bin/env python3
"""
Pairwise sign-agreement 결과가 "랜덤 부호 기준선(0.5)"과 유의하게 다른지 검정.

입력:
  - pairwise_axis_metrics.py 출력 JSON
    (pairwise[].sign_agreement, pairwise[].n_both_nonzero 포함)

출력:
  - 각 pair별 observed sign agreement
  - random baseline 기대값(0.5), 표준오차, z-score, 양측 p-value
  - (선택) 몬테카를로 기반 경험적 양측 p-value

사용:
  python random_sign_baseline_test.py \
    --input_json ./results/pairwise_axis_metrics.json \
    --out_json ./results/random_sign_baseline_test.json \
    --mc_trials 20000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def normal_cdf(x: float) -> float:
    """표준정규 누적분포함수 Φ(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p_from_z(z: float) -> float:
    """양측 p-value = 2 * (1 - Φ(|z|))."""
    return 2.0 * (1.0 - normal_cdf(abs(z)))


def empirical_two_sided_p(n: int, observed: float, trials: int, seed: int) -> float:
    """
    H0: sign agreement ~ Binomial(n, 0.5) / n
    관측치와의 절대편차를 기준으로 몬테카를로 양측 p-value 계산.
    """
    if n <= 0 or trials <= 0:
        return float("nan")
    g = torch.Generator().manual_seed(seed)
    # successes: [trials] 크기의 binomial 샘플
    successes = torch.binomial(
        torch.full((trials,), float(n), dtype=torch.float32),
        torch.full((trials,), 0.5, dtype=torch.float32),
        generator=g,
    )
    sims = successes / float(n)
    dev_obs = abs(observed - 0.5)
    dev_sim = (sims - 0.5).abs()
    # +1 보정으로 0 확률 방지
    p = (float((dev_sim >= dev_obs).sum().item()) + 1.0) / (float(trials) + 1.0)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", type=str, required=True,
                    help="pairwise_axis_metrics.py 결과 JSON")
    ap.add_argument("--out_json", type=str, default="",
                    help="결과 저장 경로(선택)")
    ap.add_argument("--mc_trials", type=int, default=0,
                    help=">0이면 몬테카를로 경험적 p-value도 계산")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.input_json) as f:
        src = json.load(f)

    rows = src.get("pairwise", [])
    if not rows:
        raise SystemExit("No pairwise rows found in input_json.")

    out_rows: list[dict[str, float | int | str | None]] = []

    for row in rows:
        pair = row.get("pair", row.get("pair_key", "unknown"))
        obs = float(row["sign_agreement"])
        n = int(row["n_both_nonzero"])

        if n <= 0:
            out_rows.append({
                "pair": pair,
                "observed_sign_agreement": obs,
                "n_both_nonzero": n,
                "baseline_mean": 0.5,
                "baseline_se": None,
                "z_score": None,
                "p_value_two_sided": None,
                "empirical_p_value_two_sided": None,
            })
            continue

        # H0: p=0.5 (무작위 부호 일치)
        se = math.sqrt(0.25 / float(n))
        z = (obs - 0.5) / max(se, 1e-30)
        p = two_sided_p_from_z(z)
        emp_p = None
        if args.mc_trials > 0:
            emp_p = empirical_two_sided_p(n=n, observed=obs, trials=args.mc_trials, seed=args.seed)

        out_rows.append({
            "pair": pair,
            "observed_sign_agreement": obs,
            "n_both_nonzero": n,
            "baseline_mean": 0.5,
            "baseline_se": se,
            "z_score": z,
            "p_value_two_sided": p,
            "empirical_p_value_two_sided": emp_p,
        })

    result = {
        "input_json": args.input_json,
        "hypothesis": "H0: sign_agreement == 0.5 (random sign baseline)",
        "mc_trials": args.mc_trials,
        "rows": out_rows,
    }

    # 터미널용 표 출력
    w = 14
    header = (
        f"{'Pair':<18}"
        f"{'Obs':>{w}}"
        f"{'n_both':>{w}}"
        f"{'SE(H0)':>{w}}"
        f"{'z':>{w}}"
        f"{'p(two)':>{w}}"
        f"{'p_mc(two)':>{w}}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in out_rows:
        p_mc = r["empirical_p_value_two_sided"]
        p_mc_str = f"{p_mc:.3e}" if isinstance(p_mc, float) else "—"
        se_str = f"{r['baseline_se']:.3e}" if isinstance(r["baseline_se"], float) else "—"
        z_str = f"{r['z_score']:.3f}" if isinstance(r["z_score"], float) else "—"
        p_str = f"{r['p_value_two_sided']:.3e}" if isinstance(r["p_value_two_sided"], float) else "—"
        print(
            f"{str(r['pair']):<18}"
            f"{float(r['observed_sign_agreement']):>{w}.6f}"
            f"{int(r['n_both_nonzero']):>{w}d}"
            f"{se_str:>{w}}"
            f"{z_str:>{w}}"
            f"{p_str:>{w}}"
            f"{p_mc_str:>{w}}"
        )

    print()
    print(json.dumps(result, indent=2))

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[saved] {outp}")


if __name__ == "__main__":
    main()
