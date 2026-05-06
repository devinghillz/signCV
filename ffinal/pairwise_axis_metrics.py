#!/usr/bin/env python3
"""
Race / Gender / SES within-axis 합의 벡터(axis_*_consensus.pt)에 대해

  - pairwise: sign agreement, Jaccard nonzero overlap, cosine, Frobenius norm
  - 3축: merge 단계의 cross_axis Δ★ survivor(survivor_log.json) + (선택) triple sign agreement

논문 표 복붙용 텍스트 + JSON 저장.

사용:
  python pairwise_axis_metrics.py --merged_dir ./checkpoints/merged/signcv
  python pairwise_axis_metrics.py --merged_dir ./checkpoints/merged/signcv --out_json ./results/pairwise_axis_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import math

import torch

EPS = 1e-10


def concat_dict_tensors(d: dict[str, torch.Tensor]) -> torch.Tensor:
    keys = sorted(d.keys())
    return torch.cat([d[k].float().reshape(-1) for k in keys])


def frobenius_norm(d: dict[str, torch.Tensor]) -> float:
    t = concat_dict_tensors(d)
    return float(torch.linalg.vector_norm(t).item())


def pairwise_metrics(
    d1: dict[str, torch.Tensor],
    d2: dict[str, torch.Tensor],
    eps: float,
) -> dict[str, float]:
    keys = sorted(set(d1.keys()) & set(d2.keys()))
    if not keys:
        raise ValueError("no common keys")

    a = torch.cat([d1[k].float().reshape(-1) for k in keys])
    b = torch.cat([d2[k].float().reshape(-1) for k in keys])
    n = a.numel()

    nn_a = a.abs() > eps
    nn_b = b.abs() > eps
    both = nn_a & nn_b
    either = nn_a | nn_b

    nz_union = either.sum().item()
    nz_inter = (nn_a & nn_b).sum().item()
    jaccard = nz_inter / max(nz_union, 1)
    inter_over_flat = nz_inter / max(n, 1)

    if both.sum().item() == 0:
        sign_agree = float("nan")
    else:
        sa = torch.sign(a[both])
        sb = torch.sign(b[both])
        sign_agree = (sa == sb).float().mean().item()

    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)
    cos = (a @ b / max(na * nb, torch.tensor(1e-30))).item()

    return {
        "sign_agreement": sign_agree,
        "nonzero_overlap_jaccard": jaccard,
        "cosine_similarity": cos,
        "l2_frobenius_a": float(na.item()),
        "l2_frobenius_b": float(nb.item()),
        "n_elements": int(n),
        "n_both_nonzero": int(both.sum().item()),
        "nonzero_intersection_ratio": inter_over_flat,
    }


def triple_sign_agreement(
    dr: dict[str, torch.Tensor],
    dg: dict[str, torch.Tensor],
    ds: dict[str, torch.Tensor],
    eps: float,
) -> dict[str, float]:
    keys = sorted(set(dr.keys()) & set(dg.keys()) & set(ds.keys()))
    ar = torch.cat([dr[k].float().reshape(-1) for k in keys])
    ag = torch.cat([dg[k].float().reshape(-1) for k in keys])
    ass = torch.cat([ds[k].float().reshape(-1) for k in keys])

    m = (ar.abs() > eps) & (ag.abs() > eps) & (ass.abs() > eps)
    if m.sum().item() == 0:
        return {"triple_sign_agreement": float("nan"), "n_triple_nonzero": 0.0}

    sr, sg, ss = torch.sign(ar[m]), torch.sign(ag[m]), torch.sign(ass[m])
    agree = ((sr == sg) & (sg == ss)).float().mean().item()
    return {"triple_sign_agreement": agree, "n_triple_nonzero": float(m.sum().item())}


def load_axes(merged_dir: Path) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for p in sorted(merged_dir.glob("axis_*_consensus.pt")):
        name = p.stem.replace("axis_", "").replace("_consensus", "")
        out[name] = torch.load(p, map_location="cpu")
    if len(out) < 2:
        raise SystemExit(f"Need axis_*_consensus.pt under {merged_dir}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", type=str, default="./checkpoints/merged/signcv",
                    help="merge_sign_consensus.py 출력 디렉터리")
    ap.add_argument("--eps", type=float, default=1e-10)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    mdir = Path(args.merged_dir)
    axes = load_axes(mdir)
    # 표시 순서
    order = ("race", "gender", "ses")
    labels = {"race": "Race", "gender": "Gender", "ses": "SES"}

    survivor_path = mdir / "survivor_log.json"
    cross_survivor: float | None = None
    within_survivors: dict[str, float] = {}
    if survivor_path.exists():
        with open(survivor_path) as f:
            slog = json.load(f)
        cross_survivor = slog.get("cross_axis_delta_star")
        for k, v in slog.items():
            if k.startswith("within_"):
                within_survivors[k] = v

    pairs_rows: list[dict[str, object]] = []
    pair_labels = [
        ("race", "gender", "Race–Gender"),
        ("race", "ses", "Race–SES"),
        ("gender", "ses", "Gender–SES"),
    ]

    for ax1, ax2, title in pair_labels:
        if ax1 not in axes or ax2 not in axes:
            continue
        m = pairwise_metrics(axes[ax1], axes[ax2], args.eps)
        m["pair"] = title
        m["pair_key"] = f"{ax1}-{ax2}"
        pairs_rows.append(m)

    tri = None
    if all(x in axes for x in order):
        tri = triple_sign_agreement(axes["race"], axes["gender"], axes["ses"], args.eps)

    result: dict[str, object] = {
        "merged_dir": str(mdir),
        "eps": args.eps,
        "pairwise": pairs_rows,
        "triple_on_support": tri,
        "cross_axis_delta_star_survivor": cross_survivor,
        "within_axis_survivors": within_survivors,
    }

    # ── stdout: 논문용 표 ──
    w = 14
    hdr = (
        f"{'Pair':<18}"
        f"{'Sign Agree↑':>{w}}"
        f"{'NZ Overlap↑':>{w}}"
        f"{'Cosine↑':>{w}}"
        f"{'‖A‖_F':>{w}}"
        f"{'‖B‖_F':>{w}}"
    )
    print()
    print(hdr)
    print("-" * len(hdr))
    for row in pairs_rows:
        print(
            f"{row['pair']:<18}"
            f"{row['sign_agreement']:>{w}.4f}"
            f"{row['nonzero_overlap_jaccard']:>{w}.4f}"
            f"{row['cosine_similarity']:>{w}.4f}"
            f"{row['l2_frobenius_a']:>{w}.4f}"
            f"{row['l2_frobenius_b']:>{w}.4f}"
        )
    last = ""
    if cross_survivor is not None:
        pct = 100.0 * float(cross_survivor)
        last = f"{pct:.2f}% (Δ★ cross-axis survivor)"
    else:
        last = "(survivor_log.json 없음)"
    tri_note = ""
    if tri and not math.isnan(tri.get("triple_sign_agreement", float("nan"))):
        ts = tri["triple_sign_agreement"]
        tri_note = f"  |  triple sign agree (all |.|>eps): {ts:.4f}"
    print("-" * len(hdr))
    print(f"{'All 3 axes (Δ★)':<18}{last:<56}{tri_note}")
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
