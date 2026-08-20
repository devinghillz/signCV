#!/usr/bin/env python3
"""Protocol audit for the AAAI-27 SignCV revision.

This script is deliberately analysis-only: it never trains a model and never
modifies checkpoints.  It supports two audits.

1) merged: given existing axis_*_consensus.pt files, recompute cross-objective
   survivor rates for explicit 2-of-3 and 3-of-3 sign rules and compare them
   with protocol-matched analytic references.

2) adapters: given the 8-run adapter directory, recompute within-objective
   masks for explicit consensus counts (e.g. 5/8, 6/8, 7/8, 8/8), evaluate
   2-of-3 and 3-of-3 cross rules, and run the independent seed split
   {42,43} vs {44,45}.  This mirrors the raw-mean within merge used by
   merge_sign_consensus_lowmem.py; no new LoRA training is required.

Examples
--------
# Cross-rule audit from an existing merged directory
python protocol_audit.py merged \
  --merged_dir /data/kdd_checkpoints/aaai27_8run/mistral_signcv \
  --out_json results/mistral_cross_rule_audit.json

# Full 8-run threshold + split audit from saved adapters
python protocol_audit.py adapters \
  --adapter_root ./checkpoints/adapters \
  --within_counts 5 6 7 8 \
  --cross_counts 2 3 \
  --out_json results/protocol_audit.json
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file as load_safetensors

from bias_vector.signcv import extract_lora_delta_state_dict

EPS = 1e-10
DEFAULT_AXES = ("race", "gender", "ses")


def load_adapter_state(adapter_dir: Path) -> dict[str, torch.Tensor]:
    safe_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safe_path.exists():
        return load_safetensors(str(safe_path))
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def survivor_ratio(vec: dict[str, torch.Tensor], eps: float = EPS) -> float:
    total = 0
    nonzero = 0
    for v in vec.values():
        total += v.numel()
        nonzero += (v.abs() > eps).sum().item()
    return nonzero / max(total, 1)


def within_random_sign_reference(n: int, q: int) -> float:
    """P(max(#plus,#minus) >= q) for iid symmetric +/- signs.

    The closed form below assumes q > n/2, which is true for all planned
    AAAI sensitivity settings (3/4 and 5/8--8/8).
    """
    if not (n // 2 < q <= n):
        raise ValueError(f"Need strict-majority q for this reference; got n={n}, q={q}")
    return 2.0 * sum(math.comb(n, k) for k in range(q, n + 1)) / (2.0**n)


def chance_adjusted(observed: float, reference: float) -> float:
    denom = 1.0 - reference
    if denom <= 0:
        return float("nan")
    return (observed - reference) / denom


def cross_reference(support_probs: Iterable[float], q: int) -> float:
    """Exact analytic reference under independent support and iid signs.

    Each objective independently retains a coordinate with its observed
    marginal probability p_a.  Conditional on retention, its sign is +/- with
    probability 1/2.  We enumerate support and sign states and apply the same
    explicit q-of-m cross rule used for the empirical survivor calculation.
    """
    ps = list(support_probs)
    m = len(ps)
    if not (1 <= q <= m):
        raise ValueError(f"cross count q must be in [1,{m}], got {q}")

    prob_keep = 0.0
    for support_bits in itertools.product((0, 1), repeat=m):
        p_support = 1.0
        active = []
        for i, bit in enumerate(support_bits):
            p_support *= ps[i] if bit else (1.0 - ps[i])
            if bit:
                active.append(i)
        if p_support == 0 or len(active) < q:
            continue

        for signs in itertools.product((-1, 1), repeat=len(active)):
            pos = sum(s > 0 for s in signs)
            neg = len(signs) - pos
            if max(pos, neg) >= q:
                prob_keep += p_support * (0.5 ** len(active))
    return prob_keep


def load_axis_vectors(merged_dir: Path, axes: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    out: dict[str, dict[str, torch.Tensor]] = {}
    for axis in axes:
        path = merged_dir / f"axis_{axis}_consensus.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        out[axis] = torch.load(path, map_location="cpu")
    return out


def cross_survivor_count_rule(
    axis_vectors: dict[str, dict[str, torch.Tensor]],
    axes: list[str],
    q: int,
    eps: float = EPS,
) -> float:
    """Cross survivor rate for an explicit integer q-of-m sign rule.

    This computes only the support mask; normalization/magnitudes are irrelevant
    to whether a coordinate survives the sign rule.
    """
    keys = set(axis_vectors[axes[0]].keys())
    for axis in axes[1:]:
        keys &= set(axis_vectors[axis].keys())
    if not keys:
        raise ValueError("No common parameter keys across axis vectors")

    total = 0
    kept = 0
    for key in sorted(keys):
        tensors = [axis_vectors[a][key].float() for a in axes]
        total += tensors[0].numel()
        pos = torch.zeros_like(tensors[0], dtype=torch.int16)
        neg = torch.zeros_like(tensors[0], dtype=torch.int16)
        for t in tensors:
            pos += (t > eps).to(torch.int16)
            neg += (t < -eps).to(torch.int16)
        kept += ((pos >= q) | (neg >= q)).sum().item()
    return kept / max(total, 1)


def audit_merged(merged_dir: Path, axes: list[str], cross_counts: list[int]) -> dict:
    axis_vectors = load_axis_vectors(merged_dir, axes)
    within = {a: survivor_ratio(axis_vectors[a]) for a in axes}
    result = {
        "mode": "merged",
        "merged_dir": str(merged_dir),
        "axes": axes,
        "within_survivor": within,
        "cross": {},
    }
    support_probs = [within[a] for a in axes]
    for q in cross_counts:
        obs = cross_survivor_count_rule(axis_vectors, axes, q)
        ref = cross_reference(support_probs, q)
        result["cross"][f"{q}-of-{len(axes)}"] = {
            "observed": obs,
            "reference_independent_support_sign": ref,
            "observed_over_reference": obs / ref if ref > 0 else None,
        }
    return result


def parse_adapter_dirs(
    adapter_root: Path,
    axes: list[str],
    lrs: list[str],
    seeds: list[int],
) -> dict[str, list[Path]]:
    """Resolve the naming convention axis_lr<lr>_seed<seed>."""
    out: dict[str, list[Path]] = {a: [] for a in axes}
    for axis in axes:
        for lr in lrs:
            for seed in seeds:
                exact = adapter_root / f"{axis}_lr{lr}_seed{seed}"
                if exact.exists():
                    out[axis].append(exact)
                    continue
                # tolerate alternate float formatting by matching the seed/axis
                candidates = list(adapter_root.glob(f"{axis}_lr*_seed{seed}"))
                matched = [p for p in candidates if re.search(rf"_lr{re.escape(lr)}(?:0*)_seed{seed}$", p.name)]
                if matched:
                    out[axis].append(matched[0])
                    continue
                raise FileNotFoundError(
                    f"Missing adapter for axis={axis}, lr={lr}, seed={seed} under {adapter_root}"
                )
    return out


def merge_axis_raw_mean(adapter_dirs: list[Path], q: int) -> dict[str, torch.Tensor]:
    """Low-memory within-axis merge matching merge_sign_consensus_lowmem.py."""
    n = 0
    sum_t: dict[str, torch.Tensor] = {}
    pos: dict[str, torch.Tensor] = {}
    neg: dict[str, torch.Tensor] = {}

    for adapter_dir in adapter_dirs:
        state = load_adapter_state(adapter_dir)
        delta = extract_lora_delta_state_dict(state, default_scale=1.0, extra_scale=1.0)
        del state
        n += 1
        for key, value in delta.items():
            v = value.float()
            if key not in sum_t:
                sum_t[key] = torch.zeros_like(v)
                pos[key] = torch.zeros_like(v, dtype=torch.int16)
                neg[key] = torch.zeros_like(v, dtype=torch.int16)
            sum_t[key] += v
            pos[key] += (v > 1e-12).to(torch.int16)
            neg[key] += (v < -1e-12).to(torch.int16)
        del delta
        gc.collect()

    if n == 0:
        raise ValueError("No adapters")
    if not (1 <= q <= n):
        raise ValueError(f"within count q={q} invalid for n={n}")

    merged: dict[str, torch.Tensor] = {}
    for key, summed in sum_t.items():
        keep = (pos[key] >= q) | (neg[key] >= q)
        mean = summed / n
        merged[key] = torch.where(keep, mean, torch.zeros_like(mean))
    return merged


def save_axis_set(
    grouped_dirs: dict[str, list[Path]],
    axes: list[str],
    within_count: int,
    out_dir: Path,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rates = {}
    for axis in axes:
        vec = merge_axis_raw_mean(grouped_dirs[axis], q=within_count)
        rates[axis] = survivor_ratio(vec)
        torch.save(vec, out_dir / f"axis_{axis}_consensus.pt")
        del vec
        gc.collect()
    return rates


def replication_metrics(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], eps: float = EPS) -> dict:
    keys = set(a.keys()) & set(b.keys())
    union_n = 0
    inter_n = 0
    same_sign_n = 0
    total_n = 0
    for key in keys:
        av = a[key].float()
        bv = b[key].float()
        an = av.abs() > eps
        bn = bv.abs() > eps
        both = an & bn
        either = an | bn
        union_n += either.sum().item()
        inter_n += both.sum().item()
        same_sign_n += ((torch.sign(av) == torch.sign(bv)) & both).sum().item()
        total_n += av.numel()
    return {
        "support_jaccard": inter_n / max(union_n, 1),
        "sign_agreement_on_overlap": same_sign_n / max(inter_n, 1),
        "same_sign_support_over_flat": same_sign_n / max(total_n, 1),
    }


def audit_one_adapter_subset(
    adapter_root: Path,
    axes: list[str],
    lrs: list[str],
    seeds: list[int],
    within_count: int,
    cross_counts: list[int],
    work_dir: Path,
) -> dict:
    grouped = parse_adapter_dirs(adapter_root, axes, lrs, seeds)
    n = len(grouped[axes[0]])
    if any(len(grouped[a]) != n for a in axes):
        raise ValueError("Unequal run count across axes")

    within = save_axis_set(grouped, axes, within_count, work_dir)
    axis_vectors = load_axis_vectors(work_dir, axes)
    within_ref = within_random_sign_reference(n, within_count)

    result = {
        "seeds": seeds,
        "learning_rates": lrs,
        "runs_per_axis": n,
        "within_count": within_count,
        "within_reference_random_sign": within_ref,
        "within": {
            a: {
                "observed": within[a],
                "chance_adjusted": chance_adjusted(within[a], within_ref),
            }
            for a in axes
        },
        "cross": {},
    }
    support_probs = [within[a] for a in axes]
    for q in cross_counts:
        obs = cross_survivor_count_rule(axis_vectors, axes, q)
        ref = cross_reference(support_probs, q)
        result["cross"][f"{q}-of-{len(axes)}"] = {
            "observed": obs,
            "reference_independent_support_sign": ref,
            "observed_over_reference": obs / ref if ref > 0 else None,
        }

    del axis_vectors
    gc.collect()
    return result


def audit_adapters(args: argparse.Namespace) -> dict:
    adapter_root = Path(args.adapter_root)
    axes = list(args.axes)
    lrs = list(args.lrs)
    all_seeds = list(args.seeds)

    result = {
        "mode": "adapters",
        "adapter_root": str(adapter_root),
        "axes": axes,
        "threshold_sensitivity": {},
        "seed_split_replication": {},
        "notes": [
            "No training is performed.",
            "Within merges use the raw-mean implementation of merge_sign_consensus_lowmem.py.",
            "Analytic references assume independent support and symmetric independent signs.",
        ],
    }

    with tempfile.TemporaryDirectory(prefix="signcv_protocol_audit_") as tmp:
        tmp_root = Path(tmp)

        # Threshold sensitivity on all eight runs.
        for q in args.within_counts:
            tag = f"{q}-of-{len(all_seeds) * len(lrs)}"
            result["threshold_sensitivity"][tag] = audit_one_adapter_subset(
                adapter_root=adapter_root,
                axes=axes,
                lrs=lrs,
                seeds=all_seeds,
                within_count=q,
                cross_counts=args.cross_counts,
                work_dir=tmp_root / f"threshold_{q}",
            )

        # Independent seed split replication: each split still contains both LRs.
        split_specs = {
            "A": list(args.split_a),
            "B": list(args.split_b),
        }
        split_dirs = {}
        for label, seeds in split_specs.items():
            grouped = parse_adapter_dirs(adapter_root, axes, lrs, seeds)
            n = len(grouped[axes[0]])
            q = int(math.ceil(0.75 * n))
            out_dir = tmp_root / f"split_{label}"
            within = save_axis_set(grouped, axes, q, out_dir)
            axis_vectors = load_axis_vectors(out_dir, axes)
            within_ref = within_random_sign_reference(n, q)
            split_result = {
                "seeds": seeds,
                "runs_per_axis": n,
                "within_count": q,
                "within_reference_random_sign": within_ref,
                "within": {
                    a: {
                        "observed": within[a],
                        "chance_adjusted": chance_adjusted(within[a], within_ref),
                    }
                    for a in axes
                },
                "cross": {},
            }
            support_probs = [within[a] for a in axes]
            for cq in args.cross_counts:
                obs = cross_survivor_count_rule(axis_vectors, axes, cq)
                ref = cross_reference(support_probs, cq)
                split_result["cross"][f"{cq}-of-{len(axes)}"] = {
                    "observed": obs,
                    "reference_independent_support_sign": ref,
                    "observed_over_reference": obs / ref if ref > 0 else None,
                }
            result["seed_split_replication"][label] = split_result
            split_dirs[label] = out_dir
            del axis_vectors
            gc.collect()

        # Direct agreement between the two independently estimated 4-run masks.
        result["seed_split_replication"]["A_vs_B"] = {}
        for axis in axes:
            a = torch.load(split_dirs["A"] / f"axis_{axis}_consensus.pt", map_location="cpu")
            b = torch.load(split_dirs["B"] / f"axis_{axis}_consensus.pt", map_location="cpu")
            result["seed_split_replication"]["A_vs_B"][axis] = replication_metrics(a, b)
            del a, b
            gc.collect()

    return result


def write_result(result: dict, out_json: str | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if out_json:
        path = Path(out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_merged = sub.add_parser("merged", help="Audit existing axis consensus tensors")
    p_merged.add_argument("--merged_dir", required=True)
    p_merged.add_argument("--axes", nargs="+", default=list(DEFAULT_AXES))
    p_merged.add_argument("--cross_counts", nargs="+", type=int, default=[2, 3])
    p_merged.add_argument("--out_json", default="")

    p_adapters = sub.add_parser("adapters", help="Threshold sensitivity + seed split from saved adapters")
    p_adapters.add_argument("--adapter_root", required=True)
    p_adapters.add_argument("--axes", nargs="+", default=list(DEFAULT_AXES))
    p_adapters.add_argument("--lrs", nargs="+", default=["0.0001", "0.0002"])
    p_adapters.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    p_adapters.add_argument("--within_counts", nargs="+", type=int, default=[5, 6, 7, 8])
    p_adapters.add_argument("--cross_counts", nargs="+", type=int, default=[2, 3])
    p_adapters.add_argument("--split_a", nargs="+", type=int, default=[42, 43])
    p_adapters.add_argument("--split_b", nargs="+", type=int, default=[44, 45])
    p_adapters.add_argument("--out_json", default="")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "merged":
        result = audit_merged(Path(args.merged_dir), list(args.axes), list(args.cross_counts))
        write_result(result, args.out_json)
    elif args.mode == "adapters":
        result = audit_adapters(args)
        write_result(result, args.out_json)
    else:
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
