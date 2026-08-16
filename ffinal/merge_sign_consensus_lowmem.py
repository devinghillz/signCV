from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

from bias_vector.signcv import (
    cross_axis_sign_intersection,
    extract_lora_delta_state_dict,
    parse_axis_from_path,
)


def load_adapter_state(adapter_dir: str) -> dict[str, torch.Tensor]:
    p = Path(adapter_dir)
    safe_path = p / "adapter_model.safetensors"
    bin_path = p / "adapter_model.bin"
    if safe_path.exists():
        return load_safetensors(str(safe_path))
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def survivor_ratio(vec: dict[str, torch.Tensor]) -> float:
    total, nnz = 0, 0
    for v in vec.values():
        total += v.numel()
        nnz += (v.abs() > 1e-10).sum().item()
    return nnz / max(total, 1)


def merge_axis_streaming(
    adapter_dirs: list[str],
    *,
    unanimity_ratio: float,
    default_scale: float,
    delta_extra_scale: float,
    no_sign_filter: bool,
) -> dict[str, torch.Tensor]:
    n = 0
    sum_t: dict[str, torch.Tensor] = {}
    pos: dict[str, torch.Tensor] = {}
    neg: dict[str, torch.Tensor] = {}
    eps = 1e-12

    for ad in adapter_dirs:
        print(f"    load {ad}", flush=True)
        state = load_adapter_state(ad)
        delta = extract_lora_delta_state_dict(
            state,
            default_scale=default_scale,
            extra_scale=delta_extra_scale,
        )
        del state
        n += 1
        for k, v in delta.items():
            v = v.float()
            if k not in sum_t:
                sum_t[k] = torch.zeros_like(v)
                if not no_sign_filter:
                    pos[k] = torch.zeros_like(v, dtype=torch.int16)
                    neg[k] = torch.zeros_like(v, dtype=torch.int16)
            sum_t[k] += v
            if not no_sign_filter:
                pos[k] += (v > eps).to(torch.int16)
                neg[k] += (v < -eps).to(torch.int16)
        del delta
        gc.collect()

    if n == 0:
        raise ValueError("no adapters")

    merged: dict[str, torch.Tensor] = {}
    if no_sign_filter:
        for k, s in sum_t.items():
            merged[k] = s / n
    else:
        thr = max(1, int(round(n * unanimity_ratio)))
        for k, s in sum_t.items():
            keep = (pos[k] >= thr) | (neg[k] >= thr)
            mean = s / n
            merged[k] = torch.where(keep, mean, torch.zeros_like(mean))
    return merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dirs", type=str, nargs="+", required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--axes", type=str, nargs="+", default=["race", "gender", "ses"])
    p.add_argument("--unanimity_ratio", type=float, default=0.75)
    p.add_argument("--tau_cross", type=float, default=0.67)
    p.add_argument("--default_scale", type=float, default=1.0)
    p.add_argument("--delta_extra_scale", type=float, default=1.0)
    p.add_argument("--no_sign_filter", action="store_true")
    p.add_argument("--no_cross_axis", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_axis: dict[str, list[str]] = defaultdict(list)
    for ad in args.adapter_dirs:
        by_axis[parse_axis_from_path(ad)].append(ad)

    axis_vecs: list[dict[str, torch.Tensor]] = []
    survivor_log = {}

    for axis in args.axes:
        dirs = by_axis.get(axis, [])
        if not dirs:
            raise ValueError(f"Missing adapters for axis='{axis}'.")
        print(f"  [{axis}] merging {len(dirs)} adapters", flush=True)
        merged = merge_axis_streaming(
            dirs,
            unanimity_ratio=args.unanimity_ratio,
            default_scale=args.default_scale,
            delta_extra_scale=args.delta_extra_scale,
            no_sign_filter=args.no_sign_filter,
        )
        sr = survivor_ratio(merged)
        survivor_log[f"within_{axis}"] = sr
        print(f"  [{axis}] within-axis survivor: {sr:.4%}", flush=True)
        torch.save(merged, out_dir / f"axis_{axis}_consensus.pt")
        axis_vecs.append(merged)
        gc.collect()

    if args.no_sign_filter or args.no_cross_axis:
        keys = set(axis_vecs[0].keys())
        for v in axis_vecs[1:]:
            keys &= set(v.keys())
        delta_star = {
            k: torch.stack([v[k].float() for v in axis_vecs], dim=0).mean(dim=0)
            for k in keys
        }
    else:
        delta_star = cross_axis_sign_intersection(axis_vecs, tau_cross=args.tau_cross)

    sr_cross = survivor_ratio(delta_star)
    survivor_log["cross_axis_delta_star"] = sr_cross
    print(f"  [Δ★] cross-axis survivor: {sr_cross:.4%}", flush=True)
    torch.save(delta_star, out_dir / "delta_star.pt")
    with open(out_dir / "survivor_log.json", "w") as f:
        json.dump(survivor_log, f, indent=2)
    print(f"Saved to {out_dir} | Survivor: {survivor_log}", flush=True)


if __name__ == "__main__":
    main()
