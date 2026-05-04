from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors

from bias_vector.signcv import (
    cross_axis_sign_intersection,
    extract_lora_delta_state_dict,
    parse_axis_from_path,
    sign_unanimity_merge,
)


def load_adapter_state(adapter_dir: str) -> dict[str, torch.Tensor]:
    p = Path(adapter_dir)
    safe_path = p / "adapter_model.safetensors"
    bin_path  = p / "adapter_model.bin"
    if safe_path.exists():
        return load_safetensors(str(safe_path))
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def survivor_ratio(vec: dict[str, torch.Tensor]) -> float:
    total, nnz = 0, 0
    for v in vec.values():
        total += v.numel()
        nnz   += (v.abs() > 1e-10).sum().item()
    return nnz / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dirs",      type=str, nargs="+", required=True)
    p.add_argument("--output_dir",        type=str, required=True)
    p.add_argument("--axes",              type=str, nargs="+",
                   default=["race", "gender", "ses"])
    p.add_argument("--unanimity_ratio",   type=float, default=0.75)
    p.add_argument("--tau_cross",         type=float, default=0.67)
    p.add_argument("--default_scale",     type=float, default=1.0)
    p.add_argument("--delta_extra_scale", type=float, default=1.0)
    p.add_argument("--no_sign_filter",  action="store_true",
                   help="Mean Delta Edit: sign filtering 없이 단순 평균")
    p.add_argument("--no_cross_axis",   action="store_true",
                   help="Within-Sign Only: cross-axis intersection 없이 axis별 평균")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(list)
    for ad in args.adapter_dirs:
        axis  = parse_axis_from_path(ad)
        state = load_adapter_state(ad)
        delta = extract_lora_delta_state_dict(
            state,
            default_scale=args.default_scale,
            extra_scale=args.delta_extra_scale,
        )
        grouped[axis].append(delta)

    axis_vecs: list[dict[str, torch.Tensor]] = []
    survivor_log = {}

    for axis in args.axes:
        if axis not in grouped or not grouped[axis]:
            raise ValueError(f"Missing adapters for axis='{axis}'.")

        if args.no_sign_filter:
            keys = set(grouped[axis][0].keys())
            for d in grouped[axis][1:]:
                keys &= set(d.keys())
            merged = {
                k: torch.stack([d[k].float() for d in grouped[axis]], dim=0).mean(dim=0)
                for k in keys
            }
        else:
            merged = sign_unanimity_merge(
                grouped[axis],
                unanimity_ratio=args.unanimity_ratio,
            )

        sr = survivor_ratio(merged)
        survivor_log[f"within_{axis}"] = sr
        print(f"  [{axis}] within-axis survivor: {sr:.4%}")
        torch.save(merged, out_dir / f"axis_{axis}_consensus.pt")
        axis_vecs.append(merged)

    if args.no_sign_filter or args.no_cross_axis:
        keys = set(axis_vecs[0].keys())
        for v in axis_vecs[1:]:
            keys &= set(v.keys())
        delta_star = {
            k: torch.stack([v[k].float() for v in axis_vecs], dim=0).mean(dim=0)
            for k in keys
        }
    else:
        delta_star = cross_axis_sign_intersection(
            axis_vecs,
            tau_cross=args.tau_cross,
        )

    sr_cross = survivor_ratio(delta_star)
    survivor_log["cross_axis_delta_star"] = sr_cross
    print(f"  [Δ★] cross-axis survivor: {sr_cross:.4%}")

    torch.save(delta_star, out_dir / "delta_star.pt")

    with open(out_dir / "survivor_log.json", "w") as f:
        json.dump(survivor_log, f, indent=2)

    print(f"Saved to {out_dir} | Survivor: {survivor_log}")


if __name__ == "__main__":
    main()
