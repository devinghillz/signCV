from __future__ import annotations

import argparse
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
    bin_path = p / "adapter_model.bin"
    if safe_path.exists():
        return load_safetensors(str(safe_path))
    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dirs", type=str, nargs="+", required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--unanimity_ratio", type=float, default=1.0)
    p.add_argument("--default_scale", type=float, default=1.0)
    p.add_argument("--delta_extra_scale", type=float, default=1.0)
    p.add_argument("--axes", type=str, nargs="+", default=["race", "gender", "religion", "profession"])
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(list)
    for ad in args.adapter_dirs:
        axis = parse_axis_from_path(ad)
        state = load_adapter_state(ad)
        delta = extract_lora_delta_state_dict(
            state,
            default_scale=args.default_scale,
            extra_scale=args.delta_extra_scale,
        )
        grouped[axis].append(delta)

    axis_vecs: list[dict[str, torch.Tensor]] = []
    for axis in args.axes:
        if axis not in grouped or not grouped[axis]:
            raise ValueError(f"Missing adapters for axis='{axis}'.")
        merged = sign_unanimity_merge(grouped[axis], unanimity_ratio=args.unanimity_ratio)
        torch.save(merged, out_dir / f"axis_{axis}_consensus.pt")
        axis_vecs.append(merged)

    delta_star = cross_axis_sign_intersection(axis_vecs)
    torch.save(delta_star, out_dir / "delta_star.pt")
    print(f"Saved merged vectors to {out_dir}")


if __name__ == "__main__":
    main()
