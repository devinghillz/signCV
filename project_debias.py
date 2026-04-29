from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bias_vector.signcv import project_out_direction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--delta_star_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--k", type=float, default=1.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    delta_star = torch.load(args.delta_star_path, map_location="cpu")

    new_state = project_out_direction(model.state_dict(), delta_star, k=args.k)
    model.load_state_dict(new_state, strict=False)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    meta = {
        "base_model": args.base_model,
        "delta_star_path": args.delta_star_path,
        "k": args.k,
    }
    torch.save(meta, Path(args.output_dir, "projection_metadata.pt"))
    print(f"Saved projected model to {args.output_dir} (k={args.k})")


if __name__ == "__main__":
    main()
