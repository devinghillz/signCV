"""
Apply Bias Vector to a pre-trained checkpoint: load θ_org, θ_bias, save θ_debias = θ_org - λ(θ_bias - θ_org).
"""

from __future__ import annotations

import argparse
import os

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

from bias_vector.debias import build_debiased_state_dict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_path", type=str, required=True, help="Original PLM (HF id or local path)")
    p.add_argument("--biased_path", type=str, required=True, help="Continual-MLM biased model directory")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--lam", type=float, default=1.0, help="Scaling factor λ")
    p.add_argument(
        "--architecture",
        type=str,
        default="masked_lm",
        choices=["masked_lm", "auto"],
        help="masked_lm: AutoModelForMaskedLM; auto: AutoModel",
    )
    p.add_argument("--skip_extra", type=str, nargs="*", default=[], help="Extra substrings in param names to skip")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.architecture == "masked_lm":
        org = AutoModelForMaskedLM.from_pretrained(args.pretrained_path)
        bias = AutoModelForMaskedLM.from_pretrained(args.biased_path)
    else:
        org = AutoModel.from_pretrained(args.pretrained_path)
        bias = AutoModel.from_pretrained(args.biased_path)

    new_state = build_debiased_state_dict(
        org.state_dict(),
        bias.state_dict(),
        lam=args.lam,
        extra_skip_substrings=args.skip_extra,
    )

    # Load structure from PLM, then load debiased weights
    if args.architecture == "masked_lm":
        debiased = AutoModelForMaskedLM.from_pretrained(args.pretrained_path)
    else:
        debiased = AutoModel.from_pretrained(args.pretrained_path)
    debiased.load_state_dict(new_state, strict=True)

    debiased.save_pretrained(args.output_dir)
    AutoConfig.from_pretrained(args.pretrained_path).save_pretrained(args.output_dir)
    try:
        tok = AutoTokenizer.from_pretrained(args.pretrained_path)
        tok.save_pretrained(args.output_dir)
    except OSError:
        pass

    print(f"Saved debiased model to {args.output_dir} (λ={args.lam})")


if __name__ == "__main__":
    main()
