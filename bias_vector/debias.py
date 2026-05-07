"""Bias Vector: θ_debias = θ_org − λ (θ_bias − θ_org). LayerNorm weights are not modified."""

from __future__ import annotations

import copy
import re
from typing import Iterable

import torch


_DEFAULT_SKIP_PATTERNS = (
    r"LayerNorm",
    r"layer_norm",
    r"layernorm",
)


def should_skip_param(name: str, extra_skip_substrings: Iterable[str] | None = None) -> bool:
    if extra_skip_substrings:
        for s in extra_skip_substrings:
            if s and s in name:
                return True
    for pat in _DEFAULT_SKIP_PATTERNS:
        if re.search(pat, name, flags=re.IGNORECASE):
            return True
    return False


def build_debiased_state_dict(
    state_org: dict[str, torch.Tensor],
    state_bias: dict[str, torch.Tensor],
    lam: float = 1.0,
    extra_skip_substrings: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    θ_debias = θ_org - lam * (θ_bias - θ_org).

    Keys must match between the two checkpoints. Skipped keys keep θ_org values.
    """
    out: dict[str, torch.Tensor] = {}
    keys_org = set(state_org.keys())
    keys_bias = set(state_bias.keys())
    missing_in_bias = keys_org - keys_bias
    if missing_in_bias:
        raise KeyError(f"Biased checkpoint missing keys: {sorted(missing_in_bias)[:20]} ...")
    extra_in_bias = keys_bias - keys_org
    if extra_in_bias:
        raise KeyError(f"Biased checkpoint has unexpected keys: {sorted(extra_in_bias)[:20]} ...")

    for name, t_org in state_org.items():
        t_org = t_org.detach().clone()
        if should_skip_param(name, extra_skip_substrings):
            out[name] = t_org
            continue
        t_bias = state_bias[name].detach()
        delta = t_bias - t_org
        out[name] = t_org - lam * delta
    return out
