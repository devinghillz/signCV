from __future__ import annotations

import re
from typing import Iterable

import torch


def _extract_lora_target_prefixes(state_dict_keys: Iterable[str]) -> set[str]:
    prefixes: set[str] = set()
    for k in state_dict_keys:
        if ".lora_A." in k:
            prefixes.add(k.split(".lora_A.")[0])
        elif ".lora_B." in k:
            prefixes.add(k.split(".lora_B.")[0])
    return prefixes


def _get_lora_matrix(state: dict[str, torch.Tensor], prefix: str, which: str) -> torch.Tensor | None:
    patterns = [
        f"{prefix}.lora_{which}.default.weight",
        f"{prefix}.lora_{which}.weight",
    ]
    for p in patterns:
        if p in state:
            return state[p]
    return None


def _get_lora_scaling(state: dict[str, torch.Tensor], prefix: str, default_scale: float = 1.0) -> float:
    alpha_key = f"{prefix}.alpha"
    rank_key = f"{prefix}.rank"
    if alpha_key in state and rank_key in state:
        alpha = float(state[alpha_key].item())
        rank = float(state[rank_key].item())
        return alpha / max(rank, 1.0)
    # Fallback: if metadata keys are absent, use caller-provided default.
    return float(default_scale)


def extract_lora_delta_state_dict(
    adapter_state_dict: dict[str, torch.Tensor],
    default_scale: float = 1.0,
    extra_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    prefixes = _extract_lora_target_prefixes(adapter_state_dict.keys())
    for prefix in prefixes:
        a = _get_lora_matrix(adapter_state_dict, prefix, "A")
        b = _get_lora_matrix(adapter_state_dict, prefix, "B")
        if a is None or b is None:
            continue
        scale = _get_lora_scaling(adapter_state_dict, prefix, default_scale=default_scale)
        delta = (b @ a) * scale * float(extra_scale)
        base_prefix = prefix
        if base_prefix.startswith("base_model.model."):
            base_prefix = base_prefix[len("base_model.model.") :]
        out[f"{base_prefix}.weight"] = delta
    return out


def sign_unanimity_merge(
    deltas: list[dict[str, torch.Tensor]],
    unanimity_ratio: float = 0.75,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """
    논문 수식 (3)(4) 구현.
    unanimity_ratio=0.75 → 4개 run 중 3개 이상 동일 부호인 좌표만 유지.
    유지된 좌표는 단순 평균 (논문 수식 4: 1/|H_a| * sum(Delta * mask)).
    """
    if not deltas:
        raise ValueError("No deltas provided.")
    keys = set(deltas[0].keys())
    for d in deltas[1:]:
        keys &= set(d.keys())
    if not keys:
        raise ValueError("No common keys across deltas.")

    n = len(deltas)
    threshold = max(1, int(round(n * unanimity_ratio)))
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        stacked = torch.stack([d[k].float() for d in deltas], dim=0)
        signs = torch.sign(stacked)
        pos = (signs > 0).sum(dim=0)
        neg = (signs < 0).sum(dim=0)
        keep = (pos >= threshold) | (neg >= threshold)
        # 논문 수식 (4): 단순 평균 (L2 정규화 없음)
        merged = stacked.mean(dim=0)
        out[k] = torch.where(keep, merged, torch.zeros_like(merged))
    return out


def cross_axis_sign_intersection(
    axis_vectors: list[dict[str, torch.Tensor]],
    tau_cross: float = 0.67,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """
    논문 수식 (5)(6) 구현.
    tau_cross=0.67 → 3개 axis 중 2개 이상 동일 부호인 좌표만 유지.
    유지된 좌표는 axis별 방향의 단순 평균으로 Δ★ 구성.
    """
    if not axis_vectors:
        raise ValueError("No axis vectors provided.")
    keys = set(axis_vectors[0].keys())
    for d in axis_vectors[1:]:
        keys &= set(d.keys())
    if not keys:
        raise ValueError("No common keys across axis vectors.")

    n_axes = len(axis_vectors)
    threshold = max(1, int(round(n_axes * tau_cross)))  # 3개 → 2

    out: dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [d[k].float() for d in axis_vectors]
        stacked = torch.stack(tensors, dim=0)
        signs = torch.sign(stacked)
        # τ_cross 이상 동의하는 좌표만 유지 (논문 수식 5)
        pos_enough = (signs > 0).sum(dim=0) >= threshold
        neg_enough = (signs < 0).sum(dim=0) >= threshold
        keep = pos_enough | neg_enough
        # 논문 수식 (6): axis별 방향의 단순 평균
        merged = stacked.mean(dim=0)
        out[k] = torch.where(keep, merged, torch.zeros_like(merged))
    return out


def project_out_direction(
    base_state_dict: dict[str, torch.Tensor],
    direction_state_dict: dict[str, torch.Tensor],
    k: float = 1.0,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    common = [kk for kk in direction_state_dict.keys() if kk in base_state_dict]
    if not common:
        raise ValueError("No overlapping keys between base model and direction vector.")

    dot = 0.0
    norm2 = 0.0
    for name in common:
        theta = base_state_dict[name].float()
        delta = direction_state_dict[name].float()
        dot += torch.sum(theta * delta).item()
        norm2 += torch.sum(delta * delta).item()
    coeff = dot / max(norm2, eps)

    out: dict[str, torch.Tensor] = {}
    for name, val in base_state_dict.items():
        t = val.detach().clone()
        if name in direction_state_dict:
            d = direction_state_dict[name].to(dtype=t.dtype, device=t.device)
            t = t - float(k) * float(coeff) * d
        out[name] = t
    return out


def parse_axis_from_path(path: str) -> str:
    m = re.search(r"(race|gender|religion|profession|ses)", path, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not infer axis from path: {path}")
    return m.group(1).lower()
