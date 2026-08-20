# AAAI-27 cross-rule sensitivity recovered from stored 4-run summaries

These values are derived from the exact stored `pairwise_axis_metrics.json`
outputs for Qwen2.5-7B-Instruct and Meta-Llama-3-8B-Instruct.  No model
training or checkpoint access is required.

## Identity used

Let A_rg, A_rs, and A_gs denote the coordinate sets on which each pair is
jointly nonzero and has the same sign.  Under an explicit 2-of-3 cross rule,
a coordinate survives iff it belongs to the union of these three sets.
Any intersection of two such pair events implies all three axes are nonzero
and share the same sign.  Therefore

```
S_2of3 = [|A_rg| + |A_rs| + |A_gs| - 2 |A_rgs|] / N,
```

where

```
|A_ij|  = n_both_nonzero_ij * sign_agreement_ij
|A_rgs| = n_triple_nonzero * triple_sign_agreement
```

(up to floating-point rounding in the stored agreement rates).

## Results

| Model | 2-of-3 observed | 2-of-3 reference | Obs./Ref | 3-of-3 observed | 3-of-3 reference | Obs./Ref | Raw 2/3-to-3/3 factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 0.511168 | 0.504571 | 1.0131 | 0.077612 | 0.070627 | 1.0989 | 6.586x |
| Meta-Llama-3-8B-Instruct | 0.514719 | 0.508197 | 1.0128 | 0.078525 | 0.071515 | 1.0980 | 6.555x |

The independent-support / symmetric-sign references use the observed marginal
within-axis survivor probabilities.  For three axes:

```
P_ref(3-of-3) = p1*p2*p3 / 4

P_ref(2-of-3) = p1*p2*p3
  + 0.5 * [p1*p2*(1-p3) + p1*p3*(1-p2) + p2*p3*(1-p1)]
```

## Paper-facing interpretation

The raw cross survivor rate changes by more than 6.5x solely by changing the
cross voting rule from unanimity to majority.  After matching the analytic
reference to the voting rule, both checkpoints remain close to the reference:
about 1.01x under 2-of-3 and 1.10x under 3-of-3.  This is direct evidence that
raw cross-survivor magnitude is highly protocol-dependent, while calibrated
departure from a protocol-matched reference is substantially more stable.

This result should replace any claim that a raw ~8% survivor rate alone
constitutes evidence of unusually sparse or conflicting multi-attribute
geometry.
