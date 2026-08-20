# AAAI-27 SignCV protocol audit

## Frozen provenance finding

The historical 4-run paper pipelines pass `tau_cross=0.67` (Mistral) or
`tau_cross=0.6666667` (Qwen/Llama-3) into `cross_axis_sign_intersection`.
The implementation computes

```python
threshold = ceil(n_axes * tau_cross)
```

with `n_axes=3`.  Both values therefore produce `threshold=3` in the
executed code.  The published 7.05/7.76/7.85/53.48 cross-survivor values
must be described as **3-of-3 sign unanimity**, not 2-of-3 majority voting.
Comments that call `0.6666667` a 2-of-3 rule are documentation errors and
must not be used as method provenance.

Do not change the historical implementation when reproducing old numbers.
Use explicit integer cross counts in `protocol_audit.py` for sensitivity
analysis.

## Why this matters

For three objectives with marginal within-survivor probabilities
`p1,p2,p3`, an independent-support / symmetric-sign reference is:

- 3-of-3: `(p1*p2*p3)/4`
- 2-of-3: `p1*p2*p3 + 0.5 * [p1*p2*(1-p3) + p1*p3*(1-p2) + p2*p3*(1-p1)]`

The reference is descriptive calibration, not a significance test.

For the within rule, the symmetric iid-sign reference is

`2 * sum_{k=q..n} C(n,k) / 2^n`

for strict-majority `q`.  In particular:

- 3-of-4: 0.625
- 6-of-8: 0.2890625

## Audit script

`protocol_audit.py` performs three analyses without new LoRA training:

1. explicit 2-of-3 vs 3-of-3 cross-rule sensitivity;
2. 8-run within-threshold sensitivity (default 5/8, 6/8, 7/8, 8/8);
3. independent seed-split replication: `{42,43}` vs `{44,45}` while retaining both learning rates.

### Existing merged tensors only

```bash
python protocol_audit.py merged \
  --merged_dir /data/kdd_checkpoints/aaai27_8run/mistral_signcv \
  --cross_counts 2 3 \
  --out_json results/mistral_cross_rule_audit.json
```

Repeat for Qwen and Llama-3.

### Saved 8-run adapters

From each model-specific experiment directory whose `checkpoints/adapters`
contains `race_lr0.0001_seed42`-style adapter folders:

```bash
python /path/to/signCV/protocol_audit.py adapters \
  --adapter_root ./checkpoints/adapters \
  --lrs 0.0001 0.0002 \
  --seeds 42 43 44 45 \
  --within_counts 5 6 7 8 \
  --cross_counts 2 3 \
  --split_a 42 43 \
  --split_b 44 45 \
  --out_json results/protocol_audit.json
```

The adapter-mode audit mirrors the raw-mean within merge in
`merge_sign_consensus_lowmem.py`; it does not train or modify any model.

## Paper-use rule

The AAAI paper should report raw survivor rate together with repeat depth,
within consensus count, and cross consensus count.  Cross-checkpoint claims
should be based on protocol-matched calibration rather than raw percentages
alone.  Structural compatibility and downstream behavioral utility remain
separate evaluation targets.
