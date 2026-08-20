#!/usr/bin/env bash
set -euo pipefail

# One-command runner for AAAI-27 SignCV protocol audits.
# Run this from the repository root on the experiment server after pulling
# branch aaai27-aia-seed-extra. No model training is performed.

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-aaai27_audit_results}"
mkdir -p "$OUT_ROOT"

run_one () {
  local name="$1"
  local adapter_root="$2"
  local out_json="$OUT_ROOT/${name}_protocol_audit.json"

  if [[ ! -d "$adapter_root" ]]; then
    echo "[SKIP] $name: adapter root not found: $adapter_root" >&2
    return 0
  fi

  echo "[RUN] $name -> $out_json"
  "$PYTHON_BIN" protocol_audit.py adapters \
    --adapter_root "$adapter_root" \
    --lrs 0.0001 0.0002 \
    --seeds 42 43 44 45 \
    --within_counts 5 6 7 8 \
    --cross_counts 2 3 \
    --split_a 42 43 \
    --split_b 44 45 \
    --out_json "$out_json"
}

# Default repository-local layouts. Override any path through environment vars
# when checkpoints live elsewhere on the server.
MISTRAL_ADAPTER_ROOT="${MISTRAL_ADAPTER_ROOT:-ffinal/checkpoints/adapters}"
QWEN_ADAPTER_ROOT="${QWEN_ADAPTER_ROOT:-ffinal_qwen/checkpoints/adapters}"
LLAMA3_ADAPTER_ROOT="${LLAMA3_ADAPTER_ROOT:-ffinal_llama/checkpoints/adapters}"

run_one mistral "$MISTRAL_ADAPTER_ROOT"
run_one qwen2_5 "$QWEN_ADAPTER_ROOT"
run_one llama3 "$LLAMA3_ADAPTER_ROOT"

"$PYTHON_BIN" aggregate_protocol_audits.py \
  --input_dir "$OUT_ROOT" \
  --out_json "$OUT_ROOT/protocol_audit_master.json" \
  --out_csv "$OUT_ROOT/protocol_audit_master.csv"

echo "[DONE] Protocol audits written to $OUT_ROOT"
