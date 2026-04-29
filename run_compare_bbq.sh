#!/usr/bin/env bash
set -euo pipefail

# Required inputs
BBQ_RAW_JSON="${BBQ_RAW_JSON:-data/bbq_raw.json}"
SIGNCV_MODEL_PATH="${SIGNCV_MODEL_PATH:-checkpoints/signcv_debiased/k_0.5}"
BIAS_MODEL_PATH="${BIAS_MODEL_PATH:-checkpoints/bias_bbq}"

# Optional inputs
BASE_BERT_MODEL="${BASE_BERT_MODEL:-google-bert/bert-base-uncased}"
AXIS="${AXIS:-}"

python prepare_bbq_shared.py \
  --data_mode json \
  --input_json "$BBQ_RAW_JSON" \
  --output_dir data

BIAS_ARGS=()
if [[ -n "$AXIS" ]]; then
  BIAS_ARGS+=(--axis "$AXIS")
fi

python train_mlm_bias.py \
  --model_name "$BASE_BERT_MODEL" \
  --output_dir "$BIAS_MODEL_PATH" \
  --data_mode json \
  --json_file data/bbq_forget_set.json \
  --json_text_field text \
  --epochs 30 \
  --batch_size 128 \
  --lr 1e-4 \
  --warmup_steps 10000 \
  --seed 42 \
  "${BIAS_ARGS[@]}"

EVAL_ARGS=()
if [[ -n "$AXIS" ]]; then
  EVAL_ARGS+=(--axis "$AXIS")
fi

python eval_compare_bbq.py \
  --model_path "$BIAS_MODEL_PATH" \
  --data_file data/bbq_eval_data.json \
  --architecture masked_lm \
  --output_json checkpoints/eval_bias_bbq.json \
  "${EVAL_ARGS[@]}"

python eval_compare_bbq.py \
  --model_path "$SIGNCV_MODEL_PATH" \
  --data_file data/bbq_eval_data.json \
  --architecture causal_lm \
  --output_json checkpoints/eval_signcv_bbq.json \
  "${EVAL_ARGS[@]}"

echo "BBQ comparison completed."
