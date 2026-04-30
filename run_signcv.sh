#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
DATA_MODE="${DATA_MODE:-json}"
TRAIN_FILE="${TRAIN_FILE:-data/forget_set.json}"
EVAL_FILE="${EVAL_FILE:-data/eval_data.json}"
AXIS_FIELD="${AXIS_FIELD:-axis}"
TEXT_FIELD="${TEXT_FIELD:-text}"
PROMPT_FIELD="${PROMPT_FIELD:-prompt}"
CHOICES_FIELD="${CHOICES_FIELD:-choices}"
LABEL_FIELD="${LABEL_FIELD:-label}"
DRY_RUN="${DRY_RUN:-true}"

AXES=(race gender religion profession)
LRS=(0.001 0.005)
SEEDS=(42 43)
K_VALUES=(0.1 0.2 0.35 0.5 0.75 1.0)

ADAPTERS_ROOT="checkpoints/signcv_adapters"
MERGED_ROOT="checkpoints/signcv_merged"
DEBIASED_ROOT="checkpoints/signcv_debiased"
EVAL_ROOT="checkpoints/signcv_eval"

mkdir -p "$ADAPTERS_ROOT" "$MERGED_ROOT" "$DEBIASED_ROOT" "$EVAL_ROOT"

MAX_SAMPLES=0
EPOCHS=10
if [[ "$DRY_RUN" == "true" ]]; then
  MAX_SAMPLES=64
  EPOCHS=1
fi

ADAPTER_DIRS=()
for axis in "${AXES[@]}"; do
  for lr in "${LRS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      tag="${axis}_lr${lr}_seed${seed}"
      out="${ADAPTERS_ROOT}/${tag}"
      ADAPTER_DIRS+=("$out")
python3 train_lora_axis.py \
        --base_model "$BASE_MODEL" \
        --data_mode "$DATA_MODE" \
        --data_file "$TRAIN_FILE" \
        --axis "$axis" \
        --axis_field "$AXIS_FIELD" \
        --text_field "$TEXT_FIELD" \
        --out_dir "$out" \
        --lr "$lr" \
        --seed "$seed" \
        --epochs "$EPOCHS" \
        --batch_size 1 \
        --grad_accum 4 \
        --max_length 128 \
        --max_samples "$MAX_SAMPLES"
    done
  done
done

python3 merge_sign_consensus.py \
  --adapter_dirs "${ADAPTER_DIRS[@]}" \
  --output_dir "$MERGED_ROOT" \
  --axes "${AXES[@]}" \
  --unanimity_ratio 1.0 \
  --delta_extra_scale 2.0

for k in "${K_VALUES[@]}"; do
  out_model="${DEBIASED_ROOT}/k_${k}"
python3 project_debias.py \
    --base_model "$BASE_MODEL" \
    --delta_star_path "${MERGED_ROOT}/delta_star.pt" \
    --output_dir "$out_model" \
    --k "$k"

  for axis in "${AXES[@]}"; do
python3 eval_bbq.py \
      --model_path "$out_model" \
      --data_mode "$DATA_MODE" \
      --data_file "$EVAL_FILE" \
      --prompt_field "$PROMPT_FIELD" \
      --choices_field "$CHOICES_FIELD" \
      --label_field "$LABEL_FIELD" \
      --axis_field "$AXIS_FIELD" \
      --axis "$axis" \
      --max_samples 200 \
      --output_json "${EVAL_ROOT}/eval_${axis}_k${k}.json"
  done
done

echo "SignCV pipeline completed."
