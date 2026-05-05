#!/usr/bin/env bash
# =============================================================
# SignCV 전체 파이프라인 — 논문 Table 2, 3, 4 수치 생성
#
# 전체 실행:  bash run_pipeline.sh
# 단계 지정:  STEP=eval bash run_pipeline.sh
#   STEP 옵션: data | train | merge | project | eval | table
# =============================================================
set -euo pipefail

# ─────────────── 논문 Table 1 기준 설정값 ───────────────
BASE_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
AXES=(race gender ses)
LRS=(0.0001 0.0002)
SEEDS=(42 43)
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_LENGTH=128
LORA_R=64
LORA_ALPHA=128
UNANIMITY_RATIO=0.75      # 논문 τ
TAU_CROSS=0.67            # 논문 τ_cross
K=1.0                     # 논문 k

# ─────────────── 경로 ───────────────
DATA_DIR="./processed_bbq"
ADAPTERS_ROOT="./checkpoints/adapters"
MERGED_ROOT="./checkpoints/merged"
RESULTS_DIR="./results"

STEP="${STEP:-all}"

mkdir -p "$ADAPTERS_ROOT" "$MERGED_ROOT" "$RESULTS_DIR"

# ═══════════════════════════════════════════════
# Step 1: 데이터 준비
# ═══════════════════════════════════════════════
if [[ "$STEP" == "all" || "$STEP" == "data" ]]; then
    echo "=========================================="
    echo "[Step 1] BBQ 데이터 전처리"
    echo "=========================================="

    # 1a. 단일축 학습/평가 데이터 (Race_ethnicity, SES, Gender_identity)
    python 전처리파일.py

    # 1b. run_signcv.sh용 combined 파일 생성 (axis 필드 포함)
    python combine_bbq_data.py

    # 1c. Held-out 다중속성 평가 데이터 (race×gender, race×SES)
    python prepare_bbq_table3_eval.py --output_dir "$DATA_DIR"

    echo "[완료] 데이터 준비"
fi

# ═══════════════════════════════════════════════
# Step 2: LoRA 학습
# ═══════════════════════════════════════════════
if [[ "$STEP" == "all" || "$STEP" == "train" ]]; then
    echo ""
    echo "=========================================="
    echo "[Step 2] LoRA 학습 (${#AXES[@]}×${#LRS[@]}×${#SEEDS[@]} runs)"
    echo "=========================================="

    for axis in "${AXES[@]}"; do
        for lr in "${LRS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                out="${ADAPTERS_ROOT}/${axis}_lr${lr}_seed${seed}"
                echo "[*] 학습: axis=${axis} lr=${lr} seed=${seed}"

                python train_lora_axis.py \
                    --base_model    "$BASE_MODEL" \
                    --data_mode     json \
                    --data_file     "${DATA_DIR}/combined_forget_set.json" \
                    --axis          "$axis" \
                    --axis_field    axis \
                    --text_field    text \
                    --out_dir       "$out" \
                    --lr            "$lr" \
                    --seed          "$seed" \
                    --epochs        "$EPOCHS" \
                    --batch_size    "$BATCH_SIZE" \
                    --grad_accum    "$GRAD_ACCUM" \
                    --max_length    "$MAX_LENGTH" \
                    --lora_r        "$LORA_R" \
                    --lora_alpha    "$LORA_ALPHA"\
                    --lora_dropout  0.1
            done
        done
    done
    echo "[완료] adapter 저장: $ADAPTERS_ROOT"
fi

# adapter 경로 재수집 (단독 실행 지원)
ADAPTER_DIRS=()
for axis in "${AXES[@]}"; do
    for lr in "${LRS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            ADAPTER_DIRS+=("${ADAPTERS_ROOT}/${axis}_lr${lr}_seed${seed}")
        done
    done
done

# ═══════════════════════════════════════════════
# Step 3: Sign-consensus merge
# Full SignCV / Mean Delta Edit / Within-Sign Only
# ═══════════════════════════════════════════════
if [[ "$STEP" == "all" || "$STEP" == "merge" ]]; then
    echo ""
    echo "=========================================="
    echo "[Step 3] Sign-consensus merge"
    echo "=========================================="

    # 3a. Full SignCV (논문 메인)
    echo "[*] Full SignCV (τ=${UNANIMITY_RATIO}, τ_cross=${TAU_CROSS})"
    python merge_sign_consensus.py \
        --adapter_dirs    "${ADAPTER_DIRS[@]}" \
        --output_dir      "${MERGED_ROOT}/signcv" \
        --axes            "${AXES[@]}" \
        --unanimity_ratio "$UNANIMITY_RATIO" \
        --tau_cross       "$TAU_CROSS" \
        --delta_extra_scale 1.0

    # 3b. Mean Delta Edit (ablation: sign filtering 없음)
    echo "[*] Mean Delta Edit"
    python merge_sign_consensus.py \
        --adapter_dirs    "${ADAPTER_DIRS[@]}" \
        --output_dir      "${MERGED_ROOT}/mean_delta" \
        --axes            "${AXES[@]}" \
        --no_sign_filter \
        --delta_extra_scale 1.0

    # 3c. Within-Sign Only (ablation: cross-axis intersection 없음)
    echo "[*] Within-Sign Only (τ=${UNANIMITY_RATIO})"
    python merge_sign_consensus.py \
        --adapter_dirs    "${ADAPTER_DIRS[@]}" \
        --output_dir      "${MERGED_ROOT}/within_only" \
        --axes            "${AXES[@]}" \
        --unanimity_ratio "$UNANIMITY_RATIO" \
        --no_cross_axis \
        --delta_extra_scale 1.0

    echo "[완료] Δ★ 저장: $MERGED_ROOT"
fi

# ═══════════════════════════════════════════════
# Step 4: Projection edit
# ═══════════════════════════════════════════════
if [[ "$STEP" == "all" || "$STEP" == "project" ]]; then
    echo ""
    echo "=========================================="
    echo "[Step 4] Projection edit (k=${K})"
    echo "=========================================="

    for variant in signcv mean_delta within_only; do
        out_model="./checkpoints/debiased_${variant}_k${K}"
        echo "[*] Projection: ${variant} → ${out_model}"
        python project_debias.py \
            --base_model      "$BASE_MODEL" \
            --delta_star_path "${MERGED_ROOT}/${variant}/delta_star.pt" \
            --output_dir      "$out_model" \
            --k               "$K"
    done
    echo "[완료] debiased model 저장"
fi

# ═══════════════════════════════════════════════
# Step 5: 평가
# ═══════════════════════════════════════════════
if [[ "$STEP" == "all" || "$STEP" == "eval" ]]; then
    echo ""
    echo "=========================================="
    echo "[Step 5] 평가"
    echo "=========================================="

    # ── 단일축 카테고리 설정 ──
    declare -A CAT_EVAL   # category → eval_data.json 경로
    declare -A CAT_RETAIN # category → retain_set.json 경로
    CAT_EVAL["Race_ethnicity"]="${DATA_DIR}/Race_ethnicity/eval_data.json"
    CAT_EVAL["SES"]="${DATA_DIR}/SES/eval_data.json"
    CAT_EVAL["Gender_identity"]="${DATA_DIR}/Gender_identity/eval_data.json"
    CAT_RETAIN["Race_ethnicity"]="${DATA_DIR}/Race_ethnicity/retain_set.json"
    CAT_RETAIN["SES"]="${DATA_DIR}/SES/retain_set.json"
    CAT_RETAIN["Gender_identity"]="${DATA_DIR}/Gender_identity/retain_set.json"

    # ── held-out 다중속성 카테고리 ──
    declare -A HELD_EVAL
    HELD_EVAL["race_x_gender"]="${DATA_DIR}/eval_race_x_gender.json"
    HELD_EVAL["race_x_ses"]="${DATA_DIR}/eval_race_x_ses.json"

    # 5a. Base model (Table 2, 3 Base 행)
    echo "[*] Base model 평가..."
    for cat in Race_ethnicity SES Gender_identity; do
        python run_eval.py \
            --model_path  "$BASE_MODEL" \
            --eval_json   "${CAT_EVAL[$cat]}" \
            --retain_json "${CAT_RETAIN[$cat]}" \
            --output_json "${RESULTS_DIR}/base_${cat}.json"
    done
    for cat in race_x_gender race_x_ses; do
        python run_eval.py \
            --model_path  "$BASE_MODEL" \
            --eval_json   "${HELD_EVAL[$cat]}" \
            --output_json "${RESULTS_DIR}/base_${cat}.json"
    done

    # 5b. Full SignCV (Table 2, 3 SignCV 행)
    echo "[*] Full SignCV 평가..."
    SC_MODEL="./checkpoints/debiased_signcv_k${K}"
    for cat in Race_ethnicity SES Gender_identity; do
        python run_eval.py \
            --model_path  "$SC_MODEL" \
            --eval_json   "${CAT_EVAL[$cat]}" \
            --retain_json "${CAT_RETAIN[$cat]}" \
            --output_json "${RESULTS_DIR}/signcv_${cat}.json"
    done
    for cat in race_x_gender race_x_ses; do
        python run_eval.py \
            --model_path  "$SC_MODEL" \
            --eval_json   "${HELD_EVAL[$cat]}" \
            --output_json "${RESULTS_DIR}/signcv_${cat}.json"
    done

    # 5c. Mean Delta Edit (Table 4 ablation)
    echo "[*] Mean Delta Edit 평가..."
    MD_MODEL="./checkpoints/debiased_mean_delta_k${K}"
    for cat in Race_ethnicity SES Gender_identity; do
        python run_eval.py \
            --model_path  "$MD_MODEL" \
            --eval_json   "${CAT_EVAL[$cat]}" \
            --retain_json "${CAT_RETAIN[$cat]}" \
            --output_json "${RESULTS_DIR}/mean_delta_${cat}.json"
    done

    # 5d. Within-Sign Only (Table 4 ablation)
    echo "[*] Within-Sign Only 평가..."
    WO_MODEL="./checkpoints/debiased_within_only_k${K}"
    for cat in Race_ethnicity SES Gender_identity; do
        python run_eval.py \
            --model_path  "$WO_MODEL" \
            --eval_json   "${CAT_EVAL[$cat]}" \
            --retain_json "${CAT_RETAIN[$cat]}" \
            --output_json "${RESULTS_DIR}/within_only_${cat}.json"
    done

    echo "[완료] 평가 결과: $RESULTS_DIR"
fi

# ═══════════════════════════════════════════════
# Step 6: 논문 테이블 출력
# ═══════════════════════════════════════════════
if [[ "$STEP" == "all" || "$STEP" == "table" ]]; then
    echo ""
    echo "=========================================="
    echo "[Step 6] 논문 테이블 출력"
    echo "=========================================="
    python print_tables.py \
        --results_dir  "$RESULTS_DIR" \
        --merged_root  "$MERGED_ROOT"
fi

echo ""
echo "✅ 파이프라인 완료"
