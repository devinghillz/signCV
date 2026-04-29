# KDD2026 Repro Pipeline

This repository now contains two comparable debiasing pipelines:

- Bias-vector baseline (`train_mlm_bias.py`, `apply_debias.py`)
- SignCV-style pipeline (`train_lora_axis.py`, `merge_sign_consensus.py`, `project_debias.py`)

Both can be aligned to a shared BBQ-format dataset for fair comparison.

## 1) Prepare shared BBQ files

```bash
python prepare_bbq_shared.py --data_mode json --input_json data/bbq_raw.json --output_dir data
```

Outputs:

- `data/bbq_forget_set.json` (training text for bias-vector baseline)
- `data/bbq_eval_data.json` (shared evaluation file)

## 2) Run SignCV pipeline (Linux)

```bash
chmod +x run_signcv.sh
DRY_RUN=true bash run_signcv.sh
```

Use `DRY_RUN=false` for full runs.

## 3) Run BBQ comparison (Linux)

```bash
chmod +x run_compare_bbq.sh
BBQ_RAW_JSON=data/bbq_raw.json SIGNCV_MODEL_PATH=checkpoints/signcv_debiased/k_0.5 bash run_compare_bbq.sh
```

Outputs:

- `checkpoints/eval_bias_bbq.json`
- `checkpoints/eval_signcv_bbq.json`

## Notes

- `run_signcv.ps1` is for Windows PowerShell.
- `run_signcv.sh` and `run_compare_bbq.sh` are for Linux/servers/containers.
- For long runs, use `tmux` or `screen` on shared servers.
