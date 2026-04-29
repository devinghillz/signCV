param(
    [switch]$DryRun = $true,
    [string]$BaseModel = "mistralai/Mistral-7B-Instruct-v0.3",
    [string]$DataMode = "json",
    [string]$DatasetName = "",
    [string]$DatasetConfig = "",
    [string]$TrainFile = "data/forget_set.json",
    [string]$EvalFile = "data/eval_data.json",
    [string]$TrainSplit = "train",
    [string]$EvalSplit = "train",
    [string]$AxisField = "axis",
    [string]$TextField = "text",
    [string]$PromptField = "prompt",
    [string]$ChoicesField = "choices",
    [string]$LabelField = "label"
)

$ErrorActionPreference = "Stop"

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )
    & python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: python $($Args -join ' ')"
    }
}

$axes = @("race", "gender", "religion", "profession")
$lrs = @(0.001, 0.005)
$seeds = @(42, 43)

$adaptersRoot = "checkpoints/signcv_adapters"
$mergedRoot = "checkpoints/signcv_merged"
$debiasedRoot = "checkpoints/signcv_debiased"
$evalRoot = "checkpoints/signcv_eval"

New-Item -ItemType Directory -Force -Path $adaptersRoot, $mergedRoot, $debiasedRoot, $evalRoot | Out-Null

$maxSamples = 0
$epochs = 10
if ($DryRun) {
    $maxSamples = 64
    $epochs = 1
}

$adapterDirs = @()
foreach ($axis in $axes) {
    foreach ($lr in $lrs) {
        foreach ($seed in $seeds) {
            $tag = "${axis}_lr${lr}_seed${seed}"
            $out = Join-Path $adaptersRoot $tag
            $adapterDirs += $out
            $trainCmd = @(
                "train_lora_axis.py",
                "--base_model", $BaseModel,
                "--data_mode", $DataMode,
                "--data_file", $TrainFile,
                "--split", $TrainSplit,
                "--axis", $axis,
                "--axis_field", $AxisField,
                "--text_field", $TextField,
                "--out_dir", $out,
                "--lr", "$lr",
                "--seed", "$seed",
                "--epochs", "$epochs",
                "--batch_size", "1",
                "--grad_accum", "4",
                "--max_length", "128",
                "--max_samples", "$maxSamples"
            )
            if ($DatasetName -ne "") {
                $trainCmd += @("--dataset_name", $DatasetName)
            }
            if ($DatasetConfig -ne "") {
                $trainCmd += @("--dataset_config", $DatasetConfig)
            }
            Invoke-PythonChecked -Args $trainCmd
        }
    }
}

$mergeCmd = @(
    "merge_sign_consensus.py",
    "--adapter_dirs"
) + $adapterDirs + @(
    "--output_dir", $mergedRoot,
    "--axes"
) + $axes + @(
    "--unanimity_ratio", "1.0",
    "--delta_extra_scale", "2.0"
)
Invoke-PythonChecked -Args $mergeCmd

$kValues = @(0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
foreach ($k in $kValues) {
    $modelOut = Join-Path $debiasedRoot "k_$k"
    $projCmd = @(
        "project_debias.py",
        "--base_model", $BaseModel,
        "--delta_star_path", (Join-Path $mergedRoot "delta_star.pt"),
        "--output_dir", $modelOut,
        "--k", "$k"
    )
    Invoke-PythonChecked -Args $projCmd

    foreach ($axis in $axes) {
        $jsonOut = Join-Path $evalRoot "eval_${axis}_k${k}.json"
        $evalCmd = @(
            "eval_bbq.py",
            "--model_path", $modelOut,
            "--data_mode", $DataMode,
            "--data_file", $EvalFile,
            "--split", $EvalSplit,
            "--prompt_field", $PromptField,
            "--choices_field", $ChoicesField,
            "--label_field", $LabelField,
            "--axis_field", $AxisField,
            "--axis", $axis,
            "--max_samples", "200",
            "--output_json", $jsonOut
        )
        if ($DatasetName -ne "") {
            $evalCmd += @("--dataset_name", $DatasetName)
        }
        if ($DatasetConfig -ne "") {
            $evalCmd += @("--dataset_config", $DatasetConfig)
        }
        Invoke-PythonChecked -Args $evalCmd
    }
}

Write-Host "SignCV pipeline completed."
