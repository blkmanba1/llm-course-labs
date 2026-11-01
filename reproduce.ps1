# One-shot reproduction (Windows PowerShell; ASCII-only body so PS 5.1 parses it correctly)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\reproduce.ps1 -Python "<path to venv python>"
# If -Python is omitted, $env:LLM_PY is used, then plain "python".

param(
    [string]$Python = "",
    [int]$Ch1Seeds = 3
)

$ErrorActionPreference = "Stop"
if (-not $Python) {
    if ($env:LLM_PY) { $Python = $env:LLM_PY } else { $Python = "python" }
}
$root = $PSScriptRoot
Write-Host "python = $Python"
& $Python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

Write-Host ""
Write-Host "=== ch1: BERT fine-tuning ==="
Push-Location "$root\labs\ch1_bert_disaster_tweets"
if ($Ch1Seeds -ge 3) { & $Python run_seeds.py } else { & $Python train_bert.py --epochs 1 --batch 8 --accum 4 --max_len 512 --seed 42 }
Pop-Location

Write-Host ""
Write-Host "=== ch7: GPT-2 text steganography ==="
Push-Location "$root\labs\ch7_gpt2_stegano"
& $Python run_stegano.py
Pop-Location

Write-Host ""
Write-Host "=== ch11: sentiment-reward RLHF ==="
Push-Location "$root\labs\ch11_rlhf_sentiment"
& $Python run_rlhf.py
Pop-Location

Write-Host ""
Write-Host "=== figures ==="
& $Python "$root\make_figures.py"

Write-Host ""
Write-Host "done: results in labs/*/results/, figures in docs/figures/"
