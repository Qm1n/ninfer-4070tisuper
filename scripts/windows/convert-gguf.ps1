# Converts a Qwen3.8-27B GGUF checkpoint pair into one NInfer artifact.
#
#   .\convert-gguf.ps1 -Frontend C:\models\Qwen3.8-27B -Gguf C:\models\model.gguf `
#                      -Mmproj C:\models\mmproj-Qwen3.8-27B-BF16.gguf -Out C:\models\qwen3_8_27b.ninfer
#
# The frontend directory supplies config.json, the six registered frontend resources, and the
# tokenizer used for the draft-head shortlist.  Quantization runs on the CPU by default; pass
# -Device cuda only when torch with CUDA support is installed.

param(
    [Parameter(Mandatory = $true)][string]$Frontend,
    [Parameter(Mandatory = $true)][string]$Gguf,
    [Parameter(Mandatory = $true)][string]$Mmproj,
    [Parameter(Mandatory = $true)][string]$Out,
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu',
    [string]$Python
)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

if (-not $Python) {
    foreach ($candidate in @(
        (Join-Path $repo '..\python312\python.exe'),
        (Join-Path $repo '..\..\python312\python.exe')
    )) {
        if (Test-Path -LiteralPath $candidate) { $Python = (Resolve-Path $candidate).Path; break }
    }
}
if (-not $Python) {
    $found = Get-Command python -ErrorAction SilentlyContinue
    if ($found) { $Python = $found.Source }
}
if (-not $Python) { throw 'no Python interpreter found; pass -Python <path>' }

foreach ($path in @($Frontend, $Gguf, $Mmproj)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "missing input: $path" }
}

Write-Host "converting with $Python (device=$Device)"
$exitCode = 1
Push-Location $repo
try {
    & $Python -u -m tools.convert.qwen3_8_27b.convert_gguf --frontend $Frontend --gguf $Gguf `
        --mmproj $Mmproj --out $Out --device $Device
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
