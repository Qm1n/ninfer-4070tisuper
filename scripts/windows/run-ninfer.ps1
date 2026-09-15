# Runs the NInfer CLI with the CUDA runtime on PATH and a profile tuned for a 16 GB Ada card.
#
#   .\run-ninfer.ps1 -Model C:\models\qwen3_8_27b_minq4.ninfer -Prompt "hello"
#   .\run-ninfer.ps1 -Model ... -Prompt ... -Profile plain -MaxContext 8192

param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$Prompt,
    [ValidateSet('plain', 'mtp')][string]$Profile = 'mtp',
    [int]$MaxContext = 16384,
    [int]$MaxNew = 512,
    [ValidateRange(16, 1024)][int]$PrefillChunk = 512,
    [string]$Exe
)

$ErrorActionPreference = 'Stop'

if (-not $Exe) {
    $Exe = Join-Path $PSScriptRoot '..\..\build\apps\ninfer.exe'
}
if (-not (Test-Path -LiteralPath $Exe)) { throw "ninfer.exe was not found: $Exe" }
if (-not (Test-Path -LiteralPath $Model)) { throw "model artifact was not found: $Model" }

# cudart64_12.dll lives in the toolkit bin directory and is the only runtime dependency.
$cudaBin = Join-Path $env:CUDA_PATH 'bin'
if (-not (Test-Path -LiteralPath (Join-Path $cudaBin 'cudart64_12.dll'))) {
    $cudaBin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin'
}
if (-not (Test-Path -LiteralPath (Join-Path $cudaBin 'cudart64_12.dll'))) {
    throw "cudart64_12.dll was not found; set CUDA_PATH to a CUDA 12 toolkit"
}
$env:PATH = "$cudaBin;$env:PATH"

$arguments = @(
    $Model,
    '--prompt', $Prompt,
    '--max-context', $MaxContext,
    '--kv-capacity', $MaxContext,
    '--max-new', $MaxNew,
    '--prefill-chunk', $PrefillChunk
)

if ($Profile -eq 'mtp') {
    # MTP reserves an extra 940 MiB, so keep the KV pool explicit instead of automatic.
    $arguments += @('--kv-dtype', 'int8', '--spec', 'mtp', '--draft-tokens', '3', '--lm-head-draft')
}
else {
    $arguments += @('--kv-dtype', 'int8')
}

& $Exe @arguments
exit $LASTEXITCODE
