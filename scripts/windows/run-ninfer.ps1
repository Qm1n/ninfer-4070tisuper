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

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

if (-not $Exe) {
    $Exe = @(
        (Join-Path $repo 'build\apps\ninfer.exe'),
        (Join-Path $repo '..\build-ninfer-sm89\apps\ninfer.exe')
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) {
    throw 'ninfer.exe was not found; build the project or pass -Exe <path>'
}
if (-not (Test-Path -LiteralPath $Model)) { throw "model artifact was not found: $Model" }

# cudart64_12.dll lives in the toolkit bin directory and is the only runtime dependency.
$cudaBin = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin' } else { $null }
if (-not $cudaBin -or -not (Test-Path -LiteralPath (Join-Path $cudaBin 'cudart64_12.dll'))) {
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
