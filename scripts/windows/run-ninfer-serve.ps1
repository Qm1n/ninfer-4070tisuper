# Starts the NInfer OpenAI-compatible server with the CUDA runtime on PATH.
#
#   .\run-ninfer-serve.ps1 -Model C:\models\qwen3_8_27b_gguf.ninfer
#   .\run-ninfer-serve.ps1 -Model ... -Port 8080 -MaxContext 32768 -Profile plain
#
# Endpoints: /v1/chat/completions, /v1/responses, /v1/messages (Anthropic), /health.
# Ctrl+C stops the server.

param(
    [string]$Model,
    [int]$Port = 8080,
    [int]$MaxContext = 16384,
    [ValidateSet('plain', 'mtp')][string]$Profile = 'mtp',
    [string]$BindHost = '127.0.0.1',
    [string]$ApiKey,
    [ValidateSet('bf16', 'int8', 'i4', 'i4-g64')][string]$KvDtype = 'int8',
    [ValidateRange(16, 1024)][int]$PrefillChunk = 512,
    [switch]$NoThinking,
    [switch]$Cors,
    [string]$Exe
)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

if (-not $Exe) {
    $candidates = @(
        (Join-Path $repo 'build\apps\ninfer-serve.exe'),
        (Join-Path $repo '..\build-ninfer-sm89\apps\ninfer-serve.exe')
    )
    $Exe = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) {
    throw 'ninfer-serve.exe was not found; build the project or pass -Exe <path>'
}

if (-not $Model) {
    $artifacts = @()
    foreach ($directory in @((Join-Path $repo '..\models'), (Join-Path $repo 'models'))) {
        if (Test-Path -LiteralPath $directory) {
            $artifacts += Get-ChildItem -LiteralPath $directory -Filter '*.ninfer' -File -ErrorAction SilentlyContinue
        }
    }
    if ($artifacts.Count -eq 1) {
        $Model = $artifacts[0].FullName
    }
    elseif ($artifacts.Count -eq 0) {
        throw 'no .ninfer artifact was found; pass -Model <path>'
    }
    else {
        Write-Host 'several artifacts were found; pass -Model <path> to choose one:'
        $artifacts | ForEach-Object { Write-Host ('  ' + $_.FullName) }
        throw 'ambiguous model artifact'
    }
}
if (-not (Test-Path -LiteralPath $Model)) { throw "model artifact was not found: $Model" }

# cudart64_12.dll lives in the toolkit bin directory and is the only runtime dependency.
$cudaBin = if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'bin' } else { $null }
if (-not $cudaBin -or -not (Test-Path -LiteralPath (Join-Path $cudaBin 'cudart64_12.dll'))) {
    $cudaBin = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin'
}
if (-not (Test-Path -LiteralPath (Join-Path $cudaBin 'cudart64_12.dll'))) {
    throw 'cudart64_12.dll was not found; set CUDA_PATH to a CUDA 12 toolkit'
}
$env:PATH = "$cudaBin;$env:PATH"

$arguments = @(
    $Model,
    '--host', $BindHost,
    '--port', $Port,
    '--max-context', $MaxContext,
    '--kv-capacity', $MaxContext,
    '--kv-dtype', $KvDtype,
    # The 1024-token default makes the GDN gating cooperative launch too large on a 16 GB Ada card
    # (cudaErrorCooperativeLaunchTooLarge) once a prompt passes roughly 3k tokens; 512 fits and is
    # faster than the smaller chunks the 16 GB profile documents.
    '--prefill-chunk', $PrefillChunk
)
if ($ApiKey) { $arguments += @('--api-key', $ApiKey) }
if ($NoThinking) { $arguments += '--no-thinking' }
# Browsers need the CORS headers when a local page talks to this server directly.
if ($Cors) { $arguments += '--cors' }

if ($Profile -eq 'mtp') {
    # MTP reserves extra runtime memory, so the KV pool stays explicit.
    $arguments += @('--spec', 'mtp', '--draft-tokens', '3', '--lm-head-draft')
}

Write-Host "serving $Model on http://$($BindHost):$Port (profile=$Profile, context=$MaxContext)"
& $Exe @arguments
exit $LASTEXITCODE
