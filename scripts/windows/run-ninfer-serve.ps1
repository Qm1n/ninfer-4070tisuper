# Starts the NInfer OpenAI-compatible server with the CUDA runtime on PATH.
#
#   .\run-ninfer-serve.ps1 -Model C:\models\qwen3_8_27b_gguf.ninfer
#   .\run-ninfer-serve.ps1 -Model ... -Port 8080 -MaxContext 32768 -Profile plain
#
# Endpoints: /v1/chat/completions, /v1/responses, /v1/messages (Anthropic), /health.
# Ctrl+C stops the server.

param(
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$Port = 8080,
    [int]$MaxContext = 16384,
    [ValidateSet('plain', 'mtp')][string]$Profile = 'mtp',
    [string]$BindHost = '127.0.0.1',
    [string]$ApiKey,
    [ValidateSet('bf16', 'int8', 'i4', 'i4-g64')][string]$KvDtype = 'int8',
    [switch]$NoThinking,
    [string]$Exe
)

$ErrorActionPreference = 'Stop'

if (-not $Exe) {
    $Exe = Join-Path $PSScriptRoot '..\..\build\apps\ninfer-serve.exe'
}
if (-not (Test-Path -LiteralPath $Exe)) { throw "ninfer-serve.exe was not found: $Exe" }
if (-not (Test-Path -LiteralPath $Model)) { throw "model artifact was not found: $Model" }

# cudart64_12.dll lives in the toolkit bin directory and is the only runtime dependency.
$cudaBin = Join-Path $env:CUDA_PATH 'bin'
if (-not (Test-Path -LiteralPath (Join-Path $cudaBin 'cudart64_12.dll'))) {
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
    '--kv-dtype', $KvDtype
)
if ($ApiKey) { $arguments += @('--api-key', $ApiKey) }
if ($NoThinking) { $arguments += '--no-thinking' }

if ($Profile -eq 'mtp') {
    # MTP reserves extra runtime memory, so the KV pool stays explicit.
    $arguments += @('--spec', 'mtp', '--draft-tokens', '3', '--lm-head-draft')
}

Write-Host "serving $Model on http://$($BindHost):$Port (profile=$Profile, context=$MaxContext)"
& $Exe @arguments
exit $LASTEXITCODE
