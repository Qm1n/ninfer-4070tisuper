# Starts the NInfer OpenAI-compatible server and opens the bundled browser chat UI.
#
#   .\start-chat-ui.ps1                 # default 8080 / 16384 context / MTP
#   .\start-chat-ui.ps1 -Port 8081 -MaxContext 32768 -Profile plain
#
# The server runs in its own window (close it, or press Ctrl+C there, to stop the model).
# The page itself is served from a local static server so the browser sees a normal HTTP origin.

param(
    [int]$Port = 8080,
    [int]$UiPort = 8090,
    [int]$MaxContext = 16384,
    [ValidateSet('plain', 'mtp')][string]$Profile = 'mtp',
    [string]$Model,
    [string]$Exe,
    [string]$Python,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$serveScript = Join-Path $PSScriptRoot 'run-ninfer-serve.ps1'
$page = Join-Path $repo 'webui\index.html'
if (-not (Test-Path -LiteralPath $serveScript)) { throw "missing $serveScript" }
if (-not (Test-Path -LiteralPath $page)) { throw "missing $page" }

$serveArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $serveScript, '-Port', $Port,
                    '-MaxContext', $MaxContext, '-Profile', $Profile, '-Cors')
if ($Model) { $serveArguments += @('-Model', $Model) }
if ($Exe) { $serveArguments += @('-Exe', $Exe) }

Write-Host "starting the model server on port $Port (a separate window shows its log)"
Start-Process -FilePath 'powershell.exe' -ArgumentList $serveArguments | Out-Null

$health = "http://127.0.0.1:$Port/health"
$ready = $false
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    Start-Sleep -Seconds 2
    try {
        $response = Invoke-WebRequest -Uri $health -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    if ($attempt % 5 -eq 4) { Write-Host 'still loading the model...' }
}
if (-not $ready) {
    Write-Warning 'the server did not become ready; check its window for details'
    exit 1
}
Write-Host 'model server ready'

if (-not $Python) {
    foreach ($candidate in @((Join-Path $repo '..\python312\python.exe'), (Join-Path $repo '..\..\python312\python.exe'))) {
        if (Test-Path -LiteralPath $candidate) { $Python = (Resolve-Path $candidate).Path; break }
    }
}

$url = (New-Object System.Uri($page)).AbsoluteUri
if ($Python) {
    # A real HTTP origin avoids the file:// restrictions some browsers apply to fetch().
    Start-Process -FilePath $Python -ArgumentList @('-m', 'http.server', $UiPort, '--bind', '127.0.0.1', '--directory', (Join-Path $repo 'webui')) -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 2
    $url = "http://127.0.0.1:$UiPort/"
}

Write-Host "chat UI: $url"
Write-Host 'stop the model by closing the server window (or Ctrl+C there)'
if (-not $NoBrowser) { Start-Process $url }
