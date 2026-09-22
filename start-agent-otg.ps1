<#!
Starts Agent OTG's local services without requiring separate terminal windows.
Run from Explorer or PowerShell: .\start-agent-otg.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSCommandPath
$backendDir = Join-Path $projectRoot 'backend'
$frontendDir = Join-Path $projectRoot 'frontend'

function Test-AgentPort {
    param([int]$Port)
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $result = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        $connected = $result.AsyncWaitHandle.WaitOne(400)
        if ($connected) { $client.EndConnect($result) }
        $client.Dispose()
        return $connected
    } catch { return $false }
}

function Start-AgentProcess {
    param([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory)
    Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory -WindowStyle Hidden | Out-Null
}

if (-not (Test-AgentPort 11434)) {
    $ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $ollama) {
        $candidate = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe'
        if (Test-Path -LiteralPath $candidate) { $ollama = $candidate }
    }
    if (-not $ollama) { throw 'Ollama was not found. Install Ollama, then run this launcher again.' }
    Start-AgentProcess -FilePath $ollama -Arguments @('serve') -WorkingDirectory $backendDir
}

if (-not (Test-AgentPort 8000)) {
    $venvPython = Join-Path $backendDir '.venv\Scripts\python.exe'
    $python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { (Get-Command python -ErrorAction Stop).Source }
    Start-AgentProcess -FilePath $python -Arguments @('-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000') -WorkingDirectory $backendDir
}

if (-not (Test-AgentPort 5173)) {
    $npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
    if (-not $npm) { $npm = (Get-Command npm -ErrorAction Stop).Source }
    Start-AgentProcess -FilePath $npm -Arguments @('run', 'dev', '--', '--host', '0.0.0.0') -WorkingDirectory $frontendDir
}

$deadline = (Get-Date).AddSeconds(25)
while ((Get-Date) -lt $deadline -and -not (Test-AgentPort 5173)) { Start-Sleep -Milliseconds 250 }
if (-not (Test-AgentPort 5173)) { throw 'The frontend did not start. Check the hidden process logs or run npm run dev in frontend once.' }

Start-Process 'http://127.0.0.1:5173'
