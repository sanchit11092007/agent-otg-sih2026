# Agent OTG — one-command demo launcher for SIH showcase
# Opens the React UI at http://127.0.0.1:5173 (backend starts automatically via Vite)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$VenvPython = Join-Path $Backend ".venv\Scripts\python.exe"

function Get-PythonCommand {
    foreach ($candidate in @($env:AGENT_OTG_PYTHON, "python", "py")) {
        if (-not $candidate) { continue }
        try {
            & $candidate --version *> $null
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch { }
    }
    throw "Python 3 was not found. Install Python 3.10+ and ensure it is on PATH."
}

function Test-PythonExecutable([string]$Path) {
    if (-not (Test-Path $Path)) { return $false }
    try {
        & $Path --version *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

Write-Host ''
Write-Host '=== Agent OTG Demo Launcher ===' -ForegroundColor Cyan

if (-not (Test-PythonExecutable $VenvPython)) {
    if (Test-Path (Join-Path $Backend ".venv")) {
        $BackupVenv = Join-Path $Backend (".venv.unusable-" + (Get-Date -Format "yyyyMMddHHmmss"))
        Write-Host "The existing virtual environment cannot run; preserving it as $BackupVenv" -ForegroundColor Yellow
        Move-Item -LiteralPath (Join-Path $Backend ".venv") -Destination $BackupVenv
    }
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    Set-Location $Backend
    $PythonCommand = Get-PythonCommand
    & $PythonCommand -m venv .venv
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
    Set-Location $Root
}

Write-Host "Checking Ollama (port 11434)..." -ForegroundColor Gray
$ollamaOk = Test-NetConnection -ComputerName 127.0.0.1 -Port 11434 -WarningAction SilentlyContinue
if (-not $ollamaOk.TcpTestSucceeded) {
    Write-Host "WARNING: Ollama is not running. Start it with: ollama serve" -ForegroundColor Red
    Write-Host "Then pull models: ollama pull qwen2.5:7b && ollama pull nomic-embed-text" -ForegroundColor Yellow
}

if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    Set-Location $Frontend
    npm install
    Set-Location $Root
}

Write-Host ''
Write-Host 'Starting frontend on all interfaces (0.0.0.0:5173)' -ForegroundColor Green
Write-Host "Local browser: http://127.0.0.1:5173" -ForegroundColor Green

try {
    $LocalIP = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } | Sort-Object -Property { if ($_.IPAddress -like "192.168.137.*") { 0 } elseif ($_.IPAddress -like "192.168.*") { 1 } else { 2 } } | Select-Object -First 1).IPAddress
    if ($LocalIP) {
        Write-Host ('Hotspot / Multi-Device URL: http://' + $LocalIP + ':5173') -ForegroundColor Cyan
        Write-Host '   (Secondary devices on same hotspot enter PIN to view live results)' -ForegroundColor Gray
        Write-Host ''
    }
} catch { }

Write-Host 'Backend API will auto-start on port 8000.' -ForegroundColor Green
Write-Host ''
Set-Location $Frontend
npm run dev
