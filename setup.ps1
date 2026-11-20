$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$PythonArgs = $null

Write-Host "Checking local development prerequisites..." -ForegroundColor Cyan

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py) was not found. Install Python 3.13 from https://www.python.org/downloads/windows/ and enable the Python Launcher."
}

foreach ($Version in @("3.13", "3.12")) {
    & py "-$Version" -c "import sys; print(sys.version)" *> $null
    if ($LASTEXITCODE -eq 0) {
        $PythonArgs = @("-$Version")
        Write-Host "Using Python $Version."
        break
    }
}
if ($null -eq $PythonArgs) {
    throw "Python 3.13 or 3.12 was not found. Install one of these versions, then run .\setup.ps1 again."
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "Node.js was not found. Install the current LTS release from https://nodejs.org/."
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Reinstall Node.js with npm and ensure it is on PATH."
}

Write-Host "Node: $(node --version)"
Write-Host "npm:  $(& npm.cmd --version)"

$BackendPath = Join-Path $ProjectRoot "backend"
$FrontendPath = Join-Path $ProjectRoot "frontend"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    & py @PythonArgs -m venv (Join-Path $ProjectRoot ".venv")
}
if ($LASTEXITCODE -ne 0) {
    throw "Could not create the Python virtual environment."
}

Write-Host "Installing CPU-compatible Python dependencies (the PyTorch wheel is large)..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
& $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r (Join-Path $BackendPath "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }

Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
Push-Location $FrontendPath
try {
    & npm.cmd install
    if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
}
finally {
    Pop-Location
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Start the backend and frontend in two separate PowerShell terminals; see README.md."
