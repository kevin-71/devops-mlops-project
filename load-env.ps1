# Loads variables from .env into the current PowerShell session.
# Usage: run this from the project root before `python -m backend.train`.
#   .\load-env.ps1

if (-not (Test-Path .env)) {
    Write-Error "No .env file found. Copy .env.example to .env and fill in your values first."
    exit 1
}

Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}

Write-Host "Environment variables loaded from .env"