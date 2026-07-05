# Load environment variables from .env file
$env_vars = @{}
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            $env_vars[$matches[1]] = $matches[2]
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
    Write-Host "✓ Environment variables loaded from .env" -ForegroundColor Green
} else {
    Write-Host "⚠ No .env file found. Make sure to set DAGSHUB_USER_TOKEN and related variables." -ForegroundColor Yellow
}

Write-Host "Starting Streamlit dashboard..."
python -m streamlit run frontend/app.py
