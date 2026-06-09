param(
    [string]$EnvFile = ".env"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $repoRoot $EnvFile

if (-not (Test-Path -LiteralPath $envPath)) {
    throw "Environment file not found: $envPath"
}

function Set-DotEnvValue {
    param([string]$Line)

    $trimmed = $Line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
        return
    }

    $match = [regex]::Match($trimmed, "^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
    if (-not $match.Success) {
        return
    }

    $name = $match.Groups[1].Value
    $value = $match.Groups[2].Value.Trim()

    if (
        ($value.StartsWith('"') -and $value.EndsWith('"')) -or
        ($value.StartsWith("'") -and $value.EndsWith("'"))
    ) {
        $value = $value.Substring(1, $value.Length - 2)
    }

    [Environment]::SetEnvironmentVariable($name, $value, "Process")
}

Get-Content -LiteralPath $envPath | ForEach-Object {
    Set-DotEnvValue $_
}

if (-not $env:HF_TOKEN) {
    throw "HF_TOKEN is missing in $envPath. Add HF_TOKEN=hf_your_token before starting MAX."
}

if (-not $env:MAX_SERVE_API_TYPES) {
    $env:MAX_SERVE_API_TYPES = '["responses"]'
}

$baseUrl = if ($env:MAX_IMAGE_BASE_URL) { $env:MAX_IMAGE_BASE_URL } elseif ($env:MAX_BASE_URL) { $env:MAX_BASE_URL } else { "http://localhost:8010" }
$model = if ($env:MAX_IMAGE_MODEL) { $env:MAX_IMAGE_MODEL } elseif ($env:NOTES_IMAGE_MODEL) { $env:NOTES_IMAGE_MODEL } else { "black-forest-labs/FLUX.2-dev" }
$devices = if ($env:MAX_IMAGE_DEVICES) { $env:MAX_IMAGE_DEVICES } else { "gpu:0" }

$uri = [Uri]$baseUrl
$port = if ($uri.Port -gt 0) { $uri.Port } else { 8010 }

Write-Host "Starting MAX image server"
Write-Host "  model: $model"
Write-Host "  port:  $port"
Write-Host "  url:   $($uri.Scheme)://$($uri.Host):$port/v1/responses"

if (-not (Get-Command max -ErrorAction SilentlyContinue)) {
    throw @"
The 'max' CLI is not installed or not available on PATH.

Modular MAX is not officially supported directly on Windows. Use WSL Ubuntu for MAX:
  1. Open Ubuntu/WSL.
  2. cd /mnt/d/Edva/eddva_ai_service
  3. Install Modular MAX there.
  4. Run: ./scripts/start_max_images.sh

Your Windows AI service can still call MAX at http://localhost:$port.
"@
}

max serve --model $model --devices $devices --port $port
