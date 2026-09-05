$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Out = Join-Path $Root "reproduced_output"

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Command $Arguments"
    }
}

if (Test-Path -LiteralPath $Out) {
    Remove-Item -LiteralPath $Out -Recurse -Force
}
New-Item -ItemType Directory -Path $Out | Out-Null

Invoke-Native python (Join-Path $Root "code\cooperative_adaptive_junction_simulator.py") `
    --outdir (Join-Path $Out "publication") `
    --precomputed-dir (Join-Path $Root "data")

Invoke-Native python (Join-Path $Root "code\BB_schottky_loadline_solver.py") `
    --outdir (Join-Path $Out "BB_loadline")

$ProfileWork = Join-Path $Out "BB_profile_work"
New-Item -ItemType Directory -Path $ProfileWork | Out-Null
Copy-Item -LiteralPath (Join-Path $Root "code\cooperative_adaptive_junction_simulator.py") -Destination $ProfileWork
Copy-Item -LiteralPath (Join-Path $Root "code\BB_schottky_loadline_solver.py") -Destination $ProfileWork
Copy-Item -LiteralPath (Join-Path $Root "code\BB_profile_validation.py") -Destination $ProfileWork

Push-Location $ProfileWork
try {
    Invoke-Native python "BB_profile_validation.py"
}
finally {
    Pop-Location
}

Write-Host "Reproduction complete: $Out"
