param([string]$OutDir = "")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrEmpty($OutDir)) { $OutDir = Join-Path $Root "reproduced_output/full_suite" }
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "1" }
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "1" }
$env:MPLBACKEND = "Agg"
function Run-Python {
    param([string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}
Run-Python -Arguments @((Join-Path $Root "code/reproduce.py"), "--full-sweeps", "--diagnostics", "--outdir", (Join-Path $OutDir "publication"))
Run-Python -Arguments @((Join-Path $Root "code/BB_schottky_loadline_solver.py"), "--outdir", (Join-Path $OutDir "BB_loadline"))
Run-Python -Arguments @((Join-Path $Root "code/BB_profile_validation.py"), "--outdir", (Join-Path $OutDir "BB_profiles"))
Run-Python -Arguments @((Join-Path $Root "code/validate_convergence.py"), "--outdir", (Join-Path $OutDir "convergence"))
Run-Python -Arguments @("-m", "unittest", "discover", "-s", (Join-Path $Root "tests"), "-v")
Write-Host "Reproduction and validation completed: $OutDir"
