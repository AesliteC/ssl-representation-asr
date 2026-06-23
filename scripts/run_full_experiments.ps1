param(
    [string]$Device = "cuda",
    [int]$LimitUtterances = 0,
    [int]$LimitTrain = 0,
    [int]$LimitDev = 0,
    [string]$LogPath = "logs/full_experiments.log"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null

function Invoke-ExperimentStep {
    param(
        [string[]]$PythonArgs
    )

    $display = "conda run -n py310 python " + ($PythonArgs -join " ")
    Write-Host "[run] $display"
    Add-Content -Path $LogPath -Value "[run] $display"
    & conda run -n py310 python @PythonArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $display"
    }
}

$prepareArgs = @("scripts/prepare_experiment_assets.py", "--device", $Device)
if ($LimitUtterances -gt 0) {
    $prepareArgs += @("--limit-utts", "$LimitUtterances")
}
Invoke-ExperimentStep -PythonArgs $prepareArgs

$trainArgs = @("scripts/run_experiments.py", "--config-dir", "configs/experiments", "--device", $Device, "--skip-existing")
if ($LimitTrain -gt 0) {
    $trainArgs += @("--limit-train", "$LimitTrain")
}
if ($LimitDev -gt 0) {
    $trainArgs += @("--limit-dev", "$LimitDev")
}
Invoke-ExperimentStep -PythonArgs $trainArgs

Invoke-ExperimentStep -PythonArgs @(
    "scripts/evaluate_experiments.py",
    "--config-dir",
    "configs/experiments",
    "--device",
    $Device,
    "--skip-missing"
)

Invoke-ExperimentStep -PythonArgs @(
    "scripts/summarize.py",
    "--outputs-dir",
    "outputs",
    "--output",
    "results/summary.csv"
)
