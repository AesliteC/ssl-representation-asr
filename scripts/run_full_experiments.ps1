param(
    [string]$Device = "cuda",
    [int]$LimitUtterances = 0,
    [int]$LimitTrain = 0,
    [int]$LimitDev = 0,
    [string]$LogPath = "logs/full_experiments.log"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null

function Write-ExperimentLogLine {
    param(
        [string]$Message
    )

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    $stream = New-Object System.IO.FileStream($LogPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    $writer = New-Object System.IO.StreamWriter($stream, $utf8NoBom)
    try {
        $writer.WriteLine($Message)
        $writer.Flush()
    }
    finally {
        $writer.Dispose()
        $stream.Dispose()
    }
}

function Invoke-ExperimentStep {
    param(
        [string[]]$PythonArgs
    )

    $display = "conda run --no-capture-output -n py310 python " + ($PythonArgs -join " ")
    Write-Host "[run] $display"
    Write-ExperimentLogLine "[run] $display"
    & conda run --no-capture-output -n py310 python @PythonArgs 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Host $line
        Write-ExperimentLogLine $line
    }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        $message = "Command failed with exit code ${exitCode}: $display"
        Write-ExperimentLogLine $message
        throw $message
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
