param(
    [string]$Python = 'D:\cjx\paperstudy\eegdenoise\.venv\Scripts\python.exe',
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Name = 'seed_confirmation_initial'
)
$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path $PSScriptRoot -Parent
$confirmationSeeds = @(45, 46, 47)
$summaryFolder = Join-Path $experimentRoot "runs/${Name}_summary"
if (Test-Path -LiteralPath $summaryFolder) { throw 'Summary output exists; choose a new Name.' }
foreach ($trainingSeed in $confirmationSeeds) {
    if (Test-Path -LiteralPath (Join-Path $experimentRoot "runs/${Name}_seed${trainingSeed}")) {
        throw 'A seed output already exists; choose a new Name.'
    }
}
$started = [DateTime]::UtcNow
$workers = foreach ($trainingSeed in $confirmationSeeds) {
    $worker = Start-Process -FilePath $Python -WorkingDirectory $experimentRoot -WindowStyle Hidden -PassThru `
        -ArgumentList @('-X', 'utf8', 'scripts/run_seed_confirmation.py', '--seed', $trainingSeed,
            '--out', "runs/${Name}_seed${trainingSeed}") `
        -RedirectStandardOutput (Join-Path $experimentRoot "runs/${Name}_seed${trainingSeed}_console.log") `
        -RedirectStandardError (Join-Path $experimentRoot "runs/${Name}_seed${trainingSeed}_stderr.log")
    [PSCustomObject]@{ Seed = $trainingSeed; Process = $worker }
}
foreach ($worker in $workers) { $worker.Process.WaitForExit() }
foreach ($worker in $workers) {
    $completionFile = Join-Path $experimentRoot "runs/${Name}_seed$($worker.Seed)/completed.json"
    if ($worker.Process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $completionFile)) {
        throw "Seed $($worker.Seed) failed or incomplete; scores not calculated. Inspect preserved logs."
    }
}
Push-Location $experimentRoot
try {
    & $Python -X utf8 scripts/summarize_seed_confirmation.py --jobs `
        "runs/${Name}_seed45" "runs/${Name}_seed46" "runs/${Name}_seed47" --out "runs/${Name}_summary"
    if ($LASTEXITCODE -ne 0) { throw 'Replication scoring failed; preserved all outputs.' }
    [PSCustomObject]@{ StartedUTC = $started.ToString('o'); FinishedUTC = [DateTime]::UtcNow.ToString('o');
        WallSeconds = ([DateTime]::UtcNow-$started).TotalSeconds; ParallelSeeds = $confirmationSeeds } |
        ConvertTo-Json | Set-Content -LiteralPath "runs/${Name}_orchestration.json" -Encoding utf8
} finally { Pop-Location }
