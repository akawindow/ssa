param(
    [ValidateSet('smoke','pilot','tests')][string]$Mode='smoke',
    [string]$Python='',
    [string]$Out='',
    [switch]$FixedOnly
)
$ErrorActionPreference='Stop'
$experimentRoot=Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $localPython=Join-Path $experimentRoot '.venv/Scripts/python.exe'
    $availablePython='D:/cjx/paperstudy/eegdenoise/.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $localPython) { $Python=$localPython }
    elseif (Test-Path -LiteralPath $availablePython) { $Python=$availablePython }
    else { $Python='python' }
}
Push-Location -LiteralPath $experimentRoot
try {
    if ($Mode -eq 'tests') { & $Python -X utf8 -m unittest discover -s tests -v }
    else {
        if (-not $Out) { $Out='runs/'+$Mode+'_'+(Get-Date -Format 'yyyyMMdd_HHmmss') }
        $experimentArgs=@('-X','utf8','-m','ssa_task_srd','--config',"configs/$Mode.json",'--out',$Out)
        if ($FixedOnly) { $experimentArgs+='--fixed-only' }
        & $Python @experimentArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "Experiment exited with code $LASTEXITCODE" }
}
finally { Pop-Location }
