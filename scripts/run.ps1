param(
    [ValidateSet('smoke','pilot','tests','round02','java_setup','java_verify','java_round','java_pilot','collaborative','ssa_only','ssa_static','ssa_structure','task_alignment','scale_fair','mlp_primary','classification')][string]$Mode='smoke',
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
    elseif ($Mode -in @('round02','java_setup','java_verify','java_round','collaborative','ssa_only','ssa_static','ssa_structure','task_alignment','scale_fair','mlp_primary','classification')) {
        if ($FixedOnly) { throw 'Selected mode does not support -FixedOnly' }
        $entryScript=@{'round02'='run_round02.py';'java_setup'='setup_java.py';'java_verify'='verify_java_srd.py';'java_round'='run_java_round.py';'collaborative'='run_collaborative_round.py';'ssa_only'='run_ssa_only.py';'ssa_static'='diagnose_ssa_only.py';'ssa_structure'='diagnose_ssa_structure.py';'task_alignment'='run_task_alignment.py';'scale_fair'='run_scale_fair.py';'mlp_primary'='run_mlp_primary.py';'classification'='run_classification.py'}[$Mode]
        $roundArgs=@('-X','utf8',('scripts/'+$entryScript))
        if ($Mode -eq 'java_setup') { $roundArgs+='--fetch' }
        if ($Mode -in @('java_setup','java_verify','ssa_static','ssa_structure') -and -not $Out) { $Out='runs/'+$Mode+'_'+(Get-Date -Format 'yyyyMMdd_HHmmss') }
        if ($Out) { $roundArgs+=@('--out',$Out) }
        & $Python @roundArgs
    }
    else {
        if (-not $Out) { $Out='runs/'+$Mode+'_'+(Get-Date -Format 'yyyyMMdd_HHmmss') }
        $experimentArgs=@('-X','utf8','-m','ssa_task_srd','--config',"configs/$Mode.json",'--out',$Out)
        if ($FixedOnly) { $experimentArgs+='--fixed-only' }
        & $Python @experimentArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "Experiment exited with code $LASTEXITCODE" }
}
finally { Pop-Location }
