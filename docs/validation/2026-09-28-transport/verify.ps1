$ErrorActionPreference='Stop'
$taskRoot=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$taskKnown=@{}
foreach($taskManifest in @('benchmark-source-manifest.json','reporting-fix-source-manifest.json')) {
    foreach($taskEntry in (Get-Content -LiteralPath (Join-Path $PSScriptRoot $taskManifest) -Raw -Encoding UTF8|ConvertFrom-Json)) {
        $taskHash=(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $PSScriptRoot $taskEntry.preserved)).Hash.ToLowerInvariant()
        if($taskHash -ne $taskEntry.sha256){throw "Archived source mismatch: $($taskEntry.path)"}
        $taskKnown[$taskEntry.path+'|'+$taskEntry.sha256]=$true
    }
}
$taskChecked=0
foreach($taskCase in @('core','two_core','late_two','full','pit_core')) {
    $taskPair=@()
    foreach($taskMethod in @('affine','triangular')) {
        $taskName=$taskCase+'_'+$taskMethod+'.json'
        $taskR=Get-Content -LiteralPath (Join-Path $PSScriptRoot $taskName) -Raw -Encoding UTF8|ConvertFrom-Json
        if($taskR.backend -ne 'd31877a10f94' -or $taskR.physics -ne 'c2c7bbf8c0ec'){throw "Target fingerprint mismatch: $taskName"}
        if(-not $taskR.graph_unchanged -or $taskR.committed -or $taskR.consistent -or $taskR.domain_validated){throw "Unexpected qualification/ownership flag: $taskName"}
        if($taskR.total_conditional_fits -ne $taskR.completed_nodes+$taskR.completed_pilots -or $taskR.total_conditional_fits -gt $taskR.settings.max_nodes){throw "Fit accounting mismatch: $taskName"}
        if($taskR.metadata.case.seed -notin @(22001,22002,22003,22004)){throw "Unexpected case: $taskName"}
        foreach($taskSource in $taskR.sources) {
            if(-not $taskKnown.ContainsKey($taskSource.path+'|'+$taskSource.sha256)){throw "Unpreserved report source: $taskName $($taskSource.path)"}
        }
        $taskPair+=($taskR.settings|ConvertTo-Json -Depth 15 -Compress)
        $taskChecked++
    }
    if($taskPair[0] -cne $taskPair[1]){throw "Unmatched settings: $taskCase"}
}
$taskBaseline=Get-Content -LiteralPath (Join-Path $PSScriptRoot '../2026-09-28-curvature/source-manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json
foreach($taskEntry in $taskBaseline) {
    if((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $taskRoot $taskEntry.path)).Hash.ToLowerInvariant() -ne $taskEntry.sha256){throw "Production/calibration source changed: $($taskEntry.path)"}
}
$taskTests=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'final_tests.json') -Raw -Encoding UTF8|ConvertFrom-Json
if($taskTests.fails -ne 0 -or $taskTests.errors -ne 0 -or $taskTests.broken -ne 0){throw 'Final tests did not all pass'}
foreach($taskEntry in $taskTests.file_hashes) {
    if((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $taskRoot ('estimator/test/'+$taskEntry.file))).Hash.ToLowerInvariant() -ne $taskEntry.sha256){throw "Test source changed: $($taskEntry.file)"}
}
foreach($taskEntry in (Get-Content -LiteralPath (Join-Path $PSScriptRoot 'source-manifest.json') -Raw -Encoding UTF8|ConvertFrom-Json)) {
    if((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $taskRoot $taskEntry.path)).Hash.ToLowerInvariant() -ne $taskEntry.sha256){throw "Final source mismatch: $($taskEntry.path)"}
}
[pscustomobject]@{matched_reports=$taskChecked;unchanged_baseline_files=$taskBaseline.Count;final_assertions=$taskTests.passes}|ConvertTo-Json
