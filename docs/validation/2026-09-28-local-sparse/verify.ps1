$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
function Assert-Hash($taskPath, $taskExpected) {
    if ((Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskExpected) {
        throw "Hash mismatch: $taskPath"
    }
}
$taskBaseline = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot '../2026-09-28-curvature/source-manifest.json') | ConvertFrom-Json
foreach ($taskEntry in $taskBaseline) { Assert-Hash (Join-Path $taskRoot $taskEntry.path) $taskEntry.sha256 }
foreach ($taskManifest in @('source-manifest.json','final-source-manifest.json')) {
    $taskSources = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot $taskManifest) | ConvertFrom-Json
    foreach ($taskEntry in $taskSources) {
        Assert-Hash (Join-Path $PSScriptRoot $taskEntry.archive) $taskEntry.sha256
        if ($taskManifest -eq 'final-source-manifest.json') { Assert-Hash (Join-Path $taskRoot $taskEntry.path) $taskEntry.sha256 }
    }
}
$taskInputs = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'input-manifest.json') | ConvertFrom-Json
foreach ($taskEntry in $taskInputs) { Assert-Hash (Join-Path $taskRoot $taskEntry.path) $taskEntry.sha256 }
$taskTests = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'regression_final.json') | ConvertFrom-Json
if ($taskTests.fails -ne 0 -or $taskTests.errors -ne 0) { throw 'Regression failures' }
foreach ($taskEntry in $taskTests.hashes) { Assert-Hash (Join-Path $taskRoot $taskEntry.path) $taskEntry.sha256 }
foreach ($taskName in @('late_two_mapped','late_two_affine','late_two_mapped_final','late_two_affine_final','full_expanding_final','core_local_final','two_core_coupled_final')) {
    $taskResult = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot ($taskName + '.json')) | ConvertFrom-Json
    if (-not $taskResult.graph_unchanged -or -not $taskResult.source_unchanged) { throw "Ownership/source failure: $taskName" }
    if ($taskResult.completed_fits -gt $taskResult.settings.max_nodes) { throw "Fit budget exceeded: $taskName" }
    if ($taskResult.committed -or $taskResult.consistent) { throw "Unexpected acceptance claim: $taskName" }
    $taskArchive = if ($taskName.EndsWith('_final')) { 'sources-final/' } else { 'sources/' }
    foreach ($taskEntry in $taskResult.sources) { Assert-Hash (Join-Path $PSScriptRoot ($taskArchive + $taskEntry.path)) $taskEntry.sha256 }
}
$taskOnline = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'online_late_two_final.json') | ConvertFrom-Json
if (-not $taskOnline.anchor_unchanged -or -not $taskOnline.full_history -or $taskOnline.completed_fits -gt 257) { throw 'Online ownership/history/budget failure' }
if ($taskOnline.committed -or $taskOnline.consistent) { throw 'Unexpected online acceptance claim' }
if ($taskOnline.outcome.status -eq 'candidate') {
    foreach ($taskCheck in $taskOnline.outcome.checks.PSObject.Properties) {
        if (-not $taskCheck.Value) { throw "Online agreement check failed: $($taskCheck.Name)" }
    }
}
$taskCommon = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'common_offgrid.json') | ConvertFrom-Json
if (-not $taskCommon.graph_unchanged -or $taskCommon.missing.Count -ne 0) { throw 'Common comparison is incomplete' }
$taskArtifacts = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'artifact-manifest.json') | ConvertFrom-Json
foreach ($taskEntry in $taskArtifacts) { Assert-Hash (Join-Path $PSScriptRoot $taskEntry.path) $taskEntry.sha256 }
"Verified $($taskBaseline.Count) unchanged baseline files, both source archives, $($taskArtifacts.Count) artifacts and $($taskTests.passes) regression assertions."
