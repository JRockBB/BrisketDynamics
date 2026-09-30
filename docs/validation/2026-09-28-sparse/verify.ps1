$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
function Assert-Hash($taskPath, $taskExpected) {
    if ((Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskExpected) {
        throw "Hash mismatch: $taskPath"
    }
}
$taskBaseline = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot '../2026-09-28-curvature/source-manifest.json') | ConvertFrom-Json
foreach ($taskEntry in $taskBaseline) { Assert-Hash (Join-Path $taskRoot $taskEntry.path) $taskEntry.sha256 }
$taskSources = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'benchmark-source-manifest.json') | ConvertFrom-Json
foreach ($taskEntry in $taskSources) { Assert-Hash (Join-Path $PSScriptRoot $taskEntry.archive) $taskEntry.sha256 }
foreach ($taskVersion in @('degree64','online','final')) {
    $taskArchive = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot ($taskVersion + '-source-archive-manifest.json')) | ConvertFrom-Json
    foreach ($taskEntry in $taskArchive) { Assert-Hash (Join-Path $PSScriptRoot $taskEntry.archive) $taskEntry.sha256 }
}
$taskFinal = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'source-manifest.json') | ConvertFrom-Json
foreach ($taskEntry in $taskFinal) { Assert-Hash (Join-Path $taskRoot $taskEntry.path) $taskEntry.sha256 }
$taskArtifacts = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'artifact-manifest.json') | ConvertFrom-Json
foreach ($taskEntry in $taskArtifacts) { Assert-Hash (Join-Path $PSScriptRoot $taskEntry.path) $taskEntry.sha256 }
$taskTests = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'final_tests.json') | ConvertFrom-Json
if ($taskTests.fails -ne 0 -or $taskTests.errors -ne 0) { throw 'Regression failures' }
foreach ($taskEntry in $taskTests.hashes) { Assert-Hash (Join-Path $taskRoot $taskEntry.path) $taskEntry.sha256 }
foreach ($taskName in @('core_tensor','core_sparse','late_two_sparse','full_sparse','full_expanding','late_two_mapped_sparse','late_two_mapped_final')) {
    $taskResult = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot ($taskName + '.json')) | ConvertFrom-Json
    if (-not $taskResult.graph_unchanged -or -not $taskResult.source_unchanged) { throw "Ownership/source failure: $taskName" }
    if ($taskResult.completed_fits -gt $taskResult.settings.max_nodes) { throw "Fit budget exceeded: $taskName" }
    if ($taskResult.committed -or $taskResult.consistent) { throw "Unexpected acceptance claim: $taskName" }
}
$taskOnline = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'online_replay_final.json') | ConvertFrom-Json
if (-not $taskOnline.anchor_unchanged -or -not $taskOnline.noise_equal -or -not $taskOnline.full_history) { throw 'Online target ownership/history failure' }
foreach ($taskSide in @('offline','online')) {
    $taskDiagnostics = $taskOnline.$taskSide
    if (-not $taskDiagnostics.resolved -or -not $taskDiagnostics.shape_pass -or $taskDiagnostics.total_graph_solves -gt 129) { throw "Online replay checks failed: $taskSide" }
}
if ($taskOnline.committed -or $taskOnline.consistent) { throw 'Unexpected online acceptance claim' }
foreach ($taskName in @('core_verified_capacity','two_core_verified_capacity')) {
    $taskResult = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot ($taskName + '.json')) | ConvertFrom-Json
    if (-not $taskResult.graph_unchanged -or -not $taskResult.diagnostics.resolved -or -not $taskResult.diagnostics.shape_pass) { throw "Capacity check failed: $taskName" }
    if ($taskResult.diagnostics.graph_solves -gt $taskResult.settings.max_nodes) { throw "Capacity fit budget exceeded: $taskName" }
    foreach ($taskEntry in $taskResult.sources) { Assert-Hash (Join-Path $PSScriptRoot ('sources/final/' + $taskEntry.path)) $taskEntry.sha256 }
}
"Verified $($taskBaseline.Count) unchanged baseline files, $($taskFinal.Count) final sources, $($taskArtifacts.Count) artifacts and $($taskTests.passes) regression assertions."
