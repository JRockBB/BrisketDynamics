$ErrorActionPreference='Stop'
$taskRows=[Collections.Generic.List[object]]::new()
foreach($taskCase in @('core','two_core','late_two','full','pit_core')) {
    foreach($taskMethod in @('affine','triangular')) {
        $taskName=$taskCase+'_'+$taskMethod+'.json'
        $taskReport=Get-Content -LiteralPath (Join-Path $PSScriptRoot $taskName) -Raw -Encoding UTF8 | ConvertFrom-Json
        $taskDiag=$taskReport.outcome.diagnostics
        $taskComplete=@($taskDiag.history | Where-Object {$_.complete})
        $taskLast=if($taskComplete.Count){$taskComplete[-1]}elseif($taskReport.last_refinement.complete){$taskReport.last_refinement}else{$null}
        $taskRows.Add([pscustomobject]@{
            report=$taskName;case=$taskCase;method=$taskMethod;t=$taskReport.metadata.t
            dimensions=$taskReport.settings.axes.Count;status=$taskReport.outcome.status
            wall_seconds=$taskReport.wall;map_seconds=$taskReport.map_seconds
            pilots=$taskReport.completed_pilots;conditional_fits=$taskReport.completed_nodes
            total_fits=$taskReport.total_conditional_fits;budget=$taskReport.settings.max_nodes
            tolerance=$taskReport.settings.tolerance;resolved=($taskDiag.resolved -eq $true)
            last_complete_indicator=$taskLast.indicator
            last_complete_nodes=$taskLast.nodes
            shape_complete=($taskDiag.shape_complete -eq $true)
            shape_defect=$taskDiag.shape_defect;shape_pass=($taskDiag.shape_pass -eq $true)
            graph_unchanged=$taskReport.graph_unchanged;error=$taskReport.outcome.error
            consistent=$false;domain_validated=$false
        })
    }
}
$taskOnline=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'mapped_online_replay.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskTests=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'final_tests.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskSummary=[pscustomobject]@{
    backend='d31877a10f94';physics='c2c7bbf8c0ec';committed=$false;consistent=$false
    paired_runs=$taskRows.ToArray()
    online=[pscustomobject]@{wall_seconds=$taskOnline.wall;tv=$taskOnline.comparison.tv_bound_estimate
        comparison_integration_change=$taskOnline.comparison.integration_change
        graph_unchanged=$taskOnline.anchor_unchanged;noise_unchanged=$taskOnline.noise_equal
        map_dimension=$taskOnline.map_dimension;nonlinear_chart_reuse_qualified=$false}
    analytic_report='analytic_comparison.json';offgrid_report='late_two_offgrid.json'
    final_assertions=$taskTests.passes;tests_report='final_tests.json';cached_shape_report='late_two_cached_shape.json'
    benchmark_sources='benchmark-source-manifest.json';final_sources='source-manifest.json'
}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'summary.json'),(ConvertTo-Json -InputObject $taskSummary -Depth 15),[Text.UTF8Encoding]::new($false))
$taskRows | Select-Object case,method,wall_seconds,total_fits,resolved,last_complete_indicator,shape_pass | Format-Table -AutoSize
