# Aggregate the same three component weights used by the frozen shape diagnostic.
# JSON numeric identifiers are explicitly converted back to integers for indexing.
$rows = @()
foreach ($suite in @('two_core', 'core', 'full', 'pit_core')) {
    $data = Get-Content -Encoding UTF8 -LiteralPath (Join-Path $PSScriptRoot ($suite + '_tight_full_curvature.json')) -Raw | ConvertFrom-Json
    $original = @{}; $observed = @{}; $corrected = @{}; $odd = @{}; $even = @{}
    foreach ($component in $data.report.components) {
        $weight = $component.weight / $data.report.checked_mass
        foreach ($direction in $component.directions) {
            $i = [int]$direction.index
            $value = $direction.values | Where-Object radius -eq 1
            $original[$i] += $weight * $value.gn_defect
            $observed[$i] += $weight * $value.observed_defect
            $odd[$i] += $weight * [math]::Abs($value.odd_remainder)
            $even[$i] += $weight * [math]::Abs($value.even_remainder)
        }
        if (-not $component.corrected.positive) { throw 'Cannot aggregate an indefinite correction' }
        foreach ($direction in $component.corrected.directions) {
            $i = [int]$direction.index
            $value = $direction.values | Where-Object radius -eq 1
            $corrected[$i] += $weight * $value.observed_defect
        }
    }
    $directions = @($original.Keys | Sort-Object | ForEach-Object {
        [pscustomobject]@{index = $_; original = $original[$_]; observed = $observed[$_]; corrected = $corrected[$_]; odd = $odd[$_]; even = $even[$_]}
    })
    $rows += [pscustomobject]@{
        suite = $suite; seed = $data.case.seed; checked_mass = $data.report.checked_mass
        dimension = $data.report.components[0].dimension
        original_defect = ($original.Values | Measure-Object -Maximum).Maximum
        observed_same_scale = ($observed.Values | Measure-Object -Maximum).Maximum
        corrected_scale_defect = ($corrected.Values | Measure-Object -Maximum).Maximum
        directions = $directions
        volume_corrections = @($data.report.components.log_volume_correction)
        hessian_step_changes = @($data.report.components | ForEach-Object { $_.checks[-1].change })
        hessian_asymmetry = @($data.report.components | ForEach-Object { $_.checks[-1].asymmetry })
    }
}
[IO.File]::WriteAllText((Join-Path $PSScriptRoot 'summary.json'), ($rows | ConvertTo-Json -Depth 8))
$rows | Select-Object suite, checked_mass, original_defect, observed_same_scale, corrected_scale_defect | ConvertTo-Json
