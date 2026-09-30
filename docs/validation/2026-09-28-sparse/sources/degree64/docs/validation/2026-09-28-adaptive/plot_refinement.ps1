Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
$plotRoot=Split-Path -Parent $PSCommandPath
$chart=New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$chart.Width=1200; $chart.Height=720; $chart.BackColor=[Drawing.Color]::White
$area=New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea 'Resolution'
$area.AxisX.Title='Conditional nodes evaluated'; $area.AxisY.Title='Joint refinement indicator (log scale)'
$area.AxisX.Minimum=0; $area.AxisX.Maximum=260; $area.AxisX.Interval=50
$area.AxisY.IsLogarithmic=$true; $area.AxisY.Minimum=0.001; $area.AxisY.Maximum=2
$area.AxisX.MajorGrid.LineColor=[Drawing.Color]::Gainsboro; $area.AxisY.MajorGrid.LineColor=[Drawing.Color]::Gainsboro
$area.AxisX.TitleFont=New-Object Drawing.Font('Segoe UI',13); $area.AxisY.TitleFont=New-Object Drawing.Font('Segoe UI',13)
$chart.ChartAreas.Add($area)
$legend=New-Object System.Windows.Forms.DataVisualization.Charting.Legend
$legend.Docking='Bottom'; $legend.Font=New-Object Drawing.Font('Segoe UI',12); $chart.Legends.Add($legend)
$definitions=@(
    @{File='core_refinement.json';Label='Original conditional solves';Color=[Drawing.Color]::Firebrick},
    @{File='core_polished_gn.json';Label='Innovation score refinement';Color=[Drawing.Color]::SteelBlue}
)
foreach($definition in $definitions) {
    $data=ConvertFrom-Json -InputObject ([IO.File]::ReadAllText((Join-Path $plotRoot $definition.File)))
    $series=New-Object System.Windows.Forms.DataVisualization.Charting.Series $definition.Label
    $series.ChartType='Line'; $series.BorderWidth=3; $series.Color=$definition.Color
    foreach($entry in $data.diagnostics.history) {
        $value=[double]$entry.indicator
        if(-not [double]::IsNaN($value) -and -not [double]::IsInfinity($value) -and $value -gt 0) {
            $null=$series.Points.AddXY([double]$entry.nodes,$value)
        }
    }
    $chart.Series.Add($series)
}
$threshold=New-Object System.Windows.Forms.DataVisualization.Charting.Series 'Declared refinement threshold'
$threshold.ChartType='Line'; $threshold.BorderDashStyle='Dash'; $threshold.BorderWidth=2; $threshold.Color=[Drawing.Color]::DimGray
$null=$threshold.Points.AddXY(0.,.002); $null=$threshold.Points.AddXY(260.,.002); $chart.Series.Add($threshold)
$title=$chart.Titles.Add('Adaptive integration: frozen one-probe cook at 510 s')
$title.Font=New-Object Drawing.Font('Segoe UI',19)
$subtitle=$chart.Titles.Add('Same data and process-noise law. This is numerical resolution, not a coverage test.')
$subtitle.Font=New-Object Drawing.Font('Segoe UI',11)
$chart.SaveImage((Join-Path $plotRoot 'refinement.png'),[System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
$chart.Dispose()
