$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
$data = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'posterior_geometry.json') -Raw | ConvertFrom-Json
$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$chart.Width = 1440
$chart.Height = 820
$chart.BackColor = [System.Drawing.Color]::White
$title = $chart.Titles.Add('Two projections of the joint posterior')
$title.Font = New-Object System.Drawing.Font('Segoe UI',18)
$subtitle = $chart.Titles.Add('Selected development cook, seed 3005. Same observations and factor model. Finite-sample diagnostic, not a calibration result.')
$subtitle.Font = New-Object System.Drawing.Font('Segoe UI',10)
$legend = $chart.Legends.Add('Legend')
$legend.Docking = 'Bottom'
$legend.Alignment = 'Center'
$legend.Font = New-Object System.Drawing.Font('Segoe UI',11)
for ($panelIndex=0; $panelIndex -lt 2; $panelIndex++) {
    $panel = $data.panels[$panelIndex]
    $area = $chart.ChartAreas.Add("Area$panelIndex")
    $area.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition((2+49*$panelIndex),17,47,70)
    $area.AxisX.IsStartedFromZero = $false
    $area.AxisY.IsStartedFromZero = $false
    $area.AxisX.TitleFont = New-Object System.Drawing.Font('Segoe UI',11)
    $area.AxisY.TitleFont = New-Object System.Drawing.Font('Segoe UI',11)
    $area.AxisX.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    $area.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::Gainsboro
    if ($panelIndex -eq 0) {
        $area.AxisX.LabelStyle.Format = 'F2'
        $area.AxisY.LabelStyle.Format = 'F3'
        $area.AxisX.Title = 'Average water (belief coordinate)'
        $area.AxisY.Title = 'Surface minus core water (belief coordinate)'
    } else {
        $area.AxisX.LabelStyle.Format = 'F1'
        $area.AxisY.LabelStyle.Format = 'F0'
        $area.AxisX.Title = 'Probe depth (normal-score coordinate)'
        $area.AxisY.Title = 'Core temperature (K)'
    }
    $particles = $chart.Series.Add("Particles$panelIndex")
    $particles.ChartArea = $area.Name
    $particles.ChartType = 'Point'
    $particles.MarkerStyle = 'Circle'
    $particles.MarkerSize = 4
    $particles.Color = [System.Drawing.Color]::FromArgb(110,39,111,173)
    $particles.LegendText = 'Nonlinear posterior particles'
    $particles.IsVisibleInLegend = ($panelIndex -eq 0)
    for ($j=0; $j -lt $panel.x.Count; $j++) { [void]$particles.Points.AddXY([double]$panel.x[$j],[double]$panel.y[$j]) }
    $ellipse = $chart.Series.Add("Ellipse$panelIndex")
    $ellipse.ChartArea = $area.Name
    $ellipse.ChartType = 'Line'
    $ellipse.BorderWidth = 3
    $ellipse.Color = [System.Drawing.Color]::FromArgb(205,98,25)
    $ellipse.LegendText = 'Gaussian 95% ellipse'
    $ellipse.IsVisibleInLegend = ($panelIndex -eq 0)
    $l11 = [Math]::Sqrt([double]$panel.cxx)
    $l21 = [double]$panel.cxy/$l11
    $conditionalVariance = [double]$panel.cyy-$l21*$l21
    if ($conditionalVariance -lt 0.0) { throw 'Projection covariance is not positive definite' }
    $l22 = [Math]::Sqrt($conditionalVariance)
    $radius = [Math]::Sqrt(5.99146454710798)
    for ($j=0; $j -le 240; $j++) {
        $angle = 2*[Math]::PI*$j/240
        $a = $radius*[Math]::Cos($angle)
        $b = $radius*[Math]::Sin($angle)
        [void]$ellipse.Points.AddXY(($panel.gaussian_mean[0]+$l11*$a),($panel.gaussian_mean[1]+$l21*$a+$l22*$b))
    }
    $truth = $chart.Series.Add("Truth$panelIndex")
    $truth.ChartArea = $area.Name
    $truth.ChartType = 'Point'
    $truth.MarkerStyle = 'Cross'
    $truth.MarkerSize = 13
    $truth.Color = [System.Drawing.Color]::Black
    $truth.LegendText = 'Simulation truth'
    $truth.IsVisibleInLegend = ($panelIndex -eq 0)
    [void]$truth.Points.AddXY([double]$panel.truth[0],[double]$panel.truth[1])
}
$output = Join-Path $PSScriptRoot 'posterior_geometry.png'
$chart.SaveImage($output,[System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
$chart.Dispose()
Write-Output $output
