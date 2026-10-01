param([string]$src, [string]$pdf)
if (Test-Path $pdf) { Remove-Item $pdf -Force }
$job = Start-Job -ScriptBlock {
  param($s, $p)
  $w = New-Object -ComObject Word.Application
  $w.Visible = $false; $w.DisplayAlerts = 0
  try { $d = $w.Documents.Open($s, $false, $true); $d.Repaginate(); "pages: " + $d.ComputeStatistics(2); $d.ExportAsFixedFormat($p, 17); $d.Close($false) }
  catch { "ERR: " + $_.Exception.Message } finally { $w.Quit() }
} -ArgumentList $src, $pdf
$done = Wait-Job $job -Timeout 120
if ($done) { Receive-Job $job } else { "TIMEOUT"; Stop-Job $job; Get-Process WINWORD -ErrorAction SilentlyContinue | Stop-Process -Force }
