# Start MLflow UI in background (Windows PowerShell Script)
# Usage: .\scripts\start_mlflow.ps1

Write-Host "Starting MLflow UI..." -ForegroundColor Green

# Check if already running
$existing = Get-Process | Where-Object {
    $_.ProcessName -eq "python" -and
    $_.CommandLine -like "*mlflow*ui*"
}

if ($existing) {
    Write-Host "MLflow is already running!" -ForegroundColor Yellow
    Write-Host "Open: http://localhost:5000" -ForegroundColor Cyan
    exit
}

# Start MLflow in background (hidden window, no logs)
Write-Host "Launching MLflow on http://localhost:5000" -ForegroundColor Cyan
Write-Host "To stop: Run .\scripts\stop_mlflow.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Note: MLflow will run in a hidden background window." -ForegroundColor Gray
Write-Host "You can safely close this terminal." -ForegroundColor Gray

# Start in completely hidden background process
Start-Process -WindowStyle Hidden -FilePath "mlflow" -ArgumentList "ui", "--port", "5000"

Write-Host ""
Write-Host "MLflow started! Opening browser..." -ForegroundColor Green
Start-Sleep -Seconds 3
Start-Process "http://localhost:5000"
