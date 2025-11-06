# Stop MLflow UI (Windows PowerShell Script)
# Usage: .\scripts\stop_mlflow.ps1

Write-Host "Stopping MLflow processes..." -ForegroundColor Yellow

# Find and kill MLflow processes
$mlflowProcesses = Get-Process | Where-Object {
    $_.ProcessName -eq "python" -and
    $_.CommandLine -like "*mlflow*ui*"
}

if ($mlflowProcesses) {
    foreach ($process in $mlflowProcesses) {
        Write-Host "Killing process $($process.Id)..." -ForegroundColor Red
        Stop-Process -Id $process.Id -Force
    }
    Write-Host "MLflow stopped successfully!" -ForegroundColor Green
} else {
    Write-Host "No MLflow processes found." -ForegroundColor Yellow
}

# Also kill uvicorn processes (MLflow uses uvicorn)
$uvicornProcesses = Get-Process | Where-Object {
    $_.ProcessName -eq "python" -and
    $_.CommandLine -like "*uvicorn*"
}

if ($uvicornProcesses) {
    foreach ($process in $uvicornProcesses) {
        Write-Host "Killing uvicorn process $($process.Id)..." -ForegroundColor Red
        Stop-Process -Id $process.Id -Force
    }
}

Write-Host "Done!" -ForegroundColor Green
