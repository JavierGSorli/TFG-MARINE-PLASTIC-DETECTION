$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$PythonCmd = "conda run -n marida python"

Write-Host "Relanzando pipeline sin recalcular RF..." -ForegroundColor Cyan

Invoke-Expression "$PythonCmd .\02_predict_unet.py --overwrite --device cpu"
Invoke-Expression "$PythonCmd .\04_predict_resnet.py --device cpu"
Invoke-Expression "$PythonCmd .\05_predict_indices.py"
Invoke-Expression "$PythonCmd .\06_unify_predictions.py"
Invoke-Expression "$PythonCmd .\07_build_xgboost_dataset.py"
Invoke-Expression "$PythonCmd .\08_train_xgboost.py"
Invoke-Expression "$PythonCmd .\09_evaluate.py"
Invoke-Expression "$PythonCmd .\10_error_analysis.py"

Write-Host "Pipeline relanzado." -ForegroundColor Green
