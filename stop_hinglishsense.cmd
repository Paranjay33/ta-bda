@echo off
REM Stop any running HinglishSense processes (Spark service + API).
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='java.exe' or Name='python.exe'\" | Where-Object { $_.CommandLine -match 'SparkInferenceServer|app.main:create_app' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host ('stopped ' + $_.ProcessId) }"
echo HinglishSense stopped.
