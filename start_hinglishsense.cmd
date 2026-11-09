@echo off
REM Start HinglishSense from Windows Command Prompt (or double-click this file).
REM Starts the Spark service (E1/E2) and the API + website on http://127.0.0.1:8000
REM Press Ctrl+C in this window to stop.
cd /d "%~dp0"
set "GITBASH=C:\Program Files\Git\bin\bash.exe"
if not exist "%GITBASH%" set "GITBASH=C:\Program Files (x86)\Git\bin\bash.exe"
if not exist "%GITBASH%" (
  echo Git Bash not found. Install Git for Windows: https://git-scm.com/download/win
  pause
  exit /b 1
)
echo Starting HinglishSense... open http://127.0.0.1:8000 when it says "API + frontend"
"%GITBASH%" scripts/start_local.sh
