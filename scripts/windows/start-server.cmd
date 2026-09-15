@echo off
rem Double-clickable launcher for the NInfer OpenAI-compatible server.
rem It reuses scripts\windows\run-ninfer-serve.ps1, which finds the built server and the
rem converted .ninfer artifact automatically when they sit in the standard workspace layout.

setlocal
set "SCRIPT=%~dp0run-ninfer-serve.ps1"

if not exist "%SCRIPT%" (
    echo run-ninfer-serve.ps1 was not found next to this launcher.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
    echo.
    echo ninfer-serve exited with code %CODE%.
    pause
)

endlocal & exit /b %CODE%
