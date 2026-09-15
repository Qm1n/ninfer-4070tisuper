@echo off
rem Double-clickable launcher: starts the NInfer server with CORS and opens the browser chat UI.
setlocal
set "SCRIPT=%~dp0start-chat-ui.ps1"
if not exist "%SCRIPT%" (
    echo start-chat-ui.ps1 was not found next to this launcher.
    pause
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" %*
if not "%ERRORLEVEL%"=="0" (
    echo.
    echo start-chat-ui exited with code %ERRORLEVEL%.
    pause
)
endlocal
