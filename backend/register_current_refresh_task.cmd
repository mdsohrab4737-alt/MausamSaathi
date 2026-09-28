@echo off
setlocal

set "TASK_NAME=MausamSaathi Daily Current Refresh"
set "RUNNER=C:\MausamSaathiSIH26\MAUSAMSAATHI_DEMO\backend\run_current_refresh.cmd"

echo ============================================================
echo REGISTER MAUSAMSAATHI DAILY REFRESH
echo ============================================================
echo Task : %TASK_NAME%
echo Time : 06:00 local computer time
echo Runner: %RUNNER%
echo.

schtasks /Create /TN "%TASK_NAME%" /SC DAILY /ST 06:00 /TR "\"%RUNNER%\"" /F

if errorlevel 1 (
    echo.
    echo [FAIL] Could not register the scheduled task.
    exit /b 1
)

echo.
echo [PASS] Scheduled task registered.
echo.
echo To test immediately:
echo   schtasks /Run /TN "%TASK_NAME%"
echo.
echo To inspect:
echo   schtasks /Query /TN "%TASK_NAME%" /V /FO LIST
echo.

exit /b 0
