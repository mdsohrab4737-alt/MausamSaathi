@echo off
setlocal

cd /d C:\MausamSaathiSIH26\MAUSAMSAATHI_DEMO

echo ============================================================
echo MAUSAMSAATHI - CURRENT DAILY REFRESH
echo ============================================================
echo.

backend\.venv\Scripts\python.exe backend\refresh_current_bilaspur.py

set "RC=%ERRORLEVEL%"

echo.
echo ============================================================
echo REFRESH EXIT CODE: %RC%
echo ============================================================

exit /b %RC%
