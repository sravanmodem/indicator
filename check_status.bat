@echo off
REM ================================================================
REM Check Server Status and API Performance
REM ================================================================

echo ========================================
echo Server Status Check
echo ========================================
echo.

set PYTHON_PATH=C:\Users\saiki\AppData\Local\Programs\Python\Python312\python.exe

REM Check if server is running
echo Checking if server is running...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Server is NOT running
    echo Run: run_server.bat
    pause
    exit /b 1
)

echo [OK] Server is running
echo.

REM API Monitoring
echo ========================================
echo API Call Statistics (Last 24 hours)
echo ========================================
curl -s http://localhost:8000/api/v1/monitoring/api-calls?hours=24
echo.
echo.

REM Cache Statistics
echo ========================================
echo Cache Performance
echo ========================================
curl -s http://localhost:8000/api/v1/cache/stats
echo.
echo.

REM Rate Limit Check
echo ========================================
echo Rate Limit Usage
echo ========================================
curl -s http://localhost:8000/api/v1/monitoring/api-costs
echo.
echo.

echo ========================================
echo Status check complete
echo ========================================
pause
