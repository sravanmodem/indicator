@echo off
REM ================================================================
REM Options Trading Indicator Server - Production Mode
REM No auto-reload, optimized for performance
REM ================================================================

echo ========================================
echo Options Trading Indicator Server
echo PRODUCTION MODE
echo ========================================
echo.

REM Set Python path (Python 3.12)
set PYTHON_PATH=C:\Users\saiki\AppData\Local\Programs\Python\Python312\python.exe

REM Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found at %PYTHON_PATH%
    echo Trying alternative path...
    set PYTHON_PATH=C:\Users\saiki\AppData\Local\Microsoft\WindowsApps\python.exe
)

if not exist "%PYTHON_PATH%" (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo Using Python: %PYTHON_PATH%
echo.

REM Change to project directory
cd /d "%~dp0"

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Start the server (production mode - no reload)
echo Starting server in PRODUCTION mode...
echo Server: http://localhost:8000
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

"%PYTHON_PATH%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

pause
