@echo off
REM ================================================================
REM Options Trading Indicator Server - Startup Script
REM Optimized API Architecture with Backend-First Design
REM ================================================================

echo ========================================
echo Options Trading Indicator Server
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
    echo Please install Python 3.12 or update the path in this script.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_PATH%
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo WARNING: No virtual environment found
    echo Consider creating one: python -m venv venv
    echo.
)

REM Display current configuration
echo ----------------------------------------
echo Configuration:
echo ----------------------------------------
echo Python: %PYTHON_PATH%
echo Working Directory: %CD%
echo Branch: optimize-api-architecture
echo.
echo Features:
echo   - Consolidated API endpoints (single call per page)
echo   - Intelligent caching (75-85%% API reduction)
echo   - 10-second auto-refresh during market hours
echo   - Zero frontend external API calls
echo ----------------------------------------
echo.

REM Check for required packages
echo Checking dependencies...
"%PYTHON_PATH%" -c "import fastapi, uvicorn, kiteconnect" 2>nul
if errorlevel 1 (
    echo.
    echo WARNING: Some packages may be missing
    echo Installing requirements...
    "%PYTHON_PATH%" -m pip install -r requirements.txt
    echo.
)

REM Start the server
echo Starting FastAPI server...
echo.
echo Server will start on: http://localhost:8000
echo Dashboard: http://localhost:8000/dashboard
echo API Monitoring: http://localhost:8000/api/v1/monitoring/api-calls
echo Cache Stats: http://localhost:8000/api/v1/cache/stats
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Run with uvicorn
"%PYTHON_PATH%" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
