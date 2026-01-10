@echo off
REM ================================================================
REM Setup Development Environment
REM ================================================================

echo ========================================
echo Environment Setup
echo ========================================
echo.

set PYTHON_PATH=C:\Users\saiki\AppData\Local\Programs\Python\Python312\python.exe

if not exist "%PYTHON_PATH%" (
    set PYTHON_PATH=C:\Users\saiki\AppData\Local\Microsoft\WindowsApps\python.exe
)

echo Using Python: %PYTHON_PATH%
echo.

REM Change to project directory
cd /d "%~dp0"

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    "%PYTHON_PATH%" -m venv venv
    echo [OK] Virtual environment created
    echo.
) else (
    echo [OK] Virtual environment already exists
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
if exist "requirements.txt" (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
    echo [OK] Dependencies installed
) else (
    echo WARNING: requirements.txt not found
    echo Installing core dependencies manually...
    pip install fastapi uvicorn kiteconnect pandas numpy loguru python-dotenv aiosqlite alpinejs-python
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Configure your .env file with Zerodha credentials
echo 2. Run: run_server.bat
echo 3. Open: http://localhost:8000/dashboard
echo.
pause
