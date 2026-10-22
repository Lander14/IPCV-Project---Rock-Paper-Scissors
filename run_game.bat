@echo off
TITLE Launcher - Rock Paper Scissors AR
cd /d "%~dp0"

echo [1/4] Checking virtual environment (.venv_win)...
if not exist ".venv_win" (
    echo Creating virtual environment for Windows...
    python -m venv .venv_win
)

echo [2/4] Activating virtual environment...
call .venv_win\Scripts\activate.bat

echo [3/4] Checking dependencies from requirements.txt...
if exist "requirements.txt" (
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
) else (
    echo Warning: requirements.txt not found.
)

echo.
echo ===================================================
echo                CAMERA CONFIGURATION
echo ===================================================
echo Enter the camera index number (e.g., 0, 1, 2)
echo or press ENTER directly to use automatic detection (--cam-auto):
set /p CAM_IDX="Camera index: "

echo.
echo [4/4] Starting the game...
cd game

if "%CAM_IDX%"=="" (
    python play_camera.py --cam-auto
) else (
    python play_camera.py --camera %CAM_IDX%
)

echo.
echo The game has finished.
pause