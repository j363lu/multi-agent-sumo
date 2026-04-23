@echo off
REM Setup script for MAPPO traffic control environment (Windows)

echo ==========================================
echo MAPPO Traffic Control - Environment Setup
echo ==========================================
echo.

REM Check if SUMO is installed
where sumo >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Warning: SUMO not found. Please install SUMO first:
    echo    Download from https://eclipse.dev/sumo/
    echo.
)

REM Check SUMO_HOME
if "%SUMO_HOME%"=="" (
    echo Warning: SUMO_HOME not set. Please set it:
    echo    set SUMO_HOME=C:\Program Files\SUMO
    echo.
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv MAPPO_venv

REM Activate virtual environment
echo Activating virtual environment...
call MAPPO_venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install PyTorch (CPU version)
echo Installing PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

REM Install other dependencies
echo Installing dependencies...
pip install numpy==1.24.3
pip install gymnasium==0.28.1
pip install pettingzoo==1.24.3
pip install sumo-rl==1.4.5
pip install pyyaml
pip install wandb
pip install tensorboard
pip install matplotlib
pip install seaborn

REM Install Tianshou
echo Installing Tianshou...
pip install tianshou==0.5.1

echo.
echo ==========================================
echo Installation Complete!
echo ==========================================
echo.
echo To activate the environment in the future, run:
echo   MAPPO_venv\Scripts\activate.bat
echo.
echo To test the installation, run:
echo   python scripts\train_mappo.py --debug
echo.
pause
