@echo off
REM Reset script for Windows: Clean checkpoints, logs, and outputs before training

echo ==========================================
echo Reset Training Environment (Windows)
echo ==========================================
echo.

REM Remove checkpoints
echo 1. Cleaning checkpoints...
if exist "checkpoints\" (
    del /Q "checkpoints\*.*" 2>nul
    echo   * checkpoints\ cleaned
) else (
    echo   - checkpoints\ does not exist
)

REM Remove logs
echo.
echo 2. Cleaning logs...
if exist "logs\" (
    del /Q "logs\*.*" 2>nul
    echo   * logs\ cleaned
) else (
    echo   - logs\ does not exist
)

REM Remove outputs
echo.
echo 3. Cleaning outputs...
if exist "outputs\" (
    del /Q /S "outputs\*.*" 2>nul
    echo   * outputs\ cleaned
) else (
    echo   - outputs\ does not exist
)

REM Remove tensorboard runs
echo.
echo 4. Cleaning tensorboard runs...
if exist "runs\" (
    rmdir /S /Q "runs" 2>nul
    mkdir "runs"
    echo   * runs\ cleaned
) else (
    echo   - runs\ does not exist
)

REM Remove Python cache
echo.
echo 5. Cleaning Python cache...
if exist "__pycache__\" (
    rmdir /S /Q "__pycache__" 2>nul
    echo   * __pycache__\ removed
) else (
    echo   - No __pycache__\ found
)

for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d" 2>nul
del /S /Q "*.pyc" 2>nul
echo   * All Python cache cleaned

echo.
echo ==========================================
echo * Reset completed!
echo ==========================================
echo.
echo You can now run:
echo   python train.py
echo   # or
echo   python train.py --demo
echo.
pause
