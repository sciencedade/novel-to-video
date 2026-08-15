@echo off
REM Novel-to-Video - build Windows exe with PyInstaller
REM Output: dist\Novel2Video.exe (GUI) and dist\Novel2Video-CLI.exe (CLI/wizard)
setlocal
cd /d "%~dp0"

echo [1/3] Installing build requirements...
python -m pip install --upgrade -r requirements-build.txt
if errorlevel 1 goto :err

echo [2/3] Building GUI exe ...
python -m PyInstaller --noconfirm --clean --onefile --windowed --name Novel2Video ^
  --add-data "workflow_templates;workflow_templates" ^
  --add-data "examples;examples" ^
  --add-data "config.yaml;." ^
  gui.py
if errorlevel 1 goto :err

echo [3/3] Building CLI/wizard exe ...
python -m PyInstaller --noconfirm --clean --onefile --console --name Novel2Video-CLI ^
  --add-data "workflow_templates;workflow_templates" ^
  --add-data "examples;examples" ^
  --add-data "config.yaml;." ^
  --hidden-import wizard ^
  main.py
if errorlevel 1 goto :err

echo.
echo Build complete: dist\Novel2Video.exe and dist\Novel2Video-CLI.exe
echo Usage:
echo   Novel2Video.exe                    GUI (recommended)
echo   Novel2Video-CLI.exe --wizard       first-run config wizard
echo   Novel2Video-CLI.exe --auto-run     CLI one-click generation
goto :eof

:err
echo.
echo Build failed. Please check Python 3.10+ and network connection.
pause
exit /b 1
