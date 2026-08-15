@echo off
REM 小说转视频 - Windows 一键安装并启动配置向导
setlocal
cd /d "%~dp0"

echo [1/4] 创建虚拟环境 venv...
if not exist venv (
    python -m venv venv
    if errorlevel 1 goto :err
)

echo [2/4] 激活虚拟环境...
call venv\Scripts\activate.bat

echo [3/4] 安装依赖...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :err

echo [4/4] 启动首次运行配置向导...
python wizard.py
goto :eof

:err
echo.
echo 安装失败，请检查 Python 3.10+ 是否已安装并加入 PATH。
pause
exit /b 1
