@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 优先使用 diulian 环境自带的 Python / pythonw，找不到则回退到系统
set "PY=D:\anaconda\envs\diulian\python.exe"
set "PYW=D:\anaconda\envs\diulian\pythonw.exe"
if not exist "%PY%" set "PY=python"
if not exist "%PYW%" set "PYW=pythonw"

REM 先用带控制台的 python 做依赖预检（失败时在本窗口显示错误并暂停）
"%PY%" -c "import sys; sys.path.insert(0, 'QT'); import qt" >nul 2>&1
if errorlevel 1 (
    echo.
    echo 依赖检查失败，请手动运行查看错误： python QT\qt.py
    pause >nul
    exit /b 1
)

REM 依赖正常：用无控制台的 pythonw 启动界面，本终端自动关闭
start "" "%PYW%" "%CD%\QT\qt.py"
exit
