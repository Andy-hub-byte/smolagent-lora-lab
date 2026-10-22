@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================================
echo   smolagent-lora-lab · Web 微调工作台
echo ============================================================
where python >nul 2>nul
if %errorlevel%==0 (
    python webui\app.py
) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
        py webui\app.py
    ) else (
        echo [错误] 未找到 Python，请先安装 Python 3.10+ 并加入 PATH
        pause
    )
)
