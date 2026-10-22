#!/bin/bash
# smolagent-lora-lab · Web 微调工作台启动脚本 (Linux / macOS)
cd "$(dirname "$0")"
echo "============================================================"
echo "  smolagent-lora-lab · Web 微调工作台"
echo "  访问地址: http://localhost:7860 (服务器重启会自动接上仍在跑的任务)"
echo "============================================================"
if command -v python3 >/dev/null 2>&1; then
    python3 webui/app.py
elif command -v python >/dev/null 2>&1; then
    python webui/app.py
else
    echo "[错误] 未找到 Python，请先安装 Python 3.10+"
    exit 1
fi
