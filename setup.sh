#!/usr/bin/env bash
# 小说转视频 - Linux/macOS 一键安装并启动配置向导
set -e
cd "$(dirname "$0")"

echo "[1/4] 创建虚拟环境 venv..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

echo "[2/4] 激活虚拟环境..."
# shellcheck disable=SC1091
source venv/bin/activate

echo "[3/4] 安装依赖..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] 启动首次运行配置向导..."
python wizard.py
