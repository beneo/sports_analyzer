#!/bin/bash

echo "🚀 简单安装 Sports Analyzer"

# 升级 pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .

echo "✅ 安装完成！"