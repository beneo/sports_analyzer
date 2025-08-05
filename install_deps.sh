#!/bin/bash

echo "🚀 Sports Analyzer 依赖安装脚本"
echo "================================"

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ 检测到虚拟环境: $VIRTUAL_ENV"
else
    echo "⚠️  未检测到虚拟环境，建议先激活虚拟环境"
fi

# 升级 pip
echo ""
echo "📦 升级 pip 到最新版本..."
python -m pip install --upgrade pip

# 安装 requirements.txt
echo ""
echo "📋 安装 requirements.txt 中的依赖..."
pip install -r requirements.txt

# 安装项目包
echo ""
echo "🔧 安装 sports_analyzer 包..."
pip install -e .

# 安装额外的开发工具（可选）
echo ""
echo "🛠️  安装开发工具..."
pip install -e ".[dev,all]"

# 验证安装
echo ""
echo "🧪 验证安装结果..."
python -c "
try:
    import sports_analyzer
    print('✅ sports_analyzer 包安装成功')
except ImportError as e:
    print(f'❌ sports_analyzer 包安装失败: {e}')

# 检查关键依赖
key_deps = ['numpy', 'torch', 'transformers', 'opencv-python', 'supervision']
for dep in key_deps:
    try:
        if dep == 'opencv-python':
            import cv2
            print(f'✅ {dep} 可用')
        else:
            __import__(dep.replace('-', '_'))
            print(f'✅ {dep} 可用')
    except ImportError:
        print(f'❌ {dep} 不可用')
"

echo ""
echo "🎉 安装完成！"
echo ""
echo "📋 下一步:"
echo "  1. cd examples/soccer"
echo "  2. ./setup.sh  # 下载预训练模型"
echo "  3. python main.py --help  # 查看使用说明"