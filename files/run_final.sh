#!/bin/bash

# LFI Viewer 最终集成版启动脚本
# 自动检测模块并运行

echo "🚀 LFI Imaging Viewer - 最终集成版"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查必需文件
echo "📋 检查文件..."

if [ ! -f "lfi_viewer_final.py" ]; then
    echo "❌ 找不到 lfi_viewer_final.py"
    exit 1
fi

if [ -f "loader.py" ] && [ -f "data_types.py" ]; then
    echo "✅ 检测到 loader.py 和 data_types.py"
    echo "   → 将使用真实数据加载模式"
else
    echo "⚠️  未检测到 loader.py 或 data_types.py"
    echo "   → 将使用模拟数据模式"
fi

echo ""

# 检查依赖
if ! command -v streamlit &> /dev/null; then
    echo "📦 首次运行，正在安装依赖..."
    pip3 install -r requirements.txt --break-system-packages
    echo "✅ 依赖安装完成！"
    echo ""
fi

# 运行应用
echo "🌐 启动应用..."
echo "💡 按 Ctrl+C 停止"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

streamlit run lfi_viewer_final.py

# 备用方案
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  启动失败，尝试备用方案..."
    python3 -m streamlit run lfi_viewer_final.py
fi
