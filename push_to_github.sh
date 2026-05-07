#!/bin/bash
# 快速推送到GitHub的脚本

echo "🚀 LFI项目GitHub设置脚本"
echo "================================"
echo ""

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 获取GitHub用户名
echo "请输入你的GitHub用户名:"
read github_username

if [ -z "$github_username" ]; then
    echo "❌ 用户名不能为空"
    exit 1
fi

echo ""
echo "📝 你输入的用户名: $github_username"
echo "仓库地址将是: https://github.com/$github_username/lfi-imaging-project"
echo ""
echo "请确认:"
echo "1. 你已经在GitHub上创建了名为 'lfi-imaging-project' 的仓库"
echo "2. 仓库是空的 (没有README, .gitignore等)"
echo ""
echo "确认继续? (y/n)"
read confirm

if [ "$confirm" != "y" ]; then
    echo "❌ 已取消"
    exit 0
fi

echo ""
echo "🔗 添加远程仓库..."
git remote add origin "https://github.com/$github_username/lfi-imaging-project.git" 2>/dev/null || \
git remote set-url origin "https://github.com/$github_username/lfi-imaging-project.git"

echo ""
echo "✅ 远程仓库已添加/更新"
git remote -v

echo ""
echo "📤 推送所有分支到GitHub..."
echo ""

# 推送main分支
echo "推送 main 分支..."
git push -u origin main

# 推送dev分支
echo "推送 dev 分支..."
git push -u origin dev

# 推送成员分支
echo "推送 zeyu/part1-data 分支..."
git push -u origin zeyu/part1-data

echo "推送 haobo/part2-preprocess 分支..."
git push -u origin haobo/part2-preprocess

echo "推送 mingxuan/part3-reconstruction 分支..."
git push -u origin mingxuan/part3-reconstruction

echo "推送 vincent/part4-ui 分支..."
git push -u origin vincent/part4-ui

echo ""
echo "✅ 所有分支已推送完成!"
echo ""
echo "🎉 设置完成!"
echo ""
echo "下一步:"
echo "1. 在GitHub仓库页面添加团队成员为collaborators"
echo "2. 设置main分支保护规则 (Settings → Branches)"
echo "3. 告诉团队成员克隆仓库:"
echo "   git clone https://github.com/$github_username/lfi-imaging-project.git"
echo ""
echo "查看详细说明: cat GITHUB_SETUP.md"
