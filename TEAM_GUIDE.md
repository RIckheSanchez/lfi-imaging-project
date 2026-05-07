# 团队成员快速上手指南

## 给团队成员的简短说明

### 第一次使用 (只需做一次)

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/lfi-imaging-project.git
cd lfi-imaging-project

# 2. 切换到你的分支
git checkout your-branch-name

# 你的分支名:
# - Zeyu: zeyu/part1-data
# - Haobo: haobo/part2-preprocess
# - Mingxuan: mingxuan/part3-reconstruction
# - Vincent: vincent/part4-ui
```

---

### 日常工作流程 (每天)

```bash
# 早上开始工作
git pull                    # 拉取最新代码

# 做修改...

# 晚上提交
git add .                   # 添加所有修改
git commit -m "今天做了什么"  # 提交
git push                    # 推送到GitHub
```

**就这么简单！**

---

### 常用命令

```bash
git status              # 查看当前状态
git log                 # 查看提交历史
git branch              # 查看当前在哪个分支 (*号)
```

---

### 如果出错了

**不小心在main分支上修改了:**
```bash
git checkout your-branch-name  # 直接切到自己的分支
```

**想撤销刚才的修改:**
```bash
git checkout .         # 撤销所有未提交的修改
```

**代码冲突了:**
```bash
# 找Vincent帮忙 😊
```

---

### 分支说明

- **main** - 稳定版本，不要直接推送！
- **dev** - 测试集成，通过Pull Request合并
- **你的分支** - 随便改，你的地盘

---

### 需要帮助?

- 查看详细文档: `cat GITHUB_SETUP.md`
- 问Vincent
- 或者Google: "git 怎么..."
