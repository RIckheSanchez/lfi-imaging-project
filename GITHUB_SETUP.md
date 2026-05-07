# GitHub 仓库设置指南

## 当前状态

✅ 本地Git仓库已初始化
✅ 所有分支已创建
✅ 项目结构已完成
✅ 初始代码已提交

**分支列表:**
- `main` - 主分支 (稳定版本)
- `dev` - 开发集成分支
- `zeyu/part1-data` - Zeyu的开发分支
- `haobo/part2-preprocess` - Haobo的开发分支
- `mingxuan/part3-reconstruction` - Mingxuan的开发分支
- `vincent/part4-ui` - Vincent的开发分支

---

## 步骤1: 在GitHub上创建仓库

### 方式A: 通过GitHub网页创建 (推荐)

1. 打开 https://github.com/new
2. 填写仓库信息:
   - **Repository name**: `lfi-imaging-project`
   - **Description**: `LFI Imaging using Laser Feedback Interferometry - UQ Final Year Project`
   - **Visibility**: 选择 Private (推荐) 或 Public
   - ⚠️ **不要勾选** "Add a README file"
   - ⚠️ **不要勾选** "Add .gitignore"
   - ⚠️ **不要勾选** "Choose a license"
3. 点击 "Create repository"

### 方式B: 使用GitHub CLI (如果已安装)

```bash
# 需要先安装: brew install gh
gh repo create lfi-imaging-project --private --source=. --remote=origin --push
```

---

## 步骤2: 连接本地仓库到GitHub

### 在你的Mac上执行:

```bash
# 1. 进入项目目录
cd /path/to/lfi-imaging-project

# 2. 添加GitHub远程仓库
# 替换 YOUR_USERNAME 为你的GitHub用户名
git remote add origin https://github.com/YOUR_USERNAME/lfi-imaging-project.git

# 3. 验证远程仓库
git remote -v

# 应该看到:
# origin  https://github.com/YOUR_USERNAME/lfi-imaging-project.git (fetch)
# origin  https://github.com/YOUR_USERNAME/lfi-imaging-project.git (push)
```

---

## 步骤3: 推送所有分支到GitHub

```bash
# 推送main分支
git push -u origin main

# 推送dev分支
git push -u origin dev

# 推送所有成员的开发分支
git push -u origin zeyu/part1-data
git push -u origin haobo/part2-preprocess
git push -u origin mingxuan/part3-reconstruction
git push -u origin vincent/part4-ui
```

**一键推送所有分支:**
```bash
git push --all origin
```

---

## 步骤4: 设置分支保护 (可选但推荐)

在GitHub网页上:

1. 进入仓库 → Settings → Branches
2. 添加分支保护规则:
   - **Branch name pattern**: `main`
   - 勾选:
     - ✅ Require a pull request before merging
     - ✅ Require approvals (至少1个reviewer)
   - 点击 "Create"

这样可以防止直接推送到main分支，必须通过Pull Request。

---

## 步骤5: 添加团队成员

1. 进入仓库 → Settings → Collaborators
2. 点击 "Add people"
3. 添加团队成员的GitHub账号:
   - Zeyu Wang
   - Yu Haobo
   - Tang Mingxuan
4. 他们会收到邀请邮件

---

## 团队成员克隆仓库

团队成员需要:

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/lfi-imaging-project.git
cd lfi-imaging-project

# 切换到自己的分支
# Zeyu:
git checkout zeyu/part1-data

# Haobo:
git checkout haobo/part2-preprocess

# Mingxuan:
git checkout mingxuan/part3-reconstruction

# Vincent:
git checkout vincent/part4-ui
```

---

## 日常工作流程

### 在自己的分支上工作

```bash
# 1. 确保在自己的分支
git checkout your-branch-name

# 2. 开始工作前，拉取最新代码
git pull origin your-branch-name

# 3. 做修改...

# 4. 查看修改
git status

# 5. 添加修改
git add .

# 6. 提交
git commit -m "描述你的修改"

# 7. 推送到GitHub
git push origin your-branch-name
```

### 合并到dev分支 (测试集成)

```bash
# 1. 在GitHub上创建Pull Request
#    从 your-branch → dev

# 2. 团队成员review代码

# 3. 合并PR

# 4. 本地更新dev分支
git checkout dev
git pull origin dev
```

### 合并到main分支 (发布)

```bash
# 只有在dev测试通过后，才合并到main
# 从 dev → main
# 需要至少一个团队成员approve
```

---

## 常见问题

### Q1: 忘记在哪个分支?
```bash
git branch  # 查看所有本地分支，*号表示当前分支
```

### Q2: 想看其他人的最新代码?
```bash
git fetch origin
git checkout zeyu/part1-data  # 切换到Zeyu的分支查看
```

### Q3: 不小心在main上做了修改?
```bash
# 不要提交！立即切换到自己的分支
git checkout your-branch-name
# 你的修改会跟着过去
```

### Q4: 代码冲突了怎么办?
```bash
# 发生冲突时
git status  # 查看哪些文件冲突了
# 手动编辑冲突文件，解决<<<<<<和>>>>>>标记的部分
git add .
git commit -m "解决冲突"
git push
```

### Q5: 想撤销最后一次提交?
```bash
# 撤销提交但保留修改
git reset --soft HEAD~1

# 完全撤销提交和修改（危险！）
git reset --hard HEAD~1
```

---

## Git基本概念速查

```
工作区 (Working Directory)
   ↓  git add
暂存区 (Staging Area)
   ↓  git commit
本地仓库 (Local Repository)
   ↓  git push
远程仓库 (Remote Repository - GitHub)
```

**常用命令:**
- `git status` - 查看当前状态
- `git log` - 查看提交历史
- `git diff` - 查看修改内容
- `git branch` - 查看/创建分支
- `git checkout` - 切换分支
- `git pull` - 拉取远程代码
- `git push` - 推送到远程

---

## 推荐的Git GUI工具 (Mac)

如果不习惯命令行，可以用:
- **GitHub Desktop** (免费) - https://desktop.github.com
- **SourceTree** (免费) - https://www.sourcetreeapp.com
- **GitKraken** (免费版) - https://www.gitkraken.com
- **VS Code** (内置Git支持)

---

## 检查清单

设置完成后确认:
- [ ] GitHub仓库已创建
- [ ] 本地仓库已连接到GitHub (`git remote -v`)
- [ ] 所有分支已推送 (`git branch -r`)
- [ ] 团队成员已添加为collaborators
- [ ] main分支已设置保护规则
- [ ] 每个人都能克隆仓库

---

## 需要帮助?

- Git官方文档: https://git-scm.com/doc
- GitHub文档: https://docs.github.com
- 或者问Vincent! 😊
