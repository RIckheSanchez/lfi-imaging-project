# Week 11 展示快速参考卡

## ⚡ 快速启动（3步）

```bash
# 1. 进入文件夹
cd ~/Downloads

# 2. 确保三个文件在一起
ls -la
# 必须看到：
# - lfi_viewer_final.py
# - loader.py
# - data_types.py

# 3. 运行
bash run_final.sh
# 或
streamlit run lfi_viewer_final.py
```

---

## 📸 必须截图的5个画面

### 1️⃣ 代码集成
打开 `lfi_viewer_final.py`，截图第5-8行：
```python
from loader import load_single_scan
from data_types import LFIRawData
```

### 2️⃣ 文件结构
终端显示：
```bash
$ ls -la
lfi_viewer_final.py  ← UI层
loader.py           ← Part 1
data_types.py       ← 共享格式
```

### 3️⃣ 模块检测成功
应用左侧栏显示：
```
✅ 检测到 loader.py 和 data_types.py
```

### 4️⃣ 数据摘要
点击展开后：
```
LFIRawData Summary
  target_type : Chip
  amplitude   : shape=(70, 1400, 400)
  z_indices   : 530 → 599 (70 slices)
```

### 5️⃣ 真实数据可视化
2D切片页面，显示：
- Z Index: 530
- 真实的芯片扫描图像

---

## 💬 关键演示话术

### 打开应用时
> "我现在运行应用。可以看到应用自动检测到了 loader.py 和 data_types.py，这两个模块由组员 Zeyu Wang 开发。"

### 上传文件时
> "我上传一个真实的HDF5文件。应用调用 load_single_scan() 函数，这是 loader.py 提供的标准接口。"

### 显示数据摘要时
> "数据摘要显示：70层扫描，每层1400×400像素，z层编号从530到599。这是真实的芯片扫描数据，不是模拟数据。"

### 展示可视化时
> "所有可视化都基于真实数据。左侧参数可以实时调节，看到立即生效。这证明UI层与数据层集成成功。"

### 强调模块化时
> "这个项目展示了模块化开发：
> - Part 1 提供数据接口
> - Part 4 调用接口显示
> - 通过标准的 LFIRawData 类通信
> - 各模块独立开发，无缝集成"

---

## 🎯 回答评委问题的准备

### Q: "你的代码能真正运行吗？"
**A:** "可以，现场演示。（打开应用，上传文件，展示功能）"

### Q: "你如何与组员协作？"
**A:** "我们定义了标准的 data_types.py，Zeyu 实现 loader，我实现 UI，通过这个共享接口对接。"

### Q: "如果组员的模块还没完成怎么办？"
**A:** "我的应用有容错机制。如果检测不到 loader.py，会自动降级到模拟数据模式，这样我可以独立开发和测试。"

### Q: "下一步计划是什么？"
**A:** "Week 12 集成 Part 2 和 Part 3 的模块，Week 13 开展 SUS 可用性评估，收集用户反馈。"

### Q: "进度是多少？"
**A:** "设计阶段100%完成，实现阶段约60%，总体进度60%。考虑到项目包含设计、实现、评估三个阶段，这是合理的进度。"

---

## 🔧 如果出现问题

### 问题1: 应用启动失败
```bash
# 重新安装依赖
pip3 install streamlit numpy plotly Pillow h5py --break-system-packages

# 检查Python版本
python3 --version  # 需要 3.8+
```

### 问题2: 找不到模块
```bash
# 确认文件在同一目录
pwd
ls -la

# 应该在同一个文件夹看到三个文件
```

### 问题3: HDF5文件加载失败
```
# 查看错误信息
# loader.py 会打印详细的错误提示
# 确认HDF5文件格式符合预期
```

---

## 📊 数据对接验证清单

确认以下内容都能正常工作：

- [ ] `from loader import load_single_scan` 导入成功
- [ ] `from data_types import LFIRawData` 导入成功
- [ ] `load_single_scan(filepath)` 返回 LFIRawData 对象
- [ ] `data.amplitude.shape` 返回 (n_slices, H, W)
- [ ] `data.z_indices` 是整数列表
- [ ] `data.get_slice(z_index)` 返回字典
- [ ] 应用显示 "✅ 检测到模块"
- [ ] 上传HDF5文件能成功加载
- [ ] 数据摘要正确显示
- [ ] 可视化显示真实数据

---

## ⏱️ 时间分配（15分钟演讲）

- **背景介绍** (2分钟) - 为什么需要这个工具
- **设计展示** (3分钟) - Figma原型5页
- **技术架构** (2分钟) - 四层架构 + 模块对接
- **实际演示** (5分钟) - 运行应用 + 真实数据
- **进度总结** (2分钟) - 已完成 + 下一步
- **Q&A** (5分钟) - 回答问题

---

**重要提醒：**
- ✅ 演示前先运行一遍，确保流畅
- ✅ 准备好HDF5文件（chip1.hdf5）
- ✅ 截图保存到PPT里
- ✅ 代码片段放大字体显示
- ✅ 提前打开终端和代码编辑器

---

**祝展示顺利！🎉**
