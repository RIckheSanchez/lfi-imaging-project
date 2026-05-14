# LFI Imaging Viewer - 最终集成版使用指南

## 🎉 完全集成组员模块！

这个版本已经完全集成了：
- ✅ **Zeyu Wang 的 loader.py** - Part 1: 数据导入模块
- ✅ **Zeyu Wang 的 data_types.py** - 标准数据格式定义
- ✅ **He Zhenyu 的 UI 层** - Part 4: Streamlit界面

---

## 📦 文件清单

你现在应该有这些文件：

```
你的文件夹/
├── lfi_viewer_final.py      ← 最终集成版应用（⭐ 推荐使用）
├── loader.py                ← Zeyu Wang 的模块
├── data_types.py            ← Zeyu Wang 的模块
├── requirements.txt         ← Python依赖
├── lfi_viewer_enhanced.py   ← 增强版（备用）
├── lfi_viewer_app.py        ← 基础版（备用）
└── README.md                ← 通用说明
```

**重要：** 这3个文件必须在同一个文件夹：
- `lfi_viewer_final.py`
- `loader.py`
- `data_types.py`

---

## 🚀 运行方法

### 第一步：安装依赖

```bash
# 打开终端，进入文件夹
cd ~/Downloads  # 或你下载文件的位置

# 安装依赖
pip3 install -r requirements.txt --break-system-packages
```

### 第二步：运行应用

```bash
streamlit run lfi_viewer_final.py
```

### 第三步：浏览器自动打开

访问：**http://localhost:8501**

---

## ✨ 功能说明

### 🎯 真实数据模式（使用 loader.py）

当 `loader.py` 和 `data_types.py` 在同一目录时：

1. **应用自动检测** → 显示 "✅ 检测到 loader.py 和 data_types.py"
2. **上传HDF5文件** → 点击文件上传按钮
3. **点击"📥 加载真实数据"** 
4. **loader.py 自动运行：**
   ```python
   # 应用内部调用
   from loader import load_single_scan
   data = load_single_scan(filepath)
   ```
5. **显示数据摘要：**
   ```
   LFIRawData Summary
     target_type : Chip
     file_path   : /tmp/chip1.hdf5
     amplitude   : shape=(70, 1400, 400)
     phase       : shape=(70, 1400, 400)
     z_indices   : 530 → 599 (70 slices)
   ```

### 🎲 模拟数据模式（备用）

如果 `loader.py` 或 `data_types.py` 缺失：
- 应用自动切换到模拟数据模式
- 显示提示：将两个文件放在同一目录
- 可以继续使用，但数据是模拟生成的

---

## 📊 数据结构对接

### Zeyu Wang 的 LFIRawData 类

```python
@dataclass
class LFIRawData:
    amplitude:   np.ndarray  # shape (n_slices, height, width)
    phase:       np.ndarray  # shape (n_slices, height, width)
    z_indices:   list        # [530, 531, ..., 599]
    target_type: str         # 'Chip', 'Leaf', etc.
    file_path:   str         # HDF5文件路径
    metadata:    dict        # 额外信息
```

### UI 层如何使用

```python
# 1. 加载数据
data = load_single_scan("chip1.hdf5")

# 2. 获取切片
slice_data = data.get_slice(530)  # 获取 z=530 的切片
amplitude_2d = slice_data['amplitude']
phase_2d = slice_data['phase']

# 3. 获取信息
print(data.n_slices)    # 70
print(data.height)      # 1400
print(data.width)       # 400
print(data.z_indices)   # [530, 531, ..., 599]
```

---

## 🎯 Week 11 展示流程

### **推荐演示脚本：**

#### 1. 展示代码集成（1分钟）

打开 `lfi_viewer_final.py`，找到这段代码展示：

```python
# Import real modules from teammates
try:
    from loader import load_single_scan
    from data_types import LFIRawData
    REAL_MODULES_AVAILABLE = True
except ImportError:
    REAL_MODULES_AVAILABLE = False
```

**说明：**
> "我的UI层直接集成了 Zeyu Wang 开发的 loader.py 模块。应用启动时会自动检测这两个模块是否存在，如果存在就使用真实的数据加载器，否则降级到模拟数据模式。"

#### 2. 展示文件结构（30秒）

在终端显示：
```bash
ls -la
# 显示三个文件在同一目录
```

**说明：**
> "这三个文件必须在同一目录：我的UI应用、loader模块、data_types模块。这展示了模块化开发的成果。"

#### 3. 运行应用（30秒）

```bash
streamlit run lfi_viewer_final.py
```

**说明：**
> "应用启动后，会自动导入 loader.py..."

#### 4. 上传真实HDF5文件（2分钟）

在浏览器中：
1. 选择 chip1.hdf5
2. 点击"加载真实数据"
3. 展示数据摘要弹窗

**说明：**
> "现在我上传一个真实的HDF5文件。应用调用 load_single_scan() 函数读取数据。可以看到数据摘要显示了70层扫描，每层1400×400像素。这是真实的芯片扫描数据。"

#### 5. 展示可视化（2分钟）

切换各个Tab：
- 2D Slice：展示真实图像
- 3D：展示体积重建
- Waveform：展示信号曲线

**说明：**
> "所有可视化功能都基于真实数据。可以看到 Z Index 从530到599，这是 loader.py 从HDF5文件中读取的真实z层编号。"

---

## 🔧 技术亮点

### 1. 模块解耦设计

```
loader.py (Part 1)          ← Zeyu Wang 负责
    ↓ 提供 load_single_scan()
data_types.py              ← 共享数据格式
    ↓ 定义 LFIRawData 类
lfi_viewer_final.py (Part 4) ← He Zhenyu 负责
    ↓ 调用上述模块
可视化界面
```

### 2. 容错机制

```python
try:
    from loader import load_single_scan
    REAL_MODULES_AVAILABLE = True
except ImportError:
    REAL_MODULES_AVAILABLE = False
    # 降级到模拟数据模式
```

### 3. 标准接口调用

```python
# 使用真实 loader
data = load_single_scan(tmp_path)

# 使用标准方法
slice_data = data.get_slice(current_z)
amplitude = slice_data['amplitude']
```

---

## 📸 必须截图的内容

### 1. 代码编辑器截图
打开 `lfi_viewer_final.py`，截图显示：
```python
from loader import load_single_scan
from data_types import LFIRawData
```

### 2. 终端截图
显示三个文件在同一目录：
```bash
$ ls -la
lfi_viewer_final.py
loader.py
data_types.py
```

### 3. 应用启动截图
显示：
```
✅ 检测到 loader.py 和 data_types.py
```

### 4. 数据摘要截图
点击展开后显示：
```
LFIRawData Summary
  target_type : Chip
  amplitude   : shape=(70, 1400, 400)
  ...
```

### 5. 可视化效果截图
- 2D切片显示真实数据
- Z Index 显示 530-599
- 3D重建效果

---

## 💬 PPT中可以说的话

### Slide: "模块集成"

> "我的UI层完全集成了组员的模块。左侧代码展示了导入语句：
> ```python
> from loader import load_single_scan
> from data_types import LFIRawData
> ```
> 
> 这两个模块由 Zeyu Wang 开发，定义了数据导入的标准接口。我的UI层只需要调用这些接口，就能加载和显示真实数据。"

### Slide: "实际运行演示"

> "这是应用的实际运行截图。可以看到左上角显示'检测到 loader.py'，证明模块集成成功。
> 
> 当我上传 chip1.hdf5 文件后，应用调用 load_single_scan() 函数，返回一个 LFIRawData 对象，包含70层扫描数据。
> 
> 所有可视化功能都基于这个真实数据对象。"

### Slide: "技术架构实现"

> "这个架构图展示了四个Part的数据流：
> 
> - Part 1 (Zeyu): loader.py 提供 load_single_scan()
> - Part 2 (Mingxuan): 信号处理（接口已定义）
> - Part 3 (Haobo): 图像重建（接口已定义）
> - Part 4 (我): UI层调用所有模块
> 
> 目前 Part 1 已经完全集成，Part 2 和 Part 3 的接口已经协调好，下周会完成集成。"

---

## ⚠️ 常见问题

### Q: 运行时提示找不到 loader.py？
**A:** 确保三个文件在同一目录：
```bash
ls -la
# 必须看到：
# lfi_viewer_final.py
# loader.py
# data_types.py
```

### Q: 上传HDF5后报错？
**A:** 检查HDF5文件格式。loader.py 期望的结构：
```
file.hdf5
├── /amplitude/530
├── /amplitude/531
...
├── /phase/530
├── /phase/531
...
```

### Q: 可以只用模拟数据吗？
**A:** 可以！如果缺少 loader.py，应用会自动切换到模拟数据模式。但 Week 11 展示时建议使用真实模块。

### Q: 如何验证集成成功？
**A:** 运行应用后，左侧栏应该显示：
```
✅ 检测到 loader.py 和 data_types.py
```

---

## 📊 进度总结

### 已完成 ✅
- [x] Figma原型设计（100%）
- [x] Streamlit UI框架（100%）
- [x] Part 1 模块集成（100%）
- [x] 真实HDF5文件支持（100%）
- [x] 2D/3D可视化（100%）
- [x] 实时参数调节（100%）
- [x] 基础导出功能（100%）

### 进行中 🔄
- [ ] Part 2/3 模块集成（Week 12）
- [ ] 性能优化（Week 12）
- [ ] SUS可用性评估（Week 13）

### 总体进度
**设计阶段：** 100% ✅  
**实现阶段：** 60% 🔄  
**评估阶段：** 0% ⏳  
**总体：** 约 60%

---

## 🎓 Week 11 核心信息

### 你可以展示的实际成果：

1. **完整的Figma原型** - 设计阶段完成
2. **真实运行的Streamlit应用** - 实现阶段进行中
3. **与组员模块的成功集成** - 协作证明
4. **真实HDF5数据加载和显示** - 功能验证

### 进度评估：

- **Progress (30%)**: ⭐⭐⭐⭐⭐ 有代码、有运行、有集成
- **Content (30%)**: ⭐⭐⭐⭐ 有设计、有实现、有计划
- **技术深度**: ⭐⭐⭐⭐ 真实模块集成、标准接口

---

**作者：** 何振宇 (49174058)  
**版本：** v1.0 Final (Fully Integrated)  
**日期：** 2026年5月14日  
**集成模块：** loader.py + data_types.py (Zeyu Wang)
