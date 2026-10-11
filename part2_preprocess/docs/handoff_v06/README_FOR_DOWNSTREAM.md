# THz LFI 预处理结果交接包 v0.6.1（给图像生成 / 可视化）

来自：Haobo Yu（预处理模块）  ·  生成日期：2026-10-11  ·  代码版本：分支 `v0.5-evaluation`，软件 v0.6.1

> 这一版的文件名是 `thz_lfi_handoff_v061_*.zip`。如果你之前收到过 `v05` 或 `v06` 开头的包，请丢弃，改用这一版。
> v05 的 `amplitude_corrected` 减掉了背景，会把大块物体的明暗抹平；v06 里 Leaf 相位用的是两倍尺度。

## 一分钟上手

```bash
pip install numpy matplotlib
python load_example.py            # 读文件、打印内容、存两张示例图
```

不需要安装我的代码，只要 numpy 就能读。

## 包里有什么

| 内容 | 数量 | 形状 | 说明 |
| --- | --- | --- | --- |
| `data/leaf_frameNNN_spatial.npz` | 91 | (1, 1200, 400) | 叶子，存储索引 0–90；其中 28–90 共 63 张相位可用 |
| `data/vegetable_frameN_spatial.npz` | 21 | (1, 400, 1300) | 蔬菜深度切片：frame 21–30 连续，40–140 每隔 10；相位保持 wrapped |
| `data/skin_frame40_spatial.npz` | 1 | (1, 400, 1300) | 皮肤样品，仅此一张；相位在信号区域内可用 |
| `vegetables_volume_21frames.npz` | 1 | (21, 400, 1300) | 21 层蔬菜叠成的体数据，带每层的毫米位置 |
| `data/*_metadata.json` | 113 | | 每个文件的来源、参数、相位状态 |
| `manifest.json` | 1 | | 全部文件的汇总 |
| `previews/*.png` | 8 | | 处理过程示例图 |

所有数组的轴顺序都是 `(n, H, W)`，dtype 为 float64（体数据文件为 float32）。

## 用哪个通道

| 通道 | 含义 | 你该怎么用 |
| --- | --- | --- |
| `amplitude_corrected` | 降噪并去掉扫描线偏移的幅度，**物体明暗保留** | **做图首选**。单位和原始幅度一样 |
| `amplitude_detail` | `amplitude_corrected` 再减去平滑背景 | 只用来看细节纹理（比如叶脉）。是相对图，会把比背景尺度大的物体抹平；无信号处为 NaN |
| `raw_amplitude` | 原始幅度 | 想自己从头处理时用 |
| `phase_corrected` | 解包裹后去掉背景和扫描线偏移的相位（弧度，相对值） | 仅当 `phase_correction_available` 为 True 时可用；无信号处为 NaN |
| `phase` | 进入空间校正前的相位 | 相位放行的图里是解包裹后的（无信号处为 NaN）；没放行的图里仍是 wrapped |
| `raw_phase` | 原始 wrapped 相位（弧度，[-π, π)） | 想自己解包裹时用 |
| `signal_mask` | 布尔数组，True 表示这里有足够信号（一整块连通区域） | 相位只在 True 的地方有意义 |
| `*_background`、`*_stripe`、`*_notch_removed` | 被去掉的成分 | 检查用 |
| `phase_correction_available` | 每帧一个布尔值 | **用相位前必须先看它** |
| `z_indices` | 帧编号 / 存储索引 | 不是毫米；毫米位置看体数据文件里的 `z_mm` |

恒等式（可用来自检）：
`amplitude_pre_spatial = amplitude_corrected + amplitude_stripe + amplitude_notch_removed`；
`amplitude_corrected = amplitude_detail + amplitude_background`（在 `signal_mask` 内）。

## 显示时请注意（否则图会"看不懂"）

1. **用灰度色图**，不要用红蓝色图显示幅度。
2. **像素不是正方形**。Skin 和 Vegetables 的整幅图宽高比是 1.965（不是 1300/400 = 3.25）；
   Leaf 要先转置（`.T`），再按宽高比 1.5 显示。`load_example.py` 里有现成写法。
3. **蔬菜不是每一层都清楚**。frame 21–30 离焦点最远，又暗又花；frame 120–140 信号最强（130 最强）。
   做展示图请用 120–140。

## 相位能不能用

| 数据集 | 能用吗 | 说明 |
| --- | --- | --- |
| Leaf | 63/91 张能（索引 28–90），仅在 `signal_mask` 内 | 相位按导师说的换算（码值 × π/8192），由我的模块解包裹。叶脉处信号太弱，是 NaN。前 28 张噪声大，没放行。**不要用 mat 文件里的 `L_P_U` / `L_P_U_M`**：它们用的是两倍尺度，而且有成块的错误 |
| Skin | 能，仅在 `signal_mask` 内 | 同样按码值 × π/8192 换算 |
| Vegetables | 不能 | 找不到一块又大又干净的连通信号区域，强行解包裹会出现几十弧度的块状错误，所以保持 wrapped |

## 三条限制

1. `amplitude_detail` 和 `phase_corrected` 是**相对图**，不能当作标定过的反射率或绝对光程。要做定量重建，请用 `amplitude_corrected` 和 `raw_phase`。
2. Leaf 的第三维含义未确认（时间还是深度）。91 张图请当作 91 张独立图像。放行的相位只是"自洽"（残差点 ≤0.1%），没有用 phantom 或第二种方法验证过。
3. 没有物理标定和临床真值，像素的毫米尺寸也没有确认。

## Vegetables 体数据文件

`vegetables_volume_21frames.npz` 包含 `amplitude_corrected`、`raw_amplitude`、`raw_phase`、`signal_mask`
（(21, 400, 1300)），`frame_ids`（21 … 140）和 `z_mm`（30.15 … 42.05，来自采集记录：起点 30.05 mm、步长 0.1 mm，frame 20 为起点）。

层间距不均匀：前 10 层间隔 0.1 mm，之后每层间隔 1 mm。做 3D 渲染时请按 `z_mm` 放置，不要按下标等距排。

## 想要其他格式或更多数据

告诉 Haobo：别的文件格式（TIFF / HDF5）、更多蔬菜帧（RDM 上共有 frame 20–155），或者别的 metadata 字段，都可以加。

也可以自己重新生成（需要原始数据和我的代码）：

```bash
git clone https://github.com/iixuo/thz-lfi-preprocessing.git && cd thz-lfi-preprocessing
git checkout v0.5-evaluation
pip install -e ".[dev,demo]"
PYTHONPATH=src python -m examples.run_spatial_processing
PYTHONPATH=src python -m examples.build_handoff
```

## 请回我一句反馈

用完后请告诉我三件事（我要写进论文）：用了哪些通道；有没有读不了或缺的字段；`amplitude_corrected` 和 `amplitude_detail` 哪个对你做图更有用。
