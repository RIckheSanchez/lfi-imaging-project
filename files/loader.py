"""
loader.py
HDF5 文件加载函数 —— Part 1 核心模块

提供:
    load_single_scan()    加载单个 HDF5 文件
    load_multiple_scans() 批量加载一个文件夹下所有 HDF5 文件
    get_z_slices()        查询文件中的 z 层编号
"""

import os
import h5py
import numpy as np
from pathlib import Path
from typing import Optional

from data_types import LFIRawData


# ── 内部工具函数 ────────────────────────────────────────────────

def _infer_target_type(filepath: str) -> str:
    """
    从文件路径自动推断 target 类型
    规则: 取父文件夹名, 例如 .../Chip/borad1.hdf5 → 'Chip'
    """
    parent = Path(filepath).parent.name
    known = {'Chip', 'Leaf', 'Skin', 'Marble', 'Mushroom',
             'Sound', 'Tomography', 'Beam Profile', 'Spectrum'}
    if parent in known:
        return parent
    # 再试文件名本身
    stem = Path(filepath).stem
    for k in known:
        if k.lower() in stem.lower():
            return k
    return 'Unknown'


def _read_hdf5(filepath: str) -> LFIRawData:
    """
    实际读取 HDF5 文件的内部函数
    假设结构:
        /amplitude/<key>  shape (H, W)
        /phase/<key>      shape (H, W)
    """
    with h5py.File(filepath, 'r') as f:

        # ── 检查顶层 keys ──────────────────────────────────────
        keys = list(f.keys())
        if 'amplitude' not in keys or 'phase' not in keys:
            raise ValueError(
                f"文件缺少 'amplitude' 或 'phase' 顶层 group。\n"
                f"实际 keys: {keys}\n路径: {filepath}"
            )

        amp_group   = f['amplitude']
        phase_group = f['phase']

        # ── 读取 z 层编号并排序 ────────────────────────────────
        z_keys = sorted(amp_group.keys(), key=lambda x: int(x))
        z_indices = [int(k) for k in z_keys]

        # ── 读取所有层，堆叠成 (n_slices, H, W) ───────────────
        amp_slices   = [amp_group[k][:] for k in z_keys]
        phase_slices = [phase_group[k][:] for k in z_keys]

        amplitude = np.stack(amp_slices,   axis=0)  # (70, 1400, 400)
        phase     = np.stack(phase_slices, axis=0)

        # ── 文件级 attributes 作为 metadata ───────────────────
        metadata = {k: v for k, v in f.attrs.items()}

    target_type = _infer_target_type(filepath)

    return LFIRawData(
        amplitude   = amplitude,
        phase       = phase,
        z_indices   = z_indices,
        target_type = target_type,
        file_path   = str(Path(filepath).resolve()),
        metadata    = metadata,
    )


# ── 公开接口 ────────────────────────────────────────────────────

def load_single_scan(filepath: str) -> LFIRawData:
    """
    加载单个 HDF5 扫描文件

    Args:
        filepath: HDF5 文件路径, 例如 '/Users/.../Chip/borad1.hdf5'

    Returns:
        LFIRawData 对象

    Raises:
        FileNotFoundError : 文件不存在
        ValueError        : 文件结构不符合预期
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    if not filepath.endswith(('.hdf5', '.h5')):
        raise ValueError(f"不支持的文件格式，请提供 .hdf5 或 .h5 文件: {filepath}")

    print(f"[load_single_scan] 正在加载: {filepath}")
    data = _read_hdf5(filepath)
    print(f"[load_single_scan] 加载完成 → {data.amplitude.shape}")
    return data


def load_multiple_scans(
    directory: str,
    target_type: Optional[str] = None,
) -> list:
    """
    批量加载一个文件夹下所有 HDF5 文件

    Args:
        directory:   文件夹路径, 例如 '/Users/.../02 Data Sets/Chip'
        target_type: 可选，只加载特定类型, 例如 'Chip'
                     None 表示加载全部

    Returns:
        LFIRawData 对象列表（按文件名排序）

    Raises:
        FileNotFoundError : 文件夹不存在
        RuntimeError      : 文件夹内没有找到任何 HDF5 文件
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"文件夹不存在: {directory}")

    # 递归搜索所有 .hdf5 / .h5 文件
    hdf5_files = sorted(
        list(dir_path.rglob('*.hdf5')) + list(dir_path.rglob('*.h5'))
    )

    if not hdf5_files:
        raise RuntimeError(f"在 {directory} 下没有找到任何 HDF5 文件")

    results = []
    failed  = []

    for fp in hdf5_files:
        try:
            data = load_single_scan(str(fp))
            # 如果指定了 target_type，则过滤
            if target_type is None or data.target_type == target_type:
                results.append(data)
        except Exception as e:
            print(f"[警告] 跳过文件 {fp.name}: {e}")
            failed.append(str(fp))

    print(f"\n[load_multiple_scans] 完成: 成功 {len(results)} 个, 失败 {len(failed)} 个")
    if failed:
        print(f"  失败文件: {failed}")

    return results


def get_z_slices(filepath: str) -> list:
    """
    快速查询文件中的 z 层编号（不加载完整数据，节省内存）

    Args:
        filepath: HDF5 文件路径

    Returns:
        z 层编号列表, 例如 [530, 531, ..., 599]
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    with h5py.File(filepath, 'r') as f:
        if 'amplitude' not in f:
            raise ValueError("文件缺少 'amplitude' group")
        z_keys = sorted(f['amplitude'].keys(), key=lambda x: int(x))
        return [int(k) for k in z_keys]
