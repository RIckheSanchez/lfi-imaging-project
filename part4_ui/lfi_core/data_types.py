"""
data_types.py
定义 LFI 数据的标准输出格式
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class LFIRawData:
    """
    LFI 单个扫描文件的标准输出格式

    Attributes:
        amplitude:    幅度数据, shape: (n_slices, height, width)
        phase:        相位数据, shape: (n_slices, height, width)
        z_indices:    z 层编号列表, 例如 [530, 531, ..., 599]
        target_type:  目标类型, 例如 'Chip', 'Leaf', 'Skin'
        file_path:    原始 HDF5 文件路径
        metadata:     其他信息（预留，可扩展）
    """
    amplitude:   np.ndarray
    phase:       np.ndarray
    z_indices:   list
    target_type: str
    file_path:   str
    metadata:    dict = field(default_factory=dict)

    def __post_init__(self):
        # 基本校验
        if self.amplitude.shape != self.phase.shape:
            raise ValueError(
                f"amplitude 和 phase 的 shape 不一致: "
                f"{self.amplitude.shape} vs {self.phase.shape}"
            )
        if len(self.z_indices) != self.amplitude.shape[0]:
            raise ValueError(
                f"z_indices 数量 ({len(self.z_indices)}) "
                f"与数据层数 ({self.amplitude.shape[0]}) 不一致"
            )

    @property
    def n_slices(self) -> int:
        """z 方向层数"""
        return self.amplitude.shape[0]

    @property
    def height(self) -> int:
        return self.amplitude.shape[1]

    @property
    def width(self) -> int:
        return self.amplitude.shape[2]

    def get_slice(self, z_index: int) -> dict:
        """
        获取某一层的幅度和相位

        Args:
            z_index: z 层编号, 例如 530

        Returns:
            {'amplitude': 2D array, 'phase': 2D array}
        """
        if z_index not in self.z_indices:
            raise KeyError(f"z_index {z_index} 不存在，可用范围: {self.z_indices}")
        i = self.z_indices.index(z_index)
        return {
            'amplitude': self.amplitude[i],
            'phase':     self.phase[i],
        }

    def summary(self) -> str:
        """打印数据摘要"""
        return (
            f"LFIRawData Summary\n"
            f"  target_type : {self.target_type}\n"
            f"  file_path   : {self.file_path}\n"
            f"  amplitude   : shape={self.amplitude.shape}, dtype={self.amplitude.dtype}\n"
            f"  phase       : shape={self.phase.shape},     dtype={self.phase.dtype}\n"
            f"  z_indices   : {self.z_indices[0]} → {self.z_indices[-1]} "
            f"({self.n_slices} slices)\n"
            f"  metadata    : {self.metadata}"
        )
