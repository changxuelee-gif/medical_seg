"""
NIfTI神经影像格式读取模块
使用nibabel库读取.nii和.nii.gz格式文件，支持3D体数据切片提取
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import nibabel as nib


def _normalize_to_01(img: np.ndarray) -> np.ndarray:
    """
    将图像数组归一化到[0, 1]范围
    
    Args:
        img: 输入图像数组
    
    Returns:
        归一化后的float32数组
    """
    img = img.astype(np.float32)
    img_min = np.min(img)
    img_max = np.max(img)
    
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.float32)
    
    normalized = (img - img_min) / (img_max - img_min)
    return normalized.astype(np.float32)


def _reorient_to_standard(data: np.ndarray) -> np.ndarray:
    """
    简单的方位校正，将NIfTI数据调整为标准放射学方向（RAS+）
    通过转置和翻转使图像方向符合常规观察习惯
    
    Args:
        data: 原始NIfTI图像数组（3D或2D）
    
    Returns:
        方位校正后的数组
    """
    if data.ndim == 2:
        data = np.flipud(data)
        data = np.fliplr(data)
    elif data.ndim == 3:
        data = np.flip(data, axis=0)
        data = np.flip(data, axis=1)
    
    return data


def load(
    file_path: Union[str, Path],
    slice_idx: Optional[int] = None,
    slice_axis: int = 2
) -> np.ndarray:
    """
    读取NIfTI格式神经影像文件（.nii或.nii.gz）
    
    Args:
        file_path: NIfTI文件路径，支持字符串或Path对象
        slice_idx: 要提取的2D切片索引，默认为None表示取中间层
        slice_axis: 切片轴方向，默认为2（即z轴方向取轴位切片）
                   0: 矢状位 (Sagittal)
                   1: 冠状位 (Coronal)
                   2: 轴位 (Axial) - 默认
    
    Returns:
        float32类型的2D numpy数组，像素值归一化到[0, 1]范围，shape为(H, W)
    
    Raises:
        FileNotFoundError: 当文件不存在时
        ValueError: 当文件格式不支持、切片索引越界或读取失败时
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"NIfTI文件不存在: {file_path}")
    
    suffix = file_path.suffix.lower()
    suffixes = [s.lower() for s in file_path.suffixes]
    
    is_nifti = False
    if suffix == '.nii':
        is_nifti = True
    elif '.nii' in suffixes and '.gz' in suffixes:
        is_nifti = True
    
    if not is_nifti:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}，仅支持.nii和.nii.gz格式")
    
    if slice_axis not in [0, 1, 2]:
        raise ValueError(f"切片轴参数slice_axis必须是0、1或2，当前值: {slice_axis}")
    
    try:
        img = nib.load(str(file_path))
        data = img.get_fdata()
        
        if data.ndim == 2:
            slice_data = data
        elif data.ndim == 3:
            if slice_idx is None:
                slice_idx = data.shape[slice_axis] // 2
            
            if slice_idx < 0 or slice_idx >= data.shape[slice_axis]:
                raise ValueError(
                    f"切片索引{slice_idx}越界，"
                    f"轴{slice_axis}的有效范围是[0, {data.shape[slice_axis] - 1}]"
                )
            
            if slice_axis == 0:
                slice_data = data[slice_idx, :, :]
            elif slice_axis == 1:
                slice_data = data[:, slice_idx, :]
            else:
                slice_data = data[:, :, slice_idx]
        elif data.ndim == 4:
            if slice_idx is None:
                slice_idx = data.shape[slice_axis] // 2
            
            if slice_idx < 0 or slice_idx >= data.shape[slice_axis]:
                raise ValueError(
                    f"切片索引{slice_idx}越界，"
                    f"轴{slice_axis}的有效范围是[0, {data.shape[slice_axis] - 1}]"
                )
            
            if slice_axis == 0:
                slice_data = data[slice_idx, :, :, 0]
            elif slice_axis == 1:
                slice_data = data[:, slice_idx, :, 0]
            else:
                slice_data = data[:, :, slice_idx, 0]
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}，仅支持2D、3D或4D数据")
        
        slice_data = _reorient_to_standard(slice_data)
        
        slice_data = _normalize_to_01(slice_data)
        
        return slice_data.astype(np.float32)
    
    except Exception as e:
        if "越界" in str(e) or "维度" in str(e):
            raise
        raise ValueError(f"读取NIfTI文件失败: {file_path}, 错误信息: {str(e)}") from e
