"""
DICOM医学影像格式读取模块
使用pydicom库读取.dcm格式文件，支持CT值（HU值）转换
"""

from pathlib import Path
from typing import Union
import numpy as np
import pydicom
from pydicom.dataset import FileDataset


def _apply_hu_conversion(ds: FileDataset, pixel_array: np.ndarray) -> np.ndarray:
    """
    应用HU值转换，将原始像素值转换为CT值（Hounsfield Units）
    
    Args:
        ds: pydicom数据集对象，包含DICOM元信息
        pixel_array: 原始像素数组
    
    Returns:
        转换为HU值后的float32数组
    """
    intercept = getattr(ds, 'RescaleIntercept', 0.0)
    slope = getattr(ds, 'RescaleSlope', 1.0)
    
    if slope != 1.0:
        pixel_array = pixel_array.astype(np.float64) * slope
        pixel_array = pixel_array.astype(np.int16)
    
    pixel_array = pixel_array.astype(np.float32) + np.float32(intercept)
    
    return pixel_array


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


def load(
    file_path: Union[str, Path],
    convert_to_hu: bool = False
) -> np.ndarray:
    """
    读取DICOM格式医学影像文件
    
    Args:
        file_path: DICOM文件路径，支持字符串或Path对象
        convert_to_hu: 是否转换为CT值（Hounsfield Units），默认为False
                      注意：仅CT模态的DICOM文件转换HU值才有意义
    
    Returns:
        float32类型的2D numpy数组，像素值归一化到[0, 1]范围，shape为(H, W)
    
    Raises:
        FileNotFoundError: 当文件不存在时
        ValueError: 当文件格式不是.dcm或读取失败时
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"DICOM文件不存在: {file_path}")
    
    if file_path.suffix.lower() != '.dcm':
        raise ValueError(f"不支持的文件格式: {file_path.suffix}，仅支持.dcm格式")
    
    try:
        ds = pydicom.dcmread(str(file_path))
        
        pixel_array = ds.pixel_array
        
        if pixel_array.ndim == 3:
            pixel_array = pixel_array[0]
        elif pixel_array.ndim > 3:
            pixel_array = pixel_array[0]
            while pixel_array.ndim > 2:
                pixel_array = pixel_array[0]
        
        pixel_array = pixel_array.astype(np.float32)
        
        if convert_to_hu:
            pixel_array = _apply_hu_conversion(ds, pixel_array)
        
        pixel_array = _normalize_to_01(pixel_array)
        
        return pixel_array.astype(np.float32)
    
    except Exception as e:
        raise ValueError(f"读取DICOM文件失败: {file_path}, 错误信息: {str(e)}") from e
