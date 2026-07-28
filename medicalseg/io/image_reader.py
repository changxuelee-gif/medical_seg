"""
普通图片读取模块，支持JPG、PNG、BMP等常见格式
使用PIL/Pillow库进行图片读取与处理
"""

from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image


def normalize(img: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """
    将图像数组归一化到指定范围
    
    Args:
        img: 输入图像数组，任意数值范围
        min_val: 归一化后的最小值，默认为0.0
        max_val: 归一化后的最大值，默认为1.0
    
    Returns:
        归一化后的float32类型数组，数值范围在[min_val, max_val]之间
    
    Raises:
        ValueError: 当图像数组所有像素值相同时（无法归一化）
    """
    img = img.astype(np.float32)
    img_min = np.min(img)
    img_max = np.max(img)
    
    if img_max == img_min:
        return np.full_like(img, min_val, dtype=np.float32)
    
    normalized = (img - img_min) / (img_max - img_min)
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized.astype(np.float32)


def load(file_path: Union[str, Path], as_gray: bool = True) -> np.ndarray:
    """
    读取普通图片文件（JPG/PNG/BMP等格式）
    
    Args:
        file_path: 图片文件路径，支持字符串或Path对象
        as_gray: 是否以灰度模式读取，True为单通道灰度图，False为三通道彩色图，默认为True
    
    Returns:
        float32类型的numpy数组，像素值归一化到[0, 1]范围
        - 灰度模式：shape为(H, W)
        - 彩色模式：shape为(H, W, 3)，通道顺序为RGB
    
    Raises:
        FileNotFoundError: 当文件不存在时
        ValueError: 当文件格式不支持或读取失败时
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {file_path}")
    
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
    if file_path.suffix.lower() not in supported_formats:
        raise ValueError(f"不支持的图片格式: {file_path.suffix}，支持的格式: {supported_formats}")
    
    try:
        img = Image.open(file_path)
        
        if as_gray:
            if img.mode != 'L':
                img = img.convert('L')
            img_array = np.array(img, dtype=np.float32)
        else:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img, dtype=np.float32)
        
        img_array = normalize(img_array, 0.0, 1.0)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"读取图片失败: {file_path}, 错误信息: {str(e)}") from e
