"""
医学影像IO统一接口模块
提供统一的图像加载函数，根据文件扩展名自动选择对应的读取器
支持格式：
- 普通图片：JPG/JPEG/PNG/BMP/TIFF等
- DICOM医学影像：.dcm
- NIfTI神经影像：.nii, .nii.gz
"""

from pathlib import Path
from typing import Union, Any
import numpy as np

from . import image_reader
from . import dicom_reader
from . import nifti_reader
from .image_reader import normalize


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
DICOM_EXTENSIONS = {'.dcm'}
NIFTI_EXTENSIONS = {'.nii'}
NIFTI_COMPRESSED_EXTENSIONS = {'.gz'}


def load_image(file_path: Union[str, Path], **kwargs: Any) -> np.ndarray:
    """
    统一的医学影像加载函数，根据文件扩展名自动选择合适的读取器
    
    支持的格式：
    - 普通图片格式：.jpg, .jpeg, .png, .bmp, .tiff, .tif, .gif
      可选参数：
        - as_gray (bool): 是否读取为灰度图，默认为True
    - DICOM格式：.dcm
      可选参数：
        - convert_to_hu (bool): 是否转换为CT值(HU)，默认为False
    - NIfTI格式：.nii, .nii.gz
      可选参数：
        - slice_idx (int, optional): 3D体数据的切片索引，默认为None(中间层)
        - slice_axis (int): 切片轴方向，0=矢状位，1=冠状位，2=轴位(默认)
    
    Args:
        file_path: 图像文件路径，支持字符串或Path对象
        **kwargs: 传递给对应读取器的额外参数
    
    Returns:
        float32类型的numpy数组，像素值归一化到[0, 1]范围
        - 灰度图/医学影像：shape为(H, W)
        - 彩色图：shape为(H, W, 3)
    
    Raises:
        FileNotFoundError: 当文件不存在时
        ValueError: 当文件格式不支持或读取失败时
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    suffix = file_path.suffix.lower()
    suffixes = [s.lower() for s in file_path.suffixes]
    
    if suffix in DICOM_EXTENSIONS:
        return dicom_reader.load(file_path, **kwargs)
    
    if suffix in NIFTI_EXTENSIONS:
        return nifti_reader.load(file_path, **kwargs)
    
    if '.nii' in suffixes and '.gz' in suffixes:
        return nifti_reader.load(file_path, **kwargs)
    
    if suffix in IMAGE_EXTENSIONS:
        return image_reader.load(file_path, **kwargs)
    
    raise ValueError(
        f"不支持的文件格式: {suffix}，"
        f"支持的格式包括: 普通图片({', '.join(IMAGE_EXTENSIONS)}), "
        f"DICOM({', '.join(DICOM_EXTENSIONS)}), "
        f"NIfTI(.nii, .nii.gz)"
    )


__all__ = ['load_image', 'normalize', 'image_reader', 'dicom_reader', 'nifti_reader']
