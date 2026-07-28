"""
医学影像分割数据集模块

提供数据集类、数据预处理/增强变换、数据集划分等功能。

主要导出：
- SegmentationDataset: 图像分割数据集类（从目录扫描）
- FileListSegDataset: 文件列表分割数据集类（预划分文件列表）
- BaseDataset: 数据集抽象基类
- Compose: 多变换组合器
- get_train_transforms: 获取训练集数据变换
- get_val_transforms: 获取验证集/测试集数据变换
- train_val_test_split: 训练/验证/测试集划分函数
- Resize, Normalize, ClaheEqualize, GaussianDenoise: 预处理变换
- RandomHorizontalFlip, RandomVerticalFlip, RandomRotation: 数据增强变换
- ToTensor: numpy转Tensor变换
"""

from .transforms import (
    Compose,
    Resize,
    Normalize,
    ClaheEqualize,
    GaussianDenoise,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    ToTensor,
    get_train_transforms,
    get_val_transforms,
)
from .base_dataset import BaseDataset
from .seg_dataset import SegmentationDataset, FileListSegDataset, train_val_test_split

__all__ = [
    'Compose',
    'Resize',
    'Normalize',
    'ClaheEqualize',
    'GaussianDenoise',
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'ToTensor',
    'get_train_transforms',
    'get_val_transforms',
    'BaseDataset',
    'SegmentationDataset',
    'FileListSegDataset',
    'train_val_test_split',
]

